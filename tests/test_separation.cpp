// SPDX-License-Identifier: 0BSD
/// @file test_separation.cpp
/// Separation's gravity (separation/gravity.hpp, estimate_gravity()), its proposal of the support
/// plane and the box (separation/proposal.hpp, propose_separation()) and the shared plane fit
/// (separation/plane_fit.hpp) on full models built here; the method is docs/methods/separation.md.
/// A full model is the points of known surfaces sampled one per voxel's square, each moved along
/// its surface's normal by Gaussian noise from a fixed seed: the support plane without the parts
/// things stand on, the box, other things standing around, walls; its trajectory is a walk of the
/// synthetic source (synthetic_source.hpp) around the box at 30 frames a second, a lap of the
/// tabletop box in 20 s and of the large box in 40 s, and its accelerometer's samples the
/// synthetic source's at 200 Hz along the walk, with the hand's acceleration. The models' frame is
/// the first camera's, as a scan's, so the plane lies tilted in it.
/// - The plane fit: a tilted plane's points give its normal and centroid; fewer than three points
///   give nothing.
/// - Gravity on the tabletop walk: as walked, shaken by 2 cm at 1.5 Hz, rolled by 60 degrees,
///   half of it rolling at 90 degrees a second, with a stretch of tracking loss whose samples
///   stay, with a gap of frames not taken whose samples go, and with the IMU's axes turned by 90
///   degrees and that rotation given: up within a measured tolerance of the known up, the samples
///   taken counted; the samples of exactly the least groups: estimated; the turned samples with
///   the identity given: no gravity, the spread too large; no sample, three seconds of samples,
///   one sample short of the least groups, all samples zero: no gravity with their outcomes;
///   refusals as values: the rotation not unit, a frame's timestamp not rising, a pose not finite,
///   a sample's time not rising, a sample not finite.
/// - A tabletop box among four other things, one of them larger than the box, one 3 cm beside it,
///   and a large box among three: gravity within the tolerance of the known up, the plane within
///   an angle and a distance of the known one, the box upright on it, its x axis along its longer
///   side, holding the known box within the margin and none of the other things' points.
/// - The box near a wall that holds more of the window's points than the support plane, for both
///   object classes, from half a walk in front of it: still the support plane, the box as above.
/// - A flat board whose top holds more of the window's points than the table seen around it, the
///   views meeting under its top: still the table.
/// - Every scene's walk rolled by 15, 30, 40, 45 and 60 degrees, and shaken: right. The large box
///   in a room's corner: right as walked and rolled by 30, 45 and 60 degrees either way, where up
///   from the cameras' axes proposed a side wall (docs/methods/separation.md, "The roll").
/// - Two things nearer than a footprint's cell join, two farther apart than 2 sqrt 2 cells do not,
///   for both object classes.
/// - A sphere alone, an empty model and a steep ramp alone: no plane. The support plane alone:
///   nothing on the plane.
/// - Parallel views: the views do not meet. Each outcome of gravity without up: no gravity, before
///   the views, with gravity's reason.
/// - Refusals as values: the voxel size and the farthest depth outside their ranges, gravity's
///   outcome no enumerator, an estimated up not unit, no pose, a pose not finite or its quaternion
///   not unit, while one within the tolerance is taken, a point not finite.

#include "camera/orbbec_imu.hpp"
#include "check.hpp"
#include "scan/object_class.hpp"
#include "separation/gravity.hpp"
#include "separation/plane_fit.hpp"
#include "separation/proposal.hpp"
#include "stage_checks.hpp"
#include "stage_frames.hpp"
#include "synthetic_source.hpp"
#include "types/gravity_estimate.hpp"
#include "types/imu_sample.hpp"
#include "types/trajectory_entry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::angle_deg;
using gca_test::check;
using gca_test::check_refusal;
using gca_test::check_within;
using gca_test::synthetic_box;
using gca_test::synthetic_sphere;
using gca_test::synthetic_surface;
using gca_test::synthetic_texture;

/// The test models' noise along a surface's normal, in voxels: σ = 0.25 v [?: a choice: about the
/// fused models' 99th percentiles of 0.60 mm at 1 mm voxels and 4.97 mm at 4 mm over 2.6
/// (docs/methods/sparse-tsdf.md, "What the extraction gives"), without their tails].
constexpr double noise_voxels = 0.25;

/// The frames of a lap at 30 frames a second: 600 around the tabletop box, 20 s, and 1200 around
/// the large box, 40 s [?: a choice, a hand-held lap].
constexpr std::size_t tabletop_lap_frames = 600;
constexpr std::size_t large_lap_frames = 1200;

/// The accelerometer's period, in µs: 200 Hz, the camera's (docs/methods/imu-samples.md).
constexpr std::uint64_t imu_period_us = 5000;

// A group of gravity's spread is one second of samples (separation/gravity.hpp,
// gravity_group_samples) only while the camera's IMU runs at orbbec_imu_rate_hz, which separation
// does not read: here both are seen, so that a change of the rate fails this test's build until
// the group's size is decided anew (docs/methods/separation.md, "The spread").
static_assert(static_cast<double>(gca::gravity_group_samples) == gca::orbbec_imu_rate_hz,
              "a group of gravity's spread is one second of the camera's IMU samples");

/// The hand's shake: its amplitude in m, its frequency in Hz, along a fixed direction that is
/// neither level nor vertical [?: a choice: a hand's sway and a walk's bob, 1.78 m/s² at its
/// peak].
constexpr double shake_m = 0.02;
constexpr double shake_hz = 1.5;

/// The rate of a walk's roll about the optical axis, in degrees a second: 90, tracking's ω_max
/// (docs/methods/tracking.md, "Parameters") [?: a choice: the fastest turn a hand gives that
/// tracking passes].
constexpr double roll_rate_deg_s = 90.0;

/// The largest angle of gravity's up from the known up, in degrees: about twice the largest over
/// every case [M 2026-11-17: this test; the largest 0.0934 deg, the large box by a wall, half a lap
/// of 20 s from rest; shaken 0.0734 deg, the tabletop box].
constexpr double gravity_tolerance_deg = 0.2;

/// What a point of a test model belongs to.
enum class part
{
    support, ///< the support plane
    object,  ///< the box the proposal is to find
    other,   ///< a thing standing around, a wall or another plane
};

/// A test scene: the known surfaces of a full model and the walk it was scanned along.
struct test_scene
{
    const char *name;          ///< printed with every measurement
    gca::object_class kind;    ///< the object class whose v and D the proposal takes
    synthetic_surface support; ///< the support plane, its normal up, or a zero extent for none
    std::optional<synthetic_box> object;    ///< the box the proposal is to find, or none
    std::vector<synthetic_box> other_boxes; ///< things standing around
    std::vector<synthetic_sphere> balls;    ///< things standing around, or the scene's only thing
    std::vector<synthetic_surface> planes;  ///< walls and other planes, each an other part
    /// The camera's poses, camera to world, one for each frame at 30 frames a second, at least one.
    std::vector<Eigen::Isometry3d> walk;
};

/// A full model built from a scene, in the models' frame, the walk's first camera's.
struct built_model
{
    std::vector<gca::model_point> points;  ///< the model
    std::vector<part> parts;               ///< what each point belongs to
    Eigen::Isometry3d to_model;            ///< from the world frame into the models' frame
    gca::separation_parameters parameters; ///< v and D of the scene's object class
};

/// A walk as a scan gives it to separation, in a model's frame.
struct walked
{
    /// Every frame of the walk, frame i at i synthetic_frame_period_us, tracked, camera to model.
    std::vector<gca::trajectory_entry> trajectory;
    std::vector<gca::camera_pose> poses;        ///< the tracked frames' poses, in their order
    std::vector<gca::imu_sample> accelerometer; ///< the samples along the walk, in the IMU's axes
};

/// Returns the level direction at a turn from the world's x axis toward its y.
/// @param deg [in] the turn, in degrees.
/// @return (cos, sin, 0).
Eigen::Vector3d level(double deg)
{
    const double rad = deg * std::numbers::pi / 180.0;
    return {std::cos(rad), std::sin(rad), 0.0};
}

/// Returns a box standing beside another, turned with it, beyond one of its sides.
/// @param box [in] the other box.
/// @param turn_deg [in] the side: 0 beyond its +x face, 90 beyond its +y face.
/// @param gap_m [in] the gap between the two facing faces, in m.
/// @param side_m [in] the new box's edge along both of its level axes, in m.
/// @param height_m [in] its height, in m.
/// @return the new box.
synthetic_box beside(const synthetic_box &box, double turn_deg, double gap_m, double side_m,
                     double height_m)
{
    const double half_m = 0.5 * (turn_deg == 0.0 ? box.size_x_m : box.size_y_m);
    const Eigen::Vector3d centre = Eigen::Vector3d(box.centre_x_m, box.centre_y_m, 0.0) +
                                   (half_m + gap_m + 0.5 * side_m) * level(box.yaw_deg + turn_deg);
    return {centre.x(), centre.y(), box.yaw_deg, side_m, side_m, height_m};
}

/// Returns a box's surfaces as the synthetic source renders them: its top and its four sides.
/// @param box [in] the box.
/// @return the surfaces, in the world frame (synthetic_source::surfaces()).
std::vector<synthetic_surface> surfaces_of(const synthetic_box &box)
{
    gca_test::synthetic_source_options options{};
    options.scene.box = box;
    return gca_test::synthetic_source(std::move(options)).surfaces();
}

/// Says whether a point of the plane z = 0 lies in a box's footprint.
/// @param box [in] the box.
/// @param point [in] the point, in the world frame, in m.
/// @return whether it lies within the box's rectangle on the plane.
bool under(const synthetic_box &box, const Eigen::Vector3d &point)
{
    const Eigen::Vector3d offset = point - Eigen::Vector3d(box.centre_x_m, box.centre_y_m, 0.0);
    return std::abs(offset.dot(level(box.yaw_deg))) <= 0.5 * box.size_x_m &&
           std::abs(offset.dot(level(box.yaw_deg + 90.0))) <= 0.5 * box.size_y_m;
}

/// Adds a point of a known surface to a model, moved along the surface's normal by the noise.
/// @param model [in,out] the model.
/// @param world [in] the point on the surface, in the world frame, in m.
/// @param normal [in] the surface's unit normal there, in the world frame.
/// @param which [in] what it belongs to.
/// @param noise [in,out] the noise's draws, of standard deviation σ in m.
/// @param generator [in,out] the generator the noise is drawn from.
void add_point(built_model &model, const Eigen::Vector3d &world, const Eigen::Vector3d &normal,
               part which, std::normal_distribution<double> &noise, std::mt19937_64 &generator)
{
    const Eigen::Vector3d moved = world + noise(generator) * normal;
    const Eigen::Vector3d n = (model.to_model.linear() * normal).normalized();
    model.points.push_back(
        {(model.to_model * moved).cast<float>(), n.cast<float>(), {128, 128, 128}});
    model.parts.push_back(which);
}

/// Samples a rectangle one point per square of the spacing, at the squares' centres.
/// @param model [in,out] the model the points are added to.
/// @param surface [in] the rectangle; its normal axis_u x axis_v.
/// @param spacing_m [in] the squares' edge, in m.
/// @param which [in] what the points belong to.
/// @param left_out [in] boxes whose footprints the points skip: what stands on the support plane.
/// @param noise [in,out] the noise's draws.
/// @param generator [in,out] the noise's generator.
void sample_rectangle(built_model &model, const synthetic_surface &surface, double spacing_m,
                      part which, const std::vector<synthetic_box> &left_out,
                      std::normal_distribution<double> &noise, std::mt19937_64 &generator)
{
    const Eigen::Vector3d normal = surface.axis_u.cross(surface.axis_v);
    const auto steps_u = static_cast<long>(std::floor(2.0 * surface.half_u_m / spacing_m));
    const auto steps_v = static_cast<long>(std::floor(2.0 * surface.half_v_m / spacing_m));
    for (long i = 0; i < steps_u; i++)
        for (long j = 0; j < steps_v; j++)
        {
            const Eigen::Vector3d world =
                surface.centre_m +
                (-surface.half_u_m + (static_cast<double>(i) + 0.5) * spacing_m) * surface.axis_u +
                (-surface.half_v_m + (static_cast<double>(j) + 0.5) * spacing_m) * surface.axis_v;
            if (std::any_of(left_out.begin(), left_out.end(),
                            [&](const synthetic_box &box) { return under(box, world); }))
                continue;
            add_point(model, world, normal, which, noise, generator);
        }
}

/// Samples the part of a sphere above the plane z = 0 at about one point per square of the spacing,
/// on a Fibonacci lattice (synthetic_source.hpp, sphere_lattice_direction()).
/// @param model [in,out] the model.
/// @param ball [in] the sphere.
/// @param spacing_m [in] the squares' edge, in m.
/// @param which [in] what the points belong to.
/// @param noise [in,out] the noise's draws.
/// @param generator [in,out] the noise's generator.
void sample_sphere(built_model &model, const synthetic_sphere &ball, double spacing_m, part which,
                   std::normal_distribution<double> &noise, std::mt19937_64 &generator)
{
    const double r = ball.radius_m;
    const auto count =
        static_cast<std::size_t>(4.0 * std::numbers::pi * r * r / (spacing_m * spacing_m));
    for (std::size_t k = 0; k < count; k++)
    {
        const Eigen::Vector3d direction = gca_test::sphere_lattice_direction(k, count);
        const Eigen::Vector3d world = ball.centre_m + r * direction;
        if (world.z() <= 0.0)
            continue;
        add_point(model, world, direction, which, noise, generator);
    }
}

/// Builds a scene's full model in the frame of its walk's first camera.
/// @param scene [in] the scene.
/// @param seed [in] the noise's seed.
/// @return the model, its parts, the object class's v and as D the far depth its tests were
///         measured at (gca_test::measured_range_of()).
built_model build(const test_scene &scene, std::uint64_t seed)
{
    const gca::object_class_parameters kind = gca::parameters_of(scene.kind);
    built_model model;
    model.parameters = {kind.full_voxel_size_m, gca_test::measured_range_of(scene.kind).far_m};
    model.to_model = scene.walk.front().inverse();
    const double v = kind.full_voxel_size_m;
    std::mt19937_64 generator(seed);
    std::normal_distribution<double> noise(0.0, noise_voxels * v);
    std::vector<synthetic_box> standing = scene.other_boxes;
    if (scene.object)
        standing.push_back(*scene.object);
    if (scene.support.half_u_m > 0.0)
        sample_rectangle(model, scene.support, v, part::support, standing, noise, generator);
    if (scene.object)
        for (const synthetic_surface &face : surfaces_of(*scene.object))
            sample_rectangle(model, face, v, part::object, {}, noise, generator);
    for (const synthetic_box &box : scene.other_boxes)
        for (const synthetic_surface &face : surfaces_of(box))
            sample_rectangle(model, face, v, part::other, {}, noise, generator);
    for (const synthetic_sphere &ball : scene.balls)
        sample_sphere(model, ball, v, part::other, noise, generator);
    for (const synthetic_surface &plane : scene.planes)
        sample_rectangle(model, plane, v, part::other, {}, noise, generator);
    return model;
}

/// Returns the known up in a model's frame.
/// @param model [in] the model, for its frame.
/// @return the world's z axis turned into the models' frame.
Eigen::Vector3d known_up(const built_model &model)
{
    return model.to_model.linear() * Eigen::Vector3d::UnitZ();
}

/// Gives a walk as a scan gives it: every frame tracked at its pose in the model's frame, and the
/// synthetic source's accelerometer samples along it with the hand's acceleration, turned into
/// the IMU's axes.
/// @param model [in] the model, for its frame.
/// @param walk [in] the poses, camera to world, frame i at i synthetic_frame_period_us.
/// @param imu_to_camera [in] the rotation from the IMU's axes into the camera frame the samples
///        are given in: each sample is Q^T of the camera frame's.
/// @return the trajectory, the poses and the samples.
walked walked_of(const built_model &model, const std::vector<Eigen::Isometry3d> &walk,
                 const Eigen::Quaterniond &imu_to_camera = Eigen::Quaterniond::Identity())
{
    walked out;
    std::vector<gca::imu_sample> gyro;
    const gca_test::synthetic_imu imu{imu_period_us, true};
    const Eigen::Matrix3d into_imu = imu_to_camera.toRotationMatrix().transpose();
    for (std::size_t i = 0; i < walk.size(); i++)
    {
        const Eigen::Isometry3d in_model = model.to_model * walk[i];
        const gca::camera_pose pose{in_model.translation(),
                                    Eigen::Quaterniond(in_model.linear()).normalized()};
        out.trajectory.push_back({i, i * gca_test::synthetic_frame_period_us, pose});
        out.poses.push_back(pose);
        gca_test::append_imu_samples(walk, i, imu, out.accelerometer, gyro);
    }
    for (gca::imu_sample &sample : out.accelerometer)
    {
        const Eigen::Vector3d turned = into_imu * Eigen::Vector3d(sample.x, sample.y, sample.z);
        sample.x = static_cast<float>(turned.x());
        sample.y = static_cast<float>(turned.y());
        sample.z = static_cast<float>(turned.z());
    }
    return out;
}

/// Returns a horizontal rectangle of the world, its normal up.
/// @param centre [in] its centre, in m.
/// @param half_x_m [in] half its extent along x, in m.
/// @param half_y_m [in] half its extent along y, in m.
/// @return the rectangle.
synthetic_surface floor_at(const Eigen::Vector3d &centre, double half_x_m, double half_y_m)
{
    return {centre,   Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(), half_x_m,
            half_y_m, synthetic_texture::plane};
}

/// Returns a wall: a vertical rectangle in the plane y = y_m from the floor up, facing -y.
/// @param y_m [in] its place along y, in m.
/// @param half_x_m [in] half its width along x, in m.
/// @param height_m [in] its height, in m.
/// @return the rectangle, its normal (0, -1, 0).
synthetic_surface wall_at(double y_m, double half_x_m, double height_m)
{
    return {Eigen::Vector3d(0.0, y_m, 0.5 * height_m),
            Eigen::Vector3d::UnitX(),
            Eigen::Vector3d::UnitZ(),
            half_x_m,
            0.5 * height_m,
            synthetic_texture::plane};
}

/// Builds a walk, or half of it, at 30 frames a second.
/// @param walk [in] the walk, its pose count the frames of a whole lap.
/// @param front_only [in] whether to keep only the poses with y below the walk's centre, the half
///        in front of a wall along +y, one stretch of the lap.
/// @return the poses, camera to world, frame i the i-th kept.
std::vector<Eigen::Isometry3d> walk_of(const gca_test::synthetic_walk &walk, bool front_only)
{
    const std::optional<std::vector<Eigen::Isometry3d>> poses = gca_test::walk_around(walk);
    check(poses.has_value(), "the walk is built");
    std::vector<Eigen::Isometry3d> kept;
    if (!poses)
        return kept;
    for (const Eigen::Isometry3d &pose : *poses)
        if (!front_only || pose.translation().y() < walk.centre_m.y())
            kept.push_back(pose);
    return kept;
}

/// Returns a walk with every camera rolled about its optical axis.
/// @param walk [in] the poses, camera to world.
/// @param roll_deg [in] the roll, in degrees, from the camera's x toward its y.
/// @return the rolled poses.
std::vector<Eigen::Isometry3d> rolled(const std::vector<Eigen::Isometry3d> &walk, double roll_deg)
{
    std::vector<Eigen::Isometry3d> out;
    for (const Eigen::Isometry3d &pose : walk)
    {
        Eigen::Isometry3d turned = pose;
        turned.linear() = pose.linear() * Eigen::AngleAxisd(roll_deg * std::numbers::pi / 180.0,
                                                            Eigen::Vector3d::UnitZ());
        out.push_back(turned);
    }
    return out;
}

/// Returns a walk whose cameras roll about their optical axes at a constant rate: frame i rolled by
/// the rate times its time, so that the camera turns about a level axis between frames, which a
/// sample's interpolated orientation follows and the earlier frame's does not
/// (docs/methods/separation.md, "Into the models' frame").
/// @param walk [in] the poses, camera to world, frame i at i synthetic_frame_period_us.
/// @param deg_per_s [in] the roll's rate, in degrees a second, from the camera's x toward its y.
/// @return the rolling poses.
std::vector<Eigen::Isometry3d> rolling(const std::vector<Eigen::Isometry3d> &walk, double deg_per_s)
{
    std::vector<Eigen::Isometry3d> out;
    for (std::size_t i = 0; i < walk.size(); i++)
    {
        const double t_s = 1e-6 * static_cast<double>(i * gca_test::synthetic_frame_period_us);
        out.push_back(rolled({walk[i]}, deg_per_s * t_s).front());
    }
    return out;
}

/// Returns a walk whose camera shakes: each position moved by shake_m sin(2π shake_hz t) along
/// (1, 2, 2) / 3 of the world, t frame i's time.
/// @param walk [in] the poses, camera to world, frame i at i synthetic_frame_period_us.
/// @return the shaken poses.
std::vector<Eigen::Isometry3d> shaken(const std::vector<Eigen::Isometry3d> &walk)
{
    const Eigen::Vector3d along = Eigen::Vector3d(1.0, 2.0, 2.0) / 3.0;
    std::vector<Eigen::Isometry3d> out;
    for (std::size_t i = 0; i < walk.size(); i++)
    {
        const double t_s = 1e-6 * static_cast<double>(i * gca_test::synthetic_frame_period_us);
        Eigen::Isometry3d moved = walk[i];
        moved.translation() += shake_m * std::sin(2.0 * std::numbers::pi * shake_hz * t_s) * along;
        out.push_back(moved);
    }
    return out;
}

/// Returns the trajectory of a walk in a model's frame.
/// @param model [in] the model, for its frame.
/// @param walk [in] the poses, camera to world.
/// @return the poses, camera to model.
std::vector<gca::camera_pose> trajectory_of(const built_model &model,
                                            const std::vector<Eigen::Isometry3d> &walk)
{
    std::vector<gca::camera_pose> out;
    for (const Eigen::Isometry3d &pose : walk)
    {
        const Eigen::Isometry3d in_model = model.to_model * pose;
        out.push_back({in_model.translation(), Eigen::Quaterniond(in_model.linear()).normalized()});
    }
    return out;
}

/// Returns an estimate that holds a known up, for the proposal's cases that are not gravity's.
/// @param up [in] the up, unit, in the models' frame.
/// @return the estimate, estimated.
gca::gravity_estimate estimate_of(const Eigen::Vector3d &up)
{
    gca::gravity_estimate known;
    known.outcome = gca::gravity_outcome::estimated;
    known.up = up;
    return known;
}

/// The tolerances a right proposal is checked against.
struct proposal_tolerance
{
    double angle_deg;       ///< the plane's and the box's z axis's angle from the known normal
    double distance_voxels; ///< the plane's distance from the known one at the known box's foot,
                            ///< and the known box's bottom below the box's
    double excess_voxels;   ///< the largest excess of the box's faces over the known box beyond μ
};

/// The tolerances, each about twice the largest over the scenes, every roll and the shaken walks
/// [M 2026-11-17: this test; the largest 0.0114 deg, by a wall on the table; 0.08 voxels, the
/// board; 1.22 voxels, the large box, the noise's farthest draw, 4.9 σ, beyond the margin].
constexpr proposal_tolerance tolerance = {0.03, 0.2, 2.0};

/// The measurements of a right proposal.
struct proposal_measure
{
    double plane_angle_deg = 0.0;   ///< the plane's normal's angle from the known one
    double plane_distance_mm = 0.0; ///< the plane's height at the known box's foot centre
    double box_angle_deg = 0.0;     ///< the box's z axis's angle from the known normal
    double least_excess_mm = 0.0;   ///< the least excess of a side or the top over the known box
    double largest_excess_mm = 0.0; ///< the largest excess of a side or the top over the known box
    double bottom_mm = 0.0;         ///< the known box's lowest corner's height in the box's frame
    std::size_t others_inside = 0;  ///< the points of other parts the box holds
};

/// Measures a proposal against a scene's known support plane and box.
/// @param scene [in] the scene, with its object.
/// @param model [in] its model.
/// @param proposal [in] a proposal with a plane and a box.
/// @return the measurements.
proposal_measure measure(const test_scene &scene, const built_model &model,
                         const gca::separation_proposal &proposal)
{
    proposal_measure m;
    const Eigen::Vector3d known_normal = model.to_model.linear() * Eigen::Vector3d::UnitZ();
    const synthetic_box &box = *scene.object;
    const Eigen::Vector3d foot =
        model.to_model * Eigen::Vector3d(box.centre_x_m, box.centre_y_m, 0.0);
    m.plane_angle_deg = angle_deg(proposal.plane.normal, known_normal);
    m.plane_distance_mm = 1e3 * std::abs(proposal.plane.height_m(foot));
    m.box_angle_deg = angle_deg(proposal.box.pose.linear().col(2), known_normal);
    const Eigen::Vector3d half = 0.5 * proposal.box.size_m;
    Eigen::Vector3d low = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
    Eigen::Vector3d high = -low;
    for (int corner = 0; corner < 8; corner++)
    {
        const Eigen::Vector3d world =
            Eigen::Vector3d(box.centre_x_m, box.centre_y_m, 0.0) +
            ((corner & 1) ? 0.5 : -0.5) * box.size_x_m * level(box.yaw_deg) +
            ((corner & 2) ? 0.5 : -0.5) * box.size_y_m * level(box.yaw_deg + 90.0) +
            ((corner & 4) ? box.height_m : 0.0) * Eigen::Vector3d::UnitZ();
        const Eigen::Vector3d q = proposal.box.pose.inverse() * (model.to_model * world);
        low = low.cwiseMin(q);
        high = high.cwiseMax(q);
    }
    const double excesses[] = {half.x() - high.x(), half.x() + low.x(), half.y() - high.y(),
                               half.y() + low.y(), proposal.box.size_m.z() - high.z()};
    m.least_excess_mm = 1e3 * *std::min_element(std::begin(excesses), std::end(excesses));
    m.largest_excess_mm = 1e3 * *std::max_element(std::begin(excesses), std::end(excesses));
    m.bottom_mm = 1e3 * low.z();
    for (std::size_t k = 0; k < model.points.size(); k++)
        if (model.parts[k] == part::other &&
            proposal.box.holds(model.points[k].position_m.cast<double>()))
            m.others_inside++;
    return m;
}

/// Checks a proposal that has to be right: proposed, the plane within the tolerance of the known
/// one, the box upright on it holding the known box within the margin and no point of another part.
/// @param scene [in] the scene, with its object.
/// @param model [in] its model.
/// @param proposal [in] the proposal.
/// @param what [in] the case, printed.
void check_proposal_right(const test_scene &scene, const built_model &model,
                          const gca::separation_proposal &proposal, const std::string &what)
{
    check(proposal.proposed(), (what + ": proposed").c_str());
    if (!proposal.proposed())
    {
        std::printf("%s: not proposed: %s\n", what.c_str(), proposal.message.c_str());
        return;
    }
    const proposal_measure m = measure(scene, model, proposal);
    const double voxel_mm = 1e3 * model.parameters.voxel_size_m;
    const double margin_mm = gca::separation_margin_voxels * voxel_mm;
    std::printf("%s: plane %zu points, object %zu points, window %.3f m\n", what.c_str(),
                proposal.plane_points, proposal.object_points, proposal.window_m);
    check_within((what + ": the plane's angle from the known one, deg").c_str(), m.plane_angle_deg,
                 tolerance.angle_deg);
    check_within((what + ": the plane's distance at the box's foot, voxels").c_str(),
                 m.plane_distance_mm / voxel_mm, tolerance.distance_voxels);
    check_within((what + ": the box's z axis's angle from the known normal, deg").c_str(),
                 m.box_angle_deg, tolerance.angle_deg);
    std::printf("%s: the box's faces exceed the known box by %.2f to %.2f mm, the margin %.1f mm\n",
                what.c_str(), m.least_excess_mm, m.largest_excess_mm, margin_mm);
    check(m.least_excess_mm >= 0.0, (what + ": the box holds the known box").c_str());
    check_within(
        (what + ": the box's largest excess over the known box beyond the margin, voxels").c_str(),
        (m.largest_excess_mm - margin_mm) / voxel_mm, tolerance.excess_voxels);
    check_within((what + ": the known box's bottom below the box's, voxels").c_str(),
                 std::abs(m.bottom_mm) / voxel_mm, tolerance.distance_voxels);
    check(proposal.box.size_m.x() >= proposal.box.size_m.y(),
          (what + ": the box's x axis along its longer side").c_str());
    check(m.others_inside == 0, (what + ": the box holds no point of another thing").c_str());
    if (m.others_inside != 0)
        std::printf("%s: the box holds %zu points of other things\n", what.c_str(),
                    m.others_inside);
}

/// Estimates gravity along a walk and checks it: estimated, up within gravity_tolerance_deg of the
/// known up.
/// @param model [in] the model, for its frame.
/// @param walk [in] the poses, camera to world, at 30 frames a second.
/// @param what [in] the case, printed.
/// @return the walk as a scan gives it, and the estimate.
std::pair<walked, gca::gravity_estimate> checked_gravity(const built_model &model,
                                                         const std::vector<Eigen::Isometry3d> &walk,
                                                         const std::string &what)
{
    walked scan = walked_of(model, walk);
    const gca::gravity_result gravity =
        gca::estimate_gravity(scan.trajectory, scan.accelerometer, gca::imu_to_camera());
    const gca::gravity_estimate &e = gravity.estimate;
    check(e.outcome == gca::gravity_outcome::estimated, (what + ": gravity estimated").c_str());
    std::printf(
        "%s: gravity of %llu samples in %llu groups, spread %.3f deg, mean %.4f m/s^2; %s\n",
        what.c_str(), static_cast<unsigned long long>(e.samples),
        static_cast<unsigned long long>(e.groups), e.spread_deg, e.mean_mps2,
        gca::reason_of(e).c_str());
    check_within((what + ": gravity's up from the known up, deg").c_str(),
                 angle_deg(e.up, known_up(model)), gravity_tolerance_deg);
    return {std::move(scan), e};
}

/// Estimates gravity along a walk and proposes from it: both right.
/// @param scene [in] the scene, with its object.
/// @param model [in] its model.
/// @param walk [in] the poses, camera to world, at 30 frames a second.
/// @param what [in] the case, printed.
/// @return the proposal.
gca::separation_proposal check_right(const test_scene &scene, const built_model &model,
                                     const std::vector<Eigen::Isometry3d> &walk,
                                     const std::string &what)
{
    const auto [scan, gravity] = checked_gravity(model, walk, what);
    const gca::separation_proposal proposal =
        gca::propose_separation(model.points, scan.poses, gravity, model.parameters);
    check_proposal_right(scene, model, proposal, what);
    return proposal;
}

/// Counts a model's points of one part within a distance of a point.
/// @param model [in] the model.
/// @param which [in] the part.
/// @param centre_m [in] the point, in the models' frame, in m.
/// @param radius_m [in] the distance, in m.
/// @return the count.
std::size_t part_points_within(const built_model &model, part which,
                               const Eigen::Vector3d &centre_m, double radius_m)
{
    std::size_t count = 0;
    for (std::size_t k = 0; k < model.points.size(); k++)
        if (model.parts[k] == which &&
            (model.points[k].position_m.cast<double>() - centre_m).norm() <= radius_m)
            count++;
    return count;
}

/// Gives a walk the frames of a lap at 30 frames a second.
/// @param walk [in] the walk.
/// @param frames [in] the frames of a whole lap.
/// @return the walk with that pose count.
gca_test::synthetic_walk lap_of(gca_test::synthetic_walk walk, std::size_t frames)
{
    walk.pose_count = frames;
    return walk;
}

/// The tabletop scene: the shared tabletop box on its plane among four other things, a tall
/// narrow box, a flat box larger than the box in points and a ball, each at least 9 cm from it, and
/// a small box 3 cm beside it [?: a choice], walked around once (tabletop_walk()) in 20 s.
/// @return the scene.
test_scene tabletop_scene()
{
    return {"tabletop",
            gca::object_class::tabletop_object,
            floor_at(Eigen::Vector3d::Zero(), gca_test::tabletop_plane.half_size_x_m,
                     gca_test::tabletop_plane.half_size_y_m),
            gca_test::tabletop_box,
            {{-0.30, -0.15, 10.0, 0.06, 0.06, 0.25},
             {0.0, 0.28, 0.0, 0.30, 0.18, 0.08},
             beside(gca_test::tabletop_box, 90.0, 0.03, 0.05, 0.08)},
            {{Eigen::Vector3d(0.30, -0.25, 0.04), 0.04}},
            {},
            walk_of(lap_of(gca_test::tabletop_walk(), tabletop_lap_frames), false)};
}

/// The large box: 1.2 m x 0.8 m x 1.0 m, a large object of 0.5 m to 2 m (GLOSSARY.md), turned by
/// 25 degrees [?: a choice].
constexpr synthetic_box large_box = {0.10, -0.05, 25.0, 1.20, 0.80, 1.00};

/// The walk once around the large box in 40 s: 2.3 m to 2.7 m from its centre, 1.5 m to 1.7 m
/// high, rolled by up to 10 degrees, looking at points within 0.1 m of its centre [?: a choice].
/// @param centre_m [in] the point walked around.
/// @return the walk.
gca_test::synthetic_walk large_walk(const Eigen::Vector3d &centre_m)
{
    return {centre_m, large_lap_frames, 2.5, 0.2, 1.6, 0.1, 10.0, 0.1};
}

/// The large scene: the large box on a floor of 4.2 m x 4.2 m among a cabinet, a crate larger than
/// the box in points and a ball, each at least 0.25 m from it, inside the walk [?: a choice].
/// @return the scene.
test_scene large_scene()
{
    return {"large",
            gca::object_class::large_object,
            floor_at(Eigen::Vector3d::Zero(), 2.1, 2.1),
            large_box,
            {{-1.30, 0.80, 0.0, 0.50, 0.50, 1.80}, {0.0, 1.45, 0.0, 1.60, 0.90, 1.20}},
            {{Eigen::Vector3d(1.40, -1.00, 0.35), 0.35}},
            {},
            walk_of(large_walk(Eigen::Vector3d(0.10, -0.05, 0.5)), false)};
}

/// The tabletop box on a small table, 0.6 m x 0.5 m, against a wall of 1.2 m x 1.0 m at the
/// table's back edge, 0.15 m behind the box, seen from the half of a steep walk in front of it:
/// 0.32 m to 0.38 m from the box, 0.61 m to 0.71 m high, looking down by about 60 degrees, half of
/// a lap of 20 s [?: a choice: where the mean of the cameras' up axes leaned by 50 degrees, which
/// up from the cameras' axes had to correct (docs/methods/separation.md, "The roll")].
/// @return the scene.
test_scene tabletop_wall_scene()
{
    const gca_test::synthetic_walk walk = {Eigen::Vector3d(gca_test::tabletop_box.centre_x_m,
                                                           gca_test::tabletop_box.centre_y_m,
                                                           0.5 * gca_test::tabletop_box.height_m),
                                           tabletop_lap_frames,
                                           0.35,
                                           0.03,
                                           0.66,
                                           0.05,
                                           10.0,
                                           0.02};
    return {"tabletop by a wall",
            gca::object_class::tabletop_object,
            floor_at(Eigen::Vector3d::Zero(), 0.30, 0.25),
            gca_test::tabletop_box,
            {},
            {},
            {wall_at(0.25, 0.6, 1.0)},
            walk_of(walk, true)};
}

/// The large box on a floor of 3 m x 3 m, against a wall of 5 m x 2.6 m at the floor's back edge,
/// 0.23 m behind the box, seen from the half of the large walk in front of it [?: a choice].
/// @return the scene.
test_scene large_wall_scene()
{
    synthetic_box box = large_box;
    box.centre_y_m = 0.55;
    return {"large by a wall",
            gca::object_class::large_object,
            floor_at(Eigen::Vector3d::Zero(), 1.5, 1.5),
            box,
            {},
            {},
            {wall_at(1.5, 2.5, 2.6)},
            walk_of(large_walk(Eigen::Vector3d(0.10, 0.55, 0.5)), true)};
}

/// A flat board, 0.8 m x 0.7 m x 0.04 m, on the tabletop plane, walked around at its middle
/// height in 20 s: its top holds more of the window's points than the table seen around it
/// [?: a choice].
/// @return the scene.
test_scene board_scene()
{
    const gca_test::synthetic_walk walk = {
        Eigen::Vector3d(0.0, 0.0, 0.02), tabletop_lap_frames, 0.65, 0.05, 0.40, 0.05, 10.0, 0.02};
    return {"a flat board",
            gca::object_class::tabletop_object,
            floor_at(Eigen::Vector3d::Zero(), gca_test::tabletop_plane.half_size_x_m,
                     gca_test::tabletop_plane.half_size_y_m),
            synthetic_box{0.0, 0.0, 0.0, 0.80, 0.70, 0.04},
            {},
            {},
            {},
            walk_of(walk, false)};
}

/// The plane fit on a tilted plane's points and on too few points.
void check_plane_fit()
{
    const Eigen::Vector3d normal = Eigen::Vector3d(0.3, -0.4, 0.866).normalized();
    const Eigen::Vector3d u = normal.cross(Eigen::Vector3d::UnitX()).normalized();
    const Eigen::Vector3d w = normal.cross(u);
    const Eigen::Vector3d centre(0.2, -0.1, 1.5);
    std::vector<Eigen::Vector3d> points;
    for (int i = -10; i <= 10; i++)
        for (int j = -5; j <= 5; j++)
            points.push_back(centre + 0.01 * i * u + 0.02 * j * w);
    const std::optional<gca::plane_fit> fit = gca::fit_plane(points);
    check(fit.has_value(), "the plane fit: a fit of 231 points");
    if (fit)
    {
        check_within("the plane fit: its normal's angle from the known one, deg",
                     std::min(angle_deg(fit->normal, normal), angle_deg(-fit->normal, normal)),
                     1e-9);
        check_within("the plane fit: its centroid's distance from the known centre, m",
                     (fit->centroid - centre).norm(), 1e-12);
        check_within("the plane fit: its least eigenvalue, m^2", std::abs(fit->eigenvalues(0)),
                     1e-12);
        check(fit->eigenvalues(0) <= fit->eigenvalues(1) &&
                  fit->eigenvalues(1) <= fit->eigenvalues(2),
              "the plane fit: its eigenvalues in increasing order");
    }
    points.resize(2);
    check(!gca::fit_plane(points).has_value(), "the plane fit: nothing from two points");
}

/// Counts the samples whose time lies in an interval.
/// @param samples [in] the samples.
/// @param after_us [in] the interval's start, in µs, not in it.
/// @param before_us [in] the interval's end, in µs, not in it.
/// @return the count of samples with after_us < t < before_us.
std::size_t samples_between(const std::vector<gca::imu_sample> &samples, std::uint64_t after_us,
                            std::uint64_t before_us)
{
    return static_cast<std::size_t>(
        std::count_if(samples.begin(), samples.end(), [&](const gca::imu_sample &s) {
            return s.timestamp_us > after_us && s.timestamp_us < before_us;
        }));
}

/// Checks an estimate that has to be estimated within the tolerance of the known up, and its
/// samples taken.
/// @param model [in] the model, for the known up.
/// @param result [in] the estimate.
/// @param samples [in] the samples it has to have taken.
/// @param what [in] the case, printed.
void check_estimated(const built_model &model, const gca::gravity_result &result,
                     std::size_t samples, const std::string &what)
{
    const gca::gravity_estimate &e = result.estimate;
    std::printf("%s: %llu samples in %llu groups, spread %.3f deg, mean %.4f m/s^2; %s%s\n",
                what.c_str(), static_cast<unsigned long long>(e.samples),
                static_cast<unsigned long long>(e.groups), e.spread_deg, e.mean_mps2,
                gca::reason_of(e).c_str(), result.refusal.c_str());
    check(e.outcome == gca::gravity_outcome::estimated, (what + ": estimated").c_str());
    check(e.samples == samples, (what + ": the samples taken").c_str());
    check(e.groups == samples / gca::gravity_group_samples, (what + ": the groups").c_str());
    check_within((what + ": gravity's up from the known up, deg").c_str(),
                 angle_deg(e.up, known_up(model)), gravity_tolerance_deg);
}

/// Checks an estimate that has to end without up.
/// @param result [in] the estimate.
/// @param outcome [in] the outcome it has to have.
/// @param what [in] the case, printed.
void check_no_up(const gca::gravity_result &result, gca::gravity_outcome outcome,
                 const std::string &what)
{
    const gca::gravity_estimate &e = result.estimate;
    std::printf("%s: %llu samples in %llu groups, spread %.3f deg, mean %.4f m/s^2: %s\n",
                what.c_str(), static_cast<unsigned long long>(e.samples),
                static_cast<unsigned long long>(e.groups), e.spread_deg, e.mean_mps2,
                gca::reason_of(e).c_str());
    check(e.outcome == outcome && e.up == Eigen::Vector3d::Zero() && result.refusal.empty(),
          (what + ": no up, its outcome").c_str());
}

/// Gravity on the tabletop walk (docs/methods/separation.md, "Gravity's up"): as walked, shaken,
/// rolled, half of it rolling, with tracking loss, with a gap, from a later first frame, with the
/// IMU's axes turned; the turned samples without their rotation, no sample, three seconds, the
/// least groups' samples exactly and one fewer, zero samples; refusals.
void check_gravity()
{
    const test_scene scene = tabletop_scene();
    const built_model model = build(scene, 71);
    const Eigen::Quaterniond identity = gca::imu_to_camera();
    const walked as_walked = walked_of(model, scene.walk);
    const std::size_t all = as_walked.accelerometer.size();
    check_estimated(model,
                    gca::estimate_gravity(as_walked.trajectory, as_walked.accelerometer, identity),
                    all, "gravity, as walked");
    const walked shake = walked_of(model, shaken(scene.walk));
    check_estimated(model, gca::estimate_gravity(shake.trajectory, shake.accelerometer, identity),
                    all, "gravity, shaken by 2 cm at 1.5 Hz");
    const walked roll = walked_of(model, rolled(scene.walk, 60.0));
    check_estimated(model, gca::estimate_gravity(roll.trajectory, roll.accelerometer, identity),
                    all, "gravity, rolled by 60 deg");
    // Half the lap, its cameras rolling at roll_rate_deg_s: between two frames the camera turns
    // about its optical axis, a level axis but for the look down, so a sample turned by the earlier
    // frame's orientation instead of the interpolated one leans by up to 3 degrees, along the
    // walk's tangent, which half a lap does not average out.
    const std::vector<Eigen::Isometry3d> half(scene.walk.begin(),
                                              scene.walk.begin() + tabletop_lap_frames / 2);
    const walked turning = walked_of(model, rolling(half, roll_rate_deg_s));
    check_estimated(model,
                    gca::estimate_gravity(turning.trajectory, turning.accelerometer, identity),
                    turning.accelerometer.size(), "gravity, half the lap rolling at 90 deg/s");

    // Frames 200 to 259 in tracking loss: the samples between frames 199 and 260 are not taken.
    walked lost = walked_of(model, shaken(scene.walk));
    for (std::size_t i = 200; i < 260; i++)
        lost.trajectory[i].pose.reset();
    check_estimated(model, gca::estimate_gravity(lost.trajectory, lost.accelerometer, identity),
                    all - samples_between(lost.accelerometer, lost.trajectory[199].timestamp_us,
                                          lost.trajectory[260].timestamp_us),
                    "gravity, shaken, frames 200 to 259 in tracking loss");

    // Frames 400 to 429 not taken: neither they nor their samples, those of (t_399, t_429], are
    // there; frames 399 and 430 enclose the rest.
    walked gap = walked_of(model, shaken(scene.walk));
    const std::uint64_t gap_after_us = gap.trajectory[399].timestamp_us;
    const std::uint64_t gap_last_us = gap.trajectory[429].timestamp_us;
    gap.trajectory.erase(gap.trajectory.begin() + 400, gap.trajectory.begin() + 430);
    std::erase_if(gap.accelerometer, [&](const gca::imu_sample &s) {
        return s.timestamp_us > gap_after_us && s.timestamp_us <= gap_last_us;
    });
    check_estimated(model, gca::estimate_gravity(gap.trajectory, gap.accelerometer, identity),
                    gap.accelerometer.size(), "gravity, shaken, frames 400 to 429 not taken");

    // The trajectory from frame 10 on: the samples before its first frame are not taken.
    walked later = walked_of(model, scene.walk);
    const std::uint64_t first_us = later.trajectory[10].timestamp_us;
    later.trajectory.erase(later.trajectory.begin(), later.trajectory.begin() + 10);
    check_estimated(model, gca::estimate_gravity(later.trajectory, later.accelerometer, identity),
                    all - samples_between(later.accelerometer, 0, first_us) - 1,
                    "gravity, the trajectory from frame 10 on");

    // The IMU's axes turned by 90 degrees about a tilted axis: right with that rotation given,
    // spread too wide with the identity.
    const Eigen::Quaterniond turned(
        Eigen::AngleAxisd(0.5 * std::numbers::pi, Eigen::Vector3d(1.0, 1.0, 1.0).normalized()));
    const walked axes = walked_of(model, scene.walk, turned);
    check_estimated(model, gca::estimate_gravity(axes.trajectory, axes.accelerometer, turned), all,
                    "gravity, the IMU's axes turned by 90 deg, the rotation given");
    check_no_up(gca::estimate_gravity(axes.trajectory, axes.accelerometer, identity),
                gca::gravity_outcome::spread_too_large,
                "gravity, the IMU's axes turned by 90 deg, the identity given");

    check_no_up(gca::estimate_gravity(as_walked.trajectory, {}, identity),
                gca::gravity_outcome::no_samples, "gravity, no sample");
    walked short_scan = walked_of(model, scene.walk);
    short_scan.trajectory.resize(90);
    std::erase_if(short_scan.accelerometer, [&](const gca::imu_sample &s) {
        return s.timestamp_us > short_scan.trajectory.back().timestamp_us;
    });
    const gca::gravity_result three =
        gca::estimate_gravity(short_scan.trajectory, short_scan.accelerometer, identity);
    check_no_up(three, gca::gravity_outcome::too_few_samples, "gravity, three seconds");
    check(three.estimate.samples == short_scan.accelerometer.size() &&
              three.estimate.groups < gca::gravity_least_groups,
          "gravity, three seconds: the samples and groups counted");
    // The least groups at its boundary: the lap's first samples filling exactly
    // gravity_least_groups groups are estimated; one sample fewer fills a group fewer, too few.
    // Their up is printed, not checked: five seconds from rest lean it by the walk's change of
    // velocity, 0.264 degrees [M 2026-11-17: this test], within the 2.3 degrees the method allows
    // one stretch of five seconds (docs/methods/separation.md, "The mean").
    walked least = walked_of(model, scene.walk);
    least.accelerometer.resize(gca::gravity_least_groups * gca::gravity_group_samples);
    const gca::gravity_result filled =
        gca::estimate_gravity(least.trajectory, least.accelerometer, identity);
    std::printf("gravity, %zu samples: %llu groups, spread %.3f deg, up %.4f deg from the known\n",
                least.accelerometer.size(), static_cast<unsigned long long>(filled.estimate.groups),
                filled.estimate.spread_deg, angle_deg(filled.estimate.up, known_up(model)));
    check(filled.estimate.outcome == gca::gravity_outcome::estimated &&
              filled.estimate.samples == least.accelerometer.size() &&
              filled.estimate.groups == gca::gravity_least_groups,
          "gravity, the least groups' samples exactly: estimated, the samples and groups counted");
    least.accelerometer.pop_back();
    const gca::gravity_result short_one =
        gca::estimate_gravity(least.trajectory, least.accelerometer, identity);
    check_no_up(short_one, gca::gravity_outcome::too_few_samples,
                "gravity, one sample short of the least groups");
    check(short_one.estimate.samples == least.accelerometer.size() &&
              short_one.estimate.groups == gca::gravity_least_groups - 1,
          "gravity, one sample short of the least groups: the samples and a group fewer counted");
    walked zero = walked_of(model, scene.walk);
    for (gca::imu_sample &s : zero.accelerometer)
        s.x = s.y = s.z = 0.0f;
    check_no_up(gca::estimate_gravity(zero.trajectory, zero.accelerometer, identity),
                gca::gravity_outcome::mean_zero, "gravity, every sample zero");

    // Refusals as values, each naming its input.
    const auto refused = [&](const std::string &what, const std::vector<gca::trajectory_entry> &t,
                             const std::vector<gca::imu_sample> &a, const Eigen::Quaterniond &q,
                             const std::string &expected) {
        const gca::gravity_result r = gca::estimate_gravity(t, a, q);
        check_refusal("gravity, " + what, r.estimate.outcome == gca::gravity_outcome::refused,
                      r.refusal,
                      r.refusal.find(expected) == 0 && r.estimate.up == Eigen::Vector3d::Zero());
    };
    const std::vector<gca::trajectory_entry> &t = as_walked.trajectory;
    const std::vector<gca::imu_sample> &a = as_walked.accelerometer;
    Eigen::Quaterniond long_q = identity;
    long_q.coeffs() *= 1.0 + 2e-9;
    refused("the rotation's norm 1 + 2e-9", t, a, long_q,
            "the rotation from the IMU's axes into the camera frame: its quaternion's norm "
            "differs from 1 by 2");
    std::vector<gca::trajectory_entry> bad_t = t;
    bad_t[5].timestamp_us = bad_t[4].timestamp_us;
    refused("a frame's timestamp not rising", bad_t, a, identity,
            "frame 5 of the trajectory: its timestamp 133332 us does not follow the frame "
            "before's 133332 us");
    bad_t = t;
    bad_t[7].pose->position_m.x() = std::numeric_limits<double>::quiet_NaN();
    refused("a pose not finite", bad_t, a, identity,
            "frame 7 of the trajectory: its position is not finite");
    bad_t = t;
    bad_t[8].pose->orientation.coeffs() *= 1.0 + 2e-9;
    refused("a quaternion of norm 1 + 2e-9", bad_t, a, identity,
            "frame 8 of the trajectory: its quaternion's norm differs from 1 by 2");
    std::vector<gca::imu_sample> bad_a = a;
    bad_a[3].timestamp_us = bad_a[2].timestamp_us;
    refused("a sample's time not rising", t, bad_a, identity,
            "accelerometer sample 3: its time 10000 us does not follow the sample before's "
            "10000 us");
    bad_a = a;
    bad_a[9].y = std::numeric_limits<float>::infinity();
    refused("a sample not finite", t, bad_a, identity,
            "accelerometer sample 9: its value is not finite");
}

/// Proposes from a scene's walk with every camera rolled: right at every roll.
/// @param scene [in] the scene, with its object.
/// @param model [in] its model.
/// @param rolls_deg [in] the rolls, in degrees.
void check_rolls(const test_scene &scene, const built_model &model,
                 const std::vector<double> &rolls_deg)
{
    for (const double roll_deg : rolls_deg)
        check_right(scene, model, rolled(scene.walk, roll_deg),
                    std::string(scene.name) + ", rolled by " +
                        std::to_string(static_cast<int>(roll_deg)) + " deg");
}

/// The large box in a room's corner: the wall behind it and a side wall on each side, 1.5 m from
/// the walk's centre, seen from the half of the large walk in front of the back wall [?: a choice,
/// the spec review's scene]: right as walked, rolled by 30, 45 and 60 degrees either way, where up
/// from the cameras' axes leaned so far toward a side wall that the wall passed the plane's
/// conditions (docs/methods/separation.md, "The roll"), and shaken.
void check_corner()
{
    test_scene scene = large_wall_scene();
    scene.name = "large in a corner";
    scene.planes.push_back({Eigen::Vector3d(1.5, 0.0, 1.3), Eigen::Vector3d::UnitZ(),
                            Eigen::Vector3d::UnitY(), 1.3, 1.5, synthetic_texture::plane});
    scene.planes.push_back({Eigen::Vector3d(-1.5, 0.0, 1.3), Eigen::Vector3d::UnitY(),
                            Eigen::Vector3d::UnitZ(), 1.5, 1.3, synthetic_texture::plane});
    const built_model model = build(scene, 5);
    std::printf("%s: %zu points, %zu poses\n", scene.name, model.points.size(), scene.walk.size());
    check_right(scene, model, scene.walk, scene.name);
    check_rolls(scene, model, {30.0, -30.0, 45.0, -45.0, 60.0, -60.0});
    check_right(scene, model, shaken(scene.walk), std::string(scene.name) + ", shaken");
}

/// The right scenes, from their walks as built, rolled and shaken: right every time.
/// @param scenes [in] the scenes, each with its object.
void check_right_scenes(const std::vector<test_scene> &scenes)
{
    std::uint64_t seed = 1;
    for (const test_scene &scene : scenes)
    {
        const built_model model = build(scene, seed++);
        std::printf("%s: %zu points, %zu poses\n", scene.name, model.points.size(),
                    scene.walk.size());
        const gca::separation_proposal proposal = check_right(scene, model, scene.walk, scene.name);
        if (!scene.planes.empty() && proposal.proposed())
        {
            const std::size_t support = part_points_within(
                model, part::support, proposal.meeting_point_m, proposal.window_m);
            const std::size_t other =
                part_points_within(model, part::other, proposal.meeting_point_m, proposal.window_m);
            std::printf(
                "%s: in the window the support plane holds %zu points, the other plane %zu\n",
                scene.name, support, other);
            check(other > support,
                  (std::string(scene.name) + ": the other plane holds more of the window's points")
                      .c_str());
        }
        check_rolls(scene, model, {15.0, 30.0, 40.0, 45.0, 60.0});
        check_right(scene, model, shaken(scene.walk), std::string(scene.name) + ", shaken");
    }
}

/// Two things nearer than a footprint's cell c join, two farther apart than 2 sqrt 2 c do not
/// (docs/methods/separation.md, "What stands on the plane"): a box beside the object, 3 voxels and
/// 18 voxels from it, for both object classes.
void check_neighbours()
{
    struct neighbour_case
    {
        test_scene scene; ///< the object on its plane, walked around
        double side_m;    ///< the neighbour's level edge, in m
        double height_m;  ///< the neighbour's height, in m
    };
    test_scene tabletop = tabletop_scene();
    tabletop.support = floor_at(Eigen::Vector3d::Zero(), 0.3, 0.3);
    tabletop.other_boxes.clear();
    tabletop.balls.clear();
    test_scene large = large_scene();
    large.support = floor_at(Eigen::Vector3d::Zero(), 1.5, 1.5);
    large.other_boxes.clear();
    large.balls.clear();
    std::uint64_t seed = 51;
    for (neighbour_case c : {neighbour_case{tabletop, 0.06, 0.12}, neighbour_case{large, 0.4, 0.8}})
        for (const double gap_voxels : {3.0, 18.0})
        {
            const double v = gca::parameters_of(c.scene.kind).full_voxel_size_m;
            c.scene.other_boxes = {
                beside(*c.scene.object, 0.0, gap_voxels * v, c.side_m, c.height_m)};
            const std::string what = std::string(c.scene.name) + ", a box " +
                                     std::to_string(static_cast<int>(gap_voxels)) +
                                     " voxels beside it";
            const built_model model = build(c.scene, seed++);
            if (gap_voxels >= gca::separation_cell_voxels)
            {
                check_right(c.scene, model, c.scene.walk, what);
                continue;
            }
            const auto [scan, gravity] = checked_gravity(model, c.scene.walk, what);
            const gca::separation_proposal proposal =
                gca::propose_separation(model.points, scan.poses, gravity, model.parameters);
            std::size_t neighbour = 0;
            std::size_t held = 0;
            for (std::size_t k = 0; k < model.points.size(); k++)
                if (model.parts[k] == part::other)
                {
                    neighbour++;
                    held += proposal.box.holds(model.points[k].position_m.cast<double>()) ? 1 : 0;
                }
            std::printf("%s: the box holds %zu of its %zu points\n", what.c_str(), held, neighbour);
            check(proposal.proposed() && held == neighbour,
                  (what + ": joined, the box holds it").c_str());
        }
}

/// Proposes for a scene without its object, gravity from its walk.
/// @param scene [in] the scene.
/// @param model [in] its model.
/// @return the proposal.
gca::separation_proposal proposal_of(const test_scene &scene, const built_model &model)
{
    const auto [scan, gravity] = checked_gravity(model, scene.walk, scene.name);
    return gca::propose_separation(model.points, scan.poses, gravity, model.parameters);
}

/// A ramp alone, 0.8 m square and steeper than the plane's cone, 55 degrees, walked around: no
/// plane, though planes through it at a slant pass the cone.
void check_ramp()
{
    test_scene ramp = tabletop_scene();
    ramp.name = "a steep ramp alone";
    ramp.support.half_u_m = 0.0;
    ramp.object.reset();
    ramp.other_boxes.clear();
    ramp.balls.clear();
    const double tilt = 55.0 * std::numbers::pi / 180.0;
    ramp.planes = {{Eigen::Vector3d(0.03, -0.02, 0.0), Eigen::Vector3d::UnitX(),
                    Eigen::Vector3d(0.0, std::cos(tilt), std::sin(tilt)), 0.4, 0.4,
                    synthetic_texture::plane}};
    const built_model model = build(ramp, 61);
    const gca::separation_proposal proposal = proposal_of(ramp, model);
    std::printf("%s: %zu points: %s\n", ramp.name, model.points.size(),
                proposal.proposed() ? "proposed" : proposal.message.c_str());
    check(proposal.end == gca::separation_end::no_plane, "a steep ramp alone: no plane");
}

/// The board: the table, not the board's top, with the views meeting under the top.
void check_board()
{
    const test_scene scene = board_scene();
    const built_model model = build(scene, 11);
    std::printf("%s: %zu points, %zu poses\n", scene.name, model.points.size(), scene.walk.size());
    const gca::separation_proposal proposal = check_right(scene, model, scene.walk, scene.name);
    const std::size_t table =
        part_points_within(model, part::support, proposal.meeting_point_m, proposal.window_m);
    const std::size_t board =
        part_points_within(model, part::object, proposal.meeting_point_m, proposal.window_m);
    std::printf("%s: in the window the table holds %zu points, the board %zu\n", scene.name, table,
                board);
    check(board > table, "a flat board: the board holds more of the window's points");
}

/// Models without a plane, and a plane with nothing on it.
void check_without_proposal()
{
    test_scene sphere = tabletop_scene();
    sphere.name = "a sphere alone";
    sphere.support.half_u_m = 0.0;
    sphere.object.reset();
    sphere.other_boxes.clear();
    sphere.balls = {{Eigen::Vector3d(0.03, -0.02, 0.10), 0.10}};
    const built_model alone = build(sphere, 21);
    const gca::separation_proposal no_plane = proposal_of(sphere, alone);
    std::printf("%s: %zu points: %s\n", sphere.name, alone.points.size(), no_plane.message.c_str());
    check(no_plane.end == gca::separation_end::no_plane, "a sphere alone: no plane");
    const gca::separation_proposal empty = gca::propose_separation(
        {}, trajectory_of(alone, sphere.walk), estimate_of(known_up(alone)), alone.parameters);
    std::printf("an empty model: %s\n", empty.message.c_str());
    check(empty.end == gca::separation_end::no_plane, "an empty model: no plane");

    test_scene bare = tabletop_scene();
    bare.name = "the table alone";
    bare.object.reset();
    bare.other_boxes.clear();
    bare.balls.clear();
    const built_model table = build(bare, 22);
    const gca::separation_proposal nothing = proposal_of(bare, table);
    std::printf("%s: %s\n", bare.name, nothing.message.c_str());
    check(nothing.end == gca::separation_end::nothing_on_plane, "the table alone: nothing on it");
    check(nothing.plane_points >= gca::separation_least_plane_points,
          "the table alone: the plane's points given");
}

/// Trajectories whose views do not meet, and gravity without up before them.
void check_views_and_no_gravity()
{
    const built_model model = build(tabletop_scene(), 31);
    std::vector<Eigen::Isometry3d> sweep;
    for (int k = 0; k < 12; k++)
    {
        const Eigen::Vector3d position(-0.3 + 0.05 * k, -0.45, 0.35);
        const std::optional<Eigen::Isometry3d> pose =
            gca_test::look_at(position, position + Eigen::Vector3d(0.0, 0.45, -0.3), 0.0);
        check(pose.has_value(), "the sweep's pose is built");
        if (pose)
            sweep.push_back(*pose);
    }
    const gca::separation_proposal parallel = gca::propose_separation(
        model.points, trajectory_of(model, sweep), estimate_of(known_up(model)), model.parameters);
    std::printf("a sweep of parallel views: %s\n", parallel.message.c_str());
    check(parallel.end == gca::separation_end::views_do_not_meet &&
              parallel.message.find("optical axes do not meet") != std::string::npos,
          "a sweep of parallel views: the views do not meet");

    // Each outcome without up ends the proposal without gravity, before the views.
    for (int k = 0; k <= static_cast<int>(gca::gravity_outcome::refused); k++)
    {
        gca::gravity_estimate none;
        none.outcome = static_cast<gca::gravity_outcome>(k);
        if (none.outcome == gca::gravity_outcome::estimated)
            continue;
        none.samples = 900;
        none.groups = 4;
        const gca::separation_proposal without = gca::propose_separation(
            model.points, trajectory_of(model, sweep), none, model.parameters);
        std::printf("outcome %d: %s\n", k, without.message.c_str());
        check(
            without.end == gca::separation_end::no_gravity &&
                without.message == gca::reason_of(none) &&
                without.message.find("no gravity: ") == 0 &&
                without.meeting_point_m == Eigen::Vector3d::Zero(),
            ("gravity's outcome " + std::to_string(k) + ": no gravity, before the views").c_str());
    }
}

/// Refusals as values, each with the reason, and a quaternion within the tolerance taken.
void check_refusals()
{
    const test_scene scene = tabletop_scene();
    const built_model model = build(scene, 41);
    const std::vector<gca::camera_pose> trajectory = trajectory_of(model, scene.walk);
    const gca::gravity_estimate up = estimate_of(known_up(model));
    const gca::separation_parameters good = model.parameters;
    struct parameter_case
    {
        const char *what;
        gca::separation_parameters parameters;
        const char *expected;
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const parameter_case parameter_cases[] = {
        {"a voxel size of 0", {0.0, good.max_depth_m}, "the voxel size 0 m is not"},
        {"a voxel size not a number", {nan, good.max_depth_m}, "the voxel size nan m is not"},
        {"a farthest depth of 0", {good.voxel_size_m, 0.0}, "the farthest depth 0 m is not"},
        {"a farthest depth infinite", {good.voxel_size_m, inf}, "the farthest depth inf m is not"},
        {"a farthest depth beyond 10^5 voxels",
         {good.voxel_size_m, 1.0001e5 * good.voxel_size_m},
         "voxels of 0.001 m"},
    };
    for (const parameter_case &c : parameter_cases)
    {
        const gca::separation_proposal p =
            gca::propose_separation(model.points, trajectory, up, c.parameters);
        check_refusal(c.what, p.end == gca::separation_end::refused, p.message,
                      p.message.find(c.expected) != std::string::npos);
    }
    const gca::separation_proposal at_largest = gca::propose_separation(
        model.points, trajectory, up,
        {good.voxel_size_m, gca::separation_largest_depth_voxels * good.voxel_size_m});
    check(at_largest.end != gca::separation_end::refused,
          "a farthest depth of 10^5 voxels is taken");

    gca::gravity_estimate odd = up;
    odd.outcome = static_cast<gca::gravity_outcome>(99);
    const gca::separation_proposal no_outcome =
        gca::propose_separation(model.points, trajectory, odd, good);
    check_refusal("gravity's outcome 99", no_outcome.end == gca::separation_end::refused,
                  no_outcome.message, no_outcome.message == "gravity's outcome 99 is no outcome");
    odd = up;
    odd.up *= 1.0 + 2e-9;
    const gca::separation_proposal long_up =
        gca::propose_separation(model.points, trajectory, odd, good);
    check_refusal("gravity's up of length 1 + 2e-9", long_up.end == gca::separation_end::refused,
                  long_up.message, long_up.message.find("gravity's up (") == 0);
    odd.up = up.up * (1.0 + 0.5e-9);
    check(gca::propose_separation(model.points, trajectory, odd, good).proposed(),
          "gravity's up of length 1 + 0.5e-9 is taken");

    const gca::separation_proposal no_pose = gca::propose_separation(model.points, {}, up, good);
    check_refusal("no pose", no_pose.end == gca::separation_end::refused, no_pose.message,
                  no_pose.message == "the trajectory holds no pose");
    std::vector<gca::camera_pose> poses = trajectory;
    poses[3].position_m.y() = nan;
    const gca::separation_proposal not_finite =
        gca::propose_separation(model.points, poses, up, good);
    check_refusal("a pose not finite", not_finite.end == gca::separation_end::refused,
                  not_finite.message, not_finite.message == "pose 3: its position is not finite");
    poses = trajectory;
    poses[5].orientation.coeffs() *= 1.0 + 2e-9;
    const gca::separation_proposal not_unit =
        gca::propose_separation(model.points, poses, up, good);
    check_refusal("a quaternion of norm 1 + 2e-9", not_unit.end == gca::separation_end::refused,
                  not_unit.message,
                  not_unit.message.find("pose 5: its quaternion's norm differs from 1 by 2") == 0);
    poses = trajectory;
    poses[5].orientation.coeffs() *= 1.0 + 0.5e-9;
    check(gca::propose_separation(model.points, poses, up, good).proposed(),
          "a quaternion of norm 1 + 0.5e-9 is taken");
    std::vector<gca::model_point> points = model.points;
    points[7].position_m.z() = static_cast<float>(nan);
    const gca::separation_proposal bad_point =
        gca::propose_separation(points, trajectory, up, good);
    check_refusal("a point not finite", bad_point.end == gca::separation_end::refused,
                  bad_point.message, bad_point.message == "point 7: its position is not finite");
}
} // namespace

int main()
{
    check_plane_fit();
    check_gravity();
    check_right_scenes(
        {tabletop_scene(), large_scene(), tabletop_wall_scene(), large_wall_scene()});
    check_corner();
    check_neighbours();
    check_ramp();
    check_board();
    check_without_proposal();
    check_views_and_no_gravity();
    check_refusals();
    return gca_test::check_summary("test_separation");
}
