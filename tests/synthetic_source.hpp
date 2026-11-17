// SPDX-License-Identifier: 0BSD
/// @file synthetic_source.hpp
/// The tests' synthetic source: a frame source (camera/frame_source.hpp) that renders a known scene
/// by the one pinhole convention (types/camera_param.hpp) and tells a test what it rendered. The
/// scene is a support plane, textured or grey, or none, a textured box of known size standing on
/// it or none, a textured sphere of known centre and radius or none, and a known trajectory: frame
/// i of an opening is seen from the trajectory's pose i. The depth may carry Gaussian noise that
/// grows with the square of the depth, from a fixed seed, and a run of frames may come without
/// depth. The source tells a test the trajectory and the known
/// surfaces, textured rectangles in the world frame, and the sphere, which are exactly what it
/// renders, the pinhole ray of any point of the picture, where any ray first meets them, the hit it
/// renders each pixel from, and how far a point lies from a known surface. It also builds the
/// trajectories its scenes are seen along: a camera's upright pose looking at a point, and a walk
/// once around a point, each refused as a value where the view is not defined. This is the one
/// renderer of the tests' frames: a test whose scene it lacks adds the scene here as one more
/// option. A test builds the source to deliver its trajectory's frames and then to report the end,
/// a timeout or a failure, or to refuse its opening, each with the message it is given. It
/// samples an IMU along the trajectory when a test asks for one. No camera and no OrbbecSDK; host
/// only.

#pragma once

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "types/camera_param.hpp"
#include "types/frame_description.hpp"
#include "types/imu_sample.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace gca_test
{
/// The textures of the scene's surfaces.
enum class synthetic_texture
{
    /// The support plane's: three sine waves in the plane, one per channel, along x, along y and
    /// diagonally, each with its own wavelength, so that the color changes along every direction
    /// of the plane and has a gradient everywhere [?: a choice].
    plane,
    /// The box's and the sphere's, painted through them like a solid: three sine waves in space,
    /// one per channel, along directions that each rise or fall by 35 to 50 degrees, with
    /// wavelengths and phases other than the plane's [?: a choice; the tables in
    /// synthetic_source.cpp]. On a face of the box of any turn every channel changes along the
    /// face, by at least 0.57 of its wave's slope [S: computed, sin 35 degrees on a side, cos 50
    /// degrees on the top].
    box,
    /// None: one grey, 128 in each channel, without a gradient, a surface whose color holds no
    /// motion [?: a choice, the grey of stage_frames.hpp's built frames].
    grey,
};

/// The support plane: the world's plane z = 0, the rectangle |x| <= half_size_x_m,
/// |y| <= half_size_y_m around the world's origin, seen from either side.
struct synthetic_plane
{
    double half_size_x_m; ///< half the rectangle's extent along the world's x axis, in m, > 0
    double half_size_y_m; ///< half the rectangle's extent along the world's y axis, in m, > 0
    /// Its texture: synthetic_texture::plane, or synthetic_texture::grey for a plane without
    /// texture.
    synthetic_texture texture = synthetic_texture::plane;
};

/// A box standing on the support plane, textured by synthetic_texture::box. Its own axes are the
/// world's x and y turned by yaw_deg about the world's z axis; its edges lie along them and along
/// the world's z. Its bottom face lies in the plane and is not rendered: the plane hides it.
struct synthetic_box
{
    double centre_x_m; ///< its footprint's centre in the plane: x in the world frame, in m
    double centre_y_m; ///< its footprint's centre in the plane: y in the world frame, in m
    double yaw_deg;    ///< the turn of its axes about the world's z axis, from x toward y, in deg
    double size_x_m;   ///< its edge length along its own x axis, in m, > 0
    double size_y_m;   ///< its edge length along its own y axis, in m, > 0
    double height_m;   ///< its edge length along the world's z axis, up from the plane, in m, > 0
};

/// A sphere, textured by synthetic_texture::box, seen from either side: from inside it the
/// camera sees its inner surface. Where it meets the plane or the box, the nearer surface along a
/// ray hides the other.
struct synthetic_sphere
{
    Eigen::Vector3d centre_m; ///< its centre, in the world frame, in m
    double radius_m;          ///< its radius, in m, > 0
};

/// The scene the synthetic source renders. The world frame is the scene's own, in m: x and y in
/// the support plane, z up from it. A test builds it with designated initializers, so that a
/// scene it does not name is none.
struct synthetic_scene
{
    std::optional<synthetic_plane> plane = std::nullopt;   ///< the support plane, or none
    std::optional<synthetic_box> box = std::nullopt;       ///< the box on the plane, or none
    std::optional<synthetic_sphere> sphere = std::nullopt; ///< the sphere, or none
    /// The trajectory: pose i maps a point from the camera frame of frame i of an opening (x
    /// right, y down, z forward, in m) into the world frame. The source delivers one frame for
    /// each pose.
    std::vector<Eigen::Isometry3d> trajectory;
};

/// Returns a texture's color at a point of a surface it covers.
/// @param texture [in] the texture.
/// @param point_m [in] the point, in the world frame, in m.
/// @return red, green and blue, unrounded, each within 28 to 228 of the 8-bit range [S: the
///         waves' mean 128 and amplitude 100, synthetic_source.cpp]; 128 each for the grey.
std::array<double, 3> texture_color(synthetic_texture texture, const Eigen::Vector3d &point_m);

/// A known surface of the scene: a textured rectangle, seen from either side.
struct synthetic_surface
{
    Eigen::Vector3d centre_m; ///< its centre, in the world frame, in m
    Eigen::Vector3d axis_u;   ///< its first axis, a unit vector in the world frame
    /// Its second axis, a unit vector in the world frame, perpendicular to axis_u. axis_u x axis_v
    /// is the surface's normal: up for the plane, out of the box for a face of the box.
    Eigen::Vector3d axis_v;
    double half_u_m;           ///< half its extent along axis_u, in m, > 0
    double half_v_m;           ///< half its extent along axis_v, in m, > 0
    synthetic_texture texture; ///< its texture
};

/// Returns the ray of a point of the picture by the one pinhole convention
/// (types/camera_param.hpp: pixel centres at whole numbers), the ray the synthetic source renders
/// each pixel along.
/// @param k [in] the intrinsics, in px.
/// @param u [in] the column, in px, any real value: between pixels too.
/// @param v [in] the row, in px, any real value.
/// @return ((u - cx) / fx, (v - cy) / fy, 1), in the camera frame.
Eigen::Vector3d pixel_ray(const gca::intrinsics &k, double u, double v);

/// Returns a point's distance from a known surface, the nearest point of its rectangle.
/// @param surface [in] the surface, in the world frame.
/// @param world [in] the point, in the world frame, in m.
/// @return the distance, in m.
double distance_from(const synthetic_surface &surface, const Eigen::Vector3d &world);

/// Gives one direction of a Fibonacci lattice on the unit sphere, on which the tests sample a
/// sphere's surface at about even spacing [S: computed: heights evenly spaced, the golden angle
/// between consecutive points].
/// @param k [in] the direction's index, 0 to count - 1.
/// @param count [in] the lattice's directions, at least 1.
/// @return the unit direction: z = 1 - 2 (k + 0.5) / count, turned by k golden angles about z.
Eigen::Vector3d sphere_lattice_direction(std::size_t k, std::size_t count);

/// Where a ray first meets a scene in front of its origin.
struct synthetic_hit
{
    double s;                  ///< the ray's parameter at the hit: the hit is origin + s ray
    std::size_t surface;       ///< the surface met: its index in the known surfaces, or their
                               ///< count for the sphere
    synthetic_texture texture; ///< the texture of the surface met
};

/// Finds where a ray first meets the known surfaces and the sphere in front of its origin, as the
/// synthetic source renders each pixel: the hit of the smallest s > 0, the earlier surface on a tie
/// and a surface before the sphere.
/// @param surfaces [in] the known surfaces, in the world frame (synthetic_source::surfaces()).
/// @param sphere [in] the sphere, in the world frame, or none.
/// @param origin [in] the ray's origin, the camera's centre in the world frame, in m.
/// @param ray [in] the ray's direction in the world frame, not zero; not unit: for a camera's ray
///        whose z in the camera frame is 1, s is the hit's depth along the camera's z, in m.
/// @return the hit; nothing when the ray meets no surface in front of its origin.
std::optional<synthetic_hit> first_hit(const std::vector<synthetic_surface> &surfaces,
                                       const std::optional<synthetic_sphere> &sphere,
                                       const Eigen::Vector3d &origin, const Eigen::Vector3d &ray);

/// Returns a camera's pose looking from a position toward a target, upright, then rolled about
/// its viewing axis. Before the roll its z axis points to the target, its x axis is level and to
/// the right of the view (z x up), and its y axis, z x x, points down in the world.
/// @param position [in] the camera's position in the world frame, in m.
/// @param target [in] the point it looks at, in the world frame, in m.
/// @param roll_deg [in] the roll about the viewing axis, in degrees, positive from the camera's x
///        toward its y; within +-90 degrees its y axis still points down in the world and its x
///        axis to the right of the view.
/// @return the pose, camera frame to world frame. Nothing when the target lies within 1e-9 m of
///         the position [?: a choice, far below a scene's size], when the view lies within 1e-4
///         rad of straight up or down, where the level x axis is not defined [?: a choice: there
///         the inputs' rounding, 2.2e-16 of them, turns the x axis by up to 2.2e-12 rad
///         (S: computed)], or when an input is not finite.
std::optional<Eigen::Isometry3d> look_at(const Eigen::Vector3d &position,
                                         const Eigen::Vector3d &target, double roll_deg);

/// A walk once around a point, as a hand-held camera goes around an object: pose k of
/// pose_count stands at the azimuth 360 k / pose_count degrees around the centre, from the
/// world's x axis toward its y, and looks toward the centre (look_at). Its distance from the
/// centre, its height, its roll and the point it looks at swing along the walk, each at its own
/// rate, so that no two views differ by a turn about the centre alone [?: a choice]:
/// distance radius_m + radius_swing_m sin 2a, height height_m + height_swing_m cos 3a, roll
/// roll_swing_deg sin 5a, the point looked at the centre moved by target_jitter_m
/// (sin 3a, cos 5a, 0), at the pose's azimuth a. Every field is the test's to set: a walk around
/// a tabletop object and one around a large object (GLOSSARY.md, "Object class") take other
/// values.
struct synthetic_walk
{
    Eigen::Vector3d centre_m; ///< the point walked around, in the world frame, in m
    std::size_t pose_count;   ///< the poses of the walk, >= 0
    /// The camera's mean distance from the centre along the plane, in m, > 0. Above
    /// radius_swing_m + 1.5 target_jitter_m it keeps every camera off the vertical through the
    /// point it looks at: that point lies within sqrt(2) target_jitter_m, less than
    /// 1.5 target_jitter_m, of the centre [S: its offset target_jitter_m (sin 3a, cos 5a, 0)], and
    /// the camera lies at least radius_m - radius_swing_m from the centre. walk_around() refuses
    /// a walk with a view nearer the vertical than look_at() allows.
    double radius_m;
    double radius_swing_m;  ///< the amplitude of that distance's swing, in m, >= 0
    double height_m;        ///< the camera's mean height above the plane z = 0, in m
    double height_swing_m;  ///< the amplitude of the height's swing, in m, >= 0
    double roll_swing_deg;  ///< the amplitude of the roll about the viewing axis, in deg, >= 0
    double target_jitter_m; ///< how far the point looked at moves along x and along y, in m, >= 0
};

/// Builds the trajectory of a walk once around a point.
/// @param walk [in] the walk.
/// @return pose_count poses, camera frame to world frame, in the walk's order; nothing when
///         look_at() refuses one of them.
std::optional<std::vector<Eigen::Isometry3d>> walk_around(const synthetic_walk &walk);

/// The tabletop scene the offline tests share, its box: 0.20 m x 0.14 m x 0.10 m, a tabletop
/// object of 10 cm to 50 cm (GLOSSARY.md), with three edge lengths, off the plane's centre and
/// turned by 25 degrees against the world's axes, so that a swap of two edges or of two axes
/// shows [?: a choice].
inline constexpr synthetic_box tabletop_box = {0.03, -0.02, 25.0, 0.20, 0.14, 0.10};

/// The tabletop scene's support plane under the box: 0.9 m x 0.8 m [?: a choice], around the box
/// in every view of tabletop_walk(), whose views also look past its edges [M 2026-11-14: in the
/// synthetic source's test, 160x120 px, 3732 to 5919 pixels of each of the walk's frames see
/// nothing].
inline constexpr synthetic_plane tabletop_plane = {0.45, 0.40};

/// The walk once around the tabletop box: 12 poses [?: a choice], 0.40 m to 0.50 m from the box's
/// centre along the plane, 0.30 m to 0.40 m above the plane, rolled by up to 10 degrees, looking
/// at points within 2 cm of the box's centre along x and y [?: a choice, a hand-held walk around
/// a tabletop box].
/// @return the walk, for walk_around().
synthetic_walk tabletop_walk();

/// The time between two frames of the synthetic source, in µs: 30 fps [?: a choice, the camera's
/// rate].
inline constexpr std::uint64_t synthetic_frame_period_us = 33333;

/// What the synthetic source reports once it has delivered its frames.
enum class synthetic_ending
{
    end,     ///< next_status::end, as a sequence at its end
    timeout, ///< next_status::timeout, as a camera that sends nothing
    failure, ///< next_status::failure, as a camera that fails
};

/// The depth noise the synthetic source adds to every pixel's depth before it rounds it to depth
/// values: Gaussian, of mean 0 and standard deviation sigma(z) = c z^2 at the true depth z, the law
/// of a stereo camera's noise (docs/methods/sparse-tsdf.md, "The truncation distance"), drawn from
/// a generator of a fixed seed [?: a choice: independent at every pixel, which the camera's noise
/// need not be].
struct synthetic_depth_noise
{
    double noise_per_m; ///< c, in 1/m, > 0
    /// The seed: frame i of an opening draws from std::mt19937_64 seeded with (seed, i) through
    /// std::seed_seq, one draw of std::normal_distribution for each pixel in the frame's order, so
    /// that a frame's noise depends on the seed and its index alone. The draws are those of this
    /// tree's standard library, the same in every run.
    std::uint32_t seed;
};

/// A run of frames without depth, as a camera whose depth stream gives none for a while: frames
/// first_frame to first_frame + frame_count - 1 of an opening have depth 0 at every pixel, their
/// color rendered as without the run.
struct synthetic_depth_gap
{
    std::uint64_t first_frame; ///< the index of the run's first frame
    std::uint64_t frame_count; ///< the frames of the run, >= 0
};

/// The standard acceleration of gravity, in m/s² [S: CGPM 1901], which the synthetic source's
/// accelerometer reads.
inline constexpr double standard_gravity_mps2 = 9.80665;

/// The IMU the synthetic source samples along its trajectory, in the camera frame, so that a test
/// knows the truth (docs/methods/imu-samples.md, "The synthetic source"): between frames i - 1 and
/// i the camera turns at the constant rate w_i = log(R_{i-1}^T R_i) / (t_i - t_{i-1}), in the
/// camera frame, R_i the rotation of the trajectory's pose i, t_i frame i's timestamp, and at a
/// time t of (t_{i-1}, t_i] its rotation is R_{i-1} exp((t - t_{i-1}) w_i). The gyroscope reads
/// w_i in rad/s, the accelerometer the specific force R(t)^T (a_i + (0, 0, standard_gravity_mps2))
/// in m/s², the world's z up, a_i the hand's acceleration when asked for, else 0. At frame 0's time
/// 0 the camera rests at pose 0. Frame i carries the samples taken in (t_{i-1}, t_i], frame 0 the
/// one at 0, each value rounded to float.
struct synthetic_imu
{
    /// The time between two samples of each sensor, in µs, > 0: sample k is taken at k period_us,
    /// from 0 on, the accelerometer's and the gyroscope's at the same times.
    std::uint64_t period_us;
    /// Whether the accelerometer reads the hand's acceleration too: for the samples of frame i,
    /// a_i = (p_{i+1} - 2 p_i + p_{i-1}) / Δt², the second difference of the trajectory's
    /// positions p at the frames' period Δt, in the world frame, the camera resting at pose 0
    /// before frame 0 and at the last pose after the last frame (p_{-1} = p_0, p_N = p_{N-1}), so
    /// that the scan's accelerations sum to no change of velocity [?: a choice].
    bool hand_acceleration = false;
};

/// Appends the IMU's samples that frame index of the synthetic source carries (synthetic_imu), so
/// that a test takes a trajectory's samples without rendering its frames.
/// @param trajectory [in] the trajectory, camera frame to world frame, in m; holds pose index.
/// @param index [in] the frame's index, its timestamp index * synthetic_frame_period_us.
/// @param imu [in] the IMU, its period > 0.
/// @param accel [in,out] gains the accelerometer's samples, in m/s² in the camera frame, rising
///        in time.
/// @param gyro [in,out] gains the gyroscope's samples, in rad/s in the camera frame, at the same
///        times.
void append_imu_samples(const std::vector<Eigen::Isometry3d> &trajectory, std::uint64_t index,
                        const synthetic_imu &imu, std::vector<gca::imu_sample> &accel,
                        std::vector<gca::imu_sample> &gyro);

/// How a test builds the synthetic source.
struct synthetic_source_options
{
    /// The size, intrinsics and depth unit of every frame; the only size the source opens for.
    gca::frame_description description;
    synthetic_scene scene; ///< what the frames show, one frame for each pose of its trajectory
    /// The depth noise, or none: then each pixel's depth is the hit's, rounded.
    std::optional<synthetic_depth_noise> depth_noise = std::nullopt;
    /// A run of frames without depth, or none.
    std::optional<synthetic_depth_gap> depth_gap = std::nullopt;
    synthetic_ending ending = synthetic_ending::end; ///< what next() reports after the frames
    std::string ending_message;                      ///< the message next() reports with it
    std::string open_refusal; ///< when not empty, open() refuses with this reason
    /// The IMU sampled along the trajectory, or none: then the frames carry no samples.
    std::optional<synthetic_imu> imu = std::nullopt;
};

/// A frame source that renders the scene of its options along its trajectory.
class synthetic_source final : public gca::frame_source
{
public:
    /// Builds the source, closed, and the scene's known surfaces.
    /// @param options [in] the frames, the scene and the ending; taken over.
    explicit synthetic_source(synthetic_source_options options);

    /// Opens the source; its next frame has index 0.
    /// @param size [in] the size asked for, in px.
    /// @return opened; or refused with open_refusal when the options give one, with both sizes
    ///         when size is not the description's, and when the source is open already.
    gca::open_result open(const gca::frame_size &size) noexcept override;

    /// Renders the next frame: frame i of an opening has index i and timestamp i times
    /// synthetic_frame_period_us and shows the known surfaces and the sphere from the trajectory's
    /// pose i. Each pixel shows the nearest surface its ray meets in front of the camera: depth
    /// is the hit's z in the camera frame, with the options' depth noise added when
    /// they give one, rounded to the nearest depth value; color is the surface's texture at the
    /// hit, rounded. A pixel whose ray meets no surface in front of the camera, or whose depth
    /// rounds below 1 or above the largest 16-bit value, 65535 [S], has depth 0 and color black.
    /// A frame of the options' depth gap then has depth 0 at every pixel, its color kept. With the
    /// options' IMU the frame carries its samples (synthetic_imu), else none. Frames come as fast
    /// as they are asked for.
    /// @param out [out] the frame, filled when the status is frame, else left as it was.
    /// @return frame for the first calls after an opening, one for each pose of the trajectory,
    ///         then the ending with its message at every call; a failure when the source is not
    ///         open.
    gca::next_result next(gca::frame &out) noexcept override;

    /// Closes the source.
    void close() noexcept override;

    /// Describes the source.
    /// @return the kind of source, no firmware and no SDK version; the same before, during and
    ///         after an opening.
    gca::source_description describe() const noexcept override;

    /// Tells a test what the source renders.
    /// @return the scene: the plane, the box and the trajectory, whose pose i is that of frame i
    ///         of an opening; valid as long as the source.
    const synthetic_scene &scene() const noexcept;

    /// Tells a test the known surfaces, the rectangles every frame is rendered from; the sphere,
    /// the one surface that is not a rectangle, is the scene's (scene()).
    /// @return the plane first when the scene has one, then, when the scene has a box, its top and
    ///         its sides on its own +x, +y, -x and -y; valid as long as the source.
    const std::vector<synthetic_surface> &surfaces() const noexcept;

private:
    synthetic_source_options m_options;
    std::vector<synthetic_surface> m_surfaces; ///< the scene's known surfaces
    bool m_open = false;
    std::uint64_t m_next_index = 0; ///< the index of the next frame of this opening
};
} // namespace gca_test
