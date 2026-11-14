// SPDX-License-Identifier: 0BSD
/// @file test_synthetic_source.cpp
/// The tests' one renderer, the synthetic source, against the test's own geometry. What a pixel
/// should hold is computed here by the pinhole formula on paths of the test's own, apart from the
/// renderer's code, from the known surfaces and the frame's pose:
/// - the plane alone, every pixel: with a depth unit that holds the whole plane, with one so fine
///   that part of it lies beyond the largest depth value, and with the camera looking away from
///   the plane, which then lies behind it;
/// - the plane seen from beside the box, looking away from it, with a face of the box behind the
///   camera on every pixel's ray: every pixel shows the plane;
/// - a walk once around a box standing on the plane: frame i is seen from the trajectory's pose
///   i, and every pixel the test names holds the depth and the color that the known box, plane,
///   textures and that pose give. The test projects its own box's corners forward into each
///   frame by the pinhole formula and names each pixel inside a face's outline, or outside the
///   box's silhouette on the plane, on the plane just beside the silhouette, or seeing nothing.
///   The source tells the trajectory and the known surfaces, which are checked against the
///   test's own box and walk;
/// - look_at() and walk_around() give upright poses, and refuse as values a view that is not
///   defined.

#include "camera/frame.hpp"
#include "check.hpp"
#include "synthetic_source.hpp"
#include "types/frame_description.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::check;
using gca_test::check_max_error;
using gca_test::check_within;
using gca_test::max_error;

/// The frames' size: small, so that the test checks every pixel at once [?: a choice].
constexpr gca::frame_size test_size = {160, 120};

/// The intrinsics: fx and fy differ, and the principal point lies off the picture's centre at
/// no whole or half pixel, so that a swap or a shift of either shows [?: a choice].
constexpr gca::intrinsics test_intrinsics = {150.0f, 140.0f, 83.3f, 56.6f};

/// The depth unit that holds the whole plane: 0.1 mm, one of the Gemini 2 L's depth precisions
/// [?: a choice]. On this scene a depth step of half a pixel is 0.5 mm near the picture's centre
/// [S: computed, 2026-11-14], five depth units.
constexpr float fine_depth_unit_m = 0.0001f;

/// The depth unit so fine that the plane's far part has no depth: the largest 16-bit value,
/// 65535 units, is 0.655 m, and the plane's depths run from 0.40 m to 0.80 m [S: computed,
/// 2026-11-14].
constexpr float finest_depth_unit_m = 0.00001f;

/// The tolerances of a pixel's depth, in depth units, and of its color, in 8-bit levels: half a
/// step from rounding to the nearest value [S], and 1e-6 for a value that lies within the two
/// paths' double rounding of half a step, about 1e-11 of a step at 65535 depth units [S: double's
/// relative step 2.2e-16]. The largest errors over the pictures and the walk are 0.499990 units
/// and 0.4999999 levels [M 2026-11-14]; rounding down instead gives errors up to one step.
constexpr double depth_tolerance_units = 0.5 + 1e-6;
constexpr double color_tolerance_levels = 0.5 + 1e-6;

/// The tolerance of the known surfaces' corners (in m), axes and normals against the test's own
/// box and plane, and of the poses' rotations off proper: about 5000 steps of double at 1 m
/// [S: double's relative step 2.2e-16], for sums of a few terms below 1. The largest are
/// 3.1e-17 m for the corners, 2e-16 for the axes and normals and 6.3e-16 for the rotations
/// [M 2026-11-14].
constexpr double geometry_tolerance = 1e-12;

/// The support plane of the pictures of the plane alone: 0.6 m x 0.5 m [?: a choice], of which
/// the picture shows part; the upper rows look past its far edge [S: computed, 2026-11-14].
constexpr gca_test::synthetic_plane test_plane = {0.30, 0.25};

/// The box: 0.20 m x 0.14 m x 0.10 m, a tabletop object within R1's 10 cm to 50 cm, with three
/// edge lengths, off the plane's centre and turned by 25 degrees against the world's axes, so
/// that a swap of two edges or of two axes shows [?: a choice].
constexpr gca_test::synthetic_box test_box = {0.03, -0.02, 25.0, 0.20, 0.14, 0.10};

/// The support plane under the box: 0.9 m x 0.8 m [?: a choice], around the box in every view of
/// the walk, whose views also look past its edges [M 2026-11-14: 3732 to 5919 pixels of each of
/// the walk's frames see nothing].
constexpr gca_test::synthetic_plane box_plane = {0.45, 0.40};

/// The margin by which a named pixel lies inside a face's outline, or outside the box's
/// silhouette, in the picture, in px [?: a choice]: far above the two paths' rounding, about
/// 1e-13 px [S: double's relative step 2.2e-16 on coordinates below 200 px], and small enough to
/// name the pixels next to the silhouette.
constexpr double outline_margin_px = 0.01;

/// The margin by which the ray of a named pixel outside the silhouette meets the plane inside or
/// outside its rectangle, in m [?: a choice]: about 0.03 px at the walk's distances [S: computed,
/// 1e-4 m at 0.5 m with fx = 150 px], far above the two paths' rounding.
constexpr double plane_margin_m = 1e-4;

/// How near the box's silhouette a pixel on the plane lies when it is just beside it, in px: the
/// pixels next to the silhouette [?: a choice].
constexpr double beside_silhouette_px = 2.0;

/// The message the walk's source ends with.
constexpr const char *walk_ending = "the walk is done";

/// Returns the pose look_at() gives, or the identity with a failed check when it refuses it.
/// @param name [in] the pose, printed when it is refused.
/// @param position [in] the camera's position, in the world frame, in m.
/// @param target [in] the point it looks at, in the world frame, in m.
/// @param roll_deg [in] the roll about the viewing axis, in degrees.
/// @return the pose, camera frame to world frame.
Eigen::Isometry3d pose_looking(const std::string &name, const Eigen::Vector3d &position,
                               const Eigen::Vector3d &target, double roll_deg)
{
    const std::optional<Eigen::Isometry3d> pose = gca_test::look_at(position, target, roll_deg);
    check(pose.has_value(), (name + ": look_at() gives the pose").c_str());
    return pose.value_or(Eigen::Isometry3d::Identity());
}

/// Returns the camera's pose that sees the plane: 0.40 m above it, looking down at about 40
/// degrees from (0.04, -0.45) toward the point (0.02, 0.03) of the plane, rolled by 12 degrees,
/// so that the depth changes along both image axes [?: a choice].
/// @return the pose, camera frame to world frame.
Eigen::Isometry3d seeing_pose()
{
    return pose_looking("the seeing pose", {0.04, -0.45, 0.40}, {0.02, 0.03, 0.0}, 12.0);
}

/// Returns a camera's pose that looks away from the plane: 0.40 m above it, looking up, so that
/// the plane lies behind the camera along most pixels' rays [?: a choice].
/// @return the pose, camera frame to world frame.
Eigen::Isometry3d looking_away_pose()
{
    return pose_looking("the pose looking away", {0.04, -0.05, 0.40}, {0.06, 0.02, 1.40}, 12.0);
}

/// Returns a camera's pose beside the box, looking away from it: 3 cm in front of the box's +x
/// side and 3 cm above the plane, below the box's top, looking along the box's +x axis and 30
/// degrees down, rolled by 6 degrees [?: a choice]. Every pixel's ray meets the plane in front
/// and the +x side behind the camera, and every corner of the box lies behind it [S: computed,
/// 2026-11-14; checked by check_beside_box()].
/// @return the pose, camera frame to world frame.
Eigen::Isometry3d beside_box_pose()
{
    const double yaw_rad = test_box.yaw_deg * M_PI / 180.0;
    const Eigen::Vector3d box_x(std::cos(yaw_rad), std::sin(yaw_rad), 0.0);
    const Eigen::Vector3d position =
        Eigen::Vector3d(test_box.centre_x_m, test_box.centre_y_m, 0.03) +
        (0.5 * test_box.size_x_m + 0.03) * box_x;
    const double down_rad = 30.0 * M_PI / 180.0;
    const Eigen::Vector3d view =
        std::cos(down_rad) * box_x - std::sin(down_rad) * Eigen::Vector3d::UnitZ();
    return pose_looking("the pose beside the box", position, position + 0.1 * view, 6.0);
}

/// Returns the walk once around the box: 12 poses [?: a choice], 0.40 m to 0.50 m from the box's
/// centre along the plane, 0.30 m to 0.40 m above the plane, rolled by up to 10 degrees, looking at
/// points within 2 cm of the box's centre along x and y [?: a choice, a hand-held walk around a
/// tabletop box].
/// @return the walk.
gca_test::synthetic_walk test_walk()
{
    return {Eigen::Vector3d(test_box.centre_x_m, test_box.centre_y_m, 0.5 * test_box.height_m),
            12,
            0.45,
            0.05,
            0.35,
            0.05,
            10.0,
            0.02};
}

/// Checks that cameras' poses are upright as look_at() makes them: each rotation proper
/// (orthonormal, determinant +1) within geometry_tolerance, its y axis pointing down in the world
/// and its x axis to the right of its view, x . (z x up) > 0.
/// @param name [in] the poses, printed in each line.
/// @param poses [in] the poses, camera frame to world frame, rolled by less than 90 degrees.
void check_upright(const std::string &name, const std::vector<Eigen::Isometry3d> &poses)
{
    max_error off_proper;
    int not_upright = 0;
    for (const Eigen::Isometry3d &pose : poses)
    {
        const Eigen::Matrix3d r = pose.linear();
        off_proper.add((r.transpose() * r - Eigen::Matrix3d::Identity()).norm());
        off_proper.add(std::abs(r.determinant() - 1.0));
        const bool y_down = r.col(1).z() < 0.0;
        const bool x_right = r.col(0).dot(r.col(2).cross(Eigen::Vector3d::UnitZ())) > 0.0;
        if (!y_down || !x_right)
            not_upright++;
    }
    check_max_error(name + ": rotations off proper", off_proper, geometry_tolerance);
    check_within((name + ": poses whose y axis is not down or x axis not to the right").c_str(),
                 not_upright, 0);
}

/// Checks that look_at() refuses a view, and prints how it came out.
/// @param what [in] the view, printed.
/// @param position [in] the camera's position, in the world frame, in m.
/// @param target [in] the point it looks at, in the world frame, in m.
void check_look_at_refuses(const std::string &what, const Eigen::Vector3d &position,
                           const Eigen::Vector3d &target)
{
    const bool refused = !gca_test::look_at(position, target, 0.0).has_value();
    std::printf("look_at(), %s: %s\n", what.c_str(), refused ? "refused" : "a pose");
    check(refused, ("look_at() refuses " + what).c_str());
}

/// Checks look_at() and walk_around() where a view is not defined, and the poses the pictures
/// use: a target at the eye or not finite, a view straight down, straight up and 0.5e-4 rad from
/// straight down are refused, below the bound of 1e-4 rad; a view 2e-4 rad from straight down
/// gives an upright pose; a walk straight above its centre is refused; the pictures' poses are
/// upright.
void check_look_at()
{
    const Eigen::Vector3d eye(0.1, -0.2, 0.4);
    const Eigen::Vector3d down = -Eigen::Vector3d::UnitZ();
    check_look_at_refuses("a target at the eye", eye, eye);
    check_look_at_refuses("a target not finite", eye,
                          Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0));
    check_look_at_refuses("a view straight down", eye, eye + 0.3 * down);
    check_look_at_refuses("a view straight up", eye, eye - 0.3 * down);
    check_look_at_refuses("a view 0.5e-4 rad from straight down", eye,
                          eye + 0.3 * Eigen::Vector3d(std::sin(0.5e-4), 0.0, -std::cos(0.5e-4)));
    const Eigen::Isometry3d near_down =
        pose_looking("a view 2e-4 rad from straight down", eye,
                     eye + 0.3 * Eigen::Vector3d(std::sin(2e-4), 0.0, -std::cos(2e-4)), 0.0);
    check_upright("look_at(), a view 2e-4 rad from straight down", {near_down});
    gca_test::synthetic_walk above = test_walk();
    above.radius_m = 0.0;
    above.radius_swing_m = 0.0;
    above.target_jitter_m = 0.0;
    const bool walk_refused = !gca_test::walk_around(above).has_value();
    std::printf("walk_around(), a walk straight above its centre: %s\n",
                walk_refused ? "refused" : "a trajectory");
    check(walk_refused, "walk_around() refuses a walk straight above its centre");
    check_upright("the pictures' poses", {seeing_pose(), looking_away_pose(), beside_box_pose()});
}

/// What a pixel should hold.
struct expected_pixel
{
    bool seen = false;             ///< whether it has depth
    double depth_m = 0.0;          ///< its depth, the camera-frame z of what it sees, in m
    std::array<double, 3> color{}; ///< its color, unrounded
};

/// The pixels of a picture of the plane by what they see.
struct pixel_counts
{
    int seen = 0;         ///< pixels that see the plane within the depth range
    int off_plane = 0;    ///< pixels whose ray, forward or backward, misses the plane's rectangle
    int behind = 0;       ///< pixels whose ray meets the plane behind the camera, or nearer than
                          ///< half a depth unit
    int beyond_range = 0; ///< pixels that see the plane farther than the largest depth value
};

/// Returns the ray of a pixel by the one pinhole convention.
/// @param k [in] the intrinsics.
/// @param u [in] the pixel's column, in px.
/// @param v [in] the pixel's row, in px.
/// @return ((u - cx) / fx, (v - cy) / fy, 1), in the camera frame: its point at parameter z has
///         the camera-frame depth z.
Eigen::Vector3d pixel_ray(const gca::intrinsics &k, int u, int v)
{
    return {(u - double{k.cx}) / k.fx, (v - double{k.cy}) / k.fy, 1.0};
}

/// Where a pixel's ray meets a plane.
struct ray_hit
{
    double z_m;                  ///< the hit's depth, its z in the camera frame, in m
    Eigen::Vector3d point_world; ///< the hit, in the world frame, in m
};

/// Finds where a pixel's ray meets a plane, on a path apart from the renderer's: the plane is
/// taken into the camera frame, n . p = d, and the ray z * ray meets it at z = d / (n . ray).
/// @param normal_world [in] the plane's normal, a unit vector in the world frame.
/// @param point_world [in] a point of the plane, in the world frame, in m.
/// @param camera_to_world [in] the camera's pose, camera frame to world frame.
/// @param ray [in] the pixel's ray (pixel_ray()), in the camera frame.
/// @return the hit; z_m is 0 or less when the plane lies behind the camera along the ray, and
///         infinite or NaN when the ray runs along the plane.
ray_hit hit_plane(const Eigen::Vector3d &normal_world, const Eigen::Vector3d &point_world,
                  const Eigen::Isometry3d &camera_to_world, const Eigen::Vector3d &ray)
{
    const Eigen::Matrix3d camera_from_world = camera_to_world.linear().transpose();
    const Eigen::Vector3d normal_camera = camera_from_world * normal_world;
    const double d =
        normal_camera.dot(camera_from_world * (point_world - camera_to_world.translation()));
    const double z = d / normal_camera.dot(ray);
    return {z, camera_to_world * Eigen::Vector3d(z * ray)};
}

/// Where a depth lies against the depth values a frame holds.
enum class depth_range
{
    below,  ///< it rounds below 1 depth unit, or it is NaN
    within, ///< it rounds to 1 to 65535 depth units
    above,  ///< it rounds above the largest 16-bit value, 65535 [S]
};

/// Returns where a depth lies against the depth values a frame holds.
/// @param z_m [in] the depth, in m.
/// @param description [in] the frame's depth unit.
/// @return below, within or above.
depth_range range_of(double z_m, const gca::frame_description &description)
{
    const double value = std::round(z_m / description.depth_unit_m);
    if (!(value >= 1.0))
        return depth_range::below;
    if (value > std::numeric_limits<std::uint16_t>::max())
        return depth_range::above;
    return depth_range::within;
}

/// Returns what a pixel that sees a surface at a hit should hold: its depth and the texture's
/// color there when the depth lies within the frame's depth values, else no depth.
/// @param hit [in] the hit.
/// @param texture [in] the surface's texture.
/// @param description [in] the frame's depth unit.
/// @return its depth and color, or not seen.
expected_pixel expect_at(const ray_hit &hit, gca_test::synthetic_texture texture,
                         const gca::frame_description &description)
{
    if (range_of(hit.z_m, description) != depth_range::within)
        return {};
    return {true, hit.z_m, gca_test::texture_color(texture, hit.point_world)};
}

/// Computes what a pixel should hold when it can see nothing but the plane, by the one pinhole
/// convention, on a path apart from the renderer's (hit_plane()).
/// @param plane [in] the plane.
/// @param camera_to_world [in] the camera's pose, camera frame to world frame.
/// @param description [in] the frame's intrinsics and depth unit.
/// @param u [in] the pixel's column, in px.
/// @param v [in] the pixel's row, in px.
/// @param counts [in,out] the pixel is counted by what it sees.
/// @return its depth and color, or not seen.
expected_pixel expect_on_plane(const gca_test::synthetic_plane &plane,
                               const Eigen::Isometry3d &camera_to_world,
                               const gca::frame_description &description, int u, int v,
                               pixel_counts &counts)
{
    const ray_hit hit = hit_plane(Eigen::Vector3d::UnitZ(), Eigen::Vector3d::Zero(),
                                  camera_to_world, pixel_ray(description.intrinsics, u, v));
    if (!(std::abs(hit.point_world.x()) <= plane.half_size_x_m &&
          std::abs(hit.point_world.y()) <= plane.half_size_y_m))
    {
        counts.off_plane++;
        return {};
    }
    switch (range_of(hit.z_m, description))
    {
    case depth_range::below:
        counts.behind++;
        return {};
    case depth_range::above:
        counts.beyond_range++;
        return {};
    case depth_range::within:
        counts.seen++;
        return expect_at(hit, gca_test::synthetic_texture::plane, description);
    }
    // Not reached: the switch names every range, and -Wswitch says when one is added.
    return {};
}

/// The errors of a set of pixels against what they should hold.
struct pixel_errors
{
    max_error depth_units;  ///< the depth errors of pixels with depth that should have it
    max_error color_levels; ///< those pixels' color errors, in 8-bit levels
    int wrongly_shown = 0;  ///< pixels with depth or color that should have neither
    int wrongly_unseen = 0; ///< pixels without depth that should have it
};

/// Compares a pixel of a frame with what it should hold and adds its errors.
/// @param frame [in] the frame.
/// @param u [in] the pixel's column, in px, within the frame.
/// @param v [in] the pixel's row, in px, within the frame.
/// @param expected [in] what it should hold.
/// @param errors [in,out] its errors are added.
void compare_pixel(const gca::frame &frame, int u, int v, const expected_pixel &expected,
                   pixel_errors &errors)
{
    const gca::frame_description &description = frame.description();
    const std::size_t pixel = static_cast<std::size_t>(v) * description.size.width + u;
    const std::uint16_t depth = frame.depth()[pixel];
    const std::uint8_t *color = &frame.color()[3 * pixel];
    if (!expected.seen)
    {
        if (depth != 0 || color[0] != 0 || color[1] != 0 || color[2] != 0)
            errors.wrongly_shown++;
        return;
    }
    if (depth == 0)
    {
        errors.wrongly_unseen++;
        return;
    }
    errors.depth_units.add(std::abs(depth - expected.depth_m / description.depth_unit_m));
    for (int c = 0; c < 3; c++)
        errors.color_levels.add(std::abs(color[c] - expected.color[c]));
}

/// Checks a set of pixels' errors against their tolerances and prints each.
/// @param name [in] the set, printed in each line.
/// @param errors [in] the errors.
void check_errors(const std::string &name, const pixel_errors &errors)
{
    check_within((name + ": pixels with depth or color that should have neither").c_str(),
                 errors.wrongly_shown, 0);
    check_within((name + ": pixels without depth that should have it").c_str(),
                 errors.wrongly_unseen, 0);
    check_max_error(name + ": depth error, in depth units", errors.depth_units,
                    depth_tolerance_units);
    check_max_error(name + ": color error, in levels", errors.color_levels, color_tolerance_levels);
}

/// Builds the options of a source of one picture of the plane alone.
/// @param depth_unit_m [in] the frame's depth unit, in m per value, > 0.
/// @param camera_to_world [in] the camera's pose, camera frame to world frame.
/// @return the options: one frame, then the end.
gca_test::synthetic_source_options picture_options(float depth_unit_m,
                                                   const Eigen::Isometry3d &camera_to_world)
{
    gca_test::synthetic_source_options options;
    options.description = {test_size, test_intrinsics, depth_unit_m};
    options.scene = {test_plane, std::nullopt, {camera_to_world}};
    options.ending = gca_test::synthetic_ending::end;
    options.ending_message = "end";
    return options;
}

/// Renders the one frame of a scene in which the camera can see nothing but the plane, and checks
/// every pixel: a pixel that sees the plane holds its depth and its color within their
/// tolerances, and a pixel that does not holds depth 0 and black.
/// @param name [in] the case, printed in each line.
/// @param options [in] the source's options: one pose, the camera seeing nothing but the plane.
/// @return the pixels by what they should see; all zero when the source gave no frame.
pixel_counts check_picture(const std::string &name,
                           const gca_test::synthetic_source_options &options)
{
    gca_test::synthetic_source source(options);
    check(source.open(test_size).opened, (name + ": opened").c_str());
    gca::frame frame;
    pixel_counts counts;
    if (source.next(frame).status != gca::next_status::frame)
    {
        check(false, (name + ": a frame").c_str());
        return counts;
    }
    const Eigen::Isometry3d &pose = source.scene().trajectory.at(frame.index());
    pixel_errors errors;
    for (int v = 0; v < static_cast<int>(test_size.height); v++)
    {
        for (int u = 0; u < static_cast<int>(test_size.width); u++)
            compare_pixel(
                frame, u, v,
                expect_on_plane(source.scene().plane, pose, frame.description(), u, v, counts),
                errors);
    }
    std::printf("%s: %d pixels see the plane, %d miss it, %d have it behind, %d see it beyond "
                "the depth range\n",
                name.c_str(), counts.seen, counts.off_plane, counts.behind, counts.beyond_range);
    check_errors(name, errors);
    return counts;
}

/// A face of the box as the test builds it.
struct test_face
{
    /// Its corners, in the world frame, in m, counter-clockwise as seen from outside the box.
    std::array<Eigen::Vector3d, 4> corners;
    Eigen::Vector3d outward; ///< its unit normal out of the box, world frame
};

/// Returns the mean of four points.
/// @param points [in] the points.
/// @return their mean, in their frame and unit.
Eigen::Vector3d mean_of(const std::array<Eigen::Vector3d, 4> &points)
{
    return 0.25 * (points[0] + points[1] + points[2] + points[3]);
}

/// Builds the faces of the box that can be seen, on a path of the test's own: its corners, the
/// footprint's centre moved by half of each edge along the box's axes, and each face's outward
/// normal as the direction from the box's middle to the face's.
/// @param box [in] the box.
/// @return the top and the sides on the box's +x, +y, -x and -y: the order of the source's
///         surfaces() after the plane.
std::array<test_face, 5> box_faces(const gca_test::synthetic_box &box)
{
    const double yaw_rad = box.yaw_deg * M_PI / 180.0;
    const Eigen::Vector3d box_x(std::cos(yaw_rad), std::sin(yaw_rad), 0.0);
    const Eigen::Vector3d box_y(-std::sin(yaw_rad), std::cos(yaw_rad), 0.0);
    const Eigen::Vector3d foot(box.centre_x_m, box.centre_y_m, 0.0);
    const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
    // The corner on the side sx (-1 or +1) of the box's x axis and sy of its y, at the bottom
    // (top 0) or the top (top 1).
    const auto corner = [&](double sx, double sy, double top) {
        return Eigen::Vector3d(foot + 0.5 * sx * box.size_x_m * box_x +
                               0.5 * sy * box.size_y_m * box_y + top * box.height_m * up);
    };
    std::array<test_face, 5> faces = {{
        {{corner(-1, -1, 1), corner(1, -1, 1), corner(1, 1, 1), corner(-1, 1, 1)}, {}},
        {{corner(1, -1, 0), corner(1, 1, 0), corner(1, 1, 1), corner(1, -1, 1)}, {}},
        {{corner(1, 1, 0), corner(-1, 1, 0), corner(-1, 1, 1), corner(1, 1, 1)}, {}},
        {{corner(-1, 1, 0), corner(-1, -1, 0), corner(-1, -1, 1), corner(-1, 1, 1)}, {}},
        {{corner(-1, -1, 0), corner(1, -1, 0), corner(1, -1, 1), corner(-1, -1, 1)}, {}},
    }};
    const Eigen::Vector3d middle = foot + 0.5 * box.height_m * up;
    for (test_face &face : faces)
        face.outward = (mean_of(face.corners) - middle).normalized();
    return faces;
}

/// Returns whether a point of a face's plane lies within the face: on the inner side of each
/// edge, the corners running counter-clockwise as seen from outside.
/// @param face [in] the face.
/// @param point [in] the point, in the world frame, in m, in the face's plane.
/// @return whether it lies within the face or on its edge.
bool within_face(const test_face &face, const Eigen::Vector3d &point)
{
    for (std::size_t c = 0; c < 4; c++)
    {
        const Eigen::Vector3d edge = face.corners[(c + 1) % 4] - face.corners[c];
        if (edge.cross(point - face.corners[c]).dot(face.outward) < 0.0)
            return false;
    }
    return true;
}

/// Checks the picture from beside the box, looking away from it: every corner of the box lies
/// behind the camera, so that the camera sees nothing but the plane; every pixel's ray meets a
/// face of the box behind the camera, so that a renderer that took a surface behind the camera
/// would show it; and every pixel shows the plane.
void check_beside_box()
{
    const Eigen::Isometry3d pose = beside_box_pose();
    const std::array<test_face, 5> faces = box_faces(test_box);
    const Eigen::Matrix3d camera_from_world = pose.linear().transpose();
    double nearest_corner_z_m = -std::numeric_limits<double>::infinity();
    for (const test_face &face : faces)
    {
        for (const Eigen::Vector3d &corner : face.corners)
            nearest_corner_z_m = std::max(nearest_corner_z_m,
                                          (camera_from_world * (corner - pose.translation())).z());
    }
    int face_behind = 0;
    for (int v = 0; v < static_cast<int>(test_size.height); v++)
    {
        for (int u = 0; u < static_cast<int>(test_size.width); u++)
        {
            const Eigen::Vector3d ray = pixel_ray(test_intrinsics, u, v);
            const bool behind = std::ranges::any_of(faces, [&](const test_face &face) {
                const ray_hit hit = hit_plane(face.outward, face.corners[0], pose, ray);
                return hit.z_m < 0.0 && within_face(face, hit.point_world);
            });
            if (behind)
                face_behind++;
        }
    }
    gca_test::synthetic_source_options options = picture_options(fine_depth_unit_m, pose);
    options.scene.plane = box_plane;
    options.scene.box = test_box;
    const pixel_counts counts = check_picture("beside the box", options);
    const int pixels = static_cast<int>(test_size.width * test_size.height);
    std::printf("beside the box: the box's corners at camera-frame z %.4f m and less, %d of %d "
                "pixels with a face of the box behind the camera\n",
                nearest_corner_z_m, face_behind, pixels);
    check(nearest_corner_z_m < 0.0, "beside the box: every corner of the box behind the camera");
    check(face_behind == pixels,
          "beside the box: every pixel's ray meets a face of the box behind the camera");
    check(counts.seen == pixels, "beside the box: every pixel sees the plane");
}

/// Checks every pixel of the pictures in which the camera can see nothing but the plane, and
/// that each shows the kinds of pixels it is meant to: the seeing pose at the fine unit sees the
/// plane and misses it; at the finest unit part of the plane lies beyond the depth range; the
/// pose looking away has the plane behind it; beside the box, the box lies behind the camera.
void check_pictures()
{
    const pixel_counts fine =
        check_picture("0.1 mm unit", picture_options(fine_depth_unit_m, seeing_pose()));
    check(fine.seen > 0 && fine.off_plane > 0 && fine.behind == 0 && fine.beyond_range == 0,
          "0.1 mm unit: pixels on and off the plane, none behind or beyond the depth range");
    const pixel_counts finest =
        check_picture("0.01 mm unit", picture_options(finest_depth_unit_m, seeing_pose()));
    check(finest.seen > 0 && finest.beyond_range > 0,
          "0.01 mm unit: pixels on the plane and beyond the depth range");
    const pixel_counts away =
        check_picture("looking away", picture_options(fine_depth_unit_m, looking_away_pose()));
    check(away.behind > 0 && away.seen == 0, "looking away: the plane behind, nothing seen");
    check_beside_box();
}

/// Returns a surface's corners.
/// @param surface [in] the surface.
/// @return its centre moved by +- half its extent along each axis, in order around it, in the
///         world frame, in m.
std::array<Eigen::Vector3d, 4> corners_of(const gca_test::synthetic_surface &surface)
{
    const Eigen::Vector3d u = surface.half_u_m * surface.axis_u;
    const Eigen::Vector3d v = surface.half_v_m * surface.axis_v;
    return {surface.centre_m - u - v, surface.centre_m + u - v, surface.centre_m + u + v,
            surface.centre_m - u + v};
}

/// Returns how far the corners of one set lie from those of another.
/// @param from [in] the first set, in m.
/// @param to [in] the second set, in the same frame, in m.
/// @return the largest distance from a corner of the first set to the nearest of the second, in m.
double farthest_corner_m(const std::array<Eigen::Vector3d, 4> &from,
                         const std::array<Eigen::Vector3d, 4> &to)
{
    double farthest = 0.0;
    for (const Eigen::Vector3d &p : from)
    {
        double nearest = std::numeric_limits<double>::infinity();
        for (const Eigen::Vector3d &q : to)
            nearest = std::min(nearest, (p - q).norm());
        farthest = std::max(farthest, nearest);
    }
    return farthest;
}

/// Checks the source's known surfaces against the test's own plane and box: the plane first, then
/// the box's top and sides in box_faces()' order; each with the expected corners, its normal
/// axis_u x axis_v up or out of the box, perpendicular unit axes, and its texture.
/// @param source [in] the source.
/// @param faces [in] the box's faces as the test builds them.
void check_known_surfaces(const gca_test::synthetic_source &source,
                          const std::array<test_face, 5> &faces)
{
    const std::vector<gca_test::synthetic_surface> &surfaces = source.surfaces();
    std::printf("known surfaces: %zu\n", surfaces.size());
    check(surfaces.size() == 1 + faces.size(),
          "known surfaces: the plane, the box's top and its four sides");
    if (surfaces.size() != 1 + faces.size())
        return;
    const double hx = box_plane.half_size_x_m;
    const double hy = box_plane.half_size_y_m;
    const test_face plane = {{Eigen::Vector3d(-hx, -hy, 0.0), Eigen::Vector3d(hx, -hy, 0.0),
                              Eigen::Vector3d(hx, hy, 0.0), Eigen::Vector3d(-hx, hy, 0.0)},
                             Eigen::Vector3d::UnitZ()};
    max_error corners_m;
    max_error axes;
    int wrong_textures = 0;
    for (std::size_t i = 0; i < surfaces.size(); i++)
    {
        const gca_test::synthetic_surface &surface = surfaces[i];
        const test_face &expected = i == 0 ? plane : faces[i - 1];
        const gca_test::synthetic_texture texture =
            i == 0 ? gca_test::synthetic_texture::plane : gca_test::synthetic_texture::box;
        const std::array<Eigen::Vector3d, 4> corners = corners_of(surface);
        corners_m.add(std::max(farthest_corner_m(corners, expected.corners),
                               farthest_corner_m(expected.corners, corners)));
        axes.add(std::abs(surface.axis_u.norm() - 1.0));
        axes.add(std::abs(surface.axis_v.norm() - 1.0));
        axes.add(std::abs(surface.axis_u.dot(surface.axis_v)));
        axes.add((surface.axis_u.cross(surface.axis_v) - expected.outward).norm());
        if (surface.texture != texture)
            wrong_textures++;
    }
    check_max_error("known surfaces: corners off the test's plane and box, in m", corners_m,
                    geometry_tolerance);
    check_max_error("known surfaces: axes off unit length or perpendicular, normals off outward",
                    axes, geometry_tolerance);
    check_within("known surfaces: textures other than the plane's and the box's", wrong_textures,
                 0);
}

/// A face of the box that faces a camera, as the camera's picture shows it.
struct front_face
{
    std::size_t face;                          ///< its index in box_faces()
    std::array<Eigen::Vector2d, 4> corners_px; ///< its corners in the picture, (u, v), in px
    double winding;                            ///< +1 or -1, the sign of the outline's area
};

/// Projects the faces of the box that face a camera into its picture, forward and apart from the
/// renderer's rays: each corner is taken into the camera frame by the inverse of the pose and
/// projected by the pinhole formula, u = fx X / Z + cx, v = fy Y / Z + cy. A face faces the
/// camera when the camera lies on its outer side.
/// @param faces [in] the box's faces.
/// @param camera_to_world [in] the camera's pose, camera frame to world frame.
/// @param k [in] the intrinsics.
/// @return the faces that face the camera; nothing when a corner of the box does not lie in
///         front of the camera, Z > 0, which the outlines need to make up the box's silhouette.
std::optional<std::vector<front_face>> project_front_faces(const std::array<test_face, 5> &faces,
                                                           const Eigen::Isometry3d &camera_to_world,
                                                           const gca::intrinsics &k)
{
    const Eigen::Matrix3d camera_from_world = camera_to_world.linear().transpose();
    const Eigen::Vector3d position = camera_to_world.translation();
    std::vector<front_face> fronts;
    for (std::size_t f = 0; f < faces.size(); f++)
    {
        front_face front{f, {}, 0.0};
        for (std::size_t c = 0; c < 4; c++)
        {
            const Eigen::Vector3d p = camera_from_world * (faces[f].corners[c] - position);
            if (!(p.z() > 0.0))
                return std::nullopt;
            front.corners_px[c] = Eigen::Vector2d(double{k.fx} * p.x() / p.z() + double{k.cx},
                                                  double{k.fy} * p.y() / p.z() + double{k.cy});
        }
        if (faces[f].outward.dot(position - mean_of(faces[f].corners)) <= 0.0)
            continue;
        double twice_area = 0.0;
        for (std::size_t c = 0; c < 4; c++)
        {
            const Eigen::Vector2d &a = front.corners_px[c];
            const Eigen::Vector2d &b = front.corners_px[(c + 1) % 4];
            twice_area += a.x() * b.y() - b.x() * a.y();
        }
        front.winding = twice_area > 0.0 ? 1.0 : -1.0;
        fronts.push_back(front);
    }
    return fronts;
}

/// Where a pixel lies against the box's outline in a picture.
struct outline_place
{
    std::optional<std::size_t> face; ///< the face whose outline holds it, or none: outside
    double margin_px; ///< its distance from that face's outline, or from the silhouette, in px
};

/// Finds where a pixel lies against the outlines of the faces that face the camera. They meet
/// without overlapping and make up the box's silhouette, the box being convex and wholly in front
/// of the camera.
/// @param fronts [in] the faces that face the camera, projected.
/// @param pixel [in] the pixel's centre (u, v), in px.
/// @return the face whose outline holds the pixel and the pixel's distance from that outline's
///         edges; or none and its distance from the nearest outline. A pixel on an outline is
///         outside at distance 0.
outline_place place_of(const std::vector<front_face> &fronts, const Eigen::Vector2d &pixel)
{
    outline_place place{std::nullopt, std::numeric_limits<double>::infinity()};
    for (const front_face &front : fronts)
    {
        double inside_px = std::numeric_limits<double>::infinity();
        double distance_px = std::numeric_limits<double>::infinity();
        for (std::size_t c = 0; c < 4; c++)
        {
            const Eigen::Vector2d &a = front.corners_px[c];
            const Eigen::Vector2d edge = front.corners_px[(c + 1) % 4] - a;
            const Eigen::Vector2d to = pixel - a;
            const double length = edge.norm();
            inside_px = std::min(inside_px,
                                 front.winding * (edge.x() * to.y() - edge.y() * to.x()) / length);
            const double along = std::clamp(edge.dot(to) / (length * length), 0.0, 1.0);
            distance_px = std::min(distance_px, (to - along * edge).norm());
        }
        if (inside_px > 0.0)
            return {front.face, inside_px};
        place.margin_px = std::min(place.margin_px, distance_px);
    }
    return place;
}

/// The named pixels of pictures of the walk, by what they see.
struct walk_counts
{
    std::array<int, 5> faces{}; ///< on each face of the box, in box_faces()' order
    int plane = 0;              ///< on the plane, away from the box's silhouette
    int beside = 0;             ///< on the plane, just beside the silhouette
    int nothing = 0;            ///< seeing nothing
    int unnamed = 0;            ///< too near an outline or the plane's edge to be named

    /// Returns the pixels named on the box.
    /// @return the sum over its faces.
    int box() const
    {
        return faces[0] + faces[1] + faces[2] + faces[3] + faces[4];
    }

    /// Adds another picture's counts to these.
    /// @param other [in] the other picture's counts.
    /// @return these counts.
    walk_counts &operator+=(const walk_counts &other)
    {
        for (std::size_t f = 0; f < faces.size(); f++)
            faces[f] += other.faces[f];
        plane += other.plane;
        beside += other.beside;
        nothing += other.nothing;
        unnamed += other.unnamed;
        return *this;
    }
};

/// Prints named pixels by what they see.
/// @param label [in] the picture or pictures, printed first.
/// @param counts [in] the counts.
void print_counts(const std::string &label, const walk_counts &counts)
{
    std::printf("%s: box %d (top %d, +x %d, +y %d, -x %d, -y %d), plane %d, beside the "
                "silhouette %d, nothing %d, not named %d\n",
                label.c_str(), counts.box(), counts.faces[0], counts.faces[1], counts.faces[2],
                counts.faces[3], counts.faces[4], counts.plane, counts.beside, counts.nothing,
                counts.unnamed);
}

/// The errors of the walk's named pixels, by what they see.
struct walk_errors
{
    pixel_errors box;     ///< on a face of the box
    pixel_errors plane;   ///< on the plane, away from the silhouette
    pixel_errors beside;  ///< on the plane, just beside the silhouette
    pixel_errors nothing; ///< seeing nothing
};

/// Names the pixels of one picture of the walk and compares each with what it should hold: a
/// pixel inside a face's outline sees that face (hit_plane() on the face's plane); one outside
/// the silhouette sees the plane where its ray meets the plane's rectangle, else nothing. A pixel
/// within outline_margin_px of an outline or plane_margin_m of the plane's edge is not named.
/// @param frame [in] the frame.
/// @param camera_to_world [in] its pose, camera frame to world frame.
/// @param faces [in] the box's faces as the test builds them.
/// @param errors [in,out] the named pixels' errors are added.
/// @return the named pixels by what they see; none, with a failed check, when the camera is not
///         above the plane or a corner of the box is not in front of it.
walk_counts check_walk_picture(const gca::frame &frame, const Eigen::Isometry3d &camera_to_world,
                               const std::array<test_face, 5> &faces, walk_errors &errors)
{
    const gca::frame_description &description = frame.description();
    const std::optional<std::vector<front_face>> fronts =
        project_front_faces(faces, camera_to_world, description.intrinsics);
    walk_counts counts;
    const bool above = camera_to_world.translation().z() > 0.0;
    check(fronts.has_value() && above,
          "walk: the camera above the plane, the whole box in front of it");
    if (!fronts || !above)
        return counts;
    for (int v = 0; v < static_cast<int>(description.size.height); v++)
    {
        for (int u = 0; u < static_cast<int>(description.size.width); u++)
        {
            const outline_place place =
                place_of(*fronts, Eigen::Vector2d(static_cast<double>(u), static_cast<double>(v)));
            const Eigen::Vector3d ray = pixel_ray(description.intrinsics, u, v);
            if (place.margin_px < outline_margin_px)
            {
                counts.unnamed++;
            }
            else if (place.face)
            {
                const test_face &face = faces[*place.face];
                counts.faces[*place.face]++;
                compare_pixel(
                    frame, u, v,
                    expect_at(hit_plane(face.outward, face.corners[0], camera_to_world, ray),
                              gca_test::synthetic_texture::box, description),
                    errors.box);
            }
            else
            {
                const ray_hit hit = hit_plane(Eigen::Vector3d::UnitZ(), Eigen::Vector3d::Zero(),
                                              camera_to_world, ray);
                const double inside_m =
                    std::min(box_plane.half_size_x_m - std::abs(hit.point_world.x()),
                             box_plane.half_size_y_m - std::abs(hit.point_world.y()));
                if (hit.z_m > 0.0 && inside_m >= plane_margin_m)
                {
                    const bool beside = place.margin_px < beside_silhouette_px;
                    (beside ? counts.beside : counts.plane)++;
                    compare_pixel(frame, u, v,
                                  expect_at(hit, gca_test::synthetic_texture::plane, description),
                                  beside ? errors.beside : errors.plane);
                }
                else if (!(hit.z_m > 0.0) || inside_m <= -plane_margin_m)
                {
                    counts.nothing++;
                    compare_pixel(frame, u, v, expected_pixel{}, errors.nothing);
                }
                else
                {
                    counts.unnamed++;
                }
            }
        }
    }
    return counts;
}

/// Checks that the walk's trajectory keeps to the walk: as many poses as it asks for, each within
/// its distances from the centre along the plane and its heights, and each upright.
/// @param walk [in] the walk.
/// @param trajectory [in] its poses, camera frame to world frame.
void check_walk_poses(const gca_test::synthetic_walk &walk,
                      const std::vector<Eigen::Isometry3d> &trajectory)
{
    double nearest_m = std::numeric_limits<double>::infinity();
    double farthest_m = 0.0;
    double lowest_m = std::numeric_limits<double>::infinity();
    double highest_m = -std::numeric_limits<double>::infinity();
    for (const Eigen::Isometry3d &pose : trajectory)
    {
        const Eigen::Vector3d p = pose.translation();
        const double along_m = std::hypot(p.x() - walk.centre_m.x(), p.y() - walk.centre_m.y());
        nearest_m = std::min(nearest_m, along_m);
        farthest_m = std::max(farthest_m, along_m);
        lowest_m = std::min(lowest_m, p.z());
        highest_m = std::max(highest_m, p.z());
    }
    std::printf("walk: %zu poses, %.4f m to %.4f m from the box's centre along the plane, %.4f m "
                "to %.4f m above it\n",
                trajectory.size(), nearest_m, farthest_m, lowest_m, highest_m);
    check(trajectory.size() == walk.pose_count, "walk: as many poses as the walk asks for");
    check(nearest_m >= walk.radius_m - walk.radius_swing_m - geometry_tolerance &&
              farthest_m <= walk.radius_m + walk.radius_swing_m + geometry_tolerance &&
              lowest_m >= walk.height_m - walk.height_swing_m - geometry_tolerance &&
              highest_m <= walk.height_m + walk.height_swing_m + geometry_tolerance,
          "walk: the poses within the walk's distances and heights");
    check_upright("walk", trajectory);
}

/// Builds the options of the source of the walk: the box on its plane, seen along a trajectory.
/// @param trajectory [in] the trajectory, camera frame to world frame.
/// @return the options: one frame for each pose, then the end with walk_ending.
gca_test::synthetic_source_options walk_options(const std::vector<Eigen::Isometry3d> &trajectory)
{
    gca_test::synthetic_source_options options;
    options.description = {test_size, test_intrinsics, fine_depth_unit_m};
    options.scene = {box_plane, test_box, trajectory};
    options.ending = gca_test::synthetic_ending::end;
    options.ending_message = walk_ending;
    return options;
}

/// Checks each frame of the walk, prints what it names, and checks that the end comes after the
/// last pose.
/// @param source [in,out] the walk's source, open.
/// @param faces [in] the box's faces as the test builds them.
/// @param errors [in,out] the named pixels' errors are added.
/// @return the named pixels of all the frames by what they see.
walk_counts check_walk_frames(gca_test::synthetic_source &source,
                              const std::array<test_face, 5> &faces, walk_errors &errors)
{
    const std::vector<Eigen::Isometry3d> &trajectory = source.scene().trajectory;
    gca::frame frame;
    walk_counts totals;
    for (std::size_t i = 0; i < trajectory.size(); i++)
    {
        if (source.next(frame).status != gca::next_status::frame)
        {
            check(false, "walk: a frame for each pose");
            return totals;
        }
        check(frame.index() == i, "walk: frame i has index i");
        const walk_counts counts =
            check_walk_picture(frame, trajectory.at(frame.index()), faces, errors);
        print_counts("walk frame " + std::to_string(frame.index()), counts);
        check(counts.box() > 0 && counts.plane > 0 && counts.beside > 0,
              "walk: pixels named on the box, on the plane and just beside the silhouette");
        totals += counts;
    }
    const gca::next_result after = source.next(frame);
    check(after.status == gca::next_status::end && after.message == walk_ending,
          "walk: the end after the last pose");
    return totals;
}

/// Prints and checks what the walk named: pixels on each of the box's five faces and pixels that
/// see nothing, and every named pixel's errors within their tolerances.
/// @param totals [in] the named pixels of all the frames.
/// @param errors [in] their errors.
void report_walk(const walk_counts &totals, const walk_errors &errors)
{
    print_counts("walk", totals);
    check(std::ranges::all_of(totals.faces, [](int n) { return n > 0; }),
          "walk: pixels named on each of the box's five faces");
    check(totals.nothing > 0, "walk: pixels named that see nothing");
    check_errors("walk, box faces", errors.box);
    check_errors("walk, plane", errors.plane);
    check_errors("walk, plane beside the silhouette", errors.beside);
    check_errors("walk, nothing", errors.nothing);
}

/// Checks the box along the walk: the trajectory keeps to the walk; the source tells the
/// trajectory it was built with and the known surfaces of the test's own box and plane; frame i
/// has index i and is seen from the trajectory's pose i; the walk's pixels as check_walk_frames()
/// and report_walk() say.
void check_walk()
{
    const gca_test::synthetic_walk walk = test_walk();
    const std::optional<std::vector<Eigen::Isometry3d>> trajectory = gca_test::walk_around(walk);
    check(trajectory.has_value(), "walk: walk_around() gives the trajectory");
    if (!trajectory)
        return;
    check_walk_poses(walk, *trajectory);
    gca_test::synthetic_source source(walk_options(*trajectory));
    const std::array<test_face, 5> faces = box_faces(test_box);
    check_known_surfaces(source, faces);
    check(std::ranges::equal(source.scene().trajectory, *trajectory,
                             [](const Eigen::Isometry3d &a, const Eigen::Isometry3d &b) {
                                 return a.matrix() == b.matrix();
                             }),
          "walk: the source tells the trajectory it was built with");
    check(source.open(test_size).opened, "walk: opened");
    walk_errors errors;
    const walk_counts totals = check_walk_frames(source, faces, errors);
    report_walk(totals, errors);
}
} // namespace

int main()
{
    check_look_at();
    check_pictures();
    check_walk();
    return gca_test::check_summary("test_synthetic_source");
}
