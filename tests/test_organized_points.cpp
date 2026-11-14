// SPDX-License-Identifier: 0BSD
/// @file test_organized_points.cpp
/// The geometry module's organized points (geometry/organized_points.hpp) on frames of the tests'
/// synthetic source, at the smaller stream size with the camera's intrinsics; the method is
/// docs/methods/organized-points.md. The expected values are computed here by the pinhole
/// formula and the known surfaces, apart from the stage's code. Before each deprojection the
/// test fills both images with 0xFF bytes, so that a pixel the stage leaves unwritten shows.
/// - An analytic plane and an analytic sphere, every pixel: a pixel is valid exactly when it has
///   depth and its z, d * unit rounded to float, lies inside the depth range; every valid point
///   is the pinhole point of its pixel and depth value, and lies on the analytic surface within
///   half a depth unit times the length of its pixel's ray; an invalid pixel's point is
///   (0, 0, 0). The plane's picture has pixels without depth, nearer than the range and farther;
///   with a range from 0 its pixels without depth stay invalid; with limits that are exactly the
///   rounded z of depth values in the picture, the pixels of those values are valid; at 641x401,
///   a size that is no multiple of the kernel's block, the last column and row are written too.
/// - Against the prototype's create_from_rgbd(), the oracle, on frames of a walk around a box,
///   with a range narrower than 0.5 m, where the oracle's bilateral filter does nothing: both
///   keep the same pixels, and every point is the oracle's once the stated half pixel
///   (0.5 z / fx, 0.5 z / fy, 0) is added to it.
/// - Refusals as values, each leaving the points as they were: empty organized points, depth
///   ranges that are none, a GPU frame that holds no frame, a frame of another size, each kind of
///   intrinsics or depth unit that cannot be used, and CUDA's last error left by an earlier call;
///   and CUDA's refusal of the launch for a frame whose rows need more blocks than a grid holds.
///   CUDA's last error is clear after each.
/// - Every buffer is requested before the pool closes, and the stage asks the closed pool for
///   nothing.

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "check.hpp"
#include "geometry/organized_points.hpp"
#include "geometry/point_cloud.hpp"
#include "gpu/cuda_container.hpp"
#include "gpu/gpu_frame.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/step_result.hpp"
#include "gpu_test_helpers.hpp"
#include "stage_frames.hpp"
#include "synthetic_source.hpp"
#include "types/camera_param.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::camera_depth_unit_m;
using gca_test::check;
using gca_test::check_max_error;
using gca_test::check_no_cuda_error;
using gca_test::check_refused;
using gca_test::check_within;
using gca_test::create_stream;
using gca_test::granted;
using gca_test::large_description;
using gca_test::max_error;
using gca_test::odd_description;
using gca_test::render;
using gca_test::small_description;
using gca_test::tabletop_box;
using gca_test::tabletop_plane;
using gca_test::tabletop_sphere;
using gca_test::tall_size;

/// The analytic pictures' depth range: 0.4005 m to 0.8005 m, each limit half a depth unit off a
/// depth value, so that whether a pixel lies within it does not depend on rounding [S: the depth
/// values are whole millimetres], and narrow enough that the plane's picture has pixels nearer
/// and farther [M 2026-11-14: 43286 nearer, 81199 inside, 9429 farther, 122086 without depth].
constexpr gca::organized_points_parameters analytic_range = {0.4005f, 0.8005f};

/// The sphere's picture's depth range: 0.3005 m to 1.2005 m [?: a choice], which holds the sphere
/// and cuts off the plane's nearest part [M 2026-11-14: 9385 valid pixels on the sphere, 115548 on
/// the plane, 13676 nearer than the range].
constexpr gca::organized_points_parameters sphere_range = {0.3005f, 1.2005f};

/// A range from 0 that holds every depth value: a pixel without depth is invalid all the same
/// [S: the method's rule 1]; 65.535 m is the largest 16-bit value in millimetres [S].
constexpr gca::organized_points_parameters range_from_zero = {0.0f, 70.0f};

/// Where the limits on rounded z values are looked for: the near one from 450 depth units up,
/// the far one from 750 down, inside the plane's picture [?: a choice].
constexpr std::uint16_t near_search_from = 450;
constexpr std::uint16_t far_search_from = 750;

/// The oracle's depth range: 0.3005 m to 0.7995 m, narrower than 0.5 m, so that the prototype's
/// bilateral filter has a radius of 0 px everywhere and returns each pixel's own depth
/// [S: cuda_point_cloud_factory.cu, __adaptive_bilateral_filter: radius floor((z - near) / 0.5 m)];
/// each limit lies half a depth unit off a depth value [S].
constexpr gca::organized_points_parameters oracle_range = {0.3005f, 0.7995f};

/// The tolerance of a valid point against the pinhole point of its pixel and depth value, in m:
/// float's rounding, a step of 1.2e-7 m at 1 m to 2 m, taken a few times [S: float32 ulp]. The
/// largest distance over the pictures is 5.3e-8 m [M 2026-11-14]; the half pixel of the
/// prototype would be 0.65 mm at 0.4 m [S: 0.5 z / fx].
constexpr double pinhole_tolerance_m = 1e-6;

/// The tolerance of a valid point's distance from the analytic surface, divided by the depth
/// unit times the length of its pixel's ray r, whose z is 1: the depth's rounding to whole units
/// moves the point by at most half a unit along z, so by at most half a unit times |r| along its
/// ray, and no farther from the surface [S]; and 1e-3 for float's rounding of the point, about
/// 1e-7 m [S: float32 ulp]. The largest is 0.4963, on the sphere [M 2026-11-14].
constexpr double surface_tolerance = 0.5 + 1e-3;

/// The tolerance of a point against the oracle's once the half pixel is added, in m: both are
/// float points of the same z, each rounded a few times [S: float32 ulp, as pinhole_tolerance_m].
/// The largest is 9.7e-8 m [M 2026-11-14].
constexpr double oracle_tolerance_m = 1e-6;

/// How far an oracle point may project from a whole pixel once the half pixel is added, in px:
/// float's rounding, a few times 4e-5 px at 640 px [S: float32's relative step 6e-8], far below
/// the whole pixel a point of another pixel lies off [?: a choice]. The largest is 3.1e-5 px
/// [M 2026-11-14].
constexpr double oracle_pixel_tolerance_px = 1e-2;

/// What the stage works on at one size: a GPU frame and organized points of that size, both from
/// the pool, and the stream the work runs on.
struct stage_buffers
{
    gca::gpu_frame frame;          ///< the GPU frame the frames are uploaded into
    gca::organized_points points;  ///< the organized points they are deprojected into
    cudaStream_t stream = nullptr; ///< the CUDA stream of the uploads and deprojections
};

/// The points and marks of organized points, downloaded.
struct downloaded_points
{
    std::vector<float3> points;     ///< in the camera frame, in m
    std::vector<std::uint8_t> mask; ///< 1 valid, 0 invalid
};

/// Builds a frame of one depth and one color, for the cases whose pictures do not matter.
/// @param description [in] its size, intrinsics and depth unit.
/// @return the frame: depth 500 units and color 128 at every pixel [?: a choice].
gca::frame uniform_frame(const gca::frame_description &description)
{
    gca::frame host;
    host.prepare(description, 0, 0);
    std::ranges::fill(host.depth(), std::uint16_t{500});
    std::ranges::fill(host.color(), std::uint8_t{128});
    return host;
}

/// Fills both images of organized points with 0xFF bytes: a NaN in every coordinate and 255 in
/// every mark, which the stage writes over at every pixel it writes.
/// @param name [in] the case, named in the check line.
/// @param points [in,out] the organized points.
/// @param stream [in] the CUDA stream the fill is queued on.
void fill_with_0xff(const std::string &name, gca::organized_points &points, cudaStream_t stream)
{
    // The test writes into the stage's images on purpose; the stage gives them only to read.
    void *coordinates = const_cast<float3 *>(points.points().data());
    void *marks = const_cast<std::uint8_t *>(points.validity_mask().data());
    const bool filled =
        cudaMemsetAsync(coordinates, 0xFF, points.points().size_bytes(), stream) == cudaSuccess &&
        cudaMemsetAsync(marks, 0xFF, points.validity_mask().size_bytes(), stream) == cudaSuccess;
    check(filled, (name + ": the images filled with 0xFF").c_str());
}

/// Uploads a frame, fills the organized points with 0xFF bytes and deprojects the frame.
/// @param name [in] the case, named in the check lines.
/// @param host [in] the frame, of the stage's size.
/// @param stage [in,out] the GPU frame, organized points and stream.
/// @param parameters [in] the depth range.
/// @return whether the upload and the deprojection were done.
bool deproject(const std::string &name, const gca::frame &host, stage_buffers &stage,
               const gca::organized_points_parameters &parameters)
{
    const gca::step_result up = stage.frame.upload(host, stage.stream);
    check(up.done, (name + ": upload, " + up.message).c_str());
    fill_with_0xff(name, stage.points, stage.stream);
    const gca::step_result made = stage.points.deproject(stage.frame, parameters, stage.stream);
    check(made.done, (name + ": deproject, " + made.message).c_str());
    return up.done && made.done;
}

/// Downloads organized points: waits for the stream's work, then copies both images.
/// @param name [in] the case, named in the check line.
/// @param points [in] the organized points.
/// @param stream [in] the CUDA stream their work was queued on; waited for.
/// @return the points and marks; empty, with a failed check, when CUDA failed.
downloaded_points download(const std::string &name, const gca::organized_points &points,
                           cudaStream_t stream)
{
    const std::size_t pixels = static_cast<std::size_t>(points.size().width) * points.size().height;
    downloaded_points result{std::vector<float3>(pixels), std::vector<std::uint8_t>(pixels)};
    const bool copied =
        cudaStreamSynchronize(stream) == cudaSuccess &&
        cudaMemcpy(result.points.data(), points.points().data(), points.points().size_bytes(),
                   cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(result.mask.data(), points.validity_mask().data(),
                   points.validity_mask().size_bytes(), cudaMemcpyDeviceToHost) == cudaSuccess;
    check(copied, (name + ": the points downloaded").c_str());
    return copied ? std::move(result) : downloaded_points{};
}

/// Returns the ray of a pixel by the one pinhole convention.
/// @param u [in] the pixel's column, in px.
/// @param v [in] the pixel's row, in px.
/// @return ((u - cx) / fx, (v - cy) / fy, 1), in the camera frame, with the test's intrinsics.
Eigen::Vector3d pixel_ray(std::uint32_t u, std::uint32_t v)
{
    const gca::intrinsics &k = small_description.intrinsics;
    return {(u - double{k.cx}) / k.fx, (v - double{k.cy}) / k.fy, 1.0};
}

/// Returns the z the method compares with the range: the depth value times the unit, rounded
/// once to float. The product of two floats is exact in double, so its cast is the float
/// product's rounding [S: IEEE 754, a float's 24-bit and a double's 53-bit significand].
/// @param d [in] the depth value, in depth units.
/// @return d * camera_depth_unit_m rounded to float, in m.
float rounded_z(std::uint16_t d)
{
    return static_cast<float>(static_cast<double>(d) * camera_depth_unit_m);
}

/// A point's distance from the nearest analytic surface, and which surface that is.
struct surface_distance
{
    double distance_m = 0.0; ///< the distance, in m
    bool on_sphere = false;  ///< whether the sphere is the nearer surface
};

/// Returns a point's distance from the nearest analytic surface of a scene of the plane and,
/// optionally, the sphere: the plane z = 0 and the sphere's surface.
/// @param scene [in] the scene.
/// @param point_world [in] the point, in the world frame, in m.
/// @return the distance, and whether the sphere is the nearer surface.
surface_distance distance_from_surface(const gca_test::synthetic_scene &scene,
                                       const Eigen::Vector3d &point_world)
{
    const double from_plane_m = std::abs(point_world.z());
    if (!scene.sphere)
        return {from_plane_m, false};
    const double from_sphere_m =
        std::abs((point_world - scene.sphere->centre_m).norm() - scene.sphere->radius_m);
    return {std::min(from_plane_m, from_sphere_m), from_sphere_m < from_plane_m};
}

/// The pixels of a picture by what their depth makes them, and the errors of its points.
struct analytic_counts
{
    int valid = 0;            ///< pixels with depth inside the range
    int without_depth = 0;    ///< pixels of depth value 0
    int nearer = 0;           ///< pixels with depth nearer than the range
    int farther = 0;          ///< pixels with depth farther than the range
    int on_near_limit = 0;    ///< pixels whose rounded z is the range's near limit exactly
    int on_far_limit = 0;     ///< pixels whose rounded z is the range's far limit exactly
    int on_sphere = 0;        ///< valid pixels whose point lies nearer the sphere than the plane
    int wrong_marks = 0;      ///< pixels marked other than their depth makes them
    int invalid_not_zero = 0; ///< invalid pixels whose point is not (0, 0, 0)
    max_error from_pinhole_m; ///< valid points' distances from their pinhole points, in m
    /// Valid points' distances from the analytic surface over depth unit x |ray|.
    max_error surface_over_unit_ray;
};

/// Checks every pixel of a frame's organized points against the analytic scene: the mark is 1
/// exactly when the pixel's depth value is not 0 and its rounded z (rounded_z()) lies inside the
/// range; a valid point is the pinhole point of its pixel and depth value, z (u - cx) / fx,
/// z (v - cy) / fy, z, and lies on the analytic surface within half a depth unit times the
/// length of its pixel's ray; an invalid point is (0, 0, 0).
/// @param frame [in] the frame the points were made from.
/// @param got [in] the points and marks, of the frame's size.
/// @param scene [in] the scene, whose pose 0 the frame is seen from.
/// @param parameters [in] the depth range.
/// @return the counts and errors.
analytic_counts compare_analytic(const gca::frame &frame, const downloaded_points &got,
                                 const gca_test::synthetic_scene &scene,
                                 const gca::organized_points_parameters &parameters)
{
    analytic_counts counts;
    const gca::frame_size &size = frame.description().size;
    const Eigen::Isometry3d &camera_to_world = scene.trajectory.at(0);
    for (std::uint32_t v = 0; v < size.height; v++)
    {
        for (std::uint32_t u = 0; u < size.width; u++)
        {
            const std::size_t pixel = static_cast<std::size_t>(v) * size.width + u;
            const std::uint16_t d = frame.depth()[pixel];
            const float z_m = rounded_z(d);
            const bool valid = d != 0 && z_m >= parameters.near_m && z_m <= parameters.far_m;
            if (d == 0)
                counts.without_depth++;
            else if (z_m < parameters.near_m)
                counts.nearer++;
            else if (z_m > parameters.far_m)
                counts.farther++;
            else
                counts.valid++;
            counts.on_near_limit += d != 0 && z_m == parameters.near_m ? 1 : 0;
            counts.on_far_limit += d != 0 && z_m == parameters.far_m ? 1 : 0;
            if (got.mask[pixel] != (valid ? 1 : 0))
                counts.wrong_marks++;
            const float3 &p = got.points[pixel];
            const Eigen::Vector3d point(p.x, p.y, p.z);
            if (!valid)
            {
                if (p.x != 0.0f || p.y != 0.0f || p.z != 0.0f)
                    counts.invalid_not_zero++;
                continue;
            }
            const Eigen::Vector3d ray = pixel_ray(u, v);
            counts.from_pinhole_m.add((point - double{z_m} * ray).norm());
            const surface_distance surface = distance_from_surface(scene, camera_to_world * point);
            counts.surface_over_unit_ray.add(surface.distance_m /
                                             (camera_depth_unit_m * ray.norm()));
            counts.on_sphere += surface.on_sphere ? 1 : 0;
        }
    }
    return counts;
}

/// Prints a picture's counts and checks its marks and points.
/// @param name [in] the case, printed in each line.
/// @param counts [in] its counts and errors.
void report_analytic(const std::string &name, const analytic_counts &counts)
{
    std::printf("%s: %d pixels valid (%d on the sphere), %d without depth, %d nearer than the "
                "range, %d farther; %d on the near limit, %d on the far limit\n",
                name.c_str(), counts.valid, counts.on_sphere, counts.without_depth, counts.nearer,
                counts.farther, counts.on_near_limit, counts.on_far_limit);
    check_within((name + ": pixels marked other than their depth makes them").c_str(),
                 counts.wrong_marks, 0);
    check_within((name + ": invalid pixels whose point is not (0, 0, 0)").c_str(),
                 counts.invalid_not_zero, 0);
    check_max_error(name + ": valid points from the pinhole point, in m", counts.from_pinhole_m,
                    pinhole_tolerance_m);
    check_max_error(name + ": valid points' distance from the analytic surface over depth unit "
                           "x |ray|",
                    counts.surface_over_unit_ray, surface_tolerance);
}

/// Deprojects the one frame of an analytic scene with a range and checks every pixel.
/// @param name [in] the case, printed in each line.
/// @param frame [in] the scene's frame, of the stage's size.
/// @param scene [in] the scene, one pose.
/// @param parameters [in] the depth range.
/// @param stage [in,out] the GPU frame, organized points and stream of the frame's size.
/// @return the counts; all zero when the frame was not deprojected.
analytic_counts check_analytic(const std::string &name, const gca::frame &frame,
                               const gca_test::synthetic_scene &scene,
                               const gca::organized_points_parameters &parameters,
                               stage_buffers &stage)
{
    if (!deproject(name, frame, stage, parameters))
        return {};
    const downloaded_points got = download(name, stage.points, stage.stream);
    if (got.points.empty())
        return {};
    const analytic_counts counts = compare_analytic(frame, got, scene, parameters);
    report_analytic(name, counts);
    return counts;
}

/// Picks a range whose limits are exactly the rounded z of two depth values in a frame: the near
/// one of a value whose z rounds up, the far one of a value whose z rounds down, so that a stage
/// that compared the exact product instead of the rounded z would drop the pixels of both.
/// @param frame [in] the frame.
/// @return the range, near and far limit each a rounded_z(); nothing when the frame holds no
///         such values between near_search_from and far_search_from.
std::optional<gca::organized_points_parameters> limits_on_rounded_z(const gca::frame &frame)
{
    std::vector<bool> present(std::numeric_limits<std::uint16_t>::max() + 1, false);
    for (const std::uint16_t d : frame.depth())
        present[d] = true;
    const auto exact = [](std::uint16_t d) { return static_cast<double>(d) * camera_depth_unit_m; };
    std::optional<std::uint16_t> near;
    for (std::uint16_t d = near_search_from; d < far_search_from && !near; d++)
    {
        if (present[d] && rounded_z(d) > exact(d))
            near = d;
    }
    std::optional<std::uint16_t> far;
    for (std::uint16_t d = far_search_from; near && d > *near && !far; d--)
    {
        if (present[d] && rounded_z(d) < exact(d))
            far = d;
    }
    if (!near || !far)
        return std::nullopt;
    std::printf("limits on rounded z: %u units, %.9g m rounded up from %.12g m; %u units, %.9g m "
                "rounded down from %.12g m\n",
                static_cast<unsigned int>(*near), rounded_z(*near), exact(*near),
                static_cast<unsigned int>(*far), rounded_z(*far), exact(*far));
    return gca::organized_points_parameters{rounded_z(*near), rounded_z(*far)};
}

/// The stages the analytic pictures run on: the test size and a size that is no multiple of the
/// kernel's block.
struct analytic_stages
{
    stage_buffers &test;      ///< of the smaller stream size
    stage_buffers &odd_sized; ///< of the odd size
};

/// Checks the analytic plane and the analytic sphere. The plane is seen from 0.65 m, looking down
/// at about 32 degrees and rolled by 10 degrees, so that its picture has pixels nearer than the
/// range, inside it, farther, and past the plane's far edge without depth [?: a choice]. It is
/// deprojected again with a range from 0, whose pixels without depth stay invalid; with limits
/// on rounded z values; and at the odd size. The sphere rests on the plane and is seen from 0.46 m,
/// looking at its centre [?: a choice].
/// @param stages [in,out] the stages at both sizes.
void check_analytic_surfaces(analytic_stages stages)
{
    const std::optional<Eigen::Isometry3d> plane_pose =
        gca_test::look_at({0.0, -0.55, 0.35}, {0.0, 0.0, 0.0}, 10.0);
    const std::optional<Eigen::Isometry3d> sphere_pose =
        gca_test::look_at({0.15, -0.40, 0.30}, tabletop_sphere.centre_m, -7.0);
    check(plane_pose && sphere_pose, "the analytic pictures: look_at() gives both poses");
    if (!plane_pose || !sphere_pose)
        return;
    const gca_test::synthetic_scene plane_scene = {.plane = tabletop_plane,
                                                   .trajectory = {*plane_pose}};
    const gca_test::synthetic_scene sphere_scene = {
        .plane = tabletop_plane, .sphere = tabletop_sphere, .trajectory = {*sphere_pose}};
    const std::vector<gca::frame> plane_frame = render(plane_scene, small_description).frames;
    const std::vector<gca::frame> odd_frame = render(plane_scene, odd_description).frames;
    const std::vector<gca::frame> sphere_frame = render(sphere_scene, small_description).frames;
    if (plane_frame.empty() || odd_frame.empty() || sphere_frame.empty())
        return;

    const analytic_counts plane =
        check_analytic("plane", plane_frame[0], plane_scene, analytic_range, stages.test);
    check(plane.valid > 0 && plane.without_depth > 0 && plane.nearer > 0 && plane.farther > 0,
          "plane: pixels valid, without depth, nearer than the range and farther");
    const analytic_counts from_zero = check_analytic("plane, range from 0", plane_frame[0],
                                                     plane_scene, range_from_zero, stages.test);
    check(from_zero.without_depth > 0 && from_zero.nearer == 0 && from_zero.farther == 0,
          "plane, range from 0: pixels without depth, every other one inside the range");
    const std::optional<gca::organized_points_parameters> on_rounded_z =
        limits_on_rounded_z(plane_frame[0]);
    check(on_rounded_z.has_value(), "plane: depth values whose z rounds up and down");
    if (on_rounded_z)
    {
        const analytic_counts limits = check_analytic("plane, limits on rounded z", plane_frame[0],
                                                      plane_scene, *on_rounded_z, stages.test);
        check(limits.on_near_limit > 0 && limits.on_far_limit > 0,
              "plane, limits on rounded z: pixels on both limits");
    }
    const analytic_counts odd = check_analytic("plane at 641x401", odd_frame[0], plane_scene,
                                               analytic_range, stages.odd_sized);
    check(odd.valid > 0 && odd.without_depth > 0,
          "plane at 641x401: pixels valid and without depth");

    const analytic_counts sphere = check_analytic("sphere on the plane", sphere_frame[0],
                                                  sphere_scene, sphere_range, stages.test);
    check(sphere.on_sphere > 0 && sphere.valid > sphere.on_sphere,
          "sphere on the plane: valid pixels on the sphere and on the plane");
}

/// Converts a frame's color from RGB8 to the BGR8 the oracle reads.
/// @param frame [in] the frame.
/// @return its color picture, BGR8, rows dense.
std::vector<std::uint8_t> bgr_of(const gca::frame &frame)
{
    std::vector<std::uint8_t> bgr(frame.color().begin(), frame.color().end());
    for (std::size_t i = 0; i + 2 < bgr.size(); i += 3)
        std::swap(bgr[i], bgr[i + 2]);
    return bgr;
}

/// Runs the oracle, the prototype's create_from_rgbd(), on a frame: depth aligned to color, both
/// with the frame's intrinsics, identity extrinsics.
/// @param frame [in] the frame, of the smaller stream size.
/// @param parameters [in] the depth range.
/// @return the oracle's points, the invalid ones dropped.
/// @note The oracle runs on the default stream and waits for the whole device.
std::vector<gca::point_t> oracle_points(const gca::frame &frame,
                                        const gca::organized_points_parameters &parameters)
{
    gca::extrinsics identity{};
    identity.rotation[0] = identity.rotation[4] = identity.rotation[8] = 1.0f;
    const gca::cuda_camera_param param(small_description.intrinsics, small_description.intrinsics,
                                       identity, small_description.size.width,
                                       small_description.size.height, camera_depth_unit_m);
    const std::vector<std::uint8_t> bgr = bgr_of(frame);
    const gca::cuda_depth_frame depth(frame.depth().data(), small_description.size.width,
                                      small_description.size.height);
    const gca::cuda_color_frame color(bgr.data(), small_description.size.width,
                                      small_description.size.height);
    return gca::point_cloud::create_from_rgbd(depth, color, param, parameters.near_m,
                                              parameters.far_m)
        ->download();
}

/// The comparison with the oracle over the frames.
struct oracle_counts
{
    std::size_t oracle_points = 0;   ///< the oracle's points
    std::size_t valid_pixels = 0;    ///< the stage's valid pixels
    std::size_t unmatched = 0;       ///< oracle points of no valid pixel, or of one taken already
    max_error off_pixel_px;          ///< oracle points' distance from their whole pixel, in px
    max_error after_half_pixel_m;    ///< |point - (oracle's + half pixel)|, in m
    double least_half_pixel_m = 0.0; ///< the least |point - oracle's|, in m
    double most_half_pixel_m = 0.0;  ///< the largest |point - oracle's|, in m
};

/// Compares a frame's organized points with the oracle's points: each oracle point is projected
/// back to its pixel with the oracle's half pixel, (u - 0.5, v - 0.5), and must belong to a valid
/// pixel no other oracle point took; there the point must be the oracle's plus
/// (0.5 z / fx, 0.5 z / fy, 0).
/// @param oracle [in] the oracle's points.
/// @param got [in] the stage's points and marks, of the smaller stream size.
/// @param counts [in,out] the counts and errors are added.
void compare_oracle(const std::vector<gca::point_t> &oracle, const downloaded_points &got,
                    oracle_counts &counts)
{
    const gca::intrinsics &k = small_description.intrinsics;
    std::vector<bool> taken(got.mask.size(), false);
    counts.oracle_points += oracle.size();
    counts.valid_pixels += static_cast<std::size_t>(std::ranges::count(got.mask, 1));
    for (const gca::point_t &q : oracle)
    {
        const double qx = q.coordinates.x;
        const double qy = q.coordinates.y;
        const double qz = q.coordinates.z;
        const double u = qx * k.fx / qz + k.cx + 0.5;
        const double v = qy * k.fy / qz + k.cy + 0.5;
        const double u_whole = std::round(u);
        const double v_whole = std::round(v);
        counts.off_pixel_px.add(std::max(std::abs(u - u_whole), std::abs(v - v_whole)));
        if (!(u_whole >= 0.0 && u_whole < small_description.size.width && v_whole >= 0.0 &&
              v_whole < small_description.size.height))
        {
            counts.unmatched++;
            continue;
        }
        const std::size_t pixel = static_cast<std::size_t>(v_whole) * small_description.size.width +
                                  static_cast<std::size_t>(u_whole);
        if (got.mask[pixel] != 1 || taken[pixel])
        {
            counts.unmatched++;
            continue;
        }
        taken[pixel] = true;
        const float3 &p = got.points[pixel];
        const Eigen::Vector3d point(p.x, p.y, p.z);
        const Eigen::Vector3d oracle_point(qx, qy, qz);
        const Eigen::Vector3d half_pixel(0.5 * qz / k.fx, 0.5 * qz / k.fy, 0.0);
        counts.after_half_pixel_m.add((point - (oracle_point + half_pixel)).norm());
        const double shift_m = (point - oracle_point).norm();
        const bool first = counts.least_half_pixel_m == 0.0;
        counts.least_half_pixel_m = first ? shift_m : std::min(counts.least_half_pixel_m, shift_m);
        counts.most_half_pixel_m = std::max(counts.most_half_pixel_m, shift_m);
    }
}

/// Compares the stage with the oracle on three frames of a walk around a box on the plane, with
/// a range narrower than 0.5 m, where the oracle's bilateral filter does nothing.
/// @param stage [in,out] the GPU frame, organized points and stream of the smaller stream size.
void check_oracle(stage_buffers &stage)
{
    // Three views of the tabletop walk, 120 degrees apart [?: a choice].
    gca_test::synthetic_walk walk = gca_test::tabletop_walk();
    walk.pose_count = 3;
    const std::optional<std::vector<Eigen::Isometry3d>> trajectory = gca_test::walk_around(walk);
    check(trajectory.has_value(), "oracle: walk_around() gives the walk");
    if (!trajectory)
        return;
    const std::vector<gca::frame> frames =
        render({.plane = tabletop_plane, .box = tabletop_box, .trajectory = *trajectory},
               small_description)
            .frames;
    oracle_counts counts;
    for (const gca::frame &frame : frames)
    {
        const std::string name = "oracle, frame " + std::to_string(frame.index());
        if (!deproject(name, frame, stage, oracle_range))
            continue;
        const downloaded_points got = download(name, stage.points, stage.stream);
        if (!got.points.empty())
            compare_oracle(oracle_points(frame, oracle_range), got, counts);
    }
    std::printf("oracle: %zu frames, %zu oracle points, %zu valid pixels, %zu oracle points "
                "unmatched\n",
                frames.size(), counts.oracle_points, counts.valid_pixels, counts.unmatched);
    std::printf("oracle: the stated half pixel, |point - oracle's point| from %.4f mm to %.4f mm\n",
                1e3 * counts.least_half_pixel_m, 1e3 * counts.most_half_pixel_m);
    check(frames.size() == 3 && counts.oracle_points > 0,
          "oracle: three frames with oracle points");
    check(counts.oracle_points == counts.valid_pixels,
          "oracle: as many oracle points as valid pixels");
    check_within("oracle: oracle points of no valid pixel or of one taken", counts.unmatched, 0);
    check_max_error("oracle: oracle points off their whole pixel, in px", counts.off_pixel_px,
                    oracle_pixel_tolerance_px);
    check_max_error("oracle: |point - (oracle's point + half pixel)|, in m",
                    counts.after_half_pixel_m, oracle_tolerance_m);
}

/// The GPU memory the refusals use, all requested before the pool closes.
struct refusal_buffers
{
    gca::gpu_frame never_uploaded; ///< a GPU frame of the test size that holds no frame
    gca::gpu_frame described;      ///< a GPU frame of the test size for the descriptions
    gca::organized_points other;   ///< organized points of the larger stream size
    gca::gpu_frame tall_frame;     ///< a GPU frame of tall_size
    gca::organized_points tall;    ///< organized points of tall_size
};

/// A frame's description that cannot be used, and the case's name.
struct unusable_description
{
    const char *what;           ///< the case
    gca::intrinsics intrinsics; ///< its intrinsics, in px
    float depth_unit_m;         ///< its depth unit, in m
};

/// Returns each kind of description the stage refuses: fx and fy not > 0 or not finite, cx and cy
/// not finite, the depth unit not > 0 or not finite, each alone [?: a choice of values].
/// @return the cases.
std::array<unusable_description, 8> unusable_descriptions()
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinite = std::numeric_limits<float>::infinity();
    const gca::intrinsics k = small_description.intrinsics;
    return {{{"fx 0", {0.0f, k.fy, k.cx, k.cy}, camera_depth_unit_m},
             {"fx infinite", {infinite, k.fy, k.cx, k.cy}, camera_depth_unit_m},
             {"fy -1", {k.fx, -1.0f, k.cx, k.cy}, camera_depth_unit_m},
             {"fy not a number", {k.fx, nan, k.cx, k.cy}, camera_depth_unit_m},
             {"cx not a number", {k.fx, k.fy, nan, k.cy}, camera_depth_unit_m},
             {"cy infinite", {k.fx, k.fy, k.cx, infinite}, camera_depth_unit_m},
             {"depth unit 0", k, 0.0f},
             {"depth unit infinite", k, infinite}}};
}

/// Checks the refusals that leave the points as they were: empty organized points, depth ranges
/// that are none, a GPU frame that holds no frame, a frame of another size, each kind of
/// description that cannot be used, and CUDA's last error left by an earlier call.
/// @param stage [in,out] the stage of the smaller stream size, its GPU frame holding a frame of the
/// scene and
///        its points made from it.
/// @param buffers [in,out] the other GPU memory the refusals use.
void check_refusals_leave_points(stage_buffers &stage, refusal_buffers &buffers)
{
    gca::organized_points &out = stage.points;
    const cudaStream_t stream = stage.stream;
    const downloaded_points before = download("refusals, before", out, stream);
    gca::organized_points empty;
    check_refused("empty organized points", empty.deproject(stage.frame, analytic_range, stream),
                  "the organized points hold no images");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::string not_a_range = "is not one";
    check_refused("a range below 0", out.deproject(stage.frame, {-0.1f, 1.0f}, stream),
                  not_a_range);
    check_refused("a range whose far limit is its near one",
                  out.deproject(stage.frame, {0.5f, 0.5f}, stream), not_a_range);
    check_refused("a range that is not finite", out.deproject(stage.frame, {0.3f, nan}, stream),
                  not_a_range);
    check_refused("a GPU frame never uploaded",
                  out.deproject(buffers.never_uploaded, analytic_range, stream),
                  "the GPU frame holds no frame");
    check_refused("an empty GPU frame", out.deproject(gca::gpu_frame{}, analytic_range, stream),
                  "the GPU frame holds no frame");
    check_refused("organized points of another size",
                  buffers.other.deproject(stage.frame, analytic_range, stream),
                  "the frame is 640x400 px, the organized points 1280x800 px");
    for (const unusable_description &c : unusable_descriptions())
    {
        const gca::step_result up = buffers.described.upload(
            uniform_frame({small_description.size, c.intrinsics, c.depth_unit_m}), stream);
        check(up.done, (std::string(c.what) + ": uploaded, " + up.message).c_str());
        check_refused(c.what, out.deproject(buffers.described, analytic_range, stream),
                      "cannot be used");
    }
    // cudaMalloc's refusal of 2^52 bytes, 4 PiB on a board of 7.4 GiB [S: docs/HARDWARE.md],
    // leaves its error as CUDA's last error.
    void *never_granted = nullptr;
    check(cudaMalloc(&never_granted, std::size_t{1} << 52) != cudaSuccess,
          "cudaMalloc refuses 2^52 bytes");
    check_refused("CUDA's last error left by an earlier call",
                  out.deproject(stage.frame, analytic_range, stream), "left by an earlier call");
    const downloaded_points after = download("refusals, after", out, stream);
    const bool same = !before.points.empty() && before.points.size() == after.points.size() &&
                      std::memcmp(before.points.data(), after.points.data(),
                                  before.points.size() * sizeof(float3)) == 0 &&
                      before.mask == after.mask;
    check(same, "refusals: the organized points left as they were, bit for bit");
    check_no_cuda_error("refusals");
}

/// Checks CUDA's refusal of the kernel's launch for a frame of 1 x 600000 px, whose rows need
/// 75000 blocks along y where a grid holds 65535: a failure with CUDA's error, which is cleared.
/// @param buffers [in,out] the GPU frame and organized points of tall_size.
/// @param stream [in] the CUDA stream the work runs on.
void check_refused_launch(refusal_buffers &buffers, cudaStream_t stream)
{
    const gca::step_result up = buffers.tall_frame.upload(
        uniform_frame({tall_size, small_description.intrinsics, camera_depth_unit_m}), stream);
    check(up.done, ("the tall frame uploaded, " + up.message).c_str());
    check_refused("a frame of 1x600000 px, 75000 blocks along y",
                  buffers.tall.deproject(buffers.tall_frame, analytic_range, stream),
                  "cudaErrorInvalidConfiguration");
    check_no_cuda_error("the refused launch");
}

/// Requests every buffer, closes the pool, runs the checks and checks that the stage asked the
/// closed pool for nothing.
/// @param stream [in] the CUDA stream the work runs on.
void check_with_closed_pool(cudaStream_t stream)
{
    gca::gpu_pool pool;
    const std::string frame_name = "a GPU frame";
    const std::string points_name = "organized points";
    stage_buffers test{
        granted(frame_name, gca::gpu_frame::request(pool, small_description.size)),
        granted(points_name, gca::organized_points::request(pool, small_description.size)), stream};
    stage_buffers odd_sized{
        granted(frame_name, gca::gpu_frame::request(pool, odd_description.size)),
        granted(points_name, gca::organized_points::request(pool, odd_description.size)), stream};
    refusal_buffers buffers{
        granted(frame_name, gca::gpu_frame::request(pool, small_description.size)),
        granted(frame_name, gca::gpu_frame::request(pool, small_description.size)),
        granted(points_name, gca::organized_points::request(pool, large_description.size)),
        granted(frame_name, gca::gpu_frame::request(pool, tall_size)),
        granted(points_name, gca::organized_points::request(pool, tall_size))};
    pool.close();
    check_analytic_surfaces({test, odd_sized});
    check_oracle(test);
    check_refusals_leave_points(test, buffers);
    check_refused_launch(buffers, stream);
    check_within("requests to the closed pool", static_cast<double>(pool.refused_while_closed()),
                 0);
}
} // namespace

/// Runs the checks of the organized points and prints the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    cudaStream_t stream = nullptr;
    if (!create_stream(stream))
        return gca_test::check_summary("test_organized_points");
    check_with_closed_pool(stream);
    check(cudaStreamDestroy(stream) == cudaSuccess, "the CUDA stream destroyed");
    return gca_test::check_summary("test_organized_points");
}
