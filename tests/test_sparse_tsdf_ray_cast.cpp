// SPDX-License-Identifier: 0BSD
/// @file test_sparse_tsdf_ray_cast.cpp
/// The fusion module's ray cast of the model's surface as a pose sees it
/// (fusion/sparse_tsdf.hpp, sparse_tsdf::ray_cast() and model_surface), on sparse TSDFs fused from
/// the tests' synthetic source at known poses and from built walls. The method is
/// docs/methods/sparse-tsdf.md, "The ray cast". The models' frame is the scene's world frame. The
/// expected surfaces, normals and intensities come from the analytic scene (synthetic_source.hpp),
/// apart from the module's code.
/// - A wall facing the camera at a whole millimetre, fused once at the tabletop object's live voxel
///   size, where its truncation is the least, mu_min: cast from its pose, every pixel inside the
///   picture's border holds the wall at its depth within float's rounding, its normal along -z, its
///   intensity and its color; seen from behind, a back face, every pixel is invalid; a view that
///   meets no block, every pixel is invalid.
/// - The colors at level 0: a wall whose color changes along its rows and columns, cast from its
///   pose and from two others, every pixel that sees it holds its color where the point lies,
///   within measured tolerances.
/// - The depth interval, which lets a ray step past the blocks before and after the depths its
///   tile's held blocks span (docs/methods/sparse-tsdf.md, "The ray cast"): the wall cast from
///   poses turned and shifted from its own, at both stream sizes, every pixel whose ray meets the
///   fused part of the wall within the range holding it at its depth with its normal; a patch in
///   front of the wall, both surfaces held around the patch's border, where tiles see the blocks of
///   both; the wall cast from inside its band with the range starting at 0, every pixel holding the
///   wall 2.5 mm away; the wall cast through a model with a patch in front of it, whose blocks
///   widen tiles' intervals, and through one without, every pixel out of the patch's reach the same
///   bit for bit in both.
/// - The normal, whose six ends read each of their voxels once (docs/methods/sparse-tsdf.md, "The
///   ray cast"): a wall fused at a slant, so that its distances change along every axis, cast at
///   both stream sizes, every pixel whose ray meets the fused part of the wall holding it at its
///   depth with its normal within measured tolerances; among them the column and the row through
///   the principal point, where float's rounding puts the end one voxel before every crossing into
///   the crossing's own cell, so that the normal takes its six interpolations there.
/// - A sphere resting on a plane, fused from a walk of 12 known poses, at the live voxel sizes of
///   both object classes: cast from a pose of the walk, from one between two of its poses and from
///   one a hand's step away from it, at both stream sizes, every valid point lies within measured
///   tolerances of the analytic surface, its normal within an angle, its intensity within a
///   tolerance of the texture's luma there and its color of the texture's; a stated share of the
///   pixels whose ray meets the scene within the depth range is valid; a pixel whose ray meets
///   nothing is invalid; a view of space never seen is invalid everywhere.
/// - Every cast: each coarser level's pixel is level 0's at twice or four times its index, bit for
///   bit, with the intrinsics halved; an invalid pixel is all 0, its color too; a valid point
///   projects onto its own pixel.
/// - Through the registration: a frame seen a hand's step from the last fused pose is registered
///   against the cast from that pose by the full method and recovers its known motion within a
///   measured tolerance, for the tabletop object, the large object and the large box turned by 1.5
///   degrees; each with level 0 alone, levels 1 and 0, and all three, printed, and against the last
///   frame's own pyramid for comparison.
/// - Refusals as values, each leaving the surface as it was and CUDA's last error clear; the casts
///   ask the closed pool for nothing.

#include "camera/frame.hpp"
#include "check.hpp"
#include "fusion/sparse_tsdf.hpp"
#include "geometry/frame_pyramid.hpp"
#include "geometry/organized_points.hpp"
#include "gpu/gpu_image.hpp"
#include "gpu/gpu_messages.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/step_result.hpp"
#include "gpu_test_helpers.hpp"
#include "registration/registration.hpp"
#include "registration_cases.hpp"
#include "scan/object_class.hpp"
#include "sparse_tsdf_helpers.hpp"
#include "sparse_tsdf_scenes.hpp"
#include "stage_checks.hpp"
#include "stage_frames.hpp"
#include "synthetic_source.hpp"
#include "tracking/tracker.hpp"
#include "transform_checks.hpp"
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
#include <initializer_list>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::angle_deg;
using gca_test::as_camera_pose;
using gca_test::check;
using gca_test::check_converged_is_known;
using gca_test::check_no_cuda_error;
using gca_test::check_refused;
using gca_test::check_within;
using gca_test::create_stream;
using gca_test::download_image;
using gca_test::expect_at;
using gca_test::expected_point;
using gca_test::flat_wall;
using gca_test::frame_buffers;
using gca_test::fuse;
using gca_test::granted;
using gca_test::large_description;
using gca_test::make_transform;
using gca_test::max_error;
using gca_test::model_case;
using gca_test::model_of;
using gca_test::normals_parameters;
using gca_test::print_result;
using gca_test::quantile;
using gca_test::range_of;
using gca_test::request_buffers;
using gca_test::small_description;
using gca_test::to_double;
using gca_test::transform_tolerance;

/// The number of levels the ray cast hands over.
constexpr std::size_t levels = gca::frame_pyramid::level_count;

/// The wall's depth and color: 500 mm, where the tabletop object's live model, 2 mm voxels, has its
/// least truncation, mu_min = 4 voxels, 8 mm, above 3 sigma = 7.5 mm [S: computed,
/// docs/methods/sparse-tsdf.md, "The truncation distance"], so that the wall's band is the
/// thinnest the method makes; the color differs in every channel [?: a choice].
constexpr std::uint16_t wall_mm = 500;
constexpr std::array<std::uint8_t, 3> wall_color = {40, 90, 200};

/// The pixels a wall's checks leave out at the picture's border: 8, where every voxel of a point's
/// central differences sees the wall [?: a choice: a voxel of 2 mm at 0.5 m spans 1.2 px].
constexpr std::uint32_t wall_border_px = 8;

/// The tolerance of the wall's depth: its distances are half floats exactly, so the crossing is
/// exact but for float's rounding of the samples' depths and of the interpolation, a few times
/// 6e-8 m at 0.5 m [S: float32 ulp]; taken 1 µm [?: a choice].
constexpr double wall_depth_tolerance_m = 1e-6;

/// The tolerance of the wall's normal: exactly (0, 0, -1) by its central differences, every voxel
/// at one depth having one distance [S]; 1e-4 degrees for float's rounding [?: a choice].
constexpr double wall_normal_tolerance_deg = 1e-4;

/// The tolerance of an intensity computed in float against the same in double: a few float steps
/// of 1, below 1e-6 [S: float32 ulp; docs/methods/frame-pyramid.md, "The intensity"].
constexpr double intensity_float_tolerance = 1e-6;

/// The tolerance of a point's projection onto its own pixel: 1e-3 px [?: a choice, above float's
/// rounding of a projection onto pixels up to 1279, a few steps of 1.2e-4 px (S: float32 ulp)];
/// at most 7.1e-5 px [M 2026-11-15: this test].
constexpr double projection_tolerance_px = 1e-3;

/// The tolerance of a normal's length: 1e-6 [S: float32 ulp, a few steps].
constexpr double unit_normal_tolerance = 1e-6;

/// The tolerances of a scene's casts, each in the units its name gives.
struct cast_tolerances
{
    double distance_median_mm; ///< the median distance of a point from the analytic surface
    double distance_max_mm;    ///< the largest such distance
    double normal_median_deg;  ///< the median angle between a normal and the surface's
    double normal_p99_deg;     ///< the 99th percentile of that angle
    double intensity_median;   ///< the median of |intensity - the texture's luma|, of [0, 1]
    double intensity_p99;      ///< the 99th percentile of that
    double color_median; ///< the median of a color's largest channel's |error| from the texture's
    double color_p99;    ///< the 99th percentile of that, in 0 to 255
    double least_seen_valid; ///< the least share of the pixels that see the scene that are valid
    double unseen_distance_max_mm; ///< the largest distance from the surface of a valid pixel
                                   ///< whose ray meets nothing
};

/// The tolerances of the scenes' casts: 1.37 to 1.60 times the worst of the six casts of each
/// scene, from a pose of the walk, from one halfway to the next and from one a hand's step off, at
/// both stream sizes, with the method's starting values and the ray cast as built; for the pixels
/// that see the scene, their share left invalid [M 2026-11-15: this test; the worst median, 99th
/// percentile and largest value of each stand above each]. The largest
/// normal's angle and intensity error are printed, not checked: they lie where the band of a
/// surface reaches past another's silhouette, as the extraction's do (docs/methods/sparse-tsdf.md,
/// "The extraction"). The valid pixels that see nothing lie at the plane's border, where a voxel
/// past the edge takes its nearest pixel's depth: 0 to 14 of a cast at 2 mm, 73 to 312 at 8 mm.
/// The tabletop scene, 2 mm voxels: distances 0.0570 / 0.864 / 1.83 mm; angles 2.04 / 8.36 / 32.4
/// degrees; intensity errors 0.00280 / 0.0198 / 0.185; a color's largest channel's errors 1.971 /
/// 9.915 / 62.61 of 255 [M 2026-11-17: this test]; at least 0.99688 of the pixels that see the
/// scene valid; a valid pixel that sees nothing within 1.14 mm, 0.57 voxels, of the plane.
constexpr cast_tolerances tabletop_tolerances = {0.08, 2.5, 3.0,  12.0,  0.004,
                                                 0.03, 3.0, 15.0, 0.995, 1.7};

/// The large scene, 8 mm voxels: distances 0.243 / 4.71 / 10.8 mm; angles 2.14 / 14.0 / 75.2
/// degrees; intensity errors 0.0277 / 0.104 / 0.481 and a color's largest channel's errors 18.73 /
/// 50.23 / 164.7 of 255 [M 2026-11-17: this test], the 8 mm voxels blurring the textures' waves
/// of 45 mm to 100 mm; at least 0.99599 of the pixels that see the scene valid; a valid pixel that
/// sees nothing within 7.27 mm, 0.91 voxels, of the plane.
constexpr cast_tolerances large_tolerances = {0.35, 15.0, 3.0,  20.0,  0.04,
                                              0.15, 28.0, 75.0, 0.994, 11.0};

/// How far a transform recovered through the cast may lie from the known motion, about twice the
/// largest errors of the full method and of the levels it was run with [M 2026-11-15: this test]:
/// the tabletop object 0.02 degrees and 0.15 mm, from 0.00926 degrees and 0.0771 mm, the
/// registration test's own tolerance (registration_cases.hpp, the frame's pyramid as the surface,
/// 0.00624 degrees and 0.0594 mm); the large object 0.04 degrees and 2 mm, from 0.0196 degrees and
/// 0.962 mm, where the 8 mm voxels blur the box's texture of 5 to 9 pixels to a wave (against the
/// frame's pyramid 0.0136 degrees and 0.671 mm); the large box turned by 1.5 degrees within it,
/// 0.00219 degrees and 0.0827 mm.
constexpr transform_tolerance tabletop_through_cast_tolerance = {0.02, 0.15e-3};
constexpr transform_tolerance large_through_cast_tolerance = {0.04, 2.0e-3};

/// A level of a ray cast, downloaded to the host.
struct cast_picture
{
    gca::frame_size size{0, 0};      ///< its width and height, in px
    gca::intrinsics k{};             ///< its intrinsics, in px
    std::vector<float3> points;      ///< its points, in the cast's camera frame, in m
    std::vector<std::uint8_t> valid; ///< its marks
    std::vector<float3> normals;     ///< its normals, in the cast's camera frame
    std::vector<float> intensity;    ///< its intensities
    std::vector<uchar3> colors;      ///< its colors, red, green and blue; level 0's alone
};

/// The levels of a ray cast, downloaded.
using cast_pictures = std::array<cast_picture, levels>;

/// Downloads every level of a model's surface.
/// @param surface [in] the surface.
/// @param stream [in] the stream its work was queued on.
/// @param what [in] the case, named in the check line.
/// @return the levels; with a failed check when a copy failed.
cast_pictures download_surface(const gca::model_surface &surface, cudaStream_t stream,
                               const std::string &what)
{
    cast_pictures pictures;
    bool copied = true;
    for (std::size_t l = 0; l < levels; l++)
    {
        cast_picture &p = pictures[l];
        p.size = surface.size(l);
        p.k = surface.level_intrinsics(l);
        copied = download_image(surface.points(l), p.points, stream) &&
                 download_image(surface.validity_mask(l), p.valid, stream) &&
                 download_image(surface.normals(l), p.normals, stream) &&
                 download_image(surface.intensity(l), p.intensity, stream) && copied;
    }
    copied = download_image(surface.colors(), pictures[0].colors, stream) && copied;
    check(copied, (what + ": the surface downloaded").c_str());
    return pictures;
}

/// Ray casts a sparse TSDF from a pose and downloads the surface.
/// @param tsdf [in] the sparse TSDF.
/// @param pose [in] the camera's pose, camera frame to the models' frame.
/// @param k [in] level 0's intrinsics.
/// @param range [in] the depth range.
/// @param surface [in,out] the surface written.
/// @param stream [in] the stream.
/// @param what [in] the case, named in the check lines.
/// @return the levels downloaded.
cast_pictures cast(const gca::sparse_tsdf &tsdf, const Eigen::Isometry3d &pose,
                   const gca::intrinsics &k, const gca::organized_points_parameters &range,
                   gca::model_surface &surface, cudaStream_t stream, const std::string &what)
{
    const gca::step_result made = tsdf.ray_cast(as_camera_pose(pose), k, range, surface, stream);
    check(made.done, (what + ": ray cast queued, " + made.message).c_str());
    return download_surface(surface, stream, what);
}

/// Says whether two float3 are the same bit for bit.
/// @param a [in] one.
/// @param b [in] the other.
/// @return whether every byte is equal.
bool same_bits(const float3 &a, const float3 &b)
{
    return std::memcmp(&a, &b, sizeof(float3)) == 0;
}

/// Checks what every cast keeps: each coarser level's pixel (u, v) is level 0's at (2^l u, 2^l v)
/// bit for bit, its intrinsics level 0's halved l times; an invalid pixel is all 0, at level 0 its
/// color too; level 0's colors are of its size; a valid pixel's point lies in front of the camera
/// within the depth range and projects onto its own pixel, its normal is of unit length or
/// (0, 0, 0) and its intensity lies in [0, 1].
/// @param what [in] the case, named in the check lines.
/// @param pictures [in] the cast's levels.
/// @param k [in] level 0's intrinsics the cast was made with.
/// @param range [in] the depth range it was made with.
void check_levels(const std::string &what, const cast_pictures &pictures, const gca::intrinsics &k,
                  const gca::organized_points_parameters &range)
{
    int odd_pixels = 0;
    int wrong_intrinsics = 0;
    int unclean_invalid = 0;
    int out_of_range = 0;
    max_error projection;
    max_error unit;
    for (std::size_t l = 0; l < levels; l++)
    {
        const cast_picture &p = pictures[l];
        const double scale = std::ldexp(1.0, -static_cast<int>(l));
        wrong_intrinsics += p.k.fx != static_cast<float>(k.fx * scale) ||
                            p.k.fy != static_cast<float>(k.fy * scale) ||
                            p.k.cx != static_cast<float>(k.cx * scale) ||
                            p.k.cy != static_cast<float>(k.cy * scale);
        const gca::frame_size expected = gca::frame_pyramid::level_size(pictures[0].size, l);
        wrong_intrinsics += p.size != expected;
        if (l == 0)
            wrong_intrinsics += p.colors.size() != p.points.size();
        if (p.points.size() != static_cast<std::size_t>(p.size.width) * p.size.height)
            continue;
        for (std::uint32_t v = 0; v < p.size.height; v++)
            for (std::uint32_t u = 0; u < p.size.width; u++)
            {
                const std::size_t i = gca_test::index_of(p.size, u, v);
                const std::size_t i0 = gca_test::index_of(pictures[0].size, u << l, v << l);
                odd_pixels +=
                    !same_bits(p.points[i], pictures[0].points[i0]) ||
                    p.valid[i] != pictures[0].valid[i0] ||
                    !same_bits(p.normals[i], pictures[0].normals[i0]) ||
                    std::memcmp(&p.intensity[i], &pictures[0].intensity[i0], sizeof(float)) != 0;
                if (p.valid[i] == 0)
                {
                    unclean_invalid += !gca_test::is_zero(p.points[i]) ||
                                       !gca_test::is_zero(p.normals[i]) || p.intensity[i] != 0.0f;
                    if (l == 0 && i < p.colors.size())
                        unclean_invalid +=
                            p.colors[i].x != 0 || p.colors[i].y != 0 || p.colors[i].z != 0;
                    continue;
                }
                const Eigen::Vector3d q = to_double(p.points[i]);
                out_of_range += !(q.z() >= range.near_m && q.z() <= range.far_m) ||
                                !(p.intensity[i] >= 0.0f && p.intensity[i] <= 1.0f);
                projection.add(std::hypot(p.k.fx * q.x() / q.z() + p.k.cx - u,
                                          p.k.fy * q.y() / q.z() + p.k.cy - v));
                if (!gca_test::is_zero(p.normals[i]))
                    unit.add(std::abs(to_double(p.normals[i]).norm() - 1.0));
            }
    }
    check_within((what + ": coarser pixels not level 0's at their centre").c_str(), odd_pixels,
                 0.0);
    check_within(
        (what + ": levels or colors of the wrong size, or levels' intrinsics wrong").c_str(),
        wrong_intrinsics, 0.0);
    check_within((what + ": invalid pixels not all 0").c_str(), unclean_invalid, 0.0);
    check_within((what + ": valid pixels out of the range or of [0, 1]").c_str(), out_of_range,
                 0.0);
    gca_test::check_max_error(what + ": a point's projection from its pixel, px", projection,
                              projection_tolerance_px);
    gca_test::check_max_error(what + ": a normal's length less 1", unit, unit_normal_tolerance);
}

/// Counts a cast's valid pixels at level 0.
/// @param picture [in] the level.
/// @return its pixels marked valid.
std::size_t valid_count(const cast_picture &picture)
{
    return static_cast<std::size_t>(std::count(picture.valid.begin(), picture.valid.end(), 1));
}

/// Gives a color's intensity in double: the Rec. 709 luma over 255 (docs/methods/frame-pyramid.md,
/// "The intensity").
/// @param color [in] red, green and blue, 0 to 255.
/// @return the intensity, in [0, 1].
double luma_of(const std::array<double, 3> &color)
{
    return (0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]) / 255.0;
}

/// Gives a cast's color's largest error from a known color.
/// @param got [in] the cast's color, red, green and blue.
/// @param known [in] the known color, red, green and blue, 0 to 255.
/// @return the largest of the three channels' |got - known|, in levels of 0 to 255.
double largest_channel_error(const uchar3 &got, const std::array<double, 3> &known)
{
    return std::max({std::abs(static_cast<double>(got.x) - known[0]),
                     std::abs(static_cast<double>(got.y) - known[1]),
                     std::abs(static_cast<double>(got.z) - known[2])});
}

/// The tabletop object's live model fused from frames at one pose, the walls' cases' model: its
/// pool, closed once the buffers, the sparse TSDF and a surface at each stream size are granted.
/// Not copied and not moved, as its pool is not.
struct fused_wall
{
    model_case m;                               ///< the parameters and the depth range
    gca::gpu_pool pool;                         ///< the pool, closed after the requests
    frame_buffers buffers;                      ///< the frame's buffers, 640x400
    gca::sparse_tsdf tsdf;                      ///< the model
    std::array<gca::model_surface, 2> surfaces; ///< the surfaces at 640x400 and 1280x800

    /// Requests the buffers, the sparse TSDF and the surfaces, closes the pool and fuses the
    /// frames, each at the same pose.
    /// @param name [in] the case, named in the check lines.
    /// @param frames [in] the frames, 640x400 with the camera's intrinsics; owned by the caller.
    /// @param stream [in] the stream.
    /// @param fused_from [in] the frames' pose, camera frame to the models' frame.
    fused_wall(const std::string &name, std::initializer_list<const gca::frame *> frames,
               cudaStream_t stream,
               const Eigen::Isometry3d &fused_from = Eigen::Isometry3d::Identity())
        : m(model_of(name, gca::object_class::tabletop_object, false))
        , buffers(request_buffers(pool, small_description.size, name))
        , tsdf(granted(name + ": the sparse TSDF", gca::sparse_tsdf::request(pool, m.tsdf, stream)))
        , surfaces{granted(name + ": the surface at 640x400",
                           gca::model_surface::request(pool, small_description.size)),
                   granted(name + ": the surface at 1280x800",
                           gca::model_surface::request(pool, large_description.size))}
    {
        pool.close();
        for (const gca::frame *frame : frames)
            fuse(tsdf, buffers, *frame, m.range, fused_from, stream, name);
    }

    fused_wall(const fused_wall &) = delete;
    fused_wall &operator=(const fused_wall &) = delete;
};

/// Gives a pose turned by 180 degrees about its camera's y axis: the same camera looking back.
/// @param pose [in] the pose, camera frame to the models' frame.
/// @return pose with the camera's x and z axes negated.
Eigen::Isometry3d turned_back(const Eigen::Isometry3d &pose)
{
    return pose * Eigen::AngleAxisd(std::numbers::pi, Eigen::Vector3d::UnitY());
}

/// The wall: fused once from the identity at the tabletop object's live voxel size; cast from the
/// identity, every pixel at least wall_border_px inside the picture is valid, at the wall's depth,
/// its normal along -z and its intensity the wall color's; cast from behind the wall, 0.4 m away
/// looking back at it, every pixel is invalid, the wall being a back face there; cast from the
/// identity looking back, at no block, every pixel is invalid.
/// @param stream [in] the stream.
void check_wall(cudaStream_t stream)
{
    const gca::frame wall = flat_wall(wall_mm, wall_color);
    fused_wall w("wall", {&wall}, stream);
    const model_case &m = w.m;
    const gca::intrinsics &k = small_description.intrinsics;
    const gca::sparse_tsdf &tsdf = w.tsdf;
    gca::model_surface &surface = w.surfaces[0];
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();

    const std::string front = "a wall, from its pose";
    const cast_pictures seen = cast(tsdf, identity, k, m.range, surface, stream, front);
    check_levels(front, seen, k, m.range);
    const cast_picture &p = seen[0];
    const double expected_intensity =
        luma_of({double{wall_color[0]}, double{wall_color[1]}, double{wall_color[2]}});
    int missed = 0;
    int without_normal = 0;
    int other_color = 0;
    max_error depth;
    max_error normal;
    max_error intensity;
    for (std::uint32_t v = wall_border_px; v + wall_border_px < p.size.height; v++)
        for (std::uint32_t u = wall_border_px; u + wall_border_px < p.size.width; u++)
        {
            const std::size_t i = gca_test::index_of(p.size, u, v);
            if (p.valid.empty() || p.valid[i] == 0)
            {
                missed++;
                continue;
            }
            depth.add(std::abs(double{p.points[i].z} - 1e-3 * wall_mm));
            without_normal += gca_test::is_zero(p.normals[i]);
            normal.add(angle_deg(to_double(p.normals[i]), Eigen::Vector3d(0.0, 0.0, -1.0)));
            intensity.add(std::abs(double{p.intensity[i]} - expected_intensity));
            other_color += i >= p.colors.size() || p.colors[i].x != wall_color[0] ||
                           p.colors[i].y != wall_color[1] || p.colors[i].z != wall_color[2];
        }
    std::printf("%s: %zu valid pixels of %zu\n", front.c_str(), valid_count(p), p.valid.size());
    check_within((front + ": pixels inside the border without the wall").c_str(), missed, 0.0);
    check_within((front + ": pixels inside the border without a normal").c_str(), without_normal,
                 0.0);
    gca_test::check_max_error(front + ": depth's error, m", depth, wall_depth_tolerance_m);
    gca_test::check_max_error(front + ": normal's angle from -z, deg", normal,
                              wall_normal_tolerance_deg);
    gca_test::check_max_error(front + ": intensity's error", intensity, intensity_float_tolerance);
    // Every voxel holds the wall's color, so every interpolation of 8 rounds back to it.
    check_within((front + ": pixels inside the border not of the wall's color").c_str(),
                 other_color, 0.0);

    const std::string behind = "a wall, from behind";
    const Eigen::Isometry3d back =
        turned_back(Eigen::Translation3d(0.0, 0.0, 1e-3 * wall_mm + 0.4) * identity);
    const cast_pictures from_behind = cast(tsdf, back, k, m.range, surface, stream, behind);
    check_levels(behind, from_behind, k, m.range);
    check_within((behind + ": valid pixels").c_str(),
                 static_cast<double>(valid_count(from_behind[0])), 0.0);

    const std::string away = "a wall, looking away from it";
    const cast_pictures nothing =
        cast(tsdf, turned_back(identity), k, m.range, surface, stream, away);
    check_levels(away, nothing, k, m.range);
    check_within((away + ": valid pixels").c_str(), static_cast<double>(valid_count(nothing[0])),
                 0.0);
    check(w.pool.refused_while_closed() == 0, "wall: the closed pool asked for nothing");
}

/// The column where the half wall's frame ends: its pixels left of it see the wall, the others
/// have no depth; next to the principal point's column, 317.41, so that the rays near the edge run
/// almost along z [?: a choice].
constexpr std::uint32_t half_wall_column = 320;

/// The pixels left of the half wall's edge within which a valid pixel without a normal lies: 3,
/// as a voxel of 2 mm spans 1.2 px at 0.5 m and a normal's central differences reach voxels up to
/// 2 voxels past its crossing, the far corners of the cell one voxel on [S: computed, 306.18 px *
/// 2 mm / 0.5 m] [?: a choice, above the 2.4 px the two give].
constexpr std::uint32_t half_wall_reach_px = 3;

/// A wall whose frame's right half has no depth, fused once from the identity: the voxels that
/// project into the right half stay unobserved. Cast from the identity, every row inside the
/// picture's border has, just left of the edge, a valid pixel without a normal: its crossing's 8
/// voxels are observed, but the central differences of its normal reach an unobserved one
/// (docs/methods/sparse-tsdf.md, "The ray cast"); no pixel without a normal lies farther left, and
/// none right of the edge is valid.
/// @param stream [in] the stream.
void check_half_wall(cudaStream_t stream)
{
    const gca::frame half = gca_test::wall_frame(
        [](std::uint32_t u, std::uint32_t) {
            return u < half_wall_column ? wall_mm : std::uint16_t{0};
        },
        wall_color);
    fused_wall w("half wall", {&half}, stream);
    const model_case &m = w.m;
    const gca::intrinsics &k = small_description.intrinsics;
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    const cast_pictures pictures =
        cast(w.tsdf, identity, k, m.range, w.surfaces[0], stream, m.name);
    check_levels(m.name, pictures, k, m.range);
    const cast_picture &p = pictures[0];
    int rows_without = 0;
    int far_from_edge = 0;
    int right_valid = 0;
    for (std::uint32_t v = wall_border_px; v + wall_border_px < p.size.height && !p.valid.empty();
         v++)
    {
        bool found = false;
        for (std::uint32_t u = wall_border_px; u + wall_border_px < p.size.width; u++)
        {
            const std::size_t i = gca_test::index_of(p.size, u, v);
            if (p.valid[i] == 0)
                continue;
            right_valid += u >= half_wall_column;
            if (!gca_test::is_zero(p.normals[i]))
                continue;
            found = true;
            far_from_edge += u + half_wall_reach_px < half_wall_column;
        }
        rows_without += !found;
    }
    std::printf("%s: %zu valid pixels\n", m.name.c_str(), valid_count(p));
    check_within((m.name + ": rows without a valid pixel without a normal at the edge").c_str(),
                 rows_without, 0.0);
    check_within((m.name + ": pixels without a normal away from the edge").c_str(), far_from_edge,
                 0.0);
    check_within((m.name + ": valid pixels right of the edge").c_str(), right_valid, 0.0);
    check(w.pool.refused_while_closed() == 0, "half wall: the closed pool asked for nothing");
}

/// What a cast of a wall gives at one pixel, where its ray meets the wall's plane inside the part
/// the fused frame saw.
struct wall_expectation
{
    bool sees = false; ///< whether the ray meets the plane wall_border_px or more inside the frame,
                       ///< within the depth range
    double depth_m = 0.0;   ///< the depth of that point along the cast's camera z, in m
    Eigen::Vector3d normal; ///< the wall's normal, -z of the fused frame's camera, in the cast's
                            ///< camera frame
};

/// The margin a wall's expected point keeps from the depth range's limits: 1 mm, so that float's
/// rounding of a depth at a limit cannot decide whether the cast takes it [?: a choice].
constexpr double range_margin_m = 1e-3;

/// Gives what a pixel of a cast of the wall sees: its ray from the cast's pose meets the plane
/// z = wall_mm of the camera the wall was fused from; the point counts when its depth lies within
/// the depth range, range_margin_m from either limit, and it projects into the fused frame, 640x400
/// with the camera's intrinsics, wall_border_px or more inside its border.
/// @param pose [in] the cast's pose, camera frame to the frame of the camera the wall was fused
///        from: the world frame for a wall fused at the identity.
/// @param k [in] the cast's intrinsics.
/// @param range [in] the cast's depth range.
/// @param u [in] the pixel's column.
/// @param v [in] its row.
/// @return the expectation.
wall_expectation wall_seen(const Eigen::Isometry3d &pose, const gca::intrinsics &k,
                           const gca::organized_points_parameters &range, std::uint32_t u,
                           std::uint32_t v)
{
    wall_expectation e;
    const Eigen::Vector3d ray = pose.linear() * gca_test::pixel_ray(k, u, v);
    if (!(std::abs(ray.z()) > 1e-9))
        return e;
    const double s = (1e-3 * wall_mm - pose.translation().z()) / ray.z();
    if (!(s >= range.near_m + range_margin_m && s <= range.far_m - range_margin_m))
        return e;
    const Eigen::Vector3d hit = pose.translation() + s * ray;
    const gca::intrinsics &fused = small_description.intrinsics;
    const double fu = fused.fx * hit.x() / hit.z() + fused.cx;
    const double fv = fused.fy * hit.y() / hit.z() + fused.cy;
    const double border = wall_border_px;
    e.sees = fu >= border && fu <= small_description.size.width - 1.0 - border && fv >= border &&
             fv <= small_description.size.height - 1.0 - border;
    e.depth_m = s;
    e.normal = pose.linear().transpose() * Eigen::Vector3d(0.0, 0.0, -1.0);
    return e;
}

/// A cast's level 0 against the wall it sees, over the pixels taken whose ray meets the part of the
/// wall the frame saw (wall_seen()).
struct wall_comparison
{
    std::size_t seen = 0; ///< those pixels
    int missed = 0;       ///< of them, those invalid or without a normal
    max_error depth;      ///< the largest error of the others' depths, in m
    max_error normal;     ///< the largest angle of their normals from the wall's, in degrees
};

/// Compares a cast's level 0 with the wall pixel by pixel.
/// @param p [in] the level.
/// @param pose [in] the cast's pose in the frame of the camera the wall was fused from
///        (wall_seen()).
/// @param range [in] the cast's depth range.
/// @param taken [in] which pixels to compare: called with the column and the row, true for those.
/// @return the comparison.
template <typename Pixels>
wall_comparison compare_with_wall(const cast_picture &p, const Eigen::Isometry3d &pose,
                                  const gca::organized_points_parameters &range, Pixels taken)
{
    wall_comparison c;
    for (std::uint32_t v = 0; v < p.size.height && !p.valid.empty(); v++)
        for (std::uint32_t u = 0; u < p.size.width; u++)
        {
            if (!taken(u, v))
                continue;
            const wall_expectation e = wall_seen(pose, p.k, range, u, v);
            if (!e.sees)
                continue;
            c.seen++;
            const std::size_t i = gca_test::index_of(p.size, u, v);
            if (p.valid[i] == 0 || gca_test::is_zero(p.normals[i]))
            {
                c.missed++;
                continue;
            }
            c.depth.add(std::abs(double{p.points[i].z} - e.depth_m));
            c.normal.add(angle_deg(to_double(p.normals[i]), e.normal));
        }
    return c;
}

/// Takes every pixel (compare_with_wall()).
/// @param u [in] the pixel's column.
/// @param v [in] its row.
/// @return true.
bool every_pixel(std::uint32_t u, std::uint32_t v)
{
    static_cast<void>(u);
    static_cast<void>(v);
    return true;
}

/// Prints how many pixels see the wall and checks a comparison: at least a number of pixels see
/// it, each holds the wall and its normal, its depth and its normal within their tolerances.
/// @param what [in] the pixels compared, printed and named in the check lines.
/// @param c [in] the comparison.
/// @param least_seen [in] the fewest pixels that must see the wall.
/// @param depth_tolerance_m [in] the largest depth's error that passes, in m.
/// @param normal_tolerance_deg [in] the largest normal's angle that passes, in degrees.
void check_wall_comparison(const std::string &what, const wall_comparison &c,
                           std::size_t least_seen, double depth_tolerance_m,
                           double normal_tolerance_deg)
{
    std::printf("%s: %zu pixels see the fused wall\n", what.c_str(), c.seen);
    check(c.seen >= least_seen, (what + ": enough pixels see the wall").c_str());
    check_within((what + ": pixels that see the wall without it or its normal").c_str(), c.missed,
                 0.0);
    gca_test::check_max_error(what + ": depth's error, m", c.depth, depth_tolerance_m);
    gca_test::check_max_error(what + ": normal's angle, deg", c.normal, normal_tolerance_deg);
}

/// The colored wall's depth: 330 mm, where a voxel of the tabletop object's live 2 mm spans 1.86 px
/// of the frame [S: computed, 306.18 px x 2 mm / 330 mm], so that a cell's two sides along a row or
/// a column differ by about 7 of its colors' levels; inside the tests' range, 0.30 m to 1.0 m
/// (stage_frames.hpp, measured_range_of()), its near limit 30 mm away [?: a choice].
constexpr std::uint16_t colored_wall_mm = 330;

/// The colored wall's slope: 4 levels a px of the frame [?: a choice, steep enough that a color
/// read from the cell's other side errs by several levels].
constexpr double colored_slope_per_px = 4.0;

/// Gives the colored wall's color where the frame it was fused from sees it: red a triangle wave
/// along the column and green along the row, from 28 to 228 at colored_slope_per_px, a period of
/// 100 px, and blue 128; whole numbers at whole pixels.
/// @param u [in] the column, in px of the fused frame, any real value.
/// @param v [in] the row, in px.
/// @return red, green and blue, 28 to 228.
std::array<double, 3> colored_wall_color(double u, double v)
{
    const auto wave = [](double x) {
        const double t = std::fmod(colored_slope_per_px * x, 400.0);
        return 28.0 + std::abs((t < 0.0 ? t + 400.0 : t) - 200.0);
    };
    return {wave(u), wave(v), 128.0};
}

/// The colored wall's tolerances of a color's largest channel's error from the wall's color where
/// the cast's point lies, in levels of 0 to 255: the median and the 90th percentile, 1.5 and 1.45
/// times the largest of the three casts, 1 and 1.721 (99th percentile 3.303, largest 4.385, where
/// a triangle wave turns) [M 2026-11-17: this test]. The voxels' colors are those of the pixels
/// their centres project to, up to half a pixel away, 2 levels at the slope [S: computed].
constexpr double colored_median_tolerance = 1.5;
constexpr double colored_p90_tolerance = 2.5;

/// The colors of a wall whose color changes along its rows and columns (colored_wall_color()),
/// fused once from the identity at the tabletop object's live voxel size: cast from its pose and
/// from two poses turned from it, at 640x400, every valid pixel whose ray meets the part of the
/// wall the frame saw, away from its border, holds the wall's color where its point lies within
/// measured tolerances. The voxels hold the colors of the pixels their centres project to, and the
/// cast interpolates the 8 around its point (docs/methods/sparse-tsdf.md, "The ray cast"); a color
/// read from another side of the cell errs by up to the cell's change of color.
/// @param stream [in] the stream.
void check_colored_wall(cudaStream_t stream)
{
    const gca::frame wall = gca_test::built_frame(
        small_description, [](std::uint32_t, std::uint32_t) { return colored_wall_mm; },
        [](std::uint32_t u, std::uint32_t v) {
            const std::array<double, 3> c = colored_wall_color(u, v);
            return std::array<std::uint8_t, 3>{static_cast<std::uint8_t>(c[0]),
                                               static_cast<std::uint8_t>(c[1]),
                                               static_cast<std::uint8_t>(c[2])};
        });
    fused_wall w("colored wall", {&wall}, stream);
    const model_case &m = w.m;
    const gca::intrinsics &fused = small_description.intrinsics;
    const std::array<std::pair<std::string, Eigen::Isometry3d>, 3> poses = {
        std::pair{std::string("from its pose"), Eigen::Isometry3d::Identity()},
        std::pair{std::string("turned 20 deg about x"),
                  make_transform(Eigen::Vector3d::UnitX(), -20.0, Eigen::Vector3d::Zero())},
        std::pair{std::string("shifted and turned 20 deg about (2, -1, 0)"),
                  make_transform(Eigen::Vector3d(2.0, -1.0, 0.0), 20.0,
                                 Eigen::Vector3d(0.03, -0.02, -0.02))}};
    for (const auto &[pose_name, pose] : poses)
    {
        const std::string what = m.name + ", " + pose_name;
        const cast_pictures pictures =
            cast(w.tsdf, pose, fused, m.range, w.surfaces[0], stream, what);
        check_levels(what, pictures, fused, m.range);
        const cast_picture &p = pictures[0];
        std::vector<double> errors;
        std::size_t missed = 0;
        for (std::uint32_t v = 0; v < p.size.height && !p.colors.empty(); v++)
            for (std::uint32_t u = 0; u < p.size.width; u++)
            {
                // Where the pixel's ray meets the wall's plane, and where the fused frame saw it.
                const Eigen::Vector3d ray = pose.linear() * gca_test::pixel_ray(p.k, u, v);
                const double s = (1e-3 * colored_wall_mm - pose.translation().z()) / ray.z();
                if (!(s >= m.range.near_m + range_margin_m && s <= m.range.far_m - range_margin_m))
                    continue;
                const Eigen::Vector3d hit = pose.translation() + s * ray;
                const double fu = fused.fx * hit.x() / hit.z() + fused.cx;
                const double fv = fused.fy * hit.y() / hit.z() + fused.cy;
                const double border = wall_border_px;
                if (!(fu >= border && fu <= p.size.width - 1.0 - border && fv >= border &&
                      fv <= p.size.height - 1.0 - border))
                    continue;
                const std::size_t i = gca_test::index_of(p.size, u, v);
                if (p.valid[i] == 0)
                {
                    missed++;
                    continue;
                }
                // The color where the cast's own point lies, as the fused frame saw it there.
                const Eigen::Vector3d point = pose * to_double(p.points[i]);
                const std::array<double, 3> known =
                    colored_wall_color(fused.fx * point.x() / point.z() + fused.cx,
                                       fused.fy * point.y() / point.z() + fused.cy);
                errors.push_back(largest_channel_error(p.colors[i], known));
            }
        std::printf("%s: %zu pixels see the fused wall, %zu of them invalid\n", what.c_str(),
                    errors.size() + missed, missed);
        check(errors.size() > p.colors.size() / 4, (what + ": enough pixels see the wall").c_str());
        check_within((what + ": pixels that see the wall invalid").c_str(),
                     static_cast<double>(missed), 0.0);
        if (errors.empty())
            continue;
        std::printf("%s: color's largest channel's error: median %.4g, 90th percentile %.4g, 99th "
                    "percentile %.4g, largest %.4g\n",
                    what.c_str(), quantile(errors, 0.5), quantile(errors, 0.9),
                    quantile(errors, 0.99), quantile(errors, 1.0));
        check_within((what + ": color's error, median").c_str(), quantile(errors, 0.5),
                     colored_median_tolerance);
        check_within((what + ": color's error, 90th percentile").c_str(), quantile(errors, 0.9),
                     colored_p90_tolerance);
    }
    check(w.pool.refused_while_closed() == 0, "colored wall: the closed pool asked for nothing");
}

/// The wall fused once from the identity, cast from poses turned and shifted from it, at both
/// stream sizes: every pixel whose ray meets the part of the wall the frame saw, away from its
/// border, holds the wall at the depth along its ray within the wall's tolerance, its normal the
/// wall's within its angle. The blocks are seen at a slant, so a tile's interval of depths spans
/// blocks a ray enters late; the cast steps past the blocks before the interval and ends past it
/// (docs/methods/sparse-tsdf.md, "The ray cast", "The depth interval").
/// @param stream [in] the stream.
void check_wall_turned(cudaStream_t stream)
{
    const gca::frame wall = flat_wall(wall_mm, wall_color);
    fused_wall w("turned wall", {&wall}, stream);
    const model_case &m = w.m;
    // Turns about the camera's axes and a shift [?: a choice]: each keeps a large part of the
    // fused wall in view.
    const std::array<std::pair<std::string, Eigen::Isometry3d>, 4> poses = {
        std::pair{std::string("turned 25 deg about y"),
                  make_transform(Eigen::Vector3d::UnitY(), 25.0, Eigen::Vector3d::Zero())},
        std::pair{std::string("turned 20 deg about x"),
                  make_transform(Eigen::Vector3d::UnitX(), -20.0, Eigen::Vector3d::Zero())},
        std::pair{std::string("turned 30 deg about (1, 2, 3)"),
                  make_transform(Eigen::Vector3d(1.0, 2.0, 3.0), 30.0, Eigen::Vector3d::Zero())},
        std::pair{std::string("shifted and turned 20 deg about (2, -1, 0)"),
                  make_transform(Eigen::Vector3d(2.0, -1.0, 0.0), 20.0,
                                 Eigen::Vector3d(0.08, -0.05, 0.12))}};
    const std::array<gca::frame_description, 2> sizes = {small_description, large_description};
    for (std::size_t s = 0; s < sizes.size(); s++)
        for (const auto &[pose_name, pose] : poses)
        {
            const std::string what =
                m.name + ", " + pose_name + ", " + gca::size_text(sizes[s].size);
            const cast_pictures pictures =
                cast(w.tsdf, pose, sizes[s].intrinsics, m.range, w.surfaces[s], stream, what);
            check_levels(what, pictures, sizes[s].intrinsics, m.range);
            const cast_picture &p = pictures[0];
            // More than a quarter of the picture sees the wall.
            check_wall_comparison(what, compare_with_wall(p, pose, m.range, every_pixel),
                                  p.valid.size() / 4 + 1, wall_depth_tolerance_m,
                                  wall_normal_tolerance_deg);
        }
    check(w.pool.refused_while_closed() == 0, "turned wall: the closed pool asked for nothing");
}

/// A coordinate of a cast's position, in m, which every crossing on a picture line whose rays keep
/// that coordinate shares, and where the normal's end one voxel after the crossing rounds into
/// another cell than the crossing's moved by one: at the tabletop object's live 2 mm voxels,
/// 1 / v = 499.99997 in float, g = fma(0.003000000026077032, 1 / v, -1/2) = 1 - 2^-24 on the grid
/// of the voxels' centres, where g + 1 rounds to 2, a tie to the even one. The cast's g is one fma
/// [B 2026-11-15: `cuobjdump -sass`, FFMA with -0.5]; with the product rounded first it would be
/// 1 - 2^-23, and the line would take no fallback [S: computed in exact rationals and float32].
constexpr double slanted_after_m = 0.003000000026077032;

/// Likewise for the end one voxel before the crossing: g = -3 - 2^-22, where g - 1 rounds to -4, a
/// tie to the even one, so that the cell before is the crossing's own moved by 0, not 1; with the
/// fma or without [S: computed in exact rationals and float32].
constexpr double slanted_before_m = -0.005000000819563866;

/// The slanted wall's tolerances: 1.65 to 1.96 times the largest errors over its three
/// orientations, 0.510 to 0.593 µm and 0.0119 to 0.0121 degrees at 640x400 and 1280x800, its
/// distances rounded to half floats; on the column and the row through the principal point 0.456 to
/// 0.461 µm and 0.0117 to 0.0120 degrees [M 2026-11-15: this test; the same pixels as the cast that
/// interpolated the six ends, bit for bit (docs/methods/sparse-tsdf.md, "On the GPU")].
constexpr double slanted_depth_tolerance_m = 1e-6;
constexpr double slanted_normal_tolerance_deg = 0.02;

/// The least number of pixels of the column and the row through the principal point that see the
/// slanted wall at 640x400, twice as many at 1280x800 [?: a choice, below the 798 and 1595
/// measured in each orientation (M 2026-11-15: this test)].
constexpr std::size_t slanted_least_on_lines = 700;

/// Gives a stream size's intrinsics with the principal point at the picture's centre, a whole
/// pixel.
/// @param d [in] the stream size and its intrinsics.
/// @return the intrinsics with cx = width / 2 and cy = height / 2, in px.
gca::intrinsics centred(const gca::frame_description &d)
{
    gca::intrinsics k = d.intrinsics;
    k.cx = static_cast<float>(d.size.width / 2);
    k.cy = static_cast<float>(d.size.height / 2);
    return k;
}

/// An orientation of the slanted wall's case: the turn of the camera that casts it, and of the
/// camera that fused it with it, and the models' axes that the rays of the column and of the row
/// through the principal point keep, a ray of the column having x = 0 and one of the row y = 0 in
/// the camera frame [S: (u - cx) / fx].
struct slanted_view
{
    std::string name;        ///< the orientation, named in the check lines
    Eigen::Quaterniond turn; ///< camera frame to the models' frame; its matrix exact in float
    int column_axis;         ///< the models' axis the column's rays keep: 0, 1 or 2
    int row_axis;            ///< the models' axis the row's rays keep, another
};

/// The wall fused once from a pose turned by 20 degrees about (1, -1, 0) at the tabletop object's
/// live voxel size, so that its distances change along every axis of the models' frame, in three
/// orientations of the scene, each with its own model: the camera looking along the models' z, x
/// and y, turned by the identity and by 120 degrees about (1, 1, 1) either way, whose matrices hold
/// only 0 and 1 [S: the quaternions (1/2, 1/2, 1/2, 1/2) and (1/2, -1/2, -1/2, -1/2)]. Each is cast
/// from its turn at a position whose column axis is slanted_after_m and whose row axis is
/// slanted_before_m, at both stream sizes with the principal point at the picture's centre
/// [?: a choice], so that every crossing on the column takes the six interpolations of the normal's
/// ends for the end after it and every crossing on the row for the end before it, along x and y,
/// y and z, z and x: all six of the fallback's conditions. Every pixel whose ray meets the part of
/// the wall the frame saw, away from its border, holds the wall at the depth along its ray, its
/// normal the wall's, each within its tolerance; and so the column and the row.
/// @param stream [in] the stream.
void check_slanted_wall(cudaStream_t stream)
{
    const gca::frame wall = flat_wall(wall_mm, wall_color);
    const Eigen::Isometry3d slant =
        make_transform(Eigen::Vector3d(1.0, -1.0, 0.0), 20.0, Eigen::Vector3d::Zero());
    const std::array<slanted_view, 3> views = {
        slanted_view{"looking along z", Eigen::Quaterniond::Identity(), 0, 1},
        slanted_view{"looking along x", Eigen::Quaterniond(0.5, 0.5, 0.5, 0.5), 1, 2},
        slanted_view{"looking along y", Eigen::Quaterniond(0.5, -0.5, -0.5, -0.5), 2, 0}};
    const std::array<gca::frame_description, 2> sizes = {small_description, large_description};
    for (const slanted_view &view : views)
    {
        const Eigen::Isometry3d turned(view.turn);
        const Eigen::Isometry3d fused_from = turned * slant;
        fused_wall w("slanted wall, " + view.name, {&wall}, stream, fused_from);
        const model_case &m = w.m;
        Eigen::Vector3d position = Eigen::Vector3d::Zero();
        position[view.column_axis] = slanted_after_m;
        position[view.row_axis] = slanted_before_m;
        const Eigen::Isometry3d pose = Eigen::Translation3d(position) * turned;
        const Eigen::Isometry3d in_wall = fused_from.inverse() * pose;
        for (std::size_t s = 0; s < sizes.size(); s++)
        {
            const std::string what = m.name + ", " + gca::size_text(sizes[s].size);
            const gca::intrinsics k = centred(sizes[s]);
            const cast_pictures pictures =
                cast(w.tsdf, pose, k, m.range, w.surfaces[s], stream, what);
            check_levels(what, pictures, k, m.range);
            const cast_picture &p = pictures[0];
            const std::uint32_t column = sizes[s].size.width / 2;
            const std::uint32_t row = sizes[s].size.height / 2;
            check_wall_comparison(what, compare_with_wall(p, in_wall, m.range, every_pixel),
                                  p.valid.size() / 4 + 1, slanted_depth_tolerance_m,
                                  slanted_normal_tolerance_deg);
            check_wall_comparison(
                what + ", the column and the row through the principal point",
                compare_with_wall(p, in_wall, m.range,
                                  [column, row](std::uint32_t u, std::uint32_t v) {
                                      return u == column || v == row;
                                  }),
                slanted_least_on_lines << s, slanted_depth_tolerance_m,
                slanted_normal_tolerance_deg);
        }
        check(w.pool.refused_while_closed() == 0,
              (m.name + ": the closed pool asked for nothing").c_str());
    }
}

/// The patch's depth in front of the wall, and its pixels in the fused frame: columns 250 to 389
/// and rows 150 to 249 [?: a choice: borders that fall inside tiles of 8 x 8 pixels].
constexpr std::uint16_t patch_mm = 400;
constexpr std::uint32_t patch_left = 250;
constexpr std::uint32_t patch_right = 390;
constexpr std::uint32_t patch_top = 150;
constexpr std::uint32_t patch_bottom = 250;

/// A patch 100 mm in front of the wall, fused once from the identity and cast from it: the tiles
/// along the patch's border see the blocks of both surfaces, and their pixels away from that border
/// hold the patch or, behind it, the wall at its depth, with the normal along -z. The rows and
/// columns within wall_border_px of the patch's border are left out: there the patch's band and
/// its silhouette's voxels meet.
/// @param stream [in] the stream.
void check_patch_before_wall(cudaStream_t stream)
{
    const gca::intrinsics &k = small_description.intrinsics;
    const auto in_patch = [](std::uint32_t u, std::uint32_t v) {
        return u >= patch_left && u < patch_right && v >= patch_top && v < patch_bottom;
    };
    // Whether a pixel's tile lies within the patch's blocks' reach: a block of 16 mm at 384 mm
    // projects 12.75 px past the patch's border [S: computed, 306.18 px * 16 mm / 384 mm].
    const auto tile_meets_patch_blocks = [](std::uint32_t u, std::uint32_t v) {
        constexpr std::uint32_t reach = 13;
        const std::uint32_t x = u / gca::depth_tile_px * gca::depth_tile_px;
        const std::uint32_t y = v / gca::depth_tile_px * gca::depth_tile_px;
        return x < patch_right + reach && x + gca::depth_tile_px + reach > patch_left &&
               y < patch_bottom + reach && y + gca::depth_tile_px + reach > patch_top;
    };
    const gca::frame frame = gca_test::wall_frame(
        [&in_patch](std::uint32_t u, std::uint32_t v) {
            return in_patch(u, v) ? patch_mm : wall_mm;
        },
        wall_color);
    fused_wall w("patch before a wall", {&frame}, stream);
    const model_case &m = w.m;
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    const cast_pictures pictures =
        cast(w.tsdf, identity, k, m.range, w.surfaces[0], stream, m.name);
    check_levels(m.name, pictures, k, m.range);
    const cast_picture &p = pictures[0];
    int missed = 0;
    std::size_t behind = 0;
    max_error depth;
    max_error normal;
    for (std::uint32_t v = wall_border_px; v + wall_border_px < p.size.height && !p.valid.empty();
         v++)
        for (std::uint32_t u = wall_border_px; u + wall_border_px < p.size.width; u++)
        {
            const bool near_border =
                u + wall_border_px >= patch_left && u < patch_right + wall_border_px &&
                v + wall_border_px >= patch_top && v < patch_bottom + wall_border_px &&
                !(u >= patch_left + wall_border_px && u + wall_border_px < patch_right &&
                  v >= patch_top + wall_border_px && v + wall_border_px < patch_bottom);
            if (near_border)
                continue;
            const std::size_t i = gca_test::index_of(p.size, u, v);
            if (p.valid[i] == 0 || gca_test::is_zero(p.normals[i]))
            {
                missed++;
                continue;
            }
            const bool patch = in_patch(u, v);
            behind += !patch && tile_meets_patch_blocks(u, v);
            depth.add(std::abs(double{p.points[i].z} - 1e-3 * (patch ? patch_mm : wall_mm)));
            normal.add(angle_deg(to_double(p.normals[i]), Eigen::Vector3d(0.0, 0.0, -1.0)));
        }
    std::printf("%s: %zu valid pixels, %zu of the wall in tiles the patch's blocks reach\n",
                m.name.c_str(), valid_count(p), behind);
    check(behind > 0, (m.name + ": pixels of the wall in tiles the patch's blocks reach").c_str());
    check_within((m.name + ": pixels away from the borders without a surface or a normal").c_str(),
                 missed, 0.0);
    gca_test::check_max_error(m.name + ": depth's error, m", depth, wall_depth_tolerance_m);
    gca_test::check_max_error(m.name + ": normal's angle from -z, deg", normal,
                              wall_normal_tolerance_deg);
    check(w.pool.refused_while_closed() == 0,
          "patch before a wall: the closed pool asked for nothing");
}

/// How far in front of the wall the camera lies inside its band: 2.5 mm, so that the block of
/// voxels around the camera's centre is held and reaches behind the camera's plane
/// [?: a choice].
constexpr double inside_band_m = 0.0025;

/// The wall fused once from the identity, cast from a camera inside its band, 2.5 mm in front of
/// it, with the depth range starting at 0: the march starts at the camera's centre, in a held block
/// that reaches behind the camera's plane and so covers the whole picture
/// (docs/methods/sparse-tsdf.md, "The ray cast", "The depth interval"); every pixel inside the
/// picture's border holds the wall 2.5 mm away, its normal along -z.
/// @param stream [in] the stream.
void check_inside_band(cudaStream_t stream)
{
    const gca::frame wall = flat_wall(wall_mm, wall_color);
    fused_wall w("inside a wall's band", {&wall}, stream);
    const model_case &m = w.m;
    const gca::intrinsics &k = small_description.intrinsics;
    const gca::organized_points_parameters from_centre = {0.0f, m.range.far_m};
    const Eigen::Isometry3d inside(Eigen::Translation3d(0.0, 0.0, 1e-3 * wall_mm - inside_band_m));
    const cast_pictures pictures =
        cast(w.tsdf, inside, k, from_centre, w.surfaces[0], stream, m.name);
    check_levels(m.name, pictures, k, from_centre);
    const cast_picture &p = pictures[0];
    int missed = 0;
    max_error depth;
    max_error normal;
    for (std::uint32_t v = wall_border_px; v + wall_border_px < p.size.height && !p.valid.empty();
         v++)
        for (std::uint32_t u = wall_border_px; u + wall_border_px < p.size.width; u++)
        {
            const std::size_t i = gca_test::index_of(p.size, u, v);
            if (p.valid[i] == 0 || gca_test::is_zero(p.normals[i]))
            {
                missed++;
                continue;
            }
            depth.add(std::abs(double{p.points[i].z} - inside_band_m));
            normal.add(angle_deg(to_double(p.normals[i]), Eigen::Vector3d(0.0, 0.0, -1.0)));
        }
    std::printf("%s: %zu valid pixels\n", m.name.c_str(), valid_count(p));
    check_within((m.name + ": pixels inside the border without the wall or its normal").c_str(),
                 missed, 0.0);
    gca_test::check_max_error(m.name + ": depth's error, m", depth, wall_depth_tolerance_m);
    gca_test::check_max_error(m.name + ": normal's angle from -z, deg", normal,
                              wall_normal_tolerance_deg);
    check(w.pool.refused_while_closed() == 0,
          "inside a wall's band: the closed pool asked for nothing");
}

/// The near patch of check_widened_tiles(): its pixels in the fused frame, columns 97 to 142 and
/// rows 97 to 134, at 350 mm, 150 mm in front of the wall [?: a choice: a corner of the picture,
/// borders inside tiles of 8 x 8 pixels].
constexpr std::uint32_t near_left = 97;
constexpr std::uint32_t near_right = 143;
constexpr std::uint32_t near_top = 97;
constexpr std::uint32_t near_bottom = 135;
constexpr std::uint16_t near_mm = 350;

/// How far from the near patch's pixels, at 640x400, a ray may come within reach of its observed
/// voxels: 8 px [?: a choice], above the 2 voxels a sample reaches at 350 mm, 3.5 px [S: computed,
/// 306.18 px * 4 mm / 350 mm]; at 1280x800 twice as many.
constexpr std::uint32_t near_reach_px = 8;

/// How far past the near patch's pixels its blocks project, at 640x400: 15 px, a block of 16 mm at
/// 336 mm, 14.6 px [S: computed, 306.18 px * 16 mm / 336 mm]; a pixel between near_reach_px and
/// this lies where the patch's blocks may widen its tile's interval.
constexpr std::uint32_t near_blocks_px = 15;

/// The wall cast from the identity through two models: the wall alone, and the wall with a patch in
/// front of it fused from a second frame whose other pixels have no depth, so that its voxels
/// change none of the wall's. The patch's blocks widen the depth intervals of the tiles they cover:
/// there the second model's march looks up blocks from the patch's depth on, where the first steps
/// past them to the wall (docs/methods/sparse-tsdf.md, "The ray cast", "The depth interval").
/// Every pixel of every level whose ray stays out of the patch's reach is the same bit for bit in
/// both casts, its point, mark, normal and intensity, at both stream sizes.
/// @param stream [in] the stream.
void check_widened_tiles(cudaStream_t stream)
{
    const gca::frame wall = flat_wall(wall_mm, wall_color);
    const gca::frame patch = gca_test::wall_frame(
        [](std::uint32_t u, std::uint32_t v) {
            return u >= near_left && u < near_right && v >= near_top && v < near_bottom
                       ? near_mm
                       : std::uint16_t{0};
        },
        wall_color);
    fused_wall alone("the wall alone", {&wall}, stream);
    fused_wall widened("the wall behind a near patch", {&wall, &patch}, stream);
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    const std::array<gca::frame_description, 2> sizes = {small_description, large_description};
    for (std::size_t s = 0; s < sizes.size(); s++)
    {
        const std::string what = "widened tiles, " + gca::size_text(sizes[s].size);
        const gca::intrinsics &k = sizes[s].intrinsics;
        const cast_pictures a =
            cast(alone.tsdf, identity, k, alone.m.range, alone.surfaces[s], stream, what);
        const cast_pictures b =
            cast(widened.tsdf, identity, k, widened.m.range, widened.surfaces[s], stream, what);
        // At 1280x800 a pixel of the fused frame spans two.
        const std::uint32_t scale = static_cast<std::uint32_t>(s) + 1;
        // Whether a pixel of level 0 lies within a distance of the patch's pixels.
        const auto within = [scale](std::uint32_t u, std::uint32_t v, std::uint32_t px) {
            return u + scale * px >= scale * near_left && u < scale * (near_right + px) &&
                   v + scale * px >= scale * near_top && v < scale * (near_bottom + px);
        };
        std::size_t compared = 0;
        std::size_t near_blocks = 0;
        std::size_t valid = 0;
        int differing = 0;
        for (std::size_t l = 0; l < levels; l++)
        {
            const cast_picture &x = a[l];
            const cast_picture &y = b[l];
            if (x.points.size() != y.points.size() || x.points.empty())
                continue;
            for (std::uint32_t v = 0; v < x.size.height; v++)
                for (std::uint32_t u = 0; u < x.size.width; u++)
                {
                    const std::uint32_t u0 = u << l;
                    const std::uint32_t v0 = v << l;
                    if (within(u0, v0, near_reach_px))
                        continue;
                    const std::size_t i = gca_test::index_of(x.size, u, v);
                    compared++;
                    near_blocks += l == 0 && within(u0, v0, near_blocks_px);
                    valid += l == 0 && x.valid[i] != 0;
                    differing += !same_bits(x.points[i], y.points[i]) || x.valid[i] != y.valid[i] ||
                                 !same_bits(x.normals[i], y.normals[i]) ||
                                 std::memcmp(&x.intensity[i], &y.intensity[i], sizeof(float)) != 0;
                }
        }
        std::printf("%s: %zu pixels of the three levels compared, %zu valid at level 0, %zu at "
                    "level 0 within the patch's blocks' reach\n",
                    what.c_str(), compared, valid, near_blocks);
        check(near_blocks > 0 && valid > compared / 2,
              (what + ": pixels compared within the patch's blocks' reach, most of them valid")
                  .c_str());
        check_within((what + ": pixels not the same bit for bit").c_str(), differing, 0.0);
    }
    check(alone.pool.refused_while_closed() == 0 && widened.pool.refused_while_closed() == 0,
          "widened tiles: the closed pools asked for nothing");
}

/// Prints the median, the 99th percentile and the largest of values, and checks the median and
/// one of the others against their tolerances.
/// @param what [in] what they are, printed and named in the check lines.
/// @param values [in] the values.
/// @param median_tolerance [in] the largest median that passes.
/// @param top [in] 0.99 to check the 99th percentile, 1 to check the largest.
/// @param top_tolerance [in] the largest value of that quantile that passes.
void check_spread(const std::string &what, const std::vector<double> &values,
                  double median_tolerance, double top, double top_tolerance)
{
    const bool finite =
        std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); });
    check(finite && !values.empty(), (what + ": values, every one finite").c_str());
    if (!finite || values.empty())
        return;
    std::printf("%s: median %.4g, 99th percentile %.4g, largest %.4g, of %zu\n", what.c_str(),
                quantile(values, 0.5), quantile(values, 0.99), quantile(values, 1.0),
                values.size());
    check_within((what + ", median").c_str(), quantile(values, 0.5), median_tolerance);
    check_within((what + (top < 1.0 ? ", 99th percentile" : ", largest")).c_str(),
                 quantile(values, top), top_tolerance);
}

/// A scene's cast compared with the analytic scene, pixel by pixel at level 0.
struct cast_comparison
{
    std::vector<double> distance_mm; ///< each valid point's distance from the nearest surface
    std::vector<double> angle_deg;   ///< each normal's angle from that surface's normal
    std::vector<double> intensity;   ///< each |intensity - the texture's luma| there
    std::vector<double> color; ///< each color's largest |channel - the texture's| there, 0 to 255
    std::vector<double> along_ray_mm; ///< |depth - the analytic hit's|, where a ray sees the scene
    std::size_t seen = 0;             ///< the pixels whose ray meets the scene within the range
    std::size_t seen_valid = 0;       ///< of these, the valid ones
    std::size_t unseen_valid = 0;     ///< the valid pixels whose ray meets nothing within it
    max_error unseen_distance_mm;     ///< the distance of those from the nearest surface
    std::size_t without_normal = 0;   ///< the valid pixels without a normal
};

/// Compares a cast of a sphere-on-plane scene with the analytic scene at level 0.
/// @param scene [in] the scene.
/// @param surfaces [in] its known surfaces, the plane first.
/// @param pose [in] the cast's pose, camera frame to the world frame.
/// @param picture [in] the cast's level 0.
/// @param range [in] the depth range of the cast.
/// @return the comparison.
cast_comparison compare_with_scene(const gca_test::synthetic_scene &scene,
                                   const std::vector<gca_test::synthetic_surface> &surfaces,
                                   const Eigen::Isometry3d &pose, const cast_picture &picture,
                                   const gca::organized_points_parameters &range)
{
    cast_comparison c;
    if (picture.valid.empty())
        return c;
    for (std::uint32_t v = 0; v < picture.size.height; v++)
        for (std::uint32_t u = 0; u < picture.size.width; u++)
        {
            const std::size_t i = gca_test::index_of(picture.size, u, v);
            const Eigen::Vector3d ray = pose.linear() * gca_test::pixel_ray(picture.k, u, v);
            const std::optional<gca_test::synthetic_hit> hit =
                gca_test::first_hit(surfaces, scene.sphere, pose.translation(), ray);
            const bool sees = hit && hit->s >= range.near_m && hit->s <= range.far_m;
            c.seen += sees;
            if (picture.valid[i] == 0)
                continue;
            c.seen_valid += sees;
            c.unseen_valid += !sees;
            const Eigen::Vector3d point = to_double(picture.points[i]);
            if (sees)
                c.along_ray_mm.push_back(1e3 * std::abs(point.z() - hit->s));
            const expected_point e = expect_at(scene, surfaces, pose * point);
            c.distance_mm.push_back(1e3 * e.distance_m);
            if (!sees)
                c.unseen_distance_mm.add(1e3 * e.distance_m);
            c.intensity.push_back(std::abs(double{picture.intensity[i]} - luma_of(e.color)));
            if (i < picture.colors.size())
            {
                c.color.push_back(largest_channel_error(picture.colors[i], e.color));
            }
            if (gca_test::is_zero(picture.normals[i]))
                c.without_normal++;
            else
                c.angle_deg.push_back(
                    angle_deg(pose.linear() * to_double(picture.normals[i]), e.normal));
        }
    return c;
}

/// A scene of a sphere on a plane and the model it is fused into, with its casts' tolerances.
struct scene_case
{
    model_case model;                ///< the sparse TSDF's parameters and the depth range
    gca_test::synthetic_scene scene; ///< the plane, the sphere and the walk's poses
    gca_test::synthetic_walk walk;   ///< the walk the scene's poses come from
    Eigen::Isometry3d step;          ///< a hand's step from a pose of the walk
    cast_tolerances tolerances;      ///< the tolerances of its casts
};

/// Gives a view of a scene's walk into space never seen: from the walk's first pose's position,
/// looking away from the walk's centre and up by 45 degrees, where no frame of the walk looked.
/// @param c [in] the case.
/// @return the pose; nothing when look_at() refuses it.
std::optional<Eigen::Isometry3d> never_seen_view(const scene_case &c)
{
    const Eigen::Vector3d position = c.scene.trajectory.front().translation();
    Eigen::Vector3d away = position - c.walk.centre_m;
    away.z() = 0.0;
    return gca_test::look_at(position, position + away.normalized() + Eigen::Vector3d::UnitZ(),
                             0.0);
}

/// Fuses a scene's walk and casts it from a pose of the walk, from one halfway to the next and from
/// one a hand's step from the first, at both stream sizes, and checks each against the analytic
/// scene; and from a view of space never seen, every pixel invalid.
/// @param c [in] the case.
/// @param stream [in] the stream.
void check_scene(const scene_case &c, cudaStream_t stream)
{
    const std::string &name = c.model.name;
    const gca_test::rendered frames = gca_test::render(c.scene, small_description);
    if (frames.frames.size() != c.scene.trajectory.size() || frames.frames.empty())
        return;
    gca_test::synthetic_walk between_walk = c.walk;
    between_walk.pose_count = 2 * c.walk.pose_count;
    const std::optional<std::vector<Eigen::Isometry3d>> between =
        gca_test::walk_around(between_walk);
    const std::optional<Eigen::Isometry3d> never = never_seen_view(c);
    check(between.has_value() && between->size() > 1 && never.has_value(),
          (name + ": the casts' poses").c_str());
    if (!between || between->size() < 2 || !never)
        return;
    gca::gpu_pool pool;
    frame_buffers buffers = request_buffers(pool, small_description.size, name);
    gca::sparse_tsdf tsdf =
        granted(name + ": the sparse TSDF", gca::sparse_tsdf::request(pool, c.model.tsdf, stream));
    std::array<gca::model_surface, 2> surfaces = {
        granted(name + ": the surface at 640x400",
                gca::model_surface::request(pool, small_description.size)),
        granted(name + ": the surface at 1280x800",
                gca::model_surface::request(pool, large_description.size))};
    pool.close();
    for (std::size_t i = 0; i < frames.frames.size(); i++)
        fuse(tsdf, buffers, frames.frames[i], c.model.range, c.scene.trajectory[i], stream,
             name + ", frame " + std::to_string(i));
    const gca::sparse_tsdf_usage usage = gca_test::read_usage(tsdf, stream, name);
    gca_test::print_usage(name, usage);
    check(usage.blocks_refused == 0 && usage.coordinates_refused == 0,
          (name + ": the pool held every block").c_str());

    const std::array<std::pair<std::string, Eigen::Isometry3d>, 3> poses = {
        std::pair{std::string("a pose of the walk"), c.scene.trajectory.front()},
        std::pair{std::string("halfway to the next pose"), (*between)[1]},
        std::pair{std::string("a hand's step off"), c.scene.trajectory.front() * c.step}};
    const std::array<gca::frame_description, 2> sizes = {small_description, large_description};
    const cast_tolerances &t = c.tolerances;
    for (std::size_t s = 0; s < sizes.size(); s++)
        for (const auto &[pose_name, pose] : poses)
        {
            const std::string what = name + ", " + pose_name + ", " + gca::size_text(sizes[s].size);
            const cast_pictures pictures =
                cast(tsdf, pose, sizes[s].intrinsics, c.model.range, surfaces[s], stream, what);
            check_levels(what, pictures, sizes[s].intrinsics, c.model.range);
            const cast_comparison cmp =
                compare_with_scene(c.scene, frames.surfaces, pose, pictures[0], c.model.range);
            const double share = static_cast<double>(cmp.seen_valid) /
                                 static_cast<double>(std::max<std::size_t>(cmp.seen, 1));
            std::printf("%s: %zu of %zu pixels that see the scene valid (%.5f), %zu valid that "
                        "see nothing, %zu without a normal\n",
                        what.c_str(), cmp.seen_valid, cmp.seen, share, cmp.unseen_valid,
                        cmp.without_normal);
            if (!cmp.along_ray_mm.empty())
                std::printf("%s: depth from the analytic hit, mm: median %.4g, 99th percentile "
                            "%.4g, largest %.4g\n",
                            what.c_str(), quantile(cmp.along_ray_mm, 0.5),
                            quantile(cmp.along_ray_mm, 0.99), quantile(cmp.along_ray_mm, 1.0));
            check_spread(what + ": distance from the surface, mm", cmp.distance_mm,
                         t.distance_median_mm, 1.0, t.distance_max_mm);
            check_spread(what + ": normal's angle, deg", cmp.angle_deg, t.normal_median_deg, 0.99,
                         t.normal_p99_deg);
            check_spread(what + ": intensity's error", cmp.intensity, t.intensity_median, 0.99,
                         t.intensity_p99);
            check_spread(what + ": color's largest channel's error", cmp.color, t.color_median,
                         0.99, t.color_p99);
            check(share >= t.least_seen_valid,
                  (what + ": the share of the pixels that see the scene valid").c_str());
            gca_test::check_max_error(what + ": a valid pixel's that sees nothing, its distance "
                                             "from the surface, mm",
                                      cmp.unseen_distance_mm, t.unseen_distance_max_mm);
        }
    const std::string never_name = name + ", a view of space never seen";
    const cast_pictures none = cast(tsdf, *never, small_description.intrinsics, c.model.range,
                                    surfaces[0], stream, never_name);
    check_levels(never_name, none, small_description.intrinsics, c.model.range);
    check_within((never_name + ": valid pixels").c_str(), static_cast<double>(valid_count(none[0])),
                 0.0);
    check(pool.refused_while_closed() == 0, (name + ": the closed pool asked for nothing").c_str());
}

/// A case of a frame registered against the cast from the last fused pose.
struct tracking_case
{
    std::string name;                        ///< the case, for the printed lines
    gca_test::synthetic_scene scene;         ///< the scene
    std::optional<Eigen::Isometry3d> pose_a; ///< the last fused pose, camera to world
    Eigen::Isometry3d delta;                 ///< the known motion from the next frame into A's
    gca::object_class kind;                  ///< the object class: voxel size, range, values
    gca::registration_parameters parameters; ///< the full method's parameters
    transform_tolerance tolerance;           ///< how far a converged transform may lie from Δ
};

/// The fused frames of a tracking case: the last pose and the poses before it, each one known
/// motion before the next [?: a choice, a camera arriving at the last pose].
constexpr int fused_frames = 5;

/// Gives parameters with level 2 skipped: levels 1 and 0.
/// @param p [in] the parameters.
/// @return p with no iteration at level 2.
gca::registration_parameters on_levels_1_and_0(gca::registration_parameters p)
{
    p.levels[2].max_iterations = 0;
    return p;
}

/// Fuses the frames arriving at pose A, ray casts the model from A and registers frame B, seen from
/// A Δ, against the cast from the identity: by the full method, which must converge within
/// its tolerance of Δ, and with level 0 alone and with levels 1 and 0, printed; and
/// against frame A's own pyramid by the full method, printed for comparison. Every converged result
/// must carry Δ within the tolerance.
/// @param c [in] the case.
/// @param stream [in] the stream.
void check_tracking(const tracking_case &c, cudaStream_t stream)
{
    check(c.pose_a.has_value(), (c.name + ": the pose").c_str());
    if (!c.pose_a)
        return;
    const model_case m = model_of(c.name, c.kind, false);
    const gca::organized_points_parameters range = range_of(c.kind);
    gca_test::synthetic_scene scene = c.scene;
    scene.trajectory.clear();
    for (int i = fused_frames - 1; i >= 0; i--)
    {
        Eigen::Isometry3d pose = *c.pose_a;
        for (int k = 0; k < i; k++)
            pose = pose * c.delta.inverse();
        scene.trajectory.push_back(pose);
    }
    scene.trajectory.push_back(*c.pose_a * c.delta);
    const gca_test::rendered frames = gca_test::render(scene, small_description);
    if (frames.frames.size() != scene.trajectory.size())
        return;
    gca::gpu_pool pool;
    frame_buffers buffers = request_buffers(pool, small_description.size, c.name);
    gca::sparse_tsdf tsdf =
        granted(c.name + ": the sparse TSDF", gca::sparse_tsdf::request(pool, m.tsdf, stream));
    gca::model_surface surface = granted(c.name + ": the surface",
                                         gca::model_surface::request(pool, small_description.size));
    gca::frame_pyramid moved = granted(c.name + ": frame B's pyramid",
                                       gca::frame_pyramid::request(pool, small_description.size));
    gca::frame_pyramid last = granted(c.name + ": frame A's pyramid",
                                      gca::frame_pyramid::request(pool, small_description.size));
    gca::registration registration = granted(
        c.name + ": the registration", gca::registration::request(pool, small_description.size));
    pool.close();
    for (int i = 0; i < fused_frames; i++)
        fuse(tsdf, buffers, frames.frames[i], range, scene.trajectory[i], stream,
             c.name + ", frame " + std::to_string(i));
    const gca::step_result made = tsdf.ray_cast(
        as_camera_pose(*c.pose_a), small_description.intrinsics, range, surface, stream);
    check(made.done, (c.name + ": ray cast queued, " + made.message).c_str());
    const gca::frame_pyramid_parameters pyramid_parameters = {range, normals_parameters};
    check(buffers.frame.upload(frames.frames.back(), stream).done &&
              moved.build(buffers.frame, pyramid_parameters, stream).done,
          (c.name + ": frame B's pyramid built").c_str());
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    const std::array<std::pair<std::string, gca::registration_parameters>, 3> configurations = {
        std::pair{std::string(", level 0 alone"), gca_test::on_level_0(c.parameters)},
        std::pair{std::string(", levels 1 and 0"), on_levels_1_and_0(c.parameters)},
        std::pair{std::string(""), c.parameters}};
    for (const auto &[suffix, parameters] : configurations)
    {
        const std::string what = c.name + " through the cast" + suffix;
        const gca::registration_result result =
            registration.align(moved, gca::surface_of(surface), identity, parameters, stream);
        print_result(what, result);
        check_converged_is_known(what, result, c.delta, c.tolerance);
        if (suffix.empty())
            check(result.converged(), (what + ": converged").c_str());
    }
    check(buffers.frame.upload(frames.frames[fused_frames - 1], stream).done &&
              last.build(buffers.frame, pyramid_parameters, stream).done,
          (c.name + ": frame A's pyramid built").c_str());
    const std::string own = c.name + " against frame A's pyramid";
    const gca::registration_result reference =
        registration.align(moved, gca::surface_of(last), identity, c.parameters, stream);
    print_result(own, reference);
    check_converged_is_known(own, reference, c.delta, c.tolerance);
    check(pool.refused_while_closed() == 0,
          (c.name + ": the closed pool asked for nothing").c_str());
}

/// Runs the tracking cases: the tabletop object from 0.5 m, the large object from 2.7 m, and the
/// large box without its sphere turned by 1.5 degrees, each at the live voxel size of its object
/// class, 640x400.
/// @param stream [in] the stream.
void check_tracking_cases(cudaStream_t stream)
{
    gca_test::synthetic_scene large_box = gca_test::large_object_scene();
    large_box.sphere = std::nullopt;
    const tracking_case cases[] = {
        {"tabletop object", gca_test::tabletop_object_scene(), gca_test::tabletop_object_pose(),
         gca_test::tabletop_delta, gca::object_class::tabletop_object,
         gca_test::tabletop_parameters, tabletop_through_cast_tolerance},
        {"large object", gca_test::large_object_scene(), gca_test::large_object_pose(),
         gca_test::large_delta, gca::object_class::large_object, gca_test::large_parameters,
         large_through_cast_tolerance},
        {"large box turned by 1.5 deg", large_box, gca_test::large_object_pose(),
         gca_test::large_turn_delta, gca::object_class::large_object, gca_test::large_parameters,
         large_through_cast_tolerance},
    };
    for (const tracking_case &c : cases)
        check_tracking(c, stream);
}

/// Says whether two downloads of a surface are the same bit for bit.
/// @param a [in] one.
/// @param b [in] the other.
/// @return whether every level's intrinsics and images are equal.
bool same_surface(const cast_pictures &a, const cast_pictures &b)
{
    for (std::size_t l = 0; l < levels; l++)
    {
        const cast_picture &x = a[l];
        const cast_picture &y = b[l];
        if (std::memcmp(&x.k, &y.k, sizeof(gca::intrinsics)) != 0 || x.valid != y.valid ||
            x.points.size() != y.points.size() || x.intensity.size() != y.intensity.size() ||
            std::memcmp(x.points.data(), y.points.data(), x.points.size() * sizeof(float3)) != 0 ||
            std::memcmp(x.normals.data(), y.normals.data(), x.normals.size() * sizeof(float3)) !=
                0 ||
            std::memcmp(x.intensity.data(), y.intensity.data(),
                        x.intensity.size() * sizeof(float)) != 0 ||
            x.colors.size() != y.colors.size() ||
            std::memcmp(x.colors.data(), y.colors.data(), x.colors.size() * sizeof(uchar3)) != 0)
            return false;
    }
    return true;
}

/// Refusals as values, each leaving the surface as it was and CUDA's last error clear: a cast from
/// an empty sparse TSDF and from one moved from, into an empty surface and one moved from, with
/// intrinsics that cannot be used, a depth range that is not one, a pose that cannot be used, and
/// after an error an earlier call left.
/// @param stream [in] the stream.
void check_refusals(cudaStream_t stream)
{
    const std::string what = "refusals";
    const model_case m = model_of(what, gca::object_class::tabletop_object, false);
    const gca::intrinsics &k = small_description.intrinsics;
    gca::gpu_pool pool;
    frame_buffers buffers = request_buffers(pool, small_description.size, what);
    gca::sparse_tsdf tsdf =
        granted(what + ": the sparse TSDF", gca::sparse_tsdf::request(pool, m.tsdf, stream));
    gca::model_surface surface =
        granted(what + ": the surface", gca::model_surface::request(pool, small_description.size));
    pool.close();
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    fuse(tsdf, buffers, flat_wall(wall_mm, wall_color), m.range, identity, stream, what);
    const cast_pictures before = cast(tsdf, identity, k, m.range, surface, stream, what);
    const gca::camera_pose usable = as_camera_pose(identity);

    gca::sparse_tsdf empty;
    check_refused("refusals, an empty sparse TSDF",
                  empty.ray_cast(usable, k, m.range, surface, stream), "holds no memory");
    gca::sparse_tsdf moved_to(std::move(tsdf));
    check_refused("refusals, a sparse TSDF moved from",
                  tsdf.ray_cast(usable, k, m.range, surface, stream), "holds no memory");
    tsdf = std::move(moved_to);
    gca::model_surface no_surface;
    check_refused("refusals, an empty surface",
                  tsdf.ray_cast(usable, k, m.range, no_surface, stream),
                  "the model's surface holds no images");
    gca::model_surface moved_surface(std::move(surface));
    check_refused("refusals, a surface moved from",
                  tsdf.ray_cast(usable, k, m.range, surface, stream),
                  "the model's surface holds no images");
    surface = std::move(moved_surface);
    gca::intrinsics unusable = k;
    unusable.fx = 0.0f;
    check_refused("refusals, intrinsics that cannot be used",
                  tsdf.ray_cast(usable, unusable, m.range, surface, stream),
                  "the intrinsics cannot be used");
    check_refused("refusals, a depth range that is not one",
                  tsdf.ray_cast(usable, k, {m.range.far_m, m.range.near_m}, surface, stream),
                  "is not one");
    const gca::camera_pose not_finite = {
        Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0),
        Eigen::Quaterniond::Identity()};
    check_refused("refusals, a pose that cannot be used",
                  tsdf.ray_cast(not_finite, k, m.range, surface, stream),
                  "the pose cannot be used");
    // An error an earlier call leaves: cudaMalloc's refusal of 2^52 bytes, more than the board
    // has [S: docs/HARDWARE.md, 7607 MB].
    void *impossible = nullptr;
    check(cudaMalloc(&impossible, std::size_t{1} << 52) != cudaSuccess,
          "refusals: cudaMalloc refuses 2^52 bytes");
    check_refused("refusals, a cast after an earlier error",
                  tsdf.ray_cast(usable, k, m.range, surface, stream), "left by an earlier call");
    check_no_cuda_error("refusals, a cast after an earlier error");
    check(same_surface(before, download_surface(surface, stream, what + ", after them")),
          "refusals: the surface as it was");
    check(pool.refused_while_closed() == 0, "refusals: the closed pool asked for nothing");
}
} // namespace

/// Runs the checks of the ray cast and prints the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    cudaStream_t stream = nullptr;
    if (!create_stream(stream))
        return gca_test::check_summary("test_sparse_tsdf_ray_cast");
    check_wall(stream);
    check_colored_wall(stream);
    check_half_wall(stream);
    check_wall_turned(stream);
    check_slanted_wall(stream);
    check_patch_before_wall(stream);
    check_inside_band(stream);
    check_widened_tiles(stream);
    check_scene({model_of("tabletop, live model", gca::object_class::tabletop_object, false),
                 gca_test::tabletop_sphere_scene(), gca_test::tabletop_walk(),
                 gca_test::tabletop_delta, tabletop_tolerances},
                stream);
    check_scene({model_of("large, live model", gca::object_class::large_object, false),
                 gca_test::large_sphere_scene(), gca_test::large_walk(), gca_test::large_delta,
                 large_tolerances},
                stream);
    check_tracking_cases(stream);
    check_refusals(stream);
    check_no_cuda_error("the end");
    check(cudaStreamDestroy(stream) == cudaSuccess, "the CUDA stream destroyed");
    return gca_test::check_summary("test_sparse_tsdf_ray_cast");
}
