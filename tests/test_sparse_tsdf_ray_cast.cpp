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
///   picture's border holds the wall at its depth within float's rounding, its normal along -z and
///   its intensity; seen from behind, a back face, every pixel is invalid; a view that meets no
///   block, every pixel is invalid.
/// - A sphere resting on a plane, fused from a walk of 12 known poses, at the live voxel sizes of
///   both object classes: cast from a pose of the walk, from one between two of its poses and from
///   one a hand's step away from it, at both stream sizes, every valid point lies within measured
///   tolerances of the analytic surface, its normal within an angle and its intensity within a
///   tolerance of the texture's luma there; a stated share of the pixels whose ray meets the scene
///   within the depth range is valid; a pixel whose ray meets nothing is invalid; a view of space
///   never seen is invalid everywhere.
/// - Every cast: each coarser level's pixel is level 0's at twice or four times its index, bit for
///   bit, with the intrinsics halved; an invalid pixel is all 0; a valid point projects onto its
///   own pixel.
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
using gca_test::surface_of;
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
    double least_seen_valid;   ///< the least share of the pixels that see the scene that are valid
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
/// degrees; intensity errors 0.00280 / 0.0198 / 0.185; at least 0.99688 of the pixels that see the
/// scene valid; a valid pixel that sees nothing within 1.14 mm, 0.57 voxels, of the plane.
constexpr cast_tolerances tabletop_tolerances = {0.08, 2.5, 3.0, 12.0, 0.004, 0.03, 0.995, 1.7};

/// The large scene, 8 mm voxels: distances 0.243 / 4.71 / 10.8 mm; angles 2.14 / 14.0 / 75.2
/// degrees; intensity errors 0.0277 / 0.104 / 0.481, the 8 mm voxels blurring the textures' waves
/// of 45 mm to 100 mm; at least 0.99599 of the pixels that see the scene valid; a valid pixel that
/// sees nothing within 7.27 mm, 0.91 voxels, of the plane.
constexpr cast_tolerances large_tolerances = {0.35, 15.0, 3.0, 20.0, 0.04, 0.15, 0.994, 11.0};

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
/// bit for bit, its intrinsics level 0's halved l times; an invalid pixel is all 0; a valid pixel's
/// point lies in front of the camera within the depth range and projects onto its own pixel, its
/// normal is of unit length or (0, 0, 0) and its intensity lies in [0, 1].
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
    check_within((what + ": levels of the wrong size or intrinsics").c_str(), wrong_intrinsics,
                 0.0);
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
    const model_case m = model_of("wall", gca::object_class::tabletop_object, false);
    const gca::intrinsics &k = small_description.intrinsics;
    gca::gpu_pool pool;
    frame_buffers buffers = request_buffers(pool, small_description.size, m.name);
    gca::sparse_tsdf tsdf =
        granted(m.name + ": the sparse TSDF", gca::sparse_tsdf::request(pool, m.tsdf, stream));
    gca::model_surface surface = granted(m.name + ": the surface",
                                         gca::model_surface::request(pool, small_description.size));
    pool.close();
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    fuse(tsdf, buffers, flat_wall(wall_mm, wall_color), m.range, identity, stream, m.name);

    const std::string front = "a wall, from its pose";
    const cast_pictures seen = cast(tsdf, identity, k, m.range, surface, stream, front);
    check_levels(front, seen, k, m.range);
    const cast_picture &p = seen[0];
    const double expected_intensity =
        luma_of({double{wall_color[0]}, double{wall_color[1]}, double{wall_color[2]}});
    int missed = 0;
    int without_normal = 0;
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
        }
    std::printf("%s: %zu valid pixels of %zu\n", front.c_str(), valid_count(p), p.valid.size());
    check_within((front + ": pixels inside the border without the wall").c_str(), missed, 0.0);
    check_within((front + ": pixels inside the border without a normal").c_str(), without_normal,
                 0.0);
    gca_test::check_max_error(front + ": depth's error, m", depth, wall_depth_tolerance_m);
    gca_test::check_max_error(front + ": normal's angle from -z, deg", normal,
                              wall_normal_tolerance_deg);
    gca_test::check_max_error(front + ": intensity's error", intensity, intensity_float_tolerance);

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
    check(pool.refused_while_closed() == 0, "wall: the closed pool asked for nothing");
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
    const model_case m = model_of("half wall", gca::object_class::tabletop_object, false);
    const gca::intrinsics &k = small_description.intrinsics;
    gca::gpu_pool pool;
    frame_buffers buffers = request_buffers(pool, small_description.size, m.name);
    gca::sparse_tsdf tsdf =
        granted(m.name + ": the sparse TSDF", gca::sparse_tsdf::request(pool, m.tsdf, stream));
    gca::model_surface surface = granted(m.name + ": the surface",
                                         gca::model_surface::request(pool, small_description.size));
    pool.close();
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    const gca::frame half = gca_test::wall_frame(
        [](std::uint32_t u, std::uint32_t) {
            return u < half_wall_column ? wall_mm : std::uint16_t{0};
        },
        wall_color);
    fuse(tsdf, buffers, half, m.range, identity, stream, m.name);
    const cast_pictures pictures = cast(tsdf, identity, k, m.range, surface, stream, m.name);
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
    check(pool.refused_while_closed() == 0, "half wall: the closed pool asked for nothing");
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
    std::vector<double> distance_mm;  ///< each valid point's distance from the nearest surface
    std::vector<double> angle_deg;    ///< each normal's angle from that surface's normal
    std::vector<double> intensity;    ///< each |intensity - the texture's luma| there
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

/// Gives a model's surface's level as a level of the registration's surface.
/// @param surface [in] the model's surface.
/// @param level [in] the level.
/// @return its points, marks, normals, intensity and intrinsics.
gca::organized_surface level_of_cast(const gca::model_surface &surface, std::size_t level)
{
    return {surface.points(level), surface.validity_mask(level), surface.normals(level),
            surface.intensity(level), surface.level_intrinsics(level)};
}

/// Gives a model's surface's levels as the registration's surface.
/// @param surface [in] the model's surface.
/// @return its levels 0, 1 and 2.
gca::surface_levels levels_of_cast(const gca::model_surface &surface)
{
    return {level_of_cast(surface, 0), level_of_cast(surface, 1), level_of_cast(surface, 2)};
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
            registration.align(moved, levels_of_cast(surface), identity, parameters, stream);
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
        registration.align(moved, surface_of(last), identity, c.parameters, stream);
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
                        x.intensity.size() * sizeof(float)) != 0)
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
    check_half_wall(stream);
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
