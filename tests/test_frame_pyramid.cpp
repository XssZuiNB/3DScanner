// SPDX-License-Identifier: 0BSD
/// @file test_frame_pyramid.cpp
/// The geometry module's frame pyramid (geometry/frame_pyramid.hpp) on frames of the tests'
/// synthetic source and on frames built here, with the camera's intrinsics; the method is
/// docs/methods/frame-pyramid.md. Before each build the test fills every level's images with 0xFF
/// bytes, so that a pixel the step leaves unwritten shows.
/// - Every picture, every level: the level's size and intrinsics, the frame's halved l times; a
///   mark is 0 or 1, and a coarser level's mark is that of its centre pixel at level 0; an invalid
///   pixel's point, normal and intensity are 0; a valid point's z is > 0, its normal a unit vector
///   toward the camera or (0, 0, 0), its intensity in [0, 1]; every point projects by its level's
///   intrinsics onto its own pixel. Each coarser level agrees with the combining computed here in
///   double from the level before as downloaded (the reference): the same points and intensities
///   within float's rounding. A pixel whose window holds a pair within premise_margin_deg of the
///   rule's threshold is left out of the reference, where float and double may decide the pair
///   differently; their share is bounded. So a coarser point combines exactly the pairs the rule
///   takes for one surface, and none across a depth edge.
/// - The intensity of known colors at level 0 is the Rec. 709 luma and the prototype's, the
///   oracle's, within float's rounding, and so is a coarser pixel's within one color.
/// - At both stream sizes and at a size that is no multiple of the kernels' block, a tilted
///   analytic plane and a plane 2 m away: every level's points lie on the plane within the
///   depth's rounding, a whole window's points project onto their pixels' centres, and every
///   level's normals agree with the plane's.
/// - Three frames of the tabletop walk, with depth edges where the box hides the plane, and one at
///   1280x800: besides the reference, every point of every level lies within 2.5 mm of a known
///   surface, a bound of what the rule leaves on this scene (between_surfaces_tolerance_m).
/// - A flat frame with an area without depth, a column without depth on a pixel that is a centre
///   at every level and one between centres: the area and the first column stay invalid at every
///   level, and no point is pulled toward them.
/// - Refusals as values, each leaving every level, filled with 0xFF bytes, as it was: an empty
///   pyramid, normals' parameters that cannot be used, the steepest incidence one float step above
///   its largest, a depth range that is none, a GPU frame that holds no frame, a frame of another
///   size, and CUDA's last error left by an earlier call; and CUDA's refusal of the launch for a
///   frame whose rows need more blocks than a grid holds. CUDA's last error is clear after each.
///   The largest steepest incidence, 89 degrees, is accepted.
/// - Every buffer is requested before the pool closes, and the step asks the closed pool for
///   nothing.

#include "camera/frame.hpp"
#include "check.hpp"
#include "geometry/frame_pyramid.hpp"
#include "geometry/organized_normals.hpp"
#include "geometry/organized_points.hpp"
#include "geometry/type.hpp"
#include "gpu/gpu_frame.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/step_result.hpp"
#include "gpu_test_helpers.hpp"
#include "stage_checks.hpp"
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
#include <numbers>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace
{
using gca_test::angle_deg;
using gca_test::built_frame;
using gca_test::camera_depth_unit_m;
using gca_test::check;
using gca_test::check_max_error;
using gca_test::check_no_cuda_error;
using gca_test::check_refused;
using gca_test::check_within;
using gca_test::create_stream;
using gca_test::distance_from;
using gca_test::download_image;
using gca_test::granted;
using gca_test::grey;
using gca_test::index_of;
using gca_test::is_zero;
using gca_test::large_description;
using gca_test::max_error;
using gca_test::median;
using gca_test::odd_description;
using gca_test::pair_side;
using gca_test::premise_margin_deg;
using gca_test::render;
using gca_test::rendered;
using gca_test::side_of;
using gca_test::small_description;
using gca_test::tabletop_box;
using gca_test::tabletop_plane;
using gca_test::tall_size;
using gca_test::to_double;

/// The number of levels.
constexpr std::size_t level_count = gca::frame_pyramid::level_count;

/// The normals' parameters at every level: the method's starting values, a 7x7 window, 25 points,
/// 75 degrees [S: docs/methods/normals.md, "Input"].
constexpr gca::organized_normals_parameters normals_parameters = {3, 25, 75.0f};

/// The tilted plane's depth range: 0.4005 m to 0.8005 m, each limit half a depth unit off a depth
/// value [?: a choice, the organized points' and the normals' tests'].
constexpr gca::organized_points_parameters tilted_range = {0.4005f, 0.8005f};

/// The plane at 2 m's depth range: 1.0005 m to 3.0005 m [?: a choice, the normals' test's].
constexpr gca::organized_points_parameters far_range = {1.0005f, 3.0005f};

/// The tabletop walk's depth range: the tabletop object class's 0.30 m to 1.0 m, each limit half a
/// unit inside [S: src/scan/object_class.cpp].
constexpr gca::organized_points_parameters box_range = {0.3005f, 0.9995f};

/// The range of the frames built here, which lie at 0.5 m [?: a choice].
constexpr gca::organized_points_parameters built_range = {0.3005f, 0.9995f};

/// The depth of the frames built here, in depth units: 0.5 m [?: a choice].
constexpr std::uint16_t built_depth = 500;

/// The plane seen from 2 m: 6 m x 6 m, larger than the view [?: a choice, the normals' test's].
constexpr gca_test::synthetic_plane far_plane = {3.0, 3.0};

/// How far a valid point may project from its pixel's centre by its level's intrinsics, in px of
/// the level: at level 0 float's rounding of the point, a few times 6e-8 of |u - cx| < 640 px
/// [S: float32 ulp]; at level 1 and 2 half a pixel, the pixel's own, the claim the method makes
/// for every steepest incidence it takes; its estimate near the picture's centre is 0.26 and 0.38
/// px at 75 degrees [S: docs/methods/frame-pyramid.md, "The intrinsics of a level"]. The largest
/// are 6.7e-5, 0.201 and 0.282 px [M 2026-11-14, this test]. A whole window's point is held far
/// tighter (plane_tolerances::whole_window_px).
constexpr std::array<double, level_count> own_pixel_tolerance_px = {1e-3, 0.5, 0.5};

/// A coarser point against the reference's, in m: float's rounding of the centre's point plus the
/// mean of offsets of a few mm, about one float step of the point, 1.2e-7 m below 2 m [S: float32
/// ulp]. The largest is 1.79e-7 m, on the plane at 2 m [M 2026-11-14, this test].
constexpr double reference_point_tolerance_m = 1e-6;

/// A coarser intensity against the reference's: float's rounding of a sum of at most 9 weighted
/// intensities, each below 4, and of the division, a few float steps of 16 [S: float32 ulp]. The
/// largest is 1.69e-7 [M 2026-11-14, this test].
constexpr double reference_intensity_tolerance = 1e-5;

/// An intensity against the Rec. 709 luma in double and against the oracle's: three products and
/// two sums of numbers below 256 and a division, each rounded once in float, a few float steps of
/// 1 [S: float32 ulp]; one step of 8 bits, 1/255, is 8000 times as large. The largest is 1.02e-7
/// against the luma, 1.19e-7 against the oracle and 1.94e-7 inside a cell at level 2
/// [M 2026-11-14, this test].
constexpr double intensity_tolerance = 4 * gca_test::float_step;

/// A point's distance from the analytic plane, divided by the depth unit times the length of its
/// pixel's ray at its level: at level 0 at most half a unit along the ray [S: the depth's rounding;
/// the organized points' test]; a coarser point is a convex combination of points of the frame
/// within 3 px of its centre pixel, whose rays are at most 3 / fx = 1 % longer [S: computed], so
/// at most half a unit times 1.01; and 1e-3 for float's rounding. The largest are 0.500, 0.318 and
/// 0.157 at level 0, 1 and 2 [M 2026-11-14, this test].
constexpr std::array<double, level_count> plane_tolerance = {0.5 + 1e-3, 0.505 + 1e-3,
                                                             0.505 + 1e-3};

/// | |n| - 1 |: 8 float steps, as the normals' test [S: float32 ulp].
constexpr double length_tolerance = 8 * gca_test::float_step;

/// What the step works on at one size: a GPU frame and a pyramid of that size, both from the
/// pool, and the stream the work runs on.
struct stage_buffers
{
    gca::gpu_frame frame;          ///< the GPU frame the frames are uploaded into
    gca::frame_pyramid pyramid;    ///< the pyramid built from them
    cudaStream_t stream = nullptr; ///< the CUDA stream of the work
};

/// One level of a pyramid, downloaded.
struct level_picture
{
    gca::frame_size size{0, 0};     ///< the level's size, in px
    gca::intrinsics k{};            ///< the level's intrinsics, in px
    std::vector<float3> points;     ///< in the camera frame, in m
    std::vector<std::uint8_t> mask; ///< 1 valid, 0 invalid
    std::vector<float3> normals;    ///< unit normals in the camera frame, or (0, 0, 0)
    std::vector<float> intensity;   ///< in [0, 1], 0 where invalid
};

/// Every level of a pyramid, downloaded; empty levels when the download failed.
using pyramid_picture = std::array<level_picture, level_count>;

/// Returns 2^level.
/// @param level [in] the level.
/// @return 2^level, as an unsigned whole number.
std::uint32_t scale_of(std::size_t level)
{
    return std::uint32_t{1} << level;
}

/// Fills every level of a pyramid with 0xFF bytes: a NaN in every coordinate and intensity and a
/// mark of 255, which the step writes over at every pixel it writes.
/// @param name [in] the case, named in the check line.
/// @param pyramid [in,out] the pyramid.
/// @param stream [in] the CUDA stream the fills are queued on.
void fill_with_0xff(const std::string &name, gca::frame_pyramid &pyramid, cudaStream_t stream)
{
    bool filled = true;
    // The test writes into the step's images on purpose; the step gives them only to read.
    const auto fill = [&](const auto &image) {
        using pixel = std::remove_cvref_t<decltype(*image.data())>;
        void *data = const_cast<pixel *>(image.data());
        filled = filled && cudaMemsetAsync(data, 0xFF, image.size_bytes(), stream) == cudaSuccess;
    };
    for (std::size_t level = 0; level < level_count; level++)
    {
        fill(pyramid.points(level).points());
        fill(pyramid.points(level).validity_mask());
        fill(pyramid.normals(level).normals());
        fill(pyramid.intensity(level));
    }
    check(filled, (name + ": the levels filled with 0xFF").c_str());
}

/// Uploads a frame, fills the pyramid with 0xFF bytes and builds it.
/// @param name [in] the case, named in the check lines.
/// @param host [in] the frame, of the pyramid's size.
/// @param stage [in,out] the GPU frame, the pyramid and the stream.
/// @param range [in] the depth range.
/// @return whether the upload and the build were done.
bool run_stage(const std::string &name, const gca::frame &host, stage_buffers &stage,
               const gca::organized_points_parameters &range)
{
    const gca::step_result up = stage.frame.upload(host, stage.stream);
    check(up.done, (name + ": upload, " + up.message).c_str());
    fill_with_0xff(name, stage.pyramid, stage.stream);
    const gca::step_result built =
        stage.pyramid.build(stage.frame, {range, normals_parameters}, stage.stream);
    check(built.done, (name + ": build, " + built.message).c_str());
    return up.done && built.done;
}

/// Downloads every level: waits for the stream's work, then copies them.
/// @param name [in] the case, named in the check line.
/// @param stage [in] the pyramid and the stream.
/// @return the levels; empty ones, with a failed check, when CUDA failed.
pyramid_picture download(const std::string &name, const stage_buffers &stage)
{
    pyramid_picture result;
    bool copied = cudaStreamSynchronize(stage.stream) == cudaSuccess;
    for (std::size_t level = 0; level < level_count && copied; level++)
    {
        level_picture &got = result[level];
        got.size = stage.pyramid.size(level);
        got.k = stage.pyramid.level_intrinsics(level);
        copied =
            download_image(stage.pyramid.points(level).points(), got.points, stage.stream) &&
            download_image(stage.pyramid.points(level).validity_mask(), got.mask, stage.stream) &&
            download_image(stage.pyramid.normals(level).normals(), got.normals, stage.stream) &&
            download_image(stage.pyramid.intensity(level), got.intensity, stage.stream);
    }
    check(copied, (name + ": downloaded").c_str());
    return copied ? std::move(result) : pyramid_picture{};
}

/// Says whether a download holds every level.
/// @param got [in] the download.
/// @return whether each level holds its points.
bool complete(const pyramid_picture &got)
{
    return std::ranges::all_of(got, [](const level_picture &l) { return !l.points.empty(); });
}

/// Returns where a point projects by a level's intrinsics.
/// @param k [in] the level's intrinsics.
/// @param p [in] the point, in the camera frame, z > 0.
/// @return its column and row, in px of the level.
Eigen::Vector2d project(const gca::intrinsics &k, const float3 &p)
{
    const Eigen::Vector3d q = to_double(p);
    return {double{k.fx} * q.x() / q.z() + k.cx, double{k.fy} * q.y() / q.z() + k.cy};
}

/// Checks every pixel of every level: the level's size and intrinsics; a mark is 0 or 1, and a
/// coarser level's mark is its centre pixel's at level 0; an invalid pixel's point, normal and
/// intensity are 0; a valid point has z > 0, a unit normal with n . p <= 0 or none, and an
/// intensity in [0, 1]; it projects onto its own pixel.
/// @param name [in] the case, printed in each line.
/// @param got [in] the levels.
/// @param description [in] the frame's description.
void check_levels(const std::string &name, const pyramid_picture &got,
                  const gca::frame_description &description)
{
    const level_picture &frame_level = got[0];
    for (std::size_t level = 0; level < level_count; level++)
    {
        const level_picture &l = got[level];
        const std::uint32_t scale = scale_of(level);
        const std::string at = name + ", level " + std::to_string(level);
        // ceil(size / 2^level) [S: halving l times, each rounded up].
        const gca::frame_size expected_size = {(description.size.width + scale - 1) / scale,
                                               (description.size.height + scale - 1) / scale};
        check(l.size == expected_size, (at + ": its size").c_str());
        const gca::intrinsics &k = description.intrinsics;
        const float s = static_cast<float>(scale);
        check(l.k.fx == k.fx / s && l.k.fy == k.fy / s && l.k.cx == k.cx / s && l.k.cy == k.cy / s,
              (at + ": its intrinsics are the frame's divided by 2^level").c_str());
        int valid = 0;
        int with_normal = 0;
        int bad_marks = 0;
        int not_centre_mark = 0;
        int invalid_not_zero = 0;
        int bad_valid = 0;
        max_error length;
        max_error off_pixel_px;
        for (std::uint32_t v = 0; v < l.size.height; v++)
        {
            for (std::uint32_t u = 0; u < l.size.width; u++)
            {
                const std::size_t pixel = index_of(l.size, u, v);
                const std::uint8_t mark = l.mask[pixel];
                bad_marks += mark > 1 ? 1 : 0;
                const std::uint8_t centre_mark =
                    frame_level.mask[index_of(frame_level.size, u * scale, v * scale)];
                not_centre_mark += mark != centre_mark ? 1 : 0;
                const float3 &p = l.points[pixel];
                const float3 &n = l.normals[pixel];
                const float intensity = l.intensity[pixel];
                if (mark != 1)
                {
                    invalid_not_zero += is_zero(p) && is_zero(n) && intensity == 0.0f ? 0 : 1;
                    continue;
                }
                valid++;
                const bool point_ok = std::isfinite(p.x) && std::isfinite(p.y) && p.z > 0.0f;
                const bool intensity_ok = intensity >= 0.0f && intensity <= 1.0f;
                bool normal_ok = is_zero(n);
                if (!normal_ok && std::isfinite(n.x) && std::isfinite(n.y) && std::isfinite(n.z))
                {
                    with_normal++;
                    length.add(std::abs(to_double(n).norm() - 1.0));
                    normal_ok = to_double(n).dot(to_double(p)) <= 0.0;
                }
                bad_valid += point_ok && intensity_ok && normal_ok ? 0 : 1;
                if (point_ok)
                {
                    const Eigen::Vector2d at_pixel = project(l.k, p);
                    off_pixel_px.add(
                        std::max(std::abs(at_pixel.x() - u), std::abs(at_pixel.y() - v)));
                }
            }
        }
        std::printf("%s: %ux%u, %d valid pixels, %d with a normal\n", at.c_str(), l.size.width,
                    l.size.height, valid, with_normal);
        check_within((at + ": marks other than 0 and 1").c_str(), bad_marks, 0);
        check_within((at + ": marks other than their centre pixel's at level 0").c_str(),
                     not_centre_mark, 0);
        check_within((at + ": invalid pixels whose point, normal or intensity is not 0").c_str(),
                     invalid_not_zero, 0);
        check_within(
            (at + ": valid pixels with z <= 0, an intensity outside [0, 1] or a normal away from "
                  "the camera")
                .c_str(),
            bad_valid, 0);
        check_max_error(at + ": | |n| - 1 |", length, length_tolerance);
        check_max_error(at + ": distance of a projected point from its pixel's centre, in px",
                        off_pixel_px, own_pixel_tolerance_px[level]);
    }
}

/// Returns the steepest incidence of the combining that reads a level, in double: θ_l with
/// tan θ_l = tan θ / 2^level [S: docs/methods/frame-pyramid.md, "One surface, at every level"].
/// @param level [in] the level read, the finer one.
/// @return θ_level, in degrees.
double incidence_of_level(std::size_t level)
{
    const double theta_rad =
        double{normals_parameters.max_incidence_deg} * std::numbers::pi / 180.0;
    return std::atan(std::tan(theta_rad) / scale_of(level)) * 180.0 / std::numbers::pi;
}

/// The largest share of a coarser level's valid pixels the reference may leave out near the rule's
/// threshold, at level 1 and 2 (index 0 unused): 2 % and 15 % [M 2026-11-14, this test: at most
/// 0.91 % at level 1, on the tabletop walk at 1280x800, and 12.0 % at level 2, on the tilted plane
/// at 1280x800, whose band seen within 1 degree of θ_1 = 61.8 degrees is wide]. The pixels left
/// out are still held to the other checks of every level.
constexpr std::array<double, level_count> undecided_share_tolerance = {0.0, 0.02, 0.15};

/// The comparison of a coarser level with the reference.
struct reference_level
{
    int decided = 0;   ///< valid pixels the reference decides
    int undecided = 0; ///< valid pixels left out: a pair near the rule's threshold
    int edges = 0;     ///< decided pixels with a valid pixel of their window across an edge
    /// For each pixel of the level: 1 when its window is whole, 9 valid pixels of the level before
    /// that are all combined, so that its point projects onto its pixel's centre up to the
    /// variation of z.
    std::vector<std::uint8_t> whole;
    max_error point_m;   ///< the point against the reference's, in m
    max_error intensity; ///< the intensity against the reference's
};

/// Computes the combining in double from the level before as downloaded, for every valid pixel
/// of each coarser level, and compares the step's points and intensities with it.
/// @param name [in] the case, printed in each line.
/// @param got [in] the levels.
/// @return the comparison of each coarser level (index 0 unused); checked: every point and
///         intensity within its tolerance, and the share of the pixels left out near the rule's
///         threshold within undecided_share_tolerance.
std::array<reference_level, level_count> check_reference(const std::string &name,
                                                         const pyramid_picture &got)
{
    std::array<reference_level, level_count> result;
    for (std::size_t level = 1; level < level_count; level++)
    {
        const level_picture &fine = got[level - 1];
        const level_picture &coarse = got[level];
        const double incidence_deg = incidence_of_level(level - 1);
        reference_level &r = result[level];
        r.whole.assign(coarse.mask.size(), 0);
        for (std::uint32_t v = 0; v < coarse.size.height; v++)
        {
            for (std::uint32_t u = 0; u < coarse.size.width; u++)
            {
                const std::size_t pixel = index_of(coarse.size, u, v);
                const std::size_t centre = index_of(fine.size, 2 * u, 2 * v);
                if (coarse.mask[pixel] != 1 || fine.mask[centre] != 1)
                    continue;
                const Eigen::Vector3d centre_point = to_double(fine.points[centre]);
                Eigen::Vector3d point_sum = Eigen::Vector3d::Zero();
                double intensity_sum = 0.0;
                double weight_sum = 0.0;
                int combined = 0;
                bool near = false;
                bool edge = false;
                for (int j = -1; j <= 1 && !near; j++)
                {
                    for (int i = -1; i <= 1; i++)
                    {
                        const long long column = 2LL * u + i;
                        const long long row = 2LL * v + j;
                        if (column < 0 || row < 0 || column >= fine.size.width ||
                            row >= fine.size.height)
                            continue;
                        const std::size_t neighbour =
                            index_of(fine.size, static_cast<std::uint32_t>(column),
                                     static_cast<std::uint32_t>(row));
                        if (fine.mask[neighbour] != 1)
                            continue;
                        const Eigen::Vector3d point = to_double(fine.points[neighbour]);
                        const pair_side side = i == 0 && j == 0
                                                   ? pair_side::one_surface
                                                   : side_of(centre_point, point, incidence_deg);
                        if (side == pair_side::near)
                        {
                            near = true;
                            break;
                        }
                        if (side == pair_side::edge)
                        {
                            edge = true;
                            continue;
                        }
                        const double weight = (2.0 - std::abs(i)) * (2.0 - std::abs(j));
                        point_sum += weight * point;
                        intensity_sum += weight * fine.intensity[neighbour];
                        weight_sum += weight;
                        combined++;
                    }
                }
                if (near)
                {
                    r.undecided++;
                    continue;
                }
                r.decided++;
                r.edges += edge ? 1 : 0;
                r.whole[pixel] = combined == 9 ? 1 : 0;
                r.point_m.add((to_double(coarse.points[pixel]) - point_sum / weight_sum).norm());
                r.intensity.add(std::abs(coarse.intensity[pixel] - intensity_sum / weight_sum));
            }
        }
        const std::string at = name + ", level " + std::to_string(level);
        std::printf("%s, reference: %d pixels decided (%d beside a depth edge), %d left out near "
                    "the rule's threshold\n",
                    at.c_str(), r.decided, r.edges, r.undecided);
        check_max_error(at + ", reference: point, in m", r.point_m, reference_point_tolerance_m);
        check_max_error(at + ", reference: intensity", r.intensity, reference_intensity_tolerance);
        const int valid = r.decided + r.undecided;
        check(valid == 0 || r.decided > 0, (at + ", reference: pixels decided").c_str());
        check_within((at + ", reference: share of the valid pixels left out").c_str(),
                     valid == 0 ? 0.0 : static_cast<double>(r.undecided) / valid,
                     undecided_share_tolerance[level]);
    }
    return result;
}

/// Runs the step on one frame, downloads every level and checks them and the reference.
/// @param name [in] the case, printed in each line.
/// @param frame [in] the frame, of the stage's size.
/// @param range [in] the depth range.
/// @param stage [in,out] the buffers and stream of the frame's size.
/// @param reference [out] the comparison with the reference.
/// @return the levels; empty ones when the step did not run.
pyramid_picture run_and_check(const std::string &name, const gca::frame &frame,
                              const gca::organized_points_parameters &range, stage_buffers &stage,
                              std::array<reference_level, level_count> &reference)
{
    if (!run_stage(name, frame, stage, range))
        return {};
    pyramid_picture got = download(name, stage);
    if (!complete(got))
        return {};
    check_levels(name, got, frame.description());
    reference = check_reference(name, got);
    return got;
}

/// The largest angle and median angle of one level's normals against the analytic ones that
/// pass, in degrees.
struct angle_tolerance
{
    double largest_deg; ///< the largest angle that passes
    double median_deg;  ///< the largest median that passes
};

/// The tolerances of one plane at one stream size: of each level's normals against the plane's,
/// and of the projection of a whole window's point.
struct plane_tolerances
{
    std::array<angle_tolerance, level_count> normals; ///< each level's normals
    double whole_window_px; ///< a whole window's point from its pixel's centre, in px
};

/// Checks one plane's picture: every level's points lie on the plane z = 0 of the world within
/// the depth's rounding, a whole window's point projects onto its pixel's centre, and every
/// level's normals agree with the plane's.
/// @param name [in] the case, printed.
/// @param frame [in] the frame, of the stage's size.
/// @param range [in] the depth range.
/// @param stage [in,out] the stage of the frame's size.
/// @param pose [in] the frame's pose, camera frame to world frame.
/// @param tolerance [in] the tolerances of the plane at this size.
void check_plane(const std::string &name, const gca::frame &frame,
                 const gca::organized_points_parameters &range, stage_buffers &stage,
                 const Eigen::Isometry3d &pose, const plane_tolerances &tolerance)
{
    std::array<reference_level, level_count> reference;
    const pyramid_picture got = run_and_check(name, frame, range, stage, reference);
    if (!complete(got))
        return;
    // The plane's normal, up from z = 0 in the world, in the camera frame; the camera is above it.
    const Eigen::Vector3d up = pose.linear().transpose() * Eigen::Vector3d::UnitZ();
    for (std::size_t level = 0; level < level_count; level++)
    {
        const level_picture &l = got[level];
        const std::string at = name + ", level " + std::to_string(level);
        max_error off_plane;
        max_error whole_off_px;
        max_error angles;
        std::vector<double> all;
        for (std::uint32_t v = 0; v < l.size.height; v++)
        {
            for (std::uint32_t u = 0; u < l.size.width; u++)
            {
                const std::size_t pixel = index_of(l.size, u, v);
                if (l.mask[pixel] != 1)
                    continue;
                const Eigen::Vector3d world = pose * to_double(l.points[pixel]);
                const double ray_length =
                    gca_test::pixel_ray(l.k, static_cast<double>(u), static_cast<double>(v)).norm();
                off_plane.add(std::abs(world.z()) / (camera_depth_unit_m * ray_length));
                if (level > 0 && reference[level].whole[pixel] == 1)
                {
                    const Eigen::Vector2d at_pixel = project(l.k, l.points[pixel]);
                    whole_off_px.add(
                        std::max(std::abs(at_pixel.x() - u), std::abs(at_pixel.y() - v)));
                }
                if (is_zero(l.normals[pixel]))
                    continue;
                const double angle = angle_deg(to_double(l.normals[pixel]), up);
                angles.add(angle);
                all.push_back(angle);
            }
        }
        check_max_error(at + ": distance from the plane over depth unit x |ray|", off_plane,
                        plane_tolerance[level]);
        if (level > 0)
            check_max_error(at + ": a whole window's point from its pixel's centre, in px",
                            whole_off_px, tolerance.whole_window_px);
        check(!all.empty(), (at + ": normals to compare with the plane's").c_str());
        check_max_error(at + ": angle to the plane's normal, in degrees", angles,
                        tolerance.normals[level].largest_deg);
        check_within((at + ": median angle to the plane's normal, in degrees").c_str(),
                     median(std::move(all)), tolerance.normals[level].median_deg);
    }
}

/// The tolerances of the planes at 640x400, each a little above its measured value [M 2026-11-14,
/// this test]. The normals, largest / median in degrees at level 0, 1 and 2: tilted plane 4.90 /
/// 0.321, 0.642 / 0.0623, 0.130 / 0.0109 (at 641x401 3.28 / 0.321 at level 0, the rest the same);
/// plane at 2 m 1.49 / 0.118, 0.395 / 0.0219, 0.0449 / 0.00424. Level 0's are the normals' test's;
/// a coarser level's window spans more of the plane over averaged depths
/// (docs/methods/frame-pyramid.md, "Accuracy"). A whole window's point from its pixel's centre at
/// level 1 and 2: tilted plane 0.00161 and 0.0235 px, plane at 2 m 0.000437 and 0.00101 px; a
/// whole window of level 2 may combine points of level 1 whose own windows are cut.
constexpr plane_tolerances small_tilted = {{{{6.0, 0.4}, {0.8, 0.08}, {0.2, 0.015}}}, 0.03};
constexpr plane_tolerances small_far = {{{{2.0, 0.15}, {0.5, 0.03}, {0.06, 0.006}}}, 0.002};

/// The tolerances of the planes at 1280x800, as at 640x400 [M 2026-11-14, this test]. The
/// normals: tilted plane 10.6 / 0.775, 1.35 / 0.103, 0.298 / 0.0160; plane at 2 m 4.53 / 0.250,
/// 0.860 / 0.0386, 0.209 / 0.00628. A whole window's point: tilted plane 0.000813 and 0.0227 px,
/// plane at 2 m 0.000234 and 0.000514 px.
constexpr plane_tolerances large_tilted = {{{{13.0, 1.0}, {1.7, 0.13}, {0.4, 0.02}}}, 0.03};
constexpr plane_tolerances large_far = {{{{5.5, 0.3}, {1.1, 0.05}, {0.3, 0.008}}}, 0.002};

/// Checks the planes: the tilted plane, seen from 0.65 m looking down at about 32 degrees and
/// rolled by 10 degrees, and a plane 2 m away, seen at 20 degrees off its normal at the picture's
/// centre and rolled by 10 degrees [?: choices, the normals' test's], at both stream sizes; and the
/// tilted plane at a size that is no multiple of the kernels' block.
/// @param small [in,out] the stage of 640x400.
/// @param large [in,out] the stage of 1280x800.
/// @param odd_sized [in,out] the stage of 641x401.
void check_planes(stage_buffers &small, stage_buffers &large, stage_buffers &odd_sized)
{
    const std::optional<Eigen::Isometry3d> tilted_pose =
        gca_test::look_at({0.0, -0.55, 0.35}, {0.0, 0.0, 0.0}, 10.0);
    const double elevation_rad = 70.0 * std::numbers::pi / 180.0;
    const std::optional<Eigen::Isometry3d> far_pose =
        gca_test::look_at({0.0, -2.0 * std::cos(elevation_rad), 2.0 * std::sin(elevation_rad)},
                          {0.0, 0.0, 0.0}, 10.0);
    check(tilted_pose && far_pose, "the planes: look_at() gives both poses");
    if (!tilted_pose || !far_pose)
        return;
    struct plane_case
    {
        const char *name;                             ///< the case, printed
        const gca::frame_description &description;    ///< its size and intrinsics
        stage_buffers &stage;                         ///< the stage of that size
        const gca_test::synthetic_plane &plane;       ///< the plane rendered
        const Eigen::Isometry3d &pose;                ///< the view
        const gca::organized_points_parameters range; ///< the depth range
        const plane_tolerances &tolerance;            ///< what it is held to
    };
    const plane_case cases[] = {
        {"tilted plane at 640x400", small_description, small, tabletop_plane, *tilted_pose,
         tilted_range, small_tilted},
        {"plane at 2 m at 640x400", small_description, small, far_plane, *far_pose, far_range,
         small_far},
        {"tilted plane at 1280x800", large_description, large, tabletop_plane, *tilted_pose,
         tilted_range, large_tilted},
        {"plane at 2 m at 1280x800", large_description, large, far_plane, *far_pose, far_range,
         large_far},
        {"tilted plane at 641x401", odd_description, odd_sized, tabletop_plane, *tilted_pose,
         tilted_range, small_tilted},
    };
    for (const plane_case &c : cases)
    {
        const rendered scene = render({.plane = c.plane, .trajectory = {c.pose}}, c.description);
        if (!scene.frames.empty())
            check_plane(c.name, scene.frames[0], c.range, c.stage, c.pose, c.tolerance);
    }
}

/// The largest distance of a point of any level from the nearest known surface of the tabletop
/// scene, in m. What this checks is that claim, every point within 2.5 mm of a known surface, not
/// the checklist's "no point of a coarser level lies between the two surfaces" itself. That one
/// rests on the reference (check_reference()): on every frame a coarser point combines exactly the
/// pairs the rule in double takes for one surface, so a point combined across a depth edge fails
/// it. This check bounds what the rule leaves: a mean across a step the rule takes for one surface
/// lies off each side by up to half the step, (z / fx) tan 75 deg / 2 = 2.75 mm at 0.45 m and more
/// farther away [S: docs/methods/frame-pyramid.md, "One surface, at every level"]. 2.5 mm lies a
/// little above the largest on these frames, 1.87 mm at level 2, 1.01 mm at level 1 and 0.63 mm at
/// level 0, the depth's rounding [M 2026-11-14, this test], so it holds for this scene's steps,
/// not for every step the method allows. Combined across the box's edges, without the rule, points
/// lie up to 62.5 mm off [M 2026-11-14: the rule dropped from the combining, a mutant].
constexpr double between_surfaces_tolerance_m = 2.5e-3;

/// Checks the frames of the tabletop scene: every level and the reference, and that every point
/// of every level lies within between_surfaces_tolerance_m of a known surface.
/// @param name [in] the case, printed.
/// @param scene [in] the scene's frames and known surfaces.
/// @param trajectory [in] the frames' poses, camera frame to world frame.
/// @param stage [in,out] the stage of the frames' size.
/// @return the coarser pixels beside a depth edge, over the frames.
int check_box_frames(const std::string &name, const rendered &scene,
                     const std::vector<Eigen::Isometry3d> &trajectory, stage_buffers &stage)
{
    int edges = 0;
    std::array<max_error, level_count> from_surface_m;
    for (std::size_t i = 0; i < scene.frames.size(); i++)
    {
        const std::string frame_name = name + ", frame " + std::to_string(i);
        std::array<reference_level, level_count> reference;
        const pyramid_picture got =
            run_and_check(frame_name, scene.frames[i], box_range, stage, reference);
        if (!complete(got))
            continue;
        for (std::size_t level = 0; level < level_count; level++)
        {
            edges += level > 0 ? reference[level].edges : 0;
            const level_picture &l = got[level];
            double largest = 0.0;
            std::uint32_t largest_u = 0;
            std::uint32_t largest_v = 0;
            for (std::uint32_t v = 0; v < l.size.height; v++)
            {
                for (std::uint32_t u = 0; u < l.size.width; u++)
                {
                    const std::size_t pixel = index_of(l.size, u, v);
                    if (l.mask[pixel] != 1)
                        continue;
                    const Eigen::Vector3d world = trajectory[i] * to_double(l.points[pixel]);
                    double nearest = std::numeric_limits<double>::infinity();
                    for (const gca_test::synthetic_surface &surface : scene.surfaces)
                        nearest = std::min(nearest, distance_from(surface, world));
                    from_surface_m[level].add(nearest);
                    if (nearest > largest)
                    {
                        largest = nearest;
                        largest_u = u;
                        largest_v = v;
                    }
                }
            }
            std::printf("%s, level %zu: the largest distance from a known surface is %.3g mm, "
                        "at pixel (%u, %u)\n",
                        frame_name.c_str(), level, largest * 1e3, largest_u, largest_v);
        }
    }
    for (std::size_t level = 0; level < level_count; level++)
        check_max_error(name + ", level " + std::to_string(level) +
                            ": distance from the nearest known surface, in m",
                        from_surface_m[level], between_surfaces_tolerance_m);
    return edges;
}

/// Checks the tabletop walk: three frames at 640x400 and the first at 1280x800, with depth edges
/// where the box hides the plane and the plane's edge where the view looks past it.
/// @param small [in,out] the stage of 640x400.
/// @param large [in,out] the stage of 1280x800.
void check_tabletop_walk(stage_buffers &small, stage_buffers &large)
{
    // Three views of the tabletop walk, 120 degrees apart [?: a choice, the normals' test's].
    gca_test::synthetic_walk walk = gca_test::tabletop_walk();
    walk.pose_count = 3;
    const std::optional<std::vector<Eigen::Isometry3d>> trajectory = gca_test::walk_around(walk);
    check(trajectory.has_value(), "tabletop walk: walk_around() gives the walk");
    if (!trajectory)
        return;
    const rendered scene =
        render({.plane = tabletop_plane, .box = tabletop_box, .trajectory = *trajectory},
               small_description);
    const int edges = check_box_frames("tabletop walk at 640x400", scene, *trajectory, small);
    std::printf("tabletop walk at 640x400: %d coarser pixels beside a depth edge\n", edges);
    check(scene.frames.size() == 3 && edges > 0,
          "tabletop walk at 640x400: three frames with depth edges at the coarser levels");
    const std::vector<Eigen::Isometry3d> first = {trajectory->front()};
    const rendered large_scene = render(
        {.plane = tabletop_plane, .box = tabletop_box, .trajectory = first}, large_description);
    const int large_edges =
        check_box_frames("tabletop walk at 1280x800", large_scene, first, large);
    check(large_scene.frames.size() == 1 && large_edges > 0,
          "tabletop walk at 1280x800: a frame with depth edges at the coarser levels");
}

/// The area without depth of the built frame: columns 200 to 263 and rows 100 to 163, 64 x 64
/// pixels, so that it holds centre pixels of every level [?: a choice].
constexpr std::uint32_t hole_first = 100;
constexpr std::uint32_t hole_columns_from = 200;
constexpr std::uint32_t hole_side = 64;

/// The built frame's column without depth on a pixel that is the centre pixel of a pixel of every
/// level, 404 = 4 x 101, and the one between centre pixels of level 1, 401 [?: choices].
constexpr std::uint32_t centre_column = 404;
constexpr std::uint32_t between_column = 401;

/// Checks a flat frame at 0.5 m with an area without depth and two columns without depth: every
/// pixel whose centre pixel lies in the area or on the column at 404 is invalid at every level;
/// the column at 401 leaves no invalid pixel at a coarser level; and every valid point of every
/// level lies at the frame's depth, none pulled toward a pixel without depth.
/// @param stage [in,out] the stage of the smaller stream size.
void check_area_without_depth(stage_buffers &stage)
{
    const std::string name = "flat frame with an area without depth";
    const auto in_hole = [](std::uint32_t u, std::uint32_t v) {
        return u >= hole_columns_from && u < hole_columns_from + hole_side && v >= hole_first &&
               v < hole_first + hole_side;
    };
    const gca::frame frame = built_frame(
        small_description,
        [&](std::uint32_t u, std::uint32_t v) -> std::uint16_t {
            return in_hole(u, v) || u == centre_column || u == between_column ? 0 : built_depth;
        },
        [](std::uint32_t, std::uint32_t) { return std::array<std::uint8_t, 3>{90, 160, 30}; });
    std::array<reference_level, level_count> reference;
    const pyramid_picture got = run_and_check(name, frame, built_range, stage, reference);
    if (!complete(got))
        return;
    const float depth_m = got[0].points[index_of(got[0].size, 0, 0)].z;
    for (std::size_t level = 0; level < level_count; level++)
    {
        const level_picture &l = got[level];
        const std::uint32_t scale = scale_of(level);
        const std::string at = name + ", level " + std::to_string(level);
        int in_area = 0;
        int in_area_valid = 0;
        int on_centre_column_valid = 0;
        int beside_between_invalid = 0;
        max_error off_depth_m;
        for (std::uint32_t v = 0; v < l.size.height; v++)
        {
            for (std::uint32_t u = 0; u < l.size.width; u++)
            {
                const std::size_t pixel = index_of(l.size, u, v);
                const bool valid = l.mask[pixel] == 1;
                if (in_hole(u * scale, v * scale))
                {
                    in_area++;
                    in_area_valid += valid ? 1 : 0;
                }
                if (u * scale == centre_column)
                    on_centre_column_valid += valid ? 1 : 0;
                if (level > 0 && u == between_column / scale && !in_hole(u * scale, v * scale))
                    beside_between_invalid += valid ? 0 : 1;
                if (valid)
                    off_depth_m.add(std::abs(double{l.points[pixel].z} - depth_m));
            }
        }
        std::printf("%s: %d pixels whose centre pixel lies in the area without depth\n", at.c_str(),
                    in_area);
        check(in_area > 0 && in_area_valid == 0,
              (at + ": every pixel of the area without depth is invalid").c_str());
        check_within((at + ": valid pixels on the column without depth at 404").c_str(),
                     on_centre_column_valid, 0);
        if (level > 0)
            check_within(
                (at + ": invalid pixels whose centre pixel is beside the column at 401").c_str(),
                beside_between_invalid, 0);
        // Every combined point lies at the same depth: their offsets' z is 0, so their mean's z is
        // the centre's [S: float32 arithmetic]; 1e-7 m, a float step of 0.5 m, allows one rounding
        // [S: float32 ulp]. The largest is 0 [M 2026-11-14, this test].
        check_max_error(at + ": z of a valid point against the frame's depth, in m", off_depth_m,
                        1e-7);
    }
}

/// The side of a cell of the known colors: 16 px, so that a pixel of level 2, whose window spans
/// 7 pixels of the frame, lies inside one cell at the cell's middle [?: a choice].
constexpr std::uint32_t color_cell = 16;

/// Returns the color of a cell of the known colors: black, white, red, green, blue and grey 128
/// first, then colors of a fixed sequence of whole numbers [?: a choice].
/// @param cell [in] the cell's number, in raster order.
/// @return red, green and blue, 0 to 255.
std::array<std::uint8_t, 3> cell_color(std::uint32_t cell)
{
    static constexpr std::array<std::array<std::uint8_t, 3>, 6> named = {
        {{0, 0, 0}, {255, 255, 255}, {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {128, 128, 128}}};
    if (cell < named.size())
        return named[cell];
    // A linear congruential sequence [S: Numerical Recipes' constants 1664525, 1013904223].
    const std::uint32_t mixed = cell * 1664525u + 1013904223u;
    return {static_cast<std::uint8_t>(mixed >> 24), static_cast<std::uint8_t>(mixed >> 16),
            static_cast<std::uint8_t>(mixed >> 8)};
}

/// Returns the Rec. 709 luma of a color in double, scaled to [0, 1] [S: ITU-R BT.709-6].
/// @param rgb [in] red, green and blue, 0 to 255.
/// @return (0.2126 R + 0.7152 G + 0.0722 B) / 255.
double rec709(const std::array<std::uint8_t, 3> &rgb)
{
    return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255.0;
}

/// Checks the intensity of known colors on a flat frame at 0.5 m tiled by cells of one color each:
/// at level 0 every pixel's intensity is the Rec. 709 luma in double and the oracle's, the
/// prototype's color3::to_intensity() of the color scaled by 1/255 as create_from_rgbd scales it;
/// at a coarser level a pixel whose window lies inside one cell has that cell's luma.
/// @param stage [in,out] the stage of the smaller stream size.
void check_known_colors(stage_buffers &stage)
{
    const std::string name = "cells of known colors";
    const std::uint32_t cells_in_row = small_description.size.width / color_cell;
    const auto cell_of = [&](std::uint32_t u, std::uint32_t v) {
        return (v / color_cell) * cells_in_row + u / color_cell;
    };
    const gca::frame frame = built_frame(
        small_description, [](std::uint32_t, std::uint32_t) { return built_depth; },
        [&](std::uint32_t u, std::uint32_t v) { return cell_color(cell_of(u, v)); });
    std::array<reference_level, level_count> reference;
    const pyramid_picture got = run_and_check(name, frame, built_range, stage, reference);
    if (!complete(got))
        return;
    max_error against_rec709;
    max_error against_oracle;
    const level_picture &frame_level = got[0];
    for (std::uint32_t v = 0; v < frame_level.size.height; v++)
    {
        for (std::uint32_t u = 0; u < frame_level.size.width; u++)
        {
            const std::array<std::uint8_t, 3> rgb = cell_color(cell_of(u, v));
            const double intensity = frame_level.intensity[index_of(frame_level.size, u, v)];
            against_rec709.add(std::abs(intensity - rec709(rgb)));
            // The oracle: create_from_rgbd scales each channel by 1/255 in float
            // (cuda_point_cloud_factory.cu), to_intensity() takes the luma in double (type.hpp).
            const float scale = 1.0f / 255.0f;
            const gca::color3 color{static_cast<float>(rgb[0]) * scale,
                                    static_cast<float>(rgb[1]) * scale,
                                    static_cast<float>(rgb[2]) * scale};
            against_oracle.add(std::abs(intensity - double{color.to_intensity()}));
        }
    }
    check_max_error(name + ", level 0: intensity against the Rec. 709 luma in double",
                    against_rec709, intensity_tolerance);
    check_max_error(name + ", level 0: intensity against the oracle's", against_oracle,
                    intensity_tolerance);
    for (std::size_t level = 1; level < level_count; level++)
    {
        const level_picture &l = got[level];
        const std::uint32_t scale = scale_of(level);
        max_error inside_cell;
        int compared = 0;
        for (std::uint32_t v = 0; v < l.size.height; v++)
        {
            for (std::uint32_t u = 0; u < l.size.width; u++)
            {
                // The window spans the frame's pixels within scale - 1 of the centre pixel.
                const std::uint32_t cu = u * scale;
                const std::uint32_t cv = v * scale;
                const std::uint32_t reach = scale - 1;
                if (cu < reach || cv < reach || cu + reach >= frame_level.size.width ||
                    cv + reach >= frame_level.size.height ||
                    cell_of(cu - reach, cv - reach) != cell_of(cu + reach, cv + reach))
                    continue;
                compared++;
                inside_cell.add(std::abs(l.intensity[index_of(l.size, u, v)] -
                                         rec709(cell_color(cell_of(cu, cv)))));
            }
        }
        const std::string at = name + ", level " + std::to_string(level);
        check(compared > 0, (at + ": pixels inside one cell").c_str());
        check_max_error(at + ": intensity inside one cell against its Rec. 709 luma", inside_cell,
                        intensity_tolerance);
    }
}

/// Says whether two downloads are equal byte for byte, intrinsics included.
/// @param a [in] the first.
/// @param b [in] the second.
/// @return whether every level's images and intrinsics are equal bit for bit.
bool same_levels(const pyramid_picture &a, const pyramid_picture &b)
{
    for (std::size_t level = 0; level < level_count; level++)
    {
        const level_picture &x = a[level];
        const level_picture &y = b[level];
        const auto same = [](const auto &p, const auto &q) {
            return p.size() == q.size() &&
                   std::memcmp(p.data(), q.data(), p.size() * sizeof(p[0])) == 0;
        };
        if (x.points.empty() || !same(x.points, y.points) || !same(x.mask, y.mask) ||
            !same(x.normals, y.normals) || !same(x.intensity, y.intensity) ||
            std::memcmp(&x.k, &y.k, sizeof(x.k)) != 0)
            return false;
    }
    return true;
}

/// The GPU memory the refusals use, all requested before the pool closes.
struct refusal_buffers
{
    gca::gpu_frame large_frame; ///< a GPU frame of the larger size, with a frame uploaded
    gca::gpu_frame tall_frame;  ///< a GPU frame of tall_size, with a frame uploaded
    gca::frame_pyramid tall;    ///< a pyramid of tall_size
};

/// Checks the refusals that leave every level as it was: an empty pyramid, normals' parameters
/// that cannot be used, the steepest incidence just above its largest among them, a depth range
/// that is none, a GPU frame that holds no frame, a frame of another size, and CUDA's last error
/// left by an earlier call. The levels are filled with 0xFF bytes first, so that any work a
/// refusal queued, which with the same frame and range would write the same values again, shows.
/// Then the largest steepest incidence itself is accepted.
/// @param stage [in,out] the stage of the smaller stream size, built from a frame.
/// @param buffers [in,out] the other GPU memory the refusals use.
void check_refusals_leave_levels(stage_buffers &stage, refusal_buffers &buffers)
{
    gca::frame_pyramid &pyramid = stage.pyramid;
    const cudaStream_t stream = stage.stream;
    const gca::frame_pyramid_parameters usable = {built_range, normals_parameters};
    fill_with_0xff("refusals", pyramid, stream);
    const pyramid_picture before = download("refusals, before", stage);
    check_refused("an empty pyramid", gca::frame_pyramid{}.build(stage.frame, usable, stream),
                  "the frame's pyramid holds no levels");
    check_refused("a window of radius 4 px",
                  pyramid.build(stage.frame, {built_range, {4, 25, 75.0f}}, stream),
                  "the window's radius 4 px is not 1 to 3");
    check_refused("an incidence of 90 deg",
                  pyramid.build(stage.frame, {built_range, {3, 25, 90.0f}}, stream),
                  "is not a finite number above 0 and at most 89");
    const float above_largest =
        std::nextafter(gca::max_steepest_incidence_deg, std::numeric_limits<float>::infinity());
    check_refused("an incidence one float step above 89 deg",
                  pyramid.build(stage.frame, {built_range, {3, 25, above_largest}}, stream),
                  "is not a finite number above 0 and at most 89");
    check_refused("a depth range from 1 m to 0.5 m",
                  pyramid.build(stage.frame, {{1.0f, 0.5f}, normals_parameters}, stream),
                  "is not one");
    check_refused("a GPU frame that holds no frame",
                  pyramid.build(gca::gpu_frame{}, usable, stream), "holds no frame");
    check_refused("a frame of another size", pyramid.build(buffers.large_frame, usable, stream),
                  "the frame is 1280x800 px, the organized points 640x400 px");
    // cudaMalloc's refusal of 2^52 bytes, 4 PiB on a board of 7.4 GiB [S: docs/HARDWARE.md],
    // leaves its error as CUDA's last error.
    void *never_granted = nullptr;
    check(cudaMalloc(&never_granted, std::size_t{1} << 52) != cudaSuccess,
          "cudaMalloc refuses 2^52 bytes");
    check_refused("CUDA's last error left by an earlier call",
                  pyramid.build(stage.frame, usable, stream), "left by an earlier call");
    const pyramid_picture after = download("refusals, after", stage);
    check(same_levels(before, after),
          "refusals: every level and its intrinsics left as they were, bit for bit");
    check_no_cuda_error("refusals");
    const gca::step_result largest =
        pyramid.build(stage.frame, {built_range, {3, 25, gca::max_steepest_incidence_deg}}, stream);
    check(largest.done, ("an incidence of 89 deg: built, " + largest.message).c_str());
    const pyramid_picture at_largest = download("an incidence of 89 deg", stage);
    if (complete(at_largest))
        check_levels("an incidence of 89 deg", at_largest, stage.frame.description());
}

/// Checks CUDA's refusal of the first launch for a frame of 1 x 600000 px, whose rows need 75000
/// blocks along y where a grid holds 65535: a failure with CUDA's error, which is cleared.
/// @param buffers [in,out] the GPU frame and the pyramid of tall_size.
/// @param stream [in] the CUDA stream the work runs on.
void check_refused_launch(refusal_buffers &buffers, cudaStream_t stream)
{
    check_refused("a frame of 1x600000 px, 75000 blocks along y",
                  buffers.tall.build(buffers.tall_frame, {built_range, normals_parameters}, stream),
                  "cudaErrorInvalidConfiguration");
    check_no_cuda_error("the refused launch");
}

/// Requests a stage's buffers of one size from the pool.
/// @param pool [in,out] the pool, open.
/// @param size [in] the size, in px.
/// @param stream [in] the CUDA stream the work runs on.
/// @return the stage's buffers; empty ones, with a failed check, where a request was refused.
stage_buffers request_stage(gca::gpu_pool &pool, const gca::frame_size &size, cudaStream_t stream)
{
    return {granted("a GPU frame", gca::gpu_frame::request(pool, size)),
            granted("a frame's pyramid", gca::frame_pyramid::request(pool, size)), stream};
}

/// Uploads a frame of a size whose depth is built_depth everywhere and waits for it.
/// @param name [in] the case, named in the check line.
/// @param description [in] the frame's description.
/// @param frame [in,out] the GPU frame of that size.
/// @param stream [in] the CUDA stream the upload is queued on.
void upload_flat(const std::string &name, const gca::frame_description &description,
                 gca::gpu_frame &frame, cudaStream_t stream)
{
    const gca::frame host = built_frame(
        description, [](std::uint32_t, std::uint32_t) { return built_depth; },
        [](std::uint32_t, std::uint32_t) { return std::array<std::uint8_t, 3>{128, 128, 128}; });
    const gca::step_result up = frame.upload(host, stream);
    check(up.done && cudaStreamSynchronize(stream) == cudaSuccess,
          (name + ": uploaded, " + up.message).c_str());
}

/// Requests every buffer, closes the pool, runs the checks and checks that the step asked the
/// closed pool for nothing.
/// @param stream [in] the CUDA stream the work runs on.
void check_with_closed_pool(cudaStream_t stream)
{
    gca::gpu_pool pool;
    stage_buffers small = request_stage(pool, small_description.size, stream);
    stage_buffers odd_sized = request_stage(pool, odd_description.size, stream);
    stage_buffers large = request_stage(pool, large_description.size, stream);
    refusal_buffers buffers{
        granted("a GPU frame", gca::gpu_frame::request(pool, large_description.size)),
        granted("a GPU frame", gca::gpu_frame::request(pool, tall_size)),
        granted("a frame's pyramid", gca::frame_pyramid::request(pool, tall_size))};
    pool.close();
    upload_flat("the larger frame", large_description, buffers.large_frame, stream);
    upload_flat("the tall frame", {tall_size, small_description.intrinsics, camera_depth_unit_m},
                buffers.tall_frame, stream);
    check_known_colors(small);
    check_area_without_depth(small);
    check_planes(small, large, odd_sized);
    check_tabletop_walk(small, large);
    check_refusals_leave_levels(small, buffers);
    check_refused_launch(buffers, stream);
    check_within("requests to the closed pool", static_cast<double>(pool.refused_while_closed()),
                 0);
}
} // namespace

/// Runs the checks of the frame's pyramid and prints the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    cudaStream_t stream = nullptr;
    if (!create_stream(stream))
        return gca_test::check_summary("test_frame_pyramid");
    check_with_closed_pool(stream);
    check(cudaStreamDestroy(stream) == cudaSuccess, "the CUDA stream destroyed");
    return gca_test::check_summary("test_frame_pyramid");
}
