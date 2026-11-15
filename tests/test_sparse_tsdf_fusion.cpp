// SPDX-License-Identifier: 0BSD
/// @file test_sparse_tsdf_fusion.cpp
/// The fusion module's integration of a frame and extraction of the surface's points
/// (fusion/sparse_tsdf.hpp, sparse_tsdf::integrate() and sparse_tsdf::extract()), on frames of the
/// tests' synthetic source at their known poses and on built frames of walls, 640x400 with the
/// camera's intrinsics. The method is docs/methods/sparse-tsdf.md, "Integration, and the weight of
/// an observation" and "The extraction". The models' frame is the scene's world frame. The
/// expected surfaces, normals and colors come from the analytic scene (synthetic_source.hpp) and
/// from the method's formulas computed here, apart from the module's code.
/// - Walls facing the camera, at whole millimetres, so that every pixel's depth is exact: one
///   frame gives its wall, normals towards the camera and its color; a second wall 4 mm behind,
///   fused once, moves the surface halfway and the color to the mean, a tie rounded to even; a
///   wall fused 64 times and then the other 64 times leaves the surface at 1 - (64/65)^64 of the
///   way, as the weight limited to 64 gives, and the colors at the method's rounded recursion.
/// - The far depth: a frame half of whose pixels lie beyond it changes nothing there.
/// - A sphere resting on a plane, fused from a walk of 12 known poses around it, at the voxel sizes
///   of both object classes, the live model's and the final models': the points' distances from
///   the analytic surface, their median and their largest, the angles of their normals from the
///   surface's and the errors of their colors against the texture there, each's median and 99th
///   percentile, lie within measured tolerances; every normal is of unit length within the model
///   file's 1e-6; all but a stated share of the samples of the surface the frames saw have a
///   point near it.
/// - Noisy frames of a plane along the same walk: the fused surface spreads less about the plane
///   than one frame's points, by a stated factor.
/// - A pool the frame uses up: the list holds the pool's blocks and every count adds up, the
///   voxels of the blocks with a number are fused, and only those give points.
/// - Refusals as values, each leaving the counts and the points as they were; a buffer of points
///   too small, reported with both counts and its first points written.
/// - Every buffer is requested before the pool closes, and the integrations and extractions ask
///   the closed pool for nothing.

#include "camera/frame.hpp"
#include "check.hpp"
#include "fusion/sparse_tsdf.hpp"
#include "geometry/organized_points.hpp"
#include "gpu/gpu_frame.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/step_result.hpp"
#include "gpu_test_helpers.hpp"
#include "scan/object_class.hpp"
#include "sparse_tsdf_helpers.hpp"
#include "stage_checks.hpp"
#include "stage_frames.hpp"
#include "synthetic_source.hpp"
#include "types/camera_pose.hpp"
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
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::angle_deg;
using gca_test::as_camera_pose;
using gca_test::check;
using gca_test::check_no_cuda_error;
using gca_test::check_refused;
using gca_test::check_within;
using gca_test::create_stream;
using gca_test::frame_buffers;
using gca_test::granted;
using gca_test::max_error;
using gca_test::method_truncation;
using gca_test::print_usage;
using gca_test::quantile;
using gca_test::read_usage;
using gca_test::request_buffers;
using gca_test::same_usage;
using gca_test::small_description;

/// The blocks of the cases' sparse TSDF, and its budget: 65536 x 4100 + 262144 x 16 + 16 bytes
/// [S: computed, docs/methods/sparse-tsdf.md, "The pool and the memory budget"], 273 MB, and a
/// room for one frame of 196608 blocks, above the bound of every case's parameters.
constexpr std::uint32_t test_blocks = 65536;
constexpr std::uint64_t test_budget = 272'891'920;

/// The points the cases' buffer holds: 2 million, 56 MB [?: a choice, above every case's count].
constexpr std::uint32_t point_capacity = 2'000'000;

/// The walls' depths, in mm, and their colors: the second wall 4 mm, 2 voxels of 2 mm, behind the
/// first, inside the truncation of both, 8 mm; the colors differ in every channel, and the green
/// channels' mean, 104.5, is a tie [?: a choice].
constexpr std::uint16_t wall_mm = 500;
constexpr std::uint16_t behind_wall_mm = 504;
constexpr std::array<std::uint8_t, 3> wall_color = {40, 90, 200};
constexpr std::array<std::uint8_t, 3> behind_wall_color = {200, 119, 60};

/// The frames of each wall in the weight's case: as many as the weight's limit, so that the voxels
/// hold max_voxel_weight observations of the first wall before the second comes.
constexpr int frames_per_wall = gca::max_voxel_weight;

/// The tolerance of a wall's depth in the cases of one and two frames: the crossings' fractions are
/// 0.5 of a voxel exactly there and the distances are half floats exactly [S: computed], so only
/// float's rounding of a point at 0.5 m remains, 6e-8 m [S: float32 ulp]; taken 1 µm
/// [?: a choice].
constexpr double exact_depth_tolerance_m = 1e-6;

/// The tolerance of the wall's depth after the weight's case, 0.05 voxels, 0.1 mm: half floats
/// round each of the 128 averages by up to 2^-11 of a voxel near the crossing, a sum of at most
/// 0.03 voxels however they fall [S: computed, 65 x 2^-11], against the 0.26 voxels between the
/// limited weight's 0.629 and an unlimited one's 0.5 of the 2 voxels' step.
constexpr double weight_depth_tolerance_m = 1e-4;

/// The tolerance of a wall's normal: exactly (0, 0, -1) by its central differences, every pixel at
/// the same depth [S]; 1e-4 degrees for float's rounding [?: a choice].
constexpr double wall_normal_tolerance_deg = 1e-4;

/// The pixels a wall's samples leave out at the picture's border, and their step: a sample every
/// 4 pixels, at least 8 pixels inside, where every voxel of a point's central differences sees the
/// wall [?: a choice: a voxel of 2 mm at 0.5 m spans 1.2 px].
constexpr std::uint32_t wall_sample_step_px = 4;
constexpr std::uint32_t wall_border_px = 8;

/// The ratio of a cell of the grid the points are looked up in to the voxel size, so that a point
/// within 2 voxels of a sample is found.
constexpr double grid_cell_voxels = 2.0;

/// Gives an object class's model as a case takes it.
struct model_case
{
    std::string name;                       ///< the case, named in the check lines
    gca::sparse_tsdf_parameters tsdf;       ///< the sparse TSDF's parameters
    gca::organized_points_parameters range; ///< the depth range of the organized points
};

/// Gives a model of an object class: its live or its final voxel size and its depth range, the
/// sparse TSDF's far depth the class's or nearer.
/// @param name [in] the case's name.
/// @param kind [in] the object class.
/// @param final_model [in] whether the final models' voxel size, else the live model's.
/// @param nearer_far_m [in] the sparse TSDF's far depth, in m, nearer than the class's; none for
///        the class's own.
/// @return the case.
model_case model_of(const std::string &name, gca::object_class kind, bool final_model,
                    std::optional<double> nearer_far_m = std::nullopt)
{
    const gca::object_class_parameters p = gca::parameters_of(kind);
    const double voxel_m = final_model ? p.full_voxel_size_m : p.live_voxel_size_m;
    return {name,
            {test_budget, static_cast<float>(voxel_m), method_truncation,
             static_cast<float>(nearer_far_m.value_or(p.max_depth_m))},
            {static_cast<float>(p.min_depth_m), static_cast<float>(p.max_depth_m)}};
}

/// The far depths of the final models' cases, nearer than their object classes', 0.7 m against
/// 1.0 m and 2.0 m against 3.0 m, so that one frame's bound fits the cases' pool of test_blocks:
/// about 163000 and 140000 blocks against a room of 196608 [S: computed, docs/methods/
/// sparse-tsdf.md, "One frame's blocks"], where the classes' far depths give 513102 and 604739.
constexpr double tabletop_final_far_m = 0.7;
constexpr double large_final_far_m = 2.0;

/// Uploads a frame, deprojects it and integrates it at its pose, and checks that the work was
/// queued.
/// @param tsdf [in,out] the sparse TSDF.
/// @param buffers [in,out] the GPU frame and organized points.
/// @param host [in] the frame.
/// @param range [in] the organized points' depth range.
/// @param pose [in] the frame's pose, camera frame to the models' frame.
/// @param stream [in] the stream.
/// @param what [in] the case, named in the check line.
void fuse(gca::sparse_tsdf &tsdf, frame_buffers &buffers, const gca::frame &host,
          const gca::organized_points_parameters &range, const Eigen::Isometry3d &pose,
          cudaStream_t stream, const std::string &what)
{
    const gca::step_result up = buffers.frame.upload(host, stream);
    const gca::step_result made = buffers.points.deproject(buffers.frame, range, stream);
    const gca::step_result fused =
        tsdf.integrate(buffers.frame, buffers.points, as_camera_pose(pose), stream);
    check(up.done && made.done && fused.done, (what + ": upload, deprojection and integration, " +
                                               up.message + made.message + fused.message)
                                                  .c_str());
}

/// Extracts a sparse TSDF's points and downloads them.
/// @param tsdf [in] the sparse TSDF.
/// @param buffer [in,out] the buffer of points.
/// @param stream [in] the stream.
/// @param what [in] the case, named in the check lines.
/// @return the points written; fewer, with a failed check, when the extraction or the download
///         failed.
std::vector<gca::surface_point> extract_points(const gca::sparse_tsdf &tsdf,
                                               gca::surface_points &buffer, cudaStream_t stream,
                                               const std::string &what)
{
    const gca::extraction_result extracted = tsdf.extract(buffer, stream);
    check(extracted.done && extracted.message.empty() && extracted.written == extracted.found,
          (what + ": extraction done, " + extracted.message).c_str());
    std::vector<gca::surface_point> points(extracted.written);
    const bool downloaded =
        points.empty() || (cudaMemcpyAsync(points.data(), buffer.points().data(),
                                           points.size() * sizeof(gca::surface_point),
                                           cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                           cudaStreamSynchronize(stream) == cudaSuccess);
    check(downloaded, (what + ": the points downloaded").c_str());
    if (!downloaded)
        points.clear();
    return points;
}

/// Prints how an extraction that has to be refused came out and checks that it was not done, with
/// a message that holds a text.
/// @param what [in] the case, printed and named in the check line.
/// @param made [in] what the extraction gave.
/// @param expected [in] a text the message holds.
void check_refusal_of(const std::string &what, const gca::extraction_result &made,
                      const std::string &expected)
{
    gca_test::check_refusal(what, !made.done, made.message,
                            made.message.find(expected) != std::string::npos);
}

/// Gives a point's position in double.
/// @param point [in] the point.
/// @return its position, in the models' frame, in m.
Eigen::Vector3d position_of(const gca::surface_point &point)
{
    return gca_test::to_double(point.position_m);
}

/// The extracted points in the cells of a grid, to find the nearest one to a sample.
class point_grid
{
public:
    /// Puts the points into the cells.
    /// @param points [in] the points.
    /// @param cell_m [in] a cell's edge, in m, > 0.
    point_grid(const std::vector<gca::surface_point> &points, double cell_m)
        : m_cell_m(cell_m)
    {
        for (const gca::surface_point &point : points)
            m_cells[key_of(cell_of(position_of(point)))].push_back(position_of(point));
    }

    /// Gives the distance from a sample to the nearest point in its cell and the 26 around it.
    /// @param sample [in] the sample, in the models' frame, in m.
    /// @return the distance, in m; infinity when none of the 27 cells holds a point, so at least
    ///         one cell's edge away.
    double nearest_m(const Eigen::Vector3d &sample) const
    {
        const gca::block_coordinate own = cell_of(sample);
        double nearest = std::numeric_limits<double>::infinity();
        for (int dz = -1; dz <= 1; dz++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    const auto found = m_cells.find(key_of({own.x + dx, own.y + dy, own.z + dz}));
                    if (found == m_cells.end())
                        continue;
                    for (const Eigen::Vector3d &p : found->second)
                        nearest = std::min(nearest, (p - sample).norm());
                }
        return nearest;
    }

private:
    /// Gives the cell a point lies in.
    /// @param p [in] the point, in m.
    /// @return floor(p / cell) on each axis.
    gca::block_coordinate cell_of(const Eigen::Vector3d &p) const
    {
        return {static_cast<std::int32_t>(std::floor(p.x() / m_cell_m)),
                static_cast<std::int32_t>(std::floor(p.y() / m_cell_m)),
                static_cast<std::int32_t>(std::floor(p.z() / m_cell_m))};
    }

    /// Gives a cell's key, the hash's key of the same coordinates (block_key()).
    /// @param c [in] the cell, each coordinate within the hash's range: the cases' scenes lie
    ///        within a few metres, a few thousand cells.
    /// @return the key.
    static std::uint64_t key_of(const gca::block_coordinate &c)
    {
        return gca::block_key(c);
    }

    double m_cell_m;
    std::unordered_map<std::uint64_t, std::vector<Eigen::Vector3d>> m_cells;
};

/// Builds a frame of a wall facing the camera.
/// @param depth_of [in] the depth value of pixel (u, v), in mm.
/// @param color [in] the wall's color.
/// @return the frame, 640x400 with the camera's intrinsics.
template <typename Depth>
gca::frame wall_frame(Depth depth_of, const std::array<std::uint8_t, 3> &color)
{
    return gca_test::built_frame(small_description, depth_of,
                                 [&color](std::uint32_t, std::uint32_t) { return color; });
}

/// Gives a frame of a wall facing the camera at one depth.
/// @param depth_mm [in] the depth of every pixel, in mm.
/// @param color [in] the wall's color.
/// @return the frame, 640x400 with the camera's intrinsics.
gca::frame flat_wall(std::uint16_t depth_mm, const std::array<std::uint8_t, 3> &color)
{
    return wall_frame([depth_mm](std::uint32_t, std::uint32_t) { return depth_mm; }, color);
}

/// Checks a wall's points: each at the expected depth within a tolerance, its normal towards the
/// camera, along -z, and its color the expected one; and the wall's samples, the points of a
/// pixel every wall_sample_step_px at least wall_border_px inside the picture, each within one
/// voxel of a point.
/// @param what [in] the case, printed and named in the check lines.
/// @param points [in] the points, in the camera frame of the walls' frames (the models' frame).
/// @param depth_m [in] the expected depth, in m.
/// @param depth_tolerance_m [in] the tolerance of the depth, in m.
/// @param color [in] the expected color.
/// @param voxel_m [in] the voxel size, in m.
void check_wall(const std::string &what, const std::vector<gca::surface_point> &points,
                double depth_m, double depth_tolerance_m, const std::array<int, 3> &color,
                double voxel_m)
{
    max_error depth_errors;
    max_error normal_errors;
    int wrong_colors = 0;
    for (const gca::surface_point &p : points)
    {
        depth_errors.add(std::abs(double{p.position_m.z} - depth_m));
        normal_errors.add(
            angle_deg(gca_test::to_double(p.normal), Eigen::Vector3d(0.0, 0.0, -1.0)));
        wrong_colors += p.color.x != color[0] || p.color.y != color[1] || p.color.z != color[2];
    }
    std::printf("%s: %zu points, the expected color (%d, %d, %d)\n", what.c_str(), points.size(),
                color[0], color[1], color[2]);
    check(!points.empty(), (what + ": points extracted").c_str());
    gca_test::check_max_error(what + ": depth's error, in m", depth_errors, depth_tolerance_m);
    gca_test::check_max_error(what + ": normal's angle from -z, in degrees", normal_errors,
                              wall_normal_tolerance_deg);
    check_within((what + ": points of another color").c_str(), wrong_colors, 0.0);
    const point_grid grid(points, grid_cell_voxels * voxel_m);
    const gca::intrinsics &k = small_description.intrinsics;
    max_error gaps;
    for (std::uint32_t v = wall_border_px; v + wall_border_px < small_description.size.height;
         v += wall_sample_step_px)
        for (std::uint32_t u = wall_border_px; u + wall_border_px < small_description.size.width;
             u += wall_sample_step_px)
            gaps.add(grid.nearest_m(depth_m * gca_test::pixel_ray(k, u, v)));
    gca_test::check_max_error(what + ": a wall's sample's distance from the nearest point, in m",
                              gaps, voxel_m);
}

/// Gives the mean of two colors as the method rounds it, to the nearest integer, a tie to the
/// even one.
/// @param a [in] one color.
/// @param b [in] the other.
/// @return the channels' means.
std::array<int, 3> mean_of(const std::array<std::uint8_t, 3> &a,
                           const std::array<std::uint8_t, 3> &b)
{
    std::array<int, 3> mean{};
    for (std::size_t c = 0; c < 3; c++)
        mean[c] = static_cast<int>(
            std::nearbyint((static_cast<double>(a[c]) + static_cast<double>(b[c])) / 2.0));
    return mean;
}

/// Gives a voxel's color after frames of one color and then frames of another, by the method's
/// running average: C <- round((W C + c) / (W + 1)), W <- min(W + 1, 64), from W = 0, each
/// rounding to the nearest integer, a tie to the even one.
/// @param first [in] the first color, fused first_count times.
/// @param first_count [in] the frames of the first color.
/// @param second [in] the second color, fused second_count times after them.
/// @param second_count [in] the frames of the second color.
/// @return the voxel's color.
std::array<int, 3> color_after(const std::array<std::uint8_t, 3> &first, int first_count,
                               const std::array<std::uint8_t, 3> &second, int second_count)
{
    std::array<int, 3> held{};
    int weight = 0;
    for (int frame = 0; frame < first_count + second_count; frame++)
    {
        const std::array<std::uint8_t, 3> &seen = frame < first_count ? first : second;
        for (std::size_t c = 0; c < 3; c++)
            held[c] = static_cast<int>(
                std::nearbyint((double(weight) * held[c] + seen[c]) / double(weight + 1)));
        weight = std::min(weight + 1, int{gca::max_voxel_weight});
    }
    return held;
}

/// The walls: one frame, then a second wall behind it; and in a new sparse TSDF, frames_per_wall
/// frames of the first wall and then as many of the second, against the weight's limit. Each at
/// the tabletop object's live voxel size, 2 mm, its far depth 1.0 m.
/// @param stream [in] the stream.
void check_walls(cudaStream_t stream)
{
    const model_case m = model_of("walls", gca::object_class::tabletop_object, false);
    const double voxel_m = m.tsdf.voxel_size_m;
    const gca::frame front = flat_wall(wall_mm, wall_color);
    const gca::frame behind = flat_wall(behind_wall_mm, behind_wall_color);
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    const double step_m = 1e-3 * (behind_wall_mm - wall_mm);
    {
        gca::gpu_pool pool;
        frame_buffers buffers = request_buffers(pool, small_description.size, m.name);
        gca::sparse_tsdf tsdf =
            granted(m.name + ": the sparse TSDF", gca::sparse_tsdf::request(pool, m.tsdf, stream));
        gca::surface_points buffer =
            granted(m.name + ": the points", gca::surface_points::request(pool, point_capacity));
        pool.close();
        fuse(tsdf, buffers, front, m.range, identity, stream, "a wall");
        check_wall("a wall", extract_points(tsdf, buffer, stream, "a wall"), 1e-3 * wall_mm,
                   exact_depth_tolerance_m,
                   {int{wall_color[0]}, int{wall_color[1]}, int{wall_color[2]}}, voxel_m);
        fuse(tsdf, buffers, behind, m.range, identity, stream, "a wall behind it");
        check_wall("a wall behind it, once each",
                   extract_points(tsdf, buffer, stream, "a wall behind it"),
                   1e-3 * wall_mm + 0.5 * step_m, exact_depth_tolerance_m,
                   mean_of(wall_color, behind_wall_color), voxel_m);
        check(pool.refused_while_closed() == 0, "walls: the closed pool asked for nothing");
    }
    gca::gpu_pool pool;
    frame_buffers buffers = request_buffers(pool, small_description.size, m.name);
    gca::sparse_tsdf tsdf =
        granted(m.name + ": the sparse TSDF", gca::sparse_tsdf::request(pool, m.tsdf, stream));
    gca::surface_points buffer =
        granted(m.name + ": the points", gca::surface_points::request(pool, point_capacity));
    pool.close();
    for (int i = 0; i < frames_per_wall; i++)
        fuse(tsdf, buffers, front, m.range, identity, stream, "the weight's limit, the first wall");
    for (int i = 0; i < frames_per_wall; i++)
        fuse(tsdf, buffers, behind, m.range, identity, stream,
             "the weight's limit, the second wall");
    // With the weight limited to 64, each of the second wall's 64 frames moves the distance by
    // 1/65 of what is left: the surface ends at 1 - (64/65)^64 = 0.629 of the step; without the
    // limit, at 64/128 = 0.5 [S: computed].
    const double share = 1.0 - std::pow(64.0 / 65.0, frames_per_wall);
    check_wall("the weight's limit, 64 frames of each wall",
               extract_points(tsdf, buffer, stream, "the weight's limit"),
               1e-3 * wall_mm + share * step_m, weight_depth_tolerance_m,
               color_after(wall_color, frames_per_wall, behind_wall_color, frames_per_wall),
               voxel_m);
    check(pool.refused_while_closed() == 0, "walls: the closed pool asked for nothing");
}

/// The far depth case's: 0.7 m, between the near wall, wall_mm, and the wall beyond it, at 0.9 m
/// inside the tabletop object's depth range of 1.0 m [?: a choice].
constexpr double far_case_far_m = 0.7;
constexpr std::uint16_t beyond_far_mm = 900;

/// The column where the far depth case's frame turns from the near wall to the wall beyond: the
/// picture's middle, 320.
constexpr std::uint32_t far_case_column = small_description.size.width / 2;

/// The far depth: a wall at 0.5 m fused, then a frame whose left half shows the same wall and whose
/// right half a wall at 0.9 m, beyond a far depth of 0.7 m but inside the organized points' range.
/// The right half's pixels observe nothing: the voxels of the near wall's blocks that project into
/// it keep the first frame's distances, so the surface stays at 0.5 m over the whole picture; and
/// the allocation counts the right half's pixels as not walked, 320 x 400 = 128000.
/// @param stream [in] the stream.
void check_far_depth(cudaStream_t stream)
{
    model_case m = model_of("far depth", gca::object_class::tabletop_object, false, far_case_far_m);
    const gca::frame front = flat_wall(wall_mm, wall_color);
    const gca::frame halves =
        wall_frame([](std::uint32_t u,
                      std::uint32_t) { return u < far_case_column ? wall_mm : beyond_far_mm; },
                   wall_color);
    const std::uint32_t right_half_px =
        (small_description.size.width - far_case_column) * small_description.size.height;
    gca::gpu_pool pool;
    frame_buffers buffers = request_buffers(pool, small_description.size, m.name);
    gca::sparse_tsdf tsdf =
        granted(m.name + ": the sparse TSDF", gca::sparse_tsdf::request(pool, m.tsdf, stream));
    gca::surface_points buffer =
        granted(m.name + ": the points", gca::surface_points::request(pool, point_capacity));
    pool.close();
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    fuse(tsdf, buffers, front, m.range, identity, stream, m.name + ", the wall");
    fuse(tsdf, buffers, halves, m.range, identity, stream, m.name + ", the two halves");
    const gca::sparse_tsdf_usage usage = read_usage(tsdf, stream, m.name);
    print_usage(m.name, usage);
    check(usage.rays_not_walked == right_half_px, "far depth: the right half's pixels not walked");
    check_wall(m.name, extract_points(tsdf, buffer, stream, m.name), 1e-3 * wall_mm,
               exact_depth_tolerance_m,
               {int{wall_color[0]}, int{wall_color[1]}, int{wall_color[2]}}, m.tsdf.voxel_size_m);
    check(pool.refused_while_closed() == 0, "far depth: the closed pool asked for nothing");
}

/// The tolerances of a scene's points, each in the units its name gives.
struct scene_tolerances
{
    double distance_median_mm; ///< the median distance from the analytic surface
    double distance_max_mm;    ///< the largest distance from the analytic surface
    double normal_median_deg;  ///< the median angle between a normal and the surface's
    double normal_p99_deg;     ///< the 99th percentile of that angle
    double color_median;       ///< the median of a point's largest channel's error, in levels
    double color_p99;          ///< the 99th percentile of that error, in levels
};

/// The tolerances of the four scenes' cases: 1.27 to 1.47 times what this test measured with
/// the method's starting values and the integration and extraction as built [M 2026-11-15: this
/// test; the measured median, 99th percentile and largest value of each stand above each]. The
/// largest normal's angle and color error are printed, not checked: the largest angles, up to 37,
/// 95, 85 and 176 degrees, lie where the band of a surface reaches past another's silhouette
/// (docs/methods/sparse-tsdf.md, "The extraction"), and the textures' waves of 45 mm to 100 mm
/// change by up to 14 levels a millimetre, so a voxel's color at its centre's projection differs
/// from the texture at the point by tens of levels at 8 mm [S: computed, 2 pi 100 / 45 mm].
/// The tabletop object's live model, 2 mm: distances 0.102 / 0.726 / 1.50 mm; angles 3.45 /
/// 15.1 / 37.0 degrees; color errors 2.11 / 20.6 / 75.7 levels.
constexpr scene_tolerances tabletop_live_tolerances = {0.15, 2.0, 5.0, 20.0, 3.0, 30.0};
/// The tabletop object's final model, 1 mm: distances 0.106 / 0.601 / 1.38 mm; angles 6.80 /
/// 30.9 / 95.2 degrees, the depth's millimetre steps showing at a voxel of 1 mm; color errors
/// 1.71 / 21.7 / 115 levels.
constexpr scene_tolerances tabletop_final_tolerances = {0.15, 2.0, 9.0, 40.0, 2.5, 30.0};
/// The large object's live model, 8 mm: distances 0.511 / 4.41 / 13.0 mm; angles 4.16 / 24.3 /
/// 84.6 degrees; color errors 16.1 / 91.8 / 183 levels.
constexpr scene_tolerances large_live_tolerances = {0.75, 17.0, 6.0, 32.0, 22.0, 120.0};
/// The large object's final model, 4 mm: distances 0.414 / 4.97 / 14.9 mm; angles 6.57 / 44.7 /
/// 176 degrees; color errors 7.37 / 98.7 / 199 levels.
constexpr scene_tolerances large_final_tolerances = {0.6, 19.0, 9.0, 58.0, 10.0, 130.0};

/// The most of a scene's samples that may lie farther than coverage_reach_voxels from every point:
/// 0.05 % [M 2026-11-15: this test, 18, 2, 5 and 1 of about 90000 in the four cases, 0.018 % at
/// most, at the plane's border, at the far depth and where the sphere meets the plane].
constexpr double uncovered_share = 0.0005;

/// The step between a scene's samples of the surface its frames saw, along each axis of the
/// picture: every fourth pixel, about 90000 samples a scene [?: a choice].
constexpr std::uint32_t scene_sample_step_px = 4;

/// The tolerance of a normal's length: 1e-6, the model file's [S: scan/model_file.hpp], so that a
/// model's writer takes the extracted normals.
constexpr double unit_normal_tolerance = 1e-6;

/// The distance from a sample within which a point covers it, in voxels: 2 [?: a choice, a voxel's
/// diagonal and more], and the tolerance of the samples' 99th percentile, 1 voxel [M 2026-11-15:
/// this test, 0.68 to 0.83 voxels in the four cases].
constexpr double coverage_reach_voxels = 2.0;
constexpr double coverage_p99_voxels = 1.0;

/// A scene of a sphere resting on a plane, the frames of a walk around it, a model of an object
/// class it is fused into and the tolerances of its points.
struct scene_case
{
    model_case model;                ///< the sparse TSDF's parameters and the points' range
    gca_test::synthetic_scene scene; ///< the plane, the sphere and the walk's poses
    scene_tolerances tolerances;     ///< the tolerances of the points
};

/// Gives the walk of a scene's case.
/// @param walk [in] the walk.
/// @return its poses; none, with a failed check, when walk_around() refuses it.
std::vector<Eigen::Isometry3d> poses_of(const gca_test::synthetic_walk &walk)
{
    const std::optional<std::vector<Eigen::Isometry3d>> poses = gca_test::walk_around(walk);
    check(poses.has_value(), "the walk's poses");
    return poses.value_or(std::vector<Eigen::Isometry3d>{});
}

/// The tabletop scene: the tabletop sphere on the tabletop plane (stage_frames.hpp,
/// synthetic_source.hpp), seen along the walk around the tabletop box, 12 poses from 0.45 m.
/// @param model [in] the model it is fused into.
/// @param tolerances [in] the tolerances of its points.
/// @return the case.
scene_case tabletop_scene(const model_case &model, const scene_tolerances &tolerances)
{
    return {model,
            {.plane = gca_test::tabletop_plane,
             .sphere = gca_test::tabletop_sphere,
             .trajectory = poses_of(gca_test::tabletop_walk())},
            tolerances};
}

/// The large scene's sphere: 0.4 m in radius, a large object of 0.5 m to 2 m (GLOSSARY.md),
/// resting on the plane at the world's origin [?: a choice].
constexpr double large_sphere_radius_m = 0.4;

/// The large scene's plane under the sphere: 3 m x 3 m [?: a choice, so that the final model's
/// blocks stay within the cases' pool: it takes 54285 of the 65536 (M 2026-11-15: this test)].
constexpr gca_test::synthetic_plane large_plane = {1.5, 1.5};

/// Gives the walk around the large scene's sphere: 12 poses, 1.5 m to 1.7 m from its centre along
/// the plane, 1.1 m to 1.3 m above it, rolled by up to 10 degrees, looking at its centre moved by
/// up to 5 cm along x and along y [?: a choice, a hand-held walk around a large object].
/// @return the walk, for walk_around().
gca_test::synthetic_walk large_walk()
{
    return {.centre_m = Eigen::Vector3d(0.0, 0.0, large_sphere_radius_m),
            .pose_count = 12,
            .radius_m = 1.6,
            .radius_swing_m = 0.1,
            .height_m = 1.2,
            .height_swing_m = 0.1,
            .roll_swing_deg = 10.0,
            .target_jitter_m = 0.05};
}

/// The large scene: the large sphere on the large plane along the large walk.
/// @param model [in] the model it is fused into.
/// @param tolerances [in] the tolerances of its points.
/// @return the case.
scene_case large_scene(const model_case &model, const scene_tolerances &tolerances)
{
    const gca_test::synthetic_walk walk = large_walk();
    return {model,
            {.plane = large_plane,
             .sphere = gca_test::synthetic_sphere{walk.centre_m, large_sphere_radius_m},
             .trajectory = poses_of(walk)},
            tolerances};
}

/// What the test knows of a point of a scene: its nearest analytic surface, the plane or the
/// sphere, and its distance, normal and color there.
struct expected_point
{
    double distance_m;           ///< the point's distance from the nearest surface, in m
    Eigen::Vector3d normal;      ///< that surface's unit normal at its nearest point, outward
    std::array<double, 3> color; ///< the texture there, 0 to 255
    bool on_sphere;              ///< whether the nearest surface is the sphere
};

/// Finds what a point should be: the nearer of the plane's rectangle and the sphere, the normal
/// up from the plane (the cameras look down on it) or out of the sphere, and the texture at the
/// nearest point.
/// @param scene [in] the scene, with a plane and a sphere.
/// @param surfaces [in] its known surfaces, the plane first.
/// @param p [in] the point, in the world frame, in m.
/// @return what it should be.
expected_point expect_at(const gca_test::synthetic_scene &scene,
                         const std::vector<gca_test::synthetic_surface> &surfaces,
                         const Eigen::Vector3d &p)
{
    const gca_test::synthetic_sphere &sphere = *scene.sphere;
    const Eigen::Vector3d out = p - sphere.centre_m;
    const double from_sphere_m = std::abs(out.norm() - sphere.radius_m);
    const double from_plane_m = gca_test::distance_from(surfaces.front(), p);
    if (from_sphere_m < from_plane_m)
    {
        const Eigen::Vector3d n = out.normalized();
        return {from_sphere_m, n,
                gca_test::texture_color(gca_test::synthetic_texture::box,
                                        sphere.centre_m + sphere.radius_m * n),
                true};
    }
    return {from_plane_m, Eigen::Vector3d::UnitZ(),
            gca_test::texture_color(gca_test::synthetic_texture::plane,
                                    Eigen::Vector3d(p.x(), p.y(), 0.0)),
            false};
}

/// Prints the median, the 99th percentile and the largest of values, and checks that each is
/// finite and that the median and either the 99th percentile or the largest lie within their
/// tolerances.
/// @param what [in] what they are, printed and named in the check lines.
/// @param values [in] the values, not empty.
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

/// Fuses a scene's frames at their poses into a sparse TSDF of the case's model, extracts its
/// points and checks them against the analytic scene.
/// @param c [in] the case.
/// @param stream [in] the stream.
void check_scene(const scene_case &c, cudaStream_t stream)
{
    const std::string &name = c.model.name;
    const gca_test::rendered frames = gca_test::render(c.scene, small_description);
    if (frames.frames.size() != c.scene.trajectory.size())
        return;
    gca::gpu_pool pool;
    frame_buffers buffers = request_buffers(pool, small_description.size, name);
    gca::sparse_tsdf tsdf =
        granted(name + ": the sparse TSDF", gca::sparse_tsdf::request(pool, c.model.tsdf, stream));
    gca::surface_points buffer =
        granted(name + ": the points", gca::surface_points::request(pool, point_capacity));
    pool.close();
    for (std::size_t i = 0; i < frames.frames.size(); i++)
        fuse(tsdf, buffers, frames.frames[i], c.model.range, c.scene.trajectory[i], stream,
             name + ", frame " + std::to_string(i));
    const gca::sparse_tsdf_usage usage = read_usage(tsdf, stream, name);
    print_usage(name, usage);
    check(usage.blocks_refused == 0 && usage.coordinates_refused == 0,
          (name + ": the pool held every block").c_str());
    const std::vector<gca::surface_point> points = extract_points(tsdf, buffer, stream, name);
    check(pool.refused_while_closed() == 0, (name + ": the closed pool asked for nothing").c_str());

    std::vector<double> distances_mm;
    std::vector<double> angles_deg;
    std::vector<double> color_levels;
    std::size_t on_sphere = 0;
    max_error unit_errors;
    for (const gca::surface_point &point : points)
    {
        unit_errors.add(std::abs(gca_test::to_double(point.normal).norm() - 1.0));
        const expected_point e = expect_at(c.scene, frames.surfaces, position_of(point));
        on_sphere += e.on_sphere;
        distances_mm.push_back(1e3 * e.distance_m);
        angles_deg.push_back(angle_deg(gca_test::to_double(point.normal), e.normal));
        const std::array<double, 3> seen = {static_cast<double>(point.color.x),
                                            static_cast<double>(point.color.y),
                                            static_cast<double>(point.color.z)};
        double worst = 0.0;
        for (std::size_t k = 0; k < 3; k++)
            worst = std::max(worst, std::abs(seen[k] - e.color[k]));
        color_levels.push_back(worst);
    }
    std::printf("%s: %zu points, %zu of them nearer the sphere\n", name.c_str(), points.size(),
                on_sphere);
    check(on_sphere > 0 && on_sphere < points.size(),
          (name + ": points on the sphere and on the plane").c_str());
    gca_test::check_max_error(name + ": a normal's length less 1", unit_errors,
                              unit_normal_tolerance);
    const scene_tolerances &t = c.tolerances;
    check_spread(name + ": distance from the surface, mm", distances_mm, t.distance_median_mm, 1.0,
                 t.distance_max_mm);
    check_spread(name + ": normal's angle, deg", angles_deg, t.normal_median_deg, 0.99,
                 t.normal_p99_deg);
    check_spread(name + ": color's error, levels", color_levels, t.color_median, 0.99, t.color_p99);

    // The samples: the analytic hit of every scene_sample_step_px-th pixel of each frame along both
    // axes, nearer than the far depth.
    const point_grid grid(points, grid_cell_voxels * c.model.tsdf.voxel_size_m);
    std::vector<double> gaps_mm;
    const gca::intrinsics &k = small_description.intrinsics;
    for (std::size_t i = 0; i < frames.frames.size(); i++)
    {
        const Eigen::Isometry3d &pose = c.scene.trajectory[i];
        for (std::uint32_t v = 0; v < small_description.size.height; v += scene_sample_step_px)
            for (std::uint32_t u = 0; u < small_description.size.width; u += scene_sample_step_px)
            {
                const Eigen::Vector3d ray = pose.linear() * gca_test::pixel_ray(k, u, v);
                const std::optional<gca_test::synthetic_hit> hit =
                    gca_test::first_hit(frames.surfaces, c.scene.sphere, pose.translation(), ray);
                if (!hit || hit->s < c.model.range.near_m || hit->s > c.model.tsdf.max_depth_m)
                    continue;
                gaps_mm.push_back(1e3 * grid.nearest_m(pose.translation() + hit->s * ray));
            }
    }
    const double voxel_mm = 1e3 * c.model.tsdf.voxel_size_m;
    const std::size_t uncovered = static_cast<std::size_t>(
        std::count_if(gaps_mm.begin(), gaps_mm.end(), [voxel_mm](double gap) {
            return !(gap <= coverage_reach_voxels * voxel_mm);
        }));
    std::printf("%s: %zu of %zu samples seen farther than %.3g mm from every point, the 99th "
                "percentile %.4g mm\n",
                name.c_str(), uncovered, gaps_mm.size(), coverage_reach_voxels * voxel_mm,
                quantile(gaps_mm, 0.99));
    check(!gaps_mm.empty(), (name + ": samples seen").c_str());
    check_within((name + ": the share of the samples seen not covered").c_str(),
                 static_cast<double>(uncovered) /
                     static_cast<double>(std::max<std::size_t>(gaps_mm.size(), 1)),
                 uncovered_share);
    check_within((name + ": the samples' 99th percentile, in voxels").c_str(),
                 quantile(gaps_mm, 0.99) / voxel_mm, coverage_p99_voxels);
}

/// The depth noise of the noisy frames: c = 0.01 /m, the method's starting value of the
/// truncation's c, so that sigma is 2.0 mm to 3.4 mm at the walk's depths of 0.45 m to 0.58 m to
/// the plane's middle [S: computed], and the band of 3 sigma is the truncation [?: a choice].
constexpr gca_test::synthetic_depth_noise test_noise = {0.01, 1};

/// The least factor by which the fused surface spreads less about the plane than one frame's
/// points: 1.8 [M 2026-11-15: this test, 2.16: one frame's points 2.19 mm, the fused surface's
/// 1.01 mm, root mean square].
constexpr double noise_reduction = 1.8;

/// Gives the root mean square of values.
/// @param values [in] the values, not empty.
/// @return sqrt(sum of squares / count); 0 when there are none.
double rms(const std::vector<double> &values)
{
    double sum = 0.0;
    for (const double x : values)
        sum += x * x;
    return values.empty() ? 0.0 : std::sqrt(sum / static_cast<double>(values.size()));
}

/// Noisy frames: the tabletop plane alone along the walk around the tabletop box, its depth with
/// noise of sigma = c z^2 from a fixed seed, fused at the tabletop object's live voxel size. The
/// spread of the fused surface about the plane, the root mean square of its points' heights above
/// it, is smaller than that of the first frame's points, deprojected here by the pinhole formula,
/// by at least noise_reduction.
/// @param stream [in] the stream.
void check_noise(cudaStream_t stream)
{
    const model_case m = model_of("noise", gca::object_class::tabletop_object, false);
    const gca_test::synthetic_scene scene = {.plane = gca_test::tabletop_plane,
                                             .trajectory = poses_of(gca_test::tabletop_walk())};
    const gca_test::rendered frames = gca_test::render(scene, small_description, test_noise);
    if (frames.frames.size() != scene.trajectory.size() || frames.frames.empty())
        return;
    gca::gpu_pool pool;
    frame_buffers buffers = request_buffers(pool, small_description.size, m.name);
    gca::sparse_tsdf tsdf =
        granted(m.name + ": the sparse TSDF", gca::sparse_tsdf::request(pool, m.tsdf, stream));
    gca::surface_points buffer =
        granted(m.name + ": the points", gca::surface_points::request(pool, point_capacity));
    pool.close();
    for (std::size_t i = 0; i < frames.frames.size(); i++)
        fuse(tsdf, buffers, frames.frames[i], m.range, scene.trajectory[i], stream,
             m.name + ", frame " + std::to_string(i));
    const std::vector<gca::surface_point> points = extract_points(tsdf, buffer, stream, m.name);
    check(pool.refused_while_closed() == 0, "noise: the closed pool asked for nothing");
    std::vector<double> fused_m;
    for (const gca::surface_point &point : points)
        fused_m.push_back(double{point.position_m.z});
    // The first frame's points in the world frame, inside the organized points' range.
    const gca::frame &first = frames.frames.front();
    const gca::intrinsics &k = small_description.intrinsics;
    std::vector<double> one_frame_m;
    for (std::uint32_t v = 0; v < small_description.size.height; v++)
        for (std::uint32_t u = 0; u < small_description.size.width; u++)
        {
            const double z = small_description.depth_unit_m *
                             static_cast<double>(
                                 first.depth()[gca_test::index_of(small_description.size, u, v)]);
            if (z < m.range.near_m || z > m.range.far_m)
                continue;
            one_frame_m.push_back(
                (scene.trajectory.front() * (z * gca_test::pixel_ray(k, u, v))).z());
        }
    const double fused_rms = rms(fused_m);
    const double one_rms = rms(one_frame_m);
    std::printf("noise: one frame's %zu points spread %.4f mm about the plane, the fused %zu "
                "points %.4f mm, %.2f times less\n",
                one_frame_m.size(), 1e3 * one_rms, fused_m.size(), 1e3 * fused_rms,
                one_rms / fused_rms);
    check(!fused_m.empty() && !one_frame_m.empty(), "noise: points fused and points of one frame");
    check(fused_rms * noise_reduction <= one_rms,
          "noise: the fused surface spreads less than one frame's points, by the stated factor");
}

/// The blocks and slots of a sparse TSDF whose pool one frame uses up: 16385 blocks, the least for
/// which the hash's 131072 slots leave room for one frame of the tabletop object's live model,
/// 114687 blocks above its bound of 87200 [S: computed, docs/methods/sparse-tsdf.md, "One frame's
/// blocks"]; and its budget, 4100 bytes a block, 16 a slot and 16 more, 69 MB.
constexpr std::uint32_t used_up_blocks = 16385;
constexpr std::uint64_t used_up_slots = 131072;
constexpr std::uint64_t used_up_budget =
    4100 * std::uint64_t{used_up_blocks} + 16 * used_up_slots + 16;

/// The depth of the wall that uses the pool up: 1.0 m, the far depth of the tabletop object, where
/// its band of 30 mm crosses about 55000 blocks [M 2026-11-15: this test, 16385 + 38531].
constexpr std::uint16_t used_up_wall_mm = 1000;

/// A pool one frame uses up: a wall facing the camera at 1.0 m, the tabletop object's live model,
/// whose band of 30 mm touches more blocks than the pool's 16385. The usage says the pool is used
/// up and the list holds as many slots as the pool has blocks, the rest unlisted, and the counts
/// add up: every slot the frame touched is a coordinate of its own, one with a block or one refused
/// for the used-up pool, none refused otherwise and every ray walked. The extraction's points lie
/// on the wall, with normals towards the camera: the slots on the list without a block are
/// skipped and no voxel outside the pool is written. A list written past its end is not seen
/// here: its 38531 entries past the end, 154 KB, land in the unused rest of the list's allocation,
/// 458 KB up to the voxels, and the counts lie elsewhere [M 2026-11-15: the addresses printed by
/// a diagnostic build of this test]; compute-sanitizer would see it (ROADMAP.md, step 2).
/// @param stream [in] the stream.
void check_used_up(cudaStream_t stream)
{
    model_case m = model_of("used-up pool", gca::object_class::tabletop_object, false);
    m.tsdf.memory_budget_bytes = used_up_budget;
    const gca::frame wall = flat_wall(used_up_wall_mm, wall_color);
    gca::gpu_pool pool;
    frame_buffers buffers = request_buffers(pool, small_description.size, m.name);
    gca::sparse_tsdf tsdf =
        granted(m.name + ": the sparse TSDF", gca::sparse_tsdf::request(pool, m.tsdf, stream));
    gca::surface_points buffer =
        granted(m.name + ": the points", gca::surface_points::request(pool, point_capacity));
    pool.close();
    check(tsdf.layout().capacity_blocks == used_up_blocks &&
              tsdf.layout().slot_count == used_up_slots,
          "used-up pool: the stated blocks and slots");
    fuse(tsdf, buffers, wall, m.range, Eigen::Isometry3d::Identity(), stream, m.name);
    const gca::sparse_tsdf_usage usage = read_usage(tsdf, stream, m.name);
    print_usage(m.name, usage);
    check(usage.used_up() && usage.blocks_refused > 0 && usage.listed_slots == used_up_blocks &&
              usage.unlisted_slots > 0,
          "used-up pool: used up, the list full and the rest unlisted");
    check(usage.listed_slots + usage.unlisted_slots == usage.blocks_in_use + usage.blocks_refused &&
              usage.coordinates_refused == 0 && usage.rays_not_walked == 0,
          "used-up pool: a slot touched for each coordinate, none refused otherwise, every ray "
          "walked");
    const std::vector<gca::surface_point> points = extract_points(tsdf, buffer, stream, m.name);
    max_error depth_errors;
    max_error normal_errors;
    for (const gca::surface_point &p : points)
    {
        depth_errors.add(std::abs(double{p.position_m.z} - 1.0));
        normal_errors.add(
            angle_deg(gca_test::to_double(p.normal), Eigen::Vector3d(0.0, 0.0, -1.0)));
    }
    std::printf("used-up pool: %zu points\n", points.size());
    check(!points.empty(), "used-up pool: points extracted");
    gca_test::check_max_error("used-up pool: depth's error, in m", depth_errors,
                              exact_depth_tolerance_m);
    gca_test::check_max_error("used-up pool: normal's angle from -z, in degrees", normal_errors,
                              wall_normal_tolerance_deg);
    check(pool.refused_while_closed() == 0, "used-up pool: the closed pool asked for nothing");
}

/// Orders points by position, then normal, then color, so that two extractions of the same
/// voxels compare as sets.
/// @param points [in,out] the points, sorted.
void sort_points(std::vector<gca::surface_point> &points)
{
    const auto as_tuple = [](const gca::surface_point &p) {
        return std::make_tuple(p.position_m.x, p.position_m.y, p.position_m.z, p.normal.x,
                               p.normal.y, p.normal.z, p.color.x, p.color.y, p.color.z);
    };
    std::sort(points.begin(), points.end(),
              [&as_tuple](const gca::surface_point &a, const gca::surface_point &b) {
                  return as_tuple(a) < as_tuple(b);
              });
}

/// Says whether two sorted sets of points are the same, bit for bit in every field.
/// @param a [in] one set.
/// @param b [in] the other.
/// @return whether they hold the same points.
bool same_points(const std::vector<gca::surface_point> &a, const std::vector<gca::surface_point> &b)
{
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(),
                      [](const gca::surface_point &x, const gca::surface_point &y) {
                          return std::memcmp(&x.position_m, &y.position_m, sizeof(float3)) == 0 &&
                                 std::memcmp(&x.normal, &y.normal, sizeof(float3)) == 0 &&
                                 x.color.x == y.color.x && x.color.y == y.color.y &&
                                 x.color.z == y.color.z;
                      });
}

/// Refusals as values, each leaving the counts and the points as they were and CUDA's last error
/// clear: the integration into an empty sparse TSDF and one moved from, of a GPU frame that holds
/// no frame, of organized points of another size, of a frame whose intrinsics cannot be used, at a
/// pose that cannot be used, and after an error an earlier call left; the extraction from an empty
/// sparse TSDF, into an empty buffer, and after an earlier error. A buffer too small: not done,
/// with the count found, its first points written, each on the wall.
/// @param stream [in] the stream.
void check_refusals(cudaStream_t stream)
{
    const std::string what = "refusals";
    const model_case m = model_of(what, gca::object_class::tabletop_object, false);
    const gca::frame wall = flat_wall(wall_mm, wall_color);
    gca::frame_description unusable = small_description;
    unusable.intrinsics.fx = 0.0f;
    const gca::frame wall_unusable = gca_test::built_frame(
        unusable, [](std::uint32_t, std::uint32_t) { return wall_mm; }, gca_test::grey);
    gca::gpu_pool pool;
    frame_buffers buffers = request_buffers(pool, small_description.size, what);
    const gca::gpu_frame never_uploaded =
        granted(what + ": a GPU frame", gca::gpu_frame::request(pool, small_description.size));
    gca::gpu_frame odd_frame =
        granted(what + ": a GPU frame", gca::gpu_frame::request(pool, small_description.size));
    const gca::organized_points odd_points =
        granted(what + ": odd points",
                gca::organized_points::request(pool, gca_test::odd_description.size));
    gca::sparse_tsdf tsdf =
        granted(what + ": the sparse TSDF", gca::sparse_tsdf::request(pool, m.tsdf, stream));
    gca::surface_points buffer =
        granted(what + ": the points", gca::surface_points::request(pool, point_capacity));
    gca::surface_points small_buffer =
        granted(what + ": ten points", gca::surface_points::request(pool, 10));
    pool.close();
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    fuse(tsdf, buffers, wall, m.range, identity, stream, what);
    check(odd_frame.upload(wall_unusable, stream).done,
          "refusals: a frame without usable intrinsics uploaded");
    const gca::sparse_tsdf_usage before = read_usage(tsdf, stream, what);
    const std::vector<gca::surface_point> points_before =
        extract_points(tsdf, buffer, stream, what + ", before");
    const gca::camera_pose usable = as_camera_pose(identity);

    gca::sparse_tsdf empty;
    check_refused("refusals, an empty sparse TSDF",
                  empty.integrate(buffers.frame, buffers.points, usable, stream),
                  "holds no memory");
    gca::sparse_tsdf moved_to(std::move(tsdf));
    check_refused("refusals, a sparse TSDF moved from",
                  tsdf.integrate(buffers.frame, buffers.points, usable, stream), "holds no memory");
    tsdf = std::move(moved_to);
    check_refused("refusals, a GPU frame that holds no frame",
                  tsdf.integrate(never_uploaded, buffers.points, usable, stream),
                  "the GPU frame holds no frame");
    check_refused("refusals, organized points of another size",
                  tsdf.integrate(buffers.frame, odd_points, usable, stream),
                  "the frame is 640x400 px, the organized points 641x401 px");
    check_refused("refusals, intrinsics that cannot be used",
                  tsdf.integrate(odd_frame, buffers.points, usable, stream),
                  "the frame's intrinsics cannot be used");
    const gca::camera_pose not_finite = {
        Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0),
        Eigen::Quaterniond::Identity()};
    check_refused("refusals, a pose that cannot be used",
                  tsdf.integrate(buffers.frame, buffers.points, not_finite, stream),
                  "the pose cannot be used");
    // An error an earlier call leaves: cudaMalloc's refusal of 2^52 bytes, more than the board
    // has [S: docs/HARDWARE.md, 7607 MB].
    void *impossible = nullptr;
    check(cudaMalloc(&impossible, std::size_t{1} << 52) != cudaSuccess,
          "refusals: cudaMalloc refuses 2^52 bytes");
    check_refused("refusals, an integration after an earlier error",
                  tsdf.integrate(buffers.frame, buffers.points, usable, stream),
                  "left by an earlier call");
    check_no_cuda_error("refusals, an integration after an earlier error");

    const gca::extraction_result from_empty = empty.extract(buffer, stream);
    check(from_empty.found == 0 && from_empty.written == 0,
          "refusals: an empty sparse TSDF finds no point");
    check_refusal_of("refusals, an extraction from an empty sparse TSDF", from_empty,
                     "holds no memory");
    gca::surface_points no_buffer;
    check_refusal_of("refusals, an extraction into an empty buffer",
                     tsdf.extract(no_buffer, stream), "the buffer of points holds no memory");
    check(cudaMalloc(&impossible, std::size_t{1} << 52) != cudaSuccess,
          "refusals: cudaMalloc refuses 2^52 bytes");
    check_refusal_of("refusals, an extraction after an earlier error", tsdf.extract(buffer, stream),
                     "left by an earlier call");
    check_no_cuda_error("refusals, an extraction after an earlier error");

    const gca::sparse_tsdf_usage after = read_usage(tsdf, stream, what);
    print_usage("refusals, after them", after);
    check(same_usage(after, before), "refusals: the counts as they were");
    std::vector<gca::surface_point> points_after =
        extract_points(tsdf, buffer, stream, what + ", after");
    std::vector<gca::surface_point> sorted_before = points_before;
    sort_points(sorted_before);
    sort_points(points_after);
    check(same_points(sorted_before, points_after), "refusals: the points as they were");

    // A buffer of ten points: not done, every point found counted, ten written.
    const gca::extraction_result small = tsdf.extract(small_buffer, stream);
    check_refusal_of("refusals, a buffer of ten points", small, "the buffer holds 10");
    check(small.found == points_before.size() && small.written == 10,
          "refusals, a buffer of ten points: every point counted, ten written");
    std::vector<gca::surface_point> ten(10);
    check(cudaMemcpy(ten.data(), small_buffer.points().data(), sizeof(gca::surface_point) * 10,
                     cudaMemcpyDeviceToHost) == cudaSuccess,
          "refusals: the ten points downloaded");
    check(std::all_of(ten.begin(), ten.end(),
                      [](const gca::surface_point &p) {
                          return std::abs(double{p.position_m.z} - 1e-3 * wall_mm) <=
                                 exact_depth_tolerance_m;
                      }),
          "refusals, a buffer of ten points: each on the wall");
    check(pool.refused_while_closed() == 0, "refusals: the closed pool asked for nothing");
}
} // namespace

/// Runs the checks of the integration and the extraction and prints the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    cudaStream_t stream = nullptr;
    if (!create_stream(stream))
        return gca_test::check_summary("test_sparse_tsdf_fusion");
    check_walls(stream);
    check_far_depth(stream);
    check_scene(
        tabletop_scene(model_of("tabletop, live model", gca::object_class::tabletop_object, false),
                       tabletop_live_tolerances),
        stream);
    check_scene(tabletop_scene(model_of("tabletop, final model", gca::object_class::tabletop_object,
                                        true, tabletop_final_far_m),
                               tabletop_final_tolerances),
                stream);
    check_scene(large_scene(model_of("large, live model", gca::object_class::large_object, false),
                            large_live_tolerances),
                stream);
    check_scene(large_scene(model_of("large, final model", gca::object_class::large_object, true,
                                     large_final_far_m),
                            large_final_tolerances),
                stream);
    check_noise(stream);
    check_used_up(stream);
    check_refusals(stream);
    check_no_cuda_error("the end");
    check(cudaStreamDestroy(stream) == cudaSuccess, "the CUDA stream destroyed");
    return gca_test::check_summary("test_sparse_tsdf_fusion");
}
