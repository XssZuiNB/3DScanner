// SPDX-License-Identifier: 0BSD
/// @file test_sparse_tsdf_allocation.cpp
/// The fusion module's allocation of the blocks near a frame's surface
/// (fusion/sparse_tsdf.hpp, sparse_tsdf::allocate_near_surface()), on frames of the tests'
/// synthetic source at their known poses, 640x400 with the camera's intrinsics, and on built
/// frames of walls; and the request's check of one frame against the hash's room. The method is
/// docs/methods/sparse-tsdf.md, "The allocation near a frame's surface". The models' frame is the
/// scene's world frame. The expected blocks come from the analytic surface, the first hit of each
/// ray (synthetic_source.hpp, first_hit()), and the truncation's rule computed here in double,
/// apart from the module's code; the allocated blocks are read through find_blocks().
/// - Two scenes, each at the live model's voxel size and depth range of its object class
///   (scan/object_class.hpp): the tabletop box on its plane, and a large box on a floor, whose
///   band reaches 270 mm at 3 m.
///   - No block farther than stated: every allocated block meets the cube of half edge eps
///     around a sample of the analytic band, eps per ray as band_margin() states; so every block
///     lies within mu(z) |r| + sqrt(3) eps of the surface.
///   - Every sample of the analytic band on a pixel's ray, the band narrowed by the depth's
///     rounding, lies in an allocated block.
///   - Between the rays, across the whole band, in every cell of four valid pixels, edges
///     between surfaces included: a sample lies in an allocated block or within h of one, h its
///     distance from the nearest of the four pixels' segments of the band that the allocation
///     must cover; the misses, their largest distance and their nearest approach to the surface
///     are printed.
///   - The frame's list holds every block once: as many slots as blocks, none unlisted.
/// - The same frame allocated again changes no number and no count; ten new sparse TSDFs given
///   the frame hold the same set of blocks and counts; a quaternion of length 2 gives what its
///   unit quaternion gives. A second frame at another pose lists as many slots as a new sparse
///   TSDF gives it blocks, and the blocks of both are the union.
/// - A far depth nearer than some of the frame's points: those points are not walked, each
///   counted.
/// - A pose far outside the hash's range: every block refused, none listed. A pose whose segments'
///   ends are not finite in float: no ray walked, each counted.
/// - Bands across more than max_blocks_per_ray blocks, in a picture of a field of view far wider
///   than the camera's: no ray walked, each counted, nothing inserted.
/// - Refusals as values, each leaving the counts as they were: an empty sparse TSDF, one moved
///   from, empty organized points, poses that are not finite in float or have an orientation of
///   zero, CUDA's last error left by an earlier call; CUDA's refusal of the launch for points
///   whose rows need more blocks than a grid holds, which resets only the list's count. The
///   request's refusals of a single value are test_sparse_tsdf_blocks.cpp's.
/// - One frame's bound (docs/methods/sparse-tsdf.md, "One frame's blocks"), computed here in
///   double from the method's formula with the camera's 1280x800 intrinsics: every object class's
///   live and final models pass the request's checks at its budget and far depth, each requested
///   from a closed pool, which refuses the first claim, so that no device memory is claimed for
///   them; each final model at the farthest depth the request takes, 10 m, is refused by the
///   bound before any claim, at its object class's budget; at the edge the request is
///   granted with a budget of B blocks and refused with B + 1, granted at a far depth whose bound
///   lies 50 blocks inside the room and refused 50 blocks outside it. The edge frame, a wall
///   facing the camera that fills 1280x800 at the far depth, at two poses: it touches no more
///   blocks than the bound and fewer than the room, and, as the pool holds fewer, the list holds
///   the pool's blocks and the rest is counted as unlisted.
/// - Every buffer is requested before the pool closes, and the allocations ask the closed pool
///   for nothing.

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
#include "stage_frames.hpp"
#include "synthetic_source.hpp"
#include "types/camera_pose.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::check;
using gca_test::check_no_cuda_error;
using gca_test::check_refusal;
using gca_test::check_refused;
using gca_test::create_stream;
using gca_test::find;
using gca_test::granted;
using gca_test::lookup;
using gca_test::make_lookup;
using gca_test::method_truncation;
using gca_test::print_usage;
using gca_test::read_usage;
using gca_test::same_usage;
using gca_test::small_description;

/// The blocks of the cases' sparse TSDF, and its budget: 32768 x 4100 + 131072 x 16 + 16 bytes
/// [S: computed, docs/methods/sparse-tsdf.md, "The pool and the memory budget"], 136 MB, more
/// blocks than the frames touch [M 2026-11-14: this test's counts, 5775 for both tabletop frames,
/// 10877 for the large one], and a room for one frame, 131072 - 32768 = 98304 blocks, above the
/// bound of both cases, 87200 and 92152 blocks [S: computed, "One frame's blocks"].
constexpr std::uint32_t test_blocks = 32768;
constexpr std::uint64_t test_budget = 136'445'968;

/// Float's rounding of a point within a few metres of the origin, as the kernel computes it,
/// about 1e-6 m [S: float32's relative step 6e-8, a few roundings]; taken ten times
/// [?: a choice], so that a point that close to an allocated block counts as in it.
constexpr double float_slack_m = 1e-5;

/// The samples along a band, from one end to the other: 17, the band in 16 steps [?: a choice].
constexpr int band_samples = 17;

/// A block's coordinates as one key, for the host's sets (the hash's key, block_key()).
using block_set = std::unordered_set<std::uint64_t>;

/// A scene, the frame size and the sparse TSDF's parameters of a case.
struct scene_case
{
    std::string name;                       ///< the case, named in the check lines
    gca_test::synthetic_scene scene;        ///< the scene and the frames' poses
    gca::frame_description description;     ///< the frames' size, intrinsics and depth unit
    gca::organized_points_parameters range; ///< the depth range of the organized points
    gca::sparse_tsdf_parameters tsdf;       ///< the sparse TSDF's parameters
};

/// Gives an object class's live voxel size and depth range as a case takes them.
/// @param kind [in] the object class.
/// @param range [out] its depth range, rounded to float.
/// @return its live model's voxel size, rounded to float, in m.
float live_model_of(gca::object_class kind, gca::organized_points_parameters &range)
{
    const gca::object_class_parameters p = gca::parameters_of(kind);
    range = {static_cast<float>(p.min_depth_m), static_cast<float>(p.max_depth_m)};
    return static_cast<float>(p.live_voxel_size_m);
}

/// The tabletop case: the tabletop box on its plane from the first two poses of the walk around
/// it, at the tabletop object's live voxel size and depth range.
/// @return the case.
scene_case tabletop_case()
{
    const std::optional<std::vector<Eigen::Isometry3d>> walk =
        gca_test::walk_around(gca_test::tabletop_walk());
    check(walk.has_value() && walk->size() >= 2, "tabletop: the walk's poses");
    const gca_test::synthetic_scene scene{
        .plane = gca_test::tabletop_plane,
        .box = gca_test::tabletop_box,
        .trajectory = walk && walk->size() >= 2
                          ? std::vector<Eigen::Isometry3d>{(*walk)[0], (*walk)[1]}
                          : std::vector<Eigen::Isometry3d>{}};
    gca::organized_points_parameters range{};
    const float voxel_m = live_model_of(gca::object_class::tabletop_object, range);
    return {"tabletop",
            scene,
            small_description,
            range,
            {test_budget, voxel_m, method_truncation, range.far_m}};
}

/// The large case's scene: a box of 1.2 m x 0.8 m x 1.0 m, a large object of 0.5 m to 2 m
/// (GLOSSARY.md), turned by 20 degrees, on a floor of 8 m x 8 m, seen from 2.7 m away and 1.5 m up
/// [?: a choice], so that the floor's far part lies beyond the depth range and its band reaches
/// 270 mm near 3 m.
/// @return the scene, its one pose that view.
gca_test::synthetic_scene large_scene()
{
    const std::optional<Eigen::Isometry3d> pose =
        gca_test::look_at(Eigen::Vector3d(2.2, -1.2, 1.5), Eigen::Vector3d(0.0, 0.0, 0.5), 0.0);
    check(pose.has_value(), "large: the camera's pose");
    return {.plane = {4.0, 4.0},
            .box = gca_test::synthetic_box{0.0, 0.0, 20.0, 1.2, 0.8, 1.0},
            .trajectory =
                pose ? std::vector<Eigen::Isometry3d>{*pose} : std::vector<Eigen::Isometry3d>{}};
}

/// The large case: the large scene at the large object's live voxel size and depth range.
/// @return the case.
scene_case large_case()
{
    gca::organized_points_parameters range{};
    const float voxel_m = live_model_of(gca::object_class::large_object, range);
    return {"large",
            large_scene(),
            small_description,
            range,
            {test_budget, voxel_m, method_truncation, range.far_m}};
}

/// Gives the truncation distance of a case at a depth, computed here in double from the method's
/// rule: mu(z) = max(mu_min v, k c z^2).
/// @param parameters [in] the sparse TSDF's parameters.
/// @param z_m [in] the depth, in m.
/// @return mu(z), in m.
double truncation_m(const gca::sparse_tsdf_parameters &parameters, double z_m)
{
    const gca::truncation_parameters &t = parameters.truncation;
    return std::max(double{t.min_voxels} * double{parameters.voxel_size_m},
                    double{t.noise_multiple} * double{t.noise_per_m} * z_m * z_m);
}

/// Gives a block's edge: 8 voxels.
/// @param parameters [in] the sparse TSDF's parameters.
/// @return 8 v, in m.
double block_edge_m(const gca::sparse_tsdf_parameters &parameters)
{
    return 8.0 * double{parameters.voxel_size_m};
}

/// Gives the block a point lies in.
/// @param p [in] the point, in the models' frame, in m.
/// @param edge_m [in] a block's edge, in m.
/// @return floor(p / edge) on each axis.
gca::block_coordinate block_of(const Eigen::Vector3d &p, double edge_m)
{
    return {static_cast<std::int32_t>(std::floor(p.x() / edge_m)),
            static_cast<std::int32_t>(std::floor(p.y() / edge_m)),
            static_cast<std::int32_t>(std::floor(p.z() / edge_m))};
}

/// Gives a point's distance from a block's cube.
/// @param p [in] the point, in the models' frame, in m.
/// @param b [in] the block.
/// @param edge_m [in] a block's edge, in m.
/// @return the distance, in m; 0 inside the cube.
double distance_from_block(const Eigen::Vector3d &p, const gca::block_coordinate &b, double edge_m)
{
    const Eigen::Vector3d low = Eigen::Vector3d(b.x, b.y, b.z) * edge_m;
    const Eigen::Vector3d high = low + Eigen::Vector3d::Constant(edge_m);
    const Eigen::Vector3d outside = (low - p).cwiseMax(p - high).cwiseMax(0.0);
    return outside.norm();
}

/// Gives a point's distance from the nearest allocated block among the blocks within reach: its
/// own and those whose cube lies within one block's edge of it.
/// @param p [in] the point, in the models' frame, in m.
/// @param allocated [in] the allocated blocks.
/// @param edge_m [in] a block's edge, in m.
/// @return the distance, in m; 0 when its own block is allocated; infinity when none within reach
///         is.
double distance_from_allocated(const Eigen::Vector3d &p, const block_set &allocated, double edge_m)
{
    const gca::block_coordinate own = block_of(p, edge_m);
    if (allocated.count(gca::block_key(own)) != 0)
        return 0.0;
    double nearest = std::numeric_limits<double>::infinity();
    for (int dx = -2; dx <= 2; dx++)
        for (int dy = -2; dy <= 2; dy++)
            for (int dz = -2; dz <= 2; dz++)
            {
                const gca::block_coordinate b{own.x + dx, own.y + dy, own.z + dz};
                if (allocated.count(gca::block_key(b)) != 0)
                    nearest = std::min(nearest, distance_from_block(p, b, edge_m));
            }
    return nearest;
}

/// A frame of a case with what the test knows of it.
struct known_frame
{
    const scene_case *from_case = nullptr;             ///< the case the frame belongs to
    std::vector<gca_test::synthetic_surface> surfaces; ///< the scene's known surfaces
    Eigen::Isometry3d pose;                            ///< camera frame to the models' frame
    std::vector<std::uint8_t> valid;                   ///< the organized points' marks, downloaded
    std::vector<std::optional<gca_test::synthetic_hit>> hits; ///< each pixel's analytic hit
};

/// Gives the ray of a point of a frame's picture (synthetic_source.hpp, pixel_ray()).
/// @param frame [in] the frame.
/// @param u [in] the column, in px, any real value.
/// @param v [in] the row, in px, any real value.
/// @return the ray in the camera frame, its z 1.
Eigen::Vector3d ray_of(const known_frame &frame, double u, double v)
{
    return gca_test::pixel_ray(frame.from_case->description.intrinsics, u, v);
}

/// Gives the analytic hit of a camera's ray.
/// @param frame [in] the frame.
/// @param ray [in] the ray in the camera frame, its z 1.
/// @return the hit, its s the depth along the camera's z, in m; nothing when the ray meets no
///         surface.
std::optional<gca_test::synthetic_hit> hit_of(const known_frame &frame, const Eigen::Vector3d &ray)
{
    return gca_test::first_hit(frame.surfaces, frame.from_case->scene.sphere,
                               frame.pose.translation(), frame.pose.linear() * ray);
}

/// Gives the depth's rounding: half a depth unit [S: the synthetic source rounds each depth to the
/// nearest unit], and float's rounding of z = d unit, below 1e-6 m within 3 m [S: float32].
/// @param frame [in] the frame.
/// @return the largest distance between a valid pixel's measured and analytic depths, in m.
double depth_rounding_m(const known_frame &frame)
{
    return 0.5 * double{frame.from_case->description.depth_unit_m} + 1e-6;
}

/// Gives the margin of a ray's analytic band within which the allocated segment of its pixel
/// lies, as a distance in the models' frame: the measured depth z_m lies within the rounding q of
/// the analytic z_a, so mu(z_m) within 2 k c (z_a + q) q of mu(z_a); the band's samples lie step /
/// 2 apart along z; both times |r|, the length of the ray of z 1; and float's rounding.
/// @param frame [in] the frame.
/// @param z_m [in] the analytic depth, in m.
/// @param ray_length [in] |r|.
/// @param step_m [in] the band's sampling step along z, in m.
/// @return eps, in m.
double band_margin(const known_frame &frame, double z_m, double ray_length, double step_m)
{
    const double q = depth_rounding_m(frame);
    const gca::truncation_parameters &t = frame.from_case->tsdf.truncation;
    const double mu_change = 2.0 * double{t.noise_multiple} * double{t.noise_per_m} * (z_m + q) * q;
    return ray_length * (0.5 * step_m + q + mu_change) + float_slack_m;
}

/// Gives the half length of the band on a pixel's ray that the allocation must cover: the analytic
/// band narrowed by the depth's rounding q at each end, as the measured depth z_m lies within q of
/// the analytic z_a and mu(z_m) >= mu(z_a - q).
/// @param frame [in] the frame.
/// @param z_a [in] the pixel's analytic depth, in m.
/// @return mu(z_a - q) - q, in m along the camera's z.
double covered_reach_m(const known_frame &frame, double z_a)
{
    const double q = depth_rounding_m(frame);
    return truncation_m(frame.from_case->tsdf, z_a - q) - q;
}

/// The blocks near the analytic band of a frame: every block that meets the cube of half edge eps
/// around a sample of the band of a valid pixel's ray, z_a +- mu(z_a), sampled every half voxel
/// along z, eps by band_margin().
struct near_band
{
    std::vector<gca::block_coordinate> blocks; ///< the blocks, each once
    double largest_margin_m = 0.0;             ///< the largest eps of a ray, in m
    std::size_t valid_without_hit = 0;         ///< valid pixels whose ray meets no surface
};

/// Collects the blocks near the analytic band of a frame.
/// @param frame [in] the frame, its marks and hits filled.
/// @return the blocks and the largest margin.
near_band blocks_near_band(const known_frame &frame)
{
    const scene_case &c = *frame.from_case;
    const double edge_m = block_edge_m(c.tsdf);
    const double step_m = 0.5 * double{c.tsdf.voxel_size_m};
    near_band result;
    block_set seen;
    for (std::uint32_t v = 0; v < c.description.size.height; v++)
        for (std::uint32_t u = 0; u < c.description.size.width; u++)
        {
            const std::size_t pixel = static_cast<std::size_t>(v) * c.description.size.width + u;
            if (frame.valid[pixel] == 0)
                continue;
            if (!frame.hits[pixel])
            {
                result.valid_without_hit++;
                continue;
            }
            const Eigen::Vector3d ray = ray_of(frame, u, v);
            const double z_a = frame.hits[pixel]->s;
            const double mu = truncation_m(c.tsdf, z_a);
            const double eps = band_margin(frame, z_a, ray.norm(), step_m);
            result.largest_margin_m = std::max(result.largest_margin_m, eps);
            const int steps = static_cast<int>(std::ceil(2.0 * mu / step_m));
            gca::block_coordinate last_low{0, 0, 0};
            gca::block_coordinate last_high{-1, -1, -1};
            for (int i = 0; i <= steps; i++)
            {
                const double z = std::min(z_a - mu + i * step_m, z_a + mu);
                const Eigen::Vector3d p = frame.pose * (z * ray);
                const gca::block_coordinate low =
                    block_of(p - Eigen::Vector3d::Constant(eps), edge_m);
                const gca::block_coordinate high =
                    block_of(p + Eigen::Vector3d::Constant(eps), edge_m);
                if (low.x == last_low.x && low.y == last_low.y && low.z == last_low.z &&
                    high.x == last_high.x && high.y == last_high.y && high.z == last_high.z)
                    continue;
                last_low = low;
                last_high = high;
                for (std::int32_t x = low.x; x <= high.x; x++)
                    for (std::int32_t y = low.y; y <= high.y; y++)
                        for (std::int32_t z_block = low.z; z_block <= high.z; z_block++)
                        {
                            const gca::block_coordinate b{x, y, z_block};
                            if (seen.insert(gca::block_key(b)).second)
                                result.blocks.push_back(b);
                        }
            }
        }
    return result;
}

/// Gives a point's distance from a segment of a ray: the ray's points between two depths.
/// @param p [in] the point, in the camera frame, in m.
/// @param ray [in] the ray, its z 1.
/// @param near_m [in] the segment's nearer depth, in m.
/// @param far_m [in] its farther depth, in m, >= near_m.
/// @return the distance, in m.
double distance_from_segment(const Eigen::Vector3d &p, const Eigen::Vector3d &ray, double near_m,
                             double far_m)
{
    const double t = std::clamp(p.dot(ray) / ray.squaredNorm(), near_m, far_m);
    return (p - t * ray).norm();
}

/// The misses between the rays of one kind of cell: samples outside every allocated block.
struct misses
{
    std::size_t count = 0;        ///< the misses
    double largest_m = 0.0;       ///< the largest distance of a miss from an allocated block, in m
    double largest_bound_m = 0.0; ///< the largest bound h of a miss, in m
    double nearest_of_mu = 1.0;   ///< the smallest |eta| / mu of a miss: how near the surface
};

/// What the samples of a frame's band found.
struct coverage
{
    std::size_t ray_samples = 0;  ///< samples on the pixels' own rays
    std::size_t ray_outside = 0;  ///< of them, farther than float's slack from every allocated
                                  ///< block
    std::size_t cells = 0;        ///< cells of four valid pixels
    std::size_t cell_samples = 0; ///< samples between the rays
    std::size_t cell_beyond = 0;  ///< of them, farther than their bound h from every block
    misses one_surface;           ///< misses of samples whose ray meets the surface of all four
                                  ///< pixels' rays
    misses at_edge; ///< misses of samples at an edge between surfaces: their ray meets
                    ///< another surface than one of the four pixels' rays
};

/// Adds a miss to its kind.
/// @param kind [in,out] the misses of the sample's kind of cell.
/// @param distance_m [in] its distance from the nearest allocated block, in m.
/// @param bound_m [in] its bound h, in m.
/// @param of_mu [in] its |eta| / mu.
void add_miss(misses &kind, double distance_m, double bound_m, double of_mu)
{
    kind.count++;
    kind.largest_m = std::max(kind.largest_m, distance_m);
    kind.largest_bound_m = std::max(kind.largest_bound_m, bound_m);
    kind.nearest_of_mu = std::min(kind.nearest_of_mu, of_mu);
}

/// Samples the band of a pixel's own ray, narrowed by the depth's rounding, against the allocated
/// blocks.
/// @param frame [in] the frame.
/// @param allocated [in] the allocated blocks.
/// @param u [in] the pixel's column.
/// @param v [in] the pixel's row.
/// @param z_a [in] its analytic depth, in m.
/// @param found [in,out] the counts, added to.
void sample_pixel_ray(const known_frame &frame, const block_set &allocated, std::uint32_t u,
                      std::uint32_t v, double z_a, coverage &found)
{
    const double edge_m = block_edge_m(frame.from_case->tsdf);
    const Eigen::Vector3d ray = ray_of(frame, u, v);
    const double reach = covered_reach_m(frame, z_a);
    for (int i = 0; i < band_samples; i++)
    {
        const double z = z_a + reach * (2.0 * i / (band_samples - 1) - 1.0);
        found.ray_samples++;
        if (distance_from_allocated(frame.pose * (z * ray), allocated, edge_m) > float_slack_m)
            found.ray_outside++;
    }
}

/// Samples the whole analytic band between the rays of a cell of four valid pixels, at the middles
/// of its top and left edges and at its centre, against the allocated blocks: a sample lies in an
/// allocated block or within h of one, h its distance from the nearest of the four pixels' covered
/// segments (covered_reach_m()), whose points each lie in an allocated block.
/// @param frame [in] the frame.
/// @param allocated [in] the allocated blocks.
/// @param u [in] the cell's top left pixel's column; u + 1 lies in the picture.
/// @param v [in] its row; v + 1 lies in the picture.
/// @param found [in,out] the counts, added to.
void sample_cell(const known_frame &frame, const block_set &allocated, std::uint32_t u,
                 std::uint32_t v, coverage &found)
{
    const scene_case &c = *frame.from_case;
    const double edge_m = block_edge_m(c.tsdf);
    const std::size_t width = c.description.size.width;
    const std::size_t pixel = static_cast<std::size_t>(v) * width + u;
    const std::size_t corners[4] = {pixel, pixel + 1, pixel + width, pixel + width + 1};
    Eigen::Vector3d corner_ray[4];
    double corner_near[4];
    double corner_far[4];
    for (int k = 0; k < 4; k++)
    {
        if (frame.valid[corners[k]] == 0 || !frame.hits[corners[k]])
            return;
        corner_ray[k] = ray_of(frame, u + k % 2, v + k / 2);
        const double z_a = frame.hits[corners[k]]->s;
        corner_near[k] = z_a - covered_reach_m(frame, z_a);
        corner_far[k] = z_a + covered_reach_m(frame, z_a);
    }
    found.cells++;
    const double offsets[3][2] = {{0.5, 0.0}, {0.0, 0.5}, {0.5, 0.5}};
    for (const auto &offset : offsets)
    {
        const Eigen::Vector3d ray = ray_of(frame, u + offset[0], v + offset[1]);
        const std::optional<gca_test::synthetic_hit> hit = hit_of(frame, ray);
        // A point between the rays whose surface lies outside the depth range is not in the
        // frame's band.
        if (!hit || hit->s < double{c.range.near_m} || hit->s > double{c.range.far_m})
            continue;
        bool one_surface = true;
        for (const std::size_t corner : corners)
            one_surface = one_surface && frame.hits[corner]->surface == hit->surface;
        const double mu = truncation_m(c.tsdf, hit->s);
        for (int i = 0; i < band_samples; i++)
        {
            const double eta = mu * (2.0 * i / (band_samples - 1) - 1.0);
            const Eigen::Vector3d p = (hit->s + eta) * ray;
            double h = std::numeric_limits<double>::infinity();
            for (int k = 0; k < 4; k++)
                h = std::min(
                    h, distance_from_segment(p, corner_ray[k], corner_near[k], corner_far[k]));
            const double d = distance_from_allocated(frame.pose * p, allocated, edge_m);
            found.cell_samples++;
            if (d > h + float_slack_m)
                found.cell_beyond++;
            if (d > float_slack_m)
                add_miss(one_surface ? found.one_surface : found.at_edge, d, h, std::abs(eta) / mu);
        }
    }
}

/// Samples the analytic band of a frame against the allocated blocks: on each valid pixel's ray,
/// and between the rays in each cell of four valid pixels.
/// @param frame [in] the frame, its marks and hits filled.
/// @param allocated [in] the allocated blocks.
/// @return what the samples found.
coverage sample_band(const known_frame &frame, const block_set &allocated)
{
    const gca::frame_size size = frame.from_case->description.size;
    coverage found;
    for (std::uint32_t v = 0; v < size.height; v++)
        for (std::uint32_t u = 0; u < size.width; u++)
        {
            const std::size_t pixel = static_cast<std::size_t>(v) * size.width + u;
            if (frame.valid[pixel] == 0 || !frame.hits[pixel])
                continue;
            sample_pixel_ray(frame, allocated, u, v, frame.hits[pixel]->s, found);
            if (u + 1 < size.width && v + 1 < size.height)
                sample_cell(frame, allocated, u, v, found);
        }
    return found;
}

/// Gives the coordinates of a lookup that have a block.
/// @param found [in] the lookup.
/// @param numbers [in] its numbers (find()).
/// @return their keys.
block_set with_block(const lookup &found, const std::vector<std::uint32_t> &numbers)
{
    block_set result;
    for (std::size_t i = 0; i < numbers.size(); i++)
        if (numbers[i] != gca::no_block)
            result.insert(gca::block_key(found.host[i]));
    return result;
}

/// Gives a pose of the camera as the sparse TSDF takes it.
/// @param pose [in] camera frame to the models' frame.
/// @return the same pose as position and orientation.
gca::camera_pose as_camera_pose(const Eigen::Isometry3d &pose)
{
    return {pose.translation(), Eigen::Quaterniond(pose.linear())};
}

/// A frame's buffers on the GPU, from the pool: the GPU frame and its organized points.
struct frame_buffers
{
    gca::gpu_frame frame;         ///< the GPU frame the frame is uploaded into
    gca::organized_points points; ///< the organized points it is deprojected into
};

/// Requests a frame's buffers from the pool.
/// @param pool [in,out] the pool, open.
/// @param size [in] the frame's size.
/// @param what [in] the case, named in the check lines.
/// @return the buffers; empty ones and a failed check when the pool refused.
frame_buffers request_buffers(gca::gpu_pool &pool, const gca::frame_size &size,
                              const std::string &what)
{
    return {granted(what + ": the GPU frame", gca::gpu_frame::request(pool, size)),
            granted(what + ": the organized points", gca::organized_points::request(pool, size))};
}

/// Uploads a frame, deprojects it and downloads its marks, and finds each pixel's analytic hit.
/// @param c [in] the case.
/// @param host [in] the frame, rendered at pose.
/// @param surfaces [in] the scene's known surfaces.
/// @param pose [in] the frame's pose, camera frame to the models' frame.
/// @param buffers [in,out] the GPU frame and organized points.
/// @param stream [in] the stream, which the download waits for.
/// @return the frame with what the test knows of it; its marks empty and a failed check when the
///         GPU failed.
known_frame prepare(const scene_case &c, const gca::frame &host,
                    const std::vector<gca_test::synthetic_surface> &surfaces,
                    const Eigen::Isometry3d &pose, frame_buffers &buffers, cudaStream_t stream)
{
    known_frame known{&c, surfaces, pose, {}, {}};
    const gca::step_result up = buffers.frame.upload(host, stream);
    check(up.done, (c.name + ": upload, " + up.message).c_str());
    const gca::step_result made = buffers.points.deproject(buffers.frame, c.range, stream);
    check(made.done, (c.name + ": deproject, " + made.message).c_str());
    const std::size_t pixels =
        static_cast<std::size_t>(c.description.size.width) * c.description.size.height;
    known.valid.resize(pixels);
    const bool downloaded =
        up.done && made.done &&
        cudaMemcpyAsync(known.valid.data(), buffers.points.validity_mask().data(), pixels,
                        cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess;
    check(downloaded, (c.name + ": the marks downloaded").c_str());
    if (!downloaded)
        known.valid.assign(pixels, 0);
    known.hits.resize(pixels);
    for (std::uint32_t v = 0; v < c.description.size.height; v++)
        for (std::uint32_t u = 0; u < c.description.size.width; u++)
        {
            const std::size_t pixel = static_cast<std::size_t>(v) * c.description.size.width + u;
            if (known.valid[pixel] != 0)
                known.hits[pixel] = hit_of(known, ray_of(known, u, v));
        }
    return known;
}

/// Allocates a frame's blocks and checks that the work was queued.
/// @param tsdf [in,out] the sparse TSDF.
/// @param points [in] the frame's organized points.
/// @param pose [in] the frame's pose.
/// @param stream [in] the stream.
/// @param what [in] the case, named in the check line.
void allocate(gca::sparse_tsdf &tsdf, const gca::organized_points &points,
              const gca::camera_pose &pose, cudaStream_t stream, const std::string &what)
{
    const gca::step_result queued = tsdf.allocate_near_surface(points, pose, stream);
    check(queued.done, (what + ": allocation queued, " + queued.message).c_str());
}

/// Checks a frame's allocated blocks against its analytic band: none farther than stated, the
/// band on every pixel's ray covered, a sample between the rays within its bound; and that the
/// frame's list holds each of its blocks once.
/// @param what [in] the case, printed and named in the check lines.
/// @param frame [in] the frame.
/// @param near [in] the blocks near its band, as looked up.
/// @param numbers [in] their numbers in a sparse TSDF given only this frame.
/// @param usage [in] that sparse TSDF's usage.
/// @return the allocated blocks.
block_set check_band(const std::string &what, const known_frame &frame, const lookup &near,
                     const std::vector<std::uint32_t> &numbers, const gca::sparse_tsdf_usage &usage)
{
    print_usage(what, usage);
    const block_set allocated = with_block(near, numbers);
    std::printf("%s: %zu blocks near the band, %zu of them allocated\n", what.c_str(),
                near.host.size(), allocated.size());
    check(usage.blocks_in_use > 0 && usage.blocks_refused == 0 && usage.coordinates_refused == 0 &&
              usage.rays_not_walked == 0,
          (what + ": blocks allocated, none refused, every ray walked").c_str());
    check(allocated.size() == usage.blocks_in_use,
          (what + ": every allocated block near the analytic band").c_str());
    check(usage.listed_slots == usage.blocks_in_use && usage.unlisted_slots == 0,
          (what + ": the frame's list holds each of its blocks once").c_str());
    const coverage found = sample_band(frame, allocated);
    std::printf("%s: %zu samples on the pixels' rays, %zu outside every block; %zu samples "
                "between the rays in %zu cells, %zu beyond their bound\n",
                what.c_str(), found.ray_samples, found.ray_outside, found.cell_samples, found.cells,
                found.cell_beyond);
    for (const auto &[kind, missed] : {std::pair{"on one surface", found.one_surface},
                                       std::pair{"at an edge between surfaces", found.at_edge}})
        std::printf("%s: %zu misses between the rays %s, up to %.4f mm from an allocated block "
                    "(their bound up to %.4f mm), the nearest to the surface at |eta| = %.3f mu\n",
                    what.c_str(), missed.count, kind, 1e3 * missed.largest_m,
                    1e3 * missed.largest_bound_m, missed.nearest_of_mu);
    check(found.ray_samples > 0 && found.ray_outside == 0,
          (what + ": every sample of the band on a pixel's ray in an allocated block").c_str());
    check(
        found.cell_samples > 0 && found.cell_beyond == 0,
        (what + ": every sample between the rays within its bound of an allocated block").c_str());
    return allocated;
}

/// Gives the slots of the hash of a number of blocks: the least power of two >= 4 B
/// [S: docs/methods/sparse-tsdf.md, "The pool and the memory budget"].
/// @param blocks [in] B, from 1 to 2^29.
/// @return S.
std::uint64_t slots_of(std::uint32_t blocks)
{
    std::uint64_t slots = 1;
    while (slots < 4 * std::uint64_t{blocks})
        slots *= 2;
    return slots;
}

/// Gives the budget of a sparse TSDF of a number of blocks: 4100 B + 16 S + 16 bytes
/// [S: docs/methods/sparse-tsdf.md, "The pool and the memory budget"].
/// @param blocks [in] B, from 1 to 2^29.
/// @return the budget, in bytes.
std::uint64_t budget_of(std::uint32_t blocks)
{
    return 4100 * std::uint64_t{blocks} + 16 * slots_of(blocks) + 16;
}

/// Gives the bound of one frame in the method's model, computed here in double apart from the
/// module's code (docs/methods/sparse-tsdf.md, "One frame's blocks"): a surface facing the camera
/// that fills the picture of 1280x800 at the camera's intrinsics there (large_description) at the
/// far depth z_max, N = (w + 2 sqrt(3) e)(h + 2 sqrt(3) e)(2 mu + 2 sqrt(3) e) / e^3 with e = 8 v,
/// mu = mu(z_max), w = 1280 (z_max + mu) / fx and h = 800 (z_max + mu) / fy.
/// @param parameters [in] the sparse TSDF's parameters.
/// @return N, in blocks.
double frame_bound_blocks(const gca::sparse_tsdf_parameters &parameters)
{
    const gca::frame_description &picture = gca_test::large_description;
    const double z = parameters.max_depth_m;
    const double mu = truncation_m(parameters, z);
    const double edge = block_edge_m(parameters);
    const double pad = 2.0 * std::sqrt(3.0) * edge;
    const double w = picture.size.width * (z + mu) / double{picture.intrinsics.fx} + pad;
    const double h = picture.size.height * (z + mu) / double{picture.intrinsics.fy} + pad;
    return w * h * (2.0 * mu + pad) / (edge * edge * edge);
}

/// Gives the room of a sparse TSDF's hash for one frame once its pool is used up.
/// @param layout [in] its layout.
/// @return S - B, in blocks.
double room_of(const gca::sparse_tsdf_layout &layout)
{
    return static_cast<double>(layout.slot_count) - static_cast<double>(layout.capacity_blocks);
}

/// Requests a sparse TSDF from a pool of its own.
/// @param parameters [in] the sparse TSDF's parameters.
/// @param stream [in] the stream its clearing is queued on.
/// @return the request's result; a granted sparse TSDF's memory is freed when it goes.
gca::request_result<gca::sparse_tsdf> request_alone(const gca::sparse_tsdf_parameters &parameters,
                                                    cudaStream_t stream)
{
    gca::gpu_pool pool;
    return gca::sparse_tsdf::request(pool, parameters, stream);
}

/// Checks that a request was refused for the bound of one frame, by its message, holding nothing.
/// @param what [in] the case, printed with the message and named in the check line.
/// @param made [in] the request's result.
void check_refused_by_bound(const std::string &what,
                            const gca::request_result<gca::sparse_tsdf> &made)
{
    check_refusal(what,
                  made.status == gca::request_status::failure &&
                      made.granted.layout().capacity_blocks == 0,
                  made.message, made.message.find("cannot hold one frame") != std::string::npos);
}

/// What a request from a closed pool gave: the request's result and the claims the pool refused.
struct unclaimed_request
{
    gca::request_result<gca::sparse_tsdf> made; ///< closed when every check of the request passed,
                                                ///< failure when one refused; nothing granted
    std::uint64_t claims_refused;               ///< 1 when every check passed: the request's
                                                ///< first claim; 0 when a check refused before it
};

/// Requests a sparse TSDF from a closed pool of its own: the request checks the parameters and
/// the budget, one frame's bound against the room among them, before its first claim, which the
/// closed pool refuses (sparse_tsdf::request()), so that no device memory is claimed.
/// @param parameters [in] the sparse TSDF's parameters.
/// @param stream [in] the stream the request would queue its clearing on; nothing is queued.
/// @return the request's result and the claims the closed pool refused.
unclaimed_request request_from_closed(const gca::sparse_tsdf_parameters &parameters,
                                      cudaStream_t stream)
{
    gca::gpu_pool pool;
    pool.close();
    unclaimed_request checked{gca::sparse_tsdf::request(pool, parameters, stream), 0};
    checked.claims_refused = pool.refused_while_closed();
    return checked;
}

/// Every object class's live and final models pass the request's checks at its budget and far
/// depth, with the method's starting values of the truncation's rule (parameters_of()), without
/// device memory: each is requested from a closed pool, comes back closed with its first claim
/// refused, and the room of its budget's layout (layout_for_budget()) lies above its bound
/// computed here; the bounds are printed against the room. Each final model at the farthest depth
/// the request takes, 10 m, is refused by the same check at its object class's budget before any
/// claim: its bound, 5.4e9 blocks at 1 mm and 8.7e7 at 4 mm, lies above the room of every budget
/// the board's 7607 MB could hold, 7.3e6 blocks at most, at 4.43 GB [S: computed].
/// @param stream [in] the stream.
void check_object_classes(cudaStream_t stream)
{
    for (const gca::object_class kind :
         {gca::object_class::tabletop_object, gca::object_class::large_object})
    {
        const gca::object_class_parameters p = gca::parameters_of(kind);
        const gca::sparse_tsdf_layout layout = gca::layout_for_budget(p.memory_budget_bytes);
        for (const auto &[model, voxel_m] :
             {std::pair{"live", p.live_voxel_size_m}, std::pair{"final", p.full_voxel_size_m}})
        {
            const std::string what = std::string(gca::name_of(kind)) + ", " + model + " model";
            const gca::sparse_tsdf_parameters parameters = {
                p.memory_budget_bytes, static_cast<float>(voxel_m), method_truncation,
                static_cast<float>(p.max_depth_m)};
            const unclaimed_request checked = request_from_closed(parameters, stream);
            std::printf("%s: %.0f mm voxels to %.1f m, one frame's bound %.1f blocks, room %.0f "
                        "(%u blocks, %u slots)\n",
                        what.c_str(), 1e3 * voxel_m, p.max_depth_m, frame_bound_blocks(parameters),
                        room_of(layout), layout.capacity_blocks, layout.slot_count);
            check(checked.made.status == gca::request_status::closed && checked.claims_refused == 1,
                  (what + " at its budget and far depth: every check passed, the first claim "
                          "refused by the closed pool")
                      .c_str());
            check(frame_bound_blocks(parameters) < room_of(layout),
                  (what + ": one frame's bound below the room").c_str());
        }
        const std::string what = std::string(gca::name_of(kind)) + ", final model at 10 m";
        const gca::sparse_tsdf_parameters farthest = {p.memory_budget_bytes,
                                                      static_cast<float>(p.full_voxel_size_m),
                                                      method_truncation, gca::farthest_max_depth_m};
        const unclaimed_request refused = request_from_closed(farthest, stream);
        std::printf("%s: one frame's bound %.3g blocks, room %.0f\n", what.c_str(),
                    frame_bound_blocks(farthest), room_of(layout));
        check_refused_by_bound(what + " at its budget", refused.made);
        check(refused.claims_refused == 0, (what + ": refused before any claim").c_str());
    }
}

/// The parameters at the edge: the large object's live voxel size, 8 mm, and a far depth of
/// 3.12 m [?: a choice], whose bound, about 105800 blocks [S: computed], lies between the least
/// and the greatest room of a hash of 131072 slots, 98304 and 114687 blocks, so that a budget of
/// B blocks has room for it and one of B + 1 not.
/// @return the parameters, the budget left 0.
gca::sparse_tsdf_parameters edge_parameters()
{
    const gca::object_class_parameters large = gca::parameters_of(gca::object_class::large_object);
    return {0, static_cast<float>(large.live_voxel_size_m), method_truncation, 3.12f};
}

/// The bound's least distance from a whole block at the edge along the budget: 0.01 blocks
/// [?: a choice], ten times the difference between this test's bound and the module's, about
/// 1e-3 blocks [S: computed: the focal lengths, floats here and doubles there, differ by 8e-9 of
/// N], so that both put the edge at the same block.
constexpr double bound_clearance_blocks = 0.01;

/// The bound's distance from the room at the two far depths of the edge along the depth: 50
/// blocks [?: a choice], far above the two bounds' difference and within a millimetre of the
/// edge's depth [M 2026-11-14: 3.1196 m and 3.1204 m].
constexpr double depth_edge_margin_blocks = 50.0;

/// The bisection's steps for a far depth: 60, which take the far depths' range of 9.75 m below
/// 1e-17 m, far below float's step at a few metres, about 2.4e-7 m [S: computed], so that it ends
/// between adjacent floats.
constexpr int bisection_steps = 60;

/// The edge of the request's check of one frame, which refuses a bound equal to the room S - B or
/// above it: along the budget, B blocks granted and B + 1 refused, B = S - floor(N) - 1, the most
/// blocks whose room exceeds N, both in the hash of 131072 slots; along the far depth, at the
/// budget of B blocks, a depth whose bound lies depth_edge_margin_blocks below the room granted
/// and one that far above it refused.
/// @param stream [in] the stream.
/// @return B, the blocks of the budget at the edge; 0 when the edge lies elsewhere.
std::uint32_t check_edges(cudaStream_t stream)
{
    gca::sparse_tsdf_parameters parameters = edge_parameters();
    const double bound = frame_bound_blocks(parameters);
    const std::uint64_t slots = 131072;
    const std::uint32_t edge_blocks = static_cast<std::uint32_t>(slots - std::floor(bound) - 1);
    const double fraction = bound - std::floor(bound);
    std::printf("edge: one frame's bound %.4f blocks at %.2f m, %u blocks granted, %u refused\n",
                bound, double{parameters.max_depth_m}, edge_blocks, edge_blocks + 1);
    // The module computes the bound with the focal lengths in double, here they are floats: the
    // two differ by about 1e-3 blocks [S: computed, 612.37 rounded to float differs by 8e-9].
    check(slots_of(edge_blocks) == slots && slots_of(edge_blocks + 1) == slots &&
              fraction > bound_clearance_blocks && fraction < 1.0 - bound_clearance_blocks,
          "edge: B and B + 1 in the hash of 131072 slots, the bound clear of a whole block");
    if (slots_of(edge_blocks) != slots || slots_of(edge_blocks + 1) != slots)
        return 0;

    parameters.memory_budget_bytes = budget_of(edge_blocks);
    const gca::sparse_tsdf inside =
        granted("edge, the budget of B blocks: the sparse TSDF", request_alone(parameters, stream));
    check(inside.layout().capacity_blocks == edge_blocks,
          "edge, the budget of B blocks: B blocks held");
    gca::sparse_tsdf_parameters one_more = parameters;
    one_more.memory_budget_bytes = budget_of(edge_blocks + 1);
    check_refused_by_bound("edge, the budget of B + 1 blocks", request_alone(one_more, stream));

    // Along the far depth: the depths whose bounds lie depth_edge_margin_blocks either side of the
    // room, found by bisection on the bound, which grows with the depth.
    const double room = room_of(inside.layout());
    const auto depth_of_bound = [&parameters](double target) {
        gca::sparse_tsdf_parameters p = parameters;
        double within = gca::nearest_max_depth_m;
        double beyond = gca::farthest_max_depth_m;
        for (int i = 0; i < bisection_steps; i++)
        {
            p.max_depth_m = static_cast<float>(0.5 * (within + beyond));
            (frame_bound_blocks(p) <= target ? within : beyond) = p.max_depth_m;
        }
        p.max_depth_m = static_cast<float>(within);
        return p;
    };
    const gca::sparse_tsdf_parameters nearer = depth_of_bound(room - depth_edge_margin_blocks);
    const gca::sparse_tsdf_parameters farther = depth_of_bound(room + depth_edge_margin_blocks);
    std::printf("edge: at the budget of B blocks, room %.0f; far depth %.6f m, bound %.1f; %.6f m, "
                "bound %.1f\n",
                room, double{nearer.max_depth_m}, frame_bound_blocks(nearer),
                double{farther.max_depth_m}, frame_bound_blocks(farther));
    check(frame_bound_blocks(nearer) < room && frame_bound_blocks(farther) > room,
          "edge: the two far depths either side of the room");
    granted("edge, the far depth inside: the sparse TSDF", request_alone(nearer, stream));
    check_refused_by_bound("edge, the far depth outside", request_alone(farther, stream));
    return edge_blocks;
}

/// Requests a sparse TSDF from a pool of its own, closes the pool, allocates a frame's blocks and
/// reads the usage.
/// @param parameters [in] the sparse TSDF's parameters.
/// @param points [in] the frame's organized points.
/// @param pose [in] the frame's pose.
/// @param stream [in] the stream.
/// @param what [in] the case, named in the check lines.
/// @param tsdf [out] the sparse TSDF, kept for lookups.
/// @return the usage after the allocation.
gca::sparse_tsdf_usage allocate_in_new(const gca::sparse_tsdf_parameters &parameters,
                                       const gca::organized_points &points,
                                       const gca::camera_pose &pose, cudaStream_t stream,
                                       const std::string &what, gca::sparse_tsdf &tsdf)
{
    gca::gpu_pool pool;
    tsdf = granted(what + ": the sparse TSDF", gca::sparse_tsdf::request(pool, parameters, stream));
    pool.close();
    allocate(tsdf, points, pose, stream, what);
    const gca::sparse_tsdf_usage usage = read_usage(tsdf, stream, what);
    check(pool.refused_while_closed() == 0, (what + ": the closed pool asked for nothing").c_str());
    return usage;
}

/// Ten new sparse TSDFs given the same frame hold the same set of blocks and the same counts; so
/// does one given the frame's pose with a quaternion of length 2, which the allocation normalizes.
/// @param c [in] the case.
/// @param frame [in] the frame.
/// @param points [in] its organized points.
/// @param near [in,out] the blocks near its band, looked up.
/// @param allocated [in] the blocks the first sparse TSDF given it holds.
/// @param usage [in] that sparse TSDF's usage.
/// @param stream [in] the stream.
void check_ten_runs(const scene_case &c, const known_frame &frame,
                    const gca::organized_points &points, lookup &near, const block_set &allocated,
                    const gca::sparse_tsdf_usage &usage, cudaStream_t stream)
{
    int same = 0;
    for (int run = 1; run <= 10; run++)
    {
        const std::string what = c.name + ", ten runs, run " + std::to_string(run);
        gca::sparse_tsdf tsdf;
        const gca::sparse_tsdf_usage again =
            allocate_in_new(c.tsdf, points, as_camera_pose(frame.pose), stream, what, tsdf);
        same += same_usage(again, usage) &&
                        with_block(near, find(tsdf, near, stream, what)) == allocated
                    ? 1
                    : 0;
    }
    std::printf("%s, ten runs: %d of 10 with the same %zu blocks and the same counts\n",
                c.name.c_str(), same, allocated.size());
    check(same == 10, (c.name + ", ten runs: the same set of blocks and counts in each").c_str());

    const std::string what = c.name + ", a quaternion of length 2";
    gca::camera_pose doubled = as_camera_pose(frame.pose);
    doubled.orientation.coeffs() *= 2.0;
    gca::sparse_tsdf tsdf;
    const gca::sparse_tsdf_usage scaled =
        allocate_in_new(c.tsdf, points, doubled, stream, what, tsdf);
    print_usage(what, scaled);
    check(same_usage(scaled, usage) &&
              with_block(near, find(tsdf, near, stream, what)) == allocated,
          (what + ": the blocks and counts of its unit quaternion").c_str());
}

/// The edge frame: a wall facing the camera that fills the picture of 1280x800 (large_description)
/// at the far depth of the edge, at its deepest whole depth unit within it, allocated into a new
/// sparse TSDF at the budget of the edge, B blocks, at two poses: the identity, and the camera
/// turned by 54.7 degrees about (1, -1, 0), its optical axis along a diagonal of the blocks' grid.
/// It touches no more blocks than the bound and fewer than the room S - B, every ray walked; as
/// it touches more blocks than the pool holds, the pool is used up, the frame's list holds B slots
/// and the rest is counted as unlisted.
/// @param edge_blocks [in] B (check_edges()).
/// @param stream [in] the stream.
void check_edge_frame(std::uint32_t edge_blocks, cudaStream_t stream)
{
    gca::sparse_tsdf_parameters parameters = edge_parameters();
    parameters.memory_budget_bytes = budget_of(edge_blocks);
    const double bound = frame_bound_blocks(parameters);
    const gca::frame_description &picture = gca_test::large_description;
    // The deepest whole depth unit whose depth, as the organized points round it to float, lies
    // within the far depth, so that no point lies past it.
    auto depth_value = static_cast<std::uint16_t>(parameters.max_depth_m / picture.depth_unit_m);
    while (static_cast<float>(depth_value) * picture.depth_unit_m > parameters.max_depth_m)
        depth_value--;
    const gca::frame wall = gca_test::built_frame(
        picture, [depth_value](std::uint32_t, std::uint32_t) { return depth_value; },
        gca_test::grey);
    gca::gpu_pool pool;
    frame_buffers buffers = request_buffers(pool, picture.size, "edge frame");
    pool.close();
    const gca::step_result up = buffers.frame.upload(wall, stream);
    const gca::step_result made = buffers.points.deproject(
        buffers.frame, {gca::nearest_max_depth_m, parameters.max_depth_m}, stream);
    check(up.done && made.done,
          ("edge frame: upload and deprojection, " + up.message + made.message).c_str());
    const Eigen::Isometry3d turned(Eigen::AngleAxisd(
        std::acos(1.0 / std::sqrt(3.0)), Eigen::Vector3d(1.0, -1.0, 0.0) / std::sqrt(2.0)));
    for (const auto &[name, pose] :
         {std::pair{"identity", Eigen::Isometry3d::Identity()}, std::pair{"turned", turned}})
    {
        const std::string what = std::string("edge frame, ") + name;
        gca::sparse_tsdf tsdf;
        const gca::sparse_tsdf_usage usage =
            allocate_in_new(parameters, buffers.points, as_camera_pose(pose), stream, what, tsdf);
        print_usage(what, usage);
        const double room = room_of(tsdf.layout());
        const std::uint32_t touched = usage.blocks_in_use + usage.blocks_refused;
        std::printf("%s: a wall at %.3f m touches %u blocks; the bound %.1f, %.3f times as many; "
                    "the room %.0f\n",
                    what.c_str(), static_cast<double>(depth_value) * picture.depth_unit_m, touched,
                    bound, bound / touched, room);
        check(usage.rays_not_walked == 0 && usage.coordinates_refused == 0,
              (what + ": every ray walked, no coordinate refused").c_str());
        check(touched <= bound && touched < room,
              (what + ": no more blocks than the bound, fewer than the room").c_str());
        check(usage.used_up() && usage.blocks_in_use == edge_blocks &&
                  usage.listed_slots == edge_blocks &&
                  usage.unlisted_slots == touched - edge_blocks,
              (what + ": the pool used up, the list full, the rest counted as unlisted").c_str());
    }
    check(pool.refused_while_closed() == 0, "edge frame: the closed pool asked for nothing");
}

/// A far depth nearer than some of a frame's points: a new sparse TSDF whose far depth is
/// 0.7005 m, between two whole depth units within the tabletop's depth range [?: a choice], does
/// not walk the frame's valid points deeper than it and counts each, and walks the nearer ones.
/// @param c [in] the case.
/// @param host [in] the frame as rendered.
/// @param frame [in] what the test knows of it.
/// @param points [in] its organized points.
/// @param touched [in] the blocks the frame touches with the case's far depth.
/// @param stream [in] the stream.
void check_depth_skip(const scene_case &c, const gca::frame &host, const known_frame &frame,
                      const gca::organized_points &points, std::uint32_t touched,
                      cudaStream_t stream)
{
    const std::string what = c.name + ", a far depth of 0.7005 m";
    gca::sparse_tsdf_parameters parameters = c.tsdf;
    parameters.max_depth_m = 0.7005f;
    // The depth as the organized points round it to float (organized_points.hpp).
    std::size_t deeper = 0;
    for (std::size_t pixel = 0; pixel < frame.valid.size(); pixel++)
        if (frame.valid[pixel] != 0 &&
            static_cast<float>(host.depth()[pixel]) * c.description.depth_unit_m >
                parameters.max_depth_m)
            deeper++;
    gca::sparse_tsdf tsdf;
    const gca::sparse_tsdf_usage usage =
        allocate_in_new(parameters, points, as_camera_pose(frame.pose), stream, what, tsdf);
    print_usage(what, usage);
    std::printf("%s: %zu valid points deeper\n", what.c_str(), deeper);
    check(deeper > 0 && usage.rays_not_walked == deeper,
          (what + ": the deeper points not walked, each counted").c_str());
    check(usage.blocks_in_use > 0 && usage.blocks_in_use < touched &&
              usage.coordinates_refused == 0,
          (what + ": the nearer points walked, fewer blocks than with every point").c_str());
}

/// Poses far away: one 100 km along x, past the hash's range of 2^20 blocks of 16 mm, 16.8 km
/// [S: computed], whose blocks are all refused, none listed; and one 1e37 m along x, whose
/// segments' ends in blocks pass float's range, 3.4e38 [S: float32], so that no ray is walked and
/// each is counted.
/// @param c [in] the case.
/// @param frame [in] the frame.
/// @param points [in] its organized points.
/// @param stream [in] the stream.
void check_far_poses(const scene_case &c, const known_frame &frame,
                     const gca::organized_points &points, cudaStream_t stream)
{
    const std::size_t valid =
        static_cast<std::size_t>(std::count(frame.valid.begin(), frame.valid.end(), 1));
    std::string what = c.name + ", a pose 100 km away";
    gca::sparse_tsdf tsdf;
    const gca::sparse_tsdf_usage far = allocate_in_new(
        c.tsdf, points, as_camera_pose(Eigen::Translation3d(1e5, 0.0, 0.0) * frame.pose), stream,
        what, tsdf);
    print_usage(what, far);
    check(far.blocks_in_use == 0 && far.blocks_refused == 0 && far.listed_slots == 0 &&
              far.unlisted_slots == 0 && far.coordinates_refused >= valid &&
              far.rays_not_walked == 0,
          (what + ": every block refused, at least one for each valid pixel, none listed").c_str());

    what = c.name + ", a pose 1e37 m away";
    const gca::sparse_tsdf_usage beyond = allocate_in_new(
        c.tsdf, points, as_camera_pose(Eigen::Translation3d(1e37, 0.0, 0.0) * frame.pose), stream,
        what, tsdf);
    print_usage(what, beyond);
    check(beyond.blocks_in_use == 0 && beyond.coordinates_refused == 0 &&
              beyond.listed_slots == 0 && beyond.rays_not_walked == valid,
          (what + ": no ray walked, each valid pixel's counted").c_str());
}

/// Refusals as values, each leaving the counts as they were and CUDA's last error clear; and
/// CUDA's refusal of the launch for points whose rows need more blocks of 8 rows than a grid
/// holds, after which only the list's count is reset, to 0.
/// @param tsdf [in,out] a sparse TSDF that has allocated a frame.
/// @param points [in] that frame's organized points.
/// @param tall [in] organized points of tall_size (stage_frames.hpp), their contents not set.
/// @param pose [in] that frame's pose.
/// @param stream [in] the stream.
void check_refusals(gca::sparse_tsdf &tsdf, const gca::organized_points &points,
                    const gca::organized_points &tall, const Eigen::Isometry3d &pose,
                    cudaStream_t stream)
{
    const std::string what = "refusals";
    const gca::sparse_tsdf_usage before = read_usage(tsdf, stream, what);
    const gca::camera_pose usable = as_camera_pose(pose);

    gca::sparse_tsdf empty;
    check_refused("refusals, an empty sparse TSDF",
                  empty.allocate_near_surface(points, usable, stream), "holds no memory");
    gca::sparse_tsdf moved_to(std::move(tsdf));
    check_refused("refusals, a sparse TSDF moved from",
                  tsdf.allocate_near_surface(points, usable, stream), "holds no memory");
    tsdf = std::move(moved_to);
    const gca::organized_points no_points;
    check_refused("refusals, empty organized points",
                  tsdf.allocate_near_surface(no_points, usable, stream),
                  "the organized points hold no images");

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    const struct
    {
        const char *what;
        gca::camera_pose pose;
    } unusable[] = {
        {"refusals, a position of NaN",
         {Eigen::Vector3d(0.0, nan, 0.0), Eigen::Quaterniond::Identity()}},
        {"refusals, a position past float's range, 1e39 m",
         {Eigen::Vector3d(1e39, 0.0, 0.0), Eigen::Quaterniond::Identity()}},
        {"refusals, an orientation of zero",
         {Eigen::Vector3d::Zero(), Eigen::Quaterniond(0.0, 0.0, 0.0, 0.0)}},
        {"refusals, an orientation with an infinite component",
         {Eigen::Vector3d::Zero(), Eigen::Quaterniond(1.0, infinity, 0.0, 0.0)}},
    };
    for (const auto &u : unusable)
        check_refused(u.what, tsdf.allocate_near_surface(points, u.pose, stream),
                      "the pose cannot be used");

    // An error an earlier call leaves: cudaMalloc's refusal of 2^52 bytes, more than the board
    // has [S: docs/HARDWARE.md, 7607 MB].
    void *impossible = nullptr;
    check(cudaMalloc(&impossible, std::size_t{1} << 52) != cudaSuccess,
          "refusals: cudaMalloc refuses 2^52 bytes");
    check_refused("refusals, an allocation after an earlier error",
                  tsdf.allocate_near_surface(points, usable, stream), "left by an earlier call");
    check_no_cuda_error("refusals, an allocation after an earlier error");

    const gca::sparse_tsdf_usage after = read_usage(tsdf, stream, what);
    print_usage("refusals, after them", after);
    check(same_usage(after, before), "refusals: the counts as they were");

    check_refused("refusals, points whose rows need more blocks than a grid holds",
                  tsdf.allocate_near_surface(tall, usable, stream), "could not launch");
    check_no_cuda_error("refusals, the launch refused");
    gca::sparse_tsdf_usage reset = before;
    reset.listed_slots = 0;
    reset.unlisted_slots = 0;
    const gca::sparse_tsdf_usage after_launch = read_usage(tsdf, stream, what);
    print_usage("refusals, after the launch refused", after_launch);
    check(same_usage(after_launch, reset),
          "refusals, the launch refused: the list's count reset, the other counts as they were");
}

/// The tabletop frames: the first frame's band and list; the same frame again; ten new sparse
/// TSDFs and a quaternion of length 2; the second frame, alone and after the first; a nearer far
/// depth; poses far away; the refusals; and the closed pool.
/// @param stream [in] the stream.
void check_tabletop(cudaStream_t stream)
{
    const scene_case c = tabletop_case();
    const gca_test::rendered frames = gca_test::render(c.scene, c.description);
    if (frames.frames.size() < 2)
        return;
    gca::gpu_pool pool;
    frame_buffers first = request_buffers(pool, c.description.size, c.name);
    frame_buffers second = request_buffers(pool, c.description.size, c.name + ", second frame");
    const gca::organized_points tall =
        granted("tabletop: the tall organized points",
                gca::organized_points::request(pool, gca_test::tall_size));
    gca::sparse_tsdf tsdf =
        granted(c.name + ": the sparse TSDF", gca::sparse_tsdf::request(pool, c.tsdf, stream));
    gca::sparse_tsdf alone = granted(c.name + ", second frame alone: the sparse TSDF",
                                     gca::sparse_tsdf::request(pool, c.tsdf, stream));
    check(tsdf.layout().capacity_blocks == test_blocks,
          "tabletop: the sparse TSDF holds the test's blocks");
    const known_frame frame0 =
        prepare(c, frames.frames[0], frames.surfaces, c.scene.trajectory[0], first, stream);
    const known_frame frame1 =
        prepare(c, frames.frames[1], frames.surfaces, c.scene.trajectory[1], second, stream);
    near_band near0 = blocks_near_band(frame0);
    near_band near1 = blocks_near_band(frame1);
    check(near0.valid_without_hit == 0 && near1.valid_without_hit == 0,
          "tabletop: every valid pixel's ray meets a surface");
    std::printf("tabletop: the largest margin of a ray %.3f mm\n",
                1e3 * std::max(near0.largest_margin_m, near1.largest_margin_m));
    lookup near = make_lookup(pool, std::move(near0.blocks), stream, c.name);
    lookup near_second = make_lookup(pool, std::move(near1.blocks), stream, c.name);
    pool.close();

    allocate(tsdf, first.points, as_camera_pose(frame0.pose), stream, c.name);
    const gca::sparse_tsdf_usage usage = read_usage(tsdf, stream, c.name);
    const std::vector<std::uint32_t> numbers = find(tsdf, near, stream, c.name);
    const block_set allocated = check_band(c.name, frame0, near, numbers, usage);

    // The same frame again: no block, number or count changes, and the list holds the same
    // slots.
    allocate(tsdf, first.points, as_camera_pose(frame0.pose), stream, c.name + ", again");
    const gca::sparse_tsdf_usage again = read_usage(tsdf, stream, c.name + ", again");
    print_usage(c.name + ", again", again);
    check(same_usage(again, usage), "tabletop, again: every count as before");
    check(find(tsdf, near, stream, c.name + ", again") == numbers,
          "tabletop, again: every block number as before");

    check_ten_runs(c, frame0, first.points, near, allocated, usage, stream);

    // The second frame, alone and then after the first: it lists as many slots as it has blocks
    // of its own, and the first's sparse TSDF holds the union of both.
    allocate(alone, second.points, as_camera_pose(frame1.pose), stream,
             c.name + ", second frame alone");
    const gca::sparse_tsdf_usage alone_usage = read_usage(alone, stream, c.name + ", second");
    const block_set allocated_second =
        check_band(c.name + ", second frame alone", frame1, near_second,
                   find(alone, near_second, stream, c.name + ", second"), alone_usage);
    allocate(tsdf, second.points, as_camera_pose(frame1.pose), stream,
             c.name + ", second frame after the first");
    const gca::sparse_tsdf_usage both = read_usage(tsdf, stream, c.name + ", both");
    print_usage(c.name + ", second frame after the first", both);
    block_set union_set = allocated;
    union_set.insert(allocated_second.begin(), allocated_second.end());
    std::printf("tabletop, second frame after the first: %zu blocks in the union\n",
                union_set.size());
    check(both.listed_slots == alone_usage.blocks_in_use && both.unlisted_slots == 0,
          "tabletop, second frame after the first: the list holds the second frame's blocks");
    // Among the blocks near the second frame's band, those of either frame.
    block_set expected_near_second;
    for (const gca::block_coordinate &b : near_second.host)
        if (union_set.count(gca::block_key(b)) != 0)
            expected_near_second.insert(gca::block_key(b));
    check(both.blocks_in_use == union_set.size() &&
              with_block(near_second, find(tsdf, near_second, stream, c.name + ", both")) ==
                  expected_near_second,
          "tabletop, second frame after the first: the blocks of both frames");

    check_depth_skip(c, frames.frames[0], frame0, first.points, usage.blocks_in_use, stream);
    check_far_poses(c, frame0, first.points, stream);
    check_refusals(tsdf, first.points, tall, frame1.pose, stream);
    check(pool.refused_while_closed() == 0, "tabletop: the closed pool asked for nothing");
}

/// The large frame's band and list, and the closed pool.
/// @param stream [in] the stream.
void check_large(cudaStream_t stream)
{
    const scene_case c = large_case();
    const gca_test::rendered frames = gca_test::render(c.scene, c.description);
    if (frames.frames.empty())
        return;
    gca::gpu_pool pool;
    frame_buffers buffers = request_buffers(pool, c.description.size, c.name);
    gca::sparse_tsdf tsdf =
        granted(c.name + ": the sparse TSDF", gca::sparse_tsdf::request(pool, c.tsdf, stream));
    const known_frame frame =
        prepare(c, frames.frames[0], frames.surfaces, c.scene.trajectory[0], buffers, stream);
    near_band band = blocks_near_band(frame);
    check(band.valid_without_hit == 0, "large: every valid pixel's ray meets a surface");
    double farthest_m = 0.0;
    for (const std::optional<gca_test::synthetic_hit> &hit : frame.hits)
        if (hit)
            farthest_m = std::max(farthest_m, hit->s);
    std::printf("large: the largest margin of a ray %.3f mm; the farthest valid depth %.3f m, mu "
                "there %.1f mm\n",
                1e3 * band.largest_margin_m, farthest_m, 1e3 * truncation_m(c.tsdf, farthest_m));
    lookup near = make_lookup(pool, std::move(band.blocks), stream, c.name);
    pool.close();

    allocate(tsdf, buffers.points, as_camera_pose(frame.pose), stream, c.name);
    const gca::sparse_tsdf_usage usage = read_usage(tsdf, stream, c.name);
    check_band(c.name, frame, near, find(tsdf, near, stream, c.name), usage);
    check(pool.refused_while_closed() == 0, "large: the closed pool asked for nothing");
}

/// Bands across more blocks than a ray may walk, in a scene the bound of one frame does not cover
/// (docs/methods/sparse-tsdf.md, "One frame's blocks"): a picture of 8 x 8 pixels [?: a choice: few
/// rays, so that a kernel that walks them anyway ends soon] whose focal lengths of 0.01 px give a
/// field of view far wider than the camera's, a wall facing it 1.0 m away, at the large object's
/// live model and the cases' budget. mu(1.0 m) = 32 mm, so each segment spans 2 mu r = 64 mm r,
/// r = ((u - cx) / fx, (v - cy) / fy, 1): r's first two components are at least 0.5 / 0.01 = 50
/// each, so the segment crosses at least 49 borders of blocks of 64 mm along each, 99 blocks or
/// more [S: computed], more than max_blocks_per_ray: no ray is walked, each is counted, nothing
/// is inserted.
/// @param stream [in] the stream.
void check_long_bands(cudaStream_t stream)
{
    const std::string what = "long bands";
    static_assert(gca::max_blocks_per_ray < 99, "the bands below cross more blocks than the cap");
    const gca::frame_description wide = {
        {8, 8}, {0.01f, 0.01f, 3.5f, 3.5f}, gca_test::camera_depth_unit_m};
    const gca::frame wall = gca_test::built_frame(
        wide, [](std::uint32_t, std::uint32_t) { return std::uint16_t{1000}; }, gca_test::grey);
    gca::organized_points_parameters range{};
    const float voxel_m = live_model_of(gca::object_class::large_object, range);
    const gca::sparse_tsdf_parameters parameters = {test_budget, voxel_m, method_truncation,
                                                    range.far_m};
    gca::gpu_pool pool;
    frame_buffers buffers = request_buffers(pool, wide.size, what);
    pool.close();
    const gca::step_result up = buffers.frame.upload(wall, stream);
    const gca::step_result made = buffers.points.deproject(buffers.frame, range, stream);
    check(up.done && made.done,
          (what + ": upload and deprojection, " + up.message + made.message).c_str());
    gca::sparse_tsdf tsdf;
    const gca::sparse_tsdf_usage usage =
        allocate_in_new(parameters, buffers.points, as_camera_pose(Eigen::Isometry3d::Identity()),
                        stream, what, tsdf);
    print_usage(what, usage);
    check(usage.rays_not_walked == 64 && usage.blocks_in_use == 0 &&
              usage.coordinates_refused == 0 && usage.listed_slots == 0,
          "long bands: no ray walked, each of the 64 counted, nothing inserted");
    check(pool.refused_while_closed() == 0, "long bands: the closed pool asked for nothing");
}
} // namespace

/// Runs the checks of the allocation near a frame's surface and prints the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    cudaStream_t stream = nullptr;
    if (!create_stream(stream))
        return gca_test::check_summary("test_sparse_tsdf_allocation");
    check_tabletop(stream);
    check_large(stream);
    check_long_bands(stream);
    check_object_classes(stream);
    const std::uint32_t edge_blocks = check_edges(stream);
    if (edge_blocks != 0)
        check_edge_frame(edge_blocks, stream);
    check_no_cuda_error("the end");
    check(cudaStreamDestroy(stream) == cudaSuccess, "the CUDA stream destroyed");
    return gca_test::check_summary("test_sparse_tsdf_allocation");
}
