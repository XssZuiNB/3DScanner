// SPDX-License-Identifier: 0BSD
/// @file sparse_tsdf.cu
/// The sparse TSDF's hash of blocks and its pool: the division of the memory budget, the request
/// with its checks of the parameters and of one frame's bound against the hash's room, the
/// clearing of the memory, the kernels that insert and find blocks, the allocation of the
/// blocks near a frame's surface with its traversal of the grid of blocks, and the reading of the
/// usage (see sparse_tsdf.hpp; the method: docs/methods/sparse-tsdf.md).

#include "fusion/sparse_tsdf.hpp"

#include "fusion/block_hash.cuh"
#include "geometry/float3_ops.cuh"
#include "gpu/gpu_messages.hpp"
#include "gpu/kernel_launch.cuh"
#include "types/camera_param.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include <Eigen/Geometry>
#include <cuda/atomic>

namespace gca
{
/// A frame's pose rounded to float, as the allocation's and the integration's kernels apply it:
/// p' = R p + t. Outside the file's anonymous namespace, so that sparse_tsdf.hpp can name it for
/// the allocation that takes a pose checked once (sparse_tsdf::allocate_at()).
struct float_pose
{
    float3 row_x;       ///< the rotation's first row, camera frame to the models' frame
    float3 row_y;       ///< its second row
    float3 row_z;       ///< its third row
    float3 translation; ///< t, the camera's centre in the models' frame, in m
};

namespace
{
/// The bytes each block of the pool takes: its 512 voxels of 8 bytes, 4096, and its entry on
/// the frame's list of slots, 4 [S: computed].
constexpr std::uint64_t block_bytes = voxels_per_block * sizeof(voxel) + sizeof(std::uint32_t);

/// The bytes each slot of the hash takes: its key, 8, its block's number, 4, and its stamp of the
/// last frame, 4 [S: computed].
constexpr std::uint64_t slot_bytes = sizeof(unsigned long long) + 2 * sizeof(std::uint32_t);

/// The most blocks a pool holds, 2^29, so that the slots, at most 2^31, fit 32 bits.
constexpr std::uint64_t max_blocks = std::uint64_t{1} << 29;

/// The picture of the bound's model of a frame: the camera's larger stream size, 1280 x 800 px,
/// with its focal lengths there, 612.37 and 612.29 px [M 2026-11-13: docs/HARDWARE.md, "What the
/// camera is and reports"]; 640x400 sees the same field within 2e-5 [S: computed, 640 / 306.18
/// and 400 / 306.14 px].
constexpr double model_width_px = 1280.0;
constexpr double model_height_px = 800.0;
constexpr double model_fx_px = 612.37;
constexpr double model_fy_px = 612.29;

/// The kernels' block: 256 threads [?: a choice; no other size was measured].
constexpr unsigned int threads_per_block = 256;

/// Why an empty sparse TSDF refuses every operation.
constexpr const char *holds_no_memory =
    "the sparse TSDF holds no memory: it was not granted by the pool, or moved from";

/// Why a pose is refused (float_pose_of()).
constexpr const char *unusable_pose = "the pose cannot be used: its position and orientation "
                                      "rounded to float must be finite and its orientation not "
                                      "zero";

static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t), "a key is one 64-bit word");

/// Gives the slots of the hash for a number of blocks: four times as many, so that the room once
/// the pool is used up holds one frame of the object classes' final models
/// (docs/methods/sparse-tsdf.md, "One frame's blocks").
/// @param blocks [in] the blocks, from 1 to max_blocks.
/// @return the least power of two >= 4 blocks.
std::uint64_t slots_for(std::uint64_t blocks) noexcept
{
    return std::bit_ceil(4 * blocks);
}

/// Gives the device memory of a sparse TSDF of a number of blocks.
/// @param blocks [in] the blocks, from 1 to max_blocks.
/// @return the pool's, the hash's, the list's and the counts' bytes together.
std::uint64_t bytes_for(std::uint64_t blocks) noexcept
{
    return blocks * block_bytes + slots_for(blocks) * slot_bytes + sizeof(block_counts);
}

/// Gives the grid of a launch with one thread for each coordinate.
/// @param count [in] the coordinates, > 0.
/// @return ceil(count / threads_per_block) blocks (blocks_for(), gpu/kernel_launch.cuh); for a
///         count past 2^32 - 1, which no device memory of this board holds, the largest unsigned
///         int, a grid CUDA refuses [S: CUDA C++ Programming Guide, "Compute Capabilities", the
///         grid's x dimension, at most 2^31 - 1], so that it comes back as CUDA's refusal.
dim3 grid_for_count(std::size_t count) noexcept
{
    if (count > std::numeric_limits<std::uint32_t>::max())
        return dim3(std::numeric_limits<unsigned int>::max());
    return dim3(blocks_for(static_cast<std::uint32_t>(count), threads_per_block));
}

/// Inserts every coordinate into the hash, one thread for each, by insert_block()
/// (fusion/block_hash.cuh).
/// @param hash [in] the hash, whose arrays and counts the insertions write; device memory owned
///        by the sparse TSDF, which holds it until the kernel has run.
/// @param coordinates [in] device memory of count coordinates, in blocks; owned by the caller,
///        which holds it until the kernel has run (sparse_tsdf::insert_blocks()).
/// @param count [in] the coordinates, > 0.
/// @note Runs on the stream of its launch, sparse_tsdf::insert_blocks()'s. A thread outside the
///       coordinates does nothing. The trap of insert_block(): a key is visible to the other
///       threads of this launch before its block's number, so no thread of it reads a number;
///       the lookups run in a later kernel. Allocates nothing.
__global__ void __launch_bounds__(threads_per_block)
    insert_kernel(block_hash_view hash, const block_coordinate *__restrict__ coordinates,
                  std::size_t count)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    insert_block(hash, coordinates[i]);
}

/// Finds every coordinate's block, one thread for each, by find_block()
/// (fusion/block_hash.cuh).
/// @param hash [in] the hash, read only; device memory owned by the sparse TSDF, which holds it
///        until the kernel has run.
/// @param coordinates [in] device memory of count coordinates, in blocks; owned by the caller,
///        which holds it until the kernel has run (sparse_tsdf::find_blocks()).
/// @param count [in] the coordinates, > 0.
/// @param found [out] device memory of count block numbers: each coordinate's, or no_block;
///        owned by the caller, as coordinates.
/// @note Runs on the stream of its launch, sparse_tsdf::find_blocks()'s, after the insertions it
///       is to see. A thread outside the coordinates writes nothing. Allocates nothing.
__global__ void __launch_bounds__(threads_per_block)
    find_kernel(block_hash_view hash, const block_coordinate *__restrict__ coordinates,
                std::size_t count, std::uint32_t *__restrict__ found)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    found[i] = find_block(hash, coordinates[i]);
}

/// The truncation distance's rule as the kernels apply it, in m: at the measured depth z,
/// mu(z) = max(mu_min v, k c z^2) (docs/methods/sparse-tsdf.md, "The truncation distance").
struct truncation_rule
{
    float mu_min_m;     ///< mu_min v, the least truncation distance, in m, > 0
    float noise_factor; ///< k c, in 1/m, > 0
};

/// Gives the truncation's rule of a sparse TSDF's parameters, in float.
/// @param parameters [in] the parameters, each in its range.
/// @return mu_min v and k c.
truncation_rule rule_of(const sparse_tsdf_parameters &parameters) noexcept
{
    const truncation_parameters &t = parameters.truncation;
    return {t.min_voxels * parameters.voxel_size_m, t.noise_multiple * t.noise_per_m};
}

/// Gives the truncation distance at a measured depth, the one copy the allocation and the
/// integration apply.
/// @param rule [in] the rule.
/// @param z_m [in] the measured depth, in m, > 0.
/// @return mu(z) = max(mu_min v, k c z^2), in m.
/// @note Device code; allocates nothing.
__device__ inline float truncation_at(const truncation_rule &rule, float z_m)
{
    return fmaxf(rule.mu_min_m, rule.noise_factor * z_m * z_m);
}

/// Rounds a frame's pose to float, its orientation normalized.
/// @param pose [in] the pose, camera frame to the models' frame.
/// @return the rotation and translation; nothing when the orientation is zero, or when a value,
///         rounded to float, is not finite.
std::optional<float_pose> float_pose_of(const camera_pose &pose)
{
    const double norm = pose.orientation.norm();
    if (!(norm > 0.0) || !std::isfinite(norm))
        return std::nullopt;
    const Eigen::Matrix3f r =
        Eigen::Quaterniond(pose.orientation.coeffs() / norm).toRotationMatrix().cast<float>();
    const Eigen::Vector3f t = pose.position_m.cast<float>();
    if (!r.allFinite() || !t.allFinite())
        return std::nullopt;
    return float_pose{make_float3(r(0, 0), r(0, 1), r(0, 2)),
                      make_float3(r(1, 0), r(1, 1), r(1, 2)),
                      make_float3(r(2, 0), r(2, 1), r(2, 2)), make_float3(t.x(), t.y(), t.z())};
}

/// Carries a point of the camera frame into the models' frame and scales it to blocks.
/// @param pose [in] the pose.
/// @param p [in] the point, in the camera frame, in m.
/// @param blocks_per_m [in] 1 / (8 v), the blocks in a metre.
/// @return (R p + t) / (8 v), in blocks of the models' frame.
/// @note Device code; allocates nothing.
__device__ inline float3 in_blocks(const float_pose &pose, const float3 &p, float blocks_per_m)
{
    return make_float3(
        (pose.row_x.x * p.x + pose.row_x.y * p.y + pose.row_x.z * p.z + pose.translation.x) *
            blocks_per_m,
        (pose.row_y.x * p.x + pose.row_y.y * p.y + pose.row_y.z * p.z + pose.translation.y) *
            blocks_per_m,
        (pose.row_z.x * p.x + pose.row_z.y * p.y + pose.row_z.z * p.z + pose.translation.z) *
            blocks_per_m);
}

/// One axis of the traversal of the grid of blocks along a segment (Amanatides and Woo 1987): the
/// block's coordinate on the axis, its step toward the segment's end, the steps left to the end's
/// block, and the segment's parameter t in [0, 1] at the next border of blocks and between two.
struct axis_walk
{
    int cell;                ///< the current block's coordinate on this axis
    int step;                ///< +1 or -1, toward the end's block
    std::uint32_t remaining; ///< the steps left on this axis to the end's block
    float t_next;            ///< t at the next border on this axis, >= 0, never NaN; infinite
                             ///< once no step is left
    float t_delta;           ///< t between two borders on this axis; infinite when none is crossed
};

/// The least and the greatest coordinate a segment's end is clamped to, in blocks: one block past
/// the hash's range on each side, so that a clamped end outside the range still lies in a block
/// the hash refuses, and every clamped end converts to an int [S: -2^20 - 1 and 2^20 are floats
/// exactly, below 2^24].
constexpr float lowest_end = static_cast<float>(block_coordinate_min) - 1.0f;
constexpr float highest_end = static_cast<float>(block_coordinate_max) + 1.0f;

/// Clamps a segment's end to one block past the hash's range on each side.
/// @param end [in] the end, in blocks of the models' frame, finite.
/// @return each coordinate clamped to [lowest_end, highest_end].
/// @note Device code; allocates nothing.
__device__ inline float3 clamped(const float3 &end)
{
    return make_float3(fminf(fmaxf(end.x, lowest_end), highest_end),
                       fminf(fmaxf(end.y, lowest_end), highest_end),
                       fminf(fmaxf(end.z, lowest_end), highest_end));
}

/// Says whether a segment's end is finite.
/// @param end [in] the end, in blocks.
/// @return true when no coordinate is infinite or NaN.
/// @note Device code; allocates nothing.
__device__ inline bool finite_end(const float3 &end)
{
    return isfinite(end.x) && isfinite(end.y) && isfinite(end.z);
}

/// Starts one axis of the traversal.
/// @param from [in] the segment's start on this axis, in blocks, clamped (clamped()).
/// @param to [in] its end on this axis, in blocks, clamped.
/// @return the axis at the start's block, with |floor(to) - floor(from)| steps left, at most
///         2^21 + 1.
/// @note Device code; allocates nothing.
__device__ inline axis_walk start_axis(float from, float to)
{
    const int cell = __float2int_rd(from);
    const int last = __float2int_rd(to);
    const long long difference = static_cast<long long>(last) - cell;
    axis_walk axis{cell, difference >= 0 ? 1 : -1,
                   static_cast<std::uint32_t>(difference >= 0 ? difference : -difference), INFINITY,
                   INFINITY};
    if (axis.remaining == 0)
        return axis;
    // floor(from) <= from < floor(from) + 1; from and to differ, so the length is > 0. A length
    // below float's range makes t_delta infinite, and fmaxf turns 0 * infinity, a NaN, into 0.
    axis.t_delta = 1.0f / fabsf(to - from);
    const float to_border =
        axis.step > 0 ? static_cast<float>(cell) + 1.0f - from : from - static_cast<float>(cell);
    axis.t_next = fmaxf(to_border * axis.t_delta, 0.0f);
    return axis;
}

/// Steps one axis of the traversal into the next block.
/// @param axis [in,out] the axis, with a step left.
/// @note Device code; allocates nothing.
__device__ inline void advance(axis_walk &axis)
{
    axis.cell += axis.step;
    axis.remaining--;
    axis.t_next = axis.remaining == 0 ? INFINITY : axis.t_next + axis.t_delta;
}

/// Inserts a block the frame's band passes through and lists its slot once for the frame: the
/// thread whose exchange of the slot's stamp changes it appends the slot to the frame's list,
/// which holds the first capacity_blocks; every append is counted (docs/methods/sparse-tsdf.md,
/// "The allocation near a frame's surface"). The stamp is read first by a relaxed atomic load: one
/// that holds this frame already was exchanged by the thread that appended the slot, so the many
/// rays that reach a slot after the first take no exchange (the times with and without:
/// docs/HARDWARE.md, "The per-frame path's stages"). The keys' trap of insert_block(): another
/// thread of this launch may have put the key there and not yet its block's number; this function
/// reads no number, and the slots it lists are read for their numbers only by work queued after
/// the allocation.
/// @param hash [in] the hash; its slots and counts are written (insert_block()).
/// @param stamps [in,out] device memory of slot_count stamps, each the last frame that touched
///        its slot; owned by the sparse TSDF.
/// @param frame_slots [out] device memory of capacity_blocks slots, the frame's list; owned by the
///        sparse TSDF.
/// @param frame [in] this frame's stamp, never every bit 1.
/// @param b [in] the block's coordinates.
/// @note Device code; allocates nothing.
__device__ inline void touch_block(const block_hash_view &hash, std::uint32_t *stamps,
                                   std::uint32_t *frame_slots, std::uint32_t frame,
                                   const block_coordinate &b)
{
    const std::uint32_t slot = insert_block(hash, b);
    if (slot == no_slot)
        return;
    if (cuda::atomic_ref<std::uint32_t, cuda::thread_scope_device>(stamps[slot])
            .load(cuda::memory_order_relaxed) == frame)
        return;
    if (atomicExch(&stamps[slot], frame) == frame)
        return;
    const std::uint32_t entry = atomicAdd(&hash.counts->listed_slots, 1u);
    if (entry < hash.capacity_blocks)
        frame_slots[entry] = slot;
}

/// Allocates the blocks near a frame's surface, one thread for each pixel: a valid pixel's point
/// p at depth z = p.z lies on its ray, so the ray's segment between the depths z - mu(z) and
/// z + mu(z) runs from p (z - mu) / z to p (z + mu) / z; carried into the models' frame and
/// scaled to blocks, the grid traversal of Amanatides and Woo visits every block it passes
/// through, in order, from the start's block to the end's, each step crossing one border
/// (docs/methods/sparse-tsdf.md, "The allocation near a frame's surface"). A point deeper than the
/// far depth, or a segment with an end that is not finite, or one that crosses more than
/// max_blocks_per_ray blocks once its ends are clamped, is not walked and is counted in
/// rays_not_walked.
/// @param points [in] device memory of size's points, rows dense, in the camera frame, in m; owned
///        by the caller's organized points, which hold them until the kernel has run.
/// @param valid [in] device memory of size's marks, 1 for a valid pixel; owned as points.
/// @param size [in] the picture's width and height, in px.
/// @param pose [in] the frame's pose, camera frame to the models' frame.
/// @param max_depth_m [in] z_max, the far depth, in m along the camera's z, > 0: a deeper point is
///        not walked, so that the frame stays within one frame's bound, which the request checked
///        (docs/methods/sparse-tsdf.md, "One frame's blocks").
/// @param blocks_per_m [in] 1 / (8 v), the blocks in a metre.
/// @param truncation [in] the truncation's rule, in m.
/// @param hash [in] the hash; its slots and counts are written.
/// @param stamps [in,out] device memory of slot_count stamps; owned by the sparse TSDF.
/// @param frame_slots [out] device memory of capacity_blocks slots, the frame's list; owned by the
///        sparse TSDF.
/// @param frame [in] this frame's stamp, never every bit 1.
/// @note Runs on the stream of its launch, sparse_tsdf::allocate_near_surface()'s, after the reset
///       of the list's count. A thread outside the picture or at an invalid pixel does nothing.
///       A thread takes at most max_blocks_per_ray steps, each an insertion (insert_block()).
///       The keys' trap of insert_block(): within this launch a key is visible before its
///       block's number, so no thread of it reads a number (touch_block()). Allocates nothing.
__global__ void __launch_bounds__(picture_block_width *picture_block_height)
    allocate_kernel(const float3 *__restrict__ points, const std::uint8_t *__restrict__ valid,
                    frame_size size, float_pose pose, float max_depth_m, float blocks_per_m,
                    truncation_rule truncation, block_hash_view hash,
                    std::uint32_t *__restrict__ stamps, std::uint32_t *__restrict__ frame_slots,
                    std::uint32_t frame)
{
    const std::uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= size.width || v >= size.height)
        return;
    const std::size_t pixel = static_cast<std::size_t>(v) * size.width + u;
    if (valid[pixel] == 0)
        return;
    const float3 p = points[pixel];
    const float z = p.z;
    if (z > max_depth_m)
    {
        atomicAdd(&hash.counts->rays_not_walked, 1u);
        return;
    }
    const float mu = truncation_at(truncation, z);
    const float near_scale = (z - mu) / z;
    const float far_scale = (z + mu) / z;
    const float3 near_end = in_blocks(
        pose, make_float3(p.x * near_scale, p.y * near_scale, p.z * near_scale), blocks_per_m);
    const float3 far_end = in_blocks(
        pose, make_float3(p.x * far_scale, p.y * far_scale, p.z * far_scale), blocks_per_m);
    if (!finite_end(near_end) || !finite_end(far_end))
    {
        atomicAdd(&hash.counts->rays_not_walked, 1u);
        return;
    }
    const float3 from = clamped(near_end);
    const float3 to = clamped(far_end);
    axis_walk along_x = start_axis(from.x, to.x);
    axis_walk along_y = start_axis(from.y, to.y);
    axis_walk along_z = start_axis(from.z, to.z);
    // Each axis has at most 2^21 + 1 steps, so the sum fits 32 bits.
    const std::uint32_t blocks = 1 + along_x.remaining + along_y.remaining + along_z.remaining;
    if (blocks > max_blocks_per_ray)
    {
        atomicAdd(&hash.counts->rays_not_walked, 1u);
        return;
    }
    touch_block(hash, stamps, frame_slots, frame,
                block_coordinate{along_x.cell, along_y.cell, along_z.cell});
    for (std::uint32_t visited = 1; visited < blocks; visited++)
    {
        // The axis whose border comes first. An axis without a step left has t_next infinite and,
        // the ends finite, no t_next is NaN (start_axis(), advance()), so when neither x nor y is
        // taken, z has a step left. The loop's count bounds the steps whatever the choice.
        if (along_x.remaining != 0 && along_x.t_next <= along_y.t_next &&
            along_x.t_next <= along_z.t_next)
            advance(along_x);
        else if (along_y.remaining != 0 && along_y.t_next <= along_z.t_next)
            advance(along_y);
        else
            advance(along_z);
        touch_block(hash, stamps, frame_slots, frame,
                    block_coordinate{along_x.cell, along_y.cell, along_z.cell});
    }
}

/// The thread blocks of the two kernels that go through blocks of voxels, the integration's over
/// the frame's list and the extraction's over the hash's slots: a fixed grid, each thread block
/// taking every grid's-th entry, so that the integration reads the list's count on the GPU and the
/// per-frame path reads nothing back. 24, as many thread blocks of 512 threads as the GPU holds at
/// once: 8 multiprocessors of 1536 threads each [M 2026-11-15: cudaGetDeviceProperties], the
/// kernels at 34 and 37 registers a thread [B 2026-11-15: cuobjdump -res-usage], 3 x 512 x 37
/// below the 65536 registers of a multiprocessor. Back to back, integrate() took 0.4 % longer to
/// 8.2 % shorter than with 96 and 7.6 % to 13.7 % shorter than with 384 (docs/HARDWARE.md, "The
/// integration of a frame").
constexpr unsigned int voxel_kernel_grid = 24;

static_assert(voxels_per_block == 512, "a kernel over blocks of voxels has a thread per voxel");

/// Gives a voxel's element in the pool, the one layout of a block's voxels: voxel (i, j, k) of
/// block n is element 512 n + i + 8 j + 64 k (docs/methods/sparse-tsdf.md, "Space, voxels and
/// blocks").
/// @param number [in] the block's number, below the pool's blocks.
/// @param local [in] (i, j, k), each 0 to 7.
/// @return the element's index in the pool.
/// @note Device code; allocates nothing.
__device__ inline std::size_t pool_index(std::uint32_t number, const int3 &local)
{
    return static_cast<std::size_t>(number) * voxels_per_block + local.x + 8 * local.y +
           64 * local.z;
}

/// Gives the voxel a thread of a kernel over blocks of voxels takes: thread t takes (i, j, k) with
/// t = i + 8 j + 64 k, so that pool_index() of a block's voxel is 512 n + t and a warp's threads
/// read and write neighbouring elements.
/// @param t [in] the thread's index in its block, below voxels_per_block.
/// @return (i, j, k), each 0 to 7.
/// @note Device code; allocates nothing.
__device__ inline int3 voxel_in_block(unsigned int t)
{
    return make_int3(static_cast<int>(t & 7u), static_cast<int>((t >> 3) & 7u),
                     static_cast<int>(t >> 6));
}

/// Gives a voxel's coordinates from its block's and its place in the block: g = 8 b + (i, j, k).
/// @param b [in] the block's coordinates, each in the hash's range, so that 8 b fits an int.
/// @param local [in] (i, j, k), each 0 to 7.
/// @return g, in voxels of the models' frame.
/// @note Device code; allocates nothing.
__device__ inline int3 voxel_of(const block_coordinate &b, const int3 &local)
{
    return make_int3(8 * b.x + local.x, 8 * b.y + local.y, 8 * b.z + local.z);
}

/// Gives a point on the grid of voxels: (g + 1/2 + offset) v, the centre of voxel g moved by an
/// offset in voxels.
/// @param g [in] the voxel, in voxels of the models' frame, each below 2^23 in magnitude, so that
///        it is a float exactly.
/// @param offset [in] the offset from its centre, in voxels.
/// @param voxel_size_m [in] v, in m, > 0.
/// @return the point, in the models' frame, in m.
/// @note Device code; allocates nothing.
__device__ inline float3 on_grid(const int3 &g, const float3 &offset, float voxel_size_m)
{
    return make_float3((static_cast<float>(g.x) + 0.5f + offset.x) * voxel_size_m,
                       (static_cast<float>(g.y) + 0.5f + offset.y) * voxel_size_m,
                       (static_cast<float>(g.z) + 0.5f + offset.z) * voxel_size_m);
}

/// Carries a point of the models' frame into the camera frame of a pose: q = R^T (p - t).
/// @param pose [in] the pose, camera frame to the models' frame.
/// @param p [in] the point, in the models' frame, in m.
/// @return q, in the camera frame, in m.
/// @note Device code; allocates nothing.
__device__ inline float3 into_camera(const float_pose &pose, const float3 &p)
{
    // R^T d is the sum of R's rows, each times d's component of its index.
    const float3 d = p - pose.translation;
    return d.x * pose.row_x + d.y * pose.row_y + d.z * pose.row_z;
}

/// A frame as the integration observes it: its points, marks and colors on the GPU, with its size
/// and intrinsics.
struct observed_frame
{
    const float3 *points;      ///< size's points, rows dense, in the camera frame, in m
    const std::uint8_t *valid; ///< size's marks, 1 for a valid pixel
    const uchar3 *color;       ///< size's colors, RGB8, rows dense
    frame_size size;           ///< the picture's width and height, in px
    intrinsics k;              ///< the intrinsics, in px, usable (usable_intrinsics())
};

/// Rounds a weighted mean of a color's channel: (w held + seen) / next, to the nearest integer,
/// a tie to the even one.
/// @param held [in] the voxel's channel, 0 to 255.
/// @param seen [in] the observation's channel, 0 to 255.
/// @param w [in] the voxel's weight W, >= 0.
/// @param next [in] W + 1.
/// @return the mean, 0 to 255: a mean of two values in that range.
/// @note Device code; allocates nothing.
__device__ inline std::uint8_t mean_channel(std::uint8_t held, unsigned char seen, float w,
                                            float next)
{
    return static_cast<std::uint8_t>(
        __float2uint_rn((w * static_cast<float>(held) + static_cast<float>(seen)) / next));
}

/// Joins one observation to a voxel's running averages with the weight 1
/// (docs/methods/sparse-tsdf.md, "Integration, and the weight of an observation"):
/// D <- (W D + d) / (W + 1), C <- (W C + c) / (W + 1), W <- min(W + 1, max_voxel_weight); D
/// rounded to the nearest half float, each of C's channels to the nearest integer, ties to even.
/// A voxel not yet observed holds 0 in every byte since the request, so its first observation
/// gives d and c themselves.
/// @param cell [in,out] the voxel, read and written in one 64-bit access each.
/// @param d [in] the observed distance, in voxels, |d| <= 20000 (sparse_tsdf::integrate()).
/// @param c [in] the observed color, RGB8.
/// @note Device code; allocates nothing. The caller's thread is the one that updates this voxel.
__device__ inline void observe(voxel &cell, float d, const uchar3 &c)
{
    const voxel held = cell;
    const float w = static_cast<float>(held.weight);
    const float next = w + 1.0f;
    const int weight = held.weight + 1;
    cell = voxel{__float2half_rn((w * __half2float(held.distance) + d) / next),
                 static_cast<std::uint16_t>(weight < max_voxel_weight ? weight : max_voxel_weight),
                 mean_channel(held.red, c.x, w, next), mean_channel(held.green, c.y, w, next),
                 mean_channel(held.blue, c.z, w, next)};
}

/// Integrates a frame into the voxels of the blocks on its list (docs/methods/sparse-tsdf.md,
/// "Integration, and the weight of an observation"): one thread for each voxel, one thread block
/// for each block, the fixed grid going through the list. Voxel g's centre is carried into the
/// camera frame and projected to its nearest pixel; a valid pixel no deeper than the far depth,
/// with depth z, observes it when eta = z - q_z >= -mu(z), with d = min(eta, mu(z)) / v and the
/// pixel's color (observe()). A list entry whose slot has no block is skipped.
/// @param hash [in] the hash, read only, and its counts, whose listed_slots the allocation queued
///        before this kernel set: the list holds the first capacity_blocks of them; device memory
///        owned by the sparse TSDF, which holds it until the kernel has run.
/// @param frame_slots [in] the frame's list of slots, capacity_blocks entries; owned as the hash.
/// @param voxels [in,out] the pool, capacity_blocks blocks of voxels_per_block voxels; owned as
///        the hash.
/// @param frame [in] the frame's points, marks, colors, size and intrinsics.
/// @param pose [in] the frame's pose, camera frame to the models' frame.
/// @param voxel_size_m [in] v, in m, > 0.
/// @param max_depth_m [in] z_max, in m: a deeper pixel observes nothing, so that |d| stays within
///        mu(z_max) / v (sparse_tsdf::integrate()).
/// @param truncation [in] the truncation's rule, in m.
/// @note Runs on the stream of its launch, sparse_tsdf::integrate()'s, after the allocation. Each
///       listed slot is on the list once, so each voxel is updated by one thread. Allocates
///       nothing.
__global__ void __launch_bounds__(voxels_per_block)
    integrate_kernel(block_hash_view hash, const std::uint32_t *__restrict__ frame_slots,
                     voxel *__restrict__ voxels, observed_frame frame, float_pose pose,
                     float voxel_size_m, float max_depth_m, truncation_rule truncation)
{
    const std::uint32_t listed = min(hash.counts->listed_slots, hash.capacity_blocks);
    const int3 local = voxel_in_block(threadIdx.x);
    const float width = static_cast<float>(frame.size.width);
    const float height = static_cast<float>(frame.size.height);
    for (std::uint32_t entry = blockIdx.x; entry < listed; entry += gridDim.x)
    {
        const held_block held = block_in_slot(hash, frame_slots[entry]);
        if (held.number == no_block)
            continue;
        const float3 q = into_camera(pose, on_grid(voxel_of(held.coordinate, local),
                                                   make_float3(0.0f, 0.0f, 0.0f), voxel_size_m));
        if (!(q.z > 0.0f))
            continue;
        const float column = frame.k.fx * q.x / q.z + frame.k.cx;
        const float row = frame.k.fy * q.y / q.z + frame.k.cy;
        // Outside (-1, width) x (-1, height) the nearest pixel lies outside the picture; inside,
        // the conversion to int is exact. A NaN fails the comparison.
        if (!(column > -1.0f && column < width && row > -1.0f && row < height))
            continue;
        const int u = __float2int_rn(column);
        const int v = __float2int_rn(row);
        if (u < 0 || v < 0 || u >= static_cast<int>(frame.size.width) ||
            v >= static_cast<int>(frame.size.height))
            continue;
        const std::size_t pixel = static_cast<std::size_t>(v) * frame.size.width + u;
        if (frame.valid[pixel] == 0)
            continue;
        const float z = frame.points[pixel].z;
        if (z > max_depth_m)
            continue;
        const float mu = truncation_at(truncation, z);
        const float eta = z - q.z;
        if (eta < -mu)
            continue;
        observe(voxels[pool_index(held.number, local)], fminf(eta, mu) / voxel_size_m,
                frame.color[pixel]);
    }
}

/// The sparse TSDF as the extraction reads it: the hash's one view and the pool's voxels.
struct voxel_lookup
{
    block_hash_view hash; ///< the hash, read only
    const voxel *voxels;  ///< the pool's voxels
};

/// Finds a voxel by its coordinates: in the block at hand without the hash, in another block
/// through it (find_block()).
/// @param tsdf [in] the hash and the voxels.
/// @param g [in] the voxel, in voxels of the models' frame.
/// @param own [in] the block at hand, with a number.
/// @return the voxel; nullptr when its block has no number.
/// @note Device code; allocates nothing. Arithmetic shifts and masks: g >> 3 is floor(g / 8) and
///       g & 7 its remainder for a negative g too [S: C++20, two's complement].
__device__ inline const voxel *voxel_at(const voxel_lookup &tsdf, const int3 &g,
                                        const held_block &own)
{
    const block_coordinate b{g.x >> 3, g.y >> 3, g.z >> 3};
    const bool at_hand =
        b.x == own.coordinate.x && b.y == own.coordinate.y && b.z == own.coordinate.z;
    const std::uint32_t number = at_hand ? own.number : find_block(tsdf.hash, b);
    if (number == no_block)
        return nullptr;
    return &tsdf.voxels[pool_index(number, make_int3(g.x & 7, g.y & 7, g.z & 7))];
}

/// Reads an observed voxel's distance.
/// @param tsdf [in] the hash and the voxels.
/// @param g [in] the voxel, in voxels of the models' frame.
/// @param own [in] the block at hand (voxel_at()).
/// @param distance [out] its distance, in voxels, when observed.
/// @return whether the voxel exists and is observed.
/// @note Device code; allocates nothing.
__device__ inline bool observed_distance(const voxel_lookup &tsdf, const int3 &g,
                                         const held_block &own, float &distance)
{
    const voxel *cell = voxel_at(tsdf, g, own);
    if (cell == nullptr || cell->weight == 0)
        return false;
    distance = __half2float(cell->distance);
    return true;
}

/// Gives the gradient of the distance at a voxel by central differences of one voxel along each
/// axis.
/// @param tsdf [in] the hash and the voxels.
/// @param g [in] the voxel, in voxels of the models' frame.
/// @param own [in] the block at hand (voxel_at()).
/// @param gradient [out] ((D(g + x) - D(g - x)) / 2, ...), in voxels per voxel, when every one of
///        the six neighbours is observed.
/// @return whether every one of the six neighbours is observed.
/// @note Device code; allocates nothing.
__device__ inline bool gradient_at(const voxel_lookup &tsdf, const int3 &g, const held_block &own,
                                   float3 &gradient)
{
    float x_after, x_before, y_after, y_before, z_after, z_before;
    if (!observed_distance(tsdf, make_int3(g.x + 1, g.y, g.z), own, x_after) ||
        !observed_distance(tsdf, make_int3(g.x - 1, g.y, g.z), own, x_before) ||
        !observed_distance(tsdf, make_int3(g.x, g.y + 1, g.z), own, y_after) ||
        !observed_distance(tsdf, make_int3(g.x, g.y - 1, g.z), own, y_before) ||
        !observed_distance(tsdf, make_int3(g.x, g.y, g.z + 1), own, z_after) ||
        !observed_distance(tsdf, make_int3(g.x, g.y, g.z - 1), own, z_before))
        return false;
    gradient = make_float3(0.5f * (x_after - x_before), 0.5f * (y_after - y_before),
                           0.5f * (z_after - z_before));
    return true;
}

/// Interpolates a color's channel between two voxels and rounds it: (1 - s) a + s b, to the nearest
/// integer, a tie to the even one.
/// @param a [in] the first voxel's channel, 0 to 255.
/// @param b [in] the second voxel's channel, 0 to 255.
/// @param s [in] the fraction of the way from the first to the second, in [0, 1].
/// @return the channel, 0 to 255.
/// @note Device code; allocates nothing.
__device__ inline unsigned char between_channels(std::uint8_t a, std::uint8_t b, float s)
{
    return static_cast<unsigned char>(
        __float2uint_rn((1.0f - s) * static_cast<float>(a) + s * static_cast<float>(b)));
}

/// Extracts the surface's points (docs/methods/sparse-tsdf.md, "The extraction"): one thread for
/// each voxel, one thread block for each slot, the fixed grid going through every slot of the
/// hash. An observed voxel g and each observed neighbour g + e, e along x, y and z, whose
/// distances differ in sign give a point at s = D_a / (D_a - D_b) of the way from g's centre to
/// the neighbour's, with the gradients by central differences at both interpolated at s and
/// normalized as its normal and the colors interpolated at s as its color; a crossing whose
/// central differences need a voxel that is missing or unobserved, or whose interpolated gradient
/// is zero, gives no point. Every point is counted; the first capacity are written.
/// @param tsdf [in] the hash and the voxels; device memory owned by the sparse TSDF, which holds it
///        until the kernel has run.
/// @param voxel_size_m [in] v, in m, > 0.
/// @param out [out] device memory of capacity points; owned by the caller's buffer, which holds it
///        until the kernel has run.
/// @param capacity [in] the points out holds.
/// @param found [in,out] device memory of one count, 0 before the launch, which ends as the points
///        found: at most 3 * 512 * capacity_blocks, below 2^32 for every pool of fewer than
///        2796203 blocks, 11.4 GB, more than the board's memory [S: computed].
/// @note Runs on the stream of its launch, sparse_tsdf::extract()'s, after the integrations.
///       Reads the voxels and writes only out and found. Allocates nothing.
__global__ void __launch_bounds__(voxels_per_block)
    extract_kernel(voxel_lookup tsdf, float voxel_size_m, surface_point *__restrict__ out,
                   std::uint32_t capacity, std::uint32_t *__restrict__ found)
{
    const int3 local = voxel_in_block(threadIdx.x);
    for (std::uint32_t slot = blockIdx.x; slot < tsdf.hash.slot_count; slot += gridDim.x)
    {
        const held_block held = block_in_slot(tsdf.hash, slot);
        if (held.number == no_block)
            continue;
        const voxel a = tsdf.voxels[pool_index(held.number, local)];
        if (a.weight == 0)
            continue;
        const int3 g = voxel_of(held.coordinate, local);
        const float distance_a = __half2float(a.distance);
        for (int axis = 0; axis < 3; axis++)
        {
            const int3 e = make_int3(axis == 0, axis == 1, axis == 2);
            const int3 h = make_int3(g.x + e.x, g.y + e.y, g.z + e.z);
            const voxel *b = voxel_at(tsdf, h, held);
            if (b == nullptr || b->weight == 0)
                continue;
            const float distance_b = __half2float(b->distance);
            // D_a >= 0 > D_b, or D_a < 0 <= D_b.
            if ((distance_a >= 0.0f) == (distance_b >= 0.0f))
                continue;
            float3 gradient_a;
            float3 gradient_b;
            if (!gradient_at(tsdf, g, held, gradient_a) || !gradient_at(tsdf, h, held, gradient_b))
                continue;
            const float s = distance_a / (distance_a - distance_b);
            const float3 gradient = (1.0f - s) * gradient_a + s * gradient_b;
            const float length = norm(gradient);
            if (!(length > 0.0f))
                continue;
            const surface_point point{
                on_grid(g,
                        make_float3(s * static_cast<float>(e.x), s * static_cast<float>(e.y),
                                    s * static_cast<float>(e.z)),
                        voxel_size_m),
                gradient / length,
                make_uchar3(between_channels(a.red, b->red, s),
                            between_channels(a.green, b->green, s),
                            between_channels(a.blue, b->blue, s))};
            const std::uint32_t index = atomicAdd(found, 1u);
            if (index < capacity)
                out[index] = point;
        }
    }
}

/// Says whether a value lies in a range; a NaN lies in none.
/// @param value [in] the value.
/// @param low [in] the range's least value.
/// @param high [in] its greatest value.
/// @return true when low <= value <= high.
bool within(float value, float low, float high)
{
    return value >= low && value <= high;
}

/// Checks the sparse TSDF's parameters other than its budget against their ranges
/// (sparse_tsdf.hpp, min_voxel_size_m and the others).
/// @param parameters [in] the parameters.
/// @return empty when the voxel size, each value of the truncation's rule and the far depth lie
///         in their ranges; else the reason.
/// @note Allocates for the reason; throws std::bad_alloc when that fails.
std::string parameters_refusal(const sparse_tsdf_parameters &parameters)
{
    const truncation_parameters &t = parameters.truncation;
    if (within(parameters.voxel_size_m, min_voxel_size_m, max_voxel_size_m) &&
        t.noise_per_m > 0.0f && t.noise_per_m <= max_noise_per_m &&
        within(t.noise_multiple, min_noise_multiple, max_noise_multiple) &&
        within(t.min_voxels, min_truncation_voxels, max_truncation_voxels) &&
        within(parameters.max_depth_m, nearest_max_depth_m, farthest_max_depth_m))
        return {};
    return "the voxel size (" + std::to_string(parameters.voxel_size_m) +
           " m), the truncation's rule (c " + std::to_string(t.noise_per_m) + " /m, k " +
           std::to_string(t.noise_multiple) + ", mu_min " + std::to_string(t.min_voxels) +
           " voxels) or the far depth (" + std::to_string(parameters.max_depth_m) +
           " m) cannot be used: the voxel size from " + std::to_string(min_voxel_size_m) + " to " +
           std::to_string(max_voxel_size_m) + " m, c above 0 up to " +
           std::to_string(max_noise_per_m) + " /m, k from " + std::to_string(min_noise_multiple) +
           " to " + std::to_string(max_noise_multiple) + ", mu_min from " +
           std::to_string(min_truncation_voxels) + " to " + std::to_string(max_truncation_voxels) +
           " voxels, the far depth from " + std::to_string(nearest_max_depth_m) + " to " +
           std::to_string(farthest_max_depth_m) + " m";
}

/// Gives the most blocks one frame can touch in the bound's model (docs/methods/sparse-tsdf.md,
/// "One frame's blocks"): a surface facing the camera that fills the model's picture, every pixel
/// at the far depth z_max, at any pose. The bands lie in the frustum's slice between the depths
/// z_max - mu and z_max + mu, mu = mu(z_max), inside a box of the camera frame of
/// w = 1280 (z_max + mu) / fx by h = 800 (z_max + mu) / fy by 2 mu. A block of edge e = 8 v that
/// meets the slice lies inside that box widened by the cube of +-e of the models' frame, which
/// reaches at most sqrt(3) e along each axis of the camera frame; the blocks are disjoint, so
/// they number at most (w + 2 sqrt(3) e)(h + 2 sqrt(3) e)(2 mu + 2 sqrt(3) e) / e^3. The bound
/// grows with the depth, so z_max is the model's worst depth.
/// @param parameters [in] the sparse TSDF's parameters, each in its range (parameters_refusal()).
/// @return the bound, in blocks, finite and > 0.
double frame_bound_blocks(const sparse_tsdf_parameters &parameters) noexcept
{
    const truncation_parameters &t = parameters.truncation;
    const double voxel_m = parameters.voxel_size_m;
    const double z = parameters.max_depth_m;
    const double mu = std::max(double{t.min_voxels} * voxel_m,
                               double{t.noise_multiple} * double{t.noise_per_m} * z * z);
    const double edge = 8.0 * voxel_m;
    const double pad = 2.0 * std::sqrt(3.0) * edge;
    const double w = model_width_px * (z + mu) / model_fx_px + pad;
    const double h = model_height_px * (z + mu) / model_fy_px + pad;
    return w * h * (2.0 * mu + pad) / (edge * edge * edge);
}
} // namespace

sparse_tsdf_layout layout_for_budget(std::uint64_t memory_budget_bytes) noexcept
{
    // bytes_for() grows with the blocks, so the largest number that fits is found by bisection
    // between 0, which always fits, and the most the budget's bytes could hold.
    std::uint64_t fits = 0;
    std::uint64_t too_many = std::min(memory_budget_bytes / block_bytes, max_blocks) + 1;
    while (too_many - fits > 1)
    {
        const std::uint64_t middle = fits + (too_many - fits) / 2;
        if (bytes_for(middle) <= memory_budget_bytes)
            fits = middle;
        else
            too_many = middle;
    }
    if (fits == 0)
        return {0, 0, 0};
    return {static_cast<std::uint32_t>(fits), static_cast<std::uint32_t>(slots_for(fits)),
            bytes_for(fits)};
}

request_result<surface_points> surface_points::request(gpu_pool &pool,
                                                       std::uint32_t capacity) noexcept
{
    request_result<gpu_array<surface_point>> points = pool.request_array<surface_point>(capacity);
    if (points.status != request_status::granted)
        return refused_as<surface_points>(points);
    request_result<gpu_array<std::uint32_t>> count = pool.request_array<std::uint32_t>(1);
    if (count.status != request_status::granted)
        return refused_as<surface_points>(count);
    request_result<pinned_buffer> count_on_host = pool.request_pinned(sizeof(std::uint32_t));
    if (count_on_host.status != request_status::granted)
        return refused_as<surface_points>(count_on_host);
    request_result<surface_points> result;
    result.status = request_status::granted;
    result.granted.m_points = std::move(points.granted);
    result.granted.m_count = std::move(count.granted);
    result.granted.m_count_on_host = std::move(count_on_host.granted);
    return result;
}

std::uint32_t surface_points::capacity() const noexcept
{
    return static_cast<std::uint32_t>(m_points.size());
}

const gpu_array<surface_point> &surface_points::points() const noexcept
{
    return m_points;
}

request_result<sparse_tsdf> sparse_tsdf::request(gpu_pool &pool,
                                                 const sparse_tsdf_parameters &parameters,
                                                 cudaStream_t stream) noexcept
{
    request_result<sparse_tsdf> result;
    std::string refusal = parameters_refusal(parameters);
    if (!refusal.empty())
    {
        result.status = request_status::failure;
        result.message = std::move(refusal);
        return result;
    }
    const sparse_tsdf_layout layout = layout_for_budget(parameters.memory_budget_bytes);
    if (layout.capacity_blocks == 0)
    {
        result.status = request_status::failure;
        result.message = "the memory budget of " + std::to_string(parameters.memory_budget_bytes) +
                         " bytes holds no block: one block with its slots, list entry and counts "
                         "needs " +
                         std::to_string(bytes_for(1)) + " bytes";
        return result;
    }
    // Once the pool is used up, a coordinate refused for it keeps its slot, so the room left for
    // one frame's new blocks is S - B; a frame that filled it would make each later insertion,
    // and each lookup of an absent block, probe all S slots. The bound has to stay below the
    // room, so that one frame touches fewer blocks than it and an empty slot is left.
    const double bound = frame_bound_blocks(parameters);
    const std::uint32_t room = layout.slot_count - layout.capacity_blocks;
    if (bound >= room)
    {
        result.status = request_status::failure;
        result.message =
            "the memory budget of " + std::to_string(parameters.memory_budget_bytes) +
            " bytes cannot hold one frame: a surface facing the camera at the far depth of " +
            std::to_string(parameters.max_depth_m) + " m can touch up to " +
            std::to_string(static_cast<std::uint64_t>(std::ceil(bound))) +
            " blocks, and the hash has room for " + std::to_string(room) +
            " once the pool is used up, its " + std::to_string(layout.slot_count) + " slots less " +
            std::to_string(layout.capacity_blocks) +
            " blocks (docs/methods/sparse-tsdf.md, \"One frame's blocks\")";
        return result;
    }
    sparse_tsdf &made = result.granted;
    made.m_layout = layout;
    made.m_parameters = parameters;
    request_result<gpu_array<unsigned long long>> keys =
        pool.request_array<unsigned long long>(layout.slot_count);
    if (keys.status != request_status::granted)
        return refused_as<sparse_tsdf>(keys);
    made.m_keys = std::move(keys.granted);
    request_result<gpu_array<std::uint32_t>> blocks =
        pool.request_array<std::uint32_t>(layout.slot_count);
    if (blocks.status != request_status::granted)
        return refused_as<sparse_tsdf>(blocks);
    made.m_blocks = std::move(blocks.granted);
    request_result<gpu_array<std::uint32_t>> stamps =
        pool.request_array<std::uint32_t>(layout.slot_count);
    if (stamps.status != request_status::granted)
        return refused_as<sparse_tsdf>(stamps);
    made.m_stamps = std::move(stamps.granted);
    request_result<gpu_array<std::uint32_t>> frame_slots =
        pool.request_array<std::uint32_t>(layout.capacity_blocks);
    if (frame_slots.status != request_status::granted)
        return refused_as<sparse_tsdf>(frame_slots);
    made.m_frame_slots = std::move(frame_slots.granted);
    request_result<gpu_array<block_counts>> counts = pool.request_array<block_counts>(1);
    if (counts.status != request_status::granted)
        return refused_as<sparse_tsdf>(counts);
    made.m_counts = std::move(counts.granted);
    request_result<gpu_array<voxel>> voxels = pool.request_array<voxel>(
        static_cast<std::size_t>(layout.capacity_blocks) * voxels_per_block);
    if (voxels.status != request_status::granted)
        return refused_as<sparse_tsdf>(voxels);
    made.m_voxels = std::move(voxels.granted);
    request_result<pinned_buffer> counts_on_host = pool.request_pinned(sizeof(block_counts));
    if (counts_on_host.status != request_status::granted)
        return refused_as<sparse_tsdf>(counts_on_host);
    made.m_counts_on_host = std::move(counts_on_host.granted);

    // Every key empty, every block's number none and every stamp no frame (all bits 1), the
    // counts 0, and every voxel unobserved (weight 0). The frame's list is written before it is
    // read.
    const cudaError_t errors[] = {
        cudaMemsetAsync(made.m_keys.data(), 0xFF, made.m_keys.size_bytes(), stream),
        cudaMemsetAsync(made.m_blocks.data(), 0xFF, made.m_blocks.size_bytes(), stream),
        cudaMemsetAsync(made.m_stamps.data(), 0xFF, made.m_stamps.size_bytes(), stream),
        cudaMemsetAsync(made.m_counts.data(), 0, made.m_counts.size_bytes(), stream),
        cudaMemsetAsync(made.m_voxels.data(), 0, made.m_voxels.size_bytes(), stream),
    };
    for (const cudaError_t error : errors)
    {
        if (error != cudaSuccess)
        {
            request_result<sparse_tsdf> refused;
            refused.message = cuda_failure("CUDA refused to clear the sparse TSDF's memory", error);
            return refused;
        }
    }
    result.status = request_status::granted;
    return result;
}

sparse_tsdf::sparse_tsdf(sparse_tsdf &&other) noexcept
    : m_layout(std::exchange(other.m_layout, sparse_tsdf_layout{0, 0, 0}))
    , m_parameters(std::exchange(other.m_parameters, sparse_tsdf_parameters{}))
    , m_frames_allocated(std::exchange(other.m_frames_allocated, 0))
    , m_keys(std::move(other.m_keys))
    , m_blocks(std::move(other.m_blocks))
    , m_stamps(std::move(other.m_stamps))
    , m_frame_slots(std::move(other.m_frame_slots))
    , m_counts(std::move(other.m_counts))
    , m_voxels(std::move(other.m_voxels))
    , m_counts_on_host(std::move(other.m_counts_on_host))
{
}

sparse_tsdf &sparse_tsdf::operator=(sparse_tsdf &&other) noexcept
{
    if (this != &other)
    {
        m_layout = std::exchange(other.m_layout, sparse_tsdf_layout{0, 0, 0});
        m_parameters = std::exchange(other.m_parameters, sparse_tsdf_parameters{});
        m_frames_allocated = std::exchange(other.m_frames_allocated, 0);
        m_keys = std::move(other.m_keys);
        m_blocks = std::move(other.m_blocks);
        m_stamps = std::move(other.m_stamps);
        m_frame_slots = std::move(other.m_frame_slots);
        m_counts = std::move(other.m_counts);
        m_voxels = std::move(other.m_voxels);
        m_counts_on_host = std::move(other.m_counts_on_host);
    }
    return *this;
}

step_result sparse_tsdf::insert_blocks(const gpu_array<block_coordinate> &coordinates,
                                       cudaStream_t stream) noexcept
{
    if (m_keys.empty())
        return {false, holds_no_memory};
    if (coordinates.empty())
        return {false, "no coordinates to insert"};
    return launch_kernel("the insertion of blocks", &insert_kernel,
                         grid_for_count(coordinates.size()), dim3(threads_per_block), stream,
                         hash_view(), coordinates.data(), coordinates.size());
}

step_result sparse_tsdf::find_blocks(const gpu_array<block_coordinate> &coordinates,
                                     gpu_array<std::uint32_t> &blocks,
                                     cudaStream_t stream) const noexcept
{
    if (m_keys.empty())
        return {false, holds_no_memory};
    if (coordinates.empty())
        return {false, "no coordinates to find"};
    if (blocks.size() != coordinates.size())
        return {false, std::to_string(coordinates.size()) + " coordinates to find, room for " +
                           std::to_string(blocks.size()) + " block numbers"};
    return launch_kernel("the lookup of blocks", &find_kernel, grid_for_count(coordinates.size()),
                         dim3(threads_per_block), stream, hash_view(), coordinates.data(),
                         coordinates.size(), blocks.data());
}

step_result sparse_tsdf::allocate_near_surface(const organized_points &points,
                                               const camera_pose &pose,
                                               cudaStream_t stream) noexcept
{
    if (m_keys.empty())
        return {false, holds_no_memory};
    if (points.points().empty())
        return {false, "the organized points hold no images"};
    const std::optional<float_pose> rounded = float_pose_of(pose);
    if (!rounded)
        return {false, unusable_pose};
    return allocate_at(points, *rounded, stream);
}

step_result sparse_tsdf::allocate_at(const organized_points &points, const float_pose &pose,
                                     cudaStream_t stream) noexcept
{
    // Nothing is queued after an earlier error, so that a refusal leaves the list's count as it
    // was; launch_kernel() asks again before its launch.
    const cudaError_t earlier = cudaPeekAtLastError();
    if (earlier != cudaSuccess)
        return {false, cuda_failure("CUDA's last error, left by an earlier call, before the "
                                    "allocation near a frame's surface",
                                    earlier)};
    void *listed = reinterpret_cast<char *>(m_counts.data()) + offsetof(block_counts, listed_slots);
    const cudaError_t reset =
        cudaMemsetAsync(listed, 0, sizeof(block_counts::listed_slots), stream);
    if (reset != cudaSuccess)
        return {false, cuda_failure("CUDA refused to reset the count of the frame's list", reset)};
    const dim3 block = picture_block();
    // This frame's stamp: the count of allocations before it, below every bit 1 (the @pre).
    const std::uint32_t frame = m_frames_allocated++;
    return launch_kernel(
        "the allocation's kernel", &allocate_kernel, grid_for(points.size(), block), block, stream,
        points.points().data(), points.validity_mask().data(), points.size(), pose,
        m_parameters.max_depth_m, 1.0f / (8.0f * m_parameters.voxel_size_m), rule_of(m_parameters),
        hash_view(), m_stamps.data(), m_frame_slots.data(), frame);
}

step_result sparse_tsdf::integrate(const gpu_frame &frame, const organized_points &points,
                                   const camera_pose &pose, cudaStream_t stream) noexcept
{
    // Every refusal comes before the allocation, so that a refused call queues nothing; the
    // allocation then asks for CUDA's last error.
    if (m_keys.empty())
        return {false, holds_no_memory};
    if (points.points().empty())
        return {false, "the organized points hold no images"};
    const frame_description &description = frame.description();
    if (frame.color().empty() || description.size != frame.color().size())
        return {false, "the GPU frame holds no frame: none was uploaded into it"};
    if (description.size != points.size())
        return {false, "the frame is " + size_text(description.size) + ", the organized points " +
                           size_text(points.size())};
    if (!usable_intrinsics(description.intrinsics))
        return {false, "the frame's intrinsics cannot be used: fx and fy must be finite and > 0, "
                       "cx and cy finite"};
    const std::optional<float_pose> rounded = float_pose_of(pose);
    if (!rounded)
        return {false, unusable_pose};
    step_result allocated = allocate_at(points, *rounded, stream);
    if (!allocated.done)
        return allocated;
    const observed_frame observed{points.points().data(), points.validity_mask().data(),
                                  frame.color().data(), description.size, description.intrinsics};
    return launch_kernel("the integration's kernel", &integrate_kernel, dim3(voxel_kernel_grid),
                         dim3(voxels_per_block), stream, hash_view(), m_frame_slots.data(),
                         m_voxels.data(), observed, *rounded, m_parameters.voxel_size_m,
                         m_parameters.max_depth_m, rule_of(m_parameters));
}

extraction_result sparse_tsdf::extract(surface_points &out, cudaStream_t stream) const noexcept
{
    extraction_result result;
    if (m_keys.empty())
    {
        result.message = holds_no_memory;
        return result;
    }
    if (out.m_points.empty())
    {
        result.message = "the buffer of points holds no memory: it was not granted by the pool, or "
                         "moved from";
        return result;
    }
    // Nothing is queued after an earlier error; launch_kernel() asks again before its launch.
    const cudaError_t earlier = cudaPeekAtLastError();
    if (earlier != cudaSuccess)
    {
        result.message = cuda_failure(
            "CUDA's last error, left by an earlier call, before the extraction", earlier);
        return result;
    }
    const cudaError_t reset =
        cudaMemsetAsync(out.m_count.data(), 0, out.m_count.size_bytes(), stream);
    if (reset != cudaSuccess)
    {
        result.message = cuda_failure("CUDA refused to reset the count of the points", reset);
        return result;
    }
    const voxel_lookup tsdf{hash_view(), m_voxels.data()};
    const step_result launched =
        launch_kernel("the extraction's kernel", &extract_kernel, dim3(voxel_kernel_grid),
                      dim3(voxels_per_block), stream, tsdf, m_parameters.voxel_size_m,
                      out.m_points.data(), out.capacity(), out.m_count.data());
    if (!launched.done)
    {
        result.message = launched.message;
        return result;
    }
    const cudaError_t copy_error =
        cudaMemcpyAsync(out.m_count_on_host.bytes().data(), out.m_count.data(),
                        out.m_count.size_bytes(), cudaMemcpyDeviceToHost, stream);
    if (copy_error != cudaSuccess)
    {
        result.message =
            cuda_failure("CUDA refused the copy of the count of the points", copy_error);
        return result;
    }
    // The one wait: the copy, and the stream's work before it, have run.
    const cudaError_t wait_error = cudaStreamSynchronize(stream);
    if (wait_error != cudaSuccess)
    {
        result.message = cuda_failure("CUDA failed while the points were extracted", wait_error);
        return result;
    }
    std::uint32_t found = 0;
    std::memcpy(&found, out.m_count_on_host.bytes().data(), sizeof(found));
    result.found = found;
    result.written = std::min(found, out.capacity());
    result.done = found <= out.capacity();
    if (!result.done)
        result.message = "the extraction found " + std::to_string(found) +
                         " points, the buffer holds " + std::to_string(out.capacity()) +
                         ": its first " + std::to_string(out.capacity()) + " are written";
    return result;
}

usage_result sparse_tsdf::read_usage(cudaStream_t stream) noexcept
{
    usage_result result;
    if (m_keys.empty())
    {
        result.message = holds_no_memory;
        return result;
    }
    const cudaError_t copy_error =
        cudaMemcpyAsync(m_counts_on_host.bytes().data(), m_counts.data(), m_counts.size_bytes(),
                        cudaMemcpyDeviceToHost, stream);
    if (copy_error != cudaSuccess)
    {
        result.message =
            cuda_failure("CUDA refused the copy of the sparse TSDF's counts", copy_error);
        return result;
    }
    // The one wait: the copy, and the stream's work before it, have run.
    const cudaError_t wait_error = cudaStreamSynchronize(stream);
    if (wait_error != cudaSuccess)
    {
        result.message =
            cuda_failure("CUDA failed while the sparse TSDF's counts were read", wait_error);
        return result;
    }
    block_counts counts{};
    std::memcpy(&counts, m_counts_on_host.bytes().data(), sizeof(counts));
    sparse_tsdf_usage usage{};
    usage.capacity_blocks = m_layout.capacity_blocks;
    usage.blocks_in_use = std::min(counts.claims, m_layout.capacity_blocks);
    usage.blocks_refused = counts.claims - usage.blocks_in_use;
    usage.coordinates_refused = counts.coordinates_refused;
    usage.listed_slots = std::min(counts.listed_slots, m_layout.capacity_blocks);
    usage.unlisted_slots = counts.listed_slots - usage.listed_slots;
    usage.rays_not_walked = counts.rays_not_walked;
    result.usage = usage;
    return result;
}

block_hash_view sparse_tsdf::hash_view() const noexcept
{
    // The view's pointers are not const, for the kernels that insert; a const sparse TSDF hands
    // it only to kernels that find blocks, which read through it and write nothing.
    return {const_cast<unsigned long long *>(m_keys.data()),
            const_cast<std::uint32_t *>(m_blocks.data()),
            const_cast<block_counts *>(m_counts.data()), m_layout.slot_count,
            m_layout.capacity_blocks};
}

sparse_tsdf_layout sparse_tsdf::layout() const noexcept
{
    return m_layout;
}
} // namespace gca
