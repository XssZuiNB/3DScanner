// SPDX-License-Identifier: 0BSD
/// @file sparse_tsdf.cu
/// The sparse TSDF's hash of blocks and its pool: the division of the memory budget, the request
/// with its checks of the parameters and of one frame's bound against the hash's room, the
/// clearing of the memory, the kernels that insert and find blocks, the allocation of the
/// blocks near a frame's surface with its traversal of the grid of blocks, the integration, the
/// ray cast's march and its surface, the extraction, and the reading of the usage (see
/// sparse_tsdf.hpp; the method: docs/methods/sparse-tsdf.md).

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

/// Why intrinsics are refused (usable_intrinsics()), after the words that name them: "the frame's"
/// in the integration, "the" in the ray cast.
constexpr const char *unusable_intrinsics =
    "intrinsics cannot be used: fx and fy must be finite and > 0, cx and cy finite";

/// Why a pose is refused (float_pose_of()).
constexpr const char *unusable_pose = "the pose cannot be used: its position and orientation "
                                      "rounded to float must be finite and its orientation not "
                                      "zero";

static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t), "a key is one 64-bit word");

/// Gives the slots of the hash for a number of blocks: four times as many, so that the room once
/// the pool is used up holds one frame of the object classes' full models' TSDFs
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

/// A block coordinate no block of a traversal has: a clamped end's block lies within one block of
/// the hash's range (lowest_end, highest_end), far above the least int.
constexpr std::int32_t no_cell = std::numeric_limits<std::int32_t>::min();

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

/// The traversal of the grid of blocks along a segment (Amanatides and Woo 1987;
/// docs/methods/sparse-tsdf.md, "The allocation near a frame's surface", step 2), the one walk the
/// allocation and the ray cast take: its three axes and the count of the blocks it visits, from the
/// start's block to the end's, each step crossing one border.
struct block_walk
{
    axis_walk x;          ///< the walk along the models' x axis
    axis_walk y;          ///< along its y axis
    axis_walk z;          ///< along its z axis
    std::uint32_t blocks; ///< the blocks visited, 1 + the steps of the three axes
};

/// Starts the walk along a segment: its ends clamped to one block past the hash's range on each
/// side (clamped()), each axis at the start's block with its steps to the end's (start_axis()).
/// @param start [in] the segment's start, in blocks of the models' frame.
/// @param end [in] its end, in blocks of the models' frame.
/// @param walk [out] the walk at the start's block, when both ends are finite.
/// @return whether both ends are finite (finite_end()); a segment with an end that is not finite is
///         not walked.
/// @note Device code; allocates nothing.
__device__ inline bool start_walk(const float3 &start, const float3 &end, block_walk &walk)
{
    if (!finite_end(start) || !finite_end(end))
        return false;
    const float3 from = clamped(start);
    const float3 to = clamped(end);
    walk.x = start_axis(from.x, to.x);
    walk.y = start_axis(from.y, to.y);
    walk.z = start_axis(from.z, to.z);
    // Each axis has at most 2^21 + 1 steps, so the sum fits 32 bits.
    walk.blocks = 1 + walk.x.remaining + walk.y.remaining + walk.z.remaining;
    return true;
}

/// Gives the block a walk is in.
/// @param walk [in] the walk.
/// @return the block's coordinates, in blocks.
/// @note Device code; allocates nothing.
__device__ inline block_coordinate walk_block(const block_walk &walk)
{
    return {walk.x.cell, walk.y.cell, walk.z.cell};
}

/// Gives where a walk leaves its block: the segment's parameter at the first border ahead.
/// @param walk [in] the walk.
/// @return t in [0, 1] at that border; 1, the segment's end, in its last block.
/// @note Device code; allocates nothing.
__device__ inline float walk_exit(const block_walk &walk)
{
    return fminf(fminf(walk.x.t_next, walk.y.t_next), fminf(walk.z.t_next, 1.0f));
}

/// Steps a walk into the next block: across the border that comes first. An axis without a step
/// left has t_next infinite and, the ends finite, no t_next is NaN (start_axis(), advance()), so
/// when neither x nor y is taken, z has a step left; the caller's count of the blocks bounds the
/// steps whatever the choice.
/// @param walk [in,out] the walk, with fewer blocks visited than walk.blocks.
/// @note Device code; allocates nothing.
__device__ inline void next_block(block_walk &walk)
{
    if (walk.x.remaining != 0 && walk.x.t_next <= walk.y.t_next && walk.x.t_next <= walk.z.t_next)
        advance(walk.x);
    else if (walk.y.remaining != 0 && walk.y.t_next <= walk.z.t_next)
        advance(walk.y);
    else
        advance(walk.z);
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
    block_walk walk;
    if (!start_walk(near_end, far_end, walk) || walk.blocks > max_blocks_per_ray)
    {
        atomicAdd(&hash.counts->rays_not_walked, 1u);
        return;
    }
    touch_block(hash, stamps, frame_slots, frame, walk_block(walk));
    for (std::uint32_t visited = 1; visited < walk.blocks; visited++)
    {
        next_block(walk);
        touch_block(hash, stamps, frame_slots, frame, walk_block(walk));
    }
}

/// The thread blocks of the two kernels that go through blocks of voxels, the integration's over
/// the frame's list and the extraction's over the hash's slots in chunks of 512: a fixed grid, each
/// thread block taking every grid's-th entry or chunk, so that the integration reads the list's
/// count on the GPU and the per-frame path reads nothing back. 24, as many thread blocks of 512
/// threads as the GPU holds at once: 8 multiprocessors of 1536 threads each [M 2026-11-15:
/// cudaGetDeviceProperties], the kernels at 34 and 40 registers a thread, the extraction's held
/// to 40 by its launch bounds [B 2026-11-16: cuobjdump -res-usage], 3 x 512 x 40 below the 65536
/// registers of a multiprocessor. Back to back, integrate() took 0.4 % longer to
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

/// Turns a direction of the models' frame into the camera frame of a pose: R^T d.
/// @param pose [in] the pose, camera frame to the models' frame.
/// @param d [in] the direction, in the models' frame.
/// @return R^T d, in the camera frame.
/// @note Device code; allocates nothing.
__device__ inline float3 turned_into_camera(const float_pose &pose, const float3 &d)
{
    // R^T d is the sum of R's rows, each times d's component of its index.
    return d.x * pose.row_x + d.y * pose.row_y + d.z * pose.row_z;
}

/// Carries a point of the models' frame into the camera frame of a pose: q = R^T (p - t).
/// @param pose [in] the pose, camera frame to the models' frame.
/// @param p [in] the point, in the models' frame, in m.
/// @return q, in the camera frame, in m.
/// @note Device code; allocates nothing.
__device__ inline float3 into_camera(const float_pose &pose, const float3 &p)
{
    return turned_into_camera(pose, p - pose.translation);
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

/// Says whether two blocks' coordinates are the same.
/// @param a [in] one.
/// @param b [in] the other.
/// @return whether x, y and z are equal.
/// @note Device code; allocates nothing.
__device__ inline bool same_block(const block_coordinate &a, const block_coordinate &b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

/// Gives the block a voxel lies in.
/// @param g [in] the voxel, in voxels of the models' frame.
/// @return floor(g / 8) on each axis, in blocks.
/// @note Device code; allocates nothing. Arithmetic shifts: g >> 3 is floor(g / 8) for a negative
///       g too [S: C++20, two's complement].
__device__ inline block_coordinate block_of_voxel(const int3 &g)
{
    return {g.x >> 3, g.y >> 3, g.z >> 3};
}

/// Gives a voxel in its block, once the block's number is found.
/// @param tsdf [in] the hash and the voxels.
/// @param number [in] the number of the block the voxel lies in (block_of_voxel()), or no_block.
/// @param g [in] the voxel, in voxels of the models' frame.
/// @return the voxel in the pool, at its place g & 7 in the block (pool_index()); nullptr when the
///         number is no_block.
/// @note Device code; allocates nothing. Masks: g & 7 is g's remainder after floor(g / 8) for a
///       negative g too [S: C++20, two's complement].
__device__ inline const voxel *voxel_in_pool(const voxel_lookup &tsdf, std::uint32_t number,
                                             const int3 &g)
{
    if (number == no_block)
        return nullptr;
    return &tsdf.voxels[pool_index(number, make_int3(g.x & 7, g.y & 7, g.z & 7))];
}

/// Finds a voxel by its coordinates: in the block at hand without the hash, in another block
/// through it (find_block()).
/// @param tsdf [in] the hash and the voxels.
/// @param g [in] the voxel, in voxels of the models' frame.
/// @param own [in] the block at hand, with a number.
/// @return the voxel; nullptr when its block has no number.
/// @note Device code; allocates nothing.
__device__ inline const voxel *voxel_at(const voxel_lookup &tsdf, const int3 &g,
                                        const held_block &own)
{
    const block_coordinate b = block_of_voxel(g);
    return voxel_in_pool(tsdf,
                         same_block(b, own.coordinate) ? own.number : find_block(tsdf.hash, b), g);
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

/// The threads of a warp [S: CUDA's warp size, 32 on every GPU so far].
constexpr unsigned int warp_threads = 32;

/// The warps of a thread block of the extraction.
constexpr unsigned int extraction_warps = voxels_per_block / warp_threads;

/// Extracts one held block's points (docs/methods/sparse-tsdf.md, "The extraction"), one thread
/// for each of its voxels, the thread's voxel being voxel_in_block(threadIdx.x): an observed voxel
/// g and each observed neighbour g + e, e along x, y and z, whose distances differ in sign give a
/// point at s = D_a / (D_a - D_b) of the way from g's centre to the neighbour's, with the
/// gradients by central differences at both interpolated at s and normalized as its normal and
/// the colors interpolated at s as its color; a crossing whose central differences need a voxel
/// that is missing or unobserved, or whose interpolated gradient is zero, gives no point. Every
/// point is counted; the first capacity are written.
/// @param tsdf [in] the hash and the voxels, as extract_kernel() takes them.
/// @param held [in] the block, with a number.
/// @param voxel_size_m [in] v, in m, > 0.
/// @param out [out] device memory of capacity points, as extract_kernel() takes it.
/// @param capacity [in] the points out holds.
/// @param found [in,out] the count of the points found, as extract_kernel() takes it.
/// @note Device code; allocates nothing and waits for no other thread.
__device__ inline void extract_block(const voxel_lookup &tsdf, const held_block &held,
                                     float voxel_size_m, surface_point *__restrict__ out,
                                     std::uint32_t capacity, std::uint32_t *__restrict__ found)
{
    const int3 local = voxel_in_block(threadIdx.x);
    const voxel a = tsdf.voxels[pool_index(held.number, local)];
    if (a.weight == 0)
        return;
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
            make_uchar3(between_channels(a.red, b->red, s), between_channels(a.green, b->green, s),
                        between_channels(a.blue, b->blue, s))};
        const std::uint32_t index = atomicAdd(found, 1u);
        if (index < capacity)
            out[index] = point;
    }
}

/// Extracts the surface's points (docs/methods/sparse-tsdf.md, "The extraction"): the fixed grid
/// goes through the hash's slots in chunks of 512 consecutive slots, one chunk to a thread block at
/// a time. Each thread reads one slot of the chunk, so that the chunk's keys come in coalesced
/// loads; the slots that hold a block with a number are listed in shared memory in slot order, each
/// at the place its warp's ballot and the earlier warps' counts give it; then the thread block
/// extracts the listed blocks one after another, one thread for each voxel (extract_block()).
/// @param tsdf [in] the hash and the voxels; device memory owned by the sparse TSDF, which holds it
///        until the kernel has run.
/// @param voxel_size_m [in] v, in m, > 0.
/// @param out [out] device memory of capacity points; owned by the caller's buffer, which holds it
///        until the kernel has run.
/// @param capacity [in] the points out holds.
/// @param found [in,out] device memory of one count, 0 before the launch, which ends as the points
///        found: at most 3 * 512 * capacity_blocks, below 2^32 for every pool of fewer than
///        2796203 blocks, 11.4 GB, more than the board's memory [S: computed].
/// @note Runs on the stream of its launch, sparse_tsdf::extract()'s, after the integrations, with
///       voxels_per_block threads a block: every warp is whole and takes part in its ballot. A
///       chunk's first slot stays below slot_count, at most 2^31, and grows by gridDim.x * 512 =
///       12288 on the fixed grid, so it never wraps [S: computed]. Two barriers a chunk, each
///       reached by every thread, as the chunk's loop runs alike for the whole thread block: the
///       first between the warps' counts and their reading, the second between the list and its
///       reading. The next chunk needs none more: it writes the counts after the second barrier,
///       which every thread passes only once it has read them, and the list after its own first
///       barrier, which every thread reaches only once it has read this chunk's list; the count
///       of the list is each thread's own. The launch bounds ask for 3 thread blocks a
///       multiprocessor, so that the fixed grid's 24 run at once (voxel_kernel_grid): 40 registers
///       a thread and 56 bytes spilled to the stack, against 57 registers and 2 blocks without the
///       bound, which took 1.3 to 1.6 times as long [B 2026-11-16: `cuobjdump -res-usage
///       build/src/libfusion.a`, the line under extract_kernel; M 2026-11-16: docs/HARDWARE.md,
///       "The extraction"]; that line is read again whenever extract_block() or the bound changes.
///       Reads the voxels and writes only out and found. Allocates nothing; 8256 bytes of shared
///       memory [S: computed, 512 blocks of 16 bytes and 16 counts of 4].
__global__ void __launch_bounds__(voxels_per_block, 3)
    extract_kernel(voxel_lookup tsdf, float voxel_size_m, surface_point *__restrict__ out,
                   std::uint32_t capacity, std::uint32_t *__restrict__ found)
{
    __shared__ held_block chunk_blocks[voxels_per_block];
    __shared__ std::uint32_t warp_held[extraction_warps];
    const unsigned int warp = threadIdx.x / warp_threads;
    const unsigned int lane = threadIdx.x % warp_threads;
    const std::uint32_t slot_count = tsdf.hash.slot_count;
    for (std::uint32_t first = blockIdx.x * voxels_per_block; first < slot_count;
         first += gridDim.x * voxels_per_block)
    {
        // The chunk's slots, one a thread; past the last slot, in a hash of fewer than 512 slots,
        // a thread holds no block. The request grants such a hash only for a tiny pool: 256 slots
        // for 64 blocks at 16 mm voxels and a far depth of 0.25 m, where one frame's bound is
        // about 180 blocks [S: computed, frame_bound_blocks()].
        const std::uint32_t slot = first + threadIdx.x;
        const held_block held =
            slot < slot_count ? block_in_slot(tsdf.hash, slot) : no_held_block();
        const bool holds = held.number != no_block;
        const unsigned int ballot = __ballot_sync(0xffffffffu, holds);
        if (lane == 0)
            warp_held[warp] = __popc(ballot);
        __syncthreads();
        // The list's places in slot order: after the earlier warps' blocks, the earlier lanes'.
        std::uint32_t place = __popc(ballot & ((1u << lane) - 1u));
        std::uint32_t listed = 0;
        for (unsigned int w = 0; w < extraction_warps; w++)
        {
            place += w < warp ? warp_held[w] : 0u;
            listed += warp_held[w];
        }
        if (holds)
            chunk_blocks[place] = held;
        __syncthreads();
        for (std::uint32_t k = 0; k < listed; k++)
            extract_block(tsdf, chunk_blocks[k], voxel_size_m, out, capacity, found);
    }
}

/// The blocks a ray's lookups of voxels last found: the block the march is in and the last other
/// block a voxel was found in, so that a sample's 8 voxels and a normal's 24 more, which lie in one
/// or a few blocks, look each block up about once.
struct block_cache
{
    held_block at_hand; ///< the march's block, its number no_block when the hash holds none
    held_block other;   ///< the last other block looked up, its number no_block when none
};

/// Finds a voxel by its coordinates for the ray cast: in the block at hand or the cached other one
/// without the hash, in any other block through it (find_block()), which then becomes the cached
/// other one.
/// @param tsdf [in] the hash and the voxels.
/// @param g [in] the voxel, in voxels of the models' frame.
/// @param cache [in,out] the blocks found last.
/// @return the voxel; nullptr when its block has no number.
/// @note Device code; allocates nothing.
__device__ inline const voxel *cast_voxel(const voxel_lookup &tsdf, const int3 &g,
                                          block_cache &cache)
{
    const block_coordinate b = block_of_voxel(g);
    std::uint32_t number;
    if (same_block(b, cache.at_hand.coordinate))
        number = cache.at_hand.number;
    else if (same_block(b, cache.other.coordinate))
        number = cache.other.number;
    else
    {
        number = find_block(tsdf.hash, b);
        cache.other = {b, number};
    }
    return voxel_in_pool(tsdf, number, g);
}

/// Reads an observed voxel's distance for the ray cast, the whole voxel in one read of 8 bytes, as
/// interpolate() reads its 8; interpolate() reads them inline, as the cast with one function that
/// hands an observed voxel out to both took 7.6 % to 8.4 % longer back to back [M 2026-11-15:
/// docs/methods/sparse-tsdf.md, "The ray cast"].
/// @param tsdf [in] the hash and the voxels.
/// @param g [in] the voxel, in voxels of the models' frame.
/// @param cache [in,out] the blocks found last (cast_voxel()).
/// @param distance [out] its distance, in voxels, when observed.
/// @return whether the voxel exists and is observed (weight > 0).
/// @note Device code; allocates nothing.
__device__ inline bool cast_distance(const voxel_lookup &tsdf, const int3 &g, block_cache &cache,
                                     float &distance)
{
    const voxel *cell = cast_voxel(tsdf, g, cache);
    if (cell == nullptr)
        return false;
    const voxel held = *cell;
    if (held.weight == 0)
        return false;
    distance = __half2float(held.distance);
    return true;
}

/// The distances of a cell's 8 voxels, which the normal's ends reuse from the crossing's
/// interpolation (crossing_gradient()): corner c is voxel cell_corner(g0, c) of the cell's lower
/// corner g0, in interpolate()'s order.
struct cell_distances
{
    float corner[8]; ///< each corner's distance, in voxels
};

/// Gives a corner of a cell.
/// @param g0 [in] the cell's lower corner, in voxels of the models' frame.
/// @param c [in] the corner, 0 to 7.
/// @return g0 + (c & 1, (c >> 1) & 1, c >> 2), in voxels.
/// @note Device code; allocates nothing.
__device__ inline int3 cell_corner(const int3 &g0, int c)
{
    return make_int3(g0.x + (c & 1), g0.y + ((c >> 1) & 1), g0.z + (c >> 2));
}

/// Gives a corner's weight in a cell's trilinear interpolation: the product of its fractions along
/// the three axes, f on the cell's upper side and 1 - f on its lower.
/// @param c [in] the corner, 0 to 7 (cell_corner()).
/// @param f [in] the point's fractions in the cell along x, y and z, each in [0, 1].
/// @return the weight, in [0, 1].
/// @note Device code; allocates nothing.
__device__ inline float corner_weight(int c, const float3 &f)
{
    return ((c & 1) != 0 ? f.x : 1.0f - f.x) * (((c >> 1) & 1) != 0 ? f.y : 1.0f - f.y) *
           ((c >> 2) != 0 ? f.z : 1.0f - f.z);
}

/// Interpolates the 8 voxels whose centres surround a point trilinearly, each weighed by the
/// product of its fractions along the three axes: the distance, and the color when asked for.
/// @param tsdf [in] the hash and the voxels.
/// @param g [in] the point on the grid of the voxels' centres, in voxels: p / v - 1/2 for the point
///        p of the models' frame, so that voxel g's centre lies at g.
/// @param cache [in,out] the blocks found last (cast_voxel()).
/// @param distance [out] the interpolated distance, in voxels, when all 8 voxels are observed.
/// @param color [out] when not nullptr, the interpolated red, green and blue, 0 to 255, unrounded,
///        when all 8 voxels are observed.
/// @param kept [out] when not nullptr, the 8 voxels' distances, when all 8 are observed.
/// @return whether all 8 voxels exist and are observed (weight > 0): a sample, or none.
/// @note Device code; allocates nothing. A point beyond float's whole numbers converts to an int
///       that saturates; its block then lies outside the hash's range and is found missing.
///       trilinear_sum() repeats this sum's operations in its order, so that the normal's ends
///       summed from held distances are these interpolations' bit for bit (crossing_gradient()):
///       a change to the weights or the sum here is made there too.
__device__ inline bool interpolate(const voxel_lookup &tsdf, const float3 &g, block_cache &cache,
                                   float &distance, float3 *color, cell_distances *kept = nullptr)
{
    const float3 base = make_float3(floorf(g.x), floorf(g.y), floorf(g.z));
    const int3 g0 =
        make_int3(__float2int_rd(base.x), __float2int_rd(base.y), __float2int_rd(base.z));
    const float3 f = g - base;
    float sum = 0.0f;
    float3 rgb = make_float3(0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int c = 0; c < 8; c++)
    {
        const voxel *corner = cast_voxel(tsdf, cell_corner(g0, c), cache);
        if (corner == nullptr)
            return false;
        const voxel held = *corner;
        if (held.weight == 0)
            return false;
        const float w = corner_weight(c, f);
        const float d = __half2float(held.distance);
        sum += w * d;
        if (kept != nullptr)
            kept->corner[c] = d;
        if (color != nullptr)
            rgb =
                rgb + w * make_float3(static_cast<float>(held.red), static_cast<float>(held.green),
                                      static_cast<float>(held.blue));
    }
    distance = sum;
    if (color != nullptr)
        *color = rgb;
    return true;
}

/// Interpolates a cell's held distances trilinearly at a point, by interpolate()'s operations in
/// its order, so that the two give the same distance bit for bit.
/// @param d [in] the cell's 8 distances, in voxels.
/// @param f [in] the point's fractions in the cell along x, y and z, each in [0, 1].
/// @return the interpolated distance, in voxels.
/// @note Device code; allocates nothing.
__device__ inline float trilinear_sum(const cell_distances &d, const float3 &f)
{
    float sum = 0.0f;
#pragma unroll
    for (int c = 0; c < 8; c++)
        sum += corner_weight(c, f) * d.corner[c];
    return sum;
}

/// Gives the six ends of the normal's central differences of one voxel along each axis, each
/// interpolated trilinearly at g + 1 or g - 1 along its axis, rounded to float (interpolate()).
/// @param tsdf [in] the hash and the voxels.
/// @param g [in] the point on the grid of the voxels' centres, in voxels (interpolate()).
/// @param cache [in,out] the blocks found last (cast_voxel()).
/// @param after [out] (D(g + x), D(g + y), D(g + z)), in voxels, when every end is a sample.
/// @param before [out] (D(g - x), D(g - y), D(g - z)), in voxels, likewise.
/// @return whether every one of the six ends is a sample (interpolate()), tried in the order
///         x after, x before, y after, y before, z after, z before up to the first that is none.
/// @note Device code; allocates nothing. Reads the 48 voxels of the six ends.
__device__ inline bool interpolated_ends(const voxel_lookup &tsdf, const float3 &g,
                                         block_cache &cache, float3 &after, float3 &before)
{
    return interpolate(tsdf, make_float3(g.x + 1.0f, g.y, g.z), cache, after.x, nullptr) &&
           interpolate(tsdf, make_float3(g.x - 1.0f, g.y, g.z), cache, before.x, nullptr) &&
           interpolate(tsdf, make_float3(g.x, g.y + 1.0f, g.z), cache, after.y, nullptr) &&
           interpolate(tsdf, make_float3(g.x, g.y - 1.0f, g.z), cache, before.y, nullptr) &&
           interpolate(tsdf, make_float3(g.x, g.y, g.z + 1.0f), cache, after.z, nullptr) &&
           interpolate(tsdf, make_float3(g.x, g.y, g.z - 1.0f), cache, before.z, nullptr);
}

/// Gives an end of the normal's central differences along one axis, g + 1 or g - 1 rounded to
/// float: its fraction in its cell as interpolate() computes it, and whether that cell is the
/// crossing's moved by one voxel.
/// @param end [in] the end's coordinate along the axis, in voxels.
/// @param cell [in] the crossing's cell's lower corner along the axis moved by one voxel to the
///        end's side, floor(g) + 1 or floor(g) - 1, in voxels.
/// @param fraction [out] end - floor(end), in [0, 1].
/// @return whether floor(end) is cell.
/// @note Device code; allocates nothing.
__device__ inline bool end_in_cell(float end, int cell, float &fraction)
{
    const float base = floorf(end);
    fraction = end - base;
    return __float2int_rd(base) == cell;
}

/// Gives the two ends of the normal's central differences along one axis, each end's cell the
/// crossing's moved by one voxel along it: of an end's 8 voxels, the 4 on the crossing's cell's
/// layer next to the end are the crossing's held distances, and the 4 on the layer beyond the
/// cell are read, each checked before the next is read; the axis's 8 new voxels, the after end's
/// first.
/// @tparam axis 0, 1 or 2: x, y or z.
/// @param tsdf [in] the hash and the voxels.
/// @param g0 [in] the crossing's cell's lower corner, in voxels of the models' frame.
/// @param own [in] the crossing's 8 distances (interpolate()).
/// @param after_f [in] the after end's fractions in its cell: the crossing's, and along the axis
///        its own (end_in_cell()).
/// @param before_f [in] the before end's fractions, likewise.
/// @param cache [in,out] the blocks found last (cast_voxel()).
/// @param after [out] the distance at g + 1 along the axis, in voxels, when the 8 new voxels are
///        observed.
/// @param before [out] the distance at g - 1 along the axis, in voxels, likewise.
/// @return whether the axis's 8 new voxels all exist and are observed (cast_distance()).
/// @note Device code; allocates nothing.
template <int axis>
__device__ inline bool ends_along(const voxel_lookup &tsdf, const int3 &g0,
                                  const cell_distances &own, const float3 &after_f,
                                  const float3 &before_f, block_cache &cache, float &after,
                                  float &before)
{
    constexpr int bit = 1 << axis;
    const int3 step = make_int3(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0, axis == 2 ? 1 : 0);
    cell_distances after_cell;
    cell_distances before_cell;
    // The after end's cell is the crossing's moved up by one: its corners on the upper side are
    // new, those on the lower side the crossing's upper ones.
#pragma unroll
    for (int c = 0; c < 8; c++)
    {
        if ((c & bit) == 0)
            continue;
        const int3 g = cell_corner(g0, c);
        if (!cast_distance(tsdf, make_int3(g.x + step.x, g.y + step.y, g.z + step.z), cache,
                           after_cell.corner[c]))
            return false;
        before_cell.corner[c] = own.corner[c & ~bit];
    }
    // The before end's cell is the crossing's moved down by one: its corners on the lower side are
    // new, those on the upper side the crossing's lower ones.
#pragma unroll
    for (int c = 0; c < 8; c++)
    {
        if ((c & bit) != 0)
            continue;
        const int3 g = cell_corner(g0, c);
        if (!cast_distance(tsdf, make_int3(g.x - step.x, g.y - step.y, g.z - step.z), cache,
                           before_cell.corner[c]))
            return false;
        after_cell.corner[c] = own.corner[c | bit];
    }
    after = trilinear_sum(after_cell, after_f);
    before = trilinear_sum(before_cell, before_f);
    return true;
}

/// Gives the gradient of the distance at a crossing by central differences of one voxel along each
/// axis, each end interpolated trilinearly: the same as the trilinear interpolation of the voxels'
/// own central differences, the interpolation being linear in the voxels' values and the same at
/// every shift of the grid [S]. Each end's cell is the crossing's moved by one voxel along the
/// end's axis, so the six ends' 48 voxels are 32 distinct ones, along each axis the 4 layers
/// floor(g) - 1 to floor(g) + 2 of the crossing's 2 x 2 voxels across it, 8 of them the
/// crossing's own, which are held [S: the lattice]: each axis reads its 8 new voxels
/// (ends_along()), and each end is summed from its 8 values in interpolate()'s operations and
/// order, so the gradient is the six interpolations' bit for bit. Where float's rounding of g + 1
/// or g - 1 carries it across a whole number into another cell than that, the six interpolations
/// are taken as they are (interpolated_ends()): g - 1 rounds to -1 at g = -2^-24, g + 1 to 2 at
/// g = 1 - 2^-24 [S: float32, rounding to the nearest, a tie to the even one]. Both can occur:
/// nvcc fuses a multiply-add into one fma by default (`--fmad=true` [S: nvcc --help]), and
/// g = p / v - 1/2 is one, rounded once [B 2026-11-15: `cuobjdump -sass`, FFMA with -0.5], so g
/// can lie that near a whole number on either side.
/// @param tsdf [in] the hash and the voxels.
/// @param g [in] the crossing on the grid of the voxels' centres, in voxels (interpolate()), its 8
///        voxels observed: so its cell lies in the hash's range of blocks, 2^20 blocks either side
///        of 0, and floor(g) + 1 and floor(g) - 1 are ints [S: block_coordinate_min and max].
/// @param own [in] the crossing's 8 distances (interpolate()).
/// @param cache [in,out] the blocks found last (cast_voxel()).
/// @param gradient [out] ((D(g + x) - D(g - x)) / 2, ...), in voxels per voxel, in the models'
///        frame, when every one of the six ends is a sample.
/// @return whether every one of the six ends is a sample: whether each of their voxels exists and
///         is observed, the 32 or, where an end's cell is another, the 48.
/// @note Device code; allocates nothing.
__device__ inline bool crossing_gradient(const voxel_lookup &tsdf, const float3 &g,
                                         const cell_distances &own, block_cache &cache,
                                         float3 &gradient)
{
    const float3 base = make_float3(floorf(g.x), floorf(g.y), floorf(g.z));
    const int3 g0 =
        make_int3(__float2int_rd(base.x), __float2int_rd(base.y), __float2int_rd(base.z));
    const float3 f = g - base;
    float3 after_f;
    float3 before_f;
    float3 after;
    float3 before;
    if (end_in_cell(g.x + 1.0f, g0.x + 1, after_f.x) &&
        end_in_cell(g.x - 1.0f, g0.x - 1, before_f.x) &&
        end_in_cell(g.y + 1.0f, g0.y + 1, after_f.y) &&
        end_in_cell(g.y - 1.0f, g0.y - 1, before_f.y) &&
        end_in_cell(g.z + 1.0f, g0.z + 1, after_f.z) &&
        end_in_cell(g.z - 1.0f, g0.z - 1, before_f.z))
    {
        if (!ends_along<0>(tsdf, g0, own, make_float3(after_f.x, f.y, f.z),
                           make_float3(before_f.x, f.y, f.z), cache, after.x, before.x) ||
            !ends_along<1>(tsdf, g0, own, make_float3(f.x, after_f.y, f.z),
                           make_float3(f.x, before_f.y, f.z), cache, after.y, before.y) ||
            !ends_along<2>(tsdf, g0, own, make_float3(f.x, f.y, after_f.z),
                           make_float3(f.x, f.y, before_f.z), cache, after.z, before.z))
            return false;
    }
    else if (!interpolated_ends(tsdf, g, cache, after, before))
        return false;
    gradient = 0.5f * (after - before);
    return true;
}

/// What a ray found where it meets the surface.
struct ray_hit
{
    float depth_m;   ///< the surface's depth along the camera's z, in m
    float3 normal;   ///< the unit normal, in the camera frame; (0, 0, 0) for none
    float intensity; ///< the Rec. 709 luma of the interpolated color, in [0, 1]
};

/// The ray cast's march along one ray (docs/methods/sparse-tsdf.md, "The ray cast"), in the frame
/// of a camera ray r of z = 1: the point at depth z is t + z R r.
struct ray_march
{
    float3 origin;      ///< t, the camera's centre in the models' frame, in m
    float3 direction;   ///< R r, the ray turned into the models' frame, |R r| = |r| >= 1
    float near_m;       ///< the depth the march starts at, along the camera's z, in m
    float far_m;        ///< the deepest crossing taken, along the camera's z, in m
    float step_z;       ///< the samples' step along the camera's z, in m: mu_min v / 2 / |r|
    float voxels_per_m; ///< 1 / v
    float held_near_m;  ///< the least depth of the held blocks the ray's tile sees, in m
    float held_far_m;   ///< their greatest depth, in m
};

/// Gives the depth of a sample of a march: the lattice near_m + i step_z along the whole ray.
/// @param march [in] the ray.
/// @param i [in] the sample's index on the lattice.
/// @return its depth along the camera's z, in m.
/// @note Device code; allocates nothing.
__device__ inline float sample_depth(const ray_march &march, std::uint32_t i)
{
    return march.near_m + static_cast<float>(i) * march.step_z;
}

/// Gives the first sample of a march at or past a depth: the index the march through every block
/// reaches there, as its samples' depths never fall from one index to the next (sample_depth(), a
/// sum and a product in float of a growing index, each rounded to the nearest [S]).
/// @param march [in] the ray.
/// @param depth [in] the depth, along the camera's z, in m, from near_m to one step past far_m.
/// @return the least i with sample_depth(march, i) >= depth.
/// @note Device code; allocates nothing. The estimate (depth - near_m) / step_z lies within a few
///       indices of the answer, which the two loops then step to.
__device__ inline std::uint32_t first_sample_at(const ray_march &march, float depth)
{
    const float estimate = (depth - march.near_m) / march.step_z;
    // Below 2^32, so that the conversion is defined; the march's indices stay far below it.
    std::uint32_t i = estimate > 0.0f ? static_cast<std::uint32_t>(fminf(estimate, 4.0e9f)) : 0u;
    while (i > 0 && !(sample_depth(march, i - 1) < depth))
        i--;
    while (sample_depth(march, i) < depth)
        i++;
    return i;
}

/// Gives the point at a depth along a march's ray on the grid of the voxels' centres.
/// @param march [in] the ray.
/// @param z [in] the depth along the camera's z, in m.
/// @return (t + z R r) / v - 1/2, in voxels (interpolate()).
/// @note Device code; allocates nothing.
__device__ inline float3 on_voxel_grid(const ray_march &march, float z)
{
    const float3 p = march.origin + z * march.direction;
    return make_float3(p.x * march.voxels_per_m - 0.5f, p.y * march.voxels_per_m - 0.5f,
                       p.z * march.voxels_per_m - 0.5f);
}

/// Gives the surface where two consecutive samples hold it: the depth where their linear
/// interpolation is 0, its normal and its intensity.
/// @param tsdf [in] the hash and the voxels.
/// @param pose [in] the camera's pose, for the normal's turn into the camera frame.
/// @param march [in] the ray.
/// @param cache [in,out] the blocks found last, the later sample's at hand (cast_voxel()).
/// @param z [in] the earlier sample's depth, in m.
/// @param before [in] its distance, >= 0, in voxels.
/// @param after [in] the later sample's distance, at z + step_z, < 0, in voxels.
/// @param hit [out] the surface, when there is one.
/// @return whether the crossing lies no deeper than the far limit and its own 8 voxels are all
///         observed, for its intensity; else no surface.
/// @note Device code; allocates nothing.
__device__ inline bool surface_between(const voxel_lookup &tsdf, const float_pose &pose,
                                       const ray_march &march, block_cache &cache, float z,
                                       float before, float after, ray_hit &hit)
{
    // KinectFusion's equation 15: the linear interpolation of the two samples.
    const float depth = z + march.step_z * before / (before - after);
    if (!(depth <= march.far_m))
        return false;
    const float3 g = on_voxel_grid(march, depth);
    float at_surface;
    float3 color;
    cell_distances own;
    if (!interpolate(tsdf, g, cache, at_surface, &color, &own))
        return false;
    hit.depth_m = depth;
    hit.intensity = frame_pyramid::intensity_of(color.x, color.y, color.z);
    hit.normal = make_float3(0.0f, 0.0f, 0.0f);
    float3 gradient;
    if (crossing_gradient(tsdf, g, own, cache, gradient))
    {
        const float3 turned = turned_into_camera(pose, gradient);
        const float length = norm(turned);
        if (length > 0.0f && isfinite(length))
            hit.normal = turned / length;
    }
    return true;
}

/// Marches one ray through the sparse TSDF to its surface (docs/methods/sparse-tsdf.md, "The ray
/// cast"): the grid traversal of the allocation (block_walk) visits the blocks its segment from the
/// near limit to one step past the far limit crosses, in order; in a block the hash holds no voxels
/// for, every sample is none, without a voxel read, and ends the pair (the lattice's indices
/// stepped through, faster than a jump to the block's exit, which took 7 % to 10 % longer
/// [M 2026-11-15: docs/HARDWARE.md, "The ray cast"]); in a block it holds, the samples at the
/// depths near_m + i step_z, one lattice along the whole ray, each the trilinear interpolation of
/// 8 observed voxels or none, which also ends the pair. The first two consecutive samples from
/// >= 0 to < 0 hold the surface (surface_between()); a change from < 0 to >= 0 before it is a back
/// face, and the ray has none. The ray's tile bounds the blocks it looks up ("The depth
/// interval"): a block it leaves before held_near_m less the margin is stepped past without a
/// lookup or a sample, and the march ends at a block it enters after held_far_m plus the margin.
/// Every block the ray crosses that the hash holds lies within that interval, so those it steps
/// past and those after its end hold no voxels: their samples would be none, the pair is ended at
/// the first block looked up, and the first sample there is the one the march through every block
/// reaches (first_sample_at()): the ray's surface is the same bit for bit. The margin covers the
/// rounding the walk's depths and the blocks' projected depths can gather: (n + 4) 2^-21 (z_end +
/// |t|) for a walk of n blocks, z_end one step past the far limit and |t| the camera's distance
/// from the models' origin, 4 times the walk's rounding after n additions and the corners'
/// [?: a choice of the factor; the rounding S: float's, 2^-24 of a value for each operation].
/// It is 0.12 mm and 0.23 mm at the live models' ranges, 125 and 116 blocks, with the camera within
/// 1 m of the origin [?: an assumption; S: computed]; farther out it grows with |t|, about 0.7 mm
/// at 10 m [S: computed], which costs lookups of a few more blocks, not a surface.
/// @param tsdf [in] the hash and the voxels.
/// @param pose [in] the camera's pose.
/// @param march [in] the ray, with its tile's interval of held blocks.
/// @param hit [out] the surface, when there is one.
/// @return whether the ray meets the surface, from the front, within the range.
/// @note Device code; allocates nothing. The thread visits the blocks the segment crosses, looks
///       up those within the interval, one lookup each, and a held block's samples, 8 voxels each.
__device__ inline bool march_to_surface(const voxel_lookup &tsdf, const float_pose &pose,
                                        const ray_march &march, ray_hit &hit)
{
    // One step past the far limit, so that a crossing at the far limit has its later sample.
    const float end_z = march.far_m + march.step_z;
    const float blocks_per_m = march.voxels_per_m / 8.0f;
    block_walk walk;
    if (!start_walk((march.origin + march.near_m * march.direction) * blocks_per_m,
                    (march.origin + end_z * march.direction) * blocks_per_m, walk))
        return false;
    const float span = end_z - march.near_m;
    const float margin =
        static_cast<float>(walk.blocks + 4) * 0x1p-21f * (end_z + norm(march.origin));
    const float first_m = march.held_near_m - margin;
    const float last_m = march.held_far_m + margin;
    // Before the first block, the cache holds no block of the walk, so that each is looked up.
    block_cache cache{{{no_cell, no_cell, no_cell}, no_block},
                      {{no_cell, no_cell, no_cell}, no_block}};
    bool paired = false;          // whether the last sample is the one before the next
    bool stepped_past = false;    // whether the last block was stepped past without a lookup
    float before = 0.0f;          // the last sample's distance, in voxels
    float entry_z = march.near_m; // the depth where the walk enters its block
    std::uint32_t i = 0;          // the next sample's index on the lattice
    for (std::uint32_t visited = 0; visited < walk.blocks; visited++)
    {
        if (visited > 0)
            next_block(walk);
        const float exit_z = march.near_m + walk_exit(walk) * span;
        if (exit_z < first_m)
        {
            stepped_past = true;
            entry_z = exit_z;
            continue;
        }
        if (entry_z > last_m)
            return false;
        if (stepped_past)
        {
            i = first_sample_at(march, entry_z);
            stepped_past = false;
        }
        entry_z = exit_z;
        const block_coordinate b = walk_block(walk);
        if (same_block(b, cache.other.coordinate))
        {
            const held_block left = cache.at_hand;
            cache.at_hand = cache.other;
            cache.other = left;
        }
        else
            cache.at_hand = {b, find_block(tsdf.hash, b)};
        for (;; i++)
        {
            const float z = sample_depth(march, i);
            if (!(z < exit_z))
                break;
            float after;
            // In a block without voxels each sample is none, without a voxel read.
            if (cache.at_hand.number == no_block ||
                !interpolate(tsdf, on_voxel_grid(march, z), cache, after, nullptr))
            {
                paired = false;
                continue;
            }
            if (paired && before >= 0.0f && after < 0.0f)
                return surface_between(tsdf, pose, march, cache, z - march.step_z, before, after,
                                       hit);
            if (paired && before < 0.0f && after >= 0.0f)
                return false;
            before = after;
            paired = true;
        }
    }
    return false;
}

/// The images a ray cast writes at one level of the model's surface (model_surface).
struct cast_level
{
    float3 *points;      ///< the level's points, rows dense
    std::uint8_t *valid; ///< its marks
    float3 *normals;     ///< its normals
    float *intensity;    ///< its intensity
    frame_size size;     ///< its width and height, in px
};

/// The images of every level a ray cast writes, by value as the kernel's argument.
struct cast_images
{
    cast_level levels[frame_pyramid::level_count]; ///< element l: level l
};

/// The tiles of the depth interval ("The depth interval"), by value as the kernels' argument.
struct depth_tiles
{
    uint2 *codes;       ///< each tile's interval, rows dense: x the code of its least depth
                        ///< (encoded_near()), y of its greatest (encoded_far()), both 0 when it
                        ///< holds no block; device memory of the caller's model_surface
    std::uint32_t wide; ///< the tiles in a row, >= 1: ceil(width / depth_tile_px)
};

/// Gives a tile.
/// @param tiles [in] the tiles.
/// @param x [in] the tile's column, below tiles.wide.
/// @param y [in] its row, below the tiles' rows.
/// @return the tile's two codes, in device memory.
/// @note Device code; allocates nothing.
__device__ inline uint2 &tile_at(const depth_tiles &tiles, std::uint32_t x, std::uint32_t y)
{
    return tiles.codes[static_cast<std::size_t>(y) * tiles.wide + x];
}

/// The picture the depth interval's tiles cover and how the march's segment lies in depth, by
/// value as the depth interval's kernel's argument ("The depth interval").
struct tile_view
{
    depth_tiles tiles; ///< the tiles, cleared to 0 before the kernel
    frame_size size;   ///< level 0's width and height, in px
    intrinsics k;      ///< level 0's intrinsics, in px, usable
    float_pose pose;   ///< the camera's pose
    float near_m;      ///< the march's first depth, in m
    float end_m;       ///< its last, one step past the far limit, in m
    float block_m;     ///< a block's edge, 8 v, in m
};

/// The least depth a corner of a block may have for the block to be projected, in m: a block that
/// reaches nearer the camera's plane while the march starts within a block of it covers the whole
/// picture [?: a choice, far below the object classes' near limits of 0.3 m and 0.4 m].
constexpr float least_projected_depth_m = 1e-3f;

/// Clamps a depth to +0 and above, so that its bits order as the depths do: a depth below 0, and
/// -0, whose sign bit would make its code beat every other one, become +0; fmaxf(-0, 0) may give
/// -0 [S: IEEE 754, the sign of a zero's maximum].
/// @param depth_m [in] the depth, in m, finite.
/// @return depth_m when > 0, else +0.
/// @note Device code; allocates nothing.
__device__ inline float at_least_zero(float depth_m)
{
    return depth_m > 0.0f ? depth_m : 0.0f;
}

/// Encodes a tile's least depth so that the atomic maximum of the codes keeps the least depth and
/// 0 is no depth: the complement of the bits of the depth clamped to +0 and above
/// (at_least_zero()), which order as the depths do [S: IEEE 754's layout].
/// @param depth_m [in] the depth, in m, finite.
/// @return its code, never 0.
/// @note Device code; allocates nothing.
__device__ inline std::uint32_t encoded_near(float depth_m)
{
    return ~__float_as_uint(at_least_zero(depth_m));
}

/// Encodes a tile's greatest depth so that the atomic maximum keeps the greatest depth and 0 is no
/// depth: the bits of the depth clamped to +0 and above, plus 1.
/// @param depth_m [in] the depth, in m, finite.
/// @return its code, never 0.
/// @note Device code; allocates nothing.
__device__ inline std::uint32_t encoded_far(float depth_m)
{
    return __float_as_uint(at_least_zero(depth_m)) + 1u;
}

/// Says whether no block widened a tile since its clearing.
/// @param tile [in] the tile's two codes.
/// @return whether its codes are 0, which no encoded depth is.
/// @note Device code; allocates nothing.
__device__ inline bool holds_no_block(const uint2 &tile)
{
    return tile.y == 0u;
}

/// Gives a tile's least depth (encoded_near()).
/// @param tile [in] the tile's two codes, of a tile that holds a block.
/// @return the depth, in m, >= 0.
/// @note Device code; allocates nothing.
__device__ inline float decoded_near(const uint2 &tile)
{
    return __uint_as_float(~tile.x);
}

/// Gives a tile's greatest depth (encoded_far()).
/// @param tile [in] the tile's two codes, of a tile that holds a block.
/// @return the depth, in m, >= 0.
/// @note Device code; allocates nothing.
__device__ inline float decoded_far(const uint2 &tile)
{
    return __uint_as_float(tile.y - 1u);
}

/// Widens a tile's interval by a block's depths: the atomic maximum of each code.
/// @param tile [in,out] the tile's codes, in device memory.
/// @param near_m [in] the block's least corner depth, in m, finite.
/// @param far_m [in] its greatest, in m, finite.
/// @note Device code; allocates nothing.
__device__ inline void widen_tile(uint2 &tile, float near_m, float far_m)
{
    atomicMax(&tile.x, encoded_near(near_m));
    atomicMax(&tile.y, encoded_far(far_m));
}

/// Projects every block the hash holds into the picture of level 0 and widens the interval of each
/// tile its projection covers by its depths (docs/methods/sparse-tsdf.md, "The ray cast", "The
/// depth interval"), one thread for each slot. A block whose corners all lie nearer than one block
/// before the march's first depth, or all farther than one block past its last, is left out; the
/// rest is clipped at the plane one block before the first depth, or at least_projected_depth_m,
/// and the rectangle around its corners' and its edges' crossings' projections, rounded out to
/// whole pixels, covers the tiles; when the march starts within a block of the camera's plane and
/// the block reaches nearer than least_projected_depth_m, it covers the whole picture. The depths
/// are the least and the greatest of its corners'.
/// @param hash [in] the hash; read only (block_in_slot()).
/// @param view [in] the picture, its tiles, the pose and the march's depths; its tiles cleared to
///        0 before the launch.
/// @note Runs on the stream of its launch, sparse_tsdf::ray_cast()'s, after the tiles' clearing and
///       the integrations it is to see. Reads the hash and writes only the tiles. A thread whose
///       block covers many tiles widens each: a block near the camera covers more. Allocates
///       nothing.
__global__ void __launch_bounds__(threads_per_block)
    depth_interval_kernel(block_hash_view hash, tile_view view)
{
    const std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= hash.slot_count)
        return;
    const held_block held = block_in_slot(hash, static_cast<std::uint32_t>(slot));
    if (held.number == no_block)
        return;
    const float block_m = view.block_m;
    const float3 origin = make_float3(static_cast<float>(held.coordinate.x) * block_m,
                                      static_cast<float>(held.coordinate.y) * block_m,
                                      static_cast<float>(held.coordinate.z) * block_m);
    float3 corners[8];
    float block_near_m = INFINITY;
    float block_far_m = -INFINITY;
#pragma unroll
    for (int c = 0; c < 8; c++)
    {
        corners[c] = into_camera(view.pose, origin + make_float3((c & 1) != 0 ? block_m : 0.0f,
                                                                 (c & 2) != 0 ? block_m : 0.0f,
                                                                 (c & 4) != 0 ? block_m : 0.0f));
        block_near_m = fminf(block_near_m, corners[c].z);
        block_far_m = fmaxf(block_far_m, corners[c].z);
    }
    const float clip_m = view.near_m - block_m;
    if (!(block_far_m >= clip_m) || !(block_near_m <= view.end_m + block_m))
        return;
    const float plane_m = fmaxf(clip_m, least_projected_depth_m);
    std::uint32_t first_x = 0;
    std::uint32_t first_y = 0;
    std::uint32_t last_x = view.tiles.wide - 1;
    std::uint32_t last_y = (view.size.height - 1) / depth_tile_px;
    if (!(block_near_m < plane_m && clip_m < least_projected_depth_m))
    {
        // The rectangle around the corners at or past the plane and the edges' crossings of it.
        float u_low = INFINITY;
        float u_high = -INFINITY;
        float v_low = INFINITY;
        float v_high = -INFINITY;
        const auto take = [&](const float3 &q) {
            const float u = view.k.fx * q.x / q.z + view.k.cx;
            const float v = view.k.fy * q.y / q.z + view.k.cy;
            u_low = fminf(u_low, u);
            u_high = fmaxf(u_high, u);
            v_low = fminf(v_low, v);
            v_high = fmaxf(v_high, v);
        };
#pragma unroll
        for (int c = 0; c < 8; c++)
        {
            if (corners[c].z >= plane_m)
                take(corners[c]);
#pragma unroll
            for (int axis = 1; axis < 8; axis <<= 1)
            {
                const int d = c | axis;
                // Each edge once, from its corner with the bit 0 to the one with it 1.
                if ((c & axis) != 0 || (corners[c].z >= plane_m) == (corners[d].z >= plane_m))
                    continue;
                const float s = (plane_m - corners[c].z) / (corners[d].z - corners[c].z);
                take(make_float3(corners[c].x + s * (corners[d].x - corners[c].x),
                                 corners[c].y + s * (corners[d].y - corners[c].y), plane_m));
            }
        }
        // Rounded out to whole pixels: a pixel's ray lies at a whole u and v, so every pixel less
        // than 1 px, less the projection's rounding, outside the rectangle is taken, which covers
        // the walk's rounding across the ray, below 0.1 px at the object classes' ranges
        // [S: computed, docs/methods/sparse-tsdf.md, "The ray cast", "The depth interval"].
        const float width = static_cast<float>(view.size.width);
        const float height = static_cast<float>(view.size.height);
        u_low = floorf(u_low);
        u_high = ceilf(u_high);
        v_low = floorf(v_low);
        v_high = ceilf(v_high);
        if (!(u_high >= 0.0f) || !(u_low <= width - 1.0f) || !(v_high >= 0.0f) ||
            !(v_low <= height - 1.0f))
            return;
        first_x = static_cast<std::uint32_t>(fmaxf(u_low, 0.0f)) / depth_tile_px;
        last_x = static_cast<std::uint32_t>(fminf(u_high, width - 1.0f)) / depth_tile_px;
        first_y = static_cast<std::uint32_t>(fmaxf(v_low, 0.0f)) / depth_tile_px;
        last_y = static_cast<std::uint32_t>(fminf(v_high, height - 1.0f)) / depth_tile_px;
    }
    for (std::uint32_t y = first_y; y <= last_y; y++)
        for (std::uint32_t x = first_x; x <= last_x; x++)
            widen_tile(tile_at(view.tiles, x, y), block_near_m, block_far_m);
}

/// The thread blocks of the ray cast's march a multiprocessor holds at once, at the least: 4 of
/// 256 threads, which holds the march's kernel at 64 registers a thread, a multiprocessor's 65536
/// registers over 1024 threads [S: CUDA C++ Programming Guide, compute capability 8.7]. Without it
/// the kernel takes 74 registers [B 2026-11-15: `cuobjdump -res-usage`], and the normal's voxels
/// read once take 9.2 % to 17.3 % off the cast back to back instead of 19.5 % to 27.9 %
/// [M 2026-11-15: docs/methods/sparse-tsdf.md, "The ray cast"].
constexpr int ray_cast_blocks_per_multiprocessor = 4;

/// Ray casts the model's surface, one thread for each pixel of level 0
/// (docs/methods/sparse-tsdf.md, "The ray cast"; march_to_surface()). A pixel whose ray meets the
/// surface gets the point at the surface's depth on its ray, in the camera frame, its mark 1, the
/// normal or (0, 0, 0) and the intensity; any other pixel all 0. Pixel (u, v) of level l is the
/// cast at K / 2^l, whose ray is level 0's at (2^l u, 2^l v) [S: (u - cx / 2^l) / (fx / 2^l) =
/// (2^l u - cx) / fx]: that pixel's thread writes it too.
/// @param tsdf [in] the hash and the voxels; device memory owned by the sparse TSDF, which holds
///        it until the kernel has run.
/// @param out [out] every level's images; owned by the caller's surface, which holds them until
///        the kernel has run.
/// @param k [in] level 0's intrinsics, in px, usable.
/// @param pose [in] the camera's pose, camera frame to the models' frame.
/// @param near_m [in] the depth the march starts at, in m, >= 0.
/// @param far_m [in] the deepest crossing taken, in m, > near_m.
/// @param voxel_size_m [in] v, in m, > 0.
/// @param step_m [in] the samples' step along the ray, mu_min v / 2, in m, > 0.
/// @param tiles [in] each tile's interval of held blocks (depth_interval_kernel()): device memory
///        of the caller's surface, which holds it until the kernel has run, rows of tiles.wide
///        tiles, >= 1, ceil(width / depth_tile_px).
/// @note Runs on the stream of its launch, sparse_tsdf::ray_cast()'s, after the integrations it is
///       to see and the depth interval's kernel. A thread outside level 0 writes nothing; a thread
///       whose tile sees no held block writes its pixel invalid without a march. Reads the tiles,
///       the hash and the voxels and writes only out. Allocates nothing. At most 64 registers a
///       thread (ray_cast_blocks_per_multiprocessor).
__global__ void __launch_bounds__(picture_block_width *picture_block_height,
                                  ray_cast_blocks_per_multiprocessor)
    ray_cast_kernel(voxel_lookup tsdf, cast_images out, intrinsics k, float_pose pose, float near_m,
                    float far_m, float voxel_size_m, float step_m, depth_tiles tiles)
{
    const std::uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;
    const frame_size size = out.levels[0].size;
    if (u >= size.width || v >= size.height)
        return;
    const float3 ray = make_float3((static_cast<float>(u) - k.cx) / k.fx,
                                   (static_cast<float>(v) - k.cy) / k.fy, 1.0f);
    const uint2 tile = tile_at(tiles, u / depth_tile_px, v / depth_tile_px);
    ray_hit hit{0.0f, make_float3(0.0f, 0.0f, 0.0f), 0.0f};
    bool found = false;
    // A tile that no held block's projection covers has no block on any of its rays to march.
    if (!holds_no_block(tile))
    {
        const ray_march march{
            pose.translation,
            make_float3(dot(pose.row_x, ray), dot(pose.row_y, ray), dot(pose.row_z, ray)),
            near_m,
            far_m,
            step_m / norm(ray),
            1.0f / voxel_size_m,
            decoded_near(tile),
            decoded_far(tile)};
        found = march_to_surface(tsdf, pose, march, hit);
    }
    const float3 point = found ? hit.depth_m * ray : make_float3(0.0f, 0.0f, 0.0f);
    for (std::uint32_t level = 0; level < frame_pyramid::level_count; level++)
    {
        const std::uint32_t stride = 1u << level;
        if (u % stride != 0 || v % stride != 0)
            break;
        const cast_level &images = out.levels[level];
        // u / stride < ceil(width / stride), the level's width, as u < width; so for the rows.
        const std::size_t pixel =
            static_cast<std::size_t>(v / stride) * images.size.width + u / stride;
        images.points[pixel] = point;
        images.valid[pixel] = found ? 1 : 0;
        images.normals[pixel] = hit.normal;
        images.intensity[pixel] = hit.intensity;
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

request_result<model_surface> model_surface::request(gpu_pool &pool,
                                                     const frame_size &size) noexcept
{
    request_result<model_surface> result;
    for (std::size_t level = 0; level < frame_pyramid::level_count; level++)
    {
        const frame_size sized = frame_pyramid::level_size(size, level);
        request_result<gpu_image<float3>> points = pool.request<float3>(sized);
        if (points.status != request_status::granted)
            return refused_as<model_surface>(points);
        request_result<gpu_image<std::uint8_t>> valid = pool.request<std::uint8_t>(sized);
        if (valid.status != request_status::granted)
            return refused_as<model_surface>(valid);
        request_result<gpu_image<float3>> normals = pool.request<float3>(sized);
        if (normals.status != request_status::granted)
            return refused_as<model_surface>(normals);
        request_result<gpu_image<float>> intensity = pool.request<float>(sized);
        if (intensity.status != request_status::granted)
            return refused_as<model_surface>(intensity);
        level_images &granted = result.granted.m_levels[level];
        granted.points = std::move(points.granted);
        granted.valid = std::move(valid.granted);
        granted.normals = std::move(normals.granted);
        granted.intensity = std::move(intensity.granted);
    }
    // A tile for each depth_tile_px x depth_tile_px pixels of level 0, the last ones in a row and
    // a column partly outside the picture (blocks_for()).
    request_result<gpu_image<uint2>> tiles = pool.request<uint2>(
        {blocks_for(size.width, depth_tile_px), blocks_for(size.height, depth_tile_px)});
    if (tiles.status != request_status::granted)
        return refused_as<model_surface>(tiles);
    result.granted.m_depth_tiles = std::move(tiles.granted);
    result.status = request_status::granted;
    return result;
}

frame_size model_surface::size(std::size_t level) const noexcept
{
    return m_levels.at(level).points.size();
}

const gpu_image<float3> &model_surface::points(std::size_t level) const noexcept
{
    return m_levels.at(level).points;
}

const gpu_image<std::uint8_t> &model_surface::validity_mask(std::size_t level) const noexcept
{
    return m_levels.at(level).valid;
}

const gpu_image<float3> &model_surface::normals(std::size_t level) const noexcept
{
    return m_levels.at(level).normals;
}

const gpu_image<float> &model_surface::intensity(std::size_t level) const noexcept
{
    return m_levels.at(level).intensity;
}

gca::intrinsics model_surface::level_intrinsics(std::size_t level) const noexcept
{
    return m_levels.at(level).intrinsics;
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
        return {false, std::string("the frame's ") + unusable_intrinsics};
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

step_result sparse_tsdf::ray_cast(const camera_pose &pose, const gca::intrinsics &k,
                                  const organized_points_parameters &range, model_surface &out,
                                  cudaStream_t stream) const noexcept
{
    if (m_keys.empty())
        return {false, holds_no_memory};
    if (out.m_levels[0].points.empty())
        return {false, "the model's surface holds no images: it was not granted by the pool, or "
                       "moved from"};
    if (!usable_intrinsics(k))
        return {false, std::string("the ") + unusable_intrinsics};
    std::string refusal = organized_points::range_refusal(range);
    if (!refusal.empty())
        return {false, std::move(refusal)};
    const std::optional<float_pose> rounded = float_pose_of(pose);
    if (!rounded)
        return {false, unusable_pose};
    cast_images images{};
    for (std::size_t level = 0; level < frame_pyramid::level_count; level++)
    {
        model_surface::level_images &l = out.m_levels[level];
        images.levels[level] = {l.points.data(), l.valid.data(), l.normals.data(),
                                l.intensity.data(), l.points.size()};
    }
    // Nothing is queued after an earlier error, so that a refusal leaves the surface as it was;
    // launch_kernel() asks again before each launch.
    const cudaError_t earlier = cudaPeekAtLastError();
    if (earlier != cudaSuccess)
        return {false, cuda_failure("CUDA's last error, left by an earlier call, before the ray "
                                    "cast",
                                    earlier)};
    gpu_image<uint2> &tile_image = out.m_depth_tiles;
    const cudaError_t cleared =
        cudaMemsetAsync(tile_image.data(), 0, tile_image.size_bytes(), stream);
    if (cleared != cudaSuccess)
        return {false, cuda_failure("CUDA refused to clear the ray cast's tiles", cleared)};
    const depth_tiles tiles{tile_image.data(), tile_image.size().width};
    const truncation_parameters &t = m_parameters.truncation;
    const float step_m = 0.5f * t.min_voxels * m_parameters.voxel_size_m;
    const frame_size size = out.m_levels[0].points.size();
    const tile_view view{tiles,
                         size,
                         k,
                         *rounded,
                         range.near_m,
                         range.far_m + step_m,
                         8.0f * m_parameters.voxel_size_m};
    const block_hash_view hash = hash_view();
    const step_result projected =
        launch_kernel("the depth interval's kernel", &depth_interval_kernel,
                      grid_for_count(hash.slot_count), dim3(threads_per_block), stream, hash, view);
    if (!projected.done)
        return projected;
    const dim3 block = picture_block();
    const step_result launched =
        launch_kernel("the ray cast's kernel", &ray_cast_kernel, grid_for(size, block), block,
                      stream, voxel_lookup{hash, m_voxels.data()}, images, k, *rounded,
                      range.near_m, range.far_m, m_parameters.voxel_size_m, step_m, tiles);
    if (launched.done)
    {
        for (std::size_t level = 0; level < frame_pyramid::level_count; level++)
            out.m_levels[level].intrinsics = frame_pyramid::intrinsics_at_level(k, level);
    }
    return launched;
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
