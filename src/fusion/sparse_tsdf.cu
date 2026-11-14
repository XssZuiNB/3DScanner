// SPDX-License-Identifier: 0BSD
/// @file sparse_tsdf.cu
/// The sparse TSDF's hash of blocks and its pool: the division of the memory budget, the request
/// and clearing of the memory, the kernels that insert and find blocks, the allocation of the
/// blocks near a frame's surface with its traversal of the grid of blocks, and the reading of the
/// usage (see sparse_tsdf.hpp; the method: docs/methods/sparse-tsdf.md).

#include "fusion/sparse_tsdf.hpp"

#include "fusion/block_hash.cuh"
#include "gpu/gpu_messages.hpp"
#include "gpu/kernel_launch.cuh"

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
namespace
{
/// The bytes each block of the pool takes: its 512 voxels of 8 bytes, 4096, and its entry on
/// the frame's list of slots, 4 [S: computed].
constexpr std::uint64_t block_bytes = voxels_per_block * sizeof(voxel) + sizeof(std::uint32_t);

/// The bytes each slot of the hash takes: its key, 8, its block's number, 4, and its stamp of the
/// last frame, 4 [S: computed].
constexpr std::uint64_t slot_bytes = sizeof(unsigned long long) + 2 * sizeof(std::uint32_t);

/// The most blocks a pool holds, 2^30, so that the slots, at most 2^31, fit 32 bits.
constexpr std::uint64_t max_blocks = std::uint64_t{1} << 30;

/// The kernels' block: 256 threads [?: a choice; no other size was measured].
constexpr unsigned int threads_per_block = 256;

/// The allocation's block: 32 x 8 threads, one warp along a row of pixels and 8 rows, as the
/// organized points' kernel [?: a choice; its time on the board stands in docs/HARDWARE.md, "The
/// per-frame path's stages"; no other block was measured].
constexpr unsigned int pixel_block_width = 32;
constexpr unsigned int pixel_block_height = 8;

/// Why an empty sparse TSDF refuses every operation.
constexpr const char *holds_no_memory =
    "the sparse TSDF holds no memory: it was not granted by the pool, or moved from";

static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t), "a key is one 64-bit word");

/// Gives the slots of the hash for a number of blocks.
/// @param blocks [in] the blocks, from 1 to max_blocks.
/// @return the least power of two >= 2 blocks.
std::uint64_t slots_for(std::uint64_t blocks) noexcept
{
    return std::bit_ceil(2 * blocks);
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

/// Finds every coordinate's block, one thread for each, by find_slot()
/// (fusion/block_hash.cuh).
/// @param keys [in] device memory of slot_count keys; owned by the sparse TSDF, which holds it
///        until the kernel has run.
/// @param blocks [in] device memory of slot_count block numbers; owned as keys.
/// @param slot_count [in] the hash's slots, a power of two >= 2.
/// @param coordinates [in] device memory of count coordinates, in blocks; owned by the caller,
///        which holds it until the kernel has run (sparse_tsdf::find_blocks()).
/// @param count [in] the coordinates, > 0.
/// @param found [out] device memory of count block numbers: each coordinate's, or no_block;
///        owned by the caller, as coordinates.
/// @note Runs on the stream of its launch, sparse_tsdf::find_blocks()'s, after the insertions it
///       is to see. A thread outside the coordinates writes nothing. Allocates nothing.
__global__ void __launch_bounds__(threads_per_block)
    find_kernel(const unsigned long long *__restrict__ keys,
                const std::uint32_t *__restrict__ blocks, std::uint32_t slot_count,
                const block_coordinate *__restrict__ coordinates, std::size_t count,
                std::uint32_t *__restrict__ found)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const std::uint32_t slot = find_slot(keys, slot_count, coordinates[i]);
    found[i] = slot == no_slot ? no_block : blocks[slot];
}

/// A frame's pose rounded to float, as the allocation's kernel applies it: p' = R p + t.
struct float_pose
{
    float3 row_x;       ///< the rotation's first row, camera frame to the models' frame
    float3 row_y;       ///< its second row
    float3 row_z;       ///< its third row
    float3 translation; ///< t, the camera's centre in the models' frame, in m
};

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
/// (docs/methods/sparse-tsdf.md, "The allocation near a frame's surface"). A segment with an end
/// that is not finite, or that crosses more than max_blocks_per_ray blocks once its ends are
/// clamped, is not walked and is counted in rays_not_walked.
/// @param points [in] device memory of size's points, rows dense, in the camera frame, in m; owned
///        by the caller's organized points, which hold them until the kernel has run.
/// @param valid [in] device memory of size's marks, 1 for a valid pixel; owned as points.
/// @param size [in] the picture's width and height, in px.
/// @param pose [in] the frame's pose, camera frame to the models' frame.
/// @param blocks_per_m [in] 1 / (8 v), the blocks in a metre.
/// @param mu_min_m [in] mu_min v, the least truncation distance, in m, > 0.
/// @param noise_factor [in] k c, in 1/m, > 0: mu(z) = max(mu_min v, k c z^2).
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
__global__ void __launch_bounds__(pixel_block_width *pixel_block_height)
    allocate_kernel(const float3 *__restrict__ points, const std::uint8_t *__restrict__ valid,
                    frame_size size, float_pose pose, float blocks_per_m, float mu_min_m,
                    float noise_factor, block_hash_view hash, std::uint32_t *__restrict__ stamps,
                    std::uint32_t *__restrict__ frame_slots, std::uint32_t frame)
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
    const float mu = fmaxf(mu_min_m, noise_factor * z * z);
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
/// @return empty when the voxel size and each value of the truncation's rule lie in their ranges;
///         else the reason.
/// @note Allocates for the reason; throws std::bad_alloc when that fails.
std::string parameters_refusal(const sparse_tsdf_parameters &parameters)
{
    const truncation_parameters &t = parameters.truncation;
    if (within(parameters.voxel_size_m, min_voxel_size_m, max_voxel_size_m) &&
        t.noise_per_m > 0.0f && t.noise_per_m <= max_noise_per_m &&
        within(t.noise_multiple, min_noise_multiple, max_noise_multiple) &&
        within(t.min_voxels, min_truncation_voxels, max_truncation_voxels))
        return {};
    return "the voxel size (" + std::to_string(parameters.voxel_size_m) +
           " m) or the truncation's rule (c " + std::to_string(t.noise_per_m) + " /m, k " +
           std::to_string(t.noise_multiple) + ", mu_min " + std::to_string(t.min_voxels) +
           " voxels) cannot be used: the voxel size from " + std::to_string(min_voxel_size_m) +
           " to " + std::to_string(max_voxel_size_m) + " m, c above 0 up to " +
           std::to_string(max_noise_per_m) + " /m, k from " + std::to_string(min_noise_multiple) +
           " to " + std::to_string(max_noise_multiple) + ", mu_min from " +
           std::to_string(min_truncation_voxels) + " to " + std::to_string(max_truncation_voxels) +
           " voxels";
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
    const block_hash_view hash{m_keys.data(), m_blocks.data(), m_counts.data(), m_layout.slot_count,
                               m_layout.capacity_blocks};
    return launch_kernel("the insertion of blocks", &insert_kernel,
                         grid_for_count(coordinates.size()), dim3(threads_per_block), stream, hash,
                         coordinates.data(), coordinates.size());
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
                         dim3(threads_per_block), stream, m_keys.data(), m_blocks.data(),
                         m_layout.slot_count, coordinates.data(), coordinates.size(),
                         blocks.data());
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
        return {false, "the pose cannot be used: its position and orientation rounded to float "
                       "must be finite and its orientation not zero"};
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
    const block_hash_view hash{m_keys.data(), m_blocks.data(), m_counts.data(), m_layout.slot_count,
                               m_layout.capacity_blocks};
    const float voxel_size_m = m_parameters.voxel_size_m;
    const truncation_parameters &t = m_parameters.truncation;
    const dim3 block(pixel_block_width, pixel_block_height);
    // This frame's stamp: the count of allocations before it, below every bit 1 (the @pre).
    const std::uint32_t frame = m_frames_allocated++;
    return launch_kernel(
        "the allocation's kernel", &allocate_kernel, grid_for(points.size(), block), block, stream,
        points.points().data(), points.validity_mask().data(), points.size(), *rounded,
        1.0f / (8.0f * voxel_size_m), t.min_voxels * voxel_size_m, t.noise_multiple * t.noise_per_m,
        hash, m_stamps.data(), m_frame_slots.data(), frame);
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

sparse_tsdf_layout sparse_tsdf::layout() const noexcept
{
    return m_layout;
}
} // namespace gca
