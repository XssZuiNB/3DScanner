// SPDX-License-Identifier: 0BSD
/// @file sparse_tsdf.cu
/// The sparse TSDF's hash of blocks and its pool: the division of the memory budget, the request
/// and clearing of the memory, the kernels that insert and find blocks, and the reading of the
/// usage (see sparse_tsdf.hpp; the method: docs/methods/sparse-tsdf.md).

#include "fusion/sparse_tsdf.hpp"

#include "fusion/block_hash.cuh"
#include "gpu/gpu_messages.hpp"
#include "gpu/kernel_launch.cuh"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

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
    const sparse_tsdf_layout layout = layout_for_budget(parameters.memory_budget_bytes);
    request_result<sparse_tsdf> result;
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
    result.usage = usage;
    return result;
}

sparse_tsdf_layout sparse_tsdf::layout() const noexcept
{
    return m_layout;
}
} // namespace gca
