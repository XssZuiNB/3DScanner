// SPDX-License-Identifier: 0BSD
/// @file sparse_tsdf.hpp
/// The sparse TSDF (docs/adr/0004-sparse-tsdf-model.md): blocks of 8 x 8 x 8 voxels near the
/// surfaces, found by their coordinates through a hash on the GPU, taken from a pool; all of its
/// device memory together is the scan's memory budget. Built so far: the hash of blocks and the
/// pool. A block is inserted and found by its coordinates, many at once from many threads; a
/// pool too small says so in its usage, a value. The device side of the hash, which every kernel
/// of the module calls, is fusion/block_hash.cuh. The method, its rules and their sources stand in
/// docs/methods/sparse-tsdf.md. All its memory comes from the pool that closes (gpu/gpu_pool.hpp)
/// before a scan starts; its operations allocate nothing.

#pragma once

#include "gpu/gpu_array.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/pinned_buffer.hpp"
#include "gpu/step_result.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>
#include <string>

namespace gca
{
/// A block's coordinates: block b is the cube [8 b, 8 b + 8) v of the models' frame, v the voxel
/// size, and the point p lies in block floor(p / (8 v)) (docs/methods/sparse-tsdf.md, "Space,
/// voxels and blocks"). The hash stores a coordinate only when each component lies in
/// [block_coordinate_min, block_coordinate_max].
struct block_coordinate
{
    std::int32_t x; ///< along the models' x axis, in blocks
    std::int32_t y; ///< along the models' y axis, in blocks
    std::int32_t z; ///< along the models' z axis, in blocks
};

/// The least component of a block's coordinates the hash stores: -2^20, so that each component
/// takes 21 bits of the key [S: docs/methods/sparse-tsdf.md, "The hash of blocks"].
constexpr std::int32_t block_coordinate_min = -(std::int32_t{1} << 20);

/// The greatest component of a block's coordinates the hash stores: 2^20 - 1.
constexpr std::int32_t block_coordinate_max = (std::int32_t{1} << 20) - 1;

/// The voxels of a block: 8 x 8 x 8 [?: a choice, the block of Nießner et al. 2013, in
/// docs/methods/sparse-tsdf.md].
constexpr std::uint32_t voxels_per_block = 8 * 8 * 8;

/// The block number of a slot or a coordinate that has no block.
constexpr std::uint32_t no_block = 0xFFFFFFFFu;

/// A voxel (docs/methods/sparse-tsdf.md, "What a voxel holds"): 8 bytes, read and written in one
/// 64-bit access; one byte is the alignment's. A voxel with weight 0 is unobserved, and its
/// distance and color mean nothing.
struct alignas(8) voxel
{
    __half distance;      ///< the signed distance, in voxels: > 0 in front of the surface
    std::uint16_t weight; ///< the observations fused, up to a limit; 0: unobserved
    std::uint8_t red;     ///< the color's red, 0 to 255
    std::uint8_t green;   ///< the color's green, 0 to 255
    std::uint8_t blue;    ///< the color's blue, 0 to 255
};
static_assert(sizeof(voxel) == 8, "a voxel is 8 bytes");

/// Says whether the hash stores a block's coordinates.
/// @param b [in] the coordinates, in blocks.
/// @return true when x, y and z each lie in [block_coordinate_min, block_coordinate_max].
__host__ __device__ constexpr bool in_key_range(const block_coordinate &b) noexcept
{
    return b.x >= block_coordinate_min && b.x <= block_coordinate_max &&
           b.y >= block_coordinate_min && b.y <= block_coordinate_max &&
           b.z >= block_coordinate_min && b.z <= block_coordinate_max;
}

/// Gives a block's key: (x + 2^20) * 2^42 + (y + 2^20) * 2^21 + (z + 2^20), one 64-bit word
/// whose top bit is 0, a different one for every coordinate in range
/// (docs/methods/sparse-tsdf.md, "The hash of blocks").
/// @param b [in] the coordinates, in blocks.
/// @return the key.
/// @pre in_key_range(b); otherwise two coordinates may share a key, so callers check first.
__host__ __device__ constexpr std::uint64_t block_key(const block_coordinate &b) noexcept
{
    // Each component minus the least is in [0, 2^21) for a coordinate in range.
    const std::uint64_t x = static_cast<std::uint64_t>(std::int64_t{b.x} - block_coordinate_min);
    const std::uint64_t y = static_cast<std::uint64_t>(std::int64_t{b.y} - block_coordinate_min);
    const std::uint64_t z = static_cast<std::uint64_t>(std::int64_t{b.z} - block_coordinate_min);
    return (x << 42) | (y << 21) | z;
}

/// Gives a block's home slot, where the hash's search for it starts: the 64-bit finalizer of
/// MurmurHash3 applied to the block's key, its low bits taken [S: A. Appleby, MurmurHash3,
/// `fmix64`, in SMHasher, placed in the public domain].
/// @param b [in] the coordinates, in blocks.
/// @param slot_count [in] the hash's slots, a power of two >= 2.
/// @return the home slot, in [0, slot_count).
/// @pre in_key_range(b), as for block_key().
__host__ __device__ constexpr std::uint32_t home_slot(const block_coordinate &b,
                                                      std::uint32_t slot_count) noexcept
{
    std::uint64_t k = block_key(b);
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return static_cast<std::uint32_t>(k & (slot_count - 1));
}

/// The parameters of the sparse TSDF's memory.
struct sparse_tsdf_parameters
{
    std::uint64_t memory_budget_bytes; ///< all of the sparse TSDF's device memory, in bytes;
                                       ///< the object class's value
};

/// How a memory budget is divided: the blocks of the pool and the slots of the hash.
struct sparse_tsdf_layout
{
    std::uint32_t capacity_blocks; ///< the blocks the pool holds; 0 when the budget holds none
    std::uint32_t slot_count;      ///< the hash's slots, the least power of two >= 2 blocks
    std::uint64_t device_bytes;    ///< the device memory of all of it, in bytes, <= the budget
};

/// Divides a memory budget: the largest number of blocks B, at most 2^30, for which
/// 4100 B + 16 S + 12 <= budget, S the least power of two >= 2 B (docs/methods/sparse-tsdf.md,
/// "The pool and the memory budget").
/// @param memory_budget_bytes [in] the budget, in bytes.
/// @return B, S and the bytes they take; all 0 when the budget holds no block (below 4144 bytes).
sparse_tsdf_layout layout_for_budget(std::uint64_t memory_budget_bytes) noexcept;

/// The counts on the GPU, from the sparse TSDF's request on.
struct block_counts
{
    std::uint32_t claims;              ///< the distinct coordinates put into the hash, each one
                                       ///< the claim of the next block's number
    std::uint32_t coordinates_refused; ///< insertions refused, the coordinate out of range or
                                       ///< the hash full
    std::uint32_t listed_slots;        ///< the slots on the frame's list of blocks; its first
                                       ///< user is the allocation near a frame's surface
                                       ///< (docs/methods/sparse-tsdf.md)
};

/// The use of the pool and the hash, read back from the GPU; it exists only for a reading that
/// succeeded (usage_result).
struct sparse_tsdf_usage
{
    std::uint32_t capacity_blocks;     ///< the blocks the pool holds, > 0
    std::uint32_t blocks_in_use;       ///< the blocks handed out, <= capacity_blocks
    std::uint32_t blocks_refused;      ///< the distinct coordinates that found the pool used up;
                                       ///< each keeps its slot, without a block
    std::uint32_t coordinates_refused; ///< insertions refused, out of range or the hash full

    /// Gives the share of the pool in use.
    /// @return blocks_in_use / capacity_blocks, from 0 to 1.
    double share_in_use() const noexcept
    {
        return static_cast<double>(blocks_in_use) / static_cast<double>(capacity_blocks);
    }

    /// Says whether the pool is used up: the next new coordinate finds no block.
    /// @return true when every block is in use.
    bool used_up() const noexcept
    {
        return blocks_in_use == capacity_blocks;
    }
};

/// What a reading of the usage gave: the usage, or why not.
struct usage_result
{
    std::optional<sparse_tsdf_usage> usage; ///< the usage; none when it could not be read
    std::string message;                    ///< why not, in plain English; empty when read
};

/// The sparse TSDF's hash of blocks and its pool, on the GPU. Its memory comes from the pool that
/// closes when it is requested, before a scan; its insertions, lookups and readings of the usage
/// allocate nothing. One exists at a time (docs/methods/sparse-tsdf.md, "The pool and the memory
/// budget"). Move-only. Not synchronized: each member function's note says whether two threads
/// may call it at once; the work it queues goes to one stream at a time, the stream the request
/// queued its clearing on or one ordered after it.
class sparse_tsdf
{
public:
    /// An empty sparse TSDF: no memory; every operation on it is refused.
    sparse_tsdf() noexcept = default;

    /// Requests all of the sparse TSDF's device memory from the pool that closes, as the memory
    /// budget divides it (layout_for_budget()): the hash's keys and block numbers, each slot's
    /// stamp of the last frame that touched it, the frame's list of slots, the counts and the
    /// pool's voxels; and the pinned buffer the usage is read through. Queues their clearing on
    /// a stream: the keys to empty, the blocks' numbers to none, the stamps to every bit 1 (no
    /// frame), the counts and every voxel to 0, unobserved; the list is written before it is
    /// read. No code uses the stamps and the list yet: their first user is the allocation near a
    /// frame's surface; and no code reads the voxels yet: their first reader is the integration
    /// (docs/methods/sparse-tsdf.md).
    /// @param pool [in,out] the pool; open, else the request is refused as closed.
    /// @param parameters [in] the memory budget.
    /// @param stream [in] the CUDA stream of the current device the clearing is queued on; owned
    ///        by the caller. Work queued on it later sees the cleared memory.
    /// @return granted, with a sparse TSDF that holds no block; failure with the reason when the
    ///         budget holds no block (below 4144 bytes); or the first of its requests that was
    ///         refused, with its status and message (gpu_pool::request_array()); or failure with
    ///         CUDA's error, then cleared, when the clearing could not be queued.
    /// @note Allocates device memory (cudaMalloc, six times) and page-locked host memory
    ///       (cudaMallocHost), so it is called before the pool is closed; whether CUDA's
    ///       allocators wait for the GPU is not stated by CUDA [?]. Queues five cudaMemsetAsync
    ///       on stream and waits for nothing else. Allocates for the message when refused; a
    ///       failed allocation ends the program (std::terminate): the function is noexcept. Two
    ///       threads may not request from one pool at once (gpu_pool::request_array()).
    static request_result<sparse_tsdf> request(gpu_pool &pool,
                                               const sparse_tsdf_parameters &parameters,
                                               cudaStream_t stream) noexcept;

    /// A sparse TSDF is not copied: a copy allocates.
    sparse_tsdf(const sparse_tsdf &) = delete;
    sparse_tsdf &operator=(const sparse_tsdf &) = delete;

    /// Takes over another sparse TSDF's memory without allocating.
    /// @param other [in,out] the sparse TSDF moved from; left empty.
    /// @note Allocates nothing and waits for nothing.
    sparse_tsdf(sparse_tsdf &&other) noexcept;

    /// Takes over another sparse TSDF's memory; this one's is freed first.
    /// @param other [in,out] the sparse TSDF moved from; left empty. Moving onto itself changes
    ///        nothing.
    /// @return this sparse TSDF.
    /// @pre No work queued on the GPU uses this one's memory any more (gpu_array's move
    ///      assignment says what happens when some does).
    /// @note Frees with cudaFree and cudaFreeHost, which may wait for the GPU
    ///       (gpu/gpu_array.hpp, gpu/pinned_buffer.hpp). Allocates nothing.
    sparse_tsdf &operator=(sparse_tsdf &&other) noexcept;

    /// Frees the memory.
    /// @pre No work queued on the GPU uses it any more (as for the move assignment).
    /// @note Frees with cudaFree and cudaFreeHost, which may wait for the GPU.
    ~sparse_tsdf() = default;

    /// Inserts blocks by their coordinates, one GPU thread for each, all at once: a coordinate
    /// not in the hash yet takes a slot and, while the pool has one, the next block; one already
    /// there changes nothing, however many threads insert it. A coordinate out of range, or one
    /// that finds the hash full, is refused and counted; one that finds the pool used up keeps
    /// its slot without a block and is counted (docs/methods/sparse-tsdf.md, "The hash of
    /// blocks"). Wait-free: each insertion ends after at most slot_count swaps and waits for no
    /// other thread. A key is visible to the other threads of the launch before its block's
    /// number (insert_block() in fusion/block_hash.cuh): a block's number is read only by work
    /// queued after this call on the same stream, find_blocks() among it.
    /// @param coordinates [in] the coordinates, in blocks, in any order, the same one any number
    ///        of times; device memory owned by the caller, which holds it until the work queued
    ///        here has run.
    /// @param stream [in] the CUDA stream of the current device the work is queued on, ordered
    ///        after the request's clearing; owned by the caller. Work queued on it later, a
    ///        lookup among it, sees the blocks inserted.
    /// @return done when the work is queued; not done with the reason when refused before it:
    ///         this sparse TSDF is empty, or the coordinates are. When refused, nothing is
    ///         queued. Not done with CUDA's error, which is then cleared
    ///         (gpu/gpu_messages.hpp): when CUDA's last error is set before the launch, by an
    ///         earlier call, with nothing queued; when CUDA refuses the launch, for one with more
    ///         than 2^32 - 1 coordinates, which no device memory of this board holds
    ///         (gpu/kernel_launch.cuh).
    /// @note One kernel on stream, which writes the hash's slots and the counts and no voxel.
    ///       Waits for nothing: cudaPeekAtLastError before the launch and cudaGetLastError after
    ///       it ask only whether an error is set. Allocates nothing when done; allocates for the
    ///       message when not, and a failed allocation ends the program (std::terminate): the
    ///       function is noexcept. Not while another thread uses this sparse TSDF.
    step_result insert_blocks(const gpu_array<block_coordinate> &coordinates,
                              cudaStream_t stream) noexcept;

    /// Finds blocks by their coordinates, one GPU thread for each.
    /// @param coordinates [in] the coordinates, in blocks; device memory owned by the caller,
    ///        which holds it until the work queued here has run.
    /// @param blocks [out] for each coordinate, at its index, its block's number in [0,
    ///        capacity_blocks), or no_block when the hash holds no block for it (never inserted,
    ///        refused, or out of range); device memory of as many elements as the coordinates,
    ///        owned by the caller as the coordinates.
    /// @param stream [in] the CUDA stream of the current device the work is queued on, ordered
    ///        after the insertions it is to see; owned by the caller.
    /// @return done when the work is queued; not done with the reason when refused before it:
    ///         this sparse TSDF is empty, the coordinates are, or blocks has another number of
    ///         elements. When refused, nothing is queued and blocks is left as it was. Not done
    ///         with CUDA's error, which is then cleared, as insert_blocks() says.
    /// @note One kernel on stream, which reads the hash and writes blocks. Waits for nothing, as
    ///       insert_blocks(). Allocates nothing when done; allocates for the message when not,
    ///       and a failed allocation ends the program (std::terminate): the function is
    ///       noexcept. Two threads may call it at once only with different blocks, and while no
    ///       thread inserts or reads the usage.
    step_result find_blocks(const gpu_array<block_coordinate> &coordinates,
                            gpu_array<std::uint32_t> &blocks, cudaStream_t stream) const noexcept;

    /// Reads the use of the pool and the hash back from the GPU, after the work queued on a
    /// stream. In the per-frame path it is called after the registration's own readback, so that
    /// its wait finds the stream's earlier work done and costs only its copy
    /// (docs/methods/sparse-tsdf.md, "The pool and the memory budget").
    /// @param stream [in] the CUDA stream of the current device whose work the reading follows;
    ///        owned by the caller.
    /// @return the usage; none, with the reason, when this sparse TSDF is empty, or with CUDA's
    ///         error, then cleared, when the copy or the wait failed. A failure gives no usage,
    ///         so it is never read as a used-up pool.
    /// @note One copy of 12 bytes into the pinned buffer on stream, then the one written wait,
    ///       cudaStreamSynchronize(stream): it waits for all the stream's earlier work. Allocates
    ///       nothing when done; allocates for the message when not, and a failed allocation ends
    ///       the program (std::terminate): the function is noexcept. Not while another thread uses
    ///       this sparse TSDF.
    usage_result read_usage(cudaStream_t stream) noexcept;

    /// Gives how the memory budget was divided.
    /// @return the blocks, the slots and the device bytes; all 0 when empty.
    sparse_tsdf_layout layout() const noexcept;

private:
    sparse_tsdf_layout m_layout{0, 0, 0};
    gpu_array<unsigned long long> m_keys;   ///< each slot's key, or every bit 1 when empty
    gpu_array<std::uint32_t> m_blocks;      ///< each slot's block number, or no_block
    gpu_array<std::uint32_t> m_stamps;      ///< each slot's last frame, every bit 1 for none
    gpu_array<std::uint32_t> m_frame_slots; ///< the frame's list of slots, one for each block
    gpu_array<block_counts> m_counts;       ///< one element
    gpu_array<voxel> m_voxels;              ///< the pool: block n holds voxels 512 n to 512 n + 511
    pinned_buffer m_counts_on_host;         ///< the counts as the last reading copied them
};
} // namespace gca
