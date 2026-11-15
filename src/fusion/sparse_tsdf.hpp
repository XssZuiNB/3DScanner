// SPDX-License-Identifier: 0BSD
/// @file sparse_tsdf.hpp
/// The sparse TSDF (docs/adr/0004-sparse-tsdf-model.md): blocks of 8 x 8 x 8 voxels near the
/// surfaces, found by their coordinates through a hash on the GPU, taken from a pool; all of its
/// device memory together is the scan's memory budget. Built so far: the hash of blocks, the pool,
/// the allocation of the blocks near a frame's surface at its pose, the integration of a frame into
/// the voxels of those blocks, and the extraction of the surface's colored points with normals. A
/// block is inserted and found by its coordinates, many at once from many threads; a frame's
/// organized points at a pose insert every block their band of the truncation distance passes
/// through and list them for the frame; the integration updates each voxel of the listed blocks
/// with the frame's depth and color; the extraction places a point on every edge of the voxel grid
/// whose two voxels' distances differ in sign. A pool too small says so in its usage, a buffer of
/// points too small in the extraction's result, both values. The device side of the hash, which
/// every kernel of the module calls, is fusion/block_hash.cuh. The method, its rules and their
/// sources stand in docs/methods/sparse-tsdf.md. All its memory comes from the pool that closes
/// (gpu/gpu_pool.hpp) before a scan starts; its operations allocate nothing.

#pragma once

#include "geometry/organized_points.hpp"
#include "gpu/gpu_array.hpp"
#include "gpu/gpu_frame.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/pinned_buffer.hpp"
#include "gpu/step_result.hpp"
#include "types/camera_pose.hpp"

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

/// The bits each component of a block's coordinates takes in its key: 21, so that three fit one
/// 64-bit word with its top bit 0 (docs/methods/sparse-tsdf.md, "The hash of blocks"). The key's
/// layout, x in the top field, then y, then z, stands in block_key() and coordinate_of_key().
constexpr unsigned int key_component_bits = 21;

/// The least component of a block's coordinates the hash stores: -2^20, so that each component
/// less it takes key_component_bits bits.
constexpr std::int32_t block_coordinate_min = -(std::int32_t{1} << (key_component_bits - 1));

/// The greatest component of a block's coordinates the hash stores: 2^20 - 1.
constexpr std::int32_t block_coordinate_max = (std::int32_t{1} << (key_component_bits - 1)) - 1;

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
/// (docs/methods/sparse-tsdf.md, "The hash of blocks"); coordinate_of_key() undoes it.
/// @param b [in] the coordinates, in blocks.
/// @return the key.
/// @pre in_key_range(b); otherwise two coordinates may share a key, so callers check first.
__host__ __device__ constexpr std::uint64_t block_key(const block_coordinate &b) noexcept
{
    // Each component minus the least is in [0, 2^21) for a coordinate in range.
    const std::uint64_t x = static_cast<std::uint64_t>(std::int64_t{b.x} - block_coordinate_min);
    const std::uint64_t y = static_cast<std::uint64_t>(std::int64_t{b.y} - block_coordinate_min);
    const std::uint64_t z = static_cast<std::uint64_t>(std::int64_t{b.z} - block_coordinate_min);
    return (x << (2 * key_component_bits)) | (y << key_component_bits) | z;
}

/// Gives the coordinates a key stands for, the inverse of block_key().
/// @param key [in] a block's key, block_key() of a coordinate in range.
/// @return the coordinates, in blocks, each in [block_coordinate_min, block_coordinate_max].
__host__ __device__ constexpr block_coordinate coordinate_of_key(std::uint64_t key) noexcept
{
    constexpr std::uint64_t field = (std::uint64_t{1} << key_component_bits) - 1;
    return {static_cast<std::int32_t>((key >> (2 * key_component_bits)) & field) +
                block_coordinate_min,
            static_cast<std::int32_t>((key >> key_component_bits) & field) + block_coordinate_min,
            static_cast<std::int32_t>(key & field) + block_coordinate_min};
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

/// The ranges the request takes the voxel size, the truncation's rule and the far depth from,
/// inclusive (docs/methods/sparse-tsdf.md, "The truncation distance", which states why)
/// [?: a choice, the user's, 2026-11-14]: the voxel size from half the object classes' finest,
/// 1 mm, to twice their coarsest, 8 mm; c up to twice the datasheet's bound, 0.01 /m; k from 1 to
/// 5 standard deviations; mu_min from 1 to 8 voxels; the far depth over the camera's depth range,
/// 0.25 m to 10 m [S: datasheet, docs/HARDWARE.md]. Besides, the request refuses a combination
/// with the memory budget whose one frame could fill the hash (sparse_tsdf::request()).
constexpr float min_voxel_size_m = 0.0005f;
constexpr float max_voxel_size_m = 0.016f;
constexpr float max_noise_per_m = 0.02f;
constexpr float min_noise_multiple = 1.0f;
constexpr float max_noise_multiple = 5.0f;
constexpr float min_truncation_voxels = 1.0f;
constexpr float max_truncation_voxels = 8.0f;
constexpr float nearest_max_depth_m = 0.25f;
constexpr float farthest_max_depth_m = 10.0f;

/// The most blocks the allocation walks along one ray's band: a ray whose band crosses more is
/// not walked and is counted (docs/methods/sparse-tsdf.md, "The allocation near a frame's
/// surface", which gives the most the starting values need, 50) [?: a choice].
constexpr std::uint32_t max_blocks_per_ray = 64;

/// The most observations a voxel's running average counts, W_max (docs/methods/sparse-tsdf.md,
/// "Integration, and the weight of an observation") [?: initial]: once a voxel holds as many, a
/// new observation counts 1 / (W_max + 1).
constexpr std::uint16_t max_voxel_weight = 64;

/// A point of the surface as the extraction gives it (docs/methods/sparse-tsdf.md, "The
/// extraction"), on the GPU: where the distance changes sign between two neighbouring voxels.
struct surface_point
{
    float3 position_m; ///< the point, in the models' frame, in m
    float3 normal;     ///< the surface's unit normal there, in the models' frame, pointing from
                       ///< negative to positive distances, towards the cameras that saw it
    uchar3 color;      ///< red, green and blue, 0 to 255, the frame's color order
};
static_assert(sizeof(surface_point) == 28, "a point is 28 bytes, the buffer's request says");

/// The truncation distance's rule (docs/methods/sparse-tsdf.md, "The truncation distance"): at the
/// measured depth z, mu(z) = max(mu_min v, k sigma(z)) with the depth noise sigma(z) = c z^2, v the
/// voxel size.
struct truncation_parameters
{
    float noise_per_m;    ///< c, the depth noise's coefficient, sigma(z) = c z^2, in 1/m; finite,
                          ///< > 0 and <= max_noise_per_m
    float noise_multiple; ///< k, the standard deviations of the noise within mu, from
                          ///< min_noise_multiple to max_noise_multiple
    float min_voxels;     ///< mu_min, the least truncation distance, in voxels, from
                          ///< min_truncation_voxels to max_truncation_voxels
};

/// The sparse TSDF's parameters: its memory, its one length, its truncation distance and the far
/// depth of the frames it takes. Its operations all take them from the sparse TSDF, which keeps
/// them from its request. The scan fills them: the budget, the voxel size and the far depth from
/// the object class (scan/object_class.hpp), the live model's voxel size or the final models'.
struct sparse_tsdf_parameters
{
    std::uint64_t memory_budget_bytes; ///< all of the sparse TSDF's device memory, in bytes;
                                       ///< the object class's value
    float voxel_size_m;                ///< v, a voxel's edge, in m, from min_voxel_size_m to
                                       ///< max_voxel_size_m
    truncation_parameters truncation;  ///< the truncation distance's rule
    float max_depth_m; ///< z_max, the farthest depth a frame contributes, along the camera's z,
                       ///< in m, from nearest_max_depth_m to farthest_max_depth_m: the object
                       ///< class's; the allocation does not walk a deeper point
};

/// How a memory budget is divided: the blocks of the pool and the slots of the hash.
struct sparse_tsdf_layout
{
    std::uint32_t capacity_blocks; ///< the blocks the pool holds; 0 when the budget holds none
    std::uint32_t slot_count;      ///< the hash's slots, the least power of two >= 4 blocks
    std::uint64_t device_bytes;    ///< the device memory of all of it, in bytes, <= the budget
};

/// Divides a memory budget: the largest number of blocks B, at most 2^29, for which
/// 4100 B + 16 S + 16 <= budget, S the least power of two >= 4 B (docs/methods/sparse-tsdf.md,
/// "The pool and the memory budget").
/// @param memory_budget_bytes [in] the budget, in bytes.
/// @return B, S and the bytes they take; all 0 when the budget holds no block (below 4180 bytes).
sparse_tsdf_layout layout_for_budget(std::uint64_t memory_budget_bytes) noexcept;

/// The counts on the GPU, from the sparse TSDF's request on.
struct block_counts
{
    std::uint32_t claims;              ///< the distinct coordinates put into the hash, each one
                                       ///< the claim of the next block's number
    std::uint32_t coordinates_refused; ///< insertions refused, the coordinate out of range or
                                       ///< the hash full
    std::uint32_t listed_slots;        ///< the slots the last allocation near a frame's surface
                                       ///< touched, each once; the frame's list holds the first
                                       ///< capacity_blocks of them
    std::uint32_t rays_not_walked;     ///< the rays the allocations did not walk: a point deeper
                                       ///< than the far depth, a band across more than
                                       ///< max_blocks_per_ray blocks, or an end that is not finite
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
    std::uint32_t listed_slots;        ///< the slots on the list of the last allocation near a
                                       ///< frame's surface, each once, <= capacity_blocks
    std::uint32_t unlisted_slots;      ///< the slots that allocation touched past the list's
                                       ///< room: counted, not listed; > 0 only once the pool is
                                       ///< used up
    std::uint32_t rays_not_walked;     ///< the rays the allocations near a frame's surface did
                                       ///< not walk, from the request on (block_counts)

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

class sparse_tsdf;

/// The hash's one view, as every kernel of the module takes it: defined in fusion/block_hash.cuh.
struct block_hash_view;

/// A frame's pose rounded to float and checked, as the allocation and the integration apply it:
/// defined in sparse_tsdf.cu.
struct float_pose;

/// The buffer an extraction writes its points into, on the GPU, of a stated size, with the count
/// of the points found and the pinned buffer that count is read back through. Its memory comes
/// from the pool that closes. Move-only. Not synchronized: one thread at a time uses one.
class surface_points
{
public:
    /// An empty buffer: no memory; an extraction into it is refused.
    surface_points() noexcept = default;

    /// Requests the buffer's memory from the pool: room for a number of points, the count on the
    /// GPU and the pinned buffer it is read back through.
    /// @param pool [in,out] the pool; open, else the request is refused as closed.
    /// @param capacity [in] the points the buffer holds, > 0.
    /// @return granted, with a buffer whose points are not set until an extraction; or the first
    ///         of its requests that was refused, with its status and message
    ///         (gpu_pool::request_array()), a capacity of 0 among them.
    /// @note Allocates device memory (cudaMalloc, twice: 28 bytes a point, and 4 bytes) and
    ///       page-locked host memory (cudaMallocHost, 4 bytes), so it is called before the pool is
    ///       closed; whether CUDA's allocators wait for the GPU is not stated by CUDA [?].
    ///       Allocates for the message when refused; a failed allocation ends the program
    ///       (std::terminate): the function is noexcept. Two threads may not request from one
    ///       pool at once.
    static request_result<surface_points> request(gpu_pool &pool, std::uint32_t capacity) noexcept;

    /// A buffer is not copied: a copy allocates.
    surface_points(const surface_points &) = delete;
    surface_points &operator=(const surface_points &) = delete;

    /// Takes over another buffer's memory without allocating.
    /// @param other [in,out] the buffer moved from; left empty.
    /// @note Allocates nothing and waits for nothing.
    surface_points(surface_points &&other) noexcept = default;

    /// Takes over another buffer's memory; this one's is freed first.
    /// @param other [in,out] the buffer moved from; left empty. Moving onto itself changes
    ///        nothing.
    /// @return this buffer.
    /// @pre No work queued on the GPU uses this one's memory any more (gpu_array's move
    ///      assignment says what happens when some does).
    /// @note Frees with cudaFree and cudaFreeHost, which may wait for the GPU (gpu/gpu_array.hpp,
    ///       gpu/pinned_buffer.hpp). Allocates nothing.
    surface_points &operator=(surface_points &&other) noexcept = default;

    /// Frees the memory.
    /// @pre No work queued on the GPU uses it any more (as for the move assignment).
    /// @note Frees with cudaFree and cudaFreeHost, which may wait for the GPU.
    ~surface_points() = default;

    /// Gives the points the buffer holds.
    /// @return its capacity; 0 when empty.
    std::uint32_t capacity() const noexcept;

    /// Gives the points for the GPU to read or for a copy to the host.
    /// @return the array of capacity() points; the first ones the last extraction wrote, as its
    ///         result says, the rest not set. Empty when the buffer is. Valid as long as the
    ///         buffer holds it.
    const gpu_array<surface_point> &points() const noexcept;

private:
    friend class sparse_tsdf; ///< the extraction writes the points and the count

    gpu_array<surface_point> m_points; ///< the points
    gpu_array<std::uint32_t> m_count;  ///< one element: the points the extraction found
    pinned_buffer m_count_on_host;     ///< the count as the last extraction read it back
};

/// What an extraction gave: the points it found and wrote, or why not.
struct extraction_result
{
    bool done = false;         ///< whether every point found was written into the buffer
    std::uint32_t found = 0;   ///< the points found; 0 when the extraction did not run
    std::uint32_t written = 0; ///< the points written, the buffer's first: found, or the
                               ///< buffer's capacity when that is smaller
    std::string message;       ///< why not done, in plain English; empty when done
};

/// The sparse TSDF's hash of blocks and its pool, on the GPU, with its parameters. Its memory comes
/// from the pool that closes when it is requested, before a scan; its insertions, lookups,
/// allocations near a frame's surface, integrations, extractions and readings of the usage
/// allocate nothing. One exists at a time (docs/methods/sparse-tsdf.md, "The pool and the memory
/// budget"). Move-only. Not synchronized: each member function's note says whether two threads may
/// call it at once; the work it queues goes to one stream at a time, the stream the request queued
/// its clearing on or one ordered after it.
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
    /// frame), the counts and every voxel to 0, unobserved: distance, weight and color 0, the
    /// state the integration's first observation of a voxel averages from
    /// (docs/methods/sparse-tsdf.md, "What a voxel holds"); the list is written before it is
    /// read. Keeps the parameters for its operations.
    /// @param pool [in,out] the pool; open, else the request is refused as closed.
    /// @param parameters [in] the memory budget, the voxel size, the truncation's rule and the far
    ///        depth.
    /// @param stream [in] the CUDA stream of the current device the clearing is queued on; owned
    ///        by the caller. Work queued on it later sees the cleared memory.
    /// @return granted, with a sparse TSDF that holds no block; failure with the reason, before
    ///         any request, when the voxel size, a value of the truncation's rule or the far depth
    ///         is not finite or lies outside its range (min_voxel_size_m and the others above),
    ///         when the budget holds no block (below 4180 bytes), or when one frame could touch as
    ///         many blocks as the hash has room for once the pool is used up, slot_count -
    ///         capacity_blocks, or more: the bound of a surface facing the camera that fills its
    ///         picture of 1280x800 at the far depth (docs/methods/sparse-tsdf.md, "One frame's
    ///         blocks"); as that room shrinks while the blocks grow within one slot_count, a budget
    ///         one block larger can be refused where a smaller one is granted. Else the first of
    ///         its requests that was refused, with its status and message
    ///         (gpu_pool::request_array()): a closed pool gives closed, with nothing allocated,
    ///         to parameters that pass every check above, so it checks them without device
    ///         memory. Or failure with CUDA's error, then cleared, when the clearing could not be
    ///         queued.
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

    /// Allocates the blocks near a frame's surface at its pose (docs/methods/sparse-tsdf.md, "The
    /// allocation near a frame's surface"), one GPU thread for each pixel: for each valid pixel
    /// with depth z, the segment of its ray between the depths z - mu(z) and z + mu(z) (the
    /// truncation's rule, kept from the request), carried into the models' frame by the pose;
    /// every block the segment passes through is inserted as insert_blocks() inserts a
    /// coordinate, refusals and a used-up pool counted alike. A point deeper than the far depth
    /// (sparse_tsdf_parameters::max_depth_m) is not walked, so that the frame stays within the
    /// bound the request checked, whatever range the points were made with. A segment with an
    /// end that is not finite is not walked; the others' ends are clamped to one block past the
    /// hash's range on each side, and one that then crosses more than max_blocks_per_ray blocks
    /// is not walked either: all three are counted in rays_not_walked, so that no thread takes
    /// more than max_blocks_per_ray steps. Lists the frame's blocks: every
    /// slot the frame touched, once, the first capacity_blocks of them on the frame's list and
    /// the rest counted, not listed. Each call is a frame of its own: allocating the same points
    /// at the same pose again changes no block and lists the same slots again. A block that the
    /// band crosses but no pixel's ray does can be missed; docs/methods/sparse-tsdf.md states how
    /// far such a miss lies from an allocated block. The keys' trap of insert_blocks() holds: a
    /// block's number is read only by work queued after this call.
    /// @param points [in] a frame's organized points and validity mask, in the camera frame, in
    ///        m; device memory owned by the caller, which holds them until the work queued here
    ///        has run. Their points written on stream or on a stream ordered before it.
    /// @param pose [in] the frame's pose: camera frame to the models' frame, as
    ///        types/camera_pose.hpp says; its position finite and in m, its orientation a finite
    ///        quaternion, not zero, normalized here. Both are rounded to float.
    /// @param stream [in] the CUDA stream of the current device the work is queued on, ordered
    ///        after the request's clearing; owned by the caller. Work queued on it later sees the
    ///        blocks and the frame's list; read_usage() reads the list's counts.
    /// @return done when the work is queued; not done with the reason when refused before it:
    ///         this sparse TSDF is empty, the organized points are, or the pose rounded to float
    ///         is not finite or its orientation is zero. When refused, nothing is queued and the
    ///         counts are left as they were. Not done with CUDA's error, which is then cleared:
    ///         when CUDA's last error is set before the call, by an earlier call, with nothing
    ///         queued; when CUDA refuses the reset of the list's count, with nothing queued; when
    ///         CUDA refuses the launch, for points whose rows need more than 65535 blocks of 8 rows
    ///         [S: CUDA C++ Programming Guide, "Compute Capabilities", the grid's y dimension],
    ///         with the list's count then reset to 0.
    /// @pre At most 2^32 - 1 allocations over the sparse TSDF's life, 4.5 years at 30 fps
    ///      [S: computed]: each takes the next number of a 32-bit count from 0 as its frame's
    ///      stamp, and the stamp with every bit 1, the count's next value after them, means "no
    ///      frame". Not checked; a scan requests its own sparse TSDF.
    /// @note A cudaMemsetAsync of 4 bytes, the list's count, then one kernel on stream, which
    ///       writes the hash's slots, the stamps, the list and the counts and no voxel. A thread
    ///       takes at most max_blocks_per_ray steps, each an insertion that probes until it finds
    ///       its key or an empty slot, at most slot_count slots when the hash is full. Waits for
    ///       nothing: cudaPeekAtLastError before and cudaGetLastError after ask only whether an
    ///       error is set. Allocates nothing when done; allocates for the message when not, and a
    ///       failed allocation ends the program (std::terminate): the function is noexcept. Not
    ///       while another thread uses this sparse TSDF.
    step_result allocate_near_surface(const organized_points &points, const camera_pose &pose,
                                      cudaStream_t stream) noexcept;

    /// Integrates a frame at its pose (docs/methods/sparse-tsdf.md, "Integration, and the weight of
    /// an observation"): allocates the blocks near its surface first (allocate_near_surface(),
    /// with the same points and pose), then updates every voxel of the blocks on the frame's list,
    /// one GPU thread for each. Voxel g's centre, (g + 1/2) v in the models' frame, is carried
    /// into the camera frame, q = R^T (c - t), and projected to its nearest pixel; a pixel that is
    /// valid and no deeper than the far depth, with depth z, observes the voxel when
    /// eta = z - q_z >= -mu(z): the distance d = min(eta, mu(z)) / v, in voxels, and the pixel's
    /// color join the voxel's running averages with the weight 1, the voxel's weight limited to
    /// max_voxel_weight. A voxel behind the camera, outside the picture, at an invalid or too deep
    /// pixel, or more than mu(z) behind the surface is left as it was. A slot on the list without
    /// a block (a used-up pool) is skipped. Each voxel is updated by one thread: the list holds
    /// each slot once.
    /// @param frame [in] the GPU frame the points were deprojected from, of their size: its color
    ///        picture (RGB8) and its intrinsics (in px); device memory owned by the caller, which
    ///        holds it until the work queued here has run. Uploaded on stream or on a stream
    ///        ordered before it.
    /// @param points [in] the frame's organized points and validity mask, in the camera frame, in
    ///        m, as allocate_near_surface() takes them; owned as frame.
    /// @param pose [in] the frame's pose, camera frame to the models' frame, as
    ///        allocate_near_surface() takes it.
    /// @param stream [in] the CUDA stream of the current device the work is queued on, ordered
    ///        after the request's clearing; owned by the caller. Work queued on it later sees the
    ///        voxels, the blocks and the frame's list.
    /// @return done when the work is queued; not done with the reason when refused before any
    ///         of it, with nothing queued: this sparse TSDF is empty, the organized points are,
    ///         the GPU frame holds no frame, its size is not the points', its intrinsics are not
    ///         usable (types/camera_param.hpp, usable_intrinsics()), or the pose is not, as
    ///         allocate_near_surface() refuses it. Not done with CUDA's error, which is then
    ///         cleared, as allocate_near_surface() returns it for the allocation; and when CUDA
    ///         refuses the integration's launch, the allocation queued then, its blocks inserted
    ///         and listed, their voxels left as they were.
    /// @pre The points were deprojected from the frame (organized_points::deproject()), so that a
    ///      pixel's point and color are the same pixel's; not checked. The same @pre on the count
    ///      of allocations as allocate_near_surface().
    /// @note allocate_near_surface()'s memset and kernel, then one kernel on stream, of a fixed
    ///       grid of thread blocks of 512 threads, which go through the frame's list and read its
    ///       count on the GPU, so that nothing is read back; it writes only the voxels of listed
    ///       blocks with a number. Waits for nothing. Allocates nothing when done; allocates for
    ///       the message when not, and a failed allocation ends the program (std::terminate): the
    ///       function is noexcept. Not while another thread uses this sparse TSDF.
    step_result integrate(const gpu_frame &frame, const organized_points &points,
                          const camera_pose &pose, cudaStream_t stream) noexcept;

    /// Extracts the surface's colored points with normals (docs/methods/sparse-tsdf.md, "The
    /// extraction"): for every block the hash holds, each voxel g and each of its neighbours
    /// g + e, e along x, y and z, found through the hash when in another block, both observed,
    /// whose distances D_a and D_b differ in sign (D_a >= 0 > D_b, or D_a < 0 <= D_b), a point at
    /// the fraction s = D_a / (D_a - D_b) of the way from g's centre to the neighbour's; its normal
    /// the gradient of the distance by central differences at both voxels, interpolated at s and
    /// normalized, so that it points from negative to positive; its color the two colors
    /// interpolated at s and rounded. A crossing whose central differences need an unobserved or
    /// missing voxel gives no point. The points come in no stated order: two runs give the same
    /// set in another order.
    /// @param out [in,out] the buffer the points are written into, the first written of them, and
    ///        its count; owned by the caller.
    /// @param stream [in] the CUDA stream of the current device the work is queued on and waited
    ///        for, ordered after the integrations it is to see; owned by the caller.
    /// @return done with the points found and written, all of them; not done, with the reason:
    ///         when the buffer holds fewer than were found, with both counts, the buffer's first
    ///         capacity points written; when refused before any work, this sparse TSDF or the
    ///         buffer is empty, with nothing queued and both counts 0; with CUDA's error, then
    ///         cleared, when CUDA's last error was set before the call (nothing queued), refused
    ///         the reset of the count, the launch or the copy, or failed while the work ran, with
    ///         both counts 0 and the buffer's points unspecified.
    /// @note A cudaMemsetAsync of the count, one kernel over every slot of the hash, of a fixed
    ///       grid of thread blocks of 512 threads, a copy of 4 bytes into the pinned buffer, then
    ///       the one written wait, cudaStreamSynchronize(stream), which waits for all the stream's
    ///       earlier work. Not for the per-frame path. Allocates nothing when done; allocates for
    ///       the message when not, and a failed allocation ends the program (std::terminate): the
    ///       function is noexcept. Not while another thread uses this sparse TSDF or the buffer.
    extraction_result extract(surface_points &out, cudaStream_t stream) const noexcept;

    /// Reads the use of the pool and the hash back from the GPU, after the work queued on a
    /// stream. In the per-frame path it is called after the registration's own readback, so that
    /// its wait finds the stream's earlier work done and costs only its copy
    /// (docs/methods/sparse-tsdf.md, "The pool and the memory budget").
    /// @param stream [in] the CUDA stream of the current device whose work the reading follows;
    ///        owned by the caller.
    /// @return the usage; none, with the reason, when this sparse TSDF is empty, or with CUDA's
    ///         error, then cleared, when the copy or the wait failed. A failure gives no usage,
    ///         so it is never read as a used-up pool.
    /// @note One copy of 16 bytes, the four counts, into the pinned buffer on stream, then the
    ///       one written wait, cudaStreamSynchronize(stream): it waits for all the stream's earlier
    ///       work. Allocates nothing when done; allocates for the message when not, and a failed
    ///       allocation ends the program (std::terminate): the function is noexcept. Not while
    ///       another thread uses this sparse TSDF.
    usage_result read_usage(cudaStream_t stream) noexcept;

    /// Gives how the memory budget was divided.
    /// @return the blocks, the slots and the device bytes; all 0 when empty.
    sparse_tsdf_layout layout() const noexcept;

private:
    /// Queues the allocation near a frame's surface at a pose already rounded to float and found
    /// usable, as allocate_near_surface() describes it after its refusals: the reset of the list's
    /// count and the allocation's kernel.
    /// @param points [in] the frame's organized points, not empty; as allocate_near_surface().
    /// @param pose [in] the frame's pose rounded to float, its values finite.
    /// @param stream [in] the stream, as allocate_near_surface().
    /// @return done when queued; not done with CUDA's error, then cleared, when CUDA's last error
    ///         was set before the call or CUDA refused the reset (nothing queued) or the launch
    ///         (the list's count then reset to 0).
    /// @pre This sparse TSDF holds memory; not checked here, its callers refuse an empty one.
    /// @note As allocate_near_surface(). Not while another thread uses this sparse TSDF.
    step_result allocate_at(const organized_points &points, const float_pose &pose,
                            cudaStream_t stream) noexcept;

    /// Gives the hash's one view, which every kernel of the module takes (fusion/block_hash.cuh).
    /// @return pointers into this sparse TSDF's keys, block numbers and counts, and its slots and
    ///         blocks; null pointers and 0 when empty.
    /// @note The pointers are not const, for the kernels that insert: a const sparse TSDF hands the
    ///       view only to kernels that find blocks, which write nothing through it. Allocates
    ///       nothing and waits for nothing.
    block_hash_view hash_view() const noexcept;

    sparse_tsdf_layout m_layout{0, 0, 0};
    sparse_tsdf_parameters m_parameters{0, 0.0f, {0.0f, 0.0f, 0.0f}, 0.0f}; ///< from the request
    std::uint32_t m_frames_allocated = 0;   ///< the allocations near a frame's surface so far, the
                                            ///< next one's stamp
    gpu_array<unsigned long long> m_keys;   ///< each slot's key, or every bit 1 when empty
    gpu_array<std::uint32_t> m_blocks;      ///< each slot's block number, or no_block
    gpu_array<std::uint32_t> m_stamps;      ///< each slot's last frame, every bit 1 for none
    gpu_array<std::uint32_t> m_frame_slots; ///< the frame's list of slots, one for each block
    gpu_array<block_counts> m_counts;       ///< one element
    gpu_array<voxel> m_voxels;              ///< the pool: block n holds voxels 512 n to 512 n + 511
    pinned_buffer m_counts_on_host;         ///< the counts as the last reading copied them
};
} // namespace gca
