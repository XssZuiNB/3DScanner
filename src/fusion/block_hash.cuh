// SPDX-License-Identifier: 0BSD
/// @file block_hash.cuh
/// The hash of blocks as device code sees it: the insertion and the lookup of one block by its
/// coordinates, and the block a slot holds, the one copy every kernel of the fusion module calls
/// when it inserts, finds or reads a block (docs/methods/sparse-tsdf.md, "The hash of blocks"), so
/// that the hash's layout stays behind these functions. Device code, header only, for the module's
/// .cu files.

#pragma once

#include "fusion/sparse_tsdf.hpp"

#include <cstdint>

#include <cuda/atomic>

namespace gca
{
/// The key of an empty slot: every bit 1, which no block's key is, as its top bit is 0
/// (block_key()).
constexpr unsigned long long empty_key = ~0ULL;

/// The slot of a coordinate the hash does not hold.
constexpr std::uint32_t no_slot = 0xFFFFFFFFu;

/// The hash's arrays and sizes as every kernel of the module takes them, by value, the one view of
/// the hash that the functions below read and write: pointers into a sparse TSDF's device memory,
/// which it owns and keeps while the kernel runs. A kernel that only finds blocks reads through
/// it and writes nothing (find_slot(), find_block(), block_in_slot()).
struct block_hash_view
{
    unsigned long long *keys;      ///< slot_count keys, empty_key for an empty slot
    std::uint32_t *blocks;         ///< slot_count block numbers, no_block for a slot without one
    block_counts *counts;          ///< the counts of claims and refusals
    std::uint32_t slot_count;      ///< the slots, a power of two >= 2
    std::uint32_t capacity_blocks; ///< the blocks the pool holds, >= 1
};

/// Inserts one block by its coordinates: tries the slots from the home slot on, swapping the
/// empty key for the block's key atomically (atomicCAS). Each slot's key is read first by a
/// relaxed atomic load: a slot that holds a key holds it for good, so a slot found holding this
/// key ends the insertion and one holding another is passed, both without the swap, which only an
/// empty slot takes. That spares the atomic for a block already held, which most of a frame's
/// rays find in the allocation near its surface (the times with and without: docs/HARDWARE.md,
/// "The per-frame path's stages"). The one thread whose swap succeeds takes the next block's
/// number (atomicAdd on the claims) and writes it as the slot's block when the pool holds that
/// block, else leaves the slot without one; a thread that finds the key already there does
/// nothing. Out of range, or after slot_count slots without room, the coordinate is refused and
/// counted. Wait-free: it ends after at most slot_count slots, a load and at most one swap each,
/// whatever the other threads do, and waits for none.
/// @param hash [in] the hash; its arrays are written as above.
/// @param b [in] the coordinates, in blocks.
/// @return the slot that holds the block's key, or no_slot when refused.
/// @note A trap for every caller: the key becomes visible to the other threads of the same
///       launch before its block's number, which the claiming thread writes by a plain store
///       after its swap. A thread that finds the key in the launch that inserts it may read
///       no_block, the slot's cleared value, as its number. A block's number is read only in a
///       kernel queued after the inserting one on the same stream, which sees every write of it.
///       Device code; allocates nothing.
__device__ inline std::uint32_t insert_block(const block_hash_view &hash, const block_coordinate &b)
{
    if (!in_key_range(b))
    {
        atomicAdd(&hash.counts->coordinates_refused, 1u);
        return no_slot;
    }
    const unsigned long long key = block_key(b);
    std::uint32_t slot = home_slot(b, hash.slot_count);
    for (std::uint32_t tried = 0; tried < hash.slot_count; tried++)
    {
        const unsigned long long held =
            cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(hash.keys[slot])
                .load(cuda::memory_order_relaxed);
        if (held == key)
            return slot;
        if (held != empty_key)
        {
            slot = (slot + 1) & (hash.slot_count - 1);
            continue;
        }
        const unsigned long long seen = atomicCAS(&hash.keys[slot], empty_key, key);
        if (seen == empty_key)
        {
            // This thread put the key there, the one thread to do so.
            const std::uint32_t number = atomicAdd(&hash.counts->claims, 1u);
            if (number < hash.capacity_blocks)
                hash.blocks[slot] = number;
            return slot;
        }
        if (seen == key)
            return slot;
        slot = (slot + 1) & (hash.slot_count - 1);
    }
    atomicAdd(&hash.counts->coordinates_refused, 1u);
    return no_slot;
}

/// Finds the slot of one block by its coordinates: the slots from the home slot on, until its
/// key, an empty slot or all slots.
/// @param hash [in] the hash; read only.
/// @param b [in] the coordinates, in blocks.
/// @return the slot that holds the block's key, whose number may be no_block (refused for a
///         used-up pool); no_slot when the hash does not hold it or it is out of range.
/// @note Reads the keys with plain loads: the insertions it is to see ran in an earlier kernel on
///       the same stream (insert_block()'s note). Device code; allocates nothing.
__device__ inline std::uint32_t find_slot(const block_hash_view &hash, const block_coordinate &b)
{
    if (!in_key_range(b))
        return no_slot;
    const unsigned long long key = block_key(b);
    std::uint32_t slot = home_slot(b, hash.slot_count);
    for (std::uint32_t tried = 0; tried < hash.slot_count; tried++)
    {
        const unsigned long long seen = hash.keys[slot];
        if (seen == key)
            return slot;
        if (seen == empty_key)
            return no_slot;
        slot = (slot + 1) & (hash.slot_count - 1);
    }
    return no_slot;
}

/// Finds the block of one coordinate: its slot (find_slot()) and that slot's block's number.
/// @param hash [in] the hash; read only.
/// @param b [in] the coordinates, in blocks.
/// @return the block's number, in [0, capacity_blocks); no_block when the hash holds no block for
///         it (never inserted, refused, or out of range).
/// @note Plain loads, as find_slot(): the insertions it is to see ran in an earlier kernel on the
///       same stream. Device code; allocates nothing.
__device__ inline std::uint32_t find_block(const block_hash_view &hash, const block_coordinate &b)
{
    const std::uint32_t slot = find_slot(hash, b);
    return slot == no_slot ? no_block : hash.blocks[slot];
}

/// A block as a slot of the hash holds it: its coordinates and its number in the pool.
struct held_block
{
    block_coordinate coordinate; ///< its coordinates, in blocks; meaningless without a number
    std::uint32_t number;        ///< its number in the pool, or no_block
};

/// Reads the block a slot holds, its coordinates from the slot's key (coordinate_of_key()).
/// @param hash [in] the hash; read only.
/// @param slot [in] the slot, below the hash's slot_count.
/// @return the block; its number no_block when the slot is empty or its key has no block (refused
///         for a used-up pool).
/// @note Plain loads: the insertions it is to see ran in an earlier kernel on the same stream
///       (insert_block()'s note). Device code; allocates nothing.
__device__ inline held_block block_in_slot(const block_hash_view &hash, std::uint32_t slot)
{
    const unsigned long long key = hash.keys[slot];
    if (key == empty_key)
        return {{0, 0, 0}, no_block};
    return {coordinate_of_key(key), hash.blocks[slot]};
}
} // namespace gca
