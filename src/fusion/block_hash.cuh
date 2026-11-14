// SPDX-License-Identifier: 0BSD
/// @file block_hash.cuh
/// The hash of blocks as device code sees it: the insertion and the lookup of one block by its
/// coordinates, the one copy every kernel of the fusion module calls when it inserts or finds a
/// block (docs/methods/sparse-tsdf.md, "The hash of blocks"). Device code, header only, for the
/// module's .cu files.

#pragma once

#include "fusion/sparse_tsdf.hpp"

#include <cstdint>

namespace gca
{
/// The key of an empty slot: every bit 1, which no block's key is, as its top bit is 0
/// (block_key()).
constexpr unsigned long long empty_key = ~0ULL;

/// The slot of a coordinate the hash does not hold.
constexpr std::uint32_t no_slot = 0xFFFFFFFFu;

/// The hash's arrays and sizes as an inserting kernel takes them, by value: pointers into a
/// sparse TSDF's device memory, which it owns and keeps while the kernel runs.
struct block_hash_view
{
    unsigned long long *keys;      ///< slot_count keys, empty_key for an empty slot
    std::uint32_t *blocks;         ///< slot_count block numbers, no_block for a slot without one
    block_counts *counts;          ///< the counts of claims and refusals
    std::uint32_t slot_count;      ///< the slots, a power of two >= 2
    std::uint32_t capacity_blocks; ///< the blocks the pool holds, >= 1
};

/// Inserts one block by its coordinates: tries the slots from the home slot on, swapping the
/// empty key for the block's key atomically (atomicCAS). The one thread whose swap succeeds takes
/// the next block's number (atomicAdd on the claims) and writes it as the slot's block when the
/// pool holds that block, else leaves the slot without one; a thread that finds the key already
/// there does nothing. Out of range, or after slot_count slots without room, the coordinate is
/// refused and counted. Wait-free: it ends after at most slot_count swaps, whatever the other
/// threads do, and waits for none.
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
/// @param keys [in] the hash's slot_count keys; device memory owned by the sparse TSDF, which
///        holds it while the caller's kernel runs.
/// @param slot_count [in] the hash's slots, a power of two >= 2.
/// @param b [in] the coordinates, in blocks.
/// @return the slot that holds the block's key, whose number may be no_block (refused for a
///         used-up pool); no_slot when the hash does not hold it or it is out of range.
/// @note Reads the keys with plain loads: the insertions it is to see ran in an earlier kernel on
///       the same stream (insert_block()'s note). Device code; allocates nothing.
__device__ inline std::uint32_t find_slot(const unsigned long long *keys, std::uint32_t slot_count,
                                          const block_coordinate &b)
{
    if (!in_key_range(b))
        return no_slot;
    const unsigned long long key = block_key(b);
    std::uint32_t slot = home_slot(b, slot_count);
    for (std::uint32_t tried = 0; tried < slot_count; tried++)
    {
        const unsigned long long seen = keys[slot];
        if (seen == key)
            return slot;
        if (seen == empty_key)
            return no_slot;
        slot = (slot + 1) & (slot_count - 1);
    }
    return no_slot;
}
} // namespace gca
