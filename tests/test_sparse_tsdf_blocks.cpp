// SPDX-License-Identifier: 0BSD
/// @file test_sparse_tsdf_blocks.cpp
/// The fusion module's hash of blocks and its pool (fusion/sparse_tsdf.hpp), fed with block
/// coordinates the test builds; the method is docs/methods/sparse-tsdf.md. Each case requests
/// its sparse TSDF and its arrays, then closes the pool, so that every operation runs with the
/// pool closed; the largest sparse TSDF here holds 16385 blocks, 69275668 bytes of device memory
/// [S: computed], far below the object classes' memory budget; the budgets of 1.0 GB and 2.0 GB
/// are divided but not allocated. The coordinates go to the GPU and the numbers come back by
/// copies on the stream the kernels run on. The request's check of one frame against the hash's
/// room is test_sparse_tsdf_allocation.cpp's; the cases here take parameters it grants at their
/// budgets.
/// - The division of a budget: at the budget of exactly 1024 blocks and one byte below it, at
///   the smallest that holds one block and one byte below it, at 1.0 GB, at 2.0 GB and at 2^53
///   bytes.
/// - Collisions on purpose and duplicates: coordinates that share a home slot, among them one
///   run that wraps from the last slot to the first and runs into another group's home, each
///   inserted by 65 threads of one launch in a shuffled order: every coordinate is found, the
///   block numbers are distinct and inside the pool, and a coordinate never inserted that shares
///   their home slot is not found. Inserting the same set again changes no number and no count.
/// - Ten runs, each a new sparse TSDF given the same insertions: the same set of coordinates
///   with a block and the same counts; with a pool too small, the same counts.
/// - A pool too small: of 1536 distinct coordinates, exactly 1024 find a block, the numbers 0 to
///   1023 once each; 512 are counted as refused; the pool is used up, its share in use 1.
/// - A hash too small: of 522 distinct coordinates, 512 take the 512 slots, 10 are refused.
/// - The coordinates' range: its limits are inserted and found, and their keys give them back
///   (coordinate_of_key()); coordinates past them are refused, and the coordinates their keys would
///   alias are not found; once those aliases hold blocks, a lookup of the coordinates past the
///   range still finds none.
/// - Refusals as values, each leaving the counts as they were: an empty sparse TSDF, no
///   coordinates, room for another number of blocks, CUDA's last error left by an earlier call,
///   a budget that holds no block, a request to the closed pool; and the GPU arrays' own requests
///   and moves.
/// - The request's ranges of the voxel size, the truncation's rule and the far depth: granted at
///   their ends, refused one float step past them, at NaN, infinity and 0.

#include "check.hpp"
#include "fusion/sparse_tsdf.hpp"
#include "gpu/gpu_array.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/step_result.hpp"
#include "gpu_test_helpers.hpp"
#include "sparse_tsdf_helpers.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using gca_test::check;
using gca_test::check_no_cuda_error;
using gca_test::check_refusal;
using gca_test::check_usage;
using gca_test::create_stream;
using gca_test::find;
using gca_test::granted;
using gca_test::lookup;
using gca_test::make_lookup;
using gca_test::read_usage;

/// The budget of the cases' sparse TSDF: 1024 blocks and 4096 slots exactly, 1024 x 4100 +
/// 4096 x 16 + 16 bytes [S: computed, docs/methods/sparse-tsdf.md, "The pool and the memory
/// budget"].
constexpr std::uint64_t test_budget = 4'263'952;

/// The blocks and slots of test_budget [S: computed].
constexpr std::uint32_t test_blocks = 1024;
constexpr std::uint32_t test_slots = 4096;

/// A budget of 128 blocks and 512 slots, 128 x 4100 + 512 x 16 + 16 bytes [S: computed], whose
/// hash fills with 512 distinct coordinates; the smallest hash the cases' parameters are granted,
/// its room for one frame 512 - 128 = 384 blocks (with_budget()).
constexpr std::uint64_t small_budget = 533'008;

/// A budget of 256 blocks and 1024 slots, 256 x 4100 + 1024 x 16 + 16 bytes [S: computed], too
/// small for the 296 coordinates of the collisions' set.
constexpr std::uint64_t too_small_budget = 1'066'000;

/// A budget of 16385 blocks and 131072 slots, 16385 x 4100 + 131072 x 16 + 16 bytes
/// [S: computed], whose room for one frame, 114687 blocks, holds the bound of every request of
/// the ranges' ends that is granted: at most 98581 blocks [S: computed,
/// docs/methods/sparse-tsdf.md, "One frame's blocks"].
constexpr std::uint64_t ranges_budget = 69'275'668;

/// Gives a sparse TSDF's parameters with a budget, the method's starting values of the
/// truncation's rule (sparse_tsdf_helpers.hpp), and the coarsest voxel size and nearest far depth
/// the request takes, whose bound of one frame, 256 blocks [S: computed,
/// docs/methods/sparse-tsdf.md, "One frame's blocks"], fits the hash of each case's budget; the
/// hash and the pool do not use them otherwise.
/// @param budget [in] the memory budget, in bytes.
/// @return the parameters.
gca::sparse_tsdf_parameters with_budget(std::uint64_t budget)
{
    return {budget, gca::max_voxel_size_m, gca_test::method_truncation, gca::nearest_max_depth_m};
}

/// How many threads insert each coordinate of the collisions' set in one launch [?: a choice:
/// more than two warps' worth, so that one warp's 32 threads may all hold one coordinate, and
/// 296 x 65 = 19240 insertions, no multiple of the kernels' 256 threads, so that the last thread
/// block is partly outside them].
constexpr int copies = 65;

/// The seeds of the coordinates scattered and of the shuffles [?: a choice].
constexpr std::uint32_t scatter_seed = 20261114;
constexpr std::uint32_t shuffle_seed = 7;

/// Block coordinates on the host.
using coordinate_list = std::vector<gca::block_coordinate>;

/// A block's coordinates as a tuple, for sets.
using coordinate_tuple = std::tuple<std::int32_t, std::int32_t, std::int32_t>;

/// Gives a block's coordinates as a tuple.
/// @param b [in] the coordinates.
/// @return (x, y, z).
coordinate_tuple as_tuple(const gca::block_coordinate &b)
{
    return {b.x, b.y, b.z};
}

/// Inserts a lookup's coordinates and checks that the work was queued.
/// @param tsdf [in,out] the sparse TSDF.
/// @param inserted [in] the coordinates.
/// @param stream [in] the stream.
/// @param what [in] the case, named in the check line.
void insert(gca::sparse_tsdf &tsdf, const lookup &inserted, cudaStream_t stream,
            const std::string &what)
{
    const gca::step_result queued = tsdf.insert_blocks(inserted.coordinates, stream);
    check(queued.done, (what + ": insertion queued, " + queued.message).c_str());
}

/// What a lookup found: how many coordinates have a block, and whether their numbers are
/// distinct and inside the pool.
struct found_blocks
{
    std::size_t found = 0; ///< coordinates with a block
    bool distinct = true;  ///< no number given twice
    bool in_pool = true;   ///< every number below the pool's blocks
};

/// Examines the block numbers of a lookup.
/// @param numbers [in] the numbers, no_block for a coordinate without a block.
/// @param capacity_blocks [in] the pool's blocks.
/// @return what was found.
found_blocks examine(const std::vector<std::uint32_t> &numbers, std::uint32_t capacity_blocks)
{
    found_blocks result;
    std::set<std::uint32_t> seen;
    for (const std::uint32_t number : numbers)
    {
        if (number == gca::no_block)
            continue;
        result.found++;
        result.in_pool = result.in_pool && number < capacity_blocks;
        result.distinct = seen.insert(number).second && result.distinct;
    }
    return result;
}

/// Prints and checks what a lookup found.
/// @param what [in] the case, printed and named in the check lines.
/// @param numbers [in] the numbers found.
/// @param capacity_blocks [in] the pool's blocks.
/// @param expected_found [in] how many coordinates are to have a block.
void check_found(const std::string &what, const std::vector<std::uint32_t> &numbers,
                 std::uint32_t capacity_blocks, std::size_t expected_found)
{
    const found_blocks found = examine(numbers, capacity_blocks);
    std::printf("%s: %zu of %zu coordinates found, numbers %s and %s\n", what.c_str(), found.found,
                numbers.size(), found.distinct ? "distinct" : "NOT distinct",
                found.in_pool ? "inside the pool" : "NOT inside the pool");
    check(found.found == expected_found, (what + ": the coordinates expected found").c_str());
    check(found.distinct, (what + ": no block number given twice").c_str());
    check(found.in_pool, (what + ": every block number inside the pool").c_str());
}

/// Collects coordinates whose home slot in the cases' hash of test_slots slots is a given one,
/// walking a cube of 128 blocks a side in a fixed order.
/// @param slot [in] the home slot, below test_slots.
/// @param count [in] how many to collect.
/// @param taken [in,out] coordinates already used, skipped; the collected ones are added.
/// @return the coordinates; fewer than count when the cube holds fewer.
coordinate_list with_home_slot(std::uint32_t slot, std::size_t count,
                               std::set<coordinate_tuple> &taken)
{
    coordinate_list collected;
    for (std::int32_t x = -64; x < 64 && collected.size() < count; x++)
        for (std::int32_t y = -64; y < 64 && collected.size() < count; y++)
            for (std::int32_t z = -64; z < 64 && collected.size() < count; z++)
            {
                const gca::block_coordinate b{x, y, z};
                if (gca::home_slot(b, test_slots) == slot && taken.insert(as_tuple(b)).second)
                    collected.push_back(b);
            }
    return collected;
}

/// Draws distinct coordinates in a cube of ±1000 blocks from a fixed seed.
/// @param count [in] how many.
/// @param seed [in] the seed.
/// @param taken [in,out] coordinates already used, skipped; the drawn ones are added.
/// @return the coordinates.
coordinate_list scattered(std::size_t count, std::uint32_t seed, std::set<coordinate_tuple> &taken)
{
    std::mt19937 random(seed);
    std::uniform_int_distribution<std::int32_t> component(-1000, 1000);
    coordinate_list drawn;
    while (drawn.size() < count)
    {
        const gca::block_coordinate b{component(random), component(random), component(random)};
        if (taken.insert(as_tuple(b)).second)
            drawn.push_back(b);
    }
    return drawn;
}

/// Repeats each coordinate and shuffles them with a fixed seed.
/// @param distinct [in] the coordinates.
/// @param times [in] how many times each.
/// @param seed [in] the shuffle's seed.
/// @return distinct.size() * times coordinates.
coordinate_list repeated(const coordinate_list &distinct, int times, std::uint32_t seed)
{
    coordinate_list all;
    for (int t = 0; t < times; t++)
        all.insert(all.end(), distinct.begin(), distinct.end());
    std::mt19937 random(seed);
    std::shuffle(all.begin(), all.end(), random);
    return all;
}

/// The collisions' set: 24 coordinates with home slot 700; 24 with home slot 4095, the last,
/// whose run wraps to slot 0; 24 with home slot 0, which that run reaches; 24 with home slot 1;
/// and 200 scattered, 296 in all.
/// @param left_out [out] one more coordinate with home slot 700, not in the set.
/// @return the set.
coordinate_list collisions_set(gca::block_coordinate &left_out)
{
    std::set<coordinate_tuple> taken;
    coordinate_list set;
    for (const std::uint32_t slot : {700u, test_slots - 1, 0u, 1u})
    {
        const coordinate_list group = with_home_slot(slot, 24, taken);
        check(group.size() == 24, "collisions: 24 coordinates found for each home slot");
        set.insert(set.end(), group.begin(), group.end());
    }
    const coordinate_list spare = with_home_slot(700, 1, taken);
    check(spare.size() == 1, "collisions: one more coordinate of home slot 700");
    left_out = spare.empty() ? gca::block_coordinate{0, 0, 0} : spare.front();
    const coordinate_list rest = scattered(200, scatter_seed, taken);
    set.insert(set.end(), rest.begin(), rest.end());
    return set;
}

/// Requests a sparse TSDF and checks that it was granted.
/// @param pool [in,out] the pool, open.
/// @param budget [in] the memory budget, in bytes.
/// @param stream [in] the stream its clearing is queued on.
/// @param what [in] the case, named in the check line.
/// @return the sparse TSDF; an empty one and a failed check when refused.
gca::sparse_tsdf requested(gca::gpu_pool &pool, std::uint64_t budget, cudaStream_t stream,
                           const std::string &what)
{
    return granted(what + ": the sparse TSDF",
                   gca::sparse_tsdf::request(pool, with_budget(budget), stream));
}

/// Checks that the pool, closed after the case's requests, was asked for nothing more.
/// @param what [in] the case, named in the check line.
/// @param pool [in] the pool.
void check_closed_pool(const std::string &what, const gca::gpu_pool &pool)
{
    check(pool.closed() && pool.refused_while_closed() == 0,
          (what + ": the closed pool asked for nothing").c_str());
}

/// Compares two layouts.
/// @param a [in] one layout.
/// @param b [in] the other.
/// @return whether blocks, slots and bytes are equal.
bool same_layout(const gca::sparse_tsdf_layout &a, const gca::sparse_tsdf_layout &b)
{
    return a.capacity_blocks == b.capacity_blocks && a.slot_count == b.slot_count &&
           a.device_bytes == b.device_bytes;
}

/// The division of budgets into blocks and slots, against values computed by hand, and that a
/// granted sparse TSDF has its budget's layout.
/// @param stream [in] the stream.
void check_layout(cudaStream_t stream)
{
    struct layout_case
    {
        const char *what;
        std::uint64_t budget;
        gca::sparse_tsdf_layout expected; // [S: computed by hand, 4100 B + 16 S + 16]
    };
    const layout_case cases[] = {
        {"layout, 1024 blocks exactly", test_budget, {test_blocks, test_slots, test_budget}},
        {"layout, one byte below 1024 blocks", test_budget - 1, {1023, 4096, 4'259'852}},
        {"layout, one block exactly", 4180, {1, 4, 4180}},
        {"layout, one byte below one block", 4179, {0, 0, 0}},
        {"layout, 1.0 GB", 1'000'000'000, {239'810, 1'048'576, 999'998'232}},
        {"layout, 2.0 GB", 2'000'000'000, {479'620, 2'097'152, 1'999'996'448}},
        {"layout, 2^53 bytes, 2^29 blocks at most",
         std::uint64_t{1} << 53,
         {std::uint32_t{1} << 29, std::uint32_t{1} << 31, 2'235'530'477'584}},
    };
    for (const layout_case &c : cases)
    {
        const gca::sparse_tsdf_layout layout = gca::layout_for_budget(c.budget);
        std::printf("%s: %llu bytes give %u blocks, %u slots, %llu bytes\n", c.what,
                    static_cast<unsigned long long>(c.budget), layout.capacity_blocks,
                    layout.slot_count, static_cast<unsigned long long>(layout.device_bytes));
        check(same_layout(layout, c.expected), c.what);
    }
    gca::gpu_pool pool;
    const gca::sparse_tsdf tsdf = requested(pool, test_budget, stream, "layout");
    check(same_layout(tsdf.layout(), gca::layout_for_budget(test_budget)),
          "layout: the granted sparse TSDF has its budget's layout");
}

/// Coordinates that share home slots, each inserted by many threads of one launch, all found;
/// inserting them again changes nothing.
/// @param stream [in] the stream.
void check_collisions(cudaStream_t stream)
{
    const std::string what = "collisions";
    gca::block_coordinate left_out{};
    const coordinate_list set = collisions_set(left_out);
    gca::gpu_pool pool;
    gca::sparse_tsdf tsdf = requested(pool, test_budget, stream, what);
    lookup all = make_lookup(pool, repeated(set, copies, shuffle_seed), stream, what);
    lookup again = make_lookup(pool, repeated(set, copies, shuffle_seed + 1), stream, what);
    lookup distinct = make_lookup(pool, set, stream, what);
    lookup absent = make_lookup(pool, {left_out}, stream, what);
    pool.close();

    insert(tsdf, all, stream, what);
    std::printf("%s: %zu coordinates, %zu insertions in one launch\n", what.c_str(), set.size(),
                all.host.size());
    const std::vector<std::uint32_t> first = find(tsdf, distinct, stream, what);
    check_found(what, first, test_blocks, set.size());
    const gca::sparse_tsdf_usage usage = read_usage(tsdf, stream, what);
    check_usage(what, usage, {test_blocks, static_cast<std::uint32_t>(set.size()), 0, 0, 0, 0, 0});
    const std::vector<std::uint32_t> not_there = find(tsdf, absent, stream, what);
    check(not_there.size() == 1 && not_there.front() == gca::no_block,
          "collisions: a coordinate of a full run's home slot, never inserted, is not found");

    insert(tsdf, again, stream, what + ", again");
    const std::vector<std::uint32_t> second = find(tsdf, distinct, stream, what + ", again");
    check(second == first, "collisions, again: every block number as before");
    check_usage(what + ", again", read_usage(tsdf, stream, what + ", again"), usage);
    check_closed_pool(what, pool);
}

/// The same insertions into ten new sparse TSDFs give the same set of blocks and the same counts;
/// into a pool too small, the same counts.
/// @param stream [in] the stream.
void check_ten_runs(cudaStream_t stream)
{
    gca::block_coordinate left_out{};
    const coordinate_list set = collisions_set(left_out);
    const coordinate_list all = repeated(set, copies, shuffle_seed);
    std::set<coordinate_tuple> first_set;
    std::vector<std::uint32_t> first_numbers;
    int same_numbers = 0;
    int same_runs = 0;
    int same_full_counts = 0;
    for (int run = 0; run < 10; run++)
    {
        const std::string what = "ten runs, run " + std::to_string(run + 1);
        gca::gpu_pool pool;
        gca::sparse_tsdf tsdf = requested(pool, test_budget, stream, what);
        gca::sparse_tsdf too_small = requested(pool, too_small_budget, stream, what);
        lookup inserted = make_lookup(pool, all, stream, what);
        lookup distinct = make_lookup(pool, set, stream, what);
        pool.close();

        insert(tsdf, inserted, stream, what);
        const std::vector<std::uint32_t> numbers = find(tsdf, distinct, stream, what);
        std::set<coordinate_tuple> with_block;
        for (std::size_t i = 0; i < numbers.size(); i++)
            if (numbers[i] != gca::no_block && numbers[i] < test_blocks)
                with_block.insert(as_tuple(set[i]));
        const gca::sparse_tsdf_usage usage = read_usage(tsdf, stream, what);
        const bool counts_as_expected = usage.blocks_in_use == set.size() &&
                                        usage.blocks_refused == 0 && usage.coordinates_refused == 0;
        if (run == 0)
        {
            first_set = with_block;
            first_numbers = numbers;
        }
        same_runs +=
            (with_block == first_set && with_block.size() == set.size() && counts_as_expected) ? 1
                                                                                               : 0;
        same_numbers += numbers == first_numbers ? 1 : 0;

        insert(too_small, inserted, stream, what + ", pool too small");
        const gca::sparse_tsdf_usage full = read_usage(too_small, stream, what);
        const std::vector<std::uint32_t> full_numbers =
            find(too_small, distinct, stream, what + ", pool too small");
        const found_blocks found = examine(full_numbers, 256);
        same_full_counts +=
            (full.blocks_in_use == 256 && full.blocks_refused == set.size() - 256 &&
             full.coordinates_refused == 0 && found.found == 256 && found.distinct && found.in_pool)
                ? 1
                : 0;
        check_closed_pool(what, pool);
    }
    std::printf("ten runs: %d of 10 with the same %zu coordinates with a block and the same "
                "counts; %d of 10 with every block number as in the first (not required)\n",
                same_runs, set.size(), same_numbers);
    check(same_runs == 10, "ten runs: the same set of blocks and counts in each");
    std::printf("ten runs, pool too small: %d of 10 with 256 blocks in use, %zu refused, 256 "
                "found, numbers distinct and inside the pool\n",
                same_full_counts, set.size() - 256);
    check(same_full_counts == 10, "ten runs, pool too small: the same counts in each");
}

/// More distinct coordinates than the pool holds: exactly its blocks are handed out, each once,
/// and the rest is counted as refused.
/// @param stream [in] the stream.
void check_pool_too_small(cudaStream_t stream)
{
    const std::string what = "pool too small";
    std::set<coordinate_tuple> taken;
    const coordinate_list set = scattered(1536, scatter_seed + 1, taken);
    const coordinate_list one_more = scattered(1, scatter_seed + 2, taken);
    gca::gpu_pool pool;
    gca::sparse_tsdf tsdf = requested(pool, test_budget, stream, what);
    lookup all = make_lookup(pool, repeated(set, 4, shuffle_seed), stream, what);
    lookup distinct = make_lookup(pool, set, stream, what);
    lookup late = make_lookup(pool, one_more, stream, what);
    pool.close();

    insert(tsdf, all, stream, what);
    const gca::sparse_tsdf_usage usage = read_usage(tsdf, stream, what);
    check_usage(what, usage, {test_blocks, test_blocks, 512, 0, 0, 0, 0});
    std::printf("%s: share in use %.3f, used up: %s\n", what.c_str(), usage.share_in_use(),
                usage.used_up() ? "yes" : "no");
    check(usage.share_in_use() == 1.0 && usage.used_up(),
          "pool too small: the share in use is 1 and the pool used up");
    std::vector<std::uint32_t> numbers = find(tsdf, distinct, stream, what);
    check_found(what, numbers, test_blocks, test_blocks);
    std::vector<std::uint32_t> handed_out;
    std::copy_if(numbers.begin(), numbers.end(), std::back_inserter(handed_out),
                 [](std::uint32_t n) { return n != gca::no_block; });
    std::sort(handed_out.begin(), handed_out.end());
    bool each_once = handed_out.size() == test_blocks;
    for (std::size_t i = 0; each_once && i < handed_out.size(); i++)
        each_once = handed_out[i] == i;
    check(each_once, "pool too small: the numbers 0 to 1023 handed out once each");

    insert(tsdf, late, stream, what + ", one more");
    check_usage(what + ", one more", read_usage(tsdf, stream, what + ", one more"),
                {test_blocks, test_blocks, 513, 0, 0, 0, 0});
    const std::vector<std::uint32_t> late_number = find(tsdf, late, stream, what + ", one more");
    check(late_number.size() == 1 && late_number.front() == gca::no_block,
          "pool too small: a coordinate inserted after the pool was used up has no block");
    check_closed_pool(what, pool);
}

/// More distinct coordinates than the hash has slots: each slot takes one, the rest is refused.
/// @param stream [in] the stream.
void check_hash_full(cudaStream_t stream)
{
    const std::string what = "hash full";
    std::set<coordinate_tuple> taken;
    const coordinate_list set = scattered(522, scatter_seed + 3, taken);
    gca::gpu_pool pool;
    gca::sparse_tsdf tsdf = requested(pool, small_budget, stream, what);
    lookup distinct = make_lookup(pool, set, stream, what);
    pool.close();
    check(same_layout(tsdf.layout(), {128, 512, small_budget}), "hash full: 128 blocks, 512 slots");

    insert(tsdf, distinct, stream, what);
    check_usage(what, read_usage(tsdf, stream, what), {128, 128, 384, 10, 0, 0, 0});
    check_found(what, find(tsdf, distinct, stream, what), 128, 128);
    insert(tsdf, distinct, stream, what + ", again");
    check_usage(what + ", again", read_usage(tsdf, stream, what + ", again"),
                {128, 128, 384, 20, 0, 0, 0});
    check_closed_pool(what, pool);
}

/// The limits of the coordinates' range are stored, and their keys give them back; coordinates past
/// them are refused, and the coordinates their keys would alias are not found.
/// @param stream [in] the stream.
void check_key_range(cudaStream_t stream)
{
    const std::string what = "key range";
    constexpr std::int32_t lo = gca::block_coordinate_min;
    constexpr std::int32_t hi = gca::block_coordinate_max;
    const coordinate_list inside = {{lo, lo, lo}, {hi, hi, hi}, {lo, hi, 0}, {0, lo, hi}};
    const coordinate_list outside = {{hi + 1, 0, 0},
                                     {0, hi + 1, 0},
                                     {0, 0, hi + 1},
                                     {lo - 1, 0, 0},
                                     {0, 0, std::numeric_limits<std::int32_t>::min()},
                                     {std::numeric_limits<std::int32_t>::max(), 0, 0}};
    // The keys of (0, 2^20, 0) and (0, 0, 2^20) carry into the next field: they would be the
    // keys of these two coordinates [S: block_key()'s fields of 21 bits].
    const coordinate_list aliased = {{1, lo, 0}, {0, 1, lo}};
    // The key gives the coordinates back, at the range's corners too (coordinate_of_key()).
    bool round_trip = true;
    for (const gca::block_coordinate &b : inside)
    {
        const gca::block_coordinate back = gca::coordinate_of_key(gca::block_key(b));
        round_trip = round_trip && back.x == b.x && back.y == b.y && back.z == b.z;
    }
    check(round_trip, "key range: each key gives its coordinates back");
    coordinate_list inserted = inside;
    inserted.insert(inserted.end(), outside.begin(), outside.end());
    gca::gpu_pool pool;
    gca::sparse_tsdf tsdf = requested(pool, test_budget, stream, what);
    lookup all = make_lookup(pool, inserted, stream, what);
    lookup would_alias = make_lookup(pool, aliased, stream, what);
    pool.close();

    insert(tsdf, all, stream, what);
    check_usage(what, read_usage(tsdf, stream, what), {test_blocks, 4, 0, 6, 0, 0, 0});
    const std::vector<std::uint32_t> numbers = find(tsdf, all, stream, what);
    bool as_expected = numbers.size() == inserted.size();
    for (std::size_t i = 0; as_expected && i < numbers.size(); i++)
        as_expected = (i < inside.size()) == (numbers[i] != gca::no_block);
    check(as_expected, "key range: the limits found, the coordinates past them not");
    const std::vector<std::uint32_t> alias_numbers = find(tsdf, would_alias, stream, what);
    check(alias_numbers.size() == 2 && alias_numbers[0] == gca::no_block &&
              alias_numbers[1] == gca::no_block,
          "key range: the coordinates a key past the range would alias are not found");

    // The aliases now hold blocks; a lookup of the coordinates past the range, whose keys would
    // be theirs, still finds none.
    insert(tsdf, would_alias, stream, what + ", the aliases");
    check_usage(what + ", the aliases", read_usage(tsdf, stream, what),
                {test_blocks, 6, 0, 6, 0, 0, 0});
    const std::vector<std::uint32_t> after = find(tsdf, all, stream, what + ", the aliases");
    bool past_range_found = false;
    for (std::size_t i = inside.size(); i < after.size(); i++)
        past_range_found = past_range_found || after[i] != gca::no_block;
    check(after.size() == inserted.size() && !past_range_found,
          "key range: with the aliases inserted, the coordinates past the range are not found");
    check_found(what + ", the aliases", find(tsdf, would_alias, stream, what), test_blocks, 2);
    check_closed_pool(what, pool);
}

/// The request's ranges of the voxel size, the truncation's rule and the far depth, as the method
/// states them (docs/methods/sparse-tsdf.md): requests with the values at their ends are granted,
/// three of them, as one request with every value at its high end has a bound of one frame no
/// budget of this board holds [S: computed, docs/methods/sparse-tsdf.md, "One frame's blocks"]:
/// every value at its low end; c at its high end and the others at their low ends; every value
/// but c at its high end, c at its least. A value one float step past an end, NaN, infinity or 0
/// is refused before any request to the pool.
/// @param pool [in,out] the pool, open.
/// @param stream [in] the stream.
void check_parameter_ranges(gca::gpu_pool &pool, cudaStream_t stream)
{
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    constexpr float infinity = std::numeric_limits<float>::infinity();
    const auto below = [](float end) { return std::nextafter(end, 0.0f); };
    const auto above = [](float end) { return std::nextafter(end, infinity); };
    // The ends as docs/methods/sparse-tsdf.md, "The truncation distance", states them, apart from
    // the module's constants, so that a range moved outward there is caught here.
    constexpr float least_voxel_m = 0.0005f;
    constexpr float greatest_voxel_m = 0.016f;
    constexpr float greatest_c = 0.02f;
    constexpr float least_k = 1.0f;
    constexpr float greatest_k = 5.0f;
    constexpr float least_mu_min = 1.0f;
    constexpr float greatest_mu_min = 8.0f;
    constexpr float nearest_far_m = 0.25f;
    constexpr float farthest_far_m = 10.0f;
    // c has no low end but 0, excluded: the least positive float stands for it [S: float32].
    const float least_c = std::numeric_limits<float>::denorm_min();
    const gca::sparse_tsdf_parameters low = {
        ranges_budget, least_voxel_m, {least_c, least_k, least_mu_min}, nearest_far_m};
    gca::sparse_tsdf_parameters high_c = low;
    high_c.truncation.noise_per_m = greatest_c;
    const gca::sparse_tsdf_parameters high = {
        ranges_budget, greatest_voxel_m, {least_c, greatest_k, greatest_mu_min}, farthest_far_m};
    for (const auto &[what, parameters] :
         {std::pair{"ranges, every value at its low end", low},
          std::pair{"ranges, c at its high end, the others at their low ends", high_c},
          std::pair{"ranges, every value but c at its high end", high}})
    {
        const gca::request_result<gca::sparse_tsdf> made =
            gca::sparse_tsdf::request(pool, parameters, stream);
        check(made.status == gca::request_status::granted &&
                  made.granted.layout().slot_count == 131072,
              (std::string(what) + ": granted, " + made.message).c_str());
    }
    // Each unusable value replaces one of the cases' parameters (with_budget()).
    const gca::sparse_tsdf_parameters base = with_budget(small_budget);
    const auto voxel = [&base](float value) {
        gca::sparse_tsdf_parameters p = base;
        p.voxel_size_m = value;
        return p;
    };
    const auto noise = [&base](float value) {
        gca::sparse_tsdf_parameters p = base;
        p.truncation.noise_per_m = value;
        return p;
    };
    const auto multiple = [&base](float value) {
        gca::sparse_tsdf_parameters p = base;
        p.truncation.noise_multiple = value;
        return p;
    };
    const auto least_truncation = [&base](float value) {
        gca::sparse_tsdf_parameters p = base;
        p.truncation.min_voxels = value;
        return p;
    };
    const auto far_depth = [&base](float value) {
        gca::sparse_tsdf_parameters p = base;
        p.max_depth_m = value;
        return p;
    };
    const struct
    {
        const char *what;
        gca::sparse_tsdf_parameters parameters;
    } unusable[] = {
        {"ranges, a voxel size of 0", voxel(0.0f)},
        {"ranges, a voxel size of NaN", voxel(nan)},
        {"ranges, a voxel size below the least", voxel(below(least_voxel_m))},
        {"ranges, a voxel size above the greatest", voxel(above(greatest_voxel_m))},
        {"ranges, c of 0", noise(0.0f)},
        {"ranges, c of -0.01 /m", noise(-0.01f)},
        {"ranges, c above the greatest", noise(above(greatest_c))},
        {"ranges, c of NaN", noise(nan)},
        {"ranges, k below the least", multiple(below(least_k))},
        {"ranges, k above the greatest", multiple(above(greatest_k))},
        {"ranges, k of infinity", multiple(infinity)},
        {"ranges, mu_min below the least", least_truncation(below(least_mu_min))},
        {"ranges, mu_min above the greatest", least_truncation(above(greatest_mu_min))},
        {"ranges, mu_min of 0", least_truncation(0.0f)},
        {"ranges, a far depth below the nearest", far_depth(below(nearest_far_m))},
        {"ranges, a far depth above the farthest", far_depth(above(farthest_far_m))},
        {"ranges, a far depth of NaN", far_depth(nan)},
        {"ranges, a far depth of 0", far_depth(0.0f)},
        {"ranges, a far depth of infinity", far_depth(infinity)},
    };
    for (const auto &u : unusable)
    {
        const gca::request_result<gca::sparse_tsdf> refused =
            gca::sparse_tsdf::request(pool, u.parameters, stream);
        check_refusal(u.what,
                      refused.status == gca::request_status::failure &&
                          refused.granted.layout().capacity_blocks == 0,
                      refused.message, refused.message.find("cannot be used") != std::string::npos);
    }
}

/// Refusals as values, each leaving the counts as they were and CUDA's last error clear.
/// @param stream [in] the stream.
void check_refusals(cudaStream_t stream)
{
    const std::string what = "refusals";
    std::set<coordinate_tuple> taken;
    gca::gpu_pool pool;
    gca::sparse_tsdf tsdf = requested(pool, test_budget, stream, what);
    lookup some = make_lookup(pool, scattered(8, scatter_seed + 4, taken), stream, what);
    gca::gpu_array<std::uint32_t> other_room =
        granted(what + ": an array of 7", pool.request_array<std::uint32_t>(7));
    gca::sparse_tsdf moved_from = requested(pool, small_budget, stream, what);
    const gca::sparse_tsdf moved_to(std::move(moved_from));
    check(moved_from.layout().capacity_blocks == 0 && moved_to.layout().capacity_blocks == 128,
          "refusals: a sparse TSDF moved from holds no blocks, the one moved to its 128");
    const gca::request_result<gca::sparse_tsdf> no_block_budget =
        gca::sparse_tsdf::request(pool, with_budget(4179), stream);
    check_refusal("refusals, a budget of 4179 bytes",
                  no_block_budget.status == gca::request_status::failure, no_block_budget.message,
                  no_block_budget.message.find("holds no block") != std::string::npos);
    check_parameter_ranges(pool, stream);
    pool.close();
    const gca::request_result<gca::sparse_tsdf> closed =
        gca::sparse_tsdf::request(pool, with_budget(test_budget), stream);
    check_refusal("refusals, a request to the closed pool",
                  closed.status == gca::request_status::closed, closed.message,
                  closed.message.find("the GPU pool is closed") != std::string::npos);
    check(pool.refused_while_closed() == 1, "refusals: the closed pool counted one request");

    insert(tsdf, some, stream, what);
    const gca::sparse_tsdf_usage before = read_usage(tsdf, stream, what);

    gca::sparse_tsdf empty;
    const gca::step_result empty_insert = empty.insert_blocks(some.coordinates, stream);
    check_refusal("refusals, insertion into an empty sparse TSDF", !empty_insert.done,
                  empty_insert.message,
                  empty_insert.message.find("holds no memory") != std::string::npos);
    const gca::step_result moved_find =
        moved_from.find_blocks(some.coordinates, some.numbers, stream);
    check_refusal("refusals, lookup in a sparse TSDF moved from", !moved_find.done,
                  moved_find.message,
                  moved_find.message.find("holds no memory") != std::string::npos);
    const gca::usage_result empty_usage = empty.read_usage(stream);
    check_refusal("refusals, the usage of an empty sparse TSDF", !empty_usage.usage.has_value(),
                  empty_usage.message,
                  empty_usage.message.find("holds no memory") != std::string::npos);
    const gca::gpu_array<gca::block_coordinate> no_coordinates;
    const gca::step_result none_inserted = tsdf.insert_blocks(no_coordinates, stream);
    check_refusal("refusals, no coordinates to insert", !none_inserted.done, none_inserted.message,
                  none_inserted.message.find("no coordinates") != std::string::npos);
    const gca::step_result none_found = tsdf.find_blocks(no_coordinates, some.numbers, stream);
    check_refusal("refusals, no coordinates to find", !none_found.done, none_found.message,
                  none_found.message.find("no coordinates") != std::string::npos);
    const gca::step_result wrong_room = tsdf.find_blocks(some.coordinates, other_room, stream);
    check_refusal("refusals, room for 7 numbers for 8 coordinates", !wrong_room.done,
                  wrong_room.message, wrong_room.message.find("room for 7") != std::string::npos);

    // An error an earlier call leaves: cudaMalloc's refusal of 2^52 bytes, more than the board
    // has [S: docs/HARDWARE.md, 7607 MB].
    void *impossible = nullptr;
    check(cudaMalloc(&impossible, std::size_t{1} << 52) != cudaSuccess,
          "refusals: cudaMalloc refuses 2^52 bytes");
    const gca::step_result after_error = tsdf.insert_blocks(some.coordinates, stream);
    check_refusal("refusals, insertion after an earlier error", !after_error.done,
                  after_error.message,
                  after_error.message.find("left by an earlier call") != std::string::npos);
    check_no_cuda_error("refusals, insertion after an earlier error");
    check(cudaMalloc(&impossible, std::size_t{1} << 52) != cudaSuccess,
          "refusals: cudaMalloc refuses 2^52 bytes again");
    const gca::step_result find_after_error =
        tsdf.find_blocks(some.coordinates, some.numbers, stream);
    check_refusal("refusals, lookup after an earlier error", !find_after_error.done,
                  find_after_error.message,
                  find_after_error.message.find("left by an earlier call") != std::string::npos);
    check_no_cuda_error("refusals, lookup after an earlier error");

    check_usage("refusals, after them", read_usage(tsdf, stream, what), before);
    check_found("refusals, after them", find(tsdf, some, stream, what), test_blocks, 8);
}

/// The GPU arrays' requests: granted, refused for no elements, for more bytes than std::size_t
/// holds, for more than CUDA can allocate, and when the pool is closed; and their moves.
void check_arrays()
{
    static_assert(!std::is_copy_constructible_v<gca::gpu_array<std::uint32_t>> &&
                      !std::is_copy_assignable_v<gca::gpu_array<std::uint32_t>>,
                  "a GPU array is not copied");
    static_assert(std::is_nothrow_move_constructible_v<gca::gpu_array<std::uint32_t>> &&
                      std::is_nothrow_move_assignable_v<gca::gpu_array<std::uint32_t>>,
                  "a GPU array moves without throwing");
    gca::gpu_pool pool;
    gca::request_result<gca::gpu_array<std::uint32_t>> granted_array =
        pool.request_array<std::uint32_t>(100);
    check(granted_array.status == gca::request_status::granted && !granted_array.granted.empty() &&
              granted_array.granted.size() == 100 && granted_array.granted.size_bytes() == 400,
          "arrays: 100 elements of 4 bytes granted");
    const std::uint32_t *memory = granted_array.granted.data();
    gca::gpu_array<std::uint32_t> constructed(std::move(granted_array.granted));
    check(granted_array.granted.empty() && granted_array.granted.size() == 0 &&
              constructed.data() == memory && constructed.size() == 100,
          "arrays: an array moved from by construction holds nothing");
    gca::gpu_array<std::uint32_t> assigned;
    assigned = std::move(constructed);
    check(constructed.empty() && constructed.size_bytes() == 0 && assigned.data() == memory,
          "arrays: an array moved from by assignment holds nothing");

    const gca::request_result<gca::gpu_array<std::uint32_t>> no_elements =
        pool.request_array<std::uint32_t>(0);
    check_refusal("arrays, 0 elements", no_elements.status == gca::request_status::failure,
                  no_elements.message,
                  no_elements.message.find("holds no element") != std::string::npos);
    const gca::request_result<gca::gpu_array<std::uint64_t>> too_many =
        pool.request_array<std::uint64_t>(std::numeric_limits<std::size_t>::max() / 4);
    check_refusal("arrays, 2^64 / 4 elements of 8 bytes",
                  too_many.status == gca::request_status::failure, too_many.message,
                  too_many.message.find("more bytes than std::size_t holds") != std::string::npos);
    // 2^49 elements of 8 bytes, 2^52 bytes, more than the board has [S: docs/HARDWARE.md].
    const gca::request_result<gca::gpu_array<std::uint64_t>> impossible =
        pool.request_array<std::uint64_t>(std::size_t{1} << 49);
    check_refusal("arrays, 2^52 bytes", impossible.status == gca::request_status::failure,
                  impossible.message,
                  impossible.message.find("CUDA could not allocate") != std::string::npos);
    check_no_cuda_error("arrays, after cudaMalloc's refusal");
    pool.close();
    const gca::request_result<gca::gpu_array<std::uint32_t>> closed =
        pool.request_array<std::uint32_t>(100);
    check_refusal("arrays, the closed pool", closed.status == gca::request_status::closed,
                  closed.message,
                  closed.message.find("the GPU pool is closed") != std::string::npos);
    check(pool.refused_while_closed() == 1 && closed.granted.empty(),
          "arrays: the closed pool counted the request and granted nothing");
}
} // namespace

/// Runs the checks of the sparse TSDF's hash of blocks and its pool and prints the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    cudaStream_t stream = nullptr;
    if (!create_stream(stream))
        return gca_test::check_summary("test_sparse_tsdf_blocks");
    check_arrays();
    check_layout(stream);
    check_collisions(stream);
    check_ten_runs(stream);
    check_pool_too_small(stream);
    check_hash_full(stream);
    check_key_range(stream);
    check_refusals(stream);
    check_no_cuda_error("the end");
    check(cudaStreamDestroy(stream) == cudaSuccess, "the CUDA stream destroyed");
    return gca_test::check_summary("test_sparse_tsdf_blocks");
}
