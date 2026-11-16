// SPDX-License-Identifier: 0BSD
/// @file test_live_model.cpp
/// The taking of a scan's live models (scan/live_model.hpp; docs/methods/tracking.md, "The live
/// model"), on the tabletop sphere's live model (2 mm), fused at its walk's known poses at 640x400
/// in a sparse TSDF of 273 MB.
/// - The spacing's rule (spacing_after()): the same within three quarters of the limit, doubled
///   while the points, a quarter at each doubling, stay above it; never less.
/// - The takes with a limit of 30000 points, below the 202488 points of every crossing: the first
///   take finds too many and raises the spacing to 4; the next takes every block at it; after more
///   frames the takes extract the blocks touched since and merge them; the live model they give
///   holds at most the limit and is an extraction of every block at that spacing but for a
///   measured share of older points within a voxel of the newer ones, the crossings of blocks not
///   touched since that read a touched neighbour's voxels.
/// - A merge over the limit, on the tabletop object class's own live model along the walk around
///   the tabletop box: too many, nothing published, the working live model as it was, the next
///   take of every block at a raised spacing, and the merge after it left no mark.
/// - The closed pool is asked for nothing.

#include "check.hpp"
#include "fusion/sparse_tsdf.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu_test_helpers.hpp"
#include "scan/live_model.hpp"
#include "scan/object_class.hpp"
#include "scan/stage_parameters.hpp"
#include "sparse_tsdf_helpers.hpp"
#include "sparse_tsdf_scenes.hpp"
#include "stage_frames.hpp"
#include "synthetic_source.hpp"
#include "types/model_point.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

namespace
{
using gca_test::check;

/// The limit of the takes: 30000 points [?: a choice], below the 50635 points of spacing 2 on the
/// scene and above the 12599 of spacing 4 [M 2026-11-16: test_sparse_tsdf_fusion's thinning case],
/// so that the spacing has to rise to 4.
constexpr std::uint32_t take_limit = 30000;

/// The share of points in one only of a merged live model and an extraction of every block at its
/// spacing: 0.01, about 6 times this test's 41 of 12599 (0.33 %, 21 only in the extraction and 20
/// only in the merged) [M 2026-11-16: this test]. They are the crossings of blocks not touched
/// since that read a touched neighbour's voxels, whose older values the merged model keeps
/// (docs/methods/tracking.md, "The live model").
constexpr double merge_share_tolerance = 0.01;

/// How far such an older point may lie from the nearest of every block's: 2 mm, a voxel, 4 times
/// this test's farthest, 0.4945 mm [M 2026-11-16: this test].
constexpr double merge_lag_tolerance_mm = 2.0;

/// The walk of the merge over the limit: tests/test_scan_interface.cpp's around the tabletop box,
/// 240 poses 1.5 degrees apart. Its frame 0, then 15 frames of the opposite side from pose
/// opposite_pose on, are fused in three steps: frame 0, then 5 of the opposite side, then 10 more
/// [?: a choice: the opposite side's surfaces lie mostly beyond frame 0's far depth of 0.6 m].
constexpr std::size_t box_walk_poses = 240;
constexpr std::size_t opposite_pose = 120;
constexpr std::array<std::size_t, 3> over_frames = {1, 6, 16};

/// The limit of the merge over the limit: 100000 points, above four thirds of frame 0's 70998
/// points at spacing 1, above the opposite side's first 5 frames' 84111 points of the blocks they
/// touched, and below both together, whose blocks hold 33739 points at spacing 2, about 135000 at
/// 1 [M 2026-11-17: this test; S: a quarter at each doubling], so that only the merge goes over it.
constexpr std::uint32_t over_limit = 100000;

/// The spacing's rule on known counts.
void check_spacing_rule()
{
    const std::uint32_t limit = 1000;
    check(gca::spacing_after(750, 1, limit) == 1, "spacing: three quarters of the limit stay");
    check(gca::spacing_after(751, 1, limit) == 2, "spacing: more doubles it");
    check(gca::spacing_after(750 * 16, 2, limit) == 8,
          "spacing: 16 times three quarters doubles it twice, from 2 to 8");
    check(gca::spacing_after(750 * 16 + 1, 4, limit) == 32,
          "spacing: past 16 times three quarters, three doublings");
    check(gca::spacing_after(0, 4, limit) == 4, "spacing: never less");
}

/// Gives a held live model with room for a limit's points and slots.
/// @param limit [in] the points.
/// @return the held live model, empty.
gca::held_live_model reserved_for(std::uint32_t limit)
{
    gca::held_live_model held;
    held.value.points.reserve(limit);
    held.slots.reserve(limit);
    return held;
}

/// Takes a live model and swaps it in when taken, as the scan interface does, and prints it.
/// @param taker [in,out] the taker.
/// @param tsdf [in] the live model's sparse TSDF.
/// @param held [in,out] the last live model taken; the new one when taken.
/// @param working [in,out] the one the take fills.
/// @param stream [in] the stream.
/// @param what [in] the take, named in the line printed.
/// @return what the take gave.
gca::live_take take_and_swap(gca::live_model_taker &taker, const gca::sparse_tsdf &tsdf,
                             gca::held_live_model &held, gca::held_live_model &working,
                             cudaStream_t stream, const std::string &what)
{
    const gca::live_take took = taker.take(tsdf, held, working, stream);
    std::printf("%s: status %d, %s, spacing %u, found %u, blocks %u, points %zu\n", what.c_str(),
                static_cast<int>(took.status), took.every_block ? "every block" : "touched blocks",
                took.spacing, took.found, took.blocks, working.value.points.size());
    if (took.status == gca::live_take_status::taken)
        std::swap(held, working);
    return took;
}

/// Downloads the points of an extraction as model points.
/// @param buffer [in] the buffer.
/// @param made [in] the extraction's result.
/// @return the points written.
std::vector<gca::model_point> model_points_of(const gca::surface_points &buffer,
                                              const gca::extraction_result &made)
{
    std::vector<gca::surface_point> points(made.written);
    check(cudaMemcpy(points.data(), buffer.points().data(),
                     sizeof(gca::surface_point) * points.size(),
                     cudaMemcpyDeviceToHost) == cudaSuccess,
          "takes: the full extraction downloaded");
    std::vector<gca::model_point> model;
    for (const gca::surface_point &p : points)
        model.push_back(gca::model_point_of(p));
    return model;
}

/// The takes on the tabletop sphere's walk.
/// @param stream [in] the stream.
void check_takes(cudaStream_t stream)
{
    const gca_test::model_case m =
        gca_test::model_of("takes", gca::object_class::tabletop_object, false);
    const gca_test::synthetic_scene scene = gca_test::tabletop_sphere_scene();
    const gca_test::rendered frames = gca_test::render(scene, gca_test::small_description);
    check(frames.frames.size() == 12, "takes: the walk's 12 frames");
    if (frames.frames.size() != 12)
        return;
    gca::gpu_pool pool;
    gca_test::frame_buffers buffers =
        gca_test::request_buffers(pool, gca_test::small_description.size, "takes");
    gca::sparse_tsdf tsdf = gca_test::granted("takes: the sparse TSDF",
                                              gca::sparse_tsdf::request(pool, m.tsdf, stream));
    gca::live_model_taker taker = gca_test::granted(
        "takes: the taker", gca::live_model_taker::request(pool, take_limit, tsdf.layout()));
    gca::surface_points every =
        gca_test::granted("takes: a buffer", gca::surface_points::request(pool, take_limit));
    pool.close();
    gca::held_live_model held = reserved_for(take_limit);
    gca::held_live_model working = reserved_for(take_limit);
    const std::vector<gca::model_point> &last = held.value.points;
    const auto take = [&](const std::string &what) {
        return take_and_swap(taker, tsdf, held, working, stream, "takes, " + what);
    };
    const auto fuse_frames = [&](std::size_t from, std::size_t to) {
        for (std::size_t i = from; i < to; i++)
            gca_test::fuse(tsdf, buffers, frames.frames[i], m.range, scene.trajectory[i], stream,
                           "takes, frame " + std::to_string(i));
    };

    fuse_frames(0, 6);
    const gca::live_take first = take("the first");
    check(first.status == gca::live_take_status::too_many && first.every_block &&
              first.spacing == 1,
          "takes: the first, every block at spacing 1, too many");
    const gca::live_take second = take("the second");
    check(second.status == gca::live_take_status::taken && second.every_block &&
              second.spacing == 4 && last.size() <= take_limit,
          "takes: the second, every block at the raised spacing 4, taken within the limit");
    fuse_frames(6, 9);
    const gca::live_take third = take("the third");
    fuse_frames(9, 12);
    const gca::live_take fourth = take("the fourth");
    check(third.status == gca::live_take_status::taken && !third.every_block &&
              fourth.status == gca::live_take_status::taken && !fourth.every_block &&
              fourth.spacing == 4 && fourth.blocks > 0 && last.size() <= take_limit,
          "takes: the third and fourth, the blocks touched since, merged within the limit");
    const gca::extraction_result full = tsdf.extract(every, {4, 0}, stream);
    check(full.done, ("takes: every block at spacing 4, " + full.message).c_str());
    const std::vector<gca_test::point_key> fresh =
        gca_test::sorted_keys(model_points_of(every, full));
    const std::vector<gca_test::point_key> merged = gca_test::sorted_keys(last);
    std::vector<gca_test::point_key> both;
    std::set_intersection(fresh.begin(), fresh.end(), merged.begin(), merged.end(),
                          std::back_inserter(both));
    std::printf("takes: every block's %zu points, the merged %zu, %zu in both\n", fresh.size(),
                merged.size(), both.size());
    // A point only in the merged live model is an older value of a crossing of a block not
    // touched since, beside a touched one: its nearest point only in every block's extraction.
    const std::vector<gca::model_point> fresh_points = model_points_of(every, full);
    double farthest_m = 0.0;
    std::size_t only_merged = 0;
    for (const gca::model_point &p : last)
    {
        const std::vector<gca::model_point> one = {p};
        if (std::binary_search(fresh.begin(), fresh.end(), gca_test::sorted_keys(one)[0]))
            continue;
        only_merged++;
        double nearest = 1e9;
        for (const gca::model_point &q : fresh_points)
            nearest = std::min(nearest, static_cast<double>((q.position_m - p.position_m).norm()));
        farthest_m = std::max(farthest_m, nearest);
    }
    std::printf("takes: %zu points only in the merged, the farthest %.4g mm from a point of every "
                "block's\n",
                only_merged, 1e3 * farthest_m);
    const std::size_t differing = (fresh.size() - both.size()) + (merged.size() - both.size());
    gca_test::check_within("takes: the share of points in one only of the merged live model and "
                           "every block's extraction",
                           static_cast<double>(differing) / static_cast<double>(fresh.size()),
                           merge_share_tolerance);
    gca_test::check_within("takes: the farthest point only in the merged from every block's, mm",
                           1e3 * farthest_m, merge_lag_tolerance_mm);
    check(held.slots.size() == last.size(), "takes: a slot for each point");
    check(pool.refused_while_closed() == 0, "takes: the closed pool asked for nothing");
}

/// A merge over the limit, on the tabletop object class's own live model along the walk around
/// the tabletop box: after frame 0 the first take keeps every block at spacing 1 within three
/// quarters of over_limit; after 5 frames of the opposite side the blocks they touched hold fewer
/// points than over_limit but more with the points kept: too many, nothing published, the working
/// live model as it was; the next take covers every block at the raised spacing; after 10 more
/// frames of that side the take of the blocks touched since merges with no mark of the refused take
/// left, so it is an extraction of every block at its spacing but for the measured share.
/// @param stream [in] the stream.
void check_merge_over_limit(cudaStream_t stream)
{
    // The tabletop object class's own live model and depth range, whose far depth of 0.6 m makes
    // the walk's model grow from its first frame (tests/test_scan_interface.cpp's live models).
    gca::object_class_parameters set = gca::parameters_of(gca::object_class::tabletop_object);
    set.memory_budget_bytes = gca_test::case_budget;
    const gca::sparse_tsdf_parameters live =
        gca::sparse_tsdf_parameters_of(set, gca::fused_model::live);
    const gca::organized_points_parameters range = gca::tracking_parameters_of(set).pyramid.range;
    gca_test::synthetic_walk walk = gca_test::tabletop_walk();
    walk.pose_count = box_walk_poses;
    const std::vector<Eigen::Isometry3d> walked = gca_test::poses_of(walk);
    // Frame 0, then the opposite side's frames from pose opposite_pose on, which see other
    // surfaces.
    std::vector<Eigen::Isometry3d> poses = {walked[0]};
    for (std::size_t k = 0; k + 1 < over_frames.back(); k++)
        poses.push_back(walked[opposite_pose + k]);
    const gca_test::synthetic_scene scene = {
        .plane = gca_test::tabletop_plane, .box = gca_test::tabletop_box, .trajectory = poses};
    const gca_test::rendered frames = gca_test::render(scene, gca_test::small_description);
    check(frames.frames.size() == over_frames.back(), "over the limit: the walk's frames");
    if (frames.frames.size() != over_frames.back())
        return;
    gca::gpu_pool pool;
    gca_test::frame_buffers buffers =
        gca_test::request_buffers(pool, gca_test::small_description.size, "over the limit");
    gca::sparse_tsdf tsdf = gca_test::granted("over the limit: the sparse TSDF",
                                              gca::sparse_tsdf::request(pool, live, stream));
    gca::live_model_taker taker =
        gca_test::granted("over the limit: the taker",
                          gca::live_model_taker::request(pool, over_limit, tsdf.layout()));
    gca::surface_points every = gca_test::granted("over the limit: a buffer",
                                                  gca::surface_points::request(pool, over_limit));
    pool.close();
    gca::held_live_model held = reserved_for(over_limit);
    gca::held_live_model working = reserved_for(over_limit);
    const auto fuse_frames = [&](std::size_t from, std::size_t to) {
        for (std::size_t i = from; i < to; i++)
            gca_test::fuse(tsdf, buffers, frames.frames[i], range, scene.trajectory[i], stream,
                           "over the limit, frame " + std::to_string(i));
    };

    fuse_frames(0, over_frames[0]);
    const gca::live_take first =
        take_and_swap(taker, tsdf, held, working, stream, "over the limit, after frame 0");
    check(first.status == gca::live_take_status::taken && first.every_block && first.spacing == 1 &&
              held.value.points.size() <= over_limit,
          "over the limit: after frame 0, every block at spacing 1, within the limit");
    fuse_frames(over_frames[0], over_frames[1]);
    // The working live model as the refused take must leave it: the one the first take swapped
    // out, empty, with a mark of its own so that a write into it shows.
    working.value.points.assign(1, held.value.points.front());
    working.slots.assign(1, 0xFFFFFFFFu);
    const std::vector<gca::model_point> held_before = held.value.points;
    const gca::live_take over =
        take_and_swap(taker, tsdf, held, working, stream, "over the limit, from the opposite side");
    check(over.status == gca::live_take_status::too_many && !over.every_block &&
              over.spacing == 1 && over.found <= over_limit,
          "over the limit: the touched blocks' points within the limit, merged with the points "
          "kept over it, too many");
    check(working.value.points.size() == 1 && working.slots.size() == 1 &&
              working.slots[0] == 0xFFFFFFFFu &&
              gca_test::sorted_keys(held.value.points) == gca_test::sorted_keys(held_before),
          "over the limit: nothing published, the working live model as it was");
    const gca::live_take raised =
        take_and_swap(taker, tsdf, held, working, stream, "over the limit, the next take");
    check(raised.status == gca::live_take_status::taken && raised.every_block &&
              raised.spacing > 1 && held.value.points.size() <= over_limit,
          "over the limit: the next take every block at a raised spacing, within the limit");
    fuse_frames(over_frames[1], over_frames[2]);
    const gca::live_take merged =
        take_and_swap(taker, tsdf, held, working, stream, "over the limit, after the last frames");
    check(merged.status == gca::live_take_status::taken && !merged.every_block,
          "over the limit: the last take merges the blocks touched since");
    const gca::extraction_result full = tsdf.extract(every, {raised.spacing, 0}, stream);
    check(full.done,
          ("over the limit: every block at the raised spacing, " + full.message).c_str());
    const std::vector<gca_test::point_key> fresh =
        gca_test::sorted_keys(model_points_of(every, full));
    const std::vector<gca_test::point_key> kept = gca_test::sorted_keys(held.value.points);
    std::vector<gca_test::point_key> both;
    std::set_intersection(fresh.begin(), fresh.end(), kept.begin(), kept.end(),
                          std::back_inserter(both));
    const std::size_t differing = (fresh.size() - both.size()) + (kept.size() - both.size());
    std::printf("over the limit: every block's %zu points, the merged %zu, %zu in both\n",
                fresh.size(), kept.size(), both.size());
    gca_test::check_within("over the limit: the share of points in one only of the merged live "
                           "model and every block's extraction, no mark left",
                           static_cast<double>(differing) / static_cast<double>(fresh.size()),
                           merge_share_tolerance);
    check(pool.refused_while_closed() == 0, "over the limit: the closed pool asked for nothing");
}
} // namespace

/// Runs the checks and prints the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    check_spacing_rule();
    cudaStream_t stream = nullptr;
    if (!gca_test::create_stream(stream))
        return gca_test::check_summary("test_live_model");
    check_takes(stream);
    check_merge_over_limit(stream);
    gca_test::check_no_cuda_error("live model");
    check(cudaStreamDestroy(stream) == cudaSuccess, "the CUDA stream destroyed");
    return gca_test::check_summary("test_live_model");
}
