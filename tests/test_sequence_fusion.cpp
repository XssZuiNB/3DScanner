// SPDX-License-Identifier: 0BSD
/// @file test_sequence_fusion.cpp
/// Refinement's fusion of a scan's sequence at the full voxel size (refinement/sequence_fusion.hpp,
/// docs/methods/refinement.md, "The fusion of the sequence"), through fuse_sequence(), on
/// sequences the test writes through the sequence's writer into a temporary directory: frames of
/// the synthetic source at 640x400 with the camera's intrinsics, the tabletop box on its plane
/// seen along a walk once around it, each frame with its known pose, and, after every fourth,
/// a frame of a wall 0.35 m in front of the camera marked as tracking loss. The tabletop object
/// class's full model (scan/stage_parameters.hpp), in a sparse TSDF of 273 MB. The poses are the
/// scene's world frame, so the model is in it too.
/// - The walk: every frame read, those with a pose fused and the walls skipped; the trajectory
///   as written, bit for bit; the full model's points within measured distances of the known
///   surfaces, their median, 99th percentile and largest, so that a wall fused shows; each face of
///   the box and the plane with points on it, every normal of unit length within the model
///   file's 1e-6.
/// - The walk cut off within a record: done, its early end said, its whole frames read, and the
///   model the same set of points, bit for bit, as that of a sequence of just those frames.
/// - A frame marked as tracked by nothing: refused with the reason, which names the frame; no
///   model and no trajectory.
/// - A cancel before the sixth frame: cancelled, five frames read, no model and no trajectory.
/// - A pool used up by walls far apart: failed with the counts, every integration done all the
///   same.
/// - Only frames of tracking loss: done, nothing fused, no point.
/// - Refusals: no file, another size, a voxel size the sparse TSDF refuses.
/// The test prints the walk's fusion time; a scan-level test ends within 60 s on the board.

#include "camera/frame.hpp"
#include "camera/sequence_format.hpp"
#include "camera/sequence_writer.hpp"
#include "check.hpp"
#include "frame_copy.hpp"
#include "fusion/sparse_tsdf.hpp"
#include "gpu_test_helpers.hpp"
#include "refinement/sequence_fusion.hpp"
#include "scan/object_class.hpp"
#include "scan/stage_parameters.hpp"
#include "sparse_tsdf_scenes.hpp"
#include "stage_checks.hpp"
#include "stage_frames.hpp"
#include "synthetic_source.hpp"
#include "test_files.hpp"
#include "types/camera_pose.hpp"
#include "types/model_point.hpp"
#include "types/trajectory_entry.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::check;
using gca_test::check_within;
using gca_test::small_description;
using gca_test::temporary_directory;

/// The walk's poses: 24, 15 degrees of azimuth apart [?: a choice: known poses need no small step,
/// and 24 views see every face of the box].
constexpr std::size_t walk_poses = 24;

/// A wall marked as tracking loss follows every fourth frame with a pose [?: a choice].
constexpr std::size_t wall_every = 4;

/// The wall's depth, 350 mm in front of the camera, nearer than every known surface of the walk's
/// views, so that a wall fused at any pose lies off them [?: a choice].
constexpr std::uint16_t wall_depth_mm = 350;

/// The records of the walk's sequence before its cut: 13 [?: a choice, within the walk].
constexpr std::size_t cut_records = 13;

/// The calls of go_on the cancel's case answers true, before its false: 5 [?: a choice].
constexpr int cancel_after = 5;

/// The case of a pool used up: four walls facing the camera at 0.55 m, fused from poses 2 m apart
/// along x, so that each frame's blocks are its own, into a budget of 140 MB, 33123 blocks with
/// 262144 slots [S: computed, docs/methods/sparse-tsdf.md, "The pool and the memory budget"]. The
/// request grants it: one frame's bound at 1 mm to 0.6 m, 103746 blocks [M 2026-11-16: the
/// request's refusal of 40 MB], lies below the hash's room of 229021 once the pool is used up; the
/// walls touch 158356 blocks, 125233 more than the pool holds [M 2026-11-16: this test, the counts
/// printed].
constexpr std::uint64_t small_budget = 140'000'000;
constexpr std::size_t used_up_walls = 4;
constexpr std::uint16_t used_up_wall_mm = 550;

/// The tolerances of the walk's full model, in mm: the median, the 99th percentile and the largest
/// distance of a point from the nearest known surface, about 1.4 to 2 times what this test
/// measured, 0.0754, 0.675 and 8.77 mm of 863311 points [M 2026-11-16: this test]. The largest lie
/// within 5 mm below the box's top edges, up to 9 mm outside its sides: the band of the
/// truncation behind a convex edge, which the extraction makes points of
/// (docs/methods/sparse-tsdf.md, "The extraction"); 0.26 % of the points lie farther than 2 mm.
constexpr double distance_median_mm = 0.15;
constexpr double distance_p99_mm = 1.0;
constexpr double distance_max_mm = 12.0;

/// The least points within a voxel of each known surface, the plane and the box's five faces
/// [?: a choice, far below what each face gives].
constexpr std::size_t least_points_on_a_face = 1000;

/// The tolerance of a normal's length: 1e-6, the model file's [S: scan/model_file.hpp].
constexpr double unit_normal_tolerance = 1e-6;

/// The tabletop object class's set (src/scan/object_class.cpp).
const gca::object_class_parameters tabletop_set =
    gca::parameters_of(gca::object_class::tabletop_object);

/// Gives the full model's parameters: the tabletop object class's (scan/stage_parameters.hpp), in a
/// sparse TSDF of 273 MB (gca_test::case_budget).
/// @return the parameters.
gca::sparse_tsdf_parameters full_model_parameters()
{
    gca::sparse_tsdf_parameters p =
        gca::sparse_tsdf_parameters_of(tabletop_set, gca::fused_model::full);
    p.memory_budget_bytes = gca_test::case_budget;
    return p;
}

/// The organized points' depth range: the tabletop object class's, 0.25 m to 0.6 m.
const gca::organized_points_parameters tabletop_range =
    gca::tracking_parameters_of(tabletop_set).pyramid.range;

/// A frame as the test writes it into a sequence: the frame and what tracking said about it.
struct written_frame
{
    gca::frame frame;             ///< the frame, its index and timestamp the sequence's
    gca::frame_tracking tracking; ///< its pose or its mark
};

/// The walk's frames as the test writes them, and the known surfaces they show.
struct walk_case
{
    std::vector<written_frame> frames;                 ///< in the sequence's order
    std::vector<gca_test::synthetic_surface> surfaces; ///< the plane, then the box's faces
};

/// Builds the walk: the tabletop box on its plane from walk_poses poses of tabletop_walk(), frame
/// k with index 2 k and its known pose, and after every wall_every-th a wall with index 2 k + 1,
/// marked as tracking loss; the timestamps at the synthetic source's 30 fps by index.
/// @return the walk; fewer frames, with a failed check, when the rendering failed.
walk_case walk()
{
    gca_test::synthetic_walk around = gca_test::tabletop_walk();
    around.pose_count = walk_poses;
    const std::vector<Eigen::Isometry3d> poses = gca_test::poses_of(around);
    const gca_test::synthetic_scene scene = {
        .plane = gca_test::tabletop_plane, .box = gca_test::tabletop_box, .trajectory = poses};
    gca_test::rendered rendered = gca_test::render(scene, small_description);
    walk_case made{{}, rendered.surfaces};
    const gca::frame wall = gca_test::flat_wall(wall_depth_mm, {200, 40, 40});
    for (std::size_t k = 0; k < rendered.frames.size(); k++)
    {
        const std::uint64_t index = 2 * k;
        made.frames.push_back({gca_test::restamped(rendered.frames[k], index,
                                                   index * gca_test::synthetic_frame_period_us),
                               gca_test::as_camera_pose(poses[k])});
        if ((k + 1) % wall_every == 0)
            made.frames.push_back(
                {gca_test::restamped(wall, index + 1,
                                     (index + 1) * gca_test::synthetic_frame_period_us),
                 gca::tracking_loss{}});
    }
    return made;
}

/// Writes frames into a new sequence through the sequence's writer.
/// @param path [in] the sequence's path, where no file stands.
/// @param frames [in] the frames with their poses or marks; copied, as a push swaps buffers.
/// @param count [in] how many of them, from the first.
/// @param what [in] the case, named in the check lines.
void write_sequence(const std::filesystem::path &path, const std::vector<written_frame> &frames,
                    std::size_t count, const std::string &what)
{
    gca::sequence_writer writer;
    const gca::open_result opened =
        writer.open(path, small_description, gca::sequence_queue_frames);
    check(opened.opened, (what + ": the writer opened; " + opened.refusal).c_str());
    std::size_t accepted = 0;
    for (std::size_t k = 0; k < count && k < frames.size(); k++)
    {
        gca::frame copy = gca_test::restamped(frames[k].frame, frames[k].frame.index(),
                                              frames[k].frame.timestamp_us());
        // A full queue is waited out: the writer's thread empties it.
        gca::push_result pushed = writer.push(copy, frames[k].tracking);
        while (pushed.status == gca::push_status::full)
            pushed = writer.push(copy, frames[k].tracking);
        accepted += pushed.status == gca::push_status::accepted;
    }
    const gca::write_result closed = writer.close();
    check(accepted == std::min(count, frames.size()) && closed.written,
          (what + ": every frame written; " + closed.reason).c_str());
}

/// Checks that the trajectory read back is the frames written, in order, bit for bit.
/// @param what [in] the case, named in the check line.
/// @param read [in] the trajectory read back.
/// @param frames [in] the frames written.
/// @param count [in] how many of them the trajectory holds.
void check_trajectory(const std::string &what, const std::vector<gca::trajectory_entry> &read,
                      const std::vector<written_frame> &frames, std::size_t count)
{
    bool same = read.size() == count && count <= frames.size();
    for (std::size_t k = 0; same && k < count; k++)
    {
        const gca::frame &f = frames[k].frame;
        const gca::camera_pose *pose = std::get_if<gca::camera_pose>(&frames[k].tracking);
        same = read[k].index == f.index() && read[k].timestamp_us == f.timestamp_us() &&
               read[k].pose.has_value() == (pose != nullptr) &&
               (pose == nullptr || gca_test::same_pose(*read[k].pose, *pose));
    }
    check(same,
          (what + ": the trajectory as written, " + std::to_string(count) + " frames, bit for bit")
              .c_str());
}

/// Gives the name of how a fusion ended, for the printed lines.
/// @param end [in] the end.
/// @return its enumerator's name.
const char *end_name(gca::sequence_fusion_end end)
{
    switch (end)
    {
    case gca::sequence_fusion_end::done:
        return "done";
    case gca::sequence_fusion_end::cancelled:
        return "cancelled";
    case gca::sequence_fusion_end::failed:
        return "failed";
    }
    return "unknown";
}

/// Fuses a sequence with the test's parameters, going on at every frame, and prints the result.
/// @param what [in] the case, printed.
/// @param path [in] the sequence.
/// @param stream [in] the stream.
/// @return the result.
gca::sequence_fusion_result fused(const std::string &what, const std::filesystem::path &path,
                                  cudaStream_t stream)
{
    const gca::sequence_fusion_result result = gca::fuse_sequence(
        path, small_description.size, full_model_parameters(), tabletop_range, [] { return true; },
        stream);
    std::printf("%s: %s, %zu frames read, %llu fused, %llu skipped, %zu points%s%s%s\n",
                what.c_str(), end_name(result.end), result.trajectory.size(),
                static_cast<unsigned long long>(result.frames_fused),
                static_cast<unsigned long long>(result.frames_skipped), result.model.size(),
                result.ended_early ? ", ended early" : "", result.message.empty() ? "" : ": ",
                result.message.c_str());
    return result;
}

/// The walk: done, every frame read, those with a pose fused and the walls skipped, the trajectory
/// as written; the points near the known surfaces, none at a wall, each surface with points on it,
/// every normal unit.
/// @param directory [in] the test's directory.
/// @param w [in] the walk.
/// @param stream [in] the stream.
void check_walk(const temporary_directory &directory, const walk_case &w, cudaStream_t stream)
{
    const std::filesystem::path path = directory / "walk.sequence";
    write_sequence(path, w.frames, w.frames.size(), "walk");
    const auto start = std::chrono::steady_clock::now();
    const gca::sequence_fusion_result result = fused("walk", path, stream);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("walk: %zu frames fused at %.0f mm voxels in %.2f s\n", w.frames.size(),
                1e3 * full_model_parameters().voxel_size_m, seconds);
    const std::size_t walls = walk_poses / wall_every;
    check(result.end == gca::sequence_fusion_end::done && result.message.empty() &&
              !result.ended_early,
          "walk: done");
    check(result.frames_fused == walk_poses && result.frames_skipped == walls,
          "walk: every frame with a pose fused, every wall skipped");
    check_trajectory("walk", result.trajectory, w.frames, w.frames.size());

    const double voxel_mm = 1e3 * tabletop_set.full_voxel_size_m;
    std::vector<double> distances_mm;
    std::vector<std::size_t> on_surface(w.surfaces.size(), 0);
    gca_test::max_error unit_errors;
    for (const gca::model_point &point : result.model)
    {
        unit_errors.add(std::abs(point.normal.cast<double>().norm() - 1.0));
        double nearest_m = std::numeric_limits<double>::infinity();
        std::size_t nearest = 0;
        for (std::size_t s = 0; s < w.surfaces.size(); s++)
        {
            const double d =
                gca_test::distance_from(w.surfaces[s], point.position_m.cast<double>());
            if (d < nearest_m)
            {
                nearest_m = d;
                nearest = s;
            }
        }
        distances_mm.push_back(1e3 * nearest_m);
        on_surface[nearest] += 1e3 * nearest_m <= voxel_mm;
    }
    check(!result.model.empty(), "walk: the full model holds points");
    gca_test::check_max_error("walk: a normal's length less 1", unit_errors, unit_normal_tolerance);
    std::printf("walk: distance from the nearest known surface: median %.4g mm, 99th percentile "
                "%.4g mm, largest %.4g mm, of %zu points\n",
                gca_test::quantile(distances_mm, 0.5), gca_test::quantile(distances_mm, 0.99),
                gca_test::quantile(distances_mm, 1.0), distances_mm.size());
    check_within("walk: the points' median distance from the known surfaces, mm",
                 gca_test::quantile(distances_mm, 0.5), distance_median_mm);
    check_within("walk: the points' 99th percentile of the distance from the known surfaces, mm",
                 gca_test::quantile(distances_mm, 0.99), distance_p99_mm);
    check_within("walk: the points' largest distance from the known surfaces, mm",
                 gca_test::quantile(distances_mm, 1.0), distance_max_mm);
    for (std::size_t s = 0; s < on_surface.size(); s++)
    {
        std::printf("walk: %zu points within a voxel of known surface %zu\n", on_surface[s], s);
        check(on_surface[s] >= least_points_on_a_face,
              ("walk: points on known surface " + std::to_string(s)).c_str());
    }
}

/// The walk cut off within its record cut_records: done, its early end said, cut_records frames
/// read, and the same model, bit for bit as a set, as a whole sequence of those frames.
/// @param directory [in] the test's directory.
/// @param w [in] the walk.
/// @param stream [in] the stream.
void check_cut(const temporary_directory &directory, const walk_case &w, cudaStream_t stream)
{
    const std::filesystem::path whole = directory / "first_frames.sequence";
    write_sequence(whole, w.frames, cut_records, "the first frames");
    const gca::sequence_fusion_result first = fused("the first frames", whole, stream);
    check(first.end == gca::sequence_fusion_end::done && !first.ended_early,
          "the first frames: done, not ended early");

    // The whole walk cut at the length of the first frames' sequence: their records whole, then
    // the next record's first bytes, as many as the end of the shorter sequence takes, fewer than
    // a record's.
    const std::filesystem::path full = directory / "walk_to_cut.sequence";
    write_sequence(full, w.frames, w.frames.size(), "the walk to cut");
    const std::filesystem::path cut = directory / "cut.sequence";
    gca_test::write_bytes(cut,
                          gca_test::bytes_of(full).substr(0, std::filesystem::file_size(whole)));
    const gca::sequence_fusion_result result = fused("cut", cut, stream);
    check(result.end == gca::sequence_fusion_end::done && result.ended_early &&
              result.message.empty(),
          "cut: done, ended early");
    check_trajectory("cut", result.trajectory, w.frames, cut_records);
    check(result.frames_fused == first.frames_fused &&
              result.frames_skipped == first.frames_skipped,
          "cut: the first frames' counts");
    check(!result.model.empty() &&
              gca_test::sorted_keys(result.model) == gca_test::sorted_keys(first.model),
          "cut: the first frames' model, bit for bit as a set");
}

/// A frame marked as tracked by nothing: failed with the reason, which names the frame; the
/// frames before it read; no model.
/// @param directory [in] the test's directory.
/// @param w [in] the walk.
/// @param stream [in] the stream.
void check_not_tracked(const temporary_directory &directory, const walk_case &w,
                       cudaStream_t stream)
{
    std::vector<written_frame> frames;
    for (std::size_t k = 0; k < 4; k++)
        frames.push_back({gca_test::restamped(w.frames[k].frame, w.frames[k].frame.index(),
                                              w.frames[k].frame.timestamp_us()),
                          k == 2 ? gca::frame_tracking{gca::not_tracked{}} : w.frames[k].tracking});
    const std::filesystem::path path = directory / "not_tracked.sequence";
    write_sequence(path, frames, frames.size(), "not tracked");
    const gca::sequence_fusion_result result = fused("not tracked", path, stream);
    const std::string expected =
        "frame " + std::to_string(frames[2].frame.index()) + " is marked as tracked by nothing";
    gca_test::check_refusal("a frame tracked by nothing",
                            result.end == gca::sequence_fusion_end::failed, result.message,
                            result.message.find(expected) != std::string::npos &&
                                result.message.find(path.string()) != std::string::npos);
    check(result.model.empty() && result.trajectory.empty() && result.frames_fused == 2,
          "not tracked: no model and no trajectory, the two frames before fused");
}

/// A cancel before the sixth frame: cancelled, five frames read, no model.
/// @param directory [in] the test's directory.
/// @param stream [in] the stream.
void check_cancel(const temporary_directory &directory, cudaStream_t stream)
{
    int asked = 0;
    const gca::sequence_fusion_result result = gca::fuse_sequence(
        directory / "walk.sequence", small_description.size, full_model_parameters(),
        tabletop_range, [&asked] { return ++asked <= cancel_after; }, stream);
    std::printf("cancel: %s after %d questions, %zu frames read\n", end_name(result.end), asked,
                result.trajectory.size());
    check(result.end == gca::sequence_fusion_end::cancelled && result.message.empty() &&
              asked == cancel_after + 1,
          "cancel: cancelled at the sixth question");
    check(result.model.empty() && result.trajectory.empty() &&
              result.frames_fused + result.frames_skipped == cancel_after,
          "cancel: five frames read, no model and no trajectory");
}

/// A pool used up: four walls at 0.55 m from poses 2 m apart into a budget of 140 MB; failed with
/// the reason and the counts, no model and no trajectory, although every integration was done.
/// @param directory [in] the test's directory.
/// @param stream [in] the stream.
void check_used_up(const temporary_directory &directory, cudaStream_t stream)
{
    const gca::frame wall = gca_test::flat_wall(used_up_wall_mm, {40, 160, 90});
    std::vector<written_frame> frames;
    for (std::size_t k = 0; k < used_up_walls; k++)
    {
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = Eigen::Vector3d(2.0 * static_cast<double>(k), 0.0, 0.0);
        frames.push_back({gca_test::restamped(wall, k, k * gca_test::synthetic_frame_period_us),
                          gca_test::as_camera_pose(pose)});
    }
    const std::filesystem::path path = directory / "walls.sequence";
    write_sequence(path, frames, frames.size(), "used up");
    gca::sparse_tsdf_parameters small = full_model_parameters();
    small.memory_budget_bytes = small_budget;
    const gca::sequence_fusion_result result = gca::fuse_sequence(
        path, small_description.size, small, tabletop_range, [] { return true; }, stream);
    gca_test::check_refusal(
        "a used-up pool", result.end == gca::sequence_fusion_end::failed, result.message,
        result.message.find("bytes was used up") != std::string::npos &&
            result.message.find("coordinates found no block") != std::string::npos);
    check(result.model.empty() && result.trajectory.empty() && result.frames_fused == used_up_walls,
          "used up: every frame fused, no model and no trajectory");
}

/// Only frames of tracking loss: done, nothing fused, no point.
/// @param directory [in] the test's directory.
/// @param w [in] the walk.
/// @param stream [in] the stream.
void check_only_lost(const temporary_directory &directory, const walk_case &w, cudaStream_t stream)
{
    std::vector<written_frame> frames;
    for (std::size_t k = 0; k < 3; k++)
        frames.push_back({gca_test::restamped(w.frames[k].frame, w.frames[k].frame.index(),
                                              w.frames[k].frame.timestamp_us()),
                          gca::tracking_loss{}});
    const std::filesystem::path path = directory / "only_lost.sequence";
    write_sequence(path, frames, frames.size(), "only lost");
    const gca::sequence_fusion_result result = fused("only lost", path, stream);
    check(result.end == gca::sequence_fusion_end::done && result.model.empty() &&
              result.frames_fused == 0 && result.frames_skipped == 3,
          "only lost: done, nothing fused, no point");
}

/// The refusals: no file, another size, a voxel size the sparse TSDF refuses; each failed with
/// its reason and no model.
/// @param directory [in] the test's directory.
/// @param stream [in] the stream.
void check_refusals(const temporary_directory &directory, cudaStream_t stream)
{
    const auto go_on = [] { return true; };
    const std::filesystem::path missing = directory / "missing.sequence";
    const gca::sequence_fusion_result no_file = gca::fuse_sequence(
        missing, small_description.size, full_model_parameters(), tabletop_range, go_on, stream);
    gca_test::check_refusal("no file", no_file.end == gca::sequence_fusion_end::failed,
                            no_file.message,
                            no_file.message.find(missing.string()) != std::string::npos);
    const gca::sequence_fusion_result other_size =
        gca::fuse_sequence(directory / "walk.sequence", {320, 200}, full_model_parameters(),
                           tabletop_range, go_on, stream);
    gca_test::check_refusal("another size", other_size.end == gca::sequence_fusion_end::failed,
                            other_size.message,
                            other_size.message.find("320x200") != std::string::npos);
    gca::sparse_tsdf_parameters coarse = full_model_parameters();
    coarse.voxel_size_m = 0.1f;
    const gca::sequence_fusion_result refused = gca::fuse_sequence(
        directory / "walk.sequence", small_description.size, coarse, tabletop_range, go_on, stream);
    gca_test::check_refusal("a voxel size of 0.1 m",
                            refused.end == gca::sequence_fusion_end::failed &&
                                refused.model.empty(),
                            refused.message, refused.message.find("voxel") != std::string::npos);
}
} // namespace

/// Runs the cases and prints the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    const auto start = std::chrono::steady_clock::now();
    const temporary_directory directory("test_sequence_fusion");
    cudaStream_t stream = nullptr;
    if (!gca_test::create_stream(stream))
        return gca_test::check_summary("test_sequence_fusion");
    const walk_case w = walk();
    check(w.frames.size() == walk_poses + walk_poses / wall_every, "the walk's frames");
    check_walk(directory, w, stream);
    check_cut(directory, w, stream);
    check_not_tracked(directory, w, stream);
    check_cancel(directory, stream);
    check_used_up(directory, stream);
    check_only_lost(directory, w, stream);
    check_refusals(directory, stream);
    gca_test::check_no_cuda_error("sequence fusion");
    cudaStreamDestroy(stream);
    std::printf("test_sequence_fusion: %.1f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    return gca_test::check_summary("test_sequence_fusion");
}
