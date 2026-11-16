// SPDX-License-Identifier: 0BSD
/// @file test_scan_interface.cpp
/// The scan interface (scan/scan_interface.hpp; the spec, "The scan interface"), in process, the
/// main seam of the spec's "Testing Decisions": the commands the page gives, on the tests'
/// synthetic source at 640x400 with the camera's intrinsics, into a temporary scans directory. The
/// scene is the tabletop box on its plane, seen along the walk once around it of
/// test_tracking.cpp, 1.5 degrees of azimuth between frames; the set is the tabletop object
/// class's own (src/scan/object_class.cpp) in a sparse TSDF of 273 MB. A listener records every
/// state, in order, and gives commands at exact frames, so that no case hangs on the board's
/// speed; a source in front of the synthetic one records its openings and closings in the same
/// log, and can leave gaps in the indices, deliver a frame of another size or describe itself with
/// a string the parameters cannot hold; a writer in front of the real one reports a full queue in
/// a fixed pattern.
/// - The walk, 240 frames, stopped after its last: the states idle, starting, scanning, refining
///   and idle, in that order; a second start during it refused with the reason, the scan going on;
///   the source closed before refining;
///   the scan's directory named by its start time with its four files and no other; the
///   parameters with the set, the commit, the source, the counts and the end; the trajectory
///   within a measured tolerance of the known one; the full model's points within measured
///   distances of the known surfaces.
/// - A discard while scanning, and one while refining (a stop then refused): nothing on disk.
/// - A source that times out, and one that fails, in the middle: the scan ends with the reason,
///   and what it had is refined.
/// - A source that refuses to open, and one whose description the parameters cannot hold:
///   refused at the start with the reason, the source closed, nothing on disk.
/// - Gaps in the indices and a writer made slow: the frames not taken counted in the state and the
///   parameters, every frame taken in the sequence with its pose or mark, bit for bit as the
///   trajectory, and those the writer had no room for in none.
/// - A frame tracking fails on: the scan ends, the frame counted as taken and written nowhere.
/// - A refinement that fails, and the scan interface's end during a scan: the scan unfinished,
///   its sequence and parameters kept, no trajectory and no full model.
/// - The "too close" rule on built frames, and its notice in the state.
/// The test prints its time; each scan-level test ends within 60 s on the board (the stage brief).

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/sequence_format.hpp"
#include "camera/sequence_reader.hpp"
#include "camera/sequence_writer.hpp"
#include "check.hpp"
#include "frame_copy.hpp"
#include "gpu_test_helpers.hpp"
#include "scan/model_file.hpp"
#include "scan/object_class.hpp"
#include "scan/parameters_file.hpp"
#include "scan/scan_interface.hpp"
#include "scan/trajectory_file.hpp"
#include "scan/utc_time.hpp"
#include "sparse_tsdf_scenes.hpp"
#include "stage_checks.hpp"
#include "stage_frames.hpp"
#include "synthetic_source.hpp"
#include "test_files.hpp"
#include "transform_checks.hpp"
#include "types/camera_pose.hpp"
#include "types/model_point.hpp"
#include "types/trajectory_entry.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::check;
using gca_test::check_within;
using gca_test::small_description;
using gca_test::temporary_directory;

/// The walk's poses: 240, 1.5 degrees of azimuth apart, as test_tracking.cpp's walk
/// [?: a choice, a hand's turn in a frame time at 30 fps, 45 degrees a second].
constexpr std::size_t walk_poses = 240;

/// The frames of a short scan: 20 of the walk's first [?: a choice].
constexpr std::size_t short_poses = 20;

/// The commit the scans' parameters hold [?: a choice, a commit's form].
const std::string test_commit = "e290a50";

/// How long a case waits for the scan interface to come back to idle: 50 s, within the 60 s of a
/// scan-level test (the stage brief) [?: a choice].
constexpr double idle_wait_s = 50.0;

/// The tolerances of the walk's tracked trajectory against the known one, in the first frame's
/// camera frame: 0.1 degrees and 1.0 mm, about twice the largest this test measured, 0.056 degrees
/// and 0.535 mm [M 2026-11-16: this test], as test_tracking.cpp's walk.
constexpr gca_test::transform_tolerance trajectory_tolerance = {0.1, 0.001};

/// The tolerances of the walk's full model, in mm: the median, the 99th percentile and the largest
/// distance of a point from the nearest known surface, 1.3 to 2.5 times what this test measured,
/// 0.0703, 0.405 and 9.17 mm of 1162157 points [M 2026-11-16: this test]; the largest lie at the
/// box's top edges, as in test_sequence_fusion.cpp's walk (docs/methods/refinement.md,
/// "Accuracy").
constexpr double distance_median_mm = 0.15;
constexpr double distance_p99_mm = 1.0;
constexpr double distance_max_mm = 12.0;

/// The set the scans run with: the tabletop object class's, in a sparse TSDF of 273 MB
/// (gca_test::case_budget), so that a case takes 273 MB of the board's memory, not 2 GB.
/// @return the set.
gca::object_class_parameters test_set()
{
    gca::object_class_parameters set = gca::parameters_of(gca::object_class::tabletop_object);
    set.memory_budget_bytes = gca_test::case_budget;
    return set;
}

/// The walk once around the tabletop box, walk_poses poses of tabletop_walk().
/// @return its poses, camera frame to world frame.
std::vector<Eigen::Isometry3d> walk_poses_of()
{
    gca_test::synthetic_walk walk = gca_test::tabletop_walk();
    walk.pose_count = walk_poses;
    return gca_test::poses_of(walk);
}

/// Gives the synthetic source's options for a scene of the tabletop box along the walk's first
/// poses.
/// @param count [in] the poses, from the walk's first.
/// @param ending [in] what the source reports after them.
/// @param message [in] the message it reports with that.
/// @return the options.
gca_test::synthetic_source_options walk_options(
    std::size_t count, gca_test::synthetic_ending ending = gca_test::synthetic_ending::end,
    const std::string &message = "the sequence's end")
{
    std::vector<Eigen::Isometry3d> poses = walk_poses_of();
    poses.resize(std::min(count, poses.size()));
    return {.description = small_description,
            .scene = {.plane = gca_test::tabletop_plane,
                      .box = gca_test::tabletop_box,
                      .trajectory = poses},
            .ending = ending,
            .ending_message = message,
            .open_refusal = {},
            .imu = std::nullopt};
}

/// What a case saw, in order, on the scan thread: the states, and the source's openings and
/// closings; guarded by its lock, as the test's thread reads it after the scan.
struct event_log
{
    std::mutex mutex;                    ///< guards the rest
    std::vector<gca::scan_state> states; ///< every state published, in order
    std::vector<std::string> events;     ///< "open", "close" and each phase as it began
};

/// A frame source in front of the synthetic source: records its openings and closings, and may
/// leave gaps in the indices, deliver a frame of another size, or describe itself with a string
/// of its case's.
class recorded_source final : public gca::frame_source
{
public:
    /// Builds the source in front of a synthetic source.
    /// @param options [in] the synthetic source's options; taken over.
    /// @param log [in,out] the case's log; outlives the source.
    recorded_source(gca_test::synthetic_source_options options, event_log &log)
        : m_inner(std::move(options))
        , m_log(log)
    {
    }

    /// Leaves out the frames of some indices: the inner source's next frame is given instead.
    /// @param indices [in] the indices left out.
    void skip(std::set<std::uint64_t> indices)
    {
        m_skipped = std::move(indices);
    }

    /// Gives a frame of another size, 4x4 px, in place of the frame of an index.
    /// @param index [in] the index.
    void odd_size_at(std::uint64_t index)
    {
        m_odd_index = index;
    }

    /// Describes the source with another camera's name.
    /// @param camera [in] the name.
    void describe_as(std::string camera)
    {
        m_camera = std::move(camera);
    }

    gca::open_result open(const gca::frame_size &size) noexcept override
    {
        record("open");
        return m_inner.open(size);
    }

    gca::next_result next(gca::frame &out) noexcept override
    {
        gca::next_result result = m_inner.next(out);
        while (result.status == gca::next_status::frame && m_skipped.count(out.index()) > 0)
            result = m_inner.next(out);
        if (result.status == gca::next_status::frame && m_odd_index == out.index())
        {
            gca::frame_description odd = out.description();
            odd.size = {4, 4};
            const std::uint64_t index = out.index();
            const std::uint64_t time = out.timestamp_us();
            out.prepare(odd, index, time);
            std::fill(out.depth().begin(), out.depth().end(), std::uint16_t{500});
            std::fill(out.color().begin(), out.color().end(), std::uint8_t{128});
        }
        return result;
    }

    void close() noexcept override
    {
        record("close");
        m_inner.close();
    }

    gca::source_description describe() const noexcept override
    {
        gca::source_description described = m_inner.describe();
        if (!m_camera.empty())
            described.camera = m_camera;
        return described;
    }

    /// Tells a test what the inner source renders.
    /// @return its known surfaces.
    const std::vector<gca_test::synthetic_surface> &surfaces() const noexcept
    {
        return m_inner.surfaces();
    }

private:
    /// Records an event in the log.
    /// @param event [in] the event.
    void record(const char *event) noexcept
    {
        const std::lock_guard<std::mutex> lock(m_log.mutex);
        m_log.events.push_back(event);
    }

    gca_test::synthetic_source m_inner;
    event_log &m_log;
    std::set<std::uint64_t> m_skipped;
    std::optional<std::uint64_t> m_odd_index;
    std::string m_camera;
};

/// A writer in front of the real one, made slow: its queue reports no room at its has_room()
/// calls 1, 4, 7 and so on, as a writer that falls behind does.
class slow_writer final : public gca::sequence_sink
{
public:
    gca::open_result open(const std::filesystem::path &path,
                          const gca::frame_description &description,
                          std::size_t queue_frames) noexcept override
    {
        return m_inner.open(path, description, queue_frames);
    }

    bool has_room() noexcept override
    {
        const std::size_t call = m_calls++;
        return call % 3 != 1 && m_inner.has_room();
    }

    gca::push_result push(gca::frame &in_out, const gca::frame_tracking &tracking) noexcept override
    {
        return m_inner.push(in_out, tracking);
    }

    gca::write_result close() noexcept override
    {
        return m_inner.close();
    }

private:
    gca::sequence_writer m_inner; ///< the real writer
    std::size_t m_calls = 0;      ///< has_room()'s calls so far
};

/// A writer in front of a real one that waits for room in its queue instead of reporting it full
/// (gca::queue_full::wait), as a disk that keeps up does, so that a case's counts do not hang on
/// the disk's speed, as during a build beside the test; it can also refuse to open, or fail at its
/// close after the real writer's, with a reason of its case's.
class patient_writer final : public gca::sequence_sink
{
public:
    /// Makes open() refuse with a reason, without opening.
    /// @param reason [in] the reason.
    void refuse_open(std::string reason)
    {
        m_open_refusal = std::move(reason);
    }

    /// Makes close() fail with a reason, after the real writer's close.
    /// @param reason [in] the reason.
    void fail_close(std::string reason)
    {
        m_close_failure = std::move(reason);
    }

    gca::open_result open(const std::filesystem::path &path,
                          const gca::frame_description &description,
                          std::size_t queue_frames) noexcept override
    {
        if (!m_open_refusal.empty())
            return {false, m_open_refusal};
        return m_inner.open(path, description, queue_frames);
    }

    bool has_room() noexcept override
    {
        return m_inner.has_room();
    }

    gca::push_result push(gca::frame &in_out, const gca::frame_tracking &tracking) noexcept override
    {
        return m_inner.push(in_out, tracking);
    }

    gca::write_result close() noexcept override
    {
        const gca::write_result closed = m_inner.close();
        if (closed.written && !m_close_failure.empty())
            return {false, m_close_failure};
        return closed;
    }

private:
    gca::sequence_writer m_inner{gca::queue_full::wait}; ///< the real writer, which waits
    std::string m_open_refusal;  ///< the reason open() refuses with; empty: it opens
    std::string m_close_failure; ///< the reason close() fails with; empty: as the real one
};

/// The phases a case's states went through, each once where it repeats.
/// @param log [in,out] the case's log.
/// @return the phases in order.
std::vector<gca::scan_phase> phases_of(event_log &log)
{
    const std::lock_guard<std::mutex> lock(log.mutex);
    std::vector<gca::scan_phase> phases;
    for (const gca::scan_state &s : log.states)
        if (phases.empty() || phases.back() != s.phase)
            phases.push_back(s.phase);
    return phases;
}

/// Gives a listener that records every state and each phase as it begins in a log, and then calls
/// a case's own reaction.
/// @param log [in,out] the case's log; outlives the scan interface.
/// @param react [in] called with each state after it is recorded; may give commands.
/// @return the listener.
std::function<void(const gca::scan_state &)> recorder(
    event_log &log, std::function<void(const gca::scan_state &)> react = {})
{
    return [&log, react](const gca::scan_state &s) {
        {
            const std::lock_guard<std::mutex> lock(log.mutex);
            if (log.states.empty() || log.states.back().phase != s.phase)
                log.events.push_back(std::string("phase ") + std::to_string(int(s.phase)));
            log.states.push_back(s);
        }
        if (react)
            react(s);
    };
}

/// Waits until the scan interface is idle after a scan, its outcome or a refusal published.
/// @param scanner [in] the scan interface.
/// @param after [in] the version of the state before the case's start.
/// @param what [in] the case, named in the check line.
/// @return the state then.
gca::scan_state wait_for_idle(const gca::scan_interface &scanner, std::uint64_t after,
                              const std::string &what)
{
    const auto begin = std::chrono::steady_clock::now();
    while (true)
    {
        const gca::scan_state s = scanner.state();
        if (s.version > after && s.phase == gca::scan_phase::idle)
            return s;
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() >
            idle_wait_s)
        {
            check(false,
                  (what + ": idle again within " + std::to_string(idle_wait_s) + " s").c_str());
            return s;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

/// Lists a directory's entries' names.
/// @param directory [in] the directory.
/// @return the names, sorted; none when it does not exist.
std::vector<std::string> names_in(const std::filesystem::path &directory)
{
    std::vector<std::string> names;
    std::error_code error;
    for (const auto &entry : std::filesystem::directory_iterator(directory, error))
        names.push_back(entry.path().filename().string());
    std::sort(names.begin(), names.end());
    return names;
}

/// Prints a case's end and outcome.
/// @param what [in] the case.
/// @param s [in] the state at its end.
void print_end(const std::string &what, const gca::scan_state &s)
{
    std::printf(
        "%s: %llu taken, %llu fused, %llu lost, %llu not taken, ended %s%s%s, outcome %d%s%s, "
        "refusal \"%s\"\n",
        what.c_str(), static_cast<unsigned long long>(s.frames.taken),
        static_cast<unsigned long long>(s.frames.fused),
        static_cast<unsigned long long>(s.frames.lost),
        static_cast<unsigned long long>(s.frames.not_taken),
        s.end ? std::string(gca::name_of(*s.end)).c_str() : "(none)",
        s.end_message.empty() ? "" : ": ", s.end_message.c_str(), int(s.outcome),
        s.outcome_message.empty() ? "" : ": ", s.outcome_message.c_str(), s.refusal.c_str());
}

// ------------------------------------------------------------------------------------ the walk

/// The walk: the states in order, the second start refused, the source closed and the live model
/// freed before refinement, the files, the parameters, the trajectory and the full model.
/// @param directory [in] the test's directory.
void check_walk(const temporary_directory &directory)
{
    const std::filesystem::path scans = directory / "walk";
    std::filesystem::create_directory(scans);
    event_log log;
    recorded_source source(walk_options(walk_poses), log);
    patient_writer writer;
    std::future<gca::command_reply> second_start;
    std::optional<gca::scan_interface> scanner;
    // The start's answer, as the listener sees it: a copy of its own, stored before the flag.
    std::shared_future<gca::command_reply> answer_seen;
    std::atomic<bool> answer_stored{false};
    bool answered_before_scanning = true;
    bool snapshots_match = true;
    std::size_t snapshots_read = 0;
    const auto react = [&](const gca::scan_state &s) {
        if (s.phase == gca::scan_phase::starting)
        {
            for (int k = 0; k < 2000 && !answer_stored; k++)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (s.phase == gca::scan_phase::scanning && s.frames.taken == 0 && answer_stored)
            answered_before_scanning =
                answer_seen.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        if (s.phase == gca::scan_phase::scanning)
        {
            // The snapshot a reader copies holds what the frame changed.
            const gca::scan_state now = scanner->state();
            snapshots_match = snapshots_match && now.version == s.version &&
                              now.frames == s.frames && now.too_close == s.too_close &&
                              now.tracking == s.tracking && now.lost_cause == s.lost_cause;
            snapshots_read++;
        }
        if (s.phase == gca::scan_phase::scanning && s.frames.taken == 5 && !second_start.valid())
            second_start = scanner->start(gca::object_class::large_object,
                                          gca::parameters_of(gca::object_class::large_object));
        if (s.phase == gca::scan_phase::scanning && s.frames.taken == walk_poses)
            (void)scanner->stop();
    };
    scanner.emplace(
        gca::scan_options{scans, small_description.size, test_commit, recorder(log, react)}, source,
        writer);
    const gca::scan_state before = scanner->state();
    check(before.phase == gca::scan_phase::idle && before.version == 0,
          "walk: idle before the start, version 0");
    const auto begin = std::chrono::steady_clock::now();
    const std::shared_future<gca::command_reply> answer =
        scanner->start(gca::object_class::tabletop_object, test_set()).share();
    answer_seen = answer;
    answer_stored = true;
    const gca::command_reply started = answer.get();
    check(started.accepted, ("walk: the start accepted; " + started.refusal).c_str());
    const gca::scan_state end = wait_for_idle(*scanner, before.version, "walk");
    check(!answered_before_scanning,
          "walk: the scanning state published before the start's answer");
    check(snapshots_match && snapshots_read > walk_poses,
          "walk: every snapshot read while scanning holds the frame's counts, tracking and notice");
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    print_end("walk", end);
    std::printf("walk: scanned and refined in %.1f s\n", seconds);

    const std::vector<gca::scan_phase> phases = phases_of(log);
    check(phases == std::vector<gca::scan_phase>{gca::scan_phase::starting,
                                                 gca::scan_phase::scanning,
                                                 gca::scan_phase::refining, gca::scan_phase::idle},
          "walk: the states starting, scanning, refining, idle, in that order");
    const gca::command_reply refused =
        second_start.valid() ? second_start.get() : gca::command_reply{true, "no second start"};
    gca_test::check_refusal("walk, a second start while scanning", !refused.accepted,
                            refused.refusal, refused.refusal == "a scan is in progress (scanning)");
    check(end.outcome == gca::scan_outcome::refined && end.end == gca::scan_end::stopped &&
              end.frames == gca::scan_frame_counts{walk_poses, walk_poses, 0, 0} &&
              end.refusal == "a scan is in progress (scanning)",
          "walk: refined, stopped, every frame fused, the second start's refusal in the state");
    {
        const std::lock_guard<std::mutex> lock(log.mutex);
        const auto at = [&log](const std::string &event) {
            return std::find(log.events.begin(), log.events.end(), event) - log.events.begin();
        };
        const std::string refining = "phase " + std::to_string(int(gca::scan_phase::refining));
        check(at("close") < at(refining) && at("open") < at("close"),
              "walk: the source opened, then closed before refining began");
    }

    const std::filesystem::path scan = scans / end.scan;
    check(names_in(scans) == std::vector<std::string>{end.scan} &&
              names_in(scan) == std::vector<std::string>{"frames.sequence", "full.ply",
                                                         "parameters.json", "trajectory.txt"},
          "walk: one scan's directory with its four files");
    const gca::read_result<gca::scan_parameters> parameters =
        gca::read_parameters(scan / "parameters.json");
    check(parameters.value.has_value(),
          ("walk: the parameters read; " + parameters.reason).c_str());
    if (parameters.value)
    {
        const gca::scan_parameters &p = *parameters.value;
        check(p.commit == test_commit && p.object_class == gca::object_class::tabletop_object &&
                  p.parameters == test_set(),
              "walk: the parameters hold the commit, the object class and its set");
        check(p.stream == small_description && p.source == source.describe() &&
                  p.frames == end.frames && p.end == gca::scan_end::stopped,
              "walk: the parameters hold the stream, the source, the counts and the end");
        check(p.start <= p.stop && gca::utc_text(p.start) == end.scan,
              "walk: the scan's directory named by its start time, which is before its stop");
    }

    const std::vector<Eigen::Isometry3d> known = walk_poses_of();
    const gca::read_result<std::vector<gca::trajectory_entry>> trajectory =
        gca::read_trajectory(scan / "trajectory.txt");
    check(trajectory.value && trajectory.value->size() == walk_poses,
          ("walk: the trajectory read, a line for each frame; " + trajectory.reason).c_str());
    gca_test::transform_error worst{0.0, 0.0};
    bool every_pose = true;
    for (std::size_t k = 0; trajectory.value && k < trajectory.value->size(); k++)
    {
        const gca::trajectory_entry &e = (*trajectory.value)[k];
        every_pose = every_pose && e.index == k && e.pose.has_value();
        if (!e.pose)
            continue;
        Eigen::Isometry3d found = Eigen::Isometry3d::Identity();
        found.linear() = e.pose->orientation.toRotationMatrix();
        found.translation() = e.pose->position_m;
        const gca_test::transform_error error =
            gca_test::error_of(found, known[0].inverse() * known[k]);
        worst.rotation_deg = std::max(worst.rotation_deg, error.rotation_deg);
        worst.translation_m = std::max(worst.translation_m, error.translation_m);
    }
    check(every_pose, "walk: every frame of the trajectory with its index and a pose");
    check_within("walk: the trajectory's largest turn from the known one, deg", worst.rotation_deg,
                 trajectory_tolerance.rotation_deg);
    check_within("walk: the trajectory's largest shift from the known one, m", worst.translation_m,
                 trajectory_tolerance.translation_m);

    const gca::read_result<std::vector<gca::model_point>> model =
        gca::read_model(scan / "full.ply");
    check(model.value && !model.value->empty(),
          ("walk: the full model read; " + model.reason).c_str());
    std::vector<double> distances_mm;
    for (std::size_t k = 0; model.value && k < model.value->size(); k++)
    {
        const Eigen::Vector3d world = known[0] * (*model.value)[k].position_m.cast<double>();
        double nearest = std::numeric_limits<double>::infinity();
        for (const gca_test::synthetic_surface &surface : source.surfaces())
            nearest = std::min(nearest, gca_test::distance_from(surface, world));
        distances_mm.push_back(1e3 * nearest);
    }
    std::printf("walk: the full model's %zu points from the nearest known surface: median %.4g "
                "mm, 99th percentile %.4g mm, largest %.4g mm\n",
                distances_mm.size(), gca_test::quantile(distances_mm, 0.5),
                gca_test::quantile(distances_mm, 0.99), gca_test::quantile(distances_mm, 1.0));
    check_within("walk: the points' median distance from the known surfaces, mm",
                 gca_test::quantile(distances_mm, 0.5), distance_median_mm);
    check_within("walk: the points' 99th percentile distance from the known surfaces, mm",
                 gca_test::quantile(distances_mm, 0.99), distance_p99_mm);
    check_within("walk: the points' largest distance from the known surfaces, mm",
                 gca_test::quantile(distances_mm, 1.0), distance_max_mm);
}

// ----------------------------------------------------------------------------- the short cases

/// A short scan's case: its scans directory, log, source, writer and scan interface.
struct short_case
{
    std::string name;                           ///< the case, named in the check lines
    std::filesystem::path scans;                ///< its scans directory
    event_log log;                              ///< its log
    recorded_source source;                     ///< its source
    patient_writer patient;                     ///< the writer, unless the case makes it slow
    std::optional<slow_writer> slow;            ///< the slow writer, when the case asks for it
    std::optional<gca::scan_interface> scanner; ///< its scan interface

    /// Builds the case and its scans directory.
    /// @param what [in] the case's name.
    /// @param directory [in] the test's directory.
    /// @param options [in] the synthetic source's options.
    short_case(const std::string &what, const temporary_directory &directory,
               gca_test::synthetic_source_options options)
        : name(what)
        , scans(directory / what)
        , source(std::move(options), log)
    {
        std::filesystem::create_directory(scans);
    }

    /// Starts the scan interface.
    /// @param react [in] the case's reaction to each state.
    /// @param slow_writing [in] whether the writer is made slow.
    void begin(std::function<void(const gca::scan_state &)> react = {}, bool slow_writing = false)
    {
        gca::sequence_sink *writer = &patient;
        if (slow_writing)
            writer = &slow.emplace();
        scanner.emplace(gca::scan_options{scans, small_description.size, test_commit,
                                          recorder(log, std::move(react))},
                        source, *writer);
    }

    /// Starts a scan and waits for idle.
    /// @param set [in] the set.
    /// @param reply [out] the start's answer.
    /// @return the state at the end.
    gca::scan_state run(const gca::object_class_parameters &set, gca::command_reply &reply)
    {
        const std::uint64_t before = scanner->state().version;
        reply = scanner->start(gca::object_class::tabletop_object, set).get();
        const gca::scan_state end = wait_for_idle(*scanner, before, name);
        print_end(name, end);
        return end;
    }
};

/// A discard while scanning, at the fifth frame: accepted, the scan ends, nothing on disk.
/// @param directory [in] the test's directory.
void check_discard_scanning(const temporary_directory &directory)
{
    short_case c("discard while scanning", directory, walk_options(short_poses));
    std::future<gca::command_reply> discarded;
    c.begin([&c, &discarded](const gca::scan_state &s) {
        if (s.phase == gca::scan_phase::scanning && s.frames.taken == 5 && !discarded.valid())
            discarded = c.scanner->discard();
    });
    gca::command_reply started;
    const gca::scan_state end = c.run(test_set(), started);
    check(started.accepted && discarded.valid() && discarded.get().accepted,
          "discard while scanning: the start and the discard accepted");
    check(end.outcome == gca::scan_outcome::discarded && end.frames.taken == 5 &&
              names_in(c.scans).empty(),
          "discard while scanning: discarded at the fifth frame, nothing on disk");
    check(phases_of(c.log) == std::vector<gca::scan_phase>{gca::scan_phase::starting,
                                                           gca::scan_phase::scanning,
                                                           gca::scan_phase::idle},
          "discard while scanning: starting, scanning, idle");
}

/// A discard while refining, a stop given with it refused: nothing on disk.
/// @param directory [in] the test's directory.
void check_discard_refining(const temporary_directory &directory)
{
    short_case c("discard while refining", directory, walk_options(short_poses));
    std::future<gca::command_reply> stopped;
    std::future<gca::command_reply> discarded;
    c.begin([&c, &stopped, &discarded](const gca::scan_state &s) {
        if (s.phase == gca::scan_phase::refining && !discarded.valid())
        {
            stopped = c.scanner->stop();
            discarded = c.scanner->discard();
        }
    });
    gca::command_reply started;
    const gca::scan_state end = c.run(test_set(), started);
    const gca::command_reply stop_reply =
        stopped.valid() ? stopped.get() : gca::command_reply{true, {}};
    gca_test::check_refusal("discard while refining, a stop while refining", !stop_reply.accepted,
                            stop_reply.refusal,
                            stop_reply.refusal.find("the scan is refining") != std::string::npos);
    check(started.accepted && discarded.valid() && discarded.get().accepted,
          "discard while refining: the start and the discard accepted");
    check(end.outcome == gca::scan_outcome::discarded && !end.end && names_in(c.scans).empty(),
          "discard while refining: discarded after the source's end, no end key, nothing on disk");
    check(phases_of(c.log) ==
              std::vector<gca::scan_phase>{gca::scan_phase::starting, gca::scan_phase::scanning,
                                           gca::scan_phase::refining, gca::scan_phase::idle},
          "discard while refining: starting, scanning, refining, idle");
}

/// A source that times out, and one that fails, after short_poses frames: the scan ends with the
/// reason, and what it had is refined.
/// @param directory [in] the test's directory.
void check_source_endings(const temporary_directory &directory)
{
    const struct
    {
        const char *what;
        gca_test::synthetic_ending ending;
        gca::scan_end end;
    } cases[] = {
        {"a timeout", gca_test::synthetic_ending::timeout, gca::scan_end::source_timeout},
        {"a failure", gca_test::synthetic_ending::failure, gca::scan_end::source_failure},
    };
    for (const auto &e : cases)
    {
        const std::string message = std::string("the camera's ") + e.what;
        short_case c(e.what, directory, walk_options(short_poses, e.ending, message));
        c.begin();
        gca::command_reply started;
        const gca::scan_state end = c.run(test_set(), started);
        check(
            started.accepted && end.end == e.end && end.end_message == message &&
                end.outcome == gca::scan_outcome::refined && end.frames.taken == short_poses,
            (std::string(e.what) + ": ended with the reason, every frame taken, refined").c_str());
        const gca::read_result<gca::scan_parameters> p =
            gca::read_parameters(c.scans / end.scan / "parameters.json");
        check(p.value && p.value->end == e.end &&
                  std::filesystem::exists(c.scans / end.scan / "full.ply"),
              (std::string(e.what) + ": the parameters' end, and the full model").c_str());
    }
}

/// A source that refuses to open, and one whose description the parameters cannot hold: the
/// start refused with the reason, idle again, the source closed, nothing on disk.
/// @param directory [in] the test's directory.
void check_start_refusals(const temporary_directory &directory)
{
    gca_test::synthetic_source_options refusing = walk_options(short_poses);
    refusing.open_refusal = "no camera on the bus";
    short_case missing("an opening refused", directory, std::move(refusing));
    missing.begin();
    gca::command_reply reply;
    gca::scan_state end = missing.run(test_set(), reply);
    gca_test::check_refusal("an opening refused", !reply.accepted, reply.refusal,
                            reply.refusal ==
                                "the frame source refused to open: no camera on the bus");
    check(end.refusal == reply.refusal && end.outcome == gca::scan_outcome::none &&
              names_in(missing.scans).empty(),
          "an opening refused: idle, the reason in the state, nothing on disk");

    short_case odd_name("a description refused", directory, walk_options(short_poses));
    odd_name.source.describe_as("sequence /home/o'neill/scan");
    odd_name.begin();
    end = odd_name.run(test_set(), reply);
    gca_test::check_refusal(
        "a description refused", !reply.accepted, reply.refusal,
        reply.refusal.find("cannot go into the scan's parameters: \"camera\" in \"source\" holds "
                           "the byte 0x27") != std::string::npos);
    bool closed = false;
    {
        const std::lock_guard<std::mutex> lock(odd_name.log.mutex);
        closed = odd_name.log.events.size() >= 2 &&
                 std::count(odd_name.log.events.begin(), odd_name.log.events.end(), "open") ==
                     std::count(odd_name.log.events.begin(), odd_name.log.events.end(), "close");
    }
    check(closed && names_in(odd_name.scans).empty(),
          "a description refused: the source closed, nothing on disk");

    short_case silent("no first frame", directory,
                      walk_options(0, gca_test::synthetic_ending::timeout, "nothing came"));
    silent.begin();
    end = silent.run(test_set(), reply);
    gca_test::check_refusal("no first frame", !reply.accepted, reply.refusal,
                            reply.refusal == "the frame source gave no first frame: nothing came");
    {
        const std::lock_guard<std::mutex> lock(silent.log.mutex);
        const auto &events = silent.log.events;
        const auto opened = std::find(events.begin(), events.end(), "open");
        closed = opened != events.end() && std::find(opened, events.end(), "close") != events.end();
    }
    check(closed && names_in(silent.scans).empty(),
          "no first frame: the source opened and closed, nothing on disk");

    short_case unwritable("a sequence refused", directory, walk_options(short_poses));
    unwritable.patient.refuse_open("the drive is read only");
    unwritable.begin();
    end = unwritable.run(test_set(), reply);
    gca_test::check_refusal("a sequence refused", !reply.accepted, reply.refusal,
                            reply.refusal == "the scan's sequence: the drive is read only");
    check(names_in(unwritable.scans).empty(),
          "a sequence refused: the scan's directory removed, nothing on disk");
}

/// A stop and a discard given while the scan starts, answered once its first frame has come,
/// and a discard while idle, refused.
/// @param directory [in] the test's directory.
void check_starting_commands(const temporary_directory &directory)
{
    short_case stopping("a stop while starting", directory, walk_options(short_poses));
    std::future<gca::command_reply> stopped;
    stopping.begin([&stopping, &stopped](const gca::scan_state &s) {
        if (s.phase == gca::scan_phase::starting && !stopped.valid())
            stopped = stopping.scanner->stop();
    });
    gca::command_reply started;
    gca::scan_state end = stopping.run(test_set(), started);
    check(started.accepted && stopped.valid() && stopped.get().accepted &&
              end.end == gca::scan_end::stopped && end.frames.taken == 1 &&
              end.outcome == gca::scan_outcome::refined,
          "a stop while starting: accepted after the first frame, stopped there, refined");

    short_case discarding("a discard while starting", directory, walk_options(short_poses));
    std::future<gca::command_reply> discarded;
    discarding.begin([&discarding, &discarded](const gca::scan_state &s) {
        if (s.phase == gca::scan_phase::starting && !discarded.valid())
            discarded = discarding.scanner->discard();
    });
    end = discarding.run(test_set(), started);
    check(started.accepted && discarded.valid() && discarded.get().accepted &&
              end.outcome == gca::scan_outcome::discarded && !end.end &&
              names_in(discarding.scans).empty(),
          "a discard while starting: accepted after the first frame, nothing on disk, no end key");

    short_case idle("a discard while idle", directory, walk_options(short_poses));
    idle.begin();
    const gca::command_reply refused = idle.scanner->discard().get();
    gca_test::check_refusal("a discard while idle", !refused.accepted, refused.refusal,
                            refused.refusal == "no scan is in progress" &&
                                idle.scanner->state().refusal == refused.refusal);
}

/// A writer that fails at its close: the scan ends writer_failed with the reason, and the
/// sequence, whole as the real writer closed it, is refined.
/// @param directory [in] the test's directory.
void check_writer_close_failure(const temporary_directory &directory)
{
    short_case c("a close failed", directory, walk_options(short_poses));
    c.patient.fail_close("the disk filled");
    c.begin();
    gca::command_reply started;
    const gca::scan_state end = c.run(test_set(), started);
    const gca::read_result<gca::scan_parameters> p =
        gca::read_parameters(c.scans / end.scan / "parameters.json");
    check(started.accepted && end.end == gca::scan_end::writer_failed &&
              end.end_message == "the disk filled" && p.value &&
              p.value->end == gca::scan_end::writer_failed,
          "a close failed: ended writer_failed with the reason, in the state and the parameters");
}

/// A discard while refining that the refinement does not see, as it fails before its first
/// frame: the scan still discards it before it is idle, nothing on disk.
/// @param directory [in] the test's directory.
void check_discard_failing_refinement(const temporary_directory &directory)
{
    short_case c("a discard while refinement fails", directory, walk_options(short_poses));
    std::future<gca::command_reply> discarded;
    c.begin([&c, &discarded](const gca::scan_state &s) {
        if (s.phase == gca::scan_phase::refining && !discarded.valid())
            discarded = c.scanner->discard();
    });
    gca::object_class_parameters set = test_set();
    set.full_voxel_size_m = 0.0001;
    gca::command_reply started;
    const gca::scan_state end = c.run(set, started);
    check(started.accepted && discarded.valid() && discarded.get().accepted &&
              end.outcome == gca::scan_outcome::discarded && names_in(c.scans).empty(),
          "a discard while refinement fails: discarded, nothing on disk");
}

/// A discard while refining whose scan's directory cannot be removed, its permissions taken away
/// by the test: the scan unfinished with the reason, never discarded.
/// @param directory [in] the test's directory.
void check_discard_not_removed(const temporary_directory &directory)
{
    short_case c("a discard not removed", directory, walk_options(short_poses));
    std::future<gca::command_reply> discarded;
    std::filesystem::path scan;
    c.begin([&c, &discarded, &scan](const gca::scan_state &s) {
        if (s.phase == gca::scan_phase::refining && !discarded.valid())
        {
            scan = c.scans / s.scan;
            std::filesystem::permissions(scan, std::filesystem::perms::owner_read |
                                                   std::filesystem::perms::owner_exec);
            discarded = c.scanner->discard();
        }
    });
    gca::command_reply started;
    const gca::scan_state end = c.run(test_set(), started);
    std::error_code ignored;
    std::filesystem::permissions(scan, std::filesystem::perms::owner_all, ignored);
    check(started.accepted && discarded.valid() && discarded.get().accepted &&
              end.outcome == gca::scan_outcome::unfinished &&
              end.outcome_message.find("the discard could not remove") == 0 &&
              std::filesystem::exists(scan / "frames.sequence"),
          "a discard not removed: unfinished with the reason, its files still there");
}

/// A command queued while the scan interface ends: refused with the reason, never left without
/// an answer. The listener, at the scan's last state, queues a start, has the test end the scan
/// interface, and waits until a command is refused at once, which says that it is ending; the
/// start, still queued, is then the end's to refuse.
/// @param directory [in] the test's directory.
void check_ending_with_a_command(const temporary_directory &directory)
{
    short_case c("ending with a command queued", directory, walk_options(short_poses));
    gca::scan_interface *self = nullptr;
    std::future<gca::command_reply> queued;
    std::atomic<bool> end_it{false};
    c.begin([&self, &queued, &end_it](const gca::scan_state &s) {
        if (s.phase != gca::scan_phase::idle || s.outcome == gca::scan_outcome::none ||
            queued.valid())
            return;
        queued = self->start(gca::object_class::tabletop_object, test_set());
        end_it = true;
        for (int k = 0; k < 5000; k++)
        {
            if (self->stop().wait_for(std::chrono::seconds(0)) == std::future_status::ready)
                return;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    self = &*c.scanner;
    const gca::command_reply started =
        c.scanner->start(gca::object_class::tabletop_object, test_set()).get();
    for (int k = 0; k < 50000 && !end_it; k++)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    c.scanner.reset();
    gca::command_reply answer{true, "no answer"};
    try
    {
        answer = queued.get();
    }
    catch (const std::future_error &e)
    {
        answer = {true, std::string("no answer: ") + e.what()};
    }
    gca_test::check_refusal("a command queued while ending", started.accepted && !answer.accepted,
                            answer.refusal, answer.refusal == "the scan interface is ending");
}

/// Gaps in the indices and a writer made slow: the counts, and the sequence holding exactly the
/// frames taken, each with the trajectory's pose or mark.
/// @param directory [in] the test's directory.
void check_not_taken(const temporary_directory &directory)
{
    // 30 frames, 7 and 8 left out: 28 delivered. The slow writer has no room at its calls 1, 4,
    // ..., 25, nine of the 28, so 19 frames are taken and 2 + 9 = 11 are not [S: computed].
    short_case c("not taken", directory, walk_options(30));
    c.source.skip({7, 8});
    c.begin({}, true);
    gca::command_reply started;
    const gca::scan_state end = c.run(test_set(), started);
    check(started.accepted && end.frames.taken == 19 && end.frames.not_taken == 11 &&
              end.frames.fused + end.frames.lost == 19 && end.outcome == gca::scan_outcome::refined,
          "not taken: 19 taken, 11 not taken, each taken frame fused or lost, refined");
    std::vector<std::uint64_t> expected;
    std::size_t call = 0;
    for (std::uint64_t index = 0; index < 30; index++)
    {
        if (index == 7 || index == 8)
            continue;
        if (call++ % 3 != 1)
            expected.push_back(index);
    }
    const std::filesystem::path scan = c.scans / end.scan;
    const gca::read_result<gca::scan_parameters> p = gca::read_parameters(scan / "parameters.json");
    check(p.value && p.value->frames == end.frames, "not taken: the parameters' counts");
    const gca::read_result<std::vector<gca::trajectory_entry>> trajectory =
        gca::read_trajectory(scan / "trajectory.txt");
    gca::sequence_reader reader(scan / "frames.sequence");
    bool same = trajectory.value && reader.open(small_description.size).opened &&
                trajectory.value->size() == expected.size();
    std::uint64_t poses = 0;
    gca::frame read;
    for (std::size_t k = 0; same && k < expected.size(); k++)
    {
        same = reader.next(read).status == gca::next_status::frame && read.index() == expected[k] &&
               (*trajectory.value)[k].index == expected[k];
        const gca::camera_pose *pose = std::get_if<gca::camera_pose>(&reader.tracking());
        const std::optional<gca::camera_pose> &written = (*trajectory.value)[k].pose;
        same = same &&
               std::holds_alternative<gca::tracking_loss>(reader.tracking()) != (pose != nullptr) &&
               written.has_value() == (pose != nullptr) &&
               (pose == nullptr || gca_test::same_pose(*pose, *written));
        poses += pose != nullptr;
    }
    same = same && reader.next(read).status == gca::next_status::end;
    check(same && poses == end.frames.fused,
          "not taken: the sequence holds exactly the frames taken, every fused one with its pose, "
          "bit for bit as the trajectory");
}

/// A frame tracking fails on, the sixth, of another size: the scan ends there, the frame taken and
/// written nowhere, what came before refined.
/// @param directory [in] the test's directory.
void check_tracking_failed(const temporary_directory &directory)
{
    short_case c("tracking failed", directory, walk_options(short_poses));
    c.source.odd_size_at(5);
    c.begin();
    gca::command_reply started;
    const gca::scan_state end = c.run(test_set(), started);
    check(started.accepted && end.end == gca::scan_end::tracking_failed &&
              end.end_message.find("is not the tracker's") != std::string::npos &&
              end.frames.taken == 6 && end.frames.fused + end.frames.lost == 5 &&
              end.outcome == gca::scan_outcome::refined,
          "tracking failed: ended at the sixth frame, which counts as taken, refined");
    const std::filesystem::path scan = c.scans / end.scan;
    const gca::read_result<gca::scan_parameters> p = gca::read_parameters(scan / "parameters.json");
    const gca::read_result<std::vector<gca::trajectory_entry>> trajectory =
        gca::read_trajectory(scan / "trajectory.txt");
    check(p.value && p.value->frames.taken == 6 && p.value->end == gca::scan_end::tracking_failed &&
              trajectory.value && trajectory.value->size() == 5,
          "tracking failed: the parameters count it, the trajectory and the sequence hold the five "
          "before");
}

/// A refinement that fails (a full voxel size the sparse TSDF refuses), and the scan interface's
/// end during a scan: unfinished, the sequence and the parameters kept, nothing else.
/// @param directory [in] the test's directory.
void check_unfinished(const temporary_directory &directory)
{
    const std::vector<std::string> unfinished_files = {"frames.sequence", "parameters.json"};
    {
        short_case c("refinement failed", directory, walk_options(short_poses));
        c.begin();
        gca::object_class_parameters set = test_set();
        set.full_voxel_size_m = 0.0001;
        gca::command_reply started;
        const gca::scan_state end = c.run(set, started);
        check(started.accepted && end.outcome == gca::scan_outcome::unfinished &&
                  end.outcome_message.find("refinement: ") == 0 &&
                  names_in(c.scans / end.scan) == unfinished_files,
              "refinement failed: unfinished with refinement's reason, its sequence and parameters "
              "kept");
    }
    short_case c("ended while scanning", directory, walk_options(walk_poses));
    std::atomic<bool> fifth{false};
    c.begin([&fifth](const gca::scan_state &s) {
        if (s.frames.taken >= 5)
            fifth = true;
    });
    const gca::command_reply started =
        c.scanner->start(gca::object_class::tabletop_object, test_set()).get();
    while (!fifth)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const std::string scan = c.scanner->state().scan;
    c.scanner.reset();
    const gca::read_result<gca::scan_parameters> p =
        gca::read_parameters(c.scans / scan / "parameters.json");
    check(started.accepted && names_in(c.scans / scan) == unfinished_files && p.value &&
              p.value->end == gca::scan_end::shut_down,
          "ended while scanning: unfinished, its sequence and parameters kept, ended shut_down");
}

/// The "too close" rule on built frames: a quarter of the pixels with depth nearer is too close,
/// fewer is not, pixels without depth do not count, a frame without depth is not; and the notice
/// in the state of a scan seen from 0.26 m.
/// @param directory [in] the test's directory.
void check_too_close(const temporary_directory &directory)
{
    const double threshold_m = 0.30;
    const std::uint32_t width = small_description.size.width;
    // Depth by column: the share of columns nearer than 0.30 m (at 299 mm), then far (at 500 mm),
    // and a share of columns without depth.
    const auto frame_of = [width](std::uint32_t nearer_columns, std::uint32_t empty_columns) {
        return gca_test::built_frame(
            small_description,
            [=](std::uint32_t u, std::uint32_t) {
                return std::uint16_t(u < nearer_columns                   ? 299
                                     : u < nearer_columns + empty_columns ? 0
                                                                          : 500);
            },
            gca_test::grey);
    };
    check(gca::too_close(frame_of(width / 4, 0), threshold_m),
          "too close: a quarter of the pixels nearer");
    check(!gca::too_close(frame_of(width / 4 - 1, 0), threshold_m),
          "too close: one column less than a quarter is not");
    check(gca::too_close(frame_of(width / 5, width / 5), threshold_m),
          "too close: a fifth nearer and a fifth without depth, a quarter of those with depth");
    check(!gca::too_close(frame_of(0, width), threshold_m), "too close: no depth is not");
    check(!gca::too_close(gca_test::built_frame(
                              small_description,
                              [](std::uint32_t, std::uint32_t) { return std::uint16_t{300}; },
                              gca_test::grey),
                          threshold_m),
          "too close: every pixel at the threshold, 300 mm, is not nearer");

    std::optional<Eigen::Isometry3d> near =
        gca_test::look_at(Eigen::Vector3d(0.0, 0.0, 0.26), Eigen::Vector3d(0.03, 0.0, 0.0), 0.0);
    check(near.has_value(), "too close: the near pose");
    gca_test::synthetic_source_options options = walk_options(0);
    options.scene.box.reset();
    options.scene.trajectory =
        std::vector<Eigen::Isometry3d>(3, near.value_or(Eigen::Isometry3d::Identity()));
    short_case c("too close", directory, std::move(options));
    bool notice = false;
    c.begin([&c, &notice](const gca::scan_state &s) {
        // The snapshot a reader copies, as the page will, not the listener's own state.
        if (s.phase == gca::scan_phase::scanning && s.frames.taken > 0)
            notice = notice || c.scanner->state().too_close;
    });
    gca::command_reply started;
    c.run(test_set(), started);
    check(started.accepted && notice, "too close: the notice in the state a reader copies");
}
} // namespace

/// Runs the cases and prints the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    const auto start = std::chrono::steady_clock::now();
    const temporary_directory directory("test_scan_interface");
    check_walk(directory);
    check_discard_scanning(directory);
    check_discard_refining(directory);
    check_source_endings(directory);
    check_start_refusals(directory);
    check_starting_commands(directory);
    check_writer_close_failure(directory);
    check_discard_failing_refinement(directory);
    check_discard_not_removed(directory);
    check_ending_with_a_command(directory);
    check_not_taken(directory);
    check_tracking_failed(directory);
    check_unfinished(directory);
    check_too_close(directory);
    gca_test::check_no_cuda_error("scan interface");
    std::printf("test_scan_interface: %.1f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    return gca_test::check_summary("test_scan_interface");
}
