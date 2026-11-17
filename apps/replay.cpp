// SPDX-License-Identifier: 0BSD
/// @file replay.cpp
/// The replay tool (R19): runs a sequence through the scanner, with no camera and no page. It
/// drives the scan interface (scan/scan_interface.hpp) as a test does: the sequence as the frame
/// source, a writer that waits for room, so that every frame is taken, as fast as the GPU and the
/// disk allow; the set of an object class or of a kept scan's parameters; and the scan's
/// directory written into the directory it is given. With --times it prints each frame's stage
/// times, which the stage clock takes by waiting for the GPU after each stage
/// (tracking/stage_clock.hpp). Its lines are an interface (docs/methods/tracking.md, "The replay
/// tool's lines"). Links no OrbbecSDK.

#include "build_commit.hpp"
#include "camera/sequence_reader.hpp"
#include "camera/sequence_writer.hpp"
#include "replay_options.hpp"
#include "scan/parameters_file.hpp"
#include "scan/scan_interface.hpp"
#include "tracking/stage_clock.hpp"
#include "tracking/tracker.hpp"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace
{
/// Gives the size of the sequence's frames, the stream size the scan is set up for.
/// @param options [in] the command line read.
/// @return the size; nothing when its header is refused, "replay: " and the reason then printed on
///     stderr.
std::optional<gca::frame_size> stream_size_of(const gca::replay_options &options)
{
    const gca::sequence_header_result header = gca::read_sequence_header(options.sequence);
    if (!header.description)
    {
        std::cerr << "replay: " << header.reason << std::endl;
        return std::nullopt;
    }
    return header.description->size;
}

/// Names what tracking made of a frame, as a time line prints it.
/// @param state [in] the frame's state; none for a frame not taken.
/// @return name_of() the state (tracking/tracker.hpp): "tracking", "lost" or "failed"; "not_taken"
///     for none; "unknown" for a value that is no enumerator.
std::string_view state_name(const std::optional<gca::tracking_state> &state)
{
    if (!state)
        return "not_taken";
    const std::string_view name = gca::name_of(*state);
    return name.empty() ? std::string_view("unknown") : name;
}

/// Prints a frame's time line on stdout: "frame <index> <state>", then each stage's name and time
/// and "total" with the frame's, in ms with three decimals (docs/methods/tracking.md, "The replay
/// tool's lines").
/// @param times [in] the frame's times.
void print_time_line(const gca::frame_times &times)
{
    const std::string_view state = state_name(times.state);
    std::printf("frame %llu %.*s", static_cast<unsigned long long>(times.index),
                static_cast<int>(state.size()), state.data());
    for (std::size_t k = 0; k < gca::path_stage_count; k++)
    {
        const std::string_view name = gca::name_of(static_cast<gca::path_stage>(k));
        std::printf(" %.*s %.3f", static_cast<int>(name.size()), name.data(), times.stage_ms[k]);
    }
    std::printf(" total %.3f\n", times.total_ms);
}

/// What the replay's listener saw of the scan, on the scan thread; the main thread reads it after
/// the scan's end, which the promise hands over.
struct scan_watch
{
    using clock = std::chrono::steady_clock;
    std::optional<clock::time_point> scanning; ///< when the scanning phase was first published
    clock::time_point last_frame{};            ///< the last state published while scanning
    bool ended = false;                        ///< whether the end was handed over
    std::promise<gca::scan_state> end;         ///< the state when the scan is idle again
};

/// Builds the listener that watches the scan: the time of the scanning phase's first state and of
/// its last, and the state that ends the scan, idle with its outcome.
/// @param watch [in,out] what it records; outlives the scan interface.
/// @return the listener.
std::function<void(const gca::scan_state &)> watcher(scan_watch &watch)
{
    return [&watch](const gca::scan_state &s) {
        if (s.phase == gca::scan_phase::scanning)
        {
            watch.last_frame = scan_watch::clock::now();
            if (!watch.scanning)
                watch.scanning = watch.last_frame;
        }
        else if (s.phase == gca::scan_phase::idle && s.outcome != gca::scan_outcome::none &&
                 !watch.ended)
        {
            watch.ended = true;
            watch.end.set_value(s);
        }
    };
}

/// Prints the scan's last line on stdout when it was refined: "replay: <OUT>/<scan>: refined, end
/// <key>, frames taken <T> fused <F> lost <L> not taken <N>, <x> ms a frame", x the scanning
/// phase's time from its first state to its last divided by the frames taken; and on stderr
/// "replay: the scan ended before the sequence's end: <message>" when it ended otherwise.
/// Otherwise prints "replay: <OUT>/<scan>: unfinished: <message>" on stderr.
/// @param options [in] the command line read.
/// @param end [in] the state at the scan's end.
/// @param watch [in] what the listener saw.
/// @return whether the scan was refined.
bool report(const gca::replay_options &options, const gca::scan_state &end, const scan_watch &watch)
{
    const std::filesystem::path directory = options.scans / end.scan;
    // A replay gives no discard, so a scan that was not refined is unfinished.
    if (end.outcome != gca::scan_outcome::refined)
    {
        std::cerr << "replay: " << directory.string() << ": unfinished: " << end.outcome_message
                  << std::endl;
        return false;
    }
    const gca::scan_frame_counts &f = end.frames;
    const double scanning_ms =
        watch.scanning
            ? std::chrono::duration<double, std::milli>(watch.last_frame - *watch.scanning).count()
            : 0.0;
    const std::string_view end_key = end.end ? gca::name_of(*end.end) : std::string_view("none");
    std::printf("replay: %s: refined, end %.*s, frames taken %llu fused %llu lost %llu not taken "
                "%llu, %.3f ms a frame\n",
                directory.c_str(), static_cast<int>(end_key.size()), end_key.data(),
                static_cast<unsigned long long>(f.taken), static_cast<unsigned long long>(f.fused),
                static_cast<unsigned long long>(f.lost),
                static_cast<unsigned long long>(f.not_taken),
                f.taken > 0 ? scanning_ms / static_cast<double>(f.taken) : 0.0);
    std::fflush(stdout);
    if (end.end != gca::scan_end::source_end)
        std::cerr << "replay: the scan ended before the sequence's end: " << end.end_message
                  << std::endl;
    return true;
}

/// Runs the scan: the scan interface on the sequence, with a writer that waits for room and, when
/// the time lines are asked for, the stage clock; starts the scan with the set, waits for its
/// end and reports it.
/// @param options [in] the command line read.
/// @param set [in] the object class and set.
/// @param size [in] the sequence's frame size, in px.
/// @return whether the scan was refined; a start refused prints "replay: the scan's start was
///     refused: <reason>" on stderr.
bool run_scan(const gca::replay_options &options, const gca::replay_set &set,
              const gca::frame_size &size)
{
    gca::sequence_reader source(options.sequence);
    // The writer waits for room, so that the replay takes every frame (docs/methods/tracking.md,
    // "The scan's loop", step 2). No test makes its queue fill: the disk keeps up with a test's
    // frames, and a slower one cannot be made without a seam of its own, so this choice is checked
    // by review; a writer that reported a full queue here would show as frames not taken in the
    // last line.
    gca::sequence_writer writer(gca::queue_full::wait);
    std::unique_ptr<gca::stage_clock> clock;
    if (options.times)
        clock = std::make_unique<gca::stage_clock>(print_time_line);
    scan_watch watch;
    std::future<gca::scan_state> ended = watch.end.get_future();
    const gca::scan_options scan_options = {.scans_directory = options.scans,
                                            .stream_size = size,
                                            .commit = std::string(gca::build_commit()),
                                            .on_state = watcher(watch),
                                            .stage_times = clock.get()};
    gca::scan_interface scanner(scan_options, source, writer);
    const gca::command_reply started = scanner.start(set.kind, set.values).get();
    if (!started.accepted)
    {
        std::cerr << "replay: the scan's start was refused: " << started.refusal << std::endl;
        return false;
    }
    const gca::scan_state end = ended.get();
    return report(options, end, watch);
}
} // namespace

/// Replays a sequence: reads the command line, the set and the sequence's frame size, then runs
/// the scan and reports it.
/// @param argc [in] the argument count.
/// @param argv [in] the arguments (replay_options.hpp).
/// @return 0 when the scan was refined, its last line printed; 1 when the parameters file or the
///     sequence's header is refused, the scan's start is refused or the scan is left unfinished,
///     "replay: " and the reason on stderr; 2 when the command line is refused, the reason and the
///     usage on stderr (replay_options.hpp).
int main(int argc, char *argv[])
{
    const std::optional<gca::replay_options> options =
        gca::read_replay_options(argc, argv, std::cerr);
    if (!options)
        return 2;
    const std::optional<gca::replay_set> set = gca::replay_set_of(*options, std::cerr);
    if (!set)
        return 1;
    const std::optional<gca::frame_size> size = stream_size_of(*options);
    if (!size)
        return 1;
    return run_scan(*options, *set, *size) ? 0 : 1;
}
