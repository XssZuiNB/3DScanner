// SPDX-License-Identifier: 0BSD
/// @file scan_interface.hpp
/// The scan interface (the spec, "The scan interface"): everything the page can do with a scan
/// and see of it. Its commands start a scan of an object class on the frame source, stop it and
/// discard it; its state is a value. One thread, the scan thread, owns the scan and its GPU work,
/// the source, the writer's calls and the scan's files. Commands reach it through a queue and
/// are answered through a future; the state leaves it as a value that a reader copies, and that
/// copy is a snapshot no one changes afterwards. A scan's phases: idle, starting (the GPU's
/// buffers, the source opened, its first frame), scanning (the per-frame path for each frame the
/// sequence's writer has room for), refining (the source closed, the live model freed, the
/// sequence fused at the full voxel size) and idle again. The scan's files go into a directory of
/// the scans directory named by its start time (docs/SCAN_FILES.md, "A scan's directory"). The
/// live model is a second value beside the state, published under a lock of its own at most once
/// a live_model_interval while the scan tracks (scan/live_model.hpp). Also the rule of the "too
/// close" notice (R36).

#pragma once

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/sequence_writer.hpp"
#include "scan/live_model.hpp"
#include "scan/object_class.hpp"
#include "scan/parameters_file.hpp"
#include "tracking/stage_clock.hpp"
#include "tracking/tracker.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace gca
{
/// Says whether a frame is too close (R36; docs/methods/tracking.md, "Parameters"): at least a
/// quarter of its pixels with depth lie nearer than the threshold. A pixel with depth is one whose
/// depth value is above 0, beyond the far depth too; a pixel without depth does not count.
/// @param frame [in] the frame, in host memory.
/// @param threshold_m [in] the object class's too_close_depth_m, in m along the camera's z, > 0.
/// @return true when the frame has a pixel with depth and 4 times the pixels nearer than the
///         threshold are at least the pixels with depth; false for a frame without depth.
/// @note Host only; one pass over the depth picture, its time on the board in docs/HARDWARE.md,
///       "The per-frame path's stages". Allocates nothing and waits for nothing.
bool too_close(const frame &frame, double threshold_m) noexcept;

/// The scan interface's phase (the spec, "The scan interface").
enum class scan_phase
{
    idle,     ///< no scan; the source is closed
    starting, ///< a start was accepted: the GPU's buffers, the source opened, its first frame
    scanning, ///< the per-frame path runs on each frame the writer has room for
    refining, ///< the source closed, the sequence fused at the full voxel size
};

/// What became of the last scan.
enum class scan_outcome
{
    none,       ///< no scan has ended since the scan interface was built
    refined,    ///< refined: its full model and trajectory stand in its directory
    unfinished, ///< kept without a full model: refinement failed, the service stopped, a file
                ///< could not be written, or a discard could not remove the scan's directory;
                ///< the message says why
    discarded,  ///< discarded: nothing of it stays on disk
};

/// The scan interface's state: a value, copied out as a snapshot.
struct scan_state
{
    std::uint64_t version = 0;           ///< rises by one with each state published
    scan_phase phase = scan_phase::idle; ///< the phase
    /// The scan in progress or the last one: its directory's name in the scans directory, its
    /// start time (scan/utc_time.hpp); empty before the first scan's directory.
    std::string scan;
    /// The tracking state of the last frame taken, tracking or lost; none before a scan's first.
    std::optional<tracking_state> tracking;
    check_outcome lost_cause = check_outcome::passed; ///< the check's condition that failed, when
                                                      ///< the last frame taken was lost
    bool too_close = false;     ///< whether the last frame taken was too close (too_close())
    scan_frame_counts frames{}; ///< the scan's frames taken, fused, lost and not taken so far
    /// Why the scan in progress or the last one ended, as its parameters say (scan_end); none
    /// before it ended, and for a scan discarded, which has no parameters.
    std::optional<scan_end> end;
    std::string end_message; ///< what ended it, in plain English: the source's or the
                             ///< tracker's or the writer's message; empty for the operator's
                             ///< stop and the service's
    scan_outcome outcome = scan_outcome::none; ///< what became of the last scan
    std::string outcome_message; ///< why the last scan is unfinished, in plain English; empty
                                 ///< otherwise
    /// The last refusal: a command refused or a start that failed, with the reason; empty before
    /// the first.
    std::string refusal;
    /// Why a live model of the scan in progress or the last one could not be taken, the last
    /// time one could not, in plain English; empty while each was taken. The scan goes on.
    std::string live_model_failure;
};

/// The answer to a command.
struct command_reply
{
    bool accepted = false; ///< whether the command was taken
    std::string refusal;   ///< why not, in plain English; empty when accepted
};

/// How the scan interface is set up, for the service's lifetime.
struct scan_options
{
    std::filesystem::path scans_directory; ///< where each scan's directory goes; it exists
    frame_size stream_size;                ///< the size the source is opened for, in px
    std::string commit;                    ///< the tree's commit, for each scan's parameters
    /// Called on the scan thread after each state is published, with that state, outside every
    /// lock: so it sees every state in order. It must return soon, as the scan waits for it, and
    /// must not throw. It may give commands and read state(), but must not wait for a command's
    /// answer, which only the scan thread gives (a stop().get() there blocks the scan thread for
    /// good, and the destructor's join with it), and must not destroy the scan interface. Empty:
    /// no call.
    std::function<void(const scan_state &)> on_state;
    /// The stage clock that times each frame's stages (tracking/stage_clock.hpp), for the replay
    /// tool's measurement alone: it waits for the GPU after each stage that queues work, which a
    /// live scan must not do. nullptr, the default and a live scan's: nothing is timed and
    /// nothing more is waited for. Owned by the caller, which keeps it alive as long as the scan
    /// interface; the scan thread alone uses it, and calls its listener for each frame.
    stage_clock *stage_times = nullptr;
    /// The most points a live model holds: 200000 [?: initial] (the spec, "The web layer and the
    /// page"), > 0.
    std::uint32_t live_model_points = 200000;
    /// The least time from one live model to the next, by the times the scan thread decides to
    /// take them: 500 ms, 2 a second [?: initial] (the spec, "The per-frame path", step 8).
    std::chrono::milliseconds live_model_interval{500};
    /// Called on the scan thread after each live model is published, with it, outside every lock,
    /// as on_state is and under its rules: it must return soon, as the scan waits for it, must not
    /// throw, must not wait for a command's answer and must not destroy the scan interface. The
    /// value it is given stays as it is until the call returns. Empty: no call.
    std::function<void(const live_model &)> on_live_model = {};
};

/// The scan interface. One scan at a time; its thread lives as long as it does.
class scan_interface
{
public:
    /// Builds the scan interface, idle, and starts its scan thread, which creates its CUDA stream
    /// on the current device.
    /// @param options [in] the scans directory, the stream size, the commit and the listener;
    ///        copied.
    /// @param source [in,out] the frame source, closed; owned by the caller, which keeps it alive
    ///        and leaves it to the scan thread until the scan interface's end.
    /// @param writer [in,out] the sequence's writer, not open; owned and left to the scan thread
    ///        as the source.
    /// @note Allocates, the room of two live models for live_model_points points with their
    ///       slots, 32 bytes a point [S: computed], and starts a thread; a failure of either ends
    ///       the program (std::terminate).
    scan_interface(const scan_options &options, frame_source &source, sequence_sink &writer);

    /// The scan interface stands for one scan thread and its devices.
    scan_interface(const scan_interface &) = delete;
    scan_interface &operator=(const scan_interface &) = delete;

    /// Ends the scan interface as the service's stop ends it (the spec, "The scan interface"): a
    /// scan in progress ends without refinement, its source closed and its sequence and
    /// parameters kept, `end` shut_down; a refinement in progress stops between two frames and the
    /// scan stays unfinished; a command not yet answered is refused. Waits for the scan thread.
    ~scan_interface();

    /// Starts a scan of an object class with a set of parameters: the page's start gives the
    /// class's own set (scan/object_class.hpp, parameters_of()); a replay or a test gives a stored
    /// or a changed one.
    /// @param kind [in] the object class.
    /// @param set [in] the set the scan runs with; copied.
    /// @return the answer, once the start is decided and its state published: accepted when the
    ///         scan is scanning, its source open and its first frame taken; refused with the
    ///         reason when a scan is in progress, the scan interface is ending, or the start
    ///         failed: the GPU's buffers were refused (the set's values among the reasons), the
    ///         source refused to open or gave no first frame, its description cannot go into the
    ///         parameters (scan/parameters_file.hpp, source_refusal()), the clock lies outside 1970
    ///         to 9999, or the scan's directory or its sequence could not be created; nothing stays
    ///         on disk then, or the reason says what could not be removed.
    /// @note Allocates for the command. Any thread may call it but the listener, which must not
    ///       wait for an answer; it waits for nothing.
    std::future<command_reply> start(object_class kind, const object_class_parameters &set);

    /// Stops the scan: it is refined, after its source is closed.
    /// @return the answer: accepted while scanning, the scan stopped before its next frame;
    ///         refused with the reason when no scan is in progress or the scan is refining. A stop
    ///         given while the scan is starting waits until the start is decided: accepted after
    ///         the first frame when the scan started, refused as no scan in progress when the
    ///         start failed.
    /// @note As start().
    std::future<command_reply> stop();

    /// Discards the scan: it ends, and nothing of it stays on disk.
    /// @return the answer: accepted while scanning or refining, before the scan is idle again;
    ///         refused with the reason when no scan is in progress. A discard given while the scan
    ///         is starting waits as a stop does. When the scan's directory cannot be removed, the
    ///         scan's outcome is unfinished with the reason, not discarded.
    /// @note As start().
    std::future<command_reply> discard();

    /// Gives the state.
    /// @return a copy of the state last published, a snapshot.
    /// @note Takes the state's lock for the copy, which allocates for its strings; any thread may
    ///       call it, the listener too.
    scan_state state() const;

    /// Gives the live model.
    /// @return a copy of the live model last published, a snapshot: empty, without a frame,
    ///         before the first live model of the scan in progress, or before the first scan.
    /// @note Takes the live model's lock for the copy, which allocates for its points, up to
    ///       live_model_points of 28 bytes [S: sizeof(model_point)], while the scan thread waits
    ///       for the lock when it publishes the next live model; any thread may call it, the
    ///       listeners too.
    live_model latest_live_model() const;

private:
    /// What a command asks.
    enum class command_kind
    {
        start,
        stop,
        discard,
    };

    /// What a start asks for.
    struct start_request
    {
        gca::object_class object_class; ///< the object class
        object_class_parameters set;    ///< the set the scan runs with
    };

    /// A command in the queue, with the promise of its answer.
    struct command
    {
        command_kind kind;                  ///< what it asks
        std::optional<start_request> start; ///< what a start asks for; none for another command
        std::promise<command_reply> reply;  ///< its answer
    };

    /// What a command during a scan asked of it.
    struct scan_requests
    {
        bool stop = false;      ///< the operator's stop
        bool discard = false;   ///< a discard
        bool shut_down = false; ///< the scan interface's end
    };

    /// What a scan's starting and scanning gave, for its refinement.
    struct scanned
    {
        bool started = false; ///< whether the scan reached scanning; when not, the start was
                              ///< refused, answered and published, and nothing more is done
        std::filesystem::path directory;  ///< the scan's directory
        frame_description frames{};       ///< the description of its frames, from the first
        source_description source;        ///< what delivered them
        utc_time start{};                 ///< when the scan started
        utc_time stop{};                  ///< when it stopped
        scan_end end = scan_end::stopped; ///< why it ended
        std::string end_message;          ///< what ended it, in plain English
        std::string gpu_failure; ///< CUDA's error when the last frame's work could not be waited
                                 ///< for; empty when it was
    };

    /// Queues a command, or refuses it at once when the scan interface is ending.
    /// @param c [in] the command; taken over.
    /// @return its answer's future.
    std::future<command_reply> post(command c);

    /// The scan thread: waits for a command, runs a start's scan, refuses the rest, until the end;
    /// then refuses every command still queued.
    void run() noexcept;

    /// Runs one scan from its start to idle again: starting and scanning (run_scanning()), then
    /// the parameters, refining and the files.
    /// @param c [in,out] the start's command, answered here.
    /// @param stream [in] the scan thread's CUDA stream, which the scan's GPU work is queued on.
    void run_scan(command &c, cudaStream_t stream) noexcept;

    /// Starts and scans: the GPU's buffers and the live model, the source opened, the directory
    /// and the sequence made, then the scan's loop, until its end; the source and the writer
    /// closed after it. The tracker, which holds the live model's sparse TSDF, is this function's
    /// own and is freed when it returns, after the last frame's work is waited for, so that the
    /// live model can never outlive the loop into refinement (one sparse TSDF at a time:
    /// refinement/sequence_fusion.hpp, fuse_sequence()'s @pre).
    /// @param c [in,out] the start's command, answered here.
    /// @param request [in] what the start asks for.
    /// @param requests [in,out] what the commands during the scan asked.
    /// @param stream [in] the scan thread's CUDA stream.
    /// @return what the scan gave; not started when the start was refused.
    scanned run_scanning(command &c, const start_request &request, scan_requests &requests,
                         cudaStream_t stream) noexcept;

    /// Ends a scan: answers the commands queued meanwhile, removes the scan's directory when a
    /// discard asks for it, and publishes the state, idle, with the outcome.
    /// @param directory [in] the scan's directory.
    /// @param requests [in,out] what the commands during the scan asked.
    /// @param outcome [in] what became of the scan, unless a discard changes it.
    /// @param message [in] why it is unfinished; empty otherwise.
    void finish(const std::filesystem::path &directory, scan_requests &requests,
                scan_outcome outcome, std::string message) noexcept;

    /// Answers the commands queued meanwhile, during a scan, without allocating when none is
    /// queued.
    /// @param during [in] the phase, scanning or refining, which says what is refused.
    /// @param requests [in,out] what they asked of the scan.
    void answer_commands(scan_phase during, scan_requests &requests) noexcept;

    /// Publishes the scan thread's working state and calls the listener.
    void publish() noexcept;

    /// Publishes the working state's frame counts, tracking state and notice alone, which a
    /// frame changes, without allocating, and calls the listener.
    void publish_frame() noexcept;

    /// Publishes an empty live model, without a frame, and calls its listener: a scan's start.
    void clear_live_model() noexcept;

    /// Takes and publishes a live model after a fused frame, when the interval has passed since
    /// the last one published or none was (docs/methods/tracking.md, "The live model"): a live
    /// model taken is swapped in under the live model's lock, then its listener is called; one of
    /// too many points is not published, and the next fused frame tries again; a failure goes into
    /// the state's live_model_failure, published, and the interval passes before the next try.
    /// @param taker [in,out] the scan's taker.
    /// @param model [in] the live model's sparse TSDF.
    /// @param frame [in] the frame's index.
    /// @param pose [in] the pose tracking gave the frame.
    /// @param stream [in] the scan thread's CUDA stream.
    /// @note Allocates nothing but a failure's message; waits for the GPU as
    ///       live_model_taker::take() says.
    void take_live_model(live_model_taker &taker, const sparse_tsdf &model, std::uint64_t frame,
                         const camera_pose &pose, cudaStream_t stream) noexcept;

    /// Refuses a command: the reason recorded and published as the last refusal, then the
    /// answer, so that a caller with the answer reads the state that goes with it.
    /// @param c [in,out] the command, answered here.
    /// @param reason [in] the reason.
    void refuse(command &c, const std::string &reason) noexcept;

    scan_options m_options;  ///< the options, for the scan interface's lifetime
    frame_source &m_source;  ///< the source; the scan thread's alone
    sequence_sink &m_writer; ///< the writer; the scan thread's alone
    scan_state m_working;    ///< the state as the scan thread builds it; the scan thread's alone
    /// The commands answer_commands() took out of the queue, answered outside the lock; the scan
    /// thread's alone, kept so that its memory is reused.
    std::deque<command> m_asked;

    // The command lock guards the queue of commands and the flag of the end; the condition wakes
    // the scan thread when either changes. A command is put in and taken out under it, and
    // answered outside it.
    std::mutex m_command_mutex;
    std::condition_variable m_command_ready;
    std::deque<command> m_commands;
    bool m_ending = false;

    // The state lock guards the published state alone: the scan thread assigns it, a reader
    // copies it. Neither holds it across a GPU call, a write to disk or the listener.
    mutable std::mutex m_state_mutex;
    scan_state m_state;

    // The live model's lock guards the published live model alone: the scan thread swaps the
    // working one in and clears it under it, a reader copies its value. The scan thread reads it
    // without the lock, as its one writer. Its points and slots and the working one's are reserved
    // for the limit at the construction, so neither the swap nor a take allocates.
    mutable std::mutex m_live_mutex;
    held_live_model m_live;
    held_live_model m_live_working; ///< the next live model as the scan thread fills it; its alone
    /// When the scan thread decided to take the last live model published or to try one that
    /// failed; none at a scan's start. The scan thread's alone.
    std::optional<std::chrono::steady_clock::time_point> m_live_last;

    std::thread m_thread; ///< the scan thread, started last
};
} // namespace gca
