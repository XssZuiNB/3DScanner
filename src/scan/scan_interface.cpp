// SPDX-License-Identifier: 0BSD
/// @file scan_interface.cpp
/// The scan interface: its scan thread, the queue of commands, the published state, a scan's
/// phases and the "too close" rule (see scan_interface.hpp; the spec, "The scan interface";
/// docs/methods/tracking.md, "The scan's loop").

#include "scan/scan_interface.hpp"

#include "camera/sequence_format.hpp"
#include "fusion/sparse_tsdf.hpp"
#include "gpu/gpu_messages.hpp"
#include "gpu/gpu_pool.hpp"
#include "refinement/sequence_fusion.hpp"
#include "scan/model_file.hpp"
#include "scan/stage_parameters.hpp"
#include "scan/trajectory_file.hpp"
#include "scan/utc_time.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <span>
#include <system_error>
#include <utility>
#include <vector>

namespace gca
{
namespace
{
/// The files of a scan's directory (docs/SCAN_FILES.md, "A scan's directory").
constexpr const char *parameters_name = "parameters.json";
constexpr const char *trajectory_name = "trajectory.txt";
constexpr const char *full_model_name = "full.ply";

/// Gives the time now, in UTC.
/// @return the system's clock, to the microsecond.
utc_time now_utc() noexcept
{
    return std::chrono::floor<std::chrono::microseconds>(std::chrono::system_clock::now());
}

/// Names a phase for a refusal.
/// @param phase [in] the phase.
/// @return its name.
const char *phase_name(scan_phase phase) noexcept
{
    switch (phase)
    {
    case scan_phase::idle:
        return "idle";
    case scan_phase::starting:
        return "starting";
    case scan_phase::scanning:
        return "scanning";
    case scan_phase::refining:
        return "refining";
    }
    return "unknown";
}

/// Gives why a scan ended when its source gave no frame.
/// @param status [in] the source's status, not a frame.
/// @return source_timeout, source_failure or source_end.
scan_end end_of(next_status status) noexcept
{
    switch (status)
    {
    case next_status::timeout:
        return scan_end::source_timeout;
    case next_status::end:
        return scan_end::source_end;
    case next_status::frame:
    case next_status::failure:
        break;
    }
    return scan_end::source_failure;
}
} // namespace

bool too_close(const frame &frame, double threshold_m) noexcept
{
    // A depth value v lies nearer than the threshold exactly when v < first_far, the least depth
    // value at or beyond it, ceil(threshold / unit) [S: computed, v is a whole number], from 1 to
    // 65536, the latter when every depth value lies nearer.
    const double limit = std::ceil(threshold_m / double{frame.description().depth_unit_m});
    const std::uint32_t first_far = static_cast<std::uint32_t>(std::clamp(limit, 1.0, 65536.0));
    // v - 1 < first_far - 1 in 16 bits: v - 1 wraps around to 65535 for a pixel without depth,
    // which is then not nearer. Counted in 32 bits a row and in 64 across the rows, which the
    // compiler vectorizes: 0.335 ms at 1280x800 against 0.704 ms with 64-bit counts and a 32-bit
    // compare [M 2026-11-16: docs/HARDWARE.md, "The per-frame path's stages"].
    const std::uint16_t nearer_limit = static_cast<std::uint16_t>(first_far - 1u);
    const frame_size size = frame.description().size;
    const std::span<const std::uint16_t> depth = frame.depth();
    std::uint64_t with_depth = 0;
    std::uint64_t nearer = 0;
    for (std::uint32_t v = 0; v < size.height; v++)
    {
        const std::uint16_t *row = depth.data() + static_cast<std::size_t>(v) * size.width;
        std::uint32_t row_with_depth = 0;
        std::uint32_t row_nearer = 0;
        for (std::uint32_t u = 0; u < size.width; u++)
        {
            row_with_depth += row[u] != 0;
            row_nearer += static_cast<std::uint16_t>(row[u] - 1u) < nearer_limit;
        }
        with_depth += row_with_depth;
        nearer += row_nearer;
    }
    return with_depth > 0 && 4 * nearer >= with_depth;
}

scan_interface::scan_interface(const scan_options &options, frame_source &source,
                               sequence_sink &writer)
    : m_options(options)
    , m_source(source)
    , m_writer(writer)
{
    m_thread = std::thread(&scan_interface::run, this);
}

scan_interface::~scan_interface()
{
    {
        const std::lock_guard<std::mutex> lock(m_command_mutex);
        m_ending = true;
    }
    m_command_ready.notify_all();
    m_thread.join();
}

std::future<command_reply> scan_interface::start(object_class kind,
                                                 const object_class_parameters &set)
{
    return post({command_kind::start, start_request{kind, set}, {}});
}

std::future<command_reply> scan_interface::stop()
{
    return post({command_kind::stop, std::nullopt, {}});
}

std::future<command_reply> scan_interface::discard()
{
    return post({command_kind::discard, std::nullopt, {}});
}

scan_state scan_interface::state() const
{
    const std::lock_guard<std::mutex> lock(m_state_mutex);
    return m_state;
}

std::future<command_reply> scan_interface::post(command c)
{
    std::future<command_reply> reply = c.reply.get_future();
    {
        const std::lock_guard<std::mutex> lock(m_command_mutex);
        if (!m_ending)
        {
            m_commands.push_back(std::move(c));
            m_command_ready.notify_one();
            return reply;
        }
    }
    c.reply.set_value({false, "the scan interface is ending"});
    return reply;
}

void scan_interface::run() noexcept
{
    cudaStream_t stream = nullptr;
    const cudaError_t created = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    const std::string no_stream =
        created == cudaSuccess ? std::string() : cuda_failure("the scan's CUDA stream", created);
    while (true)
    {
        std::unique_lock<std::mutex> lock(m_command_mutex);
        m_command_ready.wait(lock, [this] { return m_ending || !m_commands.empty(); });
        if (m_ending)
            break;
        command c = std::move(m_commands.front());
        m_commands.pop_front();
        lock.unlock();
        if (c.kind != command_kind::start)
            refuse(c, "no scan is in progress");
        else if (!no_stream.empty())
            refuse(c, no_stream);
        else
            run_scan(c, stream);
    }
    // The end: every command still queued is refused; post() refuses the later ones itself.
    std::deque<command> left;
    {
        const std::lock_guard<std::mutex> lock(m_command_mutex);
        left.swap(m_commands);
    }
    for (command &c : left)
        c.reply.set_value({false, "the scan interface is ending"});
    if (stream != nullptr)
        cudaStreamDestroy(stream);
}

void scan_interface::answer_commands(scan_phase during, scan_requests &requests) noexcept
{
    {
        const std::lock_guard<std::mutex> lock(m_command_mutex);
        requests.shut_down = requests.shut_down || m_ending;
        // Nothing is built or allocated for a frame without commands.
        if (m_commands.empty())
            return;
        m_asked.swap(m_commands);
    }
    while (!m_asked.empty())
    {
        command c = std::move(m_asked.front());
        m_asked.pop_front();
        if (c.kind == command_kind::start)
            refuse(c, std::string("a scan is in progress (") + phase_name(during) + ")");
        else if (c.kind == command_kind::stop && during == scan_phase::refining)
            refuse(c, "the scan is refining: a stop is for a scan that scans");
        else
        {
            requests.stop = requests.stop || c.kind == command_kind::stop;
            requests.discard = requests.discard || c.kind == command_kind::discard;
            c.reply.set_value({true, {}});
        }
    }
}

void scan_interface::refuse(command &c, const std::string &reason) noexcept
{
    m_working.refusal = reason;
    publish();
    c.reply.set_value({false, reason});
}

void scan_interface::publish() noexcept
{
    m_working.version++;
    {
        const std::lock_guard<std::mutex> lock(m_state_mutex);
        m_state = m_working;
    }
    if (m_options.on_state)
        m_options.on_state(m_working);
}

void scan_interface::publish_frame() noexcept
{
    m_working.version++;
    {
        // A frame changes these alone; the strings stay, so nothing is allocated.
        const std::lock_guard<std::mutex> lock(m_state_mutex);
        m_state.version = m_working.version;
        m_state.tracking = m_working.tracking;
        m_state.lost_cause = m_working.lost_cause;
        m_state.too_close = m_working.too_close;
        m_state.frames = m_working.frames;
    }
    if (m_options.on_state)
        m_options.on_state(m_working);
}

void scan_interface::finish(const std::filesystem::path &directory, scan_requests &requests,
                            scan_outcome outcome, std::string message) noexcept
{
    // The commands queued while the scan ran its last steps are the scan's to answer: a discard
    // that came during the extraction or the writes still discards it.
    answer_commands(scan_phase::refining, requests);
    if (requests.discard && outcome != scan_outcome::discarded)
    {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
        outcome = error ? scan_outcome::unfinished : scan_outcome::discarded;
        message =
            error ? "the discard could not remove " + directory.string() + ": " + error.message()
                  : std::string();
        if (!error)
        {
            m_working.end.reset();
            m_working.end_message.clear();
        }
    }
    m_working.phase = scan_phase::idle;
    m_working.outcome = outcome;
    m_working.outcome_message = std::move(message);
    publish();
}

scan_interface::scanned scan_interface::run_scanning(command &c, const start_request &request,
                                                     scan_requests &requests,
                                                     cudaStream_t stream) noexcept
{
    const object_class_parameters &set = request.set;
    scanned made;
    made.start = now_utc();
    // A start that fails leaves nothing on disk and the source closed; its state is published
    // first, then the answer, so that a caller with the answer reads the state that goes with it.
    // The live model's clearing, queued by its request, is waited for before the tracker is
    // freed with this function's return.
    const auto start_failed = [this, &c, &made, stream](const std::string &reason) {
        const cudaError_t waited = cudaStreamSynchronize(stream);
        m_working.phase = scan_phase::idle;
        refuse(c, waited == cudaSuccess
                      ? reason
                      : reason + "; " + cuda_failure("the GPU's work of the start", waited));
        return made;
    };

    // The per-frame path's buffers and the live model, before the first frame; the pool is closed
    // then, so that a stage that asks for memory during the scan fails where it is seen.
    gpu_pool pool;
    request_result<tracker> requested =
        tracker::request(pool, m_options.stream_size, tracking_parameters_of(set),
                         sparse_tsdf_parameters_of(set, fused_model::live), stream);
    if (requested.status != request_status::granted)
        return start_failed("the GPU's buffers: " + requested.message);
    pool.close();
    tracker live = std::move(requested.granted);

    const open_result opened = m_source.open(m_options.stream_size);
    if (!opened.opened)
        return start_failed("the frame source refused to open: " + opened.refusal);
    made.source = m_source.describe();
    const std::string not_held = source_refusal(made.source);
    if (!not_held.empty())
    {
        m_source.close();
        return start_failed("the frame source's description cannot go into the scan's "
                            "parameters: " +
                            not_held);
    }
    frame current;
    const next_result first = m_source.next(current);
    if (first.status != next_status::frame)
    {
        m_source.close();
        return start_failed("the frame source gave no first frame: " + first.message);
    }
    made.frames = current.description();
    const std::optional<std::string> name = utc_text(made.start);
    if (!name)
    {
        m_source.close();
        return start_failed("the clock lies outside 1970 to 9999, so the scan has no name");
    }
    made.directory = m_options.scans_directory / *name;
    std::error_code error;
    if (!std::filesystem::create_directory(made.directory, error))
    {
        m_source.close();
        return start_failed(
            "the scan's directory " + made.directory.string() +
            " could not be made: " + (error ? error.message() : std::string("it exists")));
    }
    const open_result writing =
        m_writer.open(made.directory / sequence_file_name, made.frames, sequence_queue_frames);
    if (!writing.opened)
    {
        m_source.close();
        std::filesystem::remove_all(made.directory, error);
        return start_failed("the scan's sequence: " + writing.refusal +
                            (error ? "; and its directory could not be removed: " + error.message()
                                   : std::string()));
    }
    m_working.scan = *name;
    m_working.phase = scan_phase::scanning;
    publish();
    c.reply.set_value({true, {}});
    made.started = true;

    // The scan's loop (docs/methods/tracking.md, "The scan's loop"). The source gives its frames
    // in order; the camera's gives the oldest of OrbbecSDK's queue, which drops frames when the
    // loop falls behind, so a frame the source did not deliver shows as a gap in the indices and
    // counts as not taken, as does a frame the writer had no room for (the newest frame is not
    // taken by a mailbox: the scan-interface-first-whole-scan plan check, 2026-11-16).
    // A replay's measurement times each frame's stages (docs/methods/tracking.md, "The stages'
    // times"); a live scan has no clock, and each use below is skipped.
    stage_clock *const clock = m_options.stage_times;
    std::optional<std::uint64_t> last_index;
    while (true)
    {
        if (last_index && current.index() > *last_index + 1)
            m_working.frames.not_taken += current.index() - *last_index - 1;
        last_index = current.index();
        if (clock != nullptr)
            clock->begin(current.index());
        // What tracking made of the frame; none while it is not taken.
        std::optional<tracking_state> fate;
        // Room first, then the stages, then the push, so that no frame is fused that is not
        // written (camera/sequence_writer.hpp, has_room()); a frame fused and then refused by a
        // failing writer is the one exception (docs/SCAN_FILES.md, "Parameters").
        const bool room = m_writer.has_room();
        if (clock != nullptr)
            clock->mark(path_stage::room);
        if (!room)
            m_working.frames.not_taken++;
        else
        {
            m_working.too_close = too_close(current, set.too_close_depth_m);
            if (clock != nullptr)
                clock->mark(path_stage::too_close);
            const tracking_result tracked = live.track(current, stream, clock);
            m_working.frames.taken++;
            fate = tracked.state;
            if (tracked.state == tracking_state::failed)
            {
                // Taken, neither fused nor lost, and written nowhere: the scan ends at it.
                made.end = scan_end::tracking_failed;
                made.end_message = tracked.message;
                publish_frame();
                if (clock != nullptr)
                    clock->end(fate);
                break;
            }
            const bool fused = tracked.state == tracking_state::tracking;
            m_working.frames.fused += fused;
            m_working.frames.lost += !fused;
            m_working.tracking = tracked.state;
            m_working.lost_cause = tracked.outcome;
            const push_result pushed = m_writer.push(
                current, fused ? frame_tracking{*tracked.pose} : frame_tracking{tracking_loss{}});
            if (clock != nullptr)
                clock->mark(path_stage::writer);
            if (pushed.status != push_status::accepted)
            {
                made.end = scan_end::writer_failed;
                made.end_message = pushed.reason;
                publish_frame();
                if (clock != nullptr)
                    clock->end(fate);
                break;
            }
        }
        publish_frame();
        // The commands between two frames: a stop or a discard ends the scan before the next.
        answer_commands(scan_phase::scanning, requests);
        if (clock != nullptr)
            clock->end(fate);
        if (requests.discard || requests.shut_down || requests.stop)
        {
            made.end = requests.shut_down ? scan_end::shut_down : scan_end::stopped;
            break;
        }
        const next_result next = m_source.next(current);
        if (next.status != next_status::frame)
        {
            made.end = end_of(next.status);
            made.end_message = next.message;
            break;
        }
    }
    made.stop = now_utc();
    m_source.close();
    const write_result closed = m_writer.close();
    if (!closed.written && made.end != scan_end::writer_failed)
    {
        made.end = scan_end::writer_failed;
        made.end_message = closed.reason;
    }
    // The last frame's fusion is waited for, as the tracker's freeing asks; the tracker, and the
    // live model's sparse TSDF with it, is freed as this function returns.
    const cudaError_t waited = cudaStreamSynchronize(stream);
    if (waited != cudaSuccess)
        made.gpu_failure = cuda_failure("the last frame's work on the GPU", waited);
    return made;
}

void scan_interface::run_scan(command &c, cudaStream_t stream) noexcept
{
    const start_request request = *c.start;
    m_working.phase = scan_phase::starting;
    m_working.scan.clear();
    m_working.tracking.reset();
    m_working.lost_cause = check_outcome::passed;
    m_working.too_close = false;
    m_working.frames = {};
    m_working.end.reset();
    m_working.end_message.clear();
    m_working.outcome = scan_outcome::none;
    m_working.outcome_message.clear();
    publish();

    scan_requests requests;
    // The live model is freed when run_scanning() returns: refinement below finds it gone.
    const scanned scan = run_scanning(c, request, requests, stream);
    if (!scan.started)
        return;
    if (requests.discard)
        return finish(scan.directory, requests, scan_outcome::unfinished,
                      "a discard while scanning");
    m_working.end = scan.end;
    m_working.end_message = scan.end_message;

    // The parameters at the stop, so that a scan left unfinished keeps them.
    const scan_parameters parameters = {m_options.commit, request.object_class, request.set,
                                        scan.frames,      scan.source,          scan.start,
                                        scan.stop,        m_working.frames,     scan.end};
    const write_result kept = write_parameters(scan.directory / parameters_name, parameters);
    if (!kept.written)
        return finish(scan.directory, requests, scan_outcome::unfinished,
                      "the parameters: " + kept.reason);
    if (requests.shut_down)
        return finish(scan.directory, requests, scan_outcome::unfinished,
                      "the scan interface ended during the scan");
    if (!scan.gpu_failure.empty())
        return finish(scan.directory, requests, scan_outcome::unfinished, scan.gpu_failure);

    m_working.phase = scan_phase::refining;
    publish();
    const object_class_parameters &set = request.set;
    const sequence_fusion_result refined = fuse_sequence(
        scan.directory / sequence_file_name, m_options.stream_size,
        sparse_tsdf_parameters_of(set, fused_model::full),
        tracking_parameters_of(set).pyramid.range,
        [this, &requests] {
            answer_commands(scan_phase::refining, requests);
            return !requests.discard && !requests.shut_down;
        },
        stream);
    // A discard that came during the extraction, the copy or the writes is answered by finish(),
    // which removes the scan's directory with every file written by then.
    if (refined.end == sequence_fusion_end::cancelled)
        return finish(scan.directory, requests, scan_outcome::unfinished,
                      requests.discard ? "a discard while refining"
                                       : "the scan interface ended during refinement");
    if (refined.end == sequence_fusion_end::failed)
        return finish(scan.directory, requests, scan_outcome::unfinished,
                      "refinement: " + refined.message);
    const write_result trajectory =
        write_trajectory(scan.directory / trajectory_name, refined.trajectory);
    if (!trajectory.written)
        return finish(scan.directory, requests, scan_outcome::unfinished,
                      "the trajectory: " + trajectory.reason);
    const write_result model = write_model(scan.directory / full_model_name, refined.model);
    if (!model.written)
    {
        std::error_code error;
        std::filesystem::remove(scan.directory / trajectory_name, error);
        return finish(scan.directory, requests, scan_outcome::unfinished,
                      "the full model: " + model.reason +
                          (error ? "; and the trajectory could not be removed: " + error.message()
                                 : std::string()));
    }
    finish(scan.directory, requests, scan_outcome::refined, {});
}
} // namespace gca
