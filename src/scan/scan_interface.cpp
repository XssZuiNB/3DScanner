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
#include "separation/gravity.hpp"
#include "types/gravity_estimate.hpp"
#include "types/imu_sample.hpp"

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

std::string_view name_of(scan_phase phase) noexcept
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
    return {};
}

std::string_view name_of(scan_outcome outcome) noexcept
{
    switch (outcome)
    {
    case scan_outcome::none:
        return "none";
    case scan_outcome::refined:
        return "refined";
    case scan_outcome::unfinished:
        return "unfinished";
    case scan_outcome::discarded:
        return "discarded";
    }
    return {};
}

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
    , m_views(options.stream_size, options.on_live_view)
{
    for (held_live_model *held : {&m_live, &m_live_working})
    {
        held->value.points.reserve(m_options.live_model_points);
        held->slots.reserve(m_options.live_model_points);
    }
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

live_model scan_interface::latest_live_model() const
{
    const std::lock_guard<std::mutex> lock(m_live_mutex);
    return m_live.value;
}

live_view scan_interface::latest_live_view() const
{
    return m_views.latest();
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
            refuse(c, "a scan is in progress (" + std::string(name_of(during)) + ")");
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

void scan_interface::clear_live_model() noexcept
{
    {
        const std::lock_guard<std::mutex> lock(m_live_mutex);
        live_model &value = m_live.value;
        value.version++;
        value.frame.reset();
        value.pose = {Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity()};
        value.taken = {};
        value.spacing = every_crossing;
        value.points.clear();
        m_live.slots.clear();
    }
    m_live_last.reset();
    if (m_options.on_live_model)
        m_options.on_live_model(m_live.value);
}

bool scan_interface::note_property_refusals() noexcept
{
    if (m_options.color_report == nullptr || !m_refusals_noted)
        return false;
    const std::span<const color_refusal> taken =
        m_refusals_noted->take_refusals(*m_options.color_report);
    for (const color_refusal &refusal : taken)
    {
        if (!m_working.property_refusals.empty())
            m_working.property_refusals += "; ";
        m_working.property_refusals += refusal_line(refusal);
        for (std::size_t i = 0; i < color_property_count; i++)
            m_refused[i] = m_refused[i] || color_property_table[i].property == refusal.property;
    }
    return !taken.empty();
}

void scan_interface::take_live_model(live_model_taker &taker, const sparse_tsdf &model,
                                     std::uint64_t frame, const camera_pose &pose,
                                     cudaStream_t stream) noexcept
{
    // The pacing runs on the times of the decisions, so that two live models published lie at
    // least the interval apart by the times they carry.
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    if (m_live_last && now - *m_live_last < m_options.live_model_interval)
        return;
    const live_take took = taker.take(model, m_live, m_live_working, stream);
    if (took.status == live_take_status::too_many)
        return;
    m_live_last = now;
    if (took.status == live_take_status::failed)
    {
        m_working.live_model_failure = took.message;
        publish();
        return;
    }
    live_model &next = m_live_working.value;
    next.version = m_live.value.version + 1;
    next.frame = frame;
    next.pose = pose;
    next.taken = now;
    next.spacing = took.spacing;
    {
        // The points and their slots move together.
        const std::lock_guard<std::mutex> lock(m_live_mutex);
        std::swap(m_live, m_live_working);
    }
    if (m_options.on_live_model)
        m_options.on_live_model(m_live.value);
}

void scan_interface::offer_live_view(live_view_encoder &encoder, const model_surface &surface,
                                     std::uint64_t frame, std::uint64_t timestamp_us,
                                     cudaStream_t stream) noexcept
{
    const live_view_offer offered = encoder.offer(surface, frame, timestamp_us, stream);
    if (offered.status != live_view_offer_status::failed)
        return;
    m_working.live_view_failure = offered.message;
    publish();
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
        // The hold's refusals of a source that opened and was closed again; none before.
        note_property_refusals();
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
    request_result<live_model_taker> requested_taker = live_model_taker::request(
        pool, m_options.live_model_points, requested.granted.live_model().layout());
    if (requested_taker.status != request_status::granted)
        return start_failed("the GPU's buffers: the live model's: " + requested_taker.message);
    // The encoder's views read the tracker's surface on the GPU, on the scan's stream, whose work
    // is waited for at the loop's end, before the tracker is freed.
    request_result<live_view_encoder> requested_encoder =
        live_view_encoder::request(pool, m_options.stream_size, m_views);
    if (requested_encoder.status != request_status::granted)
        return start_failed("the GPU's buffers: the live view's: " + requested_encoder.message);
    pool.close();
    tracker live = std::move(requested.granted);
    live_model_taker taker = std::move(requested_taker.granted);
    live_view_encoder encoder = std::move(requested_encoder.granted);

    const open_result opened = m_source.open(m_options.stream_size);
    if (!opened.opened)
        return start_failed("the frame source refused to open: " + opened.refusal);
    // The hold's report is this opening's from here on (camera/color_hold.hpp, report()).
    m_refusals_noted.emplace();
    note_property_refusals();
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
    note_property_refusals();
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
    // The last fused frame, whose view the encoder makes at the end when it skipped it.
    std::optional<std::uint64_t> last_fused;
    std::uint64_t last_fused_timestamp_us = 0;
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
        // written (camera/sequence_writer.hpp, has_room()). The two exceptions end the scan: a
        // frame fused and then refused by a failing writer (docs/SCAN_FILES.md, "Parameters"), and
        // a frame whose ray cast after its fusion failed, which tracking reports as failed: it is
        // in the live model, not in the sequence (docs/methods/tracking.md, "The scan's loop").
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
            // The push swaps the frame's buffers for a free slot's (camera/sequence_writer.hpp), so
            // the frame's index and timestamp are kept before it.
            const std::uint64_t index = current.index();
            const std::uint64_t timestamp_us = current.timestamp_us();
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
            // The live view, then the live model, after the frame's fusion and its push, never on
            // a lost frame (docs/methods/tracking.md, "The scan's loop", step 6).
            if (fused)
            {
                last_fused = index;
                last_fused_timestamp_us = timestamp_us;
                offer_live_view(encoder, live.surface(), index, timestamp_us, stream);
            }
            if (clock != nullptr)
            {
                const step_result waited = clock->mark_after(path_stage::live_view, stream);
                if (!waited.done)
                {
                    m_working.live_view_failure = waited.message;
                    publish();
                }
            }
            if (fused)
                take_live_model(taker, live.live_model(), index, *tracked.pose, stream);
            if (clock != nullptr)
                clock->mark(path_stage::live_model);
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
        if (note_property_refusals())
            publish();
        if (next.status != next_status::frame)
        {
            made.end = end_of(next.status);
            made.end_message = next.message;
            break;
        }
    }
    made.stop = now_utc();
    m_source.close();
    note_property_refusals();
    // The newest view shows the last fused frame (docs/methods/tracking.md, "The live view"); not
    // after a frame tracking failed on, which may leave the surface unspecified.
    if (last_fused && made.end != scan_end::tracking_failed)
    {
        const step_result finished =
            encoder.finish(live.surface(), *last_fused, last_fused_timestamp_us, stream);
        if (!finished.done)
            m_working.live_view_failure = finished.message;
    }
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
    m_working.live_model_failure.clear();
    m_working.live_view_failure.clear();
    m_working.property_refusals.clear();
    m_refusals_noted.reset();
    m_refused = {};
    clear_live_model();
    m_views.clear();
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

    // The parameters at the stop, so that a scan left unfinished keeps them; gravity is estimated
    // once refinement has read the sequence.
    scan_parameters parameters = {
        m_options.commit,  request.object_class, request.set, scan.frames,      scan.source,
        m_refused,         scan.start,           scan.stop,   m_working.frames, scan.end,
        gravity_estimate{}};
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
    // Gravity's up from the samples and the trajectory the fusion read, kept in the parameters
    // before the sequence can be removed (docs/methods/separation.md, "What it reads from a scan").
    parameters.gravity =
        estimate_gravity(refined.trajectory, refined.accelerometer, imu_to_camera()).estimate;
    const write_result estimated = write_parameters(scan.directory / parameters_name, parameters);
    if (!estimated.written)
        return finish(scan.directory, requests, scan_outcome::unfinished,
                      "the parameters with gravity: " + estimated.reason);
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
    // Without the switch only a refined scan's sequence goes (R18): the full model and trajectory
    // stand, and refinement does not read it again. A sequence that cannot be removed leaves the
    // scan refined, with the reason.
    std::string sequence_left;
    if (!m_options.keep_sequence)
    {
        std::error_code error;
        std::filesystem::remove(scan.directory / sequence_file_name, error);
        if (error)
            sequence_left = "its sequence could not be removed: " + error.message();
    }
    finish(scan.directory, requests, scan_outcome::refined, std::move(sequence_left));
}
} // namespace gca
