// SPDX-License-Identifier: 0BSD
/// @file service.cpp
/// The scanner service's parts: the stop signals, the log with its lines, the hold's lines in the
/// log, and the composition that serves until a stop signal (see service.hpp).

#include "service.hpp"

#include "camera/color_hold.hpp"
#include "camera/frame.hpp"
#include "camera/sequence_writer.hpp"
#include "scan/live_model.hpp"
#include "scan/parameters_file.hpp"
#include "stop_signals.hpp"
#include "web/web_server.hpp"

#include <atomic>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <utility>

#include <signal.h>

namespace gca
{
namespace
{
/// The stop signal the handler recorded, 0 for none. A signal may reach a thread started before
/// main(), whose mask does not block it; a lock-free atomic is safe both in a signal handler and
/// across threads [S: C++20, [support.signal]].
std::atomic<int> g_stop_signal{0};
static_assert(std::atomic<int>::is_always_lock_free,
              "a signal handler may only touch a lock-free atomic");

/// Records a stop signal that reached a thread which does not block it.
/// @param signal_number [in] the signal.
void on_stop_signal(int signal_number)
{
    g_stop_signal.store(signal_number);
}

/// Gives the set of the stop signals.
/// @return the set.
sigset_t stop_signal_set() noexcept
{
    sigset_t set;
    sigemptyset(&set);
    for (const stop_signal &stop : stop_signals)
        sigaddset(&set, stop.number);
    return set;
}

/// Gives the text after "frames" in a scan's lines.
/// @param frames [in] the scan's frames.
/// @return "frames taken <T> fused <F> lost <L> not taken <N>".
std::string frames_text(const scan_frame_counts &frames)
{
    return "frames taken " + std::to_string(frames.taken) + " fused " +
           std::to_string(frames.fused) + " lost " + std::to_string(frames.lost) + " not taken " +
           std::to_string(frames.not_taken);
}

/// Gives why a scan ended and its frames, for a scan's lines.
/// @param state [in] the state.
/// @return "end <end>, " or "end <end> (<message>), ", nothing for a scan without an end (a
///         discarded one), then frames_text().
std::string ended_text(const scan_state &state)
{
    std::string text;
    if (state.end)
    {
        text = "end " + std::string(name_of(*state.end));
        if (!state.end_message.empty())
            text += " (" + state.end_message + ")";
        text += ", ";
    }
    return text + frames_text(state.frames);
}

/// The hold in front of the camera, with its lines in the service's log: after each of its
/// calls, the readings and the refusals it added (camera/color_hold.hpp, reading_line(),
/// refusal_line()). The scan thread alone calls it, as the frame source seam says.
class logged_hold final : public frame_source
{
public:
    /// Puts the log behind the hold.
    /// @param hold [in,out] the hold; it outlives this.
    /// @param log [in,out] the log; it outlives this.
    logged_hold(color_hold &hold, service_log &log) noexcept
        : m_hold(hold)
        , m_log(log)
    {
    }

    /// Opens the hold and logs what it added: a new report when the camera opened.
    /// @param size [in] the stream size, in px.
    /// @return the hold's result.
    open_result open(const frame_size &size) noexcept override
    {
        const open_result opened = m_hold.open(size);
        if (opened.opened)
            m_logged.restart();
        log_new();
        return opened;
    }

    /// Takes the hold's next frame and logs what it added, at the hold after the settling.
    /// @param out [out] as the hold's next() fills it.
    /// @return the hold's result.
    next_result next(frame &out) noexcept override
    {
        const next_result result = m_hold.next(out);
        log_new();
        return result;
    }

    /// Closes the hold, which switches the controls on again, and logs what it added.
    void close() noexcept override
    {
        m_hold.close();
        log_new();
    }

    /// Describes the camera.
    /// @return the hold's description.
    source_description describe() const noexcept override
    {
        return m_hold.describe();
    }

private:
    /// Logs the readings and the refusals of the hold's report not logged yet, readings first;
    /// allocates nothing when there is none.
    void log_new() noexcept
    {
        const color_hold_report &report = m_hold.report();
        for (const color_reading &reading : m_logged.take_readings(report))
            m_log.line(reading_line(reading));
        for (const color_refusal &refusal : m_logged.take_refusals(report))
            m_log.line(refusal_line(refusal));
    }

    color_hold &m_hold;
    service_log &m_log;
    color_report_cursor m_logged; ///< the report's entries logged since the opening
};

/// The service's parts, built in the order of their members and ended in the reverse: the web
/// layer first, which waits for its requests, some of them for a command's answer, so the scan
/// interface outlives it; then the scan interface, which ends a scan in progress through the
/// hold and the writer, so both outlive it; the news and the log outlive the scan interface's
/// listeners.
class scanner_service
{
public:
    /// Builds the parts; the scan interface's thread starts, the web layer does not serve yet.
    /// @param settings [in] what the service runs with.
    /// @param camera [in,out] the camera's frames; it outlives this.
    /// @param properties [in,out] the camera's properties; it outlives this.
    /// @param log [in,out] the log; it outlives this.
    scanner_service(const service_settings &settings, frame_source &camera,
                    camera_properties &properties, service_log &log)
        : m_hold(camera, properties)
        , m_logged(m_hold, log)
        , m_writer(queue_full::report)
        , m_scanner({.scans_directory = settings.options.scans,
                     .stream_size = settings.stream_size,
                     .commit = settings.commit,
                     .on_state =
                         [this, &log](const scan_state &state) {
                             m_news.state_published();
                             log.on_state(state);
                         },
                     .on_live_model = [this](const live_model &) { m_news.live_model_published(); },
                     .color_report = &m_hold.report(),
                     .keep_sequence = settings.options.keep_sequence},
                    m_logged, m_writer)
        , m_web(m_scanner, m_news,
                {.address = settings.options.address,
                 .port = settings.options.port,
                 .page_directory = settings.options.page,
                 .set_of = settings.set_of})
    {
    }

    /// Starts the web layer.
    /// @return the web layer's result (web/web_server.hpp, start()).
    web_start_result start() noexcept
    {
        return m_web.start();
    }

private:
    color_hold m_hold;        ///< the hold in front of the camera
    logged_hold m_logged;     ///< the hold with its lines in the log: the scan's source
    sequence_writer m_writer; ///< the sequence's writer, which says so when its queue is full
    scan_news m_news;         ///< the counts of what the scan interface published
    scan_interface m_scanner; ///< the scan interface
    web_server m_web;         ///< the web layer, ended first
};

/// Gives the line that says the service serves.
/// @param options [in] the command line's options.
/// @param port [in] the port bound.
/// @return "serving the page on http://<address>:<port>/ from <page>; scans in <scans>,
///         sequences kept" or "..., a refined scan's sequence removed".
std::string serving_text(const service_options &options, std::uint16_t port)
{
    return "serving the page on http://" + options.address + ":" + std::to_string(port) +
           "/ from " + options.page.string() + "; scans in " + options.scans.string() + ", " +
           (options.keep_sequence ? "sequences kept" : "a refined scan's sequence removed");
}

/// Builds the parts, starts the web layer, waits for a stop signal and ends the parts.
/// @param settings [in] what the service runs with.
/// @param camera [in,out] the camera's frames.
/// @param properties [in,out] the camera's properties.
/// @param log [in,out] the log.
/// @return 0 after a stop signal; 1 when the web layer did not start.
int serve_until_stopped(const service_settings &settings, frame_source &camera,
                        camera_properties &properties, service_log &log)
{
    scanner_service service(settings, camera, properties, log);
    const web_start_result started = service.start();
    if (!started.started)
    {
        log.line("the web layer did not start: " + started.reason);
        return 1;
    }
    log.line(serving_text(settings.options, started.port));
    const stop_signal &stop = wait_for_stop_signal();
    log.line(std::string("stopping on ") + stop.name);
    return 0;
}
} // namespace

void install_stop_signals() noexcept
{
    g_stop_signal.store(0);
    struct sigaction action
    {
    };
    action.sa_handler = on_stop_signal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;
    for (const stop_signal &stop : stop_signals)
        sigaction(stop.number, &action, nullptr);
    const sigset_t set = stop_signal_set();
    pthread_sigmask(SIG_BLOCK, &set, nullptr);
    std::signal(SIGPIPE, SIG_IGN);
}

const stop_signal &wait_for_stop_signal() noexcept
{
    const sigset_t set = stop_signal_set();
    const timespec poll = {0, 100'000'000};
    for (;;)
    {
        // A blocked signal is taken here; a timeout or an interruption gives -1, and then the one
        // the handler recorded, 0 for none, is looked for: no row matches 0, and the wait goes on.
        int taken = sigtimedwait(&set, nullptr, &poll);
        if (taken <= 0)
            taken = g_stop_signal.exchange(0);
        for (const stop_signal &stop : stop_signals)
        {
            if (stop.number == taken)
                return stop;
        }
    }
}

service_log::service_log(std::function<void(const std::string &)> sink)
    : m_sink(std::move(sink))
{
}

void service_log::line(std::string_view text)
{
    const std::string whole = std::string(service_line_prefix) + std::string(text);
    const std::lock_guard<std::mutex> lock(m_mutex);
    m_sink(whole);
}

void service_log::on_state(const scan_state &state) noexcept
{
    if (state.phase != m_phase)
    {
        const std::string scan = "scan " + state.scan + ": ";
        switch (state.phase)
        {
        case scan_phase::starting:
            line("a scan starts");
            break;
        case scan_phase::scanning:
            line(scan + "scanning");
            break;
        case scan_phase::refining:
            line(scan + "refining, " + ended_text(state));
            break;
        case scan_phase::idle:
            if (m_phase == scan_phase::starting)
                line("the start failed: " + state.refusal);
            else
                line(scan + std::string(name_of(state.outcome)) +
                     (state.outcome_message.empty() ? "" : ": " + state.outcome_message) + ", " +
                     ended_text(state));
            break;
        }
        m_phase = state.phase;
    }
    if (state.live_model_failure != m_live_failure)
    {
        m_live_failure = state.live_model_failure;
        if (!m_live_failure.empty())
            line("a live model could not be taken: " + m_live_failure);
    }
}

int serve(const service_settings &settings, frame_source &camera, camera_properties &properties,
          service_log &log)
{
    const int status = serve_until_stopped(settings, camera, properties, log);
    log.line("stopped");
    return status;
}
} // namespace gca
