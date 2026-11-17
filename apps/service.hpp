// SPDX-License-Identifier: 0BSD
/// @file service.hpp
/// The scanner service's parts (the spec, "The service"; docs/SERVICE.md): the stop signals, the
/// service's log, and the composition that serves until a stop signal. The composition puts the
/// hold of the color camera's exposure and white balance (camera/color_hold.hpp) in front of the
/// camera, and builds the sequence's writer, the scan interface's news (web/web_server.hpp,
/// scan_news), the scan interface, which opens the camera only from a scan's start to its stop,
/// and the web layer, in that order; it ends them in the reverse order, so that the web layer,
/// whose requests may wait for the scan interface, ends first, and the scan interface, which
/// ends a scan in progress without refinement, its source closed through the hold, before the
/// hold and the writer. The program, apps/scanner_service.cpp, gives it the Orbbec camera; the
/// offline test gives it a stand-in on the synthetic source and drives it as the program does.
/// Links no OrbbecSDK.

#pragma once

#include "camera/camera_properties.hpp"
#include "camera/frame_source.hpp"
#include "scan/object_class.hpp"
#include "scan/scan_interface.hpp"
#include "service_options.hpp"
#include "stop_signals.hpp"
#include "types/frame_description.hpp"

#include <functional>
#include <mutex>
#include <string>
#include <string_view>

namespace gca
{
/// Makes the stop signals (stop_signals.hpp), SIGINT, SIGTERM and SIGHUP, the service's stop
/// (wait_for_stop_signal()): blocks the three in the calling thread, so that every thread it
/// starts afterwards, the scan interface's, the writer's, the web layer's and OrbbecSDK's,
/// inherits the block and none of them is interrupted by one, and installs a handler that records
/// one, for a thread started before main(), whose mask is its own [?: whether OrbbecSDK's or
/// CUDA's load-time code starts one is unverified]. A signal that arrives before
/// wait_for_stop_signal() stays pending until then; one that arrives after it took one stays
/// pending and does nothing, so only a SIGKILL ends the stop early. It also ignores SIGPIPE, so
/// that a write to a browser that closed its connection fails with EPIPE instead of ending the
/// service; cpp-httplib's server does the same when it is built [S: httplib.h, Server::Server()],
/// and this states it where the service's signals are set.
/// @pre Called by the thread that calls wait_for_stop_signal(), before it starts any other
///      thread; a thread started before is not blocked, and a signal it takes runs the handler.
/// @note Changes the process's signal dispositions and the calling thread's mask; allocates
///       nothing. Called again, it clears the signal recorded and installs the same again.
void install_stop_signals() noexcept;

/// Waits for a stop signal: SIGINT (Ctrl-C), SIGTERM (systemd's stop, a kill) or SIGHUP (a closed
/// terminal or SSH session).
/// @return the signal with its name, a row of stop_signals.
/// @pre install_stop_signals() was called by this thread.
/// @note Takes a blocked signal at once and looks for one the handler recorded every 100 ms [?: a
///       choice]; waits as long as no signal comes. Allocates nothing.
const stop_signal &wait_for_stop_signal() noexcept;

/// The service's log (docs/SERVICE.md, "Log lines"): whole lines, each service_line_prefix
/// (service_options.hpp) and its text, through one lock, so that the main thread's lines and the
/// scan thread's do not mix. The scan thread's lines say what became of each scan, from the states
/// the scan interface publishes, and what the hold of the camera's color controls read and the
/// camera refused.
class service_log
{
public:
    /// Builds the log.
    /// @param sink [in] where each line goes, whole and without its line end: the program's
    ///        standard output, which systemd's journal takes; called under the log's lock on the
    ///        thread that logs, the main thread or the scan thread, which waits for it; taken over.
    explicit service_log(std::function<void(const std::string &)> sink);

    /// The log stands for one sink and its lock.
    service_log(const service_log &) = delete;
    service_log &operator=(const service_log &) = delete;

    /// Logs a line: service_line_prefix, "scanner_service: ", and the text.
    /// @param text [in] the line's text, without a line end.
    /// @note Allocates the line; takes the log's lock. Any thread.
    void line(std::string_view text);

    /// The scan interface's listener (scan/scan_interface.hpp, on_state): logs each change of the
    /// phase, "a scan starts", "the start failed: <reason>", "scan <scan>: scanning", "scan
    /// <scan>: refining, ..." and "scan <scan>: <outcome>, ...", and a live model that could not
    /// be taken.
    /// @param state [in] the state published.
    /// @note On the scan thread alone. Allocates only for a line; compares the phase and the live
    ///       model's failure with the last ones otherwise. A failed allocation ends the program
    ///       (std::terminate): the function is noexcept.
    void on_state(const scan_state &state) noexcept;

private:
    std::mutex m_mutex;                              ///< guards the sink
    std::function<void(const std::string &)> m_sink; ///< where the lines go
    scan_phase m_phase = scan_phase::idle; ///< the last phase seen; the scan thread's alone
    std::string m_live_failure; ///< the last live model's failure seen; the scan thread's alone
};

/// What the service runs with, beside the camera.
struct service_settings
{
    service_options options; ///< the command line's: scans directory, address, port, page, switch
    frame_size stream_size;  ///< the size the camera is opened for, in px
    std::string commit;      ///< the tree's commit, for each scan's parameters
    /// Gives the set a start of an object class runs with: the class's own (parameters_of()) for
    /// the service, a test's for the test; called on the web layer's workers, several at once.
    std::function<object_class_parameters(object_class)> set_of;
};

/// Serves until a stop signal (docs/SERVICE.md): composes the hold in front of the camera, the
/// writer, the news, the scan interface and the web layer; starts the web layer and logs
/// "serving the page on http://<address>:<port>/ from <page>; scans in <scans>, <sequences>", or
/// "the web layer did not start: <reason>"; waits for a stop signal and logs "stopping on
/// <signal>"; ends the parts in the reverse order: the web layer, which waits for every request
/// in progress and every WebSocket's loop, up to 5 s for a send that makes no progress and as
/// long again for a ping [S: docs/ROUTES.md, "Threads"]; the scan interface, which ends a scan in
/// progress at its next frame without refinement, its source closed through the hold, which
/// switches the camera's automatic controls on again, and its sequence and parameters kept, and
/// stops a refinement between two frames; then the writer and the hold. Logs "stopped" last.
/// @param settings [in] what the service runs with.
/// @param camera [in,out] the camera's frames, closed; owned by the caller, which keeps it alive
///        until serve() returns; the scan thread alone uses it meanwhile.
/// @param properties [in,out] the same camera's run-time properties; as camera.
/// @param log [in,out] the service's log; owned by the caller, which keeps it alive until serve()
///        returns.
/// @return 0 after a stop signal; 1 when the web layer did not start.
/// @pre install_stop_signals() was called by this thread before any other thread was started.
/// @note Allocates and starts the parts' threads; the scan interface's creates its CUDA stream. An
///       exception from building a part (std::bad_alloc, std::system_error for a thread) passes
///       through, the parts built so far ended on its way.
int serve(const service_settings &settings, frame_source &camera, camera_properties &properties,
          service_log &log);
} // namespace gca
