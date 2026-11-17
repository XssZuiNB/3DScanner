// SPDX-License-Identifier: 0BSD
/// @file web_server.hpp
/// The web layer (the spec, "The web layer and the page"; docs/ROUTES.md): the HTTP routes that
/// carry the scan interface's commands (start a scan of an object class, stop, discard), its
/// state, and the page's static files, and the WebSocket that pushes the state and the live model
/// to every browser, served by cpp-httplib. The web layer translates a request into a command or a
/// read of the state and the answer into a message (web/messages.hpp); it holds no state of the
/// scan's own, only its options, the page's directory and the newest messages it laid out. One
/// thread of its own accepts the connections; the library's pool of workers runs the requests,
/// each of which may wait for the scan interface's answer to a command, and each WebSocket's loop
/// for as long as its browser stays. The scan interface's listeners tell the web layer what was
/// published through scan_news, which only counts, so that the scan thread never waits for a
/// browser. No authentication: the routes are for the operator's own network, not for the internet
/// (R6).

#pragma once

#include "scan/object_class.hpp"
#include "scan/scan_interface.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace httplib
{
class Server;
namespace ws
{
class WebSocket;
} // namespace ws
} // namespace httplib

namespace gca
{
/// How the web layer is set up, for its lifetime.
struct web_options
{
    /// The address it listens on, numeric: "127.0.0.1" for the board alone, "0.0.0.0" for every
    /// network the board is on.
    std::string address;
    std::uint16_t port = 0; ///< the port it listens on; 0: a free one the system picks
    /// The page's directory, whose files the static route serves; it exists. A link in it that
    /// leads outside it is refused, so the page's files are copied in, not linked.
    std::filesystem::path page_directory;
    /// Gives the set a start of an object class runs with: the class's own (parameters_of()) for
    /// the service, a changed one for a test; called on a worker's thread at each start, so it
    /// must be safe to call from several threads at once.
    std::function<object_class_parameters(object_class)> set_of;
};

/// What the scan interface's listeners tell the web layer: how many states and how many live models
/// it has published. The web layer compares the counts with those its newest messages were laid
/// out at, and reads a state or a live model from the scan interface only when one changed. Built
/// before the scan interface, whose options take its two listeners, and destroyed after it:
///
///     gca::scan_news news;
///     options.on_state = [&news](const gca::scan_state &) { news.state_published(); };
///     options.on_live_model = [&news](const gca::live_model &) { news.live_model_published(); };
///
/// Without them the web layer pushes the state and the live model a browser finds at its connection
/// and nothing after.
class scan_news
{
public:
    /// Counts a state published: the scan interface's on_state listener calls it, on the scan
    /// thread, after the state is published.
    /// @note One atomic addition: takes no lock, allocates nothing and waits for nothing. Any
    ///       thread may call it.
    void state_published() noexcept;

    /// Counts a live model published: the scan interface's on_live_model listener calls it, on the
    /// scan thread, after the live model is published.
    /// @note As state_published().
    void live_model_published() noexcept;

    /// Gives the states counted.
    /// @return how many; a state published before the count it gives is published, read through
    ///         scan_interface::state() after it.
    /// @note One atomic read; any thread may call it.
    std::uint64_t states() const noexcept;

    /// Gives the live models counted.
    /// @return how many, as states() says for the live models.
    /// @note As states().
    std::uint64_t live_models() const noexcept;

private:
    std::atomic<std::uint64_t> m_states{0};      ///< the states counted
    std::atomic<std::uint64_t> m_live_models{0}; ///< the live models counted
};

/// What start() gave.
struct web_start_result
{
    bool started = false;   ///< whether the web layer listens and serves
    std::uint16_t port = 0; ///< the port it listens on, the system's choice for port 0; 0 when not
    std::string reason;     ///< why it does not, in plain English; empty when it does
};

/// The web layer over a scan interface. Built idle; start() binds and serves, the destructor ends
/// it.
class web_server
{
public:
    /// Builds the web layer, idle: its routes, its WebSocket and the page's directory as its
    /// canonical path.
    /// @param scanner [in,out] the scan interface whose commands, state and live model the routes
    ///        and the WebSocket carry; owned by the caller, which keeps it alive until the web
    ///        layer's end, so that a request waiting for a command's answer gets it.
    /// @param news [in] the counts the scan interface's listeners keep (scan_news); owned by the
    ///        caller, which keeps it alive until the web layer's end.
    /// @param options [in] the address, the port, the page's directory and the sets; taken over.
    /// @pre options.set_of is not empty; otherwise a start request's call of it throws
    ///      std::bad_function_call, which the library answers with 500.
    /// @note Allocates; a failed allocation throws std::bad_alloc. A page directory that cannot be
    ///       resolved is not refused here but by start().
    web_server(scan_interface &scanner, const scan_news &news, web_options options);

    /// The web layer stands for one listening socket and its thread.
    web_server(const web_server &) = delete;
    web_server &operator=(const web_server &) = delete;

    /// Ends the web layer: each WebSocket's loop told to end, the listening socket closed, so the
    /// port is free, and its thread joined, which waits for every request in progress, a command's
    /// answer too, and for the library's workers. A WebSocket's loop ends within its poll, 20 ms
    /// [?: a choice], or once the send it is in ends: for a browser that stopped reading, at the
    /// library's write timeout, 5 s [S: httplib.h, CPPHTTPLIB_SERVER_WRITE_TIMEOUT_SECOND], and a
    /// ping the library sends to it at that moment may wait as long again (docs/ROUTES.md,
    /// "Threads"). The connection is closed without the WebSocket's closing handshake.
    ~web_server();

    /// Binds the address and port and starts serving on the web layer's thread.
    /// @return started with the port; or not, with the reason: the page's directory could not be
    ///         resolved or is no directory, the address and port could not be bound (a port in
    ///         use or not allowed, an address that is not the board's), the thread could not be
    ///         started, or the web layer was started already.
    /// @note Allocates and starts a thread; a failed allocation ends the program (std::terminate):
    ///       the function is noexcept. Call it from one thread, before the destructor. The
    ///       library's pool of workers starts with the serving: 8 threads on this board's 6 cores
    ///       [M 2026-11-17: nproc], up to 32 while every one is busy (httplib.h,
    ///       CPPHTTPLIB_THREAD_POOL_COUNT and _MAX_COUNT [S: cpp-httplib v0.66.0]; docs/ROUTES.md,
    ///       "Threads").
    web_start_result start() noexcept;

private:
    /// The newest message of one kind, laid out once for each version and shared by every
    /// WebSocket that sends it.
    struct newest_message
    {
        std::uint64_t counted = 0; ///< the count of the news it was taken at (scan_news)
        std::uint64_t version = 0; ///< the version of the state or the live model it holds
        std::shared_ptr<const std::string> message; ///< the message; none before the first
    };

    /// Gives the newest message of one kind, the state's or the live model's: the one laid out
    /// last, or, when the news counted a publication of its kind since, the value read again from
    /// the scan interface and laid out when its version changed.
    /// @tparam Value scan_state or live_model, each with its version.
    /// @param kind [in,out] the newest message of the kind, m_state or m_live.
    /// @param count [in] the news' count of the kind: scan_news::states or live_models.
    /// @param read [in] the scan interface's copy of the kind: scan_interface::state or
    ///        latest_live_model.
    /// @param lay_out [in] the kind's message (web/messages.hpp): state_message or
    ///        live_model_message.
    /// @return the message, the same object for one version; never empty.
    /// @note Takes the messages' lock, and inside it the scan interface's lock of the kind for one
    ///       copy: the state's, or the live model's (up to 5.6 MB [S: computed, 200000 points of 28
    ///       bytes]), which the scan thread's next live model waits for when it comes during the
    ///       copy; it comes at least the live model's interval, 500 ms [?: initial], after the one
    ///       copied. Allocates the copy and a new message; a failed allocation throws
    ///       std::bad_alloc. Any thread.
    template <typename Value>
    std::shared_ptr<const std::string> newest(newest_message &kind,
                                              std::uint64_t (scan_news::*count)() const noexcept,
                                              Value (scan_interface::*read)() const,
                                              std::string (*lay_out)(const Value &));

    /// Runs one WebSocket's loop on its worker, for as long as its browser stays (docs/ROUTES.md,
    /// "The WebSocket"): every poll, it sends the newest state message as text and the newest live
    /// model message as binary, each when it differs from the one this browser got last, outside
    /// every lock, then reads the browser's frames for up to a poll (a ping's answer, a close; a
    /// message is read and dropped). So a browser that reads slowly gets the newest of each when
    /// its send ends; the older ones it missed are dropped for it, and nothing queues.
    /// @param socket [in,out] the WebSocket, the library's, open; it lives until the loop returns.
    /// @note Returns when the browser closed or is gone (a send or a read failed), or when the web
    ///       layer ends. Allocates what newest() allocates; a failed
    ///       allocation ends the program (std::terminate): the function is noexcept.
    void serve(httplib::ws::WebSocket &socket) noexcept;

    scan_interface &m_scanner; ///< the scan interface; the caller's
    const scan_news &m_news;   ///< the scan interface's news; the caller's
    web_options m_options;     ///< the options
    /// The page's directory as its canonical path, every link resolved; empty when it could not be
    /// resolved, and then m_page_failure says why.
    std::filesystem::path m_page;
    std::string m_page_failure;                ///< why the page's directory could not be resolved
    std::unique_ptr<httplib::Server> m_server; ///< the library's server with the routes
    std::thread m_thread;                      ///< the web layer's thread, which accepts

    // The messages' lock guards the newest messages alone: a WebSocket's loop takes it to copy a
    // pointer or, the first after a change, to lay the message out; no one holds it across a send
    // or a read, and the scan thread never takes it.
    std::mutex m_newest_mutex;
    newest_message m_state; ///< the newest state message
    newest_message m_live;  ///< the newest live model message

    std::atomic<bool> m_ending{false}; ///< tells every WebSocket's loop to end
};
} // namespace gca
