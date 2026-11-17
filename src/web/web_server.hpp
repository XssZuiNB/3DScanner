// SPDX-License-Identifier: 0BSD
/// @file web_server.hpp
/// The web layer's HTTP side (the spec, "The web layer and the page"; docs/ROUTES.md): the routes
/// that carry the scan interface's commands (start a scan of an object class, stop, discard), its
/// state, and the page's static files, served by cpp-httplib. The web layer translates a request
/// into a command or a read of the state and the answer into a message (web/messages.hpp); it
/// holds no state of its own, only its options and the page's directory. One thread of its own
/// accepts the connections; the library's pool of workers runs the requests, each of which may
/// wait for the scan interface's answer to a command. No authentication: the routes are for the
/// operator's own network, not for the internet (R6).

#pragma once

#include "scan/object_class.hpp"
#include "scan/scan_interface.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace httplib
{
class Server;
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
    /// Builds the web layer, idle: its routes and the page's directory as its canonical path.
    /// @param scanner [in,out] the scan interface whose commands and state the routes carry; owned
    ///        by the caller, which keeps it alive until the web layer's end, so that a request
    ///        waiting for a command's answer gets it.
    /// @param options [in] the address, the port, the page's directory and the sets; taken over.
    /// @pre options.set_of is not empty; otherwise a start request's call of it throws
    ///      std::bad_function_call, which the library answers with 500.
    /// @note Allocates; a failed allocation throws std::bad_alloc. A page directory that cannot be
    ///       resolved is not refused here but by start().
    web_server(scan_interface &scanner, web_options options);

    /// The web layer stands for one listening socket and its thread.
    web_server(const web_server &) = delete;
    web_server &operator=(const web_server &) = delete;

    /// Ends the web layer: the listening socket closed, so the port is free, and its thread joined,
    /// which waits for every request in progress, a command's answer too, and for the library's
    /// workers.
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
    scan_interface &m_scanner; ///< the scan interface; the caller's
    web_options m_options;     ///< the options
    /// The page's directory as its canonical path, every link resolved; empty when it could not be
    /// resolved, and then m_page_failure says why.
    std::filesystem::path m_page;
    std::string m_page_failure;                ///< why the page's directory could not be resolved
    std::unique_ptr<httplib::Server> m_server; ///< the library's server with the routes
    std::thread m_thread;                      ///< the web layer's thread, which accepts
};
} // namespace gca
