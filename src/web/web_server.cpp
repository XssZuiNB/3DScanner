// SPDX-License-Identifier: 0BSD
/// @file web_server.cpp
/// The web layer's routes on cpp-httplib (see web_server.hpp; docs/ROUTES.md): the state, the
/// three commands and the page's static files, and the check that keeps the static route inside
/// the page's directory; the WebSocket's loop with the newest messages it sends, and the counts of
/// the scan interface's news.

#include "web/web_server.hpp"

#include "scan/file_io.hpp"
#include "web/messages.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>

#include <httplib.h>

namespace gca
{
namespace
{
/// The media type of every message (RFC 8259: JSON is UTF-8 and takes no charset).
constexpr const char *json_type = "application/json";

/// The longest body a request may carry, in bytes: no route takes one [?: a choice, room for a
/// client that sends a small one]. The library refuses a longer one with 413 before any route
/// runs, so that a request cannot fill the board's memory, which the scan's sparse TSDF needs; its
/// own limit is 100 MB [S: httplib.h, CPPHTTPLIB_PAYLOAD_MAX_LENGTH].
constexpr std::size_t largest_body = 1024;

/// The longest a WebSocket's loop waits for its browser's frames before it looks for newer
/// messages: 20 ms [?: a choice, below a frame's 33 ms at 30 fps]; so a state or a live model
/// reaches a browser at most that long after a reader could have it, and the loop sees the web
/// layer's end within it.
constexpr std::chrono::milliseconds websocket_poll{20};

/// The pings a browser may leave unanswered before its WebSocket is closed: 2 [?: a choice], at the
/// library's interval of 30 s [S: httplib.h, CPPHTTPLIB_WEBSOCKET_PING_INTERVAL_SECOND], so that a
/// browser gone from the network without a word (a phone out of range) frees its worker within
/// about 90 s [S: computed, 3 × 30 s].
constexpr int websocket_missed_pongs = 2;

/// Gives a page's file's media type by its extension (docs/ROUTES.md, "The page's files").
/// @param file [in] the file's path.
/// @return text/html, text/javascript or text/css in UTF-8 for .html, .js and .css,
///         application/json for .json, application/octet-stream for every other extension.
const char *media_type_of(const std::filesystem::path &file)
{
    const std::filesystem::path extension = file.extension();
    if (extension == ".html")
        return "text/html; charset=utf-8";
    if (extension == ".js")
        return "text/javascript; charset=utf-8";
    if (extension == ".css")
        return "text/css; charset=utf-8";
    if (extension == ".json")
        return json_type;
    return "application/octet-stream";
}

/// Finds the page's file a request names: the directory check of the static route
/// (docs/ROUTES.md, "The page's files"), in two steps. First the path is refused as written when,
/// after its leading "/", it holds a backslash or a NUL byte, or a segment between two "/" that
/// is empty ("//", a trailing "/"), "." or "..": so an absolute path ("//etc/hostname", or
/// "/%2Fetc/hostname" once the library decoded it) and every "." and ".." are refused before
/// anything is resolved, also those that would stay inside. Then the path is joined to the page's
/// directory and resolved, every link included, and the result must lie inside the directory,
/// compared component by component, so that a link that points out reaches no file outside it. A
/// link inside the directory that stays inside is followed. Between this check and the read a link
/// inside the directory could be changed to point out; the directory is the service's own, not the
/// client's, so that is not guarded.
/// @param page [in] the page's directory, canonical.
/// @param request_path [in] the request's path as the library decoded it, starting with "/";
///        "/" names index.html.
/// @return the file's canonical path; nothing for a path refused as written, one that leads
///         outside the directory, does not exist, cannot be resolved or is no regular file (a
///         directory). No exception: every filesystem call takes its error_code.
std::optional<std::filesystem::path> page_file(const std::filesystem::path &page,
                                               std::string_view request_path)
{
    std::string_view relative = request_path.substr(request_path.empty() ? 0 : 1);
    if (relative.empty())
        relative = "index.html";
    if (relative.find_first_of(std::string_view("\\\0", 2)) != std::string_view::npos)
        return std::nullopt;
    for (std::size_t begin = 0; begin <= relative.size();)
    {
        const std::size_t end = std::min(relative.find('/', begin), relative.size());
        const std::string_view segment = relative.substr(begin, end - begin);
        if (segment.empty() || segment == "." || segment == "..")
            return std::nullopt;
        begin = end + 1;
    }
    std::error_code error;
    const std::filesystem::path file = std::filesystem::canonical(page / relative, error);
    if (error)
        return std::nullopt;
    const auto [in_page, in_file] =
        std::mismatch(page.begin(), page.end(), file.begin(), file.end());
    if (in_page != page.end() || in_file == file.end())
        return std::nullopt;
    if (!std::filesystem::is_regular_file(file, error))
        return std::nullopt;
    return file;
}

/// Answers a command's request with the scan interface's answer: 200 when it was taken, 409 when
/// the scan interface refused it, the answer's message either way.
/// @param res [out] the response.
/// @param reply [in] the answer.
void answer(httplib::Response &res, const command_reply &reply)
{
    res.status = reply.accepted ? 200 : 409;
    res.set_content(reply_message(reply), json_type);
}
} // namespace

void scan_news::state_published() noexcept
{
    m_states.fetch_add(1, std::memory_order_release);
}

void scan_news::live_model_published() noexcept
{
    m_live_models.fetch_add(1, std::memory_order_release);
}

std::uint64_t scan_news::states() const noexcept
{
    return m_states.load(std::memory_order_acquire);
}

std::uint64_t scan_news::live_models() const noexcept
{
    return m_live_models.load(std::memory_order_acquire);
}

web_server::web_server(scan_interface &scanner, const scan_news &news, web_options options)
    : m_scanner(scanner)
    , m_news(news)
    , m_options(std::move(options))
    , m_server(std::make_unique<httplib::Server>())
{
    std::error_code error;
    m_page = std::filesystem::canonical(m_options.page_directory, error);
    const bool directory = !error && std::filesystem::is_directory(m_page, error);
    if (!error && !directory)
        error = std::make_error_code(std::errc::not_a_directory);
    if (error)
    {
        m_page.clear();
        m_page_failure = error.message();
    }

    m_server->set_payload_max_length(largest_body);
    // SO_REUSEADDR alone, in place of the library's SO_REUSEPORT (httplib.h,
    // default_socket_options()), under which a second server of the user binds the same port and
    // the kernel shares the connections between both [S: socket(7), SO_REUSEPORT]: so a second
    // service on the port is refused at its start, while a restart binds the port its predecessor
    // left in TIME_WAIT [S: socket(7), SO_REUSEADDR].
    m_server->set_socket_options(
        [](socket_t sock) { httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1); });
    // Each request runs on one of the library's workers (web_server.hpp, start()). A command's
    // request waits there for the scan interface's answer, which the scan thread gives between two
    // frames, at a start's decision (the source opened, the GPU's buffers allocated: seconds for
    // the camera and a large sparse TSDF) or, during refinement, between two fused frames; the
    // scan interface answers every command, its end included. While it waits, it holds its
    // worker: a burst of commands holds as many workers, the pool grows to 32, and requests beyond
    // wait in the library's queue until a worker is free; a state request is answered at once by
    // another worker, as state() takes only the state's lock (docs/ROUTES.md, "Threads").
    m_server->Get("/api/state", [this](const httplib::Request &, httplib::Response &res) {
        res.set_content(state_message(m_scanner.state()), json_type);
    });
    m_server->Post("/api/start", [this](const httplib::Request &req, httplib::Response &res) {
        if (!req.has_param("object_class"))
        {
            res.status = 400;
            res.set_content(
                reply_message({false, "a start names its object class: object_class=<name>"}),
                json_type);
            return;
        }
        const std::string name = req.get_param_value("object_class");
        const std::optional<object_class> kind = object_class_named(name);
        if (!kind)
        {
            res.status = 400;
            res.set_content(reply_message({false, "no object class is named \"" + name + "\""}),
                            json_type);
            return;
        }
        answer(res, m_scanner.start(*kind, m_options.set_of(*kind)).get());
    });
    m_server->Post("/api/stop", [this](const httplib::Request &, httplib::Response &res) {
        answer(res, m_scanner.stop().get());
    });
    m_server->Post("/api/discard", [this](const httplib::Request &, httplib::Response &res) {
        answer(res, m_scanner.discard().get());
    });
    // The WebSocket: a request that asks to upgrade is matched against it before the routes; each
    // browser's loop runs on the worker that took its request, for as long as the browser stays
    // (docs/ROUTES.md, "Threads").
    m_server->set_websocket_max_missed_pongs(websocket_missed_pongs);
    m_server->WebSocket(
        "/api/live", [this](const httplib::Request &, httplib::ws::WebSocket &ws) { serve(ws); });
    // The page's files: every other GET; the routes above match first, in the order they were
    // added.
    m_server->Get("/.*", [this](const httplib::Request &req, httplib::Response &res) {
        const std::optional<std::filesystem::path> file = page_file(m_page, req.path);
        read_result<std::string> bytes;
        if (file)
            bytes = read_file(*file);
        if (!bytes.value)
        {
            res.status = 404;
            return;
        }
        res.set_content(std::move(*bytes.value), media_type_of(*file));
    });
}

web_server::~web_server()
{
    // The WebSockets' loops end first, as they would hold their workers: the flag is the one thing
    // that ends a loop whose browser stays, as stop() closes only the listening socket. stop()
    // closes it whether or not the thread accepts yet; the thread's listen then ends, after the
    // library's workers have finished their requests and loops.
    m_ending.store(true);
    m_server->stop();
    if (m_thread.joinable())
        m_thread.join();
}

web_start_result web_server::start() noexcept
{
    if (m_thread.joinable())
        return {false, 0, "the web layer is started already"};
    if (m_page.empty())
        return {false, 0,
                "the page's directory " + m_options.page_directory.string() + ": " +
                    m_page_failure};
    const int port = m_options.port == 0 ? m_server->bind_to_any_port(m_options.address)
                     : m_server->bind_to_port(m_options.address, m_options.port)
                         ? int{m_options.port}
                         : -1;
    if (port <= 0)
        return {false, 0,
                "cannot listen on " + m_options.address + " port " +
                    std::to_string(m_options.port) +
                    ": the port is in use or not allowed, or the address is not the board's"};
    try
    {
        m_thread = std::thread([this] { m_server->listen_after_bind(); });
    }
    catch (const std::system_error &e)
    {
        m_server->stop();
        return {false, 0, std::string("the web layer's thread: ") + e.what()};
    }
    // Once it returns, the thread accepts; a connection made before waits in the socket's backlog.
    m_server->wait_until_ready();
    return {true, static_cast<std::uint16_t>(port), {}};
}

template <typename Value>
std::shared_ptr<const std::string> web_server::newest(newest_message &kind,
                                                      std::uint64_t (scan_news::*count)()
                                                          const noexcept,
                                                      Value (scan_interface::*read)() const,
                                                      std::string (*lay_out)(const Value &))
{
    const std::lock_guard<std::mutex> lock(m_newest_mutex);
    // The count is read before the value, so that the value copied is at least as new as the count
    // says, and a value published meanwhile raises the count past the one recorded.
    const std::uint64_t counted = (m_news.*count)();
    if (kind.message && counted == kind.counted)
        return kind.message;
    const Value value = (m_scanner.*read)();
    if (!kind.message || value.version != kind.version)
        kind.message = std::make_shared<const std::string>(lay_out(value));
    kind.counted = counted;
    kind.version = value.version;
    return kind.message;
}

void web_server::serve(httplib::ws::WebSocket &socket) noexcept
{
    // A read that times out at a frame's boundary leaves the connection as it was; one that times
    // out after a frame's first byte fails it, and the connection closes [S: httplib.h,
    // read_exact()]. The browser's frames are small, a ping's answer 6 bytes and a close at most
    // 131 [S: RFC 6455, 5.2 and 5.5], and the page reconnects (docs/ROUTES.md, "The WebSocket").
    socket.set_read_timeout(websocket_poll);
    std::shared_ptr<const std::string> sent_state;
    std::shared_ptr<const std::string> sent_live;
    while (!m_ending.load())
    {
        // Each pointer is copied under the messages' lock and sent outside it, so that a send
        // that waits for a slow browser holds no lock any other loop or the scan thread needs.
        const std::shared_ptr<const std::string> state =
            newest(m_state, &scan_news::states, &scan_interface::state, &state_message);
        if (state != sent_state)
        {
            if (!socket.send(*state))
                return;
            sent_state = state;
        }
        const std::shared_ptr<const std::string> live =
            newest(m_live, &scan_news::live_models, &scan_interface::latest_live_model,
                   &live_model_message);
        if (live != sent_live)
        {
            if (!socket.send(live->data(), live->size()))
                return;
            sent_live = live;
        }
        std::string dropped;
        if (socket.read(dropped) == httplib::ws::Fail)
            return;
    }
}
} // namespace gca
