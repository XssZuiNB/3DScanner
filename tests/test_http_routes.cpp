// SPDX-License-Identifier: 0BSD
/// @file test_http_routes.cpp
/// The web layer's routes (src/web/; docs/ROUTES.md), the spec's HTTP test: the web layer on a
/// free port of the loopback interface, over the scan interface on the tests' synthetic source at
/// 640x400 and a writer that waits for room, into a temporary scans directory, with a temporary
/// page directory; the requests made with cpp-httplib's client, the messages read with OpenCV's
/// JSON reader. The scene is the tabletop box seen along test_scan_interface.cpp's walk, its
/// first 120 poses, which a source in front of the synthetic one delivers at the camera's rate,
/// 30 frames a second, so that a scan scans long enough for the requests; the set is the
/// tabletop object class's own in a sparse TSDF of 273 MB (gca_test::case_budget, 272891920 bytes
/// [S: computed, sparse_tsdf_scenes.hpp]); these and the state's reader are the web tests' shared
/// ones (tests/web_test_helpers.hpp). The WebSocket has its own test, test_websocket.cpp.
/// - Idle: the state with each key of the document in its order; a stop and a discard refused by
///   the scan interface with its reason, which the state's last refusal repeats; a start without
///   its object class or with an unknown one refused by the web layer with its reason; a body
///   longer than the routes take refused.
/// - The page's files: index.html for "/", each media type by its extension, a link inside the
///   directory followed; refused: a path with "..", also one that would stay inside, with ".",
///   with an empty segment, with a backslash (a file of that name exists), with a NUL, an absolute
///   path, one with "%2F", a link that points out, a directory and a missing file; the file
///   outside never served.
/// - A whole scan: start, the state while scanning, a second start refused with the scan
///   interface's reason, stop, the state while refining, refined; then a second scan discarded,
///   nothing of it on disk. The scan's directory is as the same scan made in process leaves it:
///   the scan interface on the synthetic source with as many frames as the scan took, the same
///   set (tests/scan_comparison.hpp): the same files, the parameters but for start, stop and end.
/// - A second web layer on the same port, and one whose page directory does not exist: refused
///   at their start with the reason. Once the web layer is gone, nothing answers on its port.
/// The test prints its time; a scan-level test ends within 60 s on the board (the stage brief).

#include "camera/sequence_writer.hpp"
#include "check.hpp"
#include "scan/object_class.hpp"
#include "scan/parameters_file.hpp"
#include "scan/scan_interface.hpp"
#include "scan_comparison.hpp"
#include "stage_frames.hpp"
#include "test_files.hpp"
#include "web/web_server.hpp"
#include "web_test_helpers.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <opencv2/core.hpp>

namespace
{
using gca_test::check;
using gca_test::frames_keys;
using gca_test::keys_of;
using gca_test::names_in;
using gca_test::paced_source;
using gca_test::read_state;
using gca_test::state_fields;
using gca_test::state_keys;
using gca_test::temporary_directory;
using gca_test::test_set;
using gca_test::walk_options;

/// The poses a scan of the test may see: the walk's first 120 [?: a choice], 4 s at the camera's
/// rate, so that a scan stopped after 30 frames, about 1 s, has 3 s left for the stop to arrive
/// before its source ends.
constexpr std::size_t scan_poses = 120;

/// The frames the first scan has taken when the test stops it: 30 [?: a choice], so that its
/// refinement takes long enough to be seen and its model is no single view.
constexpr std::uint64_t frames_before_stop = 30;

/// The frames the second scan has taken when the test discards it: 5 [?: a choice].
constexpr std::uint64_t frames_before_discard = 5;

/// How long the test waits for a state it polls for: 30 s, within the 60 s of a scan-level test
/// (the stage brief) [?: a choice].
constexpr std::chrono::seconds poll_limit{30};

/// How long the listener holds the scan thread at the refining state until the client saw it:
/// 5 s [?: a choice, far above a few polls of 2 ms].
constexpr std::chrono::seconds hold_limit{5};

/// The content of the file outside the page's directory, which no request may get.
const std::string secret = "outside the page";

/// The hold of the refining state: the scan interface's listener holds the scan thread when it is
/// given the first refining state, until the test's client has seen that state over HTTP or the
/// hold's limit has passed, so that the client sees the scan refining however short refinement
/// is. The listener's contract asks it not to block (scan_interface.hpp, on_state); this test
/// blocks on purpose, and it cannot deadlock: the state is published, under the state's lock,
/// before the listener runs, and the state route's state() takes only that lock, which the scan
/// thread does not hold while it waits here; no command is given while it waits.
struct refining_hold
{
    std::mutex mutex;             ///< guards the rest
    std::condition_variable wake; ///< wakes the listener when the client saw the state
    bool client_saw = false;      ///< the client saw the refining state
    bool held = false;            ///< the listener held the scan thread once
    bool timed_out = false;       ///< the hold's limit passed before the client saw it

    /// The listener's part: holds at the first refining state.
    /// @param s [in] the state published.
    void on_state(const gca::scan_state &s)
    {
        if (s.phase != gca::scan_phase::refining)
            return;
        std::unique_lock<std::mutex> lock(mutex);
        if (held)
            return;
        held = true;
        timed_out = !wake.wait_for(lock, hold_limit, [this] { return client_saw; });
    }

    /// The test's part: lets the scan thread go on.
    void release()
    {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            client_saw = true;
        }
        wake.notify_all();
    }
};

/// A command's answer as the test reads it back.
struct reply_fields
{
    std::vector<std::string> keys; ///< the keys, in order
    int accepted = -1;             ///< 1 for true, 0 for false; -1 when no integer
    std::string refusal;           ///< the refusal
};

/// Reads a command's answer with OpenCV's JSON reader.
/// @param json [in] the message.
/// @return its fields; nothing when the reader refuses it.
std::optional<reply_fields> read_reply(const std::string &json)
{
    try
    {
        const cv::FileStorage storage(json, cv::FileStorage::READ | cv::FileStorage::MEMORY |
                                                cv::FileStorage::FORMAT_JSON);
        const cv::FileNode root = storage.root();
        reply_fields read;
        read.keys = keys_of(root);
        read.accepted = root["accepted"].isInt() ? static_cast<int>(root["accepted"]) : -1;
        read.refusal = root["refusal"].isString() ? static_cast<std::string>(root["refusal"])
                                                  : std::string("<no string>");
        return read;
    }
    catch (const cv::Exception &e)
    {
        std::printf("the answer refused by OpenCV: %s\n", e.what());
        return std::nullopt;
    }
}

/// The test's client of the web layer: raw request targets, sent as written.
class client
{
public:
    /// Builds the client of a port of the loopback interface.
    /// @param port [in] the port.
    explicit client(std::uint16_t port)
        : m_http("127.0.0.1", port)
    {
        m_http.set_path_encode(false);
    }

    /// Reads the state.
    /// @param what [in] the case, named in the check lines.
    /// @return the state; nothing, with a failed check, when the request or the message failed.
    std::optional<state_fields> state(const std::string &what)
    {
        const httplib::Result r = m_http.Get("/api/state");
        check(r && r->status == 200 && r->get_header_value("Content-Type") == "application/json",
              (what + ": the state, 200 and application/json").c_str());
        if (!r || r->status != 200)
            return std::nullopt;
        std::optional<state_fields> read = read_state(r->body);
        check(read && read->keys == state_keys && read->frames_keys == frames_keys,
              (what + ": the state's keys as the document's, in order; it read:\n" + r->body)
                  .c_str());
        return read;
    }

    /// Gives a command.
    /// @param target [in] the request's target, "/api/start?object_class=tabletop_object" for one.
    /// @param body [in] the request's body; empty for none.
    /// @return the response; nothing when the request failed.
    httplib::Result post(const std::string &target, const std::string &body = {})
    {
        return body.empty() ? m_http.Post(target) : m_http.Post(target, body, "text/plain");
    }

    /// Gets a page's file.
    /// @param target [in] the request's target, as sent.
    /// @return the response; nothing when the request failed.
    httplib::Result get(const std::string &target)
    {
        return m_http.Get(target);
    }

private:
    httplib::Client m_http;
};

/// Gives a command and checks its answer: the status, the message's keys, whether it was taken
/// and the refusal.
/// @param c [in,out] the client.
/// @param target [in] the command's target.
/// @param status [in] the status expected.
/// @param refusal [in] the refusal expected; empty for a command taken.
/// @param what [in] the case, named in the check line.
void check_command(client &c, const std::string &target, int status, const std::string &refusal,
                   const std::string &what)
{
    const httplib::Result r = c.post(target);
    const std::optional<reply_fields> read =
        r ? read_reply(r->body) : std::optional<reply_fields>();
    std::printf("%s: %d %s\n", what.c_str(), r ? r->status : 0, r ? r->body.c_str() : "");
    check(r && r->status == status && r->get_header_value("Content-Type") == "application/json" &&
              read && read->keys == std::vector<std::string>{"accepted", "refusal"} &&
              read->accepted == (refusal.empty() ? 1 : 0) && read->refusal == refusal,
          (what + ": status " + std::to_string(status) + ", the refusal \"" + refusal + "\"")
              .c_str());
}

/// Polls the state until a condition holds or the poll's limit passes.
/// @param c [in,out] the client.
/// @param until [in] the condition.
/// @param what [in] the case, named in the check line.
/// @param seen [out] each phase seen, once where it repeats; may be null.
/// @return the state that met the condition; the last read, with a failed check, when none did.
state_fields poll(client &c, const std::function<bool(const state_fields &)> &until,
                  const std::string &what, std::vector<std::string> *seen = nullptr)
{
    const auto begin = std::chrono::steady_clock::now();
    state_fields last;
    while (std::chrono::steady_clock::now() - begin < poll_limit)
    {
        const std::optional<state_fields> s = c.state(what);
        if (s)
        {
            last = *s;
            if (seen != nullptr && (seen->empty() || seen->back() != s->phase))
                seen->push_back(s->phase);
            if (until(*s))
                return last;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    check(false, (what + ": reached within the poll's limit").c_str());
    return last;
}

/// Writes the page's directory and what lies around it: index.html, a file of each media type,
/// a subdirectory, a file whose name holds a backslash, a link inside that stays inside, and a
/// file outside with a link to it.
/// @param directory [in] the test's directory; the page's directory is its "page".
void write_page(const std::filesystem::path &directory)
{
    const std::filesystem::path page = directory / "page";
    std::filesystem::create_directories(page / "sub");
    gca_test::write_bytes(page / "index.html", "<!doctype html><title>scan</title>\n");
    gca_test::write_bytes(page / "app.js", "export const a = 1;\n");
    gca_test::write_bytes(page / "style.css", "body { margin: 0; }\n");
    gca_test::write_bytes(page / "data.json", "{\"a\": 1}\n");
    gca_test::write_bytes(page / "model.bin", std::string("\x01\x02\x03", 3));
    gca_test::write_bytes(page / "sub" / "inner.html", "<p>inner</p>\n");
    gca_test::write_bytes(page / "back\\slash.html", "<p>a backslash in its name</p>\n");
    gca_test::write_bytes(directory / "secret.txt", secret);
    std::filesystem::create_symlink("../secret.txt", page / "out.txt");
    std::filesystem::create_symlink("sub/inner.html", page / "in.html");
}

/// The page's files: each served with its media type, every path that leads outside refused.
/// @param c [in,out] the client.
/// @param directory [in] the test's directory.
void check_page(client &c, const std::filesystem::path &directory)
{
    struct served
    {
        std::string target;
        std::filesystem::path file;
        std::string type;
    };
    const std::filesystem::path page = directory / "page";
    const std::vector<served> files = {
        {"/", page / "index.html", "text/html; charset=utf-8"},
        {"/index.html", page / "index.html", "text/html; charset=utf-8"},
        {"/app.js", page / "app.js", "text/javascript; charset=utf-8"},
        {"/style.css", page / "style.css", "text/css; charset=utf-8"},
        {"/data.json", page / "data.json", "application/json"},
        {"/model.bin", page / "model.bin", "application/octet-stream"},
        {"/sub/inner.html", page / "sub" / "inner.html", "text/html; charset=utf-8"},
        {"/in.html", page / "sub" / "inner.html", "text/html; charset=utf-8"},
    };
    for (const served &f : files)
    {
        const httplib::Result r = c.get(f.target);
        check(r && r->status == 200 && r->body == gca_test::bytes_of(f.file) &&
                  r->get_header_value("Content-Type") == f.type,
              ("the page: " + f.target + " served with " + f.type).c_str());
    }
    const std::string outside = (directory / "secret.txt").string();
    const std::vector<std::string> refused = {
        "/../secret.txt",
        "/sub/../../secret.txt",
        "/" + outside,
        "/%2F" + outside.substr(1),
        "/out.txt",
        "/sub",
        "/missing.html",
        "/sub/../app.js",
        "/./app.js",
        "/sub//inner.html",
        "/sub/",
        "/back%5Cslash.html",
        "/index.html%00.js",
    };
    for (const std::string &target : refused)
    {
        const httplib::Result r = c.get(target);
        std::printf("the page: %s: %d\n", target.c_str(), r ? r->status : 0);
        check(r && r->status == 404 && r->body.find(secret) == std::string::npos,
              ("the page: " + target + " refused with 404").c_str());
    }
}

/// The refusals while idle: the scan interface's, with its reason, and the web layer's own.
/// @param c [in,out] the client.
void check_idle_refusals(client &c)
{
    check_command(c, "/api/stop", 409, "no scan is in progress", "idle: a stop");
    check_command(c, "/api/discard", 409, "no scan is in progress", "idle: a discard");
    const std::optional<state_fields> s = c.state("idle: after the refusals");
    check(s && s->phase == "idle" && s->refusal == "no scan is in progress",
          "idle: the state's last refusal is the scan interface's");
    check_command(c, "/api/start", 400, "a start names its object class: object_class=<name>",
                  "idle: a start without its object class");
    check_command(c, "/api/start?object_class=giant", 400, "no object class is named \"giant\"",
                  "idle: a start of an unknown object class");
    const httplib::Result r = c.post("/api/stop", std::string(2048, 'x'));
    check(r && r->status == 413, "idle: a body of 2048 bytes refused with 413");
}

/// What the whole scan left.
struct scan_made
{
    std::filesystem::path directory; ///< its directory; empty when it was not refined
    std::uint64_t taken = 0;         ///< the frames it took
};

/// A whole scan through the routes: start, the state while scanning, a second start refused,
/// stop, the state while refining, refined.
/// @param c [in,out] the client.
/// @param hold [in,out] the refining state's hold.
/// @param scans [in] the scans directory.
/// @return the scan's directory and the frames it took.
scan_made check_whole_scan(client &c, refining_hold &hold, const std::filesystem::path &scans)
{
    check_command(c, "/api/start?object_class=tabletop_object", 200, "", "scan: the start");
    const state_fields scanning = poll(
        c,
        [](const state_fields &s) {
            return s.phase != "scanning" || s.frames.taken >= frames_before_stop;
        },
        "scan: scanning");
    std::printf("scan: scanning, %llu frames taken, %llu fused\n",
                static_cast<unsigned long long>(scanning.frames.taken),
                static_cast<unsigned long long>(scanning.frames.fused));
    const std::vector<std::string> on_disk = names_in(scans);
    check(scanning.phase == "scanning" && scanning.version > 0 && on_disk.size() == 1 &&
              scanning.scan == on_disk[0] && scanning.tracking == "tracking" &&
              scanning.lost_cause == "passed" &&
              (scanning.too_close == 0 || scanning.too_close == 1) &&
              scanning.frames.taken == scanning.frames.fused + scanning.frames.lost &&
              scanning.frames.not_taken == 0 && scanning.end.empty() &&
              scanning.outcome == "none" && scanning.live_model_failure.empty(),
          "scan: the state while scanning, its directory named, every frame taken fused or lost");

    check_command(c, "/api/start?object_class=tabletop_object", 409,
                  "a scan is in progress (scanning)", "scan: a second start");
    const std::optional<state_fields> after = c.state("scan: after the second start");
    check(after && after->refusal == "a scan is in progress (scanning)" &&
              after->phase == "scanning",
          "scan: the state's last refusal is the second start's, the scan going on");

    check_command(c, "/api/stop", 200, "", "scan: the stop");
    std::vector<std::string> phases;
    const state_fields refining = poll(
        c, [](const state_fields &s) { return s.phase == "refining" || s.phase == "idle"; },
        "scan: refining", &phases);
    hold.release();
    check(refining.phase == "refining" && refining.end == "stopped" &&
              refining.frames.taken >= frames_before_stop && refining.outcome == "none",
          "scan: the state while refining, the scan stopped");
    const state_fields idle =
        poll(c, [](const state_fields &s) { return s.phase == "idle"; }, "scan: idle", &phases);
    std::string seen;
    for (const std::string &p : phases)
        seen += " " + p;
    std::printf("scan: the phases seen after the stop:%s; %llu frames taken\n", seen.c_str(),
                static_cast<unsigned long long>(idle.frames.taken));
    {
        const std::lock_guard<std::mutex> lock(hold.mutex);
        check(hold.held && !hold.timed_out,
              "scan: the client saw the refining state within the hold's limit");
    }
    check(idle.outcome == "refined" && idle.end == "stopped" && idle.outcome_message.empty() &&
              idle.frames == refining.frames && idle.scan == scanning.scan,
          "scan: refined, its counts as when it was stopped");
    if (idle.outcome != "refined")
        return {};
    return {scans / idle.scan, idle.frames.taken};
}

/// A second scan, discarded while scanning: nothing of it stays on disk.
/// @param c [in,out] the client.
/// @param scans [in] the scans directory.
/// @param first [in] the first scan's directory.
void check_discarded_scan(client &c, const std::filesystem::path &scans,
                          const std::filesystem::path &first)
{
    check_command(c, "/api/start?object_class=tabletop_object", 200, "", "discard: the start");
    const state_fields scanning = poll(
        c,
        [](const state_fields &s) {
            return s.phase != "scanning" || s.frames.taken >= frames_before_discard;
        },
        "discard: scanning");
    check(scanning.phase == "scanning", "discard: scanning");
    check_command(c, "/api/discard", 200, "", "discard: the discard");
    const state_fields idle =
        poll(c, [](const state_fields &s) { return s.phase == "idle"; }, "discard: idle");
    check(idle.outcome == "discarded" && idle.scan == scanning.scan, "discard: the scan discarded");
    check(names_in(scans) == std::vector<std::string>{first.filename().string()},
          "discard: the first scan alone on disk");
}

/// The scan's directory against the same scan made in process: the synthetic source with as many
/// of the walk's poses as the scan took, the same set.
/// @param made [in] the scan made through the routes.
/// @param directory [in] the test's directory.
void check_same_as_in_process(const scan_made &made, const std::filesystem::path &directory)
{
    const std::filesystem::path scans = directory / "in_process";
    std::filesystem::create_directory(scans);
    gca_test::synthetic_source source(walk_options(made.taken));
    const gca::scan_state state =
        gca_test::scan_in_process(source, scans, test_set(gca::object_class::tabletop_object));
    check(state.outcome == gca::scan_outcome::refined,
          ("in process: refined; " + state.outcome_message).c_str());
    const std::filesystem::path in_process = gca_test::the_scan(scans);
    if (in_process.empty())
        return;
    gca_test::check_same_scan(made.directory, in_process, "same scan");
    const gca::read_result<gca::scan_parameters> a =
        gca::read_parameters(made.directory / "parameters.json");
    const gca::read_result<gca::scan_parameters> b =
        gca::read_parameters(in_process / "parameters.json");
    check(a.value && b.value, "same scan: both parameters read");
    if (!a.value || !b.value)
        return;
    gca::scan_parameters expected = *a.value;
    expected.start = b.value->start;
    expected.stop = b.value->stop;
    expected.end = gca::scan_end::source_end;
    check(a.value->end == gca::scan_end::stopped && *b.value == expected,
          "same scan: the parameters the same but for start, stop and end, stopped through the "
          "routes, at the source's end in process");
}

/// The refusals of a web layer's start: a port in use, and a page directory that does not exist.
/// @param scanner [in,out] the scan interface the other web layers are built over.
/// @param news [in] its news, which the other web layers are given too.
/// @param port [in] the port the test's web layer listens on.
/// @param directory [in] the test's directory.
void check_start_refusals(gca::scan_interface &scanner, const gca::scan_news &news,
                          std::uint16_t port, const std::filesystem::path &directory)
{
    gca::web_server same_port(scanner, news,
                              {.address = "127.0.0.1",
                               .port = port,
                               .page_directory = directory / "page",
                               .set_of = test_set});
    const gca::web_start_result taken = same_port.start();
    gca_test::check_refusal("a second web layer on the same port", !taken.started, taken.reason,
                            taken.reason == "cannot listen on 127.0.0.1 port " +
                                                std::to_string(port) +
                                                ": the port is in use or not allowed, or the "
                                                "address is not the board's");
    gca::web_server no_page(scanner, news,
                            {.address = "127.0.0.1",
                             .port = 0,
                             .page_directory = directory / "no_page",
                             .set_of = test_set});
    const gca::web_start_result missing = no_page.start();
    gca_test::check_refusal(
        "a web layer without its page directory", !missing.started, missing.reason,
        missing.reason.rfind("the page's directory " + (directory / "no_page").string() + ": ",
                             0) == 0);
}

/// Runs the test's web layer over the scan interface, the cases through its routes, then ends it.
/// @param directory [in] the test's directory, with the page written.
/// @return the scan made through the routes.
scan_made run_routes(const std::filesystem::path &directory)
{
    const std::filesystem::path scans = directory / "scans";
    std::filesystem::create_directory(scans);
    paced_source source(walk_options(scan_poses));
    gca::sequence_writer writer(gca::queue_full::wait);
    refining_hold hold;
    // The news first, as the service builds it, so that it outlives the scan interface.
    gca::scan_news news;
    const gca::scan_options options = {
        .scans_directory = scans,
        .stream_size = gca_test::small_description.size,
        .commit = gca_test::in_process_commit,
        .on_state =
            [&hold, &news](const gca::scan_state &s) {
                news.state_published();
                hold.on_state(s);
            },
        .stage_times = nullptr,
        .on_live_model = [&news](const gca::live_model &) { news.live_model_published(); }};
    gca::scan_interface scanner(options, source, writer);
    std::uint16_t port = 0;
    scan_made made;
    {
        gca::web_server server(scanner, news,
                               {.address = "127.0.0.1",
                                .port = 0,
                                .page_directory = directory / "page",
                                .set_of = test_set});
        const gca::web_start_result started = server.start();
        check(started.started && started.port != 0,
              ("the web layer listens on a free port; " + started.reason).c_str());
        if (!started.started)
            return {};
        port = started.port;
        std::printf("the web layer listens on 127.0.0.1 port %u\n", unsigned{port});
        client c(port);
        const std::optional<state_fields> idle = c.state("idle");
        check(idle && idle->phase == "idle" && idle->outcome == "none" && idle->scan.empty() &&
                  idle->tracking.empty() && idle->lost_cause == "passed" && idle->too_close == 0 &&
                  idle->frames == gca::scan_frame_counts{} && idle->end.empty() &&
                  idle->refusal.empty(),
              "idle: the state before the first scan");
        check_page(c, directory);
        check_idle_refusals(c);
        made = check_whole_scan(c, hold, scans);
        check_discarded_scan(c, scans, made.directory);
        check_start_refusals(scanner, news, port, directory);
    }
    client after(port);
    const httplib::Result r = after.get("/api/state");
    check(!r, "the web layer gone: nothing answers on its port");
    return made;
}
} // namespace

int main()
{
    const auto begin = std::chrono::steady_clock::now();
    const temporary_directory directory("test_http_routes");
    write_page(directory.path());
    const scan_made made = run_routes(directory.path());
    check(!made.directory.empty(), "a scan refined through the routes");
    if (!made.directory.empty())
        check_same_as_in_process(made, directory.path());
    std::printf("test_http_routes took %.1f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
    return gca_test::check_summary("test_http_routes");
}
