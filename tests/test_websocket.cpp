// SPDX-License-Identifier: 0BSD
/// @file test_websocket.cpp
/// The web layer's WebSocket (src/web/web_server.*; docs/ROUTES.md, "The WebSocket"): the web
/// layer on a free port of the loopback interface, over the scan interface on the tests' synthetic
/// source at 640x400, paced at the camera's 30 frames a second, and a writer that waits for room,
/// into a temporary scans directory; its browsers are cpp-httplib's WebSocket clients, which read
/// the state with OpenCV's JSON reader and the live model by the document's offsets. The scene,
/// the paced source, the set and the state's reader are the web tests' shared ones
/// (tests/web_test_helpers.hpp); the scan interface's listeners count into the web layer's news
/// (gca::scan_news), as the service's do.
/// - The live model's layout: a live model of known values, and the empty one, laid out and read
///   back by the document's offsets, give the values put in.
/// - A scan without any browser: how long its scanning takes, the base of the bound below.
/// - Two browsers and a late one: two browsers connected before a scan get the same states, the
///   phases in the same order, and the same live models byte for byte, the last the scan
///   interface's own, read back to its values. A third browser connects while the scan thread is
///   held in the middle of the scan, so that nothing changes, and gets the current state and live
///   model at once. The scanning ends within that of the scan without any browser and 2 s.
/// - A browser that stops reading: with a browser whose receive buffer is small and which reads
///   nothing until the scan's end, the scanning ends within the same bound, a reading browser gets
///   the scan's last live model, and the stopped browser got fewer live models than the reading
///   one, which shows that its sends were held up; when it reads again it gets the newest, or the
///   web layer has closed it.
/// - The web layer's end with a browser connected: the destructor returns within 1 s and the
///   browser's connection is closed. Once the web layer is gone, nothing answers on its port.
/// Each scan runs to its source's end and is discarded while refining. The test prints its time; a
/// scan-level test ends within 60 s on the board (the stage brief).

#include "camera/sequence_writer.hpp"
#include "check.hpp"
#include "scan/live_model.hpp"
#include "scan/object_class.hpp"
#include "scan/scan_interface.hpp"
#include "scan_comparison.hpp"
#include "stage_frames.hpp"
#include "test_files.hpp"
#include "types/little_endian.hpp"
#include "types/model_point.hpp"
#include "web/messages.hpp"
#include "web/web_server.hpp"
#include "web_test_helpers.hpp"

#include <atomic>
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

#include <Eigen/Geometry>
#include <httplib.h>

namespace
{
using gca_test::check;
using gca_test::paced_source;
using gca_test::read_state;
using gca_test::state_fields;
using gca_test::state_keys;
using gca_test::temporary_directory;
using gca_test::test_set;
using gca_test::walk_options;
using steady = std::chrono::steady_clock;

/// The frames each scan of the test takes before its source ends: 60 [?: a choice]. The scan
/// thread renders them on the host and takes about 5.4 s for them, 11 live models of up to 2.3 MB,
/// which fill the buffers of a browser that stops reading within the first two
/// [M 2026-11-17: this test's lines].
constexpr std::size_t scan_frames = 60;

/// The room the scanning of a scan with browsers has beyond that of the same scan without any:
/// 2 s [?: a choice] for the board's load, below the 5 s a send waits for a browser that stopped
/// reading (the library's write timeout [S: httplib.h, CPPHTTPLIB_SERVER_WRITE_TIMEOUT_SECOND]),
/// which a scan held up by such a browser would add.
constexpr std::chrono::seconds scanning_room{2};

/// The frames taken when the listener holds the scan thread for the late browser: 30
/// [?: a choice], the middle of the scan.
constexpr std::uint64_t late_after_frames = 30;

/// How long the listener holds the scan thread for the late browser: 2 s [?: a choice, far above
/// a connection and a live model's copy on the loopback interface]; after it the scan goes on and
/// the test fails.
constexpr std::chrono::seconds hold_limit{2};

/// How long the test waits for a scan's phase or a browser's message: 30 s, within the 60 s of a
/// scan-level test (the stage brief) [?: a choice].
constexpr std::chrono::seconds wait_limit{30};

/// How long the stopped browser, reading again, waits for the newest live model or its close: 10 s
/// [?: a choice], above the library's 5 s write timeout, after which the web layer closes it.
constexpr std::chrono::seconds catch_up_limit{10};

/// How long the web layer may take to end with a browser connected: 1 s [?: a choice, well under
/// the library's 5 s write timeout; the loop sees the end within its 20 ms poll].
constexpr std::chrono::seconds end_limit{1};

/// The receive buffer the stopped browser asks for: 4096 bytes [?: a choice], so that the web
/// layer's sends to it are held up once the server's own send buffer is full; the kernel doubles
/// it [S: socket(7), SO_RCVBUF].
constexpr int stopped_receive_buffer = 4096;

/// How long a browser's reading thread waits for a message before it looks at its stop: 100 ms
/// [?: a choice].
constexpr std::chrono::milliseconds browser_read_timeout{100};

/// The phases a browser connected before a scan sees, each once where it repeats: the idle one
/// before the start, then the scan's, ended by a discard while refining.
const std::vector<std::string> scan_phases = {"idle", "starting", "scanning", "refining", "idle"};

/// A live model message as the test reads it back, by the document's offsets (docs/ROUTES.md,
/// "The live model").
struct live_fields
{
    std::uint64_t version = 0;        ///< bytes 0 to 7
    std::uint64_t frame = 0;          ///< bytes 8 to 15
    std::uint32_t has_frame = 2;      ///< bytes 16 to 19, 1 or 0
    std::uint32_t n = 0;              ///< bytes 20 to 23, the points
    double pose[7] = {};              ///< bytes 24 to 79: tx, ty, tz, qx, qy, qz, qw
    std::vector<float> positions;     ///< from byte 80, x, y, z of each point
    std::vector<std::uint8_t> colors; ///< from byte 80 + 12 n, r, g, b of each point
};

/// Reads a live model message by the document's offsets, little-endian.
/// @param message [in] the message's bytes.
/// @return its fields; nothing when it is shorter than its header or its length is not
///         80 + 15 n bytes.
std::optional<live_fields> read_live_model(const std::string &message)
{
    if (message.size() < 80)
        return std::nullopt;
    const auto *const bytes = reinterpret_cast<const std::uint8_t *>(message.data());
    live_fields read;
    read.version = gca::get_little_endian<std::uint64_t>(bytes);
    read.frame = gca::get_little_endian<std::uint64_t>(bytes + 8);
    read.has_frame = gca::get_little_endian<std::uint32_t>(bytes + 16);
    read.n = gca::get_little_endian<std::uint32_t>(bytes + 20);
    for (std::size_t k = 0; k < 7; k++)
        read.pose[k] = gca::get_double(bytes + 24 + 8 * k);
    const std::size_t n = read.n;
    if (message.size() != 80 + 15 * n)
        return std::nullopt;
    read.positions.resize(3 * n);
    for (std::size_t k = 0; k < 3 * n; k++)
        read.positions[k] = gca::get_float(bytes + 80 + 4 * k);
    read.colors.assign(bytes + 80 + 12 * n, bytes + 80 + 15 * n);
    return read;
}

/// Says whether a message read back holds a live model's values, bit for bit.
/// @param read [in] the message read back.
/// @param model [in] the live model it was laid out from.
/// @return whether the version, the frame, the pose and every point's position and color are the
///         live model's, in its order, and has_frame says whether it has a frame.
bool holds(const live_fields &read, const gca::live_model &model)
{
    const Eigen::Vector3d &t = model.pose.position_m;
    const Eigen::Quaterniond &q = model.pose.orientation;
    const double pose[7] = {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()};
    bool same = read.version == model.version && read.frame == model.frame.value_or(0) &&
                read.has_frame == (model.frame ? 1u : 0u) && read.n == model.points.size();
    for (std::size_t k = 0; k < 7; k++)
        same = same && gca_test::same_bits(read.pose[k], pose[k]);
    for (std::size_t i = 0; same && i < model.points.size(); i++)
    {
        const gca::model_point &p = model.points[i];
        for (std::size_t c = 0; c < 3; c++)
            same = same && gca_test::same_bits(read.positions[3 * i + c], p.position_m[c]) &&
                   read.colors[3 * i + c] == p.color[c];
    }
    return same;
}

/// The live model's layout: a live model of known values, with a frame and three points, and the
/// empty one a scan starts with, laid out and read back by the document's offsets.
void check_layout()
{
    gca::live_model model;
    model.version = 7;
    model.frame = 42;
    // The trajectory's example pose (docs/SCAN_FILES.md), a unit quaternion.
    model.pose = {Eigen::Vector3d(0.5, -1.25, 2.0), Eigen::Quaterniond(0.88, 0.2, 0.4, 0.16)};
    model.spacing = 2;
    model.points = {
        {Eigen::Vector3f(0.125f, -0.25f, 0.375f), Eigen::Vector3f(0, 0, -1), {1, 2, 3}},
        {Eigen::Vector3f(1.5f, 2.25f, -3.0f), Eigen::Vector3f(1, 0, 0), {255, 0, 128}},
        {Eigen::Vector3f(-0.0f, 1e-3f, 0.6f), Eigen::Vector3f(0, 1, 0), {10, 20, 30}},
    };
    const std::string message = gca::live_model_message(model);
    const std::optional<live_fields> read = read_live_model(message);
    check(message.size() == 80 + 15 * 3 && read && read->has_frame == 1 && read->n == 3 &&
              holds(*read, model),
          "layout: a live model of three points read back by the document's offsets gives its "
          "values, 125 bytes");
    check(read && read->version == 7 && read->frame == 42 && read->pose[0] == 0.5 &&
              read->pose[3] == 0.2 && read->pose[6] == 0.88 && read->positions[3] == 1.5f &&
              read->colors[3] == 255 && read->colors[5] == 128,
          "layout: the version, the frame, tx, qx, qw, the second point's x and its color where "
          "the document puts them");

    gca::live_model empty;
    empty.version = 3;
    const std::string none = gca::live_model_message(empty);
    const std::optional<live_fields> read_none = read_live_model(none);
    check(none.size() == 80 && read_none && read_none->has_frame == 0 && read_none->frame == 0 &&
              read_none->n == 0 && holds(*read_none, empty) && read_none->pose[6] == 1.0,
          "layout: the empty live model, no frame and no point, 80 bytes, the identity pose");
}

/// A browser of the test: a WebSocket client of the web layer's /api/live; once it reads, it keeps
/// every message it is sent with the time it came, on a reading thread of its own.
class browser
{
public:
    /// A message as the browser got it.
    struct message
    {
        bool binary = false;     ///< a live model; else a state
        std::string bytes;       ///< the message
        steady::time_point came; ///< when it was read
    };

    /// Connects to the web layer's WebSocket; reads nothing until start_reading().
    /// @param port [in] the web layer's port on the loopback interface.
    /// @param receive_buffer [in] the receive buffer to ask for, in bytes, before the connection;
    ///        nothing for the system's own.
    explicit browser(std::uint16_t port, std::optional<int> receive_buffer = std::nullopt)
        : m_client("ws://127.0.0.1:" + std::to_string(port) + "/api/live")
    {
        if (receive_buffer)
            m_client.set_socket_options([size = *receive_buffer](socket_t sock) {
                httplib::set_socket_opt(sock, SOL_SOCKET, SO_RCVBUF, size);
            });
        // The handshake's answer waits at most 5 s [?: a choice]; the reading's timeout after it.
        m_client.set_read_timeout(std::chrono::seconds(5));
        m_connected = static_cast<bool>(m_client.connect());
        m_client.set_read_timeout(browser_read_timeout);
        m_connected_at = steady::now();
    }

    /// A browser stands for one connection and its reading thread.
    browser(const browser &) = delete;
    browser &operator=(const browser &) = delete;

    /// Stops the reading thread, then closes the connection with the closing handshake.
    ~browser()
    {
        m_stop.store(true);
        if (m_reader.joinable())
            m_reader.join();
    }

    /// Says whether the connection was made.
    /// @return whether it was.
    bool connected() const
    {
        return m_connected;
    }

    /// Gives when the connection was made.
    /// @return the time.
    steady::time_point connected_at() const
    {
        return m_connected_at;
    }

    /// Reads on the calling thread until the browser got its first state and live model, the
    /// connection ended or the wait's limit passed; before start_reading(), so that a browser that
    /// stops reading afterwards stops at a known point.
    /// @return whether it got both.
    bool read_first()
    {
        const auto both = [](const std::vector<message> &m) {
            bool state = false;
            bool live_model = false;
            for (const message &each : m)
                (each.binary ? live_model : state) = true;
            return state && live_model;
        };
        const steady::time_point limit = steady::now() + wait_limit;
        while (!both(messages()) && steady::now() < limit && read_one())
        {
        }
        return both(messages());
    }

    /// Starts the reading thread, which reads until the browser's end or the connection's.
    void start_reading()
    {
        m_reader = std::thread([this] {
            while (!m_stop.load() && read_one())
            {
            }
        });
    }

    /// Waits until the messages so far meet a condition, the connection ended or a limit passed.
    /// @param until [in] the condition.
    /// @param limit [in] the limit.
    /// @return whether they met it.
    bool wait_for(const std::function<bool(const std::vector<message> &)> &until,
                  steady::duration limit)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_came.wait_for(lock, limit, [&] { return m_closed || until(m_messages); });
        return until(m_messages);
    }

    /// Gives the messages so far.
    /// @return a copy.
    std::vector<message> messages() const
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        return m_messages;
    }

    /// Waits until the connection ended under the reading thread or a limit passed.
    /// @param limit [in] the limit.
    /// @return whether it ended.
    bool wait_closed(steady::duration limit)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_came.wait_for(lock, limit, [this] { return m_closed; });
    }

    /// Says whether the connection ended under the reading thread: the web layer closed it.
    /// @return whether it did.
    bool closed() const
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        return m_closed;
    }

private:
    /// Reads one message and keeps it, or waits for one up to the reading's timeout; one thread at
    /// a time.
    /// @return false when the connection ended; true otherwise.
    bool read_one()
    {
        std::string bytes;
        const httplib::ws::ReadResult r = m_client.read(bytes);
        if (r == httplib::ws::Timeout)
            return true;
        {
            const std::lock_guard<std::mutex> lock(m_mutex);
            if (r == httplib::ws::Fail)
                m_closed = true;
            else
                m_messages.push_back({r == httplib::ws::Binary, std::move(bytes), steady::now()});
        }
        m_came.notify_all();
        return r != httplib::ws::Fail;
    }

    httplib::ws::WebSocketClient m_client;
    bool m_connected = false;
    steady::time_point m_connected_at;
    mutable std::mutex m_mutex;     ///< guards the messages and the end
    std::condition_variable m_came; ///< wakes a wait at each message and at the end
    std::vector<message> m_messages;
    bool m_closed = false;
    std::atomic<bool> m_stop{false};
    std::thread m_reader; ///< started last, joined first
};

/// Gives the messages of one kind.
/// @param messages [in] a browser's messages.
/// @param binary [in] true for the live models, false for the states.
/// @return their bytes, in the order they came.
std::vector<std::string> of_kind(const std::vector<browser::message> &messages, bool binary)
{
    std::vector<std::string> kind;
    for (const browser::message &m : messages)
        if (m.binary == binary)
            kind.push_back(m.bytes);
    return kind;
}

/// Says whether a browser got a state of a phase.
/// @param messages [in] its messages.
/// @param phase [in] the phase.
/// @return whether it did.
bool got_phase(const std::vector<browser::message> &messages, const std::string &phase)
{
    for (const std::string &json : of_kind(messages, false))
    {
        const std::optional<state_fields> s = read_state(json);
        if (s && s->phase == phase)
            return true;
    }
    return false;
}

/// Says whether a browser got a message of one kind equal to the bytes given.
/// @param messages [in] its messages.
/// @param binary [in] the kind: true for a live model.
/// @param bytes [in] the bytes.
/// @return whether it did.
bool got(const std::vector<browser::message> &messages, bool binary, const std::string &bytes)
{
    for (const browser::message &m : messages)
        if (m.binary == binary && m.bytes == bytes)
            return true;
    return false;
}

/// What a browser's states show.
struct states_seen
{
    bool readable = true;            ///< every state read back with the document's keys in order
    bool rising = true;              ///< the versions rise strictly
    std::vector<std::string> phases; ///< the phases, each once where it repeats
};

/// Reads a browser's states back.
/// @param messages [in] its messages.
/// @return what they show.
states_seen states_of(const std::vector<browser::message> &messages)
{
    states_seen seen;
    int last = -1;
    for (const std::string &json : of_kind(messages, false))
    {
        const std::optional<state_fields> s = read_state(json);
        if (!s || s->keys != state_keys || s->frames_keys != gca_test::frames_keys)
        {
            seen.readable = false;
            continue;
        }
        seen.rising = seen.rising && s->version > last;
        last = s->version;
        if (seen.phases.empty() || seen.phases.back() != s->phase)
            seen.phases.push_back(s->phase);
    }
    return seen;
}

/// Says whether a browser's live models read back, each with versions rising strictly.
/// @param live_models [in] its live models' bytes.
/// @return whether they do.
bool live_models_rise(const std::vector<std::string> &live_models)
{
    std::optional<std::uint64_t> last;
    for (const std::string &bytes : live_models)
    {
        const std::optional<live_fields> read = read_live_model(bytes);
        if (!read || (last && read->version <= *last))
            return false;
        last = read->version;
    }
    return true;
}

/// The test's watch of a scan, on the scan thread: when its scanning ended.
struct scan_watch
{
    std::mutex mutex;                                 ///< guards the rest
    std::optional<steady::time_point> starting;       ///< when it was first starting
    std::optional<steady::time_point> scanning;       ///< when it was first scanning
    std::optional<steady::time_point> scanning_ended; ///< when it first left scanning

    /// The listener's part.
    /// @param s [in] the state published.
    void on_state(const gca::scan_state &s)
    {
        const std::lock_guard<std::mutex> lock(mutex);
        const steady::time_point now = steady::now();
        if (s.phase == gca::scan_phase::starting && !starting)
            starting = now;
        else if (s.phase == gca::scan_phase::scanning && !scanning)
            scanning = now;
        else if (s.phase != gca::scan_phase::scanning && scanning && !scanning_ended)
            scanning_ended = now;
    }

    /// Forgets the last scan, before the next one starts.
    void reset()
    {
        const std::lock_guard<std::mutex> lock(mutex);
        starting.reset();
        scanning.reset();
        scanning_ended.reset();
    }

    /// Gives how long the scan was starting, which a browser polling every 20 ms sees only when it
    /// lasts longer.
    /// @return from the starting state to the scanning one; nothing before it scanned.
    std::optional<steady::duration> starting_lasted()
    {
        const std::lock_guard<std::mutex> lock(mutex);
        if (!starting || !scanning)
            return std::nullopt;
        return *scanning - *starting;
    }

    /// Gives when the scanning ended.
    /// @return the time; nothing while it scans.
    std::optional<steady::time_point> ended()
    {
        const std::lock_guard<std::mutex> lock(mutex);
        return scanning_ended;
    }
};

/// The hold in the middle of a scan: the scan interface's listener holds the scan thread at the
/// first state with late_after_frames frames taken, after the web layer's news counted it, until
/// the test's late browser has got the current state and live model or the hold's limit has
/// passed. So nothing changes while the late browser connects, and what it gets came at its
/// connection, not at a change. The listener's contract asks it not to block (scan_interface.hpp,
/// on_state); this test blocks on purpose, and it cannot deadlock: the state and the live model are
/// published, each under its own lock, before the listener runs; the web layer's loops and the
/// test take only those locks (state(), latest_live_model()), which the scan thread does not hold
/// while it waits here, and the web layer's own messages' lock, which the scan thread never takes;
/// no command is given while it waits; and the hold's limit ends the wait whatever happens.
struct middle_hold
{
    std::mutex mutex;             ///< guards the rest
    std::condition_variable wake; ///< wakes the test when held, the listener when released
    bool armed = false;           ///< whether the scan in progress holds
    bool held = false;            ///< the listener held the scan thread
    bool released = false;        ///< the test let it go on
    bool timed_out = false;       ///< the hold's limit passed before the test let it go on

    /// The listener's part: holds once, at the first state with late_after_frames frames taken.
    /// @param s [in] the state published.
    void on_state(const gca::scan_state &s)
    {
        if (s.phase != gca::scan_phase::scanning || s.frames.taken < late_after_frames)
            return;
        std::unique_lock<std::mutex> lock(mutex);
        if (!armed || held)
            return;
        held = true;
        wake.notify_all();
        timed_out = !wake.wait_for(lock, hold_limit, [this] { return released; });
    }

    /// The test's part: waits until the listener holds.
    /// @return whether it held within the wait's limit.
    bool wait_held()
    {
        std::unique_lock<std::mutex> lock(mutex);
        return wake.wait_for(lock, wait_limit, [this] { return held; });
    }

    /// The test's part: lets the scan thread go on.
    void release()
    {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            released = true;
        }
        wake.notify_all();
    }
};

/// The test's scan interface and its web layer's port, with the watch and the hold its listeners
/// call; each the run's own.
struct scanner_under_test
{
    scan_watch &watch;            ///< the watch of the scan's scanning
    middle_hold &hold;            ///< the hold in the middle of a scan
    gca::scan_interface &scanner; ///< the scan interface
    std::uint16_t port = 0;       ///< the web layer's port on the loopback interface
    /// How long the scanning of a scan without any browser took, the bound's base; zero until
    /// measured.
    steady::duration reference{};
};

/// Starts a scan and checks that it scans.
/// @param t [in,out] the scanner under test.
/// @param what [in] the case, named in the check line.
/// @return when the start was answered; nothing, with a failed check, when it was refused.
std::optional<steady::time_point> start_scan(scanner_under_test &t, const std::string &what)
{
    t.watch.reset();
    const gca::command_reply reply =
        t.scanner
            .start(gca::object_class::tabletop_object, test_set(gca::object_class::tabletop_object))
            .get();
    check(reply.accepted, (what + ": the scan started; " + reply.refusal).c_str());
    if (!reply.accepted)
        return std::nullopt;
    return steady::now();
}

/// Waits until the scan's scanning ended at its source's end.
/// @param t [in,out] the scanner under test.
/// @param started [in] when the start was answered.
/// @param what [in] the case, named in the lines printed.
/// @return how long the scanning took from the start's answer; nothing, with a failed check, when
///         it did not end within the wait's limit.
std::optional<steady::duration> scanning_time(scanner_under_test &t, steady::time_point started,
                                              const std::string &what)
{
    const steady::time_point limit = steady::now() + wait_limit;
    std::optional<steady::time_point> ended;
    while (!(ended = t.watch.ended()) && steady::now() < limit)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    check(ended.has_value(), (what + ": the scanning ended at the source's end").c_str());
    if (!ended)
        return std::nullopt;
    const std::optional<steady::duration> starting = t.watch.starting_lasted();
    std::printf("%s: starting lasted %.3f s; the scanning took %.3f s from the start's answer\n",
                what.c_str(), starting ? std::chrono::duration<double>(*starting).count() : -1.0,
                std::chrono::duration<double>(*ended - started).count());
    return *ended - started;
}

/// Waits until the scan's scanning ended, then checks its time against the bound: the scanning time
/// of the scan without any browser and scanning_room.
/// @param t [in,out] the scanner under test, its reference measured.
/// @param started [in] when the start was answered.
/// @param what [in] the case, named in the check line.
void check_scanning_time(scanner_under_test &t, steady::time_point started, const std::string &what)
{
    const std::optional<steady::duration> took = scanning_time(t, started, what);
    const steady::duration bound = t.reference + scanning_room;
    std::printf("%s: the bound %.3f s\n", what.c_str(),
                std::chrono::duration<double>(bound).count());
    check(took && t.reference > steady::duration::zero() && *took <= bound,
          (what + ": the scanning ended within that of the scan without any browser and " +
           std::to_string(scanning_room.count()) + " s")
              .c_str());
}

/// Discards the scan once it is refining and each browser got its refining state, and waits until
/// it is idle again and each browser got the idle state of the scan discarded.
/// @param t [in,out] the scanner under test.
/// @param browsers [in,out] the reading browsers; none for a scan without any.
/// @param what [in] the case, named in the check lines.
void discard_while_refining(scanner_under_test &t, const std::vector<browser *> &browsers,
                            const std::string &what)
{
    const auto phase_reached = [&t](gca::scan_phase phase) {
        const steady::time_point limit = steady::now() + wait_limit;
        while (t.scanner.state().phase != phase && steady::now() < limit)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return t.scanner.state().phase == phase;
    };
    check(phase_reached(gca::scan_phase::refining), (what + ": the scan refining").c_str());
    for (browser *b : browsers)
        check(b->wait_for([](const auto &m) { return got_phase(m, "refining"); }, wait_limit),
              (what + ": a browser got the refining state").c_str());
    const gca::command_reply reply = t.scanner.discard().get();
    check(reply.accepted, (what + ": the discard while refining; " + reply.refusal).c_str());
    check(phase_reached(gca::scan_phase::idle) &&
              t.scanner.state().outcome == gca::scan_outcome::discarded,
          (what + ": the scan discarded").c_str());
    // The last state published, which each browser gets; an idle state of an earlier scan
    // discarded, which a browser may have got at its connection, is not it.
    const std::string last = gca::state_message(t.scanner.state());
    for (browser *b : browsers)
        check(b->wait_for([&last](const auto &m) { return got(m, false, last); }, wait_limit),
              (what + ": a browser got the idle state, the scan discarded").c_str());
}

/// The scan without any browser, whose scanning time is the bound's base.
/// @param t [in,out] the scanner under test; its reference set.
void measure_without_browsers(scanner_under_test &t)
{
    const std::string what = "no browser";
    const std::optional<steady::time_point> started = start_scan(t, what);
    if (!started)
        return;
    const std::optional<steady::duration> took = scanning_time(t, *started, what);
    if (took)
        t.reference = *took;
    discard_while_refining(t, {}, what);
}

/// Two browsers connected before a scan and a third in its middle.
/// @param t [in,out] the scanner under test.
void check_two_browsers_and_a_late_one(scanner_under_test &t)
{
    const std::string what = "two browsers";
    browser first(t.port);
    browser second(t.port);
    check(first.connected() && second.connected() && first.read_first() && second.read_first(),
          "two browsers: both connected and got the idle state and live model");
    first.start_reading();
    second.start_reading();
    {
        const std::lock_guard<std::mutex> lock(t.hold.mutex);
        t.hold.armed = true;
    }
    const std::optional<steady::time_point> started = start_scan(t, what);
    if (!started)
        return;

    check(t.hold.wait_held(), "late browser: the scan thread held in the middle of the scan");
    // The scan thread waits in the listener: the current state and live model stay as they are.
    const std::string current_state = gca::state_message(t.scanner.state());
    const gca::live_model current_live = t.scanner.latest_live_model();
    const std::string current_model = gca::live_model_message(current_live);
    {
        browser late(t.port);
        late.start_reading();
        const bool both = late.wait_for(
            [&](const auto &m) {
                return got(m, false, current_state) && got(m, true, current_model);
            },
            hold_limit);
        t.hold.release();
        const std::vector<browser::message> m = late.messages();
        const double first_after =
            m.empty() ? -1.0
                      : std::chrono::duration<double>(m.front().came - late.connected_at()).count();
        std::printf("late browser: connected at frame >= %llu, live model version %llu of %zu "
                    "points; its first message %.4f s after the connection\n",
                    static_cast<unsigned long long>(late_after_frames),
                    static_cast<unsigned long long>(current_live.version),
                    current_live.points.size(), first_after);
        {
            const std::lock_guard<std::mutex> lock(t.hold.mutex);
            check(late.connected() && both && !t.hold.timed_out && !m.empty() &&
                      !m.front().binary && m.front().bytes == current_state,
                  "late browser: connected in the middle of the scan, it got the current state "
                  "first and the current live model, while nothing changed");
            t.hold.armed = false;
        }
    }

    check_scanning_time(t, *started, what);
    discard_while_refining(t, {&first, &second}, what);

    const std::vector<browser::message> a = first.messages();
    const std::vector<browser::message> b = second.messages();
    const states_seen sa = states_of(a);
    const states_seen sb = states_of(b);
    const std::vector<std::string> la = of_kind(a, true);
    const std::vector<std::string> lb = of_kind(b, true);
    std::string phases;
    for (const std::string &p : sa.phases)
        phases += " " + p;
    std::printf("two browsers: %zu and %zu states, %zu and %zu live models; the phases:%s\n",
                of_kind(a, false).size(), of_kind(b, false).size(), la.size(), lb.size(),
                phases.c_str());
    for (const std::string &bytes : la)
    {
        const std::optional<live_fields> read = read_live_model(bytes);
        std::printf("two browsers: live model version %llu, %u points, %zu bytes\n",
                    static_cast<unsigned long long>(read ? read->version : 0), read ? read->n : 0,
                    bytes.size());
    }
    check(sa.readable && sb.readable && sa.rising && sb.rising,
          "two browsers: every state read back with the document's keys, the versions rising");
    check(sa.phases == scan_phases && sb.phases == scan_phases,
          "two browsers: both saw idle, starting, scanning, refining, idle, in that order");
    check(!of_kind(a, false).empty() && !of_kind(b, false).empty() &&
              of_kind(a, false).back() == of_kind(b, false).back(),
          "two browsers: the same last state");
    check(la == lb && la.size() >= 3 && live_models_rise(la),
          "two browsers: the same live models byte for byte, at least the idle one, the scan's "
          "empty one and one taken, the versions rising");
    const gca::live_model last = t.scanner.latest_live_model();
    const std::optional<live_fields> read = la.empty() ? std::nullopt : read_live_model(la.back());
    check(!la.empty() && la.back() == gca::live_model_message(last) && read && holds(*read, last) &&
              read->has_frame == 1 && read->n > 0,
          "two browsers: the last live model the scan interface's own, read back to its values");
}

/// A browser that stops reading during a scan, beside one that reads.
/// @param t [in,out] the scanner under test.
void check_a_browser_that_stops_reading(scanner_under_test &t)
{
    const std::string what = "stopped browser";
    browser reading(t.port);
    browser stopped(t.port, stopped_receive_buffer);
    check(reading.connected() && stopped.connected() && reading.read_first() &&
              stopped.read_first(),
          "stopped browser: both connected and got the idle state and live model; one reads no "
          "more until the scan's end");
    reading.start_reading();
    const std::optional<steady::time_point> started = start_scan(t, what);
    if (!started)
        return;
    check_scanning_time(t, *started, what);
    discard_while_refining(t, {&reading}, what);

    const std::vector<std::string> newest = of_kind(reading.messages(), true);
    const gca::live_model last = t.scanner.latest_live_model();
    check(!newest.empty() && newest.back() == gca::live_model_message(last),
          "stopped browser: the reading browser got the scan's last live model");
    stopped.start_reading();
    stopped.wait_for(
        [&](const auto &m) {
            const std::vector<std::string> got_live = of_kind(m, true);
            return !newest.empty() && !got_live.empty() && got_live.back() == newest.back();
        },
        catch_up_limit);
    const std::vector<std::string> caught = of_kind(stopped.messages(), true);
    const bool closed = stopped.closed();
    std::printf("stopped browser: %zu live models against %zu of the reading browser; %s\n",
                caught.size(), newest.size(),
                closed ? "the web layer closed it" : "it got the newest when it read again");
    check(caught.size() < newest.size(),
          "stopped browser: fewer live models than the reading browser, so its sends were held "
          "up (else the case is not exercised)");
    check(live_models_rise(caught) &&
              (closed || (!caught.empty() && caught.back() == newest.back())),
          "stopped browser: reading again, it got the newest live model, the older ones dropped, "
          "or the web layer closed it");
}

/// The web layer's end with a browser connected: the destructor ends the browser's loop, which
/// nothing but the web layer's end ends while the browser stays (the library's stop() closes only
/// the listening socket), returns within end_limit, and the browser's connection is closed. When it
/// does not return in time, the browser closes its connection, which ends its loop, so that the
/// test goes on.
/// @param server [in,out] the web layer, started; ended here.
/// @param port [in] its port on the loopback interface.
void check_end_with_a_browser(std::optional<gca::web_server> &server, std::uint16_t port)
{
    std::optional<browser> staying;
    staying.emplace(port);
    check(staying->connected() && staying->read_first(),
          "the end: a browser connected and got the state and live model");
    staying->start_reading();
    std::mutex mutex;
    std::condition_variable ended;
    std::optional<steady::time_point> end_at;
    const steady::time_point begin = steady::now();
    std::thread ender([&] {
        server.reset();
        {
            const std::lock_guard<std::mutex> lock(mutex);
            end_at = steady::now();
        }
        ended.notify_all();
    });
    bool in_time = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        in_time = ended.wait_for(lock, end_limit, [&] { return end_at.has_value(); });
    }
    if (!in_time)
        staying.reset();
    ender.join();
    std::printf("the end: the web layer ended %.4f s after its destructor began, a browser "
                "connected\n",
                std::chrono::duration<double>(*end_at - begin).count());
    check(in_time, ("the end: the web layer ended within " + std::to_string(end_limit.count()) +
                    " s with a browser connected")
                       .c_str());
    check(staying && staying->wait_closed(end_limit),
          "the end: the browser's connection closed by the web layer's end");
}

/// Runs the cases over a scan interface and its web layer, then ends them.
/// @param directory [in] the test's directory.
void run_websocket(const std::filesystem::path &directory)
{
    const std::filesystem::path scans = directory / "scans";
    const std::filesystem::path page = directory / "page";
    std::filesystem::create_directory(scans);
    std::filesystem::create_directory(page);
    paced_source source(walk_options(scan_frames));
    gca::sequence_writer writer(gca::queue_full::wait);
    // The news first, as the service builds it, so that it outlives the scan interface; the test's
    // scan_watch and middle_hold, which the listeners call after the news, live as long.
    gca::scan_news news;
    scan_watch watch;
    middle_hold hold;
    const gca::scan_options options = {
        .scans_directory = scans,
        .stream_size = gca_test::small_description.size,
        .commit = gca_test::in_process_commit,
        .on_state =
            [&](const gca::scan_state &s) {
                news.state_published();
                watch.on_state(s);
                hold.on_state(s);
            },
        .stage_times = nullptr,
        .on_live_model = [&news](const gca::live_model &) { news.live_model_published(); }};
    gca::scan_interface scanner(options, source, writer);
    std::optional<gca::web_server> server;
    server.emplace(
        scanner, news,
        gca::web_options{
            .address = "127.0.0.1", .port = 0, .page_directory = page, .set_of = test_set});
    const gca::web_start_result started = server->start();
    check(started.started && started.port != 0,
          ("the web layer listens on a free port; " + started.reason).c_str());
    if (!started.started)
        return;
    const std::uint16_t port = started.port;
    std::printf("the web layer listens on 127.0.0.1 port %u\n", unsigned{port});
    scanner_under_test t{.watch = watch, .hold = hold, .scanner = scanner, .port = port};
    measure_without_browsers(t);
    check_two_browsers_and_a_late_one(t);
    check_a_browser_that_stops_reading(t);
    check_end_with_a_browser(server, port);
    httplib::Client after("127.0.0.1", port);
    check(!after.Get("/api/state"), "the web layer gone: nothing answers on its port");
    check(gca_test::names_in(scans).empty(), "the scans discarded: nothing on disk");
}
} // namespace

int main()
{
    const auto begin = steady::now();
    check_layout();
    const temporary_directory directory("test_websocket");
    run_websocket(directory.path());
    std::printf("test_websocket took %.1f s\n",
                std::chrono::duration<double>(steady::now() - begin).count());
    return gca_test::check_summary("test_websocket");
}
