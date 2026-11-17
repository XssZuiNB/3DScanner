// SPDX-License-Identifier: 0BSD
/// @file test_scanner_service.cpp
/// The scanner service's parts (apps/service.hpp; docs/SERVICE.md) driven in process as the
/// program drives them: the stop signals made the stop before any thread starts, then serve() on
/// this thread until a stop signal, which a driver thread sends to this process with kill(). The
/// camera is the tests' stand-in (stand_in_camera.hpp), its switches on at first as the camera's
/// are [M 2026-11-13, docs/HARDWARE.md], in front of the synthetic source's tabletop box along the
/// web tests' walk at the camera's rate (web_test_helpers.hpp); the page served on a free port of
/// the loopback interface from a temporary page directory, the scans into a temporary scans
/// directory, the sequence not kept, the web tests' set in a sparse TSDF of 273 MB. The program
/// scan_in_progress, given as the first argument, is run against it through the shell.
/// - Run A, two scans in one service. Scan 1 with the stand-in refusing every read of exposure
///   (2001): scan_in_progress says no while idle (exit 1), yes while starting, scanning and
///   refining (exit 0; the log's sink holds the scan thread at the starting and the refining line
///   until it was asked), no when idle again; the state names each refusal while scanning; the
///   operator's stop refines the scan, whose parameters name exposure and whose sequence is
///   removed. Scan 2, nothing refused: the hold reads the five at its start again, and its state
///   and parameters name nothing, none of scan 1's refusals carried over. SIGINT ends the
///   service, the log's lines as docs/SERVICE.md gives them.
/// - Run B, nothing refused: SIGTERM in the middle of a scan, after the hold: the scan stays on
///   disk unfinished, end shut_down, with its sequence and parameters and without a full model;
///   the stand-in is closed, its switches written off at the hold and on again at the end, so both
///   are on as before; the log says so; once the service is gone nothing answers on its port and
///   scan_in_progress cannot tell (exit 3).
/// - Run C: SIGHUP while idle ends the service.
/// - scan_in_progress refuses an unknown argument with its usage (exit 2).
/// - The log's lines of a live model that could not be taken and a live view that could not be
///   made, from states given to the log's listener: each once when its failure changes to a reason,
///   none when it stays or is cleared.
/// The test prints its time; ctest ends it after 120 s [?: a choice, twice a scan-level test's 60 s
/// (the stage brief)], so that a stop that never ends fails it.

#include "camera/sequence_format.hpp"
#include "check.hpp"
#include "program_run.hpp"
#include "scan/object_class.hpp"
#include "scan/parameters_file.hpp"
#include "scan_comparison.hpp"
#include "service.hpp"
#include "service_options.hpp"
#include "stage_frames.hpp"
#include "stand_in_camera.hpp"
#include "test_files.hpp"
#include "web_test_helpers.hpp"

#include <chrono>
#include <condition_variable>
#include <csignal>
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
#include <unistd.h>

namespace
{
using gca_test::check;
using gca_test::paced_source;
using gca_test::read_state;
using gca_test::run_program;
using gca_test::stand_in_camera;
using gca_test::state_fields;

/// How long the test waits for a line, a state or the hold: 30 s, within a scan-level test's 60 s
/// (the stage brief) [?: a choice].
constexpr std::chrono::seconds wait_limit{30};

/// How long the log's sink holds the scan thread at the refining line: 10 s [?: a choice, far
/// above one run of scan_in_progress].
constexpr std::chrono::seconds hold_limit{10};

/// The frames scan 1 has taken when the operator stops it: 40, past the hold's 30
/// (camera/color_hold.hpp), so that white balance was held [?: a choice].
constexpr std::uint64_t frames_before_stop = 40;

/// The frames scan 2 has taken when the service is asked to stop: 45, past the hold's 30
/// [?: a choice].
constexpr std::uint64_t frames_before_signal = 45;

/// The service's log as the test reads it: each line kept and printed. Armed with a text, the sink
/// holds the thread that logs the next line holding it, the scan thread, until the test released
/// it or the hold's limit passed, so that the test sees the scan starting or refining however short
/// that phase is. The listener's contract asks it not to block (scan/scan_interface.hpp,
/// on_state); this test blocks on purpose, and it cannot deadlock: the state is published, under
/// the state's lock, before the listener runs, and the state route takes only that lock, which the
/// scan thread does not hold while it waits here; the one command given while it waits is a start
/// whose answer the test waits for on a thread of its own, and the main thread logs nothing then.
class recorded_log
{
public:
    /// Takes a line: keeps it, prints it, and holds at a line that holds the armed text.
    /// @param line [in] the line.
    void sink(const std::string &line)
    {
        std::printf("log: %s\n", line.c_str());
        std::unique_lock<std::mutex> lock(m_mutex);
        m_lines.push_back(line);
        m_changed.notify_all();
        if (m_marker.empty() || m_held || line.find(m_marker) == std::string::npos)
            return;
        m_held = true;
        m_changed.notify_all();
        if (!m_changed.wait_for(lock, hold_limit, [this] { return m_released; }))
            m_timed_out = true;
    }

    /// Gives the lines so far.
    /// @return a copy of them.
    std::vector<std::string> lines() const
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        return m_lines;
    }

    /// Gives how many lines came so far.
    /// @return the count.
    std::size_t count() const
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        return m_lines.size();
    }

    /// Waits for a line that starts with a text, among the lines from an index on.
    /// @param start [in] the text.
    /// @param from [in] the index of the first line looked at.
    /// @return the first such line; nothing when none came within the wait's limit.
    std::optional<std::string> wait_for(const std::string &start, std::size_t from)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        std::optional<std::string> found;
        m_changed.wait_for(lock, wait_limit, [&] {
            for (std::size_t k = from; k < m_lines.size(); k++)
            {
                if (m_lines[k].rfind(start, 0) == 0)
                {
                    found = m_lines[k];
                    return true;
                }
            }
            return false;
        });
        return found;
    }

    /// Arms the hold at the next line that holds a text.
    /// @param marker [in] the text, not empty.
    void arm(const std::string &marker)
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        m_marker = marker;
        m_held = false;
        m_released = false;
    }

    /// Waits until the sink holds the scan thread.
    /// @return whether it does within the wait's limit.
    bool wait_until_held()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_changed.wait_for(lock, wait_limit, [this] { return m_held; });
    }

    /// Lets the scan thread go on and disarms the hold.
    void release()
    {
        {
            const std::lock_guard<std::mutex> lock(m_mutex);
            m_released = true;
            m_marker.clear();
        }
        m_changed.notify_all();
    }

    /// Tells whether a hold's limit passed before its release.
    /// @return whether it did.
    bool timed_out() const
    {
        const std::lock_guard<std::mutex> lock(m_mutex);
        return m_timed_out;
    }

private:
    mutable std::mutex m_mutex;        ///< guards the rest
    std::condition_variable m_changed; ///< a line came, the hold began or was released
    std::vector<std::string> m_lines;  ///< every line, in order
    std::string m_marker;              ///< the text whose next line holds; empty: none
    bool m_held = false;               ///< whether the sink holds, or held, the scan thread
    bool m_released = false;           ///< whether the test released it
    bool m_timed_out = false;          ///< whether the hold's limit passed before the release
};

/// Finds the first line that starts with a text.
/// @param lines [in] the lines.
/// @param start [in] the text.
/// @return its index; the count of lines when none does.
std::size_t index_of(const std::vector<std::string> &lines, const std::string &start)
{
    std::size_t k = 0;
    while (k < lines.size() && lines[k].rfind(start, 0) != 0)
        k++;
    return k;
}

/// Tells whether the lines hold one that starts with a text.
/// @param lines [in] the lines.
/// @param start [in] the text.
/// @return whether one does.
bool has_line(const std::vector<std::string> &lines, const std::string &start)
{
    return index_of(lines, start) < lines.size();
}

/// Checks that the lines hold one that starts with a text, and says which on failure.
/// @param lines [in] the lines.
/// @param start [in] the text.
/// @param what [in] the case.
void check_line(const std::vector<std::string> &lines, const std::string &start,
                const std::string &what)
{
    check(has_line(lines, start), (what + ": the log has \"" + start + "\"").c_str());
}

/// What a test run of the service is given.
struct service_run
{
    gca::service_settings settings; ///< what the service runs with
    stand_in_camera &camera;        ///< the stand-in
    gca::service_log &log;          ///< the log, whose sink is the recorded log's
    recorded_log &recorded;         ///< the recorded log
    std::string in_progress;        ///< the path of scan_in_progress
};

/// Runs scan_in_progress against a port and checks its exit status and line.
/// @param run [in] the run, for the program's path.
/// @param port [in] the port asked.
/// @param status [in] the exit status expected.
/// @param line [in] the start of its output expected.
/// @param what [in] the case.
void check_in_progress(const service_run &run, std::uint16_t port, int status,
                       const std::string &line, const std::string &what)
{
    const gca_test::program_run asked =
        run_program(run.in_progress, {"--address", "127.0.0.1", "--port", std::to_string(port)});
    std::printf("%s: scan_in_progress exit %d: %s", what.c_str(), asked.exit_status,
                asked.output.c_str());
    check(asked.exit_status == status && asked.output.rfind(line, 0) == 0,
          (what + ": scan_in_progress exits " + std::to_string(status) + " with \"" + line + "\"")
              .c_str());
}

/// The driver's client of the service.
class client
{
public:
    /// Builds the client of a port of the loopback interface.
    /// @param port [in] the port.
    explicit client(std::uint16_t port)
        : m_http("127.0.0.1", port)
    {
    }

    /// Gives a command.
    /// @param target [in] the route with its query.
    /// @return whether it was answered with 200.
    bool command(const std::string &target)
    {
        const httplib::Result r = m_http.Post(target);
        return r && r->status == 200;
    }

    /// Polls the state until a condition holds or the wait's limit passes.
    /// @param until [in] the condition.
    /// @param what [in] the case, named in the check line.
    /// @return the state that met the condition; the last read, with a failed check, when none
    ///         did.
    state_fields poll(const std::function<bool(const state_fields &)> &until,
                      const std::string &what)
    {
        const auto begin = std::chrono::steady_clock::now();
        state_fields last;
        while (std::chrono::steady_clock::now() - begin < wait_limit)
        {
            const httplib::Result r = m_http.Get("/api/state");
            const std::optional<state_fields> s =
                r && r->status == 200 ? read_state(r->body) : std::nullopt;
            if (s)
            {
                last = *s;
                if (until(*s))
                    return last;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        check(false, (what + ": reached within the wait's limit").c_str());
        return last;
    }

private:
    httplib::Client m_http;
};

/// Waits for the service's serving line and reads its port from it.
/// @param recorded [in,out] the recorded log.
/// @param from [in] the index of the run's first line.
/// @param what [in] the case.
/// @return the port; nothing, with a failed check, when no serving line came.
std::optional<std::uint16_t> serving_port(recorded_log &recorded, std::size_t from,
                                          const std::string &what)
{
    const std::string start = "scanner_service: serving the page on http://127.0.0.1:";
    const std::optional<std::string> line = recorded.wait_for(start, from);
    check(line.has_value(), (what + ": the service serves").c_str());
    if (!line)
        return std::nullopt;
    const std::size_t end = line->find('/', start.size());
    return gca::port_of(line->substr(start.size(), end - start.size()));
}

/// Runs the service on this thread while a driver thread works it, then sends the stop signal
/// the driver asks for, when the service served.
/// @param run [in,out] the run.
/// @param drive [in] the driver's work on the port served; gives the signal to stop with.
/// @param what [in] the case.
/// @return the port served; nothing when the service did not serve.
std::optional<std::uint16_t> serve_while(service_run &run,
                                         const std::function<int(std::uint16_t)> &drive,
                                         const std::string &what)
{
    std::optional<std::uint16_t> port;
    const std::size_t from = run.recorded.count();
    std::thread driver([&] {
        port = serving_port(run.recorded, from, what);
        if (!port)
            return;
        const int signal_number = drive(*port);
        ::kill(::getpid(), signal_number);
    });
    const int status = gca::serve(run.settings, run.camera, run.camera, run.log);
    driver.join();
    check(status == 0, (what + ": serve() returned 0").c_str());
    return port;
}

/// Starts a scan while the log's sink holds the scan thread at the starting line, and checks that
/// scan_in_progress says yes then: the start's request, which is answered once the scan scans,
/// waits on a thread of its own.
/// @param run [in,out] the run.
/// @param port [in] the port served.
/// @param what [in] the case.
void start_held(service_run &run, std::uint16_t port, const std::string &what)
{
    run.recorded.arm("scanner_service: a scan starts");
    bool started = false;
    std::thread starter([&] {
        client c(port);
        started = c.command("/api/start?object_class=tabletop_object");
    });
    check(run.recorded.wait_until_held(), (what + ": held at the starting line").c_str());
    check_in_progress(run, port, 0, "scan_in_progress: yes, starting", what + ", starting");
    run.recorded.release();
    starter.join();
    check(started, (what + ": the start").c_str());
}

/// Run A: scan 1 refined after the operator's stop, exposure's reads refused; scan 2 refined in
/// the same service, nothing refused; SIGINT.
/// @param run [in,out] the run.
/// @param second [out] scan 2's name.
/// @return scan 1's name.
std::string run_refined(service_run &run, std::string &second)
{
    const std::string what = "run A";
    std::string scan;
    serve_while(
        run,
        [&](std::uint16_t port) {
            client c(port);
            check_in_progress(run, port, 1, "scan_in_progress: no, idle", what + ", idle");
            start_held(run, port, what);
            const state_fields scanning = c.poll(
                [](const state_fields &s) {
                    return s.phase != "scanning" || s.frames.taken >= frames_before_stop;
                },
                what + ": scanning");
            scan = scanning.scan;
            std::printf("%s: %s\n", what.c_str(), scanning.property_refusals.c_str());
            const std::string refused = "Color: the camera refused to read exposure (2001) at the ";
            const std::string reason = ": the stand-in refuses reads of 2001";
            check(scanning.phase == "scanning" &&
                      scanning.property_refusals == refused + "start" + reason + "; " + refused +
                                                        "hold" + reason + "; " + refused + "hold" +
                                                        reason,
                  "run A: the state names exposure (2001), refused at the start and twice at the "
                  "hold, before and while held");
            check_in_progress(run, port, 0, "scan_in_progress: yes, scanning", what + ", scanning");
            run.recorded.arm(": refining, ");
            check(c.command("/api/stop"), "run A: the stop");
            check(run.recorded.wait_until_held(), "run A: held at the refining line");
            check_in_progress(run, port, 0, "scan_in_progress: yes, refining", what + ", refining");
            run.recorded.release();
            const state_fields idle = c.poll(
                [](const state_fields &s) { return s.phase == "idle"; }, what + ": idle again");
            check(idle.outcome == "refined" && idle.outcome_message.empty() &&
                      idle.property_refusals ==
                          scanning.property_refusals + "; " + refused + "end" + reason,
                  "run A: refined, the refusals named, the end's too");
            check_in_progress(run, port, 1, "scan_in_progress: no, idle", what + ", idle again");

            run.camera.set_refuse_reads(2001, false);
            check(c.command("/api/start?object_class=tabletop_object"), "run A: scan 2's start");
            const state_fields second_scanning = c.poll(
                [](const state_fields &s) {
                    return s.phase != "scanning" || s.frames.taken >= frames_before_stop;
                },
                what + ": scan 2 scanning");
            second = second_scanning.scan;
            check(second_scanning.phase == "scanning" && second != scan &&
                      second_scanning.property_refusals.empty(),
                  "run A: scan 2 scans, its state names no refusal, none of scan 1's");
            check(c.command("/api/stop"), "run A: scan 2's stop");
            const state_fields second_idle = c.poll(
                [](const state_fields &s) { return s.phase == "idle"; }, what + ": scan 2 idle");
            check(second_idle.outcome == "refined" && second_idle.property_refusals.empty(),
                  "run A: scan 2 refined, its state names no refusal");
            return SIGINT;
        },
        what);
    check(!run.recorded.timed_out(), "run A: each hold released before its limit");
    return scan;
}

/// Run B: SIGTERM in the middle of a scan, after the hold.
/// @param run [in,out] the run.
/// @param scan [out] the scan's name.
/// @return the port served.
std::optional<std::uint16_t> run_stopped(service_run &run, std::string &scan)
{
    const std::string what = "run B";
    return serve_while(
        run,
        [&](std::uint16_t port) {
            client c(port);
            check(c.command("/api/start?object_class=tabletop_object"), "run B: the start");
            const state_fields scanning = c.poll(
                [](const state_fields &s) {
                    return s.phase != "scanning" || s.frames.taken >= frames_before_signal;
                },
                what + ": scanning");
            scan = scanning.scan;
            check(scanning.phase == "scanning" && scanning.property_refusals.empty(),
                  "run B: scanning past the hold, nothing refused");
            return SIGTERM;
        },
        what);
}

/// Checks a scan's directory: its files and its parameters' end and refused properties.
/// @param directory [in] the scan's directory.
/// @param refined [in] whether it is refined: a full model and a trajectory, else neither.
/// @param sequence [in] whether its sequence stays.
/// @param end [in] why it ended.
/// @param refused [in] the refused properties expected.
/// @param what [in] the case.
void check_scan(const std::filesystem::path &directory, bool refined, bool sequence,
                gca::scan_end end, const std::array<bool, gca::color_property_count> &refused,
                const std::string &what)
{
    const bool model = std::filesystem::exists(directory / "full.ply");
    const bool trajectory = std::filesystem::exists(directory / "trajectory.txt");
    const bool frames = std::filesystem::exists(directory / gca::sequence_file_name);
    std::printf("%s: %s: full model %d, trajectory %d, sequence %d\n", what.c_str(),
                directory.c_str(), model, trajectory, frames);
    check(model == refined && trajectory == refined,
          (what + (refined ? ": a full model and a trajectory" : ": no full model, no trajectory"))
              .c_str());
    check(frames == sequence,
          (what + (sequence ? ": its sequence kept" : ": its sequence removed")).c_str());
    const gca::read_result<gca::scan_parameters> parameters =
        gca::read_parameters(directory / "parameters.json");
    check(parameters.value && parameters.value->end == end &&
              parameters.value->refused_properties == refused,
          (what + ": its parameters' end " + std::string(gca::name_of(end)) +
           " and the properties refused; " + parameters.reason)
              .c_str());
}

/// Checks the switches the stand-in's log holds from an index on: each written on at the start,
/// off at the hold and on at the end, and nothing else written.
/// @param camera [in] the stand-in.
/// @param from [in] the log's index the scan began at.
/// @param what [in] the case.
void check_switches(const stand_in_camera &camera, std::size_t from, const std::string &what)
{
    std::vector<std::string> writes;
    for (std::size_t k = from; k < camera.log.size(); k++)
    {
        if (camera.log[k].write)
            writes.push_back(std::to_string(camera.log[k].number) + "=" +
                             std::to_string(*camera.log[k].write));
    }
    std::string text;
    for (const std::string &w : writes)
        text += w + " ";
    std::printf("%s: the switches written: %s\n", what.c_str(), text.c_str());
    check(writes ==
              std::vector<std::string>{"2000=1", "2003=1", "2000=0", "2003=0", "2000=1", "2003=1"},
          (what + ": both on at the start, off at the hold, on at the end").c_str());
    check(camera.table.at(2000).value == 1 && camera.table.at(2003).value == 1,
          (what + ": both switches on, as before the scan").c_str());
}

/// The log's lines of the live model's and the live view's failures (docs/SERVICE.md, "Log lines"),
/// from states given to the log's listener as the scan thread gives them: a line when a failure
/// changes to a reason, none when it stays as it was or is cleared.
void check_failure_lines()
{
    std::vector<std::string> lines;
    gca::service_log log([&lines](const std::string &line) { lines.push_back(line); });
    gca::scan_state state;
    state.live_model_failure = "the extraction failed";
    log.on_state(state);
    log.on_state(state);
    state.live_view_failure = "the copy was refused";
    log.on_state(state);
    log.on_state(state);
    state.live_model_failure.clear();
    state.live_view_failure.clear();
    log.on_state(state);
    state.live_view_failure = "the encoding failed";
    log.on_state(state);
    const std::vector<std::string> expected = {
        "scanner_service: a live model could not be taken: the extraction failed",
        "scanner_service: a live view could not be made: the copy was refused",
        "scanner_service: a live view could not be made: the encoding failed"};
    for (const std::string &line : lines)
        std::printf("failure lines: %s\n", line.c_str());
    check(lines == expected, "failure lines: one for each failure that changed to a reason");
}
} // namespace

/// Makes the stop signals the stop first, before any thread starts, then checks the failures' log
/// lines and runs the service's three runs and the refused command line.
/// @param argc [in] the argument count.
/// @param argv [in] argv[1]: the path of scan_in_progress.
/// @return 0 when every check passed, 1 otherwise.
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::printf("usage: test_scanner_service <scan_in_progress>\n");
        return 1;
    }
    gca::install_stop_signals();
    const auto begin = std::chrono::steady_clock::now();
    check_failure_lines();
    const gca_test::temporary_directory directory("test_scanner_service");
    const std::filesystem::path scans = directory / "scans";
    const std::filesystem::path page = directory / "page";
    std::filesystem::create_directory(scans);
    std::filesystem::create_directory(page);
    gca_test::write_bytes(page / "index.html", "<!doctype html><title>scan</title>\n");

    paced_source frames(gca_test::walk_options(gca_test::walk_poses));
    stand_in_camera camera(frames);
    camera.table.at(2000).value = 1;
    camera.table.at(2003).value = 1;
    camera.table.at(2001).refuse_reads = true;
    recorded_log recorded;
    gca::service_log log([&recorded](const std::string &line) { recorded.sink(line); });
    service_run run{.settings = {.options = {.scans = scans,
                                             .address = "127.0.0.1",
                                             .port = 0,
                                             .page = page,
                                             .keep_sequence = false},
                                 .stream_size = gca_test::small_description.size,
                                 .commit = gca_test::in_process_commit,
                                 .set_of = gca_test::test_set},
                    .camera = camera,
                    .log = log,
                    .recorded = recorded,
                    .in_progress = argv[1]};

    const gca_test::program_run refused = run_program(run.in_progress, {"--help"});
    std::printf("scan_in_progress --help: exit %d: %s", refused.exit_status,
                refused.output.c_str());
    check(refused.exit_status == 2 &&
              refused.output.rfind("scan_in_progress: unknown argument --help\nusage: ", 0) == 0,
          "scan_in_progress: an unknown argument refused with the usage, exit 2");

    std::string scan_a2;
    const std::string scan_a = run_refined(run, scan_a2);
    std::vector<std::string> lines = recorded.lines();
    check_line(lines, "scanner_service: serving the page on http://127.0.0.1:", "run A");
    check(has_line(lines, "scanner_service: serving the page on http://127.0.0.1:") &&
              lines.front().find("/ from " + page.string() + "; scans in " + scans.string() +
                                 ", a refined scan's sequence removed") != std::string::npos,
          "run A: the serving line names the page, the scans and the sequence's removal");
    check_line(lines, "scanner_service: a scan starts", "run A");
    check_line(lines, "scanner_service: scan " + scan_a + ": scanning", "run A");
    check_line(lines,
               "scanner_service: Color at the start: auto exposure on, exposure unread, gain 64, "
               "auto white balance on, white balance 4000",
               "run A");
    check_line(lines,
               "scanner_service: Color: the camera refused to read exposure (2001) at the start: "
               "the stand-in refuses reads of 2001",
               "run A");
    check_line(lines, "scanner_service: scan " + scan_a + ": refining, end stopped, frames taken ",
               "run A");
    check_line(lines, "scanner_service: scan " + scan_a + ": refined, end stopped, frames taken ",
               "run A");
    check(lines.size() >= 2 && lines[lines.size() - 2] == "scanner_service: stopping on SIGINT" &&
              lines.back() == "scanner_service: stopped",
          "run A: \"stopping on SIGINT\", then \"stopped\" last");
    check(!camera.is_open(), "run A: the stand-in closed");
    check_scan(scans / scan_a, true, false, gca::scan_end::stopped,
               {false, true, false, false, false}, "run A, scan 1");
    check_scan(scans / scan_a2, true, false, gca::scan_end::stopped, {}, "run A, scan 2");
    std::vector<std::string> starts;
    for (const std::string &line : lines)
    {
        if (line.rfind("scanner_service: Color at the start: ", 0) == 0)
            starts.push_back(line);
    }
    check(starts.size() == 2 && starts[0].find("exposure unread") != std::string::npos &&
              starts[1].find("unread") == std::string::npos,
          "run A: the five read at each scan's start, exposure unread at scan 1's only");
    check_line(lines, "scanner_service: scan " + scan_a2 + ": refined, end stopped, frames taken ",
               "run A, scan 2");

    const std::size_t log_before_b = camera.log.size();
    const std::size_t lines_before_b = lines.size();
    std::string scan_b;
    const std::optional<std::uint16_t> port_b = run_stopped(run, scan_b);
    lines = recorded.lines();
    const std::vector<std::string> lines_b(
        lines.begin() + static_cast<std::ptrdiff_t>(lines_before_b), lines.end());
    check_line(lines_b,
               "scanner_service: scan " + scan_b +
                   ": unfinished: the scan interface ended during the scan, end shut_down, "
                   "frames taken ",
               "run B");
    check_line(lines_b, "scanner_service: Color while held: auto exposure off, ", "run B");
    check_line(lines_b, "scanner_service: Color after: auto exposure on, ", "run B");
    const std::size_t stopping = index_of(lines_b, "scanner_service: stopping on SIGTERM");
    const std::size_t after = index_of(lines_b, "scanner_service: Color after: ");
    const std::size_t unfinished =
        index_of(lines_b, "scanner_service: scan " + scan_b + ": unfinished");
    check(stopping < after && after < unfinished && unfinished + 1 == lines_b.size() - 1 &&
              lines_b.back() == "scanner_service: stopped",
          "run B: \"stopping on SIGTERM\", then the controls on again, then the scan unfinished, "
          "then \"stopped\" last");
    check(!camera.is_open(), "run B: the stand-in closed");
    check_switches(camera, log_before_b, "run B");
    check_scan(scans / scan_b, false, true, gca::scan_end::shut_down, {},
               "run B, the scan stopped");
    check(gca_test::names_in(scans).size() == 3, "three scans in the scans directory");
    if (port_b)
        check_in_progress(run, *port_b, 3, "scan_in_progress: cannot tell: no answer from ",
                          "run B, the service gone");

    const std::size_t lines_before_c = recorded.lines().size();
    serve_while(run, [](std::uint16_t) { return SIGHUP; }, "run C");
    lines = recorded.lines();
    check(lines.size() == lines_before_c + 3 &&
              lines[lines.size() - 2] == "scanner_service: stopping on SIGHUP" &&
              lines.back() == "scanner_service: stopped",
          "run C: serving, \"stopping on SIGHUP\", \"stopped\"");

    std::printf("test_scanner_service: %.1f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
    return gca_test::check_summary("test_scanner_service");
}
