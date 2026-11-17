// SPDX-License-Identifier: 0BSD
/// @file scan_in_progress.cpp
/// Says whether the scanner service has a scan in progress (the spec, "The service";
/// docs/SERVICE.md, "Whether a scan is in progress"), read from the service: it asks the state
/// route, GET /api/state (docs/ROUTES.md, "The state"), and reads the phase. A scan is in progress
/// while it starts, scans or refines: the camera is the service's then, or its refinement holds
/// the GPU, and the service is not to be stopped. Before a camera program runs or the service is
/// stopped or replaced, AGENTS.md §2 asks it. Links no OrbbecSDK, no CUDA and no OpenCV, so it
/// touches no hardware.

#include "service_options.hpp"

#include <array>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

#include <httplib.h>

namespace
{
/// The address asked without --address: the board itself, which a service on 0.0.0.0 serves too.
constexpr std::string_view default_address = "127.0.0.1";

/// How long the connection and the answer may take: 2 s and 5 s [?: a choice; the state route
/// answers at once, taking only the state's lock (docs/ROUTES.md, "Threads")].
constexpr int connect_limit_s = 2;
constexpr int answer_limit_s = 5;

/// The exit statuses (docs/SERVICE.md).
enum exit_status : int
{
    in_progress = 0,  ///< a scan starts, scans or refines
    no_scan = 1,      ///< the service is idle
    refused_line = 2, ///< the command line is refused
    cannot_tell = 3,  ///< the service did not answer, or not with a state
};

/// The command line read.
struct options
{
    std::string address{default_address};           ///< --address
    std::uint16_t port = gca::default_service_port; ///< --port
};

/// Builds the usage.
/// @return the usage, lines joined by "\n", without a line end at the end.
std::string usage()
{
    return "usage: scan_in_progress [--address A] [--port N]\n"
           "  A: the service's numeric address; " +
           std::string(default_address) +
           " without it\n"
           "  N: the service's port, 1 to 65535; " +
           std::to_string(gca::default_service_port) +
           " without it\n"
           "  exit 0: a scan is in progress (starting, scanning or refining); 1: none (idle); 2: "
           "this usage; 3: cannot tell (no answer, or no state)";
}

/// Prints a refusal of the command line: the reason, then the usage, on stderr.
/// @param reason [in] why the command line is refused.
/// @return nothing.
std::optional<options> refuse(const std::string &reason)
{
    std::cerr << "scan_in_progress: " << reason << "\n" << usage() << std::endl;
    return std::nullopt;
}

/// Reads the command line: --address A and --port N, each once.
/// @param argc [in] the argument count.
/// @param argv [in] the arguments, argv[0] the program.
/// @return the options; nothing when the command line is refused: "unknown argument <ARG>"
///         (--help among them), "<OPTION> needs a value", "<OPTION> given twice", "--port <N>:
///         not a whole number from 1 to 65535".
std::optional<options> read_options(int argc, const char *const argv[])
{
    options read;
    std::array<bool, 2> given = {false, false};
    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        const std::size_t k = arg == "--address" ? 0 : arg == "--port" ? 1 : given.size();
        if (k == given.size())
            return refuse("unknown argument " + arg);
        if (given[k])
            return refuse(arg + " given twice");
        if (i + 1 >= argc)
            return refuse(arg + " needs a value");
        given[k] = true;
        const std::string value = argv[++i];
        if (k == 0)
            read.address = value;
        else if (const std::optional<std::uint16_t> port = gca::port_of(value))
            read.port = *port;
        else
            return refuse("--port " + value + ": not a whole number from 1 to 65535");
    }
    return read;
}

/// Finds the phase in a state message by its key. The message is one JSON object, and every
/// double quote inside a string of it is escaped with a backslash, so "phase":" followed by the
/// key's own quote never stands inside a string: the first such text is the key's (docs/ROUTES.md,
/// "Messages").
/// @param message [in] the state message.
/// @return the phase's name, as it stands between its quotes; nothing when the message holds no
///         such key.
std::optional<std::string> phase_of(const std::string &message)
{
    constexpr std::string_view key = "\"phase\":\"";
    const std::size_t at = message.find(key);
    if (at == std::string::npos)
        return std::nullopt;
    const std::size_t start = at + key.size();
    const std::size_t end = message.find('"', start);
    if (end == std::string::npos)
        return std::nullopt;
    return message.substr(start, end - start);
}

/// Asks the service for its state and says whether a scan is in progress, on stdout, or why it
/// cannot tell, on stderr.
/// @param asked [in] the service's address and port.
/// @return in_progress for the phases starting, scanning and refining, "scan_in_progress: yes,
///         <phase>"; no_scan for idle, "scan_in_progress: no, idle"; cannot_tell,
///         "scan_in_progress: cannot tell: <reason>", when no answer came within the limits, the
///         answer was no 200, or it held no phase of the four.
int ask(const options &asked)
{
    const std::string where =
        "http://" + asked.address + ":" + std::to_string(asked.port) + "/api/state";
    httplib::Client client(asked.address, asked.port);
    client.set_connection_timeout(connect_limit_s, 0);
    client.set_read_timeout(answer_limit_s, 0);
    client.set_write_timeout(answer_limit_s, 0);
    const httplib::Result answer = client.Get("/api/state");
    std::string reason;
    if (!answer)
        reason = "no answer from " + where + ": " + httplib::to_string(answer.error());
    else if (answer->status != 200)
        reason = where + " answered " + std::to_string(answer->status);
    else
    {
        const std::optional<std::string> phase = phase_of(answer->body);
        if (phase == "starting" || phase == "scanning" || phase == "refining")
        {
            std::cout << "scan_in_progress: yes, " << *phase << std::endl;
            return in_progress;
        }
        if (phase == "idle")
        {
            std::cout << "scan_in_progress: no, idle" << std::endl;
            return no_scan;
        }
        reason = where + " answered no state with a phase of the four";
    }
    std::cerr << "scan_in_progress: cannot tell: " << reason << std::endl;
    return cannot_tell;
}
} // namespace

/// Says whether the service has a scan in progress: reads the command line, ignores SIGPIPE, so
/// that a service that closes the connection while the request is written fails the write with
/// EPIPE, which the client reports as no answer (exit 3), instead of ending the program with 141,
/// then asks.
/// @param argc [in] the argument count.
/// @param argv [in] the arguments: [--address A] [--port N].
/// @return 0 yes, 1 no, 2 the command line refused, 3 cannot tell (ask()).
int main(int argc, char *argv[])
{
    const std::optional<options> read = read_options(argc, argv);
    if (!read)
        return refused_line;
    std::signal(SIGPIPE, SIG_IGN);
    return ask(*read);
}
