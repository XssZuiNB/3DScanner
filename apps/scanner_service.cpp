// SPDX-License-Identifier: 0BSD
/// @file scanner_service.cpp
/// The scanner service (R7; the spec, "The service"; docs/SERVICE.md): the page served on the
/// operator's own network, the scans it starts, and the Orbbec camera, opened only from a scan's
/// start to its stop, with its color exposure and white balance held during the scan. It runs
/// until SIGINT, SIGTERM or SIGHUP; a scan in progress then ends without refinement, the camera's
/// automatic controls switched on again and the camera closed, the scan kept as unfinished. The
/// options are service_options.hpp's; the parts and their order service.hpp's. Its lines on
/// standard output are an interface (docs/SERVICE.md, "Log lines"). It links OrbbecSDK, whose
/// load-time code runs before main() (AGENTS.md §3): it runs on the camera only on the user's
/// word.

#include "build_commit.hpp"
#include "camera/orbbec_device.hpp"
#include "scan/object_class.hpp"
#include "service.hpp"
#include "service_options.hpp"
#include "types/frame_description.hpp"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <string>

namespace
{
/// The camera's frame rate, in frames a second, both streams.
constexpr std::uint32_t camera_fps = 30;

/// The stream size a scan opens the camera for: 640x400 [?: a choice, until
/// per-stage-times-from-replay decides the stream size (the stage brief, decision 10)].
constexpr gca::frame_size stream_size = {640, 400};

/// Writes a line of the service's log on standard output, which systemd's journal takes, and
/// flushes it, so that a line stands in the journal when it is logged.
/// @param line [in] the line, without its line end.
void print_line(const std::string &line)
{
    std::cout << line << std::endl;
}

/// Serves with the Orbbec camera until a stop signal (service.hpp, serve()); the camera, built
/// closed, is closed again when this returns, by an exception too.
/// @param options [in] the command line read.
/// @return serve()'s: 0 after a stop signal, 1 when the web layer did not start.
int run_service(const gca::service_options &options)
{
    gca::orbbec_device camera(camera_fps);
    gca::service_log log(print_line);
    return gca::serve({.options = options,
                       .stream_size = stream_size,
                       .commit = std::string(gca::build_commit()),
                       .set_of = gca::parameters_of},
                      camera, camera, log);
}
} // namespace

/// Runs the service: reads the options first, before anything opens the GPU, a port or the camera
/// or installs a signal handler; then makes the stop signals its stop, before any thread starts;
/// then serves until one comes. An exception out of the service is caught here, so that the stack
/// unwinds and a scan in progress ends with the camera's controls switched on again; without a
/// handler the program could end without unwinding [S: C++20, [except.handle]].
/// @param argc the argument count.
/// @param argv the arguments (service_options.hpp).
/// @return 0 after a stop signal; 1 when the web layer did not start or an exception ended the
///     service; 2 when the command line is refused, the reason and the usage on stderr.
int main(int argc, char *argv[])
{
    const std::optional<gca::service_options> options = gca::read_service_options(
        argc, argv, std::getenv("HOME"), GCA_DEFAULT_PAGE_DIRECTORY, std::cerr);
    if (!options)
        return 2;
    gca::install_stop_signals();
    try
    {
        return run_service(*options);
    }
    catch (const std::exception &e)
    {
        std::cout << gca::service_line_prefix << e.what() << std::endl;
    }
    catch (...)
    {
        std::cout << gca::service_line_prefix << "an exception that is no std::exception"
                  << std::endl;
    }
    return 1;
}
