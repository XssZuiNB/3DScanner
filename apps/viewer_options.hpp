// SPDX-License-Identifier: 0BSD
/// @file viewer_options.hpp
/// orbbec_viewer's command line: the options with their defaults, and the reader that refuses
/// an argument it does not know with the usage line. Host only, without the camera and the GPU,
/// so that the offline tests drive it.

#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>

namespace gca_viewer
{
/// The viewer's options.
struct options
{
    uint32_t width = 640;   ///< color (and aligned depth) width in pixels
    uint32_t height = 400;  ///< [?: a choice] 640x480 color does not exist on USB 2 [M 2026-11-13]
    uint32_t fps = 30;      ///< frame rate of both streams
    int max_frames = -1;    ///< -1: run until quit
    bool headless = false;  ///< no window; the last of max_frames frames is saved
    bool list_only = false; ///< print the stream profiles and exit
};

/// Builds the usage line, which names every option; read_options() prints it on a refusal.
/// @return the line, without a line end.
std::string usage_line();

/// Reads the command line: --list, --headless, --frames N, --width W, --height H and --fps F,
/// each value read with std::atoi. An argument it does not know, --help among them, and an
/// option whose value is missing are refused.
/// @param argc the argument count.
/// @param argv the arguments, argv[0] the program.
/// @param headless_default the headless setting without --headless.
/// @param err where a refusal is printed: "orbbec_viewer: <reason>", then the usage line.
/// @return the options, or nothing when an argument is refused.
std::optional<options> read_options(int argc, const char *const argv[], bool headless_default,
                                    std::ostream &err);
} // namespace gca_viewer
