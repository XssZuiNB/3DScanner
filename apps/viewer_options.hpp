// SPDX-License-Identifier: 0BSD
/// @file viewer_options.hpp
/// orbbec_viewer's command line: the options with their defaults, and the reader that refuses
/// an argument it does not know, and a directory for the sequence that it cannot use, with the
/// usage line. Host only, without the camera and the GPU, so that the offline tests drive it.

#pragma once

#include <cstdint>
#include <filesystem>
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
    /// --sequence: the directory the frames taken are written to, as a sequence
    /// (docs/SEQUENCE_FILE.md) in its file sequence_file_name (viewer_loop.hpp); nothing for no
    /// sequence. read_options() gives only an existing, empty directory that can be written.
    std::optional<std::filesystem::path> sequence_directory;
    /// --hold-color: hold the color camera's exposure and white balance after the settling
    /// (camera/color_hold.hpp) and print their values; --list ignores it.
    bool hold_color = false;
};

/// Builds the usage line, which names every option; read_options() prints it on a refusal.
/// @return the line, without a line end.
std::string usage_line();

/// Reads the command line: --list, --headless, --frames N, --width W, --height H and --fps F,
/// each number read with std::atoi, --sequence DIR and --hold-color. An argument it does not
/// know, --help among them, and an option whose value is missing are refused; so is a --sequence
/// directory, the last one given, that does not exist or is no directory, that holds something
/// already, or that cannot be read or written, each with the system's message where it has one.
/// It looks at that directory and changes nothing on disk.
/// @param argc the argument count.
/// @param argv the arguments, argv[0] the program.
/// @param headless_default the headless setting without --headless.
/// @param err where a refusal is printed: "orbbec_viewer: <reason>", then the usage line; the
///        reasons for the directory are "--sequence <DIR>: no such directory", "...: is no
///        directory", "...: cannot be read: <message>", "...: holds something already" and
///        "...: cannot be written: <message>".
/// @return the options, or nothing when an argument is refused.
std::optional<options> read_options(int argc, const char *const argv[], bool headless_default,
                                    std::ostream &err);
} // namespace gca_viewer
