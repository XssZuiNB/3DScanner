// SPDX-License-Identifier: 0BSD
/// @file output_directory.hpp
/// The check a program makes of the directory its command line names for what it writes:
/// orbbec_viewer's --sequence and the replay tool's --scans. The directory must exist, be a
/// directory, hold nothing, and let the program create a file in it. Host only; it looks and
/// changes nothing on disk.

#pragma once

#include <filesystem>
#include <string>

namespace gca
{
/// Says why a program cannot write into a directory: it does not exist or is no directory, it
/// cannot be listed, it holds something already, or it cannot be written. Writing stands for
/// creating a file in it (access() with W_OK and X_OK), so a directory on a read-only mount is
/// refused too.
/// @param directory [in] the directory the command line names.
/// @return the reason, without the option's name: "no such directory", "is no directory",
///         "cannot be read: <message>", "holds something already" or "cannot be written:
///         <message>", the system's message as camera/posix_file.hpp gives it; empty when the
///         program can write there.
/// @note Allocates for the reason. Two threads may call it at once.
std::string output_directory_refusal(const std::filesystem::path &directory);
} // namespace gca
