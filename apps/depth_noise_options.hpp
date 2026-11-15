// SPDX-License-Identifier: 0BSD
/// @file depth_noise_options.hpp
/// depth_noise's command line: the directory of one sequence and nothing else; the reader refuses
/// an argument it does not know, a missing directory and one that does not exist with the reason
/// and the usage. Host only, without the camera and the GPU, so that the offline tests drive it.

#pragma once

#include <filesystem>
#include <optional>
#include <ostream>
#include <string>

namespace gca
{
/// Builds the usage, which names the one argument and the file it reads;
/// read_depth_noise_options() prints it with a refusal.
/// @return the usage, lines joined by "\n", without a line end at the end.
std::string depth_noise_usage();

/// Reads the command line: one argument, the directory of a sequence taken with the camera at
/// rest. It looks at the path and changes nothing on disk; whether DIR/frames.sequence is a
/// sequence is the sequence's reader's to say (camera/sequence_reader.hpp).
/// @param argc [in] the argument count.
/// @param argv [in] the arguments, argv[0] the program; the caller owns them.
/// @param err [in,out] where a refusal is printed: "depth_noise: <reason>", then the usage. The
///     reasons: "unknown argument <ARG>" for one that starts with "-" (--help among them), an
///     empty one and a second directory; "missing the sequence's directory"; "<DIR>: no such
///     directory" for a path that names nothing (a missing directory, or a path through a file).
/// @return the directory as given; nothing when the command line is refused.
std::optional<std::filesystem::path> read_depth_noise_options(int argc, const char *const argv[],
                                                              std::ostream &err);
} // namespace gca
