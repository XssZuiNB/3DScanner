// SPDX-License-Identifier: 0BSD
/// @file box_fit_options.hpp
/// box_fit's command line: the path of one object model's PLY and nothing else; the reader
/// refuses an argument it does not know, a missing path and a path that does not exist with the
/// reason and the usage. Host only, without the camera and the GPU, so that the offline tests
/// drive it.

#pragma once

#include <filesystem>
#include <optional>
#include <ostream>
#include <string>

namespace gca
{
/// Builds the usage, which names the one argument and the file it takes;
/// read_box_fit_options() prints it with a refusal.
/// @return the usage, lines joined by "\n", without a line end at the end.
std::string box_fit_usage();

/// Reads the command line: one argument, the path of an object model's PLY. It looks at the path
/// and changes nothing on disk; whether the file is a model the scanner wrote is the model's
/// reader's to say (scan/model_file.hpp).
/// @param argc [in] the argument count.
/// @param argv [in] the arguments, argv[0] the program; the caller owns them.
/// @param err [in,out] where a refusal is printed: "box_fit: <reason>", then the usage. The
///     reasons: "unknown argument <ARG>" for one that starts with "-" (--help among them), an
///     empty one and a second path; "missing the object model's PLY"; "<PATH>: no such file" for
///     a path that names nothing (a missing file, or a path through a file).
/// @return the path as given; nothing when the command line is refused.
std::optional<std::filesystem::path> read_box_fit_options(int argc, const char *const argv[],
                                                          std::ostream &err);
} // namespace gca
