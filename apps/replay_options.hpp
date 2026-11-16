// SPDX-License-Identifier: 0BSD
/// @file replay_options.hpp
/// The replay tool's command line: the sequence's directory, the object class or a kept scan's
/// parameters file that gives the scan's set, the directory the scan's directory goes into, and
/// the request for the time lines; the reader refuses an argument it does not know, a missing
/// sequence and a directory that holds something already, among others, with the reason and the
/// usage. Then the set the scan runs with, from the object class or the parameters file. Host
/// only, without the camera and the GPU, so that the offline tests drive it.

#pragma once

#include "scan/object_class.hpp"

#include <filesystem>
#include <optional>
#include <ostream>
#include <string>
#include <variant>

namespace gca
{
/// The replay tool's options.
struct replay_options
{
    std::filesystem::path sequence; ///< the sequence's file, DIR/frames.sequence for --sequence DIR
    /// Where the scan's set comes from: an object class's own (--object-class), or a kept scan's
    /// parameters file (--parameters), whose object class and set the scan runs with.
    std::variant<gca::object_class, std::filesystem::path> set_from;
    std::filesystem::path scans; ///< --scans: the directory the scan's directory is written into
    bool times = false;          ///< --times: print each frame's stage times
};

/// The object class and the set a replay's scan runs with.
struct replay_set
{
    gca::object_class kind;         ///< the object class
    object_class_parameters values; ///< the set
};

/// Builds the usage: its first line names every option, the lines after it each value.
/// @return the usage, lines joined by "\n", without a line end at the end.
std::string replay_usage();

/// Reads the command line: --sequence DIR, --object-class CLASS or --parameters FILE, --scans OUT
/// and --times, in any order, each once. It looks at the paths and changes nothing on disk;
/// whether the sequence's file is a sequence and the parameters file a scan's parameters are
/// their readers' to say.
/// @param argc [in] the argument count.
/// @param argv [in] the arguments, argv[0] the program; the caller owns them.
/// @param err [in,out] where a refusal is printed: "replay: <reason>", then the usage. The
///     reasons: "unknown argument <ARG>" for one that is no option (--help, a bare path and an
///     empty one among them); "<OPTION> needs a value"; "<OPTION> given twice"; "missing the set:
///     --object-class or --parameters"; "--object-class and --parameters exclude each other";
///     "--object-class <NAME>: no such object class; tabletop_object or large_object";
///     "--parameters <FILE>: no such file"; "missing --sequence"; "--sequence <DIR>:
///     <DIR>/frames.sequence: no such file"; "missing --scans"; "--scans <OUT>: " and the reason
///     output_directory_refusal() gives (output_directory.hpp). A path that names nothing is one
///     whose look gives not_found (a missing file, or a path through a file); any other path is
///     left to its reader.
/// @return the options; nothing when the command line is refused.
std::optional<replay_options> read_replay_options(int argc, const char *const argv[],
                                                  std::ostream &err);

/// Gives the set the scan runs with: the object class's own (scan/object_class.hpp,
/// parameters_of()), or the object class and set of a kept scan's parameters file as they stand
/// (scan/parameters_file.hpp, read_parameters()); the file's other keys, the facts of the scan that
/// wrote it, are not used.
/// @param options [in] the command line read.
/// @param err [in,out] where a refused parameters file is printed: "replay: " and its reader's
///     reason, which names the file; the replay turns it into exit 1.
/// @return the object class and the set; nothing when the parameters file is refused.
/// @note Reads the parameters file, host only.
std::optional<replay_set> replay_set_of(const replay_options &options, std::ostream &err);
} // namespace gca
