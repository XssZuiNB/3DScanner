// SPDX-License-Identifier: 0BSD
/// @file replay_options.cpp
/// The replay tool's command line: one table of the options, the reading of each once, the checks
/// of the paths, and the set the scan runs with (see replay_options.hpp).

#include "replay_options.hpp"

#include "camera/sequence_format.hpp"
#include "output_directory.hpp"
#include "scan/parameters_file.hpp"

#include <array>
#include <cstddef>
#include <string_view>
#include <system_error>

namespace gca
{
namespace
{
/// The options, in the usage's order; the index of each is its place in the values read.
enum option_index : std::size_t
{
    sequence_option,
    object_class_option,
    parameters_option,
    scans_option,
    times_option,
    option_count,
};

/// One option of the command line.
struct option_spec
{
    const char *name;       ///< the option, "--sequence"
    const char *value_name; ///< its value's name in the usage; nullptr for a flag
};

/// The options, at their option_index.
constexpr std::array<option_spec, option_count> option_table = {{
    {"--sequence", "DIR"},
    {"--object-class", "CLASS"},
    {"--parameters", "FILE"},
    {"--scans", "OUT"},
    {"--times", nullptr},
}};

/// Says whether a path names nothing: its look gives not_found, as a missing file and a path
/// through a file (ENOENT, ENOTDIR) do; any other error of the look is left to the path's reader.
/// @param path [in] the path.
/// @return whether it names nothing.
bool names_nothing(const std::filesystem::path &path)
{
    std::error_code ignored;
    return std::filesystem::status(path, ignored).type() == std::filesystem::file_type::not_found;
}

/// Prints a refusal: the reason, then the usage.
/// @param err [in,out] where to print.
/// @param reason [in] why the command line is refused.
/// @return nothing, what read_replay_options() returns for a refused command line.
std::optional<replay_options> refuse(std::ostream &err, const std::string &reason)
{
    err << "replay: " << reason << "\n" << replay_usage() << std::endl;
    return std::nullopt;
}
} // namespace

std::string replay_usage()
{
    return std::string("usage: replay --sequence DIR (--object-class CLASS | --parameters FILE) "
                       "--scans OUT [--times]\n"
                       "  DIR: the directory of a sequence, DIR/") +
           sequence_file_name +
           " (docs/SEQUENCE_FILE.md)\n"
           "  CLASS: " +
           std::string(name_of(object_class::tabletop_object)) + " or " +
           std::string(name_of(object_class::large_object)) +
           ", the object class whose own set the scan runs with\n"
           "  FILE: a kept scan's parameters (docs/SCAN_FILES.md), whose object class and set the "
           "scan runs with\n"
           "  OUT: an existing, empty directory the scan's directory is written into\n"
           "  --times: print each frame's stage times, waiting for the GPU after each stage";
}

std::optional<replay_options> read_replay_options(int argc, const char *const argv[],
                                                  std::ostream &err)
{
    std::array<std::optional<std::string>, option_count> given;
    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        std::size_t k = 0;
        while (k < option_count && arg != option_table[k].name)
            k++;
        if (k == option_count)
            return refuse(err, "unknown argument " + arg);
        if (given[k])
            return refuse(err, arg + " given twice");
        if (option_table[k].value_name == nullptr)
            given[k] = std::string();
        else if (i + 1 >= argc)
            return refuse(err, arg + " needs a value");
        else
            given[k] = argv[++i];
    }

    replay_options read;
    if (!given[sequence_option])
        return refuse(err, "missing --sequence");
    read.sequence = std::filesystem::path(*given[sequence_option]) / sequence_file_name;
    if (names_nothing(read.sequence))
        return refuse(err, "--sequence " + *given[sequence_option] + ": " + read.sequence.string() +
                               ": no such file");

    const std::optional<std::string> &class_name = given[object_class_option];
    const std::optional<std::string> &parameters = given[parameters_option];
    if (!class_name && !parameters)
        return refuse(err, "missing the set: --object-class or --parameters");
    if (class_name && parameters)
        return refuse(err, "--object-class and --parameters exclude each other");
    if (class_name)
    {
        const std::optional<object_class> kind = object_class_named(*class_name);
        if (!kind)
            return refuse(err, "--object-class " + *class_name + ": no such object class; " +
                                   std::string(name_of(object_class::tabletop_object)) + " or " +
                                   std::string(name_of(object_class::large_object)));
        read.set_from = *kind;
    }
    else
    {
        if (names_nothing(*parameters))
            return refuse(err, "--parameters " + *parameters + ": no such file");
        read.set_from = std::filesystem::path(*parameters);
    }

    if (!given[scans_option])
        return refuse(err, "missing --scans");
    read.scans = *given[scans_option];
    const std::string not_writable = output_directory_refusal(read.scans);
    if (!not_writable.empty())
        return refuse(err, "--scans " + *given[scans_option] + ": " + not_writable);
    read.times = given[times_option].has_value();
    return read;
}

std::optional<replay_set> replay_set_of(const replay_options &options, std::ostream &err)
{
    if (const auto *kind = std::get_if<object_class>(&options.set_from))
        return replay_set{*kind, parameters_of(*kind)};
    const read_result<scan_parameters> kept =
        read_parameters(std::get<std::filesystem::path>(options.set_from));
    if (!kept.value)
    {
        err << "replay: " << kept.reason << std::endl;
        return std::nullopt;
    }
    return replay_set{kept.value->object_class, kept.value->parameters};
}
} // namespace gca
