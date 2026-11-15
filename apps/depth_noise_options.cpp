// SPDX-License-Identifier: 0BSD
/// @file depth_noise_options.cpp
/// depth_noise's command line: the one directory and its refusals (see depth_noise_options.hpp).

#include "depth_noise_options.hpp"

#include <system_error>

namespace gca
{
namespace
{
/// Prints a refusal: the reason, then the usage.
/// @param err [in,out] where to print.
/// @param reason [in] why the command line is refused.
/// @return nothing, what read_depth_noise_options() returns for a refused command line.
std::optional<std::filesystem::path> refuse(std::ostream &err, const std::string &reason)
{
    err << "depth_noise: " << reason << "\n" << depth_noise_usage() << std::endl;
    return std::nullopt;
}
} // namespace

std::string depth_noise_usage()
{
    return "usage: depth_noise DIR\n"
           "  DIR: the directory of a sequence taken with the camera at rest, DIR/frames.sequence "
           "(docs/SEQUENCE_FILE.md)";
}

std::optional<std::filesystem::path> read_depth_noise_options(int argc, const char *const argv[],
                                                              std::ostream &err)
{
    std::optional<std::filesystem::path> directory;
    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        if (arg.empty() || arg[0] == '-' || directory)
            return refuse(err, "unknown argument " + arg);
        directory = arg;
    }
    if (!directory)
        return refuse(err, "missing the sequence's directory");
    // A path that names nothing gives not_found (ENOENT, ENOTDIR); any other error of the look,
    // and a path that names a file, is left to the sequence's reader, which names it.
    std::error_code ignored;
    if (std::filesystem::status(*directory, ignored).type() ==
        std::filesystem::file_type::not_found)
        return refuse(err, directory->string() + ": no such directory");
    return directory;
}
} // namespace gca
