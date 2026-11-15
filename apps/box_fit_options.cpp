// SPDX-License-Identifier: 0BSD
/// @file box_fit_options.cpp
/// box_fit's command line: the one path and its refusals (see box_fit_options.hpp).

#include "box_fit_options.hpp"

#include <system_error>

namespace gca
{
namespace
{
/// Prints a refusal: the reason, then the usage.
/// @param err [in,out] where to print.
/// @param reason [in] why the command line is refused.
/// @return nothing, what read_box_fit_options() returns for a refused command line.
std::optional<std::filesystem::path> refuse(std::ostream &err, const std::string &reason)
{
    err << "box_fit: " << reason << "\n" << box_fit_usage() << std::endl;
    return std::nullopt;
}
} // namespace

std::string box_fit_usage()
{
    return "usage: box_fit MODEL.ply\n"
           "  MODEL.ply: an object model as a scan keeps it (docs/SCAN_FILES.md, \"Models\")";
}

std::optional<std::filesystem::path> read_box_fit_options(int argc, const char *const argv[],
                                                          std::ostream &err)
{
    std::optional<std::filesystem::path> model;
    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        if (arg.empty() || arg[0] == '-' || model)
            return refuse(err, "unknown argument " + arg);
        model = arg;
    }
    if (!model)
        return refuse(err, "missing the object model's PLY");
    // A path that names nothing gives not_found (ENOENT, ENOTDIR); any other error of the look is
    // left to the model's reader, which names it.
    std::error_code ignored;
    if (std::filesystem::status(*model, ignored).type() == std::filesystem::file_type::not_found)
        return refuse(err, model->string() + ": no such file");
    return model;
}
} // namespace gca
