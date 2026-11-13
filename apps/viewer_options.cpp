// SPDX-License-Identifier: 0BSD
/// @file viewer_options.cpp
/// orbbec_viewer's command line: one table of the options, from which the reader and the usage
/// line both come (see viewer_options.hpp).

#include "viewer_options.hpp"

#include <cstdlib>
#include <string>

namespace gca_viewer
{
namespace
{
/// One option of the command line.
struct option_spec
{
    const char *name;       ///< the option, "--frames"
    const char *value_name; ///< its value's name in the usage line; nullptr for a flag
    /// Sets the option's field; the value is nullptr for a flag, else read with std::atoi.
    void (*set)(options &opt, const char *value);
};

/// The options, in the usage line's order.
const option_spec option_table[] = {
    {"--list", nullptr, [](options &opt, const char *) { opt.list_only = true; }},
    {"--headless", nullptr, [](options &opt, const char *) { opt.headless = true; }},
    {"--frames", "N", [](options &opt, const char *value) { opt.max_frames = std::atoi(value); }},
    {"--width", "W", [](options &opt, const char *value) { opt.width = std::atoi(value); }},
    {"--height", "H", [](options &opt, const char *value) { opt.height = std::atoi(value); }},
    {"--fps", "F", [](options &opt, const char *value) { opt.fps = std::atoi(value); }},
};

/// Finds an option by its name.
/// @param name the argument.
/// @return the option, or nullptr when the viewer has none of that name.
const option_spec *find_option(const std::string &name)
{
    for (const option_spec &spec : option_table)
    {
        if (name == spec.name)
            return &spec;
    }
    return nullptr;
}

/// Prints a refusal: the reason, then the usage line.
/// @param err where to print.
/// @param reason why the command line is refused.
/// @return nothing, what read_options() returns for a refused command line.
std::optional<options> refuse(std::ostream &err, const std::string &reason)
{
    err << "orbbec_viewer: " << reason << "\n" << usage_line() << std::endl;
    return std::nullopt;
}
} // namespace

std::string usage_line()
{
    std::string line = "usage: orbbec_viewer";
    for (const option_spec &spec : option_table)
    {
        line += std::string(" [") + spec.name;
        if (spec.value_name != nullptr)
            line += std::string(" ") + spec.value_name;
        line += "]";
    }
    return line;
}

std::optional<options> read_options(int argc, const char *const argv[], bool headless_default,
                                    std::ostream &err)
{
    options opt;
    opt.headless = headless_default;
    for (int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        const option_spec *spec = find_option(a);
        if (spec == nullptr)
            return refuse(err, "unknown argument " + a);
        const bool takes_value = spec->value_name != nullptr;
        if (takes_value && i + 1 >= argc)
            return refuse(err, a + " needs a value");
        spec->set(opt, takes_value ? argv[++i] : nullptr);
    }
    return opt;
}
} // namespace gca_viewer
