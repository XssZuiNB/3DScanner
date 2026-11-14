// SPDX-License-Identifier: 0BSD
/// @file distortion_options.cpp
/// distortion_check's command line: one table of the options, from which the reader and the
/// refusals come (see distortion_options.hpp).

#include "distortion_options.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace gca
{
namespace
{
/// Reads a whole argument as a finite number.
/// @param text [in] the argument.
/// @param field [out] the number, set when the argument is read.
/// @return whether the argument is a finite number from end to end.
template <typename T> bool read_number(const std::string &text, T &field)
{
    char *end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (text.empty() || end != text.c_str() + text.size() || !std::isfinite(value))
        return false;
    field = static_cast<T>(value);
    return true;
}

/// Reads "COLSxROWS", two whole numbers joined by an x.
/// @param text [in] the argument.
/// @param board [out] its columns and rows, set when the argument is read.
/// @return whether the argument is two whole numbers joined by an x.
bool read_corners(const std::string &text, checkerboard &board)
{
    int columns = 0;
    int rows = 0;
    int used = 0;
    if (std::sscanf(text.c_str(), "%dx%d%n", &columns, &rows, &used) != 2 ||
        used != static_cast<int>(text.size()))
        return false;
    board.columns = columns;
    board.rows = rows;
    return true;
}

/// One option of the command line; each takes a value and is required.
struct option_spec
{
    const char *name;       ///< the option, "--fx"
    const char *value_kind; ///< what its value has to be, for a refusal
    /// Sets the option's field from its value; false when the value cannot be read.
    bool (*set)(distortion_options &opt, const std::string &value);
};

/// The options, in the usage's order.
const option_spec option_table[] = {
    {"--fx", "a number",
     [](distortion_options &o, const std::string &v) { return read_number(v, o.camera.fx); }},
    {"--fy", "a number",
     [](distortion_options &o, const std::string &v) { return read_number(v, o.camera.fy); }},
    {"--cx", "a number",
     [](distortion_options &o, const std::string &v) { return read_number(v, o.camera.cx); }},
    {"--cy", "a number",
     [](distortion_options &o, const std::string &v) { return read_number(v, o.camera.cy); }},
    {"--depth-scale", "a number",
     [](distortion_options &o, const std::string &v) { return read_number(v, o.depth_scale_m); }},
    {"--square-mm", "a number",
     [](distortion_options &o, const std::string &v) { return read_number(v, o.board.square_mm); }},
    {"--corners", "COLSxROWS",
     [](distortion_options &o, const std::string &v) { return read_corners(v, o.board); }},
};

/// The number of options.
constexpr size_t option_count = sizeof(option_table) / sizeof(option_table[0]);

/// Prints a refusal: the reason, then the usage.
/// @param err [in,out] where to print.
/// @param reason [in] why the command line is refused.
/// @return nothing, what read_distortion_options() returns for a refused command line.
std::optional<distortion_options> refuse(std::ostream &err, const std::string &reason)
{
    err << "distortion_check: " << reason << "\n" << distortion_usage() << std::endl;
    return std::nullopt;
}
} // namespace

std::string distortion_usage()
{
    return "usage: distortion_check --fx PX --fy PX --cx PX --cy PX --depth-scale M_PER_UNIT "
           "--square-mm MM --corners COLSxROWS COLOR.png DEPTH.png\n"
           "  --fx, --fy, --cx, --cy (px) and --depth-scale (m/unit): as orbbec_viewer prints "
           "them at the\n"
           "    snapshot's stream size, \"Intrinsics fx=... fy=... cx=... cy=...  depth "
           "scale=... m/unit\"\n"
           "  --square-mm: the edge of one square as measured on the print, mm\n"
           "  --corners: the board's inner corners, e.g. 11x7";
}

std::optional<distortion_options> read_distortion_options(int argc, const char *const argv[],
                                                          std::ostream &err)
{
    distortion_options opt;
    bool seen[option_count] = {};
    int paths = 0;
    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        size_t option = 0;
        while (option < option_count && arg != option_table[option].name)
            option++;
        if (option == option_count && (arg.empty() || arg[0] == '-' || paths == 2))
            return refuse(err, "unknown argument " + arg);
        if (option == option_count)
        {
            (paths++ == 0 ? opt.color_path : opt.depth_path) = arg;
            continue;
        }
        if (i + 1 >= argc)
            return refuse(err, arg + " needs a value");
        const std::string value = argv[++i];
        if (!option_table[option].set(opt, value))
            return refuse(err,
                          arg + " takes " + option_table[option].value_kind + ", not " + value);
        seen[option] = true;
    }
    for (size_t option = 0; option < option_count; option++)
        if (!seen[option])
            return refuse(err, std::string("missing ") + option_table[option].name);
    if (paths < 2)
        return refuse(err,
                      paths == 0 ? "missing the color and the depth PNG" : "missing the depth PNG");
    return opt;
}
} // namespace gca
