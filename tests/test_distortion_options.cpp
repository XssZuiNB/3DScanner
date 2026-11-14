// SPDX-License-Identifier: 0BSD
/// @file test_distortion_options.cpp
/// Offline test of distortion_check's command line (distortion_options.hpp): each option's value
/// lands in its own field, in any order of the options and with the paths anywhere, the color
/// path first; each kind of argument it does not know, an option without its value, each value
/// it cannot read, each missing option and a missing path are refused with the reason and the
/// usage. Needs no GPU and no camera.

#include "check.hpp"
#include "distortion_options.hpp"

#include <cstdio>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using gca_test::check;
using gca_test::check_refusal;
using gca_test::check_summary;

namespace
{
/// What read_distortion_options() gave for one command line.
struct read_result
{
    std::optional<gca::distortion_options> options; ///< the options, or nothing when refused
    std::string printed;                            ///< what it printed to its error stream
};

/// Runs read_distortion_options() on a command line.
/// @param args [in] the arguments after the program's name.
/// @return the options and what was printed.
read_result read(const std::vector<std::string> &args)
{
    std::vector<const char *> argv = {"distortion_check"};
    for (const std::string &arg : args)
        argv.push_back(arg.c_str());
    std::ostringstream err;
    read_result result;
    result.options = gca::read_distortion_options(static_cast<int>(argv.size()), argv.data(), err);
    result.printed = err.str();
    return result;
}

/// A whole command line in the usage's order, each value distinct and exact in float, so that
/// a value landing in another option's field shows.
const std::vector<std::string> whole_line = {
    "--fx",          "611.5",  "--fy",        "612.25", "--cx",      "633.75", "--cy",  "397.5",
    "--depth-scale", "0.0005", "--square-mm", "29.5",   "--corners", "11x7",   "a.png", "b.png"};

/// Checks that a command line was accepted with whole_line's values and printed nothing.
/// @param what [in] the case, for the check lines.
/// @param r [in] what the reader gave.
void check_whole_line(const char *what, const read_result &r)
{
    if (r.options)
        std::printf("%s: fx %g fy %g cx %g cy %g, depth scale %g m/unit, square %g mm, %dx%d, "
                    "%s, %s\n",
                    what, r.options->camera.fx, r.options->camera.fy, r.options->camera.cx,
                    r.options->camera.cy, r.options->depth_scale_m, r.options->board.square_mm,
                    r.options->board.columns, r.options->board.rows, r.options->color_path.c_str(),
                    r.options->depth_path.c_str());
    const bool expected = r.options && r.options->camera.fx == 611.5f &&
                          r.options->camera.fy == 612.25f && r.options->camera.cx == 633.75f &&
                          r.options->camera.cy == 397.5f && r.options->depth_scale_m == 0.0005 &&
                          r.options->board.square_mm == 29.5 && r.options->board.columns == 11 &&
                          r.options->board.rows == 7 && r.options->color_path == "a.png" &&
                          r.options->depth_path == "b.png";
    check(expected, (std::string("each value in its field: ") + what).c_str());
    check(r.printed.empty(), (std::string("nothing printed: ") + what).c_str());
}

/// Each value lands in its field: in the usage's order, and with the options turned around and
/// the paths at the front and in the middle.
void test_accepted()
{
    check_whole_line("the usage's order", read(whole_line));
    check_whole_line("another order", read({"a.png", "--corners", "11x7", "--square-mm", "29.5",
                                            "--depth-scale", "5e-4", "b.png", "--cy", "397.5",
                                            "--cx", "633.75", "--fy", "612.25", "--fx", "611.5"}));
}

/// whole_line without one option and its value.
/// @param option [in] the option to leave out.
/// @return the arguments.
std::vector<std::string> without(const std::string &option)
{
    std::vector<std::string> args;
    for (size_t i = 0; i < whole_line.size(); i++)
    {
        if (whole_line[i] == option)
            i++;
        else
            args.push_back(whole_line[i]);
    }
    return args;
}

/// whole_line with one option's value replaced.
/// @param option [in] the option.
/// @param value [in] its new value.
/// @return the arguments.
std::vector<std::string> with_value(const std::string &option, const std::string &value)
{
    std::vector<std::string> args = whole_line;
    for (size_t i = 0; i + 1 < args.size(); i++)
        if (args[i] == option)
            args[i + 1] = value;
    return args;
}

/// A command line that has to be refused, and the reason it prints before the usage.
struct refusal_case
{
    std::string what;              ///< the case, for the check lines
    std::vector<std::string> args; ///< the arguments after the program's name
    std::string reason;            ///< the reason, after "distortion_check: "
};

/// Each refusal case gives no options and prints "distortion_check: <reason>", then the usage.
void test_refusals()
{
    std::vector<std::string> third_path = whole_line;
    third_path.push_back("c.png");
    std::vector<std::string> one_path = whole_line;
    one_path.pop_back();
    std::vector<std::string> no_path = one_path;
    no_path.pop_back();
    std::vector<std::string> no_value = no_path;
    no_value.push_back("--fx");
    std::vector<refusal_case> cases = {
        {"--help", {"--help"}, "unknown argument --help"},
        {"a short option", {"-h"}, "unknown argument -h"},
        {"an empty argument", {""}, "unknown argument "},
        {"a third path", third_path, "unknown argument c.png"},
        {"an option without its value", no_value, "--fx needs a value"},
        {"a word for a number", with_value("--fx", "abc"), "--fx takes a number, not abc"},
        {"a number with a tail", with_value("--fy", "1.5x"), "--fy takes a number, not 1.5x"},
        {"an empty number", with_value("--cx", ""), "--cx takes a number, not "},
        {"nan", with_value("--cy", "nan"), "--cy takes a number, not nan"},
        {"inf", with_value("--depth-scale", "inf"), "--depth-scale takes a number, not inf"},
        {"an option's name for a value", with_value("--square-mm", "--fx"),
         "--square-mm takes a number, not --fx"},
        {"corners with by", with_value("--corners", "11by7"),
         "--corners takes COLSxROWS, not 11by7"},
        {"corners without rows", with_value("--corners", "11x"),
         "--corners takes COLSxROWS, not 11x"},
        {"corners without columns", with_value("--corners", "x7"),
         "--corners takes COLSxROWS, not x7"},
        {"corners with a tail", with_value("--corners", "11x7x"),
         "--corners takes COLSxROWS, not 11x7x"},
        {"no depth path", one_path, "missing the depth PNG"},
        {"no path", no_path, "missing the color and the depth PNG"},
    };
    for (const char *option :
         {"--fx", "--fy", "--cx", "--cy", "--depth-scale", "--square-mm", "--corners"})
        cases.push_back(
            {std::string("no ") + option, without(option), std::string("missing ") + option});
    for (const refusal_case &refusal : cases)
    {
        const read_result r = read(refusal.args);
        // Printed without its last line end, which check_refusal() adds.
        const std::string message = r.printed.substr(0, r.printed.find_last_not_of('\n') + 1);
        check_refusal(refusal.what, !r.options.has_value(), message,
                      r.printed == "distortion_check: " + refusal.reason + "\n" +
                                       gca::distortion_usage() + "\n");
    }
}
} // namespace

/// Runs the command-line checks and prints the options read, the refusals and the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    test_accepted();
    test_refusals();
    return check_summary("test_distortion_options");
}
