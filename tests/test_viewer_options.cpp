// SPDX-License-Identifier: 0BSD
/// @file test_viewer_options.cpp
/// Offline test of orbbec_viewer's command line (viewer_options.hpp): each accepted command line
/// gives its options, the defaults among them, flags at the end of the line too, and the
/// behaviours kept from before (a value read with atoi, the last repeat winning, an option's name
/// taken as a value); each kind of argument the viewer does not know, --help and an empty one
/// among them, and an option without its value are refused with the reason and the usage line,
/// which names the six options. Needs no GPU and no camera.

#include "check.hpp"
#include "viewer_options.hpp"

#include <cstdint>
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
/// What read_options() gave for one command line.
struct read_result
{
    std::optional<gca_viewer::options> options; ///< the options, or nothing when refused
    std::string printed;                        ///< what it printed to its error stream
};

/// Runs read_options() on a command line.
/// @param args the arguments after the program's name.
/// @param headless_default the headless setting without --headless.
/// @return the options and what was printed.
read_result read(const std::vector<std::string> &args, bool headless_default)
{
    std::vector<const char *> argv = {"orbbec_viewer"};
    for (const std::string &arg : args)
        argv.push_back(arg.c_str());
    std::ostringstream err;
    read_result result;
    result.options =
        gca_viewer::read_options(static_cast<int>(argv.size()), argv.data(), headless_default, err);
    result.printed = err.str();
    return result;
}

/// A command line the viewer has to accept, and the options it has to give.
struct accepted_case
{
    const char *what;              ///< the case, for the check lines
    std::vector<std::string> args; ///< the arguments after the program's name
    bool headless_default;         ///< the headless setting without --headless
    uint32_t width;                ///< the expected width
    uint32_t height;               ///< the expected height
    uint32_t fps;                  ///< the expected frame rate
    int max_frames;                ///< the expected frame limit
    bool headless;                 ///< the expected headless setting
    bool list_only;                ///< whether --list is expected
};

/// Each accepted case gives its options and prints nothing. The defaults, 640x400 at 30 fps, no
/// frame limit, no list, are the viewer's before this test existed [S: orbbec_viewer.cpp at
/// 069a61f].
void test_accepted()
{
    const accepted_case cases[] = {
        {"no argument", {}, false, 640, 400, 30, -1, false, false},
        {"no argument, no display", {}, true, 640, 400, 30, -1, true, false},
        {"every option",
         {"--list", "--headless", "--frames", "100", "--width", "1280", "--height", "800", "--fps",
          "15"},
         false,
         1280,
         800,
         15,
         100,
         true,
         true},
        {"--list alone", {"--list"}, false, 640, 400, 30, -1, false, true},
        {"--headless alone", {"--headless"}, false, 640, 400, 30, -1, true, false},
        {"a flag after a value",
         {"--frames", "5", "--headless"},
         false,
         640,
         400,
         30,
         5,
         true,
         false},
        {"a value that is no number, read as 0",
         {"--fps", "x"},
         false,
         640,
         400,
         0,
         -1,
         false,
         false},
        {"the last repeat wins",
         {"--width", "320", "--width", "800"},
         false,
         800,
         400,
         30,
         -1,
         false,
         false},
        {"an option's name taken as a value",
         {"--frames", "--list"},
         false,
         640,
         400,
         30,
         0,
         false,
         false},
    };
    for (const accepted_case &c : cases)
    {
        const read_result r = read(c.args, c.headless_default);
        if (r.options)
            std::printf("%s: %ux%u at %u fps, frames %d, headless %d, list %d\n", c.what,
                        r.options->width, r.options->height, r.options->fps, r.options->max_frames,
                        r.options->headless, r.options->list_only);
        const bool expected =
            r.options && r.options->width == c.width && r.options->height == c.height &&
            r.options->fps == c.fps && r.options->max_frames == c.max_frames &&
            r.options->headless == c.headless && r.options->list_only == c.list_only;
        check(expected, (std::string("accepted with the expected options: ") + c.what).c_str());
        check(r.printed.empty(), (std::string("nothing printed: ") + c.what).c_str());
    }
}

/// A command line the viewer has to refuse, and the reason line it prints before the usage.
struct refusal_case
{
    const char *what;              ///< the case, for the check lines
    std::vector<std::string> args; ///< the arguments after the program's name
    const char *reason;            ///< the reason line
};

/// Each refusal case gives no options and prints its reason line, then the usage line.
void test_refusals()
{
    const refusal_case cases[] = {
        {"--help", {"--help"}, "orbbec_viewer: unknown argument --help"},
        {"misspelt option", {"--frame", "10"}, "orbbec_viewer: unknown argument --frame"},
        {"positional argument", {"cloud.ply"}, "orbbec_viewer: unknown argument cloud.ply"},
        {"unknown after an option", {"--headless", "-l"}, "orbbec_viewer: unknown argument -l"},
        {"empty argument", {""}, "orbbec_viewer: unknown argument "},
        {"--frames without its value", {"--frames"}, "orbbec_viewer: --frames needs a value"},
        {"--fps without its value",
         {"--width", "320", "--fps"},
         "orbbec_viewer: --fps needs a value"},
    };
    for (const refusal_case &refusal : cases)
    {
        const read_result r = read(refusal.args, false);
        // Printed without its last line end, which check_refusal() adds.
        const std::string message = r.printed.substr(0, r.printed.find_last_not_of('\n') + 1);
        check_refusal(refusal.what, !r.options.has_value(), message,
                      r.printed ==
                          std::string(refusal.reason) + "\n" + gca_viewer::usage_line() + "\n");
    }
}

/// The usage line names the six options and nothing else; --help is no option.
void test_usage_line()
{
    check(gca_viewer::usage_line() == "usage: orbbec_viewer [--list] [--headless] [--frames N] "
                                      "[--width W] [--height H] [--fps F]",
          "usage line: the six options");
}
} // namespace

/// Runs the command-line checks and prints the options read, the refusals and the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    test_accepted();
    test_refusals();
    test_usage_line();
    return check_summary("test_viewer_options");
}
