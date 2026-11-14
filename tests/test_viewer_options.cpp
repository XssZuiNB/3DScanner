// SPDX-License-Identifier: 0BSD
/// @file test_viewer_options.cpp
/// Offline test of orbbec_viewer's command line (viewer_options.hpp): each accepted command line
/// gives its options, the defaults among them, flags at the end of the line too, and the
/// behaviours kept from before (a value read with atoi, the last repeat winning, an option's name
/// taken as a value); each kind of argument the viewer does not know, --help and an empty one
/// among them, and an option without its value are refused with the reason and the usage line,
/// which names the seven options. --sequence takes an empty directory that can be written, in a
/// temporary directory the test removes, and refuses one that does not exist, a file, a
/// directory that holds a sequence, another file or an empty directory, one that cannot be
/// listed, one that cannot be written and one that cannot be entered, each with its reason and
/// the usage line. Needs no GPU and no camera; run as root, the
/// directories without permissions would be accepted.

#include "check.hpp"
#include "test_files.hpp"
#include "viewer_options.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
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
        const bool expected = r.options && r.options->width == c.width &&
                              r.options->height == c.height && r.options->fps == c.fps &&
                              r.options->max_frames == c.max_frames &&
                              r.options->headless == c.headless &&
                              r.options->list_only == c.list_only && !r.options->sequence_directory;
        check(expected, (std::string("accepted with the expected options: ") + c.what).c_str());
        check(r.printed.empty(), (std::string("nothing printed: ") + c.what).c_str());
    }
}

/// Checks that a command line is refused with a reason line and the usage line.
/// @param what [in] the case, for the check lines.
/// @param args [in] the arguments after the program's name.
/// @param reason [in] the reason line expected.
void check_refused(const std::string &what, const std::vector<std::string> &args,
                   const std::string &reason)
{
    const read_result r = read(args, false);
    // Printed without its last line end, which check_refusal() adds.
    const std::string message = r.printed.substr(0, r.printed.find_last_not_of('\n') + 1);
    check_refusal(what, !r.options.has_value(), message,
                  r.printed == reason + "\n" + gca_viewer::usage_line() + "\n");
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
        {"--sequence without its directory",
         {"--headless", "--sequence"},
         "orbbec_viewer: --sequence needs a value"},
    };
    for (const refusal_case &refusal : cases)
        check_refused(refusal.what, refusal.args, refusal.reason);
}

/// --sequence: an empty directory that can be written is taken, the last one given counting;
/// every other directory is refused with its reason, which names it, and the usage line.
void test_sequence_directory()
{
    const gca_test::temporary_directory root("test_viewer_options");
    const std::filesystem::path empty = root / "empty";
    const std::filesystem::path missing = root / "missing";
    const std::filesystem::path file = root / "a_file";
    const std::filesystem::path holding_sequence = root / "holding_sequence";
    const std::filesystem::path holding_file = root / "holding_file";
    const std::filesystem::path holding_directory = root / "holding_directory";
    const std::filesystem::path unlisted = root / "unlisted";
    const std::filesystem::path read_only = root / "read_only";
    const std::filesystem::path unentered = root / "unentered";
    for (const std::filesystem::path &directory :
         {empty, holding_sequence, holding_file, holding_directory, unlisted, read_only, unentered})
        std::filesystem::create_directory(directory);
    gca_test::write_bytes(file, "not a directory");
    gca_test::write_bytes(holding_sequence / "frames.sequence", "a sequence from before");
    gca_test::write_bytes(holding_file / "notes.txt", "a file of another name");
    std::filesystem::create_directory(holding_directory / "older");
    // Permissions: unlisted can be written and entered but not read, read_only read and entered
    // but not written, unentered read and written but not entered, so that no file can be
    // created in it.
    using std::filesystem::perms;
    std::filesystem::permissions(unlisted, perms::owner_write | perms::owner_exec);
    std::filesystem::permissions(read_only, perms::owner_read | perms::owner_exec);
    std::filesystem::permissions(unentered, perms::owner_read | perms::owner_write);

    const read_result taken =
        read({"--headless", "--sequence", empty.string(), "--frames", "5"}, false);
    std::printf("--sequence with an empty directory: %s\n",
                taken.options && taken.options->sequence_directory
                    ? taken.options->sequence_directory->c_str()
                    : "refused");
    check(taken.options && taken.options->sequence_directory == empty && taken.options->headless &&
              taken.options->max_frames == 5,
          "--sequence with an empty directory: taken with the other options");
    check(taken.printed.empty(), "--sequence with an empty directory: nothing printed");
    const read_result last =
        read({"--sequence", missing.string(), "--sequence", empty.string()}, false);
    check(last.options && last.options->sequence_directory == empty && last.printed.empty(),
          "--sequence given twice: the last one counts, and only it is checked");

    const std::string prefix = "orbbec_viewer: --sequence ";
    check_refused("--sequence with a missing directory", {"--sequence", missing.string()},
                  prefix + missing.string() + ": no such directory");
    check_refused("--sequence with an empty argument", {"--sequence", ""},
                  prefix + ": no such directory");
    check_refused("--sequence with a file", {"--sequence", file.string()},
                  prefix + file.string() + ": is no directory");
    check_refused("--sequence with a directory that holds a sequence",
                  {"--sequence", holding_sequence.string(), "--headless"},
                  prefix + holding_sequence.string() + ": holds something already");
    check_refused("--sequence with a directory that holds a file of another name",
                  {"--sequence", holding_file.string()},
                  prefix + holding_file.string() + ": holds something already");
    check_refused("--sequence with a directory that holds an empty directory",
                  {"--sequence", holding_directory.string()},
                  prefix + holding_directory.string() + ": holds something already");
    check_refused("--sequence with a directory that cannot be listed",
                  {"--sequence", unlisted.string()},
                  prefix + unlisted.string() + ": cannot be read: Permission denied");
    check_refused("--sequence with a directory that cannot be written",
                  {"--sequence", read_only.string()},
                  prefix + read_only.string() + ": cannot be written: Permission denied");
    check_refused("--sequence with a directory that cannot be entered",
                  {"--sequence", unentered.string()},
                  prefix + unentered.string() + ": cannot be written: Permission denied");

    // Given back, so that the temporary directory can be removed.
    for (const std::filesystem::path &directory : {unlisted, read_only, unentered})
        std::filesystem::permissions(directory, perms::owner_all);
}

/// The usage line names the seven options and nothing else; --help is no option.
void test_usage_line()
{
    check(gca_viewer::usage_line() == "usage: orbbec_viewer [--list] [--headless] [--frames N] "
                                      "[--width W] [--height H] [--fps F] [--sequence DIR]",
          "usage line: the seven options");
}
} // namespace

/// Runs the command-line checks and prints the options read, the refusals and the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    test_accepted();
    test_refusals();
    test_sequence_directory();
    test_usage_line();
    return check_summary("test_viewer_options");
}
