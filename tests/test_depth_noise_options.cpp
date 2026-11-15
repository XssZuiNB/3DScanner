// SPDX-License-Identifier: 0BSD
/// @file test_depth_noise_options.cpp
/// Offline test of depth_noise's command line (depth_noise_options.hpp): the one directory is
/// read, one made in a temporary directory the test removes; an argument it does not know
/// (--help, a short option and an empty one among them), a second directory, none, a directory
/// that does not exist and one under a file are refused with the reason and the usage, which
/// depth_noise turns into exit 2. Needs no GPU and no camera.

#include "check.hpp"
#include "depth_noise_options.hpp"
#include "test_files.hpp"

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
/// What read_depth_noise_options() gave for one command line.
struct read_result
{
    std::optional<std::filesystem::path> directory; ///< the directory, or nothing when refused
    std::string printed;                            ///< what it printed to its error stream
};

/// Runs read_depth_noise_options() on a command line.
/// @param args [in] the arguments after the program's name.
/// @return the directory and what was printed.
read_result read(const std::vector<std::string> &args)
{
    std::vector<const char *> argv = {"depth_noise"};
    for (const std::string &arg : args)
        argv.push_back(arg.c_str());
    std::ostringstream err;
    read_result result;
    result.directory =
        gca::read_depth_noise_options(static_cast<int>(argv.size()), argv.data(), err);
    result.printed = err.str();
    return result;
}

/// The path of a directory that exists is read as it is given, and nothing is printed.
/// @param directory [in] a directory that exists.
void test_accepted(const std::filesystem::path &directory)
{
    const read_result r = read({directory.string()});
    std::printf("a directory that exists: %s\n", r.directory ? r.directory->c_str() : "refused");
    check(r.directory.has_value() && *r.directory == directory,
          "the directory's path read as given");
    check(r.printed.empty(), "nothing printed for a directory that exists");
}

/// A command line that has to be refused, and the reason it prints before the usage.
struct refusal_case
{
    std::string what;              ///< the case, for the check lines
    std::vector<std::string> args; ///< the arguments after the program's name
    std::string reason;            ///< the reason, after "depth_noise: "
};

/// Each refusal case gives no directory and prints "depth_noise: <reason>", then the usage.
/// @param directory [in] a directory that exists.
/// @param file [in] a file that exists.
void test_refusals(const std::filesystem::path &directory, const std::filesystem::path &file)
{
    const std::string missing = (directory / "missing").string();
    const std::string under_a_file = (file / "sequence").string();
    const std::vector<refusal_case> cases = {
        {"no argument", {}, "missing the sequence's directory"},
        {"--help", {"--help"}, "unknown argument --help"},
        {"a short option", {"-h"}, "unknown argument -h"},
        {"an option before the directory",
         {"--verbose", directory.string()},
         "unknown argument --verbose"},
        {"an option after the directory", {directory.string(), "-v"}, "unknown argument -v"},
        {"an empty argument", {""}, "unknown argument "},
        {"a second directory", {directory.string(), "b"}, "unknown argument b"},
        {"a directory that does not exist", {missing}, missing + ": no such directory"},
        {"a path under a file", {under_a_file}, under_a_file + ": no such directory"},
    };
    for (const refusal_case &refusal : cases)
    {
        const read_result r = read(refusal.args);
        // Printed without its last line end, which check_refusal() adds.
        const std::string message = r.printed.substr(0, r.printed.find_last_not_of('\n') + 1);
        check_refusal(refusal.what, !r.directory.has_value(), message,
                      r.printed == "depth_noise: " + refusal.reason + "\n" +
                                       gca::depth_noise_usage() + "\n");
    }
}
} // namespace

/// Runs the command-line checks and prints the directory read, the refusals and the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    const gca_test::temporary_directory directory("test_depth_noise_options");
    const std::filesystem::path file = directory / "frames.sequence";
    gca_test::write_bytes(file, "3dscanner sequence");
    test_accepted(directory.path());
    test_refusals(directory.path(), file);
    return check_summary("test_depth_noise_options");
}
