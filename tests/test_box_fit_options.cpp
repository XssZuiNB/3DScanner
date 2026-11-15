// SPDX-License-Identifier: 0BSD
/// @file test_box_fit_options.cpp
/// Offline test of box_fit's command line (box_fit_options.hpp): the one path is read, a file
/// made in a temporary directory the test removes; an argument it does not know (--help, a
/// short option and an empty one among them), a second path, no path, a path that does not
/// exist and one under a file are refused with the reason and the usage, which box_fit turns
/// into exit 2. Needs no GPU and no camera.

#include "box_fit_options.hpp"
#include "check.hpp"
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
/// What read_box_fit_options() gave for one command line.
struct read_result
{
    std::optional<std::filesystem::path> model; ///< the model's path, or nothing when refused
    std::string printed;                        ///< what it printed to its error stream
};

/// Runs read_box_fit_options() on a command line.
/// @param args [in] the arguments after the program's name.
/// @return the path and what was printed.
read_result read(const std::vector<std::string> &args)
{
    std::vector<const char *> argv = {"box_fit"};
    for (const std::string &arg : args)
        argv.push_back(arg.c_str());
    std::ostringstream err;
    read_result result;
    result.model = gca::read_box_fit_options(static_cast<int>(argv.size()), argv.data(), err);
    result.printed = err.str();
    return result;
}

/// The path of a file that exists is read as it is given, and nothing is printed.
/// @param model [in] a file that exists.
void test_accepted(const std::filesystem::path &model)
{
    const read_result r = read({model.string()});
    std::printf("a file that exists: %s\n", r.model ? r.model->c_str() : "refused");
    check(r.model.has_value() && *r.model == model, "the model's path read as given");
    check(r.printed.empty(), "nothing printed for a file that exists");
}

/// A command line that has to be refused, and the reason it prints before the usage.
struct refusal_case
{
    std::string what;              ///< the case, for the check lines
    std::vector<std::string> args; ///< the arguments after the program's name
    std::string reason;            ///< the reason, after "box_fit: "
};

/// Each refusal case gives no path and prints "box_fit: <reason>", then the usage.
/// @param model [in] a file that exists.
void test_refusals(const std::filesystem::path &model)
{
    const std::string missing = (model.parent_path() / "missing.ply").string();
    const std::string under_a_file = (model / "model.ply").string();
    const std::vector<refusal_case> cases = {
        {"no argument", {}, "missing the object model's PLY"},
        {"--help", {"--help"}, "unknown argument --help"},
        {"a short option", {"-h"}, "unknown argument -h"},
        {"an option before the path", {"--verbose", model.string()}, "unknown argument --verbose"},
        {"an option after the path", {model.string(), "-v"}, "unknown argument -v"},
        {"an empty argument", {""}, "unknown argument "},
        {"a second path", {model.string(), "b.ply"}, "unknown argument b.ply"},
        {"a file that does not exist", {missing}, missing + ": no such file"},
        {"a path under a file", {under_a_file}, under_a_file + ": no such file"},
    };
    for (const refusal_case &refusal : cases)
    {
        const read_result r = read(refusal.args);
        // Printed without its last line end, which check_refusal() adds.
        const std::string message = r.printed.substr(0, r.printed.find_last_not_of('\n') + 1);
        check_refusal(refusal.what, !r.model.has_value(), message,
                      r.printed ==
                          "box_fit: " + refusal.reason + "\n" + gca::box_fit_usage() + "\n");
    }
}
} // namespace

/// Runs the command-line checks and prints the path read, the refusals and the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    const gca_test::temporary_directory directory("test_box_fit_options");
    const std::filesystem::path model = directory / "object_model.ply";
    gca_test::write_bytes(model, "ply\n");
    test_accepted(model);
    test_refusals(model);
    return check_summary("test_box_fit_options");
}
