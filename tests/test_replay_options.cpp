// SPDX-License-Identifier: 0BSD
/// @file test_replay_options.cpp
/// Offline test of the replay tool's command line (apps/replay_options.hpp), on a sequence's
/// directory, a parameters file and scans directories made in a temporary directory the test
/// removes:
/// - read: the sequence's file, the object class or the parameters file, the scans directory and
///   the time lines' request, each as given, in any order, and nothing printed;
/// - refused with the reason and the usage, which the replay turns into exit 2: an argument it
///   does not know (--help, a short option, an empty one and a bare path among them), an option
///   without its value or given twice, a missing or unknown object class, both an object class
///   and a parameters file, a parameters file that does not exist, a missing sequence (no
///   --sequence, or no frames.sequence in its directory), a missing --scans, and a scans
///   directory that does not exist, is no directory or holds something already;
/// - the set the scan runs with: each object class's own by its name, a kept scan's parameters
///   file's object class and set exactly, and a file its reader refuses, with the reason.
/// Needs no GPU and no camera.

#include "check.hpp"
#include "replay_options.hpp"
#include "scan/object_class.hpp"
#include "scan/parameters_file.hpp"
#include "scan/utc_time.hpp"
#include "test_files.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

using gca_test::check;
using gca_test::check_refusal;
using gca_test::check_summary;

namespace
{
/// What read_replay_options() gave for one command line.
struct read_result
{
    std::optional<gca::replay_options> options; ///< the options, or nothing when refused
    std::string printed;                        ///< what it printed to its error stream
};

/// Runs read_replay_options() on a command line.
/// @param args [in] the arguments after the program's name.
/// @return the options and what was printed.
read_result read(const std::vector<std::string> &args)
{
    std::vector<const char *> argv = {"replay"};
    for (const std::string &arg : args)
        argv.push_back(arg.c_str());
    std::ostringstream err;
    read_result result;
    result.options = gca::read_replay_options(static_cast<int>(argv.size()), argv.data(), err);
    result.printed = err.str();
    return result;
}

/// The files and directories the cases name.
struct places
{
    std::filesystem::path sequence;    ///< a directory that holds frames.sequence
    std::filesystem::path parameters;  ///< a file that stands for a kept scan's parameters
    std::filesystem::path scans;       ///< an empty directory
    std::filesystem::path holding;     ///< a directory that holds a file
    std::filesystem::path no_sequence; ///< a directory without frames.sequence
};

/// The command lines that are read: each value as given, the object class by its name or the
/// parameters file by its path, --times or not, in any order; nothing printed.
/// @param p [in] the places.
void test_accepted(const places &p)
{
    const read_result by_class = read({"--sequence", p.sequence.string(), "--object-class",
                                       "large_object", "--scans", p.scans.string()});
    check(by_class.options.has_value() && by_class.printed.empty(),
          "an object class: read, nothing printed");
    if (by_class.options)
    {
        const gca::replay_options &o = *by_class.options;
        check(o.sequence == p.sequence / "frames.sequence",
              "an object class: the sequence's file in its directory");
        check(std::holds_alternative<gca::object_class>(o.set_from) &&
                  std::get<gca::object_class>(o.set_from) == gca::object_class::large_object,
              "an object class: large_object");
        check(o.scans == p.scans && !o.times, "an object class: the scans directory, no times");
    }
    const read_result by_file = read({"--times", "--scans", p.scans.string(), "--parameters",
                                      p.parameters.string(), "--sequence", p.sequence.string()});
    check(by_file.options.has_value() && by_file.printed.empty(),
          "a parameters file, in another order: read, nothing printed");
    if (by_file.options)
    {
        const gca::replay_options &o = *by_file.options;
        check(std::holds_alternative<std::filesystem::path>(o.set_from) &&
                  std::get<std::filesystem::path>(o.set_from) == p.parameters,
              "a parameters file: its path as given");
        check(o.times && o.scans == p.scans && o.sequence == p.sequence / "frames.sequence",
              "a parameters file: the times asked, the scans directory, the sequence");
    }
    const read_result tabletop = read({"--object-class", "tabletop_object", "--sequence",
                                       p.sequence.string(), "--scans", p.scans.string()});
    check(tabletop.options &&
              std::holds_alternative<gca::object_class>(tabletop.options->set_from) &&
              std::get<gca::object_class>(tabletop.options->set_from) ==
                  gca::object_class::tabletop_object,
          "an object class: tabletop_object");
}

/// A command line that has to be refused, and the reason it prints before the usage.
struct refusal_case
{
    std::string what;              ///< the case, for the check lines
    std::vector<std::string> args; ///< the arguments after the program's name
    std::string reason;            ///< the reason, after "replay: "
};

/// Each refusal case gives no options and prints "replay: <reason>", then the usage.
/// @param p [in] the places.
void test_refusals(const places &p)
{
    const std::string seq = p.sequence.string();
    const std::string par = p.parameters.string();
    const std::string out = p.scans.string();
    const std::string missing = (p.scans / "missing").string();
    const std::string file = (p.sequence / "frames.sequence").string();
    const std::vector<std::string> sequence = {"--sequence", seq};
    const std::vector<std::string> scans = {"--scans", out};
    const auto line = [](std::vector<std::string> a, const std::vector<std::string> &b,
                         const std::vector<std::string> &c) {
        a.insert(a.end(), b.begin(), b.end());
        a.insert(a.end(), c.begin(), c.end());
        return a;
    };
    const std::vector<std::string> by_class = {"--object-class", "tabletop_object"};
    const std::vector<refusal_case> cases = {
        {"--help", {"--help"}, "unknown argument --help"},
        {"a short option", line({"-h"}, sequence, scans), "unknown argument -h"},
        {"an empty argument", line({""}, sequence, scans), "unknown argument "},
        {"a bare path", line(by_class, sequence, {seq}), "unknown argument " + seq},
        {"an option at the end without its value", line(by_class, sequence, {"--scans"}),
         "--scans needs a value"},
        {"an option given twice", line(by_class, sequence, line(scans, {"--sequence", seq}, {})),
         "--sequence given twice"},
        {"--times given twice", line(by_class, sequence, line(scans, {"--times", "--times"}, {})),
         "--times given twice"},
        {"no object class and no parameters", line(sequence, scans, {}),
         "missing the set: --object-class or --parameters"},
        {"both an object class and parameters",
         line(by_class, sequence, line(scans, {"--parameters", par}, {})),
         "--object-class and --parameters exclude each other"},
        {"an unknown object class", line({"--object-class", "chair"}, sequence, scans),
         "--object-class chair: no such object class; tabletop_object or large_object"},
        {"a parameters file that does not exist", line({"--parameters", missing}, sequence, scans),
         "--parameters " + missing + ": no such file"},
        {"no --sequence", line(by_class, scans, {}), "missing --sequence"},
        {"a sequence's directory without its file",
         line(by_class, {"--sequence", p.no_sequence.string()}, scans),
         "--sequence " + p.no_sequence.string() + ": " +
             (p.no_sequence / "frames.sequence").string() + ": no such file"},
        {"a sequence's directory that does not exist",
         line(by_class, {"--sequence", missing}, scans),
         "--sequence " + missing + ": " + missing + "/frames.sequence: no such file"},
        {"no --scans", line(by_class, sequence, {}), "missing --scans"},
        {"a scans directory that does not exist", line(by_class, sequence, {"--scans", missing}),
         "--scans " + missing + ": no such directory"},
        {"a scans directory that is a file", line(by_class, sequence, {"--scans", file}),
         "--scans " + file + ": is no directory"},
        {"a scans directory that holds something",
         line(by_class, sequence, {"--scans", p.holding.string()}),
         "--scans " + p.holding.string() + ": holds something already"},
    };
    for (const refusal_case &refusal : cases)
    {
        const read_result r = read(refusal.args);
        // Printed without its last line end, which check_refusal() adds.
        const std::string message = r.printed.substr(0, r.printed.find_last_not_of('\n') + 1);
        check_refusal(refusal.what, !r.options.has_value(), message,
                      r.printed == "replay: " + refusal.reason + "\n" + gca::replay_usage() + "\n");
    }
}

/// The set a scan runs with (replay_set_of()): an object class's own, by the name on the command
/// line; a kept scan's parameters file's object class and set exactly, nothing printed; a file its
/// reader refuses gives nothing and "replay: " with the reader's reason, naming the file.
/// @param p [in] the places; p.parameters holds "{}", which the reader refuses.
/// @param directory [in] the test's directory, where the kept parameters are written.
void test_set_of(const places &p, const std::filesystem::path &directory)
{
    const auto set_of = [](const gca::replay_options &options, std::string &printed) {
        std::ostringstream err;
        const std::optional<gca::replay_set> set = gca::replay_set_of(options, err);
        printed = err.str();
        return set;
    };
    std::string printed;
    check(gca::parameters_of(gca::object_class::tabletop_object) !=
              gca::parameters_of(gca::object_class::large_object),
          "set: the two object classes' sets differ, so a swap shows");
    for (const gca::object_class kind :
         {gca::object_class::tabletop_object, gca::object_class::large_object})
    {
        const std::string name(gca::name_of(kind));
        const read_result r = read({"--sequence", p.sequence.string(), "--object-class", name,
                                    "--scans", p.scans.string()});
        const std::optional<gca::replay_set> set =
            r.options ? set_of(*r.options, printed) : std::nullopt;
        check(set && set->kind == kind && set->values == gca::parameters_of(kind) &&
                  printed.empty(),
              ("set: --object-class " + name + " gives its own set").c_str());
    }

    // A kept scan's parameters of the large object, with values no object class has.
    gca::object_class_parameters values = gca::parameters_of(gca::object_class::large_object);
    values.full_voxel_size_m = 0.005;
    values.memory_budget_bytes = 123'456'789;
    const gca::utc_time time =
        std::chrono::floor<std::chrono::microseconds>(std::chrono::system_clock::now());
    const gca::scan_parameters kept = {"a commit",
                                       gca::object_class::large_object,
                                       values,
                                       {{640, 400}, {306.18f, 306.14f, 317.41f, 199.13f}, 0.001f},
                                       {"sequence of a kept scan", "", ""},
                                       {},
                                       time,
                                       time,
                                       {10, 8, 2, 0},
                                       gca::scan_end::source_end,
                                       {}};
    const std::filesystem::path kept_file = directory / "kept.json";
    check(gca::write_parameters(kept_file, kept).written, "set: test setup, the kept parameters");
    const read_result by_file = read({"--sequence", p.sequence.string(), "--parameters",
                                      kept_file.string(), "--scans", p.scans.string()});
    const std::optional<gca::replay_set> from_file =
        by_file.options ? set_of(*by_file.options, printed) : std::nullopt;
    check(from_file && from_file->kind == gca::object_class::large_object &&
              from_file->values == values && printed.empty(),
          "set: a parameters file gives its object class and set exactly");

    const read_result refused = read({"--sequence", p.sequence.string(), "--parameters",
                                      p.parameters.string(), "--scans", p.scans.string()});
    const std::optional<gca::replay_set> none =
        refused.options ? set_of(*refused.options, printed) : std::nullopt;
    check_refusal("set: a parameters file its reader refuses",
                  refused.options.has_value() && !none.has_value(),
                  printed.substr(0, printed.find_last_not_of('\n') + 1),
                  printed.rfind("replay: " + p.parameters.string() + ": ", 0) == 0 &&
                      printed.find("usage:") == std::string::npos);
}

/// The usage names every option and each object class.
void test_usage()
{
    const std::string usage = gca::replay_usage();
    std::printf("%s\n", usage.c_str());
    check(usage.rfind("usage: replay --sequence DIR (--object-class CLASS | --parameters FILE) "
                      "--scans OUT [--times]\n",
                      0) == 0,
          "the usage's first line");
    for (const char *word : {"tabletop_object", "large_object", "frames.sequence", "--times"})
        check(usage.find(word) != std::string::npos,
              (std::string("the usage names ") + word).c_str());
}
} // namespace

/// Runs the command-line checks and prints the refusals, the usage and the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    const gca_test::temporary_directory directory("test_replay_options");
    places p;
    p.sequence = directory / "sequence";
    p.no_sequence = directory / "no_sequence";
    p.scans = directory / "scans";
    p.holding = directory / "holding";
    p.parameters = directory / "parameters.json";
    for (const std::filesystem::path &made : {p.sequence, p.no_sequence, p.scans, p.holding})
        std::filesystem::create_directory(made);
    gca_test::write_bytes(p.sequence / "frames.sequence", "3dscanner sequence");
    gca_test::write_bytes(p.holding / "scan", "a scan from before");
    gca_test::write_bytes(p.parameters, "{}");
    test_accepted(p);
    test_refusals(p);
    test_set_of(p, directory.path());
    test_usage();
    return check_summary("test_replay_options");
}
