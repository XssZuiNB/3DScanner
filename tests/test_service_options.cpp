// SPDX-License-Identifier: 0BSD
/// @file test_service_options.cpp
/// Offline test of the scanner service's command line (apps/service_options.hpp), on scans
/// directories made in a temporary directory the test removes:
/// - the defaults: the scans directory in the home directory, port 8080, address 0.0.0.0, the
///   build's page directory, the sequence not kept; and each option as given, in any order,
///   nothing printed; a scans directory that holds scans already is taken;
/// - refused with the reason and the usage, which the service turns into exit 2 before it opens
///   anything: an argument it does not know (--help, a short option, an empty one and a bare path
///   among them), an option without its value or given twice, a port that is no whole number from
///   1 to 65535 (0, 65536, a sign, a space, letters, empty), no --scans without a home directory,
///   and a scans directory that does not exist, is no directory or cannot be written;
/// - the reading of a port that scan_in_progress shares, at its edges.
/// Needs no GPU and no camera.

#include "check.hpp"
#include "service_options.hpp"
#include "test_files.hpp"

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
/// The page's directory without --page, as the build would give it.
const std::filesystem::path default_page = "/build/page";

/// What read_service_options() gave for one command line.
struct read_result
{
    std::optional<gca::service_options> options; ///< the options, or nothing when refused
    std::string printed;                         ///< what it printed to its error stream
};

/// Runs read_service_options() on a command line.
/// @param args [in] the arguments after the program's name.
/// @param home [in] the home directory; nullptr for none.
/// @return the options and what was printed.
read_result read(const std::vector<std::string> &args, const char *home)
{
    std::vector<const char *> argv = {"scanner_service"};
    for (const std::string &arg : args)
        argv.push_back(arg.c_str());
    std::ostringstream err;
    read_result result;
    result.options = gca::read_service_options(static_cast<int>(argv.size()), argv.data(), home,
                                               default_page, err);
    result.printed = err.str();
    return result;
}

/// The directories the cases name.
struct places
{
    std::filesystem::path home;    ///< a home directory with 3dscanner-scans in it
    std::filesystem::path scans;   ///< a scans directory that holds a scan already
    std::filesystem::path empty;   ///< a home directory without 3dscanner-scans
    std::filesystem::path file;    ///< a file
    std::filesystem::path locked;  ///< a directory that cannot be written
    std::filesystem::path missing; ///< a path that names nothing
};

/// The command lines that are read: the defaults, then each option as given in another order.
/// @param p [in] the places.
void test_accepted(const places &p)
{
    const std::string home = p.home.string();
    const read_result defaults = read({}, home.c_str());
    check(defaults.options.has_value() && defaults.printed.empty(),
          "the defaults: read, nothing printed");
    if (defaults.options)
    {
        const gca::service_options &o = *defaults.options;
        std::printf("the defaults: scans %s, address %s, port %u, page %s, keep %d\n",
                    o.scans.c_str(), o.address.c_str(), unsigned{o.port}, o.page.c_str(),
                    o.keep_sequence);
        check(o.scans == p.home / "3dscanner-scans", "the defaults: ~/3dscanner-scans");
        check(o.port == 8080 && o.address == "0.0.0.0",
              "the defaults: port 8080 on every interface, 0.0.0.0");
        check(o.page == default_page && !o.keep_sequence,
              "the defaults: the build's page directory, the sequence not kept");
    }
    const read_result given = read({"--keep-sequence", "--page", "/srv/page", "--address",
                                    "127.0.0.1", "--port", "65535", "--scans", p.scans.string()},
                                   nullptr);
    check(given.options.has_value() && given.printed.empty(),
          "every option, without a home directory: read, nothing printed");
    if (given.options)
    {
        const gca::service_options &o = *given.options;
        check(o.scans == p.scans && o.address == "127.0.0.1" && o.port == 65535 &&
                  o.page == "/srv/page" && o.keep_sequence,
              "every option: each as given");
    }
    const read_result low = read({"--port", "1"}, home.c_str());
    check(low.options && low.options->port == 1, "the least port, 1");
}

/// A command line that has to be refused, and the reason it prints before the usage.
struct refusal_case
{
    std::string what;              ///< the case, for the check lines
    std::vector<std::string> args; ///< the arguments after the program's name
    const char *home;              ///< the home directory; nullptr for none
    std::string reason;            ///< the reason, after "scanner_service: "
};

/// Each refusal case gives no options and prints "scanner_service: <reason>", then the usage.
/// @param p [in] the places.
void test_refusals(const places &p)
{
    const std::string home = p.home.string();
    const std::string empty = p.empty.string();
    const char *const h = home.c_str();
    const std::string port_reason = ": not a whole number from 1 to 65535";
    const std::vector<refusal_case> cases = {
        {"--help", {"--help"}, h, "unknown argument --help"},
        {"a short option", {"-p", "8080"}, h, "unknown argument -p"},
        {"an empty argument", {""}, h, "unknown argument "},
        {"a bare path", {p.scans.string()}, h, "unknown argument " + p.scans.string()},
        {"an option at the end without its value",
         {"--keep-sequence", "--port"},
         h,
         "--port needs a value"},
        {"--scans without its value", {"--scans"}, h, "--scans needs a value"},
        {"an option given twice", {"--port", "8080", "--port", "8081"}, h, "--port given twice"},
        {"a flag given twice",
         {"--keep-sequence", "--keep-sequence"},
         h,
         "--keep-sequence given twice"},
        {"port 0", {"--port", "0"}, h, "--port 0" + port_reason},
        {"port 65536", {"--port", "65536"}, h, "--port 65536" + port_reason},
        {"a port far out", {"--port", "4294967297"}, h, "--port 4294967297" + port_reason},
        {"a port with a sign", {"--port", "+80"}, h, "--port +80" + port_reason},
        {"a port below 0", {"--port", "-1"}, h, "--port -1" + port_reason},
        {"a port with a space", {"--port", "80 "}, h, "--port 80 " + port_reason},
        {"a port of letters", {"--port", "http"}, h, "--port http" + port_reason},
        {"an empty port", {"--port", ""}, h, "--port " + port_reason},
        {"no --scans and no home", {}, nullptr, "no --scans, and HOME is not set"},
        {"no --scans and an empty home", {}, "", "no --scans, and HOME is not set"},
        {"a home without its scans directory",
         {},
         empty.c_str(),
         "--scans " + (p.empty / "3dscanner-scans").string() + ": no such directory"},
        {"a scans directory that does not exist",
         {"--scans", p.missing.string()},
         h,
         "--scans " + p.missing.string() + ": no such directory"},
        {"a scans directory that is a file",
         {"--scans", p.file.string()},
         h,
         "--scans " + p.file.string() + ": is no directory"},
        {"a scans directory that cannot be written",
         {"--scans", p.locked.string()},
         h,
         "--scans " + p.locked.string() + ": cannot be written: Permission denied"},
    };
    for (const refusal_case &refusal : cases)
    {
        const read_result r = read(refusal.args, refusal.home);
        // Printed without its last line end, which check_refusal() adds.
        const std::string message = r.printed.substr(0, r.printed.find_last_not_of('\n') + 1);
        check_refusal(refusal.what, !r.options.has_value(), message,
                      r.printed == "scanner_service: " + refusal.reason + "\n" +
                                       gca::service_usage(default_page) + "\n");
    }
}

/// The reading of a port at its edges: 1, 65535 and leading zeros read; 0, 65536, a sign, white
/// space, letters and an empty text refused.
void test_port_of()
{
    check(gca::port_of("1") == std::uint16_t{1} && gca::port_of("65535") == std::uint16_t{65535} &&
              gca::port_of("08080") == std::uint16_t{8080},
          "port: 1, 65535 and 08080 read");
    for (const char *refused : {"0", "65536", "+1", "-1", " 1", "1 ", "0x50", "", "1e3"})
        check(!gca::port_of(refused), (std::string("port: refused \"") + refused + "\"").c_str());
}
} // namespace

/// Runs the cases on places made in a temporary directory.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    const gca_test::temporary_directory directory("test_service_options");
    const places p = {.home = directory / "home",
                      .scans = directory / "home" / "3dscanner-scans",
                      .empty = directory / "empty_home",
                      .file = directory / "a_file",
                      .locked = directory / "locked",
                      .missing = directory / "missing"};
    std::filesystem::create_directories(p.scans / "20261117T101112.123456Z");
    std::filesystem::create_directory(p.empty);
    gca_test::write_bytes(p.file, "not a directory");
    std::filesystem::create_directory(p.locked);
    std::filesystem::permissions(p.locked, std::filesystem::perms::owner_read |
                                               std::filesystem::perms::owner_exec);
    test_accepted(p);
    test_refusals(p);
    test_port_of();
    // Writable again, so that the temporary directory can be removed.
    std::filesystem::permissions(p.locked, std::filesystem::perms::owner_all);
    return check_summary("test_service_options");
}
