// SPDX-License-Identifier: 0BSD
/// @file service_options.cpp
/// The scanner service's command line: one table of the options, the reading of each once, the
/// port's reading and the check of the scans directory (see service_options.hpp).

#include "service_options.hpp"

#include "output_directory.hpp"

#include <array>
#include <charconv>
#include <cstddef>

namespace gca
{
namespace
{
/// The options, in the usage's order; the index of each is its place in the values read.
enum option_index : std::size_t
{
    scans_option,
    port_option,
    address_option,
    page_option,
    keep_sequence_option,
    option_count,
};

/// One option of the command line.
struct option_spec
{
    const char *name;       ///< the option, "--scans"
    const char *value_name; ///< its value's name in the usage; nullptr for a flag
};

/// The options, at their option_index.
constexpr std::array<option_spec, option_count> option_table = {{
    {"--scans", "DIR"},
    {"--port", "N"},
    {"--address", "A"},
    {"--page", "DIR"},
    {"--keep-sequence", nullptr},
}};

/// Prints a refusal: the reason, then the usage.
/// @param err [in,out] where to print.
/// @param reason [in] why the command line is refused.
/// @param default_page [in] the page's directory without --page, for the usage.
/// @return nothing, what read_service_options() returns for a refused command line.
std::optional<service_options> refuse(std::ostream &err, const std::string &reason,
                                      const std::filesystem::path &default_page)
{
    err << service_line_prefix << reason << "\n" << service_usage(default_page) << std::endl;
    return std::nullopt;
}
} // namespace

std::optional<std::uint16_t> port_of(std::string_view text) noexcept
{
    // from_chars takes no sign and no white space, but a run of digits that stops early: the
    // whole text must be read.
    unsigned value = 0;
    const char *const end = text.data() + text.size();
    const std::from_chars_result read = std::from_chars(text.data(), end, value);
    if (text.empty() || read.ec != std::errc() || read.ptr != end || value < 1 || value > 65535)
        return std::nullopt;
    return static_cast<std::uint16_t>(value);
}

std::string service_usage(const std::filesystem::path &default_page)
{
    std::string usage = "usage: scanner_service";
    for (const option_spec &spec : option_table)
    {
        usage += std::string(" [") + spec.name;
        if (spec.value_name != nullptr)
            usage += std::string(" ") + spec.value_name;
        usage += "]";
    }
    return usage + "\n  --scans DIR: the scans directory, which exists and can be written; ~/" +
           std::string(default_scans_name) +
           " without it\n"
           "  --port N: the port the page is served on, 1 to 65535; " +
           std::to_string(default_service_port) +
           " without it\n"
           "  --address A: the numeric address the page is served on; " +
           std::string(default_service_address) +
           ", every interface of the board, without it; no login, for the operator's own "
           "network only\n"
           "  --page DIR: the page's directory; " +
           default_page.string() +
           " without it\n"
           "  --keep-sequence: a refined scan keeps its sequence; without it the sequence is "
           "removed once the scan is refined";
}

std::optional<service_options> read_service_options(int argc, const char *const argv[],
                                                    const char *home,
                                                    const std::filesystem::path &default_page,
                                                    std::ostream &err)
{
    std::array<std::optional<std::string>, option_count> given;
    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        std::size_t k = 0;
        while (k < option_count && arg != option_table[k].name)
            k++;
        if (k == option_count)
            return refuse(err, "unknown argument " + arg, default_page);
        if (given[k])
            return refuse(err, arg + " given twice", default_page);
        if (option_table[k].value_name == nullptr)
            given[k] = std::string();
        else if (i + 1 >= argc)
            return refuse(err, arg + " needs a value", default_page);
        else
            given[k] = argv[++i];
    }

    service_options read;
    read.port = default_service_port;
    if (const std::optional<std::string> &port = given[port_option])
    {
        const std::optional<std::uint16_t> number = port_of(*port);
        if (!number)
            return refuse(err, "--port " + *port + ": not a whole number from 1 to 65535",
                          default_page);
        read.port = *number;
    }
    read.address = given[address_option].value_or(std::string(default_service_address));
    read.page = given[page_option] ? std::filesystem::path(*given[page_option]) : default_page;
    read.keep_sequence = given[keep_sequence_option].has_value();
    if (given[scans_option])
        read.scans = *given[scans_option];
    else if (home != nullptr && *home != '\0')
        read.scans = std::filesystem::path(home) / default_scans_name;
    else
        return refuse(err, "no --scans, and HOME is not set", default_page);
    const std::string not_writable = writable_directory_refusal(read.scans);
    if (!not_writable.empty())
        return refuse(err, "--scans " + read.scans.string() + ": " + not_writable, default_page);
    return read;
}
} // namespace gca
