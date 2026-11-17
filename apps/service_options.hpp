// SPDX-License-Identifier: 0BSD
/// @file service_options.hpp
/// The scanner service's command line (docs/SERVICE.md, "Options"): the scans directory, the port
/// and the address the page is served on, the page's directory and the switch that keeps each
/// refined scan's sequence, each with its default; the reader refuses an argument it does not
/// know, a missing value, an option given twice, a port out of range and a scans directory the
/// service cannot write, with the reason and the usage. Also the reading of a port, which
/// scan_in_progress shares. Host only, without the camera and the GPU, so that the offline tests
/// drive it.

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>

namespace gca
{
/// The port the page is served on without --port: 8080 [?: a choice, the spec's "The web layer and
/// the page"].
inline constexpr std::uint16_t default_service_port = 8080;

/// The address the page is served on without --address: every interface of the board, for the
/// operator's own network (R6).
inline constexpr std::string_view default_service_address = "0.0.0.0";

/// The scans directory's name in the home directory without --scans (A8).
inline constexpr std::string_view default_scans_name = "3dscanner-scans";

/// The start of every line the service prints, its log's (docs/SERVICE.md, "Log lines") and its
/// refusals', the program's name and a colon.
inline constexpr std::string_view service_line_prefix = "scanner_service: ";

/// The service's options.
struct service_options
{
    /// --scans: the directory each scan's directory goes into; it exists and can be written.
    std::filesystem::path scans;
    std::string address;        ///< --address: the numeric address the page is served on
    std::uint16_t port = 0;     ///< --port: the port, 1 to 65535 from the command line
    std::filesystem::path page; ///< --page: the page's directory (docs/ROUTES.md)
    /// --keep-sequence: whether a refined scan keeps its sequence (R18); off, it is removed once
    /// the scan is refined.
    bool keep_sequence = false;
};

/// Reads a port.
/// @param text [in] the port as a command line gives it.
/// @return the port; nothing when the text is not a whole number from 1 to 65535 in decimal
///         digits alone.
std::optional<std::uint16_t> port_of(std::string_view text) noexcept;

/// Builds the usage: its first line names every option, the lines after it each value and its
/// default.
/// @param default_page [in] the page's directory without --page.
/// @return the usage, lines joined by "\n", without a line end at the end.
std::string service_usage(const std::filesystem::path &default_page);

/// Reads the command line: --scans DIR, --port N, --address A, --page DIR and --keep-sequence, in
/// any order, each once. It looks at the scans directory and changes nothing on disk; the address
/// and the page's directory are the web layer's to refuse when it starts.
/// @param argc [in] the argument count.
/// @param argv [in] the arguments, argv[0] the program; the caller owns them.
/// @param home [in] the home directory, the environment's HOME; nullptr or empty when it is not
///        set; the caller owns it.
/// @param default_page [in] the page's directory without --page: the build's page directory.
/// @param err [in,out] where a refusal is printed: service_line_prefix, "scanner_service: ", the
///        reason, then the usage.
///        The reasons: "unknown argument <ARG>" (--help and a bare path among them); "<OPTION>
///        needs a value"; "<OPTION> given twice"; "--port <N>: not a whole number from 1 to
///        65535"; "no --scans, and HOME is not set"; "--scans <DIR>: " and the reason
///        writable_directory_refusal() gives (output_directory.hpp).
/// @return the options: without --scans <home>/3dscanner-scans, without --port 8080, without
///         --address 0.0.0.0, without --page default_page; nothing when the command line is
///         refused.
std::optional<service_options> read_service_options(int argc, const char *const argv[],
                                                    const char *home,
                                                    const std::filesystem::path &default_page,
                                                    std::ostream &err);
} // namespace gca
