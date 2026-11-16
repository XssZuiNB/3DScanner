// SPDX-License-Identifier: 0BSD
/// @file utc_time.hpp
/// A time of a scan in UTC, to the microsecond, and its one text: ISO 8601's basic format with
/// six digits of the second's fraction, YYYYMMDDTHHMMSS.ffffffZ (docs/SCAN_FILES.md,
/// "Parameters"). A scan's parameters hold its start and stop in this text. The text is written
/// and read here and nowhere else, from 1970-01-01T00:00:00.000000Z to 9999-12-31T23:59:59.999999Z,
/// so that it has four digits of the year and sorts as its time does. Host only.

#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <string_view>

namespace gca
{
/// A time in UTC: microseconds since 1970-01-01T00:00:00Z on the system's clock, which counts no
/// leap second (std::chrono::system_clock, C++20).
using utc_time = std::chrono::sys_time<std::chrono::microseconds>;

/// Writes a time as its text.
/// @param time [in] the time, in UTC.
/// @return YYYYMMDDTHHMMSS.ffffffZ, 23 characters, e.g. 20261116T102030.123456Z; nothing when the
///         time lies before 1970-01-01T00:00:00Z or after 9999-12-31T23:59:59.999999Z.
/// @note Allocates the text. A failed allocation ends the program (std::terminate): the function
///       is noexcept.
std::optional<std::string> utc_text(utc_time time) noexcept;

/// Reads a time from its text, as utc_text() writes it.
/// @param text [in] the text.
/// @return the time; nothing when the text is not exactly 8 digits, 'T', 6 digits, '.', 6 digits
///         and 'Z', or names no time of the calendar from 1970 to 9999: a month outside 1 to 12, a
///         day the month does not have (29 February outside a leap year among them), an hour from
///         24, a minute or a second from 60 (no leap second).
/// @note Allocates nothing.
std::optional<utc_time> utc_time_of(std::string_view text) noexcept;
} // namespace gca
