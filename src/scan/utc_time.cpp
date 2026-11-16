// SPDX-License-Identifier: 0BSD
/// @file utc_time.cpp
/// A time of a scan in UTC and its text, by C++20's civil calendar (see utc_time.hpp).

#include "scan/utc_time.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace gca
{
namespace
{
using std::chrono::days;
using std::chrono::microseconds;
using std::chrono::sys_days;
using std::chrono::year;
using std::chrono::year_month_day;

/// The years the text holds: four digits, from the clock's epoch on [?: a choice: a scan's time
/// is never earlier, and four digits keep the text's length and its order the time's].
constexpr int first_year = 1970;
constexpr int last_year = 9999;

/// The text's length: 8 digits, 'T', 6 digits, '.', 6 digits, 'Z'.
constexpr std::size_t text_length = 23;

/// Reads a run of decimal digits.
/// @param text [in] the text.
/// @param from [in] the first digit's position.
/// @param count [in] the digits, at most 18, so that the value fits.
/// @return their value; nothing when a character there is no digit.
/// @pre from + count <= text.size(); utc_time_of() reads only within its 23 characters, whose
///      length it checks first. A run past the end is undefined behaviour.
std::optional<std::int64_t> digits_at(std::string_view text, std::size_t from, std::size_t count)
{
    std::int64_t value = 0;
    for (std::size_t k = from; k < from + count; k++)
    {
        if (text[k] < '0' || text[k] > '9')
            return std::nullopt;
        value = 10 * value + (text[k] - '0');
    }
    return value;
}
} // namespace

std::optional<std::string> utc_text(utc_time time) noexcept
{
    const sys_days day = std::chrono::floor<days>(time);
    const year_month_day date(day);
    if (date.year() < year(first_year) || date.year() > year(last_year))
        return std::nullopt;
    const std::chrono::hh_mm_ss<microseconds> clock(time - day);
    // Room for every int's digits, so that no field is cut: within the years above the text has
    // 23 characters.
    char text[128];
    std::snprintf(text, sizeof(text), "%04d%02u%02uT%02d%02d%02d.%06dZ", int(date.year()),
                  unsigned(date.month()), unsigned(date.day()), int(clock.hours().count()),
                  int(clock.minutes().count()), int(clock.seconds().count()),
                  int(clock.subseconds().count()));
    return std::string(text);
}

std::optional<utc_time> utc_time_of(std::string_view text) noexcept
{
    if (text.size() != text_length || text[8] != 'T' || text[15] != '.' || text[22] != 'Z')
        return std::nullopt;
    const std::optional<std::int64_t> y = digits_at(text, 0, 4);
    const std::optional<std::int64_t> mo = digits_at(text, 4, 2);
    const std::optional<std::int64_t> d = digits_at(text, 6, 2);
    const std::optional<std::int64_t> h = digits_at(text, 9, 2);
    const std::optional<std::int64_t> mi = digits_at(text, 11, 2);
    const std::optional<std::int64_t> s = digits_at(text, 13, 2);
    const std::optional<std::int64_t> us = digits_at(text, 16, 6);
    if (!y || !mo || !d || !h || !mi || !s || !us)
        return std::nullopt;
    const year_month_day date(year(static_cast<int>(*y)),
                              std::chrono::month(static_cast<unsigned>(*mo)),
                              std::chrono::day(static_cast<unsigned>(*d)));
    if (!date.ok() || *y < first_year || *h >= 24 || *mi >= 60 || *s >= 60)
        return std::nullopt;
    return utc_time(sys_days(date)) + std::chrono::hours(*h) + std::chrono::minutes(*mi) +
           std::chrono::seconds(*s) + microseconds(*us);
}
} // namespace gca
