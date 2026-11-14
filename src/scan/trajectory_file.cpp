// SPDX-License-Identifier: 0BSD
/// @file trajectory_file.cpp
/// A trajectory as text (see trajectory_file.hpp): append_number() (scan/file_io.hpp) writes
/// each number, std::from_chars reads it.

#include "scan/trajectory_file.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace gca
{
namespace
{
/// The trajectory's first line, without its line end: the format and its version.
constexpr std::string_view header = "# 3dscanner trajectory 1";

/// The mark of tracking loss, in place of a pose.
constexpr std::string_view lost_mark = "lost";

/// How far a quaternion's norm may lie from 1 [?: a choice]: a pose solve in double precision
/// leaves a unit quaternion's norm within a few 1e-16 of 1, and the file keeps every bit.
constexpr double norm_tolerance = 1e-9;

/// Reads a whole field as a number, without the locale. It reads back every bit of a double
/// that append_number() wrote [M 2026-11-14: 10^6 random doubles, g++ 11.4, a test program
/// outside the tree].
/// @tparam T std::uint64_t or double.
/// @param field [in] the field, without spaces.
/// @param value [out] the number; set only when the field is one.
/// @return whether the whole field is one number of T: decimal digits for std::uint64_t below
///         2^64, std::from_chars's general form for a double ("nan" and "inf" among them).
template <typename T> bool parse_number(std::string_view field, T &value)
{
    const std::from_chars_result end =
        std::from_chars(field.data(), field.data() + field.size(), value);
    return end.ec == std::errc() && end.ptr == field.data() + field.size() && !field.empty();
}

/// Checks that a pose is one the file holds.
/// @param pose [in] the pose.
/// @return the reason it is not; empty when its position is finite and its quaternion's norm
///         lies within norm_tolerance of 1.
std::string pose_refusal(const camera_pose &pose)
{
    if (!pose.position_m.allFinite())
        return "the position is not finite";
    const double norm = pose.orientation.norm();
    if (!(std::fabs(norm - 1.0) <= norm_tolerance))
    {
        std::string reason = "the quaternion's norm is ";
        append_number(reason, norm);
        return reason + ", not 1 within 1e-9";
    }
    return {};
}

/// Splits a line at each space.
/// @param line [in] the line, without its line end.
/// @return the fields; two spaces in a row give an empty field.
std::vector<std::string_view> fields_of(std::string_view line)
{
    std::vector<std::string_view> fields;
    std::size_t start = 0;
    for (;;)
    {
        const std::size_t space = line.find(' ', start);
        fields.push_back(line.substr(start, space - start));
        if (space == std::string_view::npos)
            return fields;
        start = space + 1;
    }
}

/// Reads a frame line.
/// @param line [in] the line, without its line end.
/// @return the frame, or nothing when the line has neither form: "<index> <timestamp_us> <tx>
///         <ty> <tz> <qx> <qy> <qz> <qw>" or "<index> <timestamp_us> lost". The pose is not
///         checked here.
std::optional<trajectory_entry> parse_line(std::string_view line)
{
    const std::vector<std::string_view> fields = fields_of(line);
    trajectory_entry entry{};
    if (fields.size() != 3 && fields.size() != 9)
        return std::nullopt;
    if (!parse_number(fields[0], entry.index) || !parse_number(fields[1], entry.timestamp_us))
        return std::nullopt;
    if (fields.size() == 3)
        return fields[2] == lost_mark ? std::optional(entry) : std::nullopt;
    double values[7];
    for (int k = 0; k < 7; k++)
    {
        if (!parse_number(fields[2 + k], values[k]))
            return std::nullopt;
    }
    // Eigen's quaternion constructor takes w first; the file holds x y z w.
    entry.pose = camera_pose{Eigen::Vector3d(values[0], values[1], values[2]),
                             Eigen::Quaterniond(values[6], values[3], values[4], values[5])};
    return entry;
}
} // namespace

write_result write_trajectory(const std::filesystem::path &path,
                              std::span<const trajectory_entry> trajectory) noexcept
{
    std::string text(header);
    text += '\n';
    for (std::size_t k = 0; k < trajectory.size(); k++)
    {
        const trajectory_entry &entry = trajectory[k];
        append_number(text, entry.index);
        text += ' ';
        append_number(text, entry.timestamp_us);
        if (!entry.pose)
        {
            text += ' ';
            text += lost_mark;
            text += '\n';
            continue;
        }
        const std::string refusal = pose_refusal(*entry.pose);
        if (!refusal.empty())
            return {false, path.string() + ": entry " + std::to_string(k) + " (frame index " +
                               std::to_string(entry.index) + "): " + refusal};
        const Eigen::Vector3d &t = entry.pose->position_m;
        const Eigen::Quaterniond &q = entry.pose->orientation;
        for (const double value : {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()})
        {
            text += ' ';
            append_number(text, value);
        }
        text += '\n';
    }
    return write_file(path, text);
}

read_result<std::vector<trajectory_entry>> read_trajectory(
    const std::filesystem::path &path) noexcept
{
    read_result<std::string> file = read_file(path);
    if (!file.value)
        return {std::nullopt, file.reason};
    const std::string_view text = *file.value;
    const auto refuse = [&](std::size_t line_number, const std::string &reason) {
        return read_result<std::vector<trajectory_entry>>{
            std::nullopt, path.string() + ": line " + std::to_string(line_number) + " " + reason};
    };
    std::vector<trajectory_entry> trajectory;
    std::size_t start = 0;
    for (std::size_t line_number = 1;; line_number++)
    {
        if (start == text.size() && line_number > 1)
            return {std::move(trajectory), {}};
        const std::size_t end = text.find('\n', start);
        if (end == std::string_view::npos)
            return refuse(line_number, line_number == 1 && start == text.size()
                                           ? "is missing: the file is empty"
                                           : "has no line end: the file ends within it");
        const std::string_view line = text.substr(start, end - start);
        start = end + 1;
        if (line_number == 1)
        {
            if (line != header)
                return refuse(1, "is not the header \"" + std::string(header) +
                                     "\" of this format and version");
            continue;
        }
        std::optional<trajectory_entry> entry = parse_line(line);
        if (!entry)
            return refuse(line_number, "is no frame line \"<index> <timestamp_us> <tx> <ty> "
                                       "<tz> <qx> <qy> <qz> <qw>\" or \"<index> <timestamp_us> "
                                       "lost\"");
        if (entry->pose)
        {
            const std::string refusal = pose_refusal(*entry->pose);
            if (!refusal.empty())
                return refuse(line_number, "holds a pose the file cannot hold: " + refusal);
        }
        trajectory.push_back(std::move(*entry));
    }
}
} // namespace gca
