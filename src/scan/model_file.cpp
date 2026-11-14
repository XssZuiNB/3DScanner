// SPDX-License-Identifier: 0BSD
/// @file model_file.cpp
/// A model as binary little-endian PLY (see model_file.hpp). Each float is written and read
/// byte by byte, the least significant byte first, so the file's byte order does not depend on
/// the host's.

#include "scan/model_file.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace gca
{
namespace
{
/// The header's lines, without their line ends; the third line ends in the point count.
constexpr std::array<std::string_view, 12> header_lines = {"ply",
                                                           "format binary_little_endian 1.0",
                                                           "element vertex ",
                                                           "property float x",
                                                           "property float y",
                                                           "property float z",
                                                           "property float nx",
                                                           "property float ny",
                                                           "property float nz",
                                                           "property uchar red",
                                                           "property uchar green",
                                                           "property uchar blue"};

/// The header's last line, without its line end.
constexpr std::string_view end_header = "end_header";

/// The index of the header line that ends in the point count.
constexpr std::size_t count_line = 2;

/// The bytes of one point: six 32-bit floats and three 8-bit colors.
constexpr std::size_t point_bytes = 6 * sizeof(float) + 3;

/// How far a normal's length may lie from 1 [?: a choice]: eight float32 steps of 1.19e-7
/// [S: FLT_EPSILON]; a normal normalized in float32 lies within a few of them.
constexpr double normal_tolerance = 1e-6;

/// Checks that a normal is a unit vector.
/// @param normal [in] the normal.
/// @param index [in] the point's index, for the reason.
/// @return the reason it is not, naming the point; empty when its length, computed in double
///         precision, lies within normal_tolerance of 1.
std::string normal_refusal(const Eigen::Vector3f &normal, std::size_t index)
{
    const double length = normal.cast<double>().norm();
    if (std::fabs(length - 1.0) <= normal_tolerance)
        return {};
    std::string reason = "point " + std::to_string(index) + ": the normal's length is ";
    append_number(reason, length);
    return reason + ", not 1 within 1e-6";
}

/// Appends a float32, the least significant byte first.
/// @param bytes [in,out] the bytes appended to.
/// @param value [in] the float.
void append_float(std::string &bytes, float value)
{
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<char>((bits >> shift) & 0xffu));
}

/// Reads a float32 stored with its least significant byte first.
/// @param bytes [in] its four bytes.
/// @return the float.
float float_at(const char *bytes)
{
    std::uint32_t bits = 0;
    for (int k = 3; k >= 0; k--)
        bits = (bits << 8) | static_cast<unsigned char>(bytes[k]);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

/// Reads the header.
/// @param text [in] the file's bytes.
/// @param count [out] the point count the header declares.
/// @param data_start [out] the offset of the first byte after the header.
/// @return the reason the header is refused, naming its line; empty when it is the writer's.
std::string read_header(std::string_view text, std::size_t &count, std::size_t &data_start)
{
    std::size_t start = 0;
    for (std::size_t k = 0; k <= header_lines.size(); k++)
    {
        const std::string number = std::to_string(k + 1);
        const std::size_t end = text.find('\n', start);
        if (end == std::string_view::npos)
            return "the file ends within header line " + number;
        const std::string_view line = text.substr(start, end - start);
        start = end + 1;
        if (k == header_lines.size())
        {
            if (line != end_header)
                return "header line " + number + " is not \"" + std::string(end_header) + "\"";
            continue;
        }
        if (k != count_line)
        {
            if (line != header_lines[k])
                return "header line " + number + " is not \"" + std::string(header_lines[k]) + "\"";
            continue;
        }
        const std::string_view digits = line.substr(std::min(line.size(), header_lines[k].size()));
        const std::from_chars_result parsed =
            std::from_chars(digits.data(), digits.data() + digits.size(), count);
        if (!line.starts_with(header_lines[k]) || digits.empty() || parsed.ec != std::errc() ||
            parsed.ptr != digits.data() + digits.size())
            return "header line " + number + " is not \"" + std::string(header_lines[k]) +
                   "<count>\"";
    }
    data_start = start;
    return {};
}
} // namespace

write_result write_model(const std::filesystem::path &path,
                         std::span<const model_point> points) noexcept
{
    std::string bytes;
    for (std::size_t k = 0; k < header_lines.size(); k++)
    {
        bytes += header_lines[k];
        if (k == count_line)
            bytes += std::to_string(points.size());
        bytes += '\n';
    }
    bytes += end_header;
    bytes += '\n';
    bytes.reserve(bytes.size() + point_bytes * points.size());
    for (std::size_t index = 0; index < points.size(); index++)
    {
        const model_point &point = points[index];
        const std::string refusal = normal_refusal(point.normal, index);
        if (!refusal.empty())
            return {false, path.string() + ": " + refusal};
        for (int k = 0; k < 3; k++)
            append_float(bytes, point.position_m[k]);
        for (int k = 0; k < 3; k++)
            append_float(bytes, point.normal[k]);
        for (const std::uint8_t channel : point.color)
            bytes.push_back(static_cast<char>(channel));
    }
    return write_file(path, bytes);
}

read_result<std::vector<model_point>> read_model(const std::filesystem::path &path) noexcept
{
    read_result<std::string> file = read_file(path);
    if (!file.value)
        return {std::nullopt, file.reason};
    const std::string_view text = *file.value;
    std::size_t count = 0;
    std::size_t data_start = 0;
    const std::string refusal = read_header(text, count, data_start);
    if (!refusal.empty())
        return {std::nullopt, path.string() + ": " + refusal};
    const std::size_t data_bytes = text.size() - data_start;
    // Compared by division, so that a count near 2^64 cannot overflow the product.
    if (data_bytes % point_bytes != 0 || data_bytes / point_bytes != count)
        return {std::nullopt, path.string() + ": the header declares " + std::to_string(count) +
                                  " points of " + std::to_string(point_bytes) + " bytes, and " +
                                  std::to_string(data_bytes) + " bytes follow it"};
    std::vector<model_point> points(count);
    const char *record = text.data() + data_start;
    for (std::size_t index = 0; index < count; index++)
    {
        model_point &point = points[index];
        for (int k = 0; k < 3; k++)
            point.position_m[k] = float_at(record + 4 * k);
        for (int k = 0; k < 3; k++)
            point.normal[k] = float_at(record + 12 + 4 * k);
        for (int k = 0; k < 3; k++)
            point.color[k] = static_cast<std::uint8_t>(record[24 + k]);
        const std::string normal = normal_refusal(point.normal, index);
        if (!normal.empty())
            return {std::nullopt, path.string() + ": " + normal};
        record += point_bytes;
    }
    return {std::move(points), {}};
}
} // namespace gca
