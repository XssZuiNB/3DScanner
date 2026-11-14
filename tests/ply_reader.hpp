// SPDX-License-Identifier: 0BSD
/// @file ply_reader.hpp
/// The offline tests' reader of the colored-ICP fixtures (data/frag_115.ply, frag_116.ply) and of
/// the scanner's models (docs/SCAN_FILES.md), apart from the scanner's code:
/// binary little-endian PLY whose first element is the vertex, with float x, y, z, nx, ny, nz
/// and uchar red, green, blue in that order [S: the fixtures' headers]. It reads the vertices and
/// stops; the elements after them (the fixtures' faces) are declared in the header and never
/// read. A header outside that layout is refused with a message naming its line. Header only;
/// each test program is one translation unit.

#pragma once

#include "geometry/type.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <istream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace gca_test
{
/// The vertices of a PLY file, in file order.
struct ply_vertices
{
    std::vector<gca::point_t> points; ///< coordinates in meters, colors in [0, 1], inactive
    std::vector<float3> normals;      ///< the file's normals, one per point
};

/// Decodes a little-endian float32 whatever the host's byte order.
/// @param bytes the four bytes, the least significant first.
/// @return the float.
inline float little_endian_float(const unsigned char *bytes)
{
    const std::uint32_t bits = std::uint32_t{bytes[0]} | std::uint32_t{bytes[1]} << 8 |
                               std::uint32_t{bytes[2]} << 16 | std::uint32_t{bytes[3]} << 24;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

/// Quotes a header line for a message, each byte outside printable ASCII written as \xNN, so a
/// stray '\r' or binary data shows.
/// @param line the header line.
/// @return the line in double quotes.
inline std::string printable(const std::string &line)
{
    static const char hex[] = "0123456789abcdef";
    std::string text = "\"";
    for (const char c : line)
    {
        const unsigned char byte = static_cast<unsigned char>(c);
        if (byte >= 0x20 && byte < 0x7f)
            text += c;
        else
            text += std::string("\\x") + hex[byte >> 4] + hex[byte & 0xf];
    }
    return text + "\"";
}

/// Parses the vertex element's header line, "element vertex <count>".
/// @param line the header line.
/// @return the count, or nothing when the line has another form or the count is not a decimal
///     number of at most gca::index_t's maximum.
inline std::optional<std::int32_t> parse_vertex_count(const std::string &line)
{
    const std::string prefix = "element vertex ";
    constexpr size_t max_digits = 10;
    if (line.rfind(prefix, 0) != 0 || line.size() == prefix.size() ||
        line.size() > prefix.size() + max_digits)
        return std::nullopt;
    std::int64_t count = 0;
    for (size_t k = prefix.size(); k < line.size(); k++)
    {
        if (line[k] < '0' || line[k] > '9')
            return std::nullopt;
        count = 10 * count + (line[k] - '0');
    }
    if (count > std::numeric_limits<gca::index_t>::max())
        return std::nullopt;
    return static_cast<std::int32_t>(count);
}

/// Reads a PLY header in the fixtures' layout. Each line has to match that layout exactly (no
/// comment lines, '\n' line ends); after the vertex's nine properties only the element and
/// property lines of later elements may follow before "end_header".
/// @param in the file, opened in binary mode, at its start; left at the first byte of data.
/// @param error the reason, set when the header is refused.
/// @return the vertex count, or nothing when the header has another layout.
inline std::optional<std::int32_t> read_ply_header(std::istream &in, std::string &error)
{
    static const char *const vertex_properties[] = {
        "property float x",   "property float y",     "property float z",
        "property float nx",  "property float ny",    "property float nz",
        "property uchar red", "property uchar green", "property uchar blue"};
    int line_number = 0;
    bool at_end = false;
    std::string line;
    const auto next_line = [&]() {
        line_number++;
        at_end = !std::getline(in, line);
        return !at_end;
    };
    const auto refusal = [&](const std::string &expected) {
        error = "PLY header line " + std::to_string(line_number) + ": expected " + expected +
                ", got " + (at_end ? std::string("the end of the file") : printable(line));
        return std::nullopt;
    };

    if (!next_line() || line != "ply")
        return refusal("\"ply\"");
    if (!next_line() || line != "format binary_little_endian 1.0")
        return refusal("\"format binary_little_endian 1.0\"");
    const std::optional<std::int32_t> count = next_line() ? parse_vertex_count(line) : std::nullopt;
    if (!count)
        return refusal("\"element vertex <count>\", the count at most 2147483647");
    for (const char *property : vertex_properties)
    {
        if (!next_line() || line != property)
            return refusal("\"" + std::string(property) + "\"");
    }
    // A property line before the next element line would be a tenth vertex property.
    bool later_element = false;
    while (next_line() && line != "end_header")
    {
        if (line.rfind("element ", 0) == 0)
            later_element = true;
        else if (!later_element || line.rfind("property ", 0) != 0)
            return refusal(later_element ? "\"element ...\", \"property ...\" or \"end_header\""
                                         : "\"element ...\" or \"end_header\"");
    }
    if (at_end)
        return refusal("\"end_header\"");
    return count;
}

/// Reads the vertex data that follows a header in the fixtures' layout. Colors are scaled by
/// 1/255 as create_from_pcl() did, CUDA_3D's way from these files to a cloud [S: point_cloud.cu
/// at 2d6eaa2].
/// @param in the file, at the first byte of data.
/// @param count the vertex count the header declared.
/// @param error the reason, set when the file ends within the vertex data.
/// @return the vertices, or nothing when the file ends within them.
inline std::optional<ply_vertices> read_ply_vertices(std::istream &in, std::int32_t count,
                                                     std::string &error)
{
    constexpr size_t vertex_bytes = 6 * sizeof(float) + 3;
    ply_vertices vertices;
    for (std::int32_t k = 0; k < count; k++)
    {
        unsigned char record[vertex_bytes];
        if (!in.read(reinterpret_cast<char *>(record), vertex_bytes))
        {
            error = "PLY data: the file ends within vertex " + std::to_string(k) + " of " +
                    std::to_string(count);
            return std::nullopt;
        }
        gca::point_t point{};
        point.coordinates =
            make_float3(little_endian_float(record), little_endian_float(record + 4),
                        little_endian_float(record + 8));
        point.color = gca::color3{static_cast<float>(record[24]) * (1.0f / 255.0f),
                                  static_cast<float>(record[25]) * (1.0f / 255.0f),
                                  static_cast<float>(record[26]) * (1.0f / 255.0f)};
        point.property = gca::point_property::inactive;
        vertices.points.push_back(point);
        vertices.normals.push_back(make_float3(little_endian_float(record + 12),
                                               little_endian_float(record + 16),
                                               little_endian_float(record + 20)));
    }
    return vertices;
}

/// Reads the vertices of a PLY file in the fixtures' layout: read_ply_header(), then
/// read_ply_vertices().
/// @param in the file, opened in binary mode, at its start.
/// @param error the reason, set when the file is refused.
/// @return the vertices, or nothing when the header has another layout or the file ends within
///     the vertex data.
inline std::optional<ply_vertices> read_ply(std::istream &in, std::string &error)
{
    const std::optional<std::int32_t> count = read_ply_header(in, error);
    if (!count)
        return std::nullopt;
    return read_ply_vertices(in, *count, error);
}

/// Reads the vertices of a PLY file on disk in the fixtures' layout; see read_ply().
/// @param path the file's path.
/// @param error the reason, prefixed by the path, set when the file cannot be opened or is
///     refused.
/// @return the vertices, or nothing.
inline std::optional<ply_vertices> read_ply_file(const std::string &path, std::string &error)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        error = path + ": cannot open";
        return std::nullopt;
    }
    std::optional<ply_vertices> vertices = read_ply(file, error);
    if (!vertices)
        error = path + ": " + error;
    return vertices;
}
} // namespace gca_test
