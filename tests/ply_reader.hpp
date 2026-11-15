// SPDX-License-Identifier: 0BSD
/// @file ply_reader.hpp
/// The offline tests' reader of the scanner's models (docs/SCAN_FILES.md, "Models"), the scan's
/// and the viewer's snapshots, apart from the scanner's code: binary little-endian PLY whose one
/// element is the vertex, with float x, y, z, nx, ny, nz and uchar red, green, blue in that order,
/// the header the models' writer writes line for line. read_ply() reads the vertices and stops. A
/// header outside that layout is refused with a message naming its line, and so is a file that
/// ends within the vertex data. Header only; each test program is one translation unit.

#pragma once

#include <array>
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
/// One vertex of a model's PLY, as the file holds it.
struct ply_vertex
{
    std::array<float, 3> position_m;   ///< x, y, z, in m
    std::array<float, 3> normal;       ///< nx, ny, nz
    std::array<std::uint8_t, 3> color; ///< red, green, blue, 0 to 255
};

/// Decodes a little-endian float32 whatever the host's byte order.
/// @param bytes the four bytes, the least significant first.
/// @return the float whose bits they are.
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
///     number of at most std::int32_t's maximum.
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
    if (count > std::numeric_limits<std::int32_t>::max())
        return std::nullopt;
    return static_cast<std::int32_t>(count);
}

/// Reads a PLY header in the models' layout. Each line has to match it exactly (no comment lines,
/// '\n' line ends), and "end_header" has to follow the vertex's nine properties.
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
    if (!next_line() || line != "end_header")
        return refusal("\"end_header\"");
    return count;
}

/// Reads the vertex data that follows a header in the models' layout.
/// @param in the file, at the first byte of data.
/// @param count the vertex count the header declared.
/// @param error the reason, set when the file ends within the vertex data.
/// @return the vertices in file order, or nothing when the file ends within them.
inline std::optional<std::vector<ply_vertex>> read_ply_vertices(std::istream &in,
                                                                std::int32_t count,
                                                                std::string &error)
{
    constexpr size_t vertex_bytes = 6 * sizeof(float) + 3;
    std::vector<ply_vertex> vertices;
    for (std::int32_t k = 0; k < count; k++)
    {
        unsigned char record[vertex_bytes];
        if (!in.read(reinterpret_cast<char *>(record), vertex_bytes))
        {
            error = "PLY data: the file ends within vertex " + std::to_string(k) + " of " +
                    std::to_string(count);
            return std::nullopt;
        }
        ply_vertex vertex{};
        for (size_t i = 0; i < 3; i++)
        {
            vertex.position_m[i] = little_endian_float(record + 4 * i);
            vertex.normal[i] = little_endian_float(record + 12 + 4 * i);
            vertex.color[i] = record[24 + i];
        }
        vertices.push_back(vertex);
    }
    return vertices;
}

/// Reads the vertices of a PLY file in the models' layout: read_ply_header(), then
/// read_ply_vertices().
/// @param in the file, opened in binary mode, at its start.
/// @param error the reason, set when the file is refused.
/// @return the vertices, or nothing when the header has another layout or the file ends within
///     the vertex data.
inline std::optional<std::vector<ply_vertex>> read_ply(std::istream &in, std::string &error)
{
    const std::optional<std::int32_t> count = read_ply_header(in, error);
    if (!count)
        return std::nullopt;
    return read_ply_vertices(in, *count, error);
}

/// Reads the vertices of a PLY file on disk in the models' layout; see read_ply().
/// @param path the file's path.
/// @param error the reason, prefixed by the path, set when the file cannot be opened or is
///     refused.
/// @return the vertices, or nothing.
inline std::optional<std::vector<ply_vertex>> read_ply_file(const std::string &path,
                                                            std::string &error)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        error = path + ": cannot open";
        return std::nullopt;
    }
    std::optional<std::vector<ply_vertex>> vertices = read_ply(file, error);
    if (!vertices)
        error = path + ": " + error;
    return vertices;
}
} // namespace gca_test
