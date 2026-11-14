// SPDX-License-Identifier: 0BSD
/// @file ply_reader.hpp
/// The offline tests' reader of the colored-ICP fixtures (data/frag_115.ply, frag_116.ply) and of
/// the scanner's models (docs/SCAN_FILES.md), apart from the scanner's code:
/// binary little-endian PLY whose first element is the vertex, with float x, y, z, nx, ny, nz
/// and uchar red, green, blue in that order [S: the fixtures' headers]. read_ply() reads the
/// vertices and stops; the elements after them are declared in the header and not read.
/// read_ply_mesh() reads the fixtures' faces too, the one element after the vertices, each a uchar
/// count of 3 and three int indices, the triangles the synthetic source renders the fixture from
/// (synthetic_source.hpp). A header outside that layout is refused with a message naming its line.
/// Header only; each test program is one translation unit.

#pragma once

#include "geometry/type.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <istream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace gca_test
{
/// The vertices of a PLY file, in file order.
struct ply_vertices
{
    std::vector<gca::point_t> points; ///< coordinates in meters, colors in [0, 1], inactive
    std::vector<float3> normals;      ///< the file's normals, one per point
};

/// The vertices and the triangles of a PLY file, in file order.
struct ply_mesh
{
    ply_vertices vertices; ///< the vertices
    /// The triangles: three indices into the vertices each, every one below their count.
    std::vector<std::array<std::uint32_t, 3>> triangles;
};

/// What a PLY header in the fixtures' layout declares.
struct ply_header
{
    std::int32_t vertex_count; ///< the vertices
    /// The lines of the elements after the vertex, from the first "element" line to the line
    /// before "end_header", without line ends.
    std::vector<std::string> later_lines;
};

/// Decodes four little-endian bytes as a 32-bit value whatever the host's byte order.
/// @tparam Value the value's type, 4 bytes: float or std::int32_t.
/// @param bytes the four bytes, the least significant first.
/// @return the value whose bits they are.
template <typename Value> inline Value little_endian(const unsigned char *bytes)
{
    static_assert(sizeof(Value) == 4, "a PLY float or int is 4 bytes");
    const std::uint32_t bits = std::uint32_t{bytes[0]} | std::uint32_t{bytes[1]} << 8 |
                               std::uint32_t{bytes[2]} << 16 | std::uint32_t{bytes[3]} << 24;
    Value value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

/// Decodes a little-endian float32 whatever the host's byte order.
/// @param bytes the four bytes, the least significant first.
/// @return the float.
inline float little_endian_float(const unsigned char *bytes)
{
    return little_endian<float>(bytes);
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

/// Parses an element's header line, "element <name> <count>".
/// @param line the header line.
/// @param name the element's name, "vertex" or "face".
/// @return the count, or nothing when the line has another form or the count is not a decimal
///     number of at most gca::index_t's maximum.
inline std::optional<std::int32_t> parse_element_count(const std::string &line,
                                                       const std::string &name)
{
    const std::string prefix = "element " + name + " ";
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
/// @return the vertex count and the later elements' lines, or nothing when the header has another
///     layout.
inline std::optional<ply_header> read_ply_header(std::istream &in, std::string &error)
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
    const std::optional<std::int32_t> count =
        next_line() ? parse_element_count(line, "vertex") : std::nullopt;
    if (!count)
        return refusal("\"element vertex <count>\", the count at most 2147483647");
    for (const char *property : vertex_properties)
    {
        if (!next_line() || line != property)
            return refusal("\"" + std::string(property) + "\"");
    }
    // A property line before the next element line would be a tenth vertex property.
    ply_header header{*count, {}};
    while (next_line() && line != "end_header")
    {
        if (line.rfind("element ", 0) != 0 &&
            (header.later_lines.empty() || line.rfind("property ", 0) != 0))
            return refusal(header.later_lines.empty()
                               ? "\"element ...\" or \"end_header\""
                               : "\"element ...\", \"property ...\" or \"end_header\"");
        header.later_lines.push_back(line);
    }
    if (at_end)
        return refusal("\"end_header\"");
    return header;
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
    const std::optional<ply_header> header = read_ply_header(in, error);
    if (!header)
        return std::nullopt;
    return read_ply_vertices(in, header->vertex_count, error);
}

/// Reads the vertices and the triangles of a PLY file in the fixtures' layout: read_ply_header(),
/// whose elements after the vertex must be exactly "element face <count>" and "property list uchar
/// int vertex_indices"; read_ply_vertices(); then each face, a uchar count of 3 and three
/// little-endian int indices, each 0 to the vertex count less 1.
/// @param in the file, opened in binary mode, at its start.
/// @param error the reason, set when the file is refused.
/// @return the vertices and the triangles, or nothing when the header has another layout or no
///     such face element, the file ends within the data, a face is not a triangle, or an index
///     lies outside the vertices.
inline std::optional<ply_mesh> read_ply_mesh(std::istream &in, std::string &error)
{
    const std::optional<ply_header> header = read_ply_header(in, error);
    if (!header)
        return std::nullopt;
    const std::optional<std::int32_t> faces =
        header->later_lines.size() == 2 &&
                header->later_lines[1] == "property list uchar int vertex_indices"
            ? parse_element_count(header->later_lines[0], "face")
            : std::nullopt;
    if (!faces)
    {
        error = "PLY header: after the vertex expected \"element face <count>\" and \"property "
                "list uchar int vertex_indices\" alone";
        return std::nullopt;
    }
    std::optional<ply_vertices> vertices = read_ply_vertices(in, header->vertex_count, error);
    if (!vertices)
        return std::nullopt;
    ply_mesh mesh{std::move(*vertices), {}};
    constexpr size_t face_bytes = 1 + 3 * sizeof(std::int32_t);
    for (std::int32_t k = 0; k < *faces; k++)
    {
        unsigned char record[face_bytes];
        const std::string which = "face " + std::to_string(k) + " of " + std::to_string(*faces);
        if (!in.read(reinterpret_cast<char *>(record), face_bytes))
        {
            error = "PLY data: the file ends within " + which;
            return std::nullopt;
        }
        if (record[0] != 3)
        {
            error = "PLY data: " + which + " has " + std::to_string(record[0]) + " corners, not 3";
            return std::nullopt;
        }
        std::array<std::uint32_t, 3> triangle{};
        for (size_t i = 0; i < 3; i++)
        {
            const std::int32_t index = little_endian<std::int32_t>(record + 1 + 4 * i);
            if (index < 0 || index >= header->vertex_count)
            {
                error = "PLY data: " + which + " names vertex " + std::to_string(index) + " of " +
                        std::to_string(header->vertex_count);
                return std::nullopt;
            }
            triangle[i] = static_cast<std::uint32_t>(index);
        }
        mesh.triangles.push_back(triangle);
    }
    return mesh;
}

/// Reads a PLY file on disk with a reader of a stream.
/// @param path the file's path.
/// @param error the reason, prefixed by the path, set when the file cannot be opened or is
///     refused.
/// @param read the reader, read_ply() or read_ply_mesh(), called as read(stream, error).
/// @return what the reader read, or nothing.
template <typename Reader>
inline auto read_file(const std::string &path, std::string &error,
                      Reader read) -> decltype(read(std::declval<std::istream &>(), error))
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        error = path + ": cannot open";
        return std::nullopt;
    }
    auto read_back = read(file, error);
    if (!read_back)
        error = path + ": " + error;
    return read_back;
}

/// Reads the vertices and the triangles of a PLY file on disk in the fixtures' layout; see
/// read_ply_mesh().
/// @param path the file's path.
/// @param error the reason, prefixed by the path, set when the file cannot be opened or is
///     refused.
/// @return the vertices and the triangles, or nothing.
inline std::optional<ply_mesh> read_ply_mesh_file(const std::string &path, std::string &error)
{
    return read_file(path, error, read_ply_mesh);
}

/// Reads the vertices of a PLY file on disk in the fixtures' layout; see read_ply().
/// @param path the file's path.
/// @param error the reason, prefixed by the path, set when the file cannot be opened or is
///     refused.
/// @return the vertices, or nothing.
inline std::optional<ply_vertices> read_ply_file(const std::string &path, std::string &error)
{
    return read_file(path, error, read_ply);
}
} // namespace gca_test
