// SPDX-License-Identifier: 0BSD
/// @file test_ply_reader.cpp
/// Offline test of the tests' PLY reader (ply_reader.hpp) on files built in memory: a file in
/// the fixtures' layout, a face element after its vertices, is read exactly, its vertices alone
/// and with its triangles; each kind of header outside that layout, vertex data cut short, a face
/// that is no triangle or names a vertex the file lacks, face data cut short, and a file that does
/// not exist are refused, each with a message that locates the reason. Needs no GPU and no camera.

#include "check.hpp"
#include "ply_reader.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using gca_test::check;
using gca_test::check_refusal;
using gca_test::check_summary;
using gca_test::float_step;
using gca_test::ply_mesh;
using gca_test::ply_vertices;
using gca_test::read_ply;
using gca_test::read_ply_file;
using gca_test::read_ply_mesh;
using gca_test::read_ply_mesh_file;

namespace
{
/// A vertex of the test file.
struct test_vertex
{
    float position[3];
    float normal[3];
    unsigned char color[3];
};

/// The test file's vertices: coordinates of the fixture's magnitudes and beyond, both signs,
/// unit normals, and colors at both ends of the byte range and between.
constexpr test_vertex test_vertices[] = {
    {{0.7304688f, 1.2382813f, 0.2304688f}, {0.6f, -0.8f, 0.0f}, {0, 128, 255}},
    {{2.8185728f, 3.5429688f, 1.7269143f}, {-0.2672612f, 0.5345225f, 0.8017837f}, {15, 4, 203}},
    {{-1.25e-3f, 4.5f, 1.0e3f}, {0.0f, 0.0f, -1.0f}, {229, 1, 77}}};

/// Appends a float32 to a byte string, the least significant byte first.
/// @param bytes the byte string.
/// @param value the float.
void append_float(std::string &bytes, float value)
{
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<char>((bits >> shift) & 0xffu));
}

/// Returns the test file's vertex data, laid out as in the fixtures.
/// @return the bytes.
std::string vertex_data()
{
    std::string bytes;
    for (const test_vertex &vertex : test_vertices)
    {
        for (float x : vertex.position)
            append_float(bytes, x);
        for (float x : vertex.normal)
            append_float(bytes, x);
        for (unsigned char c : vertex.color)
            bytes.push_back(static_cast<char>(c));
    }
    return bytes;
}

/// Returns the test file's face data: one triangle, a uchar count and three little-endian int
/// indices, 2, 0 and 1, so that an index read from the wrong place shows.
/// @return the bytes.
std::string face_data()
{
    return std::string("\x03\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00", 13);
}

/// Returns the test file's header lines: the fixtures' layout [S: data/frag_116.ply's header]
/// with three vertices and one face.
/// @return the lines, without line ends.
std::vector<std::string> valid_header()
{
    return {"ply",
            "format binary_little_endian 1.0",
            "element vertex 3",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "element face 1",
            "property list uchar int vertex_indices",
            "end_header"};
}

/// Joins header lines, each ended by '\n'.
/// @param lines the lines.
/// @return the header text.
std::string join(const std::vector<std::string> &lines)
{
    std::string text;
    for (const std::string &line : lines)
        text += line + "\n";
    return text;
}

/// Returns header lines with a run of lines replaced.
/// @param lines the lines.
/// @param first the index of the run's first line.
/// @param count the run's length; 0 inserts before lines[first].
/// @param with the lines that take the run's place; none deletes the run.
/// @return the changed lines.
std::vector<std::string> replaced(std::vector<std::string> lines, size_t first, size_t count,
                                  const std::vector<std::string> &with)
{
    lines.erase(lines.begin() + first, lines.begin() + first + count);
    lines.insert(lines.begin() + first, with.begin(), with.end());
    return lines;
}

/// read_ply() reads the test file's three vertices exactly; the face element after them does not
/// disturb it.
void test_read()
{
    std::istringstream in(join(valid_header()) + vertex_data() + face_data());
    std::string error;
    const std::optional<ply_vertices> read = read_ply(in, error);
    check(read.has_value(), ("read: the test file accepted; " + error).c_str());
    if (!read)
        return;
    const size_t n = std::size(test_vertices);
    const bool counts = read->points.size() == n && read->normals.size() == n;
    check(counts, "read: one point and one normal per vertex");
    if (!counts)
        return;
    bool exact = true;
    bool colors = true;
    bool inactive = true;
    for (size_t k = 0; k < n; k++)
    {
        const test_vertex &v = test_vertices[k];
        const gca::point_t &p = read->points[k];
        const float3 &normal = read->normals[k];
        // Exact equality also fails on NaN.
        exact = exact && p.coordinates.x == v.position[0] && p.coordinates.y == v.position[1] &&
                p.coordinates.z == v.position[2] && normal.x == v.normal[0] &&
                normal.y == v.normal[1] && normal.z == v.normal[2];
        const float read_color[3] = {p.color.r, p.color.g, p.color.b};
        for (int c = 0; c < 3; c++)
            colors = colors && std::fabs(read_color[c] - v.color[c] / 255.0) <= float_step;
        inactive = inactive && p.property == gca::point_property::inactive;
    }
    check(exact, "read: positions and normals bit-exact");
    // One float step [S: float32 ulp]: the reader multiplies by the float nearest 1/255.
    check(colors, "read: each color the byte / 255 within one float step");
    check(inactive, "read: every point inactive");
}

/// read_ply_mesh() reads the test file's three vertices as read_ply() does, and its one triangle.
void test_read_mesh()
{
    std::istringstream in(join(valid_header()) + vertex_data() + face_data());
    std::string error;
    const std::optional<ply_mesh> read = read_ply_mesh(in, error);
    check(read.has_value(), ("mesh: the test file accepted; " + error).c_str());
    if (!read)
        return;
    check(read->vertices.points.size() == std::size(test_vertices) &&
              read->vertices.points[1].coordinates.y == test_vertices[1].position[1],
          "mesh: the vertices read");
    check(read->triangles.size() == 1 && read->triangles[0][0] == 2 && read->triangles[0][1] == 0 &&
              read->triangles[0][2] == 1,
          "mesh: one triangle, its indices 2, 0 and 1 in order");
}

/// read_ply_mesh() refuses a file without the face element, with another face property, with a
/// face that is no triangle, one that names a vertex the file lacks, and face data cut short; and
/// read_ply_mesh_file() a file that does not exist; each with a message that locates the reason.
void test_mesh_refusals()
{
    const std::vector<std::string> header = valid_header();
    const std::string vertices = vertex_data();
    std::string quad = face_data();
    quad[0] = '\x04';
    std::string beyond = face_data();
    beyond[1] = '\x03';
    const struct
    {
        const char *what;
        std::string file;
        const char *location;
    } cases[] = {
        {"mesh, no face element", join(replaced(header, 12, 2, {})) + vertices, "after the vertex"},
        {"mesh, unsigned indices",
         join(replaced(header, 13, 1, {"property list uchar uint vertex_indices"})) + vertices +
             face_data(),
         "after the vertex"},
        {"mesh, a face of 4 corners", join(header) + vertices + quad, "4 corners"},
        {"mesh, an index beyond the vertices", join(header) + vertices + beyond,
         "names vertex 3 of 3"},
        {"mesh, face data one byte short",
         join(header) + vertices + face_data().substr(0, face_data().size() - 1),
         "ends within face 0 of 1"},
    };
    for (const auto &c : cases)
    {
        std::istringstream in(c.file);
        std::string error;
        const bool refused = !read_ply_mesh(in, error).has_value();
        check_refusal(c.what, refused, error, error.find(c.location) != std::string::npos);
    }
    std::string error;
    const bool refused = !read_ply_mesh_file("/nonexistent/frag.ply", error).has_value();
    check_refusal("mesh, missing file", refused, error,
                  error.find("cannot open") != std::string::npos);
}

/// A file read_ply() has to refuse, and the part of its message that locates the reason.
struct refusal_case
{
    const char *what;
    std::string file;
    const char *location;
};

/// Returns one file per kind of refusal: the test file with one change to its header, or with
/// its vertex data cut short by one byte.
/// @return the cases.
std::vector<refusal_case> refusal_cases()
{
    const std::vector<std::string> header = valid_header();
    const std::string data = vertex_data() + face_data();
    const std::string short_data = vertex_data().substr(0, vertex_data().size() - 1);
    return {
        {"empty file", "", "line 1:"},
        {"not PLY", join(replaced(header, 0, 1, {"PLY"})) + data, "line 1:"},
        {"CRLF line ends", join(replaced(header, 0, 1, {"ply\r"})) + data, "line 1:"},
        {"ASCII", join(replaced(header, 1, 1, {"format ascii 1.0"})) + data, "line 2:"},
        {"big-endian", join(replaced(header, 1, 1, {"format binary_big_endian 1.0"})) + data,
         "line 2:"},
        {"face element first",
         join(
             replaced(header, 2, 0, {"element face 1", "property list uchar int vertex_indices"})) +
             data,
         "line 3:"},
        {"empty count", join(replaced(header, 2, 1, {"element vertex "})) + data, "line 3:"},
        {"negative count", join(replaced(header, 2, 1, {"element vertex -3"})) + data, "line 3:"},
        {"count with a letter", join(replaced(header, 2, 1, {"element vertex 3x"})) + data,
         "line 3:"},
        {"count of 11 digits", join(replaced(header, 2, 1, {"element vertex 00000000003"})) + data,
         "line 3:"},
        {"count above int32", join(replaced(header, 2, 1, {"element vertex 2147483648"})) + data,
         "line 3:"},
        {"double coordinate", join(replaced(header, 3, 1, {"property double x"})) + data,
         "line 4:"},
        {"properties reordered",
         join(replaced(header, 3, 2, {"property float y", "property float x"})) + data, "line 4:"},
        {"normals missing", join(replaced(header, 6, 3, {})) + data, "line 7:"},
        {"tenth vertex property", join(replaced(header, 12, 0, {"property uchar alpha"})) + data,
         "line 13:"},
        {"comment line", join(replaced(header, 14, 0, {"comment written by hand"})) + data,
         "line 15:"},
        {"no end_header", join(replaced(header, 14, 1, {})), "line 15:"},
        {"vertex data one byte short", join(header) + short_data, "vertex 2 of 3"},
    };
}

/// read_ply() refuses each case's file with a message that locates the reason, and
/// read_ply_file() refuses a file that does not exist.
void test_refusals()
{
    for (const refusal_case &refusal : refusal_cases())
    {
        std::istringstream in(refusal.file);
        std::string error;
        const bool refused = !read_ply(in, error).has_value();
        check_refusal(refusal.what, refused, error,
                      error.find(refusal.location) != std::string::npos);
    }
    std::string error;
    const bool refused = !read_ply_file("/nonexistent/frag.ply", error).has_value();
    check_refusal("missing file", refused, error, error.find("cannot open") != std::string::npos);
}
} // namespace

/// Runs the reader's checks and prints the refusal messages and the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    test_read();
    test_refusals();
    test_read_mesh();
    test_mesh_refusals();
    return check_summary("test_ply_reader");
}
