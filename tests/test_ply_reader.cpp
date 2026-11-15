// SPDX-License-Identifier: 0BSD
/// @file test_ply_reader.cpp
/// Offline test of the tests' PLY reader (ply_reader.hpp) on files built in memory: a file in
/// the models' layout is read exactly; each kind of header outside that layout, vertex data cut
/// short, and a file that does not exist are refused, each with a message that locates the reason.
/// Needs no GPU and no camera.

#include "check.hpp"
#include "ply_reader.hpp"

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
using gca_test::ply_vertex;
using gca_test::read_ply;
using gca_test::read_ply_file;

namespace
{
/// A vertex of the test file.
struct test_vertex
{
    float position[3];
    float normal[3];
    unsigned char color[3];
};

/// The test file's vertices: coordinates of a room's magnitudes and beyond, both signs, unit
/// normals, and colors at both ends of the byte range and between.
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

/// Returns the test file's vertex data, laid out as in the models.
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

/// Returns the test file's header lines: the models' layout [S: docs/SCAN_FILES.md, "Models"] with
/// three vertices.
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

/// read_ply() reads the test file's three vertices exactly.
void test_read()
{
    std::istringstream in(join(valid_header()) + vertex_data());
    std::string error;
    const std::optional<std::vector<ply_vertex>> read = read_ply(in, error);
    check(read.has_value(), ("read: the test file accepted; " + error).c_str());
    if (!read)
        return;
    const size_t n = std::size(test_vertices);
    check(read->size() == n, "read: one vertex per vertex written");
    if (read->size() != n)
        return;
    bool exact = true;
    for (size_t k = 0; k < n; k++)
    {
        const test_vertex &v = test_vertices[k];
        const ply_vertex &got = (*read)[k];
        // Exact equality also fails on NaN.
        for (size_t i = 0; i < 3; i++)
            exact = exact && got.position_m[i] == v.position[i] && got.normal[i] == v.normal[i] &&
                    got.color[i] == v.color[i];
    }
    check(exact, "read: positions and normals bit-exact, colors byte-exact");
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
    const std::string data = vertex_data();
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
        {"face element after the vertex",
         join(replaced(header, 12, 0,
                       {"element face 1", "property list uchar int vertex_indices"})) +
             data,
         "line 13:"},
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
        {"comment line", join(replaced(header, 12, 0, {"comment written by hand"})) + data,
         "line 13:"},
        {"no end_header", join(replaced(header, 12, 1, {})), "line 13:"},
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
    const bool refused = !read_ply_file("/nonexistent/model.ply", error).has_value();
    check_refusal("missing file", refused, error, error.find("cannot open") != std::string::npos);
}
} // namespace

/// Runs the reader's checks and prints the refusal messages and the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    test_read();
    test_refusals();
    return check_summary("test_ply_reader");
}
