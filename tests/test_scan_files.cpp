// SPDX-License-Identifier: 0BSD
/// @file test_scan_files.cpp
/// The files a scan keeps (docs/SCAN_FILES.md), through the scan module's public API, each
/// written into a temporary directory that the test removes, and read back:
/// - a model is read back by the tests' own PLY reader (ply_reader.hpp, which shares no code with
///   the writer) and by the scan's reader with every position, normal and color equal; an empty
///   model too; the scan's reader refuses another header, data of another length and a normal
///   not unit; the writer refuses such a normal and writes nothing;
/// - a trajectory with tracked and lost frames is read back equal, bit for bit, the quaternion's
///   sign as given; the document's example, seven distinct values, is read as the document says
///   and written as it shows it; the reader refuses another header, a line of neither form, a
///   pose that is not finite or not unit, a file that ends within a line; the writer refuses such
///   a pose and writes nothing;
/// - the parameters of each object class, a set of neither and commits at the edges of the
///   strings the file holds are read back equal; the document's example is read as the document
///   says, also after white space, and written as it shows it; the reader refuses a text that
///   does not start with '{' (YAML, XML), a key it does not know, a key twice, a missing key,
///   another version, a value of another type, an unknown object class, a commit OpenCV does not
///   write as JSON and each value out of range; the writer refuses such a commit or set and
///   writes nothing;
/// - a writer whose file cannot be created or whose write fails says why, leaves nothing beside
///   the file, and leaves a file already there as it was; a longer part file left behind is
///   truncated; a directory at the name makes the rename fail.
/// Needs no GPU and no camera.

#include "check.hpp"
#include "ply_reader.hpp"
#include "scan/file_io.hpp"
#include "scan/model_file.hpp"
#include "scan/object_class.hpp"
#include "scan/parameters_file.hpp"
#include "scan/trajectory_file.hpp"

#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <sys/resource.h>

#include <Eigen/Geometry>

namespace
{
using gca_test::check;
using gca_test::check_refusal;

/// A directory made for this test program, removed with everything in it at its end.
class temporary_directory
{
public:
    /// Makes a new directory in the system's temporary directory.
    temporary_directory()
    {
        std::string pattern =
            (std::filesystem::temp_directory_path() / "test_scan_files.XXXXXX").string();
        if (::mkdtemp(pattern.data()) == nullptr)
        {
            std::perror("test_scan_files: mkdtemp");
            std::exit(1);
        }
        m_path = pattern;
    }

    temporary_directory(const temporary_directory &) = delete;
    temporary_directory &operator=(const temporary_directory &) = delete;

    /// Removes the directory and everything in it.
    ~temporary_directory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(m_path, ignored);
    }

    /// Gives a path in the directory.
    /// @param name [in] the file's name.
    /// @return the path.
    std::filesystem::path operator/(const std::string &name) const
    {
        return m_path / name;
    }

private:
    std::filesystem::path m_path;
};

/// Writes a file as the test gives it, for the cases written by hand.
/// @param path [in] the file's path.
/// @param bytes [in] the content.
void write_bytes(const std::filesystem::path &path, const std::string &bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    check(static_cast<bool>(file), ("test setup: wrote " + path.string()).c_str());
}

/// Reads a file's bytes as they are on disk.
/// @param path [in] the file's path.
/// @return the content; empty when the file cannot be read.
std::string bytes_of(const std::filesystem::path &path)
{
    std::ifstream file(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

/// Replaces the first occurrence of a text, for a case built from a valid file.
/// @param text [in] the valid file.
/// @param from [in] the text replaced; a failed check when it does not occur.
/// @param to [in] the text that takes its place.
/// @return the changed file.
std::string replaced(std::string text, const std::string &from, const std::string &to)
{
    const std::size_t at = text.find(from);
    check(at != std::string::npos, ("test setup: the file holds " + from).c_str());
    if (at != std::string::npos)
        text.replace(at, from.size(), to);
    return text;
}

/// Whether two floats have the same bits, so that -0 and 0 differ and NaN equals itself.
/// @param a [in] one float.
/// @param b [in] the other.
/// @return whether their bits are equal.
bool same_bits(float a, float b)
{
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}

/// Whether two doubles have the same bits.
/// @param a [in] one double.
/// @param b [in] the other.
/// @return whether their bits are equal.
bool same_bits(double a, double b)
{
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}

/// A file a reader has to refuse, and the part of the reason that says why.
struct refusal_case
{
    std::string what;   ///< the case, printed
    std::string file;   ///< the file's content
    std::string reason; ///< the part of the reason expected
};

/// A file the reader has to refuse is refused with the expected reason.
/// @tparam T the reader's value.
/// @param what [in] the case.
/// @param result [in] what the reader gave.
/// @param reason [in] the part of the reason expected.
template <typename T>
void check_read_refused(const std::string &what, const gca::read_result<T> &result,
                        const std::string &reason)
{
    check_refusal(what, !result.value.has_value(), result.reason,
                  result.reason.find(reason) != std::string::npos);
}

/// A write the writer has to refuse is refused with the expected reason, and leaves no file and
/// no part file.
/// @param what [in] the case.
/// @param result [in] what the writer gave.
/// @param path [in] the file's path.
/// @param reason [in] the part of the reason expected.
void check_write_refused(const std::string &what, const gca::write_result &result,
                         const std::filesystem::path &path, const std::string &reason)
{
    check_refusal(what, !result.written, result.reason,
                  result.reason.find(reason) != std::string::npos);
    std::filesystem::path part = path;
    part += ".part";
    check(!std::filesystem::exists(path) && !std::filesystem::exists(part),
          (what + ": no file and no part file").c_str());
}

// ------------------------------------------------------------------------------------- models

/// The test model: coordinates of both signs, of a millimetre and of a kilometre, a negative
/// zero, the smallest and largest floats, unit normals and colors at both ends of the byte range
/// and between; every float has four different bytes or is a signed zero, so a byte order other
/// than little-endian changes it [?: a choice].
std::vector<gca::model_point> test_model()
{
    return {
        {{0.7304688f, -1.2382813f, 0.2304688f}, {0.6f, -0.8f, 0.0f}, {0, 128, 255}},
        {{-2.8185728f, 3.5429688f, 1.7269143f},
         {-0.2672612f, 0.5345225f, 0.8017837f},
         {15, 4, 203}},
        {{1.25e-3f, -0.0f, 1.0e3f}, {0.0f, 0.0f, -1.0f}, {255, 1, 77}},
        {{std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(),
          -std::numeric_limits<float>::min()},
         {0.48f, 0.6f, -0.64f},
         {254, 0, 1}},
    };
}

/// Whether two models hold the same points, bit for bit.
/// @param read [in] the model read.
/// @param written [in] the model written.
/// @return whether the counts and every position, normal and color are equal.
bool same_model(const std::vector<gca::model_point> &read,
                const std::vector<gca::model_point> &written)
{
    if (read.size() != written.size())
        return false;
    for (std::size_t k = 0; k < read.size(); k++)
    {
        for (int c = 0; c < 3; c++)
        {
            if (!same_bits(read[k].position_m[c], written[k].position_m[c]) ||
                !same_bits(read[k].normal[c], written[k].normal[c]) ||
                read[k].color[c] != written[k].color[c])
                return false;
        }
    }
    return true;
}

/// A model written is read back by the tests' PLY reader and by the scan's reader with every
/// position, normal and color equal; an empty model too.
/// @param directory [in] the test's directory.
void test_model_round_trip(const temporary_directory &directory)
{
    for (const std::vector<gca::model_point> &model :
         {test_model(), std::vector<gca::model_point>{}})
    {
        const std::string what = model.empty() ? "empty model" : "model";
        const std::filesystem::path path = directory / (model.empty() ? "empty.ply" : "full.ply");
        const gca::write_result written = gca::write_model(path, model);
        check(written.written, (what + ": written; " + written.reason).c_str());

        std::string error;
        const std::optional<gca_test::ply_vertices> vertices =
            gca_test::read_ply_file(path.string(), error);
        check(vertices.has_value(), (what + ": the tests' reader reads it; " + error).c_str());
        bool equal = vertices && vertices->points.size() == model.size() &&
                     vertices->normals.size() == model.size();
        for (std::size_t k = 0; equal && k < model.size(); k++)
        {
            const gca_test::ply_vertices &v = *vertices;
            const float position[3] = {v.points[k].coordinates.x, v.points[k].coordinates.y,
                                       v.points[k].coordinates.z};
            const float normal[3] = {v.normals[k].x, v.normals[k].y, v.normals[k].z};
            // The tests' reader scales a color by 1/255; rounding back gives the byte.
            const float color[3] = {v.points[k].color.r, v.points[k].color.g, v.points[k].color.b};
            for (int c = 0; c < 3; c++)
            {
                equal = equal && same_bits(position[c], model[k].position_m[c]) &&
                        same_bits(normal[c], model[k].normal[c]) &&
                        std::lround(color[c] * 255.0f) == model[k].color[c];
            }
        }
        check(equal, (what + ": the tests' reader reads every position, normal and color").c_str());

        const gca::read_result<std::vector<gca::model_point>> read = gca::read_model(path);
        check(read.value && same_model(*read.value, model),
              (what + ": the scan's reader reads every point equal; " + read.reason).c_str());
    }
}

/// The scan's model reader refuses another header and data of another length, each with the
/// reason that locates it, and a file that does not exist.
/// @param directory [in] the test's directory.
void test_model_refusals(const temporary_directory &directory)
{
    const std::filesystem::path valid_path = directory / "valid.ply";
    gca::write_model(valid_path, test_model());
    const std::string valid = bytes_of(valid_path);
    const std::vector<refusal_case> cases = {
        {"big-endian", replaced(valid, "binary_little_endian", "binary_big_endian"),
         "header line 2 is not \"format binary_little_endian 1.0\""},
        {"a comment line", replaced(valid, "element vertex", "comment by hand\nelement vertex"),
         "header line 3 is not \"element vertex <count>\""},
        {"a count with a letter", replaced(valid, "element vertex 4", "element vertex 4x"),
         "header line 3 is not \"element vertex <count>\""},
        {"no count", replaced(valid, "element vertex 4", "element vertex "),
         "header line 3 is not \"element vertex <count>\""},
        {"a double coordinate", replaced(valid, "property float x", "property double x"),
         "header line 4 is not \"property float x\""},
        {"a tenth property", replaced(valid, "end_header", "property uchar alpha\nend_header"),
         "header line 13 is not \"end_header\""},
        {"CRLF line ends", replaced(valid, "ply\n", "ply\r\n"), "header line 1 is not \"ply\""},
        {"the header cut", valid.substr(0, valid.find("property float z")),
         "the file ends within header line 6"},
        {"the data one byte short", valid.substr(0, valid.size() - 1),
         "the header declares 4 points of 27 bytes, and 107 bytes follow it"},
        {"one byte after the data", valid + "x",
         "the header declares 4 points of 27 bytes, and 109 bytes follow it"},
        {"a count of 2^64",
         replaced(valid, "element vertex 4", "element vertex 18446744073709551616"),
         "header line 3 is not \"element vertex <count>\""},
        {"a count near 2^64",
         replaced(valid, "element vertex 4", "element vertex 18446744073709551615"),
         "the header declares 18446744073709551615 points"},
    };
    for (const refusal_case &refusal : cases)
    {
        const std::filesystem::path path = directory / "refused.ply";
        write_bytes(path, refusal.file);
        check_read_refused("model, " + refusal.what, gca::read_model(path), refusal.reason);
    }
    check_read_refused("model, a missing file", gca::read_model(directory / "missing.ply"),
                       "cannot open");
}

/// Encodes a float32 as the file holds it, the least significant byte first, apart from the
/// writer's code.
/// @param value [in] the float.
/// @return its four bytes.
std::string little_endian_bytes(float value)
{
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    std::string bytes;
    for (int shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<char>((bits >> shift) & 0xffu));
    return bytes;
}

/// The model's writer refuses a normal that is not unit within 1e-6 and writes nothing; the
/// reader refuses a file holding one, naming the point.
/// @param directory [in] the test's directory.
void test_model_normals(const temporary_directory &directory)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const struct
    {
        const char *what;
        Eigen::Vector3f normal;
        const char *reason;
    } refused_normals[] = {
        {"a normal 2e-6 too long", Eigen::Vector3f(0.0f, 0.0f, 1.000002f),
         "point 1: the normal's length is 1.00000202655792"},
        {"a zero normal", Eigen::Vector3f(0.0f, 0.0f, 0.0f),
         "point 1: the normal's length is 0, not 1 within 1e-6"},
        {"a normal not finite", Eigen::Vector3f(nan, 0.0f, 0.0f),
         "point 1: the normal's length is nan"},
    };
    for (const auto &refused : refused_normals)
    {
        const std::filesystem::path path = directory / "refused_write.ply";
        std::vector<gca::model_point> model = test_model();
        model[1].normal = refused.normal;
        check_write_refused(std::string("model writer, ") + refused.what,
                            gca::write_model(path, model), path, refused.reason);
    }

    // A valid file with point 2's normal changed to (0, 0, -2), byte by byte.
    const std::filesystem::path valid_path = directory / "valid.ply";
    gca::write_model(valid_path, test_model());
    std::string file = bytes_of(valid_path);
    const std::size_t data_start = file.find("end_header\n") + std::strlen("end_header\n");
    file.replace(data_start + 2 * 27 + 20, 4, little_endian_bytes(-2.0f));
    const std::filesystem::path path = directory / "refused.ply";
    write_bytes(path, file);
    check_read_refused("model, a normal of length 2", gca::read_model(path),
                       "point 2: the normal's length is 2, not 1 within 1e-6");
}

// --------------------------------------------------------------------------------- trajectories

/// Gives a pose.
/// @param x [in] the position's x, in m.
/// @param y [in] the position's y, in m.
/// @param z [in] the position's z, in m.
/// @param orientation [in] the orientation.
/// @return the pose.
gca::camera_pose pose(double x, double y, double z, const Eigen::Quaterniond &orientation)
{
    return gca::camera_pose{Eigen::Vector3d(x, y, z), orientation};
}

/// The test trajectory: tracked frames and lost ones, a gap in the indices, the largest index
/// and timestamp, coordinates that need all 17 digits, a negative zero, a quaternion with w < 0
/// (kept as given) and one whose norm lies 5e-10 from 1, inside the tolerance [?: a choice].
std::vector<gca::trajectory_entry> test_trajectory()
{
    const Eigen::Quaterniond turned(
        Eigen::AngleAxisd(0.7, Eigen::Vector3d(1.0, 2.0, 3.0).normalized()));
    const Eigen::Quaterniond flipped(-turned.w(), -turned.x(), -turned.y(), -turned.z());
    const Eigen::Quaterniond almost_unit(
        Eigen::Quaterniond(Eigen::AngleAxisd(3.0, Eigen::Vector3d(-0.2, 0.9, 0.1).normalized()))
            .coeffs() *
        (1.0 + 5e-10));
    const std::uint64_t largest = std::numeric_limits<std::uint64_t>::max();
    return {
        {0, 1000, pose(0.0, 0.0, 0.0, Eigen::Quaterniond::Identity())},
        {1, 34333, std::nullopt},
        {3, 101000, pose(1.0 / 3.0, -2.5e-17, 12345.678901234567, turned)},
        {4, 134333, pose(-0.0, 0.1, -0.30000000000000004, flipped)},
        {5, 167666, pose(-1.5, 2.25, 0.875, almost_unit)},
        {largest - 1, largest - 1, std::nullopt},
        {largest, largest, pose(0.5, 0.25, 0.125, turned)},
    };
}

/// Whether two trajectories are equal, bit for bit.
/// @param read [in] the trajectory read.
/// @param written [in] the trajectory written.
/// @return whether the counts, indices, timestamps, marks and every coefficient are equal.
bool same_trajectory(const std::vector<gca::trajectory_entry> &read,
                     const std::vector<gca::trajectory_entry> &written)
{
    if (read.size() != written.size())
        return false;
    for (std::size_t k = 0; k < read.size(); k++)
    {
        const gca::trajectory_entry &a = read[k];
        const gca::trajectory_entry &b = written[k];
        if (a.index != b.index || a.timestamp_us != b.timestamp_us ||
            a.pose.has_value() != b.pose.has_value())
            return false;
        if (!a.pose)
            continue;
        for (int c = 0; c < 3; c++)
        {
            if (!same_bits(a.pose->position_m[c], b.pose->position_m[c]))
                return false;
        }
        for (int c = 0; c < 4; c++)
        {
            if (!same_bits(a.pose->orientation.coeffs()[c], b.pose->orientation.coeffs()[c]))
                return false;
        }
    }
    return true;
}

/// A trajectory with tracked and lost frames is read back equal; an empty one too; the
/// document's example is read as the document says and written as the document shows it.
/// @param directory [in] the test's directory.
void test_trajectory_round_trip(const temporary_directory &directory)
{
    const std::filesystem::path path = directory / "trajectory.txt";
    const std::vector<gca::trajectory_entry> trajectory = test_trajectory();
    const gca::write_result written = gca::write_trajectory(path, trajectory);
    check(written.written, ("trajectory: written; " + written.reason).c_str());
    const gca::read_result<std::vector<gca::trajectory_entry>> read = gca::read_trajectory(path);
    check(read.value && same_trajectory(*read.value, trajectory),
          ("trajectory: read back equal, bit for bit; " + read.reason).c_str());

    const std::filesystem::path empty_path = directory / "empty.txt";
    gca::write_trajectory(empty_path, {});
    const gca::read_result<std::vector<gca::trajectory_entry>> empty =
        gca::read_trajectory(empty_path);
    check(empty.value && empty.value->empty(),
          ("trajectory: an empty one read back empty; " + empty.reason).c_str());

    // The document's example, seven distinct values, so that every field's place shows: frame 7
    // at (0.5, -1.25, 2) m, its quaternion x y z w = 0.2 0.4 0.16 0.88, a unit one (0.04 + 0.16
    // + 0.0256 + 0.7744 = 1), then frame 9 lost.
    const std::string example = "# 3dscanner trajectory 1\n"
                                "7 233333 0.5 -1.25 2 0.2 0.4 0.16 0.88\n"
                                "9 300000 lost\n";
    const std::vector<gca::trajectory_entry> expected = {
        {7, 233333, pose(0.5, -1.25, 2.0, Eigen::Quaterniond(0.88, 0.2, 0.4, 0.16))},
        {9, 300000, std::nullopt}};
    const std::filesystem::path hand_path = directory / "by_hand.txt";
    write_bytes(hand_path, example);
    const gca::read_result<std::vector<gca::trajectory_entry>> hand =
        gca::read_trajectory(hand_path);
    check(hand.value && same_trajectory(*hand.value, expected),
          ("trajectory: the document's example is read as the document says; " + hand.reason)
              .c_str());
    const std::filesystem::path example_path = directory / "example.txt";
    check(gca::write_trajectory(example_path, expected).written &&
              bytes_of(example_path) == example,
          "trajectory: the document's example is written as the document shows it");
}

/// The trajectory's reader refuses each case with the reason that names its line, and a file
/// that does not exist; the writer refuses a pose not finite or not unit and writes nothing.
/// @param directory [in] the test's directory.
void test_trajectory_refusals(const temporary_directory &directory)
{
    const std::string header = "# 3dscanner trajectory 1\n";
    const std::string no_frame_line = "line 2 is no frame line";
    const std::vector<refusal_case> cases = {
        {"another version", "# 3dscanner trajectory 2\n0 0 lost\n",
         "line 1 is not the header \"# 3dscanner trajectory 1\""},
        {"no header", "0 0 lost\n", "line 1 is not the header"},
        {"an empty file", "", "line 1 is missing: the file is empty"},
        {"the last line without its end", header + "0 0 lost", "line 2 has no line end"},
        {"a misspelled mark", header + "0 0 lots\n", no_frame_line},
        {"two fields", header + "0 0\n", no_frame_line},
        {"eight fields", header + "0 0 0 0 0 0 0 0\n", no_frame_line},
        {"ten fields", header + "0 0 0 0 0 0 0 0 1 9\n", no_frame_line},
        {"two spaces", header + "0  0 lost\n", no_frame_line},
        {"a space at the end", header + "0 0 lost \n", no_frame_line},
        {"a CRLF line end", header + "0 0 lost\r\n", no_frame_line},
        {"a negative index", header + "-1 0 lost\n", no_frame_line},
        {"an index of 2^64", header + "18446744073709551616 0 lost\n", no_frame_line},
        {"a timestamp with a fraction", header + "0 0.5 lost\n", no_frame_line},
        {"a coordinate not a number", header + "0 0 a 0 0 0 0 0 1\n", no_frame_line},
        {"an empty line", header + "\n", no_frame_line},
        {"a position not finite", header + "0 0 nan 0 0 0 0 0 1\n",
         "line 2 holds a pose the file cannot hold: the position is not finite"},
        {"a quaternion of norm 2", header + "0 0 0 0 0 0 0 0 2\n",
         "line 2 holds a pose the file cannot hold: the quaternion's norm is 2, not 1 within "
         "1e-9"},
        {"a quaternion of norm 1 + 2e-9", header + "0 0 0 0 0 0 0 0 1.000000002\n",
         "the quaternion's norm is 1.000000002"},
        {"a quaternion not finite", header + "0 0 0 0 0 inf 0 0 1\n",
         "the quaternion's norm is inf"},
        {"a bad line after good ones", header + "0 0 lost\n1 5 lost\n2 9 lots\n",
         "line 4 is no frame line"},
    };
    for (const refusal_case &refusal : cases)
    {
        const std::filesystem::path path = directory / "refused.txt";
        write_bytes(path, refusal.file);
        check_read_refused("trajectory, " + refusal.what, gca::read_trajectory(path),
                           refusal.reason);
    }
    check_read_refused("trajectory, a missing file",
                       gca::read_trajectory(directory / "missing.txt"), "cannot open");

    const Eigen::Quaterniond identity = Eigen::Quaterniond::Identity();
    const Eigen::Quaterniond too_long(identity.coeffs() * (1.0 + 2e-9));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const struct
    {
        const char *what;
        gca::camera_pose pose;
        const char *reason;
    } refused_poses[] = {
        {"a quaternion of norm 1 + 2e-9", pose(0.0, 0.0, 0.0, too_long),
         "entry 1 (frame index 8): the quaternion's norm is 1.000000002"},
        {"a position not finite", pose(0.0, nan, 0.0, identity),
         "entry 1 (frame index 8): the position is not finite"},
        {"a quaternion not finite", pose(0.0, 0.0, 0.0, Eigen::Quaterniond(nan, 0.0, 0.0, 0.0)),
         "entry 1 (frame index 8): the quaternion's norm is nan"},
    };
    for (const auto &refused : refused_poses)
    {
        const std::filesystem::path path = directory / "refused_write.txt";
        const std::vector<gca::trajectory_entry> trajectory = {
            {7, 100, pose(0.0, 0.0, 0.0, identity)}, {8, 133, refused.pose}};
        check_write_refused(std::string("trajectory writer, ") + refused.what,
                            gca::write_trajectory(path, trajectory), path, refused.reason);
    }
}

// ------------------------------------------------------------------------------- parameters

/// The document's example of the parameters, a large object's set, as written there.
const std::string parameters_by_hand = R"({
    "format_version": 1,
    "commit": "db91e0d",
    "object_class": "large_object",
    "parameters": {
        "live_voxel_size_m": 8.0000000000000002e-03,
        "full_voxel_size_m": 4.0000000000000001e-03,
        "min_depth_m": 4.0000000000000002e-01,
        "max_depth_m": 3.0,
        "memory_budget_bytes": 1000000000.0
    }
}
)";

/// The example's text of three values, which the refusal cases replace.
const std::string live_text = "8.0000000000000002e-03";
const std::string full_text = "4.0000000000000001e-03";
const std::string min_text = "4.0000000000000002e-01";

/// The parameters of each object class, and a set of neither whose doubles need all 17 digits
/// with the largest budget JSON holds exactly, are written and read back equal; commits at the
/// edges of the strings the file holds too: one that starts with '{' and holds double quotes, a
/// backslash and every other printable ASCII character but the apostrophe, an empty one and one
/// of 4095 characters; the document's example is read as the document says, also after white
/// space, and written as the document shows it.
/// @param directory [in] the test's directory.
void test_parameters_round_trip(const temporary_directory &directory)
{
    const std::vector<gca::scan_parameters> cases = {
        {"db91e0d0123456789abcdef0123456789abcdef0", gca::object_class::tabletop_object,
         gca::parameters_of(gca::object_class::tabletop_object)},
        {"831f06a", gca::object_class::large_object,
         gca::parameters_of(gca::object_class::large_object)},
        {"{\"not a map\" [",
         gca::object_class::tabletop_object,
         {1.0 / 3.0, 0.1, 0.30000000000000004, 2.9999999999999996, 9007199254740992u}},
        {" !\"#$%&()*+,-./0123456789:;<=>?@AZ[\\]^_`az{|}~", gca::object_class::large_object,
         gca::parameters_of(gca::object_class::large_object)},
        {"", gca::object_class::large_object, gca::parameters_of(gca::object_class::large_object)},
        {std::string(4095, 'x'), gca::object_class::large_object,
         gca::parameters_of(gca::object_class::large_object)},
    };
    check(!(cases[0].parameters == cases[1].parameters),
          "parameters: the two object classes' sets differ");
    for (const gca::scan_parameters &parameters : cases)
    {
        const std::string what = "parameters, " +
                                 std::string(gca::name_of(parameters.object_class)) + ", commit " +
                                 parameters.commit.substr(0, 60) + " (" +
                                 std::to_string(parameters.commit.size()) + " characters)";
        const std::filesystem::path path = directory / "parameters.json";
        const gca::write_result written = gca::write_parameters(path, parameters);
        check(written.written, (what + ": written; " + written.reason).c_str());
        const gca::read_result<gca::scan_parameters> read = gca::read_parameters(path);
        check(read.value && *read.value == parameters,
              (what + ": read back equal; " + read.reason).c_str());
    }

    const std::filesystem::path hand_path = directory / "by_hand.json";
    write_bytes(hand_path, parameters_by_hand);
    const gca::read_result<gca::scan_parameters> hand = gca::read_parameters(hand_path);
    const gca::scan_parameters expected = {
        "db91e0d", gca::object_class::large_object, {0.008, 0.004, 0.4, 3.0, 1000000000u}};
    check(hand.value && *hand.value == expected,
          ("parameters: the document's example is read as the document says; " + hand.reason)
              .c_str());
    const std::filesystem::path example_path = directory / "example.json";
    check(gca::write_parameters(example_path, expected).written &&
              bytes_of(example_path) == parameters_by_hand,
          "parameters: the document's example is written as the document shows it");
    write_bytes(hand_path, " \t\r\n" + parameters_by_hand);
    const gca::read_result<gca::scan_parameters> spaced = gca::read_parameters(hand_path);
    check(spaced.value && *spaced.value == expected,
          ("parameters: white space before the '{' is skipped; " + spaced.reason).c_str());
}

/// The parameters' writer refuses each kind of commit that OpenCV does not write as JSON, and
/// writes nothing; the reader refuses such a commit in a file written by hand.
/// @param directory [in] the test's directory.
void test_parameters_strings(const temporary_directory &directory)
{
    const struct
    {
        const char *what;
        std::string commit;
        const char *reason;
    } refused_commits[] = {
        {"an apostrophe", "it's", "\"commit\" holds the byte 0x27 at 2"},
        {"a control byte",
         "a\x01"
         "b",
         "\"commit\" holds the byte 0x01 at 1"},
        {"a tab", "a\tb", "\"commit\" holds the byte 0x09 at 1"},
        {"a byte above ASCII", "caf\xc3\xa9", "\"commit\" holds the byte 0xc3 at 3"},
        {"bounded by double quotes", "\"db91e0d\"",
         "\"commit\" starts and ends with a double quote"},
        {"one double quote", "\"", "\"commit\" starts and ends with a double quote"},
        {"4096 characters", std::string(4096, 'x'),
         "\"commit\" holds 4096 characters, more than 4095"},
    };
    for (const auto &refused : refused_commits)
    {
        const std::filesystem::path path = directory / "refused_write.json";
        const gca::scan_parameters parameters = {
            refused.commit, gca::object_class::tabletop_object,
            gca::parameters_of(gca::object_class::tabletop_object)};
        check_write_refused(std::string("parameters writer, a commit: ") + refused.what,
                            gca::write_parameters(path, parameters), path, refused.reason);
    }

    // JSON that OpenCV reads, with commits the writer refuses.
    const std::vector<refusal_case> cases = {
        {"a commit with an apostrophe", replaced(parameters_by_hand, "db91e0d", "it's"),
         "\"commit\" holds the byte 0x27 at 2"},
        {"a commit bounded by double quotes",
         replaced(parameters_by_hand, "\"db91e0d\"", "\"\\\"db91e0d\\\"\""),
         "\"commit\" starts and ends with a double quote"},
    };
    for (const refusal_case &refusal : cases)
    {
        const std::filesystem::path path = directory / "refused.json";
        write_bytes(path, refusal.file);
        check_read_refused("parameters, " + refusal.what, gca::read_parameters(path),
                           refusal.reason);
    }
}

/// The parameters' reader refuses each case with the reason that names the key, and a file that
/// does not exist; the writer refuses a set out of range and writes nothing.
/// @param directory [in] the test's directory.
void test_parameters_refusals(const temporary_directory &directory)
{
    const std::string &valid = parameters_by_hand;
    const std::string top = "at the top level";
    const std::string nested = "in \"parameters\"";
    const std::string not_real = "in \"parameters\" is no number with a decimal point or an "
                                 "exponent";
    const std::vector<refusal_case> cases = {
        {"an unknown key at the top level",
         replaced(valid, "\"commit\"", "\"colour\": \"red\",\n    \"commit\""),
         "unknown key \"colour\" " + top},
        {"an unknown key in the set",
         replaced(valid, "\"min_depth_m\"", "\"voxel_size_m\": 1.0e-3,\n        \"min_depth_m\""),
         "unknown key \"voxel_size_m\" " + nested},
        {"a key twice at the top level",
         replaced(valid, "\"commit\"", "\"commit\": \"a\",\n    \"commit\""),
         "key \"commit\" twice " + top},
        {"a key twice in the set",
         replaced(valid, "\"max_depth_m\"", "\"min_depth_m\": 0.5,\n        \"max_depth_m\""),
         "key \"min_depth_m\" twice " + nested},
        {"no commit", replaced(valid, "\"commit\": \"db91e0d\",\n", ""),
         "no key \"commit\" " + top},
        {"no object class", replaced(valid, "\"object_class\": \"large_object\",\n", ""),
         "no key \"object_class\" " + top},
        {"no max_depth_m", replaced(valid, "\"max_depth_m\": 3.0,\n", ""),
         "no key \"max_depth_m\" " + nested},
        {"version 2", replaced(valid, "\"format_version\": 1", "\"format_version\": 2"),
         "format version 2, this reader reads version 1"},
        {"version 0", replaced(valid, "\"format_version\": 1", "\"format_version\": 0"),
         "format version 0, this reader reads version 1"},
        {"no version", replaced(valid, "\"format_version\": 1,\n", ""),
         "no integer \"format_version\" " + top},
        {"the version a string",
         replaced(valid, "\"format_version\": 1", "\"format_version\": \"1\""),
         "no integer \"format_version\" " + top},
        {"the commit a number", replaced(valid, "\"db91e0d\"", "910"), "\"commit\" is no string"},
        {"the object class a number", replaced(valid, "\"large_object\"", "2"),
         "\"object_class\" is no string"},
        {"an unknown object class", replaced(valid, "\"large_object\"", "\"small_object\""),
         "\"object_class\" is \"small_object\", no object class"},
        {"the set a number", valid.substr(0, valid.find("{\n        ")) + "1.0\n}\n",
         "\"parameters\" is no JSON object"},
        {"a length written as an integer", replaced(valid, "3.0", "3"),
         "\"max_depth_m\" " + not_real},
        {"a length written as a string", replaced(valid, min_text, "\"0.4\""),
         "\"min_depth_m\" " + not_real},
        {"a budget written as an integer", replaced(valid, "1000000000.0", "1000000000"),
         "\"memory_budget_bytes\" " + not_real},
        {"a voxel size below 0", replaced(valid, full_text, "-4.0e-03"),
         "\"full_voxel_size_m\" is -0.004, not a finite number > 0"},
        {"a voxel size of 0", replaced(valid, live_text, "0.0"),
         "\"live_voxel_size_m\" is 0, not a finite number > 0"},
        {"a depth not finite", replaced(valid, min_text, ".Nan"), "\"min_depth_m\" is "},
        {"min_depth_m equal to max_depth_m", replaced(valid, min_text, "3.0"),
         "\"min_depth_m\" 3 is not less than \"max_depth_m\" 3"},
        {"min_depth_m above max_depth_m", replaced(valid, min_text, "3.5"),
         "\"min_depth_m\" 3.5 is not less than \"max_depth_m\" 3"},
        {"a budget with a fraction", replaced(valid, "1000000000.0", "1000000000.5"),
         "\"memory_budget_bytes\" is 1000000000.5, not a whole number from 1 to 2^53"},
        {"a budget of 0", replaced(valid, "1000000000.0", "0.0"),
         "\"memory_budget_bytes\" is 0, not a whole number from 1 to 2^53"},
        {"a budget above 2^53", replaced(valid, "1000000000.0", "9007199254740994.0"),
         "\"memory_budget_bytes\" is 9007199254740994, not a whole number from 1 to 2^53"},
        {"a file cut short", valid.substr(0, valid.find("\"parameters\"")),
         "no JSON that OpenCV reads"},
        {"an empty file", "", "no JSON object: the first byte after any white space is not '{'"},
        {"a list at the top level", "[1.0, 2.0]\n",
         "no JSON object: the first byte after any white space is not '{'"},
        {"YAML", "%YAML:1.0\n---\nformat_version: 1\n",
         "no JSON object: the first byte after any white space is not '{'"},
        {"XML", "<?xml version=\"1.0\"?>\n<opencv_storage></opencv_storage>\n",
         "no JSON object: the first byte after any white space is not '{'"},
    };
    for (const refusal_case &refusal : cases)
    {
        const std::filesystem::path path = directory / "refused.json";
        write_bytes(path, refusal.file);
        const gca::read_result<gca::scan_parameters> read = gca::read_parameters(path);
        check_read_refused("parameters, " + refusal.what, read, refusal.reason);
        // The not-finite case: the reason names the key and the rule; the NaN's sign is
        // OpenCV's.
        if (refusal.what == "a depth not finite")
            check(read.reason.find("not a finite number > 0") != std::string::npos,
                  "parameters, a depth not finite: the reason says not finite");
    }
    check_read_refused("parameters, a missing file",
                       gca::read_parameters(directory / "missing.json"), "cannot open");

    const gca::object_class_parameters good =
        gca::parameters_of(gca::object_class::tabletop_object);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const struct
    {
        const char *what;
        gca::object_class_parameters set;
        const char *reason;
    } refused_sets[] = {
        {"a voxel size not finite",
         {nan, good.full_voxel_size_m, good.min_depth_m, good.max_depth_m,
          good.memory_budget_bytes},
         "\"live_voxel_size_m\" is "},
        {"a depth of 0",
         {good.live_voxel_size_m, good.full_voxel_size_m, 0.0, good.max_depth_m,
          good.memory_budget_bytes},
         "\"min_depth_m\" is 0, not a finite number > 0"},
        {"min_depth_m above max_depth_m",
         {good.live_voxel_size_m, good.full_voxel_size_m, 2.0, 1.0, good.memory_budget_bytes},
         "\"min_depth_m\" 2 is not less than \"max_depth_m\" 1"},
        {"a budget of 0",
         {good.live_voxel_size_m, good.full_voxel_size_m, good.min_depth_m, good.max_depth_m, 0},
         "\"memory_budget_bytes\" is 0, not a whole number from 1 to 2^53"},
        {"a budget of 2^53 + 1",
         {good.live_voxel_size_m, good.full_voxel_size_m, good.min_depth_m, good.max_depth_m,
          9007199254740993u},
         "\"memory_budget_bytes\" is 9007199254740993, not a whole number from 1 to 2^53"},
    };
    for (const auto &refused : refused_sets)
    {
        const std::filesystem::path path = directory / "refused_write.json";
        const gca::scan_parameters parameters = {"db91e0d", gca::object_class::tabletop_object,
                                                 refused.set};
        check_write_refused(std::string("parameters writer, ") + refused.what,
                            gca::write_parameters(path, parameters), path, refused.reason);
    }
}

// ---------------------------------------------------------------------------------- writing

/// Each writer says why when its file cannot be created; a write that fails (a file size limit,
/// which stands for a full disk) says why, leaves no part file and leaves the file already there
/// as it was; a write that succeeds replaces it; a longer part file left behind is truncated; a
/// directory at the name makes the rename fail, said why, and stays.
/// @param directory [in] the test's directory.
void test_write_failures(const temporary_directory &directory)
{
    const std::filesystem::path nowhere = directory / "no_such_directory";
    const gca::scan_parameters parameters = {
        "db91e0d", gca::object_class::tabletop_object,
        gca::parameters_of(gca::object_class::tabletop_object)};
    check_write_refused("model writer, no directory",
                        gca::write_model(nowhere / "full.ply", test_model()), nowhere / "full.ply",
                        "cannot create");
    check_write_refused("trajectory writer, no directory",
                        gca::write_trajectory(nowhere / "trajectory.txt", test_trajectory()),
                        nowhere / "trajectory.txt", "cannot create");
    check_write_refused("parameters writer, no directory",
                        gca::write_parameters(nowhere / "parameters.json", parameters),
                        nowhere / "parameters.json", "cannot create");

    // A file there already, then a write cut off by a file size limit of 100 bytes, less than
    // the model's header.
    const std::filesystem::path path = directory / "kept.ply";
    const std::vector<gca::model_point> first = {test_model()[0]};
    check(gca::write_model(path, first).written, "model writer: the first model written");
    std::filesystem::path part = path;
    part += ".part";

    rlimit limit;
    check(::getrlimit(RLIMIT_FSIZE, &limit) == 0, "test setup: getrlimit");
    const rlimit saved = limit;
    limit.rlim_cur = 100;
    // Without it the kernel ends the program with SIGXFSZ instead of failing the write.
    const auto previous = std::signal(SIGXFSZ, SIG_IGN);
    check(::setrlimit(RLIMIT_FSIZE, &limit) == 0, "test setup: setrlimit");
    const gca::write_result cut = gca::write_model(path, test_model());
    check(::setrlimit(RLIMIT_FSIZE, &saved) == 0, "test setup: setrlimit restored");
    std::signal(SIGXFSZ, previous);

    check_refusal("model writer, a write cut off", !cut.written, cut.reason,
                  cut.reason.find("the write failed: File too large") != std::string::npos);
    check(!std::filesystem::exists(part), "model writer, a write cut off: no part file");
    const gca::read_result<std::vector<gca::model_point>> kept = gca::read_model(path);
    check(kept.value && same_model(*kept.value, first),
          ("model writer, a write cut off: the file there as it was; " + kept.reason).c_str());

    check(gca::write_model(path, test_model()).written, "model writer: the second model written");
    const gca::read_result<std::vector<gca::model_point>> replaced_model = gca::read_model(path);
    check(replaced_model.value && same_model(*replaced_model.value, test_model()),
          "model writer: a write replaces the file there");

    // A part file left by a power cut, longer than the new file: the write truncates it.
    const std::filesystem::path fresh = directory / "fresh.txt";
    const std::filesystem::path stale = directory / "stale.txt";
    std::filesystem::path stale_part = stale;
    stale_part += ".part";
    check(gca::write_trajectory(fresh, test_trajectory()).written,
          "trajectory writer: written where no part file stands");
    write_bytes(stale_part, std::string(bytes_of(fresh).size() + 1000, '#'));
    check(gca::write_trajectory(stale, test_trajectory()).written,
          "trajectory writer: written over a longer part file");
    check(bytes_of(stale) == bytes_of(fresh) && !std::filesystem::exists(stale_part),
          "trajectory writer: a longer part file left behind leaves nothing in the file");

    // A directory standing at the name: the rename fails, and the directory stays.
    const std::filesystem::path taken = directory / "taken.json";
    std::filesystem::create_directory(taken);
    std::filesystem::path taken_part = taken;
    taken_part += ".part";
    const gca::write_result onto_directory = gca::write_parameters(taken, parameters);
    check_refusal("parameters writer, a directory at the name", !onto_directory.written,
                  onto_directory.reason,
                  onto_directory.reason.find("the rename from " + taken_part.string() +
                                             " failed: Is a directory") != std::string::npos);
    check(std::filesystem::is_directory(taken) && !std::filesystem::exists(taken_part),
          "parameters writer, a directory at the name: the directory stays, no part file");
}
} // namespace

/// Runs the checks of the three files and prints the refusals and the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    const temporary_directory directory;
    test_model_round_trip(directory);
    test_model_refusals(directory);
    test_model_normals(directory);
    test_trajectory_round_trip(directory);
    test_trajectory_refusals(directory);
    test_parameters_round_trip(directory);
    test_parameters_refusals(directory);
    test_parameters_strings(directory);
    test_write_failures(directory);
    return gca_test::check_summary("test_scan_files");
}
