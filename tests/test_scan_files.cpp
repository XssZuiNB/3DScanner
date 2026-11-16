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
/// - a time in UTC is written and read as its text at the edges and between, worked out by hand;
///   a time outside 1970 to 9999 has no text, and a text of another form or a day the calendar
///   lacks is no time;
/// - the parameters of each object class, a set of neither, commits at the edges of the strings
///   the file holds and the scan's facts at their edges, each key of why a scan ended among them,
///   are read back equal; the document's example is read as the document says, also after white
///   space, and written as it shows it; the reader refuses a text that does not start with '{'
///   (YAML, XML), a key it does not know, a key twice, a missing key (a file written before the
///   keys of 2026-11-16 among them), another version, a value of another type, an unknown object
///   class or key of why a scan ended, a string OpenCV does not write as JSON, a time of another
///   form and each value out of range; the writer refuses such a string, set or fact and writes
///   nothing;
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
#include "scan/utc_time.hpp"
#include "test_files.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::bytes_of;
using gca_test::check;
using gca_test::check_refusal;
using gca_test::file_size_limit;
using gca_test::same_bits;
using gca_test::temporary_directory;
using gca_test::write_bytes;

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

/// Replaces a map of the top level, from its key to its closing brace, for a case built from a
/// valid file.
/// @param text [in] the valid file.
/// @param key [in] the map's key.
/// @param value [in] the text that stands after the key in the map's place.
/// @return the changed file.
std::string map_replaced(std::string text, const std::string &key, const std::string &value)
{
    const std::size_t at = text.find("\"" + key + "\": {");
    const std::size_t end = text.find("\n    }", at);
    check(at != std::string::npos && end != std::string::npos,
          ("test setup: the file holds the map " + key).c_str());
    if (at != std::string::npos && end != std::string::npos)
        text.replace(at, end + 6 - at, "\"" + key + "\": " + value);
    return text;
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
        const std::optional<std::vector<gca_test::ply_vertex>> vertices =
            gca_test::read_ply_file(path.string(), error);
        check(vertices.has_value(), (what + ": the tests' reader reads it; " + error).c_str());
        bool equal = vertices && vertices->size() == model.size();
        for (std::size_t k = 0; equal && k < model.size(); k++)
        {
            const gca_test::ply_vertex &v = (*vertices)[k];
            for (int c = 0; c < 3; c++)
            {
                equal = equal && same_bits(v.position_m[c], model[k].position_m[c]) &&
                        same_bits(v.normal[c], model[k].normal[c]) &&
                        v.color[c] == model[k].color[c];
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

// ------------------------------------------------------------------------------------- times

/// A time and its text, as the document's examples give them.
struct time_case
{
    std::int64_t us;  ///< microseconds since 1970-01-01T00:00:00Z
    std::string text; ///< its text
};

/// The times at the edges and between, each with its text worked out by hand
/// [S: computed, 365 days a year and 366 in a leap year]: the clock's epoch;
/// 2026-11-16T10:20:30.123456Z, 20773 days after it (56 years with 14 leap days, then 304 days of
/// January to October and 15 of November); the last microsecond of 2024-02-29, a leap day, 19782
/// days after the epoch; and the last microsecond of 9999-12-31, 253402300799 s after the epoch.
const std::vector<time_case> known_times = {
    {0, "19700101T000000.000000Z"},
    {(20773LL * 86400 + 10 * 3600 + 20 * 60 + 30) * 1000000 + 123456, "20261116T102030.123456Z"},
    {(19782LL * 86400 + 86399) * 1000000 + 999999, "20240229T235959.999999Z"},
    {253402300799LL * 1000000 + 999999, "99991231T235959.999999Z"},
};

/// Gives a time from its microseconds since the epoch.
/// @param us [in] the microseconds.
/// @return the time.
gca::utc_time at_us(std::int64_t us)
{
    return gca::utc_time(std::chrono::microseconds(us));
}

/// A time's text is written and read as worked out by hand at the edges and between; a time
/// before 1970 or after 9999 has no text; a text of another form, a day the calendar lacks, an
/// hour, minute or second out of range, or a year before 1970 is no time.
void test_utc_time()
{
    for (const time_case &known : known_times)
    {
        const std::optional<std::string> text = gca::utc_text(at_us(known.us));
        check(text && *text == known.text, ("utc time: " + known.text + " written").c_str());
        const std::optional<gca::utc_time> time = gca::utc_time_of(known.text);
        check(time && *time == at_us(known.us), ("utc time: " + known.text + " read").c_str());
    }
    check(!gca::utc_text(at_us(-1)), "utc time: 1 us before 1970 has no text");
    check(!gca::utc_text(at_us(253402300800LL * 1000000)), "utc time: 10000-01-01 has no text");
    const std::vector<std::string> refused = {
        "20261116T102030Z",
        "20261116T102030.12345Z",
        "20261116T102030.1234567Z",
        "2026-11-16T10:20:30.123Z",
        "20261116t102030.123456Z",
        "20261116T102030.123456z",
        "20261116T102030,123456Z",
        " 20261116T102030.12345Z",
        "2026111+T102030.123456Z",
        "20261131T000000.000000Z",
        "20250229T000000.000000Z",
        "20261100T000000.000000Z",
        "20261316T000000.000000Z",
        "20261116T240000.000000Z",
        "20261116T106000.000000Z",
        "20261116T102060.000000Z",
        "19691231T235959.999999Z",
        "00001116T102030.123456Z",
        "",
    };
    for (const std::string &text : refused)
        check(!gca::utc_time_of(text), ("utc time: \"" + text + "\" is no time").c_str());
}

// ------------------------------------------------------------------------------- parameters

/// The document's example of the parameters, a large object's scan, as written there.
const std::string parameters_by_hand = R"({
    "format_version": 1,
    "commit": "db91e0d",
    "object_class": "large_object",
    "parameters": {
        "live_voxel_size_m": 8.0000000000000002e-03,
        "full_voxel_size_m": 4.0000000000000001e-03,
        "min_depth_m": 2.5000000000000000e-01,
        "max_depth_m": 3.0,
        "memory_budget_bytes": 2000000000.0,
        "too_close_depth_m": 4.0000000000000002e-01,
        "depth_noise_per_m": 4.0000000000000001e-03,
        "truncation_noise_multiple": 2.5000000000000000e+00,
        "truncation_min_voxels": 3.0,
        "normals_radius_px": 2.0,
        "normals_min_points": 16.0,
        "normals_max_incidence_deg": 70.0,
        "max_distance_level_0_m": 1.0000000000000001e-01,
        "max_distance_level_1_m": 2.0000000000000001e-01,
        "max_distance_level_2_m": 3.4999999999999998e-01,
        "max_iterations_level_0": 20.0,
        "max_iterations_level_1": 10.0,
        "max_iterations_level_2": 5.0,
        "huber_threshold_m": 2.9999999999999999e-02,
        "point_to_plane_weight": 9.6799999999999997e-01,
        "min_fitness": 5.9999999999999998e-01,
        "max_inlier_rmse_m": 2.0000000000000000e-02,
        "max_turn_rate_deg_s": 90.0,
        "max_speed_m_s": 1.5000000000000000e+00
    },
    "stream": {
        "width_px": 640.0,
        "height_px": 400.0,
        "fx_px": 5.1725000000000000e+02,
        "fy_px": 5.1750000000000000e+02,
        "cx_px": 3.1975000000000000e+02,
        "cy_px": 2.0125000000000000e+02,
        "depth_unit_m": 1.0000000474974513e-03
    },
    "source": {
        "camera": "Orbbec Gemini 2 L",
        "firmware_version": "1.5.02",
        "sdk_version": "2.10.6"
    },
    "start_utc": "20261116T093012.250000Z",
    "stop_utc": "20261116T093512.750000Z",
    "frames": {
        "taken": 8990.0,
        "fused": 8950.0,
        "lost": 40.0,
        "not_taken": 12.0
    },
    "end": "stopped"
}
)";

/// The example's text of some values, which the refusal cases replace.
const std::string live_text = "8.0000000000000002e-03";
const std::string full_text = "4.0000000000000001e-03";
const std::string min_text = "2.5000000000000000e-01";
const std::string too_close_text = "4.0000000000000002e-01";
const std::string iterations_text = "\"max_iterations_level_0\": 20.0";
const std::string weight_text = "9.6799999999999997e-01";
const std::string fx_text = "5.1725000000000000e+02";
const std::string cx_text = "3.1975000000000000e+02";
const std::string unit_text = "1.0000000474974513e-03";
const std::string start_text = "20261116T093012.250000Z";
const std::string stop_text = "20261116T093512.750000Z";

/// The set of the document's example, a large object's scan, each value written out here and
/// every one another, so that two keys swapped show [S: docs/SCAN_FILES.md, "Parameters"].
const gca::object_class_parameters example_set = {.live_voxel_size_m = 0.008,
                                                  .full_voxel_size_m = 0.004,
                                                  .min_depth_m = 0.25,
                                                  .max_depth_m = 3.0,
                                                  .memory_budget_bytes = 2000000000u,
                                                  .too_close_depth_m = 0.4,
                                                  .depth_noise_per_m = 0.004,
                                                  .truncation_noise_multiple = 2.5,
                                                  .truncation_min_voxels = 3.0,
                                                  .normals_radius_px = 2,
                                                  .normals_min_points = 16,
                                                  .normals_max_incidence_deg = 70.0,
                                                  .max_distance_m = {0.1, 0.2, 0.35},
                                                  .max_iterations = {20, 10, 5},
                                                  .huber_threshold_m = 0.03,
                                                  .point_to_plane_weight = 0.968,
                                                  .min_fitness = 0.6,
                                                  .max_inlier_rmse_m = 0.02,
                                                  .max_turn_rate_deg_s = 90.0,
                                                  .max_speed_m_s = 1.5};

/// A set of neither object class whose doubles need all 17 digits, with the largest budget JSON
/// holds exactly and the largest counts the set holds, each value another, so that two keys
/// swapped show.
const gca::object_class_parameters odd_set = {.live_voxel_size_m = 1.0 / 3.0,
                                              .full_voxel_size_m = 0.1,
                                              .min_depth_m = 0.30000000000000004,
                                              .max_depth_m = 2.9999999999999996,
                                              .memory_budget_bytes = 9007199254740992u,
                                              .too_close_depth_m = 1.0 / 7.0 + 0.3,
                                              .depth_noise_per_m = 0.01 / 7.0,
                                              .truncation_noise_multiple = 10.0 / 3.0,
                                              .truncation_min_voxels = 11.0 / 3.0,
                                              .normals_radius_px = 4294967294u,
                                              .normals_min_points = 0,
                                              .normals_max_incidence_deg = 200.0 / 3.0,
                                              .max_distance_m = {0.1 / 3.0, 0.2 / 3.0, 0.4 / 3.0},
                                              .max_iterations = {4294967295u, 1, 7},
                                              .huber_threshold_m = 0.01 / 3.0,
                                              .point_to_plane_weight = 2.0 / 3.0,
                                              .min_fitness = 5.0 / 9.0,
                                              .max_inlier_rmse_m = 0.001 / 7.0,
                                              .max_turn_rate_deg_s = 100.0 / 7.0,
                                              .max_speed_m_s = 10.0 / 11.0};

/// The scan's facts of the document's example: 640x400 with intrinsics a float holds in few
/// digits, the depth unit 1 mm as a float, the camera with its versions, five minutes and half a
/// second, the frames counted, stopped by the operator [S: docs/SCAN_FILES.md, "Parameters"].
gca::scan_parameters example_parameters()
{
    return {"db91e0d",
            gca::object_class::large_object,
            example_set,
            {{640, 400}, {517.25f, 517.5f, 319.75f, 201.25f}, 0.001f},
            {"Orbbec Gemini 2 L", "1.5.02", "2.10.6"},
            *gca::utc_time_of(start_text),
            *gca::utc_time_of(stop_text),
            {8990, 8950, 40, 12},
            gca::scan_end::stopped};
}

/// Gives a scan's parameters with the scan's facts of the document's example.
/// @param commit [in] the commit.
/// @param kind [in] the object class.
/// @param set [in] the set.
/// @return the parameters.
gca::scan_parameters scan_of(const std::string &commit, gca::object_class kind,
                             const gca::object_class_parameters &set)
{
    gca::scan_parameters parameters = example_parameters();
    parameters.commit = commit;
    parameters.object_class = kind;
    parameters.parameters = set;
    return parameters;
}

/// The parameters of each object class, and a set of neither whose doubles need all 17 digits
/// with the largest budget JSON holds exactly and every value another, are written and read back
/// equal; commits at the edges of the strings the file holds too: one that starts with '{' and
/// holds double quotes, a backslash and every other printable ASCII character but the
/// apostrophe, an empty one and one of 4095 characters; the scan's facts at their edges: the
/// largest width and height, a float's smallest and largest values and negative coordinates in
/// the intrinsics, empty strings of the source and one of 4095 characters, the first and the last
/// time the file holds, the stop before the start, the largest counts of frames, and each key of
/// why a scan ended; the document's example is read as the document says, also after white
/// space, and written as the document shows it.
/// @param directory [in] the test's directory.
void test_parameters_round_trip(const temporary_directory &directory)
{
    std::vector<gca::scan_parameters> cases = {
        scan_of("db91e0d0123456789abcdef0123456789abcdef0", gca::object_class::tabletop_object,
                gca::parameters_of(gca::object_class::tabletop_object)),
        scan_of("831f06a", gca::object_class::large_object,
                gca::parameters_of(gca::object_class::large_object)),
        scan_of("{\"not a map\" [", gca::object_class::tabletop_object, odd_set),
        scan_of(" !\"#$%&()*+,-./0123456789:;<=>?@AZ[\\]^_`az{|}~", gca::object_class::large_object,
                gca::parameters_of(gca::object_class::large_object)),
        scan_of("", gca::object_class::large_object,
                gca::parameters_of(gca::object_class::large_object)),
        scan_of(std::string(4095, 'x'), gca::object_class::large_object,
                gca::parameters_of(gca::object_class::large_object)),
    };
    check(!(cases[0].parameters == cases[1].parameters),
          "parameters: the two object classes' sets differ");
    gca::scan_parameters edges = scan_of("db91e0d", gca::object_class::tabletop_object, odd_set);
    edges.stream = {{4294967295u, 1},
                    {std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min(),
                     -std::numeric_limits<float>::max(), -1.0f / 3.0f},
                    std::numeric_limits<float>::min()};
    edges.source = {"", std::string(4095, 'v'), ""};
    edges.start = at_us(known_times.back().us);
    edges.stop = at_us(0);
    edges.frames = {9007199254740992u, 9007199254740985u, 7, 9007199254740992u};
    cases.push_back(edges);
    for (int k = 0; !gca::name_of(static_cast<gca::scan_end>(k)).empty(); k++)
    {
        gca::scan_parameters ended = example_parameters();
        ended.end = static_cast<gca::scan_end>(k);
        // The frame tracking failed on is taken, neither fused nor lost.
        ended.frames = {ended.end == gca::scan_end::tracking_failed ? 1u : 0u, 0, 0, 0};
        ended.source = {"synthetic source", "", ""};
        cases.push_back(ended);
    }
    check(cases.size() == 14, "parameters: seven keys of why a scan ended");
    gca::scan_parameters failed = example_parameters();
    failed.frames = {120, 100, 19, 3};
    failed.end = gca::scan_end::tracking_failed;
    cases.push_back(failed);
    for (const gca::scan_parameters &parameters : cases)
    {
        const std::string what =
            "parameters, " + std::string(gca::name_of(parameters.object_class)) + ", commit " +
            parameters.commit.substr(0, 60) + " (" + std::to_string(parameters.commit.size()) +
            " characters), " + std::to_string(parameters.stream.size.width) + " px wide, ended " +
            std::string(gca::name_of(parameters.end));
        const std::filesystem::path path = directory / "parameters.json";
        const gca::write_result written = gca::write_parameters(path, parameters);
        check(written.written, (what + ": written; " + written.reason).c_str());
        const gca::read_result<gca::scan_parameters> read = gca::read_parameters(path);
        check(read.value && *read.value == parameters,
              (what + ": read back equal; " + read.reason).c_str());
        // By value is bit for bit for every number but a zero, whose sign == does not see: no
        // number of the set can be 0, and the principal point is checked bit for bit here.
        check(read.value &&
                  same_bits(read.value->stream.intrinsics.cx, parameters.stream.intrinsics.cx) &&
                  same_bits(read.value->stream.intrinsics.cy, parameters.stream.intrinsics.cy),
              (what + ": the principal point read back bit for bit").c_str());
    }

    // OpenCV writes -0.0 as 0.0 [M 2026-11-16: OpenCV 4.8.0], so a principal point of -0 comes
    // back as +0, equal by value (docs/SCAN_FILES.md, "Parameters").
    gca::scan_parameters zero = example_parameters();
    zero.stream.intrinsics.cx = -0.0f;
    const std::filesystem::path zero_path = directory / "zero.json";
    check(gca::write_parameters(zero_path, zero).written, "parameters: a cx of -0 written");
    const gca::read_result<gca::scan_parameters> zero_read = gca::read_parameters(zero_path);
    check(zero_read.value && *zero_read.value == zero &&
              same_bits(zero_read.value->stream.intrinsics.cx, 0.0f),
          "parameters: a cx of -0 read back as +0, equal by value");

    const std::filesystem::path hand_path = directory / "by_hand.json";
    write_bytes(hand_path, parameters_by_hand);
    const gca::read_result<gca::scan_parameters> hand = gca::read_parameters(hand_path);
    const gca::scan_parameters expected = example_parameters();
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

    const std::vector<std::string> end_keys = {"stopped",    "source_timeout",  "source_failure",
                                               "source_end", "tracking_failed", "writer_failed",
                                               "shut_down"};
    for (std::size_t k = 0; k < end_keys.size(); k++)
    {
        const std::optional<gca::scan_end> end = gca::scan_end_named(end_keys[k]);
        check(end && static_cast<std::size_t>(*end) == k &&
                  gca::name_of(static_cast<gca::scan_end>(k)) == end_keys[k],
              ("parameters: why a scan ended, the key " + end_keys[k]).c_str());
    }
    check(!gca::scan_end_named("paused") && !gca::scan_end_named(""),
          "parameters: no key of why a scan ended beyond the seven");
}

/// The parameters' writer refuses each kind of commit or string of the source that OpenCV does
/// not write as JSON, and writes nothing; the reader refuses such a string in a file written by
/// hand.
/// @param directory [in] the test's directory.
void test_parameters_strings(const temporary_directory &directory)
{
    const struct
    {
        const char *what;
        std::string text;
        const char *reason;
    } refused_strings[] = {
        {"an apostrophe", "it's", "holds the byte 0x27 at 2"},
        {"a control byte",
         "a\x01"
         "b",
         "holds the byte 0x01 at 1"},
        {"a tab", "a\tb", "holds the byte 0x09 at 1"},
        {"a byte above ASCII", "caf\xc3\xa9", "holds the byte 0xc3 at 3"},
        {"bounded by double quotes", "\"db91e0d\"", "starts and ends with a double quote"},
        {"one double quote", "\"", "starts and ends with a double quote"},
        {"4096 characters", std::string(4096, 'x'), "holds 4096 characters, more than 4095"},
    };
    for (const auto &refused : refused_strings)
    {
        const std::filesystem::path path = directory / "refused_write.json";
        gca::scan_parameters parameters = example_parameters();
        parameters.commit = refused.text;
        check_write_refused(std::string("parameters writer, a commit: ") + refused.what,
                            gca::write_parameters(path, parameters), path,
                            std::string("\"commit\" ") + refused.reason);
        parameters = example_parameters();
        parameters.source.firmware_version = refused.text;
        check_write_refused(std::string("parameters writer, a firmware version: ") + refused.what,
                            gca::write_parameters(path, parameters), path,
                            std::string("\"firmware_version\" in \"source\" ") + refused.reason);
    }
    gca::scan_parameters sdk = example_parameters();
    sdk.source.sdk_version = "2.10'6";
    check_write_refused("parameters writer, an SDK version with an apostrophe",
                        gca::write_parameters(directory / "refused_write.json", sdk),
                        directory / "refused_write.json",
                        "\"sdk_version\" in \"source\" holds the byte 0x27 at 4");
    gca::scan_parameters camera = example_parameters();
    camera.source.camera = "sequence /home/o'neill/scan";
    check_write_refused("parameters writer, a sequence's path with an apostrophe as the camera",
                        gca::write_parameters(directory / "refused_write.json", camera),
                        directory / "refused_write.json",
                        "\"camera\" in \"source\" holds the byte 0x27 at 16");

    // JSON that OpenCV reads, with strings the writer refuses.
    const std::vector<refusal_case> cases = {
        {"a commit with an apostrophe", replaced(parameters_by_hand, "db91e0d", "it's"),
         "\"commit\" holds the byte 0x27 at 2"},
        {"a commit bounded by double quotes",
         replaced(parameters_by_hand, "\"db91e0d\"", "\"\\\"db91e0d\\\"\""),
         "\"commit\" starts and ends with a double quote"},
        {"an SDK version with an apostrophe", replaced(parameters_by_hand, "2.10.6", "2.10'6"),
         "\"sdk_version\" in \"source\" holds the byte 0x27 at 4"},
        {"a camera bounded by double quotes",
         replaced(parameters_by_hand, "\"Orbbec Gemini 2 L\"", "\"\\\"Orbbec\\\"\""),
         "\"camera\" in \"source\" starts and ends with a double quote"},
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
/// does not exist; the writer refuses a set or a fact out of range and writes nothing.
/// @param directory [in] the test's directory.
void test_parameters_refusals(const temporary_directory &directory)
{
    const std::string &valid = parameters_by_hand;
    const std::string top = "at the top level";
    const std::string nested = "in \"parameters\"";
    const std::string set = "\" in \"parameters\" is ";
    const std::string in_stream = "\" in \"stream\" is ";
    const std::string in_frames = "\" in \"frames\" is ";
    const std::string not_real = "is no number with a decimal point or an exponent";
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
        {"the set a number", map_replaced(valid, "parameters", "1.0"),
         "\"parameters\" is no JSON object"},
        {"a length written as an integer",
         replaced(valid, "\"max_depth_m\": 3.0", "\"max_depth_m\": 3"),
         "\"max_depth_m\" in \"parameters\" " + not_real},
        {"a length written as a string", replaced(valid, min_text, "\"0.25\""),
         "\"min_depth_m\" in \"parameters\" " + not_real},
        {"a budget written as an integer", replaced(valid, "2000000000.0", "2000000000"),
         "\"memory_budget_bytes\" in \"parameters\" " + not_real},
        {"a voxel size below 0", replaced(valid, full_text, "-4.0e-03"),
         "\"full_voxel_size_m" + set + "-0.004, not a finite number > 0"},
        {"a voxel size of 0", replaced(valid, live_text, "0.0"),
         "\"live_voxel_size_m" + set + "0, not a finite number > 0"},
        {"a depth not finite", replaced(valid, min_text, ".Nan"), "\"min_depth_m" + set},
        {"min_depth_m equal to max_depth_m", replaced(valid, min_text, "3.0"),
         "\"min_depth_m\" 3 is not less than \"max_depth_m\" 3"},
        {"min_depth_m above max_depth_m", replaced(valid, min_text, "3.5"),
         "\"min_depth_m\" 3.5 is not less than \"max_depth_m\" 3"},
        {"a budget with a fraction", replaced(valid, "2000000000.0", "2000000000.5"),
         "\"memory_budget_bytes" + set + "2000000000.5, not a whole number from 1 to 2^53"},
        {"a budget of 0", replaced(valid, "2000000000.0", "0.0"),
         "\"memory_budget_bytes" + set + "0, not a whole number from 1 to 2^53"},
        {"a budget above 2^53", replaced(valid, "2000000000.0", "9007199254740994.0"),
         "\"memory_budget_bytes" + set + "9007199254740994, not a whole number from 1 to 2^53"},
        {"a file written before the set's keys of 2026-11-15",
         valid.substr(0, valid.find(",\n        \"too_close_depth_m\"")) +
             valid.substr(valid.find("\n    }", valid.find("\"too_close_depth_m\""))),
         "no key \"too_close_depth_m\" " + nested},
        {"a file written before the set's keys of 2026-11-16",
         replaced(valid, "        \"depth_noise_per_m\": 4.0000000000000001e-03,\n", ""),
         "no key \"depth_noise_per_m\" " + nested},
        {"a file written before the scan's facts of 2026-11-16",
         valid.substr(0, valid.find(",\n    \"stream\"")) + "\n}\n", "no key \"stream\" " + top},
        {"no min_fitness",
         replaced(valid, "        \"min_fitness\": 5.9999999999999998e-01,\n", ""),
         "no key \"min_fitness\" " + nested},
        {"a count written as an integer",
         replaced(valid, iterations_text, "\"max_iterations_level_0\": 20"),
         "\"max_iterations_level_0\" in \"parameters\" " + not_real},
        {"a count with a fraction",
         replaced(valid, iterations_text, "\"max_iterations_level_0\": 20.5"),
         "\"max_iterations_level_0" + set + "20.5, not a whole number from 0 to 4294967295"},
        {"a count below 0", replaced(valid, iterations_text, "\"max_iterations_level_0\": -1.0"),
         "\"max_iterations_level_0" + set + "-1, not a whole number from 0 to 4294967295"},
        {"a count above 2^32 - 1",
         replaced(valid, iterations_text, "\"max_iterations_level_0\": 4294967296.0"),
         "\"max_iterations_level_0" + set + "4294967296, not a whole number from 0 to 4294967295"},
        {"the normals' least points with a fraction",
         replaced(valid, "\"normals_min_points\": 16.0", "\"normals_min_points\": 16.5"),
         "\"normals_min_points" + set + "16.5, not a whole number from 0 to 4294967295"},
        {"the depth noise of 0",
         replaced(valid, "4.0000000000000001e-03,\n        \"trunc", "0.0,\n        \"trunc"),
         "\"depth_noise_per_m" + set + "0, not a finite number > 0"},
        {"a share above 1", replaced(valid, weight_text, "1.5"),
         "\"point_to_plane_weight" + set + "1.5, not a number in (0, 1]"},
        {"a share of 0",
         replaced(valid, "\"min_fitness\": 5.9999999999999998e-01", "\"min_fitness\": 0.0"),
         "\"min_fitness" + set + "0, not a number in (0, 1]"},
        {"a threshold of 0",
         replaced(valid, "\"max_speed_m_s\": 1.5000000000000000e+00", "\"max_speed_m_s\": 0.0"),
         "\"max_speed_m_s" + set + "0, not a finite number > 0"},
        {"too_close_depth_m equal to min_depth_m", replaced(valid, too_close_text, min_text),
         "\"too_close_depth_m\" 0.25 does not lie between \"min_depth_m\" 0.25 and \"max_depth_m\" "
         "3"},
        {"too_close_depth_m equal to max_depth_m", replaced(valid, too_close_text, "3.0"),
         "\"too_close_depth_m\" 3 does not lie between \"min_depth_m\" 0.25 and \"max_depth_m\" 3"},
        {"the stream a string", map_replaced(valid, "stream", "\"640x400\""),
         "\"stream\" is no JSON object"},
        {"no stream's depth unit", replaced(valid, ",\n        \"depth_unit_m\": " + unit_text, ""),
         "no key \"depth_unit_m\" in \"stream\""},
        {"an unknown key in the stream",
         replaced(valid, "\"width_px\"", "\"k1\": 0.1,\n        \"width_px\""),
         "unknown key \"k1\" in \"stream\""},
        {"a width of 0", replaced(valid, "\"width_px\": 640.0", "\"width_px\": 0.0"),
         "\"width_px" + in_stream + "0, not a whole number from 1 to 4294967295"},
        {"a width above 2^32 - 1",
         replaced(valid, "\"width_px\": 640.0", "\"width_px\": 4294967296.0"),
         "\"width_px" + in_stream + "4294967296, not a whole number from 1 to 4294967295"},
        {"a height with a fraction",
         replaced(valid, "\"height_px\": 400.0", "\"height_px\": 400.5"),
         "\"height_px" + in_stream + "400.5, not a whole number from 1 to 4294967295"},
        {"a width written as an integer",
         replaced(valid, "\"width_px\": 640.0", "\"width_px\": 640"),
         "\"width_px\" in \"stream\" " + not_real},
        {"a focal length of 0", replaced(valid, fx_text, "0.0"),
         "\"fx_px" + in_stream + "0, not a finite number > 0"},
        {"a focal length no float holds", replaced(valid, fx_text, "1.0e+300"),
         "\"fx_px" + in_stream + "1e+300, not a finite number that a float holds"},
        {"a principal point not finite", replaced(valid, cx_text, ".Nan"), "\"cx_px" + in_stream},
        {"a depth unit below 0", replaced(valid, unit_text, "-1.0e-03"),
         "\"depth_unit_m" + in_stream + "-0.0010000000474974513, not a finite number > 0"},
        {"a depth unit that rounds to 0 as a float", replaced(valid, unit_text, "1.0e-50"),
         "\"depth_unit_m" + in_stream + "0, not a finite number > 0"},
        {"the source a list", map_replaced(valid, "source", "[ 1.0 ]"),
         "\"source\" is no JSON object"},
        {"the frames a number", map_replaced(valid, "frames", "8990.0"),
         "\"frames\" is no JSON object"},
        {"no SDK version", replaced(valid, ",\n        \"sdk_version\": \"2.10.6\"", ""),
         "no key \"sdk_version\" in \"source\""},
        {"the camera a number", replaced(valid, "\"Orbbec Gemini 2 L\"", "2.0"),
         "\"camera\" in \"source\" is no string"},
        {"the start a number", replaced(valid, "\"" + start_text + "\"", "1.0"),
         "\"start_utc\" is no string"},
        {"the start in ISO 8601's extended format",
         replaced(valid, start_text, "2026-11-16T09:30:12.250000Z"),
         "\"start_utc\" is \"2026-11-16T09:30:12.250000Z\", not a time in UTC as "
         "YYYYMMDDTHHMMSS.ffffffZ from 1970 to 9999"},
        {"the stop on a day the calendar lacks",
         replaced(valid, stop_text, "20261131T093512.750000Z"),
         "\"stop_utc\" is \"20261131T093512.750000Z\", not a time in UTC"},
        {"no stop", replaced(valid, "    \"stop_utc\": \"" + stop_text + "\",\n", ""),
         "no key \"stop_utc\" " + top},
        {"a count of frames written as an integer",
         replaced(valid, "\"taken\": 8990.0", "\"taken\": 8990"),
         "\"taken\" in \"frames\" " + not_real},
        {"a count of frames below 0", replaced(valid, "\"not_taken\": 12.0", "\"not_taken\": -1.0"),
         "\"not_taken" + in_frames + "-1, not a whole number from 0 to 2^53"},
        {"a count of frames above 2^53",
         replaced(valid, "\"not_taken\": 12.0", "\"not_taken\": 9007199254740994.0"),
         "\"not_taken" + in_frames + "9007199254740994, not a whole number from 0 to 2^53"},
        {"frames taken that are not those fused and lost",
         replaced(valid, "\"lost\": 40.0", "\"lost\": 39.0"),
         "\"taken\" in \"frames\" 8990 is not \"fused\" 8950 and \"lost\" 39 together, for "
         "\"end\" \"stopped\""},
        {"tracking failed, and the frames taken those fused and lost alone",
         replaced(valid, "\"end\": \"stopped\"", "\"end\": \"tracking_failed\""),
         "\"taken\" in \"frames\" 8990 is not \"fused\" 8950, \"lost\" 40 and the frame tracking "
         "failed on together, for \"end\" \"tracking_failed\""},
        {"no frames not taken", replaced(valid, ",\n        \"not_taken\": 12.0", ""),
         "no key \"not_taken\" in \"frames\""},
        {"an end not of the closed set",
         replaced(valid, "\"end\": \"stopped\"", "\"end\": \"paused\""),
         "\"end\" is \"paused\", no key of why a scan ended (stopped, source_timeout, "
         "source_failure, source_end, tracking_failed, writer_failed, shut_down)"},
        {"an end as free text",
         replaced(valid, "\"end\": \"stopped\"", "\"end\": \"the camera failed: USB\""),
         "\"end\" is \"the camera failed: USB\", no key of why a scan ended"},
        {"the end a number", replaced(valid, "\"end\": \"stopped\"", "\"end\": 0"),
         "\"end\" is no string"},
        {"no end", replaced(valid, ",\n    \"end\": \"stopped\"", ""), "no key \"end\" " + top},
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
        // The not-finite cases: the reason names the key and the rule; the NaN's sign is
        // OpenCV's.
        if (refusal.what == "a depth not finite")
            check(read.reason.find("not a finite number > 0") != std::string::npos,
                  "parameters, a depth not finite: the reason says not finite");
        if (refusal.what == "a principal point not finite")
            check(read.reason.find("not a finite number that a float holds") != std::string::npos,
                  "parameters, a principal point not finite: the reason says not finite");
    }
    check_read_refused("parameters, a missing file",
                       gca::read_parameters(directory / "missing.json"), "cannot open");

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    // The document's example with one value changed.
    const auto with = [](auto change) {
        gca::scan_parameters parameters = example_parameters();
        change(parameters);
        return parameters;
    };
    const struct
    {
        const char *what;
        gca::scan_parameters parameters;
        std::string reason;
    } refused[] = {
        {"a voxel size not finite", with([nan](auto &p) { p.parameters.live_voxel_size_m = nan; }),
         "\"live_voxel_size_m" + set},
        {"a depth of 0", with([](auto &p) { p.parameters.min_depth_m = 0.0; }),
         "\"min_depth_m" + set + "0, not a finite number > 0"},
        {"min_depth_m above max_depth_m", with([](auto &p) {
             p.parameters.min_depth_m = 2.0;
             p.parameters.max_depth_m = 1.0;
         }),
         "\"min_depth_m\" 2 is not less than \"max_depth_m\" 1"},
        {"a budget of 0", with([](auto &p) { p.parameters.memory_budget_bytes = 0; }),
         "\"memory_budget_bytes" + set + "0, not a whole number from 1 to 2^53"},
        {"a budget of 2^53 + 1",
         with([](auto &p) { p.parameters.memory_budget_bytes = 9007199254740993u; }),
         "\"memory_budget_bytes" + set + "9007199254740993, not a whole number from 1 to 2^53"},
        {"too_close_depth_m beyond max_depth_m",
         with([](auto &p) { p.parameters.too_close_depth_m = 3.5; }),
         "\"too_close_depth_m\" 3.5 does not lie between \"min_depth_m\" 0.25 and "
         "\"max_depth_m\" 3"},
        {"a largest distance of 0", with([](auto &p) { p.parameters.max_distance_m[2] = 0.0; }),
         "\"max_distance_level_2_m" + set + "0, not a finite number > 0"},
        {"a least fitness above 1", with([](auto &p) { p.parameters.min_fitness = 1.25; }),
         "\"min_fitness" + set + "1.25, not a number in (0, 1]"},
        {"an RMSE threshold not finite", with([](auto &p) {
             p.parameters.max_inlier_rmse_m = std::numeric_limits<double>::infinity();
         }),
         "\"max_inlier_rmse_m" + set + "inf, not a finite number > 0"},
        {"a truncation multiple below 0",
         with([](auto &p) { p.parameters.truncation_noise_multiple = -3.0; }),
         "\"truncation_noise_multiple" + set + "-3, not a finite number > 0"},
        {"a steepest incidence not finite",
         with([nan](auto &p) { p.parameters.normals_max_incidence_deg = nan; }),
         "\"normals_max_incidence_deg" + set},
        {"a height of 0", with([](auto &p) { p.stream.size.height = 0; }),
         "\"height_px" + in_stream + "0, not a whole number from 1 to 4294967295"},
        {"a focal length not finite",
         with([](auto &p) { p.stream.intrinsics.fy = std::numeric_limits<float>::quiet_NaN(); }),
         "\"fy_px" + in_stream},
        {"a principal point not finite", with([inf](auto &p) { p.stream.intrinsics.cy = inf; }),
         "\"cy_px" + in_stream + "inf, not a finite number"},
        {"a depth unit of 0", with([](auto &p) { p.stream.depth_unit_m = 0.0f; }),
         "\"depth_unit_m" + in_stream + "0, not a finite number > 0"},
        {"a start before 1970", with([](auto &p) { p.start = at_us(-1); }),
         "\"start_utc\" is -1 us after 1970-01-01T00:00:00Z, not a time from 1970 to 9999"},
        {"a stop after 9999", with([](auto &p) { p.stop = at_us(253402300800LL * 1000000); }),
         "\"stop_utc\" is 253402300800000000 us after 1970-01-01T00:00:00Z, not a time from 1970 "
         "to 9999"},
        {"a count of frames above 2^53",
         with([](auto &p) { p.frames = {9007199254740993u, 9007199254740993u, 0, 0}; }),
         "\"taken" + in_frames + "9007199254740993, not a whole number from 0 to 2^53"},
        {"frames taken that are not those fused and lost",
         with([](auto &p) { p.frames = {10, 7, 2, 0}; }),
         "\"taken\" in \"frames\" 10 is not \"fused\" 7 and \"lost\" 2 together"},
        {"the frame tracking failed on counted, the scan stopped by the operator",
         with([](auto &p) { p.frames = {120, 100, 19, 3}; }),
         "\"taken\" in \"frames\" 120 is not \"fused\" 100 and \"lost\" 19 together, for "
         "\"end\" \"stopped\""},
        {"tracking failed, and the frame it failed on not counted", with([](auto &p) {
             p.frames = {119, 100, 19, 3};
             p.end = gca::scan_end::tracking_failed;
         }),
         "\"taken\" in \"frames\" 119 is not \"fused\" 100, \"lost\" 19 and the frame tracking "
         "failed on together, for \"end\" \"tracking_failed\""},
        {"an end that is no enumerator",
         with([](auto &p) { p.end = static_cast<gca::scan_end>(7); }),
         "\"end\" is 7, no enumerator of why a scan ended"},
    };
    for (const auto &r : refused)
    {
        const std::filesystem::path path = directory / "refused_write.json";
        check_write_refused(std::string("parameters writer, ") + r.what,
                            gca::write_parameters(path, r.parameters), path, r.reason);
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
    const gca::scan_parameters parameters = example_parameters();
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

    const gca::write_result cut = [&] {
        const file_size_limit limit(100);
        return gca::write_model(path, test_model());
    }();

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
    const temporary_directory directory("test_scan_files");
    test_model_round_trip(directory);
    test_model_refusals(directory);
    test_model_normals(directory);
    test_trajectory_round_trip(directory);
    test_trajectory_refusals(directory);
    test_utc_time();
    test_parameters_round_trip(directory);
    test_parameters_refusals(directory);
    test_parameters_strings(directory);
    test_write_failures(directory);
    return gca_test::check_summary("test_scan_files");
}
