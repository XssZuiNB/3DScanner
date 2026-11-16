// SPDX-License-Identifier: 0BSD
/// @file test_sequence.cpp
/// The sequence (docs/SEQUENCE_FILE.md) through the camera module's public API: frames written
/// by the sequence's writer into a temporary directory, which the test removes, and read back by
/// its reader as a frame source.
/// - Frames of the synthetic source, the walk around the tabletop box with its IMU at 200 Hz, with
///   poses, both marks and gaps in the indices, are read back bit for bit: depth, color, index,
///   timestamp, the IMU's samples, the pose or the mark, and the header's description; then the
///   end, at every later call. A reader opened again, in the middle or after the end, starts at
///   the first frame.
/// - The writer writes the bytes the document lays out, built here from its tables, samples with
///   a negative zero and a NaN's payload among them, and a frame without samples.
/// - A queue of 2 frames reports full to the caller, through has_room() and push(); a push after
///   has_room() said yes is accepted; after close() every accepted frame is on disk, in order,
///   and no other, and frames_written() counts them.
/// - The writer refuses, as a value with the reason: not open, a queue of 0, a file that cannot
///   be created (it exists, no directory, a directory it cannot write), a header the disk
///   refuses (no file is left), a second open, a frame of other intrinsics or another depth
///   unit, an index or a timestamp that does not increase, a sample that does not follow its
///   sensor's last or lies after its frame's timestamp, more samples of a sensor than a record
///   holds; a frame refused or not taken is left as it was. A disk that refuses a write (a file
///   size limit) reaches push() and close() as the reason, also when it refuses the end alone, and
///   the frames written before it are read back and counted by frames_written().
/// - The reader refuses, as a value with the reason: a missing file, a file that is no sequence,
///   another version, version 1 among them, a header cut short, another size; a file cut at every
///   byte, within the samples too, gives every whole frame before the cut, then the early end with
///   the record's number, which ended_early() tells apart, the frame given left as it was; an
///   unknown kind, an index or a timestamp that does not increase, a sample that does not follow
///   its sensor's last or lies after its frame's timestamp, bytes after the end, none of them an
///   early end.
/// Every writer's thread ends within its test. Needs no GPU and no camera.

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/sequence_format.hpp"
#include "camera/sequence_reader.hpp"
#include "camera/sequence_writer.hpp"
#include "check.hpp"
#include "frame_copy.hpp"
#include "synthetic_source.hpp"
#include "test_files.hpp"
#include "types/camera_pose.hpp"
#include "types/frame_description.hpp"
#include "types/imu_sample.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::bytes_of;
using gca_test::check;
using gca_test::check_refusal;
using gca_test::same_samples;
using gca_test::temporary_directory;
using gca_test::write_bytes;

/// The walk's pictures: small, so that the test stays fast [?: a choice].
constexpr gca::frame_size walk_size = {160, 120};

/// The walk's intrinsics, each with bits a rounding would change [?: a choice].
constexpr gca::intrinsics walk_intrinsics = {150.3f, 140.7f, 83.3f, 56.6f};

/// The walk's depth unit, 0.1 mm [?: a choice].
constexpr float walk_depth_unit_m = 0.0001f;

/// The walk's IMU period, 200 Hz [?: a choice, the camera's rate, docs/methods/imu-samples.md].
constexpr std::uint64_t walk_imu_period_us = 5000;

/// One frame as the test pushed it, kept for the comparison after reading.
struct pushed_frame
{
    std::uint64_t index;                ///< its index
    std::uint64_t timestamp_us;         ///< its timestamp, in µs
    std::vector<std::uint16_t> depth;   ///< its depth values
    std::vector<std::uint8_t> color;    ///< its color bytes
    std::vector<gca::imu_sample> accel; ///< its accelerometer's samples
    std::vector<gca::imu_sample> gyro;  ///< its gyroscope's samples
    gca::frame_tracking tracking;       ///< its pose or mark
};

/// Keeps a copy of a frame before the push swaps its buffers away.
/// @param one_frame [in] the frame.
/// @param tracking [in] its pose or mark.
/// @return the copy.
pushed_frame copy_of(const gca::frame &one_frame, const gca::frame_tracking &tracking)
{
    return {one_frame.index(),
            one_frame.timestamp_us(),
            {one_frame.depth().begin(), one_frame.depth().end()},
            {one_frame.color().begin(), one_frame.color().end()},
            {one_frame.accel().begin(), one_frame.accel().end()},
            {one_frame.gyro().begin(), one_frame.gyro().end()},
            tracking};
}

/// Whether two trackings are the same: the same mark, or poses with the same bits.
/// @param a [in] one tracking.
/// @param b [in] the other.
/// @return whether they are equal, bit for bit.
bool same_tracking(const gca::frame_tracking &a, const gca::frame_tracking &b)
{
    if (a.index() != b.index())
        return false;
    const gca::camera_pose *pa = std::get_if<gca::camera_pose>(&a);
    const gca::camera_pose *pb = std::get_if<gca::camera_pose>(&b);
    return !pa || gca_test::same_pose(*pa, *pb);
}

/// Whether a frame read is the frame pushed, bit for bit.
/// @param read [in] the frame the reader gave.
/// @param tracking [in] the tracking the reader gave with it.
/// @param pushed [in] the frame the test pushed.
/// @return whether index, timestamp, every depth value and color byte, every sample bit for bit
///         and the tracking are equal.
bool same_frame(const gca::frame &read, const gca::frame_tracking &tracking,
                const pushed_frame &pushed)
{
    return read.index() == pushed.index && read.timestamp_us() == pushed.timestamp_us &&
           std::equal(read.depth().begin(), read.depth().end(), pushed.depth.begin(),
                      pushed.depth.end()) &&
           std::equal(read.color().begin(), read.color().end(), pushed.color.begin(),
                      pushed.color.end()) &&
           same_samples(read.accel(), pushed.accel) && same_samples(read.gyro(), pushed.gyro) &&
           same_tracking(tracking, pushed.tracking);
}

/// Gives a pose from a pose of the synthetic source's trajectory.
/// @param camera_to_world [in] the pose, camera frame to the scene's frame.
/// @return the same pose as the sequence holds it.
gca::camera_pose pose_of(const Eigen::Isometry3d &camera_to_world)
{
    return {camera_to_world.translation(), Eigen::Quaterniond(camera_to_world.rotation())};
}

/// Renders the walk once around the tabletop box (tests/synthetic_source.hpp): 12 frames of the
/// synthetic source, with its IMU's samples at 200 Hz.
/// @param trajectory [out] the walk's trajectory, camera frame to the scene's frame.
/// @return the frames, index 0 to 11; empty when the walk could not be built.
std::vector<gca::frame> render_walk(std::vector<Eigen::Isometry3d> &trajectory)
{
    const std::optional<std::vector<Eigen::Isometry3d>> walk =
        gca_test::walk_around(gca_test::tabletop_walk());
    check(walk.has_value(), "test setup: the walk around the box");
    if (!walk)
        return {};
    trajectory = *walk;
    gca_test::synthetic_source_options options;
    options.description = {walk_size, walk_intrinsics, walk_depth_unit_m};
    options.scene = {
        .plane = gca_test::tabletop_plane, .box = gca_test::tabletop_box, .trajectory = trajectory};
    options.imu = gca_test::synthetic_imu{walk_imu_period_us};
    gca_test::synthetic_source source(options);
    check(source.open(walk_size).opened, "test setup: the synthetic source opened");
    std::vector<gca::frame> frames(trajectory.size());
    for (gca::frame &one_frame : frames)
        check(source.next(one_frame).status == gca::next_status::frame,
              "test setup: a frame of the walk");
    source.close();
    return frames;
}

/// Reads a whole sequence and compares it with the frames pushed.
/// @param name [in] the case, printed in the check lines.
/// @param path [in] the sequence.
/// @param size [in] the size its frames have.
/// @param pushed [in] the frames pushed and accepted, in order.
/// @param description [in] the description every frame read has, bit for bit.
void check_read_back(const std::string &name, const std::filesystem::path &path,
                     const gca::frame_size &size, const std::vector<pushed_frame> &pushed,
                     const gca::frame_description &description)
{
    gca::sequence_reader reader(path);
    const gca::open_result opened = reader.open(size);
    check(opened.opened, (name + ": the reader opened; " + opened.refusal).c_str());
    gca::frame read;
    std::size_t equal = 0;
    for (const pushed_frame &expected : pushed)
    {
        const gca::next_result result = reader.next(read);
        if (result.status != gca::next_status::frame)
        {
            std::printf("%s: no frame where index %llu was pushed: %s\n", name.c_str(),
                        static_cast<unsigned long long>(expected.index), result.message.c_str());
            break;
        }
        if (same_frame(read, reader.tracking(), expected) &&
            same_description(read.description(), description))
            equal++;
    }
    std::printf("%s: %zu of %zu frames read back equal\n", name.c_str(), equal, pushed.size());
    check(equal == pushed.size(), (name + ": every frame read back equal, bit for bit").c_str());
    const gca::next_result end = reader.next(read);
    const gca::next_result again = reader.next(read);
    std::printf("%s: then %s\n", name.c_str(), end.message.c_str());
    check(end.status == gca::next_status::end && again.status == gca::next_status::end &&
              end.message == again.message && !reader.ended_early(),
          (name + ": then the end, at every later call, and no early end").c_str());
}

/// Frames of a walk around a box with poses, both marks and gaps in the indices are read back
/// bit for bit; the reader describes itself by the path, and an opening again starts at the
/// first frame.
/// @param directory [in] the test's directory.
void test_round_trip(const temporary_directory &directory)
{
    std::vector<Eigen::Isometry3d> trajectory;
    std::vector<gca::frame> frames = render_walk(trajectory);
    if (frames.empty())
        return;
    const gca::frame_description description = frames[0].description();

    // Pose, tracking loss, not tracked in turn; frames 4 and 7 are not pushed, gaps in the
    // indices. Two poses carry bits a careless copy would lose: a negative zero, and the
    // quaternion -q for q, which the file keeps as given.
    std::vector<gca::frame_tracking> trackings;
    for (std::size_t k = 0; k < frames.size(); k++)
    {
        if (k % 3 == 0)
            trackings.push_back(pose_of(trajectory[k]));
        else if (k % 3 == 1)
            trackings.push_back(gca::tracking_loss{});
        else
            trackings.push_back(gca::not_tracked{});
    }
    gca::camera_pose odd = pose_of(trajectory[3]);
    odd.position_m.x() = -0.0;
    odd.orientation.coeffs() = -odd.orientation.coeffs();
    trackings[3] = odd;
    gca::camera_pose third = pose_of(trajectory[9]);
    third.position_m.z() = 1.0 / 3.0;
    trackings[9] = third;

    const std::filesystem::path path = directory / "walk.sequence";
    std::vector<pushed_frame> pushed;
    {
        gca::sequence_writer writer;
        const gca::open_result opened = writer.open(path, description, gca::sequence_queue_frames);
        check(opened.opened, ("walk: the writer opened; " + opened.refusal).c_str());
        // Every copy first, then the pushes one after another, so that close() finds frames
        // still in the queue.
        for (std::size_t k = 0; k < frames.size(); k++)
        {
            if (k != 4 && k != 7)
                pushed.push_back(copy_of(frames[k], trackings[k]));
        }
        std::size_t accepted = 0;
        for (std::size_t k = 0; k < frames.size(); k++)
        {
            if (k != 4 && k != 7)
                accepted +=
                    writer.push(frames[k], trackings[k]).status == gca::push_status::accepted;
        }
        check(accepted == pushed.size(), "walk: every push accepted");
        const gca::write_result closed = writer.close();
        check(closed.written, ("walk: the writer closed; " + closed.reason).c_str());
        check(writer.frames_written() == pushed.size(), "walk: every frame counted as written");
    }
    check_read_back("walk", path, walk_size, pushed, description);

    gca::sequence_reader reader(path);
    const gca::source_description described = reader.describe();
    std::printf("walk: the reader describes itself as \"%s\"\n", described.camera.c_str());
    check(described.camera == "sequence " + path.string() && described.firmware_version.empty() &&
              described.sdk_version.empty(),
          "walk: the reader describes itself by the path, without versions");
    gca::frame read;
    check(reader.open(walk_size).opened && reader.next(read).status == gca::next_status::frame,
          "walk: the first opening's first frame");
    reader.close();
    const bool reopened = reader.open(walk_size).opened &&
                          reader.next(read).status == gca::next_status::frame &&
                          same_frame(read, reader.tracking(), pushed[0]);
    check(reopened, "walk: an opening again starts at the first frame");

    // Read to the end, then open again: the first frame again, not the end.
    gca::next_result result;
    while ((result = reader.next(read)).status == gca::next_status::frame)
    {
    }
    check(result.status == gca::next_status::end && !reader.ended_early(),
          "walk: the end, which is no early end");
    reader.close();
    const bool after_end = reader.open(walk_size).opened &&
                           reader.next(read).status == gca::next_status::frame &&
                           same_frame(read, reader.tracking(), pushed[0]);
    check(after_end, "walk: an opening again after the end starts at the first frame");
}

/// Appends a number's bytes, the least significant first, as docs/SEQUENCE_FILE.md lays it out.
/// @param bytes [in,out] the file's bytes so far.
/// @param value [in] the number.
/// @param count [in] its bytes: 2, 4 or 8.
void append_little_endian(std::string &bytes, std::uint64_t value, int count)
{
    for (int k = 0; k < count; k++)
        bytes += static_cast<char>((value >> (8 * k)) & 0xff);
}

/// The writer writes the bytes docs/SEQUENCE_FILE.md lays out, built here from the document's
/// tables apart from the writer's code: a header, a frame with a pose and samples of both sensors,
/// one in tracking loss with a gyroscope's sample alone, one not tracked without samples, the end.
/// @param directory [in] the test's directory.
void test_bytes_as_documented(const temporary_directory &directory)
{
    const gca::frame_description description = {{2, 1}, {1.5f, -2.25f, 0.5f, -0.0f}, 0.001f};
    const gca::camera_pose pose = {Eigen::Vector3d(0.5, -1.25, 2.0),
                                   Eigen::Quaterniond(0.88, 0.2, 0.4, 0.16)};
    const std::uint64_t indices[3] = {5, 0x0102030405060708, 0x0102030405060709};
    const std::uint64_t timestamps[3] = {1000, 0x1112131415161718, 0x1112131415161719};
    const gca::frame_tracking trackings[3] = {pose, gca::tracking_loss{}, gca::not_tracked{}};
    // Values whose bits a careless copy would change: a negative zero, a NaN's payload, a
    // subnormal.
    const float nan_with_payload = std::bit_cast<float>(std::uint32_t{0x7fc12345});
    const std::vector<gca::imu_sample> accels[3] = {
        {{990, 1.5f, -0.0f, nan_with_payload}, {1000, -9.80665f, 0.25f, 3e-39f}}, {}, {}};
    const std::vector<gca::imu_sample> gyros[3] = {
        {{995, 0.125f, -2.5f, 1e10f}}, {{0x1112131415161700, 7.0f, 8.0f, 9.0f}}, {}};

    const std::filesystem::path path = directory / "documented.sequence";
    {
        gca::sequence_writer writer;
        check(writer.open(path, description, 2).opened, "documented: the writer opened");
        for (int k = 0; k < 3; k++)
        {
            gca::frame one_frame;
            one_frame.prepare(description, indices[k], timestamps[k]);
            one_frame.accel().assign(accels[k].begin(), accels[k].end());
            one_frame.gyro().assign(gyros[k].begin(), gyros[k].end());
            one_frame.depth()[0] = static_cast<std::uint16_t>(0x1234 + k);
            one_frame.depth()[1] = 0xabcd;
            for (std::size_t c = 0; c < 6; c++)
                one_frame.color()[c] = static_cast<std::uint8_t>(10 * k + c);
            // A queue of 2 can be full: wait until the frame is taken.
            while (writer.push(one_frame, trackings[k]).status == gca::push_status::full)
                std::this_thread::yield();
        }
        check(writer.close().written, "documented: the writer closed");
    }

    std::string expected = "3dscanner sequence";
    append_little_endian(expected, 2, 4);
    append_little_endian(expected, 2, 4);
    append_little_endian(expected, 1, 4);
    for (const float value : {1.5f, -2.25f, 0.5f, -0.0f, 0.001f})
        append_little_endian(expected, std::bit_cast<std::uint32_t>(value), 4);
    const char kinds[3] = {'P', 'L', 'N'};
    for (int k = 0; k < 3; k++)
    {
        expected += kinds[k];
        append_little_endian(expected, indices[k], 8);
        append_little_endian(expected, timestamps[k], 8);
        if (k == 0)
        {
            for (const double value : {0.5, -1.25, 2.0, 0.2, 0.4, 0.16, 0.88})
                append_little_endian(expected, std::bit_cast<std::uint64_t>(value), 8);
        }
        append_little_endian(expected, accels[k].size(), 2);
        append_little_endian(expected, gyros[k].size(), 2);
        for (const std::vector<gca::imu_sample> *samples : {&accels[k], &gyros[k]})
        {
            for (const gca::imu_sample &sample : *samples)
            {
                append_little_endian(expected, sample.timestamp_us, 8);
                for (const float value : {sample.x, sample.y, sample.z})
                    append_little_endian(expected, std::bit_cast<std::uint32_t>(value), 4);
            }
        }
        append_little_endian(expected, 0x1234 + k, 2);
        append_little_endian(expected, 0xabcd, 2);
        for (int c = 0; c < 6; c++)
            expected += static_cast<char>(10 * k + c);
    }
    expected += 'E';
    const std::string written = bytes_of(path);
    std::printf("documented: %zu bytes written, %zu laid out by the document\n", written.size(),
                expected.size());
    check(written == expected, "documented: the writer writes the document's bytes");
}

/// Stamps a frame's index into its first pixels, so that the reading can tell which frame's
/// pixels follow a record's lead: the index's four 16-bit parts into depth 0 to 3, its eight
/// bytes into color 0 to 7.
/// @param one_frame [in,out] the frame, at least 4 pixels.
void stamp_index(gca::frame &one_frame)
{
    for (int k = 0; k < 4; k++)
        one_frame.depth()[k] = static_cast<std::uint16_t>(one_frame.index() >> (16 * k));
    for (int k = 0; k < 8; k++)
        one_frame.color()[k] = static_cast<std::uint8_t>(one_frame.index() >> (8 * k));
}

/// Whether a frame holds the stamp of its own index.
/// @param one_frame [in] the frame.
/// @return whether its first pixels hold its index as stamp_index() writes it.
bool has_own_stamp(const gca::frame &one_frame)
{
    std::uint64_t from_depth = 0;
    std::uint64_t from_color = 0;
    for (int k = 0; k < 4; k++)
        from_depth |= static_cast<std::uint64_t>(one_frame.depth()[k]) << (16 * k);
    for (int k = 0; k < 8; k++)
        from_color |= static_cast<std::uint64_t>(one_frame.color()[k]) << (8 * k);
    return from_depth == one_frame.index() && from_color == one_frame.index();
}

/// A queue of 2 frames reports full to the caller, by has_room() before a push and by push(); a
/// push after has_room() said yes is accepted; after close() every accepted frame is on disk, in
/// order, and no frame that was not accepted. It fills because a push is a swap of
/// buffers under a lock, a few microseconds, while the writer's thread writes each frame of
/// 640x400, 1.28 MB, into the file [S: computed]: frames pushed back to back outrun it. The test
/// fails, and does not wait, when no full comes within 64 pushes. A frame not taken is dropped,
/// as the scanner drops it, and the test waits 2 ms after it, as the scanner's next frame comes
/// later, so that frames pass through the ring many times; the last 16 pushes come back to back,
/// so that close() finds the queue full.
/// @param directory [in] the test's directory.
void test_full_queue(const temporary_directory &directory)
{
    const gca::frame_description description = {
        {640, 400}, {500.0f, 500.0f, 319.5f, 199.5f}, 0.001f};
    const std::filesystem::path path = directory / "full.sequence";
    std::vector<std::uint64_t> accepted;
    int fulls = 0;
    int no_room = 0;
    int room_lost = 0;
    {
        gca::sequence_writer writer;
        check(writer.open(path, description, 2).opened, "full queue: the writer opened");
        gca::frame one_frame;
        for (std::uint64_t index = 0; index < 64; index++)
        {
            one_frame.prepare(description, index, 1000 + 33333 * index);
            stamp_index(one_frame);
            // Room seen stays until the push: the writer's thread only empties the queue.
            const bool room = writer.has_room();
            no_room += !room;
            const gca::push_result result = writer.push(one_frame, gca::not_tracked{});
            room_lost += room && result.status != gca::push_status::accepted;
            if (result.status == gca::push_status::accepted)
                accepted.push_back(index);
            else if (result.status == gca::push_status::full)
            {
                fulls++;
                check(one_frame.index() == index && has_own_stamp(one_frame),
                      "full queue: a frame not taken is left as it was");
                if (index < 48)
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            else
                check(false, ("full queue: a push failed: " + result.reason).c_str());
        }
        // Closed at once after the pushes back to back, with the queue full.
        const gca::write_result closed = writer.close();
        check(closed.written, ("full queue: the writer closed; " + closed.reason).c_str());
        check(writer.frames_written() == accepted.size(),
              "full queue: every accepted frame counted as written");
        check(!writer.has_room(), "full queue: no room once the writer is closed");
        const gca::push_result after = writer.push(one_frame, gca::not_tracked{});
        check_refusal("full queue, a push after close()", after.status == gca::push_status::failure,
                      after.reason, after.reason == "the sequence writer is not open");
    }
    std::printf("full queue: %d pushes found the queue full; %zu frames accepted; has_room() said "
                "no %d times, and %d pushes after a yes were not accepted\n",
                fulls, accepted.size(), no_room, room_lost);
    check(fulls > 0, "full queue: a full queue is reported within 64 pushes");
    check(no_room > 0, "full queue: has_room() says when the queue is full");
    check(room_lost == 0, "full queue: a push after has_room() said yes is accepted");

    gca::sequence_reader reader(path);
    check(reader.open(description.size).opened, "full queue: the reader opened");
    gca::frame read;
    std::vector<std::uint64_t> found;
    bool stamps = true;
    gca::next_result result;
    while ((result = reader.next(read)).status == gca::next_status::frame)
    {
        found.push_back(read.index());
        stamps =
            stamps && has_own_stamp(read) && read.timestamp_us() == 1000 + 33333 * read.index();
    }
    std::printf("full queue: %zu frames on disk, then %s\n", found.size(), result.message.c_str());
    check(found == accepted, "full queue: every accepted frame on disk, in order, and no other");
    check(stamps, "full queue: each record's pixels and timestamp are its own frame's");
    check(result.status == gca::next_status::end, "full queue: then the end");
}

/// Gives a small frame of a description, its pixels made from its index, with one sample of the
/// accelerometer, 10 µs before its timestamp, and two of the gyroscope, 20 µs and 5 µs before it,
/// 60 bytes of samples in its record [S: docs/SEQUENCE_FILE.md, 20 bytes a sample].
/// @param description [in] its description.
/// @param index [in] its index; its timestamp is 1000 + 33333 index µs.
/// @return the frame.
gca::frame small_frame(const gca::frame_description &description, std::uint64_t index)
{
    gca::frame one_frame;
    const std::uint64_t timestamp_us = 1000 + 33333 * index;
    one_frame.prepare(description, index, timestamp_us);
    const float value = static_cast<float>(index);
    one_frame.accel().push_back({timestamp_us - 10, value, 0.5f, -9.75f});
    one_frame.gyro().push_back({timestamp_us - 20, -value, 0.125f, 0.0f});
    one_frame.gyro().push_back({timestamp_us - 5, value, -0.125f, 1.0f});
    for (std::size_t k = 0; k < one_frame.depth().size(); k++)
        one_frame.depth()[k] = static_cast<std::uint16_t>(1000 * index + k);
    for (std::size_t k = 0; k < one_frame.color().size(); k++)
        one_frame.color()[k] = static_cast<std::uint8_t>(31 * index + k);
    return one_frame;
}

/// The small frames' description: 4x3 px [?: a choice].
const gca::frame_description small = {{4, 3}, {3.0f, 3.0f, 1.5f, 1.0f}, 0.001f};

/// A push refused by the writer: failure with the expected part of the reason, and the frame
/// left as it was.
/// @param what [in] the case.
/// @param writer [in,out] the writer.
/// @param one_frame [in,out] the frame pushed.
/// @param reason [in] the part of the reason expected.
void check_push_refused(const std::string &what, gca::sequence_writer &writer,
                        gca::frame &one_frame, const std::string &reason)
{
    const std::uint64_t index = one_frame.index();
    const std::vector<std::uint16_t> depth(one_frame.depth().begin(), one_frame.depth().end());
    const gca::push_result result = writer.push(one_frame, gca::tracking_loss{});
    check_refusal(what, result.status == gca::push_status::failure, result.reason,
                  result.reason.find(reason) != std::string::npos);
    check(one_frame.index() == index &&
              std::equal(depth.begin(), depth.end(), one_frame.depth().begin(),
                         one_frame.depth().end()),
          (what + ": the frame left as it was").c_str());
}

/// An opening the writer refuses: not opened, with the expected part of the reason.
/// @param what [in] the case.
/// @param result [in] what open() gave.
/// @param reason [in] the part of the reason expected.
void check_open_refused(const std::string &what, const gca::open_result &result,
                        const std::string &reason)
{
    check_refusal(what, !result.opened, result.refusal,
                  result.refusal.find(reason) != std::string::npos);
}

/// The writer refuses what does not fit, each as a value with the reason: a header the disk
/// refuses leaves no file; frames refused do not reach the file, and the frames accepted around
/// them do. It has no room before open() and room once open.
/// @param directory [in] the test's directory.
void test_writer_refusals(const temporary_directory &directory)
{
    gca::sequence_writer writer;
    check(!writer.has_room(), "writer: no room before open()");
    gca::frame one_frame = small_frame(small, 1);
    check_push_refused("writer, a push before open()", writer, one_frame,
                       "the sequence writer is not open");
    const gca::write_result closed = writer.close();
    check_refusal("writer, close() before open()", !closed.written, closed.reason,
                  closed.reason == "the sequence writer is not open");

    const std::filesystem::path path = directory / "refusals.sequence";
    check_open_refused("writer, a queue of 0 frames", writer.open(path, small, 0),
                       "a queue of 0 frames takes no frame");
    check(!std::filesystem::exists(path), "writer, a queue of 0 frames: no file created");
    write_bytes(directory / "taken.sequence", "a file there");
    check_open_refused("writer, a file there already",
                       writer.open(directory / "taken.sequence", small, 2), "File exists");
    check(bytes_of(directory / "taken.sequence") == "a file there",
          "writer, a file there already: left as it was");
    check_open_refused("writer, no directory",
                       writer.open(directory / "nowhere" / "a.sequence", small, 2),
                       "cannot create: No such file or directory");
    const std::filesystem::path locked = directory / "locked";
    std::filesystem::create_directory(locked);
    std::filesystem::permissions(locked, std::filesystem::perms::owner_read |
                                             std::filesystem::perms::owner_exec);
    check_open_refused("writer, a directory it cannot write",
                       writer.open(locked / "a.sequence", small, 2),
                       "cannot create: Permission denied");
    std::filesystem::permissions(locked, std::filesystem::perms::owner_all);
    {
        // A disk that refuses the header: a file size limit of 20 bytes, less than its 50.
        const gca_test::file_size_limit limit(20);
        check_open_refused("writer, a header the disk refuses", writer.open(path, small, 2),
                           "the header's write failed: File too large");
    }
    check(!std::filesystem::exists(path), "writer, a header the disk refuses: no file left");

    check(writer.open(path, small, 2).opened, "writer: opened where the header was refused");
    check(writer.has_room(), "writer: room once open");
    check_open_refused("writer, a second open()",
                       writer.open(directory / "second.sequence", small, 2),
                       "the writer is open already, writing " + path.string());
    check(!std::filesystem::exists(directory / "second.sequence"),
          "writer, a second open(): no file created");

    std::vector<pushed_frame> pushed;
    gca::frame first = small_frame(small, 5);
    pushed.push_back(copy_of(first, gca::not_tracked{}));
    check(writer.push(first, gca::not_tracked{}).status == gca::push_status::accepted,
          "writer: frame 5 accepted");
    gca::frame_description other = small;
    other.intrinsics.cx = 1.25f;
    gca::frame wrong = small_frame(other, 6);
    check_push_refused("writer, other intrinsics", writer, wrong,
                       "frame index 6: its size, intrinsics or depth unit are not the sequence's");
    gca::frame_description other_unit = small;
    other_unit.depth_unit_m = 0.0001f;
    gca::frame finer = small_frame(other_unit, 6);
    check_push_refused("writer, another depth unit", writer, finer,
                       "frame index 6: its size, intrinsics or depth unit are not the sequence's");
    gca::frame again = small_frame(small, 5);
    check_push_refused("writer, an index that does not increase", writer, again,
                       ": frame index 5 does not increase; the frame before has index 5");
    gca::frame earlier = small_frame(small, 6);
    earlier.prepare(small, 6, 1000 + 33333 * 5);
    check_push_refused("writer, a timestamp that does not increase", writer, earlier,
                       ": timestamp 167665 µs does not increase; the frame before was taken at "
                       "167665 µs");
    // Frame 5's accelerometer sample was taken at 167655 µs; frame 7's timestamp is 234331 µs.
    gca::frame repeated = small_frame(small, 7);
    repeated.accel()[0].timestamp_us = 167655;
    check_push_refused(
        "writer, an accelerometer sample that does not follow the last", writer, repeated,
        ": frame index 7: accelerometer sample 0 at 167655 µs does not increase; the "
        "sample before was taken at 167655 µs");
    gca::frame swapped = small_frame(small, 7);
    std::swap(swapped.gyro()[0], swapped.gyro()[1]);
    check_push_refused("writer, a frame's gyroscope samples out of order", writer, swapped,
                       ": frame index 7: gyroscope sample 1 at 234311 µs does not increase; the "
                       "sample before was taken at 234326 µs");
    gca::frame ahead = small_frame(small, 7);
    ahead.gyro()[1].timestamp_us = 234332;
    check_push_refused("writer, a gyroscope sample after its frame's timestamp", writer, ahead,
                       ": frame index 7: gyroscope sample 1 at 234332 µs lies after its frame's "
                       "timestamp, 234331 µs");
    gca::frame crowded = small_frame(small, 7);
    crowded.accel().assign(gca::imu_samples_per_record_max + 1, crowded.accel()[0]);
    check_push_refused("writer, more accelerometer samples than a record holds", writer, crowded,
                       ": frame index 7 holds 65536 accelerometer samples; a record holds at "
                       "most 65535");
    gca::frame later = small_frame(small, 9);
    pushed.push_back(copy_of(later, gca::tracking_loss{}));
    while (writer.push(later, gca::tracking_loss{}).status == gca::push_status::full)
        std::this_thread::yield();
    check(writer.close().written, "writer: closed");
    check_read_back("writer, after the refusals", path, small.size, pushed, small);
}

/// A disk that refuses a write, stood for by a file size limit that cuts the second frame:
/// push() gives the reason once the writer's thread has met it, close() gives it, and the
/// reader gives the frame written before, then the early end. A limit that refuses only the
/// end's byte: close() says so, and the reader gives both frames, then the early end.
/// @param directory [in] the test's directory.
void test_disk_refuses(const temporary_directory &directory)
{
    const std::filesystem::path path = directory / "cut.sequence";
    // The header, 50 bytes, and a frame in tracking loss, 21 + 3 * 20 + 5 * 12 bytes with its
    // three samples; then half of the next [S: docs/SEQUENCE_FILE.md].
    const std::size_t record_bytes = 21 + 3 * 20 + 5 * 12;
    const std::size_t limit_bytes = 50 + record_bytes + record_bytes / 2;
    std::vector<pushed_frame> pushed;
    gca::push_result refused;
    gca::write_result closed;
    std::uint64_t written = 0;
    {
        const gca_test::file_size_limit limit(limit_bytes);
        gca::sequence_writer writer;
        check(writer.frames_written() == 0, "a disk that refuses: none written before open()");
        check(writer.open(path, small, 4).opened, "a disk that refuses: the writer opened");
        std::uint64_t index = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline)
        {
            gca::frame one_frame = small_frame(small, index);
            const pushed_frame copy = copy_of(one_frame, gca::tracking_loss{});
            refused = writer.push(one_frame, gca::tracking_loss{});
            if (refused.status == gca::push_status::failure)
                break;
            if (refused.status == gca::push_status::accepted)
            {
                pushed.push_back(copy);
                index++;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        closed = writer.close();
        written = writer.frames_written();
    }
    std::printf("a disk that refuses: %zu frames accepted, %llu written\n", pushed.size(),
                static_cast<unsigned long long>(written));
    check(written == 1 && pushed.size() > written,
          "a disk that refuses: the frame before the failure counted as written, the accepted "
          "frames after it not");
    check_refusal("a disk that refuses, push()", refused.status == gca::push_status::failure,
                  refused.reason,
                  refused.reason.find(": the write of frame index 1 failed: File too large") !=
                      std::string::npos);
    check_refusal("a disk that refuses, close()", !closed.written, closed.reason,
                  closed.reason == refused.reason);
    check(std::filesystem::file_size(path) == limit_bytes,
          "a disk that refuses: the file holds what the limit let through");

    gca::sequence_reader reader(path);
    check(reader.open(small.size).opened, "a disk that refuses: the reader opened");
    gca::frame read;
    const bool first = reader.next(read).status == gca::next_status::frame && !pushed.empty() &&
                       same_frame(read, reader.tracking(), pushed[0]);
    check(first, "a disk that refuses: the frame before the failure read back");
    const gca::next_result early = reader.next(read);
    check_refusal("a disk that refuses, the reader", early.status == gca::next_status::failure,
                  early.message,
                  early.message ==
                      path.string() + ": ends early: within record 2, before the sequence's end");
    check(reader.ended_early(), "a disk that refuses: the reader says the file ended early");

    // A limit that lets two whole frames through and refuses the end's one byte.
    const std::filesystem::path endless = directory / "endless.sequence";
    std::vector<pushed_frame> two;
    gca::write_result end_refused;
    {
        const gca_test::file_size_limit limit(50 + 2 * record_bytes);
        gca::sequence_writer writer;
        check(writer.open(endless, small, 4).opened, "an end the disk refuses: the writer opened");
        for (std::uint64_t index = 0; index < 2; index++)
        {
            gca::frame one_frame = small_frame(small, index);
            two.push_back(copy_of(one_frame, gca::tracking_loss{}));
            check(writer.push(one_frame, gca::tracking_loss{}).status == gca::push_status::accepted,
                  "an end the disk refuses: a frame accepted");
        }
        end_refused = writer.close();
        check(writer.frames_written() == 2,
              "an end the disk refuses: both frames counted as written");
    }
    check_refusal("an end the disk refuses, close()", !end_refused.written, end_refused.reason,
                  end_refused.reason ==
                      endless.string() + ": the end's write failed: File too large");
    gca::sequence_reader endless_reader(endless);
    bool both = endless_reader.open(small.size).opened;
    for (std::size_t k = 0; both && k < 2; k++)
        both = endless_reader.next(read).status == gca::next_status::frame &&
               same_frame(read, endless_reader.tracking(), two[k]);
    check(both, "an end the disk refuses: both frames read back");
    const gca::next_result no_end = endless_reader.next(read);
    check_refusal("an end the disk refuses, the reader",
                  no_end.status == gca::next_status::failure && endless_reader.ended_early(),
                  no_end.message,
                  no_end.message == endless.string() +
                                        ": ends early: after record 2, without the sequence's end");
}

/// A file the reader has to refuse at open(): not opened, with the expected reason.
/// @param what [in] the case.
/// @param path [in] the file.
/// @param size [in] the size asked for.
/// @param reason [in] the reason expected, after the path.
void check_reader_refuses(const std::string &what, const std::filesystem::path &path,
                          const gca::frame_size &size, const std::string &reason)
{
    gca::sequence_reader reader(path);
    const gca::open_result result = reader.open(size);
    check_refusal(what, !result.opened, result.refusal, result.refusal == path.string() + reason);
}

/// The bytes of a small sequence: a frame with a pose (index 1), one in tracking loss (index 2),
/// one not tracked (index 4), and the end.
/// @param directory [in] the test's directory, where the writer writes it.
/// @param pushed [out] the frames, as pushed.
/// @return the file's bytes.
std::string small_sequence(const temporary_directory &directory, std::vector<pushed_frame> &pushed)
{
    const std::filesystem::path path = directory / "small.sequence";
    const gca::frame_tracking trackings[3] = {
        gca::camera_pose{Eigen::Vector3d(0.1, 0.2, 0.3), Eigen::Quaterniond::Identity()},
        gca::tracking_loss{}, gca::not_tracked{}};
    const std::uint64_t indices[3] = {1, 2, 4};
    gca::sequence_writer writer;
    check(writer.open(path, small, 4).opened, "test setup: the small sequence's writer opened");
    for (int k = 0; k < 3; k++)
    {
        gca::frame one_frame = small_frame(small, indices[k]);
        pushed.push_back(copy_of(one_frame, trackings[k]));
        check(writer.push(one_frame, trackings[k]).status == gca::push_status::accepted,
              "test setup: the small sequence's frame accepted");
    }
    check(writer.close().written, "test setup: the small sequence written");
    return bytes_of(path);
}

/// A sequence cut at every byte after its header: the reader gives every whole frame before
/// the cut, then the early end, naming the record, at this and the next call; the frame it was
/// given is left as it was.
/// @param directory [in] the test's directory.
/// @param bytes [in] the whole sequence's bytes.
/// @param pushed [in] its frames, as pushed.
void check_every_cut(const temporary_directory &directory, const std::string &bytes,
                     const std::vector<pushed_frame> &pushed)
{
    // Where each record ends [S: docs/SEQUENCE_FILE.md]: the header's 50 bytes, a frame with a
    // pose 77 + 3 * 20 + 5 * 12, the others 21 + 3 * 20 + 5 * 12, the end 1.
    const std::size_t ends[3] = {50 + 197, 50 + 197 + 141, 50 + 197 + 141 + 141};
    const std::filesystem::path path = directory / "cut_short.sequence";
    int wrong = 0;
    for (std::size_t cut = 50; cut < bytes.size(); cut++)
    {
        write_bytes(path, bytes.substr(0, cut));
        const std::size_t whole = static_cast<std::size_t>(
            std::count_if(ends, ends + 3, [&](std::size_t end) { return end <= cut; }));
        const bool at_boundary = cut == 50 || std::find(ends, ends + 3, cut) != ends + 3;
        const std::string expected =
            path.string() + ": ends early: " +
            (at_boundary
                 ? "after " +
                       (whole == 0 ? std::string("its header")
                                   : "record " + std::to_string(whole)) +
                       ", without the sequence's end"
                 : "within record " + std::to_string(whole + 1) + ", before the sequence's end");
        gca::sequence_reader reader(path);
        bool right = reader.open(small.size).opened;
        gca::frame read;
        for (std::size_t k = 0; right && k < whole; k++)
            right = reader.next(read).status == gca::next_status::frame &&
                    same_frame(read, reader.tracking(), pushed[k]);
        const gca::next_result early = reader.next(read);
        const gca::next_result again = reader.next(read);
        right = right && early.status == gca::next_status::failure && early.message == expected &&
                reader.ended_early() && again.message == expected &&
                (whole == 0 || same_frame(read, reader.tracking(), pushed[whole - 1]));
        if (!right)
        {
            if (wrong < 5)
                std::printf("cut at byte %zu: \"%s\", expected \"%s\"\n", cut,
                            early.message.c_str(), expected.c_str());
            wrong++;
        }
    }
    std::printf("cut at each of bytes 50 to %zu: %d cases wrong\n", bytes.size() - 1, wrong);
    check(wrong == 0, "every cut: the whole frames before it, then the early end with the "
                      "record, the frame left as it was");
}

/// A sequence that reads up to one record and then fails there: the frames before it, then the
/// failure with the expected reason.
/// @param what [in] the case.
/// @param directory [in] the test's directory.
/// @param bytes [in] the file's bytes.
/// @param pushed [in] the frames before the failure, as pushed.
/// @param frames_before [in] how many frames come before the failure.
/// @param reason [in] the reason expected, after the path and ": ".
void check_fails_after(const std::string &what, const temporary_directory &directory,
                       const std::string &bytes, const std::vector<pushed_frame> &pushed,
                       std::size_t frames_before, const std::string &reason)
{
    const std::filesystem::path path = directory / "changed.sequence";
    write_bytes(path, bytes);
    gca::sequence_reader reader(path);
    bool before = reader.open(small.size).opened;
    gca::frame read;
    for (std::size_t k = 0; before && k < frames_before; k++)
        before = reader.next(read).status == gca::next_status::frame &&
                 same_frame(read, reader.tracking(), pushed[k]);
    check(before, (what + ": the frames before read back").c_str());
    const gca::next_result result = reader.next(read);
    check_refusal(what, result.status == gca::next_status::failure && !reader.ended_early(),
                  result.message, result.message == path.string() + ": " + reason);
}

/// The reader refuses what is not a sequence of this version and size, and fails, after the
/// frames before, where a sequence is cut short or holds what its writer never writes.
/// @param directory [in] the test's directory.
void test_reader_refusals(const temporary_directory &directory)
{
    std::vector<pushed_frame> pushed;
    const std::string bytes = small_sequence(directory, pushed);
    check_read_back("small sequence", directory / "small.sequence", small.size, pushed, small);

    const std::filesystem::path path = directory / "refused.sequence";
    check_reader_refuses("reader, a missing file", directory / "missing.sequence", small.size,
                         ": cannot open: No such file or directory");
    write_bytes(path, "");
    check_reader_refuses("reader, an empty file", path, small.size,
                         " ends early: within its header, after 0 of its 50 bytes");
    write_bytes(path, "# 3dscanner trajectory 1\n7 233333 lost\n");
    check_reader_refuses("reader, a trajectory", path, small.size,
                         " is no sequence: it does not start with \"3dscanner sequence\"");
    write_bytes(path, "3dscanner");
    check_reader_refuses("reader, the magic's start only", path, small.size,
                         " ends early: within its header, after 9 of its 50 bytes");
    // Version 1, the format before the IMU's samples, is not read (AGENTS.md §7 rule 6).
    std::string version_1 = bytes;
    version_1[18] = 1;
    write_bytes(path, version_1);
    check_reader_refuses("reader, version 1", path, small.size,
                         " is of the sequence format's version 1; this reader knows version 2 "
                         "only");
    write_bytes(path, version_1.substr(0, 22));
    check_reader_refuses("reader, version 1 with a header of 22 bytes", path, small.size,
                         " is of the sequence format's version 1; this reader knows version 2 "
                         "only");
    std::string version_3 = bytes;
    version_3[18] = 3;
    write_bytes(path, version_3);
    check_reader_refuses("reader, version 3", path, small.size,
                         " is of the sequence format's version 3; this reader knows version 2 "
                         "only");
    write_bytes(path, bytes.substr(0, 30));
    check_reader_refuses("reader, a header cut short", path, small.size,
                         " ends early: within its header, after 30 of its 50 bytes");
    write_bytes(path, bytes);
    check_reader_refuses("reader, another size asked", path, {5, 3},
                         ": its frames are 4x3 px, not the 5x3 px asked");
    std::filesystem::create_directory(directory / "a_directory.sequence");
    check_reader_refuses("reader, a directory", directory / "a_directory.sequence", small.size,
                         ": cannot read: Is a directory");

    gca::sequence_reader reader(path);
    gca::frame read;
    const gca::next_result closed = reader.next(read);
    check_refusal("reader, next() before open()", closed.status == gca::next_status::failure,
                  closed.message,
                  closed.message == path.string() + ": the sequence reader is not open");
    check(reader.open(small.size).opened, "reader: opened");
    const gca::open_result twice = reader.open(small.size);
    check_refusal("reader, a second open()", !twice.opened, twice.refusal,
                  twice.refusal == path.string() + ": the reader is open already");
    reader.close();

    check_every_cut(directory, bytes, pushed);

    // Record 2 starts after the header and the frame with a pose; its accelerometer's sample
    // after its lead of 21 bytes, its gyroscope's two after that. Record 1's accelerometer sample
    // starts after its lead of 77 bytes.
    const std::size_t second = 50 + 197;
    const std::size_t second_accel = second + 21;
    const std::size_t second_gyro = second_accel + 20;
    const std::size_t first_accel = 50 + 77;
    std::string unknown = bytes;
    unknown[second] = 'X';
    check_fails_after("reader, an unknown kind", directory, unknown, pushed, 1,
                      "record 2 has the unknown kind 0x58");
    std::string same_index = bytes;
    same_index[second + 1] = 1;
    check_fails_after("reader, an index that does not increase", directory, same_index, pushed, 1,
                      "record 2: frame index 1 does not increase; the frame before has index 1");
    std::string same_time = bytes;
    same_time.replace(second + 9, 8, bytes.substr(50 + 9, 8));
    check_fails_after("reader, a timestamp that does not increase", directory, same_time, pushed, 1,
                      "record 2: timestamp 34333 µs does not increase; the frame before was "
                      "taken at 34333 µs");
    std::string repeated_sample = bytes;
    repeated_sample.replace(second_accel, 8, bytes.substr(first_accel, 8));
    check_fails_after("reader, an accelerometer sample that does not follow the last", directory,
                      repeated_sample, pushed, 1,
                      "record 2: accelerometer sample 0 at 34323 µs does not increase; the sample "
                      "before was taken at 34323 µs");
    // Record 2's frame was taken at 67666 µs; its second gyroscope sample one µs after it.
    std::string ahead = bytes;
    std::string later_time;
    append_little_endian(later_time, 67667, 8);
    ahead.replace(second_gyro + 20, 8, later_time);
    check_fails_after("reader, a gyroscope sample after its frame's timestamp", directory, ahead,
                      pushed, 1,
                      "record 2: gyroscope sample 1 at 67667 µs lies after its frame's timestamp, "
                      "67666 µs");
    check_fails_after("reader, a byte after the end", directory, bytes + "E", pushed, 3,
                      "bytes follow the sequence's end, record 4; the end is the file's last "
                      "byte");
}
} // namespace

/// Runs the sequence's checks and prints the measured values, the refusals and the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    const temporary_directory directory("test_sequence");
    test_round_trip(directory);
    test_bytes_as_documented(directory);
    test_full_queue(directory);
    test_writer_refusals(directory);
    test_disk_refuses(directory);
    test_reader_refusals(directory);
    return gca_test::check_summary("test_sequence");
}
