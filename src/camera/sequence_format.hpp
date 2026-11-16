// SPDX-License-Identifier: 0BSD
/// @file sequence_format.hpp
/// The layout of a sequence on disk (GLOSSARY.md, "Sequence"; docs/SEQUENCE_FILE.md), in one
/// place for its writer (camera/sequence_writer.hpp) and its reader (camera/sequence_reader.hpp):
/// what tracking said about each frame, the header with the frames' description and the format's
/// version, the comparison of a frame's description with the header's, each record's kind and
/// lead, the bytes before its samples, an IMU sample's bytes, the order in which frames follow one
/// another and the order of each sensor's samples. Every number is little-endian. After a frame's
/// lead come its samples, the accelerometer's, then the gyroscope's, and then its pictures as they
/// lie in a frame's buffers (camera/frame.hpp): depth, then color.

#pragma once

#include "types/camera_pose.hpp"
#include "types/frame_description.hpp"
#include "types/imu_sample.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>

namespace gca
{
// A frame's depth values are written and read as they lie in memory, which is the file's byte
// order only on a little-endian host, as the board's aarch64 is [S: aarch64 Linux's ABI].
static_assert(std::endian::native == std::endian::little,
              "the sequence's depth is written as it lies in memory, little-endian");

/// The name of a sequence's file in its directory: a scan's (docs/SCAN_FILES.md, "A scan's
/// directory"), the one orbbec_viewer's --sequence names, the one depth_noise and the replay tool
/// read [?: a choice; the ending is the one ctest `sequence` gives its files].
inline constexpr const char *sequence_file_name = "frames.sequence";

/// The mark of tracking loss: tracking gave the frame no pose (GLOSSARY.md, "Tracking loss").
struct tracking_loss
{
};

/// The mark that nothing tracked the frame, as in a sequence the viewer writes.
struct not_tracked
{
};

/// What tracking said about a frame: its pose (types/camera_pose.hpp), the mark of tracking
/// loss, or the mark that nothing tracked it.
using frame_tracking = std::variant<camera_pose, tracking_loss, not_tracked>;

/// The bytes of the header: the magic, the format's version and the frames' description.
inline constexpr std::size_t sequence_header_bytes = 50;

/// The header of a sequence that holds frames of a description.
/// @param description [in] the size, intrinsics and depth unit of every frame.
/// @return the header's bytes, as docs/SEQUENCE_FILE.md, "Header", lays them out.
/// @note Allocates nothing. Two threads may call it at once.
std::array<std::uint8_t, sequence_header_bytes> encode_sequence_header(
    const frame_description &description) noexcept;

/// What decode_sequence_header() gave.
struct sequence_header_result
{
    std::optional<frame_description> description; ///< the frames' description; nothing when
                                                  ///< the header was refused
    std::string reason; ///< why it was refused, in plain English; empty when it was read
};

/// Reads a header. It checks the magic first and the version next, so that a file of another
/// version is refused as such whatever its header's length.
/// @param bytes [in] the file's first bytes, sequence_header_bytes of them, or all of the file
///        when it is shorter.
/// @return the description; or the reason: the bytes are not the magic "3dscanner sequence", the
///         version is not 2, the file ends within the header. Version 1, the format before the
///         IMU's samples, is refused as any other.
/// @note Allocates for a reason. A failed allocation ends the program (std::terminate): the
///       function is noexcept. Two threads may call it at once.
sequence_header_result decode_sequence_header(std::span<const std::uint8_t> bytes) noexcept;

/// The kind byte of the record that ends a sequence; it is the file's last byte.
inline constexpr std::uint8_t sequence_end_kind = 'E';

/// The most bytes a frame's lead holds: its kind, its index, its timestamp, a pose and the counts
/// of its samples.
inline constexpr std::size_t frame_lead_max_bytes = 77;

/// The most samples of one sensor a frame's record holds: its count is 16 bits [S]. 65535 samples
/// are 327 s at the camera's 200 Hz [S: computed], far more than a frame carries
/// (docs/methods/imu-samples.md); the writer refuses a frame with more.
inline constexpr std::size_t imu_samples_per_record_max = 65535;

/// What a frame's lead says: the record's bytes before its samples.
struct frame_lead
{
    std::uint64_t index;        ///< the frame's index
    std::uint64_t timestamp_us; ///< the frame's timestamp, in µs on its source's clock
    frame_tracking tracking;    ///< the frame's pose, bit for bit, or its mark
    std::uint16_t accel_count;  ///< the accelerometer's samples that follow the lead
    std::uint16_t gyro_count;   ///< the gyroscope's samples that follow the accelerometer's
};

/// A frame's lead as bytes.
struct frame_lead_bytes
{
    std::array<std::uint8_t, frame_lead_max_bytes> bytes; ///< the lead, in its first size bytes
    std::size_t size;                                     ///< 77 with a pose, 21 with a mark
};

/// The bytes of a frame's lead.
/// @param lead [in] the frame's index, timestamp, pose or mark and its counts of samples; a pose's
///        bits are kept as given.
/// @return the kind byte ('P', 'L' or 'N'), the index, the timestamp, with 'P' the pose, and the
///         two counts (docs/SEQUENCE_FILE.md, "Layout").
/// @note Allocates nothing. Two threads may call it at once.
frame_lead_bytes encode_frame_lead(const frame_lead &lead) noexcept;

/// How many bytes of a frame's lead follow its kind byte.
/// @param kind [in] a record's first byte.
/// @return 76 for 'P', 20 for 'L' and 'N'; nothing for the end's kind and for a byte that is
///         no record's kind.
/// @note Allocates nothing. Two threads may call it at once.
std::optional<std::size_t> frame_lead_bytes_after(std::uint8_t kind) noexcept;

/// Reads a frame's lead.
/// @param kind [in] its kind byte.
/// @param rest [in] the bytes that follow the kind; the first frame_lead_bytes_after(kind) of
///        them are read, the others not.
/// @return the index and the timestamp, the pose, bit for bit, with 'P', the mark of tracking
///         loss with 'L', the mark that nothing tracked the frame with 'N', and the counts of its
///         samples; nothing for any other byte, which is no frame's kind.
/// @note Allocates nothing. Two threads may call it at once.
std::optional<frame_lead> decode_frame_lead(
    std::uint8_t kind, std::span<const std::uint8_t, frame_lead_max_bytes - 1> rest) noexcept;

/// The bytes of one IMU sample in a record: its timestamp and its three values.
inline constexpr std::size_t imu_sample_bytes = 20;

/// The bytes of an IMU sample.
/// @param sample [in] the sample; its values' bits are kept, NaN's payload and -0 among them.
/// @return the timestamp, unsigned 64-bit, then x, y, z, three floats (docs/SEQUENCE_FILE.md,
///         "Layout").
/// @note Allocates nothing. Two threads may call it at once.
std::array<std::uint8_t, imu_sample_bytes> encode_imu_sample(const imu_sample &sample) noexcept;

/// Reads an IMU sample written by encode_imu_sample().
/// @param bytes [in] its imu_sample_bytes bytes.
/// @return the sample, bit for bit.
/// @note Allocates nothing. Two threads may call it at once.
imu_sample decode_imu_sample(std::span<const std::uint8_t, imu_sample_bytes> bytes) noexcept;

/// The samples the writer encodes and the reader decodes at once, in a buffer on its stack
/// [?: a choice]: 64 samples, 1280 bytes, a frame's samples at the camera's rate in one write or
/// read [S: docs/methods/imu-samples.md, 6 or 7 a frame].
inline constexpr std::size_t imu_samples_per_block = 64;

/// The order of a sequence's IMU samples, the rule of the frame source seam
/// (camera/frame_source.hpp): each sample of a sensor follows the sensor's sample before it, in
/// its frame or in an earlier frame of the sequence, and none lies after its frame's timestamp.
/// It holds each sensor's last sample so far; the writer and the reader both check each frame
/// with it and advance it with each frame they take.
class imu_sample_order
{
public:
    /// Checks a frame's samples against the sensors' last samples so far, the accelerometer's
    /// first; changes nothing.
    /// @param accel [in] the frame's accelerometer samples, in the record's order.
    /// @param gyro [in] its gyroscope samples, in the record's order.
    /// @param frame_timestamp_us [in] the frame's timestamp, in µs.
    /// @return empty when the samples keep the rule; otherwise the reason for the first that does
    ///         not, as "gyroscope sample 1 at 5 µs does not increase; the sample before was taken
    ///         at 9 µs" or "accelerometer sample 0 at 7 µs lies after its frame's timestamp, 6
    ///         µs", counting the frame's samples of the sensor from 0.
    /// @note Allocates only for a reason. A failed allocation ends the program (std::terminate):
    ///       the function is noexcept.
    std::string refusal(std::span<const imu_sample> accel, std::span<const imu_sample> gyro,
                        std::uint64_t frame_timestamp_us) const noexcept;

    /// Takes a frame's samples as the sensors' last so far: each sensor's last sample of the
    /// frame, when it has one.
    /// @param accel [in] the frame's accelerometer samples, which refusal() accepted.
    /// @param gyro [in] its gyroscope samples, the same.
    /// @note Allocates nothing.
    void advance(std::span<const imu_sample> accel, std::span<const imu_sample> gyro) noexcept;

private:
    /// The timestamps of the last accelerometer's and gyroscope's samples so far, in µs; nothing
    /// while the frames so far held none.
    std::optional<std::uint64_t> m_last_accel_us;
    std::optional<std::uint64_t> m_last_gyro_us;
};

/// Checks that a frame may follow the frame before it in a sequence: the frames of a frame
/// source's opening have strictly increasing indices and timestamps, and a gap in the indices
/// stands for frames not taken (camera/frame_source.hpp). The writer and the reader both check
/// it.
/// @param index [in] the frame's index.
/// @param timestamp_us [in] the frame's timestamp, in µs.
/// @param index_before [in] the index of the frame before it.
/// @param timestamp_before_us [in] the timestamp of the frame before it, in µs.
/// @return empty when both increase; otherwise the reason, as "frame index 5 does not increase;
///         the frame before has index 5".
/// @note Allocates only for a reason. A failed allocation ends the program (std::terminate): the
///       function is noexcept. Two threads may call it at once.
std::string order_refusal(std::uint64_t index, std::uint64_t timestamp_us,
                          std::uint64_t index_before, std::uint64_t timestamp_before_us) noexcept;

/// Compares two frames' descriptions bit for bit, as a sequence's header keeps them: the size, each
/// intrinsic and the depth unit, so that -0 and 0 differ and a NaN equals itself. The writer takes
/// only frames of its header's description; the offline tests compare the frames read back with
/// those they rendered.
/// @param a [in] one description.
/// @param b [in] the other.
/// @return whether every field has the same bits.
/// @note Allocates nothing. Two threads may call it at once.
bool same_description(const frame_description &a, const frame_description &b) noexcept;
} // namespace gca
