// SPDX-License-Identifier: 0BSD
/// @file sequence_format.hpp
/// The layout of a sequence on disk (GLOSSARY.md, "Sequence"; docs/SEQUENCE_FILE.md), in one
/// place for its writer (camera/sequence_writer.hpp) and its reader (camera/sequence_reader.hpp):
/// what tracking said about each frame, the header with the frames' description and the format's
/// version, the comparison of a frame's description with the header's, each record's kind and
/// lead, the bytes before its pictures, and the order in which frames follow one another. Every
/// number is little-endian. The pictures follow a frame's lead as they lie in a frame's buffers
/// (camera/frame.hpp): depth, then color.

#pragma once

#include "types/camera_pose.hpp"
#include "types/frame_description.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <variant>

namespace gca
{
// A frame's depth values are written and read as they lie in memory, which is the file's byte
// order only on a little-endian host, as the board's aarch64 is [S: aarch64 Linux's ABI].
static_assert(std::endian::native == std::endian::little,
              "the sequence's depth is written as it lies in memory, little-endian");

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
///         version is not 1, the file ends within the header.
/// @note Allocates for a reason. A failed allocation ends the program (std::terminate): the
///       function is noexcept. Two threads may call it at once.
sequence_header_result decode_sequence_header(std::span<const std::uint8_t> bytes) noexcept;

/// The kind byte of the record that ends a sequence; it is the file's last byte.
inline constexpr std::uint8_t sequence_end_kind = 'E';

/// The most bytes a frame's lead holds: its kind, its index, its timestamp and a pose.
inline constexpr std::size_t frame_lead_max_bytes = 73;

/// What a frame's lead says: the record's bytes before its pictures.
struct frame_lead
{
    std::uint64_t index;        ///< the frame's index
    std::uint64_t timestamp_us; ///< the frame's timestamp, in µs on its source's clock
    frame_tracking tracking;    ///< the frame's pose, bit for bit, or its mark
};

/// A frame's lead as bytes.
struct frame_lead_bytes
{
    std::array<std::uint8_t, frame_lead_max_bytes> bytes; ///< the lead, in its first size bytes
    std::size_t size;                                     ///< 73 with a pose, 17 with a mark
};

/// The bytes of a frame's lead.
/// @param lead [in] the frame's index, timestamp and pose or mark; a pose's bits are kept as
///        given.
/// @return the kind byte ('P', 'L' or 'N'), the index, the timestamp and, with 'P', the pose
///         (docs/SEQUENCE_FILE.md, "Layout").
/// @note Allocates nothing. Two threads may call it at once.
frame_lead_bytes encode_frame_lead(const frame_lead &lead) noexcept;

/// How many bytes of a frame's lead follow its kind byte.
/// @param kind [in] a record's first byte.
/// @return 72 for 'P', 16 for 'L' and 'N'; nothing for the end's kind and for a byte that is
///         no record's kind.
/// @note Allocates nothing. Two threads may call it at once.
std::optional<std::size_t> frame_lead_bytes_after(std::uint8_t kind) noexcept;

/// Reads a frame's lead.
/// @param kind [in] its kind byte.
/// @param rest [in] the bytes that follow the kind; the first frame_lead_bytes_after(kind) of
///        them are read, the others not.
/// @return the index and the timestamp, and the pose, bit for bit, with 'P', the mark of tracking
///         loss with 'L', the mark that nothing tracked the frame with 'N'; nothing for any other
///         byte, which is no frame's kind.
/// @note Allocates nothing. Two threads may call it at once.
std::optional<frame_lead> decode_frame_lead(
    std::uint8_t kind, std::span<const std::uint8_t, frame_lead_max_bytes - 1> rest) noexcept;

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
