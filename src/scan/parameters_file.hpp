// SPDX-License-Identifier: 0BSD
/// @file parameters_file.hpp
/// A scan's parameters on disk: JSON with a format version, the tree's commit, the object class,
/// the set of parameters in force, and the scan's facts: the stream's size, intrinsics and depth
/// unit, what delivered the frames, the camera's run-time properties it refused, the times of its
/// start and stop, the frames taken, fused, lost and not taken, and why it ended
/// (docs/SCAN_FILES.md, "Parameters"). Why a scan ended is one key of a closed set, and a property
/// refused one of the five names of camera/camera_properties.hpp's table, never free text: OpenCV
/// writes only some strings as JSON. The writer
/// and the reader use OpenCV's FileStorage on memory and do the file's I/O themselves
/// (scan/file_io.hpp). The reader refuses, with the reason, a file of another version, a key it
/// does not know, a key twice or missing, a value of another type, a string that OpenCV does not
/// write as JSON, and a set or a fact the scanner cannot have run with; the writer refuses those
/// strings, sets and facts too, so it writes nothing its reader refuses.

#pragma once

#include "camera/camera_properties.hpp"
#include "camera/frame_source.hpp"
#include "scan/file_io.hpp"
#include "scan/object_class.hpp"
#include "scan/utc_time.hpp"
#include "types/frame_description.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace gca
{
/// Why a scan ended (docs/SCAN_FILES.md, "Parameters"): one key of a closed set, which the file
/// holds instead of the message; the message with its details is the scan's state's. The
/// enumerators run from 0 without a gap, and each has its row in parameters_file.cpp's table, a
/// switch that -Wswitch checks.
enum class scan_end
{
    stopped,         ///< the operator's stop
    source_timeout,  ///< the frame source gave no frame within its time limit
    source_failure,  ///< the frame source failed (camera/frame_source.hpp)
    source_end,      ///< the frame source had no frame left, as a sequence at its end
    tracking_failed, ///< tracking failed on a frame (tracking/tracker.hpp, tracking_state::failed)
    writer_failed,   ///< the sequence's writer failed (camera/sequence_writer.hpp)
    shut_down,       ///< the service was asked to stop during the scan
};

/// Gives the key of why a scan ended, as the file writes it.
/// @param end [in] why it ended, one of the enumerators.
/// @return the enumerator's name, a static string (docs/SCAN_FILES.md lists them); empty for a
///         value that is no enumerator.
std::string_view name_of(scan_end end) noexcept;

/// Finds why a scan ended from its key.
/// @param name [in] the key, as name_of() gives it.
/// @return why it ended; nothing when the name is no key of the set.
std::optional<scan_end> scan_end_named(std::string_view name) noexcept;

/// The frames of a scan, counted as the scanner took them or not.
struct scan_frame_counts
{
    /// The frames that entered the per-frame path: each fused or lost, but for the frame tracking
    /// failed on when the scan ended because of it (scan_end::tracking_failed), so fused + lost,
    /// and 1 more then; at most 2^53, so that JSON holds it.
    std::uint64_t taken;
    std::uint64_t fused; ///< the frames tracking passed and fused into the live model, <= 2^53
    std::uint64_t lost;  ///< the frames in tracking loss, nothing fused, <= 2^53
    /// The frames the scanner did not take: those the frame source delivered when the sequence's
    /// writer had no room, and the gaps in the frame indices, frames the source did not deliver;
    /// at most 2^53.
    std::uint64_t not_taken;

    /// Compares two counts value by value.
    /// @param other [in] the other counts.
    /// @return whether every count is equal.
    bool operator==(const scan_frame_counts &other) const = default;
};

/// What a scan's parameters file holds.
struct scan_parameters
{
    /// The tree's commit the scanner was built from: at most 4095 characters [M 2026-11-14:
    /// OpenCV 4.8.0 reads no longer string back] of printable ASCII, 0x20 to 0x7e [S: ASCII],
    /// without the apostrophe, not bounded by double quotes (docs/SCAN_FILES.md, "Strings").
    std::string commit;
    gca::object_class object_class;     ///< the object class the operator chose
    object_class_parameters parameters; ///< the set in force, which need not be the class's own
    /// The frames' size, intrinsics and depth unit, the stream size the source was opened for:
    /// width and height >= 1, fx and fy finite and > 0, cx and cy finite, the depth unit finite
    /// and > 0.
    frame_description stream;
    /// What delivered the frames and its versions (camera/frame_source.hpp), each string one the
    /// commit's rule admits; an empty string where the source has no such version.
    source_description source;
    /// The camera's run-time properties the camera refused a read or a write of while the scan's
    /// source was open (camera/color_hold.hpp): one flag for each row of color_property_table, in
    /// its order, true for a property refused at least once; all false for a source without the
    /// hold, a sequence's or the tests' synthetic one.
    std::array<bool, color_property_count> refused_properties;
    utc_time start;           ///< when the scan started, from 1970 to 9999 (scan/utc_time.hpp)
    utc_time stop;            ///< when it stopped, from 1970 to 9999
    scan_frame_counts frames; ///< the frames taken, fused, lost and not taken
    scan_end end;             ///< why the scan ended

    /// Compares two scans' parameters value by value.
    /// @param other [in] the other parameters.
    /// @return whether every value is equal.
    bool operator==(const scan_parameters &other) const = default;
};

/// Checks that the file holds each string of a source's description: at most 4095 characters
/// [M 2026-11-14: OpenCV 4.8.0 reads no longer string back] of printable ASCII, 0x20 to 0x7e
/// [S: ASCII], without the apostrophe, not bounded by double quotes (docs/SCAN_FILES.md,
/// "Strings"). A scan checks its source's description with it at its start,
/// before it writes anything, so that a scan whose parameters could not be written never starts.
/// @param source [in] the description, as the source gives it (camera/frame_source.hpp).
/// @return the reason one is not, naming its key in "source"; empty when each is.
/// @note Allocates for the reason. A failed allocation ends the program (std::terminate): the
///       function is noexcept.
std::string source_refusal(const source_description &source) noexcept;

/// Writes a scan's parameters as JSON (docs/SCAN_FILES.md), the doubles exactly.
/// @param path [in] the file's path; its directory exists. A file there is replaced.
/// @param parameters [in] the parameters.
/// @return written; or the reason, naming the path and the key: a length or threshold of the set
///         is not finite and > 0, a share not in (0, 1], min_depth_m is not less than max_depth_m,
///         too_close_depth_m does not lie between them, the budget is not from 1 to 2^53 bytes;
///         the stream's width or height is 0, fx or fy is not finite and > 0, cx or cy not finite,
///         the depth unit not finite and > 0; a time lies outside 1970 to 9999; why the scan
///         ended is no enumerator; a count of frames lies above 2^53, or taken is not fused + lost,
///         and 1 more when the scan ended because tracking failed; the commit or a string of the
///         source is one the file does not hold (source_refusal() says which; nothing is written
///         then); or OpenCV could not write the JSON; or the file could not be written
///         (scan/file_io.hpp, write_file()).
/// @note Allocates. A failed allocation ends the program (std::terminate): the function is
///       noexcept.
write_result write_parameters(const std::filesystem::path &path,
                              const scan_parameters &parameters) noexcept;

/// Reads a scan's parameters written by write_parameters().
/// @param path [in] the file's path.
/// @return the parameters; or the reason, naming the path and the key: the file cannot be read,
///         does not start with '{' after any white space, is no JSON that OpenCV reads, is of
///         another format version, holds a key the reader does not know or a key twice, lacks a
///         key, holds a value of another type, an unknown object class, a refused property that
///         is no name of color_property_table or stands twice, a count of iterations that is no
///         whole number from 0 to 2^32 - 1, a width or height that is no whole number from 1 to
///         2^32 - 1, a count of frames that is no whole number from 0 to 2^53, a value
///         of the stream that no float holds, a time that is not the text of utc_time_of(), an
///         unknown key of why the scan ended, or a string, a set or a fact the writer refuses; a
///         file written before the scan's facts and the set's six keys of 2026-11-16, or before
///         "refused_properties" of 2026-11-17, among them, which lacks a key.
/// @note Allocates. A failed allocation ends the program (std::terminate): the function is
///       noexcept.
read_result<scan_parameters> read_parameters(const std::filesystem::path &path) noexcept;
} // namespace gca
