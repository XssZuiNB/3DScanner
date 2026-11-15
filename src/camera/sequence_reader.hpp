// SPDX-License-Identifier: 0BSD
/// @file sequence_reader.hpp
/// The sequence's reader (GLOSSARY.md, "Sequence"; docs/SEQUENCE_FILE.md): a frame source
/// (camera/frame_source.hpp) that gives a sequence's frames in the order written, bit for bit,
/// with their IMU samples, as fast as they are asked, and with each frame the pose tracking gave
/// it or its mark. A
/// sequence whose writer was cut off still gives every whole frame it holds, and then says that
/// it ends early. Every failure is a value with its reason; no exception leaves the reader.

#pragma once

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/sequence_format.hpp"
#include "types/frame_description.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace gca
{
/// A sequence on disk as a frame source.
class sequence_reader final : public frame_source
{
public:
    /// Builds the reader of a sequence, closed; the file is opened by open().
    /// @param path [in] the sequence's path; copied.
    explicit sequence_reader(std::filesystem::path path);

    /// Closes the reader when it is open.
    ~sequence_reader() override;

    /// Opens the sequence and reads its header; the next frame is the sequence's first.
    /// @param size [in] the size of the pictures asked for, in px; the sequence's own.
    /// @return opened; or not, with the reason, naming the file: the reader is open already; the
    ///         file cannot be opened or read, with the system's message; it is no sequence (its
    ///         first bytes are not the magic); it is of a version other than 2, version 1 among
    ///         them; it ends within its header; its frames are of another size than asked (both
    ///         sizes named).
    /// @note Allocates the reader's own frame, of the sequence's size, when it opens, unless an
    ///       opening before allocated it; allocates for a refusal's reason. A failed allocation
    ///       ends the program (std::terminate): the function is noexcept.
    open_result open(const frame_size &size) noexcept override;

    /// Gives the next frame of the sequence. The record is read into a frame of the reader's own,
    /// which is swapped with out once it is whole, so that out is left as it was on every other
    /// status.
    /// @param out [out] the next frame, its index, timestamp, description, samples and pixels as
    ///            written; left as it was on every other status. Owned by the caller.
    /// @return frame; end once the sequence's end is read, at this and every later call; or
    ///         failure with the reason, naming the file and the record (the first after the
    ///         header is record 1), at this and every later call: the reader is not open; the
    ///         file ends early, before the sequence's end, within a record or between two (its
    ///         writer was cut off, and every whole frame before has been given; ended_early()
    ///         says so); a record's kind is unknown; a frame's index or timestamp does not
    ///         increase; a sample does not follow its sensor's last or lies after its frame's
    ///         timestamp (imu_sample_order); bytes follow the end; the file cannot be read, with
    ///         the system's message.
    /// @note Allocates nothing when out already has the sequence's size and its lists of samples
    ///       hold the record's samples, as the reader's frame and out change buffers. When out's
    ///       buffers are smaller, they become the reader's frame, which grows them once, at the
    ///       next frame. Allocates for the message of a status other than frame. A failed
    ///       allocation ends the program (std::terminate): the function is noexcept.
    next_result next(frame &out) noexcept override;

    /// Closes the sequence's file. A closed reader stays closed.
    void close() noexcept override;

    /// Describes the reader.
    /// @return "sequence " and the path as the camera; no firmware or SDK version, which the
    ///         sequence does not hold. A scan's parameters file holds only some strings
    ///         (docs/SCAN_FILES.md, "Strings"): a path with an apostrophe, or one that starts and
    ///         ends with a double quote, cannot go into it as it is.
    /// @note Allocates for the strings. A failed allocation ends the program (std::terminate):
    ///       the function is noexcept.
    source_description describe() const noexcept override;

    /// Tells what tracking said about the frame next() gave last.
    /// @return its pose, bit for bit, or its mark; not tracked before the first frame of an
    ///         opening. Valid until the next call of next().
    const frame_tracking &tracking() const noexcept;

    /// Tells whether the sequence ended early: next() reported that the file ends before the
    /// sequence's end, as a file whose writer was cut off does, an unfinished scan's among them.
    /// The frames before were given whole, so a caller may go on with them.
    /// @return true after next() reported the early end, until the next open(); false before,
    ///         and after the end or any other failure.
    bool ended_early() const noexcept;

private:
    /// Ends the sequence's reading with a status other than frame, kept for every later call.
    /// @param status [in] end or failure.
    /// @param message [in] the message, without the file's path.
    /// @return the result, with the path before the message.
    next_result finish(next_status status, const std::string &message);

    std::filesystem::path m_path;              ///< the sequence's path
    int m_fd = -1;                             ///< the open file's descriptor; -1 when closed
    frame_description m_description{};         ///< the description of every frame, from the header
    frame m_spare;                             ///< the frame a record is read into before the swap
    frame_tracking m_tracking = not_tracked{}; ///< what tracking said about the last frame
    std::uint64_t m_records = 0;               ///< the frames given since open()
    std::uint64_t m_last_index = 0;            ///< the index of the frame given last
    std::uint64_t m_last_timestamp_us = 0;     ///< its timestamp, in µs
    imu_sample_order m_sample_order;           ///< each sensor's last sample given
    bool m_finished = false;                   ///< whether the end or a failure was reached
    bool m_ended_early = false;                ///< whether the file ended before the end
    next_result m_ending;                      ///< what every next() gives once finished
};
} // namespace gca
