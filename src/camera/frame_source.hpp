// SPDX-License-Identifier: 0BSD
/// @file frame_source.hpp
/// The camera seam: a frame source (GLOSSARY.md) is opened for a stream size, gives one frame
/// after another, each with the IMU's samples taken since the frame before, and is closed again,
/// and it describes itself. When it has no frame it says
/// why, as a value: it refused to open, a timeout, a failure of the device with its message,
/// the end. No exception leaves it: its implementations catch what their device throws, and
/// every function of the seam is noexcept. Today the sequence (camera/sequence_reader.hpp), the
/// Orbbec camera (camera/orbbec_device.hpp) and the tests' synthetic source
/// (tests/synthetic_source.hpp) implement it. This header and the frame build and link without
/// OrbbecSDK (src/CMakeLists.txt, the library camera).

#pragma once

#include "camera/frame.hpp"
#include "types/frame_description.hpp"

#include <string>

namespace gca
{
/// What open() gave: the source is open, or the reason it refused.
struct open_result
{
    bool opened = false; ///< whether the source is open now
    std::string refusal; ///< why it did not open, in plain English; empty when it opened
};

/// Whether next() gave a frame, and why not when it did not.
enum class next_status
{
    frame,   ///< the frame was filled with the next frame
    timeout, ///< no frame came within the source's time limit; the source stays open
    failure, ///< the device failed, or the source is not open; the message says how
    end,     ///< the source has no frame left, as a sequence at its end
};

/// What next() gave.
struct next_result
{
    next_status status = next_status::failure; ///< a frame, or why there is none
    std::string message; ///< why there is no frame, in plain English; empty with a frame
};

/// What a source says about itself, for a scan's parameters.
struct source_description
{
    std::string camera;           ///< what delivers the frames: the camera's name, or the kind
                                  ///< of source
    std::string firmware_version; ///< the camera's firmware; empty for a source without one
    std::string sdk_version;      ///< OrbbecSDK's version; empty for a source without it

    /// Compares two descriptions string by string.
    /// @param other [in] the other description.
    /// @return whether the three strings are equal.
    bool operator==(const source_description &other) const = default;
};

/// The seam every source of frames stands behind. One thread at a time may use a source.
class frame_source
{
public:
    /// Builds the seam's part of a source, which holds nothing.
    frame_source() = default;

    /// A source is not copied: it stands for one device, file or scene and its state.
    frame_source(const frame_source &) = delete;
    frame_source &operator=(const frame_source &) = delete;

    /// Ends the source; an implementation's destructor closes it when it is open.
    virtual ~frame_source() = default;

    /// Opens the source for a stream size.
    /// @param size [in] the size of the pictures asked for, in px.
    /// @return opened, or not opened with the reason: the source cannot deliver that size, its
    ///         device is missing or held, or the source is open already.
    /// @pre The source is closed; when it is open, the call is refused and the source stays
    ///      open as it was.
    /// @note Allocates for a refusal's reason, and an implementation may allocate what it opens
    ///       (the Orbbec camera allocates OrbbecSDK's context, device and pipeline,
    ///       camera/orbbec_device.hpp). A failed allocation ends the program (std::terminate):
    ///       the function is noexcept.
    virtual open_result open(const frame_size &size) noexcept = 0;

    /// Gives the next frame.
    /// @param out [out] filled with the next frame when the status is frame (frame::prepare(),
    ///            then every pixel); left as it was on every other status. Owned by the caller,
    ///            which may pass the same frame again and again.
    /// @return frame; or timeout, failure or end, each with its message. The frames of one
    ///         opening have the size the source was opened for and strictly increasing indices
    ///         and timestamps; a gap in the indices stands for frames the source did not
    ///         deliver. Each frame carries the IMU's samples not given with an earlier frame whose
    ///         timestamps are at most its own, for each sensor in strictly increasing time, so
    ///         that the samples of an opening's frames, in frame order, are each sensor's stream
    ///         (docs/methods/imu-samples.md, "Which frame carries a sample"); a source without an
    ///         IMU, or whose IMU did not open, gives none.
    /// @pre The source is open; when it is not, the result is a failure saying so.
    /// @note Allocates for the message of a status other than frame, and for out's buffers when
    ///       they grow (frame::prepare()); beyond that, a delivered frame of the size out already
    ///       has allocates nothing in the source's own code. An implementation may allocate inside
    ///       a vendor's SDK: the Orbbec camera's next() does, in OrbbecSDK, for every frameset
    ///       (camera/orbbec_device.hpp). A failed allocation ends the program (std::terminate):
    ///       the function is noexcept.
    virtual next_result next(frame &out) noexcept = 0;

    /// Closes the source. A closed source stays closed.
    virtual void close() noexcept = 0;

    /// Describes the source.
    /// @return the camera or the kind of source, and the versions of its firmware and of
    ///         OrbbecSDK where it has them. A version the source reads from its device is empty
    ///         until an opening has reached the device, so before the first open() it is empty.
    /// @note Allocates for the strings. A failed allocation ends the program (std::terminate):
    ///       the function is noexcept.
    virtual source_description describe() const noexcept = 0;
};
} // namespace gca
