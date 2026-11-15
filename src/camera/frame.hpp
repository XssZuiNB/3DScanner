// SPDX-License-Identifier: 0BSD
/// @file frame.hpp
/// A frame (GLOSSARY.md): one color picture and the depth picture aligned to it, taken at one
/// time. It owns both buffers in host memory and carries its index, its timestamp and its
/// description: the size, the intrinsics as the one plain camera struct, and the depth unit.
/// Color is RGB8: 3 bytes per pixel in the order red, green, blue, rows top to bottom, pixels
/// left to right, row v starting at byte 3 * v * width. Depth is 16 bits per pixel in the same
/// order, value * depth_unit_m meters along the camera's z axis; 0 means no depth. It also
/// carries the IMU's samples taken since the frame before, the accelerometer's and the
/// gyroscope's, each in a list of its own (types/imu_sample.hpp, docs/methods/imu-samples.md). A
/// frame source fills a frame its caller owns (camera/frame_source.hpp), so a caller that reuses
/// one frame allocates its buffers once.

#pragma once

#include "types/frame_description.hpp"
#include "types/imu_sample.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace gca
{
/// The samples of each IMU sensor a prepared frame holds without allocating [?: a choice]: twice
/// the 33 samples a frame gets at the IMU's fastest rate, 1 kHz, and 30 fps [S: computed,
/// docs/methods/imu-samples.md, "On the per-frame path"]; at the camera's 200 Hz a frame gets 6 or
/// 7 [S: computed, 200 / 30].
inline constexpr std::size_t frame_imu_samples_reserved = 64;

/// One color picture and the aligned depth picture, with their index, timestamp and description,
/// and the IMU's samples since the frame before.
/// Move-only, so that no copy allocates unseen. Not synchronized: while one thread prepares or
/// writes a frame, no other thread uses it.
class frame
{
public:
    /// An empty frame: size 0 x 0, no buffers, index 0, timestamp 0. A frame source gives it its
    /// content.
    frame() = default;

    /// A frame is not copied: a copy allocates both buffers.
    frame(const frame &) = delete;
    frame &operator=(const frame &) = delete;

    /// Takes over another frame's buffers and values without allocating.
    /// @param other [in,out] the frame moved from; left valid, its buffers unspecified.
    frame(frame &&other) noexcept = default;

    /// Takes over another frame's buffers and values without allocating; this frame's buffers
    /// are released.
    /// @param other [in,out] the frame moved from; left valid, its buffers unspecified.
    /// @return this frame.
    frame &operator=(frame &&other) noexcept = default;

    /// Releases the buffers.
    ~frame() = default;

    /// Prepares the frame for a frame source to fill: sets its description, index and timestamp,
    /// sizes the color and depth buffers for the description's size and empties the IMU's two
    /// lists of samples. The buffers' contents are left as they were; the source writes every
    /// pixel of both, and the samples it has.
    /// @param description [in] the frame's size, intrinsics and depth unit; copied.
    /// @param index [in] the frame's index among the frames of its source's opening.
    /// @param timestamp_us [in] the time the frame was taken, in µs on the source's clock.
    /// @note Allocates only when a buffer's capacity is less than the size needs, and when a list
    ///       of samples holds less than frame_imu_samples_reserved, which it reserves then; a frame
    ///       prepared for the same size again does not allocate. Throws std::bad_alloc when an
    ///       allocation fails.
    void prepare(const frame_description &description, std::uint64_t index,
                 std::uint64_t timestamp_us);

    /// Gives the frame's description.
    /// @return the size, intrinsics and depth unit it was prepared with, valid as long as the
    ///         frame and until its next prepare().
    const frame_description &description() const noexcept;

    /// Gives the frame's index.
    /// @return its index among the frames of its source's opening: the indices of one opening
    ///         increase strictly, and a gap stands for frames the source did not deliver.
    std::uint64_t index() const noexcept;

    /// Gives the frame's timestamp.
    /// @return the time it was taken, in µs on the source's clock; the timestamps of one opening
    ///         increase strictly.
    std::uint64_t timestamp_us() const noexcept;

    /// Gives the color picture for a source to write.
    /// @return 3 * width * height bytes, RGB8 in the file's order, owned by the frame; valid
    ///         until its next prepare() or its end.
    std::span<std::uint8_t> color() noexcept;

    /// Gives the color picture.
    /// @return 3 * width * height bytes, RGB8 in the file's order, owned by the frame; valid
    ///         until its next prepare() or its end.
    std::span<const std::uint8_t> color() const noexcept;

    /// Gives the depth picture for a source to write.
    /// @return width * height depth values in the file's order, owned by the frame; valid until
    ///         its next prepare() or its end.
    std::span<std::uint16_t> depth() noexcept;

    /// Gives the depth picture.
    /// @return width * height depth values in the file's order, owned by the frame; valid until
    ///         its next prepare() or its end.
    std::span<const std::uint16_t> depth() const noexcept;

    /// Gives the accelerometer's samples for a source to fill: empty after prepare(), the source
    /// appends the samples the frame carries, in increasing time (camera/frame_source.hpp).
    /// @return the list, owned by the frame; valid until its end. Appending within its capacity,
    ///         frame_imu_samples_reserved after prepare() or what the list held before, allocates
    ///         nothing; beyond it the list grows, and std::bad_alloc is thrown when that fails.
    std::vector<imu_sample> &accel() noexcept;

    /// Gives the accelerometer's samples.
    /// @return the samples taken since the frame before, in increasing time, along the IMU's axes
    ///         in m/s² (docs/methods/imu-samples.md); empty for a source without an IMU. Owned by
    ///         the frame; valid until its next prepare() or its end.
    std::span<const imu_sample> accel() const noexcept;

    /// Gives the gyroscope's samples for a source to fill, as accel() does the accelerometer's.
    /// @return the list, owned by the frame; valid until its end. Allocates as accel() says.
    std::vector<imu_sample> &gyro() noexcept;

    /// Gives the gyroscope's samples.
    /// @return the samples taken since the frame before, in increasing time, along the IMU's axes
    ///         in rad/s (docs/methods/imu-samples.md); empty for a source without an IMU. Owned by
    ///         the frame; valid until its next prepare() or its end.
    std::span<const imu_sample> gyro() const noexcept;

private:
    frame_description m_description{};
    std::uint64_t m_index = 0;
    std::uint64_t m_timestamp_us = 0;
    std::vector<std::uint8_t> m_color;
    std::vector<std::uint16_t> m_depth;
    std::vector<imu_sample> m_accel; ///< the accelerometer's samples since the frame before
    std::vector<imu_sample> m_gyro;  ///< the gyroscope's samples since the frame before
};
} // namespace gca
