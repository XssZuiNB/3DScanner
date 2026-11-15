// SPDX-License-Identifier: 0BSD
/// @file frame_copy.hpp
/// A copy of a frame's index, timestamp, pixels and IMU samples, for the tests that check a frame
/// source left the caller's frame as it was or compare frames read back with frames rendered: a
/// frame itself cannot be copied (camera/frame.hpp). Also the comparison of two lists of samples
/// bit for bit. Header only.

#pragma once

#include "camera/frame.hpp"
#include "types/imu_sample.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <span>
#include <vector>

namespace gca_test
{
/// A frame's index, timestamp, pixels and IMU samples, copied to compare the frame with later.
struct frame_copy
{
    std::uint64_t index;                ///< the frame's index
    std::uint64_t timestamp_us;         ///< its timestamp, in µs
    std::vector<std::uint8_t> color;    ///< its color picture
    std::vector<std::uint16_t> depth;   ///< its depth picture
    std::vector<gca::imu_sample> accel; ///< its accelerometer's samples
    std::vector<gca::imu_sample> gyro;  ///< its gyroscope's samples
};

/// Copies a frame's index, timestamp, pixels and samples.
/// @param frame [in] the frame.
/// @return the copy.
inline frame_copy copy_of(const gca::frame &frame)
{
    return {frame.index(),
            frame.timestamp_us(),
            std::vector<std::uint8_t>(frame.color().begin(), frame.color().end()),
            std::vector<std::uint16_t>(frame.depth().begin(), frame.depth().end()),
            std::vector<gca::imu_sample>(frame.accel().begin(), frame.accel().end()),
            std::vector<gca::imu_sample>(frame.gyro().begin(), frame.gyro().end())};
}

/// Returns whether two lists of IMU samples are equal bit for bit: the same count, and each
/// sample's timestamp and the bits of its three values, so that -0 and 0 differ and a NaN equals
/// itself.
/// @param a [in] one list.
/// @param b [in] the other.
/// @return whether they are equal.
inline bool same_samples(std::span<const gca::imu_sample> a, std::span<const gca::imu_sample> b)
{
    const auto bits = [](float value) { return std::bit_cast<std::uint32_t>(value); };
    return std::ranges::equal(a, b, [&](const gca::imu_sample &p, const gca::imu_sample &q) {
        return p.timestamp_us == q.timestamp_us && bits(p.x) == bits(q.x) &&
               bits(p.y) == bits(q.y) && bits(p.z) == bits(q.z);
    });
}

/// Returns whether a frame holds what a copy holds.
/// @param frame [in] the frame.
/// @param copy [in] the copy.
/// @return whether the index, the timestamp, every pixel of color and depth and every sample, bit
///         for bit, are equal.
inline bool holds(const gca::frame &frame, const frame_copy &copy)
{
    return frame.index() == copy.index && frame.timestamp_us() == copy.timestamp_us &&
           std::ranges::equal(frame.color(), copy.color) &&
           std::ranges::equal(frame.depth(), copy.depth) &&
           same_samples(frame.accel(), copy.accel) && same_samples(frame.gyro(), copy.gyro);
}
} // namespace gca_test
