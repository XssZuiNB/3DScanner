// SPDX-License-Identifier: 0BSD
/// @file imu_sample.hpp
/// One sample of the camera's IMU, its accelerometer's or its gyroscope's: three values along the
/// IMU's axes and the time they were taken on the frames' clock (docs/methods/imu-samples.md). A
/// frame carries the samples of each sensor taken since the frame before (camera/frame.hpp), and
/// the sequence keeps them (docs/SEQUENCE_FILE.md); separation is to read them, which links only
/// this library. A plain struct, header only.

#pragma once

#include <cstdint>

namespace gca
{
/// One sample of the accelerometer or of the gyroscope. The values are the camera's, bit for bit,
/// along the IMU's axes, which docs/methods/imu-samples.md, "The axes", relates to the camera
/// frame; no source rotates them.
struct imu_sample
{
    /// When it was taken, in µs on the frames' clock: the clock of the timestamps of the frames
    /// its frame source delivers (docs/methods/imu-samples.md, "The clock").
    std::uint64_t timestamp_us;
    float x; ///< along the IMU's x axis: m/s² of specific force, or rad/s of turn rate
    float y; ///< along the IMU's y axis, in the same unit
    float z; ///< along the IMU's z axis, in the same unit
};
} // namespace gca
