// SPDX-License-Identifier: 0BSD
/// @file imu_sample.hpp
/// One sample of the camera's IMU, its accelerometer's or its gyroscope's: three values along the
/// IMU's axes and the time they were taken on the frames' clock (docs/methods/imu-samples.md). A
/// frame carries the samples of each sensor taken since the frame before (camera/frame.hpp), and
/// the sequence keeps them (docs/SEQUENCE_FILE.md); separation's gravity reads the
/// accelerometer's (separation/gravity.hpp). Beside it the rotation from the IMU's axes into the
/// camera frame, which a reader of the samples applies. A plain struct and a constant, header
/// only.

#pragma once

#include <cstdint>

#include <Eigen/Geometry>

namespace gca
{
/// One sample of the accelerometer or of the gyroscope. The values are the camera's, bit for bit,
/// along the IMU's axes, which imu_to_camera() relates to the camera frame; no source rotates
/// them.
struct imu_sample
{
    /// When it was taken, in µs on the frames' clock: the clock of the timestamps of the frames
    /// its frame source delivers (docs/methods/imu-samples.md, "The clock").
    std::uint64_t timestamp_us;
    float x; ///< along the IMU's x axis: m/s² of specific force, or rad/s of turn rate
    float y; ///< along the IMU's y axis, in the same unit
    float z; ///< along the IMU's z axis, in the same unit
};

/// Gives the rotation from the IMU's axes into the camera frame (x right, y down, z forward), the
/// camera's: the identity [?: initial: OrbbecSDK reports the IMU's extrinsics equal to the depth
/// camera's (M 2026-11-13), an early fit put the axes 5.2 degrees apart within its error, and the
/// dominant level plane of the recording's full model lies 0.86 degrees from the gravity found with
/// it (M 2026-11-16 recording, early); docs/methods/imu-samples.md, "The axes"].
/// @return the rotation as a unit quaternion (Hamilton, Eigen's): it turns a sample's values into
///         the camera frame.
inline Eigen::Quaterniond imu_to_camera() noexcept
{
    return Eigen::Quaterniond::Identity();
}
} // namespace gca
