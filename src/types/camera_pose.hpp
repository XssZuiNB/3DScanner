// SPDX-License-Identifier: 0BSD
/// @file camera_pose.hpp
/// The camera's pose at a frame, as tracking gives it: where the camera stands and how it is
/// turned in the models' frame, the camera frame of the first fused frame. The trajectory of a
/// scan (scan/trajectory_file.hpp, docs/SCAN_FILES.md) and the sequence
/// (camera/sequence_format.hpp, docs/SEQUENCE_FILE.md) both hold it, and separation's proposal
/// reads a trajectory of them. A plain struct of Eigen's value types and the one check of a pose
/// that the trajectory's file and separation take, header only.

#pragma once

#include <cmath>

#include <Eigen/Geometry>

namespace gca
{
/// The camera's pose at a frame: it maps a point of the camera frame (x right, y down, z
/// forward, in m) into the models' frame, the camera frame of the first fused frame, as
/// orientation * point + position_m.
struct camera_pose
{
    Eigen::Vector3d position_m;     ///< the camera's centre in the models' frame, in m, finite
    Eigen::Quaterniond orientation; ///< the rotation from the camera frame into the models'
                                    ///< frame, a unit quaternion (Hamilton, Eigen's); q and -q
                                    ///< are the same rotation, and a file keeps the sign given
};

/// How far a pose's quaternion's norm may lie from 1: 1e-9 [?: a choice]: a pose solve in double
/// precision leaves a unit quaternion's norm within a few 1e-16 of 1, and the trajectory's file
/// keeps every bit (docs/SCAN_FILES.md, "Trajectory").
inline constexpr double pose_norm_tolerance = 1e-9;

/// What is wrong with a pose, when anything is.
enum class pose_fault
{
    none,                ///< a finite position and a quaternion of norm 1 within the tolerance
    position_not_finite, ///< a coordinate of the position is not finite
    norm_not_one, ///< the quaternion's norm differs from 1 by more than pose_norm_tolerance, or is
                  ///< NaN
};

/// Checks a pose as the trajectory's file and separation take it: its position first.
/// @param pose [in] the pose.
/// @return none, or the first fault found.
/// @note Host only; allocates nothing.
inline pose_fault fault_of(const camera_pose &pose) noexcept
{
    if (!pose.position_m.allFinite())
        return pose_fault::position_not_finite;
    if (!(std::abs(pose.orientation.norm() - 1.0) <= pose_norm_tolerance))
        return pose_fault::norm_not_one;
    return pose_fault::none;
}
} // namespace gca
