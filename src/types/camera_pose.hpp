// SPDX-License-Identifier: 0BSD
/// @file camera_pose.hpp
/// The camera's pose at a frame, as tracking gives it: where the camera stands and how it is
/// turned in the models' frame, the camera frame of the first fused frame. The trajectory of a
/// scan (scan/trajectory_file.hpp, docs/SCAN_FILES.md) and the sequence
/// (camera/sequence_format.hpp, docs/SEQUENCE_FILE.md) both hold it. A plain struct of Eigen's
/// value types, header only.

#pragma once

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
} // namespace gca
