// SPDX-License-Identifier: 0BSD
/// @file trajectory_entry.hpp
/// One frame of a scan's trajectory (GLOSSARY.md): the frame's index and timestamp with the
/// camera's pose tracking gave it, or none when the frame is in tracking loss. The trajectory's
/// file (scan/trajectory_file.hpp, docs/SCAN_FILES.md, "Trajectory") writes and reads a list of
/// them, one for each frame the scanner took. A plain struct, header only.

#pragma once

#include "types/camera_pose.hpp"

#include <cstdint>
#include <optional>

namespace gca
{
/// One frame the scanner took.
struct trajectory_entry
{
    std::uint64_t index;             ///< the frame's index (camera/frame.hpp, frame::index())
    std::uint64_t timestamp_us;      ///< the frame's timestamp, in µs on the source's clock
    std::optional<camera_pose> pose; ///< the camera's pose (types/camera_pose.hpp); nothing
                                     ///< when the frame is in tracking loss
};
} // namespace gca
