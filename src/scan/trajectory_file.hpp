// SPDX-License-Identifier: 0BSD
/// @file trajectory_file.hpp
/// A scan's trajectory (GLOSSARY.md) on disk: text, a header line with the format's version,
/// then one line for each frame the scanner took, with its index and timestamp and either the
/// camera's pose in the models' frame or the mark of tracking loss (docs/SCAN_FILES.md,
/// "Trajectory"). Numbers are written in their shortest form that reads back exactly, whatever
/// the locale. The writer refuses, and the reader refuses with the line's number, a pose that is
/// not finite or whose quaternion's norm differs from 1 by more than 1e-9.

#pragma once

#include "scan/file_io.hpp"
#include "types/camera_pose.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

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

/// Writes a trajectory (docs/SCAN_FILES.md, "Trajectory").
/// @param path [in] the file's path; its directory exists. A file there is replaced.
/// @param trajectory [in] the frames in the order the scanner took them.
/// @return written; or the reason, naming the path and the entry: a pose not finite or a
///         quaternion whose norm differs from 1 by more than 1e-9 (nothing is written then); or
///         the file could not be written (scan/file_io.hpp, write_file()).
/// @note Allocates. A failed allocation ends the program (std::terminate): the function is
///       noexcept.
write_result write_trajectory(const std::filesystem::path &path,
                              std::span<const trajectory_entry> trajectory) noexcept;

/// Reads a trajectory written by write_trajectory().
/// @param path [in] the file's path.
/// @return the frames in the file's order; or the reason, naming the path and the line: the file
///         cannot be read, its first line is not this format's header, a line is no frame line,
///         a pose is not finite or its quaternion is not unit within 1e-9, the file ends within a
///         line.
/// @note Allocates. A failed allocation ends the program (std::terminate): the function is
///       noexcept.
read_result<std::vector<trajectory_entry>> read_trajectory(
    const std::filesystem::path &path) noexcept;
} // namespace gca
