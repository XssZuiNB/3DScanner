// SPDX-License-Identifier: 0BSD
/// @file model_file.hpp
/// A model on disk, the full model or the object model (GLOSSARY.md): binary little-endian PLY,
/// for each point its position in m and its unit normal as 32-bit floats and its color as 8-bit
/// RGB (docs/SCAN_FILES.md, "Models"). The bytes are the same on any host. The reader reads
/// exactly the header the writer writes, and refuses another header and data of another length;
/// the writer and the reader refuse a normal whose length differs from 1 by more than 1e-6.

#pragma once

#include "scan/file_io.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include <Eigen/Core>

namespace gca
{
/// One point of a model, in the models' frame: the camera frame of the first fused frame (x
/// right, y down, z forward, in m). It lives here while only the scan module uses it; when the
/// sparse TSDF (ADR 0004) produces model points, it moves to the lowest layer that needs it
/// (AGENTS.md §7 rule 4).
struct model_point
{
    Eigen::Vector3f position_m;        ///< the point, in m
    Eigen::Vector3f normal;            ///< the surface's unit normal at the point, its length
                                       ///< within 1e-6 of 1
    std::array<std::uint8_t, 3> color; ///< red, green, blue, 0 to 255
};

/// Writes a model (docs/SCAN_FILES.md, "Models").
/// @param path [in] the file's path; its directory exists. A file there is replaced.
/// @param points [in] the points, written in this order and as they are.
/// @return written; or the reason, naming the path: a point's normal is not unit within 1e-6
///         (nothing is written then), or the file could not be written (scan/file_io.hpp,
///         write_file()).
/// @note Allocates the file's bytes, 27 per point, before it writes them. A failed allocation
///       ends the program (std::terminate): the function is noexcept.
write_result write_model(const std::filesystem::path &path,
                         std::span<const model_point> points) noexcept;

/// Reads a model written by write_model().
/// @param path [in] the file's path.
/// @return the points in the file's order; or the reason, naming the path: the file cannot be
///         read, a header line differs from the writer's (its number given), the data after the
///         header is not 27 bytes for each point the header declares, a point's normal is not
///         unit within 1e-6 (the point named).
/// @note Allocates the file's size and the points. A failed allocation ends the program
///       (std::terminate): the function is noexcept.
read_result<std::vector<model_point>> read_model(const std::filesystem::path &path) noexcept;
} // namespace gca
