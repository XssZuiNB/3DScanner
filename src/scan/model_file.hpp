// SPDX-License-Identifier: 0BSD
/// @file model_file.hpp
/// A model on disk, the full model or the object model (GLOSSARY.md): binary little-endian PLY,
/// for each point its position in m and its unit normal as 32-bit floats and its color as 8-bit
/// RGB (docs/SCAN_FILES.md, "Models"). The bytes are the same on any host. The reader reads
/// exactly the header the writer writes, and refuses another header and data of another length;
/// the writer and the reader refuse a normal whose length differs from 1 by more than 1e-6. A
/// model's point is types/model_point.hpp.

#pragma once

#include "scan/file_io.hpp"
#include "types/model_point.hpp"

#include <filesystem>
#include <span>
#include <vector>

namespace gca
{
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
