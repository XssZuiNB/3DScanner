// SPDX-License-Identifier: 0BSD
/// @file parameters_file.hpp
/// A scan's parameters on disk: JSON with a format version, the tree's commit, the object class
/// and the set of parameters in force (docs/SCAN_FILES.md, "Parameters"). The writer and the
/// reader use OpenCV's FileStorage on memory and do the file's I/O themselves
/// (scan/file_io.hpp). The reader refuses, with the reason, a file of another version, a key it
/// does not know, a key twice or missing, a value of another type, a commit that OpenCV does not
/// write as JSON and a set the scanner cannot run with; the writer refuses that commit and that
/// set too, so it writes nothing its reader refuses.

#pragma once

#include "scan/file_io.hpp"
#include "scan/object_class.hpp"

#include <filesystem>
#include <string>

namespace gca
{
/// What a scan's parameters file holds.
struct scan_parameters
{
    std::string commit; ///< the tree's commit the scanner was built from: at most 4095
                        ///< characters of printable ASCII (0x20 to 0x7e) without the
                        ///< apostrophe, not bounded by double quotes (docs/SCAN_FILES.md)
    gca::object_class object_class;     ///< the object class the operator chose
    object_class_parameters parameters; ///< the set in force, which need not be the class's own

    /// Compares two scans' parameters value by value.
    /// @param other [in] the other parameters.
    /// @return whether every value is equal.
    bool operator==(const scan_parameters &other) const = default;
};

/// Writes a scan's parameters as JSON (docs/SCAN_FILES.md), the doubles exactly.
/// @param path [in] the file's path; its directory exists. A file there is replaced.
/// @param parameters [in] the parameters.
/// @return written; or the reason, naming the path and the key: a length or threshold of the set
///         is not finite and > 0, a share not in (0, 1], min_depth_m is not less than max_depth_m,
///         too_close_depth_m does not lie between them, the budget is not from 1 to 2^53 bytes,
///         the commit holds more than 4095 characters, a byte outside printable ASCII or
///         an apostrophe, or starts and ends with a double quote (nothing is written then); or
///         OpenCV could not write the JSON; or the file could not be written (scan/file_io.hpp,
///         write_file()).
/// @note Allocates. A failed allocation ends the program (std::terminate): the function is
///       noexcept.
write_result write_parameters(const std::filesystem::path &path,
                              const scan_parameters &parameters) noexcept;

/// Reads a scan's parameters written by write_parameters().
/// @param path [in] the file's path.
/// @return the parameters; or the reason, naming the path and the key: the file cannot be read,
///         does not start with '{' after any white space, is no JSON that OpenCV reads, is of
///         another format version, holds a key the reader does not know or a key twice, lacks a
///         key, holds a value of another type, an unknown object class, a count of iterations
///         that is no whole number from 0 to 2^32 - 1, or a commit or a set the writer refuses;
///         a file written before the set's 13 keys of 2026-11-15 among them, which lacks a key.
/// @note Allocates. A failed allocation ends the program (std::terminate): the function is
///       noexcept.
read_result<scan_parameters> read_parameters(const std::filesystem::path &path) noexcept;
} // namespace gca
