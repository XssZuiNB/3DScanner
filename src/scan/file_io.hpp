// SPDX-License-Identifier: 0BSD
/// @file file_io.hpp
/// What the writers and readers of a scan's files (docs/SCAN_FILES.md) share: what they give
/// back, a file written or the reason it was not, a value read or the reason the file was
/// refused; a whole file written or read at once; and a number written in its shortest exact
/// form. A file is written beside its name and renamed onto it once it is whole and on disk, so
/// a file found under its name is whole. A failure is a value (CODING_STANDARDS.md, "The
/// rewrite's code"); no exception leaves these functions.

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace gca
{
/// What a writer of a scan's file gave.
struct write_result
{
    bool written = false; ///< whether the whole file stands under its name, synced to disk
    std::string reason;   ///< why it does not, in plain English, naming the file; empty when
                          ///< written
};

/// What a reader of a scan's file gave.
/// @tparam T the value the file holds.
template <typename T> struct read_result
{
    std::optional<T> value; ///< the file's content; nothing when the file was refused
    std::string reason;     ///< why it was refused, in plain English, naming the file and the
                            ///< place in it; empty when it was read
};

/// Writes a whole file: the bytes go into "<path>.part", created or truncated, which is synced
/// to disk and renamed onto the path, replacing a file there; then the directory is synced, so
/// that the rename survives a power cut. Until a write fails, nothing stands under the path but
/// the file that stood there; on a failure "<path>.part" is removed. A power cut during a write
/// can leave "<path>.part", which no reader takes and the next write truncates.
/// @param path [in] the file's path; its directory exists.
/// @param bytes [in] the file's content.
/// @return written; or the reason with the path and the system's message: "<path>.part" cannot
///         be created, a write, its sync, its close or the rename failed (a full disk, a file
///         size limit, a directory at the path), each leaving the path as it was; or the
///         directory's sync failed, after which the new file stands under the path but a power
///         cut may undo the rename.
/// @note Allocates for the reason and the part's path. A failed allocation ends the program
///       (std::terminate): the function is noexcept.
write_result write_file(const std::filesystem::path &path, std::string_view bytes) noexcept;

/// Reads a whole file into memory.
/// @param path [in] the file's path.
/// @return its bytes, or the reason with the path and the system's message: the file cannot be
///         opened or read (missing, a directory, no permission).
/// @note Allocates the file's size. A failed allocation ends the program (std::terminate): the
///       function is noexcept.
read_result<std::string> read_file(const std::filesystem::path &path) noexcept;

/// Appends a number in its shortest decimal form that reads back to the same value
/// (std::to_chars), whatever the locale; std::from_chars reads it back bit for bit
/// [M 2026-11-14: 10^6 random doubles, g++ 11.4, a test program outside the tree].
/// @param text [in,out] the text appended to.
/// @param value [in] the number; "nan", "-nan", "inf" or "-inf" when it is not finite.
/// @note Allocates when the text grows. A failed allocation ends the program (std::terminate):
///       the function is noexcept.
void append_number(std::string &text, double value) noexcept;

/// Appends a whole number in decimal (std::to_chars).
/// @param text [in,out] the text appended to.
/// @param value [in] the number.
/// @note Allocates when the text grows. A failed allocation ends the program (std::terminate):
///       the function is noexcept.
void append_number(std::string &text, std::uint64_t value) noexcept;
} // namespace gca
