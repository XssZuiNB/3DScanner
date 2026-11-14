// SPDX-License-Identifier: 0BSD
/// @file posix_file.hpp
/// The POSIX file calls that the writers and readers of the scanner's files share: the sequence
/// (camera/sequence_writer.hpp, camera/sequence_reader.hpp) and the files a scan keeps
/// (scan/file_io.hpp): the system's message, a write of all bytes, a read up to a size, a
/// directory's sync. Each failure comes back with the system's message, and what a writer gave
/// is a value: the file written, or the reason it was not (CODING_STANDARDS.md, "The rewrite's
/// code"). No exception leaves these functions.

#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace gca
{
/// What a writer of a file gave.
struct write_result
{
    bool written = false; ///< whether the whole file stands under its name, synced to disk
    std::string reason;   ///< why it does not, in plain English, naming the file; empty when
                          ///< written
};

/// Gives the system's message for the error in errno. It uses GNU's strerror_r(), which writes
/// into the caller's buffer or returns a static string, so two threads may call it at once;
/// strerror(), which libstdc++'s std::error_code::message() calls [B: objdump -T
/// libstdc++.so.6.0.30], shares one buffer.
/// @return the message, as "No space left on device".
/// @note Allocates for the message. A failed allocation ends the program (std::terminate): the
///       function is noexcept. Two threads may call it at once.
std::string system_message() noexcept;

/// Writes all bytes to a file, again after a partial write or an interrupted call.
/// @param fd [in] the descriptor of a file open for writing.
/// @param bytes [in] the bytes; owned by the caller, read during the call only.
/// @return whether every byte was written; on false errno holds the reason, and the bytes before
///         the failure may stand in the file.
/// @note Allocates nothing. Two threads may call it at once, each on a file of its own.
bool write_all(int fd, std::string_view bytes) noexcept;

/// Reads up to a number of bytes, again after a short read or an interrupted call, until that
/// number or the file's end.
/// @param fd [in] the descriptor of a file open for reading.
/// @param buffer [out] where the bytes go, at least size of them; owned by the caller.
/// @param size [in] the bytes asked for.
/// @return the bytes read, fewer than size only at the file's end; nothing on an error, with
///         errno holding the reason.
/// @note Allocates nothing. Two threads may call it at once, each on a file of its own.
std::optional<std::size_t> read_up_to(int fd, void *buffer, std::size_t size) noexcept;

/// Syncs a directory, so that a file created or renamed in it keeps its name after a power cut.
/// @param directory [in] the directory; empty for the working directory.
/// @return whether it was synced; on false errno holds the reason.
/// @note Allocates nothing. Two threads may call it at once.
bool sync_directory(const std::filesystem::path &directory) noexcept;
} // namespace gca
