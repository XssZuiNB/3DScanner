// SPDX-License-Identifier: 0BSD
/// @file file_io.cpp
/// A whole file written beside its name and renamed onto it, and a whole file read, with POSIX
/// calls, so that each failure comes back with the system's message; and the shortest exact
/// form of a number (see file_io.hpp).

#include "scan/file_io.hpp"

#include <cerrno>
#include <charconv>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gca
{
namespace
{
/// Gives the system's message for the error in errno. GNU's strerror_r() writes into the
/// caller's buffer or returns a static string, so two threads may call it at once; strerror(),
/// which libstdc++'s std::error_code::message() calls [B: objdump -T libstdc++.so.6.0.30],
/// shares one buffer.
/// @return the message.
std::string system_message()
{
    char buffer[256];
    return ::strerror_r(errno, buffer, sizeof(buffer));
}

/// Writes all bytes to a file, again after a partial write or an interrupted call.
/// @param fd [in] the open file's descriptor.
/// @param bytes [in] the bytes.
/// @return whether every byte was written; on false errno holds the reason.
bool write_all(int fd, std::string_view bytes)
{
    std::size_t done = 0;
    while (done < bytes.size())
    {
        const ssize_t n = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (n < 0 && errno == EINTR)
            continue;
        if (n < 0)
            return false;
        done += static_cast<std::size_t>(n);
    }
    return true;
}

/// Reads all bytes of a file, again after a short read or an interrupted call.
/// @param fd [in] the open file's descriptor, at its start.
/// @param out [out] the file's bytes, appended.
/// @return whether the file was read to its end; on false errno holds the reason.
bool read_all(int fd, std::string &out)
{
    char buffer[1 << 16];
    for (;;)
    {
        const ssize_t n = ::read(fd, buffer, sizeof(buffer));
        if (n < 0 && errno == EINTR)
            continue;
        if (n < 0)
            return false;
        if (n == 0)
            return true;
        out.append(buffer, static_cast<std::size_t>(n));
    }
}

/// Syncs a directory, so that a rename in it survives a power cut.
/// @param directory [in] the directory; empty for the working directory.
/// @return whether it was synced; on false errno holds the reason.
bool sync_directory(const std::filesystem::path &directory)
{
    const int fd =
        ::open(directory.empty() ? "." : directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0)
        return false;
    const bool synced = ::fsync(fd) == 0;
    const int saved = errno;
    ::close(fd);
    errno = saved;
    return synced;
}

/// Appends a number in its shortest form that reads back exactly.
/// @tparam T double or std::uint64_t.
/// @param text [in,out] the text appended to.
/// @param value [in] the number.
template <typename T> void append_digits(std::string &text, T value)
{
    char digits[32];
    const std::to_chars_result end = std::to_chars(digits, digits + sizeof(digits), value);
    text.append(digits, end.ptr);
}
} // namespace

write_result write_file(const std::filesystem::path &path, std::string_view bytes) noexcept
{
    std::filesystem::path part = path;
    part += ".part";
    const int fd = ::open(part.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0)
        return {false,
                path.string() + ": cannot create " + part.string() + ": " + system_message()};
    std::string failure;
    if (!write_all(fd, bytes))
        failure = "the write failed: " + system_message();
    else if (::fsync(fd) != 0)
        failure = "the sync to disk failed: " + system_message();
    if (::close(fd) != 0 && failure.empty())
        failure = "the close failed: " + system_message();
    if (failure.empty() && std::rename(part.c_str(), path.c_str()) != 0)
        failure = "the rename from " + part.string() + " failed: " + system_message();
    if (!failure.empty())
    {
        ::unlink(part.c_str());
        return {false, path.string() + ": " + failure};
    }
    if (!sync_directory(path.parent_path()))
        return {false, path.string() + ": the sync of its directory failed: " + system_message() +
                           "; the file stands under its name, but a power cut may undo the rename"};
    return {true, {}};
}

read_result<std::string> read_file(const std::filesystem::path &path) noexcept
{
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return {std::nullopt, path.string() + ": cannot open: " + system_message()};
    std::string bytes;
    struct stat status;
    if (::fstat(fd, &status) == 0 && status.st_size > 0)
        bytes.reserve(static_cast<std::size_t>(status.st_size));
    const bool read = read_all(fd, bytes);
    const std::string reason = read ? std::string() : system_message();
    ::close(fd);
    if (!read)
        return {std::nullopt, path.string() + ": cannot read: " + reason};
    return {std::move(bytes), {}};
}

void append_number(std::string &text, double value) noexcept
{
    append_digits(text, value);
}

void append_number(std::string &text, std::uint64_t value) noexcept
{
    append_digits(text, value);
}
} // namespace gca
