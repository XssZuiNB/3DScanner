// SPDX-License-Identifier: 0BSD
/// @file file_io.cpp
/// A whole file written beside its name and renamed onto it, and a whole file read, with POSIX
/// calls (camera/posix_file.hpp), so that each failure comes back with the system's message; and
/// the shortest exact form of a number (see file_io.hpp).

#include "scan/file_io.hpp"

#include <charconv>
#include <cstddef>
#include <cstdio>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gca
{
namespace
{
/// Reads all bytes of a file.
/// @param fd [in] the open file's descriptor, at its start.
/// @param out [out] the file's bytes, appended.
/// @return whether the file was read to its end; on false errno holds the reason.
bool read_all(int fd, std::string &out)
{
    char buffer[1 << 16];
    for (;;)
    {
        const std::optional<std::size_t> n = read_up_to(fd, buffer, sizeof(buffer));
        if (!n)
            return false;
        out.append(buffer, *n);
        if (*n < sizeof(buffer))
            return true;
    }
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
