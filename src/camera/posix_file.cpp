// SPDX-License-Identifier: 0BSD
/// @file posix_file.cpp
/// The system's message for errno, a write of all bytes, a read up to a size and a directory's
/// sync, with POSIX calls (see posix_file.hpp).

#include "camera/posix_file.hpp"

#include <cerrno>
#include <cstddef>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>

namespace gca
{
std::string system_message() noexcept
{
    char buffer[256];
    return ::strerror_r(errno, buffer, sizeof(buffer));
}

bool write_all(int fd, std::string_view bytes) noexcept
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

std::optional<std::size_t> read_up_to(int fd, void *buffer, std::size_t size) noexcept
{
    std::size_t done = 0;
    while (done < size)
    {
        const ssize_t n = ::read(fd, static_cast<char *>(buffer) + done, size - done);
        if (n < 0 && errno == EINTR)
            continue;
        if (n < 0)
            return std::nullopt;
        if (n == 0)
            break;
        done += static_cast<std::size_t>(n);
    }
    return done;
}

bool sync_directory(const std::filesystem::path &directory) noexcept
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
} // namespace gca
