// SPDX-License-Identifier: 0BSD
/// @file output_directory.cpp
/// The check of the directory a program writes into (see output_directory.hpp).

#include "output_directory.hpp"

#include "camera/posix_file.hpp"

#include <cerrno>
#include <cstring>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gca
{
std::string output_directory_refusal(const std::filesystem::path &directory)
{
    struct stat status
    {
    };
    if (::stat(directory.c_str(), &status) != 0)
        return errno == ENOENT ? "no such directory" : "cannot be read: " + system_message();
    if (!S_ISDIR(status.st_mode))
        return "is no directory";
    DIR *listing = ::opendir(directory.c_str());
    if (listing == nullptr)
        return "cannot be read: " + system_message();
    bool holds_something = false;
    int read_error = 0;
    for (;;)
    {
        errno = 0;
        const dirent *entry = ::readdir(listing);
        if (entry == nullptr)
        {
            read_error = errno;
            break;
        }
        if (std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0)
        {
            holds_something = true;
            break;
        }
    }
    ::closedir(listing);
    if (read_error != 0)
    {
        errno = read_error;
        return "cannot be read: " + system_message();
    }
    if (holds_something)
        return "holds something already";
    if (::access(directory.c_str(), W_OK | X_OK) != 0)
        return "cannot be written: " + system_message();
    return {};
}
} // namespace gca
