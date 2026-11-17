// SPDX-License-Identifier: 0BSD
/// @file output_directory.cpp
/// The checks of the directory a program writes into (see output_directory.hpp).

#include "output_directory.hpp"

#include "camera/posix_file.hpp"

#include <cerrno>
#include <cstring>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gca
{
namespace
{
/// Says whether a directory holds something, an entry other than "." and "..".
/// @param directory [in] the directory, which exists and is one.
/// @param refusal [out] "cannot be read: <message>" when it cannot be listed; left as it was
///        otherwise.
/// @return whether it holds something; false when it cannot be listed.
bool holds_something(const std::filesystem::path &directory, std::string &refusal)
{
    DIR *listing = ::opendir(directory.c_str());
    if (listing == nullptr)
    {
        refusal = "cannot be read: " + system_message();
        return false;
    }
    bool found = false;
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
            found = true;
            break;
        }
    }
    ::closedir(listing);
    if (read_error != 0)
    {
        errno = read_error;
        refusal = "cannot be read: " + system_message();
    }
    return found;
}

/// The checks both kinds of directory share, in order: it exists and is a directory, it holds
/// nothing when it must, and it can be written.
/// @param directory [in] the directory the command line names.
/// @param must_be_empty [in] whether it must hold nothing.
/// @return the reason, as output_directory_refusal() and writable_directory_refusal() give it;
///         empty when the program can write there.
std::string directory_refusal(const std::filesystem::path &directory, bool must_be_empty)
{
    struct stat status
    {
    };
    if (::stat(directory.c_str(), &status) != 0)
        return errno == ENOENT ? "no such directory" : "cannot be read: " + system_message();
    if (!S_ISDIR(status.st_mode))
        return "is no directory";
    if (must_be_empty)
    {
        std::string refusal;
        const bool found = holds_something(directory, refusal);
        if (!refusal.empty())
            return refusal;
        if (found)
            return "holds something already";
    }
    if (::access(directory.c_str(), W_OK | X_OK) != 0)
        return "cannot be written: " + system_message();
    return {};
}
} // namespace

std::string output_directory_refusal(const std::filesystem::path &directory)
{
    return directory_refusal(directory, true);
}

std::string writable_directory_refusal(const std::filesystem::path &directory)
{
    return directory_refusal(directory, false);
}
} // namespace gca
