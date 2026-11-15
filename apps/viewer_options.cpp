// SPDX-License-Identifier: 0BSD
/// @file viewer_options.cpp
/// orbbec_viewer's command line: one table of the options, from which the reader and the usage
/// line both come, and the check of the sequence's directory (see viewer_options.hpp).

#include "viewer_options.hpp"

#include "camera/posix_file.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gca_viewer
{
namespace
{
/// One option of the command line.
struct option_spec
{
    const char *name;       ///< the option, "--frames"
    const char *value_name; ///< its value's name in the usage line; nullptr for a flag
    /// Sets the option's field; the value is nullptr for a flag, else a number read with
    /// std::atoi or a path taken as it is.
    void (*set)(options &opt, const char *value);
};

/// The options, in the usage line's order.
const option_spec option_table[] = {
    {"--list", nullptr, [](options &opt, const char *) { opt.list_only = true; }},
    {"--headless", nullptr, [](options &opt, const char *) { opt.headless = true; }},
    {"--frames", "N", [](options &opt, const char *value) { opt.max_frames = std::atoi(value); }},
    {"--width", "W", [](options &opt, const char *value) { opt.width = std::atoi(value); }},
    {"--height", "H", [](options &opt, const char *value) { opt.height = std::atoi(value); }},
    {"--fps", "F", [](options &opt, const char *value) { opt.fps = std::atoi(value); }},
    {"--sequence", "DIR", [](options &opt, const char *value) { opt.sequence_directory = value; }},
    {"--hold-color", nullptr, [](options &opt, const char *) { opt.hold_color = true; }},
};

/// Finds an option by its name.
/// @param name the argument.
/// @return the option, or nullptr when the viewer has none of that name.
const option_spec *find_option(const std::string &name)
{
    for (const option_spec &spec : option_table)
    {
        if (name == spec.name)
            return &spec;
    }
    return nullptr;
}

/// Says why the viewer cannot write its sequence into a directory: it does not exist or is no
/// directory, it cannot be listed, it holds something already, or it cannot be written. Writing
/// stands for creating a file in it (access() with W_OK and X_OK), so a directory on a read-only
/// mount is refused too. It looks only and changes nothing on disk.
/// @param directory [in] the directory --sequence names.
/// @return the reason, without the option's name; empty when the viewer can write there.
std::string directory_refusal(const std::filesystem::path &directory)
{
    struct stat status
    {
    };
    if (::stat(directory.c_str(), &status) != 0)
        return errno == ENOENT ? "no such directory" : "cannot be read: " + gca::system_message();
    if (!S_ISDIR(status.st_mode))
        return "is no directory";
    DIR *listing = ::opendir(directory.c_str());
    if (listing == nullptr)
        return "cannot be read: " + gca::system_message();
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
        return "cannot be read: " + gca::system_message();
    }
    if (holds_something)
        return "holds something already";
    if (::access(directory.c_str(), W_OK | X_OK) != 0)
        return "cannot be written: " + gca::system_message();
    return {};
}

/// Prints a refusal: the reason, then the usage line.
/// @param err where to print.
/// @param reason why the command line is refused.
/// @return nothing, what read_options() returns for a refused command line.
std::optional<options> refuse(std::ostream &err, const std::string &reason)
{
    err << "orbbec_viewer: " << reason << "\n" << usage_line() << std::endl;
    return std::nullopt;
}
} // namespace

std::string usage_line()
{
    std::string line = "usage: orbbec_viewer";
    for (const option_spec &spec : option_table)
    {
        line += std::string(" [") + spec.name;
        if (spec.value_name != nullptr)
            line += std::string(" ") + spec.value_name;
        line += "]";
    }
    return line;
}

std::optional<options> read_options(int argc, const char *const argv[], bool headless_default,
                                    std::ostream &err)
{
    options opt;
    opt.headless = headless_default;
    for (int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        const option_spec *spec = find_option(a);
        if (spec == nullptr)
            return refuse(err, "unknown argument " + a);
        const bool takes_value = spec->value_name != nullptr;
        if (takes_value && i + 1 >= argc)
            return refuse(err, a + " needs a value");
        spec->set(opt, takes_value ? argv[++i] : nullptr);
    }
    if (opt.sequence_directory)
    {
        const std::string refusal = directory_refusal(*opt.sequence_directory);
        if (!refusal.empty())
            return refuse(err, "--sequence " + opt.sequence_directory->string() + ": " + refusal);
    }
    return opt;
}
} // namespace gca_viewer
