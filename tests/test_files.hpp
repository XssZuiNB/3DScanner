// SPDX-License-Identifier: 0BSD
/// @file test_files.hpp
/// The offline tests' helpers for files: a temporary directory that the test removes with
/// everything in it, a file's bytes written or read as they are, a comparison of two numbers
/// bit for bit, and a limit on the size of the files the test program writes, which stands for
/// a full disk. Header only; each test program is one translation unit.

#pragma once

#include "check.hpp"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>

#include <stdlib.h>
#include <sys/resource.h>

namespace gca_test
{
/// A directory made for a test program, removed with everything in it at its end.
class temporary_directory
{
public:
    /// Makes a new directory in the system's temporary directory; ends the program when it
    /// cannot.
    /// @param program [in] the test program's name, the start of the directory's name.
    explicit temporary_directory(const std::string &program)
    {
        std::string pattern =
            (std::filesystem::temp_directory_path() / (program + ".XXXXXX")).string();
        if (::mkdtemp(pattern.data()) == nullptr)
        {
            std::perror((program + ": mkdtemp").c_str());
            std::exit(1);
        }
        m_path = pattern;
    }

    temporary_directory(const temporary_directory &) = delete;
    temporary_directory &operator=(const temporary_directory &) = delete;

    /// Removes the directory and everything in it.
    ~temporary_directory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(m_path, ignored);
    }

    /// Gives a path in the directory.
    /// @param name [in] the file's name.
    /// @return the path.
    std::filesystem::path operator/(const std::string &name) const
    {
        return m_path / name;
    }

    /// Gives the directory's own path.
    /// @return the path, valid as long as the directory.
    const std::filesystem::path &path() const noexcept
    {
        return m_path;
    }

private:
    std::filesystem::path m_path;
};

/// Writes a file as the test gives it, for the cases written by hand.
/// @param path [in] the file's path.
/// @param bytes [in] the content.
inline void write_bytes(const std::filesystem::path &path, const std::string &bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    check(static_cast<bool>(file), ("test setup: wrote " + path.string()).c_str());
}

/// Reads a file's bytes as they are on disk.
/// @param path [in] the file's path.
/// @return the content; empty when the file cannot be read.
inline std::string bytes_of(const std::filesystem::path &path)
{
    std::ifstream file(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

/// Whether two floats have the same bits, so that -0 and 0 differ and NaN equals itself.
/// @param a [in] one float.
/// @param b [in] the other.
/// @return whether their bits are equal.
inline bool same_bits(float a, float b)
{
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}

/// Whether two doubles have the same bits.
/// @param a [in] one double.
/// @param b [in] the other.
/// @return whether their bits are equal.
inline bool same_bits(double a, double b)
{
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}

/// A limit on the size of every file the test program writes, all its threads included, for as
/// long as it lives: a write past it fails with EFBIG ("File too large"), as a full disk fails
/// a write with ENOSPC. The kernel would end the program with SIGXFSZ instead, so the signal is
/// ignored meanwhile.
class file_size_limit
{
public:
    /// Sets the limit.
    /// @param bytes [in] the largest size a file may reach, in bytes.
    explicit file_size_limit(rlim_t bytes)
    {
        check(::getrlimit(RLIMIT_FSIZE, &m_saved) == 0, "test setup: getrlimit");
        rlimit limit = m_saved;
        limit.rlim_cur = bytes;
        m_previous = std::signal(SIGXFSZ, SIG_IGN);
        check(::setrlimit(RLIMIT_FSIZE, &limit) == 0, "test setup: setrlimit");
    }

    file_size_limit(const file_size_limit &) = delete;
    file_size_limit &operator=(const file_size_limit &) = delete;

    /// Restores the limit and the signal's handling as they were.
    ~file_size_limit()
    {
        check(::setrlimit(RLIMIT_FSIZE, &m_saved) == 0, "test setup: setrlimit restored");
        std::signal(SIGXFSZ, m_previous);
    }

private:
    rlimit m_saved{};
    void (*m_previous)(int) = SIG_DFL;
};
} // namespace gca_test
