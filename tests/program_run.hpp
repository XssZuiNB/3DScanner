// SPDX-License-Identifier: 0BSD
/// @file program_run.hpp
/// The offline tests' run of a program through the shell, for the tests of a program's exits: its
/// exit status and what it printed, stdout and stderr together. Header only; each test program is
/// one translation unit.

#pragma once

#include <cstddef>
#include <cstdio>
#include <string>

#include <sys/wait.h>

namespace gca_test
{
/// What a run of the program gave.
struct program_run
{
    int exit_status;    ///< its exit status, or -1 when it did not exit
    std::string output; ///< what it printed, stdout and stderr together
};

/// Runs the program on one argument through the shell.
/// @param program [in] the program's path, without a single quote.
/// @param argument [in] its argument, without a single quote.
/// @return its exit status and what it printed.
inline program_run run_program(const std::string &program, const std::string &argument)
{
    const std::string command = "'" + program + "' '" + argument + "' 2>&1";
    FILE *pipe = ::popen(command.c_str(), "r");
    if (pipe == nullptr)
        return {-1, "popen failed"};
    std::string output;
    char buffer[4096];
    for (std::size_t n; (n = std::fread(buffer, 1, sizeof(buffer), pipe)) > 0;)
        output.append(buffer, n);
    const int status = ::pclose(pipe);
    return {WIFEXITED(status) ? WEXITSTATUS(status) : -1, output};
}
} // namespace gca_test
