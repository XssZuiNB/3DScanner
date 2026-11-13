// SPDX-License-Identifier: 0BSD
/// @file check.hpp
/// The offline tests' check helpers: a failed check prints what was checked and is counted, a
/// measured value is printed with the tolerance it is checked against, and the count becomes
/// the test program's summary line and exit status. Header only; each test program is one
/// translation unit.

#pragma once

#include <cstdio>

namespace gca_test
{
/// The number of failed checks in this test program.
inline int g_failures = 0;

/// Records a failed check.
/// @param ok the condition.
/// @param what what was checked, printed on failure.
inline void check(bool ok, const char *what)
{
    if (!ok)
    {
        std::fprintf(stderr, "FAIL: %s\n", what);
        g_failures++;
    }
}

/// Prints a measured value with its tolerance and records a failed check when the value exceeds
/// the tolerance or is NaN.
/// @param what what was measured.
/// @param measured the measured value.
/// @param tolerance the largest value that passes.
inline void check_within(const char *what, double measured, double tolerance)
{
    std::printf("%s: %.3g (tolerance %.3g)\n", what, measured, tolerance);
    check(measured <= tolerance, what);
}

/// Prints the test program's summary line when every check passed.
/// @param program the test program's name, printed in the summary line.
/// @return the program's exit status: 0 when every check passed, 1 otherwise.
inline int check_summary(const char *program)
{
    if (g_failures == 0)
        std::printf("%s: all checks passed\n", program);
    return g_failures == 0 ? 0 : 1;
}
} // namespace gca_test
