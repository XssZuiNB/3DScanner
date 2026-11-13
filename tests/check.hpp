// SPDX-License-Identifier: 0BSD
/// @file check.hpp
/// The offline tests' check helpers: a failed check prints what was checked and is counted, a
/// measured value is printed with the tolerance it is checked against, a set of errors is
/// checked by its largest and by every one being finite, and the count becomes the test
/// program's summary line and exit status. Header only; each test program is one translation
/// unit.

#pragma once

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

namespace gca_test
{
/// One float step (ulp) between 1 and 2, FLT_EPSILON: the largest rounding step of a coordinate
/// below 2 m, of a unit normal's components and of a color in [0, 1] [S: float32 ulp].
constexpr double float_step = std::numeric_limits<float>::epsilon();

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

/// The largest of a set of errors, and whether every error was finite. A NaN or infinite value
/// downloaded from the GPU makes its error non-finite, which a maximum alone would drop.
struct max_error
{
    double max = 0.0;
    bool all_finite = true;

    /// Adds one error.
    /// @param error the error.
    void add(double error)
    {
        if (std::isfinite(error))
            max = std::fmax(max, error);
        else
            all_finite = false;
    }
};

/// Checks that every error was finite and that the largest is within the tolerance, and prints
/// the largest.
/// @param what what was measured.
/// @param errors the errors.
/// @param tolerance the largest error that passes.
inline void check_max_error(const std::string &what, const max_error &errors, double tolerance)
{
    check(errors.all_finite, (what + ": every value finite").c_str());
    check_within(what.c_str(), errors.max, tolerance);
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
