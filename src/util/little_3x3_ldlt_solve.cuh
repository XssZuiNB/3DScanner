// SPDX-License-Identifier: 0BSD
/// @file little_3x3_ldlt_solve.cuh
/// The solution of a small symmetric positive definite 3x3 system in device code, by an LDL^T
/// factorization without pivoting.

#pragma once

#include <Eigen/Core>

#include <cuda_runtime.h>

namespace gca
{
/// Solves a x = b by a = L D L^T, L unit lower triangular and D diagonal, without pivoting.
/// Reads only a's lower triangle.
/// @param a the matrix, symmetric positive definite: an entry of D that is zero or tiny makes x
///     infinite or inaccurate, since nothing pivots around it.
/// @param b the right-hand side.
/// @return x.
__device__ __forceinline__ Eigen::Vector3f solve_ldlt_3x3(const Eigen::Matrix3f &a,
                                                          const Eigen::Vector3f &b)
{
    const float d0 = a(0, 0);
    const float l10 = a(1, 0) / d0;
    const float l20 = a(2, 0) / d0;
    const float d1 = a(1, 1) - l10 * l10 * d0;
    const float l21 = (a(2, 1) - l10 * l20 * d0) / d1;
    const float d2 = a(2, 2) - l20 * l20 * d0 - l21 * l21 * d1;

    // L y = b, then D z = y, then L^T x = z, all in x.
    Eigen::Vector3f x;
    x(0) = b(0);
    x(1) = b(1) - l10 * x(0);
    x(2) = b(2) - l20 * x(0) - l21 * x(1);
    x(0) /= d0;
    x(1) /= d1;
    x(2) /= d2;
    x(1) -= l21 * x(2);
    x(0) -= l10 * x(1) + l20 * x(2);
    return x;
}
} // namespace gca
