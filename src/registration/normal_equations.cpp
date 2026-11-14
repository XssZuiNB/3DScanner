// SPDX-License-Identifier: 0BSD
/// @file normal_equations.cpp
/// One iteration's normal equations solved on the host (see normal_equations.hpp; the method:
/// docs/methods/registration.md, "The system, centred and scaled" and "The degeneracy test and the
/// damped solve").

#include "registration/normal_equations.hpp"

#include "registration/se3.hpp"

#include <cmath>

#include <Eigen/Eigenvalues>

namespace gca
{
namespace
{
using matrix6 = Eigen::Matrix<double, 6, 6>;
using vector6 = Eigen::Matrix<double, 6, 1>;

/// Unpacks the system from the sums: H from its upper triangle, row by row, and g.
/// @param sums [in] the sums, at normal_equations.hpp's positions.
/// @param h [out] H, symmetric.
/// @param g [out] g.
void unpack_system(const normal_sums &sums, matrix6 &h, vector6 &g)
{
    std::size_t k = sum_hessian;
    for (int i = 0; i < 6; i++)
    {
        for (int j = i; j < 6; j++)
        {
            h(i, j) = sums[k];
            h(j, i) = sums[k];
            k++;
        }
    }
    for (int i = 0; i < 6; i++)
        g(i) = sums[sum_gradient + static_cast<std::size_t>(i)];
}
} // namespace

system_solution solve_normal_equations(const normal_sums &sums,
                                       const Eigen::Vector3d &centre_m) noexcept
{
    system_solution solution;
    solution.centroid_m = centre_m;
    const double candidates = sums[sum_candidates];
    const double correspondences = sums[sum_correspondences];
    solution.fitness = candidates > 0.0 ? correspondences / candidates : 0.0;
    solution.inlier_rmse_m =
        correspondences > 0.0 ? std::sqrt(sums[sum_squared_residual] / correspondences) : 0.0;
    const double weight = sums[sum_weight];
    // Written so that a sum that is not a number fails each test toward no step.
    if (!(correspondences > 0.0) || !(weight > 0.0) ||
        !(solution.fitness >= registration_min_fitness))
    {
        solution.verdict = system_verdict::few_correspondences;
        return solution;
    }

    // The shift from c to the centroid c' = c + d, exactly: (s - c - d) x n = (s - c) x n - d x n,
    // and the same for the color term's d_q in place of n.
    const Eigen::Vector3d d(sums[sum_offset] / weight, sums[sum_offset + 1] / weight,
                            sums[sum_offset + 2] / weight);
    solution.centroid_m = centre_m + d;
    matrix6 h;
    vector6 g;
    unpack_system(sums, h, g);
    matrix6 shift = matrix6::Identity();
    shift.block<3, 3>(0, 3) = -cross_matrix(d);
    h = shift * h * shift.transpose();
    g = shift * g;
    const double spread2 = sums[sum_spread] / weight - d.squaredNorm();
    solution.verdict = system_verdict::unsolvable;
    if (!(spread2 > 0.0))
        return solution;

    // The rotation in units of the spread ρ, and the whole divided by W.
    const double spread = std::sqrt(spread2);
    vector6 scale;
    scale << 1.0 / spread, 1.0 / spread, 1.0 / spread, 1.0, 1.0, 1.0;
    const matrix6 a_hat = scale.asDiagonal() * h * scale.asDiagonal() / weight;
    const vector6 b_hat = scale.asDiagonal() * g / weight;
    if (!a_hat.allFinite() || !b_hat.allFinite())
        return solution;
    const Eigen::SelfAdjointEigenSolver<matrix6> eigen(a_hat);
    if (eigen.info() != Eigen::Success)
        return solution;
    const vector6 &lambda = eigen.eigenvalues();
    for (int k = 0; k < 6; k++)
        solution.eigenvalues[static_cast<std::size_t>(k)] = lambda(k);
    solution.degenerate = !(lambda(0) >= registration_degeneracy_threshold);

    // The damped step along each eigenvector: -(v . b̂) / (λ + μ).
    const matrix6 &v = eigen.eigenvectors();
    const vector6 along = (v.transpose() * b_hat).array() / (lambda.array() + registration_damping);
    const vector6 step_scaled = -(v * along);
    twist xi;
    xi.head<3>() = step_scaled.head<3>() / spread;
    xi.tail<3>() = step_scaled.tail<3>();
    if (!xi.allFinite())
        return solution;
    solution.step = Eigen::Translation3d(solution.centroid_m) * se3_exp(xi) *
                    Eigen::Translation3d(-solution.centroid_m);
    const bool small = xi.head<3>().norm() < registration_converged_rotation_rad &&
                       xi.tail<3>().norm() < registration_converged_translation_m;
    solution.verdict = small ? system_verdict::last_step : system_verdict::step;
    return solution;
}
} // namespace gca
