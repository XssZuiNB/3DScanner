// SPDX-License-Identifier: 0BSD
/// @file plane_fit.cpp
/// The orthogonal least-squares plane through a set of points (see plane_fit.hpp).

#include "separation/plane_fit.hpp"

#include <Eigen/Eigenvalues>

namespace gca
{
std::optional<plane_fit> fit_plane(std::span<const Eigen::Vector3d> points) noexcept
{
    if (points.size() < 3)
        return std::nullopt;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d &p : points)
        centroid += p;
    centroid /= static_cast<double>(points.size());
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const Eigen::Vector3d &p : points)
        covariance += (p - centroid) * (p - centroid).transpose();
    // Eigen's eigenvalues come in increasing order: column 0 is the plane's normal.
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    return plane_fit{centroid, solver.eigenvectors().col(0), solver.eigenvalues()};
}
} // namespace gca
