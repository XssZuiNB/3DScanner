// SPDX-License-Identifier: 0BSD
/// @file se3.cpp
/// The SE(3) exponential (see se3.hpp; the method: docs/methods/registration.md, "The update: the
/// SE(3) exponential").

#include "registration/se3.hpp"

#include <cmath>

namespace gca
{
namespace
{
/// The angle below which the exponential's coefficients are their series: there the series' next
/// terms, below θ⁴ / 120 = 8.3e-19, lie under double's rounding of 1, and the closed forms would
/// divide by a θ³ near 0 [S: computed; docs/methods/registration.md, "The update"].
constexpr double series_below_rad = 1e-4;
} // namespace

Eigen::Matrix3d cross_matrix(const Eigen::Vector3d &w) noexcept
{
    Eigen::Matrix3d m;
    m << 0.0, -w.z(), w.y(), w.z(), 0.0, -w.x(), -w.y(), w.x(), 0.0;
    return m;
}

Eigen::Isometry3d se3_exp(const twist &xi) noexcept
{
    const Eigen::Vector3d omega = xi.head<3>();
    const Eigen::Vector3d upsilon = xi.tail<3>();
    const double theta2 = omega.squaredNorm();
    const double theta = std::sqrt(theta2);
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    if (theta < series_below_rad)
    {
        a = 1.0 - theta2 / 6.0;
        b = 0.5 - theta2 / 24.0;
        c = 1.0 / 6.0 - theta2 / 120.0;
    }
    else
    {
        const double sine = std::sin(theta);
        a = sine / theta;
        b = (1.0 - std::cos(theta)) / theta2;
        c = (theta - sine) / (theta2 * theta);
    }
    const Eigen::Matrix3d w = cross_matrix(omega);
    const Eigen::Matrix3d w2 = w * w;
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.linear() = Eigen::Matrix3d::Identity() + a * w + b * w2;
    transform.translation() = (Eigen::Matrix3d::Identity() + b * w + c * w2) * upsilon;
    return transform;
}
} // namespace gca
