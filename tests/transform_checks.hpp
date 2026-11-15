// SPDX-License-Identifier: 0BSD
/// @file transform_checks.hpp
/// The offline tests' rigid transforms in double: one built from an axis, an angle and a shift,
/// how far a transform lies from a known one, the tolerance of that, and the check that prints it.
/// Header only; each test program is one translation unit.

#pragma once

#include "check.hpp"

#include <cmath>
#include <numbers>
#include <string>

#include <Eigen/Geometry>

namespace gca_test
{
/// Builds a rigid transform: a turn about an axis through the origin, then a shift.
/// @param axis [in] the axis, not zero; normalized here.
/// @param angle_deg [in] the angle, in degrees.
/// @param shift_m [in] the shift, in m.
/// @return the transform p -> R p + shift.
inline Eigen::Isometry3d make_transform(const Eigen::Vector3d &axis, double angle_deg,
                                        const Eigen::Vector3d &shift_m)
{
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.linear() =
        Eigen::AngleAxisd(angle_deg * std::numbers::pi / 180.0, axis.normalized()).matrix();
    transform.translation() = shift_m;
    return transform;
}

/// How far a transform lies from a known one.
struct transform_error
{
    double rotation_deg;  ///< the angle of the turn between them, in degrees
    double translation_m; ///< the length of the shift between them at the camera's centre, in m
};

/// Measures how far a transform lies from a known one: the turn and the shift of known^-1 found.
/// @param found [in] the transform found.
/// @param known [in] the known transform.
/// @return the angle of the turn, in degrees, and the length of the shift, in m.
inline transform_error error_of(const Eigen::Isometry3d &found, const Eigen::Isometry3d &known)
{
    const Eigen::Isometry3d e = known.inverse() * found;
    const Eigen::AngleAxisd turn(e.linear());
    return {std::abs(turn.angle()) * 180.0 / std::numbers::pi, e.translation().norm()};
}

/// Tolerances of a recovered transform.
struct transform_tolerance
{
    double rotation_deg;  ///< the largest turn from the known transform, in degrees
    double translation_m; ///< the largest shift from it at the camera's centre, in m
};

/// Checks that a transform lies within a tolerance of a known one, and prints how far it lies.
/// @param what [in] the case.
/// @param found [in] the transform found.
/// @param known [in] the known transform.
/// @param tolerance [in] how far it may lie.
inline void check_near(const std::string &what, const Eigen::Isometry3d &found,
                       const Eigen::Isometry3d &known, const transform_tolerance &tolerance)
{
    const transform_error error = error_of(found, known);
    check_within((what + ": rotation error (deg)").c_str(), error.rotation_deg,
                 tolerance.rotation_deg);
    check_within((what + ": translation error (m)").c_str(), error.translation_m,
                 tolerance.translation_m);
}
} // namespace gca_test
