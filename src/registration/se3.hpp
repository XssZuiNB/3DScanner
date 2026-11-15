// SPDX-License-Identifier: 0BSD
/// @file se3.hpp
/// The SE(3) exponential: the rigid transform a twist, a rotation vector and a translation,
/// generates; the matrix of a cross product it is built from; and the check that a transform is
/// rigid. It is the registration's pose update (registration/registration.hpp) and the pose graph's
/// (refinement/pose_graph.hpp), the one copy of the exponential and of the check.
/// The method and its sources: docs/methods/registration.md, "The update: the SE(3) exponential".
/// Host only, in double.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gca
{
/// A twist ξ = (ω, υ): the rotation vector ω in its first three components, in rad, and the
/// translation υ in its last three, in m, both in the frame the transform it generates acts in.
using twist = Eigen::Matrix<double, 6, 1>;

/// Returns the matrix of the cross product with a vector.
/// @param w [in] the vector, in any unit.
/// @return [w]x, with [w]x v = w x v for every v; its unit is w's.
/// @note Allocates nothing; host only.
Eigen::Matrix3d cross_matrix(const Eigen::Vector3d &w) noexcept;

/// Returns the rigid transform a twist generates, exp(ξ^) = [R, V υ; 0, 1] with
/// R = I + A [ω]x + B [ω]x², V = I + B [ω]x + C [ω]x², θ = |ω|, A = sin θ / θ,
/// B = (1 - cos θ) / θ², C = (θ - sin θ) / θ³, each by its series 1 - θ²/6, 1/2 - θ²/24,
/// 1/6 - θ²/120 below θ = 1e-4 rad (docs/methods/registration.md, "The update").
/// @param xi [in] the twist, (ω in rad, υ in m); finite. A twist that is not gives a transform
///        that is not either.
/// @return the transform: R a rotation to double's rounding, the translation V υ in m.
/// @note Allocates nothing; host only.
Eigen::Isometry3d se3_exp(const twist &xi) noexcept;

/// How far a rigid transform's linear part may lie from a rotation: 1e-6 for each entry of
/// R^T R - I [?: a choice, far above double's rounding of a product of rotations, far below a
/// mistake].
inline constexpr double rigid_rotation_tolerance = 1e-6;

/// Says whether a transform is rigid: finite, its linear part a rotation within
/// rigid_rotation_tolerance.
/// @param transform [in] the transform, in any unit of length.
/// @return whether every entry is finite, every entry of R^T R - I within rigid_rotation_tolerance
///         of 0, and det R > 0.
/// @note Allocates nothing; host only.
bool is_rigid(const Eigen::Isometry3d &transform) noexcept;
} // namespace gca
