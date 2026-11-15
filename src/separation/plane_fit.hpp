// SPDX-License-Identifier: 0BSD
/// @file plane_fit.hpp
/// The orthogonal least-squares plane through a set of points, the one copy of it (AGENTS.md §7
/// rule 4): through the points' centroid, its normal the eigenvector of the least eigenvalue of
/// their centred covariance, summed in double [S: K. Pearson, "On Lines and Planes of Closest Fit
/// to Systems of Points in Space", Philosophical Magazine 2(11), 1901]. Separation fits the
/// support plane with it (docs/methods/separation.md), distortion_check a checkerboard's corners
/// (apps/checkerboard_measure.cpp). Host only.

#pragma once

#include <optional>
#include <span>

#include <Eigen/Core>

namespace gca
{
/// The plane fitted to a set of points, in the points' frame and unit.
struct plane_fit
{
    Eigen::Vector3d centroid; ///< the points' mean, a point of the plane
    /// The plane's unit normal: the eigenvector of the centred covariance's least eigenvalue, of
    /// either sign.
    Eigen::Vector3d normal;
    /// The centred covariance's eigenvalues, the sums over the points of the squared distances
    /// from the centroid along each eigenvector, in increasing order, in the points' unit squared:
    /// the first the sum of squared distances from the plane, the last along the points' main line.
    Eigen::Vector3d eigenvalues;
};

/// Fits the orthogonal least-squares plane to a set of points: the centroid, then the centred
/// covariance Σ (p - centroid)(p - centroid)^T, both summed in double in the points' order, and its
/// eigenvectors by Eigen's self-adjoint eigensolver.
/// @param points [in] the points, in any frame and unit, finite, else the fit's values are NaN or
///        infinite; host memory, read during the call.
/// @return the fit; nothing when fewer than three points are given. Points on a line give a fit
///         whose second eigenvalue is 0 and whose normal is any direction across the line: the
///         caller judges the spread by the eigenvalues.
/// @note Host only; allocates nothing. Holds no state: two threads may fit at once.
std::optional<plane_fit> fit_plane(std::span<const Eigen::Vector3d> points) noexcept;
} // namespace gca
