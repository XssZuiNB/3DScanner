// SPDX-License-Identifier: 0BSD
/// @file normal_equations.hpp
/// One iteration's normal equations of the registration, on the host in double: the system and
/// its sums as the GPU reads them back, about the centre its kernel used (their positions below);
/// the fitness and the inlier RMSE; the shift to the correspondences' centroid, the rotation's
/// scale and the normalization; the degeneracy test on the eigenvalues; the damped step as a rigid
/// transform; and whether that step ends the iterations. The constants that judge a system stand
/// here, each [?: initial]. The method: docs/methods/registration.md, "The system,
/// centred and scaled", "The degeneracy test and the damped solve", "The iterations and the end".

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cstddef>

namespace gca
{
/// The positions of the 35 numbers one iteration reads back, the system and its sums over its
/// correspondences, each point s and centre c in the surface's frame, in m (docs/methods/
/// registration.md, "The system, centred and scaled"). The kernel writes them in this order. The
/// system holds both terms of colored ICP, each residual with its weight w, λ w_G for the
/// point-to-plane term and (1 - λ) w_C for the color term; the weight sums, the offset and the
/// spread are the point-to-plane term's alone, its weights λ w_G.
inline constexpr std::size_t sum_hessian = 0; ///< 21: H = Σ w J J^T, its upper triangle row by row
inline constexpr std::size_t sum_gradient = 21;         ///< 6: g = Σ w J r
inline constexpr std::size_t sum_weight = 27;           ///< W = Σ λ w_G
inline constexpr std::size_t sum_offset = 28;           ///< 3: Σ λ w_G (s - c), in m
inline constexpr std::size_t sum_spread = 31;           ///< Σ λ w_G |s - c|², in m²
inline constexpr std::size_t sum_squared_residual = 32; ///< Σ r_G², unweighted, in m²
inline constexpr std::size_t sum_correspondences = 33;  ///< the correspondences' count
inline constexpr std::size_t sum_candidates = 34; ///< the frame's pixels with a point and a normal
inline constexpr std::size_t sum_count = 35;      ///< the numbers read back

/// The system and its sums of one iteration, at the positions above; J = [(s - c) x n; n] for a
/// correspondence's point-to-plane residual r_G of point s and normal n, J = [(s - c) x d_q; d_q]
/// for its color residual r_C of gradient d_q, in the surface's frame.
using normal_sums = std::array<double, sum_count>;

/// f_min, the least fitness of a system that is solved: 0.1 [?: initial; docs/methods/
/// registration.md, "Parameters"].
inline constexpr double registration_min_fitness = 0.1;

/// τ, the least eigenvalue of the centred, scaled and normalized system that is not degenerate:
/// along its weakest motion 0.01 of what the point-to-plane term holds per correspondence, the
/// system being normalized by that term's weights [?: initial; the planes' and the test scenes'
/// smallest eigenvalues stand in docs/methods/registration.md, "Accuracy"].
inline constexpr double registration_degeneracy_threshold = 0.01;

/// μ, the damping added to the normalized system's diagonal: τ / 10, so that a step along an
/// eigenvalue of at least τ keeps at least 0.91 of the Gauss-Newton step [?: initial; S: computed,
/// τ / (τ + μ)].
inline constexpr double registration_damping = registration_degeneracy_threshold / 10.0;

/// ε_ω, the turn below which a step ends the iterations, in rad: 1e-5, 10 µm at 1 m
/// [?: initial].
inline constexpr double registration_converged_rotation_rad = 1e-5;

/// ε_υ, the move of the correspondences' centroid below which a step ends the iterations,
/// in m: 10 µm, a hundredth of a depth unit of 1 mm [?: initial].
inline constexpr double registration_converged_translation_m = 1e-5;

/// What one iteration's system gives.
enum class system_verdict
{
    few_correspondences, ///< no correspondence, or a fitness below registration_min_fitness
    unsolvable,          ///< no spread (ρ² <= 0), or a system or step that is not a number
    step,                ///< a damped step, after which the iterations go on
    last_step,           ///< a damped step below ε_ω and ε_υ: the iterations end with it
};

/// One iteration's system solved: its verdict, its fitness, RMSE and conditioning, and its step.
struct system_solution
{
    system_verdict verdict = system_verdict::few_correspondences; ///< what the system gives
    double fitness = 0.0;       ///< correspondences / candidates, 0 to 1; 0 with no candidate
    double inlier_rmse_m = 0.0; ///< sqrt(Σ r_G² / correspondences), in m; 0 with none
    /// The eigenvalues of the centred, scaled and normalized system Â, ascending: the
    /// point-to-plane term's part holds each within 0 to 2 up to rounding, and the color term only
    /// adds; zeros when the verdict is few_correspondences or the spread is none.
    std::array<double, 6> eigenvalues{};
    /// Whether the system is degenerate: its smallest eigenvalue below τ, or not a number. A step
    /// is taken all the same, damped; the registration judges its last system by it.
    bool degenerate = true;
    /// The correspondences' centroid c' in the surface's frame, in m, the next iteration's centre;
    /// the centre the sums were taken about when there is no correspondence.
    Eigen::Vector3d centroid_m = Eigen::Vector3d::Zero();
    /// The step ΔT = Tr(c') exp((ω, υ_c)^) Tr(-c'), applied on the left of the transform, in the
    /// surface's frame; the identity unless the verdict is step or last_step.
    Eigen::Isometry3d step = Eigen::Isometry3d::Identity();
};

/// Solves one iteration's normal equations (docs/methods/registration.md): the fitness and the
/// RMSE; with too few correspondences nothing more; else the shift of the system from the centre to
/// the centroid c', the rotation scaled by the spread ρ and the whole divided by W; unsolvable when
/// ρ² <= 0; its eigenvalues, and whether the smallest lies below τ; the damped step
/// -(Â + μ I)^-1 b̂ by the eigenvectors, as a twist about c', which ends the iterations when it is
/// below ε_ω and ε_υ.
/// @param sums [in] the system and its sums, at the positions above, about centre_m; read back
///        from the GPU.
/// @param centre_m [in] the centre c the kernel took the sums about, in the surface's frame, in m,
///        finite.
/// @return the verdict, the fitness, RMSE, eigenvalues and degeneracy, the centroid and the step. A
///         sum that is not a number makes the verdict few_correspondences or unsolvable, never a
///         step.
/// @note Allocates nothing; host only.
system_solution solve_normal_equations(const normal_sums &sums,
                                       const Eigen::Vector3d &centre_m) noexcept;
} // namespace gca
