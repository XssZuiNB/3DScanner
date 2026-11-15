// SPDX-License-Identifier: 0BSD
/// @file pose_graph.hpp
/// Refinement's pose graph and its solve, on the host in double: the keyframes' poses at its
/// nodes, the measured transforms between keyframes with their information at its edges, and the
/// damped Gauss-Newton on SE(3) with Huber's weight that brings the poses to the edges, node 0
/// held fixed. The result is a value for every call: how the solve ended, the poses, each edge's
/// last weight and the iterations. The constants stand here, each [?: initial]. The method, its
/// sources and its measurements: docs/methods/refinement.md.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gca
{
/// k, Huber's threshold on an edge's normalized residual s = sqrt(r^T Ω r), in standard
/// deviations: 3.548, the square root of 12.592, the 0.95 quantile of χ² with 6 degrees of
/// freedom, so that an edge that is right keeps the weight 1 in 95 of 100 cases [?: initial; S:
/// computed; docs/methods/refinement.md, "The robust weight"].
inline constexpr double pose_graph_huber_threshold = 3.548;

/// μ_0, the damping's start, a share of the system's diagonal: 1e-4, near the Gauss-Newton step
/// [?: initial; docs/methods/refinement.md, "The damped Gauss-Newton"].
inline constexpr double pose_graph_initial_damping = 1e-4;

/// ε_ω, the turn of every node below which a step ends the solve, in rad: 1e-6 [?: initial].
inline constexpr double pose_graph_converged_rotation_rad = 1e-6;

/// ε_υ, the move of every node's camera centre below which a step ends the solve, in m: 1e-6, a
/// thousandth of a depth unit of 1 mm [?: initial].
inline constexpr double pose_graph_converged_translation_m = 1e-6;

/// The most iterations a solve may be given: 1000 [?: a choice].
inline constexpr std::uint32_t pose_graph_largest_max_iterations = 1000;

/// An edge of the pose graph: a measured transform between two keyframes and its information
/// (docs/methods/refinement.md, "The graph"). Where the poses T and the edge disagree,
/// T_target^-1 T_source = exp(ε^) transform, ε = (ω, υ) the edge's error, a twist in the target's
/// camera frame.
struct pose_graph_edge
{
    /// i, the node whose camera frame the transform carries into: below the graph's node count.
    std::size_t target;
    /// j, the node whose camera frame the transform carries from: below the graph's node count,
    /// not target.
    std::size_t source;
    /// Z_ij, from the source keyframe's camera frame into the target's: its linear part a rotation
    /// within 1e-6 (is_rigid(), registration/se3.hpp), its translation in m; finite.
    Eigen::Isometry3d transform;
    /// Ω_ij, the inverse of ε's covariance, rotation first: the turns' block in rad^-2, the shifts'
    /// in m^-2, the blocks between them in rad^-1 m^-1; finite, symmetric within 1e-9 of its
    /// largest entry [?: a choice, far above the rounding of a product such as A^T H A, far below a
    /// mistake], positive definite.
    Eigen::Matrix<double, 6, 6> information;
};

/// A pose graph: a node for each keyframe with its pose, and the edges between them.
struct pose_graph
{
    /// T_k, keyframe k's pose: from its camera frame (x right, y down, z forward, in m) into the
    /// models' frame; finite and rigid. The start of the solve; node 0 is held fixed at its pose.
    std::vector<Eigen::Isometry3d> poses;
    std::vector<pose_graph_edge> edges; ///< the edges, each between two of the nodes
};

/// The solve's parameters (docs/methods/refinement.md, "Constants and parameters").
struct pose_graph_parameters
{
    /// The most iterations, each one factorization and its step, 1 to
    /// pose_graph_largest_max_iterations.
    std::uint32_t max_iterations;
};

/// How a solve ended (docs/methods/refinement.md, "The end"); only converged gives the keyframes'
/// optimized poses.
enum class pose_graph_end
{
    /// A step under which every node turned by less than ε_ω and moved by less than ε_υ, and the
    /// Gauss-Newton step without damping at the poses it ended with under them too [?: the
    /// orchestrator's choice, 2026-11-15, option (a) over a refusal of Ω's span].
    converged,
    iterations_used_up, ///< max_iterations steps without such a pair of steps
    /// The graph has no edge, or a node no chain of edges joins to node 0; nothing was solved.
    disconnected,
    refused, ///< the inputs, before anything was solved; the message says why
};

/// A solve's result, a value for every call (docs/methods/refinement.md, "The end").
struct pose_graph_result
{
    pose_graph_end end = pose_graph_end::refused; ///< how it ended
    /// Why it was refused or is disconnected, in plain English; empty otherwise.
    std::string message;
    /// The poses, one for each node, in the graph's order and frames, node 0's as given: the
    /// keyframes' optimized poses when converged, the poses after the last accepted step when the
    /// iterations were used up; empty when disconnected or refused.
    std::vector<Eigen::Isometry3d> poses;
    /// Each edge's Huber weight at those poses, in the graph's order: 1 for an edge within k
    /// standard deviations, k / s below 1 for one weighed down; empty when poses is.
    std::vector<double> edge_weights;
    std::uint32_t iterations = 0; ///< the factorizations, each with its step; 0 when not solved

    /// Says whether the solve converged, the one end whose poses are the keyframes' optimized
    /// poses.
    /// @return end == converged.
    bool converged() const noexcept
    {
        return end == pose_graph_end::converged;
    }
};

/// Optimizes a pose graph (docs/methods/refinement.md): checks the inputs and that the edges join
/// every node to node 0, then iterates Levenberg-Marquardt on SE(3): each edge's residual, Huber's
/// weight and exact Jacobians at the poses, the sparse system in double without node 0, solved by
/// a sparse Cholesky decomposition with Marquardt's damping, each node moved in its own camera
/// frame by the SE(3) exponential when the step lowers the robust objective, until a step and the
/// undamped Gauss-Newton step at the poses it ends with are both small enough, or max_iterations
/// are used up.
/// @param graph [in] the graph: at least one node; every pose finite and rigid; every edge between
///        two distinct nodes of the graph, its transform finite and rigid, its information finite,
///        symmetric and positive definite.
/// @param parameters [in] the most iterations, 1 to pose_graph_largest_max_iterations.
/// @return the result (pose_graph_result): converged, with the poses and the edges' weights; the
///         iterations used up, with the poses after the last accepted step and their weights, not
///         optimized poses; disconnected, with the reason and no poses, when the graph has no edge
///         or a node no chain of edges joins to node 0; or refused, with the reason and no poses,
///         when an input breaks one of the conditions above.
/// @note Host only; allocates for the system, its factorization, the poses and the message, and
///       throws std::bad_alloc when an allocation fails. Holds no state: two threads may solve two
///       graphs at once.
pose_graph_result optimize_pose_graph(const pose_graph &graph,
                                      const pose_graph_parameters &parameters);
} // namespace gca
