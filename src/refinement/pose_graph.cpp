// SPDX-License-Identifier: 0BSD
/// @file pose_graph.cpp
/// Refinement's pose graph solved on the host (see pose_graph.hpp; the method:
/// docs/methods/refinement.md).

#include "refinement/pose_graph.hpp"

#include "registration/se3.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

namespace gca
{
namespace
{
using matrix6 = Eigen::Matrix<double, 6, 6>;
using vector6 = Eigen::Matrix<double, 6, 1>;
using sparse_matrix = Eigen::SparseMatrix<double>;

/// The angle below which β of the rotation's inverse left Jacobian is its series 1/12 + θ²/720:
/// the next term, θ⁴/30240 < 3.4e-21, lies under double's rounding of 1/12, and the closed form
/// would divide by a θ² near 0 [S: computed; docs/methods/refinement.md, "The update and the
/// Jacobians"].
constexpr double series_below_rad = 1e-4;

/// How far an information matrix may lie from symmetric: each entry of Ω - Ω^T within 1e-9 of
/// Ω's largest entry [?: a choice, far above the rounding of a product such as A^T H A, far below
/// a mistake].
constexpr double symmetry_tolerance = 1e-9;

/// ν_0, the factor by which the first rejected step after an accepted one grows the damping: 2
/// [S: Nielsen 1999; docs/methods/refinement.md, "The damped Gauss-Newton"].
constexpr double nielsen_first_growth = 2.0;

/// The factor by which each further rejection in a row grows ν, so that the damping grows ever
/// faster: 2 [S: Nielsen 1999].
constexpr double nielsen_growth_of_growth = 2.0;

/// The least factor an accepted step shrinks the damping by: 1/3, at a gain ratio of 1 [S: Nielsen
/// 1999].
constexpr double nielsen_least_shrink = 1.0 / 3.0;

/// One edge linearized at the poses: its residual, its weight and its Jacobians.
struct linearized_edge
{
    vector6 residual; ///< r = (ω_E, t_E), in rad and m, in the target's camera frame
    double weight;    ///< Huber's weight w, in (0, 1]
    /// J_i = ∂r/∂δ_i, rows (ω_E in rad, t_E in m) by columns (the target's turn in rad, its shift
    /// in m): its turn-by-turn and shift-by-shift blocks without unit, the shift-by-turn block in
    /// m, the turn-by-shift block 0.
    matrix6 jacobian_target;
    /// J_j = ∂r/∂δ_j, likewise for the source's turn and shift; its turn-by-shift block 0, its
    /// shift-by-turn block in m.
    matrix6 jacobian_source;
};

/// Returns an edge's residual at two poses: r = (ω_E, t_E) of E = T_i^-1 T_j Z^-1.
/// @param target_pose [in] T_i, the target node's pose, camera to models' frame, in m.
/// @param source_pose [in] T_j, the source node's pose, likewise.
/// @param transform [in] Z, the edge's transform from the source's camera frame into the target's.
/// @return ω_E, the rotation vector of E's rotation, its angle in [0, π], in rad; t_E, E's
///         translation, in m; both in the target's camera frame.
vector6 edge_residual(const Eigen::Isometry3d &target_pose, const Eigen::Isometry3d &source_pose,
                      const Eigen::Isometry3d &transform)
{
    const Eigen::Isometry3d e = target_pose.inverse() * source_pose * transform.inverse();
    const Eigen::AngleAxisd turn(e.linear());
    vector6 r;
    r.head<3>() = turn.angle() * turn.axis();
    r.tail<3>() = e.translation();
    return r;
}

/// Returns an edge's normalized residual.
/// @param residual [in] r, in rad and m.
/// @param information [in] Ω, positive definite.
/// @return s = sqrt(r^T Ω r), in standard deviations.
double normalized_residual(const vector6 &residual, const matrix6 &information)
{
    return std::sqrt(std::max(0.0, residual.dot(information * residual)));
}

/// Returns Huber's weight of a normalized residual.
/// @param s [in] the normalized residual, >= 0.
/// @return 1 for s <= k, k / s beyond.
double huber_weight(double s)
{
    return s <= pose_graph_huber_threshold ? 1.0 : pose_graph_huber_threshold / s;
}

/// Returns Huber's loss of a normalized residual.
/// @param s [in] the normalized residual, >= 0.
/// @return s²/2 for s <= k, k s - k²/2 beyond.
double huber_loss(double s)
{
    constexpr double k = pose_graph_huber_threshold;
    return s <= k ? 0.5 * s * s : k * s - 0.5 * k * k;
}

/// Returns the inverse of the rotation's left Jacobian, which carries a small left turn a of a
/// rotation with rotation vector φ into the change of that vector, J_l^-1(φ) a.
/// @param phi [in] φ, the rotation vector, its angle in [0, π], in rad.
/// @return I - ½ [φ]x + β [φ]x², β = 1/θ² - (1 + cos θ) / (2 θ sin θ), its series below
///         series_below_rad.
Eigen::Matrix3d left_jacobian_inverse(const Eigen::Vector3d &phi)
{
    const double theta2 = phi.squaredNorm();
    const double theta = std::sqrt(theta2);
    const double beta =
        theta < series_below_rad
            ? 1.0 / 12.0 + theta2 / 720.0
            : 1.0 / theta2 - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
    const Eigen::Matrix3d w = cross_matrix(phi);
    return Eigen::Matrix3d::Identity() - 0.5 * w + beta * w * w;
}

/// Returns the adjoint of a rigid transform, which carries a twist through it.
/// @param transform [in] T = (R, t), t in m.
/// @return Ad(T) = [R 0; [t]x R  R], with T exp(δ^) T^-1 = exp((Ad(T) δ)^) for a twist δ
///         (ω first, in rad; υ last, in m).
matrix6 adjoint(const Eigen::Isometry3d &transform)
{
    const Eigen::Matrix3d r = transform.linear();
    matrix6 a = matrix6::Zero();
    a.block<3, 3>(0, 0) = r;
    a.block<3, 3>(3, 3) = r;
    a.block<3, 3>(3, 0) = cross_matrix(transform.translation()) * r;
    return a;
}

/// Linearizes an edge at the poses: its residual, Huber's weight, and its Jacobians for each node
/// moved in its own camera frame, T ← T exp(δ^) (docs/methods/refinement.md, "The update and the
/// Jacobians").
/// @param edge [in] the edge, its nodes in poses.
/// @param poses [in] the poses, camera to models' frame, in m.
/// @return r; w; J_i = -M(r) and J_j = M(r) Ad(T_i^-1 T_j), M(r) = [J_l^-1(ω_E) 0; -[t_E]x I].
linearized_edge linearize(const pose_graph_edge &edge, const std::vector<Eigen::Isometry3d> &poses)
{
    const Eigen::Isometry3d &target_pose = poses[edge.target];
    const Eigen::Isometry3d &source_pose = poses[edge.source];
    linearized_edge l;
    l.residual = edge_residual(target_pose, source_pose, edge.transform);
    l.weight = huber_weight(normalized_residual(l.residual, edge.information));
    matrix6 m = matrix6::Zero();
    m.block<3, 3>(0, 0) = left_jacobian_inverse(l.residual.head<3>());
    m.block<3, 3>(3, 0) = -cross_matrix(l.residual.tail<3>());
    m.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    l.jacobian_target = -m;
    l.jacobian_source = m * adjoint(target_pose.inverse() * source_pose);
    return l;
}

/// Returns the robust objective at the poses.
/// @param edges [in] the graph's edges, their nodes in poses.
/// @param poses [in] the poses, one for each node, camera to models' frame, in m.
/// @return F = Σ ρ(s_e) over the edges, ρ Huber's loss, >= 0.
double robust_objective(const std::vector<pose_graph_edge> &edges,
                        const std::vector<Eigen::Isometry3d> &poses)
{
    double f = 0.0;
    for (const pose_graph_edge &edge : edges)
    {
        const vector6 r = edge_residual(poses[edge.target], poses[edge.source], edge.transform);
        f += huber_loss(normalized_residual(r, edge.information));
    }
    return f;
}

/// Returns where a node's six unknowns start in the system, node 0 left out.
/// @param node [in] the node, >= 1.
/// @return 6 (node - 1): its turn's three unknowns from there, then its shift's three.
Eigen::Index first_unknown(std::size_t node)
{
    return static_cast<Eigen::Index>(6 * (node - 1));
}

/// The weighted system of one linearization, node 0 left out: node n >= 1 owns the unknowns
/// first_unknown(n) to first_unknown(n) + 5.
struct linear_system
{
    sparse_matrix h;             ///< H = Σ w J^T Ω J, symmetric, both triangles stored
    Eigen::VectorXd g;           ///< g = Σ w J^T Ω r
    Eigen::VectorXd diagonal;    ///< H's diagonal, D
    std::vector<double> weights; ///< each edge's weight, in the graph's order
};

/// Adds a 6 x 6 block to the triplets of a matrix, unless it belongs to node 0.
/// @param triplets [in,out] the triplets.
/// @param row_node [in] the node of the block's rows, below the graph's node count; 0 adds
///        nothing.
/// @param column_node [in] the node of the block's columns, likewise.
/// @param block [in] the block.
void add_block(std::vector<Eigen::Triplet<double>> &triplets, std::size_t row_node,
               std::size_t column_node, const matrix6 &block)
{
    if (row_node == 0 || column_node == 0)
        return;
    const auto row = static_cast<int>(first_unknown(row_node));
    const auto column = static_cast<int>(first_unknown(column_node));
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            triplets.emplace_back(row + i, column + j, block(i, j));
}

/// Builds the weighted system at the poses (docs/methods/refinement.md, "The damped
/// Gauss-Newton"), in double.
/// @param edges [in] the graph's edges, which join every node to node 0, their nodes in poses.
/// @param poses [in] the poses, at least two, camera to models' frame, in m.
/// @return H, g, H's diagonal and each edge's weight.
linear_system build_system(const std::vector<pose_graph_edge> &edges,
                           const std::vector<Eigen::Isometry3d> &poses)
{
    // Where a node after the last would start: the unknowns' count.
    const Eigen::Index unknowns = first_unknown(poses.size());
    linear_system system;
    system.g = Eigen::VectorXd::Zero(unknowns);
    system.weights.reserve(edges.size());
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(4 * 36 * edges.size());
    for (const pose_graph_edge &edge : edges)
    {
        const linearized_edge l = linearize(edge, poses);
        system.weights.push_back(l.weight);
        const matrix6 wi = l.weight * l.jacobian_target.transpose() * edge.information;
        const matrix6 wj = l.weight * l.jacobian_source.transpose() * edge.information;
        add_block(triplets, edge.target, edge.target, wi * l.jacobian_target);
        add_block(triplets, edge.source, edge.source, wj * l.jacobian_source);
        add_block(triplets, edge.target, edge.source, wi * l.jacobian_source);
        add_block(triplets, edge.source, edge.target, wj * l.jacobian_target);
        if (edge.target != 0)
            system.g.segment<6>(first_unknown(edge.target)) += wi * l.residual;
        if (edge.source != 0)
            system.g.segment<6>(first_unknown(edge.source)) += wj * l.residual;
    }
    system.h.resize(unknowns, unknowns);
    system.h.setFromTriplets(triplets.begin(), triplets.end());
    system.diagonal = system.h.diagonal();
    return system;
}

/// Moves every node but node 0 by its part of a step, each in its own camera frame.
/// @param poses [in] the poses, one for each node.
/// @param step [in] h, six unknowns (ω in rad, υ in m) for each node from node 1 on.
/// @return the poses T_n exp(δ_n^), node 0's as it was.
std::vector<Eigen::Isometry3d> moved(const std::vector<Eigen::Isometry3d> &poses,
                                     const Eigen::VectorXd &step)
{
    std::vector<Eigen::Isometry3d> result = poses;
    for (std::size_t n = 1; n < poses.size(); n++)
    {
        const twist delta = step.segment<6>(first_unknown(n));
        result[n] = poses[n] * se3_exp(delta);
    }
    return result;
}

/// Says whether a step ends the solve: every node turned by less than ε_ω and moved by less
/// than ε_υ.
/// @param step [in] h, six unknowns (ω in rad, υ in m) for each node from node 1 on.
/// @return whether each node's |ω| < ε_ω and |υ| < ε_υ.
bool is_small(const Eigen::VectorXd &step)
{
    for (Eigen::Index n = 0; n < step.size(); n += 6)
    {
        if (!(step.segment<3>(n).norm() < pose_graph_converged_rotation_rad) ||
            !(step.segment<3>(n + 3).norm() < pose_graph_converged_translation_m))
            return false;
    }
    return true;
}

/// Says whether the Gauss-Newton step without damping is small: the check that a damped step under
/// ε_ω and ε_υ ends the solve at a minimum and not where the damping shrank it, which a damping by
/// H's diagonal does in the weak directions of an Ω whose eigenvalues span far [?: the
/// user's decision, 2026-11-15, option (a) over a refusal of Ω's span;
/// docs/methods/refinement.md, "The end"].
/// @param system [in] the system at the poses the solve would end with.
/// @return whether H h = -g factorizes and its h is_small(); false when the factorization fails
///         or h is not finite.
/// @note Allocates for the factorization and the step; throws std::bad_alloc when that fails.
bool gauss_newton_step_is_small(const linear_system &system)
{
    const Eigen::SimplicialLLT<sparse_matrix, Eigen::Lower, Eigen::AMDOrdering<int>> cholesky(
        system.h);
    if (cholesky.info() != Eigen::Success)
        return false;
    const Eigen::VectorXd step = cholesky.solve(-system.g);
    return step.allFinite() && is_small(step);
}

/// Levenberg-Marquardt's damping μ with Nielsen's update (docs/methods/refinement.md, "The damped
/// Gauss-Newton"): shrunk after an accepted step by how well the model predicted it, grown ever
/// faster after rejected steps in a row.
struct damping_schedule
{
    double mu = pose_graph_initial_damping; ///< μ, the share of H's diagonal added to it, > 0
    double nu = nielsen_first_growth;       ///< ν, the factor the next rejection grows μ by

    /// Updates the damping after an accepted step: μ ← μ max(1/3, 1 - (2ϱ - 1)³), ν ← ν_0.
    /// @param gain [in] ϱ, the step's gain ratio, > 0.
    void accepted(double gain)
    {
        const double centred = 2.0 * gain - 1.0;
        mu *= std::max(nielsen_least_shrink, 1.0 - centred * centred * centred);
        nu = nielsen_first_growth;
    }

    /// Updates the damping after a rejected step, or a factorization that failed: μ ← μ ν,
    /// ν ← 2 ν.
    void rejected()
    {
        mu *= nu;
        nu *= nielsen_growth_of_growth;
    }
};

/// Says why an edge cannot be solved, if it cannot.
/// @param edge [in] the edge.
/// @param index [in] its position in the graph's edges, for the reason.
/// @param nodes [in] the graph's nodes.
/// @return empty when its nodes are two distinct nodes of the graph, its transform is finite and
///         rigid and its information finite, symmetric within symmetry_tolerance and positive
///         definite; else the reason.
/// @note Allocates for the reason; throws std::bad_alloc when that fails.
std::string edge_refusal(const pose_graph_edge &edge, std::size_t index, std::size_t nodes)
{
    const std::string name = "edge " + std::to_string(index);
    for (const std::size_t node : {edge.target, edge.source})
    {
        if (node >= nodes)
            return name + " joins node " + std::to_string(node) + ", which is not in the graph";
    }
    if (edge.target == edge.source)
        return name + " joins node " + std::to_string(edge.target) + " to itself";
    if (!is_rigid(edge.transform))
        return name + "'s transform is not finite or not rigid";
    const matrix6 &omega = edge.information;
    if (!omega.allFinite())
        return name + "'s information is not finite";
    if ((omega - omega.transpose()).cwiseAbs().maxCoeff() >
        symmetry_tolerance * omega.cwiseAbs().maxCoeff())
        return name + "'s information is not symmetric";
    if (Eigen::LLT<matrix6>(omega).info() != Eigen::Success)
        return name + "'s information is not positive definite";
    return {};
}

/// Says why the edges do not join every node to node 0, if they do not.
/// @param graph [in] the graph, its edges' nodes in it.
/// @return empty when a chain of edges joins every node to node 0; else the reason: no edge, or
///         the first node no chain joins.
/// @note Allocates; throws std::bad_alloc when that fails.
std::string connection_refusal(const pose_graph &graph)
{
    if (graph.edges.empty())
        return "the graph has no edges";
    const std::size_t nodes = graph.poses.size();
    std::vector<std::vector<std::size_t>> neighbours(nodes);
    for (const pose_graph_edge &edge : graph.edges)
    {
        neighbours[edge.target].push_back(edge.source);
        neighbours[edge.source].push_back(edge.target);
    }
    // A breadth-first walk from node 0.
    std::vector<bool> reached(nodes, false);
    std::vector<std::size_t> queue{0};
    reached[0] = true;
    for (std::size_t next = 0; next < queue.size(); next++)
    {
        for (const std::size_t neighbour : neighbours[queue[next]])
        {
            if (!reached[neighbour])
            {
                reached[neighbour] = true;
                queue.push_back(neighbour);
            }
        }
    }
    for (std::size_t n = 0; n < nodes; n++)
    {
        if (!reached[n])
            return "node " + std::to_string(n) + " is not joined to node 0 by the edges";
    }
    return {};
}

/// Says why a graph and the parameters cannot be solved, if they cannot.
/// @param graph [in] the graph.
/// @param parameters [in] the parameters.
/// @return empty when the inputs keep optimize_pose_graph()'s conditions; else the reason.
/// @note Allocates for the reason; throws std::bad_alloc when that fails.
std::string input_refusal(const pose_graph &graph, const pose_graph_parameters &parameters)
{
    if (graph.poses.empty())
        return "the graph has no node";
    if (parameters.max_iterations < 1 ||
        parameters.max_iterations > pose_graph_largest_max_iterations)
        return "max_iterations " + std::to_string(parameters.max_iterations) +
               " lies outside 1 to " + std::to_string(pose_graph_largest_max_iterations);
    for (std::size_t n = 0; n < graph.poses.size(); n++)
    {
        if (!is_rigid(graph.poses[n]))
            return "pose " + std::to_string(n) + " is not finite or not rigid";
    }
    for (std::size_t e = 0; e < graph.edges.size(); e++)
    {
        std::string reason = edge_refusal(graph.edges[e], e, graph.poses.size());
        if (!reason.empty())
            return reason;
    }
    return {};
}
} // namespace

pose_graph_result optimize_pose_graph(const pose_graph &graph,
                                      const pose_graph_parameters &parameters)
{
    pose_graph_result result;
    result.message = input_refusal(graph, parameters);
    if (!result.message.empty())
        return result;
    result.message = connection_refusal(graph);
    if (!result.message.empty())
    {
        result.end = pose_graph_end::disconnected;
        return result;
    }

    // Levenberg-Marquardt with Marquardt's scaling and Nielsen's update of the damping
    // (docs/methods/refinement.md, "The damped Gauss-Newton").
    std::vector<Eigen::Isometry3d> poses = graph.poses;
    double objective = robust_objective(graph.edges, poses);
    linear_system system = build_system(graph.edges, poses);
    damping_schedule damping;
    result.end = pose_graph_end::iterations_used_up;
    Eigen::SimplicialLLT<sparse_matrix, Eigen::Lower, Eigen::AMDOrdering<int>> cholesky;
    for (std::uint32_t iteration = 1; iteration <= parameters.max_iterations; iteration++)
    {
        result.iterations = iteration;
        sparse_matrix damped = system.h;
        for (Eigen::Index k = 0; k < damped.rows(); k++)
            damped.coeffRef(k, k) += damping.mu * system.diagonal(k);
        cholesky.compute(damped);
        Eigen::VectorXd step;
        if (cholesky.info() == Eigen::Success)
            step = cholesky.solve(-system.g);
        if (cholesky.info() != Eigen::Success || !step.allFinite())
        {
            // A factorization that fails counts as a rejected step [?: untested, no input found
            // that makes one fail; docs/methods/refinement.md, "The damped Gauss-Newton"].
            damping.rejected();
            continue;
        }
        std::vector<Eigen::Isometry3d> candidate = moved(poses, step);
        const double candidate_objective = robust_objective(graph.edges, candidate);
        const double predicted =
            0.5 * step.dot(damping.mu * system.diagonal.cwiseProduct(step) - system.g);
        const double gain = (objective - candidate_objective) / predicted;
        if (gain > 0.0)
        {
            poses = std::move(candidate);
            objective = candidate_objective;
            system = build_system(graph.edges, poses);
            damping.accepted(gain);
        }
        else
        {
            damping.rejected();
        }
        // Converged needs both steps small: the damped one, and the undamped one at the poses
        // the solve would end with (docs/methods/refinement.md, "The end").
        if (is_small(step) && gauss_newton_step_is_small(system))
        {
            result.end = pose_graph_end::converged;
            break;
        }
    }
    result.poses = std::move(poses);
    result.edge_weights = std::move(system.weights);
    return result;
}
} // namespace gca
