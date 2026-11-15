// SPDX-License-Identifier: 0BSD
/// @file test_pose_graph.cpp
/// The refinement module's pose graph solve (refinement/pose_graph.hpp, optimize_pose_graph()) on
/// graphs built here from known poses; the method is docs/methods/refinement.md. The known poses
/// are the keyframes of a walk once around the tabletop box (synthetic_source.hpp,
/// tabletop_walk()); an edge between two of them carries their known relative transform with an
/// error the test puts in, and its standard deviations are 1 mrad and 1 mm on each axis but where
/// a case says otherwise. The start is the known walk with a drift put in: each step from one
/// keyframe to the next turned and shifted a little more, as tracking drifts. Node 0 starts at its
/// known pose.
/// - A ring with one edge closing the loop, its edges right: converged within a bound of
///   iterations, every pose at the known one, node 0 as given bit for bit, every weight 1. The same
///   started at the known poses: converged at the first iteration, where it started.
/// - The same ring with noisy edges and an information that correlates the axes: converged, every
///   pose within a tolerance of the known one, and the poses a stationary point of the robust
///   objective, checked by central differences of the objective as docs/methods/refinement.md
///   states it, computed here.
/// - One wrong edge among many right ones: weighed down by Huber's weight, every right edge at
///   the weight 1, every pose within a tolerance of the known one.
/// - A graph in two pieces, one without edges, and one node without edges: disconnected, the
///   reason, no poses, no iterations.
/// - The same ring with an information whose eigenvalues span 1e8: converged at the known poses,
///   through rejected steps. With a span of 1e12: never converged away from the known poses.
/// - The iterations used up: one iteration on the drifted ring, the poses after it.
/// - 300 keyframes of a walk, each joined to the keyframes 1, 10 and 25 after it, the edges noisy:
///   converged, every pose within a tolerance of the known one.
/// - Refusals as values, with no poses: no node, max_iterations outside its range while the
///   largest is taken, a pose that is not finite or not rigid, an edge whose node is not in the
///   graph or that joins a node to itself, whose transform is not finite or not rigid, whose
///   information is not finite, not symmetric or not positive definite, while an information
///   symmetric within the tolerance is taken.

#include "check.hpp"
#include "refinement/pose_graph.hpp"
#include "registration/se3.hpp"
#include "synthetic_source.hpp"
#include "transform_checks.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace
{
using gca_test::check;
using gca_test::check_refusal;
using gca_test::check_within;
using gca_test::error_of;
using gca_test::make_transform;
using gca_test::transform_error;
using gca_test::transform_tolerance;

using pose_list = std::vector<Eigen::Isometry3d>;
using matrix6 = Eigen::Matrix<double, 6, 6>;

/// The edges' standard deviation of the turn, in rad, on each axis: 1 mrad [?: a choice, about a
/// registration's error between keyframes].
constexpr double sigma_rad = 1e-3;

/// The edges' standard deviation of the shift, in m, on each axis: 1 mm [?: a choice].
constexpr double sigma_m = 1e-3;

/// The most iterations the tests give a solve [?: initial; docs/methods/refinement.md].
constexpr gca::pose_graph_parameters parameters = {100};

/// Returns the edges' information: the inverse of the covariance of standard deviations sigma_rad
/// and sigma_m on each axis.
/// @return diag(1/σ_rad² x 3, 1/σ_m² x 3), in rad^-2 and m^-2.
matrix6 test_information()
{
    matrix6 omega = matrix6::Zero();
    omega.diagonal() << 1.0 / (sigma_rad * sigma_rad), 1.0 / (sigma_rad * sigma_rad),
        1.0 / (sigma_rad * sigma_rad), 1.0 / (sigma_m * sigma_m), 1.0 / (sigma_m * sigma_m),
        1.0 / (sigma_m * sigma_m);
    return omega;
}

/// Returns an information that is neither diagonal nor the same on each axis, as a registration's
/// system gives [?: a choice]: Ω = (S P S)^-1, S the standard deviations 0.5, 1 and 2 mrad of the
/// turn and 2, 1 and 0.5 mm of the shift, P the correlations 0.4^|a - b| between axes a and b.
/// With it the rotation's inverse left Jacobian changes an edge's gradient: for an Ω whose turns'
/// block is a multiple of I and that holds no correlation between a turn and a shift, it does not,
/// since J_l^-1(φ)^T φ = φ [S: computed; [φ]x φ = 0].
/// @return Ω, in rad^-2, rad^-1 m^-1 and m^-2; symmetric and positive definite.
matrix6 correlated_information()
{
    Eigen::Matrix<double, 6, 1> sigma;
    sigma << 0.5e-3, 1e-3, 2e-3, 2e-3, 1e-3, 0.5e-3;
    matrix6 covariance;
    for (int a = 0; a < 6; a++)
        for (int b = 0; b < 6; b++)
            covariance(a, b) = sigma(a) * sigma(b) * std::pow(0.4, std::abs(a - b));
    const matrix6 omega = covariance.inverse();
    return 0.5 * (omega + omega.transpose());
}

/// Returns the keyframes of a walk once around the tabletop box.
/// @param count [in] the keyframes, >= 1.
/// @return count poses, camera frame to world frame, in the walk's order; nothing when the walk
///         is refused.
std::optional<pose_list> walk_keyframes(std::size_t count)
{
    gca_test::synthetic_walk walk = gca_test::tabletop_walk();
    walk.pose_count = count;
    return gca_test::walk_around(walk);
}

/// Returns the error put into an edge: a twist of each axis's standard deviation times a
/// different sine of the edge's position, so that the errors are fixed and look random [?: a
/// choice].
/// @param index [in] the edge's position in the graph.
/// @param scale [in] the error's size in standard deviations on each axis.
/// @return the twist, ω in rad and υ in m.
gca::twist edge_noise(std::size_t index, double scale)
{
    const double x = static_cast<double>(index);
    gca::twist n;
    n << sigma_rad * std::sin(1.7 * x + 0.3), sigma_rad * std::cos(2.3 * x + 0.1),
        sigma_rad * std::sin(3.1 * x + 0.7), sigma_m * std::cos(1.1 * x + 0.5),
        sigma_m * std::sin(2.9 * x + 0.2), sigma_m * std::cos(3.7 * x + 0.9);
    return scale * n;
}

/// Builds the edge between two known keyframes with an error put in: where the poses are the known
/// ones, the edge's residual is the error.
/// @param known [in] the known poses.
/// @param target [in] the edge's target node, in known.
/// @param source [in] the edge's source node, in known.
/// @param error [in] ε, the error, ω in rad and υ in m, in the target's camera frame.
/// @return the edge, Z = exp(-ε^) T_target^-1 T_source, with test_information().
gca::pose_graph_edge edge_between(const pose_list &known, std::size_t target, std::size_t source,
                                  const gca::twist &error)
{
    return {target, source, gca::se3_exp(-error) * known[target].inverse() * known[source],
            test_information()};
}

/// The drift put into each step of the start in the rings: a turn of 0.15 degrees and a shift of
/// 1.5 mm [?: a choice]; the ring's start lies up to 1.34 degrees and 23 mm from the known poses
/// [M 2026-11-15, this test].
gca::twist ring_drift()
{
    gca::twist d;
    d.head<3>() = Eigen::Vector3d(0.3, 0.8, -0.5).normalized() * (0.15 * std::numbers::pi / 180.0);
    d.tail<3>() = Eigen::Vector3d(1.0, -0.5, 0.3).normalized() * 1.5e-3;
    return d;
}

/// Builds a start with a drift: node 0 at its known pose, each next node at the known step from the
/// one before, moved by the drift in its own camera frame.
/// @param known [in] the known poses.
/// @param drift [in] the drift of each step, ω in rad and υ in m.
/// @return the poses, start_k = start_{k-1} T_{k-1}^-1 T_k exp(drift^).
pose_list drifted(const pose_list &known, const gca::twist &drift)
{
    pose_list start = known;
    for (std::size_t k = 1; k < known.size(); k++)
        start[k] = start[k - 1] * known[k - 1].inverse() * known[k] * gca::se3_exp(drift);
    return start;
}

/// Measures how far the poses lie from the known ones, the largest turn and shift over the nodes.
/// @param found [in] the poses found, one for each known one.
/// @param known [in] the known poses.
/// @return the largest angle, in degrees, and the largest shift of a camera's centre, in m.
transform_error largest_error(const pose_list &found, const pose_list &known)
{
    transform_error largest = {0.0, 0.0};
    for (std::size_t k = 0; k < known.size(); k++)
    {
        const transform_error e = error_of(found[k], known[k]);
        largest.rotation_deg = std::max(largest.rotation_deg, e.rotation_deg);
        largest.translation_m = std::max(largest.translation_m, e.translation_m);
    }
    return largest;
}

/// Checks that the poses lie within a tolerance of the known ones, and prints how far.
/// @param what [in] the case.
/// @param found [in] the poses found; a count other than known's fails.
/// @param known [in] the known poses.
/// @param tolerance [in] how far each may lie.
void check_poses_near(const std::string &what, const pose_list &found, const pose_list &known,
                      const transform_tolerance &tolerance)
{
    check(found.size() == known.size(), (what + ": a pose for each node").c_str());
    if (found.size() != known.size())
        return;
    const transform_error e = largest_error(found, known);
    check_within((what + ": largest rotation error (deg)").c_str(), e.rotation_deg,
                 tolerance.rotation_deg);
    check_within((what + ": largest translation error (m)").c_str(), e.translation_m,
                 tolerance.translation_m);
}

/// The robust objective of docs/methods/refinement.md, computed here from its statement: for each
/// edge E = T_i^-1 T_j Z^-1, r = (rotation vector of E's rotation, E's translation),
/// s = sqrt(r^T Ω r), Huber's loss of s with the module's threshold.
/// @param graph [in] the graph.
/// @param poses [in] the poses, one for each node.
/// @return F = Σ ρ(s).
double objective(const gca::pose_graph &graph, const pose_list &poses)
{
    constexpr double k = gca::pose_graph_huber_threshold;
    double f = 0.0;
    for (const gca::pose_graph_edge &edge : graph.edges)
    {
        const Eigen::Isometry3d e =
            poses[edge.target].inverse() * poses[edge.source] * edge.transform.inverse();
        const Eigen::AngleAxisd turn(e.linear());
        gca::twist r;
        r.head<3>() = turn.angle() * turn.axis();
        r.tail<3>() = e.translation();
        const double s = std::sqrt(r.dot(edge.information * r));
        f += s <= k ? 0.5 * s * s : k * s - 0.5 * k * k;
    }
    return f;
}

/// Returns the largest derivative of the robust objective over every node but node 0, each moved
/// in its own camera frame along each axis, by central differences of objective().
/// @param graph [in] the graph.
/// @param poses [in] the poses, one for each node.
/// @return max |(F(T_n exp(h e_a)) - F(T_n exp(-h e_a))) / 2h|, h = 1e-6 rad or m.
double largest_derivative(const gca::pose_graph &graph, const pose_list &poses)
{
    // h, the step of the central differences: 1e-6 rad or m, near the cube root of double's
    // rounding, 4.8e-6, where a central difference's truncation, h², and rounding, u / h, are
    // balanced [S: computed]. The solved noisy ring's share below is flat there: 7.4e-10 to
    // 7.5e-10 with steps of 1e-5, 1e-6, 1e-7 and 1e-8, and 9.8e-9 with 1e-4, the truncation's
    // [M 2026-11-15, this test with each step].
    constexpr double h = 1e-6;
    double largest = 0.0;
    for (std::size_t n = 1; n < poses.size(); n++)
    {
        for (int a = 0; a < 6; a++)
        {
            gca::twist step = gca::twist::Zero();
            step(a) = h;
            pose_list plus = poses;
            pose_list minus = poses;
            plus[n] = poses[n] * gca::se3_exp(step);
            minus[n] = poses[n] * gca::se3_exp(-step);
            const double d = (objective(graph, plus) - objective(graph, minus)) / (2.0 * h);
            largest = std::max(largest, std::abs(d));
        }
    }
    return largest;
}

/// Names a solve's end.
/// @param end [in] the end.
/// @return its name.
const char *end_name(gca::pose_graph_end end)
{
    switch (end)
    {
    case gca::pose_graph_end::converged:
        return "converged";
    case gca::pose_graph_end::iterations_used_up:
        return "iterations used up";
    case gca::pose_graph_end::disconnected:
        return "disconnected";
    case gca::pose_graph_end::refused:
        return "refused";
    }
    return "unknown";
}

/// Prints a result's end and iterations.
/// @param what [in] the case.
/// @param result [in] the result.
void print_result(const std::string &what, const gca::pose_graph_result &result)
{
    std::printf("%s: %s after %u iterations%s%s\n", what.c_str(), end_name(result.end),
                result.iterations, result.message.empty() ? "" : ", ", result.message.c_str());
}

/// Builds the ring: the keyframes, an edge from each to the next and one from the last to the
/// first that closes the loop, each with the error edge_noise() of its position times a scale.
/// @param known [in] the known poses, >= 2.
/// @param noise_scale [in] the edges' error in standard deviations; 0 for right edges.
/// @param start [in] the start's poses.
/// @return the graph.
gca::pose_graph ring_graph(const pose_list &known, double noise_scale, const pose_list &start)
{
    gca::pose_graph graph;
    graph.poses = start;
    const std::size_t n = known.size();
    for (std::size_t k = 0; k < n; k++)
        graph.edges.push_back(edge_between(known, k, (k + 1) % n, edge_noise(k, noise_scale)));
    return graph;
}

/// The ring's keyframes [?: a choice: 15 degrees apart around the tabletop box].
constexpr std::size_t ring_keyframes = 24;

/// How far a ring with right edges may end from the known poses: 1e-8 degrees and 1e-9 m. From
/// its drifted start it ended 1.9e-10 degrees and 5.0e-12 m off; started at the known
/// poses, 1.6e-15 degrees and 0 m off [M 2026-11-15, this test]: the last step's remainder and
/// double's rounding.
constexpr transform_tolerance exact_ring_tolerance = {1e-8, 1e-9};

/// The most iterations the drifted ring with right edges may take: 6. It took 4 [M 2026-11-15,
/// this test], a step within 1e-6 at the fourth: the Gauss-Newton step's quadratic convergence on
/// a graph whose minimum leaves every residual 0, which a wrong Jacobian loses.
constexpr std::uint32_t exact_ring_iterations = 6;

/// A ring with right edges and one edge closing the loop, from a drifted start.
/// @param known [in] the ring's known poses.
void check_ring_with_right_edges(const pose_list &known)
{
    const gca::pose_graph graph = ring_graph(known, 0.0, drifted(known, ring_drift()));
    const transform_error start = largest_error(graph.poses, known);
    std::printf("ring, right edges: the start %.3g deg and %.3g mm from the known poses\n",
                start.rotation_deg, 1e3 * start.translation_m);
    const gca::pose_graph_result result = gca::optimize_pose_graph(graph, parameters);
    print_result("ring, right edges", result);
    check(result.converged(), "ring, right edges: converged");
    check(result.iterations <= exact_ring_iterations,
          "ring, right edges: within the iterations stated");
    check_poses_near("ring, right edges", result.poses, known, exact_ring_tolerance);
    check(!result.poses.empty() && result.poses[0].matrix() == graph.poses[0].matrix(),
          "ring, right edges: node 0 as given, bit for bit");
    check(result.edge_weights.size() == graph.edges.size() &&
              std::all_of(result.edge_weights.begin(), result.edge_weights.end(),
                          [](double w) { return w == 1.0; }),
          "ring, right edges: every edge's weight 1");

    gca::pose_graph at_known = graph;
    at_known.poses = known;
    const gca::pose_graph_result again = gca::optimize_pose_graph(at_known, parameters);
    print_result("ring, right edges, from the known poses", again);
    check(again.converged() && again.iterations == 1,
          "ring, right edges, from the known poses: converged at the first iteration");
    check_poses_near("ring, right edges, from the known poses", again.poses, known,
                     exact_ring_tolerance);
}

/// The noisy ring's edge errors, in edge_noise()'s standard deviations of 1 mrad and 1 mm: 3, so
/// that at the minimum the edges' turns are large enough for the rotation's inverse left Jacobian
/// to show in the gradient [?: a choice; at 1 the probe that drops it gave 9.1e-8 against the
/// solve's 1.2e-9 below, at 3 4.9e-7 against 7.5e-10, M 2026-11-15].
constexpr double noisy_ring_error = 3.0;

/// How far a ring with noisy edges may end from the known poses: 0.3 degrees and 7 mm. It ended
/// 0.262 degrees and 5.86 mm off, from a start 1.34 degrees and 23 mm off [M 2026-11-15, this
/// test]: the minimum of edges with errors of 3 standard deviations lies there, not at the known
/// poses.
constexpr transform_tolerance noisy_ring_tolerance = {0.3, 7e-3};

/// The largest derivative of the robust objective at a solved ring with noisy edges, as a share of
/// that at its start: 2e-8. The solve left 7.5e-10, 9.6e-6 against 1.27e4, the remainder after
/// its last step below 1e-6; the same solve with the rotation's inverse left Jacobian taken as I
/// left 4.9e-7 [M 2026-11-15, this test and that mutant].
constexpr double noisy_ring_stationarity = 2e-8;

/// A ring with noisy edges and one edge closing the loop, from a drifted start, each edge with the
/// correlated information.
/// @param known [in] the ring's known poses.
void check_ring_with_noisy_edges(const pose_list &known)
{
    gca::pose_graph graph = ring_graph(known, noisy_ring_error, drifted(known, ring_drift()));
    for (gca::pose_graph_edge &edge : graph.edges)
        edge.information = correlated_information();
    const gca::pose_graph_result result = gca::optimize_pose_graph(graph, parameters);
    print_result("ring, noisy edges", result);
    check(result.converged(), "ring, noisy edges: converged");
    check_poses_near("ring, noisy edges", result.poses, known, noisy_ring_tolerance);
    if (result.poses.size() != known.size())
        return;
    const double at_start = largest_derivative(graph, graph.poses);
    const double at_end = largest_derivative(graph, result.poses);
    std::printf("ring, noisy edges: largest derivative of the objective %.3g at the start, %.3g at "
                "the end\n",
                at_start, at_end);
    check_within("ring, noisy edges: largest derivative at the end, a share of the start's",
                 at_end / at_start, noisy_ring_stationarity);
}

/// How far the ring with one wrong edge may end from the known poses: 0.05 degrees and 0.6 mm. It
/// ended 0.038 degrees and 0.41 mm off [M 2026-11-15, this test]: the bounded pull of the wrong
/// edge, k standard deviations, against the right ones.
constexpr transform_tolerance wrong_edge_tolerance = {0.05, 0.6e-3};

/// The largest weight the wrong edge may end with: 0.05. It ended at 0.035 [M 2026-11-15, this
/// test], k / s with s about 100 standard deviations.
constexpr double wrong_edge_weight = 0.05;

/// One wrong edge among many right ones: the ring's keyframes each joined to the keyframes 1, 2
/// and 6 after it, the edges right, and one edge from keyframe 3 to 11 off by a turn of 5 degrees
/// and a shift of 54 mm [?: a choice, a registration that converged on a wrong surface].
/// @param known [in] the ring's known poses.
void check_one_wrong_edge(const pose_list &known)
{
    gca::pose_graph graph;
    graph.poses = drifted(known, ring_drift());
    const std::size_t n = known.size();
    for (const std::size_t ahead : {1, 2, 6})
        for (std::size_t k = 0; k < n; k++)
            graph.edges.push_back(edge_between(known, k, (k + ahead) % n, gca::twist::Zero()));
    const std::size_t wrong = graph.edges.size();
    gca::pose_graph_edge edge = edge_between(known, 3, 11, gca::twist::Zero());
    edge.transform =
        make_transform(Eigen::Vector3d(0.2, -0.6, 0.8), 5.0, Eigen::Vector3d(0.03, -0.02, 0.04)) *
        edge.transform;
    graph.edges.push_back(edge);

    const gca::pose_graph_result result = gca::optimize_pose_graph(graph, parameters);
    print_result("one wrong edge", result);
    check(result.converged(), "one wrong edge: converged");
    check_poses_near("one wrong edge", result.poses, known, wrong_edge_tolerance);
    if (result.edge_weights.size() != graph.edges.size())
    {
        check(false, "one wrong edge: a weight for each edge");
        return;
    }
    check_within("one wrong edge: its weight", result.edge_weights[wrong], wrong_edge_weight);
    const double least_right =
        *std::min_element(result.edge_weights.begin(), result.edge_weights.begin() + wrong);
    std::printf("one wrong edge: the right edges' least weight %.6g\n", least_right);
    check(least_right == 1.0, "one wrong edge: every right edge's weight 1");
}

/// Checks a result that is disconnected: the reason, no poses, no weights, no iterations.
/// @param what [in] the case.
/// @param result [in] the result.
/// @param expected [in] the reason expected.
void check_disconnected(const std::string &what, const gca::pose_graph_result &result,
                        const std::string &expected)
{
    print_result(what, result);
    check(result.end == gca::pose_graph_end::disconnected && result.message == expected,
          (what + ": disconnected with the expected reason").c_str());
    check(result.poses.empty() && result.edge_weights.empty() && result.iterations == 0,
          (what + ": no poses, no weights, no iterations").c_str());
}

/// A graph in two pieces, a graph without edges, and one node without edges.
/// @param known [in] the ring's known poses, >= 12.
void check_graphs_not_in_one_piece(const pose_list &known)
{
    gca::pose_graph pieces;
    pieces.poses = pose_list(known.begin(), known.begin() + 12);
    for (std::size_t k = 0; k + 1 < 12; k++)
    {
        if (k != 5)
            pieces.edges.push_back(edge_between(known, k, k + 1, gca::twist::Zero()));
    }
    check_disconnected("two pieces, nodes 0 to 5 and 6 to 11",
                       gca::optimize_pose_graph(pieces, parameters),
                       "node 6 is not joined to node 0 by the edges");

    gca::pose_graph no_edges;
    no_edges.poses = pose_list(known.begin(), known.begin() + 3);
    check_disconnected("three nodes without edges", gca::optimize_pose_graph(no_edges, parameters),
                       "the graph has no edges");
    no_edges.poses.resize(1);
    check_disconnected("one node without edges", gca::optimize_pose_graph(no_edges, parameters),
                       "the graph has no edges");
}

/// Returns an information whose eigenvalues span far: Ω = 1e6 Q diag(span, 1, 1, 1, 1, 1) Q^T, Q
/// the orthogonal factor of the matrix sin(1.3 a + 2.1 b + 0.4) of the axes a, b, so that five
/// directions keep standard deviations of 1 mrad or 1 mm and one direction, turned against every
/// axis, is sqrt(span) times as sure [?: a choice, informations far worse conditioned than the
/// tests' other ones].
/// @param span [in] the ratio of the largest eigenvalue to the others, >= 1.
/// @return Ω, in rad^-2, rad^-1 m^-1 and m^-2; symmetric and positive definite.
matrix6 ill_conditioned_information(double span)
{
    matrix6 a;
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            a(i, j) = std::sin(1.3 * i + 2.1 * j + 0.4);
    const matrix6 q = a.householderQr().householderQ();
    Eigen::Matrix<double, 6, 1> eigenvalues;
    eigenvalues << span, 1.0, 1.0, 1.0, 1.0, 1.0;
    const matrix6 omega = 1e6 * q * eigenvalues.asDiagonal() * q.transpose();
    return 0.5 * (omega + omega.transpose());
}

/// How far the ring with an ill-conditioned information may end from the known poses: 1e-6
/// degrees and 1e-8 m. It ended 1.4e-8 degrees and 3.9e-10 m off after 61 iterations, of which 12
/// were rejected steps [M 2026-11-15, this test, and the rejections counted by a scratch copy of
/// the module].
constexpr transform_tolerance ill_conditioned_tolerance = {1e-6, 1e-8};

/// The ring with right edges whose information is ill-conditioned, from a drifted start: the
/// damping shrinks over the accepted steps until a step overshoots and is rejected, which grows it
/// again (docs/methods/refinement.md, "Accuracy"); converged, at the known poses.
/// @param known [in] the ring's known poses.
void check_rejected_steps(const pose_list &known)
{
    gca::pose_graph graph = ring_graph(known, 0.0, drifted(known, ring_drift()));
    for (gca::pose_graph_edge &edge : graph.edges)
        edge.information = ill_conditioned_information(1e8);
    const gca::pose_graph_result result = gca::optimize_pose_graph(graph, parameters);
    print_result("ring, right edges, ill-conditioned information", result);
    check(result.converged(), "ring, right edges, ill-conditioned information: converged");
    check_poses_near("ring, right edges, ill-conditioned information", result.poses, known,
                     ill_conditioned_tolerance);
}

/// The ring with right edges whose information's eigenvalues span 1e12, from a drifted start:
/// there the damping by H's diagonal shrinks the steps in the weak directions under ε far from the
/// minimum, which the undamped Gauss-Newton step tells apart (docs/methods/refinement.md, "The
/// end"). It must not end converged away from the known poses: it ends with its iterations used up,
/// 2.08 degrees and 23.9 mm off; without the undamped step's check it ended converged after 3
/// iterations 2.66 degrees and 24 mm off [M 2026-11-15, this test and a mutant without the check].
/// @param known [in] the ring's known poses.
void check_no_false_convergence(const pose_list &known)
{
    gca::pose_graph graph = ring_graph(known, 0.0, drifted(known, ring_drift()));
    for (gca::pose_graph_edge &edge : graph.edges)
        edge.information = ill_conditioned_information(1e12);
    const gca::pose_graph_result result = gca::optimize_pose_graph(graph, parameters);
    print_result("ring, right edges, information spanning 1e12", result);
    check(result.poses.size() == known.size(),
          "ring, right edges, information spanning 1e12: a pose for each node");
    if (result.poses.size() != known.size())
        return;
    const transform_error e = largest_error(result.poses, known);
    std::printf("ring, right edges, information spanning 1e12: %.3g deg and %.3g mm from the known "
                "poses\n",
                e.rotation_deg, 1e3 * e.translation_m);
    if (result.converged())
        check_poses_near("ring, right edges, information spanning 1e12, converged", result.poses,
                         known, ill_conditioned_tolerance);
}

/// One iteration on the drifted ring with right edges: the iterations used up, with the poses.
/// @param known [in] the ring's known poses.
void check_iterations_used_up(const pose_list &known)
{
    const gca::pose_graph graph = ring_graph(known, 0.0, drifted(known, ring_drift()));
    const gca::pose_graph_result result = gca::optimize_pose_graph(graph, {1});
    print_result("ring, right edges, one iteration", result);
    check(result.end == gca::pose_graph_end::iterations_used_up && result.iterations == 1 &&
              result.message.empty(),
          "ring, right edges, one iteration: the iterations used up");
    check(result.poses.size() == known.size() && result.edge_weights.size() == graph.edges.size(),
          "ring, right edges, one iteration: a pose for each node, a weight for each edge");
    if (result.poses.size() != known.size())
        return;
    check(result.poses[0].matrix() == graph.poses[0].matrix(),
          "ring, right edges, one iteration: node 0 as given");
    const transform_error start = largest_error(graph.poses, known);
    const transform_error after = largest_error(result.poses, known);
    std::printf("ring, right edges, one iteration: %.3g mm from the known poses, %.3g mm at the "
                "start\n",
                1e3 * after.translation_m, 1e3 * start.translation_m);
}

/// The keyframes of the large graph [?: a choice, 1.2 degrees apart around the tabletop box].
constexpr std::size_t large_keyframes = 300;

/// How far the large graph may end from the known poses: 0.1 degrees and 2 mm. It ended 0.085
/// degrees and 1.55 mm off, from a start 1.71 degrees and 29 mm off [M 2026-11-15, this test].
constexpr transform_tolerance large_graph_tolerance = {0.1, 2e-3};

/// 300 keyframes of a walk, each joined to the keyframes 1, 10 and 25 after it, around the loop,
/// the edges noisy, the start drifted by a tenth of the ring's drift at each step.
void check_300_keyframes()
{
    const std::optional<pose_list> known = walk_keyframes(large_keyframes);
    check(known.has_value(), "300 keyframes: the walk");
    if (!known)
        return;
    gca::pose_graph graph;
    graph.poses = drifted(*known, 0.1 * ring_drift());
    for (const std::size_t ahead : {1, 10, 25})
        for (std::size_t k = 0; k < large_keyframes; k++)
            graph.edges.push_back(edge_between(*known, k, (k + ahead) % large_keyframes,
                                               edge_noise(graph.edges.size(), 1.0)));
    const transform_error start = largest_error(graph.poses, *known);
    std::printf("300 keyframes, %zu edges: the start %.3g deg and %.3g mm from the known poses\n",
                graph.edges.size(), start.rotation_deg, 1e3 * start.translation_m);
    const gca::pose_graph_result result = gca::optimize_pose_graph(graph, parameters);
    print_result("300 keyframes", result);
    check(result.converged(), "300 keyframes: converged");
    check_poses_near("300 keyframes", result.poses, *known, large_graph_tolerance);
}

/// Checks that a graph is refused with a reason and without poses.
/// @param what [in] the case.
/// @param graph [in] the graph.
/// @param params [in] the parameters.
/// @param expected [in] the reason expected.
void check_refused(const std::string &what, const gca::pose_graph &graph,
                   const gca::pose_graph_parameters &params, const std::string &expected)
{
    const gca::pose_graph_result result = gca::optimize_pose_graph(graph, params);
    check_refusal(what, result.end == gca::pose_graph_end::refused, result.message,
                  result.message == expected);
    check(result.poses.empty() && result.edge_weights.empty() && result.iterations == 0,
          (what + ": no poses, no weights, no iterations").c_str());
}

/// Checks that a graph is taken: solved to convergence.
/// @param what [in] the case.
/// @param graph [in] the graph.
/// @param params [in] the parameters.
void check_taken(const std::string &what, const gca::pose_graph &graph,
                 const gca::pose_graph_parameters &params)
{
    const gca::pose_graph_result result = gca::optimize_pose_graph(graph, params);
    print_result(what, result);
    check(result.converged(), (what + ": taken and converged").c_str());
}

/// The refusals, each on the ring with right edges with one input broken.
/// @param known [in] the ring's known poses.
void check_refusals(const pose_list &known)
{
    const gca::pose_graph ring = ring_graph(known, 0.0, drifted(known, ring_drift()));
    const double nan = std::numeric_limits<double>::quiet_NaN();

    check_refused("no node", gca::pose_graph{}, parameters, "the graph has no node");
    check_refused("max_iterations 0", ring, {0}, "max_iterations 0 lies outside 1 to 1000");
    check_refused("max_iterations 1001", ring, {1001},
                  "max_iterations 1001 lies outside 1 to 1000");
    check_taken("max_iterations 1000", ring, {1000});

    gca::pose_graph g = ring;
    g.poses[5].linear() *= 1.01;
    check_refused("pose 5 scaled", g, parameters, "pose 5 is not finite or not rigid");
    g = ring;
    g.poses[5].translation().x() = nan;
    check_refused("pose 5 not finite", g, parameters, "pose 5 is not finite or not rigid");

    g = ring;
    g.edges[4].target = known.size();
    check_refused("edge 4's target not in the graph", g, parameters,
                  "edge 4 joins node 24, which is not in the graph");
    g = ring;
    g.edges[4].source = known.size() + 3;
    check_refused("edge 4's source not in the graph", g, parameters,
                  "edge 4 joins node 27, which is not in the graph");
    g = ring;
    g.edges[4].source = g.edges[4].target;
    check_refused("edge 4 from a node to itself", g, parameters, "edge 4 joins node 4 to itself");

    g = ring;
    g.edges[4].transform.translation().y() = nan;
    check_refused("edge 4's transform not finite", g, parameters,
                  "edge 4's transform is not finite or not rigid");
    g = ring;
    g.edges[4].transform.linear() *= 0.99;
    check_refused("edge 4's transform scaled", g, parameters,
                  "edge 4's transform is not finite or not rigid");

    g = ring;
    g.edges[4].information(2, 3) = nan;
    check_refused("edge 4's information not finite", g, parameters,
                  "edge 4's information is not finite");
    g = ring;
    g.edges[4].information(1, 4) = 1e-3 * g.edges[4].information(4, 4);
    check_refused("edge 4's information not symmetric", g, parameters,
                  "edge 4's information is not symmetric");
    g = ring;
    g.edges[4].information(1, 4) = 1e-12 * g.edges[4].information(4, 4);
    check_taken("edge 4's information symmetric within the tolerance", g, parameters);
    g = ring;
    g.edges[4].information.setZero();
    check_refused("edge 4's information zero", g, parameters,
                  "edge 4's information is not positive definite");
    g = ring;
    g.edges[4].information(5, 5) = 0.0;
    check_refused("edge 4's information without the shift along z", g, parameters,
                  "edge 4's information is not positive definite");
    g = ring;
    g.edges[4].information(0, 0) = -1.0;
    check_refused("edge 4's information negative along a turn", g, parameters,
                  "edge 4's information is not positive definite");
}
} // namespace

int main()
{
    const std::optional<pose_list> known = walk_keyframes(ring_keyframes);
    check(known.has_value(), "the ring's walk");
    if (known)
    {
        check_ring_with_right_edges(*known);
        check_ring_with_noisy_edges(*known);
        check_one_wrong_edge(*known);
        check_rejected_steps(*known);
        check_no_false_convergence(*known);
        check_graphs_not_in_one_piece(*known);
        check_iterations_used_up(*known);
        check_refusals(*known);
    }
    check_300_keyframes();
    return gca_test::check_summary("test_pose_graph");
}
