// SPDX-License-Identifier: 0BSD
/// @file proposal.cpp
/// Separation's proposal of the support plane and the box (see proposal.hpp and
/// docs/methods/separation.md), in named steps: the checks of the inputs, gravity's up, the
/// trajectory's meeting point, the window and its sample, the plane's hypotheses, fits and
/// judgement, the footprint's cells and pieces, the object's choice and the box.

#include "separation/proposal.hpp"

#include "separation/gravity.hpp"
#include "separation/plane_fit.hpp"
#include "separation/separation_messages.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>

namespace gca
{
namespace
{
/// The seed of the generator the plane's sample and hypotheses are drawn from [?: a choice: any
/// fixed value, so that a proposal is the same in every run].
constexpr std::uint64_t sample_seed = 20261115;

/// Says why the inputs are refused (docs/methods/separation.md, "The ends"), before any is used.
/// @param full_model [in] the full model's points.
/// @param trajectory [in] the trajectory.
/// @param gravity [in] gravity's estimate.
/// @param parameters [in] v and D.
/// @return the reason, naming the parameter, gravity, the pose or the point; empty when every
///         input is accepted.
std::string refusal_of(std::span<const model_point> full_model,
                       std::span<const camera_pose> trajectory, const gravity_estimate &gravity,
                       const separation_parameters &parameters)
{
    const double v = parameters.voxel_size_m;
    const double depth = parameters.max_depth_m;
    if (!(std::isfinite(v) && v > 0.0))
        return "the voxel size " + text_of(v) + " m is not a finite number > 0";
    if (!(std::isfinite(depth) && depth > 0.0))
        return "the farthest depth " + text_of(depth) + " m is not a finite number > 0";
    if (depth > separation_largest_depth_voxels * v)
        return "the farthest depth " + text_of(depth) + " m is more than " +
               text_of(separation_largest_depth_voxels) + " voxels of " + text_of(v) + " m";
    if (!is_outcome(gravity.outcome))
        return "gravity's outcome " + std::to_string(static_cast<int>(gravity.outcome)) +
               " is no outcome";
    if (gravity.outcome == gravity_outcome::estimated && !up_is_unit(gravity.up))
        return "gravity's up (" + text_of(gravity.up.x()) + ", " + text_of(gravity.up.y()) + ", " +
               text_of(gravity.up.z()) + ") is not finite with a length within " +
               text_of(gravity_up_tolerance) + " of 1";
    if (trajectory.empty())
        return "the trajectory holds no pose";
    for (std::size_t i = 0; i < trajectory.size(); i++)
    {
        const std::string fault = fault_text(trajectory[i]);
        if (!fault.empty())
            return "pose " + std::to_string(i) + ": " + fault;
    }
    for (std::size_t k = 0; k < full_model.size(); k++)
        if (!full_model[k].position_m.allFinite())
            return "point " + std::to_string(k) + ": its position is not finite";
    return "";
}

/// Where the views meet and how far the window reaches, with gravity's up.
struct view_geometry
{
    Eigen::Vector3d up;              ///< u, gravity's up, unit, in the models' frame
    Eigen::Vector3d meeting_point_m; ///< m, in the models' frame, in m
    double window_m;                 ///< W, in m
};

/// The trajectory's geometry, or why it has none.
struct view_result
{
    std::optional<view_geometry> geometry; ///< the geometry; nothing when the views do not meet
    std::string reason;                    ///< why the views do not meet; empty otherwise
};

/// Finds the views' meeting point and the window (docs/methods/separation.md, "The views' meeting
/// point", "The window").
/// @param trajectory [in] the trajectory, camera to model, in the models' frame and m: at least one
///        pose, each finite with a unit quaternion.
/// @param up [in] gravity's up, unit, in the models' frame.
/// @param max_depth_m [in] D, in m, finite and > 0.
/// @return the geometry; or nothing with the reason when the least eigenvalue of
///         (1/N) Σ (I - z z^T) lies below separation_least_axis_spread.
view_result views_of(std::span<const camera_pose> trajectory, const Eigen::Vector3d &up,
                     double max_depth_m)
{
    const double count = static_cast<double>(trajectory.size());
    Eigen::Matrix3d across = Eigen::Matrix3d::Zero();
    Eigen::Vector3d across_centres = Eigen::Vector3d::Zero();
    for (const camera_pose &pose : trajectory)
    {
        const Eigen::Vector3d z = pose.orientation.toRotationMatrix().col(2);
        const Eigen::Matrix3d off_axis = Eigen::Matrix3d::Identity() - z * z.transpose();
        across += off_axis;
        across_centres += off_axis * pose.position_m;
    }

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> spread(across / count);
    if (!(spread.eigenvalues()(0) >= separation_least_axis_spread))
        return {std::nullopt,
                "the views' optical axes do not meet: the least eigenvalue of their spread is " +
                    text_of(spread.eigenvalues()(0)) + ", below " +
                    text_of(separation_least_axis_spread) + ", sin² 5 degrees"};
    // m = A^-1 b by A's eigenvectors; every eigenvalue is at least N sin² 5 degrees.
    const Eigen::Matrix3d &vectors = spread.eigenvectors();
    const Eigen::Vector3d meeting = vectors * ((vectors.transpose() * across_centres).array() /
                                               (count * spread.eigenvalues().array()))
                                                  .matrix();

    std::vector<double> distances;
    distances.reserve(trajectory.size());
    for (const camera_pose &pose : trajectory)
        distances.push_back((pose.position_m - meeting).norm());
    // The median, the upper of the two middle distances for an even count.
    const auto middle = distances.begin() + static_cast<std::ptrdiff_t>(distances.size() / 2);
    std::nth_element(distances.begin(), middle, distances.end());
    return {view_geometry{up, meeting, std::min(*middle, max_depth_m)}, ""};
}

/// Turns a plane's normal to the side of gravity's up.
/// @param normal [in] the normal, unit, in the models' frame.
/// @param up [in] u, unit, in the models' frame.
/// @return normal, or -normal when normal · up < 0.
Eigen::Vector3d toward(const Eigen::Vector3d &normal, const Eigen::Vector3d &up)
{
    return normal.dot(up) < 0.0 ? Eigen::Vector3d(-normal) : normal;
}

/// Gathers the window: the model's points within W of m (docs/methods/separation.md, "The window").
/// @param full_model [in] the full model's points, in the models' frame, in m.
/// @param views [in] m and W.
/// @return the window's points in double, in the models' frame, in m, in the model's order.
std::vector<Eigen::Vector3d> window_of(std::span<const model_point> full_model,
                                       const view_geometry &views)
{
    std::vector<Eigen::Vector3d> window;
    for (const model_point &point : full_model)
    {
        const Eigen::Vector3d p = point.position_m.cast<double>();
        if ((p - views.meeting_point_m).norm() <= views.window_m)
            window.push_back(p);
    }
    return window;
}

/// Draws the sample the hypotheses are scored on: separation_sample_points of the window's points
/// with replacement, each index the generator's output modulo the window's count.
/// @param window [in] the window's points, in the models' frame, in m.
/// @param generator [in,out] the generator.
/// @return the sample; the whole window when it holds no more points than the sample.
std::vector<Eigen::Vector3d> sample_of(const std::vector<Eigen::Vector3d> &window,
                                       std::mt19937_64 &generator)
{
    if (window.size() <= separation_sample_points)
        return window;
    std::vector<Eigen::Vector3d> sample;
    sample.reserve(separation_sample_points);
    for (std::size_t s = 0; s < separation_sample_points; s++)
        sample.push_back(window[generator() % window.size()]);
    return sample;
}

/// What lies in a plane's band.
struct band_points
{
    std::size_t count = 0;             ///< the points in the band
    double squared_distances_m2 = 0.0; ///< the sum of their squared distances from the plane, in m²
};

/// Counts the points within the plane's band and sums their squared distances from it, and keeps
/// them when asked.
/// @param points [in] the points, in the models' frame, in m.
/// @param plane [in] the plane, in the models' frame.
/// @param band_m [in] d_in, in m.
/// @param kept [out] the points in the band, when not null; cleared first.
/// @return the count and the sum.
band_points band_of(std::span<const Eigen::Vector3d> points, const support_plane &plane,
                    double band_m, std::vector<Eigen::Vector3d> *kept)
{
    if (kept)
        kept->clear();
    band_points band;
    for (const Eigen::Vector3d &p : points)
    {
        const double distance = plane.height_m(p);
        if (std::abs(distance) <= band_m)
        {
            band.count++;
            band.squared_distances_m2 += distance * distance;
            if (kept)
                kept->push_back(p);
        }
    }
    return band;
}

/// Says whether a plane lies under the object (docs/methods/separation.md, "The support plane"):
/// its normal within separation_largest_tilt_deg of up, the meeting point above it.
/// @param plane [in] the plane, its normal turned toward up, in the models' frame.
/// @param views [in] gravity's up and the meeting point.
/// @return whether both conditions hold.
bool under_object(const support_plane &plane, const view_geometry &views)
{
    const double least_cosine = std::cos(separation_largest_tilt_deg * std::numbers::pi / 180.0);
    return plane.normal.dot(views.up) >= least_cosine &&
           plane.height_m(views.meeting_point_m) > 0.0;
}

/// Finds the best hypothesis of the support plane (docs/methods/separation.md, "The support
/// plane", "Hypotheses"): separation_hypotheses planes through three points of the sample, each a
/// candidate when its normal, turned toward up, lies within separation_largest_tilt_deg of up and
/// the meeting point lies above it; the candidate with the most sampled points within the band.
/// @param sample [in] the sampled points of the window, in the models' frame, in m.
/// @param views [in] gravity's up and the meeting point.
/// @param band_m [in] d_in, in m.
/// @param generator [in,out] the generator the three points are drawn from.
/// @return the best candidate, the first on a tie; nothing when no hypothesis is a candidate, also
///         when the sample holds fewer than three points.
std::optional<support_plane> best_hypothesis(std::span<const Eigen::Vector3d> sample,
                                             const view_geometry &views, double band_m,
                                             std::mt19937_64 &generator)
{
    if (sample.size() < 3)
        return std::nullopt;
    std::optional<support_plane> best;
    std::size_t best_count = 0;
    for (std::uint32_t h = 0; h < separation_hypotheses; h++)
    {
        const Eigen::Vector3d &a = sample[generator() % sample.size()];
        const Eigen::Vector3d &b = sample[generator() % sample.size()];
        const Eigen::Vector3d &c = sample[generator() % sample.size()];
        const Eigen::Vector3d across = (b - a).cross(c - a);
        const double length = across.norm();
        if (!(length > 0.0))
            continue;
        const Eigen::Vector3d normal = toward(across / length, views.up);
        const support_plane plane{normal, normal.dot(a)};
        if (!under_object(plane, views))
            continue;
        const std::size_t count = band_of(sample, plane, band_m, nullptr).count;
        if (!best || count > best_count)
        {
            best = plane;
            best_count = count;
        }
    }
    return best;
}

/// The support plane after its fits, or why there is none.
struct plane_result
{
    std::optional<support_plane> plane; ///< the plane; nothing when there is no plane
    std::size_t points = 0;             ///< the window's points in its band, the last counted
    std::string reason;                 ///< why there is no plane; empty otherwise
};

/// Fits the support plane to the window's points in the band of the plane before it, three times,
/// and judges it (docs/methods/separation.md, "Fits", "No plane"): the band holds at least
/// separation_least_plane_points points before each fit and after the last; the fitted plane still
/// lies under the object; the last band's points lie within an RMS distance of
/// separation_largest_band_spread_voxels v of it.
/// @param window [in] the window's points, in the models' frame, in m.
/// @param hypothesis [in] the best hypothesis, in the models' frame.
/// @param views [in] gravity's up and the meeting point.
/// @param voxel_m [in] v, in m.
/// @return the plane, its normal turned toward up, and its band's count; or nothing with the count
///         of the band that fell short or was judged, and the reason.
plane_result fitted_plane(const std::vector<Eigen::Vector3d> &window,
                          const support_plane &hypothesis, const view_geometry &views,
                          double voxel_m)
{
    const double band_m = separation_band_voxels * voxel_m;
    support_plane plane = hypothesis;
    std::vector<Eigen::Vector3d> band;
    band_points last;
    static_assert(separation_least_plane_points >= 3, "a fit needs three points");
    for (int fit = 0; fit <= separation_plane_fits; fit++)
    {
        last = band_of(window, plane, band_m, fit < separation_plane_fits ? &band : nullptr);
        if (last.count < separation_least_plane_points)
            return {std::nullopt, last.count,
                    "no plane under the views' meeting point: the best holds " +
                        std::to_string(last.count) + " points of the window within " +
                        text_of(band_m) + " m, fewer than " +
                        std::to_string(separation_least_plane_points)};
        if (fit == separation_plane_fits)
            break;
        // The band holds at least separation_least_plane_points >= 3 points: the fit exists.
        const plane_fit fitted = *fit_plane(band);
        const Eigen::Vector3d normal = toward(fitted.normal, views.up);
        plane = {normal, normal.dot(fitted.centroid)};
    }
    // A hypothesis that cuts across a surface at a slant holds a strip of it, whose fit is that
    // surface: the fit can leave the conditions.
    if (!under_object(plane, views))
        return {std::nullopt, last.count,
                "no plane under the views' meeting point: the best plane, fitted, lies " +
                    text_of(std::acos(std::clamp(plane.normal.dot(views.up), -1.0, 1.0)) * 180.0 /
                            std::numbers::pi) +
                    " degrees from up, the meeting point " +
                    text_of(plane.height_m(views.meeting_point_m)) +
                    " m above it: it leaves the conditions"};
    // A fit that keeps crossing surfaces at a slant spreads its band's points across the band; a
    // surface lies within its noise of its plane.
    const double spread_m = std::sqrt(last.squared_distances_m2 / static_cast<double>(last.count));
    const double largest_m = separation_largest_band_spread_voxels * voxel_m;
    if (!(spread_m <= largest_m))
        return {std::nullopt, last.count,
                "no plane under the views' meeting point: the best plane's " +
                    std::to_string(last.count) + " points of the window lie " + text_of(spread_m) +
                    " m from it in the root mean square, more than " + text_of(largest_m) +
                    " m: it cuts across surfaces"};
    return {plane, last.count, ""};
}

/// The plane's frame on the plane: the foot of the meeting point and two axes in the plane.
struct plane_frame
{
    Eigen::Vector3d origin_m; ///< o, the meeting point's foot on the plane, models' frame, in m
    Eigen::Vector3d e1;       ///< the models' axis least aligned with n, projected, unit
    Eigen::Vector3d e2;       ///< n x e1
};

/// Builds the plane's frame (docs/methods/separation.md, "What stands on the plane").
/// @param plane [in] the support plane, in the models' frame.
/// @param meeting_point_m [in] m, in the models' frame, in m.
/// @return o, e1 and e2.
plane_frame frame_of(const support_plane &plane, const Eigen::Vector3d &meeting_point_m)
{
    Eigen::Index least = 0;
    plane.normal.cwiseAbs().minCoeff(&least);
    const Eigen::Vector3d axis = Eigen::Vector3d::Unit(least);
    const Eigen::Vector3d e1 = (axis - axis.dot(plane.normal) * plane.normal).normalized();
    return {meeting_point_m - plane.height_m(meeting_point_m) * plane.normal, e1,
            plane.normal.cross(e1)};
}

/// Packs a cell's coordinates into a key that sorts by i, then by j.
/// @param i [in] the cell's index along e1, within the int32 range.
/// @param j [in] the cell's index along e2, within the int32 range.
/// @return the key.
std::uint64_t key_of(std::int64_t i, std::int64_t j)
{
    constexpr std::int64_t shift = std::int64_t{1} << 31;
    return (static_cast<std::uint64_t>(i + shift) << 32) | static_cast<std::uint64_t>(j + shift);
}

/// Unpacks a cell's coordinates.
/// @param key [in] the key, as key_of() packs it.
/// @return (i, j).
std::pair<std::int64_t, std::int64_t> cell_of(std::uint64_t key)
{
    constexpr std::int64_t shift = std::int64_t{1} << 31;
    return {static_cast<std::int64_t>(key >> 32) - shift,
            static_cast<std::int64_t>(key & 0xffffffffu) - shift};
}

/// What stands on the plane, in cells of the plane's frame.
struct footprint
{
    /// Each standing point's (a, b, h): its coordinates along e1 and e2 from o and its height above
    /// the plane, in m.
    std::vector<Eigen::Vector3d> standing;
    /// The occupied cells' keys (key_of()), sorted, each once.
    std::vector<std::uint64_t> keys;
    /// The standing points' indices, grouped by cell in the keys' order.
    std::vector<std::uint32_t> items;
    /// Where each cell's points begin in items, and items' size at the end: one more than keys.
    std::vector<std::size_t> first;
};

/// Gathers the points that stand on the plane, higher than d_in, and their cells of c
/// (docs/methods/separation.md, "What stands on the plane").
/// @param window [in] the window's points, in the models' frame, in m.
/// @param plane [in] the support plane.
/// @param frame [in] the plane's frame.
/// @param voxel_m [in] v, in m.
/// @return the footprint; no standing point and no cell when nothing stands on the plane.
footprint footprint_of(const std::vector<Eigen::Vector3d> &window, const support_plane &plane,
                       const plane_frame &frame, double voxel_m)
{
    const double band_m = separation_band_voxels * voxel_m;
    const double cell_m = separation_cell_voxels * voxel_m;
    footprint result;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> entries; // (key, standing point)
    for (const Eigen::Vector3d &p : window)
    {
        const double h = plane.height_m(p);
        if (!(h > band_m))
            continue;
        const Eigen::Vector3d offset = p - frame.origin_m;
        const double a = frame.e1.dot(offset);
        const double b = frame.e2.dot(offset);
        entries.emplace_back(key_of(static_cast<std::int64_t>(std::floor(a / cell_m)),
                                    static_cast<std::int64_t>(std::floor(b / cell_m))),
                             static_cast<std::uint32_t>(result.standing.size()));
        result.standing.emplace_back(a, b, h);
    }
    std::sort(entries.begin(), entries.end());
    for (std::size_t e = 0; e < entries.size(); e++)
    {
        if (e == 0 || entries[e].first != entries[e - 1].first)
        {
            result.keys.push_back(entries[e].first);
            result.first.push_back(e);
        }
        result.items.push_back(entries[e].second);
    }
    result.first.push_back(entries.size());
    return result;
}

/// Finds a cell's root in the union-find forest, halving the path on the way [S: Tarjan 1975].
/// @param parent [in,out] each cell's parent; a root is its own parent.
/// @param cell [in] the cell.
/// @return its root.
std::uint32_t root_of(std::vector<std::uint32_t> &parent, std::uint32_t cell)
{
    while (parent[cell] != cell)
    {
        parent[cell] = parent[parent[cell]];
        cell = parent[cell];
    }
    return cell;
}

/// Joins the occupied cells to their occupied neighbours into pieces: each cell to the four of its
/// eight neighbours with a greater key, which with the lower ones joining it covers all eight; the
/// lower root becomes the root of both.
/// @param keys [in] the occupied cells' keys, sorted.
/// @return each cell's root: equal roots, one piece.
std::vector<std::uint32_t> pieces_of(const std::vector<std::uint64_t> &keys)
{
    std::vector<std::uint32_t> parent(keys.size());
    for (std::uint32_t c = 0; c < parent.size(); c++)
        parent[c] = c;
    constexpr std::array<std::array<std::int64_t, 2>, 4> ahead = {
        {{0, 1}, {1, -1}, {1, 0}, {1, 1}}};
    for (std::uint32_t c = 0; c < keys.size(); c++)
    {
        const auto [i, j] = cell_of(keys[c]);
        for (const auto &step : ahead)
        {
            const std::uint64_t neighbour = key_of(i + step[0], j + step[1]);
            const auto found = std::lower_bound(keys.begin(), keys.end(), neighbour);
            if (found == keys.end() || *found != neighbour)
                continue;
            const std::uint32_t root_a = root_of(parent, c);
            const std::uint32_t root_b =
                root_of(parent, static_cast<std::uint32_t>(found - keys.begin()));
            parent[std::max(root_a, root_b)] = std::min(root_a, root_b);
        }
    }
    for (std::uint32_t c = 0; c < parent.size(); c++)
        parent[c] = root_of(parent, c);
    return parent;
}

/// Chooses the object's piece: the piece of the occupied cell nearest to o, the distance from o to
/// the cell's square, 0 for o's own cell, exact in integers.
/// @param keys [in] the occupied cells' keys, sorted, at least one.
/// @param roots [in] each cell's root (pieces_of()).
/// @return the object's root; the first cell in sorted order's on a tie.
std::uint32_t object_of(const std::vector<std::uint64_t> &keys,
                        const std::vector<std::uint32_t> &roots)
{
    std::uint32_t nearest = 0;
    std::int64_t nearest_distance = std::numeric_limits<std::int64_t>::max();
    for (std::uint32_t c = 0; c < keys.size(); c++)
    {
        const auto [i, j] = cell_of(keys[c]);
        const std::int64_t di = std::max<std::int64_t>({i, -i - 1, 0});
        const std::int64_t dj = std::max<std::int64_t>({j, -j - 1, 0});
        if (di * di + dj * dj < nearest_distance)
        {
            nearest = c;
            nearest_distance = di * di + dj * dj;
        }
    }
    return roots[nearest];
}

/// The z component of the cross product of (b - o) and (c - o) in the plane.
/// @param o [in] the common point, (a, b) in m.
/// @param b [in] the first point, (a, b) in m.
/// @param c [in] the second point, (a, b) in m.
/// @return > 0 when o, b, c turn counterclockwise, 0 on a line; in m².
double turn_of(const Eigen::Vector2d &o, const Eigen::Vector2d &b, const Eigen::Vector2d &c)
{
    return (b.x() - o.x()) * (c.y() - o.y()) - (b.y() - o.y()) * (c.x() - o.x());
}

/// Builds the convex hull of points in the plane by Andrew's monotone chain [S: A. M. Andrew,
/// Information Processing Letters 9(5), 1979].
/// @param points [in] the points, (a, b) in the plane's frame, in m; sorted and taken over.
/// @return the hull's vertices counterclockwise, in the same frame and unit, without points on its
///         edges; one vertex for points that all coincide, two for points on a line.
std::vector<Eigen::Vector2d> hull_of(std::vector<Eigen::Vector2d> points)
{
    std::sort(points.begin(), points.end(), [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
        return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
    });
    points.erase(std::unique(points.begin(), points.end()), points.end());
    if (points.size() < 3)
        return points;
    std::vector<Eigen::Vector2d> hull(2 * points.size());
    std::size_t k = 0;
    for (const Eigen::Vector2d &p : points)
    {
        while (k >= 2 && turn_of(hull[k - 2], hull[k - 1], p) <= 0.0)
            k--;
        hull[k++] = p;
    }
    const std::size_t lower = k + 1;
    for (std::size_t i = points.size() - 1; i-- > 0;)
    {
        while (k >= lower && turn_of(hull[k - 2], hull[k - 1], points[i]) <= 0.0)
            k--;
        hull[k++] = points[i];
    }
    hull.resize(k - 1);
    return hull;
}

/// The least rectangle around a footprint, in the plane's frame (a, b).
struct footprint_rectangle
{
    Eigen::Vector2d along;  ///< the direction of its first side, unit, in (a, b)
    Eigen::Vector2d centre; ///< its centre, (a, b) in m
    double length_along;    ///< its extent along `along`, in m, >= 0
    double length_across;   ///< its extent across `along`, in m, >= 0
};

/// Finds the minimum-area rectangle around a convex hull: one of its sides lies along an edge of
/// the hull [S: H. Freeman, R. Shapira, Communications of the ACM 18(7), 1975], so each edge is
/// tried with every vertex projected onto it and across it.
/// @param hull [in] the hull's vertices, (a, b) in the plane's frame, in m, at least one.
/// @return the rectangle of least area, in the same frame, the first edge's on a tie; along (1, 0)
///         when the hull is one point.
footprint_rectangle least_rectangle(const std::vector<Eigen::Vector2d> &hull)
{
    footprint_rectangle best{Eigen::Vector2d::UnitX(), hull.front(), 0.0, 0.0};
    double best_area = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < hull.size(); k++)
    {
        const Eigen::Vector2d edge = hull[(k + 1) % hull.size()] - hull[k];
        const double length = edge.norm();
        if (!(length > 0.0))
            continue;
        const Eigen::Vector2d along = edge / length;
        const Eigen::Vector2d across(-along.y(), along.x());
        double low_along = std::numeric_limits<double>::infinity();
        double high_along = -low_along;
        double low_across = low_along;
        double high_across = -low_along;
        for (const Eigen::Vector2d &p : hull)
        {
            low_along = std::min(low_along, along.dot(p));
            high_along = std::max(high_along, along.dot(p));
            low_across = std::min(low_across, across.dot(p));
            high_across = std::max(high_across, across.dot(p));
        }
        const double area = (high_along - low_along) * (high_across - low_across);
        if (area < best_area)
        {
            best_area = area;
            best = {along,
                    0.5 * (low_along + high_along) * along +
                        0.5 * (low_across + high_across) * across,
                    high_along - low_along, high_across - low_across};
        }
    }
    return best;
}

/// Builds the box around the object's piece (docs/methods/separation.md, "The box"): the least
/// rectangle around its footprint's hull grown by μ, its x axis along the longer side, its height
/// the piece's highest point plus μ, its bottom in the plane.
/// @param shape [in] the footprint.
/// @param roots [in] each cell's root.
/// @param object [in] the object's root.
/// @param plane [in] the support plane.
/// @param frame [in] the plane's frame.
/// @param voxel_m [in] v, in m.
/// @param object_points [out] the points of the object's piece.
/// @return the box, in the models' frame.
separation_box box_of(const footprint &shape, const std::vector<std::uint32_t> &roots,
                      std::uint32_t object, const support_plane &plane, const plane_frame &frame,
                      double voxel_m, std::size_t &object_points)
{
    std::vector<Eigen::Vector2d> points;
    double top_m = 0.0;
    for (std::uint32_t c = 0; c < shape.keys.size(); c++)
    {
        if (roots[c] != object)
            continue;
        for (std::size_t e = shape.first[c]; e < shape.first[c + 1]; e++)
        {
            const Eigen::Vector3d &s = shape.standing[shape.items[e]];
            points.emplace_back(s.x(), s.y());
            top_m = std::max(top_m, s.z());
        }
    }
    object_points = points.size();
    const footprint_rectangle rectangle = least_rectangle(hull_of(std::move(points)));
    const double margin_m = separation_margin_voxels * voxel_m;
    const double length_along = rectangle.length_along + 2.0 * margin_m;
    const double length_across = rectangle.length_across + 2.0 * margin_m;
    const Eigen::Vector2d x_axis = length_along >= length_across
                                       ? rectangle.along
                                       : Eigen::Vector2d(-rectangle.along.y(), rectangle.along.x());
    const Eigen::Vector3d x = x_axis.x() * frame.e1 + x_axis.y() * frame.e2;
    separation_box box{Eigen::Isometry3d::Identity(),
                       Eigen::Vector3d(std::max(length_along, length_across),
                                       std::min(length_along, length_across), top_m + margin_m)};
    box.pose.linear().col(0) = x;
    box.pose.linear().col(1) = plane.normal.cross(x);
    box.pose.linear().col(2) = plane.normal;
    box.pose.translation() =
        frame.origin_m + rectangle.centre.x() * frame.e1 + rectangle.centre.y() * frame.e2;
    return box;
}
} // namespace

separation_proposal propose_separation(std::span<const model_point> full_model,
                                       std::span<const camera_pose> trajectory,
                                       const gravity_estimate &gravity,
                                       const separation_parameters &parameters)
{
    separation_proposal result;
    result.message = refusal_of(full_model, trajectory, gravity, parameters);
    if (!result.message.empty())
        return result;
    if (gravity.outcome != gravity_outcome::estimated)
    {
        result.end = separation_end::no_gravity;
        result.message = reason_of(gravity);
        return result;
    }
    const view_result views = views_of(trajectory, gravity.up, parameters.max_depth_m);
    if (!views.geometry)
    {
        result.end = separation_end::views_do_not_meet;
        result.message = views.reason;
        return result;
    }
    result.up = views.geometry->up;
    result.meeting_point_m = views.geometry->meeting_point_m;
    result.window_m = views.geometry->window_m;
    const double v = parameters.voxel_size_m;

    const std::vector<Eigen::Vector3d> window = window_of(full_model, *views.geometry);
    std::mt19937_64 generator(sample_seed);
    const std::vector<Eigen::Vector3d> sample = sample_of(window, generator);
    const std::optional<support_plane> hypothesis =
        best_hypothesis(sample, *views.geometry, separation_band_voxels * v, generator);
    if (!hypothesis)
    {
        result.end = separation_end::no_plane;
        result.message =
            "no plane under the views' meeting point: no plane through three of " +
            std::to_string(sample.size()) + " sampled points of the window lies within " +
            text_of(separation_largest_tilt_deg) + " degrees of up with the meeting point above it";
        return result;
    }
    const plane_result fitted = fitted_plane(window, *hypothesis, *views.geometry, v);
    result.plane_points = fitted.points;
    if (!fitted.plane)
    {
        result.end = separation_end::no_plane;
        result.message = fitted.reason;
        return result;
    }
    const support_plane &plane = *fitted.plane;

    const plane_frame frame = frame_of(plane, result.meeting_point_m);
    const footprint shape = footprint_of(window, plane, frame, v);
    if (shape.keys.empty())
    {
        result.end = separation_end::nothing_on_plane;
        result.message = "nothing stands on the plane within " + text_of(result.window_m) +
                         " m of the views' meeting point";
        return result;
    }
    const std::vector<std::uint32_t> roots = pieces_of(shape.keys);
    result.box =
        box_of(shape, roots, object_of(shape.keys, roots), plane, frame, v, result.object_points);
    result.plane = plane;
    result.end = separation_end::proposed;
    return result;
}
} // namespace gca
