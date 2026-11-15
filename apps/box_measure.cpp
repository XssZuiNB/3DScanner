// SPDX-License-Identifier: 0BSD
/// @file box_measure.cpp
/// The box fitted to an object model: its axes from the normals, its faces, their planes and
/// spreads, its edges, the refusals and the report's lines (see box_measure.hpp).

#include "box_measure.hpp"

#include "separation/plane_fit.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <numbers>
#include <sstream>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

namespace gca
{
namespace
{
/// The half-angle of a face's cone, in degrees [?: a choice]: a scanned box's normals scatter
/// by a few degrees about its face's (the test tilts them by up to 5 deg), and a sphere puts
/// 3 (1 - cos 20 deg) = 18.1 % of its normals within 20 deg of six axes [S: a spherical cap's
/// area]. Under 45 deg, so that a normal lies in one face's cone at most.
constexpr double face_angle_deg = 20.0;

/// The cosine of face_angle_deg: a normal lies in an axis's cone when its dot product with the
/// axis, or with the axis's opposite, is at least this.
const double cos_face_angle = std::cos(face_angle_deg * std::numbers::pi / 180.0);

/// The normals the axes' search starts from, spread evenly over the points' order
/// [?: a choice]: an axis needs one of them on a face across it.
constexpr std::size_t seed_count = 64;

/// The steps of the mean shift that moves an axis to its normals' mean [?: a choice]: the axes
/// only sort the normals into the faces' cones, 20 deg wide, and the faces' planes give the
/// edges, so a step more or less moves no edge measurably [?].
constexpr int mean_shift_steps = 8;

/// How far from its median offset a face keeps a point, in robust standard deviations
/// [?: a choice]: a normal distribution has 6.3e-5 of its draws farther out [S: its tail at
/// 4 sigma], so the spread loses nothing measurable, while a remnant of the support plane lies a
/// box's height below the top.
constexpr double band_sigmas = 4.0;

/// A normal distribution's standard deviation over its median absolute deviation, 1 / 0.6745
/// [S: the normal distribution's third quartile lies 0.6745 sigma above its median].
constexpr double mad_to_sigma = 1.4826;

/// The least share of the model's points a face keeps to count [?: a choice]: the narrowest face
/// of a box of 300 x 200 x 30 mm holds 6.7 % of the five faces' area [S: computed, 6000 of
/// 90000 mm^2], and the stray normals near a face that was not scanned hold far less [?].
constexpr double min_face_share = 0.01;

/// The fewest points a face keeps to count: a plane needs three (separation/plane_fit.hpp).
constexpr std::size_t min_face_points = 3;

/// The faces a scanned box shows: its top and four sides (ADR 0003).
constexpr int faces_of_a_box = 5;

/// m in mm.
constexpr double mm_per_m = 1000.0;

/// The faces of a box: one along +axis and one along -axis for each of the three axes.
constexpr int face_count = 6;

/// Gives a face's index.
/// @param axis [in] the axis it lies across, 0 to 2.
/// @param positive [in] whether it lies along +axis rather than -axis.
/// @return the index, 0 to face_count - 1.
constexpr int face_along(int axis, bool positive)
{
    return 2 * axis + (positive ? 0 : 1);
}

/// Gives the axis a face lies across.
/// @param face [in] the face's index.
/// @return the axis, 0 to 2.
constexpr int axis_of(int face)
{
    return face / 2;
}

/// Gives whether a face lies along +axis.
/// @param face [in] the face's index.
/// @return true along +axis, false along -axis.
constexpr bool is_positive(int face)
{
    return face % 2 == 0;
}

/// Gives the face across the box from a face.
/// @param face [in] the face's index.
/// @return the other face across the same axis.
constexpr int opposite(int face)
{
    return face_along(axis_of(face), !is_positive(face));
}

/// Counts the normals in an axis's cone, of either sign.
/// @param normals [in] the unit normals.
/// @param axis [in] the axis, a unit vector, or zero, whose cone holds no normal.
/// @return the normals within face_angle_deg of the axis or of its opposite.
std::size_t count_in_cone(const std::vector<Eigen::Vector3d> &normals, const Eigen::Vector3d &axis)
{
    std::size_t count = 0;
    for (const Eigen::Vector3d &normal : normals)
        if (std::abs(normal.dot(axis)) >= cos_face_angle)
            count++;
    return count;
}

/// Moves an axis to the mean of the normals in its cone, each turned to the axis's side, for
/// mean_shift_steps steps, held square to an axis found before.
/// @param normals [in] the unit normals.
/// @param axis [in] the axis to start from, a unit vector, square to square_to if given.
/// @param square_to [in] the axis found before, a unit vector; nothing for the first axis.
/// @return the axis, a unit vector; the one given when its cone holds no normal.
Eigen::Vector3d mean_shift(const std::vector<Eigen::Vector3d> &normals, Eigen::Vector3d axis,
                           const std::optional<Eigen::Vector3d> &square_to)
{
    for (int step = 0; step < mean_shift_steps; step++)
    {
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        for (const Eigen::Vector3d &normal : normals)
        {
            const double along = normal.dot(axis);
            if (std::abs(along) >= cos_face_angle)
                sum += along >= 0.0 ? normal : Eigen::Vector3d(-normal);
        }
        if (square_to)
            sum -= sum.dot(*square_to) * *square_to;
        if (sum.norm() == 0.0)
            break;
        axis = sum.normalized();
    }
    return axis;
}

/// Finds one of the box's axes: of seed_count normals spread evenly over the points' order, the
/// one whose cone holds the most normals, mean-shifted. For the second axis each seed is first
/// turned square to the first axis; a seed along the first axis leaves a short remainder of any
/// direction, which is judged by its cone like any other, or none, whose zero vector
/// (Eigen's normalized() gives zero for zero) holds no normal.
/// @param normals [in] the unit normals.
/// @param square_to [in] the first axis, a unit vector; nothing when the first is sought.
/// @return the axis, a unit vector, square to square_to if given; when no seed's cone holds a
///     normal (an empty model, or one whose normals all lie along the first axis), an axis square
///     to square_to, or z, whose faces then do not count.
Eigen::Vector3d find_axis(const std::vector<Eigen::Vector3d> &normals,
                          const std::optional<Eigen::Vector3d> &square_to)
{
    Eigen::Vector3d best = square_to ? square_to->unitOrthogonal() : Eigen::Vector3d::UnitZ();
    std::size_t best_count = 0;
    const std::size_t seeds = std::min(seed_count, normals.size());
    for (std::size_t s = 0; s < seeds; s++)
    {
        Eigen::Vector3d seed = normals[s * normals.size() / seeds];
        if (square_to)
            seed = (seed - seed.dot(*square_to) * *square_to).normalized();
        const std::size_t count = count_in_cone(normals, seed);
        if (count > best_count)
        {
            best = seed;
            best_count = count;
        }
    }
    return mean_shift(normals, best, square_to);
}

/// The median of a set of values; the upper one of the middle two for an even count.
/// @param values [in,out] the values, reordered.
/// @pre values is not empty: every caller passes at least one value; for an empty set
///     std::nth_element would be given an end as the middle, and the read past it is undefined.
/// @return the median.
double median_of(std::vector<double> &values)
{
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

/// Keeps the points of a face's cone whose offset along its axis lies within band_sigmas robust
/// standard deviations of the median offset.
/// @param positions [in] the model's positions, in m.
/// @param indices [in] the cone's points, as indices into positions.
/// @param axis [in] the face's axis, a unit vector.
/// @return the positions kept, in m, in the cone's order; empty for an empty cone.
std::vector<Eigen::Vector3d> keep_band(const std::vector<Eigen::Vector3d> &positions,
                                       const std::vector<std::size_t> &indices,
                                       const Eigen::Vector3d &axis)
{
    if (indices.empty())
        return {};
    std::vector<double> offsets;
    offsets.reserve(indices.size());
    for (const std::size_t i : indices)
        offsets.push_back(positions[i].dot(axis));
    std::vector<double> deviations = offsets;
    const double median = median_of(deviations);
    for (double &deviation : deviations)
        deviation = std::abs(deviation - median);
    const double band = band_sigmas * mad_to_sigma * median_of(deviations);
    std::vector<Eigen::Vector3d> kept;
    kept.reserve(indices.size());
    for (std::size_t j = 0; j < indices.size(); j++)
        if (std::abs(offsets[j] - median) <= band)
            kept.push_back(positions[indices[j]]);
    return kept;
}

/// A face that counts: its points and its plane.
struct face_fit
{
    std::vector<Eigen::Vector3d> points_m; ///< the points it keeps, in m
    plane_fit plane;                       ///< their plane, its normal pointing out of the box
};

/// Fits a face's plane and turns its normal out of the box.
/// @param points_m [in] the points the face keeps, in m, at least min_face_points; taken over,
///     the face holds them.
/// @param outward [in] the face's direction, its axis or the axis's opposite, a unit vector.
/// @return the face with its points and its plane.
face_fit fit_face(std::vector<Eigen::Vector3d> points_m, const Eigen::Vector3d &outward)
{
    // At least three points: fit_plane() gives a plane.
    plane_fit plane = *fit_plane(points_m);
    if (plane.normal.dot(outward) < 0.0)
        plane.normal = -plane.normal;
    return {std::move(points_m), plane};
}

/// Describes a face for the measurement.
/// @param face [in] the face.
/// @return its count, centroid, outward normal and spread.
box_face face_of(const face_fit &face)
{
    const double count = static_cast<double>(face.points_m.size());
    // The least eigenvalue is the sum of the squared distances from the plane; rounding can
    // leave it a little below 0.
    const double mean_square = std::max(0.0, face.plane.eigenvalues(0)) / count;
    return {face.points_m.size(), face.plane.centroid, face.plane.normal,
            std::sqrt(mean_square) * mm_per_m};
}

/// Measures the edge between two opposite faces: the distance between their planes along
/// their mean outward normal, between their centroids.
/// @param plus [in] the face along +axis.
/// @param minus [in] the face along -axis.
/// @return the edge, in mm.
double edge_between(const face_fit &plus, const face_fit &minus)
{
    const Eigen::Vector3d across = (plus.plane.normal - minus.plane.normal).normalized();
    return (plus.plane.centroid - minus.plane.centroid).dot(across) * mm_per_m;
}

/// How far the four side faces reach along the top's normal.
struct side_reach
{
    double foot_m; ///< the median of the four sides' lowest points' offsets, in m
    double top_m;  ///< the median of the four sides' highest points' offsets, in m
};

/// Measures how far the four side faces reach along the top's normal: the median of their lowest
/// points (median_of(): the upper of the middle two), so that one side seen only in part does not
/// move the foot [?: a choice], and the median of their highest points.
/// @param up [in] the top's outward normal, a unit vector.
/// @param sides [in] the four faces square to the top; borrowed for the call, each holding at
///     least min_face_points points.
/// @return the foot and the sides' top, as offsets along up, in m.
side_reach reach_of(const Eigen::Vector3d &up, const std::array<const face_fit *, 4> &sides)
{
    std::vector<double> lowest;
    std::vector<double> highest;
    for (const face_fit *face : sides)
    {
        const auto [low, high] =
            std::minmax_element(face->points_m.begin(), face->points_m.end(),
                                [&up](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                                    return a.dot(up) < b.dot(up);
                                });
        lowest.push_back(low->dot(up));
        highest.push_back(high->dot(up));
    }
    return {median_of(lowest), median_of(highest)};
}

/// Writes one face's line of the report.
/// @param out [in,out] the report.
/// @param name [in] the face's name.
/// @param face [in] the face.
void write_face(std::ostringstream &out, const char *name, const box_face &face)
{
    out << "\n" << name << ": " << face.points << " points, spread " << face.spread_mm << " mm";
}
} // namespace

box_result measure_box(std::span<const model_point> model) noexcept
{
    std::vector<Eigen::Vector3d> positions;
    std::vector<Eigen::Vector3d> normals;
    positions.reserve(model.size());
    normals.reserve(model.size());
    for (const model_point &point : model)
    {
        positions.push_back(point.position_m.cast<double>());
        normals.push_back(point.normal.cast<double>());
    }

    // 1. The axes from the normals.
    std::array<Eigen::Vector3d, 3> axes;
    axes[0] = find_axis(normals, std::nullopt);
    axes[1] = find_axis(normals, axes[0]);
    axes[2] = axes[0].cross(axes[1]);

    // 2. The faces: the cones, the band about each median offset, the faces that count.
    std::array<std::vector<std::size_t>, face_count> cones;
    for (std::size_t i = 0; i < normals.size(); i++)
    {
        for (int k = 0; k < 3; k++)
        {
            const double along = normals[i].dot(axes[k]);
            if (along >= cos_face_angle)
                cones[face_along(k, true)].push_back(i);
            else if (along <= -cos_face_angle)
                cones[face_along(k, false)].push_back(i);
        }
    }
    std::array<std::vector<Eigen::Vector3d>, face_count> kept;
    std::array<bool, face_count> is_counted{};
    std::size_t points_on_faces = 0;
    int faces_shown = 0;
    for (int f = 0; f < face_count; f++)
    {
        kept[f] = keep_band(positions, cones[f], axes[axis_of(f)]);
        const std::size_t n = kept[f].size();
        is_counted[f] =
            n >= min_face_points &&
            static_cast<double>(n) >= min_face_share * static_cast<double>(model.size());
        if (is_counted[f])
        {
            points_on_faces += n;
            faces_shown++;
        }
    }

    // The refusals. At least half of the points on the faces that count [?: a choice]: a sphere
    // puts 18.1 % there (face_angle_deg), a box all but those on the rounding of its edges [?];
    // a shape between the two can pass, and its faces' spreads then show it.
    if (2 * points_on_faces < model.size())
    {
        std::ostringstream reason;
        reason << std::fixed << std::setprecision(1)
               << 100.0 * static_cast<double>(points_on_faces) / static_cast<double>(model.size())
               << " % of the " << model.size()
               << " points lie on the faces found; a box needs at least half";
        return {std::nullopt, reason.str()};
    }
    // Exactly five faces count past this refusal, one of them missing: it also guards every
    // dereference of faces[] below, which reads only the five that count.
    if (faces_shown != faces_of_a_box)
        return {std::nullopt, "a box shows five faces, its top and four sides, as the side it "
                              "stands on is not scanned (ADR 0003); this model shows " +
                                  std::to_string(faces_shown)};

    // 3. The planes. The face that does not count is the one the box stands on; its axis is the
    // height's, the other two run across the four sides.
    const int bottom = static_cast<int>(std::find(is_counted.begin(), is_counted.end(), false) -
                                        is_counted.begin());
    std::array<std::optional<face_fit>, face_count> faces;
    for (int f = 0; f < face_count; f++)
        if (is_counted[f])
            faces[f] =
                fit_face(std::move(kept[f]),
                         is_positive(f) ? axes[axis_of(f)] : Eigen::Vector3d(-axes[axis_of(f)]));
    const face_fit &top = *faces[opposite(bottom)];
    const int a = (axis_of(bottom) + 1) % 3;
    const int b = (axis_of(bottom) + 2) % 3;
    const face_fit &a_plus = *faces[face_along(a, true)];
    const face_fit &a_minus = *faces[face_along(a, false)];
    const face_fit &b_plus = *faces[face_along(b, true)];
    const face_fit &b_minus = *faces[face_along(b, false)];

    // The top has to lie at the upper end of the sides. A remnant of the support plane with more
    // points than the top takes the top's band (the median lies in it) and lies at their foot
    // [M 2026-11-15: ctest box_measure, a remnant 80 mm wide around a box 50 mm high, measured
    // 1.0 mm high before this refusal].
    const Eigen::Vector3d &up = top.plane.normal;
    const side_reach reach = reach_of(up, {&a_plus, &a_minus, &b_plus, &b_minus});
    const double top_m = top.plane.centroid.dot(up);
    if (2.0 * (top_m - reach.foot_m) < reach.top_m - reach.foot_m)
        return {std::nullopt, "the points facing up lie in the lower half of the sides: a "
                              "remnant of the support plane outweighs the top"};

    // 4. The edges: the longer across the four sides is the length, between the ends.
    const double across_a = edge_between(a_plus, a_minus);
    const double across_b = edge_between(b_plus, b_minus);
    const bool a_is_length = across_a >= across_b;
    box_measurement measurement;
    measurement.length_mm = std::max(across_a, across_b);
    measurement.width_mm = std::min(across_a, across_b);
    measurement.height_mm = (top_m - reach.foot_m) * mm_per_m;
    measurement.top = face_of(top);
    measurement.ends = a_is_length ? std::array{face_of(a_plus), face_of(a_minus)}
                                   : std::array{face_of(b_plus), face_of(b_minus)};
    measurement.sides = a_is_length ? std::array{face_of(b_plus), face_of(b_minus)}
                                    : std::array{face_of(a_plus), face_of(a_minus)};
    measurement.points = model.size();
    measurement.points_on_faces = points_on_faces;
    return {measurement, ""};
}

std::string box_report(const std::string &name, const box_measurement &measurement)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "box_fit: " << name << ": " << measurement.points << " points, "
        << measurement.points_on_faces << " on five faces";
    out << "\nlength: " << measurement.length_mm << " mm";
    out << "\nwidth: " << measurement.width_mm << " mm";
    out << "\nheight: " << measurement.height_mm << " mm";
    write_face(out, "top", measurement.top);
    write_face(out, "side 1", measurement.sides[0]);
    write_face(out, "side 2", measurement.sides[1]);
    write_face(out, "end 1", measurement.ends[0]);
    write_face(out, "end 2", measurement.ends[1]);
    return out.str();
}
} // namespace gca
