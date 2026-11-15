// SPDX-License-Identifier: 0BSD
/// @file test_box_measure.cpp
/// Offline test of the acceptance tool's box fit (box_measure.hpp) and of the program box_fit on
/// object models the test builds: the five faces of a box of each object class's size, a point
/// every voxel of the class's full model, in four orientations and up to 2.5 m away, each face
/// with its own noise along its normal and its normals tilted at random, from a fixed seed. The
/// three edges agree with the box's, each face's spread with its noise, each face's normal with
/// the box's. They stay so with an end seen only in its upper half, with a few stray normals
/// facing down and with a remnant of the support plane around the box's foot; a remnant that
/// outweighs the top is refused. A sphere, a box missing a side beside its bottom, a box with all
/// six faces, one plane and an empty model are refused with the reason; the report's lines are
/// pinned. The program, given as the test's argument, prints a written box's lines with exit 0,
/// refuses a written sphere with exit 1 and a missing file with exit 2. Needs no GPU and no
/// camera.

#include "box_measure.hpp"
#include "check.hpp"
#include "program_run.hpp"
#include "scan/model_file.hpp"
#include "synthetic_source.hpp"
#include "test_files.hpp"
#include "transform_checks.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Geometry>

using gca_test::check;
using gca_test::check_max_error;
using gca_test::check_refusal;
using gca_test::check_summary;
using gca_test::max_error;
using gca_test::program_run;
using gca_test::run_program;

namespace
{
/// A face of a box the test builds, in the box's frame: x across the length, y across the
/// width, z up from the foot at z = 0.
enum box_face_id : int
{
    plus_x,  ///< an end
    minus_x, ///< the other end
    plus_y,  ///< a side
    minus_y, ///< the other side
    top,     ///< the top, at z = height
    bottom,  ///< the side the box stands on, at z = 0, which a scan does not hold
};

/// The number of a box's faces.
constexpr int face_count = 6;

/// The names of the faces, for the check lines.
constexpr std::array<const char *, face_count> face_names = {"+x end",  "-x end", "+y side",
                                                             "-y side", "top",    "bottom"};

/// Gives the axis of the box's frame a face lies across.
/// @param face [in] the face, by box_face_id.
/// @return 0 for x, 1 for y, 2 for z.
int axis_of(int face)
{
    return face / 2;
}

/// Gives the side of its axis a face lies on.
/// @param face [in] the face, by box_face_id.
/// @return +1 for the face along +axis, -1 for the one along -axis.
double sign_of(int face)
{
    return face % 2 == 0 ? 1.0 : -1.0;
}

/// A box the test builds.
struct box_spec
{
    double length_mm;  ///< its edge along x, the longest, in mm
    double width_mm;   ///< its edge along y, in mm
    double height_mm;  ///< its edge along z, in mm
    double spacing_mm; ///< the points' spacing on each face, the object class's final voxel, mm
    /// Each face's noise along its normal: the standard deviation, in mm, by box_face_id.
    std::array<double, face_count> sigma_mm;
    std::array<bool, face_count> built; ///< which faces carry points, by box_face_id
    /// For each face, by box_face_id, the height in the box's frame below which it carries no
    /// points, in mm: 0 for a face seen whole.
    std::array<double, face_count> seen_from_z_mm{};
};

/// The faces a scan holds: all but the bottom.
constexpr std::array<bool, face_count> five_faces = {true, true, true, true, true, false};

/// The largest tilt of a point's normal from its face's, in degrees [?: a choice]: the fusion's
/// normals scatter about a flat face's.
constexpr double normal_tilt_deg = 5.0;

/// A face's outward normal in the box's frame.
/// @param face [in] the face, by box_face_id.
/// @return the unit normal.
Eigen::Vector3d normal_of(int face)
{
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    normal(axis_of(face)) = sign_of(face);
    return normal;
}

/// Appends one point of a model in the model's frame.
/// @param model [in,out] the model.
/// @param pose [in] the box's frame to the model's, in m.
/// @param position_mm [in] the point in the box's frame, in mm.
/// @param normal [in] its unit normal in the box's frame.
void add_point(std::vector<gca::model_point> &model, const Eigen::Isometry3d &pose,
               const Eigen::Vector3d &position_mm, const Eigen::Vector3d &normal)
{
    model.push_back({(pose * (position_mm / 1000.0)).cast<float>(),
                     (pose.linear() * normal).normalized().cast<float>(),
                     {200, 120, 40}});
}

/// Builds the faces of a box: on each face built, a grid of points from edge to edge at about
/// the spacing, from the height the face is seen from up, each moved along the face's normal by
/// a draw of the face's noise, its normal tilted by an angle drawn up to normal_tilt_deg towards
/// a direction drawn in the face, all from one generator of a fixed seed in the points' order.
/// @param box [in] the box.
/// @param pose [in] the box's frame to the model's: the box's foot at the origin, in m.
/// @param seed [in] the generator's seed.
/// @return the model's points, face by face.
std::vector<gca::model_point> build_box(const box_spec &box, const Eigen::Isometry3d &pose,
                                        std::uint64_t seed)
{
    const Eigen::Vector3d low(-box.length_mm / 2, -box.width_mm / 2, 0.0);
    const Eigen::Vector3d high(box.length_mm / 2, box.width_mm / 2, box.height_mm);
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<gca::model_point> model;
    for (int face = 0; face < face_count; face++)
    {
        if (!box.built[face])
            continue;
        const int k = axis_of(face);
        const int u = (k + 1) % 3;
        const int v = (k + 2) % 3;
        const Eigen::Vector3d normal = normal_of(face);
        std::normal_distribution<double> noise(0.0, box.sigma_mm[face]);
        const int steps_u = static_cast<int>(std::lround((high(u) - low(u)) / box.spacing_mm));
        const int steps_v = static_cast<int>(std::lround((high(v) - low(v)) / box.spacing_mm));
        for (int i = 0; i <= steps_u; i++)
        {
            for (int j = 0; j <= steps_v; j++)
            {
                Eigen::Vector3d p;
                p(k) = sign_of(face) > 0.0 ? high(k) : low(k);
                p(u) = low(u) + (high(u) - low(u)) * i / steps_u;
                p(v) = low(v) + (high(v) - low(v)) * j / steps_v;
                if (p.z() < box.seen_from_z_mm[face])
                    continue;
                p += noise(generator) * normal;
                const double tilt = normal_tilt_deg * unit(generator) * std::numbers::pi / 180.0;
                const double turn = 2.0 * std::numbers::pi * unit(generator);
                Eigen::Vector3d toward = Eigen::Vector3d::Zero();
                toward(u) = std::cos(turn);
                toward(v) = std::sin(turn);
                add_point(model, pose, p, std::cos(tilt) * normal + std::sin(tilt) * toward);
            }
        }
    }
    return model;
}

/// The boxes of the two object classes [?: a choice of size within the glossary's 10 cm to 50 cm
/// and 0.5 m to 2 m], a point every final voxel (scan/object_class.cpp: 1 mm and 4 mm), each
/// face's noise its own, so that a spread given to the wrong face shows [?: a choice of noise].
struct object_class_box
{
    const char *name;      ///< the object class
    box_spec box;          ///< the box
    double edge_tolerance; ///< how far an edge may lie from the box's, mm
};

/// The tabletop object's box: 300 x 200 x 150 mm, noise 0.5 to 1.5 mm, 212105 points.
const box_spec tabletop_box = {300.0,     200.0, 150.0, 1.0, {0.5, 0.75, 1.0, 1.25, 1.5, 1.0},
                               five_faces};

/// The large object's box: 1200 x 800 x 600 mm, noise 2 to 6 mm, 212105 points.
const box_spec large_box = {1200.0, 800.0, 600.0, 4.0, {2.0, 3.0, 4.0, 5.0, 6.0, 4.0}, five_faces};

/// How far an edge of the tabletop object's box may lie from the box's, mm [M 2026-11-15: this
/// test, at most 0.0141 mm over the four orientations, 0.0142 mm with an end seen in part,
/// 0.0123 mm with the strays, 0.0043 mm with the remnant]: about three times that, 1/40 of R21's
/// 2 mm.
constexpr double tabletop_edge_tolerance_mm = 0.05;

/// How far an edge of the large object's box may lie from the box's, mm [M 2026-11-15: this
/// test, at most 0.069 mm over the four orientations]: about three times that, 1/50 of R21's
/// 10 mm.
constexpr double large_edge_tolerance_mm = 0.2;

/// How far a face's spread may lie from its noise, over the noise [M 2026-11-15: this test, at
/// most 0.0078 for the tabletop object's box, 0.0059 for the large object's, 0.0083 with the
/// remnant]: the spread of a sample's standard deviation over 30000 points is about
/// sqrt(1 / 60000) = 0.0041 of it [S: a normal sample's standard deviation], and the tolerance
/// is twice the largest.
constexpr double spread_tolerance = 0.02;

/// How far a face's normal may turn from the box's, deg [M 2026-11-15: this test, at most
/// 0.0138 deg]: about three times that.
constexpr double normal_tolerance_deg = 0.05;

/// The orientations of a box in the model's frame (x right, y down, z forward from the first
/// camera): first upright before the camera, the box's axes along the frame's, so that the
/// normals lie on the frame's axes; then turned about three slanted axes, the last 2.5 m away
/// [?: a choice; CODING_STANDARDS.md "Numerics" asks for points 2 m away].
/// @return the box's frames to the model's.
std::vector<Eigen::Isometry3d> orientations()
{
    using gca_test::make_transform;
    return {make_transform({1.0, 0.0, 0.0}, 90.0, {0.05, 0.25, 0.9}),
            make_transform({1.0, 2.0, 3.0}, 47.0, {-0.3, 0.2, 1.4}),
            make_transform({-2.0, 1.0, 0.5}, 123.0, {0.4, -0.3, 2.0}),
            make_transform({0.3, -1.0, 0.2}, 200.0, {-0.2, 0.1, 2.5})};
}

/// Finds the box's face whose normal a fitted face's lies nearest.
/// @param pose [in] the box's frame to the model's.
/// @param normal [in] the fitted face's outward normal, in the model's frame.
/// @return the face, by box_face_id.
int nearest_face(const Eigen::Isometry3d &pose, const Eigen::Vector3d &normal)
{
    int nearest = 0;
    for (int face = 1; face < face_count; face++)
        if ((pose.linear() * normal_of(face)).dot(normal) >
            (pose.linear() * normal_of(nearest)).dot(normal))
            nearest = face;
    return nearest;
}

/// How far a measurement lies from its box, over the cases of one object class.
struct class_errors
{
    max_error edge_mm;         ///< an edge's distance from the box's, mm
    max_error spread_relative; ///< a spread's distance from its face's noise, over the noise
    max_error normal_deg;      ///< a face's normal's angle from the box's, deg
};

/// Checks one fitted face: that its normal is the expected face's and that its spread is that
/// face's noise.
/// @param what [in] the case and the face's name in the report.
/// @param face [in] the fitted face.
/// @param expected [in] the box's face it has to be, by box_face_id.
/// @param box [in] the box.
/// @param pose [in] the box's frame to the model's.
/// @param errors [in,out] the class's errors.
void check_face(const std::string &what, const gca::box_face &face, int expected,
                const box_spec &box, const Eigen::Isometry3d &pose, class_errors &errors)
{
    const int found = nearest_face(pose, face.normal);
    const Eigen::Vector3d known = pose.linear() * normal_of(expected);
    // The angle in double here: stage_checks.hpp's takes CUDA's float3 and pulls in CUDA.
    const double angle_deg =
        std::acos(std::clamp(known.dot(face.normal), -1.0, 1.0)) * 180.0 / std::numbers::pi;
    std::printf("  %s: %zu points, spread %.4f mm (noise %.2f), %.4f deg from the %s\n",
                what.c_str(), face.points, face.spread_mm, box.sigma_mm[expected], angle_deg,
                face_names[expected]);
    check(found == expected, (what + ": the box's " + face_names[expected]).c_str());
    errors.normal_deg.add(angle_deg);
    errors.spread_relative.add(std::abs(face.spread_mm - box.sigma_mm[expected]) /
                               box.sigma_mm[expected]);
}

/// Measures a model of a box and checks its edges and faces against the box: the length along
/// x between the ends, the width along y between the sides, the height along z.
/// @param what [in] the case.
/// @param model [in] the model.
/// @param box [in] the box.
/// @param pose [in] the box's frame to the model's.
/// @param errors [in,out] the class's errors.
void check_box(const std::string &what, const std::vector<gca::model_point> &model,
               const box_spec &box, const Eigen::Isometry3d &pose, class_errors &errors)
{
    const gca::box_result result = gca::measure_box(model);
    check(result.measurement.has_value(), (what + ": measured").c_str());
    if (!result.measurement)
    {
        std::printf("%s: refused: %s\n", what.c_str(), result.refusal.c_str());
        return;
    }
    const gca::box_measurement &m = *result.measurement;
    std::printf("%s: %zu points, %zu on five faces; length %.4f, width %.4f, height %.4f mm\n",
                what.c_str(), m.points, m.points_on_faces, m.length_mm, m.width_mm, m.height_mm);
    check(result.refusal.empty(), (what + ": no refusal").c_str());
    check(m.points == model.size(), (what + ": the model's points counted").c_str());
    errors.edge_mm.add(std::abs(m.length_mm - box.length_mm));
    errors.edge_mm.add(std::abs(m.width_mm - box.width_mm));
    errors.edge_mm.add(std::abs(m.height_mm - box.height_mm));
    check_face(what + ", top", m.top, top, box, pose, errors);
    // The faces of either order: the one along +x or +y first or second.
    const int first_end = nearest_face(pose, m.ends[0].normal) == plus_x ? 0 : 1;
    const int first_side = nearest_face(pose, m.sides[0].normal) == plus_y ? 0 : 1;
    check_face(what + ", end", m.ends[first_end], plus_x, box, pose, errors);
    check_face(what + ", end", m.ends[1 - first_end], minus_x, box, pose, errors);
    check_face(what + ", side", m.sides[first_side], plus_y, box, pose, errors);
    check_face(what + ", side", m.sides[1 - first_side], minus_y, box, pose, errors);
}

/// Checks a tabletop box's case: its edges within the tabletop tolerance, its spreads and its
/// faces' normals within theirs.
/// @param what [in] the case.
/// @param model [in] the model.
/// @param box [in] the box.
/// @param pose [in] the box's frame to the model's.
void check_tabletop_case(const std::string &what, const std::vector<gca::model_point> &model,
                         const box_spec &box, const Eigen::Isometry3d &pose)
{
    class_errors errors;
    check_box(what, model, box, pose, errors);
    check_max_error(what + ": an edge's error, mm", errors.edge_mm, tabletop_edge_tolerance_mm);
    check_max_error(what + ": a spread's error over its noise", errors.spread_relative,
                    spread_tolerance);
    check_max_error(what + ": a face's normal's angle, deg", errors.normal_deg,
                    normal_tolerance_deg);
}

/// The boxes of both object classes in every orientation: the edges within the class's
/// tolerance, small beside R21's 2 mm and 10 mm, each spread within a share of its face's noise,
/// each face's normal within an angle of the box's.
void test_boxes()
{
    const object_class_box classes[] = {
        {"tabletop object", tabletop_box, tabletop_edge_tolerance_mm},
        {"large object", large_box, large_edge_tolerance_mm}};
    std::uint64_t seed = 1;
    for (const object_class_box &object_class : classes)
    {
        class_errors errors;
        int orientation = 0;
        for (const Eigen::Isometry3d &pose : orientations())
        {
            const std::string what =
                std::string(object_class.name) + ", orientation " + std::to_string(orientation++);
            check_box(what, build_box(object_class.box, pose, seed++), object_class.box, pose,
                      errors);
        }
        const std::string name = object_class.name;
        check_max_error(name + ": an edge's error, mm", errors.edge_mm,
                        object_class.edge_tolerance);
        check_max_error(name + ": a spread's error over its noise", errors.spread_relative,
                        spread_tolerance);
        check_max_error(name + ": a face's normal's angle, deg", errors.normal_deg,
                        normal_tolerance_deg);
    }
}

/// An end seen only in its upper half, as one a scan did not reach down to the foot: its
/// centroid lies 37.5 mm above the other end's, so the length has to be the distance between
/// the ends' planes, not between their centroids (which would give sqrt(300^2 + 37.5^2) - 300 =
/// 2.33 mm more [S: computed]), and the height has to come from the sides that reach the foot.
void test_end_seen_in_part()
{
    box_spec box = tabletop_box;
    box.seen_from_z_mm[plus_x] = box.height_mm / 2;
    const Eigen::Isometry3d pose = orientations()[1];
    check_tabletop_case("tabletop object, an end seen in its upper half", build_box(box, pose, 301),
                        box, pose);
}

/// Adds ten stray points facing down at the box's foot, along its middle line, where the face it
/// stands on would be, as stray normals near a face that was not scanned [?: a choice].
/// @param model [in,out] the model.
/// @param box [in] the box.
/// @param pose [in] the box's frame to the model's.
void add_strays_facing_down(std::vector<gca::model_point> &model, const box_spec &box,
                            const Eigen::Isometry3d &pose)
{
    constexpr int count = 10;
    for (int i = 0; i < count; i++)
        add_point(model, pose, {box.length_mm * (i + 0.5) / count - box.length_mm / 2, 0.0, 0.0},
                  -Eigen::Vector3d::UnitZ());
}

/// Ten stray points facing down, far under 1 % of the points: the face the box stands on does not
/// count, so the model still shows five faces and measures as before.
void test_strays_facing_down()
{
    const Eigen::Isometry3d pose = orientations()[2];
    std::vector<gca::model_point> model = build_box(tabletop_box, pose, 401);
    add_strays_facing_down(model, tabletop_box, pose);
    check_tabletop_case("tabletop object with 10 stray points facing down", model, tabletop_box,
                        pose);
}

/// Adds a remnant of the support plane: a ring of points around the box's foot, one voxel above
/// it, as the separation's cut leaves the plane's highest points (docs/methods/separation.md),
/// its normals the top's.
/// @param model [in,out] the model.
/// @param box [in] the box.
/// @param pose [in] the box's frame to the model's.
/// @param ring_mm [in] the ring's width beyond the foot, mm.
void add_support_remnant(std::vector<gca::model_point> &model, const box_spec &box,
                         const Eigen::Isometry3d &pose, double ring_mm)
{
    const double half_x = box.length_mm / 2 + ring_mm;
    const double half_y = box.width_mm / 2 + ring_mm;
    for (double x = -half_x; x <= half_x; x += box.spacing_mm)
        for (double y = -half_y; y <= half_y; y += box.spacing_mm)
            if (std::abs(x) > box.length_mm / 2 || std::abs(y) > box.width_mm / 2)
                add_point(model, pose, {x, y, box.spacing_mm}, Eigen::Vector3d::UnitZ());
}

/// A remnant of the support plane 20 mm wide around the tabletop box's foot, a third as many
/// points as the top [?: a choice], in its cone: the band leaves it out, and the edges and the
/// top's spread stay as they are.
void test_support_remnant()
{
    const Eigen::Isometry3d pose = orientations()[1];
    std::vector<gca::model_point> model = build_box(tabletop_box, pose, 101);
    const std::size_t box_points = model.size();
    add_support_remnant(model, tabletop_box, pose, 20.0);
    std::printf("the remnant: %zu points beside the box's %zu\n", model.size() - box_points,
                box_points);
    check_tabletop_case("tabletop object with a remnant of the support plane", model, tabletop_box,
                        pose);
}

/// A remnant 80 mm wide around a box 50 mm high, more points than the top [?: a choice]: the
/// band about the median keeps the remnant in the top's place, so the model is refused
/// [?: a choice, rather than a top sought at the sides' upper end], not given a height of about
/// one voxel [M 2026-11-15: this case before the refusal, length 299.99, width 200.01, height
/// 1.00 mm].
void test_remnant_outweighs_top()
{
    box_spec low_box = tabletop_box;
    low_box.height_mm = 50.0;
    const Eigen::Isometry3d pose = orientations()[3];
    std::vector<gca::model_point> model = build_box(low_box, pose, 501);
    const std::size_t box_points = model.size();
    add_support_remnant(model, low_box, pose, 80.0);
    const gca::box_result result = gca::measure_box(model);
    std::printf("a remnant of %zu points beside a 50 mm box's %zu", model.size() - box_points,
                box_points);
    if (result.measurement)
        std::printf(": length %.4f, width %.4f, height %.4f mm\n", result.measurement->length_mm,
                    result.measurement->width_mm, result.measurement->height_mm);
    else
        std::printf("\n");
    check_refusal("a remnant that outweighs the top", !result.measurement.has_value(),
                  result.refusal,
                  result.refusal == "the points facing up lie in the lower half of the sides: a "
                                    "remnant of the support plane outweighs the top");
}

/// Builds a sphere's surface of radius 150 mm, a tabletop object's size: 100000 points on a
/// Fibonacci lattice (synthetic_source.hpp), their normals radial, each moved along its normal
/// by a draw of noise of 0.5 mm from a fixed seed [?: a choice].
/// @return the model, centred 0.8 m before the camera.
std::vector<gca::model_point> build_sphere()
{
    constexpr double radius_mm = 150.0;
    constexpr std::size_t count = 100000;
    const Eigen::Isometry3d pose = gca_test::make_transform({0.0, 0.0, 1.0}, 0.0, {0, 0, 0.8});
    std::mt19937_64 generator(7);
    std::normal_distribution<double> noise(0.0, 0.5);
    std::vector<gca::model_point> model;
    for (std::size_t k = 0; k < count; k++)
    {
        const Eigen::Vector3d normal = gca_test::sphere_lattice_direction(k, count);
        add_point(model, pose, (radius_mm + noise(generator)) * normal, normal);
    }
    return model;
}

/// Checks that a model is refused as showing a number of faces.
/// @param what [in] the case.
/// @param model [in] the model.
/// @param faces [in] the faces it shows.
void check_faces_refused(const std::string &what, const std::vector<gca::model_point> &model,
                         int faces)
{
    const gca::box_result result = gca::measure_box(model);
    check_refusal(what, !result.measurement.has_value(), result.refusal,
                  result.refusal == "a box shows five faces, its top and four sides, as the side "
                                    "it stands on is not scanned (ADR 0003); this model shows " +
                                        std::to_string(faces));
}

/// Models that are no box are refused with the reason: a sphere, whose faces hold too few of its
/// points (3 (1 - cos 20 deg) = 18.1 % of a sphere's normals lie within 20 deg of six axes
/// [S: a spherical cap's area]); a box missing a side beside its bottom, a box with all six
/// faces, one plane and an empty model, by the faces they show.
void test_refusals()
{
    const gca::box_result result = gca::measure_box(build_sphere());
    const std::string suffix = " % of the 100000 points lie on the faces found; a box needs at "
                               "least half";
    const double share = std::atof(result.refusal.c_str());
    check_refusal("a sphere", !result.measurement.has_value(), result.refusal,
                  result.refusal.size() > suffix.size() &&
                      result.refusal.compare(result.refusal.size() - suffix.size(), suffix.size(),
                                             suffix) == 0 &&
                      std::abs(share - 18.1) <= 1.0);

    const Eigen::Isometry3d pose = orientations()[2];
    box_spec four = tabletop_box;
    four.built[minus_y] = false;
    check_faces_refused("a box missing a side beside its bottom", build_box(four, pose, 201), 4);
    box_spec six = tabletop_box;
    six.built[bottom] = true;
    check_faces_refused("a box with all six faces", build_box(six, pose, 202), 6);
    box_spec plane = tabletop_box;
    plane.built = {false, false, false, false, true, false};
    check_faces_refused("one plane", build_box(plane, pose, 203), 1);
    check_faces_refused("an empty model", {}, 0);
}

/// The report's lines, which the hardware acceptance reads: a measurement written by hand, each
/// length rounded to 0.01 mm.
void test_report()
{
    gca::box_measurement m;
    m.length_mm = 300.004;
    m.width_mm = 199.996;
    m.height_mm = 150.5;
    const auto face = [](std::size_t points, double spread_mm) {
        return gca::box_face{points, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ(), spread_mm};
    };
    m.top = face(60501, 1.499);
    m.sides = {face(45451, 1.0), face(45452, 1.251)};
    m.ends = {face(30351, 0.5), face(30352, 0.7449)};
    m.points = 212105;
    m.points_on_faces = 212000;
    const std::string expected = "box_fit: scan/object_model.ply: 212105 points, 212000 on five "
                                 "faces\n"
                                 "length: 300.00 mm\n"
                                 "width: 200.00 mm\n"
                                 "height: 150.50 mm\n"
                                 "top: 60501 points, spread 1.50 mm\n"
                                 "side 1: 45451 points, spread 1.00 mm\n"
                                 "side 2: 45452 points, spread 1.25 mm\n"
                                 "end 1: 30351 points, spread 0.50 mm\n"
                                 "end 2: 30352 points, spread 0.74 mm";
    const std::string report = gca::box_report("scan/object_model.ply", m);
    std::printf("the report:\n%s\n", report.c_str());
    check(report == expected, "the report's lines");
}

/// The program on models the test writes (scan/model_file.hpp): a box prints the report of the
/// box measured from the model, exit 0; a sphere prints "no box" with the reason, exit 1; a file
/// that does not exist is refused with the usage, exit 2.
/// @param program [in] box_fit's path.
void test_program(const std::string &program)
{
    const gca_test::temporary_directory directory("test_box_measure");
    const std::filesystem::path box_path = directory / "box.ply";
    const std::filesystem::path sphere_path = directory / "sphere.ply";
    const std::vector<gca::model_point> box = build_box(tabletop_box, orientations()[0], 601);
    const std::vector<gca::model_point> sphere = build_sphere();
    check(gca::write_model(box_path, box).written, "test setup: the box's model written");
    check(gca::write_model(sphere_path, sphere).written, "test setup: the sphere's model written");

    const program_run measured = run_program(program, box_path.string());
    const gca::box_result box_result = gca::measure_box(box);
    std::printf("box_fit on a box, exit %d:\n%s", measured.exit_status, measured.output.c_str());
    check(measured.exit_status == 0 && box_result.measurement &&
              measured.output == gca::box_report(box_path.string(), *box_result.measurement) + "\n",
          "box_fit on a box: its lines, exit 0");

    const program_run refused = run_program(program, sphere_path.string());
    std::printf("box_fit on a sphere, exit %d: %s", refused.exit_status, refused.output.c_str());
    check(refused.exit_status == 1 &&
              refused.output == "box_fit: " + sphere_path.string() +
                                    ": no box: " + gca::measure_box(sphere).refusal + "\n",
          "box_fit on a sphere: no box, exit 1");

    const std::string missing = (directory / "missing.ply").string();
    const program_run unread = run_program(program, missing);
    std::printf("box_fit on a missing file, exit %d: %s", unread.exit_status,
                unread.output.c_str());
    check(unread.exit_status == 2 &&
              unread.output.rfind("box_fit: " + missing + ": no such file\nusage: ", 0) == 0,
          "box_fit on a missing file: the usage, exit 2");
}
} // namespace

/// Runs the box fit's checks and prints each case's measurement, the errors and the summary line.
/// @param argc [in] the argument count, 2.
/// @param argv [in] the arguments: argv[1] the path of the program box_fit.
/// @return 0 when every check passed, 1 otherwise; 2 without the program's path.
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::fprintf(stderr, "usage: test_box_measure BOX_FIT\n");
        return 2;
    }
    test_boxes();
    test_end_seen_in_part();
    test_strays_facing_down();
    test_support_remnant();
    test_remnant_outweighs_top();
    test_refusals();
    test_report();
    test_program(argv[1]);
    return check_summary("test_box_measure");
}
