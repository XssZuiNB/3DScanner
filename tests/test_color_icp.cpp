// SPDX-License-Identifier: 0BSD
/// @file test_color_icp.cpp
/// Offline test of gca::color_icp: a source cloud, moved by a known small rigid transform and
/// uploaded in another order than its target, is aligned back onto the target, and the recovered
/// transform undoes the known one within a tolerance. Three cases:
/// - the fixture data/frag_116.ply, built as CUDA_3D's pipeline built its frames (voxel
///   downsampling, normals), its own points the source: a real scene, whose geometry constrains
///   every degree of freedom [?: not measured with the color term off];
/// - a textured plane, its own points the source, moved within the plane: the geometric term is
///   blind to that move, so only the color term recovers it (textured_plane_transform());
/// - the same plane resampled on an offset grid as the source, its colors taken from the
///   texture: where the source's points lie between the target's, the alignment depends on the
///   value of the target's color gradients, not only on their direction
///   (resampled_plane_tolerance).
/// With its own points as the source, the known transform aligns the source exactly. Each run
/// prints the errors with their tolerances. Needs the GPU, no camera; the fixture's path is the
/// program's argument.

#include "check.hpp"
#include "geometry/point_cloud.hpp"
#include "host_geometry.hpp"
#include "ply_reader.hpp"
#include "registration/color_icp.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

using gca_test::add_scaled;
using gca_test::check;
using gca_test::check_max_error;
using gca_test::check_summary;
using gca_test::distance;
using gca_test::dot;
using gca_test::float_step;
using gca_test::from_matrix4f;
using gca_test::make_plane_axes;
using gca_test::make_plane_grid;
using gca_test::make_point;
using gca_test::make_rigid_transform;
using gca_test::max_error;
using gca_test::plane_axes;
using gca_test::plane_grid_point;
using gca_test::plane_normal;
using gca_test::ply_vertices;
using gca_test::positive_on_every_axis;
using gca_test::read_ply_file;
using gca_test::rigid_transform;
using gca_test::rotate_direction;
using gca_test::scale;
using gca_test::to_vec3;
using gca_test::transform_point;
using gca_test::vec3;

namespace
{
/// CUDA_3D's pipeline from a camera frame to colored ICP [S: apps/app.cpp:91, 124, 128 at
/// 2d6eaa2]: voxel downsampling at 8 mm, normals from the neighbours within 4 cm, and colored
/// ICP of at most 10 iterations, correspondences within 5 cm and color gradients from the
/// neighbours within 5 cm. Its radius outlier removal (app.cpp:126) is left out: the test pins
/// colored ICP, not that filter.
constexpr float voxel_size_m = 0.008f;
constexpr float normal_radius_m = 0.04f;
constexpr size_t icp_max_iterations = 10;
constexpr float icp_max_correspondence_m = 0.05f;
constexpr float icp_color_gradient_radius_m = 0.05f;

/// The moved source's upload order: point k of the upload is point (upload_stride * k) mod n of
/// the source, every point once when the stride and n are coprime (checked). In the target's own
/// order, a step that paired points by index would align a copy without the nearest-neighbour
/// search.
constexpr size_t upload_stride = 7919;

/// Builds the rigid transform that rotates by an angle about an axis through a pivot, then
/// shifts.
/// @param axis the rotation axis, not zero.
/// @param angle_rad the rotation angle in radians.
/// @param pivot the point the axis runs through, in meters.
/// @param shift the shift after the rotation, in meters.
/// @return the transform, R p + (pivot - R pivot + shift), rounded to float.
rigid_transform make_rotation_about(const vec3 &axis, double angle_rad, const vec3 &pivot,
                                    const vec3 &shift)
{
    rigid_transform transform = make_rigid_transform(axis, angle_rad, {0.0, 0.0, 0.0});
    const vec3 t =
        add_scaled(add_scaled(pivot, -1.0, rotate_direction(transform, pivot)), 1.0, shift);
    transform.translation[0] = static_cast<float>(t.x);
    transform.translation[1] = static_cast<float>(t.y);
    transform.translation[2] = static_cast<float>(t.z);
    return transform;
}

/// A source moved by a known transform, in upload order.
struct moved_source
{
    std::vector<gca::point_t> points; ///< the moved points, in upload order
    max_error partner_distance_m;     ///< each moved point's distance from its partner
    bool all_positive = true;         ///< every moved point positive on every axis
};

/// Moves every source point by the transform, in double with the float values the transform
/// holds, rounds each coordinate to float once, keeps colors and properties, and permutes the
/// order by upload_stride.
/// @param source the source's points.
/// @param partners the target point each source point lies on or next to, by index.
/// @param transform the known transform.
/// @return the moved source.
moved_source move_source(const std::vector<gca::point_t> &source,
                         const std::vector<gca::point_t> &partners,
                         const rigid_transform &transform)
{
    moved_source moved;
    const size_t n = source.size();
    for (size_t k = 0; k < n; k++)
    {
        const size_t index = upload_stride * k % n;
        const vec3 p = transform_point(transform, to_vec3(source[index].coordinates));
        gca::point_t point = source[index];
        point.coordinates =
            make_float3(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
        moved.points.push_back(point);
        moved.partner_distance_m.add(distance(p, to_vec3(partners[index].coordinates)));
        moved.all_positive = moved.all_positive && positive_on_every_axis(p);
    }
    return moved;
}

/// Aligns the moved source onto the target with colored ICP and CUDA_3D's parameters: at least
/// one iteration (with none, align() reads an uninitialized value [S: color_icp.cu:184, 231 at
/// 2d6eaa2]) and one fresh object per alignment.
/// @param source the moved source.
/// @param target the target, its normals estimated.
/// @return the transform align() recovered, which moves the source onto the target, or nothing
///     when align() failed.
std::optional<Eigen::Matrix4f> align_source(const std::shared_ptr<const gca::point_cloud> &source,
                                            const std::shared_ptr<const gca::point_cloud> &target)
{
    gca::color_icp icp(icp_max_iterations, icp_max_correspondence_m, icp_color_gradient_radius_m);
    icp.set_source_point_cloud(source);
    icp.set_target_point_cloud(target);
    if (!icp.align())
        return std::nullopt;
    return icp.get_final_transformation_matrix();
}

/// How far the recovered transform is from undoing the known one. E = recovered * known is the
/// identity when ICP undid the move exactly.
struct alignment_error
{
    bool homogeneous = false; ///< every entry finite and the last row (0, 0, 0, 1)
    max_error rotation_deg;   ///< E's rotation angle
    max_error point_m;        ///< |E p - p| over the target's points
};

/// Compares the recovered transform with the known one, in double.
/// @param recovered the transform align() recovered.
/// @param known the transform the source was moved by.
/// @param target the target's points.
/// @return the comparison.
alignment_error compare_alignment(const Eigen::Matrix4f &recovered, const rigid_transform &known,
                                  const std::vector<gca::point_t> &target)
{
    alignment_error error;
    bool finite = true;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            finite = finite && std::isfinite(recovered(i, j));
    error.homogeneous = finite && recovered(3, 0) == 0.0f && recovered(3, 1) == 0.0f &&
                        recovered(3, 2) == 0.0f && recovered(3, 3) == 1.0f;

    const rigid_transform back = from_matrix4f(recovered);
    // E's rotation, column by column, and its angle from the skew part and the trace, accurate at
    // small angles where acos is not.
    const vec3 x = rotate_direction(back, rotate_direction(known, {1.0, 0.0, 0.0}));
    const vec3 y = rotate_direction(back, rotate_direction(known, {0.0, 1.0, 0.0}));
    const vec3 z = rotate_direction(back, rotate_direction(known, {0.0, 0.0, 1.0}));
    const vec3 skew = {y.z - z.y, z.x - x.z, x.y - y.x};
    error.rotation_deg.add(
        std::atan2(0.5 * std::sqrt(dot(skew, skew)), 0.5 * (x.x + y.y + z.z - 1.0)) * 180.0 / M_PI);
    for (const gca::point_t &point : target)
    {
        const vec3 p = to_vec3(point.coordinates);
        error.point_m.add(distance(transform_point(back, transform_point(known, p)), p));
    }
    return error;
}

/// The tolerances of one case.
struct alignment_tolerance
{
    double rotation_deg; ///< of E's rotation angle
    double point_m;      ///< of the largest |E p - p|
};

/// Checks that every normal of the target is finite: one NaN normal would make ICP's sums NaN
/// [S: the target's normal enters each point's Jacobian, cuda_color_icp_build_least_square.cu:
/// 197-198, and transform_reduce adds them up over all points, :467, at 2d6eaa2].
/// @param name the case's name, the prefix of its check lines.
/// @param target the target, its normals estimated.
void check_normals_finite(const std::string &name, const gca::point_cloud &target)
{
    const std::vector<float3> normals = target.download_normals();
    bool finite = normals.size() == target.points_number();
    for (const float3 &n : normals)
        finite = finite && std::isfinite(n.x) && std::isfinite(n.y) && std::isfinite(n.z);
    check(finite, (name + ": one finite normal per point").c_str());
}

/// Moves the source by the known transform, aligns it back onto the target with colored ICP and
/// checks that the recovered transform undoes the known one. Every moved source point lies
/// within the correspondence distance of its partner, so none is without a correspondence at the
/// first iteration (with none, the sums are zero and the zero step meets align()'s stop rule [S:
/// cuda_color_icp_build_least_square.cu:159-165, color_icp.cu:224-228 at 2d6eaa2]); and the
/// moved source stays positive on every axis (make_point()'s comment).
/// @param name the case's name, the prefix of its check lines.
/// @param target the target, its normals estimated.
/// @param source the source's points before the move, point k next to the target's point k.
/// @param known the transform the source is moved by.
/// @param tolerance the case's tolerances.
void check_alignment(const std::string &name, const std::shared_ptr<gca::point_cloud> &target,
                     const std::vector<gca::point_t> &source, const rigid_transform &known,
                     const alignment_tolerance &tolerance)
{
    const std::vector<gca::point_t> partners = target->download();
    std::printf("%s: %zu points\n", name.c_str(), partners.size());
    check(source.size() == partners.size(), (name + ": one partner per source point").c_str());
    check(std::gcd(upload_stride, source.size()) == 1,
          (name + ": the upload order holds every point once").c_str());
    if (source.size() != partners.size())
        return;
    const moved_source moved = move_source(source, partners, known);
    check_max_error(name + ": max distance of the moved source from its partners (m)",
                    moved.partner_distance_m, icp_max_correspondence_m);
    check(moved.all_positive, (name + ": the moved source positive on every axis").c_str());

    const std::optional<Eigen::Matrix4f> recovered =
        align_source(gca::point_cloud::create_from_points(moved.points), target);
    check(recovered.has_value(), (name + ": align() succeeded").c_str());
    if (!recovered)
        return;
    const alignment_error error = compare_alignment(*recovered, known, partners);
    check(error.homogeneous,
          (name + ": recovered transform finite with last row (0, 0, 0, 1)").c_str());
    check_max_error(name + ": rotation error (deg)", error.rotation_deg, tolerance.rotation_deg);
    check_max_error(name + ": max point error (m)", error.point_m, tolerance.point_m);
}

/// The fixture's vertex count [S: data/frag_116.ply's header].
constexpr size_t fixture_vertices = 103388;

/// Returns the fixture case's known transform: 1 degree about the axis (2, -3, 1) through the
/// pivot (1.75, 2.57, 1.12) m, near the fixture's centroid (1.753, 2.575, 1.124) m, then
/// (6, -4, 3) mm. Its points move 7.2 to 28.1 mm, and the lowest coordinate afterwards is 0.24 m
/// [S: computed in double over the fixture's 103 388 vertices, 2026-11-13]. The identity misses
/// it by 1 degree, 12 times the rotation step below which align() may stop (0.081 degrees, with
/// a step below 1 mm, color_icp.hpp:47-48 at 2d6eaa2 [S]), so ICP has to iterate.
/// @return the transform.
rigid_transform fixture_transform()
{
    return make_rotation_about({2.0, -3.0, 1.0}, 1.0 * M_PI / 180.0, {1.75, 2.57, 1.12},
                               {0.006, -0.004, 0.003});
}

/// The fixture case's tolerances, 3 times the errors they were set from, 2.25e-4 degrees and
/// 1.05e-5 m [M 2026-11-13, library at 2d6eaa2, the same over 3 runs]; after
/// replace-math-cuh-with-eigen 2.25e-4 degrees and 1.06e-5 m [M 2026-11-13]. Those errors are not
/// rounding but what Gauss-Newton leaves after the step that met align()'s stop rule (a step below
/// 1 mm and 0.081 degrees), so they depend on where that rule fires: of 15 transforms around this
/// one (0.8 to 1.2 degrees, the shift scaled by 0.8 to 1.25, this one included), 10 end with such a
/// remainder, 1.03e-5 to 1.74e-5 m and 2.25e-4 to 4.44e-4 degrees, and 5 at rounding level,
/// 1.4e-7 to 3.5e-7 m and 1.0e-6 to 3.3e-6 degrees, presumably one step later [M 2026-11-13; ?:
/// the iterations are not visible]. Undone by nothing, the move leaves 1 degree and up to 28 mm.
constexpr alignment_tolerance fixture_tolerance = {7e-4, 3e-5};

/// Colored ICP aligns a moved copy of the fixture back onto it. The fixture's coordinates are
/// positive on every axis: x 0.73 to 2.82 m, y 1.24 to 3.54 m, z 0.23 to 1.73 m [S: computed
/// over the fixture, 2026-11-13]; the read checks it.
/// @param path the fixture's path.
void test_fixture(const std::string &path)
{
    std::string error;
    const std::optional<ply_vertices> fixture = read_ply_file(path, error);
    check(fixture.has_value(), ("fixture: read; " + error).c_str());
    if (!fixture)
        return;
    bool all_positive = fixture->points.size() == fixture_vertices;
    for (const gca::point_t &point : fixture->points)
        all_positive = all_positive && positive_on_every_axis(to_vec3(point.coordinates));
    check(all_positive, "fixture: every vertex read, positive on every axis");

    const std::shared_ptr<gca::point_cloud> cloud =
        gca::point_cloud::create_from_points(fixture->points)->voxel_grid_down_sample(voxel_size_m);
    check(cloud != nullptr, "fixture: voxel downsampling gave a cloud");
    if (cloud == nullptr)
        return;
    check(cloud->estimate_normals(normal_radius_m), "fixture: normals estimated");
    check_normals_finite("fixture", *cloud);
    check_alignment("fixture", cloud, cloud->download(), fixture_transform(), fixture_tolerance);
}

/// The textured plane's center in meters, and its grid: (2 * half_u + 1) points along u and
/// (2 * half_v + 1) along v, spacing apart. It is positive on every axis: x 0.33 to 0.67 m,
/// y 0.25 to 0.75 m, z 0.90 to 1.10 m [S: computed from the inputs, 2026-11-13]. Within the
/// normals' radius lie about 140 points, within the color gradient's about 220 (pi r^2 /
/// spacing^2 [S: computed]).
constexpr vec3 textured_plane_center = {0.5, 0.5, 1.0};
constexpr int textured_plane_half_u = 40;
constexpr int textured_plane_half_v = 30;
constexpr double textured_plane_spacing_m = 0.006;

/// One wave of the plane's texture.
struct texture_wave
{
    double direction_deg; ///< its direction in the plane, from u toward v
    double wavelength_m;  ///< its wavelength
    double phase_rad;     ///< its phase at the plane's center
};

/// The texture: two waves in different directions, so the color constrains both in-plane
/// directions and the rotation about the normal. Each is long against the color gradient's
/// 5 cm radius: in one dimension, a least-squares slope over +-r keeps 3 (sin x - x cos x) / x^3
/// of a wave's slope, x = 2 pi r / wavelength, 0.89 and 0.84 here [S: computed]; a slope fitted
/// much too flat would make Gauss-Newton overshoot the move [?].
constexpr texture_wave texture_waves[] = {{20.0, 0.30, 0.4}, {115.0, 0.24, 1.1}};
/// Each wave's amplitude around the mean gray 0.5: the gray stays within 0.1 to 0.9.
constexpr double texture_amplitude = 0.2;

/// Returns the texture's gray at an in-plane position.
/// @param s_m the position along u from the plane's center.
/// @param t_m the position along v from the plane's center.
/// @return the gray, in [0, 1].
double texture_gray(double s_m, double t_m)
{
    double gray = 0.5;
    for (const texture_wave &wave : texture_waves)
    {
        const double a = wave.direction_deg * M_PI / 180.0;
        const double along_m = s_m * std::cos(a) + t_m * std::sin(a);
        gray +=
            texture_amplitude * std::sin(2.0 * M_PI * along_m / wave.wavelength_m + wave.phase_rad);
    }
    return gray;
}

/// Builds the textured plane on a grid along make_plane_axes(), each point gray with the
/// texture's value at its in-plane position.
/// @param offset_u the grid's shift along u, in cells.
/// @param offset_v the grid's shift along v, in cells.
/// @return the points, column by column.
std::vector<gca::point_t> make_textured_plane(double offset_u, double offset_v)
{
    std::vector<gca::point_t> points;
    for (const plane_grid_point &grid :
         make_plane_grid(textured_plane_center, textured_plane_half_u, textured_plane_half_v,
                         textured_plane_spacing_m, offset_u, offset_v))
    {
        const float g = static_cast<float>(texture_gray(grid.s_m, grid.t_m));
        points.push_back(make_point(grid.p, gca::color3{g, g, g}));
    }
    return points;
}

/// Returns the textured-plane cases' known transform: 0.5 degrees about the plane's normal
/// through its center, then 3 mm along u and -2 mm along v. The moved source stays in the plane,
/// which leaves the geometric term blind to the move [S: its residual (s - q) . n is zero for
/// points of the plane, cuda_color_icp_build_least_square.cu:193 at 2d6eaa2]; its points move 1.4
/// to 6.1 mm [S: computed from the inputs, 2026-11-13], all below the 5 cm correspondence
/// distance.
/// @return the transform.
rigid_transform textured_plane_transform()
{
    const plane_axes axes = make_plane_axes();
    return make_rotation_about(plane_normal(), 0.5 * M_PI / 180.0, textured_plane_center,
                               add_scaled(scale(0.003, axes.u), -0.002, axes.v));
}

/// The textured-plane case's tolerances: 8 float steps [S: float32 ulp], 9.5e-7 rad (5.5e-5
/// degrees) for the angle and 9.5e-7 m for points below 2 m. The errors they were set from are
/// rounding: 1.22e-5 degrees (1.8 steps) and 7.4e-8 m (0.6 steps) [M 2026-11-13, library at
/// 2d6eaa2, the same over 3 runs]; after replace-math-cuh-with-eigen 8.49e-6 degrees and 8.5e-8 m
/// [M 2026-11-13]; 18 transforms around this one (0 to 0.7 degrees, the shift scaled by 0.8 to
/// 1.25) end within 2.0e-5 degrees (3.0 steps) and 5.1e-7 m (4.3 steps) [M 2026-11-13]. Undone
/// by nothing, the move leaves 0.5 degrees and up to 6.1 mm.
constexpr alignment_tolerance textured_plane_tolerance = {8 * float_step * 180.0 / M_PI,
                                                          8 * float_step};

/// The resampled source's grid shift, in cells: 0.3 along u and 0.15 along v, 2.0 mm from its
/// partner, the target point of the same column and row; the next target point is 4.3 mm away
/// [S: computed from the inputs, 2026-11-13], so the partner is the nearest.
constexpr double resample_offset_u = 0.3;
constexpr double resample_offset_v = 0.15;

/// The resampled case's tolerances, 3 times the errors they were set from, 0.0119 degrees and
/// 2.04e-4 m [M 2026-11-13, library at 2d6eaa2, the same over 3 runs]; after
/// replace-math-cuh-with-eigen 0.0119 degrees and 2.06e-4 m [M 2026-11-13]. Those errors are no
/// convergence remainder: all 18 transforms around this one (0 to 0.7 degrees, the shift scaled by
/// 0.8 to 1.25) end at the same errors [M 2026-11-13], presumably the bias of a color model whose
/// gradients are fitted flatter than the texture's (texture_waves' comment), which leaves color
/// residuals at the known transform where the source's points lie between the target's [?]. With
/// the gradients doubled (Atb doubled in cuda_compute_color_gradient.cu), the same 18 end at
/// 1.75 to 1.96 mm, 2.9 times the tolerance and more [M 2026-11-13]. Undone by nothing, the move
/// leaves 0.5 degrees and up to 6.1 mm.
constexpr alignment_tolerance resampled_plane_tolerance = {0.035, 6e-4};

/// Colored ICP aligns two sources moved within the textured plane back onto it: the plane's own
/// points, and the plane resampled between them.
void test_textured_plane()
{
    const std::vector<gca::point_t> points = make_textured_plane(0.0, 0.0);
    const std::shared_ptr<gca::point_cloud> cloud = gca::point_cloud::create_from_points(points);
    check(cloud->estimate_normals(normal_radius_m), "textured plane: normals estimated");
    check_normals_finite("textured plane", *cloud);
    check_alignment("textured plane", cloud, points, textured_plane_transform(),
                    textured_plane_tolerance);
    check_alignment("resampled plane", cloud,
                    make_textured_plane(resample_offset_u, resample_offset_v),
                    textured_plane_transform(), resampled_plane_tolerance);
}
} // namespace

/// Runs the colored-ICP checks and prints their measurements and the summary line.
/// @param argc the number of arguments.
/// @param argv the arguments: the program, then the path of data/frag_116.ply.
/// @return 0 when every check passed, 1 otherwise.
int main(int argc, char **argv)
{
    check(argc == 2, "one argument, the path of data/frag_116.ply");
    if (argc == 2)
        test_fixture(argv[1]);
    test_textured_plane();
    return check_summary("test_color_icp");
}
