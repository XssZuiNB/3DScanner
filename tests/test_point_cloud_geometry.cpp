// SPDX-License-Identifier: 0BSD
/// @file test_point_cloud_geometry.cpp
/// Offline test of gca::point_cloud's geometry on synthetic clouds uploaded with
/// create_from_points(): transform() moves points, normals and bounds by a known rigid
/// transform; estimate_normals() finds the analytic normals of a tilted plane and of a sphere,
/// oriented as the code orients them today; voxel_grid_down_sample() gives the known voxel
/// centroids and count. Each run prints the measured maxima with their tolerances. Needs the
/// GPU, no camera.

#include "check.hpp"
#include "geometry/point_cloud.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using gca_test::check;
using gca_test::check_summary;
using gca_test::check_within;

namespace
{
/// One float step (ulp) between 1 and 2, FLT_EPSILON: the largest rounding step of a coordinate
/// below 2 m, of a unit normal's components and of a color in [0, 1] [S: float32 ulp].
constexpr double float_step = std::numeric_limits<float>::epsilon();

/// A vector in double, for the analytic values.
struct vec3
{
    double x;
    double y;
    double z;
};

/// Returns a + s * b.
/// @param a the first vector.
/// @param s the factor of b.
/// @param b the second vector.
/// @return a + s * b.
vec3 add_scaled(const vec3 &a, double s, const vec3 &b)
{
    return {a.x + s * b.x, a.y + s * b.y, a.z + s * b.z};
}

/// Returns the vector scaled by a factor.
/// @param s the factor.
/// @param a the vector.
/// @return s * a.
vec3 scale(double s, const vec3 &a)
{
    return {s * a.x, s * a.y, s * a.z};
}

/// Returns the dot product.
/// @param a the first vector.
/// @param b the second vector.
/// @return a . b.
double dot(const vec3 &a, const vec3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// Returns the cross product.
/// @param a the first vector.
/// @param b the second vector.
/// @return a x b.
vec3 cross(const vec3 &a, const vec3 &b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

/// Returns the vector scaled to unit length.
/// @param a the vector, not zero.
/// @return a / |a|.
vec3 normalized(const vec3 &a)
{
    return scale(1.0 / std::sqrt(dot(a, a)), a);
}

/// Returns the distance between two points.
/// @param a the first point.
/// @param b the second point.
/// @return |a - b|.
double distance(const vec3 &a, const vec3 &b)
{
    const vec3 d = add_scaled(a, -1.0, b);
    return std::sqrt(dot(d, d));
}

/// Returns the angle between two directions.
/// @param a the first direction, not zero.
/// @param b the second direction, not zero.
/// @return the angle in degrees, 0 to 180.
double angle_deg(const vec3 &a, const vec3 &b)
{
    const vec3 c = cross(a, b);
    return std::atan2(std::sqrt(dot(c, c)), dot(a, b)) * 180.0 / M_PI;
}

/// Converts a float3 to double.
/// @param a the float3.
/// @return a in double.
vec3 to_vec3(const float3 &a)
{
    return {a.x, a.y, a.z};
}

/// Returns a point's color as a vector.
/// @param point the point.
/// @return (r, g, b).
vec3 color_of(const gca::point_t &point)
{
    return {point.color.r, point.color.g, point.color.b};
}

/// The largest of a set of errors, and whether every error was finite. A NaN or infinite value
/// downloaded from the GPU makes its error non-finite, which a maximum alone would drop.
struct max_error
{
    double max = 0.0;
    bool all_finite = true;

    /// Adds one error.
    /// @param error the error.
    void add(double error)
    {
        if (std::isfinite(error))
            max = std::fmax(max, error);
        else
            all_finite = false;
    }
};

/// Checks that every error was finite and that the largest is within the tolerance, and prints
/// the largest.
/// @param what what was measured.
/// @param errors the errors.
/// @param tolerance the largest error that passes.
void check_max_error(const std::string &what, const max_error &errors, double tolerance)
{
    check(errors.all_finite, (what + ": every value finite").c_str());
    check_within(what.c_str(), errors.max, tolerance);
}

/// Builds a point to upload with create_from_points(). Every point the test builds is positive
/// on every axis: today's max bound starts at FLT_MIN (geometry_util.cuh:81, 92), so a cloud
/// with no positive coordinate on an axis gets a wrong bound. No cloud comes from
/// create_from_rgbd(), whose default minimum depth 0 makes NaN points [S].
/// @param p the coordinates in meters, rounded to float.
/// @param color the color.
/// @return the point, property inactive like the points create_from_rgbd() keeps.
gca::point_t make_point(const vec3 &p, const gca::color3 &color)
{
    gca::point_t point{};
    point.coordinates =
        make_float3(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
    point.color = color;
    point.property = gca::point_property::inactive;
    return point;
}

/// The plane's center in meters.
constexpr vec3 plane_center = {0.3, 0.2, 1.0};
/// The plane's grid: (2 * plane_grid_half + 1)^2 points, plane_grid_spacing_m apart.
constexpr int plane_grid_half = 20;
constexpr double plane_grid_spacing_m = 0.005;

/// Returns the plane's unit normal, pointing away from the camera (n_z > 0) so that the code has
/// to flip it, and tilted off every axis with components of distinct magnitude, which keeps the
/// eigensolver clear of two bugs [S]. An axis-aligned plane has a diagonal covariance, whose
/// eigenvalues fast_eigen_val_vec.cuh:196-204 returns unsorted; compute_normal_functor's strict
/// < then picks a wrong eigenvector if the smallest eigenvalue ties another. max_coeff
/// (math.cuh:100) returns the x entry when the two largest of its three entries tie, and
/// fast_eigen_val_vec.cuh:34 divides by that entry's root: a less accurate eigenvector, or NaN
/// when the entry is 0. On a plane those entries, the squared cross products of the shifted
/// covariance's rows, are proportional to the squared components of the eigenvector being
/// solved for (the adjugate of a rank-2 symmetric matrix): the normal, components 0.28, 0.19 and
/// 0.94, or, where a neighbourhood elongated along the grid's edge makes the solver start from
/// the largest eigenvalue, the edge direction u (0, 0.98, 0.20) or v (0.96, 0.06, 0.28) in
/// magnitude [S: computed from the inputs, 2026-11-13]; none has a tie.
/// @return the normal.
vec3 plane_normal()
{
    return normalized({0.3, -0.2, 1.0});
}

/// Builds the tilted plane: a square grid around plane_center, the color varying with the grid
/// position.
/// @return the points.
std::vector<gca::point_t> make_plane()
{
    const vec3 n = plane_normal();
    const vec3 u = normalized(cross(n, {1.0, 0.0, 0.0}));
    const vec3 v = cross(n, u);
    std::vector<gca::point_t> points;
    for (int i = -plane_grid_half; i <= plane_grid_half; i++)
    {
        for (int j = -plane_grid_half; j <= plane_grid_half; j++)
        {
            const vec3 p = add_scaled(add_scaled(plane_center, i * plane_grid_spacing_m, u),
                                      j * plane_grid_spacing_m, v);
            const float r = static_cast<float>(i + plane_grid_half) / (2 * plane_grid_half);
            const float g = static_cast<float>(j + plane_grid_half) / (2 * plane_grid_half);
            points.push_back(make_point(p, gca::color3{r, g, 0.5f}));
        }
    }
    return points;
}

/// The sphere's center and radius in meters, and its number of points.
constexpr vec3 sphere_center = {0.3, 0.2, 1.0};
constexpr double sphere_radius_m = 0.1;
constexpr int sphere_points = 4000;

/// Builds the sphere: a Fibonacci lattice, the points spread evenly over the whole surface,
/// about 5.6 mm apart (the root of the area per point, 4 pi R^2 / 4000 [S: computed from the
/// inputs, 2026-11-13]).
/// @return the points.
std::vector<gca::point_t> make_sphere()
{
    const double golden_angle = M_PI * (3.0 - std::sqrt(5.0));
    std::vector<gca::point_t> points;
    for (int k = 0; k < sphere_points; k++)
    {
        const double z = 1.0 - 2.0 * (k + 0.5) / sphere_points;
        const double r = std::sqrt(1.0 - z * z);
        const vec3 direction = {r * std::cos(golden_angle * k), r * std::sin(golden_angle * k), z};
        points.push_back(make_point(add_scaled(sphere_center, sphere_radius_m, direction),
                                    gca::color3{0.5f, 0.5f, 0.5f}));
    }
    return points;
}

/// The normals' search radius in meters: 15 to 49 neighbours per point on the plane, 37 to 43 on
/// the sphere [S: counted in double over the inputs, 2026-11-13]; fewer than 3 give a zero
/// normal. Every point is its own neighbour unless it is NaN, and none is: a point without a
/// neighbour would shift every later normal, since cuda_estimate_normals.cu:184-191 pairs the
/// per-neighbourhood sums with the points by index [S].
constexpr float normal_radius_m = 0.02f;

/// How estimated normals compare with the analytic ones.
struct normal_errors
{
    max_error angle_deg;           ///< to the oriented analytic normal, outside the band
    max_error axis_angle_deg;      ///< to the analytic normal's line, either sign
    max_error length;              ///< | |n| - 1 |
    bool all_nonzero = true;       ///< no zero normal
    bool all_toward_camera = true; ///< n_z <= 0 everywhere, today's orientation rule
};

/// Estimates the cloud's normals and compares them with analytic ones. Today's code flips every
/// normal with n . (0, 0, -1) < 0, toward a camera at the origin looking along +z; the analytic
/// normal is oriented the same way. Where the analytic normal's |n_z| is below
/// orientation_band, noise decides the estimate's z sign, so only the axis angle counts there.
/// @param points the cloud.
/// @param analytic_normal the analytic unit normal at a point, either sign.
/// @param orientation_band the |n_z| below which the orientation is not compared.
/// @return the comparison, or nothing when estimate_normals() failed or did not give one normal
///     per point.
template <typename normal_fn>
std::optional<normal_errors> compare_normals(const std::vector<gca::point_t> &points,
                                             normal_fn analytic_normal, double orientation_band)
{
    const auto cloud = gca::point_cloud::create_from_points(points);
    if (!cloud->estimate_normals(normal_radius_m) || !cloud->has_normals())
        return std::nullopt;
    const std::vector<float3> normals = cloud->download_normals();
    if (normals.size() != points.size())
        return std::nullopt;
    normal_errors errors;
    for (size_t k = 0; k < points.size(); k++)
    {
        const vec3 n = to_vec3(normals[k]);
        const double length = std::sqrt(dot(n, n));
        errors.length.add(std::fabs(length - 1.0));
        if (length == 0.0)
        {
            errors.all_nonzero = false;
            continue;
        }
        const vec3 axis = analytic_normal(to_vec3(points[k].coordinates));
        const vec3 expected = scale(axis.z > 0.0 ? -1.0 : 1.0, axis);
        const double angle = angle_deg(n, expected);
        errors.all_toward_camera = errors.all_toward_camera && normals[k].z <= 0.0f;
        errors.axis_angle_deg.add(std::fmin(angle, 180.0 - angle));
        if (std::fabs(expected.z) >= orientation_band)
            errors.angle_deg.add(angle);
    }
    return errors;
}

/// The tolerance of | |n| - 1 |: 8 float steps [S: float32 ulp]. Today's maxima are 3.04e-7
/// (2.6 steps) on the plane and 1.07e-7 on the sphere [M 2026-11-13, b5c70dc]: the eigensolver
/// normalizes a cross product of two computed eigenvectors, and another normalization path
/// rounds by a few steps more or less.
constexpr double normal_length_tolerance = 8 * float_step;

/// estimate_normals() on the tilted plane gives the plane's normal everywhere, flipped toward
/// the camera.
void test_normals_plane()
{
    // Today's maximum on this board [M 2026-11-13, b5c70dc]: 0.154 degrees, from the raw moments
    // summed in float at 1 m (PCA in double over the same neighbours gives 0.0002 degrees [S:
    // computed from the inputs, 2026-11-13]); tolerance 3.2 times that.
    constexpr double angle_tolerance_deg = 0.5;
    const std::optional<normal_errors> errors =
        compare_normals(make_plane(), [](const vec3 &) { return plane_normal(); }, 0.0);
    check(errors.has_value(), "normals plane: one normal per point");
    if (!errors)
        return;
    check(errors->all_nonzero, "normals plane: no zero normal");
    check(errors->all_toward_camera, "normals plane: every normal has n_z <= 0");
    check_max_error("normals plane: max angle (deg)", errors->angle_deg, angle_tolerance_deg);
    check_max_error("normals plane: max | |n| - 1 |", errors->length, normal_length_tolerance);
}

/// estimate_normals() on the sphere gives the radial direction, flipped toward the camera. Its
/// neighbourhoods take every orientation, so the max_coeff tie of plane_normal()'s comment can
/// occur where a normal's two largest components have equal magnitude: a less accurate normal,
/// and NaN where the returned entry is 0, which the finite check catches. No normal is NaN today
/// [M 2026-11-13].
void test_normals_sphere()
{
    // The equator band |n_z| < 0.2 reaches 11.5 degrees on each side of the equator (asin 0.2 [S:
    // computed]), far beyond the estimate's angle error, so outside it the orientation rule, not
    // noise, decides the sign.
    constexpr double orientation_band = 0.2;
    // Today's maximum on this board [M 2026-11-13, b5c70dc]: 0.717 degrees, of which 0.688 is the
    // sampling's own (PCA in double over the same neighbours [S: computed from the inputs,
    // 2026-11-13]); tolerance 2.1 times that.
    constexpr double angle_tolerance_deg = 1.5;
    const std::optional<normal_errors> errors = compare_normals(
        make_sphere(), [](const vec3 &p) { return normalized(add_scaled(p, -1.0, sphere_center)); },
        orientation_band);
    check(errors.has_value(), "normals sphere: one normal per point");
    if (!errors)
        return;
    check(errors->all_nonzero, "normals sphere: no zero normal");
    check(errors->all_toward_camera, "normals sphere: every normal has n_z <= 0");
    check_max_error("normals sphere: max angle outside the band (deg)", errors->angle_deg,
                    angle_tolerance_deg);
    check_max_error("normals sphere: max axis angle (deg)", errors->axis_angle_deg,
                    angle_tolerance_deg);
    check_max_error("normals sphere: max | |n| - 1 |", errors->length, normal_length_tolerance);
}

/// A rigid transform as the test passes it to the library, in float.
struct rigid_transform
{
    float rotation[3][3];
    float translation[3];
};

/// Builds the rigid transform that rotates by an angle about an axis through the origin, then
/// translates (Rodrigues' formula, rounded to float once).
/// @param axis the rotation axis, not zero.
/// @param angle_rad the rotation angle in radians.
/// @param translation the translation in meters.
/// @return the transform.
rigid_transform make_rigid_transform(const vec3 &axis, double angle_rad, const vec3 &translation)
{
    const vec3 k = normalized(axis);
    const double c = std::cos(angle_rad);
    const double s = std::sin(angle_rad);
    const double kv[3] = {k.x, k.y, k.z};
    const double kx[3][3] = {{0.0, -k.z, k.y}, {k.z, 0.0, -k.x}, {-k.y, k.x, 0.0}};
    rigid_transform result{};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            result.rotation[i][j] =
                static_cast<float>((i == j ? c : 0.0) + s * kx[i][j] + (1.0 - c) * kv[i] * kv[j]);
    result.translation[0] = static_cast<float>(translation.x);
    result.translation[1] = static_cast<float>(translation.y);
    result.translation[2] = static_cast<float>(translation.z);
    return result;
}

/// Converts the transform to the library's 4x4 matrix.
/// @param transform the transform.
/// @return the homogeneous matrix [R t; 0 1].
mat4x4 to_mat4x4(const rigid_transform &transform)
{
    mat4x4 m;
    for (unsigned int i = 0; i < 3; i++)
    {
        for (unsigned int j = 0; j < 3; j++)
            m(i, j) = transform.rotation[i][j];
        m(i, 3) = transform.translation[i];
        m(3, i) = 0.0f;
    }
    m(3, 3) = 1.0f;
    return m;
}

/// Rotates a direction by the transform, in double, with the float values the library receives.
/// @param transform the transform.
/// @param d the direction.
/// @return R d.
vec3 rotate_direction(const rigid_transform &transform, const vec3 &d)
{
    const auto row = [&](int i) {
        return transform.rotation[i][0] * d.x + transform.rotation[i][1] * d.y +
               transform.rotation[i][2] * d.z;
    };
    return {row(0), row(1), row(2)};
}

/// Moves a point by the transform, in double, with the float values the library receives.
/// @param transform the transform.
/// @param p the point.
/// @return R p + t.
vec3 transform_point(const rigid_transform &transform, const vec3 &p)
{
    const vec3 t = {transform.translation[0], transform.translation[1], transform.translation[2]};
    return add_scaled(rotate_direction(transform, p), 1.0, t);
}

/// transform() moves every point of the plane to R p + t, turns its normal to R n, keeps the
/// colors and properties, and leaves the cloud's bounds those of the moved points (the normals
/// estimated first made the cloud cache its bounds). The plane stays positive on every axis:
/// x 0.48 to 0.69 m, y 0.30 to 0.57 m, z 0.78 to 0.95 m afterwards [S: computed from the inputs,
/// 2026-11-13].
void test_transform()
{
    // 4 float steps at 1 m for positions and bounds, 4 for unit normals [S: float32 ulp]. Today's
    // maxima on this board [M 2026-11-13, b5c70dc]: positions 7.9e-8 m, normals 5.89e-8,
    // bounds 2.14e-8 m.
    constexpr double position_tolerance_m = 4 * float_step;
    constexpr double normal_tolerance = 4 * float_step;
    const std::vector<gca::point_t> before = make_plane();
    const auto cloud = gca::point_cloud::create_from_points(before);
    check(cloud->estimate_normals(normal_radius_m), "transform: normals estimated first");
    const std::vector<float3> normals_before = cloud->download_normals();
    const rigid_transform transform =
        make_rigid_transform({1.0, 2.0, 3.0}, 30.0 * M_PI / 180.0, {0.1, 0.2, -0.05});
    cloud->transform(to_mat4x4(transform));
    const std::vector<gca::point_t> after = cloud->download();
    const std::vector<float3> normals_after = cloud->download_normals();

    const bool sizes_kept = after.size() == before.size() &&
                            normals_before.size() == before.size() &&
                            normals_after.size() == before.size();
    check(sizes_kept, "transform: point and normal counts kept");
    if (!sizes_kept)
        return;
    max_error position_m;
    max_error normal;
    vec3 min_expected = {INFINITY, INFINITY, INFINITY};
    vec3 max_expected = {-INFINITY, -INFINITY, -INFINITY};
    bool attributes_kept = true;
    for (size_t k = 0; k < before.size(); k++)
    {
        const vec3 p = transform_point(transform, to_vec3(before[k].coordinates));
        position_m.add(distance(to_vec3(after[k].coordinates), p));
        normal.add(distance(to_vec3(normals_after[k]),
                            rotate_direction(transform, to_vec3(normals_before[k]))));
        min_expected = {std::fmin(min_expected.x, p.x), std::fmin(min_expected.y, p.y),
                        std::fmin(min_expected.z, p.z)};
        max_expected = {std::fmax(max_expected.x, p.x), std::fmax(max_expected.y, p.y),
                        std::fmax(max_expected.z, p.z)};
        // Exact equality also fails on a NaN color.
        attributes_kept = attributes_kept && after[k].color.r == before[k].color.r &&
                          after[k].color.g == before[k].color.g &&
                          after[k].color.b == before[k].color.b &&
                          after[k].property == before[k].property;
    }
    max_error bound_m;
    bound_m.add(distance(to_vec3(cloud->get_min_bound()), min_expected));
    bound_m.add(distance(to_vec3(cloud->get_max_bound()), max_expected));
    check_max_error("transform: max position error (m)", position_m, position_tolerance_m);
    check_max_error("transform: max normal error", normal, normal_tolerance);
    check_max_error("transform: max bound error (m)", bound_m, position_tolerance_m);
    check(attributes_kept, "transform: colors and properties kept");
}

/// The voxel size in meters: a power of two, so its multiples are exact in float.
constexpr float voxel_size_m = 0.125f;
/// The cloud's min corner, one of its points: 2, 4 and 8 voxels from the origin.
constexpr vec3 voxel_min_corner = {0.25, 0.5, 1.0};
/// The occupied voxels, counted from the min corner.
constexpr int voxel_cells[][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
                                  {2, 1, 0}, {1, 2, 3}, {3, 0, 2}, {2, 3, 1}};
/// Positions within a voxel, in voxels: each at least 0.1 voxel from every face of its voxel.
constexpr double voxel_offsets[][3] = {{0.2, 0.3, 0.7},  {0.8, 0.15, 0.4},  {0.5, 0.85, 0.25},
                                       {0.35, 0.6, 0.9}, {0.1, 0.45, 0.55}, {0.65, 0.75, 0.15}};
constexpr int voxel_offsets_n = sizeof(voxel_offsets) / sizeof(voxel_offsets[0]);
/// The upload order's stride: point k of the upload is point 7k mod n of the voxel-by-voxel
/// order. A voxel holds at most 6 consecutive points of that order, so no two neighbours of the
/// upload share a voxel, and the voxel code has to sort.
constexpr size_t voxel_upload_stride = 7;

/// One occupied voxel's expected output point.
struct voxel_cluster
{
    vec3 centroid; ///< the mean of its points' float coordinates
    vec3 color;    ///< the mean of its points' colors
};

/// The voxel test's cloud and its expected output.
struct voxel_input
{
    std::vector<gca::point_t> points;    ///< in upload order
    std::vector<voxel_cluster> clusters; ///< one per occupied voxel
};

/// Builds the voxel test's cloud: 4 to 6 points in each occupied voxel, the first voxel also
/// holding the min corner itself, 40 points in all, uploaded interleaved. Every point but the
/// min corner lies at least 0.1 voxel from every voxel face, so float rounding of
/// (p - min) / size cannot move it across one. Today's lattice starts at the cloud's min bound;
/// that bound lies on multiples of the voxel size, so a lattice anchored at the world origin
/// groups the points the same way.
/// @return the cloud and the expected output.
voxel_input make_voxel_input()
{
    voxel_input input;
    std::vector<gca::point_t> by_voxel;
    const int cells_n = sizeof(voxel_cells) / sizeof(voxel_cells[0]);
    for (int c = 0; c < cells_n; c++)
    {
        std::vector<vec3> offsets;
        if (c == 0)
            offsets.push_back({0.0, 0.0, 0.0});
        for (int k = 0; k < 4 + c % 3; k++)
        {
            const double *o = voxel_offsets[(c + k) % voxel_offsets_n];
            offsets.push_back({o[0], o[1], o[2]});
        }
        vec3 position_sum = {0.0, 0.0, 0.0};
        vec3 color_sum = {0.0, 0.0, 0.0};
        for (const vec3 &o : offsets)
        {
            const int i = static_cast<int>(by_voxel.size());
            const vec3 cell = {voxel_cells[c][0] + o.x, voxel_cells[c][1] + o.y,
                               voxel_cells[c][2] + o.z};
            const gca::color3 color = {(i % 5) / 4.0f, (i % 3) / 2.0f, (i % 7) / 6.0f};
            by_voxel.push_back(make_point(add_scaled(voxel_min_corner, voxel_size_m, cell), color));
            position_sum = add_scaled(position_sum, 1.0, to_vec3(by_voxel.back().coordinates));
            color_sum = add_scaled(color_sum, 1.0, color_of(by_voxel.back()));
        }
        input.clusters.push_back(
            {scale(1.0 / offsets.size(), position_sum), scale(1.0 / offsets.size(), color_sum)});
    }
    for (size_t k = 0; k < by_voxel.size(); k++)
        input.points.push_back(by_voxel[voxel_upload_stride * k % by_voxel.size()]);
    return input;
}

/// voxel_grid_down_sample() gives one point per occupied voxel, at the mean position and with
/// the mean color of the voxel's points. The output order is not checked.
void test_voxel_down_sample()
{
    // 4 float steps at 1 m for centroids, 4 for colors in [0, 1] [S: float32 ulp]. Today's maxima
    // on this board [M 2026-11-13, b5c70dc, interleaved upload]: centroids 8.55e-8 m, colors
    // 3.73e-8.
    constexpr double centroid_tolerance_m = 4 * float_step;
    constexpr double color_tolerance = 4 * float_step;
    const voxel_input input = make_voxel_input();
    check(input.points.size() % voxel_upload_stride != 0,
          "voxel: the upload order holds every point once");
    const auto down =
        gca::point_cloud::create_from_points(input.points)->voxel_grid_down_sample(voxel_size_m);
    check(down != nullptr, "voxel: a result");
    if (down == nullptr)
        return;
    const std::vector<gca::point_t> result = down->download();
    check(result.size() == input.clusters.size(), "voxel: one point per occupied voxel");

    std::vector<int> matches(input.clusters.size(), 0);
    max_error centroid_m;
    max_error color;
    for (const gca::point_t &p : result)
    {
        size_t nearest = 0;
        double nearest_m = INFINITY;
        for (size_t c = 0; c < input.clusters.size(); c++)
        {
            const double d = distance(to_vec3(p.coordinates), input.clusters[c].centroid);
            if (d < nearest_m)
            {
                nearest = c;
                nearest_m = d;
            }
        }
        matches[nearest]++;
        centroid_m.add(nearest_m);
        color.add(distance(color_of(p), input.clusters[nearest].color));
    }
    bool one_each = true;
    for (int m : matches)
        one_each = one_each && m == 1;
    check(one_each, "voxel: each occupied voxel matched by exactly one output point");
    check_max_error("voxel: max centroid error (m)", centroid_m, centroid_tolerance_m);
    check_max_error("voxel: max color error", color, color_tolerance);
}
} // namespace

/// Runs the geometry checks and prints their measurements and the summary line.
/// @return 0 when every check passed, 1 otherwise.
int main()
{
    test_transform();
    test_normals_plane();
    test_normals_sphere();
    test_voxel_down_sample();
    return check_summary("test_point_cloud_geometry");
}
