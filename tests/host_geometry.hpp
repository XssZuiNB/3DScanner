// SPDX-License-Identifier: 0BSD
/// @file host_geometry.hpp
/// The offline tests' analytic side: vectors and rigid transforms in double on the host, their
/// conversions to and from the library's float types, and the tilted plane the tests build
/// clouds on. Header only; each test program is one translation unit.

#pragma once

#include "geometry/type.hpp"

#include <cmath>
#include <vector>

#include <Eigen/Core>

namespace gca_test
{
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
inline vec3 add_scaled(const vec3 &a, double s, const vec3 &b)
{
    return {a.x + s * b.x, a.y + s * b.y, a.z + s * b.z};
}

/// Returns the vector scaled by a factor.
/// @param s the factor.
/// @param a the vector.
/// @return s * a.
inline vec3 scale(double s, const vec3 &a)
{
    return {s * a.x, s * a.y, s * a.z};
}

/// Returns the dot product.
/// @param a the first vector.
/// @param b the second vector.
/// @return a . b.
inline double dot(const vec3 &a, const vec3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// Returns the cross product.
/// @param a the first vector.
/// @param b the second vector.
/// @return a x b.
inline vec3 cross(const vec3 &a, const vec3 &b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

/// Returns the vector scaled to unit length.
/// @param a the vector, not zero.
/// @return a / |a|.
inline vec3 normalized(const vec3 &a)
{
    return scale(1.0 / std::sqrt(dot(a, a)), a);
}

/// Returns the distance between two points.
/// @param a the first point.
/// @param b the second point.
/// @return |a - b|.
inline double distance(const vec3 &a, const vec3 &b)
{
    const vec3 d = add_scaled(a, -1.0, b);
    return std::sqrt(dot(d, d));
}

/// Returns the angle between two directions.
/// @param a the first direction, not zero.
/// @param b the second direction, not zero.
/// @return the angle in degrees, 0 to 180.
inline double angle_deg(const vec3 &a, const vec3 &b)
{
    const vec3 c = cross(a, b);
    return std::atan2(std::sqrt(dot(c, c)), dot(a, b)) * 180.0 / M_PI;
}

/// Converts a float3 to double.
/// @param a the float3.
/// @return a in double.
inline vec3 to_vec3(const float3 &a)
{
    return {a.x, a.y, a.z};
}

/// Builds a point to upload with create_from_points(). Every point the tests build is positive
/// on every axis: today's max bound starts at FLT_MIN (geometry_util.cuh:81, 92), so a cloud
/// with no positive coordinate on an axis gets a wrong bound. No cloud of the tests that build
/// points comes from create_from_rgbd(), whose default minimum depth 0 makes NaN points [S].
/// @param p the coordinates in meters, rounded to float.
/// @param color the color.
/// @return the point, property inactive like the points create_from_rgbd() keeps.
inline gca::point_t make_point(const vec3 &p, const gca::color3 &color)
{
    gca::point_t point{};
    point.coordinates =
        make_float3(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
    point.color = color;
    point.property = gca::point_property::inactive;
    return point;
}

/// Returns whether a point is positive on every axis, as make_point()'s comment asks of every
/// point the tests upload.
/// @param p the point.
/// @return whether x, y and z are all positive; false when one is NaN.
inline bool positive_on_every_axis(const vec3 &p)
{
    return p.x > 0.0 && p.y > 0.0 && p.z > 0.0;
}

/// Returns the tests' plane's unit normal, pointing away from the camera (n_z > 0) so that the
/// code has to flip it, and tilted off every axis, which keeps the eigensolver clear of a bug [S].
/// An axis-aligned plane has a diagonal covariance, whose eigenvalues the eigensolver returns in
/// the diagonal's order (fast_eigen_val_vec.cuh, solve_symmetric_eigen_3x3()), not sorted;
/// compute_normal_functor's strict < then picks a wrong eigenvector if the smallest eigenvalue
/// ties another.
/// @return the normal.
inline vec3 plane_normal()
{
    return normalized({0.3, -0.2, 1.0});
}

/// Two orthonormal directions in the tests' plane.
struct plane_axes
{
    vec3 u; ///< perpendicular to the normal and to the x axis
    vec3 v; ///< normal x u
};

/// Returns the in-plane directions the tests lay their grids along.
/// @return u and v.
inline plane_axes make_plane_axes()
{
    const vec3 n = plane_normal();
    const vec3 u = normalized(cross(n, {1.0, 0.0, 0.0}));
    return {u, cross(n, u)};
}

/// One point of a grid on the tests' plane.
struct plane_grid_point
{
    int i;      ///< its column, counted along u from the center
    int j;      ///< its row, counted along v from the center
    double s_m; ///< its in-plane coordinate along u from the center
    double t_m; ///< its in-plane coordinate along v from the center
    vec3 p;     ///< the point
};

/// Lays a grid on the tests' plane: (2 half_u + 1) columns along u and (2 half_v + 1) rows along
/// v, point (i, j) at center + (i + offset_u) spacing u + (j + offset_v) spacing v.
/// @param center the grid's center, a point of the plane, in meters.
/// @param half_u the columns on each side of the center.
/// @param half_v the rows on each side of the center.
/// @param spacing_m the distance between neighbouring columns and rows.
/// @param offset_u the shift of every column along u, in cells.
/// @param offset_v the shift of every row along v, in cells.
/// @return the points, column by column.
inline std::vector<plane_grid_point> make_plane_grid(const vec3 &center, int half_u, int half_v,
                                                     double spacing_m, double offset_u,
                                                     double offset_v)
{
    const plane_axes axes = make_plane_axes();
    std::vector<plane_grid_point> grid;
    for (int i = -half_u; i <= half_u; i++)
    {
        for (int j = -half_v; j <= half_v; j++)
        {
            const double s = (i + offset_u) * spacing_m;
            const double t = (j + offset_v) * spacing_m;
            grid.push_back({i, j, s, t, add_scaled(add_scaled(center, s, axes.u), t, axes.v)});
        }
    }
    return grid;
}

/// A rigid transform as the tests pass it to the library, in float.
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
inline rigid_transform make_rigid_transform(const vec3 &axis, double angle_rad,
                                            const vec3 &translation)
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
inline Eigen::Matrix4f to_matrix4f(const rigid_transform &transform)
{
    Eigen::Matrix4f m;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            m(i, j) = transform.rotation[i][j];
        m(i, 3) = transform.translation[i];
        m(3, i) = 0.0f;
    }
    m(3, 3) = 1.0f;
    return m;
}

/// Converts the library's 4x4 matrix to a rigid transform; the last row is left to the caller.
/// @param m the homogeneous matrix [R t; 0 1].
/// @return R and t, as the matrix holds them.
inline rigid_transform from_matrix4f(const Eigen::Matrix4f &m)
{
    rigid_transform transform{};
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            transform.rotation[i][j] = m(i, j);
        transform.translation[i] = m(i, 3);
    }
    return transform;
}

/// Rotates a direction by the transform, in double, with the float values the library receives.
/// @param transform the transform.
/// @param d the direction.
/// @return R d.
inline vec3 rotate_direction(const rigid_transform &transform, const vec3 &d)
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
inline vec3 transform_point(const rigid_transform &transform, const vec3 &p)
{
    const vec3 t = {transform.translation[0], transform.translation[1], transform.translation[2]};
    return add_scaled(rotate_direction(transform, p), 1.0, t);
}
} // namespace gca_test
