// SPDX-License-Identifier: 0BSD
/// @file stage_checks.hpp
/// What the per-frame path's stages' tests share to check an organized picture on the host: a
/// pixel's index, a float3 in double, the zero vector that marks an invalid point and a missing
/// normal, the angle between two directions, a quantile and the median of values, and the rule of
/// one surface
/// in double, which places a pair of points against its threshold with a margin, so that a test's
/// own reference in double leaves out a pair that float and double may decide differently
/// (docs/methods/normals.md, "One surface, or a depth edge"). Header only; each test program is
/// one translation unit.

#pragma once

#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace gca_test
{
/// How near the rule's threshold a pair may lie before a test's reference leaves its pixel out:
/// 1 degree of the angle between the pair's segment and the line of sight [?: a choice, far above
/// float's rounding of that angle, a few 1e-5 degrees for points mm apart at 2 m (S: float32
/// ulp)].
inline constexpr double premise_margin_deg = 1.0;

/// Returns pixel (u, v)'s index.
/// @param size [in] the picture's size, in px.
/// @param u [in] the column, in px.
/// @param v [in] the row, in px.
/// @return v * width + u.
inline std::size_t index_of(const gca::frame_size &size, std::uint32_t u, std::uint32_t v)
{
    return static_cast<std::size_t>(v) * size.width + u;
}

/// Converts a float3 to double.
/// @param a [in] the float3.
/// @return a in double.
inline Eigen::Vector3d to_double(const float3 &a)
{
    return {a.x, a.y, a.z};
}

/// Says whether a vector is (0, 0, 0): an invalid pixel's point, a pixel without a normal.
/// @param n [in] the vector.
/// @return whether every component is 0; false for a NaN.
inline bool is_zero(const float3 &n)
{
    return n.x == 0.0f && n.y == 0.0f && n.z == 0.0f;
}

/// Returns the angle between two directions.
/// @param a [in] the first direction, not zero.
/// @param b [in] the second direction, not zero.
/// @return the angle in degrees, 0 to 180; NaN when either is NaN.
inline double angle_deg(const Eigen::Vector3d &a, const Eigen::Vector3d &b)
{
    return std::atan2(a.cross(b).norm(), a.dot(b)) * 180.0 / std::numbers::pi;
}

/// Returns a quantile of values: the value at index floor(q n) of the n values sorted, the largest
/// for q = 1.
/// @param values [in] the values, none a NaN; taken over and reordered.
/// @param q [in] the quantile, from 0 to 1.
/// @return the value; 0 when there are none.
inline double quantile(std::vector<double> values, double q)
{
    if (values.empty())
        return 0.0;
    const std::size_t index = std::min(
        static_cast<std::size_t>(q * static_cast<double>(values.size())), values.size() - 1);
    const auto at = values.begin() + static_cast<std::ptrdiff_t>(index);
    std::nth_element(values.begin(), at, values.end());
    return *at;
}

/// Returns the median of values: quantile() at 1/2, the value at index n / 2.
/// @param values [in] the values, none a NaN; taken over and reordered.
/// @return their median; 0 when there are none.
inline double median(std::vector<double> values)
{
    return quantile(std::move(values), 0.5);
}

/// How a pair of points stands to the rule of one surface.
enum class pair_side
{
    one_surface, ///< inside the rule by more than premise_margin_deg
    edge,        ///< beyond it by more than premise_margin_deg
    near,        ///< within premise_margin_deg of its threshold
};

/// Places a pair of points against the rule of one surface with a steepest incidence, in double:
/// the angle between the segment d = b - a and the line of sight m = a + b, against 90 degrees
/// less the incidence.
/// @param a [in] the first point, in the camera frame, in m.
/// @param b [in] the second point, in the camera frame, in m, not a.
/// @param max_incidence_deg [in] the steepest incidence, in degrees.
/// @return its side, with premise_margin_deg on each side of the threshold.
inline pair_side side_of(const Eigen::Vector3d &a, const Eigen::Vector3d &b,
                         double max_incidence_deg)
{
    const Eigen::Vector3d d = b - a;
    const Eigen::Vector3d m = a + b;
    const double from_sight_deg =
        std::acos(std::min(std::abs(d.dot(m)) / (d.norm() * m.norm()), 1.0)) * 180.0 /
        std::numbers::pi;
    const double threshold_deg = 90.0 - max_incidence_deg;
    if (from_sight_deg > threshold_deg + premise_margin_deg)
        return pair_side::one_surface;
    if (from_sight_deg < threshold_deg - premise_margin_deg)
        return pair_side::edge;
    return pair_side::near;
}
} // namespace gca_test
