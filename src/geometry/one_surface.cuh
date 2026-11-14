// SPDX-License-Identifier: 0BSD
/// @file one_surface.cuh
/// The rule of one surface, which the per-frame path's stages share: two points of an organized
/// picture lie on one surface unless the segment between them points nearly along the line of
/// sight, the mark of a depth edge between two surfaces, one behind the other. The normals keep a
/// pixel's neighbourhood to its surface by it (geometry/organized_normals.hpp), and a coarser
/// level of a frame's pyramid combines only points it finds on one surface
/// (geometry/frame_pyramid.hpp). The rule and its threshold: docs/methods/normals.md, "One
/// surface, or a depth edge". Header only, for CUDA translation units.

#pragma once

#include "geometry/float3_ops.cuh"

#include <cuda_runtime.h>

namespace gca
{
/// Says whether two points lie on one surface: the segment between them makes an angle of at
/// least 90 degrees less the steepest incidence with the line of sight to their midpoint
/// (docs/methods/normals.md, "One surface, or a depth edge"). Symmetric in a and b; a point lies
/// on one surface with itself.
/// @param a [in] the first point, in the camera frame, in m, z > 0.
/// @param b [in] the second point, in the camera frame, in m, z > 0.
/// @param sin2_max_incidence [in] the square of the sine of the steepest incidence, in (0, 1),
///        below 1 by far more than the rounding of the products below
///        (geometry/organized_normals.hpp, max_steepest_incidence_deg).
/// @return whether (d . m)^2 <= sin^2 * |d|^2 * |m|^2, d = b - a, m = a + b.
/// @note Device code; allocates nothing and waits for nothing.
__device__ __forceinline__ bool on_one_surface(const float3 &a, const float3 &b,
                                               float sin2_max_incidence)
{
    const float3 d = b - a;
    const float3 m = a + b;
    const float along = dot(d, m);
    return along * along <= sin2_max_incidence * dot(d, d) * dot(m, m);
}
} // namespace gca
