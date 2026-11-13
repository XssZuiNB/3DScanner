// SPDX-License-Identifier: 0BSD
/// @file float3_eigen.cuh
/// The conversions between CUDA's `float3`, in which the library stores points, normals and
/// color gradients, and Eigen's `Vector3f`, in which it does matrix arithmetic, usable in host and
/// device code.

#pragma once

#include <Eigen/Core>

#include <cuda_runtime.h>

namespace gca
{
/// Copies a float3 into an Eigen vector.
/// @param v the vector.
/// @return (v.x, v.y, v.z).
__host__ __device__ __forceinline__ Eigen::Vector3f to_eigen(const float3 &v)
{
    return Eigen::Vector3f(v.x, v.y, v.z);
}

/// Copies an Eigen vector into a float3.
/// @param v the vector.
/// @return (v(0), v(1), v(2)).
__host__ __device__ __forceinline__ float3 to_float3(const Eigen::Vector3f &v)
{
    return make_float3(v(0), v(1), v(2));
}
} // namespace gca
