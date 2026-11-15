// SPDX-License-Identifier: 0BSD
/// @file float3_eigen.cuh
/// The conversion of Eigen's `Vector3f`, in which the library does matrix arithmetic, into CUDA's
/// `float3`, in which it stores points and normals, usable in host and device code.

#pragma once

#include <Eigen/Core>

#include <cuda_runtime.h>

namespace gca
{
/// Copies an Eigen vector into a float3.
/// @param v the vector.
/// @return (v(0), v(1), v(2)).
__host__ __device__ __forceinline__ float3 to_float3(const Eigen::Vector3f &v)
{
    return make_float3(v(0), v(1), v(2));
}
} // namespace gca
