// SPDX-License-Identifier: 0BSD
/// @file float3_ops.cuh
/// The arithmetic of CUDA's `float3` that the library uses, usable in host and device code: sum,
/// difference, scaling, division by a number, scaling in place, dot product, cross product and
/// Euclidean norm. Each is written from its mathematical definition. The functions live in the
/// global namespace because `float3` does, so that argument-dependent lookup finds them.

#pragma once

#include <cmath>

#include <cuda_runtime.h>

/// Adds two vectors component by component.
/// @param a the first vector.
/// @param b the second vector.
/// @return a + b.
__host__ __device__ __forceinline__ float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/// Subtracts two vectors component by component.
/// @param a the vector to subtract from.
/// @param b the vector to subtract.
/// @return a - b.
__host__ __device__ __forceinline__ float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

/// Scales a vector by a number.
/// @param v the vector.
/// @param s the factor.
/// @return v scaled by s.
__host__ __device__ __forceinline__ float3 operator*(const float3 &v, float s)
{
    return make_float3(v.x * s, v.y * s, v.z * s);
}

/// Scales a vector by a number, the number written first.
/// @param s the factor.
/// @param v the vector.
/// @return v scaled by s.
__host__ __device__ __forceinline__ float3 operator*(float s, const float3 &v)
{
    return v * s;
}

/// Divides a vector by a number component by component. A zero divisor gives infinities or NaNs
/// as float division does [S: IEEE 754].
/// @param v the vector.
/// @param s the divisor.
/// @return the vector whose components are v's divided by s.
__host__ __device__ __forceinline__ float3 operator/(const float3 &v, float s)
{
    return make_float3(v.x / s, v.y / s, v.z / s);
}

/// Scales a vector by a number in place.
/// @param v the vector, replaced by its scaled value.
/// @param s the factor.
/// @return v after scaling.
__host__ __device__ __forceinline__ float3 &operator*=(float3 &v, float s)
{
    v = v * s;
    return v;
}

/// Computes the dot product of two vectors.
/// @param a the first vector.
/// @param b the second vector.
/// @return the sum of the products of the matching components.
__host__ __device__ __forceinline__ float dot(const float3 &a, const float3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// Computes the cross product of two vectors in a right-handed frame.
/// @param a the first vector.
/// @param b the second vector.
/// @return the vector orthogonal to both whose length is the area of their parallelogram, so
/// that cross(x axis, y axis) is the z axis.
__host__ __device__ __forceinline__ float3 cross(const float3 &a, const float3 &b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

/// Computes the Euclidean length of a vector.
/// @param v the vector.
/// @return the square root of v's dot product with itself.
__host__ __device__ __forceinline__ float norm(const float3 &v)
{
    return sqrtf(dot(v, v));
}
