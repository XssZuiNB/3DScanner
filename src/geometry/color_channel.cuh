// SPDX-License-Identifier: 0BSD
/// @file color_channel.cuh
/// The rounding of a color's channel to a byte on the GPU, the one copy that the sparse TSDF's
/// integration, extraction and ray cast (fusion/sparse_tsdf.hpp) and the live view's shading
/// (scan/live_view.hpp) take a channel of 0 to 255 from. Header only, for CUDA translation units.

#pragma once

#include <cuda_runtime.h>

namespace gca
{
/// Rounds a color's channel to a byte: to the nearest integer, a tie to the even one
/// (__float2uint_rn).
/// @param channel [in] the channel: a weighted mean of channels of 0 to 255 whose weights sum to 1
///        but for float's rounding, or such a channel times a factor of at most 1, so within
///        [0, 255.5); a larger value keeps its low byte, a negative one gives 0 [S: PTX ISA,
///        cvt: a float's conversion to an integer saturates].
/// @return the channel, 0 to 255.
/// @note Device code; allocates nothing.
__device__ __forceinline__ unsigned char rounded_channel(float channel)
{
    return static_cast<unsigned char>(__float2uint_rn(channel));
}
} // namespace gca
