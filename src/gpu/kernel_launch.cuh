// SPDX-License-Identifier: 0BSD
/// @file kernel_launch.cuh
/// The launch of a stage's kernel over a picture, shared by the per-frame path's stages (such as
/// geometry/organized_points.hpp and geometry/organized_normals.hpp): the block of threads, the
/// grid of blocks that covers the picture, and the launch on the caller's stream between two
/// questions to CUDA, whether an earlier call left an error and whether CUDA accepted the launch,
/// neither of which waits. A failure comes back as a value (gpu/step_result.hpp) with CUDA's
/// error, which is cleared (gpu/gpu_messages.hpp). Header only, for CUDA translation units: it
/// launches with <<< >>>.

#pragma once

#include "gpu/gpu_messages.hpp"
#include "gpu/step_result.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>
#include <utility>

namespace gca
{
/// The block of threads of a stage's kernel over a picture: 32 x 8 threads, one warp along a row
/// and 8 rows, one thread for each pixel [?: a choice, the user's, 2026-11-14; the stages' times
/// on the board stand in docs/HARDWARE.md, "The per-frame path's stages"; no other block was
/// measured]. A kernel declares it in its __launch_bounds__ and launches with picture_block().
inline constexpr unsigned int picture_block_width = 32;
inline constexpr unsigned int picture_block_height = 8;

/// Returns the block of threads of a stage's kernel over a picture.
/// @return picture_block_width x picture_block_height threads, depth 1.
inline dim3 picture_block()
{
    return dim3(picture_block_width, picture_block_height);
}

/// Returns the number of blocks that cover a length.
/// @param length [in] the length, in px.
/// @param block [in] a block's length, in px, > 0.
/// @return ceil(length / block), without overflow for any 32-bit length.
inline unsigned int blocks_for(std::uint32_t length, unsigned int block)
{
    return static_cast<unsigned int>((std::uint64_t{length} + block - 1) / block);
}

/// Returns the grid of blocks that covers a picture, one thread for each pixel.
/// @param size [in] the picture's width and height, in px.
/// @param block [in] a block's width and height in threads, each > 0, depth 1.
/// @return ceil(width / block.x) x ceil(height / block.y) blocks; a grid CUDA refuses, when the
///         picture needs more blocks along a dimension than a grid holds, is refused at the launch.
inline dim3 grid_for(const frame_size &size, const dim3 &block)
{
    return dim3(blocks_for(size.width, block.x), blocks_for(size.height, block.y));
}

/// Launches a kernel on a stream, after asking CUDA whether an earlier call left an error and
/// before asking whether CUDA accepted the launch. An error an earlier call left would otherwise
/// come back from the second question as if the launch had failed.
/// @param kernel_name [in] the kernel, for the message: "the organized points' kernel".
/// @param kernel [in] the kernel, a __global__ function.
/// @param grid [in] the grid of blocks (grid_for()).
/// @param block [in] the block of threads.
/// @param stream [in] the CUDA stream of the current device the kernel is queued on; owned by the
///        caller.
/// @param arguments [in] the kernel's arguments, copied to the GPU at the launch.
/// @return done when the kernel is queued; not done with CUDA's error, which is then cleared:
///         when CUDA's last error is set before the launch, with nothing queued ("CUDA's last
///         error, left by an earlier call, before <kernel_name>"); when CUDA refuses the launch
///         ("CUDA could not launch <kernel_name>"), for one a grid larger than CUDA allows.
/// @note Waits for nothing: cudaPeekAtLastError and cudaGetLastError ask only whether an error
///       is set. Allocates nothing when done; allocates for the message when not, and a failed
///       allocation ends the program (std::terminate): the function is noexcept.
template <typename... Parameters, typename... Arguments>
step_result launch_kernel(const char *kernel_name, void (*kernel)(Parameters...), const dim3 &grid,
                          const dim3 &block, cudaStream_t stream, Arguments &&...arguments) noexcept
{
    const cudaError_t earlier = cudaPeekAtLastError();
    if (earlier != cudaSuccess)
        return {false, cuda_failure(("CUDA's last error, left by an earlier call, before " +
                                     std::string(kernel_name))
                                        .c_str(),
                                    earlier)};
    kernel<<<grid, block, 0, stream>>>(std::forward<Arguments>(arguments)...);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return {false,
                cuda_failure(("CUDA could not launch " + std::string(kernel_name)).c_str(), error)};
    return {true, {}};
}
} // namespace gca
