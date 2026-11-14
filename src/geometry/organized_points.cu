// SPDX-License-Identifier: 0BSD
/// @file organized_points.cu
/// The organized points: their request from the pool, the checks before a deprojection, and its
/// kernel (see organized_points.hpp; the method: docs/methods/organized-points.md).

#include "geometry/organized_points.hpp"

#include "gpu/gpu_messages.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace gca
{
namespace
{
/// The kernel's block: 32 x 8 threads, one warp along a row and 8 rows [?: a choice; its time on
/// the board stands in docs/HARDWARE.md, "The per-frame path's stages"; no other block was
/// measured].
constexpr unsigned int block_width = 32;
constexpr unsigned int block_height = 8;

/// Deprojects every pixel of a depth picture by the one pinhole convention and marks it valid or
/// not (docs/methods/organized-points.md).
/// @param depth [in] the depth picture, device memory of width * height values in depth units, 0
///        for no depth, rows dense; owned by the GPU frame deprojected, which holds it until the
///        kernel has run (organized_points::deproject()'s @param frame).
/// @param size [in] the picture's width and height, in px.
/// @param k [in] the intrinsics, in px: fx, fy finite and > 0, cx, cy finite.
/// @param depth_unit_m [in] meters per depth value, finite and > 0.
/// @param near_m [in] the depth range's lower limit, in m, >= 0.
/// @param far_m [in] the depth range's upper limit, in m, > near_m.
/// @param points [out] device memory of width * height points, rows dense: pixel (u, v)'s point
///        in the camera frame, in m, or (0, 0, 0) when it is invalid; owned by the organized
///        points, which hold it until the kernel has run (their end's @pre).
/// @param valid [out] device memory of width * height marks, rows dense: 1 for a valid pixel, 0
///        for an invalid one; owned by the organized points, as points.
/// @note One thread for each pixel, on the launch's stream; a thread outside the picture writes
///       nothing. Allocates nothing.
__global__ void __launch_bounds__(block_width *block_height)
    deproject_kernel(const std::uint16_t *__restrict__ depth, frame_size size, intrinsics k,
                     float depth_unit_m, float near_m, float far_m, float3 *__restrict__ points,
                     std::uint8_t *__restrict__ valid)
{
    const std::uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= size.width || v >= size.height)
        return;
    const std::size_t pixel = static_cast<std::size_t>(v) * size.width + u;
    const std::uint16_t d = depth[pixel];
    // d converts to float exactly (d <= 65535 < 2^24); z is rounded once, and the range is
    // compared on the z the point keeps.
    const float z = static_cast<float>(d) * depth_unit_m;
    const bool is_valid = d != 0 && z >= near_m && z <= far_m;
    points[pixel] = is_valid ? make_float3((static_cast<float>(u) - k.cx) * z / k.fx,
                                           (static_cast<float>(v) - k.cy) * z / k.fy, z)
                             : make_float3(0.0f, 0.0f, 0.0f);
    valid[pixel] = is_valid ? 1 : 0;
}

/// Returns the number of blocks that cover a length.
/// @param length [in] the length, in px.
/// @param block [in] a block's length, in px, > 0.
/// @return ceil(length / block), without overflow for any 32-bit length.
unsigned int blocks_for(std::uint32_t length, unsigned int block)
{
    return static_cast<unsigned int>((std::uint64_t{length} + block - 1) / block);
}

/// Checks the depth range.
/// @param parameters [in] the range.
/// @return empty when it is usable: both limits finite, 0 <= near_m < far_m; else the reason.
/// @note Allocates for the reason; throws std::bad_alloc when that fails.
std::string range_refusal(const organized_points_parameters &parameters)
{
    const float near_m = parameters.near_m;
    const float far_m = parameters.far_m;
    if (std::isfinite(near_m) && std::isfinite(far_m) && near_m >= 0.0f && far_m > near_m)
        return {};
    return "the depth range " + std::to_string(near_m) + " m to " + std::to_string(far_m) +
           " m is not one: both limits finite, 0 <= near < far";
}

/// Checks the frame's intrinsics and depth unit.
/// @param description [in] the frame's description.
/// @return empty when they are usable: fx and fy finite and > 0, cx and cy finite, the depth
///         unit finite and > 0; else the reason.
/// @note Allocates for the reason; throws std::bad_alloc when that fails.
std::string description_refusal(const frame_description &description)
{
    const intrinsics &k = description.intrinsics;
    const float unit_m = description.depth_unit_m;
    if (std::isfinite(k.fx) && k.fx > 0.0f && std::isfinite(k.fy) && k.fy > 0.0f &&
        std::isfinite(k.cx) && std::isfinite(k.cy) && std::isfinite(unit_m) && unit_m > 0.0f)
        return {};
    return "the frame's intrinsics (fx " + std::to_string(k.fx) + ", fy " + std::to_string(k.fy) +
           ", cx " + std::to_string(k.cx) + ", cy " + std::to_string(k.cy) +
           " px) or depth unit (" + std::to_string(unit_m) +
           " m) cannot be used: fx, fy and the unit finite and > 0, cx and cy finite";
}
} // namespace

request_result<organized_points> organized_points::request(gpu_pool &pool,
                                                           const frame_size &size) noexcept
{
    request_result<gpu_image<float3>> points = pool.request<float3>(size);
    if (points.status != request_status::granted)
        return refused_as<organized_points>(points);
    request_result<gpu_image<std::uint8_t>> marks = pool.request<std::uint8_t>(size);
    if (marks.status != request_status::granted)
        return refused_as<organized_points>(marks);
    request_result<organized_points> result;
    result.status = request_status::granted;
    result.granted.m_points = std::move(points.granted);
    result.granted.m_validity_mask = std::move(marks.granted);
    return result;
}

step_result organized_points::deproject(const gpu_frame &frame,
                                        const organized_points_parameters &parameters,
                                        cudaStream_t stream) noexcept
{
    if (m_points.empty())
        return {false, "the organized points hold no images"};
    std::string refusal = range_refusal(parameters);
    if (!refusal.empty())
        return {false, std::move(refusal)};
    const frame_description &description = frame.description();
    if (frame.depth().empty() || description.size != frame.depth().size())
        return {false, "the GPU frame holds no frame: none was uploaded into it"};
    if (description.size != size())
        return {false, "the frame is " + size_text(description.size) + ", the organized points " +
                           size_text(size())};
    refusal = description_refusal(description);
    if (!refusal.empty())
        return {false, std::move(refusal)};
    // An error an earlier call left as CUDA's last error would come back from the check after
    // the launch as if the launch had failed: it is reported, and cleared, before it.
    const cudaError_t earlier = cudaPeekAtLastError();
    if (earlier != cudaSuccess)
        return {false, cuda_failure("CUDA's last error, left by an earlier call, before the "
                                    "organized points' kernel",
                                    earlier)};
    const dim3 block(block_width, block_height);
    const dim3 grid(blocks_for(description.size.width, block_width),
                    blocks_for(description.size.height, block_height));
    deproject_kernel<<<grid, block, 0, stream>>>(
        frame.depth().data(), description.size, description.intrinsics, description.depth_unit_m,
        parameters.near_m, parameters.far_m, m_points.data(), m_validity_mask.data());
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        return {false, cuda_failure("CUDA could not launch the organized points' kernel", error)};
    return {true, {}};
}

frame_size organized_points::size() const noexcept
{
    return m_points.size();
}

const gpu_image<float3> &organized_points::points() const noexcept
{
    return m_points;
}

const gpu_image<std::uint8_t> &organized_points::validity_mask() const noexcept
{
    return m_validity_mask;
}
} // namespace gca
