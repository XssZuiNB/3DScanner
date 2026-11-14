// SPDX-License-Identifier: 0BSD
/// @file frame_pyramid.cu
/// The frame's pyramid: its request from the pool, the build of its levels in order, and the
/// kernels of level 0's intensity and of a coarser level's combining (see frame_pyramid.hpp; the
/// method: docs/methods/frame-pyramid.md).

#include "geometry/frame_pyramid.hpp"

#include "geometry/float3_ops.cuh"
#include "geometry/one_surface.cuh"
#include "gpu/kernel_launch.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <string>
#include <utility>

namespace gca
{
namespace
{
/// Computes every pixel's intensity at level 0: the Rec. 709 luma of its color, scaled to [0, 1],
/// at a valid pixel, and 0 at an invalid one (docs/methods/frame-pyramid.md, "The intensity").
/// @param color [in] device memory of width * height colors, RGB, 8 bits a channel (x red, y
///        green, z blue), rows dense; owned by the GPU frame built from, which holds it until the
///        kernel has run (frame_pyramid::build()'s @param frame).
/// @param valid [in] device memory of width * height marks, 1 valid, 0 invalid, rows dense; level
///        0's validity mask, owned by the pyramid.
/// @param size [in] the picture's width and height, in px.
/// @param intensity [out] device memory of width * height intensities, rows dense; owned by the
///        pyramid, which holds it until the kernel has run (its end's @pre).
/// @note One thread for each pixel, on the launch's stream; a thread outside the picture writes
///       nothing. Allocates nothing.
__global__ void __launch_bounds__(picture_block_width *picture_block_height)
    intensity_kernel(const uchar3 *__restrict__ color, const std::uint8_t *__restrict__ valid,
                     frame_size size, float *__restrict__ intensity)
{
    const std::uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= size.width || v >= size.height)
        return;
    const std::size_t pixel = static_cast<std::size_t>(v) * size.width + u;
    if (valid[pixel] == 0)
    {
        intensity[pixel] = 0.0f;
        return;
    }
    const uchar3 c = color[pixel];
    // Rec. 709 luma [S: ITU-R BT.709-6]: three products and two sums of numbers below 256, then
    // the division, each rounded once, a few float steps of 1 [S: float32 ulp].
    const float luma = 0.2126f * static_cast<float>(c.x) + 0.7152f * static_cast<float>(c.y) +
                       0.0722f * static_cast<float>(c.z);
    intensity[pixel] = luma / 255.0f;
}

/// Combines every pixel of a coarser level from the level before (docs/methods/frame-pyramid.md,
/// "A coarser level"): pixel (u, v) is valid exactly when its centre pixel (2u, 2v) of the finer
/// level is; its point and intensity are the means, weighted by the triangle (2 - |i|)(2 - |j|),
/// over the centre and each valid pixel of its 3 x 3 window that lies on one surface with it.
/// @param fine_points [in] device memory of the finer level's points, rows dense, in the camera
///        frame, in m, z > 0 where valid; owned by the pyramid, as every image here.
/// @param fine_valid [in] device memory of the finer level's marks, 1 valid, 0 invalid.
/// @param fine_intensity [in] device memory of the finer level's intensities, in [0, 1].
/// @param fine [in] the finer level's width and height, in px.
/// @param coarse [in] the coarser level's width and height, in px: ceil(fine / 2) each, so that
///        every centre pixel lies in the finer level.
/// @param sin2_max_incidence [in] the square of the sine of the finer level's steepest incidence,
///        in (0, 1).
/// @param coarse_points [out] device memory of the coarser level's points; (0, 0, 0) where
///        invalid.
/// @param coarse_valid [out] device memory of the coarser level's marks.
/// @param coarse_intensity [out] device memory of the coarser level's intensities; 0 where
///        invalid.
/// @note One thread for each pixel of the coarser level, on the launch's stream; a thread outside
///       it writes nothing. Allocates nothing.
__global__ void __launch_bounds__(picture_block_width *picture_block_height)
    combine_kernel(const float3 *__restrict__ fine_points,
                   const std::uint8_t *__restrict__ fine_valid,
                   const float *__restrict__ fine_intensity, frame_size fine, frame_size coarse,
                   float sin2_max_incidence, float3 *__restrict__ coarse_points,
                   std::uint8_t *__restrict__ coarse_valid, float *__restrict__ coarse_intensity)
{
    const std::uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= coarse.width || v >= coarse.height)
        return;
    const std::size_t pixel = static_cast<std::size_t>(v) * coarse.width + u;
    const int centre_u = 2 * static_cast<int>(u);
    const int centre_v = 2 * static_cast<int>(v);
    const std::size_t centre = static_cast<std::size_t>(centre_v) * fine.width + centre_u;
    if (fine_valid[centre] == 0)
    {
        coarse_points[pixel] = make_float3(0.0f, 0.0f, 0.0f);
        coarse_valid[pixel] = 0;
        coarse_intensity[pixel] = 0.0f;
        return;
    }
    const float3 centre_point = fine_points[centre];
    // The mean of the offsets from the centre's point, a few mm, so that the mean is rounded at
    // the scale of the point only once [S: float32 ulp]. The weights are whole numbers whose sum,
    // at most 16, is exact.
    float3 offset_sum = make_float3(0.0f, 0.0f, 0.0f);
    float intensity_sum = 0.0f;
    float weight_sum = 0.0f;
    for (int j = -1; j <= 1; j++)
    {
        const int row = centre_v + j;
        if (row < 0 || row >= static_cast<int>(fine.height))
            continue;
        for (int i = -1; i <= 1; i++)
        {
            const int column = centre_u + i;
            if (column < 0 || column >= static_cast<int>(fine.width))
                continue;
            const std::size_t neighbour = static_cast<std::size_t>(row) * fine.width + column;
            // An invalid pixel is never combined. Its point, (0, 0, 0), would fail the rule as
            // well, its segment to the centre lying along the line of sight, as long as sin^2 of
            // the incidence lies below 1 by more than the rule's rounding, which the normals'
            // largest steepest incidence keeps it (max_steepest_incidence_deg).
            if (fine_valid[neighbour] == 0)
                continue;
            const float3 point = fine_points[neighbour];
            // The centre lies on one surface with itself (the rule's d is 0).
            if (!on_one_surface(centre_point, point, sin2_max_incidence))
                continue;
            const float weight = static_cast<float>((2 - abs(i)) * (2 - abs(j)));
            offset_sum = offset_sum + weight * (point - centre_point);
            intensity_sum += weight * fine_intensity[neighbour];
            weight_sum += weight;
        }
    }
    coarse_points[pixel] = centre_point + offset_sum / weight_sum;
    coarse_valid[pixel] = 1;
    coarse_intensity[pixel] = intensity_sum / weight_sum;
}

/// Returns the size of the level after a level: half its rows and columns, rounded up, so that
/// every pixel of the next level has its centre pixel, twice its index, in the level.
/// @param size [in] the level's width and height, in px.
/// @return ceil(width / 2) x ceil(height / 2), in px; 0 stays 0.
frame_size half_size(const frame_size &size)
{
    return {size.width / 2 + size.width % 2, size.height / 2 + size.height % 2};
}

/// Returns the square of the sine of a level's steepest incidence for the combining: the
/// incidence θ_l with tan θ_l = tan θ / 2^level (docs/methods/frame-pyramid.md, "One surface, at
/// every level").
/// @param max_incidence_deg [in] the normals' steepest incidence θ, in degrees, finite, > 0, at
///        most max_steepest_incidence_deg.
/// @param level [in] the level the combining reads, the finer one, >= 0.
/// @return sin^2 θ_level, in (0, 1), at most sin^2 of max_steepest_incidence_deg, 0.99970
///         [S: computed], computed in double and rounded to float once.
float sin2_of_level(float max_incidence_deg, std::size_t level)
{
    const double tangent = std::tan(double{max_incidence_deg} * std::numbers::pi / 180.0) /
                           std::ldexp(1.0, static_cast<int>(level));
    return static_cast<float>(tangent * tangent / (1.0 + tangent * tangent));
}
} // namespace

request_result<frame_pyramid> frame_pyramid::request(gpu_pool &pool,
                                                     const frame_size &size) noexcept
{
    request_result<frame_pyramid> result;
    frame_size level_size = size;
    for (std::size_t level = 0; level < level_count; level++)
    {
        request_result<organized_points> points = organized_points::request(pool, level_size);
        if (points.status != request_status::granted)
            return refused_as<frame_pyramid>(points);
        request_result<organized_normals> normals = organized_normals::request(pool, level_size);
        if (normals.status != request_status::granted)
            return refused_as<frame_pyramid>(normals);
        request_result<gpu_image<float>> intensity = pool.request<float>(level_size);
        if (intensity.status != request_status::granted)
            return refused_as<frame_pyramid>(intensity);
        level_images &granted = result.granted.m_levels[level];
        granted.points = std::move(points.granted);
        granted.normals = std::move(normals.granted);
        granted.intensity = std::move(intensity.granted);
        level_size = half_size(level_size);
    }
    result.status = request_status::granted;
    return result;
}

step_result frame_pyramid::build(const gpu_frame &frame, const frame_pyramid_parameters &parameters,
                                 cudaStream_t stream) noexcept
{
    level_images &frame_level = m_levels[0];
    if (frame_level.points.points().empty())
        return {false, "the frame's pyramid holds no levels"};
    std::string refusal = organized_normals::parameters_refusal(parameters.normals);
    if (!refusal.empty())
        return {false, std::move(refusal)};
    // The first work queued: its refusals leave every level as it was.
    step_result made = frame_level.points.deproject(frame, parameters.range, stream);
    if (!made.done)
        return made;
    const gca::intrinsics &k = frame.description().intrinsics;
    for (std::size_t level = 0; level < level_count; level++)
    {
        const float scale = std::ldexp(1.0f, -static_cast<int>(level));
        m_levels[level].intrinsics = {k.fx * scale, k.fy * scale, k.cx * scale, k.cy * scale};
    }
    const dim3 block = picture_block();
    made =
        launch_kernel("the intensity's kernel", &intensity_kernel, grid_for(size(0), block), block,
                      stream, frame.color().data(), frame_level.points.validity_mask().data(),
                      size(0), frame_level.intensity.data());
    if (!made.done)
        return made;
    for (std::size_t level = 0; level < level_count; level++)
    {
        if (level > 0)
        {
            made = combine(level, sin2_of_level(parameters.normals.max_incidence_deg, level - 1),
                           stream);
            if (!made.done)
                return made;
        }
        level_images &l = m_levels[level];
        made = l.normals.estimate(l.points, parameters.normals, stream);
        if (!made.done)
            return made;
    }
    return made;
}

step_result frame_pyramid::combine(std::size_t level, float sin2_max_incidence,
                                   cudaStream_t stream) noexcept
{
    const level_images &fine = m_levels[level - 1];
    level_images &coarse = m_levels[level];
    const dim3 block = picture_block();
    return launch_kernel(
        "the pyramid's combining kernel", &combine_kernel, grid_for(coarse.points.size(), block),
        block, stream, fine.points.points().data(), fine.points.validity_mask().data(),
        fine.intensity.data(), fine.points.size(), coarse.points.size(), sin2_max_incidence,
        coarse.points.points_to_write(organized_points::pyramid_key()).data(),
        coarse.points.validity_mask_to_write(organized_points::pyramid_key()).data(),
        coarse.intensity.data());
}

frame_size frame_pyramid::size(std::size_t level) const noexcept
{
    return m_levels.at(level).points.size();
}

const organized_points &frame_pyramid::points(std::size_t level) const noexcept
{
    return m_levels.at(level).points;
}

const organized_normals &frame_pyramid::normals(std::size_t level) const noexcept
{
    return m_levels.at(level).normals;
}

const gpu_image<float> &frame_pyramid::intensity(std::size_t level) const noexcept
{
    return m_levels.at(level).intensity;
}

gca::intrinsics frame_pyramid::level_intrinsics(std::size_t level) const noexcept
{
    return m_levels.at(level).intrinsics;
}
} // namespace gca
