// SPDX-License-Identifier: 0BSD
/// @file organized_normals.cu
/// The organized normals: their request from the pool, the checks before an estimate, and its
/// kernel (see organized_normals.hpp; the method: docs/methods/normals.md).

#include "geometry/organized_normals.hpp"

#include "geometry/float3_eigen.cuh"
#include "geometry/float3_ops.cuh"
#include "geometry/symmetric_eigen_3x3.cuh"
#include "gpu/gpu_messages.hpp"
#include "gpu/kernel_launch.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <string>
#include <utility>

#include <Eigen/Core>

namespace gca
{
namespace
{
/// The kernel's block: 32 x 8 threads, one warp along a row and 8 rows [?: a choice, the
/// organized points' block; its time on the board stands in docs/HARDWARE.md, "The per-frame
/// path's stages"; no other block was measured].
constexpr unsigned int block_width = 32;
constexpr unsigned int block_height = 8;

/// The largest window half size: (2 * 3 + 1)^2 = 49 pixels, whose membership fits the kernel's one
/// 64-bit word [S: docs/methods/normals.md, "The covariance, centred first"].
constexpr std::uint32_t max_radius_px = 3;

/// Says whether two points lie on one surface: the segment between them makes an angle of at
/// least 90 degrees less the steepest incidence with the line of sight to their midpoint
/// (docs/methods/normals.md, "One surface, or a depth edge"). Symmetric in a and b; a point lies
/// on one surface with itself.
/// @param a [in] the first point, in the camera frame, in m, z > 0.
/// @param b [in] the second point, in the camera frame, in m, z > 0.
/// @param sin2_max_incidence [in] the square of the sine of the steepest incidence, in (0, 1).
/// @return whether (d . m)^2 <= sin^2 * |d|^2 * |m|^2, d = b - a, m = a + b.
__device__ __forceinline__ bool on_one_surface(const float3 &a, const float3 &b,
                                               float sin2_max_incidence)
{
    const float3 d = b - a;
    const float3 m = a + b;
    const float along = dot(d, m);
    return along * along <= sin2_max_incidence * dot(d, d) * dot(m, m);
}

/// Estimates every pixel's normal from its window's points (docs/methods/normals.md): the
/// neighbourhood and its mean in a first pass over the window, the centred covariance in a second
/// over the pixels the first kept, then the eigenvector of the smallest eigenvalue, turned toward
/// the camera.
/// @param points [in] device memory of width * height points, rows dense: pixel (u, v)'s point
///        in the camera frame, in m, z > 0 where valid; owned by the organized points estimated
///        from, which hold it until the kernel has run (organized_normals::estimate()'s @param
///        points).
/// @param valid [in] device memory of width * height marks, rows dense: 1 for a valid pixel, 0 for
///        an invalid one; owned as points.
/// @param size [in] the picture's width and height, in px, each below 2^31 (an image that large
///        would need 24 GiB, more than the board holds [S: docs/HARDWARE.md]).
/// @param radius [in] the window's half size R, in px, 1 to max_radius_px.
/// @param min_points [in] the fewest points of a neighbourhood that give a normal, 2R + 2 to
///        (2R + 1)^2.
/// @param sin2_max_incidence [in] the square of the sine of the steepest incidence, in (0, 1).
/// @param normals [out] device memory of width * height normals, rows dense: pixel (u, v)'s unit
///        normal in the camera frame, or (0, 0, 0) when it has none; owned by the organized
///        normals, which hold it until the kernel has run (their end's @pre).
/// @note One thread for each pixel, on the launch's stream; a thread outside the picture writes
///       nothing. Allocates nothing.
__global__ void __launch_bounds__(block_width *block_height)
    estimate_kernel(const float3 *__restrict__ points, const std::uint8_t *__restrict__ valid,
                    frame_size size, int radius, std::uint32_t min_points, float sin2_max_incidence,
                    float3 *__restrict__ normals)
{
    const std::uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= size.width || v >= size.height)
        return;
    const std::size_t pixel = static_cast<std::size_t>(v) * size.width + u;
    const float3 none = make_float3(0.0f, 0.0f, 0.0f);
    if (valid[pixel] == 0)
    {
        normals[pixel] = none;
        return;
    }
    const float3 centre = points[pixel];
    const int width = static_cast<int>(size.width);
    const int height = static_cast<int>(size.height);
    const int side = 2 * radius + 1;
    const int first_row = max(static_cast<int>(v) - radius, 0);
    const int last_row = min(static_cast<int>(v) + radius, height - 1);
    const int first_column = max(static_cast<int>(u) - radius, 0);
    const int last_column = min(static_cast<int>(u) + radius, width - 1);

    // The first pass: the neighbourhood, one bit per window pixel, and the sum of its points'
    // offsets from the pixel's own. The offsets are a few mm to cm, so that the mean,
    // centre + offset_sum / count, is rounded at the scale of the pixel's point only once, by half
    // a float step of it [S: float32 ulp; docs/methods/normals.md, "The covariance, centred
    // first"].
    std::uint64_t members = 0;
    std::uint32_t count = 0;
    float3 offset_sum = none;
    for (int row = first_row; row <= last_row; row++)
    {
        for (int column = first_column; column <= last_column; column++)
        {
            const std::size_t neighbour = static_cast<std::size_t>(row) * size.width + column;
            if (valid[neighbour] == 0)
                continue;
            const float3 point = points[neighbour];
            if (!on_one_surface(centre, point, sin2_max_incidence))
            {
                // An adjacent pixel across a depth edge: the pixel lies at the edge.
                if (abs(row - static_cast<int>(v)) <= 1 && abs(column - static_cast<int>(u)) <= 1)
                {
                    normals[pixel] = none;
                    return;
                }
                continue;
            }
            const int bit = (row - static_cast<int>(v) + radius) * side +
                            (column - static_cast<int>(u) + radius);
            members |= std::uint64_t{1} << bit;
            count++;
            offset_sum = offset_sum + (point - centre);
        }
    }
    if (count < min_points)
    {
        normals[pixel] = none;
        return;
    }
    const float3 mean = centre + offset_sum / static_cast<float>(count);

    // The second pass: the sum of the outer products of the kept points less the mean. The
    // covariance's factor 1 / count changes no eigenvector and is left out. Float suffices: the
    // products are of centred coordinates, so their rounding is relative to the window's spread,
    // not to the distance, and the normals agree with the same sums in double within 7.1e-4
    // degrees on every picture of the test, at both stream sizes [M 2026-11-14:
    // docs/methods/normals.md].
    float xx = 0.0f;
    float xy = 0.0f;
    float xz = 0.0f;
    float yy = 0.0f;
    float yz = 0.0f;
    float zz = 0.0f;
    for (int row = first_row; row <= last_row; row++)
    {
        for (int column = first_column; column <= last_column; column++)
        {
            const int bit = (row - static_cast<int>(v) + radius) * side +
                            (column - static_cast<int>(u) + radius);
            if (((members >> bit) & 1) == 0)
                continue;
            const float3 q = points[static_cast<std::size_t>(row) * size.width + column] - mean;
            xx += q.x * q.x;
            xy += q.x * q.y;
            xz += q.x * q.z;
            yy += q.y * q.y;
            yz += q.y * q.z;
            zz += q.z * q.z;
        }
    }
    Eigen::Matrix3f covariance;
    covariance << xx, xy, xz, xy, yy, yz, xz, yz, zz;
    const symmetric_eigen_3x3 eigen = solve_symmetric_eigen_3x3(covariance);
    // The values ascend except for a diagonal matrix (symmetric_eigen_3x3.cuh): take the least.
    // Choosing its column by comparison instead cut the stack from 88 to 40 bytes and changed the
    // time by about +1 %, so it was not kept (docs/HARDWARE.md, "The per-frame path's stages").
    Eigen::Index least = 0;
    eigen.values.minCoeff(&least);
    float3 normal = to_float3(eigen.vectors.col(least));
    if (dot(normal, centre) > 0.0f)
        normal = -1.0f * normal;
    normals[pixel] = normal;
}

/// Checks the parameters.
/// @param parameters [in] the parameters.
/// @return empty when they are usable: radius_px 1 to max_radius_px, min_points 2 radius_px + 2
///         to (2 radius_px + 1)^2, max_incidence_deg finite, > 0 and < 90; else the reason.
/// @note Allocates for the reason; throws std::bad_alloc when that fails.
std::string parameters_refusal(const organized_normals_parameters &parameters)
{
    const std::uint32_t radius = parameters.radius_px;
    if (radius < 1 || radius > max_radius_px)
        return "the window's radius " + std::to_string(radius) + " px is not 1 to " +
               std::to_string(max_radius_px);
    const std::uint32_t fewest = 2 * radius + 2;
    const std::uint32_t window = (2 * radius + 1) * (2 * radius + 1);
    if (parameters.min_points < fewest || parameters.min_points > window)
        return "the fewest points " + std::to_string(parameters.min_points) + " is not " +
               std::to_string(fewest) + " to " + std::to_string(window) +
               " for a window of radius " + std::to_string(radius) + " px";
    const float incidence = parameters.max_incidence_deg;
    if (!(std::isfinite(incidence) && incidence > 0.0f && incidence < 90.0f))
        return "the steepest incidence " + std::to_string(incidence) +
               " deg is not a finite number above 0 and below 90";
    return {};
}
} // namespace

request_result<organized_normals> organized_normals::request(gpu_pool &pool,
                                                             const frame_size &size) noexcept
{
    request_result<gpu_image<float3>> normals = pool.request<float3>(size);
    if (normals.status != request_status::granted)
        return refused_as<organized_normals>(normals);
    request_result<organized_normals> result;
    result.status = request_status::granted;
    result.granted.m_normals = std::move(normals.granted);
    return result;
}

step_result organized_normals::estimate(const organized_points &points,
                                        const organized_normals_parameters &parameters,
                                        cudaStream_t stream) noexcept
{
    if (m_normals.empty())
        return {false, "the organized normals hold no image"};
    std::string refusal = parameters_refusal(parameters);
    if (!refusal.empty())
        return {false, std::move(refusal)};
    if (points.points().empty())
        return {false, "the organized points hold no images"};
    if (points.size() != size())
        return {false, "the organized points are " + size_text(points.size()) +
                           ", the organized normals " + size_text(size())};
    const double sine = std::sin(double{parameters.max_incidence_deg} * std::numbers::pi / 180.0);
    const dim3 block(block_width, block_height);
    return launch_kernel("the organized normals' kernel", &estimate_kernel, grid_for(size(), block),
                         block, stream, points.points().data(), points.validity_mask().data(),
                         size(), static_cast<int>(parameters.radius_px), parameters.min_points,
                         static_cast<float>(sine * sine), m_normals.data());
}

frame_size organized_normals::size() const noexcept
{
    return m_normals.size();
}

const gpu_image<float3> &organized_normals::normals() const noexcept
{
    return m_normals;
}
} // namespace gca
