// SPDX-License-Identifier: 0BSD
/// @file registration.cu
/// The registration of a frame against a surface: its request from the pool, the checks before an
/// alignment, the kernel of the surface's intensity gradient, the kernels that associate and build
/// one iteration's system and sums of both terms, and the levels' iterations (see registration.hpp;
/// the method: docs/methods/registration.md).

#include "registration/registration.hpp"

#include "geometry/float3_ops.cuh"
#include "gpu/gpu_messages.hpp"
#include "gpu/kernel_launch.cuh"
#include "registration/normal_equations.hpp"
#include "types/camera_param.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numbers>
#include <string>
#include <utility>

namespace gca
{
namespace
{
static_assert(picture_block_width == 32, "each row of the kernel's block is one warp");
static_assert(picture_block_width * picture_block_height >= sum_count,
              "the block's first 35 threads add its warps' sums, one sum each");

/// The warps of the system kernel's block: one along each of its rows.
constexpr unsigned int block_warps = picture_block_height;

/// The threads of each block of the kernel that adds the blocks' sums, one block for each of the 35
/// sums [?: a choice; its time is part of the iteration's, docs/HARDWARE.md].
constexpr unsigned int total_threads = 256;
static_assert((total_threads & (total_threads - 1)) == 0, "total_kernel's tree halves the threads");

/// The largest d_max the registration takes, in m: beyond it the association by projection pairs
/// points of surfaces that are not the same [?: a choice; docs/methods/registration.md,
/// "Parameters"].
constexpr float largest_max_distance_m = 0.5f;

/// The most iterations the registration takes [?: a choice; docs/methods/registration.md,
/// "Parameters"].
constexpr std::uint32_t largest_max_iterations = 100;

/// The fewest neighbours of a pixel's gradient window that give it a gradient: 3, so that the 2 x 2
/// system of the fit has more equations than unknowns [?: initial; docs/methods/registration.md,
/// "The color term"].
constexpr unsigned int min_gradient_neighbours = 3;

/// The least ratio of the smaller to the larger eigenvalue of the gradient's 2 x 2 system: below
/// it the neighbours lie nearly on one line of the tangent plane, which holds the gradient across
/// it too weakly [?: a choice, far below a window's spread over two rows, far above float's
/// rounding].
constexpr float min_gradient_eigenvalue_ratio = 1e-3f;

/// How far the initial transform's linear part may lie from a rotation: |R^T R - I| per entry
/// [?: a choice, far above double's rounding of a product of rotations, far below a mistake].
constexpr double rotation_tolerance = 1e-6;

/// The refusal when the registration holds no buffers.
constexpr char holds_no_buffers[] = "the registration holds no buffers";

/// A picture's images as a kernel reads them: device memory of size.width * size.height pixels,
/// rows dense, owned by the caller of align(), which holds them until the kernels have run.
struct picture
{
    const float3 *points;      ///< pixel (u, v)'s point, in its camera frame, in m
    const std::uint8_t *valid; ///< pixel (u, v)'s validity mark, 1 or 0
    const float3 *normals;     ///< pixel (u, v)'s unit normal, or (0, 0, 0) for none
    const float *intensity;    ///< pixel (u, v)'s intensity, in [0, 1] where valid
    frame_size size;           ///< the picture's width and height, in px
};

/// What one iteration's kernel takes besides the pictures: the current transform, the centre, the
/// association's thresholds and the terms' weights, in float.
struct iteration_inputs
{
    float3 rotation_row_0;       ///< R's first row: the transform's rotation, F to S
    float3 rotation_row_1;       ///< R's second row
    float3 rotation_row_2;       ///< R's third row
    float3 translation_m;        ///< t, F to S, in m
    float3 centre_m;             ///< c, the centre the products are taken about, in S, in m
    gca::intrinsics intrinsics;  ///< the surface level's intrinsics, in px
    float max_distance2_m2;      ///< d_max² of the level, in m²
    float min_normal_cos;        ///< cos θ_n
    float huber_threshold_m;     ///< δ_G, in m
    float point_to_plane_weight; ///< λ, in (0, 1]
    /// 1 - λ, in [0, 1); 0 leaves the color term out, so that it adds nothing, not even a NaN.
    float color_weight;
};

/// Says whether a vector is (0, 0, 0), a missing normal.
/// @param n [in] the vector.
/// @return whether every component is 0.
/// @note Device code; allocates nothing.
__device__ __forceinline__ bool is_zero(const float3 &n)
{
    return n.x == 0.0f && n.y == 0.0f && n.z == 0.0f;
}

/// Says how strongly Huber's weight takes a residual: 1 up to the threshold, threshold / |r|
/// beyond (docs/methods/registration.md, "The robust weight").
/// @param r [in] the residual, finite.
/// @param threshold [in] δ, > 0, in the residual's unit.
/// @return the weight, in (0, 1].
/// @note Device code; allocates nothing.
__device__ __forceinline__ float huber_weight(float r, float threshold)
{
    const float abs_r = fabsf(r);
    return abs_r <= threshold ? 1.0f : threshold / abs_r;
}

/// Writes or adds one residual's terms of the system: w J J^T's upper triangle and w J r,
/// J = [arm x j; j]. The point-to-plane term writes, so that its products are rounded as they were
/// before the color term joined, each product its own rounding [S: nvcc may fuse a product with
/// the addition that follows it into one rounding, CUDA C++ Programming Guide, "Floating-Point
/// Standard"]; the color term adds onto them.
/// @tparam add whether the terms are added to the pixel's (the color term) or written (the
///         point-to-plane term).
/// @param arm [in] s - c, the point less the centre, in S, in m.
/// @param j [in] the residual's derivative by a translation: the normal n for the point-to-plane
///        term, the gradient d_q for the color term, in S.
/// @param r [in] the residual: r_G in m for the point-to-plane term, r_C in intensity units for the
///        color term, finite.
/// @param w [in] its weight: λ w_G for the point-to-plane term, (1 - λ) w_C for the color term,
///        each Huber's weight in (0, 1]; so in (0, 1].
/// @param terms [in,out] the pixel's terms, at normal_equations.hpp's positions; written or added
///        to.
/// @note Device code; allocates nothing.
template <bool add>
__device__ __forceinline__ void residual_terms(const float3 &arm, const float3 &j, float r, float w,
                                               float (&terms)[sum_count])
{
    const float3 lever = cross(arm, j);
    const float jacobian[6] = {lever.x, lever.y, lever.z, j.x, j.y, j.z};
    std::size_t k = sum_hessian;
#pragma unroll
    for (std::size_t row_j = 0; row_j < 6; row_j++)
    {
        const float weighted = w * jacobian[row_j];
#pragma unroll
        for (std::size_t column_j = row_j; column_j < 6; column_j++)
        {
            if constexpr (add)
                terms[k++] += weighted * jacobian[column_j];
            else
                terms[k++] = weighted * jacobian[column_j];
        }
        if constexpr (add)
            terms[sum_gradient + row_j] += weighted * r;
        else
            terms[sum_gradient + row_j] = weighted * r;
    }
}

/// Forms one pixel's terms of the system and its sums (docs/methods/registration.md, "The
/// association by projection", "The residuals and their Jacobians" and "The system, centred and
/// scaled"): nothing for a pixel that is no candidate; the candidates' count alone for a candidate
/// without a correspondence; and for a correspondence of point s = R p + t with the surface's point
/// q, normal n, intensity I_q and gradient d_q: the point-to-plane residual r_G = n . (s - q) with
/// the weight λ w_G, w_G Huber's at δ_G; where d_q is not (0, 0, 0) and 1 - λ > 0, the color
/// residual r_C = I_q + d_q . (s - q) - I_p with the weight (1 - λ) w_C, w_C Huber's at δ_C; both
/// terms' λ-weighted w J J^T's upper triangle and w J r about c; and the point-to-plane term's
/// λ w_G, λ w_G (s - c), λ w_G |s - c|², r_G² and the correspondences' count. All in float.
/// @param frame [in] the frame's level, in F.
/// @param surface [in] the surface's level, in S.
/// @param gradients [in] device memory of the surface level's gradients, d_q at q's index, in S, in
///        intensity units per m, (0, 0, 0) for none; owned by the registration.
/// @param in [in] the transform, the centre, the thresholds and the weights.
/// @param u [in] the frame's column, in px; any value, outside the picture too.
/// @param v [in] the frame's row, in px; any value.
/// @param terms [out] the terms, at normal_equations.hpp's positions; 0 where the pixel has none.
/// @note Device code; allocates nothing.
__device__ __forceinline__ void pixel_terms(const picture &frame, const picture &surface,
                                            const float3 *gradients, const iteration_inputs &in,
                                            std::uint32_t u, std::uint32_t v,
                                            float (&terms)[sum_count])
{
#pragma unroll
    for (std::size_t k = 0; k < sum_count; k++)
        terms[k] = 0.0f;
    if (u >= frame.size.width || v >= frame.size.height)
        return;
    const std::size_t pixel = static_cast<std::size_t>(v) * frame.size.width + u;
    const float3 m = frame.normals[pixel];
    if (frame.valid[pixel] == 0 || is_zero(m))
        return;
    terms[sum_candidates] = 1.0f;

    const float3 p = frame.points[pixel];
    const float3 s = make_float3(dot(in.rotation_row_0, p), dot(in.rotation_row_1, p),
                                 dot(in.rotation_row_2, p)) +
                     in.translation_m;
    if (!(s.z > 0.0f))
        return;
    const float column = in.intrinsics.fx * s.x / s.z + in.intrinsics.cx;
    const float row = in.intrinsics.fy * s.y / s.z + in.intrinsics.cy;
    // The nearest pixel lies inside the picture; a value that is not a number fails here.
    if (!(column >= -0.5f && column < static_cast<float>(surface.size.width) - 0.5f &&
          row >= -0.5f && row < static_cast<float>(surface.size.height) - 0.5f))
        return;
    // min(): x + 0.5 may round up to the width for x just below width - 0.5 [S: float32 ulp].
    const std::uint32_t partner_u =
        min(static_cast<std::uint32_t>(floorf(column + 0.5f)), surface.size.width - 1);
    const std::uint32_t partner_v =
        min(static_cast<std::uint32_t>(floorf(row + 0.5f)), surface.size.height - 1);
    const std::size_t partner =
        static_cast<std::size_t>(partner_v) * surface.size.width + partner_u;
    const float3 n = surface.normals[partner];
    if (surface.valid[partner] == 0 || is_zero(n))
        return;
    const float3 d = s - surface.points[partner];
    if (!(dot(d, d) <= in.max_distance2_m2))
        return;
    const float3 turned_m = make_float3(dot(in.rotation_row_0, m), dot(in.rotation_row_1, m),
                                        dot(in.rotation_row_2, m));
    if (!(dot(n, turned_m) >= in.min_normal_cos))
        return;

    const float r = dot(n, d);
    const float w = in.point_to_plane_weight * huber_weight(r, in.huber_threshold_m);
    const float3 arm = s - in.centre_m;
    residual_terms<false>(arm, n, r, w, terms);
    terms[sum_weight] = w;
    terms[sum_offset] = w * arm.x;
    terms[sum_offset + 1] = w * arm.y;
    terms[sum_offset + 2] = w * arm.z;
    terms[sum_spread] = w * dot(arm, arm);
    terms[sum_squared_residual] = r * r;
    terms[sum_correspondences] = 1.0f;

    // The color term, left out entirely at λ = 1: 0 times a term would still add a NaN.
    if (!(in.color_weight > 0.0f))
        return;
    const float3 gradient = gradients[partner];
    if (is_zero(gradient))
        return;
    const float r_color = surface.intensity[partner] + dot(gradient, d) - frame.intensity[pixel];
    residual_terms<true>(
        arm, gradient, r_color,
        in.color_weight * huber_weight(r_color, registration::color_huber_threshold), terms);
}

/// Gives a unit vector perpendicular to a unit normal, the first axis of its tangent plane's basis:
/// n x e normalized, e the coordinate axis along which n has its smallest component, so that
/// |n x e| >= sqrt(2/3) [S: computed, the largest of a unit vector's two other components].
/// @param n [in] the unit normal, in the surface's camera frame S.
/// @return the unit vector, perpendicular to n, in S.
/// @note Device code; allocates nothing.
__device__ __forceinline__ float3 tangent_axis(const float3 &n)
{
    const float x = fabsf(n.x);
    const float y = fabsf(n.y);
    const float z = fabsf(n.z);
    const float3 e = x <= y && x <= z ? make_float3(1.0f, 0.0f, 0.0f)
                     : y <= z         ? make_float3(0.0f, 1.0f, 0.0f)
                                      : make_float3(0.0f, 0.0f, 1.0f);
    const float3 a = cross(n, e);
    return a / norm(a);
}

/// Fits every surface pixel's intensity gradient in its tangent plane (docs/methods/
/// registration.md, "The color term"): over the valid neighbours of its (2 R_C + 1)^2 window, the
/// least squares of I_q' - I_q against the offsets q' - q in the tangent plane's basis a, b,
/// anchored at q; the gradient α a + β b. (0, 0, 0) for a pixel that is invalid, has no normal, has
/// fewer than min_gradient_neighbours valid neighbours, or whose 2 x 2 system's eigenvalues lie
/// further apart than min_gradient_eigenvalue_ratio, which a point that is not a number fails too.
/// No rule of one surface keeps the window off a depth edge: a pixel next to one has no normal when
/// the surface's normals mark it, as the normals stage's do, and a rule showed no benefit where
/// they did not (docs/methods/registration.md, "The color term").
/// @param surface [in] the surface's level, in S; its intensity finite at a valid pixel.
/// @param gradients [out] device memory of the level's width * height gradients, rows dense, in S,
///        in intensity units per m; owned by the registration.
/// @note One thread for each pixel of the level in the stages' block of 32 x 8 threads, on the
///       launch's stream; a thread outside the picture writes nothing. Allocates nothing.
__global__ void __launch_bounds__(picture_block_width *picture_block_height)
    gradient_kernel(picture surface, float3 *__restrict__ gradients)
{
    constexpr int radius = registration::gradient_radius_px;
    const std::uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= surface.size.width || v >= surface.size.height)
        return;
    const std::size_t pixel = static_cast<std::size_t>(v) * surface.size.width + u;
    const float3 none = make_float3(0.0f, 0.0f, 0.0f);
    const float3 n = surface.normals[pixel];
    if (surface.valid[pixel] == 0 || is_zero(n))
    {
        gradients[pixel] = none;
        return;
    }
    const float3 q = surface.points[pixel];
    const float intensity = surface.intensity[pixel];
    const float3 a = tangent_axis(n);
    const float3 b = cross(n, a);
    const int width = static_cast<int>(surface.size.width);
    const int height = static_cast<int>(surface.size.height);
    const int first_row = max(static_cast<int>(v) - radius, 0);
    const int last_row = min(static_cast<int>(v) + radius, height - 1);
    const int first_column = max(static_cast<int>(u) - radius, 0);
    const int last_column = min(static_cast<int>(u) + radius, width - 1);
    // The 2 x 2 normal equations of the offsets x = (a . (q' - q), b . (q' - q)) and the intensity
    // differences: Σ x x^T and Σ x (I_q' - I_q). The offsets are a few mm to cm, their products
    // rounded relative to the window's spread, not to the distance [S: float32 ulp].
    float aa = 0.0f;
    float ab = 0.0f;
    float bb = 0.0f;
    float ai = 0.0f;
    float bi = 0.0f;
    unsigned int count = 0;
    for (int row = first_row; row <= last_row; row++)
    {
        for (int column = first_column; column <= last_column; column++)
        {
            const std::size_t neighbour =
                static_cast<std::size_t>(row) * surface.size.width + column;
            if (neighbour == pixel || surface.valid[neighbour] == 0)
                continue;
            const float3 offset = surface.points[neighbour] - q;
            const float x_a = dot(a, offset);
            const float x_b = dot(b, offset);
            const float difference = surface.intensity[neighbour] - intensity;
            aa += x_a * x_a;
            ab += x_a * x_b;
            bb += x_b * x_b;
            ai += x_a * difference;
            bi += x_b * difference;
            count++;
        }
    }
    const float half_trace = 0.5f * (aa + bb);
    const float determinant = aa * bb - ab * ab;
    const float larger = half_trace + sqrtf(fmaxf(half_trace * half_trace - determinant, 0.0f));
    // The smaller eigenvalue is determinant / larger: at least the ratio times the larger.
    if (count < min_gradient_neighbours || !(larger > 0.0f) ||
        !(determinant >= min_gradient_eigenvalue_ratio * larger * larger))
    {
        gradients[pixel] = none;
        return;
    }
    const float alpha = (bb * ai - ab * bi) / determinant;
    const float beta = (aa * bi - ab * ai) / determinant;
    gradients[pixel] = alpha * a + beta * b;
}

/// Builds one iteration's system and sums for each block of the frame's pixels: each thread forms
/// its pixel's terms in float (pixel_terms()), each warp, a row of the block, adds its 32 threads'
/// terms in float by a tree of five shuffles, and the block adds its warps' sums in double, in a
/// fixed order (docs/methods/registration.md, "Precision": in double within the warp, an iteration
/// took 11 to 13 times as long).
/// @param frame [in] the frame's level, in F.
/// @param surface [in] the surface's level, in S.
/// @param gradients [in] device memory of the surface level's gradients (gradient_kernel()); owned
///        by the registration.
/// @param in [in] the transform, the centre, the thresholds and the weights.
/// @param block_sums [out] device memory of 35 sums for each block of the grid: sum k of block
///        b = blockIdx.y * gridDim.x + blockIdx.x at k * gridDim.x * gridDim.y + b; owned by the
///        registration, which holds it until the kernel has run.
/// @note One thread for each pixel of the frame's level in the stages' block of 32 x 8 threads, on
///       the launch's stream; a thread outside the picture adds zeros. Every thread takes part in
///       its warp's shuffles. Allocates nothing; 1120 bytes of shared memory [S: computed, 8 x 35
///       floats].
__global__ void __launch_bounds__(picture_block_width *picture_block_height)
    system_kernel(picture frame, picture surface, const float3 *__restrict__ gradients,
                  iteration_inputs in, double *__restrict__ block_sums)
{
    float terms[sum_count];
    pixel_terms(frame, surface, gradients, in, blockIdx.x * blockDim.x + threadIdx.x,
                blockIdx.y * blockDim.y + threadIdx.y, terms);
    __shared__ float warp_sums[block_warps][sum_count];
#pragma unroll
    for (std::size_t k = 0; k < sum_count; k++)
    {
        float x = terms[k];
#pragma unroll
        for (unsigned int offset = 16; offset > 0; offset /= 2)
            x += __shfl_down_sync(0xffffffffu, x, offset);
        if (threadIdx.x == 0)
            warp_sums[threadIdx.y][k] = x;
    }
    __syncthreads();
    // The warps' sums become double here, by 35 threads: a float converts to double exactly, and
    // a conversion on the GPU's FP64 unit costs a whole warp's instruction even for one lane.
    const unsigned int thread = threadIdx.y * blockDim.x + threadIdx.x;
    if (thread < sum_count)
    {
        double total = 0.0;
        for (unsigned int warp = 0; warp < block_warps; warp++)
            total += static_cast<double>(warp_sums[warp][thread]);
        const std::size_t blocks = static_cast<std::size_t>(gridDim.x) * gridDim.y;
        const std::size_t block = static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
        block_sums[thread * blocks + block] = total;
    }
}

/// Adds the blocks' sums into the 35 totals in double, each in a fixed order: each thread adds the
/// blocks it strides over, then a tree in shared memory, so that the same input gives the same
/// totals bit for bit.
/// @param block_sums [in] device memory of 35 sums for each of blocks blocks, sum k of block b at
///        k * blocks + b; owned by the registration.
/// @param blocks [in] the number of blocks the system kernel ran, > 0.
/// @param sums [out] device memory of the 35 totals, at normal_equations.hpp's positions; owned by
///        the registration.
/// @note One block of total_threads threads for each sum, on the launch's stream. Allocates
///       nothing; 2048 bytes of shared memory [S: computed, 256 doubles].
__global__ void __launch_bounds__(total_threads)
    total_kernel(const double *__restrict__ block_sums, std::uint32_t blocks,
                 double *__restrict__ sums)
{
    const std::size_t k = blockIdx.x;
    double partial = 0.0;
    for (std::uint32_t block = threadIdx.x; block < blocks; block += total_threads)
        partial += block_sums[k * blocks + block];
    __shared__ double partials[total_threads];
    partials[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned int stride = total_threads / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        sums[k] = partials[0];
}

/// Says whether a transform is rigid: finite, its linear part a rotation within rotation_tolerance.
/// @param transform [in] the transform.
/// @return whether every entry is finite, every entry of R^T R - I within rotation_tolerance of 0,
///         and det R > 0.
bool is_rigid(const Eigen::Isometry3d &transform)
{
    if (!transform.matrix().allFinite())
        return false;
    const Eigen::Matrix3d r = transform.linear();
    const double off = (r.transpose() * r - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff();
    return off <= rotation_tolerance && r.determinant() > 0.0;
}

/// Says why a surface's level cannot be registered against, if it cannot.
/// @param surface [in] the surface's level.
/// @param largest [in] the largest size the registration's buffers hold at that level, in px.
/// @return empty when its images hold memory, are of one size, at most largest along each side,
///         and its intrinsics can be used; else the reason.
/// @note Allocates for the reason; throws std::bad_alloc when that fails.
std::string surface_refusal(const organized_surface &surface, const frame_size &largest)
{
    if (surface.points.empty() || surface.validity_mask.empty() || surface.normals.empty() ||
        surface.intensity.empty())
        return "the surface's images are empty";
    const frame_size size = surface.points.size();
    if (!(surface.validity_mask.size() == size) || !(surface.normals.size() == size) ||
        !(surface.intensity.size() == size))
        return "the surface's points, marks, normals and intensity are not of one size: " +
               size_text(size) + ", " + size_text(surface.validity_mask.size()) + ", " +
               size_text(surface.normals.size()) + " and " + size_text(surface.intensity.size());
    if (size.width > largest.width || size.height > largest.height)
        return "the surface's " + size_text(size) + " are larger than the registration's " +
               size_text(largest);
    if (!usable_intrinsics(surface.intrinsics))
        return "the surface's intrinsics are not finite with fx and fy > 0";
    return {};
}

/// Says why the parameters cannot be used, if they cannot.
/// @param parameters [in] the parameters.
/// @return empty when each lies in its range (registration_parameters); else the reason.
/// @note Allocates only for the reason; throws std::bad_alloc when that fails.
std::string parameters_refusal(const registration_parameters &parameters)
{
    // The level's name is built only for a refusal: a call that is taken allocates nothing.
    const auto name = [](std::size_t level) { return "level " + std::to_string(level) + "'s "; };
    float smallest_d_max = largest_max_distance_m;
    for (std::size_t level = 0; level < frame_pyramid::level_count; level++)
    {
        const registration_level_parameters &at = parameters.levels[level];
        const float d_max = at.max_distance_m;
        if (!std::isfinite(d_max) || !(d_max > 0.0f) || d_max > largest_max_distance_m)
            return name(level) + "largest distance " + std::to_string(d_max) +
                   " m is not within (0, " + std::to_string(largest_max_distance_m) + "] m";
        smallest_d_max = std::min(smallest_d_max, d_max);
        // Level 0 gives the result: it iterates at least once.
        const std::uint32_t fewest = level == 0 ? 1 : 0;
        if (at.max_iterations < fewest || at.max_iterations > largest_max_iterations)
            return name(level) + "most iterations " + std::to_string(at.max_iterations) +
                   " are not within " + std::to_string(fewest) + " to " +
                   std::to_string(largest_max_iterations);
    }
    const float huber = parameters.huber_threshold_m;
    if (!std::isfinite(huber) || !(huber > 0.0f) || huber > smallest_d_max)
        return "Huber's threshold " + std::to_string(huber) +
               " m is not within (0, the smallest largest distance]";
    const float lambda = parameters.point_to_plane_weight;
    if (!std::isfinite(lambda) || !(lambda > 0.0f) || lambda > 1.0f)
        return "the point-to-plane term's weight " + std::to_string(lambda) +
               " is not within (0, 1]";
    return {};
}

/// Rounds the transform, the centre, the thresholds and the weights to the kernel's floats.
/// @param transform [in] the current transform, F to S, in m.
/// @param centre_m [in] the centre c, in S, in m.
/// @param intrinsics [in] the surface level's intrinsics, in px.
/// @param max_distance_m [in] the level's d_max, in m.
/// @param parameters [in] δ_G and λ.
/// @return the kernel's inputs, each value rounded to float once; 1 - λ computed in double first,
///         exactly 0 at λ = 1.
iteration_inputs inputs_of(const Eigen::Isometry3d &transform, const Eigen::Vector3d &centre_m,
                           const gca::intrinsics &intrinsics, float max_distance_m,
                           const registration_parameters &parameters)
{
    const Eigen::Matrix3d r = transform.linear();
    const Eigen::Vector3d t = transform.translation();
    const auto row = [&r](int i) {
        return make_float3(static_cast<float>(r(i, 0)), static_cast<float>(r(i, 1)),
                           static_cast<float>(r(i, 2)));
    };
    const auto to_float3 = [](const Eigen::Vector3d &x) {
        return make_float3(static_cast<float>(x.x()), static_cast<float>(x.y()),
                           static_cast<float>(x.z()));
    };
    const double angle_rad = double{registration::max_normal_angle_deg} * std::numbers::pi / 180.0;
    return {row(0),
            row(1),
            row(2),
            to_float3(t),
            to_float3(centre_m),
            intrinsics,
            max_distance_m * max_distance_m,
            static_cast<float>(std::cos(angle_rad)),
            parameters.huber_threshold_m,
            parameters.point_to_plane_weight,
            static_cast<float>(1.0 - double{parameters.point_to_plane_weight})};
}

/// Gives a frame's or a surface's level as the kernels read it.
/// @param points [in] the level's organized points.
/// @param valid [in] the level's validity marks, of the points' size.
/// @param normals [in] the level's normals, of the points' size.
/// @param intensity [in] the level's intensity, of the points' size.
/// @return the level's device pointers and size; valid as long as the images.
picture picture_of(const gpu_image<float3> &points, const gpu_image<std::uint8_t> &valid,
                   const gpu_image<float3> &normals, const gpu_image<float> &intensity)
{
    return {points.data(), valid.data(), normals.data(), intensity.data(), points.size()};
}

} // namespace

request_result<registration> registration::request(gpu_pool &pool, const frame_size &size) noexcept
{
    request_result<registration> result;
    const dim3 grid = grid_for(size, picture_block());
    const std::size_t blocks = static_cast<std::size_t>(grid.x) * grid.y;
    request_result<gpu_array<double>> block_sums = pool.request_array<double>(blocks * sum_count);
    if (block_sums.status != request_status::granted)
        return refused_as<registration>(block_sums);
    request_result<gpu_array<double>> sums = pool.request_array<double>(sum_count);
    if (sums.status != request_status::granted)
        return refused_as<registration>(sums);
    request_result<pinned_buffer> on_host = pool.request_pinned(sum_count * sizeof(double));
    if (on_host.status != request_status::granted)
        return refused_as<registration>(on_host);
    registration &made = result.granted;
    for (std::size_t level = 0; level < frame_pyramid::level_count; level++)
    {
        const frame_size sized = frame_pyramid::level_size(size, level);
        request_result<gpu_array<float3>> gradients =
            pool.request_array<float3>(static_cast<std::size_t>(sized.width) * sized.height);
        if (gradients.status != request_status::granted)
            return refused_as<registration>(gradients);
        made.m_gradients[level] = std::move(gradients.granted);
    }
    made.m_size = size;
    made.m_block_sums = std::move(block_sums.granted);
    made.m_sums = std::move(sums.granted);
    made.m_sums_on_host = std::move(on_host.granted);
    result.status = request_status::granted;
    return result;
}

registration_result registration::align(const frame_pyramid &frame, const surface_levels &surface,
                                        const Eigen::Isometry3d &initial,
                                        const registration_parameters &parameters,
                                        cudaStream_t stream) noexcept
{
    registration_result result;
    result.transform = initial;
    const auto refuse = [&result](std::string reason) {
        result.end = registration_end::refused;
        result.message = std::move(reason);
        return result;
    };
    if (m_block_sums.empty())
        return refuse(holds_no_buffers);
    if (frame.points(0).points().empty())
        return refuse("the frame's pyramid holds no levels");
    const frame_size frame_size_0 = frame.size(0);
    if (frame_size_0.width > m_size.width || frame_size_0.height > m_size.height)
        return refuse("the frame's level 0 of " + size_text(frame_size_0) +
                      " is larger than the registration's " + size_text(m_size));
    for (std::size_t level = 0; level < frame_pyramid::level_count; level++)
    {
        std::string reason =
            surface_refusal(surface[level], frame_pyramid::level_size(m_size, level));
        if (!reason.empty())
            return refuse("level " + std::to_string(level) +
                          " of the surface: " + std::move(reason));
    }
    if (!is_rigid(initial))
        return refuse("the initial transform is not finite, or its linear part is no rotation");
    std::string reason = parameters_refusal(parameters);
    if (!reason.empty())
        return refuse(std::move(reason));

    const auto fail = [&result](std::string message) {
        result.end = registration_end::failed;
        result.message = std::move(message);
        return result;
    };
    const dim3 block = picture_block();
    std::array<picture, frame_pyramid::level_count> surface_pictures{};
    for (std::size_t level = 0; level < frame_pyramid::level_count; level++)
    {
        const organized_surface &at = surface[level];
        surface_pictures[level] = picture_of(at.points, at.validity_mask, at.normals, at.intensity);
        if (parameters.levels[level].max_iterations == 0)
            continue;
        const step_result made =
            launch_kernel("the registration's gradient kernel", &gradient_kernel,
                          grid_for(surface_pictures[level].size, block), block, stream,
                          surface_pictures[level], m_gradients[level].data());
        if (!made.done)
            return fail(made.message);
    }

    // The first iteration centres on S's origin; each later one, at every level, on the last
    // centroid, which is metric and carries over from level to level.
    Eigen::Vector3d centre_m = Eigen::Vector3d::Zero();
    for (std::size_t step = 0; step < frame_pyramid::level_count; step++)
    {
        const std::size_t level = frame_pyramid::level_count - 1 - step;
        const registration_level_parameters &at = parameters.levels[level];
        if (at.max_iterations == 0)
            continue;
        const picture frame_picture =
            picture_of(frame.points(level).points(), frame.points(level).validity_mask(),
                       frame.normals(level).normals(), frame.intensity(level));
        const dim3 grid = grid_for(frame_picture.size, block);
        const std::uint32_t blocks = grid.x * grid.y;
        // The level's end; level 0's is the registration's.
        registration_end end = registration_end::degenerate;
        bool last_degenerate = true;
        for (std::uint32_t iteration = 0; iteration < at.max_iterations; iteration++)
        {
            const iteration_inputs in =
                inputs_of(result.transform, centre_m, surface[level].intrinsics, at.max_distance_m,
                          parameters);
            step_result made =
                launch_kernel("the registration's system kernel", &system_kernel, grid, block,
                              stream, frame_picture, surface_pictures[level],
                              m_gradients[level].data(), in, m_block_sums.data());
            if (!made.done)
                return fail(std::move(made.message));
            made = launch_kernel("the registration's total kernel", &total_kernel,
                                 dim3(static_cast<unsigned int>(sum_count)), dim3(total_threads),
                                 stream, m_block_sums.data(), blocks, m_sums.data());
            if (!made.done)
                return fail(std::move(made.message));
            const cudaError_t copy_error =
                cudaMemcpyAsync(m_sums_on_host.bytes().data(), m_sums.data(), m_sums.size_bytes(),
                                cudaMemcpyDeviceToHost, stream);
            if (copy_error != cudaSuccess)
                return fail(
                    cuda_failure("CUDA refused the copy of the registration's sums", copy_error));
            // The one wait of an iteration: the copy, and the stream's work before it, have run.
            const cudaError_t wait_error = cudaStreamSynchronize(stream);
            if (wait_error != cudaSuccess)
                return fail(cuda_failure("CUDA failed while the registration's sums were read",
                                         wait_error));
            result.iterations++;

            normal_sums sums{};
            std::memcpy(sums.data(), m_sums_on_host.bytes().data(), sizeof(sums));
            const system_solution solution = solve_normal_equations(sums, centre_m);
            result.fitness = solution.fitness;
            result.inlier_rmse_m = solution.inlier_rmse_m;
            result.eigenvalues = solution.eigenvalues;
            last_degenerate = solution.degenerate;
            if (solution.verdict == system_verdict::few_correspondences)
            {
                end = registration_end::few_correspondences;
                break;
            }
            if (solution.verdict == system_verdict::unsolvable)
            {
                end = registration_end::degenerate;
                break;
            }
            result.transform = solution.step * result.transform;
            centre_m = solution.centroid_m;
            // The last system decides: a small step on a degenerate system is no pose.
            if (solution.verdict == system_verdict::last_step)
            {
                end = last_degenerate ? registration_end::degenerate : registration_end::converged;
                break;
            }
            end = last_degenerate ? registration_end::degenerate
                                  : registration_end::iterations_used_up;
        }
        // A coarser level passes on its last transform whatever its end [?: a choice].
        if (level == 0)
            result.end = end;
    }
    return result;
}
} // namespace gca
