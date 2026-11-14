// SPDX-License-Identifier: 0BSD
/// @file registration.cu
/// The registration of a frame against a surface: its request from the pool, the checks before an
/// alignment, the kernels that associate and build one iteration's system and sums, and the
/// iterations (see registration.hpp; the method: docs/methods/registration.md).

#include "registration/registration.hpp"

#include "geometry/float3_ops.cuh"
#include "gpu/gpu_messages.hpp"
#include "gpu/kernel_launch.cuh"
#include "registration/normal_equations.hpp"
#include "types/camera_param.hpp"

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
    frame_size size;           ///< the picture's width and height, in px
};

/// What one iteration's kernel takes besides the pictures: the current transform, the centre and
/// the association's thresholds, in float.
struct iteration_inputs
{
    float3 rotation_row_0;      ///< R's first row: the transform's rotation, F to S
    float3 rotation_row_1;      ///< R's second row
    float3 rotation_row_2;      ///< R's third row
    float3 translation_m;       ///< t, F to S, in m
    float3 centre_m;            ///< c, the centre the products are taken about, in S, in m
    gca::intrinsics intrinsics; ///< the surface's intrinsics, in px
    float max_distance2_m2;     ///< d_max², in m²
    float min_normal_cos;       ///< cos θ_n
    float huber_threshold_m;    ///< δ_G, in m
};

/// Says whether a vector is (0, 0, 0), a missing normal.
/// @param n [in] the vector.
/// @return whether every component is 0.
/// @note Device code; allocates nothing.
__device__ __forceinline__ bool is_zero(const float3 &n)
{
    return n.x == 0.0f && n.y == 0.0f && n.z == 0.0f;
}

/// Forms one pixel's terms of the system and its sums (docs/methods/registration.md, "The
/// association by projection" and "The system, centred and scaled"): nothing for a pixel that is
/// no candidate; the candidates' count alone for a candidate without a correspondence; and for a
/// correspondence of point s = R p + t, normal n, residual r = n . (s - q) and Huber's weight w,
/// with J = [(s - c) x n; n]: w J J^T's upper triangle, w J r, w, w (s - c), w |s - c|², r² and the
/// correspondences' count. All in float.
/// @param frame [in] the frame's level, in F.
/// @param surface [in] the surface, in S.
/// @param in [in] the transform, the centre and the thresholds.
/// @param u [in] the frame's column, in px; any value, outside the picture too.
/// @param v [in] the frame's row, in px; any value.
/// @param terms [out] the terms, at normal_equations.hpp's positions; 0 where the pixel has none.
/// @note Device code; allocates nothing.
__device__ __forceinline__ void pixel_terms(const picture &frame, const picture &surface,
                                            const iteration_inputs &in, std::uint32_t u,
                                            std::uint32_t v, float (&terms)[sum_count])
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
    const float abs_r = fabsf(r);
    const float w = abs_r <= in.huber_threshold_m ? 1.0f : in.huber_threshold_m / abs_r;
    const float3 arm = s - in.centre_m;
    const float3 lever = cross(arm, n);
    const float j[6] = {lever.x, lever.y, lever.z, n.x, n.y, n.z};
    std::size_t k = sum_hessian;
#pragma unroll
    for (std::size_t row_j = 0; row_j < 6; row_j++)
    {
        const float weighted = w * j[row_j];
#pragma unroll
        for (std::size_t column_j = row_j; column_j < 6; column_j++)
            terms[k++] = weighted * j[column_j];
        terms[sum_gradient + row_j] = weighted * r;
    }
    terms[sum_weight] = w;
    terms[sum_offset] = w * arm.x;
    terms[sum_offset + 1] = w * arm.y;
    terms[sum_offset + 2] = w * arm.z;
    terms[sum_spread] = w * dot(arm, arm);
    terms[sum_squared_residual] = r * r;
    terms[sum_correspondences] = 1.0f;
}

/// Builds one iteration's system and sums for each block of the frame's pixels: each thread forms
/// its pixel's terms in float (pixel_terms()), each warp, a row of the block, adds its 32 threads'
/// terms in float by a tree of five shuffles, and the block adds its warps' sums in double, in a
/// fixed order (docs/methods/registration.md, "Precision": in double within the warp, an iteration
/// took 11 to 13 times as long).
/// @param frame [in] the frame's level, in F.
/// @param surface [in] the surface, in S.
/// @param in [in] the transform, the centre and the thresholds.
/// @param block_sums [out] device memory of 35 sums for each block of the grid: sum k of block
///        b = blockIdx.y * gridDim.x + blockIdx.x at k * gridDim.x * gridDim.y + b; owned by the
///        registration, which holds it until the kernel has run.
/// @note One thread for each pixel of the frame's level in the stages' block of 32 x 8 threads, on
///       the launch's stream; a thread outside the picture adds zeros. Every thread takes part in
///       its warp's shuffles. Allocates nothing; 1120 bytes of shared memory [S: computed, 8 x 35
///       floats].
__global__ void __launch_bounds__(picture_block_width *picture_block_height)
    system_kernel(picture frame, picture surface, iteration_inputs in,
                  double *__restrict__ block_sums)
{
    float terms[sum_count];
    pixel_terms(frame, surface, in, blockIdx.x * blockDim.x + threadIdx.x,
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

/// Says why a surface cannot be registered against, if it cannot.
/// @param surface [in] the surface.
/// @return empty when its images hold memory, are of one size, and its intrinsics can be used;
///         else the reason.
/// @note Allocates for the reason; throws std::bad_alloc when that fails.
std::string surface_refusal(const organized_surface &surface)
{
    if (surface.points.empty() || surface.validity_mask.empty() || surface.normals.empty())
        return "the surface's images are empty";
    const frame_size size = surface.points.size();
    if (!(surface.validity_mask.size() == size) || !(surface.normals.size() == size))
        return "the surface's points, marks and normals are not of one size: " + size_text(size) +
               ", " + size_text(surface.validity_mask.size()) + " and " +
               size_text(surface.normals.size());
    if (!usable_intrinsics(surface.intrinsics))
        return "the surface's intrinsics are not finite with fx and fy > 0";
    return {};
}

/// Says why the parameters cannot be used, if they cannot.
/// @param parameters [in] the parameters.
/// @return empty when each lies in its range (registration_parameters); else the reason.
/// @note Allocates for the reason; throws std::bad_alloc when that fails.
std::string parameters_refusal(const registration_parameters &parameters)
{
    const float d_max = parameters.max_distance_m;
    if (!std::isfinite(d_max) || !(d_max > 0.0f) || d_max > largest_max_distance_m)
        return "the largest distance " + std::to_string(d_max) + " m is not within (0, " +
               std::to_string(largest_max_distance_m) + "] m";
    const float huber = parameters.huber_threshold_m;
    if (!std::isfinite(huber) || !(huber > 0.0f) || huber > d_max)
        return "Huber's threshold " + std::to_string(huber) +
               " m is not within (0, the largest distance]";
    if (parameters.max_iterations < 1 || parameters.max_iterations > largest_max_iterations)
        return "the most iterations " + std::to_string(parameters.max_iterations) +
               " are not within 1 to " + std::to_string(largest_max_iterations);
    return {};
}

/// Rounds the transform, the centre and the thresholds to the kernel's floats.
/// @param transform [in] the current transform, F to S, in m.
/// @param centre_m [in] the centre c, in S, in m.
/// @param intrinsics [in] the surface's intrinsics, in px.
/// @param parameters [in] d_max and δ_G.
/// @return the kernel's inputs, each value rounded to float once.
iteration_inputs inputs_of(const Eigen::Isometry3d &transform, const Eigen::Vector3d &centre_m,
                           const gca::intrinsics &intrinsics,
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
    const float d_max = parameters.max_distance_m;
    return {row(0),
            row(1),
            row(2),
            to_float3(t),
            to_float3(centre_m),
            intrinsics,
            d_max * d_max,
            static_cast<float>(std::cos(angle_rad)),
            parameters.huber_threshold_m};
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
    made.m_size = size;
    made.m_block_sums = std::move(block_sums.granted);
    made.m_sums = std::move(sums.granted);
    made.m_sums_on_host = std::move(on_host.granted);
    result.status = request_status::granted;
    return result;
}

registration_result registration::align(const frame_pyramid &frame, std::size_t level,
                                        const organized_surface &surface,
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
    if (level >= frame_pyramid::level_count)
        return refuse("level " + std::to_string(level) + " is not one of the frame pyramid's " +
                      std::to_string(frame_pyramid::level_count) + " levels");
    const frame_size level_size = frame.size(level);
    if (frame.points(level).points().empty())
        return refuse("the frame's pyramid holds no levels");
    if (level_size.width > m_size.width || level_size.height > m_size.height)
        return refuse("the frame's level of " + size_text(level_size) +
                      " is larger than the registration's " + size_text(m_size));
    std::string reason = surface_refusal(surface);
    if (!reason.empty())
        return refuse(std::move(reason));
    if (!is_rigid(initial))
        return refuse("the initial transform is not finite, or its linear part is no rotation");
    reason = parameters_refusal(parameters);
    if (!reason.empty())
        return refuse(std::move(reason));

    const picture frame_picture{frame.points(level).points().data(),
                                frame.points(level).validity_mask().data(),
                                frame.normals(level).normals().data(), level_size};
    const picture surface_picture{surface.points.data(), surface.validity_mask.data(),
                                  surface.normals.data(), surface.points.size()};
    const dim3 block = picture_block();
    const dim3 grid = grid_for(level_size, block);
    const std::uint32_t blocks = grid.x * grid.y;
    const auto fail = [&result](std::string message) {
        result.end = registration_end::failed;
        result.message = std::move(message);
        return result;
    };
    // The first iteration centres on S's origin; each later one on the last centroid.
    Eigen::Vector3d centre_m = Eigen::Vector3d::Zero();
    bool last_degenerate = true;
    for (std::uint32_t iteration = 0; iteration < parameters.max_iterations; iteration++)
    {
        const iteration_inputs in =
            inputs_of(result.transform, centre_m, surface.intrinsics, parameters);
        step_result made =
            launch_kernel("the registration's system kernel", &system_kernel, grid, block, stream,
                          frame_picture, surface_picture, in, m_block_sums.data());
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
            return fail(
                cuda_failure("CUDA failed while the registration's sums were read", wait_error));
        result.iterations++;

        normal_sums sums{};
        std::memcpy(sums.data(), m_sums_on_host.bytes().data(), sizeof(sums));
        const system_solution solution = solve_normal_equations(sums, centre_m);
        result.fitness = solution.fitness;
        result.inlier_rmse_m = solution.inlier_rmse_m;
        result.eigenvalues = solution.eigenvalues;
        last_degenerate = solution.degenerate;
        switch (solution.verdict)
        {
        case system_verdict::few_correspondences:
            result.end = registration_end::few_correspondences;
            return result;
        case system_verdict::unsolvable:
            result.end = registration_end::degenerate;
            return result;
        case system_verdict::step:
        case system_verdict::last_step:
            break;
        }
        result.transform = solution.step * result.transform;
        centre_m = solution.centroid_m;
        // The last system decides: a small step on a degenerate system is no pose.
        if (solution.verdict == system_verdict::last_step)
        {
            result.end =
                last_degenerate ? registration_end::degenerate : registration_end::converged;
            return result;
        }
    }
    result.end =
        last_degenerate ? registration_end::degenerate : registration_end::iterations_used_up;
    return result;
}
} // namespace gca
