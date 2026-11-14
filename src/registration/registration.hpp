// SPDX-License-Identifier: 0BSD
/// @file registration.hpp
/// The registration of a frame against a surface, the per-frame path's step after the ray cast:
/// from an initial transform, the transform that carries a frame's pyramid onto organized points
/// with normals and intensity, by colored ICP, its point-to-plane term and its color term weighed
/// by λ, coarse to fine over the pyramid's levels, with the association by projection, Huber's
/// weight on each term, a centred and scaled system judged by its eigenvalues, a damped step and
/// the SE(3) exponential; and the evidence of how well it is known, as a value. The method, its
/// parameters and its sources stand in docs/methods/registration.md. The buffers come from the pool
/// (gpu/gpu_pool.hpp) before it closes; the registration allocates nothing, and each iteration
/// reads back only its system and its sums, 280 bytes [S: computed, 35 doubles], after its one
/// written wait.

#pragma once

#include "geometry/frame_pyramid.hpp"
#include "gpu/gpu_array.hpp"
#include "gpu/gpu_image.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/pinned_buffer.hpp"
#include "types/camera_param.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include <Eigen/Geometry>

namespace gca
{
/// The surface a frame is registered against at one level, as the registration reads it: organized
/// points with normals and intensity in the surface's camera frame S (x right, y down, z forward,
/// in m) and the intrinsics of the camera whose picture they fill. A view of images others own, a
/// ray cast's or a level of another frame's pyramid (docs/methods/registration.md, "Input"); it
/// holds no memory, and the images must hold theirs until the work queued on them has run.
struct organized_surface
{
    /// The points: pixel (u, v) at index v * width + u, in S, in m; (0, 0, 0) where invalid.
    const gpu_image<float3> &points;
    /// The validity marks, of the points' size: 1 for a valid pixel, 0 for an invalid one.
    const gpu_image<std::uint8_t> &validity_mask;
    /// The normals, of the points' size: unit vectors in S toward the camera, or (0, 0, 0) for
    /// none, which a pixel then has no correspondence with.
    const gpu_image<float3> &normals;
    /// The intensity, of the points' size: in [0, 1] at a valid pixel, the frame pyramid's
    /// (docs/methods/frame-pyramid.md, "The intensity"); read only at valid pixels.
    const gpu_image<float> &intensity;
    /// The camera's intrinsics, in px of the picture, by the one pinhole convention
    /// (types/camera_param.hpp); fx, fy finite and > 0, cx, cy finite.
    gca::intrinsics intrinsics;
};

/// The surface at each level of the frame's pyramid: element l is level l, registered against the
/// frame's level l, with its own intrinsics, K_l of the level for a level of another frame's
/// pyramid (docs/methods/registration.md, "Coarse to fine").
using surface_levels = std::array<organized_surface, frame_pyramid::level_count>;

/// The registration's parameters at one level of the pyramid (docs/methods/registration.md,
/// "Parameters").
struct registration_level_parameters
{
    /// d_max_l, the association's largest distance between a frame's point and its partner at this
    /// level, in m; finite, > 0, at most 0.5.
    float max_distance_m;
    /// n_l, the most iterations at this level: 1 to 100 at level 0, 0 to 100 at a coarser level,
    /// where 0 skips the level.
    std::uint32_t max_iterations;
};

/// The registration's parameters for one call (docs/methods/registration.md, "Parameters"): the
/// values that differ between the object classes or the levels. align() refuses a value outside
/// its range.
struct registration_parameters
{
    /// Each level's parameters: element l for level l.
    std::array<registration_level_parameters, frame_pyramid::level_count> levels;
    /// δ_G, the point-to-plane residual beyond which Huber's weight falls below 1, in m; finite,
    /// > 0, at most the smallest of the levels' max_distance_m.
    float huber_threshold_m;
    /// λ, the point-to-plane term's weight; the color term's is 1 - λ. Finite, > 0, at most 1;
    /// 1 leaves the color term out, and the registration is the point-to-plane term's alone.
    float point_to_plane_weight;
};

/// How a registration ended (docs/methods/registration.md, "Output"); only converged gives a pose.
enum class registration_end
{
    /// A step below ε_ω and ε_υ on a last system whose smallest eigenvalue is at least τ: the
    /// transform is the minimum near the start, and the surface holds every motion.
    converged,
    /// Level 0's max_iterations steps, none small enough, the last system not degenerate.
    iterations_used_up,
    few_correspondences, ///< no correspondence, or a fitness below f_min
    /// The surface holds some motion too weakly: the last system's smallest eigenvalue below τ, or
    /// a system with no spread or that is not a number.
    degenerate,
    refused, ///< the inputs, before anything was queued; the message says why
    failed,  ///< CUDA's error, with its message
};

/// A registration's result, a value for every call (docs/methods/registration.md, "Output").
struct registration_result
{
    /// How it ended: level 0's end. A coarser level passes on its last transform whatever its end
    /// [?: a choice]; its end is not reported.
    registration_end end = registration_end::refused;
    std::string message; ///< why it was refused or failed, in plain English; empty otherwise
    /// The transform from the frame's camera frame F into the surface's frame S, in m: the last
    /// estimate, the initial transform when refused or failed before a step. A pose only when the
    /// registration converged. Each coarser level hands its last transform to the next finer one
    /// whatever its end [?: a choice]: after a coarse level that ended degenerate, it holds that
    /// level's damped steps, each shortened along the motions the level held weakly
    /// (docs/methods/registration.md, "Coarse to fine").
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    /// The share of the frame's candidates at level 0, its pixels with a valid point and a normal,
    /// that found a correspondence in the last system, 0 to 1; 0 with no candidate or no system.
    double fitness = 0.0;
    /// The root mean square of the last system's point-to-plane residuals, unweighted, in m; 0
    /// with no correspondence.
    double inlier_rmse_m = 0.0;
    std::uint32_t iterations = 0; ///< the systems built at every level; 0 when refused
    /// The conditioning: the eigenvalues of the last system, centred, scaled and normalized,
    /// ascending; the point-to-plane term's part holds each within 0 to 2, and the color term only
    /// adds (docs/methods/registration.md, "The system, centred and scaled"); zeros when no system
    /// was solved for them.
    std::array<double, 6> eigenvalues{};

    /// Says whether the registration converged, the one end whose transform is a pose.
    /// @return end == converged.
    bool converged() const noexcept
    {
        return end == registration_end::converged;
    }
};

/// The registration's buffers on the GPU and in pinned memory, for frames up to a size, and its
/// one operation, align(). Move-only. Not synchronized: one thread at a time uses one.
class registration
{
public:
    /// θ_n, the association's largest angle between a surface's normal and a frame's normal turned
    /// into the surface's frame, in degrees: 30 [?: initial; docs/methods/registration.md,
    /// "Parameters"].
    static constexpr float max_normal_angle_deg = 30.0f;

    /// δ_C, the color residual beyond which Huber's weight falls below 1, in intensity units of
    /// [0, 1]: 0.1 of the intensity's range [?: initial]; at 0.03 the fixture's real pair did not
    /// converge [M 2026-11-14: docs/methods/registration.md, "The robust weight"].
    static constexpr float color_huber_threshold = 0.1f;

    /// R_C, the half size of the window the surface's intensity gradient is fitted over, in px of
    /// the level: the (2 R_C + 1)^2 pixels around a pixel, 1; with 3 three of the tests' cases did
    /// not converge, with 2 every case converged in more iterations and further off
    /// [M 2026-11-14: docs/methods/registration.md, "The color term"].
    static constexpr int gradient_radius_px = 1;

    /// An empty registration: no buffers; align() with it is refused.
    registration() noexcept = default;

    /// Requests the buffers from the pool, for the pyramids of frames up to a size: the sums of
    /// each block of the kernel's threads, the totals, the pinned buffer they are read through, and
    /// the surface's intensity gradient at each level.
    /// @param pool [in,out] the pool; open, else the request is refused as closed.
    /// @param size [in] the largest frame the registration takes, in px, each > 0: a pyramid's
    ///        level 0; the gradients' levels are of frame_pyramid::level_size(size, l).
    /// @return granted, with a registration whose buffers' contents are not set; or the first of
    ///         the six requests that was refused, with its status and message
    ///         (gpu_pool::request_array(), gpu_pool::request_pinned()).
    /// @note Allocates device memory (cudaMalloc, five times: 280 bytes for each block of 32 x 8
    ///       pixels of size, 280 bytes, and 12 bytes for each pixel of each level for the
    ///       gradients, 4.03 MB at 640x400 and 16.1 MB at 1280x800) and 280 bytes of pinned host
    ///       memory (cudaMallocHost) [S: computed, 35 doubles for each block and once, a float3
    ///       over 1 + 1/4 + 1/16 of the pixels], so it is called before the pool is closed.
    ///       Whether those wait for the GPU is not stated by CUDA [?]; nothing else waits.
    ///       Allocates for the message when refused; a failed allocation ends the program
    ///       (std::terminate): the function is noexcept. Two threads may not request from one pool
    ///       at once (gpu_pool::request()).
    static request_result<registration> request(gpu_pool &pool, const frame_size &size) noexcept;

    /// A registration is not copied: a copy allocates.
    registration(const registration &) = delete;
    registration &operator=(const registration &) = delete;

    /// Takes over another registration's buffers without allocating.
    /// @param other [in,out] the registration moved from; left empty.
    /// @note Allocates nothing and waits for nothing.
    registration(registration &&other) noexcept = default;

    /// Takes over another registration's buffers; this one's are freed first.
    /// @param other [in,out] the registration moved from; left empty. Moving onto itself changes
    ///        nothing.
    /// @return this registration.
    /// @pre No work queued on the GPU uses this one's buffers any more (gpu_array's move
    ///      assignment says what happens when some does).
    /// @note Frees with cudaFree and cudaFreeHost, which may wait for the GPU (gpu/gpu_array.hpp,
    ///       gpu/pinned_buffer.hpp). Allocates nothing.
    registration &operator=(registration &&other) noexcept = default;

    /// Frees the buffers.
    /// @pre No work queued on the GPU uses them any more (as for the move assignment).
    /// @note Frees with cudaFree and cudaFreeHost, which may wait for the GPU.
    ~registration() = default;

    /// Registers a frame's pyramid against a surface's levels, coarse to fine, from an initial
    /// transform (docs/methods/registration.md, "Coarse to fine", "The iterations and the end").
    /// First the surface's intensity gradient at each level that iterates; then levels 2, 1 and 0
    /// in turn, each from the transform the coarser one ended with: each iteration associates the
    /// level's candidates with the surface's pixels by projection at the current transform, builds
    /// the system of both terms with Huber's weights on the GPU, reads it back and solves it on the
    /// host (registration/normal_equations.hpp) with a damped step, until a step is small enough, a
    /// system has too few correspondences or cannot step, or the level's iterations are used up. A
    /// coarser level passes on its last transform whatever its end [?: a choice]: one that ended
    /// degenerate leaves its damped steps behind, each shortened along the motions it held weakly,
    /// and one that ended with too few correspondences the transform it started from. Level 0's end
    /// is the result's, its last system's eigenvalues deciding whether it is degenerate.
    /// @param frame [in] the frame's pyramid, built on stream or on a stream the caller has ordered
    ///        before it; its points in the frame's camera frame F, in m, level 0 at most the size
    ///        the registration was requested for. Owned by the caller.
    /// @param surface [in] the surface's levels, their images made on stream or on a stream
    ///        ordered before it; level l's images of one size, at most
    ///        frame_pyramid::level_size() of the requested size at l; owned by the caller.
    /// @param initial [in] the initial transform from F into the surface's frame S, in m; finite,
    ///        its linear part a rotation within 1e-6 (|R^T R - I| per entry, det R > 0).
    /// @param parameters [in] each level's d_max and most iterations, δ_G and λ, each within its
    ///        range.
    /// @param stream [in] the CUDA stream of the current device the work is queued on and waited
    ///        for; owned by the caller.
    /// @return the result (registration_result): converged, with the transform and its evidence;
    ///         or ended with level 0's iterations used up, too few correspondences or as
    ///         degenerate, each with the last transform and the last system's evidence and never a
    ///         pose; refused, with the reason and the initial transform, before anything is queued,
    ///         when the registration is empty, the pyramid holds no levels or its level 0 is larger
    ///         than the requested size, a level of the surface has empty images or images not of
    ///         one size, intrinsics that cannot be used or a size larger than the registration's at
    ///         that level, the initial transform is not finite or not rigid, or a parameter lies
    ///         outside its range; or failed, with CUDA's error, which is cleared
    ///         (gpu/gpu_messages.hpp), when CUDA's last error is set before the first launch by an
    ///         earlier call, when CUDA refuses a launch, or when the copy or the wait fails, with
    ///         the transform reached before.
    /// @note One kernel on stream for each level that iterates, then per iteration two kernels, a
    ///       copy of 280 bytes [S: computed, 35 doubles] into the pinned buffer and one written
    ///       wait, cudaStreamSynchronize(stream), which also waits for the stream's earlier work.
    ///       Allocates nothing when it ends other than refused or failed; allocates for the message
    ///       then, and a failed allocation ends the program (std::terminate): the function is
    ///       noexcept. Not while another thread uses this registration.
    registration_result align(const frame_pyramid &frame, const surface_levels &surface,
                              const Eigen::Isometry3d &initial,
                              const registration_parameters &parameters,
                              cudaStream_t stream) noexcept;

private:
    frame_size m_size{0, 0};        ///< level 0's largest size, from the request
    gpu_array<double> m_block_sums; ///< 35 sums of each block: sum k of block b at k * blocks + b
    gpu_array<double> m_sums;       ///< the 35 totals (registration/normal_equations.hpp)
    pinned_buffer m_sums_on_host;   ///< the totals as the last iteration copied them
    /// The surface's intensity gradient at each level, in S, in intensity units per m, (0, 0, 0)
    /// for none: pixel (u, v) of a surface level of width W at u + v * W, for levels up to
    /// frame_pyramid::level_size(m_size, l).
    std::array<gpu_array<float3>, frame_pyramid::level_count> m_gradients;
};
} // namespace gca
