// SPDX-License-Identifier: 0BSD
/// @file organized_normals.hpp
/// The organized normals, the per-frame path's stage after the organized points: each valid
/// pixel's normal from the points of its neighbours in the picture, by the eigenvector of their
/// covariance, centred first, turned toward the camera, at the pixel's own index. A pixel at a
/// depth edge or with too few neighbours on its surface has no normal, the zero vector. The
/// method, the rules and their sources stand in docs/methods/normals.md. The image comes from the
/// pool (gpu/gpu_pool.hpp) before it closes; the stage itself allocates nothing, waits for
/// nothing and reads nothing back.

#pragma once

#include "geometry/organized_points.hpp"
#include "gpu/gpu_image.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/step_result.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

namespace gca
{
/// The steepest incidence the stage takes, in degrees: 89 [?: a choice, the user's, 2026-11-14].
/// The rule of one surface needs the square of its sine below 1 in float, far enough that the
/// rounding of the rule's products cannot close the gap: at 89 degrees it is 0.99970, 3.0e-4 below
/// 1, against a few float steps, 1e-7 each; from 89.990 degrees on it rounds to 1, and the rule
/// takes every pair for one surface, a point and the camera's centre included [S: computed;
/// docs/methods/normals.md, "One surface, or a depth edge"].
inline constexpr float max_steepest_incidence_deg = 89.0f;

/// The stage's parameters (docs/methods/normals.md, "Input"): the window, the fewest points of a
/// neighbourhood that give a normal, and the steepest incidence that still counts as one surface.
struct organized_normals_parameters
{
    /// The window's half size R, in px: the (2R + 1)^2 pixels around a pixel; 1 to 3, so that the
    /// window's membership fits one 64-bit word (49 bits at R = 3).
    std::uint32_t radius_px;
    /// The fewest points of a neighbourhood, the pixel's own included, that give a normal; 2R + 2
    /// to (2R + 1)^2, so that the neighbourhood never lies on one line of the picture.
    std::uint32_t min_points;
    /// The steepest incidence, the angle between a surface's normal and the line of sight, at
    /// which two neighbouring points still count as one surface, in degrees; finite, > 0, at most
    /// max_steepest_incidence_deg.
    float max_incidence_deg;
};

/// A frame's organized normals on the GPU, of one size. Pixel (u, v) is element v * width + u, the
/// frame's own index: a unit normal in the camera frame (x right, y down, z forward) with
/// n . p <= 0 for the pixel's point p, or (0, 0, 0) when the pixel has no normal
/// (docs/methods/normals.md). Move-only. Not synchronized: one thread at a time uses one.
class organized_normals
{
public:
    /// Empty organized normals: no image; estimate() into them is refused.
    organized_normals() noexcept = default;

    /// Requests the image from the pool: the normals of a size.
    /// @param pool [in,out] the pool; open, else the request is refused as closed.
    /// @param size [in] the size of the organized points they will be estimated from, in px, each
    ///        > 0.
    /// @return granted, with organized normals of that size whose contents are not set until the
    ///         first estimate(); or the request's refusal, with its status and message
    ///         (gpu_pool::request()).
    /// @note Allocates device memory (cudaMalloc, 12 bytes per pixel), so it is called before the
    ///       pool is closed. Whether cudaMalloc waits for the GPU is not stated by CUDA [?];
    ///       nothing else waits. Allocates for the message when refused; a failed allocation ends
    ///       the program (std::terminate): the function is noexcept. Two threads may not request
    ///       from one pool at once (gpu_pool::request()).
    static request_result<organized_normals> request(gpu_pool &pool,
                                                     const frame_size &size) noexcept;

    /// Checks the stage's parameters, as estimate() does before it queues anything, so that a step
    /// that runs the normals among other work (geometry/frame_pyramid.hpp) refuses them before it
    /// queues its first kernel.
    /// @param parameters [in] the parameters.
    /// @return empty when each field lies in the range organized_normals_parameters states; else
    ///         the reason, the message estimate() refuses them with.
    /// @note Allocates for the reason; throws std::bad_alloc when that fails. Waits for nothing.
    static std::string parameters_refusal(const organized_normals_parameters &parameters);

    /// Organized normals are not copied: a copy allocates.
    organized_normals(const organized_normals &) = delete;
    organized_normals &operator=(const organized_normals &) = delete;

    /// Takes over other organized normals' image without allocating.
    /// @param other [in,out] the organized normals moved from; left empty.
    /// @note Allocates nothing and waits for nothing.
    organized_normals(organized_normals &&other) noexcept = default;

    /// Takes over other organized normals' image; this one's is freed first.
    /// @param other [in,out] the organized normals moved from; left empty. Moving onto itself
    ///        changes nothing.
    /// @return these organized normals.
    /// @pre No work queued on the GPU uses this one's image any more (gpu_image's move assignment
    ///      says what happens when some does).
    /// @note Frees with cudaFree, which may wait for the GPU (gpu/gpu_image.hpp). Allocates
    ///       nothing.
    organized_normals &operator=(organized_normals &&other) noexcept = default;

    /// Frees the image.
    /// @pre No work queued on the GPU uses it any more (as for the move assignment).
    /// @note Frees with cudaFree, which may wait for the GPU (gpu/gpu_image.hpp).
    ~organized_normals() = default;

    /// Estimates every pixel's normal from organized points. A valid pixel's neighbourhood is
    /// itself and the valid pixels of its window that lie on its surface; it has a normal unless
    /// one of its 8 adjacent pixels is valid and not on its surface (a depth edge) or its
    /// neighbourhood holds fewer than min_points points. The normal is the unit eigenvector of the
    /// smallest eigenvalue of the neighbourhood's covariance, centred on its mean, negated when
    /// it points away from the camera (docs/methods/normals.md).
    /// @param points [in] the organized points, of these normals' size, made on stream or on a
    ///        stream the caller has ordered before it; in the camera frame, in m. Owned by the
    ///        caller; they must hold their images until the work queued here has run.
    /// @param parameters [in] the window, the fewest points and the steepest incidence.
    /// @param stream [in] the CUDA stream of the current device the work is queued on; owned by
    ///        the caller. Work queued on it later sees the normals.
    /// @return done when the work is queued; not done with the reason when refused before it:
    ///         these normals are empty; the parameters are refused (parameters_refusal()); the
    ///         organized points are empty; their size is not these normals'. When refused, nothing
    ///         is queued and the image is left as it was. Not done with CUDA's error, which is
    ///         then cleared (gpu/gpu_messages.hpp): when CUDA's last error is set before the
    ///         launch, by an earlier call, with nothing queued and the image left as it was; when
    ///         CUDA refuses the launch, for one when the picture's rows need more than 65535
    ///         blocks of 8 rows [S: CUDA C++ Programming Guide, "Compute Capabilities", the grid's
    ///         y dimension], with the image's contents then unspecified.
    /// @note One kernel on stream, which reads the points and marks and writes the normals. Waits
    ///       for nothing: cudaPeekAtLastError before the launch and cudaGetLastError after it ask
    ///       only whether an error is set and whether CUDA accepted the launch. Allocates nothing
    ///       when done; allocates for the message when not, and a failed allocation ends the
    ///       program (std::terminate): the function is noexcept. Two threads may call it at once
    ///       only for different organized normals.
    step_result estimate(const organized_points &points,
                         const organized_normals_parameters &parameters,
                         cudaStream_t stream) noexcept;

    /// Gives the size the organized normals were requested with.
    /// @return their width and height, in px; 0 x 0 when empty.
    frame_size size() const noexcept;

    /// Gives the normals for the GPU to read.
    /// @return the image of the normals, unit vectors in the camera frame, of the size(); (0, 0, 0)
    ///         at a pixel without a normal, which a reader skips, since a residual along it is 0
    ///         (docs/methods/normals.md, "Output"); empty when the organized normals are. Valid as
    ///         long as they hold it.
    const gpu_image<float3> &normals() const noexcept;

private:
    gpu_image<float3> m_normals;
};
} // namespace gca
