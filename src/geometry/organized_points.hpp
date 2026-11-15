// SPDX-License-Identifier: 0BSD
/// @file organized_points.hpp
/// The organized points, the per-frame path's first stage after the upload: a frame's depth
/// picture on the GPU becomes a point in the camera frame for every pixel, at the pixel's own
/// index, and a validity mask, by the one pinhole convention (types/camera_param.hpp) inside a
/// depth range. The method, the rules and their sources stand in
/// docs/methods/organized-points.md. The images come from the pool (gpu/gpu_pool.hpp) before it
/// closes; the stage itself allocates nothing, waits for nothing and reads nothing back.

#pragma once

#include "gpu/gpu_frame.hpp"
#include "gpu/gpu_image.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/step_result.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

namespace gca
{
/// The stage's parameters: the depth range. A pixel is valid only when its depth lies within it.
struct organized_points_parameters
{
    float near_m; ///< the range's lower limit, in m along the camera's z axis, finite, >= 0
    float far_m;  ///< the range's upper limit, in m along the camera's z axis, finite, > near_m
};

/// A frame's organized points and their validity mask on the GPU, of one size. Pixel (u, v) is
/// element v * width + u of both images, the frame's own index. A valid pixel's point is
/// ((u - cx) z / fx, (v - cy) z / fy, z) in the camera frame (x right, y down, z forward, in m)
/// with z = depth value * depth unit, and its mark is 1; an invalid pixel's point is (0, 0, 0)
/// and its mark 0 (docs/methods/organized-points.md). Move-only. Not synchronized: one thread at
/// a time uses one.
///
/// Two writers fill the images: deproject(), from a frame; and, for a coarser level of a frame's
/// pyramid, frame_pyramid::combine() (geometry/frame_pyramid.hpp), from the level before, where a
/// valid pixel's point is the weighted mean of the points it combines, with the same meaning of
/// point and mark (docs/methods/frame-pyramid.md). The pyramid writes through points_to_write()
/// and validity_mask_to_write(), which take a key only the pyramid can make; everything else only
/// reads the images.
class organized_points
{
public:
    /// The key to the images' writing for a coarser level of a frame's pyramid. Only
    /// frame_pyramid makes one: its constructor is private, and the pyramid is its one friend, so
    /// that the pyramid reaches no private member of the organized points themselves.
    class pyramid_key
    {
        friend class frame_pyramid;

        /// Makes a key; for the frame's pyramid alone.
        pyramid_key() noexcept = default;
    };

    /// Empty organized points: no images; deproject() into them is refused.
    organized_points() noexcept = default;

    /// Requests the images from the pool: the points and the marks of a size.
    /// @param pool [in,out] the pool; open, else the request is refused as closed.
    /// @param size [in] the size of the frames they will be made from, in px, each > 0.
    /// @return granted, with organized points of that size whose contents are not set until the
    ///         first deproject(); or the first of the two requests that was refused, with its
    ///         status and message (gpu_pool::request()).
    /// @note Allocates device memory (cudaMalloc, twice: 12 and 1 bytes per pixel), so it is
    ///       called before the pool is closed. Whether cudaMalloc waits for the GPU is not stated
    ///       by CUDA [?]; nothing else waits. Allocates for the message when refused; a failed
    ///       allocation ends the program (std::terminate): the function is noexcept. Two threads
    ///       may not request from one pool at once (gpu_pool::request()).
    static request_result<organized_points> request(gpu_pool &pool,
                                                    const frame_size &size) noexcept;

    /// Checks a depth range, as deproject() does before it queues anything, so that a step that
    /// takes the same range (fusion/sparse_tsdf.hpp, sparse_tsdf::ray_cast()) refuses it by the
    /// same rule and message.
    /// @param parameters [in] the range.
    /// @return empty when it is usable: both limits finite, 0 <= near_m < far_m; else the reason,
    ///         the message deproject() refuses it with.
    /// @note Allocates for the reason; throws std::bad_alloc when that fails. Waits for nothing.
    static std::string range_refusal(const organized_points_parameters &parameters);

    /// Organized points are not copied: a copy allocates.
    organized_points(const organized_points &) = delete;
    organized_points &operator=(const organized_points &) = delete;

    /// Takes over other organized points' images without allocating.
    /// @param other [in,out] the organized points moved from; left empty.
    /// @note Allocates nothing and waits for nothing.
    organized_points(organized_points &&other) noexcept = default;

    /// Takes over other organized points' images; this one's are freed first.
    /// @param other [in,out] the organized points moved from; left empty. Moving onto itself
    ///        changes nothing.
    /// @return these organized points.
    /// @pre No work queued on the GPU uses this one's images any more (gpu_image's move
    ///      assignment says what happens when some does).
    /// @note Frees with cudaFree, which may wait for the GPU (gpu/gpu_image.hpp). Allocates
    ///       nothing.
    organized_points &operator=(organized_points &&other) noexcept = default;

    /// Frees the images.
    /// @pre No work queued on the GPU uses them any more (as for the move assignment).
    /// @note Frees with cudaFree, which may wait for the GPU (gpu/gpu_image.hpp).
    ~organized_points() = default;

    /// Deprojects a frame's depth into the points and marks each pixel valid or not: valid
    /// exactly when its depth value d is not 0 and near_m <= z <= far_m, z = d * depth unit
    /// rounded to float, the point's own z.
    /// @param frame [in] the GPU frame, of these points' size, with a frame uploaded into it on
    ///        stream or on a stream the caller has ordered before it; its description gives the
    ///        intrinsics (in px) and the depth unit (in m). Owned by the caller; it must hold
    ///        the frame until the work queued here has run.
    /// @param parameters [in] the depth range.
    /// @param stream [in] the CUDA stream of the current device the work is queued on; owned by
    ///        the caller. Work queued on it later sees the points and the marks.
    /// @return done when the work is queued; not done with the reason when refused before it:
    ///         these organized points are empty; near_m or far_m is not finite, near_m < 0 or
    ///         far_m <= near_m; the GPU frame holds no frame (nothing uploaded, or empty); its
    ///         size is not these points'; its fx or fy is not a finite number > 0, its cx or cy
    ///         not finite, its depth unit not a finite number > 0. When refused, nothing is
    ///         queued and the images are left as they were. Not done with CUDA's error, which is
    ///         then cleared (gpu/gpu_messages.hpp): when CUDA's last error is set before the
    ///         launch, by an earlier call (gpu_image's end leaves cudaFree's there), with nothing
    ///         queued and the images left as they were; when CUDA refuses the launch, for one
    ///         when a frame's rows need more than 65535 blocks of 8 rows [S: CUDA C++ Programming
    ///         Guide, "Compute Capabilities", the grid's y dimension], with the images' contents
    ///         then unspecified.
    /// @note One kernel on stream, which reads the depth picture and writes both images. Waits
    ///       for nothing: cudaPeekAtLastError before the launch and cudaGetLastError after it
    ///       ask only whether an error is set and whether CUDA accepted the launch. Allocates
    ///       nothing when done; allocates for the message when not, and a failed allocation ends
    ///       the program (std::terminate): the function is noexcept. Two threads may call it at
    ///       once only for different organized points.
    step_result deproject(const gpu_frame &frame, const organized_points_parameters &parameters,
                          cudaStream_t stream) noexcept;

    /// Gives the size the organized points were requested with.
    /// @return their width and height, in px; 0 x 0 when empty.
    frame_size size() const noexcept;

    /// Gives the points for the GPU to read.
    /// @return the image of the points, in the camera frame, in m, of the size(); (0, 0, 0) at an
    ///         invalid pixel; empty when the organized points are. Valid as long as they hold it.
    const gpu_image<float3> &points() const noexcept;

    /// Gives the validity mask for the GPU to read.
    /// @return the image of the marks, 1 for a valid pixel and 0 for an invalid one, of the
    ///         size(); empty when the organized points are. Valid as long as they hold it.
    const gpu_image<std::uint8_t> &validity_mask() const noexcept;

    /// Gives the points for the GPU to write: the frame's pyramid's combining of a coarser level.
    /// @param key [in] the key only the frame's pyramid makes.
    /// @return the image of the points, of the size(); empty when the organized points are. Valid
    ///         as long as they hold it.
    gpu_image<float3> &points_to_write(pyramid_key key) noexcept;

    /// Gives the validity mask for the GPU to write: the frame's pyramid's combining of a coarser
    /// level.
    /// @param key [in] the key only the frame's pyramid makes.
    /// @return the image of the marks, of the size(); empty when the organized points are. Valid
    ///         as long as they hold it.
    gpu_image<std::uint8_t> &validity_mask_to_write(pyramid_key key) noexcept;

private:
    gpu_image<float3> m_points;
    gpu_image<std::uint8_t> m_validity_mask;
};
} // namespace gca
