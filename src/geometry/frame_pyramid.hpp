// SPDX-License-Identifier: 0BSD
/// @file frame_pyramid.hpp
/// The frame's pyramid, the per-frame path's step after the upload as one call: a frame on the
/// GPU becomes three levels, each with its organized points and validity mask
/// (geometry/organized_points.hpp), its normals (geometry/organized_normals.hpp), its intensity,
/// the Rec. 709 luma of the frame's color, and its intrinsics. Level 0 is the frame's size; each
/// coarser level combines the level before at half its rows and columns, never across an invalid
/// pixel or a depth edge. The method, the rules and their sources stand in
/// docs/methods/frame-pyramid.md. Every image comes from the pool (gpu/gpu_pool.hpp) before it
/// closes; the step itself allocates nothing, waits for nothing and reads nothing back.

#pragma once

#include "geometry/organized_normals.hpp"
#include "geometry/organized_points.hpp"
#include "gpu/gpu_frame.hpp"
#include "gpu/gpu_image.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/step_result.hpp"
#include "types/camera_param.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>

namespace gca
{
/// The step's parameters (docs/methods/frame-pyramid.md, "Input"), the same at every level.
struct frame_pyramid_parameters
{
    organized_points_parameters range; ///< level 0's depth range (organized_points::deproject())
    /// Every level's normals (organized_normals::estimate()). Their steepest incidence θ also sets
    /// which points a coarser level combines: two adjacent points of level l are combined when
    /// they lie on one surface by the steepest incidence θ_l, tan θ_l = tan θ / 2^l.
    organized_normals_parameters normals;
};

/// A frame's pyramid on the GPU: level_count levels, level 0 of the size it was requested with,
/// each next level of half the rows and columns, rounded up. Level l holds, for each of its
/// pixels at index v * width + u, a point in the camera frame (x right, y down, z forward, in m)
/// and a validity mark, a normal or (0, 0, 0), and an intensity in [0, 1], 0 at an invalid pixel;
/// and its intrinsics, the frame's divided by 2^l (docs/methods/frame-pyramid.md). Move-only.
/// Not synchronized: one thread at a time uses one.
class frame_pyramid
{
public:
    /// The number of levels, 3 [?: initial; docs/methods/frame-pyramid.md, "Input"].
    static constexpr std::size_t level_count = 3;

    /// Gives the size of a level of the pyramid of frames of a size: level 0 the frame's, each next
    /// level half the rows and columns of the one before, rounded up, so that every pixel of the
    /// next level has its centre pixel, twice its index, in the level before
    /// (docs/methods/frame-pyramid.md, "Output").
    /// @param size [in] level 0's width and height, in px.
    /// @param level [in] the level, any value: level_count and above go on halving.
    /// @return the level's width and height, in px; 0 stays 0.
    /// @note Host only; allocates nothing.
    static frame_size level_size(const frame_size &size, std::size_t level) noexcept;

    /// Gives the intrinsics of a level of the pyramid of frames with given intrinsics: each of fx,
    /// fy, cx, cy divided by 2^level, exact in float (docs/methods/frame-pyramid.md, "The
    /// intrinsics of a level"), the one copy the pyramid's build() and the sparse TSDF's ray cast
    /// (fusion/sparse_tsdf.hpp) take a level's intrinsics from.
    /// @param k [in] level 0's intrinsics, in px.
    /// @param level [in] the level, any value below 127, so that 2^-level is a float.
    /// @return the level's intrinsics, in px of the level, pixel centres at whole numbers.
    /// @note Host only; allocates nothing.
    static gca::intrinsics intrinsics_at_level(const gca::intrinsics &k,
                                               std::size_t level) noexcept;

    /// Gives the intensity of a color: its Rec. 709 luma, scaled to [0, 1]
    /// (docs/methods/frame-pyramid.md, "The intensity"), in float: three products and two sums,
    /// then the division by 255. The one copy that the pyramid's level 0 and the sparse TSDF's ray
    /// cast (fusion/sparse_tsdf.hpp) take an intensity from.
    /// @param red [in] the red channel, 0 to 255; a ray cast's interpolated channel need not be a
    ///        whole number.
    /// @param green [in] the green channel, 0 to 255.
    /// @param blue [in] the blue channel, 0 to 255.
    /// @return (0.2126 red + 0.7152 green + 0.0722 blue) / 255, in [0, 1] up to float's rounding
    ///         [S: ITU-R BT.709-6, the luminance signal's coefficients].
    /// @note Host and device code; allocates nothing.
    __host__ __device__ static float intensity_of(float red, float green, float blue) noexcept
    {
        return (0.2126f * red + 0.7152f * green + 0.0722f * blue) / 255.0f;
    }

    /// An empty pyramid: no levels; build() into it is refused.
    frame_pyramid() noexcept = default;

    /// Requests every level's images from the pool: at each level the organized points, the
    /// organized normals and the intensity.
    /// @param pool [in,out] the pool; open, else the request is refused as closed.
    /// @param size [in] the size of the frames it will be built from, level 0's, in px, each > 0.
    /// @return granted, with a pyramid whose contents and intrinsics are not set until the first
    ///         build(); or the first of its requests that was refused, with its status and message
    ///         (gpu_pool::request()).
    /// @note Allocates device memory (cudaMalloc, four times per level: 29 bytes per pixel of
    ///       the level [S: 12 for a point, 1 for a mark, 12 for a normal, 4 for an intensity]), so
    ///       it is called before the pool is closed. Whether cudaMalloc waits for
    ///       the GPU is not stated by CUDA [?]; nothing else waits. Allocates for the message when
    ///       refused; a failed allocation ends the program (std::terminate): the function is
    ///       noexcept. Two threads may not request from one pool at once (gpu_pool::request()).
    static request_result<frame_pyramid> request(gpu_pool &pool, const frame_size &size) noexcept;

    /// A pyramid is not copied: a copy allocates.
    frame_pyramid(const frame_pyramid &) = delete;
    frame_pyramid &operator=(const frame_pyramid &) = delete;

    /// Takes over another pyramid's levels without allocating.
    /// @param other [in,out] the pyramid moved from; left empty.
    /// @note Allocates nothing and waits for nothing.
    frame_pyramid(frame_pyramid &&other) noexcept = default;

    /// Takes over another pyramid's levels; this one's are freed first.
    /// @param other [in,out] the pyramid moved from; left empty. Moving onto itself changes
    ///        nothing.
    /// @return this pyramid.
    /// @pre No work queued on the GPU uses this one's images any more (gpu_image's move
    ///      assignment says what happens when some does).
    /// @note Frees with cudaFree, which may wait for the GPU (gpu/gpu_image.hpp). Allocates
    ///       nothing.
    frame_pyramid &operator=(frame_pyramid &&other) noexcept = default;

    /// Frees the levels' images.
    /// @pre No work queued on the GPU uses them any more (as for the move assignment).
    /// @note Frees with cudaFree, which may wait for the GPU (gpu/gpu_image.hpp).
    ~frame_pyramid() = default;

    /// Builds every level from a frame: level 0's organized points from its depth, level 0's
    /// intensity from its color, level 0's normals; then for each coarser level its points, marks
    /// and intensity combined from the level before, and its normals.
    /// @param frame [in] the GPU frame, of level 0's size, with a frame uploaded into it on stream
    ///        or on a stream the caller has ordered before it; its description gives the
    ///        intrinsics (in px) and the depth unit (in m). Owned by the caller; it must hold the
    ///        frame until the work queued here has run.
    /// @param parameters [in] the depth range and the normals' parameters.
    /// @param stream [in] the CUDA stream of the current device the work is queued on; owned by
    ///        the caller. Work queued on it later sees every level.
    /// @return done when the work is queued, and the levels' intrinsics are the frame's divided
    ///         by 2^l; not done with the reason when refused before anything is queued, with every
    ///         level and its intrinsics left as they were: the pyramid is empty; the normals'
    ///         parameters cannot be used (organized_normals::parameters_refusal()); the depth
    ///         range or the frame
    ///         cannot be used or the frame's size is not level 0's, as
    ///         organized_points::deproject() refuses them; CUDA's last error is set before the
    ///         first launch, by an earlier call, with CUDA's error, which is then cleared
    ///         (gpu/gpu_messages.hpp). Not done with CUDA's error, cleared, when CUDA refuses a
    ///         later launch, with the levels' contents then unspecified.
    /// @note Seven kernels on stream, each through the stages' one launch
    ///       (gpu/kernel_launch.cuh). Waits for nothing: cudaPeekAtLastError before a launch and
    ///       cudaGetLastError after it ask only whether an error is set and whether CUDA accepted
    ///       the launch. Allocates nothing when done; allocates for the message when not, and a
    ///       failed allocation ends the program (std::terminate): the function is noexcept. Two
    ///       threads may call it at once only for different pyramids.
    step_result build(const gpu_frame &frame, const frame_pyramid_parameters &parameters,
                      cudaStream_t stream) noexcept;

    /// Gives a level's size.
    /// @param level [in] the level.
    /// @return its width and height, in px; 0 x 0 when the pyramid is empty.
    /// @pre level < level_count; another level ends the program (std::terminate): the function is
    ///      noexcept. The same holds for every accessor of a level below.
    frame_size size(std::size_t level) const noexcept;

    /// Gives a level's organized points and validity mask for the GPU to read.
    /// @param level [in] the level.
    /// @return the level's organized points, in the camera frame, in m: at level 0 the frame's,
    ///         at a coarser level the combined ones; empty when the pyramid is. Valid as long as
    ///         the pyramid holds them.
    /// @pre level < level_count, as for size().
    const organized_points &points(std::size_t level) const noexcept;

    /// Gives a level's normals for the GPU to read.
    /// @param level [in] the level.
    /// @return the level's organized normals, estimated from its points; empty when the pyramid
    ///         is. Valid as long as the pyramid holds them.
    /// @pre level < level_count, as for size().
    const organized_normals &normals(std::size_t level) const noexcept;

    /// Gives a level's intensity for the GPU to read.
    /// @param level [in] the level.
    /// @return the image of the intensity, the Rec. 709 luma in [0, 1] at a valid pixel and 0 at
    ///         an invalid one, of the level's size; empty when the pyramid is. Valid as long as the
    ///         pyramid holds it.
    /// @pre level < level_count, as for size().
    const gpu_image<float> &intensity(std::size_t level) const noexcept;

    /// Gives a level's intrinsics: those of the frame built last, each divided by 2^level.
    /// @param level [in] the level.
    /// @return fx, fy, cx, cy in px of the level, pixel centres at whole numbers; zero before the
    ///         first build().
    /// @pre level < level_count, as for size().
    gca::intrinsics level_intrinsics(std::size_t level) const noexcept;

private:
    /// One level: its images and its intrinsics.
    struct level_images
    {
        organized_points points;      ///< the points and marks
        organized_normals normals;    ///< the normals
        gpu_image<float> intensity;   ///< the intensity
        gca::intrinsics intrinsics{}; ///< the intrinsics, in px; zero before the first build()
    };

    /// Combines a level from the level before: its points, marks and intensity
    /// (docs/methods/frame-pyramid.md, "A coarser level"). The one code that writes organized
    /// points other than organized_points::deproject(), through the key only the pyramid makes
    /// (organized_points::pyramid_key).
    /// @param level [in] the level written, 1 to level_count - 1.
    /// @param sin2_max_incidence [in] the square of the sine of the level before's steepest
    ///        incidence θ_{level - 1}, in (0, 1): below 1 by the normals' largest steepest
    ///        incidence (max_steepest_incidence_deg).
    /// @param stream [in] the CUDA stream the kernel is queued on; owned by the caller.
    /// @return done when the kernel is queued; not done with CUDA's error, which is cleared.
    /// @note One kernel on stream through gpu/kernel_launch.cuh; waits for nothing, allocates
    ///       nothing when done.
    step_result combine(std::size_t level, float sin2_max_incidence, cudaStream_t stream) noexcept;

    std::array<level_images, level_count> m_levels;
};
} // namespace gca
