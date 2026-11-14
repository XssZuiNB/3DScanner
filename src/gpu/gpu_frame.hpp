// SPDX-License-Identifier: 0BSD
/// @file gpu_frame.hpp
/// A frame on the GPU (GLOSSARY.md, "Frame"): the color and the depth picture of a frame
/// (camera/frame.hpp) as GPU images (gpu/gpu_image.hpp), with the frame's index, timestamp and
/// description on the host. The upload, the per-frame path's first step, copies a frame into it;
/// the download copies it back into a frame, for the tests. All its memory comes from the pool
/// (gpu/gpu_pool.hpp) before the pool is closed.
///
/// The upload stages through a pinned host buffer of the GPU frame's own: it copies the frame
/// into the buffer on the host, then queues the copies to the GPU, which do not make the host
/// wait. A copy straight from the frame's pageable memory may wait for the stream's earlier work
/// [S: cuda_runtime_api.h, "API synchronization behavior", CUDA 12.6], and does on this board.
/// Behind about 22 ms of queued work at 1280x800, the two copies from pageable memory block the
/// host for a median 22.89 ms, this upload for 0.91 ms, its copy into the staging buffer; at
/// 640x400 neither waits for the queued work, 0.25 and 0.26 ms. On an idle stream staging costs
/// time: from the call to the copies' end 1.52 ms against 0.60 ms at 1280x800, 0.34 against
/// 0.22 ms at 640x400 [M 2026-11-14, MAXN_SUPER, nvhost_podgov, 10 s per case]. The frame's own
/// buffers stay plain host memory: the camera module links no CUDA.
/// One staging buffer per GPU frame: before the upload writes into it, it waits until the copies
/// of the last upload out of it have run. A caller that waits for a frame's work before it
/// uploads the next, as reading back a frame's tracking result does, finds that wait returning at
/// once; a second staging buffer would help only a caller that uploads before that, and is not
/// built without a measured benefit (AGENTS.md §7 rule 6).

#pragma once

#include "camera/frame.hpp"
#include "gpu/gpu_image.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/pinned_buffer.hpp"
#include "gpu/step_result.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace gca
{
/// A frame's pictures on the GPU, its index, timestamp and description, and the pinned host
/// buffer its upload stages through. Its images keep the size it was requested with, and its
/// description is the one of the frame uploaded last, of that size. Move-only. Not
/// synchronized: one thread at a time uses a GPU frame.
class gpu_frame
{
public:
    /// An empty GPU frame: no images, no staging buffer; an upload into it is refused.
    gpu_frame() noexcept = default;

    /// Requests a GPU frame's memory from the pool: the color and the depth image of a size, and
    /// the staging buffer of their bytes; and creates the event its uploads record.
    /// @param pool [in,out] the pool; open, else the request is refused as closed.
    /// @param size [in] the size of the frames it will hold, in px, each > 0.
    /// @return granted, with a GPU frame of that size that holds no frame yet; or the first of
    ///         its requests that was refused, with its status and message (gpu_pool::request()),
    ///         or failure when CUDA could not create the event, with CUDA's error.
    /// @note Allocates device memory (cudaMalloc, twice), page-locked host memory
    ///       (cudaMallocHost) and an event (cudaEventCreateWithFlags), so it is called before the
    ///       pool is closed. Whether CUDA's allocators wait for the GPU is not stated by CUDA [?];
    ///       nothing else waits. When CUDA fails, its last error is cleared.
    ///       Allocates for the message when refused; a failed allocation ends the program
    ///       (std::terminate): the function is noexcept.
    static request_result<gpu_frame> request(gpu_pool &pool, const frame_size &size) noexcept;

    /// A GPU frame is not copied: a copy allocates.
    gpu_frame(const gpu_frame &) = delete;
    gpu_frame &operator=(const gpu_frame &) = delete;

    /// Takes over another GPU frame's memory, event and values without allocating.
    /// @param other [in,out] the GPU frame moved from; left empty.
    /// @note Allocates nothing and waits for nothing.
    gpu_frame(gpu_frame &&other) noexcept;

    /// Takes over another GPU frame's memory, event and values; this one's are released first.
    /// @param other [in,out] the GPU frame moved from; left empty. Moving a GPU frame onto itself
    ///        changes nothing.
    /// @return this GPU frame.
    /// @pre No work queued on the GPU uses this GPU frame's images any more (gpu_image's move
    ///      assignment says what happens when some does).
    /// @note Waits until this GPU frame's last upload has copied out of its staging buffer
    ///       (cudaEventSynchronize), so that the buffer is not freed under a copy; frees with
    ///       cudaFree and cudaFreeHost, which may wait as gpu_image and pinned_buffer say.
    ///       Allocates nothing.
    gpu_frame &operator=(gpu_frame &&other) noexcept;

    /// Releases the GPU frame's memory and event.
    /// @pre No work queued on the GPU uses its images any more (as for the move assignment).
    /// @note Waits until its last upload has copied out of its staging buffer, as the move
    ///       assignment does, then frees as it does.
    ~gpu_frame();

    /// Uploads a frame: copies its pictures into the staging buffer, queues their copies into
    /// the images on a stream, and copies its index, timestamp and description.
    /// @param host [in] the frame, in host memory, of the GPU frame's size; owned by the caller.
    ///        It is read only during the call: when the call returns, the frame may be changed or
    ///        reused at once.
    /// @param stream [in] the CUDA stream of the current device the copies are queued on; owned
    ///        by the caller. Work queued on it later sees the frame's pictures in the images.
    /// @return done; or not done with the reason: the GPU frame is empty, the frame's size is
    ///         not the GPU frame's, the frame's buffers do not hold its size (a frame moved
    ///         from), or CUDA failed, with CUDA's error. When refused before a copy, the GPU
    ///         frame is left as it was; when CUDA fails, its images' contents are unspecified
    ///         and its index, timestamp and description are left as they were.
    /// @note One written wait: cudaEventSynchronize on the event the last upload recorded after
    ///       its copies, which waits until those copies out of the staging buffer have run on
    ///       their stream (and for nothing queued after them); before the first upload it
    ///       returns at once [S: cuda_runtime_api.h, cudaEventRecord, CUDA 12.6]. Then a host
    ///       copy of 5 bytes per pixel into the staging buffer, two cudaMemcpyAsync from it,
    ///       host to device, on stream, and cudaEventRecord on stream. No other wait: the copies
    ///       from pinned memory "should be fully asynchronous" [S: as above, "API
    ///       synchronization behavior"]. Allocates nothing when done; allocates for the message
    ///       when not, and a failed allocation ends the program (std::terminate): the function
    ///       is noexcept. When CUDA fails, its last error is cleared.
    step_result upload(const frame &host, cudaStream_t stream) noexcept;

    /// Downloads the GPU frame into a frame, for the tests: its description, index, timestamp
    /// and both pictures, and waits until they are in host memory.
    /// @param host [out] the frame, prepared for the GPU frame's description, index and timestamp
    ///        (frame::prepare()) and filled with its pictures when done; when not done, prepared
    ///        but with its pictures unspecified, or left as it was when refused before the
    ///        preparation. Owned by the caller.
    /// @param stream [in] the CUDA stream of the current device the copies are queued on and
    ///        waited for; owned by the caller. Its earlier work, an upload included, runs first.
    /// @return done; or not done with the reason: the GPU frame is empty or holds no frame yet,
    ///         or CUDA failed, with CUDA's error.
    /// @note One written wait: cudaStreamSynchronize after two cudaMemcpyAsync, device to host,
    ///       on stream, which waits for all of the stream's work, done or not, so that no copy
    ///       writes into host's buffers once the call has returned. The copies go into the
    ///       frame's pageable memory and may make the host wait before that [S: as for upload()].
    ///       Allocates host memory when host's buffers grow (frame::prepare()) and for the
    ///       message when not done; a failed allocation ends the program (std::terminate): the
    ///       function is noexcept. When CUDA fails, its last error is cleared. Two threads may
    ///       call it at once only for different GPU frames and different frames. Not for the
    ///       per-frame path: it reads back whole pictures and waits.
    step_result download(frame &host, cudaStream_t stream) const noexcept;

    /// Gives the description of the frame uploaded last.
    /// @return its size, intrinsics and depth unit; 0 x 0, zero intrinsics and depth unit before
    ///         the first upload. Valid until the next upload or move.
    const frame_description &description() const noexcept;

    /// Gives the index of the frame uploaded last.
    /// @return its index among its source's frames; 0 before the first upload.
    std::uint64_t index() const noexcept;

    /// Gives the timestamp of the frame uploaded last.
    /// @return its timestamp, in µs on its source's clock; 0 before the first upload.
    std::uint64_t timestamp_us() const noexcept;

    /// Gives the color picture for the GPU to read.
    /// @return the image, RGB8 (x red, y green, z blue) in the frame's order, of the GPU frame's
    ///         size; empty for an empty GPU frame. Valid as long as the GPU frame holds it.
    const gpu_image<uchar3> &color() const noexcept;

    /// Gives the depth picture for the GPU to read.
    /// @return the image, in depth units in the frame's order, 0 for no depth, of the GPU
    ///         frame's size; empty for an empty GPU frame. Valid as long as the GPU frame holds
    ///         it.
    const gpu_image<std::uint16_t> &depth() const noexcept;

private:
    /// Waits until the last upload has copied out of the staging buffer, then destroys the
    /// event.
    /// @note The written wait of the move assignment and the destructor: cudaEventSynchronize.
    ///       A failure of either call is not reported; its error is left as CUDA's last error.
    void release_event() noexcept;

    frame_description m_description{};
    std::uint64_t m_index = 0;
    std::uint64_t m_timestamp_us = 0;
    gpu_image<uchar3> m_color;
    gpu_image<std::uint16_t> m_depth;
    pinned_buffer m_staging;                ///< depth's 2 bytes per pixel, then color's 3
    cudaEvent_t m_staging_copied = nullptr; ///< recorded after each upload's copies
};
} // namespace gca
