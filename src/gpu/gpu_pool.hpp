// SPDX-License-Identifier: 0BSD
/// @file gpu_pool.hpp
/// The pool that closes: the gate the per-frame path's buffers pass through (CODING_STANDARDS.md,
/// "The rewrite's code": the path's buffers exist before a scan starts, and it allocates
/// nothing). While it is open, a request is granted a GPU image (gpu/gpu_image.hpp) of its own
/// device memory, or a pinned host buffer (gpu/pinned_buffer.hpp) of its own page-locked memory;
/// every buffer of the path is requested before a scan's first frame, and the pool is closed
/// then. After that every request is refused as a value and counted, so a stage that asks for
/// memory during a scan fails where its caller sees it, and a test sees the count. The pool holds
/// no memory itself: what it granted is the sum of its requests, and each buffer frees its
/// memory at its own end, before or after the pool's. It is not the pool of the sparse TSDF's
/// blocks (docs/adr/0004-sparse-tsdf-model.md).

#pragma once

#include "gpu/gpu_image.hpp"
#include "gpu/pinned_buffer.hpp"
#include "types/frame_description.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace gca
{
/// Whether a request to the pool was granted, and why not when it was not.
enum class request_status
{
    granted, ///< the buffer holds memory of the size asked for
    closed,  ///< the pool is closed: no memory is granted any more
    failure, ///< the size is zero or too large, or CUDA could not allocate; the message says which
};

/// What a request gave: a buffer, or why not.
template <typename Buffer> struct request_result
{
    request_status status = request_status::failure; ///< granted, or why not
    std::string message; ///< why nothing was granted, in plain English; empty when granted
    Buffer granted;      ///< the granted buffer; empty unless granted
};

/// Carries a refused request's status and message over into the result of something made of
/// several requests, such as a GPU frame (gpu/gpu_frame.hpp).
/// @param refused [in] the refused request.
/// @return its status and message, with nothing granted.
/// @note Allocates for the message; a failed allocation ends the program (std::terminate): the
///       function is noexcept. Waits for nothing.
template <typename Made, typename Buffer>
request_result<Made> refused_as(const request_result<Buffer> &refused) noexcept
{
    request_result<Made> result;
    result.status = refused.status;
    result.message = refused.message;
    return result;
}

/// The gate of the per-frame path's buffers: open when built, closed once by close(), never
/// opened again; a scan builds a pool of its own. Not copied and not moved, so that a stage that
/// keeps a reference to the pool asks the one the scan closed. Not synchronized: one thread at a
/// time requests, closes or reads the count.
class gpu_pool
{
public:
    /// Builds an open pool, which has granted nothing.
    gpu_pool() = default;

    /// A pool is not copied or moved: it stands for one scan's gate.
    gpu_pool(const gpu_pool &) = delete;
    gpu_pool &operator=(const gpu_pool &) = delete;

    /// Ends the pool; the buffers it granted keep their memory.
    ~gpu_pool() = default;

    /// Requests a GPU image of a size.
    /// @param size [in] the image's width and height, in px, each > 0.
    /// @return granted, with an image of that size whose contents are not cleared; closed, with
    ///         an empty image, when the pool is closed, whatever the size; failure, with an empty
    ///         image and the reason, when width or height is 0, when the image's bytes exceed
    ///         what std::size_t holds, or when CUDA cannot allocate them.
    /// @note Allocates device memory with cudaMalloc when granted, and nothing when closed;
    ///       whether cudaMalloc waits for the GPU's work is not stated by CUDA [?], which is one
    ///       reason it is called only before a scan's first frame. Not stream-ordered. When CUDA
    ///       fails, its last error is cleared (gpu/gpu_messages.hpp). Allocates for the message
    ///       when it refuses; a failed allocation ends the program (std::terminate): the
    ///       function is noexcept. Two threads may not request at once.
    template <typename Pixel>
    request_result<gpu_image<Pixel>> request(const frame_size &size) noexcept
    {
        request_result<gpu_image<Pixel>> result;
        void *data = nullptr;
        result.status = allocate_image(size, sizeof(Pixel), data, result.message);
        if (result.status == request_status::granted)
        {
            // cudaMalloc's memory is "suitably aligned for any kind of variable" [S:
            // cuda_runtime_api.h, cudaMalloc, CUDA 12.6], so it holds Pixels.
            result.granted = gpu_image<Pixel>(static_cast<Pixel *>(data), size);
        }
        return result;
    }

    /// Requests a pinned host buffer.
    /// @param bytes [in] its size, in bytes, > 0.
    /// @return granted, with a buffer of that size whose contents are not cleared; closed, with
    ///         an empty buffer, when the pool is closed, whatever the size; failure, with an
    ///         empty buffer and the reason, when bytes is 0 or CUDA cannot allocate them.
    /// @note Allocates page-locked host memory with cudaMallocHost when granted, and nothing when
    ///       closed; whether cudaMallocHost waits for the GPU's work is not stated by CUDA [?].
    ///       When CUDA fails, its last error is cleared. Allocates for the message when it
    ///       refuses; a failed allocation ends the program (std::terminate): the function is
    ///       noexcept. Two threads may not request at once.
    request_result<pinned_buffer> request_pinned(std::size_t bytes) noexcept;

    /// Closes the pool: every later request is refused with the status closed. Closing a closed
    /// pool changes nothing.
    /// @note Allocates nothing and waits for nothing. Not while another thread requests.
    void close() noexcept;

    /// Says whether the pool is closed.
    /// @return true after close(), false before it.
    bool closed() const noexcept;

    /// Counts the requests refused because the pool was closed, so that a test of a scan sees a
    /// stage that asked for memory during it.
    /// @return the requests made after close(), each counted once; 0 while the pool is open.
    std::uint64_t refused_while_closed() const noexcept;

private:
    /// Counts a request made to the closed pool and writes its refusal.
    /// @param what [in] what was requested, for the message: "an image of 640x400 px".
    /// @return the refusal's message.
    /// @pre The pool is closed; the request is counted all the same when it is not.
    std::string refuse_closed(const std::string &what) noexcept;

    /// Allocates the device memory of an image request, unless the pool is closed or the size is
    /// not one to allocate.
    /// @param size [in] the image's width and height, in px.
    /// @param pixel_bytes [in] the size of one pixel, in bytes, > 0.
    /// @param data [out] the allocated device memory when granted, owned by the caller from then
    ///        on; nullptr otherwise.
    /// @param message [out] why nothing was granted; left empty when granted.
    /// @return granted, closed or failure, as request() says.
    /// @note Allocates with cudaMalloc as request() says.
    request_status allocate_image(const frame_size &size, std::size_t pixel_bytes, void *&data,
                                  std::string &message) noexcept;

    bool m_closed = false;
    std::uint64_t m_refused_while_closed = 0;
};
} // namespace gca
