// SPDX-License-Identifier: 0BSD
/// @file gpu_pool.cpp
/// The pool that closes: its gate, its count of refused requests and the allocation of what it
/// grants (see gpu_pool.hpp).

#include "gpu/gpu_pool.hpp"

#include "gpu/gpu_messages.hpp"

#include <cuda_runtime_api.h>

#include <limits>
#include <span>
#include <string>

namespace gca
{
request_result<pinned_buffer> gpu_pool::request_pinned(std::size_t bytes) noexcept
{
    request_result<pinned_buffer> result;
    // Written only for a refusal, so that a granted request allocates no string.
    const auto what = [bytes] {
        return "a pinned host buffer of " + std::to_string(bytes) + " bytes";
    };
    // Before every other check, so that a closed pool calls no allocator at all.
    if (m_closed)
    {
        result.status = request_status::closed;
        result.message = refuse_closed(what());
        return result;
    }
    if (bytes == 0)
    {
        result.message = what() + " holds nothing";
        return result;
    }
    void *data = nullptr;
    const cudaError_t error = cudaMallocHost(&data, bytes);
    if (error != cudaSuccess)
    {
        result.message = cuda_failure(("CUDA could not allocate " + what()).c_str(), error);
        return result;
    }
    result.status = request_status::granted;
    result.granted = pinned_buffer(std::span<std::byte>(static_cast<std::byte *>(data), bytes));
    return result;
}

void gpu_pool::close() noexcept
{
    m_closed = true;
}

bool gpu_pool::closed() const noexcept
{
    return m_closed;
}

std::uint64_t gpu_pool::refused_while_closed() const noexcept
{
    return m_refused_while_closed;
}

std::string gpu_pool::refuse_closed(const std::string &what) noexcept
{
    ++m_refused_while_closed;
    return "the GPU pool is closed: " + what +
           " was requested after its closing, and the per-frame path allocates nothing";
}

request_status gpu_pool::allocate_image(const frame_size &size, std::size_t pixel_bytes,
                                        void *&data, std::string &message) noexcept
{
    data = nullptr;
    // Before every other check, so that a closed pool calls no allocator at all.
    if (m_closed)
    {
        message = refuse_closed("an image of " + size_text(size));
        return request_status::closed;
    }
    if (size.width == 0 || size.height == 0)
    {
        message = "an image of " + size_text(size) + " holds no pixel";
        return request_status::failure;
    }
    // width * height is at most (2^32 - 1)^2 < 2^64 [S], so it fits; the bytes may not.
    static_assert(sizeof(std::size_t) >= 8, "the pixel count of two 32-bit factors fits");
    const std::size_t pixels = static_cast<std::size_t>(size.width) * size.height;
    if (pixels > std::numeric_limits<std::size_t>::max() / pixel_bytes)
    {
        message = "an image of " + size_text(size) + " of " + std::to_string(pixel_bytes) +
                  " bytes each has more bytes than std::size_t holds";
        return request_status::failure;
    }
    const cudaError_t error = cudaMalloc(&data, pixels * pixel_bytes);
    if (error != cudaSuccess)
    {
        data = nullptr;
        message =
            cuda_failure(("CUDA could not allocate an image of " + size_text(size)).c_str(), error);
        return request_status::failure;
    }
    return request_status::granted;
}
} // namespace gca
