// SPDX-License-Identifier: 0BSD
/// @file gpu_frame.cpp
/// A frame on the GPU: its request from the pool, its moves and end, and the checks, staging and
/// copies of an upload and a download (see gpu_frame.hpp).

#include "gpu/gpu_frame.hpp"

#include "gpu/gpu_messages.hpp"

#include <cstddef>
#include <cstring>
#include <string>
#include <utility>

namespace gca
{
namespace
{
// A frame's color is 3 bytes per pixel (camera/frame.hpp), copied byte by byte into uchar3.
static_assert(sizeof(uchar3) == 3, "a color pixel is 3 bytes, as in a frame");

/// Copies a request's refusal into the result of another type.
/// @param refused [in] the refused request.
/// @return its status and message, with an empty GPU frame.
template <typename Buffer>
request_result<gpu_frame> refused_as(const request_result<Buffer> &refused) noexcept
{
    request_result<gpu_frame> result;
    result.status = refused.status;
    result.message = refused.message;
    return result;
}
} // namespace

request_result<gpu_frame> gpu_frame::request(gpu_pool &pool, const frame_size &size) noexcept
{
    request_result<gpu_image<uchar3>> color = pool.request<uchar3>(size);
    if (color.status != request_status::granted)
        return refused_as(color);
    request_result<gpu_image<std::uint16_t>> depth = pool.request<std::uint16_t>(size);
    if (depth.status != request_status::granted)
        return refused_as(depth);
    request_result<pinned_buffer> staging =
        pool.request_pinned(color.granted.size_bytes() + depth.granted.size_bytes());
    if (staging.status != request_status::granted)
        return refused_as(staging);
    cudaEvent_t event = nullptr;
    const cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    request_result<gpu_frame> result;
    if (error != cudaSuccess)
    {
        result.message = cuda_failure("CUDA could not create a GPU frame's event", error);
        return result;
    }
    result.status = request_status::granted;
    result.granted.m_color = std::move(color.granted);
    result.granted.m_depth = std::move(depth.granted);
    result.granted.m_staging = std::move(staging.granted);
    result.granted.m_staging_copied = event;
    return result;
}

gpu_frame::gpu_frame(gpu_frame &&other) noexcept
    : m_description(std::exchange(other.m_description, frame_description{}))
    , m_index(std::exchange(other.m_index, 0))
    , m_timestamp_us(std::exchange(other.m_timestamp_us, 0))
    , m_color(std::move(other.m_color))
    , m_depth(std::move(other.m_depth))
    , m_staging(std::move(other.m_staging))
    , m_staging_copied(std::exchange(other.m_staging_copied, nullptr))
{
}

gpu_frame &gpu_frame::operator=(gpu_frame &&other) noexcept
{
    if (this != &other)
    {
        // The event first: its wait keeps the staging buffer alive until its copies have run.
        release_event();
        m_description = std::exchange(other.m_description, frame_description{});
        m_index = std::exchange(other.m_index, 0);
        m_timestamp_us = std::exchange(other.m_timestamp_us, 0);
        m_color = std::move(other.m_color);
        m_depth = std::move(other.m_depth);
        m_staging = std::move(other.m_staging);
        m_staging_copied = std::exchange(other.m_staging_copied, nullptr);
    }
    return *this;
}

gpu_frame::~gpu_frame()
{
    // The members, the staging buffer among them, are freed after this body.
    release_event();
}

void gpu_frame::release_event() noexcept
{
    if (m_staging_copied != nullptr)
    {
        cudaEventSynchronize(m_staging_copied);
        cudaEventDestroy(m_staging_copied);
    }
    m_staging_copied = nullptr;
}

copy_result gpu_frame::upload(const frame &host, cudaStream_t stream) noexcept
{
    if (m_staging.empty())
    {
        return {false, "the GPU frame is empty: it was not granted by the pool, or moved from"};
    }
    const frame_size &size = host.description().size;
    if (size != m_color.size())
    {
        return {false,
                "the frame is " + size_text(size) + ", the GPU frame " + size_text(m_color.size())};
    }
    // A frame moved from keeps its description, not its buffers (camera/frame.hpp).
    if (host.color().size_bytes() != m_color.size_bytes() ||
        host.depth().size_bytes() != m_depth.size_bytes())
    {
        return {false, "the frame's buffers hold " + std::to_string(host.color().size_bytes()) +
                           " and " + std::to_string(host.depth().size_bytes()) +
                           " bytes, not its size's (a frame moved from)"};
    }
    // The one wait: the last upload's copies out of the staging buffer have run.
    const cudaError_t wait_error = cudaEventSynchronize(m_staging_copied);
    if (wait_error != cudaSuccess)
    {
        return {false, cuda_failure("CUDA failed while the last upload's copies ran", wait_error)};
    }
    std::byte *const staged_depth = m_staging.bytes().data();
    std::byte *const staged_color = staged_depth + m_depth.size_bytes();
    std::memcpy(staged_depth, host.depth().data(), m_depth.size_bytes());
    std::memcpy(staged_color, host.color().data(), m_color.size_bytes());
    const cudaError_t depth_error = cudaMemcpyAsync(
        m_depth.data(), staged_depth, m_depth.size_bytes(), cudaMemcpyHostToDevice, stream);
    const cudaError_t color_error = cudaMemcpyAsync(
        m_color.data(), staged_color, m_color.size_bytes(), cudaMemcpyHostToDevice, stream);
    // Recorded even after a failed copy, so that the next upload waits for the one that ran.
    const cudaError_t record_error = cudaEventRecord(m_staging_copied, stream);
    if (depth_error != cudaSuccess)
    {
        return {false, cuda_failure("CUDA refused the depth picture's upload", depth_error)};
    }
    if (color_error != cudaSuccess)
    {
        return {false, cuda_failure("CUDA refused the color picture's upload", color_error)};
    }
    if (record_error != cudaSuccess)
    {
        return {false, cuda_failure("CUDA refused the upload's event", record_error)};
    }
    m_description = host.description();
    m_index = host.index();
    m_timestamp_us = host.timestamp_us();
    return {true, {}};
}

copy_result gpu_frame::download(frame &host, cudaStream_t stream) const noexcept
{
    if (m_staging.empty())
    {
        return {false, "the GPU frame is empty: it was not granted by the pool, or moved from"};
    }
    // Only an upload sets the description, and only to the images' size.
    if (m_description.size != m_color.size())
    {
        return {false, "the GPU frame holds no frame yet"};
    }
    host.prepare(m_description, m_index, m_timestamp_us);
    const cudaError_t color_error = cudaMemcpyAsync(
        host.color().data(), m_color.data(), m_color.size_bytes(), cudaMemcpyDeviceToHost, stream);
    const cudaError_t depth_error = cudaMemcpyAsync(
        host.depth().data(), m_depth.data(), m_depth.size_bytes(), cudaMemcpyDeviceToHost, stream);
    // The one wait: the host reads the pictures next, and no copy may still write into host's
    // buffers once the call has returned, done or not.
    const cudaError_t wait_error = cudaStreamSynchronize(stream);
    if (color_error != cudaSuccess)
    {
        return {false, cuda_failure("CUDA refused the color picture's download", color_error)};
    }
    if (depth_error != cudaSuccess)
    {
        return {false, cuda_failure("CUDA refused the depth picture's download", depth_error)};
    }
    if (wait_error != cudaSuccess)
    {
        return {false,
                cuda_failure("CUDA failed while the stream ran to the download", wait_error)};
    }
    return {true, {}};
}

const frame_description &gpu_frame::description() const noexcept
{
    return m_description;
}

std::uint64_t gpu_frame::index() const noexcept
{
    return m_index;
}

std::uint64_t gpu_frame::timestamp_us() const noexcept
{
    return m_timestamp_us;
}

const gpu_image<uchar3> &gpu_frame::color() const noexcept
{
    return m_color;
}

const gpu_image<std::uint16_t> &gpu_frame::depth() const noexcept
{
    return m_depth;
}
} // namespace gca
