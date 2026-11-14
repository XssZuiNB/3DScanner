// SPDX-License-Identifier: 0BSD
/// @file gpu_image.hpp
/// The GPU image: one picture in device memory, typed by its pixel, owned by one object and
/// moved, never copied. Its rows are dense, with no pitch: pixel (u, v) is element v * width + u,
/// rows top to bottom, pixels left to right, as in a frame (camera/frame.hpp). Only the pool
/// (gpu/gpu_pool.hpp) makes an image that holds memory, so every buffer of the per-frame path
/// comes from the pool and exists before the pool is closed. A color image has pixel uchar3 (x
/// red, y green, z blue, 8 bits each), a depth image std::uint16_t in the frame's depth units.
/// Header only.

#pragma once

#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <type_traits>

namespace gca
{
class gpu_pool;

/// One picture in device memory, of pixels of type Pixel, which is copied byte by byte.
/// Move-only, so that no copy allocates unseen. Not synchronized: while one thread moves or ends
/// an image, no other thread uses it.
template <typename Pixel> class gpu_image
{
    static_assert(std::is_trivially_copyable_v<Pixel>, "a pixel is copied byte by byte");

public:
    /// An empty image: 0 x 0 pixels, no memory.
    gpu_image() noexcept = default;

    /// An image is not copied: a copy allocates device memory.
    gpu_image(const gpu_image &) = delete;
    gpu_image &operator=(const gpu_image &) = delete;

    /// Takes over another image's memory without allocating.
    /// @param other [in,out] the image moved from; left empty, 0 x 0 with no memory.
    /// @note Allocates nothing and waits for nothing.
    gpu_image(gpu_image &&other) noexcept
        : m_data(other.m_data)
        , m_size(other.m_size)
    {
        other.m_data = nullptr;
        other.m_size = {0, 0};
    }

    /// Takes over another image's memory; this image's own memory is freed first.
    /// @param other [in,out] the image moved from; left empty, 0 x 0 with no memory. Moving an
    ///        image onto itself changes nothing.
    /// @return this image.
    /// @pre No work queued on the GPU uses this image's own memory any more. When some does,
    ///      cudaFree is not promised to wait for it [S: cuda_runtime_api.h, cudaFree, CUDA 12.6:
    ///      it "may perform implicit synchronization"], and that work may then use freed memory.
    /// @note Frees with cudaFree, which may wait for the GPU [S: as above], so the per-frame path
    ///       moves no image onto one that holds memory. Allocates nothing.
    gpu_image &operator=(gpu_image &&other) noexcept
    {
        if (this != &other)
        {
            release();
            m_data = other.m_data;
            m_size = other.m_size;
            other.m_data = nullptr;
            other.m_size = {0, 0};
        }
        return *this;
    }

    /// Frees the image's memory.
    /// @pre No work queued on the GPU uses the image any more. When some does, cudaFree is not
    ///      promised to wait for it [S: as for the move assignment], and that work may then use
    ///      freed memory.
    /// @note Frees with cudaFree, which may wait for the GPU [S: as for the move assignment], so
    ///       an image of the per-frame path ends after a scan, not during it. A failure of
    ///       cudaFree is not reported; its error is left as CUDA's last error.
    ~gpu_image()
    {
        release();
    }

    /// Gives the image's size.
    /// @return its width and height, in px; 0 x 0 for an empty image.
    frame_size size() const noexcept
    {
        return m_size;
    }

    /// Gives the image's pixels for the GPU to write.
    /// @return device memory of width * height pixels, owned by the image and valid as long as it
    ///         holds them; nullptr for an empty image. Not to be read or written by the host.
    Pixel *data() noexcept
    {
        return m_data;
    }

    /// Gives the image's pixels for the GPU to read.
    /// @return device memory of width * height pixels, owned by the image and valid as long as it
    ///         holds them; nullptr for an empty image. Not to be read by the host.
    const Pixel *data() const noexcept
    {
        return m_data;
    }

    /// Gives the size of the image's pixels.
    /// @return width * height * sizeof(Pixel), in bytes; 0 for an empty image.
    std::size_t size_bytes() const noexcept
    {
        return static_cast<std::size_t>(m_size.width) * m_size.height * sizeof(Pixel);
    }

    /// Says whether the image holds memory.
    /// @return true for an image made empty or moved from, false for one the pool granted.
    bool empty() const noexcept
    {
        return m_data == nullptr;
    }

private:
    friend class gpu_pool;

    /// Takes over device memory the pool allocated.
    /// @param data [in] device memory of width * height pixels from cudaMalloc, not nullptr;
    ///        owned by the image from now on.
    /// @param size [in] its width and height, in px, each > 0.
    gpu_image(Pixel *data, const frame_size &size) noexcept
        : m_data(data)
        , m_size(size)
    {
    }

    /// Frees the image's memory, if it holds any, and leaves it empty.
    /// @note cudaFree may wait for the GPU [S: as for the move assignment]; its failure is not
    ///       reported, as nothing can be done about it here.
    void release() noexcept
    {
        if (m_data != nullptr)
        {
            cudaFree(m_data);
        }
        m_data = nullptr;
        m_size = {0, 0};
    }

    Pixel *m_data = nullptr;
    frame_size m_size{0, 0};
};
} // namespace gca
