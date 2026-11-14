// SPDX-License-Identifier: 0BSD
/// @file gpu_image.hpp
/// The GPU image: one picture in device memory, typed by its pixel, owned by one object and
/// moved, never copied: a GPU array (gpu/gpu_array.hpp) of its pixels, which owns the memory,
/// with the picture's width and height. Its rows are dense, with no pitch: pixel (u, v) is
/// element v * width + u, rows top to bottom, pixels left to right, as in a frame
/// (camera/frame.hpp). Only the pool (gpu/gpu_pool.hpp) makes an image that holds memory, so every
/// buffer of the per-frame path comes from the pool and exists before the pool is closed. A color
/// image has pixel uchar3 (x red, y green, z blue, 8 bits each), a depth image std::uint16_t in
/// the frame's depth units. Header only.

#pragma once

#include "gpu/gpu_array.hpp"
#include "types/frame_description.hpp"

#include <cstddef>
#include <utility>

namespace gca
{
class gpu_pool;

/// One picture in device memory, of pixels of type Pixel, which is copied byte by byte.
/// Move-only, so that no copy allocates unseen. Not synchronized: while one thread moves or ends
/// an image, no other thread uses it.
template <typename Pixel> class gpu_image
{
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
        : m_pixels(std::move(other.m_pixels))
        , m_size(other.m_size)
    {
        other.m_size = {0, 0};
    }

    /// Takes over another image's memory; this image's own memory is freed first.
    /// @param other [in,out] the image moved from; left empty, 0 x 0 with no memory. Moving an
    ///        image onto itself changes nothing.
    /// @return this image.
    /// @pre No work queued on the GPU uses this image's own memory any more; when some does, it
    ///      may use freed memory (gpu_array's move assignment).
    /// @note Frees with cudaFree, which may wait for the GPU (gpu_array's move assignment), so
    ///       the per-frame path moves no image onto one that holds memory. Allocates nothing.
    gpu_image &operator=(gpu_image &&other) noexcept
    {
        if (this != &other)
        {
            m_pixels = std::move(other.m_pixels);
            m_size = other.m_size;
            other.m_size = {0, 0};
        }
        return *this;
    }

    /// Frees the image's memory.
    /// @pre No work queued on the GPU uses the image any more; when some does, it may use freed
    ///      memory (gpu_array's end).
    /// @note Frees with cudaFree, which may wait for the GPU (gpu_array's end), so an image of
    ///       the per-frame path ends after a scan, not during it. A failure of cudaFree is not
    ///       reported; its error is left as CUDA's last error.
    ~gpu_image() = default;

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
        return m_pixels.data();
    }

    /// Gives the image's pixels for the GPU to read.
    /// @return device memory of width * height pixels, owned by the image and valid as long as it
    ///         holds them; nullptr for an empty image. Not to be read by the host.
    const Pixel *data() const noexcept
    {
        return m_pixels.data();
    }

    /// Gives the size of the image's pixels.
    /// @return width * height * sizeof(Pixel), in bytes; 0 for an empty image.
    std::size_t size_bytes() const noexcept
    {
        return m_pixels.size_bytes();
    }

    /// Says whether the image holds memory.
    /// @return true for an image made empty or moved from, false for one the pool granted.
    bool empty() const noexcept
    {
        return m_pixels.empty();
    }

private:
    friend class gpu_pool;

    /// Takes over the pixels the pool allocated.
    /// @param pixels [in,out] an array of width * height pixels, not empty; taken over, left
    ///        empty.
    /// @param size [in] its width and height, in px, each > 0.
    gpu_image(gpu_array<Pixel> &&pixels, const frame_size &size) noexcept
        : m_pixels(std::move(pixels))
        , m_size(size)
    {
    }

    gpu_array<Pixel> m_pixels;
    frame_size m_size{0, 0};
};
} // namespace gca
