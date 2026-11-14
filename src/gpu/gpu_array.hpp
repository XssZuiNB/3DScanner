// SPDX-License-Identifier: 0BSD
/// @file gpu_array.hpp
/// The GPU array: a number of elements of one type in device memory, owned by one object and
/// moved, never copied. It is the one owner of device memory in the rewrite: a GPU image
/// (gpu/gpu_image.hpp) is an array of its pixels with their width and height. Only the pool
/// (gpu/gpu_pool.hpp) makes an array that holds memory, so every buffer of the per-frame path
/// comes from the pool and exists before the pool is closed. Header only.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <type_traits>

namespace gca
{
class gpu_pool;

/// Elements of type Element in device memory, which are copied byte by byte. Move-only, so that
/// no copy allocates unseen. Not synchronized: while one thread moves or ends an array, no other
/// thread uses it.
template <typename Element> class gpu_array
{
    static_assert(std::is_trivially_copyable_v<Element>, "an element is copied byte by byte");

public:
    /// An empty array: no elements, no memory.
    gpu_array() noexcept = default;

    /// An array is not copied: a copy allocates device memory.
    gpu_array(const gpu_array &) = delete;
    gpu_array &operator=(const gpu_array &) = delete;

    /// Takes over another array's memory without allocating.
    /// @param other [in,out] the array moved from; left empty, no elements and no memory.
    /// @note Allocates nothing and waits for nothing.
    gpu_array(gpu_array &&other) noexcept
        : m_data(other.m_data)
        , m_size(other.m_size)
    {
        other.m_data = nullptr;
        other.m_size = 0;
    }

    /// Takes over another array's memory; this array's own memory is freed first.
    /// @param other [in,out] the array moved from; left empty, no elements and no memory. Moving
    ///        an array onto itself changes nothing.
    /// @return this array.
    /// @pre No work queued on the GPU uses this array's own memory any more. When some does,
    ///      cudaFree is not promised to wait for it [S: cuda_runtime_api.h, cudaFree, CUDA 12.6:
    ///      it "may perform implicit synchronization"], and that work may then use freed memory.
    /// @note Frees with cudaFree, which may wait for the GPU [S: as above], so the per-frame path
    ///       moves no array onto one that holds memory. Allocates nothing.
    gpu_array &operator=(gpu_array &&other) noexcept
    {
        if (this != &other)
        {
            release();
            m_data = other.m_data;
            m_size = other.m_size;
            other.m_data = nullptr;
            other.m_size = 0;
        }
        return *this;
    }

    /// Frees the array's memory.
    /// @pre No work queued on the GPU uses the array any more. When some does, cudaFree is not
    ///      promised to wait for it [S: as for the move assignment], and that work may then use
    ///      freed memory.
    /// @note Frees with cudaFree, which may wait for the GPU [S: as for the move assignment], so
    ///       an array of the per-frame path ends after a scan, not during it. A failure of
    ///       cudaFree is not reported; its error is left as CUDA's last error.
    ~gpu_array()
    {
        release();
    }

    /// Gives the number of elements.
    /// @return the elements the array holds; 0 for an empty array.
    std::size_t size() const noexcept
    {
        return m_size;
    }

    /// Gives the elements for the GPU to write.
    /// @return device memory of size() elements, owned by the array and valid as long as it holds
    ///         them; nullptr for an empty array. Not to be read or written by the host.
    Element *data() noexcept
    {
        return m_data;
    }

    /// Gives the elements for the GPU to read.
    /// @return device memory of size() elements, owned by the array and valid as long as it holds
    ///         them; nullptr for an empty array. Not to be read by the host.
    const Element *data() const noexcept
    {
        return m_data;
    }

    /// Gives the size of the elements.
    /// @return size() * sizeof(Element), in bytes; 0 for an empty array.
    std::size_t size_bytes() const noexcept
    {
        return m_size * sizeof(Element);
    }

    /// Says whether the array holds memory.
    /// @return true for an array made empty or moved from, false for one the pool granted.
    bool empty() const noexcept
    {
        return m_data == nullptr;
    }

private:
    friend class gpu_pool;

    /// Takes over device memory the pool allocated.
    /// @param data [in] device memory of size elements from cudaMalloc, not nullptr; owned by the
    ///        array from now on.
    /// @param size [in] the number of elements, > 0.
    gpu_array(Element *data, std::size_t size) noexcept
        : m_data(data)
        , m_size(size)
    {
    }

    /// Frees the array's memory, if it holds any, and leaves it empty.
    /// @note cudaFree may wait for the GPU [S: as for the move assignment]; its failure is not
    ///       reported, as nothing can be done about it here.
    void release() noexcept
    {
        if (m_data != nullptr)
        {
            cudaFree(m_data);
        }
        m_data = nullptr;
        m_size = 0;
    }

    Element *m_data = nullptr;
    std::size_t m_size = 0;
};
} // namespace gca
