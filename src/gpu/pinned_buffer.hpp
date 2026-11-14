// SPDX-License-Identifier: 0BSD
/// @file pinned_buffer.hpp
/// The pinned host buffer: page-locked host memory from cudaMallocHost, owned by one object and
/// moved, never copied. A copy between it and the GPU queued with cudaMemcpyAsync "should be
/// fully asynchronous", where one from pageable memory "might be synchronous" and may wait for
/// the stream's earlier work [S: cuda_runtime_api.h, "API synchronization behavior", CUDA 12.6].
/// Only the pool (gpu/gpu_pool.hpp) makes one that holds memory, before it is closed. Header
/// only.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <span>

namespace gca
{
class gpu_pool;

/// Page-locked host memory, a number of bytes. Move-only, so that no copy allocates unseen. Not
/// synchronized: while one thread writes, moves or ends a buffer, no other thread uses it.
class pinned_buffer
{
public:
    /// An empty buffer: no bytes, no memory.
    pinned_buffer() noexcept = default;

    /// A buffer is not copied: a copy allocates page-locked memory.
    pinned_buffer(const pinned_buffer &) = delete;
    pinned_buffer &operator=(const pinned_buffer &) = delete;

    /// Takes over another buffer's memory without allocating.
    /// @param other [in,out] the buffer moved from; left empty, with no memory.
    /// @note Allocates nothing and waits for nothing.
    pinned_buffer(pinned_buffer &&other) noexcept
        : m_bytes(other.m_bytes)
    {
        other.m_bytes = {};
    }

    /// Takes over another buffer's memory; this buffer's own memory is freed first.
    /// @param other [in,out] the buffer moved from; left empty, with no memory. Moving a buffer
    ///        onto itself changes nothing.
    /// @return this buffer.
    /// @pre No copy queued on the GPU reads or writes this buffer's own memory any more. When
    ///      one does, it may use freed memory: cudaFreeHost is not documented to wait for it
    ///      [S: cuda_runtime_api.h, cudaFreeHost, CUDA 12.6].
    /// @note Frees with cudaFreeHost; whether that waits for the GPU is not documented [?].
    ///       Allocates nothing.
    pinned_buffer &operator=(pinned_buffer &&other) noexcept
    {
        if (this != &other)
        {
            release();
            m_bytes = other.m_bytes;
            other.m_bytes = {};
        }
        return *this;
    }

    /// Frees the buffer's memory.
    /// @pre No copy queued on the GPU reads or writes the buffer any more. When one does, it may
    ///      use freed memory [S: as for the move assignment].
    /// @note Frees with cudaFreeHost; whether that waits for the GPU is not documented [?]. A
    ///       failure of cudaFreeHost is not reported; its error is left as CUDA's last error.
    ~pinned_buffer()
    {
        release();
    }

    /// Gives the buffer's bytes.
    /// @return page-locked host memory owned by the buffer, valid as long as it holds it; empty
    ///         for an empty buffer.
    std::span<std::byte> bytes() noexcept
    {
        return m_bytes;
    }

    /// Gives the buffer's bytes.
    /// @return page-locked host memory owned by the buffer, valid as long as it holds it; empty
    ///         for an empty buffer.
    std::span<const std::byte> bytes() const noexcept
    {
        return m_bytes;
    }

    /// Says whether the buffer holds memory.
    /// @return true for a buffer made empty or moved from, false for one the pool granted.
    bool empty() const noexcept
    {
        return m_bytes.data() == nullptr;
    }

private:
    friend class gpu_pool;

    /// Takes over page-locked memory the pool allocated.
    /// @param bytes [in] memory from cudaMallocHost, not empty; owned by the buffer from now on.
    explicit pinned_buffer(std::span<std::byte> bytes) noexcept
        : m_bytes(bytes)
    {
    }

    /// Frees the buffer's memory, if it holds any, and leaves it empty.
    /// @note cudaFreeHost's failure is not reported, as nothing can be done about it here.
    void release() noexcept
    {
        if (m_bytes.data() != nullptr)
        {
            cudaFreeHost(m_bytes.data());
        }
        m_bytes = {};
    }

    std::span<std::byte> m_bytes;
};
} // namespace gca
