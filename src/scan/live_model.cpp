// SPDX-License-Identifier: 0BSD
/// @file live_model.cpp
/// The taking of a scan's live models (see live_model.hpp; docs/methods/tracking.md, "The live
/// model").

#include "scan/live_model.hpp"

#include "gpu/gpu_messages.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <utility>

namespace gca
{
std::uint32_t spacing_after(std::uint64_t points, std::uint32_t spacing,
                            std::uint32_t limit) noexcept
{
    const double most = live_model_fill * static_cast<double>(limit);
    double expected = static_cast<double>(points);
    std::uint32_t next = spacing;
    // A doubling of the spacing keeps a quarter of a surface's edges [S: a surface's crossings
    // along one axis lie on the grid of the other two coordinates, each thinned by half].
    while (expected > most && next < (1u << 30))
    {
        next *= 2;
        expected /= 4.0;
    }
    return next;
}

request_result<live_model_taker> live_model_taker::request(
    gpu_pool &pool, std::uint32_t limit, const sparse_tsdf_layout &layout) noexcept
{
    request_result<live_model_taker> result;
    request_result<surface_points> extracted =
        surface_points::request(pool, limit, layout.capacity_blocks);
    if (extracted.status != request_status::granted)
        return refused_as<live_model_taker>(extracted);
    request_result<pinned_buffer> points =
        pool.request_pinned(std::size_t{limit} * sizeof(surface_point));
    if (points.status != request_status::granted)
        return refused_as<live_model_taker>(points);
    request_result<pinned_buffer> slots =
        pool.request_pinned(std::size_t{limit} * sizeof(std::uint32_t));
    if (slots.status != request_status::granted)
        return refused_as<live_model_taker>(slots);
    request_result<pinned_buffer> blocks =
        pool.request_pinned(std::size_t{layout.capacity_blocks} * sizeof(std::uint32_t));
    if (blocks.status != request_status::granted)
        return refused_as<live_model_taker>(blocks);
    result.status = request_status::granted;
    live_model_taker &made = result.granted;
    made.m_limit = limit;
    made.m_extracted = std::move(extracted.granted);
    made.m_points_on_host = std::move(points.granted);
    made.m_slots_on_host = std::move(slots.granted);
    made.m_blocks_on_host = std::move(blocks.granted);
    made.m_touched.resize(layout.slot_count);
    return result;
}

live_take live_model_taker::take(const sparse_tsdf &model, const held_live_model &last,
                                 held_live_model &next, cudaStream_t stream) noexcept
{
    const std::vector<model_point> &last_points = last.value.points;
    std::vector<model_point> &points = next.value.points;
    std::vector<std::uint32_t> &slots = next.slots;
    live_take made;
    made.every_block = m_every_block;
    made.spacing = m_spacing;
    if (m_limit == 0)
    {
        made.message = "the live model's buffers hold no memory";
        return made;
    }
    // The stamp the allocations after this extraction take, and so the next take's since.
    const std::uint32_t next_since = model.frames_allocated();
    const extraction_result extracted =
        model.extract(m_extracted, {m_spacing, m_every_block ? 0u : m_since}, stream);
    made.found = extracted.found;
    made.blocks = extracted.blocks;
    if (extracted.found > m_extracted.capacity())
    {
        // More than the limit is more than live_model_fill of it: the spacing rises.
        m_spacing = spacing_after(extracted.found, m_spacing, m_limit);
        m_every_block = true;
        made.status = live_take_status::too_many;
        return made;
    }
    if (!extracted.done)
    {
        made.message = "the extraction: " + extracted.message;
        return made;
    }

    // The points, their slots and the blocks gone through, in one copy each, then one wait.
    const std::size_t written = extracted.written;
    cudaError_t error =
        cudaMemcpyAsync(m_points_on_host.bytes().data(), m_extracted.points().data(),
                        written * sizeof(surface_point), cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(m_slots_on_host.bytes().data(), m_extracted.slots().data(),
                                written * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess && !m_every_block)
        error = cudaMemcpyAsync(m_blocks_on_host.bytes().data(), m_extracted.blocks().data(),
                                std::size_t{extracted.blocks} * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess)
        error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess)
    {
        made.message = cuda_failure("the copy of the live model's points", error);
        return made;
    }
    const auto *new_points =
        reinterpret_cast<const surface_point *>(m_points_on_host.bytes().data());
    const auto *new_slots = reinterpret_cast<const std::uint32_t *>(m_slots_on_host.bytes().data());

    // The points kept: none when every block was taken, else the last live model's points of the
    // blocks not gone through, the touched blocks marked and the marks taken back after.
    std::size_t kept = 0;
    if (!m_every_block)
    {
        const auto *touched =
            reinterpret_cast<const std::uint32_t *>(m_blocks_on_host.bytes().data());
        m_touched.mark(touched, extracted.blocks);
        for (const std::uint32_t slot : last.slots)
            kept += !m_touched.marked(slot);
        if (kept + written > m_limit)
        {
            m_touched.clear(touched, extracted.blocks);
            m_spacing = spacing_after(kept + written, m_spacing, m_limit);
            m_every_block = true;
            made.status = live_take_status::too_many;
            return made;
        }
        points.resize(kept + written);
        slots.resize(kept + written);
        std::size_t to = 0;
        for (std::size_t k = 0; k < last_points.size(); k++)
            if (!m_touched.marked(last.slots[k]))
            {
                points[to] = last_points[k];
                slots[to] = last.slots[k];
                to++;
            }
        m_touched.clear(touched, extracted.blocks);
    }
    else
    {
        points.resize(written);
        slots.resize(written);
    }
    for (std::size_t k = 0; k < written; k++)
    {
        points[kept + k] = model_point_of(new_points[k]);
        slots[kept + k] = new_slots[k];
    }

    m_since = next_since;
    const std::uint32_t next_spacing = spacing_after(points.size(), m_spacing, m_limit);
    m_every_block = next_spacing != m_spacing;
    m_spacing = next_spacing;
    made.status = live_take_status::taken;
    return made;
}
} // namespace gca
