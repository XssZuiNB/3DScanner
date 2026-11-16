// SPDX-License-Identifier: 0BSD
/// @file stage_clock.cpp
/// The stage clock: the stages' names, the marks with their one wait, a frame's end (see
/// stage_clock.hpp).

#include "tracking/stage_clock.hpp"

#include "gpu/gpu_messages.hpp"

#include <utility>

namespace gca
{
namespace
{
/// The milliseconds between two times.
/// @param from [in] the earlier time.
/// @param to [in] the later time.
/// @return to - from, in ms.
double milliseconds(std::chrono::steady_clock::time_point from,
                    std::chrono::steady_clock::time_point to) noexcept
{
    return std::chrono::duration<double, std::milli>(to - from).count();
}
} // namespace

std::string_view name_of(path_stage stage) noexcept
{
    switch (stage)
    {
    case path_stage::room:
        return "room";
    case path_stage::too_close:
        return "too_close";
    case path_stage::upload:
        return "upload";
    case path_stage::pyramid:
        return "pyramid";
    case path_stage::ray_cast:
        return "ray_cast";
    case path_stage::registration:
        return "registration";
    case path_stage::check:
        return "check";
    case path_stage::fusion:
        return "fusion";
    case path_stage::writer:
        return "writer";
    }
    return {};
}

stage_clock::stage_clock(std::function<void(const frame_times &)> on_frame)
    : m_on_frame(std::move(on_frame))
{
}

void stage_clock::begin(std::uint64_t index) noexcept
{
    m_times = {};
    m_times.index = index;
    m_start = clock::now();
    m_last = m_start;
}

void stage_clock::mark(path_stage stage) noexcept
{
    const clock::time_point now = clock::now();
    m_times.stage_ms[static_cast<std::size_t>(stage)] = milliseconds(m_last, now);
    m_last = now;
}

step_result stage_clock::mark_after(path_stage stage, cudaStream_t stream) noexcept
{
    const cudaError_t waited = cudaStreamSynchronize(stream);
    if (waited != cudaSuccess)
        return {false, cuda_failure("the stage clock's wait for the stream", waited)};
    mark(stage);
    return {true, {}};
}

void stage_clock::end(std::optional<tracking_state> state) noexcept
{
    m_times.total_ms = milliseconds(m_start, clock::now());
    m_times.state = state;
    if (m_on_frame)
        m_on_frame(m_times);
}
} // namespace gca
