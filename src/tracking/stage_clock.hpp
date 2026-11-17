// SPDX-License-Identifier: 0BSD
/// @file stage_clock.hpp
/// The stage clock (docs/methods/tracking.md, "The stages' times"): the time of each stage of the
/// per-frame path, frame after frame, for the replay tool's measurement only. The scan's loop and
/// the tracker mark each stage's end; after a stage that queues work on the GPU the clock waits
/// for the stream, so that the stage's time holds that work, and a stage's time is the time
/// since the mark before. A live scan has no clock: the path then waits only where it is written
/// to wait. When a frame ends, its times go to a listener, which the replay prints as a line.

#pragma once

#include "gpu/step_result.hpp"
#include "tracking/tracker.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>

namespace gca
{
/// The stages of the per-frame path, in the order of the time lines' columns
/// (docs/methods/tracking.md, "The stages' times"; the path runs the ray cast after the fusion and
/// the live view before the live model). The enumerators run from 0 without a gap, live_view the
/// last, and each has its row in stage_clock.cpp's table, a switch that -Wswitch checks.
enum class path_stage
{
    room,         ///< the writer's room (the scan's loop, step 2)
    too_close,    ///< the "too close" notice (the scan's loop, step 3)
    upload,       ///< the upload (the path, step 1)
    pyramid,      ///< the pyramid (the path, step 2)
    ray_cast,     ///< the surface, the live model's ray cast after the fusion (the path, step 3)
    registration, ///< the registration (the path, step 4)
    check,        ///< the check and the frame's pose (the path, step 5)
    fusion,       ///< the fusion (the path, step 6)
    writer,       ///< the writer taking the frame (the scan's loop, step 5)
    live_model,   ///< the live model, when the frame takes one (the scan's loop, step 6)
    live_view,    ///< the live view's shading and copy, when a free encoder takes the frame's
                  ///< (the scan's loop, step 6)
};

/// The number of the path's stages.
inline constexpr std::size_t path_stage_count = static_cast<std::size_t>(path_stage::live_view) + 1;

/// Gives a stage's name, as the replay's time lines print it.
/// @param stage [in] the stage, one of the enumerators.
/// @return "room", "too_close", "upload", "pyramid", "ray_cast", "registration", "check",
///         "fusion", "writer", "live_model" or "live_view"; a static string; empty for a value
///         that is no enumerator.
std::string_view name_of(path_stage stage) noexcept;

/// One frame's times.
struct frame_times
{
    std::uint64_t index = 0; ///< the frame's index
    /// What tracking made of it; none for a frame not taken, which found no room.
    std::optional<tracking_state> state;
    /// Each stage's time, in ms, in the order of path_stage; 0 for a stage the frame did not run:
    /// the fusion and the ray cast of a frame in tracking loss, the stages after the one that
    /// failed; the live model's on a frame that takes none is the decision's alone, near 0, and
    /// the live view's on a frame whose encoder is busy or that is lost the offer's alone.
    std::array<double, path_stage_count> stage_ms{};
    double total_ms = 0.0; ///< the frame's time from begin() to end(), in ms
};

/// The stage clock. One thread uses it: the scan thread.
class stage_clock
{
public:
    /// Builds the clock.
    /// @param on_frame [in] called at each frame's end() with its times, on the thread that ends
    ///        it, after the frame's time is taken; it must return soon and must not throw. Copied.
    explicit stage_clock(std::function<void(const frame_times &)> on_frame);

    /// Begins a frame: its times set to 0, its start and the last mark now.
    /// @param index [in] the frame's index.
    /// @note Allocates nothing and waits for nothing.
    void begin(std::uint64_t index) noexcept;

    /// Marks the end of a stage that runs on the host: its time is the time since the last mark.
    /// @param stage [in] the stage.
    /// @note Allocates nothing and waits for nothing.
    void mark(path_stage stage) noexcept;

    /// Marks the end of a stage that queues work on the GPU: waits for the stream, then the
    /// stage's time is the time since the last mark.
    /// @param stage [in] the stage.
    /// @param stream [in] the CUDA stream the stage queued its work on; owned by the caller.
    /// @return done; or CUDA's error of the wait, the stage's time then not taken.
    /// @note Waits for every piece of work on the stream (cudaStreamSynchronize): the wait the
    ///       measurement adds. Allocates nothing but for a failure's message.
    step_result mark_after(path_stage stage, cudaStream_t stream) noexcept;

    /// Ends the frame: its time from begin() to now, then the listener with its times.
    /// @param state [in] what tracking made of it; none for a frame not taken.
    /// @note Allocates nothing; calls the listener, and waits for it.
    void end(std::optional<tracking_state> state) noexcept;

private:
    using clock = std::chrono::steady_clock;

    std::function<void(const frame_times &)> m_on_frame; ///< the listener
    frame_times m_times;                                 ///< the frame's times so far
    clock::time_point m_start;                           ///< the frame's begin()
    clock::time_point m_last;                            ///< the last mark, or the begin()
};
} // namespace gca
