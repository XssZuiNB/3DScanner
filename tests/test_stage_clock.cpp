// SPDX-License-Identifier: 0BSD
/// @file test_stage_clock.cpp
/// The stage clock (tracking/stage_clock.hpp; docs/methods/tracking.md, "The stages' times"),
/// which times the per-frame path's stages for a replay's measurement:
/// - the stages' names, as the replay's time lines print them, in the path's order;
/// - a stage that queues work on the GPU: its time holds that work, two memsets of a buffer
///   timed by CUDA's events, because the clock waits for the stream; and the stream is idle after
///   the mark;
/// - host stages: each stage's time is the time since the mark before, a stage not marked is 0,
///   the stages add up to the frame's time, and the frame's times go to the listener with its
///   index and state when the frame ends, anew for each frame.
/// Needs the GPU and no camera.

#include "check.hpp"
#include "gpu_test_helpers.hpp"
#include "tracking/stage_clock.hpp"
#include "tracking/tracker.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace
{
using gca_test::check;

/// The names the time lines print, in the path's order (docs/methods/tracking.md, "The stages'
/// times").
const std::vector<std::string> expected_names = {"room",     "too_close",    "upload", "pyramid",
                                                 "ray_cast", "registration", "check",  "fusion",
                                                 "writer",   "live_model"};

/// Each stage's name, in the order of the enumerators.
void test_names()
{
    check(gca::path_stage_count == expected_names.size(), "names: ten stages");
    for (std::size_t k = 0; k < gca::path_stage_count; k++)
        check(gca::name_of(static_cast<gca::path_stage>(k)) == expected_names[k],
              ("names: stage " + std::to_string(k) + " is " + expected_names[k]).c_str());
}

/// The buffer the GPU stage's memsets write: 256 MiB [?: a choice], so that two memsets take 18.4
/// ms on the GPU [M 2026-11-16: this test], far above a mark that returns at once.
constexpr std::size_t gpu_stage_bytes = std::size_t{256} << 20;

/// The least time the GPU stage's work takes, in ms: 0.5 ms [?: a choice], 37 times below the
/// 18.4 ms measured [M 2026-11-16: this test], so that the work is long against the clock's reads.
constexpr float gpu_stage_least_ms = 0.5f;

/// How far the host stages' sum may fall below their frame's time, in ms: the reads of the clock
/// between begin() and end() and the first mark, microseconds each [?: a choice, 1 ms].
constexpr double host_sum_margin_ms = 1.0;

/// The longest the second host frame, which sleeps not at all, may take, in ms: half the first
/// stage's 20 ms of sleep [?: a choice], so that times carried over from the first frame show.
constexpr double second_frame_longest_ms = 10.0;

/// A stage that queues work on the GPU: two memsets of gpu_stage_bytes, between two CUDA events,
/// then the mark that waits for the stream. The stage's time holds the work's time on the GPU,
/// which the events give; a mark that did not wait would return in microseconds, before the work.
void test_gpu_stage()
{
    cudaStream_t stream = nullptr;
    if (!gca_test::create_stream(stream))
        return;
    constexpr std::size_t bytes = gpu_stage_bytes;
    void *buffer = nullptr;
    cudaEvent_t before = nullptr;
    cudaEvent_t after = nullptr;
    check(cudaMalloc(&buffer, bytes) == cudaSuccess && cudaEventCreate(&before) == cudaSuccess &&
              cudaEventCreate(&after) == cudaSuccess,
          "GPU stage: test setup, the buffer and the events");
    std::vector<gca::frame_times> seen;
    gca::stage_clock clock([&seen](const gca::frame_times &times) { seen.push_back(times); });
    clock.begin(7);
    check(cudaEventRecord(before, stream) == cudaSuccess &&
              cudaMemsetAsync(buffer, 1, bytes, stream) == cudaSuccess &&
              cudaMemsetAsync(buffer, 2, bytes, stream) == cudaSuccess &&
              cudaEventRecord(after, stream) == cudaSuccess,
          "GPU stage: test setup, the work queued");
    const gca::step_result waited = clock.mark_after(gca::path_stage::fusion, stream);
    check(waited.done, ("GPU stage: the wait; " + waited.message).c_str());
    check(cudaStreamQuery(stream) == cudaSuccess, "GPU stage: the stream is idle after the mark");
    clock.end(gca::tracking_state::tracking);
    float gpu_ms = 0.0f;
    check(cudaEventElapsedTime(&gpu_ms, before, after) == cudaSuccess,
          "GPU stage: the work's time from the events");
    check(seen.size() == 1, "GPU stage: one frame's times");
    if (seen.size() == 1)
    {
        const double stage_ms = seen[0].stage_ms[static_cast<std::size_t>(gca::path_stage::fusion)];
        std::printf("GPU stage: the stage %.3f ms, the work on the GPU %.3f ms\n", stage_ms,
                    double{gpu_ms});
        check(gpu_ms > gpu_stage_least_ms, "GPU stage: the work takes more than 0.5 ms on the GPU");
        check(stage_ms >= double{gpu_ms}, "GPU stage: the stage's time holds the work's");
    }
    cudaEventDestroy(before);
    cudaEventDestroy(after);
    cudaFree(buffer);
    cudaStreamDestroy(stream);
    gca_test::check_no_cuda_error("GPU stage");
}

/// Host stages over two frames: a stage's time is the time since the mark before (20 ms and
/// 10 ms of sleep, [?: a choice] long against the clock's own reads), a stage not marked is 0,
/// the stages add up to the frame's time, and the listener gets each frame's index and state when
/// it ends; the second frame's times start anew.
void test_host_stages()
{
    std::vector<gca::frame_times> seen;
    gca::stage_clock clock([&seen](const gca::frame_times &times) { seen.push_back(times); });
    clock.begin(3);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    clock.mark(gca::path_stage::room);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    clock.mark(gca::path_stage::check);
    clock.end(gca::tracking_state::lost);
    clock.begin(4);
    clock.mark(gca::path_stage::writer);
    clock.end(std::nullopt);
    check(seen.size() == 2, "host stages: two frames' times");
    if (seen.size() != 2)
        return;
    const gca::frame_times &first = seen[0];
    const double room = first.stage_ms[static_cast<std::size_t>(gca::path_stage::room)];
    const double checked = first.stage_ms[static_cast<std::size_t>(gca::path_stage::check)];
    double sum = 0.0;
    int zeros = 0;
    for (const double ms : first.stage_ms)
    {
        sum += ms;
        zeros += ms == 0.0;
    }
    std::printf("host stages: room %.3f ms, check %.3f ms, total %.3f ms\n", room, checked,
                first.total_ms);
    check(first.index == 3 && first.state == gca::tracking_state::lost,
          "host stages: the first frame's index and state");
    check(room >= 20.0 && checked >= 10.0, "host stages: each stage's time since the mark before");
    check(zeros == 8, "host stages: the stages not marked are 0");
    check(sum <= first.total_ms && first.total_ms - sum < host_sum_margin_ms,
          "host stages: the stages add up to the frame's time");
    const gca::frame_times &second = seen[1];
    check(second.index == 4 && !second.state, "host stages: the second frame, not taken");
    check(second.stage_ms[static_cast<std::size_t>(gca::path_stage::room)] == 0.0 &&
              second.total_ms < second_frame_longest_ms,
          "host stages: the second frame's times start anew");
}
} // namespace

int main()
{
    test_names();
    test_gpu_stage();
    test_host_stages();
    return gca_test::check_summary("test_stage_clock");
}
