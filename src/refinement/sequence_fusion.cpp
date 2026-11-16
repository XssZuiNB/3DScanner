// SPDX-License-Identifier: 0BSD
/// @file sequence_fusion.cpp
/// Refinement's fusion of a scan's sequence at the full voxel size (see sequence_fusion.hpp,
/// docs/methods/refinement.md, "The fusion of the sequence").

#include "refinement/sequence_fusion.hpp"

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/sequence_format.hpp"
#include "camera/sequence_reader.hpp"
#include "gpu/gpu_frame.hpp"
#include "gpu/gpu_messages.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/step_result.hpp"

#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace gca
{
namespace
{
/// Ends a fusion other than done: no model and no trajectory, so that a caller cannot keep a
/// part of either; the counts as they are.
/// @param result [in,out] the result so far.
/// @param end [in] failed or cancelled.
/// @param message [in] why it failed, in plain English, naming the sequence; empty when
///        cancelled.
/// @return the result.
sequence_fusion_result ended_with(sequence_fusion_result &result, sequence_fusion_end end,
                                  std::string message)
{
    result.end = end;
    result.message = std::move(message);
    result.model.clear();
    result.trajectory.clear();
    return std::move(result);
}

/// Ends a fusion as failed (ended_with()).
/// @param result [in,out] the result so far.
/// @param message [in] why, in plain English, naming the sequence.
/// @return the result.
sequence_fusion_result failed_with(sequence_fusion_result &result, std::string message)
{
    return ended_with(result, sequence_fusion_end::failed, std::move(message));
}

/// Checks that the full model's pool held every block its frames touched: an integration skips a
/// slot the pool had no block for and is done all the same (fusion/sparse_tsdf.hpp, integrate()),
/// so a used-up pool would give a full model that lacks surface.
/// @param tsdf [in,out] the full model's sparse TSDF, after its frames' integrations.
/// @param budget_bytes [in] its memory budget, for the reason.
/// @param stream [in] the stream the integrations were queued on.
/// @return the reason it did not, with the counts; empty when it did.
/// @note The one written wait of sparse_tsdf::read_usage(), which waits for the integrations.
std::string used_up_refusal(sparse_tsdf &tsdf, std::uint64_t budget_bytes, cudaStream_t stream)
{
    const usage_result read = tsdf.read_usage(stream);
    if (!read.usage)
        return "the full model's usage: " + read.message;
    const sparse_tsdf_usage &usage = *read.usage;
    if (usage.blocks_refused == 0 && usage.coordinates_refused == 0)
        return {};
    return "the full model's memory budget of " + std::to_string(budget_bytes) +
           " bytes was used up: " + std::to_string(usage.blocks_in_use) + " of " +
           std::to_string(usage.capacity_blocks) + " blocks in use, " +
           std::to_string(usage.blocks_refused) + " coordinates found no block and " +
           std::to_string(usage.coordinates_refused) +
           " insertions were refused, so the full model would lack surface";
}

/// Extracts the full model and copies it to the host: the extraction into a buffer of one point
/// counts the surface's points, and the extraction into a buffer of that count writes them all.
/// @param tsdf [in] the full model's sparse TSDF.
/// @param pool [in,out] the call's pool, open, which grants the second buffer.
/// @param counter [in,out] the buffer of one point.
/// @param stream [in] the stream the extractions run and are waited for on.
/// @param reason [out] why the extraction or the copy failed; set only then.
/// @return the points; nothing when a step failed.
std::optional<std::vector<model_point>> extracted(const sparse_tsdf &tsdf, gpu_pool &pool,
                                                  surface_points &counter, cudaStream_t stream,
                                                  std::string &reason)
{
    const extraction_result counted = tsdf.extract(counter, extraction_choice{}, stream);
    // Not done with points found is a buffer too small; not done without any is a failure.
    if (!counted.done && counted.found == 0)
    {
        reason = "the extraction: " + counted.message;
        return std::nullopt;
    }
    if (counted.found == 0)
        return std::vector<model_point>{};
    surface_points sized;
    surface_points *buffer = &counter;
    if (!counted.done)
    {
        request_result<surface_points> granted = surface_points::request(pool, counted.found);
        if (granted.status != request_status::granted)
        {
            reason = "the extraction's buffer of " + std::to_string(counted.found) +
                     " points: " + granted.message;
            return std::nullopt;
        }
        sized = std::move(granted.granted);
        buffer = &sized;
        const extraction_result written = tsdf.extract(sized, extraction_choice{}, stream);
        if (!written.done)
        {
            reason = "the extraction: " + written.message;
            return std::nullopt;
        }
    }
    std::vector<surface_point> on_host(counted.found);
    const cudaError_t copied =
        cudaMemcpyAsync(on_host.data(), buffer->points().data(),
                        on_host.size() * sizeof(surface_point), cudaMemcpyDeviceToHost, stream);
    const cudaError_t waited = copied == cudaSuccess ? cudaStreamSynchronize(stream) : copied;
    if (waited != cudaSuccess)
    {
        reason = cuda_failure("the copy of the full model to the host", waited);
        return std::nullopt;
    }
    std::vector<model_point> model;
    model.reserve(on_host.size());
    for (const surface_point &point : on_host)
        model.push_back(model_point_of(point));
    return model;
}
} // namespace

sequence_fusion_result fuse_sequence(const std::filesystem::path &sequence, const frame_size &size,
                                     const sparse_tsdf_parameters &full_model,
                                     const organized_points_parameters &range,
                                     const std::function<bool()> &go_on,
                                     cudaStream_t stream) noexcept
{
    sequence_fusion_result result;
    const std::string named = sequence.string() + ": ";
    sequence_reader reader(sequence);
    const open_result opened = reader.open(size);
    if (!opened.opened)
        return failed_with(result, opened.refusal);

    // Every buffer before the first frame.
    gpu_pool pool;
    request_result<sparse_tsdf> tsdf = sparse_tsdf::request(pool, full_model, stream);
    if (tsdf.status != request_status::granted)
        return failed_with(result, named + "the full model's sparse TSDF: " + tsdf.message);
    request_result<gpu_frame> uploaded = gpu_frame::request(pool, size);
    if (uploaded.status != request_status::granted)
        return failed_with(result, named + "the GPU frame: " + uploaded.message);
    request_result<organized_points> points = organized_points::request(pool, size);
    if (points.status != request_status::granted)
        return failed_with(result, named + "the organized points: " + points.message);
    request_result<surface_points> counter = surface_points::request(pool, 1);
    if (counter.status != request_status::granted)
        return failed_with(result, named + "the extraction's buffer: " + counter.message);

    frame host;
    while (true)
    {
        if (!go_on())
            return ended_with(result, sequence_fusion_end::cancelled, {});
        const next_result next = reader.next(host);
        if (next.status == next_status::end)
            break;
        if (next.status != next_status::frame)
        {
            // A sequence whose writer was cut off: every whole frame before the cut has come.
            if (reader.ended_early())
            {
                result.ended_early = true;
                break;
            }
            return failed_with(result, next.message);
        }
        const std::string frame_named = named + "frame " + std::to_string(host.index());
        const frame_tracking &tracking = reader.tracking();
        if (std::holds_alternative<not_tracked>(tracking))
            return failed_with(result, frame_named +
                                           " is marked as tracked by nothing: refinement fuses a "
                                           "scan's sequence, each of whose frames has a pose or "
                                           "the mark of tracking loss");
        const camera_pose *pose = std::get_if<camera_pose>(&tracking);
        result.trajectory.push_back(
            {host.index(), host.timestamp_us(),
             pose != nullptr ? std::optional<camera_pose>(*pose) : std::nullopt});
        if (pose == nullptr)
        {
            result.frames_skipped++;
            continue;
        }
        step_result made = uploaded.granted.upload(host, stream);
        if (!made.done)
            return failed_with(result, frame_named + ": the upload: " + made.message);
        made = points.granted.deproject(uploaded.granted, range, stream);
        if (!made.done)
            return failed_with(result, frame_named + ": the organized points: " + made.message);
        made = tsdf.granted.integrate(uploaded.granted, points.granted, *pose, stream);
        if (!made.done)
            return failed_with(result, frame_named + ": the fusion: " + made.message);
        result.frames_fused++;
    }

    std::string reason = used_up_refusal(tsdf.granted, full_model.memory_budget_bytes, stream);
    if (!reason.empty())
        return failed_with(result, named + reason);
    std::optional<std::vector<model_point>> model =
        extracted(tsdf.granted, pool, counter.granted, stream, reason);
    if (!model)
        return failed_with(result, named + reason);
    result.model = std::move(*model);
    result.end = sequence_fusion_end::done;
    return result;
}
} // namespace gca
