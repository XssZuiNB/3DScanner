// SPDX-License-Identifier: 0BSD
/// @file sequence_fusion.hpp
/// Refinement's fusion of a scan's sequence at the full voxel size (docs/methods/refinement.md,
/// "The fusion of the sequence"): the sequence is read back in order, each frame tracking gave a
/// pose is fused at that pose into a sparse TSDF of the full model's parameters, a frame of
/// tracking loss is skipped, and the full model is extracted and copied to the host as model
/// points (types/model_point.hpp), with the trajectory as the sequence holds it. A sequence that
/// holds a frame nothing tracked is refused, and so is a fusion whose pool was used up; one cut off
/// is fused up to its last whole frame; the caller can stop the fusion between two frames. A
/// failure is a value with its reason, and gives neither a model nor a trajectory.

#pragma once

#include "fusion/sparse_tsdf.hpp"
#include "geometry/organized_points.hpp"
#include "types/frame_description.hpp"
#include "types/model_point.hpp"
#include "types/trajectory_entry.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace gca
{
/// How the fusion of a sequence ended.
enum class sequence_fusion_end
{
    done,      ///< every whole frame was read and the full model extracted
    cancelled, ///< the caller stopped it between two frames; no model and no trajectory
    failed,    ///< the sequence, a step or CUDA failed, or the pool was used up; the message says
               ///< why; no model and no trajectory
};

/// What the fusion of a sequence gave.
struct sequence_fusion_result
{
    sequence_fusion_end end = sequence_fusion_end::failed; ///< done, cancelled or failed
    std::string message; ///< why it failed, in plain English; empty when done or cancelled
    /// The full model's points, in the poses' frame, in no stated order; filled only when done,
    /// and empty then when no frame was fused.
    std::vector<model_point> model;
    /// The frames read, in the sequence's order: each one's index, timestamp and pose, or none
    /// for tracking loss, as the sequence holds them, bit for bit; filled only when done, so that
    /// no caller keeps a part of it.
    std::vector<trajectory_entry> trajectory;
    std::uint64_t frames_fused = 0;   ///< the frames fused at their poses, up to the end
    std::uint64_t frames_skipped = 0; ///< the frames of tracking loss, not fused, up to the end
    /// Whether the sequence ended early, cut off within or after a frame's record as a writer
    /// that was stopped leaves it (camera/sequence_reader.hpp, ended_early()); its whole frames
    /// were fused.
    bool ended_early = false;
};

/// Fuses a scan's sequence at the full voxel size and extracts the full model
/// (docs/methods/refinement.md, "The fusion of the sequence"). The buffers are requested from a
/// pool of the call's own: the sparse TSDF of the parameters given, the GPU frame and the
/// organized points of the stream's size, and the points' buffer, once of one point to count the
/// surface's points and then of that count. For each frame in the sequence's order, go_on is
/// asked first; a frame with a pose is uploaded, its organized points are made within the depth
/// range and it is integrated at its pose; a frame of tracking loss is read and skipped. After the
/// last frame the pool's usage is read: a coordinate that found no block, or an insertion the hash
/// refused, fails the fusion, as the full model would lack surface.
/// @param sequence [in] the sequence's path (docs/SEQUENCE_FILE.md).
/// @param size [in] the size of its frames, the stream size of the scan, in px.
/// @param full_model [in] the full model's sparse TSDF: the memory budget, the full voxel size, the
///        truncation and the far depth (scan/stage_parameters.hpp, fused_model::full).
/// @param range [in] the organized points' depth range, the object class's, in m.
/// @param go_on [in] asked before each frame, on the calling thread; false stops the fusion,
///        which ends cancelled. It must not throw.
/// @param stream [in] the CUDA stream of the current device the work is queued on and waited for;
///        owned by the caller.
/// @return done, with the model, the trajectory and the counts; cancelled when go_on said false,
///         with the counts of the frames read before; failed with the reason, naming the sequence
///         and the frame where there is one: the sequence cannot be opened (the reader's reason:
///         no file, another version, another size), its reading fails other than by its early end,
///         a frame is marked as tracked by nothing, a request is refused (the sparse TSDF's
///         parameters, a budget the board cannot grant), an upload, a deprojection or an
///         integration is refused or fails, the pool was used up (the counts named), or the
///         reading of the usage, the extraction or the copy to the host fails.
/// @pre No other sparse TSDF exists: the live model's has been freed (docs/methods/sparse-tsdf.md,
///      "The pool and the memory budget": one at a time). The call does not check it: with the
///      live model's still held, the full model's request either fails as a value, when CUDA
///      cannot grant it, or takes a second budget of the memory the board's CPU and GPU share.
/// @note Allocates device memory, pinned host memory, and host memory for the trajectory and the
///       model: at its peak about 56 bytes a point on the host, the points as copied and as model
///       points, 28 bytes each, besides 28 bytes a point on the GPU; not a step of the per-frame
///       path. A failed host allocation ends the program (std::terminate): the function is
///       noexcept. Waits where the upload waits for its staging buffer, for the usage, for the
///       extraction twice, and for the points' copy. Not while another thread uses the stream.
sequence_fusion_result fuse_sequence(const std::filesystem::path &sequence, const frame_size &size,
                                     const sparse_tsdf_parameters &full_model,
                                     const organized_points_parameters &range,
                                     const std::function<bool()> &go_on,
                                     cudaStream_t stream) noexcept;
} // namespace gca
