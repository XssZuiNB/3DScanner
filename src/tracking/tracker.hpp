// SPDX-License-Identifier: 0BSD
/// @file tracker.hpp
/// Tracking (docs/methods/tracking.md): the per-frame path's steps 1 to 6 as one call for each
/// frame of a scan. A frame is uploaded, its pyramid built, the live model's surface ray cast from
/// the last tracked pose, the frame registered against that surface from the identity, the
/// registration's result checked, and the frame fused at its pose only when the check passes; each
/// frame comes back with its tracking state as a value: tracking with its pose, tracking loss with
/// the first condition of the check that failed, or the tracker's failure with the reason. Until a
/// frame is fused the model is empty, and a frame is registered against its own pyramid, so that
/// a first view that does not hold every motion starts no model. The check is a host function of
/// the registration's result alone, check_registration(). The tracker holds every buffer of the
/// path and the live model's sparse TSDF, all requested from the pool that closes before the
/// first frame; track() allocates nothing, launches no kernel of its own and waits only where the
/// upload and the registration wait, and, for the replay tool's measurement alone, where a stage
/// clock waits after each stage (tracking/stage_clock.hpp).

#pragma once

#include "camera/frame.hpp"
#include "fusion/sparse_tsdf.hpp"
#include "geometry/frame_pyramid.hpp"
#include "gpu/gpu_frame.hpp"
#include "gpu/gpu_pool.hpp"
#include "registration/registration.hpp"
#include "types/camera_pose.hpp"
#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <Eigen/Geometry>

namespace gca
{
class stage_clock;

/// The check's thresholds (docs/methods/tracking.md, "Parameters"), each the object class's
/// (scan/object_class.hpp); the tracker refuses a value outside its range.
struct tracking_thresholds
{
    double min_fitness;         ///< f_track, the least fitness, in (0, 1]
    double max_inlier_rmse_m;   ///< e_track, the largest inlier RMSE, in m, finite and > 0
    double max_turn_rate_deg_s; ///< ω_max, the hand's largest turn rate, in °/s, finite and > 0
    double max_speed_m_s;       ///< v_max, the hand's largest speed, in m/s, finite and > 0
};

/// Tracking's parameters (docs/methods/tracking.md, "Input"): the pyramid's, whose depth range
/// the ray cast takes too, the registration's and the check's thresholds. The live model's sparse
/// TSDF takes its own (fusion/sparse_tsdf.hpp).
struct tracking_parameters
{
    frame_pyramid_parameters pyramid;     ///< the depth range and the normals' parameters
    registration_parameters registration; ///< each level's d_max and most iterations, δ_G, λ
    tracking_thresholds thresholds;       ///< the check's thresholds
};

/// The tracking state of a frame (docs/methods/tracking.md, "Output").
enum class tracking_state
{
    tracking, ///< the check passed; the frame was fused at its pose
    lost,     ///< tracking loss: a condition of the check failed; nothing was fused
    failed,   ///< the tracker refused the frame or its state, or CUDA failed; the message says why
};

/// What the check found: passed, or the first of its conditions that failed, in the check's order
/// (docs/methods/tracking.md, "The check").
enum class check_outcome
{
    passed,           ///< every condition held
    registration_end, ///< the registration ended neither converged nor with its iterations used up
    fitness,          ///< the fitness lay below f_track
    inlier_rmse,      ///< the inlier RMSE lay above e_track
    step,             ///< the step's turn or shift lay beyond what a hand makes in Δt
};

/// Gives a tracking state's name, as the replay's time lines and the web layer's state print it.
/// @param state [in] the state, one of the enumerators.
/// @return "tracking", "lost" or "failed"; a static string; empty for a value that is no
///         enumerator.
std::string_view name_of(tracking_state state) noexcept;

/// Gives a check's outcome's name, as the web layer's state prints it.
/// @param outcome [in] the outcome, one of the enumerators.
/// @return "passed", "registration_end", "fitness", "inlier_rmse" or "step"; a static string;
///         empty for a value that is no enumerator.
std::string_view name_of(check_outcome outcome) noexcept;

/// The step of a registration's transform T, the frame's camera frame into the camera frame at the
/// last tracked pose (docs/methods/tracking.md, "The step a hand can make").
struct tracking_step
{
    double turn_deg = 0.0; ///< θ, the angle of T's rotation, in degrees, 0 to 180
    double shift_m = 0.0;  ///< s = |t_T|, the distance the camera's centre moved, in m
};

/// Gives the step of a transform.
/// @param transform [in] T, rigid, in m.
/// @return the angle of its rotation, in degrees, and the length of its translation, in m.
/// @note Host only; allocates nothing.
tracking_step step_of(const Eigen::Isometry3d &transform) noexcept;

/// Checks a registration's result (docs/methods/tracking.md, "The check"): in this order, that
/// the registration ended converged or with its iterations used up, that its fitness is at least
/// f_track, that its inlier RMSE is at most e_track, and, when there is a last tracked frame, that
/// the step of its transform turns by at most ω_max Δt and shifts by at most v_max Δt.
/// @param result [in] the registration's result; a refused or failed one ends neither converged
///        nor with its iterations used up, and fails the first condition.
/// @param interval_s [in] Δt, the time from the last tracked frame to this one, in s, > 0; none
///        when no frame has been tracked, and then the step is not judged.
/// @param thresholds [in] the thresholds, each in its range.
/// @return passed, or the first condition that failed. A fitness, an RMSE or a step that is not a
///         number fails its condition.
/// @note Host only; allocates nothing and waits for nothing.
check_outcome check_registration(const registration_result &result,
                                 const std::optional<double> &interval_s,
                                 const tracking_thresholds &thresholds) noexcept;

/// Gives the live model's surface as the registration takes it: each level's points, marks,
/// normals, intensity and intrinsics, as the last ray cast wrote them. The one copy of this view;
/// the frame's pyramid's is the registration's (registration/registration.hpp).
/// @param surface [in] the ray cast's surface (fusion/sparse_tsdf.hpp); it holds the images, which
///        must outlive the view.
/// @return its levels 0, 1 and 2, each viewing the surface's images.
/// @note Host only; allocates nothing and waits for nothing.
surface_levels surface_of(const model_surface &surface) noexcept;

/// What tracking gave for a frame: its state, with its pose when tracking, the cause when lost and
/// the reason when failed, and the evidence the check judged (docs/methods/tracking.md, "Output").
struct tracking_result
{
    tracking_state state = tracking_state::failed; ///< tracking, lost or failed
    /// The check's outcome: passed when tracking, the first condition that failed when lost;
    /// passed when failed.
    check_outcome outcome = check_outcome::passed;
    std::string message; ///< why it failed, in plain English; empty when tracking or lost
    /// The frame's pose, its camera frame into the models' frame (types/camera_pose.hpp), the pose
    /// it was fused at; present only when tracking.
    std::optional<camera_pose> pose;
    /// The registration's result: its end, transform T, fitness, inlier RMSE, iterations,
    /// eigenvalues and last step (registration/registration.hpp); ended refused, with nothing
    /// else, when the frame failed before the registration.
    registration_result registration;
    tracking_step step; ///< the step of T; zeros when the frame failed before the registration
    /// Δt, the time from the last tracked frame to this one, in s; none while the model is empty
    /// or when the frame failed before the registration.
    std::optional<double> interval_s;
};

/// The tracker: the buffers of the per-frame path for frames of one size, the live model's sparse
/// TSDF, the last tracked pose and its time. Move-only. Not synchronized: one thread at a time
/// uses one, and the work it queues goes to one stream, the stream it was requested with or one
/// ordered after it.
class tracker
{
public:
    /// An empty tracker: no buffers; track() with it fails.
    tracker() noexcept = default;

    /// Checks tracking's parameters as request() does before it takes memory: the depth range
    /// (organized_points::range_refusal()), the normals' parameters
    /// (organized_normals::parameters_refusal()), the registration's
    /// (registration::parameters_refusal()) and the thresholds' ranges (tracking_thresholds).
    /// @param parameters [in] the parameters.
    /// @return empty when each is usable; else the reason, naming the value.
    /// @note Allocates for the reason; throws std::bad_alloc when that fails. Waits for nothing.
    static std::string parameters_refusal(const tracking_parameters &parameters);

    /// Requests every buffer of the path from the pool, for frames of one size: first the live
    /// model's sparse TSDF, whose clearing it queues on stream, so that a closed pool checks every
    /// parameter without device memory (sparse_tsdf::request()); then the GPU frame, the frame's
    /// pyramid, the ray cast's surface and the registration's buffers. The model is empty and no
    /// frame is tracked.
    /// @param pool [in,out] the pool; open, else the request is refused as closed.
    /// @param size [in] the frames' size, in px, each > 0.
    /// @param parameters [in] tracking's parameters, each in its range (parameters_refusal()).
    /// @param model [in] the live model's parameters: the memory budget, the voxel size, the
    ///        truncation's rule and the far depth, each in its range (sparse_tsdf::request()).
    /// @param stream [in] the CUDA stream of the current device the model's clearing is queued on;
    ///        owned by the caller. track() queues its work on it or on a stream ordered after it.
    /// @return granted, with the tracker; failure with the reason, before any request, when the
    ///         size is 0 or parameters_refusal() refuses the parameters; else the first of the
    ///         requests that was refused, with its status and message: the sparse TSDF's checks of
    ///         its parameters among them, and closed for a closed pool.
    /// @note Allocates device memory (cudaMalloc, the sparse TSDF's budget and 25.9 MB at 640x400,
    ///       103.5 MB at 1280x800) and pinned host memory (cudaMallocHost, 1.28 MB and 5.12 MB)
    ///       [S: computed, docs/methods/tracking.md, "The path of a frame"], so it is called before
    ///       the pool is closed; whether CUDA's allocators wait for the GPU is not stated by CUDA
    ///       [?]. Allocates for the message when refused; a failed allocation ends the program
    ///       (std::terminate): the function is noexcept. Two threads may not request from one pool
    ///       at once.
    static request_result<tracker> request(gpu_pool &pool, const frame_size &size,
                                           const tracking_parameters &parameters,
                                           const sparse_tsdf_parameters &model,
                                           cudaStream_t stream) noexcept;

    /// A tracker is not copied: a copy allocates.
    tracker(const tracker &) = delete;
    tracker &operator=(const tracker &) = delete;

    /// Takes over another tracker's buffers and state without allocating.
    /// @param other [in,out] the tracker moved from; left empty.
    /// @note Allocates nothing and waits for nothing.
    tracker(tracker &&other) noexcept = default;

    /// Takes over another tracker's buffers and state; this one's are freed first.
    /// @param other [in,out] the tracker moved from; left empty. Moving onto itself changes
    ///        nothing.
    /// @return this tracker.
    /// @pre No work queued on the GPU uses this one's buffers any more.
    /// @note Frees with cudaFree and cudaFreeHost, which may wait for the GPU, and waits for the
    ///       GPU frame's last upload (gpu/gpu_frame.hpp). Allocates nothing.
    tracker &operator=(tracker &&other) noexcept = default;

    /// Frees the buffers.
    /// @pre No work queued on the GPU uses them any more.
    /// @note Frees as the move assignment does.
    ~tracker() = default;

    /// Tracks a frame (docs/methods/tracking.md, "The path of a frame"): uploads it, builds its
    /// pyramid, registers the pyramid from the identity against the surface, the live model as the
    /// last tracked pose P_L sees it, cast after the last fusion, and checks the result, Δt counted
    /// from the last tracked frame; while the model is empty, registers the pyramid against itself
    /// and checks the result without a step. When the check passes, fuses the frame at P = P_L T
    /// (the identity while the model is empty), composed in double with its rotation kept as a
    /// unit quaternion; P becomes the last tracked pose, the frame's timestamp the last tracked
    /// time, and the live model is ray cast from P at the frame's intrinsics and the pyramid's
    /// depth range into the surface, for the next frame and the live view (surface()). When it
    /// fails, nothing is fused or cast and the last tracked pose and time stay.
    /// @param frame [in] the frame, in host memory, of the tracker's size, its timestamp later
    ///        than that of the frame before, as a frame source delivers them
    ///        (camera/frame_source.hpp); owned by the caller and read only during the call.
    /// @param stream [in] the CUDA stream of the current device the work is queued on, the
    ///        request's or one ordered after it; owned by the caller.
    /// @param clock [in,out] the stage clock that times the stages (tracking/stage_clock.hpp), for
    ///        the replay tool's measurement only; owned by the caller. nullptr, the default and a
    ///        live scan's: nothing is timed and nothing more is waited for.
    /// @return tracking, with the pose; lost, with the condition that failed; each with the
    ///         registration's result, the step and Δt. Failed, with the reason and nothing fused or
    ///         changed, when the tracker is empty, the frame's size is not the tracker's, or its
    ///         timestamp is not later than that of the frame before it; failed with the step's
    ///         reason when the upload, the pyramid or the registration is refused or fails, CUDA's
    ///         error then cleared, with nothing fused; failed with the fusion's reason when the
    ///         fusion is refused or fails, which leaves what sparse_tsdf::integrate() says, and the
    ///         last tracked pose as it was; failed with the ray cast's reason when the cast after
    ///         the fusion is refused or fails, which leaves the frame fused, P the last tracked
    ///         pose and the surface unspecified. A frame that passes the timestamp's check
    ///         counts as the frame before for the next call, whatever follows.
    /// @note Queues the upload's copies, the pyramid's seven kernels, the registration's kernels
    ///       and copies, and when the check passes the fusion's memset and two kernels and the ray
    ///       cast's clearing and two kernels, on stream. Waits where the upload waits for its
    ///       staging buffer and where the registration waits once per iteration
    ///       (cudaStreamSynchronize, which also waits for the stream's earlier work, the frame
    ///       before's fusion and cast included); the fusion and the cast are queued and not waited
    ///       for. With a stage clock it also waits for the stream after the upload, the pyramid,
    ///       the registration, the fusion and the ray cast (stage_clock::mark_after()), the
    ///       measurement's waits,
    ///       which a live scan must not have; a failed wait fails the frame with CUDA's error,
    ///       naming the stage. Reads back only the registration's 280 bytes per iteration.
    ///       Allocates nothing when tracking or lost; allocates for the message when failed, and a
    ///       failed allocation ends the program (std::terminate): the function is noexcept. Not
    ///       while another thread uses this tracker.
    tracking_result track(const frame &frame, cudaStream_t stream,
                          stage_clock *clock = nullptr) noexcept;

    /// Gives the live model, for its extraction or its usage (fusion/sparse_tsdf.hpp).
    /// @return the sparse TSDF the frames are fused into; empty when the tracker is. Valid as
    ///         long as the tracker holds it.
    const sparse_tsdf &live_model() const noexcept;

    /// Gives the live model's surface as the last tracked pose sees it, cast after the last
    /// fusion: the next frame's surface, and the live view's picture with its colors at level 0
    /// (docs/methods/tracking.md, "The live view").
    /// @return the surface; its images are unspecified until the first frame is fused, and after
    ///         a frame whose cast failed. Valid as long as the tracker holds it; the next track()
    ///         writes it when it fuses.
    /// @note Its images are written by work queued on track()'s stream: work that reads them is
    ///       queued on that stream, or one ordered after it.
    const model_surface &surface() const noexcept;

private:
    tracking_parameters m_parameters{}; ///< the parameters from the request
    frame_size m_size{0, 0};            ///< the frames' size, from the request
    sparse_tsdf m_model;                ///< the live model
    gpu_frame m_frame;                  ///< the frame uploaded last
    frame_pyramid m_pyramid;            ///< its pyramid
    model_surface m_surface;            ///< the live model's surface as P_L sees it
    registration m_registration;        ///< the registration's buffers
    bool m_model_holds_frame = false;   ///< whether a frame has been fused
    camera_pose m_last_pose{Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity()}; ///< P_L
    std::uint64_t m_last_tracked_us = 0; ///< the last tracked frame's timestamp, in µs
    /// The timestamp of the frame before, in µs: the last frame that passed the timestamp's
    /// check; none before the first.
    std::optional<std::uint64_t> m_last_frame_us;
};
} // namespace gca
