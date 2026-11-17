// SPDX-License-Identifier: 0BSD
/// @file tracker.cpp
/// Tracking: the check of a registration's result and the tracker's path of a frame (see
/// tracker.hpp, docs/methods/tracking.md).

#include "tracking/tracker.hpp"

#include "geometry/organized_normals.hpp"
#include "geometry/organized_points.hpp"
#include "tracking/stage_clock.hpp"

#include <cmath>
#include <numbers>
#include <string>
#include <string_view>
#include <utility>

namespace gca
{
namespace
{
/// The microseconds of a second.
constexpr double microseconds_per_second = 1e6;

/// Says whether a value is finite and > 0.
/// @param value [in] the value.
/// @return whether it is.
bool positive(double value) noexcept
{
    return std::isfinite(value) && value > 0.0;
}

/// Writes a frame's size for a reason.
/// @param size [in] the size.
/// @return "<width>x<height> px".
std::string size_text(const frame_size &size)
{
    return std::to_string(size.width) + "x" + std::to_string(size.height) + " px";
}

/// Composes the last tracked pose with a registration's transform: P = P_L T, its rotation the
/// unit quaternion q_L q_T, its position t_L + R_L t_T, in double.
/// @param last [in] P_L, the camera frame at the last tracked pose into the models' frame.
/// @param transform [in] T, the frame's camera frame into the camera frame at P_L, rigid.
/// @return P, the frame's camera frame into the models' frame, its quaternion normalized.
camera_pose composed(const camera_pose &last, const Eigen::Isometry3d &transform) noexcept
{
    const Eigen::Quaterniond turn(transform.linear());
    return {last.position_m + last.orientation * transform.translation(),
            (last.orientation * turn).normalized()};
}
} // namespace

std::string_view name_of(tracking_state state) noexcept
{
    switch (state)
    {
    case tracking_state::tracking:
        return "tracking";
    case tracking_state::lost:
        return "lost";
    case tracking_state::failed:
        return "failed";
    }
    return {};
}

std::string_view name_of(check_outcome outcome) noexcept
{
    switch (outcome)
    {
    case check_outcome::passed:
        return "passed";
    case check_outcome::registration_end:
        return "registration_end";
    case check_outcome::fitness:
        return "fitness";
    case check_outcome::inlier_rmse:
        return "inlier_rmse";
    case check_outcome::step:
        return "step";
    }
    return {};
}

tracking_step step_of(const Eigen::Isometry3d &transform) noexcept
{
    const Eigen::AngleAxisd turn(transform.linear());
    return {std::abs(turn.angle()) * 180.0 / std::numbers::pi, transform.translation().norm()};
}

check_outcome check_registration(const registration_result &result,
                                 const std::optional<double> &interval_s,
                                 const tracking_thresholds &thresholds) noexcept
{
    // Both ends leave a last system whose smallest eigenvalue is at least τ (the user's decision
    // of 2026-11-15, evening; docs/methods/tracking.md, "The check").
    if (result.end != registration_end::converged &&
        result.end != registration_end::iterations_used_up)
        return check_outcome::registration_end;
    // Written so that a value that is not a number fails its condition.
    if (!(result.fitness >= thresholds.min_fitness))
        return check_outcome::fitness;
    if (!(result.inlier_rmse_m <= thresholds.max_inlier_rmse_m))
        return check_outcome::inlier_rmse;
    if (interval_s)
    {
        const tracking_step step = step_of(result.transform);
        if (!(step.turn_deg <= thresholds.max_turn_rate_deg_s * *interval_s) ||
            !(step.shift_m <= thresholds.max_speed_m_s * *interval_s))
            return check_outcome::step;
    }
    return check_outcome::passed;
}

surface_levels surface_of(const model_surface &surface) noexcept
{
    const auto level = [&surface](std::size_t l) {
        return organized_surface{surface.points(l), surface.validity_mask(l), surface.normals(l),
                                 surface.intensity(l), surface.level_intrinsics(l)};
    };
    return {level(0), level(1), level(2)};
}

std::string tracker::parameters_refusal(const tracking_parameters &parameters)
{
    std::string reason = organized_points::range_refusal(parameters.pyramid.range);
    if (!reason.empty())
        return "the depth range: " + reason;
    reason = organized_normals::parameters_refusal(parameters.pyramid.normals);
    if (!reason.empty())
        return "the normals: " + reason;
    reason = registration::parameters_refusal(parameters.registration);
    if (!reason.empty())
        return "the registration: " + reason;
    const tracking_thresholds &t = parameters.thresholds;
    if (!(positive(t.min_fitness) && t.min_fitness <= 1.0))
        return "the least fitness " + std::to_string(t.min_fitness) + " is not within (0, 1]";
    if (!positive(t.max_inlier_rmse_m))
        return "the largest inlier RMSE " + std::to_string(t.max_inlier_rmse_m) +
               " m is not a finite number > 0";
    if (!positive(t.max_turn_rate_deg_s))
        return "the largest turn rate " + std::to_string(t.max_turn_rate_deg_s) +
               " deg/s is not a finite number > 0";
    if (!positive(t.max_speed_m_s))
        return "the largest speed " + std::to_string(t.max_speed_m_s) +
               " m/s is not a finite number > 0";
    return {};
}

request_result<tracker> tracker::request(gpu_pool &pool, const frame_size &size,
                                         const tracking_parameters &parameters,
                                         const sparse_tsdf_parameters &model,
                                         cudaStream_t stream) noexcept
{
    request_result<tracker> result;
    if (size.width == 0 || size.height == 0)
    {
        result.message = "the frames' size " + size_text(size) + " is not > 0 along each side";
        return result;
    }
    result.message = parameters_refusal(parameters);
    if (!result.message.empty())
        return result;

    // The live model first: its request checks its parameters before it allocates, so a closed
    // pool checks them without device memory.
    request_result<sparse_tsdf> live_model = sparse_tsdf::request(pool, model, stream);
    if (live_model.status != request_status::granted)
        return refused_as<tracker>(live_model);
    request_result<gpu_frame> frame = gpu_frame::request(pool, size);
    if (frame.status != request_status::granted)
        return refused_as<tracker>(frame);
    request_result<frame_pyramid> pyramid = frame_pyramid::request(pool, size);
    if (pyramid.status != request_status::granted)
        return refused_as<tracker>(pyramid);
    request_result<model_surface> surface = model_surface::request(pool, size);
    if (surface.status != request_status::granted)
        return refused_as<tracker>(surface);
    request_result<registration> aligner = registration::request(pool, size);
    if (aligner.status != request_status::granted)
        return refused_as<tracker>(aligner);

    tracker &made = result.granted;
    made.m_parameters = parameters;
    made.m_size = size;
    made.m_model = std::move(live_model.granted);
    made.m_frame = std::move(frame.granted);
    made.m_pyramid = std::move(pyramid.granted);
    made.m_surface = std::move(surface.granted);
    made.m_registration = std::move(aligner.granted);
    result.status = request_status::granted;
    return result;
}

tracking_result tracker::track(const frame &frame, cudaStream_t stream, stage_clock *clock) noexcept
{
    tracking_result result;
    const auto fail = [&result](std::string message) {
        result.state = tracking_state::failed;
        result.message = std::move(message);
        return result;
    };
    // The measurement's wait after a stage that queues work on the GPU, with a stage clock alone.
    const auto timed = [clock, stream](path_stage stage) {
        return clock == nullptr ? step_result{true, {}} : clock->mark_after(stage, stream);
    };
    if (m_frame.color().empty())
        return fail("the tracker holds no buffers");
    if (!(frame.description().size == m_size))
        return fail("the frame's size " + size_text(frame.description().size) +
                    " is not the tracker's " + size_text(m_size));
    const std::uint64_t time_us = frame.timestamp_us();
    if (m_last_frame_us && time_us <= *m_last_frame_us)
        return fail("the frame's timestamp " + std::to_string(time_us) +
                    " us is not later than that of the frame before it, " +
                    std::to_string(*m_last_frame_us) + " us");
    m_last_frame_us = time_us;

    step_result made = m_frame.upload(frame, stream);
    if (made.done)
        made = timed(path_stage::upload);
    if (!made.done)
        return fail("the upload: " + made.message);
    made = m_pyramid.build(m_frame, m_parameters.pyramid, stream);
    if (made.done)
        made = timed(path_stage::pyramid);
    if (!made.done)
        return fail("the pyramid: " + made.message);
    // The surface as P_L sees it was cast after the last fusion; while the model is empty, the
    // frame against its own pyramid (docs/methods/tracking.md, "The models' frame and the first
    // frame").
    const surface_levels surface =
        m_model_holds_frame ? surface_of(m_surface) : surface_of(m_pyramid);
    result.registration = m_registration.align(m_pyramid, surface, Eigen::Isometry3d::Identity(),
                                               m_parameters.registration, stream);
    if (result.registration.end == registration_end::refused ||
        result.registration.end == registration_end::failed)
        return fail("the registration: " + result.registration.message);
    made = timed(path_stage::registration);
    if (!made.done)
        return fail("the registration: " + made.message);
    result.step = step_of(result.registration.transform);
    if (m_model_holds_frame)
        result.interval_s =
            static_cast<double>(time_us - m_last_tracked_us) / microseconds_per_second;

    result.outcome =
        check_registration(result.registration, result.interval_s, m_parameters.thresholds);
    if (clock != nullptr)
        clock->mark(path_stage::check);
    if (result.outcome != check_outcome::passed)
    {
        result.state = tracking_state::lost;
        return result;
    }
    // The first fused frame's camera frame is the models' frame: its pose is the identity.
    const camera_pose pose =
        m_model_holds_frame ? composed(m_last_pose, result.registration.transform)
                            : camera_pose{Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity()};
    made = m_model.integrate(m_frame, m_pyramid.points(0), pose, stream);
    if (made.done)
        made = timed(path_stage::fusion);
    if (!made.done)
        return fail("the fusion: " + made.message);
    m_model_holds_frame = true;
    m_last_pose = pose;
    m_last_tracked_us = time_us;
    // The surface for the next frame, cast now that the model and P_L have changed: the cast that
    // frame would make, and the live view of this one (docs/methods/tracking.md, "The path of a
    // frame", step 3).
    made = m_model.ray_cast(m_last_pose, m_frame.description().intrinsics,
                            m_parameters.pyramid.range, m_surface, stream);
    if (made.done)
        made = timed(path_stage::ray_cast);
    if (!made.done)
        return fail("the ray cast: " + made.message);
    result.state = tracking_state::tracking;
    result.pose = pose;
    return result;
}

const sparse_tsdf &tracker::live_model() const noexcept
{
    return m_model;
}

const model_surface &tracker::surface() const noexcept
{
    return m_surface;
}
} // namespace gca
