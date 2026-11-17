// SPDX-License-Identifier: 0BSD
/// @file gravity.cpp
/// Gravity's up from the accelerometer (see gravity.hpp and docs/methods/separation.md, "Gravity's
/// up"), in named steps: the checks of the inputs, the samples taken and turned into the models'
/// frame, their mean, the groups' spread and the judgement.

#include "separation/gravity.hpp"

#include "separation/separation_messages.hpp"
#include "types/camera_pose.hpp"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <vector>

namespace gca
{
namespace
{
/// Says why the inputs are refused (gravity.hpp, estimate_gravity()), before any is used.
/// @param trajectory [in] the frames the scanner took.
/// @param accelerometer [in] the accelerometer's samples.
/// @param imu_to_camera [in] Q.
/// @return the reason, naming Q, the frame or the sample; empty when every input is accepted.
std::string refusal_of(std::span<const trajectory_entry> trajectory,
                       std::span<const imu_sample> accelerometer,
                       const Eigen::Quaterniond &imu_to_camera)
{
    if (!(std::abs(imu_to_camera.norm() - 1.0) <= pose_norm_tolerance))
        return "the rotation from the IMU's axes into the camera frame: its quaternion's norm "
               "differs from 1 by " +
               text_of(imu_to_camera.norm() - 1.0) + ", more than " + text_of(pose_norm_tolerance);
    for (std::size_t j = 0; j < trajectory.size(); j++)
    {
        const std::string named = "frame " + std::to_string(j) + " of the trajectory";
        if (j > 0 && !(trajectory[j].timestamp_us > trajectory[j - 1].timestamp_us))
            return named + ": its timestamp " + std::to_string(trajectory[j].timestamp_us) +
                   " us does not follow the frame before's " +
                   std::to_string(trajectory[j - 1].timestamp_us) + " us";
        if (!trajectory[j].pose)
            continue;
        const std::string fault = fault_text(*trajectory[j].pose);
        if (!fault.empty())
            return named + ": " + fault;
    }
    for (std::size_t k = 0; k < accelerometer.size(); k++)
    {
        const imu_sample &sample = accelerometer[k];
        const std::string named = "accelerometer sample " + std::to_string(k);
        if (k > 0 && !(sample.timestamp_us > accelerometer[k - 1].timestamp_us))
            return named + ": its time " + std::to_string(sample.timestamp_us) +
                   " us does not follow the sample before's " +
                   std::to_string(accelerometer[k - 1].timestamp_us) + " us";
        if (!(std::isfinite(sample.x) && std::isfinite(sample.y) && std::isfinite(sample.z)))
            return named + ": its value is not finite";
    }
    return "";
}

/// Finds the camera's orientation at a sample's time (docs/methods/separation.md, "Into the
/// models' frame"): the orientation of a tracked frame at that time, or the spherical linear
/// interpolation between the two consecutive frames that enclose it, both tracked.
/// @param trajectory [in] the frames, timestamps strictly rising, each pose checked.
/// @param after [in,out] the index of the first frame whose timestamp is not below the time of the
///        sample before, or 0; moved on to that of this sample. Samples come in rising time.
/// @param time_us [in] the sample's time, in µs on the frames' clock.
/// @return the rotation from the camera frame into the models' frame; nothing when the sample lies
///         before the first frame or after the last, or next to a frame in tracking loss.
std::optional<Eigen::Matrix3d> orientation_at(std::span<const trajectory_entry> trajectory,
                                              std::size_t &after, std::uint64_t time_us)
{
    while (after < trajectory.size() && trajectory[after].timestamp_us < time_us)
        after++;
    if (after == trajectory.size())
        return std::nullopt;
    const trajectory_entry &next = trajectory[after];
    if (next.timestamp_us == time_us)
    {
        if (!next.pose)
            return std::nullopt;
        return next.pose->orientation.normalized().toRotationMatrix();
    }
    if (after == 0)
        return std::nullopt;
    const trajectory_entry &before = trajectory[after - 1];
    if (!before.pose || !next.pose)
        return std::nullopt;
    // t_j < time_us < t_{j+1}: the share lies in (0, 1).
    const double share = static_cast<double>(time_us - before.timestamp_us) /
                         static_cast<double>(next.timestamp_us - before.timestamp_us);
    // Eigen's slerp takes the shorter way, whatever the quaternions' signs.
    return before.pose->orientation.normalized()
        .slerp(share, next.pose->orientation.normalized())
        .toRotationMatrix();
}

/// Gives the angle between two vectors.
/// @param a [in] the first vector.
/// @param b [in] the second vector.
/// @return atan2(|a x b|, a · b), in rad: 0 to pi, 0 when either is zero.
double angle_between(const Eigen::Vector3d &a, const Eigen::Vector3d &b)
{
    return std::atan2(a.cross(b).norm(), a.dot(b));
}
} // namespace

gravity_result estimate_gravity(std::span<const trajectory_entry> trajectory,
                                std::span<const imu_sample> accelerometer,
                                const Eigen::Quaterniond &imu_to_camera)
{
    gravity_result result;
    result.refusal = refusal_of(trajectory, accelerometer, imu_to_camera);
    if (!result.refusal.empty())
    {
        result.estimate.outcome = gravity_outcome::refused;
        return result;
    }
    gravity_estimate &estimate = result.estimate;
    if (accelerometer.empty())
    {
        estimate.outcome = gravity_outcome::no_samples;
        return result;
    }

    // Each sample between tracked frames, s = R(t) Q f, in the models' frame.
    const Eigen::Matrix3d q = imu_to_camera.normalized().toRotationMatrix();
    std::vector<Eigen::Vector3d> taken;
    taken.reserve(accelerometer.size());
    std::size_t after = 0;
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const imu_sample &sample : accelerometer)
    {
        const std::optional<Eigen::Matrix3d> rotation =
            orientation_at(trajectory, after, sample.timestamp_us);
        if (!rotation)
            continue;
        taken.push_back(*rotation * (q * Eigen::Vector3d(sample.x, sample.y, sample.z)));
        sum += taken.back();
    }
    estimate.samples = taken.size();
    estimate.groups = taken.size() / gravity_group_samples;
    if (taken.empty())
    {
        estimate.outcome = gravity_outcome::too_few_samples;
        return result;
    }
    const Eigen::Vector3d mean = sum / static_cast<double>(taken.size());
    estimate.mean_mps2 = mean.norm();
    if (estimate.groups < gravity_least_groups)
    {
        estimate.outcome = gravity_outcome::too_few_samples;
        return result;
    }
    if (!(estimate.mean_mps2 > 0.0))
    {
        estimate.outcome = gravity_outcome::mean_zero;
        return result;
    }

    // The spread: the root mean square of the groups' means' angles from the mean.
    double squares = 0.0;
    for (std::size_t b = 0; b < estimate.groups; b++)
    {
        Eigen::Vector3d group = Eigen::Vector3d::Zero();
        for (std::size_t k = b * gravity_group_samples; k < (b + 1) * gravity_group_samples; k++)
            group += taken[k];
        const double angle = angle_between(group, mean);
        squares += angle * angle;
    }
    estimate.spread_deg =
        std::sqrt(squares / static_cast<double>(estimate.groups)) * 180.0 / std::numbers::pi;
    if (!(estimate.spread_deg <= gravity_largest_spread_deg))
    {
        estimate.outcome = gravity_outcome::spread_too_large;
        return result;
    }
    estimate.up = mean / estimate.mean_mps2;
    estimate.outcome = gravity_outcome::estimated;
    return result;
}

std::string reason_of(const gravity_estimate &estimate)
{
    switch (estimate.outcome)
    {
    case gravity_outcome::estimated:
        return "";
    case gravity_outcome::not_estimated:
        return "no gravity: none was estimated, as for a scan not yet refined";
    case gravity_outcome::no_samples:
        return "no gravity: the scan's frames carry no accelerometer sample, its camera having "
               "opened without its IMU";
    case gravity_outcome::too_few_samples:
        return "no gravity: " + std::to_string(estimate.samples) +
               " accelerometer samples lie between tracked frames, " +
               std::to_string(estimate.groups) + " groups of " +
               std::to_string(gravity_group_samples) + ", fewer than " +
               std::to_string(gravity_least_groups);
    case gravity_outcome::mean_zero:
        return "no gravity: the mean of the " + std::to_string(estimate.samples) +
               " accelerometer samples between tracked frames is 0";
    case gravity_outcome::spread_too_large:
        return "no gravity: the means of the accelerometer's " + std::to_string(estimate.groups) +
               " groups spread by " + text_of(estimate.spread_deg) +
               " degrees about their mean, more than " + text_of(gravity_largest_spread_deg);
    case gravity_outcome::refused:
        return "no gravity: its estimate refused the scan's trajectory, its samples or the IMU's "
               "rotation";
    }
    return "no gravity: the estimate's outcome " +
           std::to_string(static_cast<int>(estimate.outcome)) + " is no outcome";
}
} // namespace gca
