// SPDX-License-Identifier: 0BSD
/// @file gravity.hpp
/// Gravity's up from the camera's accelerometer (docs/methods/separation.md, "Gravity's up"): each
/// accelerometer sample between two tracked frames of the trajectory is turned into the camera
/// frame by the IMU's rotation and into the models' frame by the camera's orientation at its time,
/// interpolated between the two frames' orientations; up is the direction of the samples' mean,
/// in which the hand's acceleration averages out. The samples in groups of one second give the
/// spread that refuses a gravity it cannot trust; a scan without samples has no gravity. The
/// result is a value for every call, its outcome a key of a closed set that the scan's parameters
/// keep (types/gravity_estimate.hpp). The constants stand here, each [?: initial]. On the host in
/// double.

#pragma once

#include "types/gravity_estimate.hpp"
#include "types/imu_sample.hpp"
#include "types/trajectory_entry.hpp"

#include <cstddef>
#include <span>
#include <string>

#include <Eigen/Geometry>

namespace gca
{
/// The samples of one group of gravity's spread: 200, one second at the IMU's 200 Hz
/// [?: initial; docs/methods/separation.md, "The spread"]. Separation does not read the camera's
/// rate (camera/orbbec_imu.hpp, orbbec_imu_rate_hz); tests/test_separation.cpp holds the two
/// equal in a static_assert, so that a change of the rate fails its build.
inline constexpr std::size_t gravity_group_samples = 200;

/// The least groups the samples between tracked frames must fill: 5, five seconds, in which one
/// stretch of tracked frames leans up by at most 2.3 degrees [?: initial; S: computed,
/// docs/methods/separation.md, "The mean"].
inline constexpr std::size_t gravity_least_groups = 5;

/// The largest spread of the groups' means about their mean, in degrees: 10 [?: initial: 23 times
/// the 0.43 degrees of a hand-held recording (M 2026-11-16 recording, early), far below the plane's
/// 45 degrees; docs/methods/separation.md, "The spread"].
inline constexpr double gravity_largest_spread_deg = 10.0;

/// An estimate and, when it was refused, why.
struct gravity_result
{
    gravity_estimate estimate; ///< the estimate (types/gravity_estimate.hpp)
    /// Why the inputs were refused, naming the frame or the sample, in plain English; empty unless
    /// the outcome is refused.
    std::string refusal;
};

/// Estimates gravity's up in the models' frame (docs/methods/separation.md, "Gravity's up"). A
/// sample at t is taken when two consecutive frames of the trajectory enclose it, t_j <= t <=
/// t_{j+1}, both tracked, or when it lies at a tracked frame's own time; its rotation is the
/// spherical linear interpolation of their orientations at (t - t_j) / (t_{j+1} - t_j), the
/// shorter way. Each taken sample s = R(t) Q f; the mean of the s; the spread, the root mean square
/// of the angles atan2(|m x mean|, m · mean) of the means m of consecutive groups of
/// gravity_group_samples taken samples, the last group of fewer left out.
/// @param trajectory [in] the frames the scanner took, in the order taken
/// (types/trajectory_entry.hpp):
///        timestamps in µs on the frames' clock, strictly rising, else refused; a tracked frame's
///        pose camera to model, its position finite and its quaternion of norm 1 within
///        pose_norm_tolerance, else refused; frames in tracking loss without a pose. Host memory,
///        read during the call.
/// @param accelerometer [in] the accelerometer's samples of the scan's frames, in m/s² along the
///        IMU's axes, their timestamps on the frames' clock strictly rising and their values
///        finite, else refused; empty for a camera that opened without its IMU. Host memory, read
///        during the call.
/// @param imu_to_camera [in] Q, the rotation from the IMU's axes into the camera frame, the
///        camera's (types/imu_sample.hpp, imu_to_camera()), its norm within pose_norm_tolerance of
///        1, else refused.
/// @return the estimate: estimated, with up (unit, models' frame), the samples taken, the groups,
///         the spread in degrees and the mean's length in m/s²; no_samples for an empty
///         accelerometer; too_few_samples when the taken samples fill fewer than
///         gravity_least_groups groups; mean_zero when their mean is 0; spread_too_large when the
///         spread exceeds gravity_largest_spread_deg; refused, with the refusal naming the input,
///         before anything is computed. Up is zero unless estimated.
/// @note Host only; allocates the taken samples, 24 bytes each, and throws std::bad_alloc when the
///       allocation fails. Holds no state: two threads may estimate at once.
gravity_result estimate_gravity(std::span<const trajectory_entry> trajectory,
                                std::span<const imu_sample> accelerometer,
                                const Eigen::Quaterniond &imu_to_camera);

/// Says in plain English why an estimate gives no up, for a proposal's message and the page.
/// @param estimate [in] the estimate, as estimate_gravity() gives it or the scan's parameters hold
///        it.
/// @return the reason, naming the counts or the spread where they say it; empty when estimated.
/// @note Allocates the text.
std::string reason_of(const gravity_estimate &estimate);
} // namespace gca
