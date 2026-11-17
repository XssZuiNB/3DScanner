// SPDX-License-Identifier: 0BSD
/// @file gravity_estimate.hpp
/// Gravity's up as a scan found it: the direction of the accelerometer's mean specific force in the
/// models' frame, with the samples, the groups and the spread it was judged by, or the outcome that
/// says why there is none (docs/methods/separation.md, "Gravity's up"). Separation estimates it
/// (separation/gravity.hpp) and proposes from it (separation/proposal.hpp); the scan's parameters
/// keep it (scan/parameters_file.hpp, docs/SCAN_FILES.md). A plain struct, the check of an outcome
/// and the one check of its up, header only.

#pragma once

#include <cmath>
#include <cstdint>

#include <Eigen/Core>

namespace gca
{
/// How a gravity estimate ended, a closed set whose keys the scan's parameters hold. The
/// enumerators run from 0 without a gap.
enum class gravity_outcome
{
    not_estimated,    ///< no estimate was made, as for a scan not yet refined
    estimated,        ///< up holds gravity's up
    no_samples,       ///< no accelerometer sample: the camera opened without its IMU
    too_few_samples,  ///< the samples between tracked frames fill too few groups
    mean_zero,        ///< the samples' mean is 0: no direction
    spread_too_large, ///< the groups' means spread wider than the limit
    refused,          ///< the trajectory, the samples or the rotation were refused
};

/// How far the length of an estimated up may lie from 1: 1e-9 [?: a choice, as a pose's
/// quaternion (types/camera_pose.hpp): a normalization in double leaves a few 1e-16, and the
/// parameters' file keeps every bit].
inline constexpr double gravity_up_tolerance = 1e-9;

/// Gravity's up and its evidence.
struct gravity_estimate
{
    gravity_outcome outcome = gravity_outcome::not_estimated; ///< how it ended
    /// Up, the unit direction opposite gravity, in the models' frame, when estimated; zero
    /// otherwise.
    Eigen::Vector3d up = Eigen::Vector3d::Zero();
    /// The accelerometer's samples taken, those between two tracked frames or at a tracked frame's
    /// time; 0 when not estimated or without samples.
    std::uint64_t samples = 0;
    std::uint64_t groups = 0; ///< the groups of the spread the samples fill
    /// The root mean square of the angles between the groups' means and the mean, in degrees, >=
    /// 0; 0 where it was not computed (too few groups, a mean of 0, not estimated).
    double spread_deg = 0.0;
    /// The length of the samples' mean, in m/s², >= 0: gravity's g plus what is left of the hand's
    /// acceleration, in the accelerometer's scale; 0 without samples taken.
    double mean_mps2 = 0.0;

    /// Compares two estimates value by value.
    /// @param other [in] the other estimate.
    /// @return whether every value is equal.
    bool operator==(const gravity_estimate &other) const = default;
};

/// Says whether a value is one of the outcomes: one case for each, so -Wswitch names an
/// enumerator added without its case (tools/verify.sh fails on the warning).
/// @param outcome [in] the value.
/// @return whether it is an enumerator of gravity_outcome.
constexpr bool is_outcome(gravity_outcome outcome) noexcept
{
    switch (outcome)
    {
    case gravity_outcome::not_estimated:
    case gravity_outcome::estimated:
    case gravity_outcome::no_samples:
    case gravity_outcome::too_few_samples:
    case gravity_outcome::mean_zero:
    case gravity_outcome::spread_too_large:
    case gravity_outcome::refused:
        return true;
    }
    return false;
}

/// Checks an estimated up, as the proposal and the parameters' file take it.
/// @param up [in] the up, in the models' frame.
/// @return whether it is finite and its length lies within gravity_up_tolerance of 1.
/// @note Host only; allocates nothing.
inline bool up_is_unit(const Eigen::Vector3d &up) noexcept
{
    return up.allFinite() && std::abs(up.norm() - 1.0) <= gravity_up_tolerance;
}
} // namespace gca
