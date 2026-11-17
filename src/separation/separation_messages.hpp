// SPDX-License-Identifier: 0BSD
/// @file separation_messages.hpp
/// The separation library's words in its refusals and reasons, which gravity's estimate
/// (separation/gravity.hpp) and the proposal (separation/proposal.hpp) share: a number's text, and
/// what is wrong with a pose (types/camera_pose.hpp, fault_of()). Host only.

#pragma once

#include "types/camera_pose.hpp"

#include <string>

namespace gca
{
/// Writes a number for a message, in up to six significant digits.
/// @param value [in] the number.
/// @return its text as printf's %.6g writes it, "nan" or "inf" included.
/// @note Allocates for the string; throws std::bad_alloc when that fails.
std::string text_of(double value);

/// Says what is wrong with a pose, as the caller's refusal goes on after the pose's name and ": ".
/// @param pose [in] the pose.
/// @return empty when fault_of() finds nothing; "its position is not finite"; or "its quaternion's
///         norm differs from 1 by <d>, more than <pose_norm_tolerance>", each number as text_of()
///         writes it.
/// @note Allocates for the string; throws std::bad_alloc when that fails.
std::string fault_text(const camera_pose &pose);
} // namespace gca
