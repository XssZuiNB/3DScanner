// SPDX-License-Identifier: 0BSD
/// @file separation_messages.cpp
/// The separation library's words for a number and a pose's fault (see separation_messages.hpp).

#include "separation/separation_messages.hpp"

#include <array>
#include <cstdio>

namespace gca
{
std::string text_of(double value)
{
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), "%.6g", value);
    return text.data();
}

std::string fault_text(const camera_pose &pose)
{
    switch (fault_of(pose))
    {
    case pose_fault::none:
        return "";
    case pose_fault::position_not_finite:
        return "its position is not finite";
    case pose_fault::norm_not_one:
        break;
    }
    return "its quaternion's norm differs from 1 by " + text_of(pose.orientation.norm() - 1.0) +
           ", more than " + text_of(pose_norm_tolerance);
}
} // namespace gca
