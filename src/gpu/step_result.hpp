// SPDX-License-Identifier: 0BSD
/// @file step_result.hpp
/// What a step on the GPU gave: done, or why not, as a value (CODING_STANDARDS.md, "The
/// rewrite's code": a failure is a value). A frame's upload and download (gpu/gpu_frame.hpp) and
/// the stages of the per-frame path, such as the organized points
/// (geometry/organized_points.hpp), return it. Header only.

#pragma once

#include <string>

namespace gca
{
/// What a step on the GPU gave: whether it was done, and why not when it was not. What "done"
/// means is the step's own: its work queued on the stream, or finished when the step waits.
struct step_result
{
    bool done = false;   ///< whether the step was done, as the step's comment says
    std::string message; ///< why not, in plain English; empty when done
};
} // namespace gca
