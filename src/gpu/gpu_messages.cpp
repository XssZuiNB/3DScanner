// SPDX-License-Identifier: 0BSD
/// @file gpu_messages.cpp
/// The gpu module's refusal messages (see gpu_messages.hpp).

#include "gpu/gpu_messages.hpp"

#include <string>

namespace gca
{
std::string size_text(const frame_size &size)
{
    return std::to_string(size.width) + "x" + std::to_string(size.height) + " px";
}

std::string cuda_failure(const char *what, cudaError_t error)
{
    // Cleared here, at the one place every failure of the gpu module passes through.
    cudaGetLastError();
    return std::string(what) + ": " + cudaGetErrorName(error) + ", " + cudaGetErrorString(error);
}
} // namespace gca
