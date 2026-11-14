// SPDX-License-Identifier: 0BSD
/// @file gpu_messages.hpp
/// The gpu module's refusal messages: a picture's size, and CUDA's error, which is cleared as it
/// is written, so that a failure the gpu module reported as a value does not fail the next CUDA
/// call's check elsewhere.

#pragma once

#include "types/frame_description.hpp"

#include <cuda_runtime_api.h>

#include <string>

namespace gca
{
/// Writes a picture's size for a message.
/// @param size [in] the size, in px.
/// @return "<width>x<height> px".
/// @note Allocates for the string; throws std::bad_alloc when that fails.
std::string size_text(const frame_size &size);

/// Writes what CUDA refused and its error for a message, and clears the error CUDA keeps as the
/// calling thread's last one (cudaGetLastError resets it to cudaSuccess [S: cuda_runtime_api.h,
/// CUDA 12.6]), so that the next check of it elsewhere does not see it again. An error that
/// leaves "the process in an inconsistent state" [S: driver_types.h] comes back at every later
/// CUDA call all the same.
/// @param what [in] what CUDA refused, in plain English.
/// @param error [in] CUDA's error, not cudaSuccess.
/// @return what, then CUDA's name and description of the error.
/// @note Allocates for the string; throws std::bad_alloc when that fails. Waits for nothing.
std::string cuda_failure(const char *what, cudaError_t error);
} // namespace gca
