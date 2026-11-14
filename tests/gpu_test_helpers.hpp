// SPDX-License-Identifier: 0BSD
/// @file gpu_test_helpers.hpp
/// The offline tests' GPU helpers, beside check.hpp's: a CUDA stream to queue the work on, what
/// the pool granted, and the check that CUDA's last error is clear. A failure is a failed check,
/// counted by check.hpp. Header only; each test program is one translation unit.

#pragma once

#include "check.hpp"
#include "gpu/gpu_pool.hpp"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <string>
#include <utility>

namespace gca_test
{
/// Creates the CUDA stream a test queues its work on.
/// @param stream [out] the stream, not blocking with the default stream; nullptr on failure.
/// @return whether it was created; a failed check and CUDA's error printed when not.
inline bool create_stream(cudaStream_t &stream)
{
    stream = nullptr;
    const cudaError_t created = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    check(created == cudaSuccess, "a CUDA stream");
    if (created != cudaSuccess)
        std::fprintf(stderr, "cudaStreamCreateWithFlags: %s\n", cudaGetErrorString(created));
    return created == cudaSuccess;
}

/// Checks that CUDA's last error is clear, so that a refusal left nothing for the next check.
/// @param what [in] the case, named in the check line.
inline void check_no_cuda_error(const std::string &what)
{
    const cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess)
        std::printf("%s: CUDA's last error is %s\n", what.c_str(), cudaGetErrorName(error));
    check(error == cudaSuccess, (what + ": CUDA's last error clear").c_str());
}

/// Takes what a request to the pool gave, and checks that it was granted.
/// @param what [in] what was requested, named in the check line: "a GPU frame".
/// @param requested [in] the request's result (gpu_pool::request(), gpu_frame::request(), ...);
///        taken over.
/// @return what was granted; an empty one, and a failed check, when the request was refused.
template <typename Made> Made granted(const std::string &what, gca::request_result<Made> requested)
{
    check(requested.status == gca::request_status::granted,
          (what + " granted, " + requested.message).c_str());
    return std::move(requested.granted);
}
} // namespace gca_test
