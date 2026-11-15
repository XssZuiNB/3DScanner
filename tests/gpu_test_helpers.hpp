// SPDX-License-Identifier: 0BSD
/// @file gpu_test_helpers.hpp
/// The offline tests' GPU helpers, beside check.hpp's: a CUDA stream to queue the work on, what
/// the pool granted, the check that CUDA's last error is clear, the check of a GPU step's refusal,
/// and the download of an image after the work queued on its stream. A failure is a failed check,
/// counted by check.hpp. Header only; each test program is one translation unit.

#pragma once

#include "check.hpp"
#include "gpu/gpu_image.hpp"
#include "gpu/gpu_pool.hpp"
#include "gpu/step_result.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

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

/// Checks that a GPU step was refused with a message that holds a text.
/// @param what [in] the case, printed and named in the check line.
/// @param made [in] what the step gave.
/// @param expected [in] a text the message holds.
inline void check_refused(const std::string &what, const gca::step_result &made,
                          const std::string &expected)
{
    check_refusal(what, !made.done, made.message, made.message.find(expected) != std::string::npos);
}

/// Copies one image to the host after the work queued on a stream: the copy is queued on the
/// stream, which is then waited for.
/// @param image [in] the image.
/// @param out [out] its pixels, resized to them.
/// @param stream [in] the stream the image's work was queued on.
/// @return whether CUDA copied it.
template <typename Pixel>
bool download_image(const gca::gpu_image<Pixel> &image, std::vector<Pixel> &out,
                    cudaStream_t stream)
{
    out.resize(static_cast<std::size_t>(image.size().width) * image.size().height);
    return cudaMemcpyAsync(out.data(), image.data(), image.size_bytes(), cudaMemcpyDeviceToHost,
                           stream) == cudaSuccess &&
           cudaStreamSynchronize(stream) == cudaSuccess;
}
} // namespace gca_test
