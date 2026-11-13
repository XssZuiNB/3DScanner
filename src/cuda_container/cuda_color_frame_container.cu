#include "cuda_container/cuda_container.hpp"

#include "cuda_container/cuda_frame.cuh"
#include "util/cuda_util.cuh"

#include <cuda_runtime_api.h>

namespace gca
{
cuda_color_frame::cuda_color_frame(uint32_t width, uint32_t height)
    : __m_impl(new cuda_color_frame_impl(width, height))
{
}

cuda_color_frame::cuda_color_frame(const uint8_t *frame, uint32_t width, uint32_t height)
    : __m_impl(new cuda_color_frame_impl(frame, width, height))
{
}

const thrust::device_vector<uint8_t> &cuda_color_frame::get_color_frame_vec() const
{
    return __m_impl->get_frame_vec();
}

void cuda_color_frame::upload(const uint8_t *src, uint32_t width, uint32_t height)
{
    __m_impl->upload(src, width, height);
}

cuda_color_frame::~cuda_color_frame()
{
    if (__m_impl)
    {
        delete __m_impl;
    }
    __m_impl = nullptr;
}
} // namespace gca
