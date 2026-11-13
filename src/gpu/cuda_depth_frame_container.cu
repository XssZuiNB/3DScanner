#include "gpu/cuda_container.hpp"
#include "gpu/cuda_frame.cuh"
#include "gpu/cuda_util.cuh"

#include <cuda_runtime_api.h>

namespace gca
{
cuda_depth_frame::cuda_depth_frame(uint32_t width, uint32_t height)
    : __m_impl(new cuda_depth_frame_impl(width, height))
{
}

cuda_depth_frame::cuda_depth_frame(const uint16_t *frame, uint32_t width, uint32_t height)
    : __m_impl(new cuda_depth_frame_impl(frame, width, height))
{
}

const thrust::device_vector<uint16_t> &cuda_depth_frame::get_depth_frame_vec() const
{
    return __m_impl->get_frame_vec();
}

void cuda_depth_frame::upload(const uint16_t *src, uint32_t width, uint32_t height)
{
    __m_impl->upload(src, width, height);
}

cuda_depth_frame::~cuda_depth_frame()
{
    if (__m_impl)
    {
        delete __m_impl;
    }
    __m_impl = nullptr;
}
} // namespace gca
