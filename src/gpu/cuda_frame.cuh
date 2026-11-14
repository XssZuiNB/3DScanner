#pragma once

#include "gpu/cuda_util.cuh"
#include "types/camera_param.hpp"

#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

namespace gca
{
template <typename T, size_t N_CHANNEL> class cuda_frame
{
public:
    cuda_frame(uint32_t width, uint32_t height)
        : m_data_vec(width * height * N_CHANNEL)
    {
    }

    cuda_frame(const T *frame, uint32_t width, uint32_t height)
        : m_data_vec(width * height * N_CHANNEL)
    {
        if (frame)
        {
            thrust::copy(frame, frame + m_data_vec.size(), m_data_vec.begin());
        }
    }

    void upload(const T *frame, uint32_t width, uint32_t height)
    {
        if (!frame)
        {
            return;
        }

        auto new_size = width * height * N_CHANNEL;
        if (new_size != m_data_vec.size())
        {
            m_data_vec.resize(new_size);
        }

        thrust::copy(frame, frame + new_size, m_data_vec.begin());
    }

    const thrust::device_vector<T> &get_frame_vec() const
    {
        return m_data_vec;
    }

private:
    thrust::device_vector<T> m_data_vec;
};
} // namespace gca
