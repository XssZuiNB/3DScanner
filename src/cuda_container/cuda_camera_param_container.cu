#include "cuda_container/cuda_container.hpp"
#include "util/cuda_util.cuh"

#include <cuda_runtime_api.h>

namespace gca
{
cuda_camera_param::cuda_camera_param(const device &device)
    : cuda_camera_param(device.get_depth_intrinsics(), device.get_color_intrinsics(),
                        device.get_depth_to_color_extrinsics(), device.get_width(),
                        device.get_height(), device.get_depth_scale(),
                        device.get_depth_frame_format(), device.get_color_frame_format())
{
}

cuda_camera_param::cuda_camera_param(const gca::intrinsics &depth_intrin,
                                     const gca::intrinsics &color_intrin,
                                     const gca::extrinsics &depth2color_extrin,
                                     const uint32_t width, const uint32_t height,
                                     const float depth_scale, const gca::frame_format depth_format,
                                     const gca::frame_format color_format)
    : m_width(width)
    , m_height(height)
    , m_depth_scale(depth_scale)
    , m_depth_frame_format(depth_format)
    , m_color_frame_format(color_format)
{
    auto err = cudaMalloc(&m_depth_intrin_ptr, sizeof(gca::intrinsics));
    check_cuda_error(err, __FILE__, __LINE__);
    err = cudaMalloc(&m_color_intrin_ptr, sizeof(gca::intrinsics));
    check_cuda_error(err, __FILE__, __LINE__);
    err = cudaMalloc(&m_depth2color_extrin_ptr, sizeof(gca::extrinsics));
    check_cuda_error(err, __FILE__, __LINE__);

    cudaMemcpy(m_depth_intrin_ptr, &depth_intrin, sizeof(gca::intrinsics), cudaMemcpyDefault);
    cudaMemcpy(m_color_intrin_ptr, &color_intrin, sizeof(gca::intrinsics), cudaMemcpyDefault);
    cudaMemcpy(m_depth2color_extrin_ptr, &depth2color_extrin, sizeof(gca::extrinsics),
               cudaMemcpyDefault);
}

const gca::intrinsics *cuda_camera_param::get_depth_intrinsics_ptr() const
{
    return m_depth_intrin_ptr;
}
const gca::intrinsics *cuda_camera_param::get_color_intrinsics_ptr() const
{
    return m_color_intrin_ptr;
}
const gca::extrinsics *cuda_camera_param::get_depth2color_extrinsics_ptr() const
{
    return m_depth2color_extrin_ptr;
}
uint32_t cuda_camera_param::get_width() const
{
    return m_width;
}
uint32_t cuda_camera_param::get_height() const
{
    return m_height;
}
float cuda_camera_param::get_depth_scale() const
{
    return m_depth_scale;
}
gca::frame_format cuda_camera_param::get_depth_frame_format() const
{
    return m_depth_frame_format;
}
gca::frame_format cuda_camera_param::get_color_frame_format() const
{
    return m_color_frame_format;
}

cuda_camera_param::~cuda_camera_param()
{
    if (m_depth_intrin_ptr)
    {
        cudaFree(m_depth_intrin_ptr);
    }
    m_depth_intrin_ptr = nullptr;

    if (m_color_intrin_ptr)
    {
        cudaFree(m_color_intrin_ptr);
    }
    m_color_intrin_ptr = nullptr;

    if (m_depth2color_extrin_ptr)
    {
        cudaFree(m_depth2color_extrin_ptr);
    }
    m_depth2color_extrin_ptr = nullptr;
}
} // namespace gca
