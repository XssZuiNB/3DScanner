#pragma once

#include "geometry/type.hpp"

#include <Eigen/Core>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace gca
{
::cudaError_t cuda_build_gauss_newton_color_icp(
    Eigen::Matrix<float, 6, 6> &JTJ, Eigen::Vector<float, 6> &JTr,
    const thrust::device_vector<gca::point_t> &src_points,
    const thrust::device_vector<gca::point_t> &tgt_points,
    const thrust::device_vector<float3> &tgt_normals,
    const thrust::device_vector<float3> &tgt_color_gradient,
    const thrust::device_vector<gca::index_t> &nn_src_tgt, const float lambda);
} // namespace gca
