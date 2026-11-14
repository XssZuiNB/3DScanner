# SPDX-License-Identifier: 0BSD
#
# src/CMakeLists.txt - the libraries, one per directory, each linking only the ones below it;
# types is header only, and camera/ holds two, so that only the Orbbec camera links OrbbecSDK:
#   types <- camera (the frame and the frame source seam)
#   types <- orbbec_camera (OrbbecSDK, OpenCV)
#   types <- camera <- gpu (CUDA runtime) <- geometry (Eigen) <- registration
# Every library includes from src/ ("gpu/cuda_container.hpp"); types carries that directory.

# ---------------------------------------------------------------- types (header only)
# The plain structs, the camera parameters and a frame's description, and the console colors,
# which camera, orbbec_camera and gpu use.
add_library(types INTERFACE)
target_include_directories(types INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------- camera
# The frame and the frame source seam (camera/frame_source.hpp), built and linked without
# OrbbecSDK, so that the offline tests and the programs that need no camera load none.
add_library(
    camera
    STATIC
    camera/frame.cpp
    camera/frame.hpp
    camera/frame_source.hpp
)
target_link_libraries(camera PUBLIC types)

# The Orbbec camera, the one part of the camera module that links OrbbecSDK.
if(GCA_WITH_ORBBEC)
    add_library(
        orbbec_camera
        STATIC
        camera/camera_device.hpp
        camera/orbbec_device.cpp
        camera/orbbec_device.hpp
    )
    target_link_libraries(
        orbbec_camera
        PUBLIC
        types
        opencv_core
        ob::OrbbecSDK
        PRIVATE
        opencv_imgcodecs
        opencv_imgproc
    )
endif()

# ---------------------------------------------------------------- gpu
add_library(
    gpu
    STATIC
    gpu/cuda_camera_param_container.cu
    gpu/cuda_color_frame_container.cu
    gpu/cuda_container.hpp
    gpu/cuda_depth_frame_container.cu
    gpu/cuda_frame.cuh
    gpu/cuda_util.cuh
    gpu/gpu_check.cu
    gpu/gpu_check.hpp
    gpu/gpu_frame.cpp
    gpu/gpu_frame.hpp
    gpu/gpu_image.hpp
    gpu/gpu_messages.cpp
    gpu/gpu_messages.hpp
    gpu/gpu_pool.cpp
    gpu/gpu_pool.hpp
    gpu/pinned_buffer.hpp
)
# PUBLIC: the headers hold thrust and CUDA types, and gpu_frame.hpp the frame of camera.
target_link_libraries(gpu PUBLIC types camera CUDA::cudart)

# ---------------------------------------------------------------- geometry
add_library(
    geometry
    STATIC
    geometry/cuda_estimate_normals.cu
    geometry/cuda_estimate_normals.cuh
    geometry/cuda_nn_search.cu
    geometry/cuda_nn_search.cuh
    geometry/cuda_point_cloud_factory.cu
    geometry/cuda_point_cloud_factory.cuh
    geometry/cuda_voxel_grid_down_sample.cu
    geometry/cuda_voxel_grid_down_sample.cuh
    geometry/float3_eigen.cuh
    geometry/float3_ops.cuh
    geometry/geometry_util.cuh
    geometry/point_cloud.cu
    geometry/point_cloud.hpp
    geometry/symmetric_eigen_3x3.cuh
    geometry/type.hpp
)
target_link_libraries(geometry PUBLIC gpu Eigen3::Eigen)

# ---------------------------------------------------------------- registration
add_library(
    registration
    STATIC
    registration/color_icp.cu
    registration/color_icp.hpp
    registration/cuda_color_icp_build_least_square.cu
    registration/cuda_color_icp_build_least_square.cuh
    registration/cuda_compute_color_gradient.cu
    registration/cuda_compute_color_gradient.cuh
    registration/eigen_solver.hpp
    registration/ldlt_3x3.cuh
)
target_link_libraries(registration PUBLIC geometry Eigen3::Eigen)

# Separable compilation stays off: no device function or variable is used outside the
# translation unit that defines it, so nothing needs a device link [M 2026-11-13: every test
# passes; the colored-ICP fixture's rotation error prints 0.000224 deg instead of 0.000225, every
# other printed value is equal]. It guards nothing more [M 2026-11-13: nvcc 12.6 probes]: a
# __host__ __device__ function calling a host-only one draws warning #20011-D (#20014-D when the
# callee's body is visible) in either mode, and the device code drops the call, device link or
# not; a __host__ __device__ function left undefined stops ptxas without it, nvlink with it.
set_target_properties(
    gpu
    geometry
    registration
    PROPERTIES
    CUDA_ARCHITECTURES "${GCA_CUDA_ARCHITECTURES}"
)
