# SPDX-License-Identifier: 0BSD
#
# src/CMakeLists.txt - the libraries, one per directory, each linking only the ones below it;
# types is header only, and camera/ holds two, so that only the Orbbec camera links OrbbecSDK:
#   types (Eigen) <- camera (the frame, the frame source seam, the Orbbec camera's rules, the
#                    sequence; threads)
#   camera <- orbbec_camera (OrbbecSDK, OpenCV)
#   types <- camera <- gpu (CUDA runtime) <- geometry (Eigen) <- registration
#   geometry <- fusion (the sparse TSDF, which takes a frame's organized points)
#   types <- camera <- scan (Eigen, OpenCV's core)
# Every library includes from src/ ("gpu/cuda_container.hpp"); types carries that directory.

# ---------------------------------------------------------------- types (header only)
# The plain structs, the camera parameters, a frame's description and the camera's pose, and the
# console colors, which camera, orbbec_camera and gpu use. types carries Eigen because the
# camera's pose (types/camera_pose.hpp) holds Eigen's vector and quaternion, plain value types
# that the sequence (camera) and the trajectory (scan) share.
add_library(types INTERFACE)
target_include_directories(types INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(types INTERFACE Eigen3::Eigen)

# ---------------------------------------------------------------- camera
# The frame, the frame source seam (camera/frame_source.hpp), the sequence with its writer's
# thread, the POSIX file calls that scan shares, and the Orbbec camera's decisions that need no
# camera (camera/orbbec_rules.hpp), built and linked without OrbbecSDK, so that the offline
# tests and the programs that need no camera load none.
find_package(Threads REQUIRED)
add_library(
    camera
    STATIC
    camera/frame.cpp
    camera/frame.hpp
    camera/frame_source.hpp
    camera/orbbec_rules.cpp
    camera/orbbec_rules.hpp
    camera/posix_file.cpp
    camera/posix_file.hpp
    camera/sequence_format.cpp
    camera/sequence_format.hpp
    camera/sequence_reader.cpp
    camera/sequence_reader.hpp
    camera/sequence_writer.cpp
    camera/sequence_writer.hpp
)
target_link_libraries(camera PUBLIC types PRIVATE Threads::Threads)

# The Orbbec camera as a frame source: the one part of the camera module that links OrbbecSDK.
if(GCA_WITH_ORBBEC)
    add_library(
        orbbec_camera
        STATIC
        camera/orbbec_device.cpp
        camera/orbbec_device.hpp
    )
    target_link_libraries(
        orbbec_camera
        PUBLIC
        camera
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
    gpu/gpu_array.hpp
    gpu/gpu_check.cu
    gpu/gpu_check.hpp
    gpu/gpu_frame.cpp
    gpu/gpu_frame.hpp
    gpu/gpu_image.hpp
    gpu/gpu_messages.cpp
    gpu/gpu_messages.hpp
    gpu/gpu_pool.cpp
    gpu/gpu_pool.hpp
    gpu/kernel_launch.cuh
    gpu/pinned_buffer.hpp
    gpu/step_result.hpp
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
    geometry/organized_normals.cu
    geometry/organized_normals.hpp
    geometry/organized_points.cu
    geometry/organized_points.hpp
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

# ---------------------------------------------------------------- fusion
# The sparse TSDF (docs/methods/sparse-tsdf.md): today its hash of blocks, its pool and the
# allocation of the blocks near a frame's surface.
add_library(fusion STATIC fusion/block_hash.cuh fusion/sparse_tsdf.cu fusion/sparse_tsdf.hpp)
# PUBLIC: the header holds gpu's arrays, its pool, CUDA's half float, geometry's organized points
# and the camera's pose of types.
target_link_libraries(fusion PUBLIC geometry)

# ---------------------------------------------------------------- scan
# The files a scan keeps and the parameters of each object class (docs/SCAN_FILES.md); host only.
# OpenCV's core reads and writes the parameters' JSON.
add_library(
    scan
    STATIC
    scan/file_io.cpp
    scan/file_io.hpp
    scan/model_file.cpp
    scan/model_file.hpp
    scan/object_class.cpp
    scan/object_class.hpp
    scan/parameters_file.cpp
    scan/parameters_file.hpp
    scan/trajectory_file.cpp
    scan/trajectory_file.hpp
)
# PUBLIC: the headers hold Eigen types and camera's write_result (camera/posix_file.hpp).
target_link_libraries(scan PUBLIC types camera Eigen3::Eigen PRIVATE opencv_core)

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
    fusion
    PROPERTIES
    CUDA_ARCHITECTURES "${GCA_CUDA_ARCHITECTURES}"
)
