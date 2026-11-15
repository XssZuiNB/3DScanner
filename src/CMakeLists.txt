# SPDX-License-Identifier: 0BSD
#
# src/CMakeLists.txt - the libraries, one per directory, each linking only the ones below it;
# types is header only, and camera/ holds two, so that only the Orbbec camera links OrbbecSDK:
#   types (Eigen) <- camera (the frame, the frame source seam, the Orbbec camera's rules, the
#                    sequence; threads)
#   camera <- orbbec_camera (OrbbecSDK, OpenCV)
#   types <- camera <- gpu (CUDA runtime) <- geometry (Eigen) <- registration
#   geometry <- fusion (the sparse TSDF, which takes a frame's organized points)
#   fusion, registration <- tracking (the per-frame path's steps 1 to 6 and the check; no kernel)
#   registration <- refinement (the pose graph, on the host; registration's SE(3) exponential)
#   types <- separation (the proposal of the support plane and the box, on the host; Eigen)
#   types <- camera <- scan (Eigen, OpenCV's core)
# Every library includes from src/ ("gpu/gpu_pool.hpp"); types carries that directory.

# ---------------------------------------------------------------- types (header only)
# The plain structs, the camera parameters, a frame's description, the camera's pose and a model's
# point, and the console colors, which camera, orbbec_camera and gpu use. types carries Eigen
# because the camera's pose (types/camera_pose.hpp) holds Eigen's vector and quaternion, plain value
# types that the sequence (camera) and the trajectory (scan) share, and a model's point
# (types/model_point.hpp) Eigen's vectors, which the model's file (scan) and separation share.
add_library(types INTERFACE)
target_include_directories(types INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(types INTERFACE Eigen3::Eigen)

# ---------------------------------------------------------------- camera
# The frame, the frame source seam (camera/frame_source.hpp), the sequence with its writer's
# thread, the POSIX file calls that scan shares, the Orbbec camera's decisions that need no
# camera (camera/orbbec_rules.hpp) with its IMU's queue (camera/orbbec_imu.hpp), and the seam of
# the camera's run-time properties with the hold of the color exposure and white balance
# (camera/camera_properties.hpp, camera/color_hold.hpp), built and linked without OrbbecSDK, so
# that the offline tests and the programs that need no camera load none.
find_package(Threads REQUIRED)
add_library(
    camera
    STATIC
    camera/camera_properties.hpp
    camera/color_hold.cpp
    camera/color_hold.hpp
    camera/frame.cpp
    camera/frame.hpp
    camera/frame_source.hpp
    camera/orbbec_imu.cpp
    camera/orbbec_imu.hpp
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
    gpu/gpu_array.hpp
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
# PUBLIC: the headers hold CUDA types, and gpu_frame.hpp the frame of camera.
target_link_libraries(gpu PUBLIC types camera CUDA::cudart)

# ---------------------------------------------------------------- geometry
add_library(
    geometry
    STATIC
    geometry/float3_eigen.cuh
    geometry/float3_ops.cuh
    geometry/frame_pyramid.cu
    geometry/frame_pyramid.hpp
    geometry/one_surface.cuh
    geometry/organized_normals.cu
    geometry/organized_normals.hpp
    geometry/organized_points.cu
    geometry/organized_points.hpp
    geometry/symmetric_eigen_3x3.cuh
)
target_link_libraries(geometry PUBLIC gpu Eigen3::Eigen)

# ---------------------------------------------------------------- registration
# The registration of a frame against a surface (docs/methods/registration.md).
add_library(
    registration
    STATIC
    registration/normal_equations.cpp
    registration/normal_equations.hpp
    registration/registration.cu
    registration/registration.hpp
    registration/se3.cpp
    registration/se3.hpp
)
target_link_libraries(registration PUBLIC geometry Eigen3::Eigen)

# ---------------------------------------------------------------- fusion
# The sparse TSDF (docs/methods/sparse-tsdf.md): its hash of blocks, its pool, the allocation of the
# blocks near a frame's surface, the integration, the ray cast and the extraction.
add_library(fusion STATIC fusion/block_hash.cuh fusion/sparse_tsdf.cu fusion/sparse_tsdf.hpp)
# PUBLIC: the header holds gpu's arrays, images and pool, CUDA's half float, geometry's organized
# points and frame pyramid, and the camera's pose and intrinsics of types.
target_link_libraries(fusion PUBLIC geometry)

# ---------------------------------------------------------------- tracking
# Tracking (docs/methods/tracking.md): each frame uploaded, its pyramid built, the live model ray
# cast from the last tracked pose, the registration, the check, and the fusion when it passes; it
# launches no kernel of its own.
add_library(tracking STATIC tracking/tracker.cpp tracking/tracker.hpp)
# PUBLIC: the header holds fusion's sparse TSDF and surface, registration's parameters and result,
# the frame of camera and the pose of types.
target_link_libraries(tracking PUBLIC fusion registration Eigen3::Eigen)

# ---------------------------------------------------------------- refinement
# Refinement (docs/methods/refinement.md): today the pose graph of keyframes and its solve, on the
# host in double; the SE(3) exponential and the check of a rigid transform are registration's
# (registration/se3.hpp), their one copy.
add_library(refinement STATIC refinement/pose_graph.cpp refinement/pose_graph.hpp)
# PUBLIC: the header holds Eigen's types; types carries src/.
target_link_libraries(refinement PUBLIC types Eigen3::Eigen PRIVATE registration)

# ---------------------------------------------------------------- separation
# Separation (docs/methods/separation.md): today the proposal of the support plane and the box from
# the full model and its trajectory, and the orthogonal least-squares plane, which
# distortion_check's measurement shares; on the host in double.
add_library(
    separation
    STATIC
    separation/plane_fit.cpp
    separation/plane_fit.hpp
    separation/proposal.cpp
    separation/proposal.hpp
)
# PUBLIC: the headers hold Eigen's types and types' model point and camera pose.
target_link_libraries(separation PUBLIC types Eigen3::Eigen)

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
# passes; the prototype's colored-ICP test, deleted since, printed one rotation error of 0.000224
# deg instead of 0.000225, every other printed value equal]. It guards nothing more
# [M 2026-11-13: nvcc 12.6 probes]: a __host__ __device__ function calling a host-only one draws
# warning #20011-D (#20014-D when the callee's body is visible) in either mode, and the device code
# drops the call, device link or not; a __host__ __device__ function left undefined stops ptxas
# without it, nvlink with it.
set_target_properties(
    geometry
    registration
    fusion
    PROPERTIES
    CUDA_ARCHITECTURES "${GCA_CUDA_ARCHITECTURES}"
)
