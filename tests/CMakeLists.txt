# SPDX-License-Identifier: 0BSD
#
# tests/CMakeLists.txt - the offline tests, one program per ctest, and the synthetic source of
# frames they share; none touches the camera (AGENTS.md §4).

add_executable(test_point_cloud_from_rgbd test_point_cloud_from_rgbd.cpp)
target_link_libraries(test_point_cloud_from_rgbd PRIVATE geometry)
add_test(NAME point_cloud_from_rgbd COMMAND test_point_cloud_from_rgbd)

add_executable(test_point_cloud_geometry test_point_cloud_geometry.cpp)
target_link_libraries(test_point_cloud_geometry PRIVATE geometry)
add_test(NAME point_cloud_geometry COMMAND test_point_cloud_geometry)

add_executable(test_ply_reader test_ply_reader.cpp)
target_link_libraries(test_ply_reader PRIVATE geometry)
add_test(NAME ply_reader COMMAND test_ply_reader)

# The files a scan keeps; the tests' PLY reader takes geometry's point type.
add_executable(test_scan_files test_scan_files.cpp)
target_link_libraries(test_scan_files PRIVATE scan geometry)
add_test(NAME scan_files COMMAND test_scan_files)

add_executable(test_color_icp test_color_icp.cpp)
target_link_libraries(test_color_icp PRIVATE registration)
add_test(NAME color_icp COMMAND test_color_icp "${PROJECT_SOURCE_DIR}/data/frag_116.ply")

add_executable(test_viewer_options test_viewer_options.cpp)
target_link_libraries(test_viewer_options PRIVATE viewer_options)
add_test(NAME viewer_options COMMAND test_viewer_options)

# The tests' synthetic source of frames, for every test that needs frames; host only.
add_library(synthetic_source STATIC synthetic_source.cpp synthetic_source.hpp)
target_link_libraries(synthetic_source PUBLIC camera Eigen3::Eigen)

add_executable(test_frame_source test_frame_source.cpp)
target_link_libraries(test_frame_source PRIVATE synthetic_source)
add_test(NAME frame_source COMMAND test_frame_source)

# The Orbbec camera's decisions that need no camera, fed with plain data; links no OrbbecSDK.
add_executable(test_orbbec_rules test_orbbec_rules.cpp)
target_link_libraries(test_orbbec_rules PRIVATE camera)
add_test(NAME orbbec_rules COMMAND test_orbbec_rules)

# The viewer's loop on the synthetic source; links no OrbbecSDK.
add_executable(test_viewer_loop test_viewer_loop.cpp)
target_link_libraries(test_viewer_loop PRIVATE viewer_loop synthetic_source opencv_core opencv_imgcodecs)
add_test(NAME viewer_loop COMMAND test_viewer_loop)

add_executable(test_synthetic_source test_synthetic_source.cpp)
target_link_libraries(test_synthetic_source PRIVATE synthetic_source)
add_test(NAME synthetic_source COMMAND test_synthetic_source)

# The sequence on disk: frames of the synthetic source written and read back.
add_executable(test_sequence test_sequence.cpp)
target_link_libraries(test_sequence PRIVATE synthetic_source)
add_test(NAME sequence COMMAND test_sequence)

add_executable(test_checkerboard_measure test_checkerboard_measure.cpp)
target_link_libraries(test_checkerboard_measure PRIVATE checkerboard_measure Eigen3::Eigen)
add_test(NAME checkerboard_measure COMMAND test_checkerboard_measure)

add_executable(test_distortion_options test_distortion_options.cpp)
target_link_libraries(test_distortion_options PRIVATE distortion_options)
add_test(NAME distortion_options COMMAND test_distortion_options)

add_executable(test_gpu_images test_gpu_images.cpp)
target_link_libraries(test_gpu_images PRIVATE gpu synthetic_source)
add_test(NAME gpu_images COMMAND test_gpu_images)

add_executable(test_organized_points test_organized_points.cpp)
target_link_libraries(test_organized_points PRIVATE geometry synthetic_source)
add_test(NAME organized_points COMMAND test_organized_points)

add_executable(test_organized_normals test_organized_normals.cpp)
target_link_libraries(test_organized_normals PRIVATE geometry synthetic_source)
add_test(NAME organized_normals COMMAND test_organized_normals)

# The frame's pyramid: the intensity and the coarser levels, on frames of the synthetic source and
# frames built in the test; the prototype's intensity is the oracle.
add_executable(test_frame_pyramid test_frame_pyramid.cpp)
target_link_libraries(test_frame_pyramid PRIVATE geometry synthetic_source)
add_test(NAME frame_pyramid COMMAND test_frame_pyramid)

# The registration of a frame against a surface, on pairs of frames of the synthetic source whose
# transform is known, and against the prototype's colored ICP on the textured plane and on the
# fixture, which the synthetic source renders from its faces; scan gives the object classes' depth
# ranges.
add_executable(test_registration test_registration.cpp)
target_link_libraries(test_registration PRIVATE registration scan synthetic_source)
add_test(NAME registration COMMAND test_registration "${PROJECT_SOURCE_DIR}/data/frag_115.ply"
         "${PROJECT_SOURCE_DIR}/data/frag_116.ply")

# The sparse TSDF's hash of blocks and its pool, fed with block coordinates the test builds.
add_executable(test_sparse_tsdf_blocks test_sparse_tsdf_blocks.cpp)
target_link_libraries(test_sparse_tsdf_blocks PRIVATE fusion)
add_test(NAME sparse_tsdf_blocks COMMAND test_sparse_tsdf_blocks)

# The sparse TSDF's allocation near a frame's surface, on frames of the synthetic source at their
# known poses; scan gives the object classes' voxel sizes and depth ranges.
add_executable(test_sparse_tsdf_allocation test_sparse_tsdf_allocation.cpp)
target_link_libraries(test_sparse_tsdf_allocation PRIVATE fusion scan synthetic_source)
add_test(NAME sparse_tsdf_allocation COMMAND test_sparse_tsdf_allocation)

# Refinement's pose graph solve, on graphs built from the keyframes of a walk of the synthetic
# source whose poses are known.
add_executable(test_pose_graph test_pose_graph.cpp)
target_link_libraries(test_pose_graph PRIVATE refinement registration synthetic_source)
add_test(NAME pose_graph COMMAND test_pose_graph)
