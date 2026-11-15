# SPDX-License-Identifier: 0BSD
#
# tests/CMakeLists.txt - the offline tests, one program per ctest, and the synthetic source of
# frames they share; none touches the camera (AGENTS.md §4).

# The tests' PLY reader, header only, on files built in memory; links nothing of the tree.
add_executable(test_ply_reader test_ply_reader.cpp)
add_test(NAME ply_reader COMMAND test_ply_reader)

# The files a scan keeps, read back by the scan's readers and by the tests' PLY reader.
add_executable(test_scan_files test_scan_files.cpp)
target_link_libraries(test_scan_files PRIVATE scan)
add_test(NAME scan_files COMMAND test_scan_files)

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

# The viewer's loop on the synthetic source; links no OrbbecSDK. CUDA's runtime lets the test leave
# CUDA's last error set, as a failed call would, before a frame.
add_executable(test_viewer_loop test_viewer_loop.cpp)
target_link_libraries(
    test_viewer_loop
    PRIVATE
    viewer_loop
    synthetic_source
    CUDA::cudart
    opencv_core
    opencv_imgcodecs
)
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
# frames built in the test.
add_executable(test_frame_pyramid test_frame_pyramid.cpp)
target_link_libraries(test_frame_pyramid PRIVATE geometry synthetic_source)
add_test(NAME frame_pyramid COMMAND test_frame_pyramid)

# The registration of a frame against a surface, on pairs of frames of the synthetic source whose
# transform is known; scan gives the object classes' depth ranges.
add_executable(test_registration test_registration.cpp)
target_link_libraries(test_registration PRIVATE registration scan synthetic_source)
add_test(NAME registration COMMAND test_registration)

# The sparse TSDF's hash of blocks and its pool, fed with block coordinates the test builds.
add_executable(test_sparse_tsdf_blocks test_sparse_tsdf_blocks.cpp)
target_link_libraries(test_sparse_tsdf_blocks PRIVATE fusion)
add_test(NAME sparse_tsdf_blocks COMMAND test_sparse_tsdf_blocks)

# The sparse TSDF's allocation near a frame's surface, on frames of the synthetic source at their
# known poses; scan gives the object classes' voxel sizes and depth ranges.
add_executable(test_sparse_tsdf_allocation test_sparse_tsdf_allocation.cpp)
target_link_libraries(test_sparse_tsdf_allocation PRIVATE fusion scan synthetic_source)
add_test(NAME sparse_tsdf_allocation COMMAND test_sparse_tsdf_allocation)

# The sparse TSDF's integration of a frame and extraction of the surface's points, on frames of the
# synthetic source at their known poses and on built walls; scan gives the object classes' voxel
# sizes and depth ranges.
add_executable(test_sparse_tsdf_fusion test_sparse_tsdf_fusion.cpp)
target_link_libraries(test_sparse_tsdf_fusion PRIVATE fusion scan synthetic_source)
add_test(NAME sparse_tsdf_fusion COMMAND test_sparse_tsdf_fusion)

# Separation's proposal of the support plane and the box, on full models built from the synthetic
# source's surfaces along its walks; scan gives the object classes' voxel sizes and depth ranges,
# CUDA the headers of the angle in stage_checks.hpp, whose other checks take CUDA's float3.
add_executable(test_separation test_separation.cpp)
target_link_libraries(test_separation PRIVATE separation scan synthetic_source CUDA::cudart)
add_test(NAME separation COMMAND test_separation)

# Refinement's pose graph solve, on graphs built from the keyframes of a walk of the synthetic
# source whose poses are known.
add_executable(test_pose_graph test_pose_graph.cpp)
target_link_libraries(test_pose_graph PRIVATE refinement registration synthetic_source)
add_test(NAME pose_graph COMMAND test_pose_graph)

# The acceptance tool's box fit, on object models the test builds: boxes of both object classes'
# sizes in several orientations with noise from a fixed seed, a remnant of the support plane, and
# models that are no box; and the program box_fit on models the test writes. The synthetic source
# gives the sphere's lattice, scan writes the models.
add_executable(test_box_measure test_box_measure.cpp)
target_link_libraries(test_box_measure PRIVATE box_measure scan synthetic_source Eigen3::Eigen)
add_test(NAME box_measure COMMAND test_box_measure $<TARGET_FILE:box_fit>)

add_executable(test_box_fit_options test_box_fit_options.cpp)
target_link_libraries(test_box_fit_options PRIVATE box_fit_options)
add_test(NAME box_fit_options COMMAND test_box_fit_options)

# The sparse TSDF's ray cast of the model's surface, on sparse TSDFs fused from frames of the
# synthetic source at their known poses and from built walls, and through the registration of a
# frame against it; scan gives the object classes' voxel sizes and depth ranges.
add_executable(test_sparse_tsdf_ray_cast test_sparse_tsdf_ray_cast.cpp)
target_link_libraries(test_sparse_tsdf_ray_cast PRIVATE fusion registration scan synthetic_source)
add_test(NAME sparse_tsdf_ray_cast COMMAND test_sparse_tsdf_ray_cast)
