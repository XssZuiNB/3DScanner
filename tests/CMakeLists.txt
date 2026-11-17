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

# The replay tool's command line, on paths the test makes; needs no GPU.
add_executable(test_replay_options test_replay_options.cpp)
target_link_libraries(test_replay_options PRIVATE replay_options)
add_test(NAME replay_options COMMAND test_replay_options)

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

# The Orbbec camera's IMU without the camera: its streams' choice, its refusal, its queue of
# readings given to the frames, one thread adding while another takes; links no OrbbecSDK.
add_executable(test_orbbec_imu test_orbbec_imu.cpp)
target_link_libraries(test_orbbec_imu PRIVATE camera)
add_test(NAME orbbec_imu COMMAND test_orbbec_imu)

# The hold of the color camera's exposure and white balance, in front of a stand-in camera: the
# synthetic source's frames and a table of the camera's properties; links no OrbbecSDK.
add_executable(test_color_hold test_color_hold.cpp)
target_link_libraries(test_color_hold PRIVATE synthetic_source)
add_test(NAME color_hold COMMAND test_color_hold)

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
# transform is known; scan_interface gives the object classes' values of the registration.
add_executable(test_registration test_registration.cpp)
target_link_libraries(test_registration PRIVATE registration scan_interface synthetic_source)
add_test(NAME registration COMMAND test_registration)

# The sparse TSDF's hash of blocks and its pool, fed with block coordinates the test builds.
add_executable(test_sparse_tsdf_blocks test_sparse_tsdf_blocks.cpp)
target_link_libraries(test_sparse_tsdf_blocks PRIVATE fusion)
add_test(NAME sparse_tsdf_blocks COMMAND test_sparse_tsdf_blocks)

# The sparse TSDF's allocation near a frame's surface, on frames of the synthetic source at their
# known poses; scan gives the object classes' voxel sizes and far depths.
add_executable(test_sparse_tsdf_allocation test_sparse_tsdf_allocation.cpp)
target_link_libraries(test_sparse_tsdf_allocation PRIVATE fusion scan synthetic_source)
add_test(NAME sparse_tsdf_allocation COMMAND test_sparse_tsdf_allocation)

# The sparse TSDF's integration of a frame and extraction of the surface's points, on frames of the
# synthetic source at their known poses and on built walls; scan gives the object classes' voxel
# sizes.
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

# Refinement's fusion of a scan's sequence at the full voxel size, on sequences the test writes
# from frames of the synthetic source with their known poses; scan_interface gives the tabletop
# object class's full model and depth range.
add_executable(test_sequence_fusion test_sequence_fusion.cpp)
target_link_libraries(test_sequence_fusion PRIVATE refinement scan_interface synthetic_source)
add_test(NAME sequence_fusion COMMAND test_sequence_fusion)

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

# depth_noise's measurement, on sequences of a plane the test writes through the sequence's writer
# from the synthetic source with its depth noise of a known c, and the program depth_noise on
# sequences the test writes.
add_executable(test_depth_noise_measure test_depth_noise_measure.cpp)
target_link_libraries(test_depth_noise_measure PRIVATE depth_noise_measure synthetic_source)
add_test(NAME depth_noise_measure COMMAND test_depth_noise_measure $<TARGET_FILE:depth_noise>)

add_executable(test_depth_noise_options test_depth_noise_options.cpp)
target_link_libraries(test_depth_noise_options PRIVATE depth_noise_options)
add_test(NAME depth_noise_options COMMAND test_depth_noise_options)

# The sparse TSDF's ray cast of the model's surface, on sparse TSDFs fused from frames of the
# synthetic source at their known poses and from built walls, and through the registration of a
# frame against it, the cast's levels as tracking hands them over; scan_interface gives the object
# classes' voxel sizes and registration's values.
add_executable(test_sparse_tsdf_ray_cast test_sparse_tsdf_ray_cast.cpp)
target_link_libraries(
    test_sparse_tsdf_ray_cast
    PRIVATE
    fusion
    registration
    tracking
    scan_interface
    synthetic_source
)
add_test(NAME sparse_tsdf_ray_cast COMMAND test_sparse_tsdf_ray_cast)

# Tracking: each frame of the synthetic source registered against the live model, checked, and
# fused only when the check passes, with the pool closed before the first frame; scan_interface
# gives the tabletop object class's values as tracking's and the live model's parameters.
add_executable(test_tracking test_tracking.cpp)
target_link_libraries(test_tracking PRIVATE tracking scan_interface synthetic_source)
add_test(NAME tracking COMMAND test_tracking)

# The stage clock that times the per-frame path's stages for the replay tool: its names, a stage
# that queues work on the GPU, the host's stages.
add_executable(test_stage_clock test_stage_clock.cpp)
target_link_libraries(test_stage_clock PRIVATE tracking)
add_test(NAME stage_clock COMMAND test_stage_clock)

# The stages' parameters of an object class's set, and each stage taking each object class's from
# a closed pool, without device memory; the stage frames give the tests' stream size.
add_executable(test_stage_parameters test_stage_parameters.cpp)
target_link_libraries(test_stage_parameters PRIVATE scan_interface synthetic_source)
add_test(NAME stage_parameters COMMAND test_stage_parameters)

# The taking of a scan's live models: the spacing's rule, and the takes on frames of the synthetic
# source fused at their known poses, against an extraction of every block.
add_executable(test_live_model test_live_model.cpp)
target_link_libraries(test_live_model PRIVATE scan_interface synthetic_source)
add_test(NAME live_model COMMAND test_live_model)

# The scan interface in process, the spec's main seam: whole scans on the synthetic source into a
# temporary scans directory, with a source and a writer in front of the real ones that the test
# controls.
add_executable(test_scan_interface test_scan_interface.cpp)
target_link_libraries(test_scan_interface PRIVATE scan_interface synthetic_source)
add_test(NAME scan_interface COMMAND test_scan_interface)

# The replay tool run as a program on a short synthetic sequence the test writes, against the same
# scan made in process through the scan interface; replay_options gives its usage, build_commit
# the commit its scans' parameters hold.
add_executable(test_replay test_replay.cpp)
target_link_libraries(
    test_replay
    PRIVATE
    build_commit
    replay_options
    scan_interface
    synthetic_source
)
add_test(NAME replay COMMAND test_replay $<TARGET_FILE:replay>)

# The web layer's routes, the spec's HTTP test: a whole scan through them on a free port of the
# loopback interface, over the scan interface on the synthetic source, with cpp-httplib's client
# and OpenCV's JSON reader for the messages; the scan against the same scan made in process.
add_executable(test_http_routes test_http_routes.cpp)
target_link_libraries(test_http_routes PRIVATE web synthetic_source httplib::httplib opencv_core)
add_test(NAME http_routes COMMAND test_http_routes)
