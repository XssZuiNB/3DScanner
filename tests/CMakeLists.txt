# SPDX-License-Identifier: 0BSD
#
# tests/CMakeLists.txt - the offline tests, one program per ctest; none touches the camera
# (AGENTS.md §4).

add_executable(test_point_cloud_from_rgbd test_point_cloud_from_rgbd.cpp)
target_link_libraries(test_point_cloud_from_rgbd PRIVATE geometry)
add_test(NAME point_cloud_from_rgbd COMMAND test_point_cloud_from_rgbd)

add_executable(test_point_cloud_geometry test_point_cloud_geometry.cpp)
target_link_libraries(test_point_cloud_geometry PRIVATE geometry)
add_test(NAME point_cloud_geometry COMMAND test_point_cloud_geometry)

add_executable(test_ply_reader test_ply_reader.cpp)
target_link_libraries(test_ply_reader PRIVATE geometry)
add_test(NAME ply_reader COMMAND test_ply_reader)

add_executable(test_color_icp test_color_icp.cpp)
target_link_libraries(test_color_icp PRIVATE registration)
add_test(NAME color_icp COMMAND test_color_icp "${PROJECT_SOURCE_DIR}/data/frag_116.ply")

add_executable(test_viewer_options test_viewer_options.cpp)
target_link_libraries(test_viewer_options PRIVATE viewer_options)
add_test(NAME viewer_options COMMAND test_viewer_options)
