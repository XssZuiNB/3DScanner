# SPDX-License-Identifier: 0BSD
#
# apps/CMakeLists.txt - the programs: orbbec_viewer, the camera test, its command-line reader
# viewer_options and its loop on a frame source viewer_loop, which need no camera, so the offline
# tests drive them; distortion_check, which measures a snapshot of a checkerboard, its
# measurement checkerboard_measure and its command-line reader distortion_options, which the
# offline tests drive too.

# The viewer's command line, host only: built and tested without the camera backend too; camera
# gives it the system's message for a directory it refuses.
add_library(viewer_options STATIC viewer_options.cpp viewer_options.hpp)
target_include_directories(viewer_options PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(viewer_options PRIVATE camera)

# The viewer's loop on any frame source, without the camera backend: the offline tests run it on
# the synthetic source.
add_library(viewer_loop STATIC viewer_loop.cpp viewer_loop.hpp)
target_link_libraries(
    viewer_loop
    PUBLIC
    camera
    viewer_options
    PRIVATE
    geometry
    opencv_core
    opencv_highgui
    opencv_imgcodecs
    opencv_imgproc
)

# The checkerboard measurement behind distortion_check, host only: built and tested without the
# camera backend too.
add_library(checkerboard_measure STATIC checkerboard_measure.cpp checkerboard_measure.hpp)
target_include_directories(checkerboard_measure PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    checkerboard_measure
    PUBLIC
    types
    opencv_core
    PRIVATE
    Eigen3::Eigen
    opencv_calib3d
    opencv_imgproc
)

# distortion_check's command line, host only.
add_library(distortion_options STATIC distortion_options.cpp distortion_options.hpp)
target_link_libraries(distortion_options PUBLIC checkerboard_measure)

# Whether the aligned frames are distorted, from a snapshot of orbbec_viewer; opens no camera.
add_executable(distortion_check distortion_check.cpp)
target_link_libraries(
    distortion_check
    PRIVATE
    checkerboard_measure
    distortion_options
    opencv_core
    opencv_imgcodecs
)

# Orbbec camera test / live point cloud viewer
if(GCA_WITH_ORBBEC)
    add_executable(orbbec_viewer orbbec_viewer.cpp)
    target_link_libraries(
        orbbec_viewer
        PRIVATE
        gpu
        orbbec_camera
        viewer_loop
        viewer_options
    )
    set_target_properties(orbbec_viewer PROPERTIES
        BUILD_RPATH "${OrbbecSDK_DIR}")
endif()
