# SPDX-License-Identifier: 0BSD
#
# apps/CMakeLists.txt - the programs: orbbec_viewer, the camera test, and its command-line
# reader viewer_options, which needs no camera, so the offline tests drive it.

# The viewer's command line, host only: built and tested without the camera backend too.
add_library(viewer_options STATIC viewer_options.cpp viewer_options.hpp)
target_include_directories(viewer_options PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Orbbec camera test / live point cloud viewer
if(GCA_WITH_ORBBEC)
    add_executable(orbbec_viewer orbbec_viewer.cpp)
    target_link_libraries(
        orbbec_viewer
        PRIVATE
        geometry
        gpu
        viewer_options
        opencv_core
        opencv_highgui
        opencv_imgcodecs
        opencv_imgproc
        orbbec_camera
    )
    set_target_properties(orbbec_viewer PROPERTIES
        BUILD_RPATH "${OrbbecSDK_DIR}")
endif()
