# SPDX-License-Identifier: 0BSD
#
# apps/CMakeLists.txt - the programs: orbbec_viewer, the camera test, its command-line reader
# viewer_options and its loop on a frame source viewer_loop, which need no camera, so the offline
# tests drive them; distortion_check, which measures a snapshot of a checkerboard, its
# measurement checkerboard_measure and its command-line reader distortion_options, which the
# offline tests drive too; box_fit, the acceptance tool, which fits a box to an object model, its
# measurement box_measure and its command-line reader box_fit_options, which the offline tests
# drive too; depth_noise, which takes the depth noise's c from a still sequence, its measurement
# depth_noise_measure and its command-line reader depth_noise_options, which the offline tests
# drive too; replay, the replay tool, which runs a sequence through the scan interface, its
# command-line reader replay_options, which the offline tests drive too, and build_commit, the
# tree's commit the programs are built from; output_directory, the check of the directory a
# command line names for what a program writes, which the viewer and the replay share.

# The check of the directory a program's command line names for what it writes, host only;
# camera gives it the system's message for a directory it refuses.
add_library(output_directory STATIC output_directory.cpp output_directory.hpp)
target_include_directories(output_directory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(output_directory PRIVATE camera)

# The viewer's command line, host only: built and tested without the camera backend too.
add_library(viewer_options STATIC viewer_options.cpp viewer_options.hpp)
target_include_directories(viewer_options PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(viewer_options PRIVATE output_directory)

# The viewer's loop on any frame source, without the camera backend: the offline tests run it on
# the synthetic source. geometry gives it the organized points and normals, scan the models' PLY
# writer its snapshot uses.
add_library(viewer_loop STATIC viewer_loop.cpp viewer_loop.hpp)
target_link_libraries(
    viewer_loop
    PUBLIC
    camera
    viewer_options
    PRIVATE
    geometry
    scan
    opencv_core
    opencv_highgui
    opencv_imgcodecs
    opencv_imgproc
)

# The checkerboard measurement behind distortion_check, host only: built and tested without the
# camera backend too; separation gives it the plane fit (separation/plane_fit.hpp, the one copy).
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
    separation
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

# The acceptance tool's box fit (R21), host only: built and tested without the camera backend
# too; separation gives it the plane fit (separation/plane_fit.hpp, the one copy).
add_library(box_measure STATIC box_measure.cpp box_measure.hpp)
target_include_directories(box_measure PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(box_measure PUBLIC types PRIVATE separation)

# box_fit's command line, host only.
add_library(box_fit_options STATIC box_fit_options.cpp box_fit_options.hpp)
target_include_directories(box_fit_options PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# The acceptance tool: the edges of the box in an object model; opens no camera and no GPU. scan
# reads the model.
add_executable(box_fit box_fit.cpp)
target_link_libraries(box_fit PRIVATE box_fit_options box_measure scan)

# The depth noise's c from a still sequence, host only: built and tested without the camera
# backend too; camera reads the sequence, separation gives it the plane fit
# (separation/plane_fit.hpp, the one copy).
add_library(depth_noise_measure STATIC depth_noise_measure.cpp depth_noise_measure.hpp)
target_include_directories(depth_noise_measure PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(depth_noise_measure PUBLIC types PRIVATE camera separation Eigen3::Eigen)

# depth_noise's command line, host only.
add_library(depth_noise_options STATIC depth_noise_options.cpp depth_noise_options.hpp)
target_include_directories(depth_noise_options PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# The depth noise's c from a still sequence; opens no camera and no GPU.
add_executable(depth_noise depth_noise.cpp)
target_link_libraries(depth_noise PRIVATE depth_noise_measure depth_noise_options)

# The tree's commit the programs are built from (build_commit.hpp): build_commit.cmake writes its
# definition into the build directory at every build, replacing it only when its text changes, so
# that an unchanged tree relinks no program.
set(GCA_BUILD_COMMIT_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/build_commit.cpp")
add_custom_target(
    build_commit_stamp
    COMMAND
        ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR} -DOUTPUT=${GCA_BUILD_COMMIT_SOURCE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/build_commit.cmake
    BYPRODUCTS ${GCA_BUILD_COMMIT_SOURCE}
    VERBATIM
)
add_library(build_commit STATIC ${GCA_BUILD_COMMIT_SOURCE} build_commit.hpp)
add_dependencies(build_commit build_commit_stamp)
target_include_directories(build_commit PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# The replay tool's command line, host only: built and tested without the GPU; scan gives it the
# object classes' names, camera the sequence's file name.
add_library(replay_options STATIC replay_options.cpp replay_options.hpp)
target_include_directories(replay_options PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(replay_options PUBLIC scan PRIVATE camera output_directory)

# The replay tool (R19): a sequence through the scan interface, its scan's directory written into
# a directory it is given; links no OrbbecSDK.
add_executable(replay replay.cpp)
target_link_libraries(replay PRIVATE build_commit replay_options scan_interface)

# The Orbbec camera test: the viewer's loop on the camera.
if(GCA_WITH_ORBBEC)
    add_executable(orbbec_viewer orbbec_viewer.cpp)
    target_link_libraries(
        orbbec_viewer
        PRIVATE
        orbbec_camera
        viewer_loop
        viewer_options
    )
    set_target_properties(orbbec_viewer PROPERTIES
        BUILD_RPATH "${OrbbecSDK_DIR}")
endif()
