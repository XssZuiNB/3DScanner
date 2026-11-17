# SPDX-License-Identifier: 0BSD
#
# CMakeLists.txt - the tree's build: the C++20 dialect for C++ and CUDA, the build type and
# CUDA architecture, the warning set, the dependencies (CUDA, Eigen, cpp-httplib, three.js, OpenCV,
# OrbbecSDK), then the libraries and the page (src/), the programs (apps/) and the tests (tests/).

# 3.25.2 is the first CMake that knows the CUDA dialect C++20 with nvcc 12 (CODING_STANDARDS.md,
# "What the guides leave to this tree", C++20).
cmake_minimum_required(VERSION 3.25.2)

project(3DScanner LANGUAGES C CXX CUDA)

# C++20 for C++ and CUDA, on every target of the tree and required; extensions stay at CMake's
# default, which gives g++ -std=gnu++20 and nvcc -std=c++20 (CODING_STANDARDS.md, same bullet).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 87 = Jetson Orin, 89 = Ada, 61 = Pascal
set(GCA_CUDA_ARCHITECTURES "87" CACHE STRING "CUDA architectures to build for")

# One CUDA runtime in every program: the shared libcudart.so, which the gpu library links as
# CUDA::cudart. CMake's default, the static runtime, links libcudart_static.a beside it
# [M 2026-11-13: the link line of the one shared library before the split had both].
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# ---------------------------------------------------------------- Warnings
# The warning set of CODING_STANDARDS.md, on every target of the tree: -Wall -Wextra for host
# code and for nvcc's host compiler; nvcc's own warnings are its defaults, none suppressed, and
# -Wreorder, -Wdefault-stream-launch (a launch that names no stream) and -Wmissing-launch-bounds (a
# kernel without __launch_bounds__). tools/verify.sh fails on any warning in the configure or build
# log.
add_compile_options(
    "$<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra;-Wreorder>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Wdefault-stream-launch;-Wmissing-launch-bounds>"
)

# The dependencies are found here, so that their targets are seen in src/, apps/ and tests/.
find_package(CUDAToolkit REQUIRED)

# Downloaded dependencies live outside the tree (tools/fetch_deps.sh).
if(DEFINED ENV{GCA_DEPS_DIR})
    set(_gca_deps_default "$ENV{GCA_DEPS_DIR}")
else()
    set(_gca_deps_default "$ENV{HOME}/.local/share/3dscanner")
endif()
set(GCA_DEPS_DIR "${_gca_deps_default}" CACHE PATH "Directory of tools/fetch_deps.sh's downloads")

# ---------------------------------------------------------------- Eigen (header only)
# The geometry and registration libraries use Eigen in device code. nvcc compiles it without
# --expt-relaxed-constexpr [M 2026-11-13: nvcc 12.6, Eigen 3.4.0; no error, no warning, every
# test passes], so the flag is not set. Eigen 3.4.0 makes nvcc warn #20012-D 16 times in each
# translation unit that includes it unless the CUDA toolkit's include directory is a system one,
# which linking CUDA::cudart makes it [M 2026-11-13: nvcc 12.6]; the gpu library links it PUBLIC,
# so every library and program above it gets the system directory.
# The code needs Eigen 3.4 (Eigen::Vector<T, N>), and the measurements above are on 3.4.0.
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    set(GCA_EIGEN_DIR "${GCA_DEPS_DIR}/eigen-3.4.0" CACHE PATH "Eigen source dir")
    if(NOT EXISTS "${GCA_EIGEN_DIR}/Eigen/Core")
        message(FATAL_ERROR "Eigen3 not found. Run tools/fetch_deps.sh, install libeigen3-dev "
            "or set GCA_EIGEN_DIR.")
    endif()
    # A source copy states its version only in Macros.h; accept 3.4 or a later 3.x, as
    # find_package above does [S: Eigen 3.4.0's CMakeLists.txt, SameMajorVersion].
    file(STRINGS "${GCA_EIGEN_DIR}/Eigen/src/Core/util/Macros.h" _gca_eigen_defines
        REGEX "^#define EIGEN_(WORLD|MAJOR)_VERSION [0-9]+$")
    string(REGEX REPLACE ".*EIGEN_WORLD_VERSION ([0-9]+).*" "\\1" _gca_eigen_world
        "${_gca_eigen_defines}")
    string(REGEX REPLACE ".*EIGEN_MAJOR_VERSION ([0-9]+).*" "\\1" _gca_eigen_major
        "${_gca_eigen_defines}")
    if(NOT _gca_eigen_world STREQUAL "3" OR NOT _gca_eigen_major GREATER_EQUAL 4)
        message(FATAL_ERROR "Eigen 3.4 needed; ${GCA_EIGEN_DIR} holds "
            "${_gca_eigen_world}.${_gca_eigen_major}. Run tools/fetch_deps.sh or set "
            "GCA_EIGEN_DIR.")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${GCA_EIGEN_DIR}")
    message(STATUS "Using bundled Eigen: ${GCA_EIGEN_DIR}")
endif()
# Only Eigen's MPL2 files are used: with this definition an include of one of its LGPL files
# stops the compile, for an installed Eigen and the bundled one alike (CODING_STANDARDS.md,
# "What the guides leave to this tree", Eigen).
set_property(TARGET Eigen3::Eigen APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS EIGEN_MPL2_ONLY)

# ---------------------------------------------------------------- cpp-httplib (header only)
# The web layer's HTTP server (src/web/) and the HTTP test's client: cpp-httplib v0.66.0, the
# user's choice (README.md, "License"), one header from tools/fetch_deps.sh. An IMPORTED target, as
# Eigen's, so that its directory is a system one and the warning rule of tools/verify.sh holds the
# tree's own code: the header compiled without a warning under the tree's flags as a plain -I
# directory [M 2026-11-14: g++ 11.4, choose-web-libraries' probe], and a later version's warnings
# do not fail the build. Its TLS, zlib and brotli stay off: none of their macros is defined.
set(GCA_HTTPLIB_DIR "${GCA_DEPS_DIR}/cpp-httplib-0.66.0" CACHE PATH "cpp-httplib source dir")
if(NOT EXISTS "${GCA_HTTPLIB_DIR}/httplib.h")
    message(FATAL_ERROR "cpp-httplib 0.66.0 not found in ${GCA_HTTPLIB_DIR}. Run "
        "tools/fetch_deps.sh or set GCA_HTTPLIB_DIR.")
endif()
add_library(httplib::httplib INTERFACE IMPORTED)
set_target_properties(httplib::httplib PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${GCA_HTTPLIB_DIR}")

# ---------------------------------------------------------------- three.js (the page's)
# The page's 3D view (src/web/page/live_view.js): three.js 0.187.0, the user's choice (README.md,
# "License"), npm's package from tools/fetch_deps.sh. The build copies four of its files into the
# page's directory (src/CMakeLists.txt, "The page"); nothing of it is compiled.
set(GCA_THREE_DIR "${GCA_DEPS_DIR}/three-0.187.0" CACHE PATH "three.js package dir")
if(NOT EXISTS "${GCA_THREE_DIR}/build/three.module.js")
    message(FATAL_ERROR "three.js 0.187.0 not found in ${GCA_THREE_DIR}. Run "
        "tools/fetch_deps.sh or set GCA_THREE_DIR.")
endif()

# The page's directory in the build: the page's files and three.js's, which the web layer serves
# (docs/ROUTES.md, "The page's files") and the HTTP test fetches.
set(GCA_PAGE_DIRECTORY "${CMAKE_BINARY_DIR}/page")

# ---------------------------------------------------------------- OpenCV
# The Orbbec camera library (orbbec_camera) and the viewer decode, save and show images with it;
# distortion_check finds the checkerboard and fits its homography with calib3d, which links
# features2d and flann [M 2026-11-14: ldd of distortion_check, OpenCV 4.8.0].
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui calib3d features2d flann)

# ---------------------------------------------------------------- Camera backends
# The Orbbec camera library (orbbec_camera) and the viewer are built with the Orbbec backend;
# the camera library, the frame and its seam, is built without it too.
option(GCA_WITH_ORBBEC "Build Orbbec camera support (OrbbecSDK v2)" ON)
if(GCA_WITH_ORBBEC)
    if(NOT OrbbecSDK_DIR)
        file(GLOB _ob_sdk_dirs "${GCA_DEPS_DIR}/OrbbecSDK_v2*/lib")
        if(NOT _ob_sdk_dirs)
            message(FATAL_ERROR "OrbbecSDK not found in ${GCA_DEPS_DIR}. Run tools/fetch_deps.sh.")
        endif()
        list(GET _ob_sdk_dirs 0 _ob_sdk_dir)
        set(OrbbecSDK_DIR "${_ob_sdk_dir}" CACHE PATH "OrbbecSDK cmake config dir")
    endif()
    find_package(OrbbecSDK REQUIRED)
endif()

message(STATUS "GCA: Orbbec=${GCA_WITH_ORBBEC}")

add_subdirectory(src)
add_subdirectory(apps)

enable_testing()
add_subdirectory(tests)
