find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# Downloaded dependencies live outside the tree (tools/fetch_deps.sh).
if(DEFINED ENV{GCA_DEPS_DIR})
    set(_gca_deps_default "$ENV{GCA_DEPS_DIR}")
else()
    set(_gca_deps_default "$ENV{HOME}/.local/share/3dscanner")
endif()
set(GCA_DEPS_DIR "${_gca_deps_default}" CACHE PATH "Directory of tools/fetch_deps.sh's downloads")

# ---------------------------------------------------------------- Eigen (header only)
# The library uses Eigen in device code. nvcc compiles it without --expt-relaxed-constexpr
# [M 2026-11-13: nvcc 12.6, Eigen 3.4.0; no error, no warning, every test passes], so the
# flag is not set. Eigen 3.4.0 makes nvcc warn #20012-D 16 times in each translation unit
# that includes it unless the CUDA toolkit's include directory is a system one, which linking
# CUDA::cudart makes it [M 2026-11-13: nvcc 12.6].
# The code needs Eigen 3.4 (Eigen::Vector<T, N>), and the measurements above are on 3.4.0.
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    set(GCA_EIGEN_DIR "${GCA_DEPS_DIR}/eigen-3.4.0" CACHE PATH "Eigen source dir")
    if(NOT EXISTS "${GCA_EIGEN_DIR}/Eigen/Core")
        message(FATAL_ERROR "Eigen3 not found. Run tools/fetch_deps.sh, install libeigen3-dev or set GCA_EIGEN_DIR.")
    endif()
    # A source copy states its version only in Macros.h; accept 3.4 or a later 3.x, as
    # find_package above does [S: Eigen 3.4.0's CMakeLists.txt, SameMajorVersion].
    file(STRINGS "${GCA_EIGEN_DIR}/Eigen/src/Core/util/Macros.h" _gca_eigen_defines
        REGEX "^#define EIGEN_(WORLD|MAJOR)_VERSION [0-9]+$")
    string(REGEX REPLACE ".*EIGEN_WORLD_VERSION ([0-9]+).*" "\\1" _gca_eigen_world "${_gca_eigen_defines}")
    string(REGEX REPLACE ".*EIGEN_MAJOR_VERSION ([0-9]+).*" "\\1" _gca_eigen_major "${_gca_eigen_defines}")
    if(NOT _gca_eigen_world STREQUAL "3" OR NOT _gca_eigen_major GREATER_EQUAL 4)
        message(FATAL_ERROR "Eigen 3.4 needed; ${GCA_EIGEN_DIR} holds ${_gca_eigen_world}.${_gca_eigen_major}. Run tools/fetch_deps.sh or set GCA_EIGEN_DIR.")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${GCA_EIGEN_DIR}")
    message(STATUS "Using bundled Eigen: ${GCA_EIGEN_DIR}")
endif()

# ---------------------------------------------------------------- Camera backends
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

# ---------------------------------------------------------------- Sources
set(CAMERA_SRC camera/camera_device.hpp camera/camera_param.hpp)
if(GCA_WITH_ORBBEC)
    list(APPEND CAMERA_SRC camera/orbbec_device.cpp camera/orbbec_device.hpp)
endif()

file(GLOB CUDA_CONTAINER_SRC "cuda_container/*.hpp" "cuda_container/*.cu" "cuda_container/*.cuh")
file(GLOB GEOMETRY_SRC "geometry/*.cpp" "geometry/*.hpp" "geometry/*.cu" "geometry/*.cuh")
file(GLOB REGISTRATION_SRC
    "registration/*.cpp" "registration/*.hpp" "registration/*.cu" "registration/*.cuh")
file(GLOB UTIL_SRC "util/*.cpp" "util/*.hpp" "util/*.cu" "util/*.cuh")

add_library(
    gca_lib
    SHARED
    ${CAMERA_SRC}
    ${CUDA_CONTAINER_SRC}
    ${GEOMETRY_SRC}
    ${REGISTRATION_SRC}
    ${UTIL_SRC}
)

target_compile_features(gca_lib PUBLIC cxx_std_17)

target_include_directories(
    gca_lib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    gca_lib
    PUBLIC
    CUDA::cudart
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

if(GCA_WITH_ORBBEC)
    target_link_libraries(gca_lib PUBLIC ob::OrbbecSDK)
    target_compile_definitions(gca_lib PUBLIC GCA_WITH_ORBBEC)
endif()

set_target_properties(
    gca_lib
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${GCA_CUDA_ARCHITECTURES}"
)
