cmake_minimum_required(VERSION 3.18)

project(3DScanner LANGUAGES C CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# 87 = Jetson Orin, 89 = Ada, 61 = Pascal
set(GCA_CUDA_ARCHITECTURES "87" CACHE STRING "CUDA architectures to build for")

include_directories(src)

add_subdirectory(src)
add_subdirectory(apps)

enable_testing()
add_subdirectory(tests)
