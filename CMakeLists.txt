cmake_minimum_required(VERSION 3.18)

project(3DScanner LANGUAGES C CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 87 = Jetson Orin, 89 = Ada, 61 = Pascal
set(GCA_CUDA_ARCHITECTURES "87" CACHE STRING "CUDA architectures to build for")

add_subdirectory(src)
add_subdirectory(apps)

enable_testing()
add_subdirectory(tests)
