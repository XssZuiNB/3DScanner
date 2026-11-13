#pragma once

#include "util/console_color.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

#include <memory>

__forceinline__ static int div_up(int total, int grain)
{
    return (total + grain - 1) / grain;
}

__forceinline__ static void check_cuda_error(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        std::cout << RED << "CUDA ERR! "
                  << (std::string(file) + ":" + std::to_string(line) + ": " +
                      cudaGetErrorString(err));
    }
}
