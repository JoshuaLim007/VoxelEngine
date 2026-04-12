#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_math.h"
#include "VolumeRaytracer.cuh"

using namespace GPUDDA;

__global__ void PopulateVoxels(BitArray voxels, uint3 size);

static VoxelBuffer3D CreateVoxels(uint3 size)
{
    VoxelBuffer3D voxels;
    voxels.dimensions[0] = size.x;
    voxels.dimensions[1] = size.y;
    voxels.dimensions[2] = size.z;
    size_t buffer_size = static_cast<size_t>(size.x) * size.y * size.z;
    voxels.grid = BitArray(buffer_size);

    BitArray temp = BitArray(buffer_size, true);
    auto threads = dim3(8, 8, 8);
    auto scaled_size = make_uint3(size.x, size.y, size.z);
    auto dim = dim3((scaled_size.x / 8 + threads.x - 1) / threads.x, (scaled_size.y + threads.y - 1) / threads.y,
        (scaled_size.z + threads.z - 1) / threads.z);

    PopulateVoxels << <dim, threads >> > (temp, scaled_size);
    cudaDeviceSynchronize();
    cudaMemcpy(voxels.grid.Raw(), temp.Raw(), temp.ByteSize(), cudaMemcpyDeviceToHost);
    cudaFree(temp.Raw());

    return voxels;
}