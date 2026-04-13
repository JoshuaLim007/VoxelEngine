#include "VoxelWorldBuilder.cuh"
#include "cuda_noise.cuh"

__device__ float PerlinNoise(float x, float y, float z)
{
    float noise = cudaNoise::repeaterPerlin(make_float3(x, y, z), 1.0f, 0x71889283, 32, 2.0f, 0.5f);
    return noise;
}

__global__ void PopulateVoxels(BitArray voxels, uint3 size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    size_t max = static_cast<size_t>(size.x) * size.y * size.z;
    float scale = 0.01;

    // 1d to 3d index
    float fx = x * scale;
    float fy = y * scale;
    float fz = z * scale;
    float t = PerlinNoise(fx, fy, fz) * 1000;
    t = fmaxf(t, 0);
    auto newIdx = x + y * size.x + z * size.x * size.y;
    newIdx = GetSampleIndex(x, y, z, size.x, size.y);
    if (y > t)
    {
        voxels[newIdx] = (0);
    }
    else
    {
        voxels[newIdx] = (1);
    }
}
