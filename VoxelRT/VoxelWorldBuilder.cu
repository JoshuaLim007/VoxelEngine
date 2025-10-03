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

    x *= 8;
    size_t idx = static_cast<size_t>(z) * size.x * size.y + static_cast<size_t>(y) * size.x + x;

    size_t max = static_cast<size_t>(size.x) * size.y * size.z;
    for (size_t i = 0; i < 8; i++)
    {
        if (idx + i >= max)
        {
            return;
        }
        float scale = 0.002;

        float3 offsets[] = {make_float3(1.5f, 0, 0),     make_float3(0, 1.5f, 0),  make_float3(0, 0, 1.5f),
                            make_float3(-1.5f, 0, 1.5f), make_float3(0, -1.5f, 0), make_float3(0, 0, -1.5f)};
        bool nextToAir = false;
        for (int j = 0; j < 6; j++)
        {
            int x = (idx + i) % size.x;
            int y = ((idx + i) / size.x) % size.y;
            int z = (idx + i) / (size.x * size.y);
            x += offsets[j].x;
            y += offsets[j].y;
            z += offsets[j].z;
            float fx = x * scale;
            float fy = y * scale;
            float fz = z * scale;
            float t = PerlinNoise(fx, fy, fz) * 1000;
            t = fmaxf(t, 0);
            bool isAir = y > t;
            if (isAir)
            {
                nextToAir = true;
                break;
            }
        }

        if (nextToAir)
        {
            // 1d to 3d index
            int x = (idx + i) % size.x;
            int y = ((idx + i) / size.x) % size.y;
            int z = (idx + i) / (size.x * size.y);
            float fx = x * scale;
            float fy = y * scale;
            float fz = z * scale;
            float t = PerlinNoise(fx, fy, fz) * 1000;
            t = fmaxf(t, 0);
            if (y > t)
            {
                voxels[idx + i] = (0);
            }
            else
            {
                voxels[idx + i] = (1);
            }
        }
        else
        {
            voxels[idx + i] = (0);
        }
    }
}
