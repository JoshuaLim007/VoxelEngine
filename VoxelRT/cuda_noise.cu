#include "cuda_noise.cuh"

// Define the constant memory variable
__device__ __constant__ float gradMap[16][3] = {{1.0f, 1.0f, 0.0f},   {-1.0f, 1.0f, 0.0f},  {1.0f, -1.0f, 0.0f},
                                                {-1.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 1.0f},   {-1.0f, 0.0f, 1.0f},
                                                {1.0f, 0.0f, -1.0f},  {-1.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 1.0f},
                                                {0.0f, -1.0f, 1.0f},  {0.0f, 1.0f, -1.0f},  {0.0f, -1.0f, -1.0f}};
