#include "VoxelWorldBuilder.cuh"
#include "cuda_noise.cuh"
#define DEBUG

__device__ float PerlinNoise(float x, float y, float z) {
	float noise = cudaNoise::repeaterPerlin(make_float3(x, y, z), 1.0f, 0x71889283, 32, 2.0f, 0.5f);
	return noise;
}

__global__ void PopulateVoxels(BitArray voxels, uint3 size) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	x *= 8;
	size_t idx = static_cast<size_t>(z) * size.x * size.y + static_cast<size_t>(y) * size.x + x;

	size_t max = static_cast<size_t>(size.x) * size.y * size.z;
	for (size_t i = 0; i < 8; i++)
	{
		if (idx + i >= max) {
			return;
		}
		float scale = 0.002;
		int xSpheres = 8;
		int ySpheres = 8;
		int zSpheres = 8;

		float3 offsets[] = {
			make_float3(1, 0, 0),
			make_float3(0, 1, 0),
			make_float3(0, 0, 1),
			make_float3(-1, 0, 1),
			make_float3(0, -1, 0),
			make_float3(0, 0, -1)
		};
		bool nextToAir = false;
		for (int j = 0; j < 6; j++) {
			int x = (idx + i) % size.x;
			int y = ((idx + i) / size.x) % size.y;
			int z = (idx + i) / (size.x * size.y);
			x += offsets[j].x * 1.5f;
			y += offsets[j].y * 1.5f;
			z += offsets[j].z * 1.5f;

#ifndef DEBUG
			float fx = x * scale;
			float fy = y * scale;
			float fz = z * scale;
			float t = PerlinNoise(fx, fy, fz) * 1000;
			t = fmaxf(t, 0);
			bool isAir = t > 0.9f;
			if (isAir) {
				nextToAir = true;
				break;
			}
#endif


			float fx = x;
			float fy = y;
			float fz = z;

			float3 sphereSize = make_float3((float)size.x / xSpheres, (float)size.y / ySpheres, (float)size.z / zSpheres);

			float tx = floorf(fx / sphereSize.x) * sphereSize.x + (sphereSize.x / 2.0f);
			float ty = floorf(fy / sphereSize.y) * sphereSize.y + (sphereSize.y / 2.0f);
			float tz = floorf(fz / sphereSize.z) * sphereSize.z + (sphereSize.z / 2.0f);

			float3 sphereCenter = make_float3(tx, ty, tz);
			float dist = length(make_float3(x, y, z) - sphereCenter);
			if (dist > fminf(fminf(sphereSize.x, sphereSize.y), sphereSize.z) / 4.0f) {
				nextToAir = true;
				break;
			}
		}

		if (nextToAir) {

			//1d to 3d index
			int x = (idx + i) % size.x;
			int y = ((idx + i) / size.x) % size.y;
			int z = (idx + i) / (size.x * size.y);
#ifndef DEBUG
			float fx = x * scale;
			float fy = y * scale;
			float fz = z * scale;
			float t = PerlinNoise(fx, fy, fz) * 1000;
			t = fmaxf(t, 0);
			if (t > 0.999f) {
				voxels[idx + i] = (0);
			}
			else {
				voxels[idx + i] = (1);
			}
#endif


			float fx = x;
			float fy = y;
			float fz = z;

			float3 sphereSize = make_float3((float)size.x / xSpheres, (float)size.y / ySpheres, (float)size.z / zSpheres);

			float tx = floorf(fx / sphereSize.x) * sphereSize.x + (sphereSize.x / 2.0f);
			float ty = floorf(fy / sphereSize.y) * sphereSize.y + (sphereSize.y / 2.0f);
			float tz = floorf(fz / sphereSize.z) * sphereSize.z + (sphereSize.z / 2.0f);

			float3 sphereCenter = make_float3(tx, ty, tz);
			float dist = length(make_float3(x, y, z) - sphereCenter);
			if (dist > fminf(fminf(sphereSize.x, sphereSize.y), sphereSize.z) / 4.0f) {
				voxels[idx + i] = (0);
			}
			else {
				voxels[idx + i] = (1);
			}
		}
		else {
			voxels[idx + i] = (0);
		}

	}
}