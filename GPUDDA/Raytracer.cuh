#ifndef __GPUDDA_RAYTRACER_CUH__
#define __GPUDDA_RAYTRACER_CUH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <chrono>
#include <iostream>
#include "helper_math.h"
#include "DDA.cuh"

//#define DEBUG_VIEW
//#define ORTHO

#define CUDA_SAFE_CALL(x) { \
	cudaError_t err = x; \
	if (err != cudaSuccess) { \
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
		std::cerr << "Error in file: " << __FILE__ << std::endl; \
		std::cerr << "Error in line: " << __LINE__ << std::endl; \
		exit(EXIT_FAILURE); \
	} \
}

namespace GPUDDA
{
	namespace Graphics
	{
		struct RenderParams {
			VoxelRaytracer3D* Raytracer;
			uint32_t screen_width;
			uint32_t screen_height;
			
			void* d_screen_texture;
			void* d_normal_texture;
			void* d_depth_texture;
			void* d_sm_texture;

			float3 origin;
			float3 camera_fwd;
			float3 camera_up;
			float3 camera_right;
		};

		struct BGRA8888 {
			uint8_t b, g, r, a;
		};
		typedef uint32_t Normal1010102;
		typedef float Depth32f;
		typedef uint32_t S16fM16f;

		struct Environment {
			float3 LightDirection;
			float3 LightColor;
			float3 AmbientColor;
		};

		__host__ __device__ void getDirections(float3 eularAngles, float3* forwad, float3* up, float3* right);

		void SetEnvironment(const Environment &env);

		void SetFOV(float fov);

		void SetOrthoWindowSize(float2 windowSize);

		void RaytraceScreen(const RenderParams&);

		void AllocateScreenColorTexture(uint32_t screen_width, uint32_t screen_height, void** d_screen_texture);
		void AllocateNormalTexture(uint32_t screen_width, uint32_t screen_height, void** d_screen_texture);
		void AllocateDepthTexture(uint32_t screen_width, uint32_t screen_height, void** d_screen_texture);
		void AllocateSMTexture(uint32_t screen_width, uint32_t screen_height, void** d_screen_texture);
		void ClearDepthTexture(void* d_depth_texture, uint32_t screen_width, uint32_t screen_height);
	}
}
#endif