// DDARaytracer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "../SDLRenderer/SDLRenderer.h"
#include "../GPUDDA/DDA.cuh"
#include "../GPUDDA/Raytracer.cuh"
#include "../GPUDDA/VoxelWorldBuilder.cuh"
#include <fstream>
#include <sstream>

#define SWIDTH 1280
#define SHEIGHT 720

using namespace GPUDDA::Graphics;
using namespace GPUDDA;

VoxelBuffer<3> CreateVoxels(uint3 size, float3 originOffset) {
	VoxelBuffer<3> voxels;
	voxels.dimensions[0] = size.x;
	voxels.dimensions[1] = size.y;
	voxels.dimensions[2] = size.z;
	size_t buffer_size = static_cast<size_t>(size.x) * size.y * size.z;
	voxels.grid = BitArray(buffer_size);

	BitArray temp = BitArray(buffer_size, true);
	auto threads = dim3(16, 8, 8);
	auto scaled_size = make_uint3(size.x, size.y, size.z);
	auto dim = dim3(
		(scaled_size.x / 8 + threads.x - 1) / threads.x, 
		(scaled_size.y + threads.y - 1) / threads.y,
		(scaled_size.z + threads.z - 1) / threads.z);

	PopulateVoxels << <dim, threads >> > (temp, scaled_size, originOffset);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	cudaMemcpy(voxels.grid.raw(), temp.raw(), temp.byte_size(), cudaMemcpyDeviceToHost);
	cudaFree(temp.raw());

	return voxels;
}


struct RenderEntities {
	VoxelRaytracer3D* raytracer;
	VoxelBuffer<3> voxelBuffer;
	VoxelBuffer<3>* voxelBufferData;
	int voxelBufferDataCount;
	Bounds<float3>* voxelBufferDataBounds;
	int voxelBufferDataBoundsCount;
	int factor;
	float3 originOffset;
	void Free() {
		if (raytracer) {
			raytracer->Free();
			delete raytracer;
		}
		delete voxelBuffer.grid.raw();
		for (size_t i = 0; i < voxelBufferDataCount; i++)
		{
			delete voxelBufferData[i].grid.raw();
		}
		delete[] voxelBufferData;
		delete[] voxelBufferDataBounds;
	}
};

int main()
{
	int chunks = 9;
	float3 offsets[] = {
		make_float3(0,0,0),

		make_float3(1,0,0),
		make_float3(0,0,1),
		make_float3(1,0,1),

		make_float3(-1,0,0),
		make_float3(0,0,-1),
		make_float3(-1,0,-1),

		make_float3(1,0,-1),
		make_float3(-1,0,1),
	};

	std::vector<RenderEntities> renderEntities;

	for (size_t i = 0; i < chunks; i++)
	{
		float3 originOffset = make_float3(512, 0, 512) * offsets[i];
		//TODO: goal, 128k x 512 x 128k
		int targetRatio = 16;
		auto t0 = std::chrono::high_resolution_clock::now();
		auto buffer = CreateVoxels(make_uint3(512, 512, 512), originOffset);
		int factor = 512 / targetRatio;
		auto t1 = std::chrono::high_resolution_clock::now();
		auto td = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
		std::cout << "Voxel generation time: " << td << "ms" << std::endl;

		auto t2 = std::chrono::high_resolution_clock::now();
		auto buffers = createBuffersFromVoxels(buffer, factor);
		auto t3 = std::chrono::high_resolution_clock::now();
		auto td2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
		std::cout << "Buffer generation time: " << td2 << "ms" << std::endl;

		auto low_res_buffer = std::get<0>(buffers);
		auto low_res_grid_data = std::get<1>(buffers);
		auto bounds = std::get<2>(buffers);
		auto count = low_res_buffer.dimensions[0] * low_res_buffer.dimensions[1] * low_res_buffer.dimensions[2];

		RenderEntities temp;
		temp.factor = factor;
		temp.voxelBuffer = low_res_buffer;
		temp.voxelBufferData = low_res_grid_data;
		temp.voxelBufferDataCount = count;
		temp.voxelBufferDataBounds = bounds;
		temp.voxelBufferDataBoundsCount = count;
		temp.originOffset = originOffset;

		delete[] buffer.grid.raw();
	
		VoxelRaytracer3D* raytracer = new GPUDDA::VoxelRaytracer3D();
		temp.raytracer = raytracer;

		renderEntities.push_back(temp);
	}


	Renderer renderer("SDL Window");
	if (!renderer.init(SWIDTH, SHEIGHT)) {
		return 1;
	}

	void* d_pixels;
	void* d_normals;
	void* d_depth;
	void* d_s16fM16f;

	float3 cam_pos = { 0, 0, 0 };
	float3 cam_up = { 0, 1, 0 };
	float3 cam_right = { 1, 0, 0 };
	float3 cam_forward = { 0, 0, 1 };
	float3 cam_eular = { 0, 0, 0 };

	Graphics::Environment env;
	env.LightDirection = { 1, 1, 1};
	env.LightDirection = normalize(env.LightDirection);
	env.LightColor = { 2, 2, 2 };
	env.AmbientColor = { 0.5f, 0.5f, 0.5f };
	SetEnvironment(env);
	SetFOV(90);
	auto orthoWindowSize = make_float2(10, 10);
	SetOrthoWindowSize(orthoWindowSize);

	Graphics::AllocateScreenColorTexture(SWIDTH, SHEIGHT, &d_pixels);
	Graphics::AllocateNormalTexture(SWIDTH, SHEIGHT, &d_normals);
	Graphics::AllocateDepthTexture(SWIDTH, SHEIGHT, &d_depth);
	Graphics::AllocateSMTexture(SWIDTH, SHEIGHT, &d_s16fM16f);
		
	bool clicking = false;
	renderer.AddUpdateEventCallback([&](const CallbackData& data) {
		auto inputT0 = std::chrono::high_resolution_clock::now();

		SDL_Event e;
		while (SDL_PollEvent(&e)) {
			if (e.type == SDL_QUIT) {
				data.renderer->close();
			}
			if (e.type == SDL_MOUSEBUTTONDOWN) {
				if (e.button.button == SDL_BUTTON_LEFT) {
					clicking = true;
				}
			}
			if (e.type == SDL_MOUSEBUTTONUP) {
				if (e.button.button == SDL_BUTTON_LEFT) {
					clicking = false;
				}
			}
			if (e.type == SDL_MOUSEWHEEL) {
				if (e.wheel.y > 0) { // scroll up
					orthoWindowSize.x -= 10;
					orthoWindowSize.y -= 10;
				}
				else if (e.wheel.y < 0) { // scroll down
					orthoWindowSize.x += 10;
					orthoWindowSize.y += 10;
				}
				SetOrthoWindowSize(orthoWindowSize);
			}

		}

		float cam_speed = 0.2;
		const Uint8* currentKeyStates = SDL_GetKeyboardState(NULL);
		//shift
		if (currentKeyStates[SDL_SCANCODE_LSHIFT]) {
			cam_speed *= 10;
		}
		if (currentKeyStates[SDL_SCANCODE_W]) {
			cam_pos += cam_forward * cam_speed;
		}
		if (currentKeyStates[SDL_SCANCODE_S]) {
			cam_pos -= cam_forward * cam_speed;
		}

		SetOrthoWindowSize(orthoWindowSize);

		if (currentKeyStates[SDL_SCANCODE_A]) {
			cam_pos -= cam_right * cam_speed;
		}
		if (currentKeyStates[SDL_SCANCODE_D]) {
			cam_pos += cam_right * cam_speed;
		}
		if (currentKeyStates[SDL_SCANCODE_Q]) {
			cam_pos -= cam_up * cam_speed;
		}
		if (currentKeyStates[SDL_SCANCODE_E]) {
			cam_pos += cam_up * cam_speed;
		}
		if (currentKeyStates[SDL_SCANCODE_R]) {
			//read from text file 
			std::fstream file("C:\\Users\\joshu\\Desktop\\camData.txt", std::ios::in);
			//first line contains rotation
			std::string line;
			std::getline(file, line);
			std::istringstream iss(line);
			std::string token;
			std::getline(iss, token, ',');
			cam_eular.x = std::stof(token);
			std::getline(iss, token, ',');
			cam_eular.y = std::stof(token);
			std::getline(iss, token, ',');
			cam_eular.z = std::stof(token);
			//second line contains position
			std::getline(file, line);
			std::istringstream iss2(line);
			std::getline(iss2, token, ',');
			cam_pos.x = std::stof(token);
			std::getline(iss2, token, ',');
			cam_pos.y = std::stof(token);
			std::getline(iss2, token, ',');
			cam_pos.z = std::stof(token);
			file.close();
		}

		std::cout << "Cam pos: " << cam_pos.x << ", " << cam_pos.y << ", " << cam_pos.z << std::endl;
		std::cout << "Cam eular: " << cam_eular.x << ", " << cam_eular.y << ", " << cam_eular.z << std::endl;

		static int last_x = 0, last_y = 0;
		int x, y;
		SDL_GetMouseState(&x, &y);
		if (clicking) {
			//mouse movement
			int dx = x - last_x;
			int dy = y - last_y;
			cam_eular.x += dy * 0.001f;
			cam_eular.y += dx * 0.001f;
		}
		last_x = x;
		last_y = y;

		auto inputT1 = std::chrono::high_resolution_clock::now();
		auto inputTd = std::chrono::duration_cast<std::chrono::microseconds>(inputT1 - inputT0).count() / 1000.0f;
		getDirections(cam_eular, &cam_forward, &cam_up, &cam_right);
		printf("Input time: %fms\n", inputTd);
	});
	ClearDepthTexture(d_depth, SWIDTH, SHEIGHT);

	renderer.AddRenderEventCallback([&](const CallbackData& data) {

		RenderParams params;
		params.screen_width = SWIDTH;
		params.screen_height = SHEIGHT;
		params.d_screen_texture = d_pixels;
		params.d_normal_texture = d_normals;
		params.d_depth_texture = d_depth;
		params.d_sm_texture = d_s16fM16f;
		params.camera_fwd = cam_forward;
		params.camera_up = cam_up;
		params.camera_right = cam_right;

		for (size_t i = 0; i < chunks; i++)
		{
			auto& entity = renderEntities[i];
			params.origin = cam_pos - entity.originOffset;
			params.Raytracer = entity.raytracer;
			auto raytracer = entity.raytracer;

			if (raytracer->IsUploadVoxelBufferAsyncComplete() == VoxelRaytracer3D::ASYNC_STATUS_COMPLETE) {
				raytracer->FinishAsyncUpload();
				RaytraceScreen(params);
			}
			if(raytracer->IsUploadVoxelBufferAsyncComplete() == VoxelRaytracer3D::ASYNC_STATUS_LOADING) {
				continue;
			}
			if (raytracer->IsUploadVoxelBufferAsyncComplete() == VoxelRaytracer3D::ASYNC_STATUS_IDLE) {
				raytracer->UploadBuffersAsync(
					entity.voxelBuffer, 
					entity.voxelBufferData, 
					entity.voxelBufferDataCount,
					entity.voxelBufferDataBounds,
					entity.voxelBufferDataBoundsCount
				);
				raytracer->SetFactor(entity.factor);
			}

			//raytracer->UploadVoxelBuffer(entity.voxelBuffer);
			//raytracer->UploadVoxelBufferDatas(entity.voxelBufferData, entity.voxelBufferDataCount);
			//raytracer->UploadVoxelBufferDataBounds(entity.voxelBufferDataBounds, entity.voxelBufferDataBoundsCount);
			//raytracer->SetFactor(entity.factor);
			//RaytraceScreen(params);

		}

		cudaMemcpy(data.pixels, d_pixels, SWIDTH* SHEIGHT * sizeof(PixelData), cudaMemcpyDeviceToHost);
	});

	bool running = true;
	while (running) {
		running = renderer.Tick();
		auto rT = renderer.GetRenderFrameTime();
		printf("Render frame time: %fms\n", rT);
		auto fps = 1000.0f / rT;
		std::cout << "Render FPS: " << fps << std::endl;
	}
	cudaFree(d_pixels);
	cudaFree(d_normals);
	cudaFree(d_depth);
	cudaFree(d_s16fM16f);

	for(auto & entity : renderEntities)
	{
		entity.Free();
	}

}
