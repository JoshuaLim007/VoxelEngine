// DDARaytracer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "../SDLRenderer/SDLRenderer.h"
#include "../GPUDDA/DDA.cuh"
#include "../GPUDDA/Raytracer.cuh"
#include "../GPUDDA/VoxelWorldBuilder.cuh"
#include <fstream>
#include <sstream>

using namespace GPUDDA::Graphics;
using namespace GPUDDA;

constexpr uint32_t width = 1920;
constexpr uint32_t height = 1080;

VoxelBuffer<3> CreateVoxels(uint3 size) {
	VoxelBuffer<3> voxels;
	voxels.dimensions[0] = size.x;
	voxels.dimensions[1] = size.y;
	voxels.dimensions[2] = size.z;
	size_t buffer_size = static_cast<size_t>(size.x) * size.y * size.z;
	voxels.grid = BitArray(buffer_size);

	BitArray temp = BitArray(buffer_size, true);
	auto threads = dim3(8, 8, 8);
	auto scaled_size = make_uint3(size.x, size.y, size.z);
	auto dim = dim3(
		(scaled_size.x / 8 + threads.x - 1) / threads.x, 
		(scaled_size.y + threads.y - 1) / threads.y,
		(scaled_size.z + threads.z - 1) / threads.z);

	PopulateVoxels << <dim, threads >> > (temp, scaled_size);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	cudaMemcpy(voxels.grid.raw(), temp.raw(), temp.byte_size(), cudaMemcpyDeviceToHost);
	cudaFree(temp.raw());

	return voxels;
}
int main()
{
	//TODO: goal, 128k x 512 x 128k
	int factor = 32;
	auto t0 = std::chrono::high_resolution_clock::now();
	auto buffer = CreateVoxels(make_uint3(512, 512, 512));
	auto t1 = std::chrono::high_resolution_clock::now();
	auto td = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
	std::cout << "Voxel generation time: " << td << "ms" << std::endl;

	auto t2 = std::chrono::high_resolution_clock::now();
	auto buffers = createBuffersFromVoxels(buffer, factor);
	auto t3 = std::chrono::high_resolution_clock::now();
	auto td2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
	std::cout << "Buffer generation time: " << td2 << "ms" << std::endl;

	delete[] buffer.grid.raw();
	Renderer renderer("SDL Window");
	if (!renderer.init(width, height)) {
		return 1;
	}

	VoxelRaytracer3D* raytracer = new GPUDDA::VoxelRaytracer3D(1);
	auto low_res_buffer = std::get<0>(buffers);
	auto low_res_grid_data = std::get<1>(buffers);
	auto bounds = std::get<2>(buffers);
	auto count = low_res_buffer.dimensions[0] * low_res_buffer.dimensions[1] * low_res_buffer.dimensions[2];
	raytracer->UploadVoxelBuffer(low_res_buffer);
	raytracer->UploadVoxelBufferDatas(low_res_grid_data, count);
	raytracer->UploadVoxelBufferDataBounds(bounds, count);
	raytracer->SetFactor(factor);

	void* d_pixels;
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

	cudaMalloc(&d_pixels, width * height * sizeof(PixelData));
	cudaMemset(d_pixels, 255, width * height * sizeof(PixelData));
	bool clicking = false;
	renderer.AddRenderEventCallback([&](const CallbackData& data) {
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
				} else if (e.wheel.y < 0) { // scroll down
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

			//cam_pos = { 138.252, 101.042, 503.197 };
			//cam_eular = { -0.522997, 0.512, 0 };
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

		std::cout << "Cam Forward: " << cam_forward.x << ", " << cam_forward.y << ", " << cam_forward.z << std::endl;

		getDirections(cam_eular, &cam_forward, &cam_up, &cam_right);
		RaytraceScreen(raytracer, width, height, d_pixels, cam_pos, cam_forward, cam_up, cam_right);
		cudaMemcpy(data.pixels, d_pixels, width * height * sizeof(PixelData), cudaMemcpyDeviceToHost);
		});

	bool running = true;
	double avgFrameTime = 0.0f;
	while (running) {
		auto t0 = std::chrono::high_resolution_clock::now();
		running = renderer.render();
		auto t1 = std::chrono::high_resolution_clock::now();
		auto td = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

		avgFrameTime = avgFrameTime * (1 - (1 / 1000.0)) + td * (1 / 1000.0);

		printf("Frame time: %dms\n", td);
		printf("Avg Frame time: %dms\n", avgFrameTime);

		auto fps = 1000.0f / td;
		auto avgfps = 1000.0f / avgFrameTime;
		std::cout << "FPS: " << fps << std::endl;
		std::cout << "Avg FPS: " << avgfps << std::endl;

		std::stringstream stream;
		stream << "Avg FPS: " << avgfps << "\n";
		std::string temp = stream.str();

		auto window = renderer.GetWindow();
		SDL_SetWindowTitle(window, temp.c_str());
	}
	cudaFree(d_pixels);
	delete raytracer;
}
