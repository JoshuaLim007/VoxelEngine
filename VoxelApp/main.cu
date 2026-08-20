// DDARaytracer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "VolumeRaytracer.cuh"
#include "Renderer.cuh"
#include "SDLRenderer.h"
#include "ChunkStreamingManager.cuh"
#include <fstream>
#include <iostream>
#include <sstream>
#include "Logger.h"

constexpr uint32_t width = 1280;
constexpr uint32_t height = 720;

int main()
{
    constexpr int factor = GPUDDA::STREAM_BRICK_DIM;
    Renderer renderer("SDL Window");
    if (!renderer.Init(width, height, 1.0f))
    {
        return 1;
    }
    Logger::getInstance().SetInterval(0.25f);
    GPUDDA::VoxelRaytracer3D *raytracer = new GPUDDA::VoxelRaytracer3D(1);

    // Streaming manager owns all GPU voxel memory
    GPUDDA::ChunkStreamingManager streamingMgr;
    const uint32_t brick_words =
        ((uint32_t)factor * (uint32_t)factor * (uint32_t)factor + 31u) / 32u;
    streamingMgr.Init(raytracer, brick_words);

    void *d_pixels;
    float3 cam_pos     = {0, 400, 0};
    float3 cam_up      = {0, 1, 0};
    float3 cam_right   = {1, 0, 0};
    float3 cam_forward = {0, 0, 1};
    float3 cam_eular   = {0, 0, 0};

    // Block until enough chunks around the camera are loaded to show something
    streamingMgr.WaitForInitialChunks(raytracer, cam_pos, /*min_chunks=*/4);

    GPUDDA::Graphics::Environment env{};
    env.LightDirection = {1, 1, 1};
    env.LightDirection = normalize(env.LightDirection);
    env.LightColor = {2, 2, 2};
    env.AmbientColor = {0.5f, 0.5f, 0.5f};
    GPUDDA::Graphics::SetEnvironment(env);
    GPUDDA::Graphics::SetFOV(90);
    auto orthoWindowSize = make_float2(10, 10);
    GPUDDA::Graphics::SetOrthoWindowSize(orthoWindowSize);
    auto minFloatingOriginBounds = make_float3(
        0,
        0,
        0);
    auto maxFloatingOriginBounds = make_float3(
        minFloatingOriginBounds.x + GPUDDA::SC_VOXEL_DIM,
        100000.0f,
        minFloatingOriginBounds.z + GPUDDA::SC_VOXEL_DIM);
    GetGlobalRegistry().set("constantOriginOffset", make_float3(GPUDDA::WORLD_SC_X / 2 * GPUDDA::SC_VOXEL_DIM, 0, GPUDDA::WORLD_SC_Z / 2 * GPUDDA::SC_VOXEL_DIM));
	GPUDDA::Graphics::EnableFloatingOrigin(true, minFloatingOriginBounds, maxFloatingOriginBounds);
    cudaMalloc(&d_pixels, width * height * sizeof(PixelData));
    cudaMemset(d_pixels, 255, width * height * sizeof(PixelData));
    bool clicking = false;
    
    Keyboard keyboard{};

    renderer.AddRenderEventCallback([&](const CallbackData &data) {
        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT)
            {
                data.renderer->Close();
            }
            if (e.type == SDL_MOUSEBUTTONDOWN)
            {
                if (e.button.button == SDL_BUTTON_LEFT)
                {
                    clicking = true;
                }
            }
            if (e.type == SDL_MOUSEBUTTONUP)
            {
                if (e.button.button == SDL_BUTTON_LEFT)
                {
                    clicking = false;
                }
            }
        }
        keyboard.update();

        float cam_speed = 0.2;
        const Uint8 *currentKeyStates = SDL_GetKeyboardState(NULL);
        // shift
        if (keyboard.held(SDL_SCANCODE_LSHIFT))
        {
            cam_speed *= 10;
        }
        
        if (keyboard.held(SDL_SCANCODE_W))
        {
            cam_pos += cam_forward * cam_speed;
        }
        if (keyboard.held(SDL_SCANCODE_S))
        {
            cam_pos -= cam_forward * cam_speed;
        }

        if (keyboard.held(SDL_SCANCODE_A))
        {
            cam_pos -= cam_right * cam_speed;
        }
        if (keyboard.held(SDL_SCANCODE_D))
        {
            cam_pos += cam_right * cam_speed;
        }
        if (keyboard.held(SDL_SCANCODE_Q))
        {
            cam_pos -= make_float3(0,1,0) * cam_speed;
        }
        if (keyboard.held(SDL_SCANCODE_E))
        {
            cam_pos += make_float3(0, 1, 0) * cam_speed;
        }

        //std::cout << "Cam pos: " << cam_pos.x << ", " << cam_pos.y << ", " << cam_pos.z << std::endl;
        //std::cout << "Cam eular: " << cam_eular.x << ", " << cam_eular.y << ", " << cam_eular.z << std::endl;

        static int last_x = 0, last_y = 0;
        int x, y;
        SDL_GetMouseState(&x, &y);
        if (clicking)
        {
            // mouse movement
            int dx = x - last_x;
            int dy = y - last_y;
            cam_eular.x += dy * 0.004f;
            cam_eular.y += dx * 0.004f;
        }
        last_x = x;
        last_y = y;

        //std::cout << "Cam Forward: " << cam_forward.x << ", " << cam_forward.y << ", " << cam_forward.z << std::endl;

        GPUDDA::Graphics::GetDirections(cam_eular, &cam_forward, &cam_up, &cam_right);
        streamingMgr.UpdateCamera(cam_pos, cam_forward);
        streamingMgr.FlushUploads(raytracer);
        GPUDDA::Graphics::RenderScreenFast(raytracer, width, height, d_pixels, cam_pos, cam_forward, cam_up, cam_right);
        cudaMemcpy(data.pixels, d_pixels, width * height * sizeof(PixelData), cudaMemcpyDeviceToHost);
    });

    bool running = true;
    double avgFrameTime = 0.0f;
    while (running)
    {
		double frameTime = 0.0f;
        running = renderer.Render(frameTime);
		Logger::getInstance().write("Frame time", std::to_string(frameTime * 1000.0f) + " ms");
		Logger::getInstance().write("FPS", std::to_string(1.0f / frameTime));
        Logger::getInstance().update();
    }
    cudaFree(d_pixels);
    delete raytracer;
}
