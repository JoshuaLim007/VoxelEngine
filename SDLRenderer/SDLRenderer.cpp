// SDLRenderer.cpp : Defines the functions for the static library.
//

#include "SDLRenderer.h"
#include <SDL.h>
#include <functional>
#undef main

Renderer::Renderer(std::string title) : title(title) {
}
bool Renderer::init(int width, int height) {
	w = width;
	h = height;
    int val = SDL_Init(SDL_INIT_VIDEO);
    // Create a window
    window = SDL_CreateWindow(
        title.c_str(),  // Window title
        SDL_WINDOWPOS_CENTERED, // X position (centered)
        SDL_WINDOWPOS_CENTERED, // Y position (centered)
        width,                    // Width
        height,                    // Height
        SDL_WINDOW_SHOWN        // Flags
    );

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    tex = SDL_CreateTexture(renderer, SDL_PixelFormatEnum::SDL_PIXELFORMAT_ARGB8888, SDL_TextureAccess::SDL_TEXTUREACCESS_STREAMING, width * texScale, height * texScale);
    SDL_SetTextureScaleMode(tex, SDL_ScaleMode::SDL_ScaleModeLinear);
    SDL_RenderSetVSync(renderer, 0);
    return val == 0;
}
void Renderer::AddRenderEventCallback(std::function<void(const CallbackData&)> callback) {
	callbacks.push_back(callback);
}
void Renderer::AddUpdateEventCallback(std::function<void(const CallbackData&)> callback) {
    updateCallbacks.push_back(callback);
}
Renderer::~Renderer() {
    close();
    for (auto& thread : threads) {
        if(thread.joinable() == false) continue;
        thread.join();
	}
    threads.clear();
	SDL_DestroyWindow(window);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyTexture(tex);
	SDL_Quit();
}
void Renderer::close() {
    closing = true;
}
double Renderer::GetRenderFrameTime() {
    return RenderFrameTime;
}
void RenderThread(Renderer* renderer) {
    while (!renderer->closing) {
        auto t0 = std::chrono::high_resolution_clock::now();
        void* pixels;
        int pitch = 0;
        SDL_LockTexture(renderer->tex, NULL, &pixels, &pitch);
        CallbackData data;
        data.renderer = renderer;
        data.pixels = reinterpret_cast<PixelData*>(pixels);
        auto callbacks = renderer->callbacks;
        for (auto& callback : callbacks) {
			callback(data);
        }
        SDL_Rect destRect = { 0, 0, renderer->w, renderer->h };
        SDL_UnlockTexture(renderer->tex);
        SDL_RenderCopy(renderer->renderer, renderer->tex, NULL, &destRect);
        SDL_RenderPresent(renderer->renderer);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;
        renderer->RenderFrameTime = elapsed;
    }
}
bool Renderer::Tick() {
    if (threads.size() == 0) {
        threads.push_back(std::thread(RenderThread, this));
    }
	bool quit = false;
    CallbackData data;
    data.renderer = this;
	for (auto& callback : updateCallbacks) {
		callback(data);
	}
    if (closing == true) {
        //wait for all threads to finish
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
    return !closing;
}