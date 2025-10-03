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
Renderer::~Renderer() {
	SDL_DestroyWindow(window);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyTexture(tex);
	SDL_Quit();
}
void Renderer::close() {
    closing = true;
}
bool Renderer::render() {

	bool quit = false;
    void* pixels;
    int pitch = 0;
    SDL_LockTexture(tex, NULL, &pixels, &pitch);
    CallbackData data;
    data.renderer = this;
	data.pixels = reinterpret_cast<PixelData*>(pixels);
	for (auto& callback : callbacks) {
		callback(data);
	}
    SDL_UnlockTexture(tex);
    SDL_Rect destRect = { 0, 0, w, h };
    SDL_RenderCopy(renderer, tex, NULL, &destRect);
    SDL_RenderPresent(renderer);

    return !closing;
}