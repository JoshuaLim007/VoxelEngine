#pragma once
#include <string>
#include <SDL.h>
#include <vector>
#include <thread>
#include <functional>
#undef main

struct PixelData {
	uint8_t b, g, r, a;
};
class Renderer;
struct CallbackData {
	Renderer* renderer;
	PixelData* pixels;
};
class Renderer {
	bool closing = false;
	std::string title{};
	SDL_Window* window = nullptr;
	SDL_Renderer* renderer = nullptr;
	std::vector<std::thread> threads{};
	SDL_Texture* tex = nullptr;
	std::vector<std::function<void(const CallbackData&)>> callbacks;
	std::vector<std::function<void(const CallbackData&)>> updateCallbacks;
	int w = 0, h = 0;
	double RenderFrameTime = 0.0;
public:
	friend void RenderThread(Renderer*);
	float texScale = 1.0f;
	Renderer(std::string);
	double GetRenderFrameTime();
	void AddRenderEventCallback(std::function<void(const CallbackData&)>);
	void AddUpdateEventCallback(std::function<void(const CallbackData&)>);
	~Renderer();
	bool init(int width, int height);
	void close();
	bool Tick();
};
