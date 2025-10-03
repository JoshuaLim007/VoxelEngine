#pragma once
#include <SDL.h>
#include <functional>
#include <string>
#include <vector>
#undef main

struct PixelData
{
    uint8_t b, g, r, a;
};
class Renderer;
struct CallbackData
{
    Renderer *renderer;
    PixelData *pixels;
};
class Renderer
{
    bool closing = false;
    std::string title{};
    SDL_Window *window = nullptr;
    SDL_Renderer *renderer = nullptr;
    SDL_Texture *tex = nullptr;
    std::vector<std::function<void(const CallbackData &)>> callbacks;
    int w = 0, h = 0;

  public:
    inline SDL_Window *GetWindow()
    {
        return window;
    }
    float texScale = 1.0f;
    Renderer(std::string);
    void AddRenderEventCallback(std::function<void(const CallbackData &)>);
    ~Renderer();
    bool Init(int width, int height);
    void Close();
    bool Render();
};
