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
    double deltaTime;
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
	double lastFrameTime = 0.0;

  public:
    inline SDL_Window *GetWindow()
    {
        return window;
    }
    Renderer(std::string);
    void AddRenderEventCallback(std::function<void(const CallbackData &)>);
    ~Renderer();
    bool Init(int width, int height, float scale);
    void Close();
    bool Render(double& frameTime);
};

#include <array>
class Keyboard
{
public:
    void update()
    {
        previous = current;

        const Uint8* state = SDL_GetKeyboardState(nullptr);

        std::copy(
            state,
            state + SDL_NUM_SCANCODES,
            current.begin()
        );
    }

    // Key was pressed this frame
    bool down(SDL_Scancode key) const
    {
        return current[key] && !previous[key];
    }

    // Key was released this frame
    bool up(SDL_Scancode key) const
    {
        return !current[key] && previous[key];
    }

    // Key is currently being held
    bool held(SDL_Scancode key) const
    {
        return current[key];
    }

private:
    std::array<Uint8, SDL_NUM_SCANCODES> current{};
    std::array<Uint8, SDL_NUM_SCANCODES> previous{};
};
