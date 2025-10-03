

#include <fstream>
#include <iostream>
#include <SDL.h>
#include "../GPUDDA/DDA.cuh"
using namespace GPUDDA;

#undef main
#include <string>
#include <vector>
#include <tuple>
#include <cmath>
#include <limits>
#ifdef _DEBUG
#define RECORD_INTERSECTED_POINTS
#else
#undef RECORD_INTERSECTED_POINTS
#endif

#define RAYS 1000000

#ifdef RECORD_INTERSECTED_POINTS
std::vector<float2> interSectedPoints = {};
#endif
void DDARayTraversal(
    VoxelBuffer<2> VoxelBuffer,
    float2 start,
    float2 direction,
    float4* bounds,
    int max_steps,
    float4* per_voxel_bounds,
    int per_voxel_bounds_scale,

    //return values
    bool* out_hit,
    bool* out_isOutOfBounds,
    float2* out_HitCell,
    float2* out_HitIntersectedPoint,
    int* out_stepsTaken
) {
    float x = start.x;
    float y = start.y;
    float dx = direction.x;
    float dy = direction.y;

    int cell_x = static_cast<int>(x);
    int cell_y = static_cast<int>(y);

    int step_x = (dx > 0) ? 1 : -1;
    int step_y = (dy > 0) ? 1 : -1;

    float tDelta_x = (dx != 0) ? fabs(1.0f / dx) : std::numeric_limits<float>::infinity();
    float tDelta_y = (dy != 0) ? fabs(1.0f / dy) : std::numeric_limits<float>::infinity();

    float tMax_x = (dx != 0) ? (((cell_x + (step_x > 0)) - x) / dx) : std::numeric_limits<float>::infinity();
    float tMax_y = (dy != 0) ? (((cell_y + (step_y > 0)) - y) / dy) : std::numeric_limits<float>::infinity();

    *out_HitIntersectedPoint = make_float2(x, y);
    *out_hit = false;
    *out_isOutOfBounds = false;
    *out_stepsTaken = 0;

    int rows = VoxelBuffer.dimensions[1];
    int cols = VoxelBuffer.dimensions[0];
    auto grid = VoxelBuffer.grid;

    for (int step = 0; step < max_steps; ++step) {
        if (0 <= cell_x && cell_x < cols && 0 <= cell_y && cell_y < rows) {
            *out_HitCell = make_float2(cell_x, cell_y);
            if (per_voxel_bounds) {
                int idx = (cell_y * cols + cell_x);
                float bmin_x = per_voxel_bounds[idx].x + cell_x * per_voxel_bounds_scale;
                float bmin_y = per_voxel_bounds[idx].y + cell_y * per_voxel_bounds_scale;
                float bmax_x = per_voxel_bounds[idx].z + 1 + cell_x * per_voxel_bounds_scale;
                float bmax_y = per_voxel_bounds[idx].w + 1 + cell_y * per_voxel_bounds_scale;
                if (grid[cell_y * cols + cell_x] == 1 && bmin_x <= bmax_x) {
                    float temp_x = start.x * per_voxel_bounds_scale;
                    float temp_y = start.y * per_voxel_bounds_scale;
                    if (RayIntersectsAABB(make_float2(temp_x, temp_y), direction, make_float2(bmin_x, bmin_y), make_float2(bmax_x, bmax_y), nullptr, nullptr)) {
                        *out_hit = true;
                        break;
                    }
                }
            }
            else {
                if (grid[cell_y * cols + cell_x] == 1) {
                    *out_hit = true;
                    break;
                }
            }
        }
        else {
            *out_isOutOfBounds = true;
            break;
        }

        float intersect_x = 0;
        float intersect_y = 0;
        if (tMax_x < tMax_y) {
            intersect_x = cell_x + (step_x > 0);
            intersect_y = y + (tMax_x * dy);
            cell_x += step_x;
            tMax_x += tDelta_x;
        }
        else {
            intersect_x = x + (tMax_y * dx);
            intersect_y = cell_y + (step_y > 0);
            cell_y += step_y;
            tMax_y += tDelta_y;
        }

        if (bounds) {
            int min_x = bounds->x;
            int min_y = bounds->y;
            int max_x = bounds->z;
            int max_y = bounds->w;
            if (!(min_x <= intersect_x && intersect_x <= max_x && min_y <= intersect_y && intersect_y <= max_y)) {
                *out_isOutOfBounds = true;
                break;
            }
        }

        *out_stepsTaken += 1;
#ifdef RECORD_INTERSECTED_POINTS
        interSectedPoints.push_back(make_float2(intersect_x, intersect_y));
#endif
        *out_HitIntersectedPoint = make_float2(intersect_x, intersect_y);
    }
}

float2 Raytrace(float2 origin, float2 ray, VoxelBuffer<2> chunks, VoxelBuffer<2>* chunksData, float4* chunkBoundingBoxes, int factor) {
	float rayLen = sqrt(ray.x * ray.x + ray.y * ray.y);
    ray.x /= rayLen;
    ray.y /= rayLen;
#ifdef RECORD_INTERSECTED_POINTS
    interSectedPoints.clear();
#endif
    float2 previous_cell = make_float2(-1, -1);
    int total_steps = 0;

    //in chunk space
    float2 start = origin;// make_float2(0, 0);
	start.x /= factor;
	start.y /= factor;
    float2 direction = ray;// make_float2(1, 1);
	float eps = 1e-5;
    if (!(start.x >= 0 && start.y >= 0 && start.x < chunks.dimensions[1] && start.y < chunks.dimensions[0])) {
        float2 intersect;
        if (RayIntersectsAABB(start, direction, make_float2(0, 0), make_float2(chunks.dimensions[1], chunks.dimensions[0]), &intersect, nullptr)) {
			if (intersect.x == chunks.dimensions[0])
                intersect.x -= eps;
			if (intersect.y == chunks.dimensions[1]) 
                intersect.y -= eps;
            start = intersect;
        }
    }

    float2 hitPosition = make_float2(0,0);
    while (true) {
        bool hit = false;
        bool isOutOfBounds = false;
        float2 hitCell;
        float2 hitIntersectedPoint;
        int stepsTaken;
        float2 start_high_res;
#ifdef RECORD_INTERSECTED_POINTS
		int sidx = interSectedPoints.size();
#endif
        DDARayTraversal(chunks, start, direction, nullptr, 100, chunkBoundingBoxes, factor,
            &hit, &isOutOfBounds, &hitCell, &hitIntersectedPoint, &stepsTaken);
#ifdef RECORD_INTERSECTED_POINTS
        for (size_t i = sidx; i < interSectedPoints.size(); i++)
        {
            interSectedPoints[i].x *= factor;
            interSectedPoints[i].y *= factor;
        }
#endif
        total_steps += stepsTaken;
        start_high_res = make_float2(hitIntersectedPoint.x * factor, hitIntersectedPoint.y * factor);
		hitPosition = start_high_res;

        if (hit && !isOutOfBounds) {
            float4 chunkBounds;
            if (previous_cell.x == hitCell.x && previous_cell.y == hitCell.y) {
                break;
            }
            previous_cell = hitCell;
            chunkBounds.x = 0;
            chunkBounds.y = 0;
            chunkBounds.z = factor;
            chunkBounds.w = factor;

            bool hit1 = false;
            bool isOutOfBounds1 = false;
            float2 hitCell1;
            float2 hitIntersectedPoint1;
            int stepsTaken1;

            start_high_res.x -= hitCell.x * factor;
            start_high_res.y -= hitCell.y * factor;
            VoxelBuffer chunkData = chunksData[(int)(hitCell.y * chunks.dimensions[0] + hitCell.x)];

#ifdef RECORD_INTERSECTED_POINTS
            int sidx = interSectedPoints.size();
#endif

			if (start_high_res.x == factor)
				start_high_res.x -= eps;
			if (start_high_res.y == factor)
				start_high_res.y -= eps;

            DDARayTraversal(chunkData, start_high_res, direction, &chunkBounds, 100, nullptr, 0,
                &hit1, &isOutOfBounds1, &hitCell1, &hitIntersectedPoint1, &stepsTaken1);

#ifdef RECORD_INTERSECTED_POINTS
            for (size_t i = sidx; i < interSectedPoints.size(); i++)
            {
                interSectedPoints[i].x += hitCell.x * factor;
                interSectedPoints[i].y += hitCell.y * factor;
            }
#endif
            total_steps += stepsTaken1;

            hitPosition = hitIntersectedPoint1;
            hitPosition.x += hitCell.x * factor;
            hitPosition.y += hitCell.y * factor;

            if (!hit1) {
                start = make_float2(
                    hitIntersectedPoint1.x + eps * direction.x + hitCell.x * factor,
                    hitIntersectedPoint1.y + eps * direction.y + hitCell.y * factor
                );
				start.x /= factor;
				start.y /= factor;
                continue;
            }
            else {
                break;
            }
        }
        else {
            break;
        }
    }
    return hitPosition;
}

// Function to initialize SDL and create a window
bool initSDL(SDL_Window** window, SDL_Renderer** renderer, int width, int height) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    *window = SDL_CreateWindow("SDL Window", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);
    if (*window == nullptr) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    *renderer = SDL_CreateRenderer(*window, -1, SDL_RENDERER_ACCELERATED);
    if (*renderer == nullptr) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    return true;
}

// Function to close SDL and destroy the window
void closeSDL(SDL_Window* window, SDL_Renderer* renderer) {
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

int main() {
    int rows = 4024, cols = 4024;
    std::vector<char> grid(rows * cols);

    std::ifstream infile("C:\\Users\\joshu\\source\\repos\\PythonTest\\DDATest\\voxel_buffer.txt");
    if (!infile) {
        std::cerr << "Unable to open file voxel_buffer.txt";
        return 1;
    }
    for (int i = 0; i < rows * cols; ++i) {
        int temp;
        infile >> temp;
		grid[i] = (temp != 0);
    }
    infile.Close();

    VoxelBuffer<2>* voxelBuffer = new VoxelBuffer<2>();
    voxelBuffer->grid = BitArray(grid.size());
	for (size_t i = 0; i < grid.size(); ++i) {
		voxelBuffer->grid[i] = grid[i];
	}
    voxelBuffer->dimensions[1] = rows;
    voxelBuffer->dimensions[0] = cols;
    int factor = 8;
    auto datas = createBuffersFromVoxels(*voxelBuffer, factor);

	delete[] voxelBuffer->grid.Raw();
    delete voxelBuffer;
    grid.clear();

    auto low_res_buffer = std::get<0>(datas);
    auto low_res_grid_data = std::get<1>(datas);
    auto low_res_chunk_bounds = std::get<2>(datas);

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;

    if (!initSDL(&window, &renderer, 1920, 1080)) {
        return 1;
    }

    bool quit = false;
    SDL_Event e;

    SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
    SDL_RenderClear(renderer);

    int window_width, window_height;
    SDL_GetWindowSize(window, &window_width, &window_height);

	VoxelRaytracer2D raytracer(RAYS);
    raytracer.UploadVoxelBuffer(low_res_buffer);
    raytracer.UploadVoxelBufferDataBounds(reinterpret_cast<Bounds<float2>*>(low_res_chunk_bounds), low_res_buffer.dimensions[0] * low_res_buffer.dimensions[1]);
    raytracer.UploadVoxelBufferDatas(low_res_grid_data, low_res_buffer.dimensions[0] * low_res_buffer.dimensions[1]);
	raytracer.SetFactor(factor);

    // Add these variables at the beginning of the main function
    int offsetX = 0, offsetY = 0;
    bool dragging = false;
    int lastMouseX = 0, lastMouseY = 0;
    float2 origin = make_float2(0, 0);
    float2 ray = make_float2(1, 1);

    // Variables to track setting position and ray direction
    bool setPosition = false;
    bool setRayDirection = false;
    auto scale = 8;

    int rays = RAYS;
	std::vector<float2> rayDirections(rays);
	std::vector<float2> origins(rays);

    // Modify the event loop to handle mouse and keyboard events
    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
            else if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
                dragging = true;
                lastMouseX = e.button.x;
                lastMouseY = e.button.y;
            }
            else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {
                dragging = false;
            }
            else if (e.type == SDL_MOUSEMOTION && dragging) {
                int deltaX = e.motion.x - lastMouseX;
                int deltaY = e.motion.y - lastMouseY;
                offsetX += deltaX;
                offsetY += deltaY;
                lastMouseX = e.motion.x;
                lastMouseY = e.motion.y;
            }

            else if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_t) {
                    setPosition = true;
                }
                else if (e.key.keysym.sym == SDLK_f) {
                    setRayDirection = true;
                }
            }
			else if (e.type == SDL_KEYUP) {
				if (e.key.keysym.sym == SDLK_t) {
					setPosition = false;
				}
				else if (e.key.keysym.sym == SDLK_f) {
					setRayDirection = false;
				}
			}

            if (setPosition) {
                int mouseX, mouseY;
                SDL_GetMouseState(&mouseX, &mouseY);
                origin = make_float2((float)(mouseX - offsetX) / scale, (float)(mouseY - offsetY) / scale);
            }
			if (setRayDirection) {
				int mouseX, mouseY;
				SDL_GetMouseState(&mouseX, &mouseY);
				auto dir = make_float2((float)(mouseX - offsetX) / scale - origin.x, (float)(mouseY - offsetY) / scale - origin.y);
                ray = dir;
			}
        }


        for (int i = 0; i < rays; ++i) {
            origins[i] = origin;
            float angle = (2.0f * M_PI / rays) * i;
            rayDirections[i] = make_float2(cos(angle), sin(angle));
        }

		//auto tempPos = make_float2(origin.x + offsetX / scale, origin.y + offsetY / scale);
        //auto hitPos = Raytrace(origin, ray, low_res_buffer, low_res_grid_data, low_res_chunk_bounds, factor);
		auto gpuHitPos = raytracer.Raytrace(origins, rayDirections);
        float2 hitPos = gpuHitPos.hitPoint[0];

        SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
        SDL_RenderClear(renderer);

        int window_width, window_height;
        SDL_GetWindowSize(window, &window_width, &window_height);

        // Draw each voxel in the low res buffer as a rectangle
        for (int y = 0; y < low_res_buffer.dimensions[1]; ++y) {
            for (int x = 0; x < low_res_buffer.dimensions[0]; ++x) {
                SDL_Rect rect;
                rect.x = static_cast<int>(x * factor * scale) + offsetX;
                rect.y = static_cast<int>(y * factor * scale) + offsetY;
                rect.w = static_cast<int>(factor * scale + 1);
                rect.h = static_cast<int>(factor * scale + 1);

				if (rect.x + rect.w < 0 || rect.x > window_width || rect.y + rect.h < 0 || rect.y > window_height) {
					continue; // Skip rendering if the rectangle is outside the window
				}

                if (low_res_buffer.grid[y * low_res_buffer.dimensions[0] + x] != 1) {
                    SDL_SetRenderDrawColor(renderer, 0xAF, 0xD8, 0xE6, 0xFF); // Light pastel blue color
                    SDL_RenderFillRect(renderer, &rect);
                }

                float4 bounds = low_res_chunk_bounds[y * low_res_buffer.dimensions[0] + x];
                SDL_Rect boundsRect;

                // Draw the smaller voxels within each chunk
                VoxelBuffer<2> chunkData = low_res_grid_data[y * low_res_buffer.dimensions[0] + x];
                for (int dy = 0; dy < chunkData.dimensions[1]; ++dy) {
                    for (int dx = 0; dx < chunkData.dimensions[0]; ++dx) {
                        SDL_Rect smallRect;
                        smallRect.x = rect.x + dx * scale;
                        smallRect.y = rect.y + dy * scale;
                        smallRect.w = scale + 1;
                        smallRect.h = scale + 1;

                        //draw voxels
                        if (chunkData.grid[dy * chunkData.dimensions[0] + dx] == 1) {
                            SDL_SetRenderDrawColor(renderer, 0xFF, 0xC0, 0xCB, 0xFF); // Light pastel red color
                            SDL_RenderFillRect(renderer, &smallRect);
                            SDL_SetRenderDrawColor(renderer, 0x77, 0x77, 0x77, 0x22); // Lighter black color
                            SDL_RenderDrawRect(renderer, &smallRect);
                        }
                        else {
                            SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xE0, 0xFF); // Light pastel yellow color
                            SDL_RenderFillRect(renderer, &smallRect);
                            SDL_SetRenderDrawColor(renderer, 0x77, 0x77, 0x77, 0x22); // Lighter black color
                            SDL_RenderDrawRect(renderer, &smallRect);
                        }
                    }
                }
            }
        }

        for (int y = 0; y < low_res_buffer.dimensions[1]; ++y) {
            for (int x = 0; x < low_res_buffer.dimensions[0]; ++x) {

                SDL_Rect rect;
                rect.x = static_cast<int>(x * factor * scale) + offsetX;
                rect.y = static_cast<int>(y * factor * scale) + offsetY;
                rect.w = static_cast<int>(factor * scale + 1);
                rect.h = static_cast<int>(factor * scale + 1);

				if (rect.x + rect.w < 0 || rect.x > window_width || rect.y + rect.h < 0 || rect.y > window_height) {
					continue; // Skip rendering if the rectangle is outside the window
				}

                SDL_Rect boundsRect;
                float4 bounds = low_res_chunk_bounds[y * low_res_buffer.dimensions[0] + x];

                //draw borders around chunks
                if (low_res_buffer.grid[y * low_res_buffer.dimensions[0] + x] != 1) {
                    SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0xFF); // Black color
                    SDL_RenderDrawRect(renderer, &rect);
                }

                // Draw the bounding box of the chunk
                boundsRect.x = static_cast<int>(x * factor * scale + bounds.x * scale) + offsetX;
                boundsRect.y = static_cast<int>(y * factor * scale + bounds.y * scale) + offsetY;
                boundsRect.w = static_cast<int>((bounds.z - bounds.x + 1) * scale + 1);
                boundsRect.h = static_cast<int>((bounds.w - bounds.y + 1) * scale + 1);
                SDL_SetRenderDrawColor(renderer, 0x80, 0x00, 0x80, 0xFF); // Purple color
                SDL_RenderDrawRect(renderer, &boundsRect);

                boundsRect.x -= 1;
                boundsRect.y -= 1;
                boundsRect.w += 2;
                boundsRect.h += 2;
                SDL_RenderDrawRect(renderer, &boundsRect);
            }
        }


        // Draw the line from origin to ray * inf

        // Draw a ray for each ray in rayDirections buffer
        int is = 10000;
        for (int i = 0; i < rays; i += is) {
			if (!gpuHitPos.valid[i]) continue; // Skip if no hit
            SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0xFF, 0xFF); // Blue color
            SDL_RenderDrawLine(renderer, 
                origins[i].x * scale + offsetX, 
                origins[i].y * scale + offsetY, 
                origins[i].x * scale + offsetX + rayDirections[i].x * scale * gpuHitPos.distance[i],
                origins[i].y * scale + offsetY + rayDirections[i].y * scale * gpuHitPos.distance[i]
            );
			float2 hit = gpuHitPos.hitPoint[i];
			float2 normal = gpuHitPos.normal[i];
			SDL_SetRenderDrawColor(renderer, 0x00, 0xFF, 0x00, 0xFF); // Green color
			SDL_RenderDrawLine(renderer,
				hit.x* scale + offsetX,
				hit.y* scale + offsetY,
				hit.x* scale + offsetX - normal.x * scale * 10,
				hit.y* scale + offsetY - normal.y * scale * 10
			);
        }

        int radius = 5;

#ifdef RECORD_INTERSECTED_POINTS
        for (size_t i = 0; i < interSectedPoints.size(); i++)
        {
			auto point = interSectedPoints[i];
            for (int w = 0; w < radius * 2; w++) {
                for (int h = 0; h < radius * 2; h++) {
                    int dx = radius - w; // horizontal offset
                    int dy = radius - h; // vertical offset
                    if ((dx * dx + dy * dy) <= (radius * radius)) {
                        SDL_RenderDrawPoint(renderer, point.x * scale + dx + offsetX, point.y * scale + dy + offsetY);
                    }
                }
            }
        }
#endif
        for (size_t i = 0; i < rays; i += is)
        {
            SDL_SetRenderDrawColor(renderer, 0xFF, 0x00, 0x00, 0xFF); // Red color
            for (int w = 0; w < radius * 2; w++) {
                for (int h = 0; h < radius * 2; h++) {
                    int dx = radius - w; // horizontal offset
                    int dy = radius - h; // vertical offset
                    if ((dx * dx + dy * dy) <= (radius * radius)) {
                        SDL_RenderDrawPoint(renderer, gpuHitPos.hitPoint[i].x* scale + dx + offsetX, gpuHitPos.hitPoint[i].y* scale + dy + offsetY);
                    }
                }
            }
        }

        int avg_steps = 0;
		for (int i = 0; i < rays; ++i) {
			avg_steps += gpuHitPos.steps[i];
		}
		avg_steps /= rays;
		std::string text = "Avg Steps: " + std::to_string(avg_steps);
		std::cout << text << std::endl;

        SDL_RenderPresent(renderer);
    }

    closeSDL(window, renderer);

    delete[] low_res_buffer.grid.Raw();
    delete[] low_res_grid_data;
    delete[] low_res_chunk_bounds;

    return 0;
}