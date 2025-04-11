#include "Raytracer.cuh"
#include "cuda_noise.cuh"
using namespace GPUDDA;

struct RenderParams {
	uint2 Resolution;
	size_t FrameNumber;
	float Fov;
	float2 OrthoSize;
	__device__ __host__ RenderParams(uint2 r, size_t n, float fov, float2 size) {
		Resolution = r;
		FrameNumber = n;
		Fov = fov;
		OrthoSize = size;
	}
	__device__ __host__ RenderParams() {}
};
__device__ RenderParams d_params;
RenderParams h_params = RenderParams(make_uint2(0,0), 0, 90, make_float2(10,10));

__host__ __device__ void Graphics::getDirections(float3 eularAngles, float3* forwad, float3* up, float3* right)
{
	float3 fwd = make_float3(0, 0, 0);
	float3 upVec = make_float3(0, 0, 0);
	float3 rgt = make_float3(0, 0, 0);
	fwd.x = cos(eularAngles.x) * sin(eularAngles.y);
	fwd.y = -sin(eularAngles.x);
	fwd.z = cos(eularAngles.x) * cos(eularAngles.y);
	rgt.x = cos(eularAngles.y);
	rgt.y = 0;
	rgt.z = -sin(eularAngles.y);
	upVec = cross(fwd, rgt);
	*forwad = fwd * -1;
	*up = upVec * -1;
	*right = rgt;
}

__device__ float3 getRayDirection(float3 fwd, float3 up, float3 right, uint2 screen_dim, float3 uv, float FOV) {
	float aspectRatio = (float)screen_dim.x / (float)screen_dim.y;
	uv.x = uv.x * 2 - 1;
	uv.y = uv.y * 2 - 1;
	uv.z = 1;
	float fov = FOV * 3.1415 / 180.0;
	float scale_x = tanf(fov / 2.0f) * aspectRatio;
	float scale_y = tanf(fov / 2.0f);
	float3 ray_dir{};
	ray_dir.x = fwd.x + uv.x * scale_x * right.x + uv.y * scale_y * up.x;
	ray_dir.y = fwd.y + uv.x * scale_x * right.y + uv.y * scale_y * up.y;
	ray_dir.z = fwd.z + uv.x * scale_x * right.z + uv.y * scale_y * up.z;
	ray_dir = normalize(ray_dir);
	return ray_dir;
}

__device__ void getRayDirectionOrtho(
	float3 fwd, 
	float3 up, 
	float3 right, 
	float2 uv, 
	float2 screen_size,
	float3 origin,
	float3& out_rayDir,
	float3& out_rayOrigin) {

	float ratio = static_cast<float>(d_params.Resolution.x) / d_params.Resolution.y;
	out_rayDir = fwd;
	out_rayOrigin = origin;
	out_rayOrigin += right * (uv.x * 2 - 1) * screen_size.x * ratio;
	out_rayOrigin += up * (uv.y * 2 - 1) * screen_size.y;
}

template<typename T>
__device__ void setPixelColor(void* screen_texture, uint32_t screen_width, uint32_t screen_height, int x, int y, float3 color) {
	T* pixels = (T*)screen_texture;
	if (x < screen_width && y < screen_height) {
		T* pixel = &pixels[y * screen_width + x];
		color.x = fminf(fmaxf(color.x, 0), 1);
		color.y = fminf(fmaxf(color.y, 0), 1);
		color.z = fminf(fmaxf(color.z, 0), 1);

		pixel->r = color.x * 255;
		pixel->g = color.y * 255;
		pixel->b = color.z * 255;
		pixel->a = 255;
	}
}

__device__ Graphics::Environment g_env;
__device__ float3 calculateColor(float3 camPos, float3 normal, float3 position,
	VoxelBuffer<3>* chunks,
	VoxelBuffer<3>* chunksData,
	Bounds<float3>* chunkBoundingBoxes,
	int factor,
	int& out_steps) {
	out_steps = 0;

	//shadow
	float3 shadowRay = normalize(g_env.LightDirection);
	float3 shadowPos = position + g_env.LightDirection * 0.01f;
	float3 shadowNormal;
	int steps;
	bool hit = raytrace(MAX_STEPS, shadowPos, shadowRay, chunks[0], chunksData, chunkBoundingBoxes, factor, steps, shadowNormal, shadowPos);
	out_steps += steps;

	float lDot = fmaxf(dot(normal, g_env.LightDirection), 0) * (hit ? 0 : 1);
	float3 diffuse = lDot * g_env.LightColor;
	float3 ambient = g_env.AmbientColor * lerp(0.25,1.0, dot(normal, make_float3(0, 1, 0)) * 0.5 + 0.5 );
	float3 color = diffuse + ambient;

	//specular
	if (!hit) {
		float3 viewDir = normalize(position - camPos);
		float3 reflectDir = reflect(g_env.LightDirection, normal);
		float spec = powf(fmaxf(dot(viewDir, reflectDir), 0), 32);
		color.x += spec * g_env.LightColor.x;
		color.y += spec * g_env.LightColor.y;
		color.z += spec * g_env.LightColor.z;
	}

	//Ambient Occlusion
	if (lDot == 0) {
		constexpr int samples = 8;
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int seed = y * 1920 + x;
		float occlusion = 0.0f;
		for (int i = 0; i < samples; i++) {
			int si = seed + i * 1000 + (d_params.FrameNumber + 1) * 1000;
			float3 sampleDir = make_float3(
				cudaNoise::randomFloat(si) * 2 - 1,
				cudaNoise::randomFloat(si * 10) * 2 - 1,
				cudaNoise::randomFloat(si * 100) * 2 - 1);
			sampleDir = normalize(sampleDir);
			if (dot(sampleDir, normal) < 0) {
				sampleDir = reflect(sampleDir, normal);
			}

			float3 samplePos = position + sampleDir * 0.01f;
			float3 sampleNormal;
			bool hit = raytrace(8, samplePos, sampleDir, chunks[0], chunksData, chunkBoundingBoxes, factor, steps, sampleNormal, samplePos);
			if (hit) {
				float dist = length(samplePos - position);
				float occlusion = 1 - fminf(1 / (dist * 10.0f), 1.0f);
				occlusion += occlusion;
			}
			else {
				occlusion += 1.0f;
			}
		}
		occlusion /= samples;
		color *= occlusion;
	}

	return color;
}

__device__ float3 Tonemap(float3 color) {
	float3 tonemappedColor = color / (color + make_float3(1.0f));
	tonemappedColor.x = fminf(fmaxf(tonemappedColor.x, 0), 1);
	tonemappedColor.y = fminf(fmaxf(tonemappedColor.y, 0), 1);
	tonemappedColor.z = fminf(fmaxf(tonemappedColor.z, 0), 1);
	return tonemappedColor;
}

__global__ void screenDispatch(
	float3 origin,
	float3 camera_fwd,
	float3 camera_up,
	float3 camera_right,
	uint32_t screen_width,
	uint32_t screen_height,
	void* screen_texture,

	VoxelBuffer<3>* chunks,
	VoxelBuffer<3>* chunksData,
	Bounds<float3>* chunkBoundingBoxes,
	int factor) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < screen_width && y < screen_height) {
		float2 uv = make_float2(x / (float)screen_width, y / (float)screen_height);
#ifdef ORTHO
		float3 ray_dir;
		getRayDirectionOrtho(camera_fwd, camera_up, camera_right, uv, d_params.OrthoSize, origin, ray_dir, origin);
#else
		auto ray_dir = getRayDirection(camera_fwd, camera_up, camera_right, make_uint2(screen_width, screen_height), make_float3(uv.x, uv.y, 0), d_params.Fov);
#endif
		int steps;
		float3 normal;
		float3 hitPos;
		bool hit = raytrace(MAX_STEPS, origin, ray_dir, chunks[0], chunksData, chunkBoundingBoxes, factor, steps, normal, hitPos);
		normal = -normal;
		if (hit) {
#ifdef DEBUG_VIEW
			float dist = length(hitPos - origin);
			hitPos.x = (hitPos.x) / 128.0f;
			hitPos.y = (hitPos.y) / 128.0f;
			hitPos.z = (hitPos.z) / 128.0f;
			hitPos.x = fmodf(hitPos.x, 1.0f + FLT_EPS_DDA);
			hitPos.y = fmodf(hitPos.y, 1.0f + FLT_EPS_DDA);
			hitPos.z = fmodf(hitPos.z, 1.0f + FLT_EPS_DDA);

			////top left
			//if (x < screen_width >> 1 && y < screen_height >> 1) {
			//	setPixelColor<Graphics::BGRA8888>(screen_texture, screen_width, screen_height, x, y, make_float3(normal.x, normal.y, normal.z));
			//}
			////top right
			//else if(x > screen_width >> 1 && y < screen_height >> 1){
			//	setPixelColor<Graphics::BGRA8888>(screen_texture, screen_width, screen_height, x, y, make_float3(hitPos.x, hitPos.y, hitPos.z));
			//}
			////bottom left
			//else if (x < screen_width >> 1) {
			//	//nothing
			//}
			////bottom right
			//else {
			//	setPixelColor<Graphics::BGRA8888>(screen_texture, screen_width, screen_height, x, y, make_float3(dist * 0.01f, 0, 0));
			//}

			setPixelColor<Graphics::BGRA8888>(screen_texture, screen_width, screen_height, x, y, make_float3(fmodf(dist, 1.0f),0,0));

#else
			int color_steps = 0;
			float3 color = calculateColor(origin, normal, hitPos, chunks, chunksData, chunkBoundingBoxes, factor, color_steps);
			color = Tonemap(color);
			steps += color_steps;
			setPixelColor<Graphics::BGRA8888>(screen_texture, screen_width, screen_height, x, y, make_float3(color.x, color.y, color.z));
#endif
		}
		else {
			setPixelColor<Graphics::BGRA8888>(screen_texture, screen_width, screen_height, x, y, make_float3(ray_dir.x, ray_dir.y, ray_dir.z));
		}
#ifdef DEBUG_VIEW
		//if (x < screen_width >> 1 && y > screen_height >> 1) {
		//	setPixelColor<Graphics::BGRA8888>(screen_texture, screen_width, screen_height, x, y, make_float3(steps / 256.0f, 0, 0));
		//}
#endif
	}
}

void Graphics::SetEnvironment(const Environment& env_v) {
	void* d_env;
	cudaGetSymbolAddress(&d_env, g_env);
	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
	}
	cudaMemcpy(d_env, &env_v, sizeof(Environment), cudaMemcpyHostToDevice);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
	}
}

void Graphics::SetFOV(float fov) {
	h_params.Fov = fov;
}

void Graphics::SetOrthoWindowSize(float2 size) {
	h_params.OrthoSize = size;
}

void Graphics::RaytraceScreen(
	VoxelRaytracer3D* rt,
	uint32_t screen_width,
	uint32_t screen_height,
	void* d_screen_texture,
	float3 origin,
	float3 camera_fwd,
	float3 camera_up,
	float3 camera_right) {

	dim3 blockSize(8, 8, 1);
	dim3 numBlocks((screen_width + blockSize.x - 1) / blockSize.x, (screen_height + blockSize.y - 1) / blockSize.y, 1);

	auto buffer = rt->GetVoxelBuffer();
	auto bufferDataBounds = rt->GetVoxelBufferDataBounds();
	auto bufferData = rt->GetVoxelBufferDatas();
	auto factor = rt->GetFactor();
	
	h_params.Resolution = make_uint2(screen_width, screen_height);
	cudaMemcpyToSymbol(d_params, &h_params, sizeof(RenderParams));
	h_params.FrameNumber++;

	screenDispatch << < numBlocks, blockSize >> > (
		origin, camera_fwd, camera_up, camera_right,
		screen_width, screen_height, d_screen_texture,
		buffer, bufferData, bufferDataBounds, factor);

	CUDA_SAFE_CALL(cudaDeviceSynchronize());
}