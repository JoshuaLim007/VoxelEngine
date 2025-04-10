#pragma once
#ifndef DDA_CUH
#define DDA_CUH

#include <limits>
#include <math.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <tuple>
#include <thread>
#include <mutex>

constexpr auto FLT_EPS_DDA = 1e-4;
constexpr auto FLT_INF = std::numeric_limits<float>::infinity();
constexpr auto FLT_EPS = std::numeric_limits<float>::epsilon();

namespace GPUDDA {
	template<class T>
	struct Bounds {
		T min; // minimum bounds
		T max; // maximum bounds
	};
	
	template <typename T>
	class RayTraceResults {
	public:
		std::shared_ptr<bool[]> valid{};  // Indicates if the ray hit a voxel
		std::shared_ptr<T[]> hitPoint{};  // Hit point in world space
		std::shared_ptr<T[]> normal{};    // Normal at the hit point
		std::shared_ptr<float[]> distance{};  // Distance from ray origin to hit point
		std::shared_ptr<int[]> voxelIndex{};  // Index of the voxel that was hit
		std::shared_ptr<int[]> steps{};       // Number of steps taken to find the hit point

		// Constructor
		RayTraceResults(size_t count) {
			if (count == 0) {
				return; // No need to allocate if count is zero
			}
			valid = std::shared_ptr<bool[]>(new bool[count]());
			hitPoint = std::shared_ptr<T[]>(new T[count]());
			normal = std::shared_ptr<T[]>(new T[count]());
			distance = std::shared_ptr<float[]>(new float[count]());
			voxelIndex = std::shared_ptr<int[]>(new int[count]());
			steps = std::shared_ptr<int[]>(new int[count]());
		}

	};
	
	struct BitRef {
		uint8_t* byte = nullptr;
		size_t index = 0;
		__device__ __host__ operator bool() const;
		__device__ __host__ BitRef& operator=(bool value);
	};
	
	struct BitArray {
	private:
		size_t size = 0;
		uint8_t* data = nullptr;
	public:
		__device__ __host__ BitArray();
		__device__ __host__ BitArray(size_t num_bits);
		__device__ __host__ BitArray(const BitArray& other, bool isGPU);
		__device__ __host__ BitArray(size_t num_bits, bool isGPU);
		__device__ __host__ bool operator[](size_t index) const;
		__device__ __host__ BitRef operator[](size_t index);
		__device__ __host__ uint8_t* raw();
		__device__ __host__ size_t bit_size() const;
		__device__ __host__ size_t byte_size() const;
	};

	std::ostream& operator<<(std::ostream& os, const BitArray& bits);

	template<size_t D>
	struct VoxelBuffer {
		BitArray grid{};
		uint16_t dimensions[D]{};
	};
	
	constexpr size_t MAX_STEPS = 2048;

	template<typename T, size_t N>
	struct DDARayParams {
		VoxelBuffer<N> VoxelBuffer;
		T start;
		T direction;
		Bounds<T>* bounds;
		int max_steps;
		Bounds<T>* per_voxel_bounds;
		int per_voxel_bounds_scale;

		__device__ __host__ static DDARayParams Default(
			const GPUDDA::VoxelBuffer<N>& buffer,
			const T& start,
			const T& direction) {

			DDARayParams Params;
			Params.VoxelBuffer = buffer;
			Params.start = start;
			Params.direction = direction;
			Params.bounds = nullptr;
			Params.max_steps = MAX_STEPS;
			Params.per_voxel_bounds = nullptr;
			Params.per_voxel_bounds_scale = 0;
			return Params;
		}
	};

	template<typename T>
	struct DDARayResults {
		bool hit;
		bool isOutOfBounds;
		T HitCell;
		T HitIntersectedPoint;
		T HitNormal;
		int stepsTaken;
	};

	__device__ bool ray_intersects_aabb(float2 start, float2 direction, float2 bmin, float2 bmax, float2* out_intersect, float2* out_normal);
	__device__ bool ray_intersects_aabb(float3 start, float3 direction, float3 bmin, float3 bmax, float3* out_intersect, float3* out_normal);

	__device__ void dda_ray_traversal(
		DDARayParams<float3, 3> Params,
		DDARayResults<float3>& Results
	);

	__device__ void dda_ray_traversal(
		DDARayParams<float2, 2> Params,
		DDARayResults<float2>& Results
	);

	__device__ bool raytrace(int maxSteps, float3 origin, float3 ray, VoxelBuffer<3> chunks, VoxelBuffer<3>* chunksData, Bounds<float3>* chunkBoundingBoxes, int factor,
		int& out_steps, float3& out_normal, float3& out_hit);

	__device__ float2 raytrace(float2 origin, float2 ray, VoxelBuffer<2> chunks, VoxelBuffer<2>* chunksData, Bounds<float2>* chunkBoundingBoxes, int factor,
		int& out_steps, float2& out_normal);

	template<class T, size_t N>
	class VoxelRayTracerBase {
	private:
		VoxelRayTracerBase(const VoxelRayTracerBase&) = delete;
		VoxelRayTracerBase& operator=(const VoxelRayTracerBase&) = delete;
		VoxelRayTracerBase(VoxelRayTracerBase&&) = delete;
		VoxelRayTracerBase& operator=(VoxelRayTracerBase&&) = delete;
		VoxelRayTracerBase() = default;

	protected:
		VoxelRayTracerBase(size_t count) {
			resultsCPU = RayTraceResults<T>(count);

			d_results = nullptr;
			cudaMalloc((void**)&d_results, sizeof(T) * count);
			d_results_normal = nullptr;
			cudaMalloc((void**)&d_results_normal, sizeof(T) * count);
			d_results_steps = nullptr;
			cudaMalloc((void**)&d_results_steps, sizeof(int) * count);
			d_origins = nullptr;
			d_rays = nullptr;
			cudaMalloc((void**)&d_origins, sizeof(T) * count);
			cudaMalloc((void**)&d_rays, sizeof(T) * count);
		}
		~VoxelRayTracerBase() {
			Free();
		}

		// CPU resources
		RayTraceResults<T> resultsCPU = RayTraceResults<T>(0);

		// GPU resources
		T* d_results;
		T* d_results_normal;
		int* d_results_steps;
		T* d_origins;
		T* d_rays;
		int factor = 1;
	
		// GPU resources
		GPUDDA::VoxelBuffer<N>* gpu_VoxelBuffer = nullptr;
		GPUDDA::VoxelBuffer<N>* gpu_VoxelBufferDatas = nullptr;
		Bounds<T>* gpu_VoxelBufferDataBounds = nullptr;
		T dimensions{};

	public:
		GPUDDA::VoxelBuffer<N>* GetVoxelBuffer() {
			return gpu_VoxelBuffer;
		}
		GPUDDA::VoxelBuffer<N>* GetVoxelBufferDatas() {
			return gpu_VoxelBufferDatas;
		}
		Bounds<T>* GetVoxelBufferDataBounds() {
			return gpu_VoxelBufferDataBounds;
		}

		int GetFactor() {
			return factor;
		}
		void SetFactor(int f) {
			factor = f;
		}
		void Free() {
			if (gpu_VoxelBuffer != nullptr)
				cudaFree(gpu_VoxelBuffer);
			if (gpu_VoxelBufferDatas != nullptr)
				cudaFree(gpu_VoxelBufferDatas);
			if (gpu_VoxelBufferDataBounds != nullptr)
				cudaFree(gpu_VoxelBufferDataBounds);

			if (d_results != nullptr)
				cudaFree(d_results);
			if (d_results_normal != nullptr)
				cudaFree(d_results_normal);
			if (d_results_steps != nullptr)
				cudaFree(d_results_steps);
			if (d_origins != nullptr)
				cudaFree(d_origins);
			if (d_rays != nullptr)
				cudaFree(d_rays);

			resultsCPU = RayTraceResults<T>(0); // Reset CPU results
		}
		virtual void UploadVoxelBuffer(const GPUDDA::VoxelBuffer<N>& buff) = 0;
		virtual void UploadVoxelBufferDatas(GPUDDA::VoxelBuffer<N>* buff, size_t count) = 0;
		virtual void UploadVoxelBufferDataBounds(Bounds<T>* bounds, size_t count) = 0;
		virtual RayTraceResults<T> Raytrace(std::vector<T> origin, std::vector<T> ray) = 0;
	};
	
	class VoxelRaytracer2D : public VoxelRayTracerBase<float2, 2> {
	public:
		VoxelRaytracer2D(size_t count) : VoxelRayTracerBase<float2, 2>(count){}
		void UploadVoxelBuffer(const GPUDDA::VoxelBuffer<2>& buff) override;
		void UploadVoxelBufferDatas(GPUDDA::VoxelBuffer<2>* buff, size_t count) override;
		void UploadVoxelBufferDataBounds(Bounds<float2>* bounds, size_t count) override;
		RayTraceResults<float2> Raytrace(std::vector<float2> origin, std::vector<float2> ray) override;
	};
	
	class VoxelRaytracer3D : public VoxelRayTracerBase<float3, 3> {
	public:
		VoxelRaytracer3D(size_t count) : VoxelRayTracerBase<float3, 3>(count){}
		void UploadVoxelBuffer(const GPUDDA::VoxelBuffer<3>& buff) override;
		void UploadVoxelBufferDatas(GPUDDA::VoxelBuffer<3>* buff, size_t count) override;
		void UploadVoxelBufferDataBounds(Bounds<float3>* bounds, size_t count) override;
		RayTraceResults<float3> Raytrace(std::vector<float3> origin, std::vector<float3> ray) override;
	};

	struct ThreadParams{
		size_t threadId = 0;
		size_t maxCount = 0;
		size_t countPerThread = 0;

		size_t rows = 0;
		size_t cols = 0;
		size_t slices = 0;
		size_t factor = 0;

		VoxelBuffer<3>* low_res_grid_data = nullptr;
		Bounds<float3>* low_res_per_chunk_bounds = nullptr;
		const VoxelBuffer<3>* high_res_buffer = nullptr;
		bool* low_res_grid_contains_voxels = nullptr;
	};

#define MULTI_THREADING 1
	static void HandleThread(ThreadParams params) {
		static std::mutex mutex;
		auto low_res_grid_data = params.low_res_grid_data;
		auto low_res_per_chunk_bounds = params.low_res_per_chunk_bounds;
		auto high_res_buffer = params.high_res_buffer;
		size_t factor = params.factor;
		size_t low_res_rows = params.rows;
		size_t low_res_cols = params.cols;
		size_t low_res_slices = params.slices;

		size_t index = params.threadId * params.countPerThread;
		for (size_t i = 0; i < params.countPerThread; i++) {
			size_t tI = index + i;
			size_t x = tI % params.cols;
			size_t y = (tI / params.cols) % params.rows;
			size_t z = tI / (params.cols * params.rows);
			if (tI >= params.maxCount) {
				break;
			}

			bool any = false;
			auto temp = &low_res_grid_data[z * low_res_rows * low_res_cols + y * low_res_cols + x];
			temp->grid = BitArray(factor * factor * factor);
			temp->dimensions[2] = factor;
			temp->dimensions[1] = factor;
			temp->dimensions[0] = factor;

			int min_x = std::numeric_limits<int>::max();
			int min_y = std::numeric_limits<int>::max();
			int min_z = std::numeric_limits<int>::max();
			int max_x = std::numeric_limits<int>::min();
			int max_y = std::numeric_limits<int>::min();
			int max_z = std::numeric_limits<int>::min();

			for (int dz = 0; dz < factor; ++dz) {
				for (int dy = 0; dy < factor; ++dy) {
					for (int dx = 0; dx < factor; ++dx) {
						// Copy the data from the high-res buffer to the low-res buffer
						temp->grid[static_cast<size_t>(dz) * factor * factor + static_cast<size_t>(dy) * factor + dx] =
							high_res_buffer->grid[(static_cast<size_t>(z) * factor + dz) * high_res_buffer->dimensions[1] * 
							high_res_buffer->dimensions[0] + (static_cast<size_t>(y) * factor + dy) * 
							high_res_buffer->dimensions[0] + (static_cast<size_t>(x) * factor + dx)];

						// Check if the voxel is occupied
						int px = x * factor + dx;
						int py = y * factor + dy;
						int pz = z * factor + dz;
						size_t idx = static_cast<size_t>(pz) * high_res_buffer->dimensions[1] * high_res_buffer->dimensions[0] + static_cast<size_t>(py) * high_res_buffer->dimensions[0] + px;
						if (high_res_buffer->grid[idx] != 0) {
							any = true;
							min_x = std::min(min_x, dx);
							min_y = std::min(min_y, dy);
							min_z = std::min(min_z, dz);
							max_x = std::max(max_x, dx);
							max_y = std::max(max_y, dy);
							max_z = std::max(max_z, dz);
						}
					}
				}
			}

			if (any == false) {
				min_x = 0;
				min_y = 0;
				min_z = 0;
				max_x = -1;
				max_y = -1;
				max_z = -1;
				temp->dimensions[0] = 0;
				temp->dimensions[1] = 0;
				temp->dimensions[2] = 0;
				delete[] temp->grid.raw();
			}
			auto idx = z * low_res_rows * low_res_cols + y * low_res_cols + x;
			low_res_per_chunk_bounds[idx].max = make_float3(max_x, max_y, max_z);
			low_res_per_chunk_bounds[idx].min = make_float3(min_x, min_y, min_z);
			params.low_res_grid_contains_voxels[idx] = any ? 1 : 0;
		}
	}

	static std::tuple<VoxelBuffer<3>, VoxelBuffer<3>*, Bounds<float3>*> createBuffersFromVoxels(const VoxelBuffer<3>& high_res_buffer, int factor = 4) {
		size_t low_res_cols = high_res_buffer.dimensions[0] / (size_t)factor;
		size_t low_res_rows = high_res_buffer.dimensions[1] / (size_t)factor;
		size_t low_res_slices = high_res_buffer.dimensions[2] / (size_t)factor;
		VoxelBuffer<3>* low_res_grid_data = new VoxelBuffer<3>[low_res_rows * low_res_cols * low_res_slices] {};
		Bounds<float3>* low_res_per_chunk_bounds = new Bounds<float3>[low_res_rows * low_res_cols * low_res_slices]{};
		
#if MULTI_THREADING == 0
		BitArray low_res_grid = BitArray(low_res_rows * low_res_cols * low_res_slices);
		for (size_t z = 0; z < low_res_slices; z++) {
			for (size_t y = 0; y < low_res_rows; y++) {
				for (size_t x = 0; x < low_res_cols; x++) {
					bool any = false;
					auto temp = &low_res_grid_data[z * low_res_rows * low_res_cols + y * low_res_cols + x];
					temp->grid = BitArray(factor * factor * factor);
					temp->dimensions[2] = factor;
					temp->dimensions[1] = factor;
					temp->dimensions[0] = factor;

					int min_x = std::numeric_limits<int>::max();
					int min_y = std::numeric_limits<int>::max();
					int min_z = std::numeric_limits<int>::max();
					int max_x = std::numeric_limits<int>::min();
					int max_y = std::numeric_limits<int>::min();
					int max_z = std::numeric_limits<int>::min();

					for (int dz = 0; dz < factor; ++dz) {
						for (int dy = 0; dy < factor; ++dy) {
							for (int dx = 0; dx < factor; ++dx) {
								// Copy the data from the high-res buffer to the low-res buffer
								temp->grid[static_cast<size_t>(dz) * factor * factor + static_cast<size_t>(dy) * factor + dx] =
									high_res_buffer.grid[(static_cast<size_t>(z) * factor + dz) * high_res_buffer.dimensions[1] * 
									high_res_buffer.dimensions[0] + (static_cast<size_t>(y) * factor + dy) * 
									high_res_buffer.dimensions[0] + (static_cast<size_t>(x) * factor + dx)];

								// Check if the voxel is occupied
								int px = x * factor + dx;
								int py = y * factor + dy;
								int pz = z * factor + dz;
								size_t idx = static_cast<size_t>(pz) * high_res_buffer.dimensions[1] * high_res_buffer.dimensions[0] + static_cast<size_t>(py) * high_res_buffer.dimensions[0] + px;
								if (high_res_buffer.grid[idx] != 0) {
									any = true;
									min_x = std::min(min_x, dx);
									min_y = std::min(min_y, dy);
									min_z = std::min(min_z, dz);
									max_x = std::max(max_x, dx);
									max_y = std::max(max_y, dy);
									max_z = std::max(max_z, dz);
								}
							}
						}
					}

					if (any == false) {
						min_x = 0;
						min_y = 0;
						min_z = 0;
						max_x = -1;
						max_y = -1;
						max_z = -1;
						temp->dimensions[0] = 0;
						temp->dimensions[1] = 0;
						temp->dimensions[2] = 0;
						delete[] temp->grid.raw();
					}
					auto idx = z * low_res_rows * low_res_cols + y * low_res_cols + x;
					low_res_per_chunk_bounds[idx].max = make_float3(max_x, max_y, max_z);
					low_res_per_chunk_bounds[idx].min = make_float3(min_x, min_y, min_z);
					low_res_grid[idx] = any ? 1 : 0;
				}
			}
		}
#elif MULTI_THREADING == 1
		size_t max_count = low_res_cols * low_res_rows * low_res_slices;
		size_t threads_count = std::thread::hardware_concurrency();
		size_t count_per_thread = (max_count + threads_count - 1) / threads_count;
		std::vector<std::thread> threads;
		bool* temp = new bool[max_count] {};
		for (int i = 0; i < threads_count; i++) {

			ThreadParams params{};
			params.threadId = i;
			params.maxCount = max_count;
			params.countPerThread = count_per_thread;
			params.rows = low_res_rows;
			params.cols = low_res_cols;
			params.slices = low_res_slices;
			params.factor = factor;
			params.low_res_grid_data = low_res_grid_data;
			params.low_res_per_chunk_bounds = low_res_per_chunk_bounds;
			params.low_res_grid_contains_voxels = temp;
			params.high_res_buffer = &high_res_buffer;
			threads.push_back(std::thread(HandleThread, params));
		}

		for (auto& thread : threads) {
			thread.join();
		}

		BitArray low_res_grid = BitArray(low_res_rows * low_res_cols * low_res_slices);
		for (size_t i = 0; i < max_count; i++) {
			low_res_grid[i] = temp[i];
		}
		delete[] temp;
#endif

		VoxelBuffer<3> low_res_buffer;
		low_res_buffer.grid = low_res_grid;
		low_res_buffer.dimensions[2] = low_res_slices;
		low_res_buffer.dimensions[1] = low_res_rows;
		low_res_buffer.dimensions[0] = low_res_cols;
		return std::make_tuple(low_res_buffer, low_res_grid_data, low_res_per_chunk_bounds);
	}

	static std::tuple<VoxelBuffer<2>, VoxelBuffer<2>*, float4*> createBuffersFromVoxels(const VoxelBuffer<2>& high_res_buffer, int factor = 4) {
		int low_res_rows = high_res_buffer.dimensions[1] / factor;
		int low_res_cols = high_res_buffer.dimensions[0] / factor;
		BitArray low_res_grid = BitArray(low_res_rows * low_res_cols);
		VoxelBuffer<2>* low_res_grid_data = new VoxelBuffer<2>[low_res_rows * low_res_cols] {};
		float4* low_res_per_chunk_bounds = new float4[low_res_rows * low_res_cols]{};

		for (int y = 0; y < low_res_rows; ++y) {
			for (int x = 0; x < low_res_cols; ++x) {
				bool any = false;
				auto temp = &low_res_grid_data[y * low_res_cols + x];
				temp->grid = BitArray(factor * factor);
				temp->dimensions[1] = factor;
				temp->dimensions[0] = factor;
				for (int dy = 0; dy < factor; ++dy) {
					for (int dx = 0; dx < factor; ++dx) {
						temp->grid[dy * factor + dx] = high_res_buffer.grid[(y * factor + dy) * high_res_buffer.dimensions[0] + (x * factor + dx)];
					}
				}

				int min_x = std::numeric_limits<int>::max();
				int min_y = std::numeric_limits<int>::max();
				int max_x = std::numeric_limits<int>::min();
				int max_y = std::numeric_limits<int>::min();

				for (int dy = 0; dy < factor; ++dy) {
					for (int dx = 0; dx < factor; ++dx) {

						int px = x * factor + dx;
						int py = y * factor + dy;
						int idx = py * high_res_buffer.dimensions[0] + px;

						if (high_res_buffer.grid[idx] != 0) {
							any = true;
							min_x = std::min(min_x, dx);
							min_y = std::min(min_y, dy);
							max_x = std::max(max_x, dx);
							max_y = std::max(max_y, dy);
						}
					}
				}

				if (any == false) {
					min_x = 0;
					min_y = 0;
					max_x = -1;
					max_y = -1;
					temp->dimensions[0] = 0;
					temp->dimensions[1] = 0;
					delete[] temp->grid.raw();
				}

				low_res_per_chunk_bounds[y * low_res_cols + x] = make_float4(min_x, min_y, max_x, max_y);
				low_res_grid[y * low_res_cols + x] = any ? 1 : 0;
			}
		}

		VoxelBuffer<2> low_res_buffer;
		low_res_buffer.grid = low_res_grid;
		low_res_buffer.dimensions[1] = low_res_rows;
		low_res_buffer.dimensions[0] = low_res_cols;

		return std::make_tuple(low_res_buffer, low_res_grid_data, low_res_per_chunk_bounds);
	}
};

#endif 