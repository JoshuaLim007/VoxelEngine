#include "DDA.cuh"

#include "helper_math.h"
#include <chrono>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

namespace GPUDDA
{
__device__ __host__ BitRef::operator bool() const
{
    return (*byte >> index) & 1;
}

__host__ __device__ BitRef &BitRef::operator=(bool value)
{
    if (value)
        *byte |= (1 << index); // Set bit
    else
        *byte &= ~(1 << index); // Clear bit
    return *this;
}
__device__ __host__ BitArray::BitArray() : size(0), data(nullptr)
{
}
__device__ __host__ BitArray::BitArray(size_t num_bits) : size(num_bits)
{
    data = new uint8_t[(size + 7) / 8]();
}
__device__ __host__ BitArray::BitArray(const BitArray &other, bool isGPU) : size(other.size)
{
    if (isGPU)
    {
        cudaMalloc((void **)&data, (size + 7) / 8);
        cudaMemcpy(data, other.data, (size + 7) / 8, cudaMemcpyHostToDevice);
        return;
    }
    data = new uint8_t[(size + 7) / 8];
    std::copy(other.data, other.data + (size + 7) / 8, data);
}
__device__ __host__ BitArray::BitArray(size_t num_bits, bool isGPU) : size(num_bits)
{
    if (isGPU)
    {
        cudaMalloc((void **)&data, (size + 7) / 8);
        return;
    }
    data = new uint8_t[(size + 7) / 8];
}

__device__ __host__ bool BitArray::operator[](size_t index) const
{
    if (index >= size)
    {
        return false; // Out of bounds
    }
    return (data[index / 8] >> (index % 8)) & 1;
}

__device__ __host__ BitRef BitArray::operator[](size_t index)
{
    return BitRef{&data[index / 8], static_cast<size_t>(index % 8)};
}
__device__ __host__ uint8_t *BitArray::Raw()
{
    return data;
}
__device__ __host__ size_t BitArray::BitSize() const
{
    return size;
}
__device__ __host__ size_t BitArray::ByteSize() const
{
    return (size + 7) / 8;
}
std::ostream &operator<<(std::ostream &os, const BitArray &bits)
{
    for (size_t i = 0; i < bits.BitSize(); ++i)
    {
        os << bits[i]; // Print each bit
    }
    return os;
}

// 2D
__device__ bool RayIntersectsAABB(float2 start, float2 direction, float2 bmin, float2 bmax, float2 *out_intersect,
                                  float2 *out_normal)
{
    float inv_dir_x = 1.0f / (direction.x == 0 ? FLT_EPS : direction.x);
    float inv_dir_y = 1.0f / (direction.y == 0 ? FLT_EPS : direction.y);

    float t_min_x = (bmin.x - start.x) * inv_dir_x;
    float t_max_x = (bmax.x - start.x) * inv_dir_x;
    float t_min_y = (bmin.y - start.y) * inv_dir_y;
    float t_max_y = (bmax.y - start.y) * inv_dir_y;

    float t1_x = fminf(t_min_x, t_max_x);
    float t2_x = fmaxf(t_min_x, t_max_x);
    float t1_y = fminf(t_min_y, t_max_y);
    float t2_y = fmaxf(t_min_y, t_max_y);

    float t_min = fmaxf(t1_x, t1_y); // Largest entering time
    float t_max = fminf(t2_x, t2_y); // Smallest exiting time

    if (t_max < fmaxf(t_min, 0.0f))
    {
        return false; // No intersection
    }
    if (out_intersect)
    {
        *out_intersect = make_float2(start.x + t_min * direction.x, start.y + t_min * direction.y);
    }
    if (out_normal)
    {
        // Determine the axis the intersection happened on
        if (t1_x > t1_y)
        {
            *out_normal = make_float2((inv_dir_x < 0.0f) ? -1.0f : 1.0f, 0.0f);
        }
        else
        {
            *out_normal = make_float2(0.0f, (inv_dir_y < 0.0f) ? -1.0f : 1.0f);
        }
    }
    return true;
}

///////// 2D /////////
__device__ void DDARayTraversal(DDARayParams<float2, 2> Params, DDARayResults<float2> &Results)
{
    float x = Params.start.x;
    float y = Params.start.y;
    float dx = Params.direction.x;
    float dy = Params.direction.y;

    int cell_x = static_cast<int>(x);
    int cell_y = static_cast<int>(y);

    int step_x = (dx > 0) ? 1 : -1;
    int step_y = (dy > 0) ? 1 : -1;

    float tDelta_x = (dx != 0) ? fabs(1.0f / dx) : FLT_INF;
    float tDelta_y = (dy != 0) ? fabs(1.0f / dy) : FLT_INF;

    float tMax_x = (dx != 0) ? (((cell_x + (step_x > 0)) - x) / dx) : FLT_INF;
    float tMax_y = (dy != 0) ? (((cell_y + (step_y > 0)) - y) / dy) : FLT_INF;

    Results.HitIntersectedPoint = make_float2(x, y);
    Results.hit = false;
    Results.isOutOfBounds = false;
    Results.stepsTaken = 0;

    int rows = Params.VoxelBuffer.dimensions[1];
    int cols = Params.VoxelBuffer.dimensions[0];
    auto grid = Params.VoxelBuffer.grid;

    for (int step = 0; step < Params.max_steps; ++step)
    {
        if (0 <= cell_x && cell_x < cols && 0 <= cell_y && cell_y < rows)
        {
            Results.HitCell = make_float2(cell_x, cell_y);
            int idx = (cell_y * cols + cell_x);
            if (Params.per_voxel_bounds)
            {
                float bmin_x = Params.per_voxel_bounds[idx].min.x + cell_x * Params.per_voxel_bounds_scale;
                float bmin_y = Params.per_voxel_bounds[idx].min.y + cell_y * Params.per_voxel_bounds_scale;
                float bmax_x = Params.per_voxel_bounds[idx].max.x + 1 + cell_x * Params.per_voxel_bounds_scale;
                float bmax_y = Params.per_voxel_bounds[idx].max.y + 1 + cell_y * Params.per_voxel_bounds_scale;
                if (grid[idx] == 1 && bmin_x <= bmax_x)
                {
                    float temp_x = Params.start.x * Params.per_voxel_bounds_scale;
                    float temp_y = Params.start.y * Params.per_voxel_bounds_scale;
                    float2 aabb_normal = make_float2(0, 0);
                    if (GPUDDA::RayIntersectsAABB(make_float2(temp_x, temp_y), Params.direction,
                                                  make_float2(bmin_x, bmin_y), make_float2(bmax_x, bmax_y), nullptr,
                                                  &aabb_normal))
                    {
                        Results.hit = true;
                        if (step == 0)
                        {
                            Results.HitNormal = aabb_normal;
                        }
                        break;
                    }
                }
            }
            else
            {
                if (grid[idx] == 1)
                {
                    Results.hit = true;
                    break;
                }
            }
        }
        else
        {
            Results.isOutOfBounds = true;
            break;
        }

        float intersect_x = 0;
        float intersect_y = 0;
        if (tMax_x < tMax_y)
        {
            intersect_x = cell_x + (step_x > 0);
            intersect_y = y + (tMax_x * dy);
            cell_x += step_x;
            tMax_x += tDelta_x;
            Results.HitNormal = make_float2(step_x, 0);
        }
        else
        {
            intersect_x = x + (tMax_y * dx);
            intersect_y = cell_y + (step_y > 0);
            cell_y += step_y;
            tMax_y += tDelta_y;
            Results.HitNormal = make_float2(0, step_y);
        }

        if (Params.bounds)
        {
            int min_x = Params.bounds->min.x;
            int min_y = Params.bounds->min.y;
            int max_x = Params.bounds->max.x;
            int max_y = Params.bounds->max.y;
            bool isOutOfBounds =
                (intersect_x < min_x || intersect_x > max_x || intersect_y < min_y || intersect_y > max_y);
            if (isOutOfBounds)
            {
                Results.isOutOfBounds = true;
                break;
            }
        }

        Results.stepsTaken += 1;
        Results.HitIntersectedPoint = make_float2(intersect_x, intersect_y);
    }
}

__device__ float2 Raytrace(float2 origin, float2 ray, VoxelBuffer<2> chunks, VoxelBuffer<2> *chunksData,
                           Bounds<float2> *chunkBoundingBoxes, int factor, int &out_steps, float2 &out_normal)
{
    float rayLen = sqrt(ray.x * ray.x + ray.y * ray.y);
    ray.x /= rayLen;
    ray.y /= rayLen;

    float2 previous_cell = make_float2(-1, -1);
    int total_steps = 0;

    // in chunk space
    float2 start = origin;
    start.x /= factor;
    start.y /= factor;
    float2 direction = normalize(ray);
    float eps = FLT_EPS_DDA;
    if (!(start.x >= 0 && start.y >= 0 && start.x < chunks.dimensions[0] && start.y < chunks.dimensions[1]))
    {
        float2 intersect;
        if (GPUDDA::RayIntersectsAABB(start, direction, make_float2(0, 0),
                                      make_float2(chunks.dimensions[0], chunks.dimensions[1]), &intersect, nullptr))
        {
            if (intersect.x == chunks.dimensions[0])
                intersect.x -= 1;
            if (intersect.y == chunks.dimensions[1])
                intersect.y -= 1;
            start = intersect;
        }
    }
    out_normal = make_float2(0, 0);
    float2 hitPosition = make_float2(0, 0);
    bool hit = false;
    while (true)
    {

        float2 start_high_res;
        DDARayParams<float2, 2> params = DDARayParams<float2, 2>::Default(chunks, start, direction);
        params.per_voxel_bounds = chunkBoundingBoxes;
        params.per_voxel_bounds_scale = factor;
        DDARayResults<float2> results;
        DDARayTraversal(params, results);

        total_steps += results.stepsTaken;
        start_high_res = make_float2(results.HitIntersectedPoint.x * factor, results.HitIntersectedPoint.y * factor);
        hitPosition = start_high_res;

        if (results.hit && !results.isOutOfBounds)
        {
            float4 chunkBounds{};
            if (previous_cell.x == results.HitCell.x && previous_cell.y == results.HitCell.y)
            {
                break;
            }
            previous_cell = results.HitCell;
            chunkBounds.x = 0;
            chunkBounds.y = 0;
            chunkBounds.z = factor;
            chunkBounds.w = factor;

            start_high_res.x -= results.HitCell.x * factor;
            start_high_res.y -= results.HitCell.y * factor;
            VoxelBuffer<2> chunkData = chunksData[(int)(results.HitCell.y * chunks.dimensions[0] + results.HitCell.x)];

            if (start_high_res.x == factor)
                start_high_res.x -= eps;
            if (start_high_res.y == factor)
                start_high_res.y -= eps;

            DDARayParams<float2, 2> params_hr = DDARayParams<float2, 2>::Default(chunkData, start_high_res, direction);
            params_hr.bounds = reinterpret_cast<Bounds<float2> *>(&chunkBounds);
            DDARayResults<float2> results_hr;

            DDARayTraversal(params_hr, results_hr);

            total_steps += results_hr.stepsTaken;

            hitPosition = results_hr.HitIntersectedPoint;
            hitPosition.x += results.HitCell.x * factor;
            hitPosition.y += results.HitCell.y * factor;

            if (!results_hr.hit)
            {

                start = make_float2(results_hr.HitIntersectedPoint.x + results.HitCell.x * factor,
                                    results_hr.HitIntersectedPoint.y + results.HitCell.y * factor);
                start.x /= factor;
                start.y /= factor;

                if (direction.x < 0)
                    start.x -= eps;
                if (direction.y < 0)
                    start.y -= eps;

                continue;
            }
            else
            {
                if (results_hr.stepsTaken == 0)
                {
                    out_normal = results.HitNormal;
                }
                else
                {
                    out_normal = results_hr.HitNormal;
                }
                hit = true;
                break;
            }
        }
        else
        {
            break;
        }
    }
    out_steps = total_steps;
    return hit ? hitPosition : make_float2(FLT_INF, FLT_INF);
}

__global__ void dispatch(float2 *origins, float2 *rays, VoxelBuffer<2> *chunks, VoxelBuffer<2> *chunksData,
                         Bounds<float2> *chunkBoundingBoxes, int factor, float2 *results_point, float2 *results_normal,
                         int *results_steps, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        int steps;
        float2 normal;
        results_point[idx] =
            Raytrace(origins[idx], rays[idx], chunks[0], chunksData, chunkBoundingBoxes, factor, steps, normal);
        results_steps[idx] = steps;
        results_normal[idx] = normal;
    }
}

void VoxelRaytracer2D::UploadVoxelBuffer(const GPUDDA::VoxelBuffer<2> &buff)
{
    if (gpu_VoxelBuffer != nullptr)
    {
        VoxelBuffer<2> *temp;
        cudaMemcpy(temp, gpu_VoxelBuffer, sizeof(GPUDDA::VoxelBuffer<2>), cudaMemcpyDeviceToHost);
        cudaFree(temp->grid.Raw());
        cudaFree(gpu_VoxelBuffer);
    }

    dimensions.x = buff.dimensions[0];
    dimensions.y = buff.dimensions[1];

    BitArray bitArray(buff.grid, true);

    VoxelBuffer<2> temp;
    temp.dimensions[0] = buff.dimensions[0];
    temp.dimensions[1] = buff.dimensions[1];
    temp.grid = bitArray;

    cudaMalloc((void **)&gpu_VoxelBuffer, sizeof(GPUDDA::VoxelBuffer<2>));
    cudaMemcpy(gpu_VoxelBuffer, &temp, sizeof(GPUDDA::VoxelBuffer<2>), cudaMemcpyHostToDevice);
}

void VoxelRaytracer2D::UploadVoxelBufferDatas(GPUDDA::VoxelBuffer<2> *buff, size_t count)
{
    if (gpu_VoxelBufferDatas != nullptr)
        cudaFree(gpu_VoxelBufferDatas);

    GPUDDA::VoxelBuffer<2> *temp = new GPUDDA::VoxelBuffer<2>[count];
    for (size_t i = 0; i < count; i++)
    {
        temp[i].dimensions[0] = buff[i].dimensions[0];
        temp[i].dimensions[1] = buff[i].dimensions[1];
        temp[i].grid = BitArray(buff[i].grid, true);
    }

    auto memSize = sizeof(GPUDDA::VoxelBuffer<2>) * count;
    cudaMalloc((void **)&gpu_VoxelBufferDatas, memSize);
    cudaMemcpy(gpu_VoxelBufferDatas, temp, memSize, cudaMemcpyHostToDevice);
}

void VoxelRaytracer2D::UploadVoxelBufferDataBounds(Bounds<float2> *bounds, size_t count)
{
    if (gpu_VoxelBufferDataBounds != nullptr)
        cudaFree(gpu_VoxelBufferDataBounds);

    auto memSize = sizeof(Bounds<float2>) * count;
    cudaMalloc((void **)&gpu_VoxelBufferDataBounds, memSize);
    cudaMemcpy(gpu_VoxelBufferDataBounds, bounds, memSize, cudaMemcpyHostToDevice);
}

RayTraceResults<float2> VoxelRaytracer2D::Raytrace(std::vector<float2> origin, std::vector<float2> ray)
{
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();

    auto result = resultsCPU;
    int count = origin.size();

    cudaMemcpy(d_origins, origin.data(), sizeof(float2) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rays, ray.data(), sizeof(float2) * count, cudaMemcpyHostToDevice);

    dim3 blockSize(8, 1, 1);
    dim3 numBlocks((count + (count - 1) / blockSize.x), 1, 1);

    dispatch<<<numBlocks, blockSize>>>(d_origins, d_rays, gpu_VoxelBuffer, gpu_VoxelBufferDatas,
                                       gpu_VoxelBufferDataBounds, factor, d_results, d_results_normal, d_results_steps,
                                       count);

    cudaMemcpy(result.hitPoint.get(), d_results, sizeof(float2) * count, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.normal.get(), d_results_normal, sizeof(float2) * count, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.steps.get(), d_results_steps, sizeof(int) * count, cudaMemcpyDeviceToHost);

    auto validPtr = result.valid.get();
    auto pointPtr = result.hitPoint.get();
    auto distancePtr = result.distance.get();
    auto voxelPtr = result.voxelIndex.get();
    for (size_t i = 0; i < count; i++)
    {
        validPtr[i] = (pointPtr[i].x != FLT_INF && pointPtr[i].y != FLT_INF);
        if (validPtr[i])
        {
            float dtx = origin[i].x - pointPtr[i].x;
            float dty = origin[i].y - pointPtr[i].y;
            distancePtr[i] = sqrt(dtx * dtx + dty * dty);
            voxelPtr[i] = (int)(pointPtr[i].y * dimensions.x + pointPtr[i].x);
        }
    }

    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "Raytracing time: " << dt / 1000.0f << " ms" << std::endl;

    return result;
}

///////// 3D /////////
__global__ void dispatch(float3 *origins, float3 *rays, VoxelBuffer<3> *chunks, VoxelBuffer<3> *chunksData,
                         Bounds<float3> *chunkBoundingBoxes, int factor, float3 *results_point, float3 *results_normal,
                         int *results_steps, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        int steps;
        float3 normal;
        float3 pos;
        if (Raytrace(MAX_STEPS, origins[idx], rays[idx], chunks[0], chunksData, chunkBoundingBoxes, factor, steps,
                     normal, pos))
        {
            results_point[idx] = pos;
        }
        else
        {
            results_point[idx] = make_float3(FLT_INF, FLT_INF, FLT_INF);
        }
        results_steps[idx] = steps;
        results_normal[idx] = normal;
    }
}

__device__ bool aabb_contains(const float3 &pos, const float3 &min, const float3 &max)
{
    return pos.x >= min.x && pos.x <= max.x && pos.y >= min.y && pos.y <= max.y && pos.z >= min.z && pos.z <= max.z;
}

__device__ bool RayIntersectsAABB(const float3 &start, const float3 &direction, const float3 &bmin, const float3 &bmax,
                                  float3 *out_intersect, float3 *out_normal)
{
    float inv_dir_x = 1.0f / (direction.x == 0 ? FLT_EPS : direction.x);
    float inv_dir_y = 1.0f / (direction.y == 0 ? FLT_EPS : direction.y);
    float inv_dir_z = 1.0f / (direction.z == 0 ? FLT_EPS : direction.z);

    float t_min_x = (bmin.x - start.x) * inv_dir_x;
    float t_max_x = (bmax.x - start.x) * inv_dir_x;
    float t_min_y = (bmin.y - start.y) * inv_dir_y;
    float t_max_y = (bmax.y - start.y) * inv_dir_y;
    float t_min_z = (bmin.z - start.z) * inv_dir_z;
    float t_max_z = (bmax.z - start.z) * inv_dir_z;

    float t1_x = fminf(t_min_x, t_max_x);
    float t2_x = fmaxf(t_min_x, t_max_x);
    float t1_y = fminf(t_min_y, t_max_y);
    float t2_y = fmaxf(t_min_y, t_max_y);
    float t1_z = fminf(t_min_z, t_max_z);
    float t2_z = fmaxf(t_min_z, t_max_z);

    float t_min = fmaxf(fmaxf(t1_x, t1_y), t1_z); // Largest entering time
    float t_max = fminf(fminf(t2_x, t2_y), t2_z); // Smallest exiting time

    if (t_max < fmaxf(t_min, 0.0f))
    {
        return false; // No intersection
    }
    if (out_intersect)
    {
        *out_intersect =
            make_float3(start.x + t_min * direction.x, start.y + t_min * direction.y, start.z + t_min * direction.z);
    }

    if (out_normal)
    {
        if (t_min == t1_x)
        {
            *out_normal = make_float3((inv_dir_x < 0.0f) ? -1.0f : 1.0f, 0.0f, 0.0f);
        }
        else if (t_min == t1_y)
        {
            *out_normal = make_float3(0.0f, (inv_dir_y < 0.0f) ? -1.0f : 1.0f, 0.0f);
        }
        else
        { // t_min == t1_z
            *out_normal = make_float3(0.0f, 0.0f, (inv_dir_z < 0.0f) ? -1.0f : 1.0f);
        }
    }

    return true;
}

__device__ void DDARayTraversal(const DDARayParams<float3, 3> &Params, DDARayResults<float3> &Results)
{
    float x = Params.start.x;
    float y = Params.start.y;
    float z = Params.start.z;

    float dx = Params.direction.x;
    float dy = Params.direction.y;
    float dz = Params.direction.z;

    int cell_x = static_cast<int>(x);
    int cell_y = static_cast<int>(y);
    int cell_z = static_cast<int>(z);

    // start tracing
    int depth = Params.VoxelBuffer.dimensions[2];
    int rows = Params.VoxelBuffer.dimensions[1];
    int cols = Params.VoxelBuffer.dimensions[0];

    int step_x = (dx > 0) ? 1 : -1;
    int step_y = (dy > 0) ? 1 : -1;
    int step_z = (dz > 0) ? 1 : -1;

    float tDelta_x = (dx != 0) ? fabs(1.0f / dx) : FLT_INF;
    float tDelta_y = (dy != 0) ? fabs(1.0f / dy) : FLT_INF;
    float tDelta_z = (dz != 0) ? fabs(1.0f / dz) : FLT_INF;

    float tMax_x = (dx != 0) ? (((cell_x + (step_x > 0)) - x) / dx) : FLT_INF;
    float tMax_y = (dy != 0) ? (((cell_y + (step_y > 0)) - y) / dy) : FLT_INF;
    float tMax_z = (dz != 0) ? (((cell_z + (step_z > 0)) - z) / dz) : FLT_INF;

    Results.HitIntersectedPoint = make_float3(x, y, z);
    Results.hit = false;
    Results.isOutOfBounds = false;
    Results.stepsTaken = 0;

    auto grid = Params.VoxelBuffer.grid;
    bool exit = false;

    bool IsOnEdge = cell_x == cols || cell_y == rows || cell_z == depth;
    float3 edgePadding = make_float3(0, 0, 0);
    if (IsOnEdge)
    {
        if (dx < 0)
        {
            edgePadding.x = 1;
        }
        if (dy < 0)
        {
            edgePadding.y = 1;
        }
        if (dz < 0)
        {
            edgePadding.z = 1;
        }
    }

    for (int step = 0; step < Params.max_steps; ++step)
    {
        bool skipCheck = Params.takeInitialStep == true && step == 0;

        if (skipCheck == false)
        {
            if (0 <= cell_x && cell_x < cols + edgePadding.x && 0 <= cell_y && cell_y < rows + edgePadding.y &&
                0 <= cell_z && cell_z < depth + edgePadding.z)
            {

                int clamped_x = min(max(cell_x, 0), cols - 1);
                int clamped_y = min(max(cell_y, 0), rows - 1);
                int clamped_z = min(max(cell_z, 0), depth - 1);
                Results.HitCell = make_float3(clamped_x, clamped_y, clamped_z);
                int idx = (clamped_z * rows * cols + clamped_y * cols + clamped_x);
                if (Params.per_voxel_bounds)
                {
                    float bmin_x = Params.per_voxel_bounds[idx].min.x + clamped_x * Params.per_voxel_bounds_scale;
                    float bmin_y = Params.per_voxel_bounds[idx].min.y + clamped_y * Params.per_voxel_bounds_scale;
                    float bmin_z = Params.per_voxel_bounds[idx].min.z + clamped_z * Params.per_voxel_bounds_scale;
                    float bmax_x = Params.per_voxel_bounds[idx].max.x + 1 + clamped_x * Params.per_voxel_bounds_scale;
                    float bmax_y = Params.per_voxel_bounds[idx].max.y + 1 + clamped_y * Params.per_voxel_bounds_scale;
                    float bmax_z = Params.per_voxel_bounds[idx].max.z + 1 + clamped_z * Params.per_voxel_bounds_scale;
                    if (grid[idx] == 1 && bmin_x <= bmax_x)
                    {
                        float temp_x = Params.start.x * Params.per_voxel_bounds_scale;
                        float temp_y = Params.start.y * Params.per_voxel_bounds_scale;
                        float temp_z = Params.start.z * Params.per_voxel_bounds_scale;
                        float3 aabb_normal = make_float3(0, 0, 0);
                        float3 aabb_pos = make_float3(0, 0, 0);
                        if (RayIntersectsAABB(make_float3(temp_x, temp_y, temp_z), Params.direction,
                                              make_float3(bmin_x, bmin_y, bmin_z), make_float3(bmax_x, bmax_y, bmax_z),
                                              &aabb_pos, &aabb_normal))
                        {
                            Results.hit = true;
                            if (step == 0)
                            {
                                Results.HitNormal = aabb_normal;
                            }
                            exit = true;
                        }
                    }
                }
                else
                {
                    if (grid[idx] == 1)
                    {
                        Results.hit = true;
                        exit = true;
                    }
                }
            }
            else
            {
                Results.isOutOfBounds = true;
                exit = true;
            }
        }

        float intersect_x = 0;
        float intersect_y = 0;
        float intersect_z = 0;
        if (tMax_x < tMax_y && tMax_x < tMax_z)
        {
            intersect_x = cell_x + (step_x > 0);
            intersect_y = y + (tMax_x * dy);
            intersect_z = z + (tMax_x * dz);
            cell_x += step_x;
            tMax_x += tDelta_x;
            if (!exit)
                Results.HitNormal = make_float3(step_x, 0, 0);
        }
        else if (tMax_y <= tMax_x && tMax_y < tMax_z)
        {
            intersect_x = x + (tMax_y * dx);
            intersect_y = cell_y + (step_y > 0);
            intersect_z = z + (tMax_y * dz);
            cell_y += step_y;
            tMax_y += tDelta_y;
            if (!exit)
                Results.HitNormal = make_float3(0, step_y, 0);
        }
        else
        {
            intersect_x = x + (tMax_z * dx);
            intersect_y = y + (tMax_z * dy);
            intersect_z = cell_z + (step_z > 0);
            cell_z += step_z;
            tMax_z += tDelta_z;
            if (!exit)
                Results.HitNormal = make_float3(0, 0, step_z);
        }
        if (!exit)
        {
            if (Params.bounds)
            {
                int min_x = Params.bounds->min.x;
                int min_y = Params.bounds->min.y;
                int min_z = Params.bounds->min.z;
                int max_x = Params.bounds->max.x;
                int max_y = Params.bounds->max.y;
                int max_z = Params.bounds->max.z;
                // Check if the intersection point is within the bounds
                bool isOutOfBounds = (intersect_x < min_x || intersect_x > max_x || intersect_y < min_y ||
                                      intersect_y > max_y || intersect_z < min_z || intersect_z > max_z);
                if (isOutOfBounds)
                {
                    Results.isOutOfBounds = true;
                    break;
                }
            }
            Results.stepsTaken += 1;
            Results.HitIntersectedPoint = make_float3(intersect_x, intersect_y, intersect_z);
        }
        else
        {
            Results.NextCell = make_float3(cell_x, cell_y, cell_z);
            Results.NextInterSectedPoint = make_float3(intersect_x, intersect_y, intersect_z);
            break;
        }
    }
}

//__device__ bool debugPrint;
//__device__ void DebugPrint(bool val) {
//    debugPrint = val;
//}

__device__ bool Raytrace(int maxSteps, float3 origin, float3 ray, VoxelBuffer<3> chunks, VoxelBuffer<3> *chunksData,
                         Bounds<float3> *chunkBoundingBoxes, int factor, int &out_steps, float3 &out_normal,
                         float3 &out_pos)
{

    float3 previous_cell = make_float3(-1, -1, -1);
    int total_steps = 0;

    float3 start = origin;
    start.x /= factor;
    start.y /= factor;
    start.z /= factor;

    float3 direction = normalize(ray);
    float3 start_normal = make_float3(0, 0, 0);
    if (!(start.x >= 0 && start.y >= 0 && start.z >= 0 && start.x < chunks.dimensions[0] &&
          start.y < chunks.dimensions[1] && start.z < chunks.dimensions[2]))
    {
        float3 intersect;
        if (RayIntersectsAABB(make_float3(start.x, start.y, start.z), direction,
                              make_float3(FLT_EPS_DDA, FLT_EPS_DDA, FLT_EPS_DDA),
                              make_float3(chunks.dimensions[0] - FLT_EPS_DDA, chunks.dimensions[1] - FLT_EPS_DDA,
                                          chunks.dimensions[2] - FLT_EPS_DDA),
                              &intersect, &start_normal))
        {
            start = intersect;
        }
    }
    out_normal = make_float3(0, 0, 0);
    float3 hitPosition = make_float3(0, 0, 0);
    bool hit = false;
    //     auto tx = threadIdx.x + blockIdx.x * blockDim.x;
    //     auto ty = threadIdx.y + blockIdx.y * blockDim.y;
    //     bool doDebugPrint = tx == 1920 >> 1 && ty == 1080 >> 1 && debugPrint;
    //     if (doDebugPrint) {
    // printf("----------START-----------\n");
    //     }
    // int i = 0;
    while (total_steps < maxSteps)
    {
        float3 start_high_res;
        DDARayParams<float3, 3> params = DDARayParams<float3, 3>::Default(chunks, start, direction);
        params.per_voxel_bounds = chunkBoundingBoxes;
        params.per_voxel_bounds_scale = factor;
        DDARayResults<float3> results;
        DDARayTraversal(params, results);

        //        if (doDebugPrint) {
        //            printf("---CHUNK---: %d \n", i);
        // printf("HitCell: %d %d %d\n", (int)results.HitCell.x, (int)results.HitCell.y, (int)results.HitCell.z);
        // printf("HitIntersectedPoint: %f %f %f\n", results.HitIntersectedPoint.x, results.HitIntersectedPoint.y,
        // results.HitIntersectedPoint.z); printf("NextCell: %f %f %f\n", results.NextCell.x, results.NextCell.y,
        // results.NextCell.z); printf("Hit: %d\n", results.hit); printf("isOutOfBounds: %d\n", results.isOutOfBounds);
        // printf("stepsTaken: %d\n", results.stepsTaken);
        // printf("start: %f %f %f\n", start.x, start.y, start.z);
        //        }

        total_steps += results.stepsTaken;
        start_high_res = make_float3(results.HitIntersectedPoint.x * factor, results.HitIntersectedPoint.y * factor,
                                     results.HitIntersectedPoint.z * factor);
        hitPosition = start_high_res;
        if (results.hit && !results.isOutOfBounds)
        {
            Bounds<float3> chunkBounds{};
            if (previous_cell.x == results.HitCell.x && previous_cell.y == results.HitCell.y &&
                previous_cell.z == results.HitCell.z)
            {

                // if (doDebugPrint) {
                //     printf("Same cell\n");
                // }

                break;
            }
            previous_cell = results.HitCell;
            chunkBounds.min.x = 0;
            chunkBounds.min.y = 0;
            chunkBounds.min.z = 0;
            chunkBounds.max.x = factor;
            chunkBounds.max.y = factor;
            chunkBounds.max.z = factor;
            start_high_res.x -= results.HitCell.x * factor;
            start_high_res.y -= results.HitCell.y * factor;
            start_high_res.z -= results.HitCell.z * factor;

            VoxelBuffer<3> chunkData =
                chunksData[(int)(results.HitCell.z * chunks.dimensions[1] * chunks.dimensions[0] +
                                 results.HitCell.y * chunks.dimensions[0] + results.HitCell.x)];
            DDARayParams<float3, 3> params_hr = DDARayParams<float3, 3>::Default(chunkData, start_high_res, direction);
            params_hr.bounds = &chunkBounds;
            DDARayResults<float3> results_hr;
            DDARayTraversal(params_hr, results_hr);

            // if (doDebugPrint) {
            //     printf("---VOXEL---:\n");
            //     printf("HitCell: %d %d %d\n", (int)results_hr.HitCell.x, (int)results_hr.HitCell.y,
            //     (int)results_hr.HitCell.z); printf("HitIntersectedPoint: %f %f %f\n",
            //     results_hr.HitIntersectedPoint.x, results_hr.HitIntersectedPoint.y,
            //     results_hr.HitIntersectedPoint.z); printf("NextCell: %f %f %f\n", results_hr.NextCell.x,
            //     results_hr.NextCell.y, results_hr.NextCell.z); printf("Hit: %d\n", results_hr.hit);
            //     printf("isOutOfBounds: %d\n", results_hr.isOutOfBounds);
            //     printf("stepsTaken: %d\n", results_hr.stepsTaken);
            //     printf("start: %f %f %f\n", start_high_res.x, start_high_res.y, start_high_res.z);
            // }

            total_steps += results_hr.stepsTaken;
            hitPosition = make_float3(results_hr.HitIntersectedPoint.x + results.HitCell.x * factor,
                                      results_hr.HitIntersectedPoint.y + results.HitCell.y * factor,
                                      results_hr.HitIntersectedPoint.z + results.HitCell.z * factor);

            if (!results_hr.hit)
            {
                start = hitPosition;
                start.x /= factor;
                start.y /= factor;
                start.z /= factor;

                // if (doDebugPrint) {
                //     printf("Next start: %f %f %f\n", start.x, start.y, start.z);
                // }

                if (results_hr.isOutOfBounds)
                {
                    // projected cell
                    int cx = static_cast<int>(start.x);
                    int cy = static_cast<int>(start.y);
                    int cz = static_cast<int>(start.z);
                    bool projectedCellIsSame =
                        results.HitCell.x == cx && results.HitCell.y == cy && results.HitCell.z == cz;

                    // apply the smallest diff to start
                    if (projectedCellIsSame)
                    {

                        // first apply eps to see if it crosses chunk border
                        if (results.HitCell.x == cx)
                        {
                            start.x = direction.x < 0 ? nextafterf(start.x, -FLT_INF) : nextafterf(start.x, FLT_INF);
                        }
                        if (results.HitCell.y == cy)
                        {
                            start.y = direction.y < 0 ? nextafterf(start.y, -FLT_INF) : nextafterf(start.y, FLT_INF);
                        }
                        if (results.HitCell.z == cz)
                        {
                            start.z = direction.z < 0 ? nextafterf(start.z, -FLT_INF) : nextafterf(start.z, FLT_INF);
                        }

                        // projected cell
                        int cx = static_cast<int>(start.x);
                        int cy = static_cast<int>(start.y);
                        int cz = static_cast<int>(start.z);
                        projectedCellIsSame =
                            results.HitCell.x == cx && results.HitCell.y == cy && results.HitCell.z == cz;

                        // if projected cell is still the same, apply the smallest diff
                        if (projectedCellIsSame)
                        {
                            // find the smallest diff to the next cell
                            float3 diff = make_float3(results.NextCell.x - start.x, results.NextCell.y - start.y,
                                                      results.NextCell.z - start.z);
                            float3 absDiff = make_float3(fabsf(diff.x), fabsf(diff.y), fabsf(diff.z));
                            if (absDiff.x < absDiff.y && absDiff.x < absDiff.z)
                            {
                                start.x += diff.x;
                            }
                            else if (absDiff.y < absDiff.x && absDiff.y < absDiff.z)
                            {
                                start.y += diff.y;
                            }
                            else
                            {
                                start.z += diff.z;
                            }
                        }
                    }
                }

                //              if (doDebugPrint) {
                // printf("Adjusted Next start: %f %f %f\n", start.x, start.y, start.z);
                //              }
                //              i++;
                continue;
            }
            else
            {
                // steps taken was 0, use chunk's normal
                if (results_hr.stepsTaken == 0)
                {
                    out_normal = results.HitNormal;
                }
                else
                {
                    out_normal = results_hr.HitNormal;
                }
                hit = true;
                break;
            }
        }
        else
        {

            // if (doDebugPrint) {
            //     printf("results.hit false, results.isOutOfBounds true");
            // }

            break;
        }
        // i++;
    }

    // if (doDebugPrint) {
    //	printf("----------END-----------\n");
    // }

    out_steps = total_steps;
    if (hit)
    {
        out_pos = hitPosition;
        if (total_steps == 0)
        {
            out_pos = start * factor;
            out_normal = start_normal;
        }
    }
    return hit;
}

void VoxelRaytracer3D::UploadVoxelBuffer(const GPUDDA::VoxelBuffer<3> &buff)
{

    if (gpu_VoxelBuffer != nullptr)
    {
        VoxelBuffer<3> *temp;
        cudaMemcpy(temp, gpu_VoxelBuffer, sizeof(GPUDDA::VoxelBuffer<3>), cudaMemcpyDeviceToHost);
        cudaFree(temp->grid.Raw());
        cudaFree(gpu_VoxelBuffer);
    }

    dimensions.x = buff.dimensions[0];
    dimensions.y = buff.dimensions[1];
    dimensions.z = buff.dimensions[2];

    VoxelBuffer<3> temp;
    temp.grid = BitArray(buff.grid, true);
    temp.dimensions[0] = buff.dimensions[0];
    temp.dimensions[1] = buff.dimensions[1];
    temp.dimensions[2] = buff.dimensions[2];

    cudaMalloc((void **)&gpu_VoxelBuffer, sizeof(GPUDDA::VoxelBuffer<3>));
    cudaMemcpy(gpu_VoxelBuffer, &temp, sizeof(GPUDDA::VoxelBuffer<3>), cudaMemcpyHostToDevice);
}

void VoxelRaytracer3D::UploadVoxelBufferDatas(GPUDDA::VoxelBuffer<3> *buff, size_t count)
{
    GPUDDA::VoxelBuffer<3> *temp = new GPUDDA::VoxelBuffer<3>[count];
    for (size_t i = 0; i < count; i++)
    {
        temp[i].dimensions[0] = buff[i].dimensions[0];
        temp[i].dimensions[1] = buff[i].dimensions[1];
        temp[i].dimensions[2] = buff[i].dimensions[2];
        temp[i].grid = BitArray(buff[i].grid, true);
    }
    auto memSize = sizeof(GPUDDA::VoxelBuffer<3>) * count;
    cudaMalloc((void **)&gpu_VoxelBufferDatas, memSize);
    cudaMemcpy(gpu_VoxelBufferDatas, temp, memSize, cudaMemcpyHostToDevice);
}

void VoxelRaytracer3D::UploadVoxelBufferDataBounds(Bounds<float3> *bounds, size_t count)
{
    auto memSize = sizeof(Bounds<float3>) * count;
    cudaMalloc((void **)&gpu_VoxelBufferDataBounds, memSize);
    cudaMemcpy(gpu_VoxelBufferDataBounds, bounds, memSize, cudaMemcpyHostToDevice);
}

RayTraceResults<float3> VoxelRaytracer3D::Raytrace(std::vector<float3> origin, std::vector<float3> ray)
{
    auto result = resultsCPU;
    int count = origin.size();

    cudaMemcpy(d_origins, origin.data(), sizeof(float3) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rays, ray.data(), sizeof(float3) * count, cudaMemcpyHostToDevice);

    dim3 blockSize(8, 1, 1);
    dim3 numBlocks((count + (count - 1) / blockSize.x), 1, 1);

    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    dispatch<<<numBlocks, blockSize>>>(d_origins, d_rays, gpu_VoxelBuffer, gpu_VoxelBufferDatas,
                                       gpu_VoxelBufferDataBounds, factor, d_results, d_results_normal, d_results_steps,
                                       count);

    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "Raytracing time: " << dt / 1000.0f << " ms" << std::endl;

    cudaMemcpy(result.hitPoint.get(), d_results, sizeof(float3) * count, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.normal.get(), d_results_normal, sizeof(float3) * count, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.steps.get(), d_results_steps, sizeof(int) * count, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < count; i++)
    {
        result.valid[i] =
            (result.hitPoint[i].x != FLT_INF && result.hitPoint[i].y != FLT_INF && result.hitPoint[i].z != FLT_INF);
        if (result.valid)
        {
            float dtx = origin[i].x - result.hitPoint[i].x;
            float dty = origin[i].y - result.hitPoint[i].y;
            float dtz = origin[i].z - result.hitPoint[i].z;
            result.distance[i] = sqrt(dtx * dtx + dty * dty + dtz * dtz);
            result.voxelIndex[i] = (int)(result.hitPoint[i].z * dimensions.x * dimensions.y +
                                         result.hitPoint[i].y * dimensions.x + result.hitPoint[i].x);
        }
    }

    cudaDeviceSynchronize();
    return result;
}

} // namespace GPUDDA
