#include "VolumeRaytracer.cuh"

#include "helper_math.h"
#include <chrono>
#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
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
__host__ __device__ BitRef& BitRef::operator=(bool value)
{
#if defined(__CUDA_ARCH__)
    uint32_t mask = 1u << (index & 31);
    if (value)
        atomicOr(byte, mask);
    else
        atomicAnd(byte, ~mask);
#else
    uint32_t mask = 1u << (index & 31);
    std::atomic<uint32_t>* ref = reinterpret_cast<std::atomic<uint32_t>*>(byte);
    if (value)
        ref->fetch_or(mask, std::memory_order_relaxed);
    else
        ref->fetch_and(~mask, std::memory_order_relaxed);
#endif
    return *this;
}
__device__ __host__ BitArray::BitArray() : size(0), data(nullptr)
{
}
__device__ __host__ BitArray::BitArray(const BitArray &other, bool isGPU) : size(other.size)
{
    if (isGPU)
    {
        cudaMalloc((void **)&data, (size + 31) / 32 * sizeof(uint32_t));
        cudaMemcpy(data, other.data, (size + 31) / 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        return;
    }
    data = new uint32_t[(size + 31) / 32];
    std::copy(other.data, other.data + (size + 31) / 32, data);
}
__device__ __host__ BitArray::BitArray(size_t num_bits, bool isGPU) : size(num_bits)
{
    if (isGPU)
    {
        cudaMalloc((void **)&data, (size + 31) / 32 * sizeof(uint32_t));
        return;
    }
    data = new uint32_t[(size + 31) / 32];
}

__device__ __host__ bool BitArray::operator[](size_t index) const
{
    if (index >= size)
    {
        return false; // Out of bounds
    }
    return (data[index / 32] >> (index % 32)) & 1;
}

__device__ __host__ BitRef BitArray::operator[](size_t index)
{
    return BitRef{&data[index / 32], static_cast<size_t>(index % 32)};
}
__device__ __host__ uint32_t *BitArray::Raw()
{
    return data;
}
__device__ __host__ size_t BitArray::BitSize() const
{
    return size;
}
__device__ __host__ size_t BitArray::ByteSize() const
{
    return ((size + 31) / 32) * sizeof(uint32_t);
}
std::ostream &operator<<(std::ostream &os, const BitArray &bits)
{
    for (size_t i = 0; i < bits.BitSize(); ++i)
    {
        os << bits[i]; // Print each bit
    }
    return os;
}

__global__ void dispatch(const float3 *__restrict__ origins, const float3 *__restrict__ rays,
                         VoxelBuffer3D *__restrict__ chunks, VoxelBuffer3D *__restrict__ chunksData,
                         Bounds3Df *__restrict__ chunkBoundingBoxes, int factor,
                         float3 *__restrict__ results_point, float3 *__restrict__ results_normal,
                         int *__restrict__ results_steps, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        int steps;
        float3 normal;
        float3 pos;
        const float3 rayDir = normalize(rays[idx]);
        if (Raytrace(MAX_STEPS, origins[idx], rayDir, chunks[0], chunksData, chunkBoundingBoxes, factor, steps,
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
        *out_intersect = make_float3(start.x + t_min * direction.x, start.y + t_min * direction.y, start.z + t_min * direction.z);
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

    DDARayResults<float3> returnResults;
    returnResults.HitIntersectedPoint = make_float3(x, y, z);
    returnResults.hit = false;
    returnResults.isOutOfBounds = false;
    returnResults.stepsTaken = 0;

    auto grid = Params.VoxelBuffer.grid;
    uint32_t* gridRaw = grid.Raw();
    bool exit = false;
    const bool hasPerVoxelBounds = (Params.per_voxel_bounds != nullptr);
    const bool hasTraversalBounds = (Params.bounds != nullptr);
    const float invPerVoxelBoundsScale = hasPerVoxelBounds ? (1.0f / Params.per_voxel_bounds_scale) : 0.0f;

    int bounds_min_x = 0;
    int bounds_min_y = 0;
    int bounds_min_z = 0;
    int bounds_max_x = 0;
    int bounds_max_y = 0;
    int bounds_max_z = 0;
    if (hasTraversalBounds)
    {
        bounds_min_x = Params.bounds->min.x;
        bounds_min_y = Params.bounds->min.y;
        bounds_min_z = Params.bounds->min.z;
        bounds_max_x = Params.bounds->max.x;
        bounds_max_y = Params.bounds->max.y;
        bounds_max_z = Params.bounds->max.z;
    }

    bool IsOnEdge = cell_x == cols || cell_y == rows || cell_z == depth;
    int edgePadding_x = 0;
    int edgePadding_y = 0;
    int edgePadding_z = 0;
    if (IsOnEdge)
    {
        if (dx < 0)
        {
            edgePadding_x = 1;
        }
        if (dy < 0)
        {
            edgePadding_y = 1;
        }
        if (dz < 0)
        {
            edgePadding_z = 1;
        }
    }

    for (int step = 0; step < Params.max_steps; ++step)
    {
        bool skipCheck = Params.takeInitialStep && step == 0;

        if (!skipCheck)
        {
            if ((unsigned)cell_x < (unsigned)(cols + edgePadding_x) &&
                (unsigned)cell_y < (unsigned)(rows + edgePadding_y) &&
                (unsigned)cell_z < (unsigned)(depth + edgePadding_z))
            {
                int clamped_x = (cell_x < 0) ? 0 : ((cell_x >= cols) ? (cols - 1) : cell_x);
                int clamped_y = (cell_y < 0) ? 0 : ((cell_y >= rows) ? (rows - 1) : cell_y);
                int clamped_z = (cell_z < 0) ? 0 : ((cell_z >= depth) ? (depth - 1) : cell_z);
                returnResults.HitCell = make_float3(clamped_x, clamped_y, clamped_z);

                int idx = GetSampleIndex(clamped_x, clamped_y, clamped_z, cols, rows);
                bool occupied = ((gridRaw[idx >> 5] >> (idx & 31)) & 1u) != 0;

                if (hasPerVoxelBounds)
                {
                    if (occupied)
                    {
                        const auto& voxelBound = Params.per_voxel_bounds[idx];
                        float bmin_x = voxelBound.min.x * invPerVoxelBoundsScale + clamped_x;
                        float bmin_y = voxelBound.min.y * invPerVoxelBoundsScale + clamped_y;
                        float bmin_z = voxelBound.min.z * invPerVoxelBoundsScale + clamped_z;
                        float bmax_x = (voxelBound.max.x + 1.0f) * invPerVoxelBoundsScale + clamped_x;
                        float bmax_y = (voxelBound.max.y + 1.0f) * invPerVoxelBoundsScale + clamped_y;
                        float bmax_z = (voxelBound.max.z + 1.0f) * invPerVoxelBoundsScale + clamped_z;
                        if (bmin_x <= bmax_x)
                        {
                            float3 aabb_normal = make_float3(0, 0, 0);
                            float3 aabb_pos = make_float3(0, 0, 0);
                            if (RayIntersectsAABB(Params.start, Params.direction,
                                                  make_float3(bmin_x, bmin_y, bmin_z), make_float3(bmax_x, bmax_y, bmax_z),
                                                  &aabb_pos, &aabb_normal))
                            {
                                returnResults.hit = true;
                                returnResults.HitNormal = aabb_normal;
                                if (step != 0)
                                {
                                    returnResults.HitIntersectedPoint = aabb_pos;
                                }
                                exit = true;
                            }
                        }
                    }
                }
                else
                {
                    if (occupied)
                    {
                        returnResults.hit = true;
                        exit = true;
                    }
                }
            }
            else
            {
                returnResults.isOutOfBounds = true;
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
                returnResults.HitNormal = make_float3(step_x, 0, 0);
        }
        else if (tMax_y <= tMax_x && tMax_y < tMax_z)
        {
            intersect_x = x + (tMax_y * dx);
            intersect_y = cell_y + (step_y > 0);
            intersect_z = z + (tMax_y * dz);
            cell_y += step_y;
            tMax_y += tDelta_y;
            if (!exit)
                returnResults.HitNormal = make_float3(0, step_y, 0);
        }
        else
        {
            intersect_x = x + (tMax_z * dx);
            intersect_y = y + (tMax_z * dy);
            intersect_z = cell_z + (step_z > 0);
            cell_z += step_z;
            tMax_z += tDelta_z;
            if (!exit)
                returnResults.HitNormal = make_float3(0, 0, step_z);
        }
        if (!exit)
        {
            if (hasTraversalBounds)
            {
                // Check if the intersection point is within the bounds
                bool isOutOfBounds = (intersect_x < bounds_min_x || intersect_x > bounds_max_x ||
                                      intersect_y < bounds_min_y || intersect_y > bounds_max_y ||
                                      intersect_z < bounds_min_z || intersect_z > bounds_max_z);
                if (isOutOfBounds)
                {
                    returnResults.isOutOfBounds = true;
                    break;
                }
            }
            returnResults.stepsTaken += 1;
            returnResults.HitIntersectedPoint = make_float3(intersect_x, intersect_y, intersect_z);
        }
        else
        {
            returnResults.NextCell = make_float3(cell_x, cell_y, cell_z);
            break;
        }
    }
    Results = returnResults;
}

__device__ bool Raytrace(int maxSteps, float3 origin, float3 ray, VoxelBuffer3D chunks, VoxelBuffer3D *chunksData,
                         Bounds3Df *chunkBoundingBoxes, int factor, int &out_steps, float3 &out_normal,
                         float3 &out_pos)
{

    int3 previous_cell = make_int3(-1, -1, -1);
    int total_steps = 0;

    float3 start = origin;
    const float invFactor = 1.0f / factor;
    start.x *= invFactor;
    start.y *= invFactor;
    start.z *= invFactor;

    float3 direction = ray;
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
    Bounds3Df chunkBounds{};
    chunkBounds.min = make_float3(0, 0, 0);
    chunkBounds.max = make_float3(factor, factor, factor);
    DDARayParams<float3, 3> params = DDARayParams<float3, 3>::Default(chunks, start, direction);
    params.per_voxel_bounds = chunkBoundingBoxes;
    params.per_voxel_bounds_scale = factor;

    while (total_steps < maxSteps)
    {
        float3 start_high_res;
        params.start = start;
        DDARayResults<float3> results;
        DDARayTraversal(params, results);

        total_steps += results.stepsTaken;
        start_high_res = make_float3(results.HitIntersectedPoint.x * factor, results.HitIntersectedPoint.y * factor,
                                     results.HitIntersectedPoint.z * factor);
        hitPosition = start_high_res;
        if (results.hit && !results.isOutOfBounds)
        {
            int hitCellX = static_cast<int>(results.HitCell.x);
            int hitCellY = static_cast<int>(results.HitCell.y);
            int hitCellZ = static_cast<int>(results.HitCell.z);

            if (previous_cell.x == hitCellX &&
                previous_cell.y == hitCellY &&
                previous_cell.z == hitCellZ)
            {
                break;
            }
            previous_cell = make_int3(hitCellX, hitCellY, hitCellZ);

            start_high_res.x -= hitCellX * factor;
            start_high_res.y -= hitCellY * factor;
            start_high_res.z -= hitCellZ * factor;

            int index = GetSampleIndex(hitCellX, hitCellY, hitCellZ, chunks.dimensions[0], chunks.dimensions[1]);
            VoxelBuffer3D chunkData = chunksData[index];
            DDARayParams<float3, 3> params_hr = DDARayParams<float3, 3>::Default(chunkData, start_high_res, direction);
            params_hr.bounds = &chunkBounds;
            DDARayResults<float3> results_hr;
            DDARayTraversal(params_hr, results_hr);

            total_steps += results_hr.stepsTaken;
            hitPosition = make_float3(results_hr.HitIntersectedPoint.x + hitCellX * factor,
                                      results_hr.HitIntersectedPoint.y + hitCellY * factor,
                                      results_hr.HitIntersectedPoint.z + hitCellZ * factor);

            if (!results_hr.hit)
            {
                start = hitPosition;
                start.x *= invFactor;
                start.y *= invFactor;
                start.z *= invFactor;

                if (results_hr.isOutOfBounds)
                {
                    // projected cell
                    int cx = static_cast<int>(start.x);
                    int cy = static_cast<int>(start.y);
                    int cz = static_cast<int>(start.z);
                    bool projectedCellIsSame = hitCellX == cx && hitCellY == cy && hitCellZ == cz;

                    // apply the smallest diff to start
                    if (projectedCellIsSame)
                    {
                        // first apply eps to see if it crosses chunk border
                        if (hitCellX == cx)
                        {
                            start.x = direction.x < 0 ? nextafterf(start.x, -FLT_INF) : nextafterf(start.x, FLT_INF);
                        }
                        if (hitCellY == cy)
                        {
                            start.y = direction.y < 0 ? nextafterf(start.y, -FLT_INF) : nextafterf(start.y, FLT_INF);
                        }
                        if (hitCellZ == cz)
                        {
                            start.z = direction.z < 0 ? nextafterf(start.z, -FLT_INF) : nextafterf(start.z, FLT_INF);
                        }

                        // projected cell
                        int cx = static_cast<int>(start.x);
                        int cy = static_cast<int>(start.y);
                        int cz = static_cast<int>(start.z);
                        projectedCellIsSame = hitCellX == cx && hitCellY == cy && hitCellZ == cz;

                        // if projected cell is still the same, apply the smallest diff
                        if (projectedCellIsSame)
                        {
                            // find the smallest diff to the next cell
                            float3 diff = make_float3(results.NextCell.x - start.x, results.NextCell.y - start.y, results.NextCell.z - start.z);
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
            break;
        }
    }

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

void VoxelRaytracer3D::UploadVoxelBuffer(const GPUDDA::VoxelBuffer3D &buff)
{

    if (gpu_VoxelBuffer != nullptr)
    {
        if (gpu_VoxelBufferGridData != nullptr)
        {
            cudaFree(gpu_VoxelBufferGridData);
            gpu_VoxelBufferGridData = nullptr;
        }
        cudaFree(gpu_VoxelBuffer);
        gpu_VoxelBuffer = nullptr;
    }

    dimensions.x = buff.dimensions[0];
    dimensions.y = buff.dimensions[1];
    dimensions.z = buff.dimensions[2];

    VoxelBuffer3D temp;
    temp.grid = BitArray(buff.grid, true);
    gpu_VoxelBufferGridData = temp.grid.Raw();
    temp.dimensions[0] = buff.dimensions[0];
    temp.dimensions[1] = buff.dimensions[1];
    temp.dimensions[2] = buff.dimensions[2];

    cudaMalloc((void **)&gpu_VoxelBuffer, sizeof(GPUDDA::VoxelBuffer3D));
    cudaMemcpy(gpu_VoxelBuffer, &temp, sizeof(GPUDDA::VoxelBuffer3D), cudaMemcpyHostToDevice);
}

void VoxelRaytracer3D::UploadVoxelBufferDatas(GPUDDA::VoxelBuffer3D *buff, size_t count)
{
    if (gpu_VoxelBufferDatas != nullptr)
    {
        for (auto *ptr : gpu_VoxelBufferDataGridPointers)
        {
            if (ptr != nullptr)
            {
                cudaFree(ptr);
            }
        }
        gpu_VoxelBufferDataGridPointers.clear();
        cudaFree(gpu_VoxelBufferDatas);
        gpu_VoxelBufferDatas = nullptr;
    }

    GPUDDA::VoxelBuffer3D *temp = new GPUDDA::VoxelBuffer3D[count];
    gpu_VoxelBufferDataGridPointers.reserve(count);
    for (size_t i = 0; i < count; i++)
    {
        temp[i].dimensions[0] = buff[i].dimensions[0];
        temp[i].dimensions[1] = buff[i].dimensions[1];
        temp[i].dimensions[2] = buff[i].dimensions[2];
        temp[i].grid = BitArray(buff[i].grid, true);
        gpu_VoxelBufferDataGridPointers.push_back(temp[i].grid.Raw());
    }
    auto memSize = sizeof(GPUDDA::VoxelBuffer3D) * count;
    cudaMalloc((void **)&gpu_VoxelBufferDatas, memSize);
    cudaMemcpy(gpu_VoxelBufferDatas, temp, memSize, cudaMemcpyHostToDevice);
    delete[] temp;
}

void VoxelRaytracer3D::UploadVoxelBufferDataBounds(Bounds3Df *bounds, size_t count)
{
    if (gpu_VoxelBufferDataBounds != nullptr)
    {
        cudaFree(gpu_VoxelBufferDataBounds);
        gpu_VoxelBufferDataBounds = nullptr;
    }
    auto memSize = sizeof(Bounds3Df) * count;
    cudaMalloc((void **)&gpu_VoxelBufferDataBounds, memSize);
    cudaMemcpy(gpu_VoxelBufferDataBounds, bounds, memSize, cudaMemcpyHostToDevice);
}

RayTraceResults<float3> VoxelRaytracer3D::Raytrace(std::vector<float3> origin, std::vector<float3> ray)
{
    auto& result = resultsCPU;
    int count = origin.size();

    cudaMemcpy(d_origins, origin.data(), sizeof(float3) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rays, ray.data(), sizeof(float3) * count, cudaMemcpyHostToDevice);

    dim3 blockSize(128, 1, 1);
    dim3 numBlocks((count + blockSize.x - 1) / blockSize.x, 1, 1);

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
        if (result.valid[i])
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
