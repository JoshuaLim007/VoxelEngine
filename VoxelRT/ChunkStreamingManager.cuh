#pragma once
#ifndef CHUNK_STREAMING_MANAGER_CUH
#define CHUNK_STREAMING_MANAGER_CUH

// ============================================================
// ChunkStreamingManager
// Camera-driven async voxel chunk streaming system.
//
// Design overview
// ---------------
// The world (4096 x 512 x 4096 voxels) is partitioned into
// SuperChunks (SC_VOXEL_DIM^3 = 256^3 voxels each).  The
// manager keeps only the SCs nearest to the camera loaded on
// the GPU; the rest are generated on demand by a CPU thread
// pool and transferred asynchronously, never blocking the
// render thread.
//
// Chunk priority = view-alignment * distance weight, so chunks
// directly ahead of the camera are built before ones behind.
//
// GPU memory layout (owned entirely by this manager)
// ---------------------------------------------------
//  d_pool_data_    : flat packed-bit array, MAX_POOL_BRICKS slots
//  d_brick_indices_: per-brick slot (UINT32_MAX = not loaded)
//  d_brick_bounds_ : tight occupied-voxel bounds for each pool slot
//  d_dist_field_   : Chebyshev distance for empty-space skip
//  d_lowres_bits_  : packed-bit occupancy for the low-res DDA grid
//
// Thread safety
// -------------
// UpdateCamera() and FlushUploads() must be called from the
// main (render) thread.  Background workers only write to the
// upload_queue_ (mutex-protected).  All GPU and CPU state
// arrays are touched exclusively on the main thread.
// ============================================================

#include <atomic>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include "VolumeRaytracer.cuh"

namespace GPUDDA {

// ============================================================
// World / chunk dimension constants
// ============================================================
constexpr uint32_t STREAM_WORLD_VOXEL_X  = 4096;
constexpr uint32_t STREAM_WORLD_VOXEL_Y  = 512;
constexpr uint32_t STREAM_WORLD_VOXEL_Z  = 4096;

// brick_dim must match the factor used in the renderer (32)
constexpr uint32_t STREAM_BRICK_DIM      = 32;
// Number of bricks per super-chunk axis
constexpr uint32_t SC_BRICK_DIM          = 8;
// Voxels per super-chunk axis
constexpr uint32_t SC_VOXEL_DIM          = STREAM_BRICK_DIM * SC_BRICK_DIM; // 256

// World dimensions in super-chunks
constexpr uint32_t WORLD_SC_X            = STREAM_WORLD_VOXEL_X / SC_VOXEL_DIM; // 16
constexpr uint32_t WORLD_SC_Y            = STREAM_WORLD_VOXEL_Y / SC_VOXEL_DIM; // 2
constexpr uint32_t WORLD_SC_Z            = STREAM_WORLD_VOXEL_Z / SC_VOXEL_DIM; // 16

// Low-res grid dimensions (world in bricks, matches RaytraceFast)
constexpr uint32_t LR_DIM_X              = STREAM_WORLD_VOXEL_X / STREAM_BRICK_DIM; // 128
constexpr uint32_t LR_DIM_Y              = STREAM_WORLD_VOXEL_Y / STREAM_BRICK_DIM; // 16
constexpr uint32_t LR_DIM_Z              = STREAM_WORLD_VOXEL_Z / STREAM_BRICK_DIM; // 128
constexpr uint32_t LR_TOTAL              = LR_DIM_X * LR_DIM_Y * LR_DIM_Z;          // 262144

// Bricks per super-chunk
constexpr uint32_t BRICKS_PER_SC        = SC_BRICK_DIM * SC_BRICK_DIM * SC_BRICK_DIM; // 512

// GPU pool: pre-allocated brick slots.
// Sized to the world upper bound so pool exhaustion cannot happen under
// the current streaming policy (ignoring memory pressure by design).
constexpr uint32_t MAX_POOL_BRICKS       =
    WORLD_SC_X * WORLD_SC_Y * WORLD_SC_Z * BRICKS_PER_SC;

// Camera-relative render distance in world voxels.
// Scheduling/filtering is done in super-chunk coordinates derived from this.
constexpr float    STREAM_LOAD_RADIUS      = 1024.0f * 2;
constexpr int      STREAM_RENDER_RADIUS_SC =
    static_cast<int>(STREAM_LOAD_RADIUS / static_cast<float>(SC_VOXEL_DIM));

// Maximum count of in-flight chunk build requests (queued + currently building).
constexpr uint32_t MAX_QUEUED_CHUNKS     = 128;

// Background worker threads for chunk generation.
// Uses hardware_concurrency - 1 (capped), or at least 1.
constexpr uint32_t MAX_WORKER_THREADS    = 4;

// ============================================================
// Structs
// ============================================================

struct ChunkKey {
    uint16_t x, y, z;
    bool operator==(const ChunkKey& o) const {
        return x == o.x && y == o.y && z == o.z;
    }
};

struct ChunkKeyHash {
    size_t operator()(const ChunkKey& k) const {
        size_t h = static_cast<size_t>(k.x);
        h = h * 2654435761u ^ static_cast<size_t>(k.y);
        h = h * 2654435761u ^ static_cast<size_t>(k.z);
        return h;
    }
};

// Per-brick occupancy count within a finished super-chunk.
// local_brick_seq[i]  = sequential index of brick i inside brick_data, or
//                       UINT32_MAX when brick i is completely empty.
// brick_data stores occupied-brick packed-bit data back to back;
// sequential_index k starts at brick_data[k * brick_words].
struct ChunkBuildResult {
    ChunkKey              key;
    std::vector<uint32_t> brick_data;        // occupied_count * brick_words uint32_ts
    std::vector<BrickBounds> brick_bounds;   // occupied_count, matches brick_data order
    std::vector<uint32_t> local_brick_seq;   // BRICKS_PER_SC entries
    uint32_t              occupied_count = 0;
    bool                  canceled = false;
};

// Priority policy interface. Higher score means higher scheduling priority.
class IChunkScoringPolicy {
public:
    virtual ~IChunkScoringPolicy() = default;
    virtual float Score(const ChunkKey& camera_sc, const ChunkKey& candidate_sc) const = 0;
};

// Current default policy: Manhattan distance in super-chunk space.
// This is quantized to chunk coordinates (not world-space float position).
class ManhattanChunkScoringPolicy final : public IChunkScoringPolicy {
public:
    float Score(const ChunkKey& camera_sc, const ChunkKey& candidate_sc) const override;
};

// ============================================================
// ChunkStreamingManager
// ============================================================
class ChunkStreamingManager {
public:
    ChunkStreamingManager();
    ~ChunkStreamingManager();

    // Allocate GPU pool, bind it to the raytracer, and launch workers.
    // brick_words = ceil(STREAM_BRICK_DIM^3 / 32).
    void Init(VoxelRaytracer3D* raytracer, uint32_t brick_words);

    // Call once per frame: schedules loads based on camera state.
    void UpdateCamera(float3 cam_pos, float3 cam_fwd);

    // Call once per frame: integrates all completed chunks into the GPU pool.
    // Completed chunks outside render distance are dropped immediately.
    // Returns true if GPU data changed.
    bool FlushUploads(VoxelRaytracer3D* raytracer);

    // Blocking helper: processes uploads in a tight loop until at least
    // min_chunks SCs near cam_pos are loaded.  Call before entering the
    // render loop to avoid showing an empty world on startup.
    void WaitForInitialChunks(VoxelRaytracer3D* raytracer, float3 cam_pos, int min_chunks);

    // GPU resource accessors (pointers remain valid for lifetime of manager)
    uint32_t* GetGPUBrickData()    const { return d_pool_data_;    }
    uint32_t* GetGPUBrickIndices() const { return d_brick_indices_; }
    BrickBounds* GetGPUBrickBounds() const { return d_brick_bounds_; }
    uint8_t*  GetGPUDistField()    const { return d_dist_field_;    }
    uint32_t* GetGPULowResBits()   const { return d_lowres_bits_;   }
    uint32_t  BrickWords()         const { return brick_words_;      }

private:
    // ---- Worker thread ----
    void WorkerThread();
    ChunkBuildResult BuildChunk(const ChunkKey& key);

    // ---- Per-frame state integration ----
    void IntegrateResult(const ChunkBuildResult& result);
    void EvictChunk(const ChunkKey& key);
    void UploadGPUState();
    bool IsWithinRenderDistance(const ChunkKey& key, const ChunkKey& cam_sc) const;
    bool IsBuildCancelled(const ChunkKey& key);
    void RequestBuildCancel(const ChunkKey& key);

    // ---- Distance field (rebuilt off the main thread) ----
    // When occupancy changes, IntegrateResult/EvictChunk sets df_rebuild_requested_
    // and signals df_cv_.  DFThread wakes, snapshots occupancy under occ_mutex_,
    // computes the Chebyshev DF in the background, and stores the result.
    // UploadGPUState picks it up on the next call without stalling the render loop.
    void DFThread();
    void BuildDFFromSnapshot(const std::vector<uint8_t>& occ,
                              const std::vector<bool>&    loaded,
                              std::vector<uint8_t>&       out);

    std::thread              df_thread_;
    std::atomic<bool>        df_shutdown_{false};
    // df_mutex_ guards df_rebuild_requested_, df_result_, df_result_ready_
    std::mutex               df_mutex_;
    std::condition_variable  df_cv_;
    bool                     df_rebuild_requested_ = false;
    std::vector<uint8_t>     df_result_;
    bool                     df_result_ready_      = false;

    // ---- Slot allocator (contiguous-range free list) ----
    struct FreeRange { uint32_t start, length; };
    uint32_t AllocContig(uint32_t count);   // returns base slot or UINT32_MAX
    void     FreeContig (uint32_t base, uint32_t count);

    // ---- Priority ----
    float ChunkPriority(const ChunkKey& camera_sc, const ChunkKey& key) const;
    std::unique_ptr<IChunkScoringPolicy> scoring_policy_;

    // ---- GPU resources (owned by this manager) ----
    uint32_t* d_pool_data_    = nullptr; // MAX_POOL_BRICKS * brick_words_ uint32_ts
    uint32_t* d_brick_indices_= nullptr; // LR_TOTAL uint32_ts
    BrickBounds* d_brick_bounds_ = nullptr; // MAX_POOL_BRICKS entries
    uint8_t*  d_dist_field_   = nullptr; // LR_TOTAL uint8_ts
    uint32_t* d_lowres_bits_  = nullptr; // (LR_TOTAL + 31)/32 uint32_ts

    // ---- CPU shadow of GPU state ----
    uint32_t  brick_words_ = 0;
    uint32_t  lr_words_    = 0;   // (LR_TOTAL + 31) / 32

    // Pinned (page-locked) host buffers.  cudaMallocHost guarantees DMA-direct
    // transfers, so cudaMemcpyAsync truly returns to the CPU immediately.
    // Written only on the main thread; GPU reads via async DMA.
    uint32_t* pin_brick_indices_ = nullptr;  // LR_TOTAL entries
    uint8_t*  pin_dist_field_    = nullptr;  // LR_TOTAL entries
    uint32_t* pin_lr_bits_       = nullptr;  // lr_words_ entries (maintained incrementally)

    // Per-brick occupancy and per-SC load state (main thread only).
    // Protected by occ_mutex_ for the brief snapshot the DF thread takes.
    std::vector<uint8_t> cpu_occupancy_;  // LR_TOTAL entries (0 or 1)
    std::vector<bool>    cpu_sc_loaded_;  // WORLD_SC_X * Y * Z entries
    mutable std::mutex   occ_mutex_;

    bool gpu_state_dirty_ = false;

    // ---- Slot allocator ----
    std::list<FreeRange>  free_ranges_;

    // ---- Loaded chunk registry (main thread only) ----
    struct LoadedSC {
        uint32_t base_slot;    // first GPU pool slot for this SC's occupied bricks
        uint32_t brick_count;  // number of occupied bricks
    };
    std::unordered_map<ChunkKey, LoadedSC, ChunkKeyHash> loaded_;

    // ---- In-flight set (chunks being built; protected by in_flight_mutex_) ----
    std::unordered_set<ChunkKey, ChunkKeyHash> in_flight_;
    std::mutex in_flight_mutex_;

    // ---- Active build tracking + cancellation (worker + main thread) ----
    std::unordered_set<ChunkKey, ChunkKeyHash> building_;
    std::unordered_set<ChunkKey, ChunkKeyHash> cancel_requested_;
    std::mutex                                  build_state_mutex_;

    // ---- Worker thread pool ----
    std::vector<std::thread> workers_;
    std::atomic<bool>        shutdown_{false};

    // ---- Build request priority queue (max-heap by priority) ----
    struct BuildRequest {
        ChunkKey key;
        float    priority;
        bool operator<(const BuildRequest& o) const { return priority < o.priority; }
    };
    std::priority_queue<BuildRequest> build_queue_;
    std::mutex                        build_queue_mutex_;
    std::condition_variable           build_queue_cv_;

    // ---- Completed build results (workers → main thread) ----
    std::queue<ChunkBuildResult> upload_queue_;
    std::mutex                   upload_queue_mutex_;

    // ---- Camera snapshot (updated by UpdateCamera, read by FlushUploads) ----
    mutable std::mutex cam_mutex_;
    ChunkKey           cam_sc_snapshot_{0u, 0u, 0u};
};

} // namespace GPUDDA

#endif // CHUNK_STREAMING_MANAGER_CUH
