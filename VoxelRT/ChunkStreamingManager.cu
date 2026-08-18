#include "ChunkStreamingManager.cuh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>

namespace GPUDDA {

// ============================================================
// CPU Perlin noise — exact port of cuda_noise.cuh functions.
// repeaterPerlin() ignores its `seed` parameter (matching the
// GPU implementation), so terrain is pixel-identical to the
// GPU-generated world.
// ============================================================
namespace CPUNoise {

inline unsigned int hash_fn(unsigned int seed) {
    seed = (seed + 0x7ed55d16u) + (seed << 12);
    seed = (seed ^ 0xc761c23cu) ^ (seed >> 19);
    seed = (seed + 0x165667b1u) + (seed << 5);
    seed = (seed + 0xd3a2646cu) ^ (seed << 9);
    seed = (seed + 0xfd7046c5u) + (seed << 3);
    seed = (seed ^ 0xb55a4f09u) ^ (seed >> 16);
    return seed;
}

inline float randomFloat(unsigned int seed) {
    return static_cast<float>(hash_fn(seed)) / static_cast<float>(0xffffffffu);
}

inline unsigned int randomIntGrid(float x, float y, float z, float seed = 0.f) {
    return hash_fn(static_cast<unsigned int>(x * 1723.f + y * 93241.f + z * 149812.f + 3824.f + seed));
}

inline float grad(int h, float x, float y, float z) {
    switch (h & 0xF) {
        case 0x0: return  x + y;
        case 0x1: return -x + y;
        case 0x2: return  x - y;
        case 0x3: return -x - y;
        case 0x4: return  x + z;
        case 0x5: return -x + z;
        case 0x6: return  x - z;
        case 0x7: return -x - z;
        case 0x8: return  y + z;
        case 0x9: return -y + z;
        case 0xA: return  y - z;
        case 0xB: return -y - z;
        case 0xC: return  y + x;
        case 0xD: return -y + z;
        case 0xE: return  y - x;
        case 0xF: return -y - z;
        default:  return 0.f;
    }
}

inline float fade(float t) {
    return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
}

inline float lerp(float a, float b, float t) {
    return a * (1.f - t) + b * t;
}

// Matches cudaNoise::perlinNoise(pos, scale, seed)
inline float perlinNoise(float px, float py, float pz, float scale, int seed) {
    const float fseed = static_cast<float>(seed);
    px *= scale; py *= scale; pz *= scale;
    float ix = std::floor(px), iy = std::floor(py), iz = std::floor(pz);
    px -= ix; py -= iy; pz -= iz;
    const float u = fade(px), v = fade(py), w = fade(pz);

    const float i000 = grad(randomIntGrid(ix,      iy,      iz,      fseed), px,      py,      pz     );
    const float i100 = grad(randomIntGrid(ix+1.f,  iy,      iz,      fseed), px-1.f,  py,      pz     );
    const float i010 = grad(randomIntGrid(ix,      iy+1.f,  iz,      fseed), px,      py-1.f,  pz     );
    const float i110 = grad(randomIntGrid(ix+1.f,  iy+1.f,  iz,      fseed), px-1.f,  py-1.f,  pz     );
    const float i001 = grad(randomIntGrid(ix,      iy,      iz+1.f,  fseed), px,      py,      pz-1.f );
    const float i101 = grad(randomIntGrid(ix+1.f,  iy,      iz+1.f,  fseed), px-1.f,  py,      pz-1.f );
    const float i011 = grad(randomIntGrid(ix,      iy+1.f,  iz+1.f,  fseed), px,      py-1.f,  pz-1.f );
    const float i111 = grad(randomIntGrid(ix+1.f,  iy+1.f,  iz+1.f,  fseed), px-1.f,  py-1.f,  pz-1.f);

    const float x00 = lerp(i000, i100, u);
    const float x10 = lerp(i010, i110, u);
    const float x01 = lerp(i001, i101, u);
    const float x11 = lerp(i011, i111, u);
    return lerp(lerp(x00, x10, v), lerp(x01, x11, v), w);
}

// Matches cudaNoise::repeaterPerlin(pos, scale, seed, n, lacunarity, decay).
// NOTE: `seed` is intentionally unused — the GPU version ignores it too.
inline float repeaterPerlin(float px, float py, float pz,
                             float scale, int /*seed*/,
                             int n, float lacunarity, float decay) {
    float acc = 0.f, amp = 1.f;
    for (int i = 0; i < n; ++i) {
        acc += perlinNoise(px * scale, py * scale, pz * scale, 1.f, (i + 38) * 27389482) * amp;
        scale *= lacunarity;
        amp   *= decay;
    }
    return acc;
}

} // namespace CPUNoise

float ManhattanChunkScoringPolicy::Score(const ChunkKey& camera_sc,
                                         const ChunkKey& candidate_sc) const {
    const int dx = std::abs(static_cast<int>(candidate_sc.x) - static_cast<int>(camera_sc.x));
    //const int dy = std::abs(static_cast<int>(candidate_sc.y) - static_cast<int>(camera_sc.y));
    const int dz = std::abs(static_cast<int>(candidate_sc.z) - static_cast<int>(camera_sc.z));
    const int manhattan = dx + 0 + dz;
    // Higher score means more urgent. Add 1 to avoid divide-by-zero at camera chunk.
    return 1.0f / static_cast<float>(manhattan + 1);
}

// ============================================================
// ChunkStreamingManager
// ============================================================

ChunkStreamingManager::ChunkStreamingManager()
    : scoring_policy_(std::make_unique<ManhattanChunkScoringPolicy>()) {}

ChunkStreamingManager::~ChunkStreamingManager() {
    // Shut down worker threads
    shutdown_.store(true, std::memory_order_relaxed);
    build_queue_cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();

    // Shut down DF thread
    df_shutdown_.store(true, std::memory_order_relaxed);
    df_cv_.notify_all();
    if (df_thread_.joinable()) df_thread_.join();

    // Free GPU resources
    if (d_pool_data_)     { cudaFree(d_pool_data_);     d_pool_data_     = nullptr; }
    if (d_brick_indices_) { cudaFree(d_brick_indices_); d_brick_indices_ = nullptr; }
    if (d_brick_bounds_)  { cudaFree(d_brick_bounds_);  d_brick_bounds_  = nullptr; }
    if (d_dist_field_)    { cudaFree(d_dist_field_);    d_dist_field_    = nullptr; }
    if (d_lowres_bits_)   { cudaFree(d_lowres_bits_);   d_lowres_bits_   = nullptr; }

    // Free pinned host buffers
    if (pin_brick_indices_) { cudaFreeHost(pin_brick_indices_); pin_brick_indices_ = nullptr; }
    if (pin_dist_field_)    { cudaFreeHost(pin_dist_field_);    pin_dist_field_    = nullptr; }
    if (pin_lr_bits_)       { cudaFreeHost(pin_lr_bits_);       pin_lr_bits_       = nullptr; }
}

// ============================================================
void ChunkStreamingManager::Init(VoxelRaytracer3D* raytracer, uint32_t brick_words) {
    brick_words_ = brick_words;

    const size_t pool_bytes    = static_cast<size_t>(MAX_POOL_BRICKS) * brick_words * sizeof(uint32_t);
    const size_t indices_bytes = LR_TOTAL * sizeof(uint32_t);
    const size_t bounds_bytes  = static_cast<size_t>(MAX_POOL_BRICKS) * sizeof(BrickBounds);
    const size_t df_bytes      = LR_TOTAL * sizeof(uint8_t);
    const size_t lr_words      = (LR_TOTAL + 31u) / 32u;
    const size_t lr_bits_bytes = lr_words * sizeof(uint32_t);

    cudaMalloc(reinterpret_cast<void**>(&d_pool_data_),     pool_bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_brick_indices_), indices_bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_brick_bounds_),  bounds_bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_dist_field_),    df_bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_lowres_bits_),   lr_bits_bytes);

    // UINT32_MAX (0xFFFFFFFF) per index → all unloaded
    cudaMemset(d_brick_indices_, 0xFF, indices_bytes);
    // dist = 1 everywhere: no accidental skipping through unloaded regions
    cudaMemset(d_dist_field_, 0x01, df_bytes);
    // occupancy = 0: all empty
    cudaMemset(d_lowres_bits_, 0x00, lr_bits_bytes);

    lr_words_ = static_cast<uint32_t>(lr_words);

    // Pinned host staging buffers — page-locked memory lets cudaMemcpyAsync
    // DMA directly without an internal staging copy, so it truly returns
    // to the CPU immediately.
    cudaMallocHost(reinterpret_cast<void**>(&pin_brick_indices_), indices_bytes);
    cudaMallocHost(reinterpret_cast<void**>(&pin_dist_field_),    df_bytes);
    cudaMallocHost(reinterpret_cast<void**>(&pin_lr_bits_),       lr_bits_bytes);

    std::memset(pin_brick_indices_, 0xFF, indices_bytes);  // UINT32_MAX: all unloaded
    std::memset(pin_dist_field_,    0x01, df_bytes);       // dist = 1: safe, no skip
    std::memset(pin_lr_bits_,       0x00, lr_bits_bytes);  // all empty

    // CPU-only state (not uploaded to GPU directly)
    cpu_occupancy_.assign(LR_TOTAL, 0u);
    cpu_sc_loaded_.assign(WORLD_SC_X * WORLD_SC_Y * WORLD_SC_Z, false);

    // Slot allocator: one free range covering the full pool
    free_ranges_.push_back({0u, MAX_POOL_BRICKS});

    // Bind GPU resources to the raytracer (single binding, pointers never change)
    raytracer->BindStreamingResources(
        d_pool_data_, d_brick_indices_, d_brick_bounds_, brick_words, STREAM_BRICK_DIM,
        d_dist_field_,
        d_lowres_bits_,
        static_cast<uint16_t>(LR_DIM_X),
        static_cast<uint16_t>(LR_DIM_Y),
        static_cast<uint16_t>(LR_DIM_Z));

    // Worker threads: leave cores for render thread + DF thread
    const uint32_t hw = std::thread::hardware_concurrency();
    const uint32_t nw = std::min(MAX_WORKER_THREADS, hw > 2u ? hw - 2u : 1u);
    workers_.reserve(nw);
    for (uint32_t i = 0; i < nw; ++i) {
        workers_.emplace_back(&ChunkStreamingManager::WorkerThread, this);
    }

    // Dedicated DF rebuild thread (keeps BFS off the render thread entirely)
    df_thread_ = std::thread(&ChunkStreamingManager::DFThread, this);

    std::cout << "[Streaming] Initialized: "
              << nw << " worker threads, "
              << (pool_bytes / (1024 * 1024)) << " MB pool\n";
}

// ============================================================
void ChunkStreamingManager::UpdateCamera(float3 cam_pos, float3 cam_fwd) {
    (void)cam_fwd;

    // Camera super-chunk position
    const int sc_cx = WORLD_SC_X / 2;// static_cast<int>(cam_pos.x / SC_VOXEL_DIM);
    const int sc_cy = WORLD_SC_Y / 2;//static_cast<int>(cam_pos.y / SC_VOXEL_DIM);
    const int sc_cz = WORLD_SC_Z / 2;//static_cast<int>(cam_pos.z / SC_VOXEL_DIM);

    const ChunkKey cam_sc{static_cast<uint16_t>(sc_cx),
                          static_cast<uint16_t>(sc_cy),
                          static_cast<uint16_t>(sc_cz)};

    {
        std::lock_guard<std::mutex> lk(cam_mutex_);
        cam_sc_snapshot_ = cam_sc;
    }

    // Re-score queued (not currently building) requests so nearer/higher-score
    // chunks can move ahead as the camera changes.
    {
        std::lock_guard<std::mutex> lk(build_queue_mutex_);
        if (!build_queue_.empty()) {
            std::vector<BuildRequest> pending;
            pending.reserve(build_queue_.size());
            while (!build_queue_.empty()) {
                auto req = build_queue_.top();
                build_queue_.pop();
                req.priority = ChunkPriority(cam_sc, req.key);
                pending.push_back(req);
            }
            for (const auto& req : pending) {
                build_queue_.push(req);
            }
        }
    }

    // Evict loaded chunks that are now outside render distance.
    // This is the only steady-state eviction policy.
    {
        std::vector<ChunkKey> to_evict;
        to_evict.reserve(loaded_.size());
        for (const auto& kv : loaded_) {
            if (!IsWithinRenderDistance(kv.first, cam_sc)) {
                to_evict.push_back(kv.first);
            }
        }
        for (const auto& key : to_evict) {
            EvictChunk(key);
        }
    }

    // Candidate SCs within render distance, sorted by policy score
    const int sc_radius = STREAM_RENDER_RADIUS_SC;

    std::vector<BuildRequest> candidates;
    for (int dz = -sc_radius; dz <= sc_radius; ++dz) {
        for (int dy = -sc_radius; dy <= sc_radius; ++dy) {
            for (int dx = -sc_radius; dx <= sc_radius; ++dx) {
                const int sx = sc_cx + dx, sy = sc_cy + dy, sz = sc_cz + dz;
                if (sx < 0 || sy < 0 || sz < 0) continue;
                if (static_cast<uint32_t>(sx) >= WORLD_SC_X ||
                    static_cast<uint32_t>(sy) >= WORLD_SC_Y ||
                    static_cast<uint32_t>(sz) >= WORLD_SC_Z) continue;

                const ChunkKey key{ static_cast<uint16_t>(sx),
                                    static_cast<uint16_t>(sy),
                                    static_cast<uint16_t>(sz) };

                if (loaded_.count(key)) continue;
                {
                    std::lock_guard<std::mutex> lk(in_flight_mutex_);
                    if (in_flight_.count(key)) continue;
                }

                if (!IsWithinRenderDistance(key, cam_sc)) continue;

                candidates.push_back({key, ChunkPriority(cam_sc, key)});
            }
        }
    }

    if (!candidates.empty()) {
        std::sort(candidates.begin(), candidates.end(),
                  [](const BuildRequest& a, const BuildRequest& b) {
                      return a.priority > b.priority;
                  });

        // Preempt currently building chunks when they are lower priority than
        // top candidates that are not yet in-flight.
        {
            std::vector<ChunkKey> current_building;
            {
                std::lock_guard<std::mutex> lk(build_state_mutex_);
                current_building.reserve(building_.size());
                for (const auto& key : building_) current_building.push_back(key);
            }

            if (!current_building.empty()) {
                std::vector<float> build_scores;
                build_scores.reserve(current_building.size());
                for (const auto& key : current_building) {
                    build_scores.push_back(ChunkPriority(cam_sc, key));
                }
                std::sort(build_scores.begin(), build_scores.end());

                size_t target_idx = 0;
                for (const auto& cand : candidates) {
                    if (target_idx >= build_scores.size()) break;

                    bool already_in_flight = false;
                    {
                        std::lock_guard<std::mutex> lk(in_flight_mutex_);
                        already_in_flight = in_flight_.count(cand.key) != 0;
                    }
                    if (already_in_flight) continue;

                    const float weakest_building = build_scores[target_idx];
                    if (cand.priority > weakest_building) {
                        // Cancel one currently building chunk that is at or below
                        // the weakest score bucket.
                        ChunkKey to_cancel{};
                        bool found = false;
                        {
                            std::lock_guard<std::mutex> lk(build_state_mutex_);
                            for (const auto& bk : building_) {
                                if (ChunkPriority(cam_sc, bk) <= weakest_building) {
                                    to_cancel = bk;
                                    found = true;
                                    break;
                                }
                            }
                        }
                        if (found) {
                            RequestBuildCancel(to_cancel);
                            ++target_idx;
                        }
                    }
                }
            }
        }

        std::vector<BuildRequest> to_enqueue;
        {
            std::lock_guard<std::mutex> lk(in_flight_mutex_);
            const size_t in_flight_count = in_flight_.size();
            if (in_flight_count < MAX_QUEUED_CHUNKS) {
                const size_t capacity = static_cast<size_t>(MAX_QUEUED_CHUNKS) - in_flight_count;
                const size_t take = std::min(capacity, candidates.size());
                to_enqueue.reserve(take);
                for (size_t i = 0; i < take; ++i) {
                    in_flight_.insert(candidates[i].key);
                    to_enqueue.push_back(candidates[i]);
                }
            }
        }

        if (!to_enqueue.empty()) {
            std::lock_guard<std::mutex> lk(build_queue_mutex_);
            for (const auto& req : to_enqueue) {
                build_queue_.push(req);
            }
            build_queue_cv_.notify_all();
        }
    }
}

// ============================================================
bool ChunkStreamingManager::FlushUploads(VoxelRaytracer3D* /*raytracer*/) {
    // Drain all completed chunk builds from the upload queue.
    // Completed chunks that are now out of render distance are dropped.
    std::vector<ChunkBuildResult> results;
    {
        std::lock_guard<std::mutex> lk(upload_queue_mutex_);
        while (!upload_queue_.empty()) {
            results.push_back(std::move(upload_queue_.front()));
            upload_queue_.pop();
        }
    }

    ChunkKey cam_sc;
    {
        std::lock_guard<std::mutex> lk(cam_mutex_);
        cam_sc = cam_sc_snapshot_;
    }

    for (auto& r : results) {
        if (!IsWithinRenderDistance(r.key, cam_sc)) {
            continue;
        }
        IntegrateResult(r);
    }

    // Upload if chunks changed this frame OR a background DF rebuild finished
    bool df_ready = false;
    {
        std::lock_guard<std::mutex> lk(df_mutex_);
        df_ready = df_result_ready_;
    }

    if (gpu_state_dirty_ || df_ready) {
        UploadGPUState();
        gpu_state_dirty_ = false;
        return true;
    }
    return false;
}

// ============================================================
void ChunkStreamingManager::WaitForInitialChunks(VoxelRaytracer3D* raytracer,
                                                  float3 cam_pos, int min_chunks) {
    std::cout << "[Streaming] Waiting for " << min_chunks << " initial chunks...\n";
    while (static_cast<int>(loaded_.size()) < min_chunks) {
        UpdateCamera(cam_pos, {0.f, 0.f, 1.f});
        FlushUploads(raytracer);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::cout << "[Streaming] " << loaded_.size() << " chunks loaded.\n";
}

// ============================================================
// Internal: worker thread
// ============================================================
void ChunkStreamingManager::WorkerThread() {
    while (!shutdown_.load(std::memory_order_relaxed)) {
        ChunkKey key;
        {
            std::unique_lock<std::mutex> lk(build_queue_mutex_);
            build_queue_cv_.wait(lk, [this] {
                return shutdown_.load(std::memory_order_relaxed)
                    || !build_queue_.empty();
            });
            if (shutdown_.load(std::memory_order_relaxed)) break;
            if (build_queue_.empty()) continue;
            key = build_queue_.top().key;
            build_queue_.pop();
        }

        {
            std::lock_guard<std::mutex> lk(build_state_mutex_);
            building_.insert(key);
            cancel_requested_.erase(key);
        }

        ChunkBuildResult result = BuildChunk(key);

        {
            std::lock_guard<std::mutex> lk(build_state_mutex_);
            building_.erase(key);
            cancel_requested_.erase(key);
        }

        {
            std::lock_guard<std::mutex> lk(upload_queue_mutex_);
            upload_queue_.push(std::move(result));
        }
        {
            std::lock_guard<std::mutex> lk(in_flight_mutex_);
            in_flight_.erase(key);
        }
    }
}

// ============================================================
// Internal: build one super-chunk on a worker thread
// Terrain is identical to the PopulateVoxels GPU kernel:
//   height(x,z) = (repeaterPerlin(x*0.001,0,z*0.001,...)*0.5+0.5)*512
//   solid iff world_y <= height
// Optimisation: compute height once per (x,z) column to avoid
// 32 redundant noise evaluations per Y-voxel.
// ============================================================
ChunkBuildResult ChunkStreamingManager::BuildChunk(const ChunkKey& key) {
    ChunkBuildResult result;
    result.key = key;
    result.local_brick_seq.assign(BRICKS_PER_SC, UINT32_MAX);

    if (IsBuildCancelled(key)) {
        result.canceled = true;
        return result;
    }

    // World-space voxel origin of this super-chunk
    const uint32_t wx0 = key.x * SC_VOXEL_DIM;
    const uint32_t wy0 = key.y * SC_VOXEL_DIM;
    const uint32_t wz0 = key.z * SC_VOXEL_DIM;

    // Precompute 2D height map for the SC's XZ footprint (256x256 values)
    // stored as heights[sc_local_z * SC_VOXEL_DIM + sc_local_x]
    static thread_local std::vector<float> heights;
    const uint32_t hmap_size = SC_VOXEL_DIM * SC_VOXEL_DIM;
    heights.resize(hmap_size);

    for (uint32_t lz = 0; lz < SC_VOXEL_DIM; ++lz) {
        if (IsBuildCancelled(key)) {
            result.canceled = true;
            return result;
        }
        for (uint32_t lx = 0; lx < SC_VOXEL_DIM; ++lx) {
            const float wx = (wx0 + lx) * 0.001f;
            const float wz = (wz0 + lz) * 0.001f;
            // Match PopulateVoxels exactly: noise at (fx, 0, fz)
            const float n = CPUNoise::repeaterPerlin(wx, 0.f, wz, 1.f,
                                                     0x71889283, 32, 2.f, 0.5f);
            heights[lz * SC_VOXEL_DIM + lx] = std::max((n * 0.5f + 0.5f) * 512.f, 0.f);
        }
    }

    // Build each brick in the super-chunk
    uint32_t seq_idx = 0;
    static thread_local std::vector<uint32_t> brick_buf;
    brick_buf.resize(brick_words_);

    for (uint32_t lbz = 0; lbz < SC_BRICK_DIM; ++lbz) {
        if (IsBuildCancelled(key)) {
            result.canceled = true;
            return result;
        }
        for (uint32_t lby = 0; lby < SC_BRICK_DIM; ++lby) {
            for (uint32_t lbx = 0; lbx < SC_BRICK_DIM; ++lbx) {
                const uint32_t local_brick_idx =
                    lbx + lby * SC_BRICK_DIM + lbz * SC_BRICK_DIM * SC_BRICK_DIM;

                // World-space voxel origin of this brick
                const uint32_t bwx = wx0 + lbx * STREAM_BRICK_DIM;
                const uint32_t bwy = wy0 + lby * STREAM_BRICK_DIM;
                const uint32_t bwz = wz0 + lbz * STREAM_BRICK_DIM;

                std::fill(brick_buf.begin(), brick_buf.end(), 0u);
                bool any = false;
                BrickBounds bounds{STREAM_BRICK_DIM, STREAM_BRICK_DIM, STREAM_BRICK_DIM, 0u, 0u, 0u};

                for (uint32_t dvz = 0; dvz < STREAM_BRICK_DIM; ++dvz) {
                    for (uint32_t dvy = 0; dvy < STREAM_BRICK_DIM; ++dvy) {
                        // Height row for this z column
                        const uint32_t lz = lbz * STREAM_BRICK_DIM + dvz;
                        const float* hrow = heights.data() + lz * SC_VOXEL_DIM
                                            + lbx * STREAM_BRICK_DIM;
                        const float world_y = static_cast<float>(bwy + dvy);

                        for (uint32_t dvx = 0; dvx < STREAM_BRICK_DIM; ++dvx) {
                            if (world_y <= hrow[dvx]) {
                                const uint32_t lo = GetSampleIndex(
                                    dvx, dvy, dvz,
                                    STREAM_BRICK_DIM, STREAM_BRICK_DIM);
                                brick_buf[lo >> 5] |= 1u << (lo & 31);
                                any = true;
                                bounds.min_x = std::min(bounds.min_x, static_cast<uint8_t>(dvx));
                                bounds.min_y = std::min(bounds.min_y, static_cast<uint8_t>(dvy));
                                bounds.min_z = std::min(bounds.min_z, static_cast<uint8_t>(dvz));
                                bounds.max_x = std::max(bounds.max_x, static_cast<uint8_t>(dvx));
                                bounds.max_y = std::max(bounds.max_y, static_cast<uint8_t>(dvy));
                                bounds.max_z = std::max(bounds.max_z, static_cast<uint8_t>(dvz));
                            }
                        }
                    }
                }

                if (any) {
                    result.local_brick_seq[local_brick_idx] = seq_idx++;
                    result.brick_data.insert(
                        result.brick_data.end(),
                        brick_buf.begin(), brick_buf.end());
                    result.brick_bounds.push_back(bounds);
                }
            }
        }
    }

    result.occupied_count = seq_idx;
    return result;
}

bool ChunkStreamingManager::IsBuildCancelled(const ChunkKey& key) {
    std::lock_guard<std::mutex> lk(build_state_mutex_);
    return cancel_requested_.count(key) != 0;
}

void ChunkStreamingManager::RequestBuildCancel(const ChunkKey& key) {
    std::lock_guard<std::mutex> lk(build_state_mutex_);
    if (building_.count(key)) {
        cancel_requested_.insert(key);
    }
}

// ============================================================
// Internal: integrate a completed SC into CPU shadow state
// ============================================================
void ChunkStreamingManager::IntegrateResult(const ChunkBuildResult& result) {
    if (result.canceled) return;

    // Discard if already loaded (duplicate delivery from in-flight requeue)
    if (loaded_.count(result.key)) return;

    const uint32_t occupied = result.occupied_count;
    uint32_t base_slot = 0;

    if (occupied > 0) {
        base_slot = AllocContig(occupied);
        if (base_slot == UINT32_MAX) {
            // Should be unreachable with MAX_POOL_BRICKS sized to world upper bound.
            // Keep this as a fail-safe to avoid writing invalid GPU addresses.
            return;
        }

        // Upload all brick data in one contiguous cudaMemcpy
        cudaMemcpy(
            d_pool_data_ + static_cast<size_t>(base_slot) * brick_words_,
            result.brick_data.data(),
            result.brick_data.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice);
        cudaMemcpy(
            d_brick_bounds_ + base_slot,
            result.brick_bounds.data(),
            result.brick_bounds.size() * sizeof(BrickBounds),
            cudaMemcpyHostToDevice);
    }

    // Update pinned indices, packed lr_bits, and occupancy for each brick in this SC
    const uint32_t bx0 = result.key.x * SC_BRICK_DIM;
    const uint32_t by0 = result.key.y * SC_BRICK_DIM;
    const uint32_t bz0 = result.key.z * SC_BRICK_DIM;

    {
        std::lock_guard<std::mutex> occ_lk(occ_mutex_);
        for (uint32_t lbz = 0; lbz < SC_BRICK_DIM; ++lbz) {
            for (uint32_t lby = 0; lby < SC_BRICK_DIM; ++lby) {
                for (uint32_t lbx = 0; lbx < SC_BRICK_DIM; ++lbx) {
                    const uint32_t li =
                        lbx + lby * SC_BRICK_DIM + lbz * SC_BRICK_DIM * SC_BRICK_DIM;
                    const uint32_t lr_idx = GetSampleIndex(
                        bx0 + lbx, by0 + lby, bz0 + lbz, LR_DIM_X, LR_DIM_Y);

                    const uint32_t seq = result.local_brick_seq[li];
                    if (seq == UINT32_MAX) {
                        pin_brick_indices_[lr_idx] = UINT32_MAX;
                        cpu_occupancy_[lr_idx]     = 0u;
                        pin_lr_bits_[lr_idx >> 5] &= ~(1u << (lr_idx & 31u));
                    } else {
                        pin_brick_indices_[lr_idx] = base_slot + seq;
                        cpu_occupancy_[lr_idx]     = 1u;
                        pin_lr_bits_[lr_idx >> 5] |=  (1u << (lr_idx & 31u));
                    }
                }
            }
        }

        const uint32_t sc_idx =
            result.key.x + result.key.y * WORLD_SC_X + result.key.z * WORLD_SC_X * WORLD_SC_Y;
        cpu_sc_loaded_[sc_idx] = true;
    }

    loaded_[result.key] = {base_slot, occupied};
    gpu_state_dirty_     = true;

    // Signal the background DF thread to rebuild with fresh occupancy data
    {
        std::lock_guard<std::mutex> lk(df_mutex_);
        df_rebuild_requested_ = true;
    }
    df_cv_.notify_one();
}

// ============================================================
// Internal: evict a super-chunk from CPU state
// ============================================================
void ChunkStreamingManager::EvictChunk(const ChunkKey& key) {
    auto it = loaded_.find(key);
    if (it == loaded_.end()) return;

    const LoadedSC& lc = it->second;
    if (lc.brick_count > 0) FreeContig(lc.base_slot, lc.brick_count);

    const uint32_t bx0 = key.x * SC_BRICK_DIM;
    const uint32_t by0 = key.y * SC_BRICK_DIM;
    const uint32_t bz0 = key.z * SC_BRICK_DIM;

    {
        std::lock_guard<std::mutex> occ_lk(occ_mutex_);
        for (uint32_t lbz = 0; lbz < SC_BRICK_DIM; ++lbz) {
            for (uint32_t lby = 0; lby < SC_BRICK_DIM; ++lby) {
                for (uint32_t lbx = 0; lbx < SC_BRICK_DIM; ++lbx) {
                    const uint32_t lr_idx = GetSampleIndex(
                        bx0 + lbx, by0 + lby, bz0 + lbz, LR_DIM_X, LR_DIM_Y);
                    pin_brick_indices_[lr_idx]    = UINT32_MAX;
                    cpu_occupancy_[lr_idx]        = 0u;
                    pin_lr_bits_[lr_idx >> 5] &= ~(1u << (lr_idx & 31u));
                }
            }
        }

        const uint32_t sc_idx =
            key.x + key.y * WORLD_SC_X + key.z * WORLD_SC_X * WORLD_SC_Y;
        cpu_sc_loaded_[sc_idx] = false;
    }

    loaded_.erase(it);
    gpu_state_dirty_ = true;

    // Signal the background DF thread
    {
        std::lock_guard<std::mutex> lk(df_mutex_);
        df_rebuild_requested_ = true;
    }
    df_cv_.notify_one();
}

// ============================================================
// Internal: push GPU state using async DMA transfers.
// Applies a fresh DF result from the background thread if one is ready.
// cudaMemcpyAsync on the null stream returns to the CPU immediately;
// the GPU serialises these transfers with the subsequent render kernel
// automatically, so no explicit stream sync is required.
// ============================================================
void ChunkStreamingManager::UploadGPUState() {
    // If the DF thread has finished a rebuild, copy it to the pinned buffer
    {
        std::lock_guard<std::mutex> lk(df_mutex_);
        if (df_result_ready_) {
            std::memcpy(pin_dist_field_, df_result_.data(), LR_TOTAL * sizeof(uint8_t));
            df_result_ready_ = false;
        }
    }

    cudaMemcpyAsync(d_brick_indices_, pin_brick_indices_,
                    LR_TOTAL  * sizeof(uint32_t), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_dist_field_,    pin_dist_field_,
                    LR_TOTAL  * sizeof(uint8_t),  cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_lowres_bits_,   pin_lr_bits_,
                    lr_words_ * sizeof(uint32_t), cudaMemcpyHostToDevice, 0);
}

// ============================================================
// Internal: dedicated distance-field rebuild thread.
// Sleeps until notified, snapshots occupancy under occ_mutex_ (brief),
// then runs the BFS without holding any lock.
// ============================================================
void ChunkStreamingManager::DFThread() {
    std::vector<uint8_t> occ_snap;
    std::vector<bool>    loaded_snap;

    while (true) {
        {
            std::unique_lock<std::mutex> lk(df_mutex_);
            df_cv_.wait(lk, [this] {
                return df_shutdown_.load(std::memory_order_relaxed)
                    || df_rebuild_requested_;
            });
            if (df_shutdown_.load(std::memory_order_relaxed)) break;
            df_rebuild_requested_ = false;
        }

        // Take a snapshot of occupancy while holding occ_mutex_ briefly
        {
            std::lock_guard<std::mutex> occ_lk(occ_mutex_);
            occ_snap    = cpu_occupancy_;   // 262 KB copy
            loaded_snap = cpu_sc_loaded_;   // 512-entry copy
        }

        // BFS with no locks held
        std::vector<uint8_t> new_df;
        BuildDFFromSnapshot(occ_snap, loaded_snap, new_df);

        // Publish result
        {
            std::lock_guard<std::mutex> lk(df_mutex_);
            df_result_       = std::move(new_df);
            df_result_ready_ = true;
        }
    }
}

// ============================================================
// Internal: BFS Chebyshev distance field computed from a snapshot.
// Cells in unloaded SCs are capped at 1 so the DDA cannot skip
// through regions we haven't streamed yet.
// ============================================================
void ChunkStreamingManager::BuildDFFromSnapshot(
    const std::vector<uint8_t>& occ,
    const std::vector<bool>&    loaded,
    std::vector<uint8_t>&       out)
{
    out.assign(LR_TOTAL, 255u);
    for (uint32_t i = 0; i < LR_TOTAL; ++i) {
        if (occ[i]) out[i] = 0u;
    }

    auto getDF = [&](int x, int y, int z) -> uint8_t {
        if (x < 0 || y < 0 || z < 0 ||
            static_cast<uint32_t>(x) >= LR_DIM_X ||
            static_cast<uint32_t>(y) >= LR_DIM_Y ||
            static_cast<uint32_t>(z) >= LR_DIM_Z) return 255u;
        return out[GetSampleIndex(x, y, z, LR_DIM_X, LR_DIM_Y)];
    };

    for (int pass = 0; pass < 3; ++pass) {
        // Forward sweep
        for (int z = 0; z < static_cast<int>(LR_DIM_Z); ++z)
        for (int y = 0; y < static_cast<int>(LR_DIM_Y); ++y)
        for (int x = 0; x < static_cast<int>(LR_DIM_X); ++x) {
            const uint32_t idx = GetSampleIndex(x, y, z, LR_DIM_X, LR_DIM_Y);
            if (out[idx] == 0u) continue;
            uint8_t best = 255u;
            for (int dz=-1; dz<=1; ++dz)
            for (int dy=-1; dy<=1; ++dy)
            for (int dx=-1; dx<=1; ++dx) {
                if (!dx && !dy && !dz) continue;
                const uint8_t n = getDF(x+dx, y+dy, z+dz);
                if (n < 255u) best = std::min(best, static_cast<uint8_t>(n + 1u));
            }
            if (best < out[idx]) out[idx] = best;
        }
        // Backward sweep
        for (int z = static_cast<int>(LR_DIM_Z)-1; z >= 0; --z)
        for (int y = static_cast<int>(LR_DIM_Y)-1; y >= 0; --y)
        for (int x = static_cast<int>(LR_DIM_X)-1; x >= 0; --x) {
            const uint32_t idx = GetSampleIndex(x, y, z, LR_DIM_X, LR_DIM_Y);
            if (out[idx] == 0u) continue;
            uint8_t best = 255u;
            for (int dz=-1; dz<=1; ++dz)
            for (int dy=-1; dy<=1; ++dy)
            for (int dx=-1; dx<=1; ++dx) {
                if (!dx && !dy && !dz) continue;
                const uint8_t n = getDF(x+dx, y+dy, z+dz);
                if (n < 255u) best = std::min(best, static_cast<uint8_t>(n + 1u));
            }
            if (best < out[idx]) out[idx] = best;
        }
    }

    // Cap distance to 1 for bricks in unloaded SCs so the DDA cannot
    // accidentally skip through unknown terrain.
    for (int z = 0; z < static_cast<int>(LR_DIM_Z); ++z) {
        const uint32_t sc_z = static_cast<uint32_t>(z) / SC_BRICK_DIM;
        for (int y = 0; y < static_cast<int>(LR_DIM_Y); ++y) {
            const uint32_t sc_y = static_cast<uint32_t>(y) / SC_BRICK_DIM;
            for (int x = 0; x < static_cast<int>(LR_DIM_X); ++x) {
                const uint32_t sc_x   = static_cast<uint32_t>(x) / SC_BRICK_DIM;
                const uint32_t sc_idx = sc_x + sc_y * WORLD_SC_X
                                      + sc_z * WORLD_SC_X * WORLD_SC_Y;
                if (!loaded[sc_idx]) {
                    const uint32_t idx = GetSampleIndex(x, y, z, LR_DIM_X, LR_DIM_Y);
                    if (out[idx] > 1u) out[idx] = 1u;
                }
            }
        }
    }
}

// ============================================================
// Internal: chunk priority
// Higher priority = more urgent to build.
// Forward-facing chunks score higher; closer chunks score higher.
// ============================================================
float ChunkStreamingManager::ChunkPriority(const ChunkKey& camera_sc, const ChunkKey& key) const {
    if (!scoring_policy_) return 0.0f;
    return scoring_policy_->Score(camera_sc, key);
}

bool ChunkStreamingManager::IsWithinRenderDistance(const ChunkKey& key, const ChunkKey& cam_sc) const {
    const int dx = std::abs(static_cast<int>(key.x) - static_cast<int>(cam_sc.x));
    const int dz = std::abs(static_cast<int>(key.z) - static_cast<int>(cam_sc.z));
    // Cylindrical inclusion in chunk space: radius is applied in XZ only.
    const int dist2 = dx * dx + dz * dz;
    const int radius2 = STREAM_RENDER_RADIUS_SC * STREAM_RENDER_RADIUS_SC;
    return dist2 <= radius2;
}

// ============================================================
// Internal: contiguous slot allocator (first-fit, sorted free list)
// ============================================================
uint32_t ChunkStreamingManager::AllocContig(uint32_t count) {
    for (auto it = free_ranges_.begin(); it != free_ranges_.end(); ++it) {
        if (it->length >= count) {
            const uint32_t base = it->start;
            it->start  += count;
            it->length -= count;
            if (it->length == 0u) free_ranges_.erase(it);
            return base;
        }
    }
    return UINT32_MAX; // pool exhausted
}

void ChunkStreamingManager::FreeContig(uint32_t base, uint32_t count) {
    FreeRange nr{base, count};
    // Insert sorted by start address
    auto it = free_ranges_.begin();
    while (it != free_ranges_.end() && it->start < base) ++it;
    auto ins = free_ranges_.insert(it, nr);
    // Merge with successor
    auto nxt = std::next(ins);
    if (nxt != free_ranges_.end() && ins->start + ins->length == nxt->start) {
        ins->length += nxt->length;
        free_ranges_.erase(nxt);
    }
    // Merge with predecessor
    if (ins != free_ranges_.begin()) {
        auto prv = std::prev(ins);
        if (prv->start + prv->length == ins->start) {
            prv->length += ins->length;
            free_ranges_.erase(ins);
        }
    }
}

} // namespace GPUDDA
