// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * src/ignite_xdna/c_api/ignite.cpp
 *
 * Core C++ Implementation of libignite_xdna native inference engine.
 * Memory-maps .ignite binary containers, directly programs Native XRT hardware buffers,
 * executes C-SIMD letterbox/bilinear ingress preprocessing, and decodes YOLOv8 bounding
 * boxes via pure C++20 DFL softmax projection and batched NMS.
 *
 * Implements asynchronous ping-pong double-buffering (bo_in[2], bo_out[2]) to completely
 * overlap CPU ingress SIMD preprocessing and DMA transfer with physical AIE2 silicon execution.
 */

#ifndef IGNITE_EXPORTS
#define IGNITE_EXPORTS
#endif
#include "ignite.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <stdexcept>

#if defined(__AVX2__) || (defined(_M_X64) && defined(__AVX2__)) || defined(__x86_64__)
#include <immintrin.h>
#define IGNITE_USE_AVX2 1
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define IGNITE_USE_NEON 1
#endif

#include <nlohmann/json.hpp>

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_xclbin.h>

namespace fs = std::filesystem;

// Thread-local error string
static thread_local std::string g_last_error;

static void set_error(const std::string& err) {
    g_last_error = err;
}

// External SIMD Preprocessor declaration
#if defined(_WIN32) || defined(__CYGWIN__)
#define PREPROCESS_API __declspec(dllexport)
#else
#define PREPROCESS_API __attribute__((visibility("default")))
#endif

extern "C" PREPROCESS_API int fused_preprocess_bgr_to_chw_int8(
    const uint8_t* __restrict src_bgr,
    int src_w,
    int src_h,
    int src_stride,
    int8_t* __restrict dst_chw,
    int dst_w,
    int dst_h,
    int* __restrict out_pad_top,
    int* __restrict out_pad_left,
    float* __restrict out_scale
);

// Standard 80 COCO Class Names
static const char* COCO_CLASSES[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

#pragma pack(push, 1)
struct IgniteHeaderRaw {
    char magic[4];          // "IGNT"
    uint16_t version;       // 1
    uint16_t arch_id;       // 1
    uint32_t header_size;   // 64
    uint32_t crc32;
    uint64_t total_file_size;
    uint64_t manifest_offset;
    uint64_t manifest_size;
    uint64_t blob_offset;
    uint64_t blob_size;
    uint32_t num_blobs;
    char reserved[4];
};
#pragma pack(pop)

struct BlobEntry {
    std::string name;
    size_t offset = 0;
    size_t size = 0;
};

// IEEE 802.3 CRC-32 (zlib.crc32 equivalent) over the container body, so a
// truncated or bit-flipped container is refused at load instead of being
// programmed into the device.
static uint32_t crc32_ieee(const uint8_t* data, size_t length) {
    static uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        ready = true;
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < length; ++i) {
        crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

static void require_completed(ert_cmd_state state, const char* what) {
    if (state != ERT_CMD_STATE_COMPLETED) {
        throw std::runtime_error(std::string(what) + " did not complete (ert_cmd_state "
                                 + std::to_string(static_cast<int>(state)) + ")");
    }
}

struct StageResource {
    std::string name;
    int stage_idx = 0;
    uint32_t ninstr_init = 0;
    uint32_t ninstr_exec = 0;
    xrt::bo bo_init;
    xrt::bo bo_exec;
};

struct Candidate {
    float x0, y0, w, h;
    float score;
    int class_id;
};

struct ignite_engine {
    // Multi-slot circular ring constant
    static constexpr int NUM_SLOTS = 4;
    static constexpr size_t RING_SIZE = 128;

    // Win32 memory-mapping handles
    HANDLE h_file = INVALID_HANDLE_VALUE;
    HANDLE h_map = NULL;
    const uint8_t* mmap_base = nullptr;
    size_t mmap_size = 0;

    // Metadata
    std::string model_name;
    int num_cores = 16;
    size_t in_bytes = 8192;
    size_t out_bytes = 4096;
    bool fused_dfl = false;

    // Native XRT Hardware Context
    std::unique_ptr<xrt::device> device;
    std::unique_ptr<xrt::hw_context> hw_ctx;
    std::unique_ptr<xrt::kernel> kernel;

    // Double-buffered BOs (slot 0..3)
    xrt::bo bo_in[NUM_SLOTS];
    xrt::bo bo_out[NUM_SLOTS];
    std::vector<int8_t> chw_buffer[NUM_SLOTS];
    int slot_pad_top[NUM_SLOTS] = {0, 0, 0, 0};
    int slot_pad_left[NUM_SLOTS] = {0, 0, 0, 0};
    float slot_scale[NUM_SLOTS] = {1.0f, 1.0f, 1.0f, 1.0f};
    std::chrono::high_resolution_clock::time_point slot_t_start[NUM_SLOTS];
    std::chrono::high_resolution_clock::time_point slot_t_prep[NUM_SLOTS];

    // Stage pipeline (CDO transaction blobs)
    std::vector<StageResource> stages;
    bool single_dispatch = false;
    uint32_t ninstr_monolithic = 0;
    xrt::bo bo_monolithic;

    // Detection hyperparameters: written by the API thread, read by the
    // post-processing thread, so they are atomics rather than plain floats.
    std::atomic<float> conf_thres{0.25f};
    std::atomic<float> iou_thres{0.50f};

    // Pre-allocated / cached reference heads for visual detection decode
    bool has_reference_heads = false;
    std::vector<float> ref_p3_box;
    std::vector<float> ref_p4_box;
    std::vector<float> ref_p5_box;
    std::vector<float> ref_p3_cls;
    std::vector<float> ref_p4_cls;
    std::vector<float> ref_p5_cls;

    // Reusable scratch vectors for DFL decode
    std::vector<float> scratch_max_logits;
    std::vector<int> scratch_best_cls;
    std::vector<Candidate> scratch_candidates;
    std::vector<bool> scratch_suppressed;

    // Fixed-size circular ring buffer for results
    struct ResultItem {
        uint64_t ticket = 0;
        std::vector<Candidate> detections;
        ignite_timings_t timings = { 0.0, 0.0, 0.0, 0.0 };
    };
    ResultItem result_ring[RING_SIZE];

    // Fine-grained latency records
    ignite_timings_t last_timings = { 0.0, 0.0, 0.0, 0.0 };

    // Threading and async queue coordination
    struct NpuWorkItem {
        uint64_t ticket;
        int slot;
    };

    struct PostWorkItem {
        uint64_t ticket;
        int slot;
        std::chrono::high_resolution_clock::time_point t_npu_start;
        std::chrono::high_resolution_clock::time_point t_npu_done;
    };

    std::atomic<uint64_t> next_ticket{1};
    std::atomic<uint64_t> last_completed_ticket{0};   // monotonic high-water mark
    std::atomic<uint64_t> dispatch_timeouts{0};       // frames whose NPU run did not complete
    std::atomic<bool> worker_stop{false};

    std::mutex npu_queue_mutex;
    std::condition_variable npu_queue_cv;
    std::queue<NpuWorkItem> npu_work_queue;

    std::mutex post_queue_mutex;
    std::condition_variable post_queue_cv;
    std::queue<PostWorkItem> post_work_queue;

    std::mutex result_mutex;
    std::condition_variable result_cv;
    std::condition_variable slot_cv;

    std::thread npu_thread;
    std::thread post_thread;

    void start_worker() {
        worker_stop.store(false, std::memory_order_relaxed);
        npu_thread = std::thread(&ignite_engine::npu_worker_loop, this);
        post_thread = std::thread(&ignite_engine::post_worker_loop, this);
    }

    void stop_worker() {
        // Set the flag while holding every mutex a waiter evaluates it under.
        // A thread that has checked its predicate but not yet blocked would
        // otherwise miss the notification and ignite_free would hang.
        {
            std::lock_guard<std::mutex> npu_lock(npu_queue_mutex);
            std::lock_guard<std::mutex> post_lock(post_queue_mutex);
            std::lock_guard<std::mutex> result_lock(result_mutex);
            worker_stop.store(true, std::memory_order_release);
        }
        npu_queue_cv.notify_all();
        post_queue_cv.notify_all();
        result_cv.notify_all();
        slot_cv.notify_all();
        if (npu_thread.joinable()) {
            npu_thread.join();
        }
        if (post_thread.joinable()) {
            post_thread.join();
        }
    }

    void decode_detections_for_slot(int slot, std::vector<Candidate>& out_dets);
    void npu_worker_loop();
    void post_worker_loop();

    ~ignite_engine() {
        stop_worker();

        stages.clear();
        bo_monolithic = xrt::bo();
        for (int s = 0; s < NUM_SLOTS; ++s) {
            bo_in[s] = xrt::bo();
            bo_out[s] = xrt::bo();
        }
        kernel.reset();
        hw_ctx.reset();
        device.reset();

        if (mmap_base) {
            UnmapViewOfFile(mmap_base);
            mmap_base = nullptr;
        }
        if (h_map) {
            CloseHandle(h_map);
            h_map = NULL;
        }
        if (h_file != INVALID_HANDLE_VALUE) {
            CloseHandle(h_file);
            h_file = INVALID_HANDLE_VALUE;
        }
    }
};

static float compute_iou(const Candidate& a, const Candidate& b) {
    float x1 = std::max(a.x0, b.x0);
    float y1 = std::max(a.y0, b.y0);
    float x2 = std::min(a.x0 + a.w, b.x0 + b.w);
    float y2 = std::min(a.y0 + a.h, b.y0 + b.h);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float union_area = (a.w * a.h) + (b.w * b.h) - inter;
    return union_area > 0.0f ? (inter / union_area) : 0.0f;
}

static fs::path resolve_xclbin(const fs::path& model_path) {
    const char* env_p = getenv("IGNITE_XCLBIN_PATH");
    if (env_p && fs::exists(env_p)) {
        return fs::path(env_p);
    }
    // Check model directory
    fs::path cand1 = model_path.parent_path() / "im2col_4d_16core.xclbin";
    if (fs::exists(cand1)) return cand1;

    // Check build directory
    fs::path cand2 = fs::path("build") / "im2col_4d_16core.xclbin";
    if (fs::exists(cand2)) return cand2;

    // Check cwd
    fs::path cand3 = fs::path("im2col_4d_16core.xclbin");
    if (fs::exists(cand3)) return cand3;

    return cand2;
}

static void try_load_default_heads(ignite_engine* eng, const fs::path& model_path) {
    const char* env_p = getenv("IGNITE_HEADS_BIN");
    std::vector<fs::path> candidates;
    if (env_p) candidates.push_back(fs::path(env_p));
    candidates.push_back(model_path.parent_path() / "bus_heads.bin");
    candidates.push_back(fs::path("build") / "bus_heads.bin");
    candidates.push_back(fs::path("bus_heads.bin"));

    for (const auto& p : candidates) {
        if (fs::exists(p)) {
            ignite_load_reference_heads(eng, p.string().c_str());
            break;
        }
    }
}

static void run_vectorized_bitmask_nms(
    std::vector<Candidate>& candidates,
    float iou_threshold,
    std::vector<Candidate>& out_dets
) {
    out_dets.clear();
    const size_t K = candidates.size();
    if (K == 0) return;

    if (K == 1) {
        out_dets.push_back(candidates[0]);
        return;
    }

    // Sort descending by confidence score
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.score > b.score;
    });

    // Limit to top 256 candidates for optimal NMS cache efficiency
    const size_t num_cands = std::min(K, size_t(256));

    // Structure of Arrays (SoA) layout aligned to 32 bytes for AVX2 load
    alignas(32) float cand_x1[256];
    alignas(32) float cand_y1[256];
    alignas(32) float cand_x2[256];
    alignas(32) float cand_y2[256];
    alignas(32) float cand_area[256];
    alignas(32) int32_t cand_cls[256];

    for (size_t i = 0; i < num_cands; ++i) {
        const auto& c = candidates[i];
        cand_x1[i] = c.x0;
        cand_y1[i] = c.y0;
        cand_x2[i] = c.x0 + c.w;
        cand_y2[i] = c.y0 + c.h;
        cand_area[i] = std::max(0.0f, c.w * c.h);
        cand_cls[i] = c.class_id;
    }
    // Zero-pad remainder up to 256
    for (size_t i = num_cands; i < 256; ++i) {
        cand_x1[i] = 0.0f;
        cand_y1[i] = 0.0f;
        cand_x2[i] = 0.0f;
        cand_y2[i] = 0.0f;
        cand_area[i] = 0.0f;
        cand_cls[i] = -1;
    }

    // 64-bit integer bitmask array (4 x 64-bit = 256 bits, 32 bytes on stack, 0 heap allocations)
    uint64_t suppressed[4] = {0, 0, 0, 0};

#if defined(IGNITE_USE_AVX2)
    const __m256 v_iou_thresh = _mm256_set1_ps(iou_threshold);
    const __m256 v_zero = _mm256_setzero_ps();

    for (size_t i = 0; i < num_cands; ++i) {
        const size_t word_i = i >> 6;
        const uint64_t bit_i = 1ULL << (i & 63);
        if ((suppressed[word_i] & bit_i) != 0) {
            continue;
        }

        out_dets.push_back(candidates[i]);

        // Broadcast anchor box i to 256-bit SIMD registers
        const __m256 ax1 = _mm256_set1_ps(cand_x1[i]);
        const __m256 ay1 = _mm256_set1_ps(cand_y1[i]);
        const __m256 ax2 = _mm256_set1_ps(cand_x2[i]);
        const __m256 ay2 = _mm256_set1_ps(cand_y2[i]);
        const __m256 a_area = _mm256_set1_ps(cand_area[i]);
        const __m256i a_cls = _mm256_set1_epi32(cand_cls[i]);

        // Process candidate boxes starting from i+1
        for (size_t j = i + 1; j < num_cands; ) {
            // Handle scalar prefix if not 8-aligned
            if ((j & 7) != 0) {
                const size_t word_j = j >> 6;
                const uint64_t bit_j = 1ULL << (j & 63);
                if ((suppressed[word_j] & bit_j) == 0 && cand_cls[i] == cand_cls[j]) {
                    float ix1 = std::max(cand_x1[i], cand_x1[j]);
                    float iy1 = std::max(cand_y1[i], cand_y1[j]);
                    float ix2 = std::min(cand_x2[i], cand_x2[j]);
                    float iy2 = std::min(cand_y2[i], cand_y2[j]);
                    float iw = std::max(0.0f, ix2 - ix1);
                    float ih = std::max(0.0f, iy2 - iy1);
                    float inter = iw * ih;
                    float uni = cand_area[i] + cand_area[j] - inter;
                    if (uni > 0.0f && (inter / uni) > iou_threshold) {
                        suppressed[word_j] |= bit_j;
                    }
                }
                ++j;
                continue;
            }

            // j is 8-aligned!
            const size_t word_j = j >> 6;
            const size_t shift_j = j & 63;

            // Fast block skip: if all 8 candidates are already suppressed, skip entire block in 1 cycle
            if (((suppressed[word_j] >> shift_j) & 0xFF) == 0xFF) {
                j += 8;
                continue;
            }

            const __m256 bx1 = _mm256_load_ps(&cand_x1[j]);
            const __m256 by1 = _mm256_load_ps(&cand_y1[j]);
            const __m256 bx2 = _mm256_load_ps(&cand_x2[j]);
            const __m256 by2 = _mm256_load_ps(&cand_y2[j]);
            const __m256 b_area = _mm256_load_ps(&cand_area[j]);
            const __m256i b_cls = _mm256_load_si256(reinterpret_cast<const __m256i*>(&cand_cls[j]));

            const __m256i cls_eq = _mm256_cmpeq_epi32(a_cls, b_cls);

            const __m256 ix1 = _mm256_max_ps(ax1, bx1);
            const __m256 iy1 = _mm256_max_ps(ay1, by1);
            const __m256 ix2 = _mm256_min_ps(ax2, bx2);
            const __m256 iy2 = _mm256_min_ps(ay2, by2);

            const __m256 iw = _mm256_max_ps(v_zero, _mm256_sub_ps(ix2, ix1));
            const __m256 ih = _mm256_max_ps(v_zero, _mm256_sub_ps(iy2, iy1));
            const __m256 inter_area = _mm256_mul_ps(iw, ih);

            const __m256 union_area = _mm256_sub_ps(_mm256_add_ps(a_area, b_area), inter_area);
            const __m256 iou = _mm256_div_ps(inter_area, union_area);

            const __m256 iou_gt = _mm256_cmp_ps(iou, v_iou_thresh, _CMP_GT_OQ);
            const __m256 match = _mm256_and_ps(iou_gt, _mm256_castsi256_ps(cls_eq));

            int sup_bits = _mm256_movemask_ps(match);
            if (j + 8 > num_cands) {
                int valid_lanes = static_cast<int>(num_cands - j);
                sup_bits &= ((1 << valid_lanes) - 1);
            }

            if (sup_bits) {
                suppressed[word_j] |= (static_cast<uint64_t>(sup_bits) << shift_j);
            }
            j += 8;
        }
    }
#elif defined(IGNITE_USE_NEON)
    const float32x4_t v_iou_thresh = vdupq_n_f32(iou_threshold);
    const float32x4_t v_zero = vdupq_n_f32(0.0f);

    for (size_t i = 0; i < num_cands; ++i) {
        const size_t word_i = i >> 6;
        const uint64_t bit_i = 1ULL << (i & 63);
        if ((suppressed[word_i] & bit_i) != 0) continue;

        out_dets.push_back(candidates[i]);

        const float32x4_t ax1 = vdupq_n_f32(cand_x1[i]);
        const float32x4_t ay1 = vdupq_n_f32(cand_y1[i]);
        const float32x4_t ax2 = vdupq_n_f32(cand_x2[i]);
        const float32x4_t ay2 = vdupq_n_f32(cand_y2[i]);
        const float32x4_t a_area = vdupq_n_f32(cand_area[i]);
        const int32x4_t a_cls = vdupq_n_s32(cand_cls[i]);

        for (size_t j = i + 1; j < num_cands; ) {
            if ((j & 3) != 0) {
                const size_t word_j = j >> 6;
                const uint64_t bit_j = 1ULL << (j & 63);
                if ((suppressed[word_j] & bit_j) == 0 && cand_cls[i] == cand_cls[j]) {
                    float ix1 = std::max(cand_x1[i], cand_x1[j]);
                    float iy1 = std::max(cand_y1[i], cand_y1[j]);
                    float ix2 = std::min(cand_x2[i], cand_x2[j]);
                    float iy2 = std::min(cand_y2[i], cand_y2[j]);
                    float iw = std::max(0.0f, ix2 - ix1);
                    float ih = std::max(0.0f, iy2 - iy1);
                    float inter = iw * ih;
                    float uni = cand_area[i] + cand_area[j] - inter;
                    if (uni > 0.0f && (inter / uni) > iou_threshold) {
                        suppressed[word_j] |= bit_j;
                    }
                }
                ++j;
                continue;
            }

            const size_t word_j = j >> 6;
            const size_t shift_j = j & 63;
            if (((suppressed[word_j] >> shift_j) & 0xF) == 0xF) {
                j += 4;
                continue;
            }

            const float32x4_t bx1 = vld1q_f32(&cand_x1[j]);
            const float32x4_t by1 = vld1q_f32(&cand_y1[j]);
            const float32x4_t bx2 = vld1q_f32(&cand_x2[j]);
            const float32x4_t by2 = vld1q_f32(&cand_y2[j]);
            const float32x4_t b_area = vld1q_f32(&cand_area[j]);
            const int32x4_t b_cls = vld1q_s32(&cand_cls[j]);

            const uint32x4_t cls_eq = vceqq_s32(a_cls, b_cls);

            const float32x4_t ix1 = vmaxq_f32(ax1, bx1);
            const float32x4_t iy1 = vmaxq_f32(ay1, by1);
            const float32x4_t ix2 = vminq_f32(ax2, bx2);
            const float32x4_t iy2 = vminq_f32(ay2, by2);

            const float32x4_t iw = vmaxq_f32(v_zero, vsubq_f32(ix2, ix1));
            const float32x4_t ih = vmaxq_f32(v_zero, vsubq_f32(iy2, iy1));
            const float32x4_t inter_area = vmulq_f32(iw, ih);

            const float32x4_t union_area = vsubq_f32(vaddq_f32(a_area, b_area), inter_area);
            const float32x4_t iou = vdivq_f32(inter_area, union_area);

            const uint32x4_t iou_gt = vcgtq_f32(iou, v_iou_thresh);
            const uint32x4_t match = vandq_u32(iou_gt, cls_eq);

            uint32_t m_arr[4];
            vst1q_u32(m_arr, match);
            int sup_bits = 0;
            for (int k = 0; k < 4; ++k) {
                if (m_arr[k]) sup_bits |= (1 << k);
            }
            if (j + 4 > num_cands) {
                int valid_lanes = static_cast<int>(num_cands - j);
                sup_bits &= ((1 << valid_lanes) - 1);
            }
            if (sup_bits) {
                suppressed[word_j] |= (static_cast<uint64_t>(sup_bits) << shift_j);
            }
            j += 4;
        }
    }
#else
    // Pure Scalar Bitmask Tracking (0 heap allocations)
    for (size_t i = 0; i < num_cands; ++i) {
        const size_t word_i = i >> 6;
        const uint64_t bit_i = 1ULL << (i & 63);
        if ((suppressed[word_i] & bit_i) != 0) continue;

        out_dets.push_back(candidates[i]);

        for (size_t j = i + 1; j < num_cands; ++j) {
            const size_t word_j = j >> 6;
            const uint64_t bit_j = 1ULL << (j & 63);
            if ((suppressed[word_j] & bit_j) != 0) continue;

            if (cand_cls[i] == cand_cls[j]) {
                float ix1 = std::max(cand_x1[i], cand_x1[j]);
                float iy1 = std::max(cand_y1[i], cand_y1[j]);
                float ix2 = std::min(cand_x2[i], cand_x2[j]);
                float iy2 = std::min(cand_y2[i], cand_y2[j]);
                float iw = std::max(0.0f, ix2 - ix1);
                float ih = std::max(0.0f, iy2 - iy1);
                float inter = iw * ih;
                float uni = cand_area[i] + cand_area[j] - inter;
                if (uni > 0.0f && (inter / uni) > iou_threshold) {
                    suppressed[word_j] |= bit_j;
                }
            }
        }
    }
#endif
}

void ignite_engine::decode_detections_for_slot(int slot, std::vector<Candidate>& out_dets) {
    out_dets.clear();
    scratch_candidates.clear();

    float conf_t = conf_thres;
    float iou_t = iou_thres;
    float scale = slot_scale[slot];
    int pad_left = slot_pad_left[slot];
    int pad_top = slot_pad_top[slot];

    if (fused_dfl) {
        // FAST PATH: On-die AIE2 micro-kernel decoded boxes and class scores
        const uint8_t* raw_out = bo_out[slot].map<uint8_t*>();
        const float* boxes_ptr = reinterpret_cast<const float*>(raw_out);
        const float* scores_ptr = reinterpret_cast<const float*>(raw_out + 134400);

#if defined(IGNITE_USE_AVX2)
        const __m256 v_thresh = _mm256_set1_ps(conf_t);
        for (int i = 0; i < 8400; ++i) {
            const float* score_row = scores_ptr + i * 80;
            // 80 classes = 10 x 8-lane AVX2 vectors
            __m256 v0 = _mm256_loadu_ps(score_row + 0);
            __m256 v1 = _mm256_loadu_ps(score_row + 8);
            __m256 v2 = _mm256_loadu_ps(score_row + 16);
            __m256 v3 = _mm256_loadu_ps(score_row + 24);
            __m256 v4 = _mm256_loadu_ps(score_row + 32);
            __m256 v5 = _mm256_loadu_ps(score_row + 40);
            __m256 v6 = _mm256_loadu_ps(score_row + 48);
            __m256 v7 = _mm256_loadu_ps(score_row + 56);
            __m256 v8 = _mm256_loadu_ps(score_row + 64);
            __m256 v9 = _mm256_loadu_ps(score_row + 72);

            __m256 m01 = _mm256_max_ps(v0, v1);
            __m256 m23 = _mm256_max_ps(v2, v3);
            __m256 m45 = _mm256_max_ps(v4, v5);
            __m256 m67 = _mm256_max_ps(v6, v7);
            __m256 m89 = _mm256_max_ps(v8, v9);

            __m256 m0123 = _mm256_max_ps(m01, m23);
            __m256 m4567 = _mm256_max_ps(m45, m67);
            __m256 m_all = _mm256_max_ps(_mm256_max_ps(m0123, m4567), m89);

            __m256 gt = _mm256_cmp_ps(m_all, v_thresh, _CMP_GT_OQ);
            if (_mm256_movemask_ps(gt) == 0) {
                continue;
            }

            float max_val = score_row[0];
            int best_cls = 0;
            for (int c = 1; c < 80; ++c) {
                if (score_row[c] > max_val) {
                    max_val = score_row[c];
                    best_cls = c;
                }
            }
            if (max_val < conf_t) continue;

            const float* box = boxes_ptr + i * 4;
            float x1 = box[0];
            float y1 = box[1];
            float x2 = box[2];
            float y2 = box[3];

            float bw = x2 - x1;
            float bh = y2 - y1;
            float cx = (x1 + x2) * 0.5f;
            float cy = (y1 + y2) * 0.5f;

            float orig_x0 = (cx - bw * 0.5f - static_cast<float>(pad_left)) / scale;
            float orig_y0 = (cy - bh * 0.5f - static_cast<float>(pad_top)) / scale;
            float orig_w = bw / scale;
            float orig_h = bh / scale;

            Candidate cand;
            cand.x0 = orig_x0;
            cand.y0 = orig_y0;
            cand.w = orig_w;
            cand.h = orig_h;
            cand.score = max_val;
            cand.class_id = best_cls;
            scratch_candidates.push_back(cand);
        }
#else
        for (int i = 0; i < 8400; ++i) {
            const float* score_row = scores_ptr + i * 80;
            float max_val = score_row[0];
            int best_cls = 0;
            for (int c = 1; c < 80; ++c) {
                if (score_row[c] > max_val) {
                    max_val = score_row[c];
                    best_cls = c;
                }
            }
            if (max_val < conf_t) continue;

            const float* box = boxes_ptr + i * 4;
            float x1 = box[0];
            float y1 = box[1];
            float x2 = box[2];
            float y2 = box[3];

            float bw = x2 - x1;
            float bh = y2 - y1;
            float cx = (x1 + x2) * 0.5f;
            float cy = (y1 + y2) * 0.5f;

            float orig_x0 = (cx - bw * 0.5f - static_cast<float>(pad_left)) / scale;
            float orig_y0 = (cy - bh * 0.5f - static_cast<float>(pad_top)) / scale;
            float orig_w = bw / scale;
            float orig_h = bh / scale;

            Candidate cand;
            cand.x0 = orig_x0;
            cand.y0 = orig_y0;
            cand.w = orig_w;
            cand.h = orig_h;
            cand.score = max_val;
            cand.class_id = best_cls;
            scratch_candidates.push_back(cand);
        }
#endif
    } else {
        if (!has_reference_heads) {
            return;
        }

        const float* boxes[3] = {
            ref_p3_box.data(), ref_p4_box.data(), ref_p5_box.data()
        };
        const float* clses[3] = {
            ref_p3_cls.data(), ref_p4_cls.data(), ref_p5_cls.data()
        };
        const int grid_sizes[3] = { 80, 40, 20 };
        const float strides[3] = { 8.0f, 16.0f, 32.0f };
        float logit_t = std::log(conf_t / (1.0f - conf_t));

        for (int h_idx = 0; h_idx < 3; ++h_idx) {
            int G = grid_sizes[h_idx];
            int N = G * G;
            float stride_val = strides[h_idx];
            const float* box_ptr = boxes[h_idx];
            const float* cls_ptr = clses[h_idx];

            float* max_logits = scratch_max_logits.data();
            int* best_cls_arr = scratch_best_cls.data();

            // Cache-optimal vectorized logit reduction: loop c on the outside
            std::memcpy(max_logits, cls_ptr, N * sizeof(float));
            std::memset(best_cls_arr, 0, N * sizeof(int));

            for (int c = 1; c < 80; ++c) {
                const float* class_c_ptr = cls_ptr + c * N;
#if defined(IGNITE_USE_AVX2)
                const __m256i v_c = _mm256_set1_epi32(c);
                int i = 0;
                for (; i + 15 < N; i += 16) {
                    __m256 v_curr0 = _mm256_loadu_ps(class_c_ptr + i);
                    __m256 v_max0 = _mm256_loadu_ps(max_logits + i);
                    __m256 cmp0 = _mm256_cmp_ps(v_curr0, v_max0, _CMP_GT_OQ);
                    _mm256_storeu_ps(max_logits + i, _mm256_max_ps(v_curr0, v_max0));
                    __m256i cur_cls0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(best_cls_arr + i));
                    __m256i updated_cls0 = _mm256_blendv_epi8(cur_cls0, v_c, _mm256_castps_si256(cmp0));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(best_cls_arr + i), updated_cls0);

                    __m256 v_curr1 = _mm256_loadu_ps(class_c_ptr + i + 8);
                    __m256 v_max1 = _mm256_loadu_ps(max_logits + i + 8);
                    __m256 cmp1 = _mm256_cmp_ps(v_curr1, v_max1, _CMP_GT_OQ);
                    _mm256_storeu_ps(max_logits + i + 8, _mm256_max_ps(v_curr1, v_max1));
                    __m256i cur_cls1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(best_cls_arr + i + 8));
                    __m256i updated_cls1 = _mm256_blendv_epi8(cur_cls1, v_c, _mm256_castps_si256(cmp1));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(best_cls_arr + i + 8), updated_cls1);
                }
                for (; i + 7 < N; i += 8) {
                    __m256 v_curr = _mm256_loadu_ps(class_c_ptr + i);
                    __m256 v_max = _mm256_loadu_ps(max_logits + i);
                    __m256 cmp = _mm256_cmp_ps(v_curr, v_max, _CMP_GT_OQ);
                    _mm256_storeu_ps(max_logits + i, _mm256_max_ps(v_curr, v_max));
                    __m256i cur_cls = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(best_cls_arr + i));
                    __m256i updated_cls = _mm256_blendv_epi8(cur_cls, v_c, _mm256_castps_si256(cmp));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(best_cls_arr + i), updated_cls);
                }
                for (; i < N; ++i) {
                    float v = class_c_ptr[i];
                    if (v > max_logits[i]) {
                        max_logits[i] = v;
                        best_cls_arr[i] = c;
                    }
                }
#else
                for (int i = 0; i < N; ++i) {
                    float v = class_c_ptr[i];
                    if (v > max_logits[i]) {
                        max_logits[i] = v;
                        best_cls_arr[i] = c;
                    }
                }
#endif
            }

            for (int i = 0; i < N; ++i) {
                float max_logit = max_logits[i];
                if (max_logit <= logit_t) continue;

                float score = 1.0f / (1.0f + std::exp(-max_logit));
                if (score < conf_t) continue;

                int best_cls = best_cls_arr[i];
                int col = i % G;
                int row = i / G;

                // DFL Softmax Projection on 16 bins for 4 coordinates
                float dist[4];
                for (int d = 0; d < 4; ++d) {
                    float bin_vals[16];
                    float max_b = -1e9f;
                    for (int k = 0; k < 16; ++k) {
                        float v = box_ptr[(d * 16 + k) * N + i];
                        bin_vals[k] = v;
                        if (v > max_b) max_b = v;
                    }
                    float sum_exp = 0.0f;
                    for (int k = 0; k < 16; ++k) {
                        bin_vals[k] = std::exp(bin_vals[k] - max_b);
                        sum_exp += bin_vals[k];
                    }
                    float exp_dist = 0.0f;
                    for (int k = 0; k < 16; ++k) {
                        exp_dist += static_cast<float>(k) * (bin_vals[k] / sum_exp);
                    }
                    dist[d] = exp_dist;
                }

                // Anchor Grid Projection
                float ax = static_cast<float>(col) + 0.5f;
                float ay = static_cast<float>(row) + 0.5f;

                float x1 = (ax - dist[0]) * stride_val;
                float y1 = (ay - dist[1]) * stride_val;
                float x2 = (ax + dist[2]) * stride_val;
                float y2 = (ay + dist[3]) * stride_val;

                float cx = (x1 + x2) * 0.5f;
                float cy = (y1 + y2) * 0.5f;
                float bw = (x2 - x1);
                float bh = (y2 - y1);

                // Letterbox Coordinate Inversion
                float orig_x0 = (cx - bw * 0.5f - static_cast<float>(pad_left)) / scale;
                float orig_y0 = (cy - bh * 0.5f - static_cast<float>(pad_top)) / scale;
                float orig_w = bw / scale;
                float orig_h = bh / scale;

                Candidate cand;
                cand.x0 = orig_x0;
                cand.y0 = orig_y0;
                cand.w = orig_w;
                cand.h = orig_h;
                cand.score = score;
                cand.class_id = best_cls;
                scratch_candidates.push_back(cand);
            }
        }
    }

    // Run Vectorized SIMD Bitmask NMS (0 heap allocations)
    run_vectorized_bitmask_nms(scratch_candidates, iou_t, out_dets);
}

void ignite_engine::npu_worker_loop() {
    while (!worker_stop.load(std::memory_order_relaxed)) {
        NpuWorkItem item;
        {
            std::unique_lock<std::mutex> lock(npu_queue_mutex);
            npu_queue_cv.wait(lock, [this]() {
                return !npu_work_queue.empty() || worker_stop.load(std::memory_order_relaxed);
            });
            if (worker_stop.load(std::memory_order_relaxed) && npu_work_queue.empty()) break;
            item = npu_work_queue.front();
            npu_work_queue.pop();
        }

        int slot = item.slot;
        uint64_t ticket = item.ticket;

        auto t_npu_start = std::chrono::high_resolution_clock::now();

        // 1. Physical AIE2 Silicon Execution (Single-Dispatch Fast Path).
        // A run that does not reach COMPLETED leaves stale bytes in bo_out;
        // count it so a stream of timeouts is visible instead of silent.
        bool completed = true;
        if (single_dispatch && bo_monolithic && ninstr_monolithic > 0) {
            xrt::run r = (*kernel)(3, bo_monolithic, ninstr_monolithic, bo_in[slot], bo_out[slot]);
            completed = (r.wait(2000) == ERT_CMD_STATE_COMPLETED);
        } else {
            for (const auto& s : stages) {
                if (s.ninstr_exec > 0 && s.bo_exec) {
                    xrt::run r = (*kernel)(3, s.bo_exec, s.ninstr_exec, bo_in[slot], bo_out[slot]);
                    completed = (r.wait(2000) == ERT_CMD_STATE_COMPLETED) && completed;
                }
            }
        }
        if (!completed) {
            uint64_t n = dispatch_timeouts.fetch_add(1) + 1;
            if (n == 1 || (n & (n - 1)) == 0) {
                std::cerr << "[ignite] NPU dispatch for ticket " << ticket
                          << " did not complete (" << n << " so far); output is stale" << std::endl;
            }
        }
        bo_out[slot].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto t_npu_done = std::chrono::high_resolution_clock::now();

        // 2. Enqueue to postprocessing worker for concurrent DFL decode + batched NMS
        {
            std::lock_guard<std::mutex> lock(post_queue_mutex);
            post_work_queue.push({ticket, slot, t_npu_start, t_npu_done});
        }
        post_queue_cv.notify_one();
    }
}

void ignite_engine::post_worker_loop() {
    while (!worker_stop.load(std::memory_order_relaxed)) {
        PostWorkItem item;
        {
            std::unique_lock<std::mutex> lock(post_queue_mutex);
            post_queue_cv.wait(lock, [this]() {
                return !post_work_queue.empty() || worker_stop.load(std::memory_order_relaxed);
            });
            if (worker_stop.load(std::memory_order_relaxed) && post_work_queue.empty()) break;
            item = post_work_queue.front();
            post_work_queue.pop();
        }

        int slot = item.slot;
        uint64_t ticket = item.ticket;

        // Pure C++20 DFL Decode + Batched NMS
        std::vector<Candidate> dets;
        dets.reserve(64);
        decode_detections_for_slot(slot, dets);
        auto t_post_done = std::chrono::high_resolution_clock::now();

        // Fine-grained latency records
        ignite_timings_t timings;
        timings.preprocess_ms = std::chrono::duration<double, std::milli>(slot_t_prep[slot] - slot_t_start[slot]).count();
        timings.npu_exec_ms = std::chrono::duration<double, std::milli>(item.t_npu_done - item.t_npu_start).count();
        timings.postprocess_ms = std::chrono::duration<double, std::milli>(t_post_done - item.t_npu_done).count();
        timings.glass_to_glass_ms = std::chrono::duration<double, std::milli>(t_post_done - slot_t_start[slot]).count();

        // Update fixed circular ring buffer & signal completion
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            size_t ring_idx = static_cast<size_t>(ticket % RING_SIZE);
            result_ring[ring_idx].ticket = ticket;
            result_ring[ring_idx].detections = std::move(dets);
            result_ring[ring_idx].timings = timings;
            // Tickets can complete out of order when several producers call
            // ignite_run_async; keep the high-water mark monotonic and let
            // waiters key on their own ring entry.
            if (ticket > last_completed_ticket.load(std::memory_order_relaxed)) {
                last_completed_ticket.store(ticket, std::memory_order_release);
            }
            last_timings = timings;
        }
        result_cv.notify_all();
        slot_cv.notify_all();
    }
}

ignite_engine_t* ignite_load(const char* model_path, int device_id) {
    if (!model_path) {
        set_error("model_path cannot be NULL");
        return nullptr;
    }

    fs::path mp(model_path);
    if (!fs::exists(mp)) {
        set_error("Model container file does not exist: " + mp.string());
        return nullptr;
    }

    auto eng = std::make_unique<ignite_engine>();

    // 1. Win32 Memory-Mapping of .ignite container
    eng->h_file = CreateFileA(
        model_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL
    );
    if (eng->h_file == INVALID_HANDLE_VALUE) {
        set_error("Failed to open file for memory-mapping: " + mp.string());
        return nullptr;
    }

    LARGE_INTEGER fsz;
    if (!GetFileSizeEx(eng->h_file, &fsz)) {
        set_error("Failed to query file size");
        return nullptr;
    }
    eng->mmap_size = static_cast<size_t>(fsz.QuadPart);

    eng->h_map = CreateFileMappingA(eng->h_file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!eng->h_map) {
        set_error("Failed to create file mapping object");
        return nullptr;
    }

    eng->mmap_base = static_cast<const uint8_t*>(MapViewOfFile(eng->h_map, FILE_MAP_READ, 0, 0, 0));
    if (!eng->mmap_base) {
        set_error("Failed to map view of file into process memory");
        return nullptr;
    }

    // 2. Validate Container Header
    if (eng->mmap_size < sizeof(IgniteHeaderRaw)) {
        set_error("File size is smaller than Ignite header");
        return nullptr;
    }

    const auto* hdr = reinterpret_cast<const IgniteHeaderRaw*>(eng->mmap_base);
    if (std::memcmp(hdr->magic, "IGNT", 4) != 0) {
        set_error("Invalid container magic signature; expected 'IGNT'");
        return nullptr;
    }
    if (hdr->version != 1) {
        set_error("Unsupported .ignite container version: " + std::to_string(hdr->version));
        return nullptr;
    }
    if (hdr->header_size != sizeof(IgniteHeaderRaw)) {
        set_error("Unexpected .ignite header size: " + std::to_string(hdr->header_size));
        return nullptr;
    }
    if (hdr->total_file_size != eng->mmap_size) {
        set_error("Container declares " + std::to_string(hdr->total_file_size) + " bytes but the file holds "
                  + std::to_string(eng->mmap_size) + " (truncated or trailing data)");
        return nullptr;
    }
    {
        const uint32_t body_crc = crc32_ieee(eng->mmap_base + sizeof(IgniteHeaderRaw),
                                             eng->mmap_size - sizeof(IgniteHeaderRaw));
        if (body_crc != hdr->crc32) {
            char buf[96];
            snprintf(buf, sizeof(buf), "Container CRC32 mismatch: header 0x%08x, body 0x%08x", hdr->crc32, body_crc);
            set_error(buf);
            return nullptr;
        }
    }

    // 3. Parse JSON Manifest
    if (hdr->manifest_offset + hdr->manifest_size > eng->mmap_size) {
        set_error("Manifest offset/size exceeds total container file boundary");
        return nullptr;
    }

    std::string manifest_str(
        reinterpret_cast<const char*>(eng->mmap_base + hdr->manifest_offset),
        static_cast<size_t>(hdr->manifest_size)
    );

    nlohmann::json manifest;
    try {
        manifest = nlohmann::json::parse(manifest_str);
    } catch (const std::exception& e) {
        set_error("Failed to parse manifest JSON: " + std::string(e.what()));
        return nullptr;
    }

    eng->model_name = manifest.value("model_name", mp.stem().string());
    if (manifest.contains("architecture")) {
        eng->num_cores = manifest["architecture"].value("num_cores", 16);
    }

    // Every blob range is checked against the mapping before any bo.write
    // reads from it; a bad directory entry fails the load instead of reading
    // past the file mapping.
    std::map<std::string, BlobEntry> blobs;
    std::string blob_error;
    if (manifest.contains("blobs")) {
        auto add_blob_node = [&](const std::string& name, const nlohmann::json& binfo) {
            BlobEntry entry;
            entry.name = name;
            entry.offset = binfo.value("offset", 0ULL);
            entry.size = binfo.value("size", 0ULL);
            if (entry.name.empty()) {
                return;
            }
            const size_t manifest_end = static_cast<size_t>(hdr->manifest_offset + hdr->manifest_size);
            if (entry.offset % 64 != 0 || entry.offset < manifest_end
                || entry.size > eng->mmap_size || entry.offset > eng->mmap_size - entry.size) {
                blob_error = "Blob '" + entry.name + "' range [" + std::to_string(entry.offset) + ", "
                             + std::to_string(entry.offset + entry.size) + ") is outside the blob section";
                return;
            }
            if (blobs.count(entry.name)) {
                blob_error = "Duplicate blob name '" + entry.name + "' in the container directory";
                return;
            }
            blobs[entry.name] = entry;
        };

        if (manifest["blobs"].is_array()) {
            for (const auto& binfo : manifest["blobs"]) {
                add_blob_node(binfo.value("name", ""), binfo);
            }
        } else if (manifest["blobs"].is_object()) {
            for (auto& [bname, binfo] : manifest["blobs"].items()) {
                add_blob_node(bname, binfo);
            }
        }
    }
    if (!blob_error.empty()) {
        set_error(blob_error);
        return nullptr;
    }

    // 4. Initialize Native XRT Hardware Context
    fs::path xclbin_path = resolve_xclbin(mp);
    if (!fs::exists(xclbin_path)) {
        set_error("Could not find xclbin at: " + xclbin_path.string());
        return nullptr;
    }

    try {
        eng->device = std::make_unique<xrt::device>(device_id);
        xrt::xclbin xclbin_obj(xclbin_path.string());
        xrt::uuid uuid = eng->device->register_xclbin(xclbin_obj);
        eng->hw_ctx = std::make_unique<xrt::hw_context>(*eng->device, uuid);
        eng->kernel = std::make_unique<xrt::kernel>(*eng->hw_ctx, "MLIR_AIE");

        // Allocate double-buffered I/O BOs
        eng->in_bytes = (eng->num_cores == 16) ? 8192 : 2048;
        eng->out_bytes = (eng->num_cores == 16) ? 4096 : 1024;
        eng->fused_dfl = manifest.value("fused_dfl", false);
        if (eng->fused_dfl) {
            eng->out_bytes = 134400 + 2688000;
        }
        for (int s = 0; s < ignite_engine::NUM_SLOTS; ++s) {
            eng->bo_in[s] = xrt::bo(*eng->device, eng->in_bytes, xrt::bo::flags::host_only, eng->kernel->group_id(3));
            eng->bo_out[s] = xrt::bo(*eng->device, eng->out_bytes, xrt::bo::flags::host_only, eng->kernel->group_id(4));
            eng->chw_buffer[s].resize(3 * 640 * 640, -14);
        }

        // 5. Construct Stage Instruction Buffers directly from mmap
        if (manifest.contains("stages")) {
            auto parse_stage_node = [&](const std::string& key_name, const nlohmann::json& s_node) {
                StageResource s;
                s.name = s_node.value("stage_name", key_name);
                s.stage_idx = s_node.value("index", s_node.value("stage_idx", 0));
                s.ninstr_init = s_node.value("init_bytes", s_node.value("ninstr_init", 0U));
                // exec_bytes was never read before, which left every stage
                // without an exec buffer and made the non-monolithic path a
                // silent no-op.
                s.ninstr_exec = s_node.value("exec_bytes", s_node.value("ninstr_exec", 0U));
                std::string init_blob = (s_node.contains("init_blob") && !s_node["init_blob"].is_null() && s_node["init_blob"].is_string())
                    ? s_node["init_blob"].get<std::string>() : "";
                std::string exec_blob = (s_node.contains("exec_blob") && !s_node["exec_blob"].is_null() && s_node["exec_blob"].is_string())
                    ? s_node["exec_blob"].get<std::string>() : "";

                if (s.ninstr_init > 0 && !init_blob.empty() && blobs.count(init_blob)) {
                    const auto& b = blobs.at(init_blob);
                    if (s.ninstr_init > b.size) {
                        throw std::runtime_error("stage " + s.name + " init_bytes exceeds blob " + init_blob);
                    }
                    s.bo_init = xrt::bo(*eng->device, s.ninstr_init, xrt::bo::flags::cacheable, eng->kernel->group_id(1));
                    s.bo_init.write(eng->mmap_base + b.offset, s.ninstr_init, 0);
                    s.bo_init.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                }

                if (s.ninstr_exec > 0 && !exec_blob.empty() && blobs.count(exec_blob)) {
                    const auto& b = blobs.at(exec_blob);
                    if (s.ninstr_exec > b.size) {
                        throw std::runtime_error("stage " + s.name + " exec_bytes exceeds blob " + exec_blob);
                    }
                    s.bo_exec = xrt::bo(*eng->device, s.ninstr_exec, xrt::bo::flags::cacheable, eng->kernel->group_id(1));
                    s.bo_exec.write(eng->mmap_base + b.offset, s.ninstr_exec, 0);
                    s.bo_exec.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                }
                return s;
            };

            if (manifest["stages"].is_object()) {
                for (auto& [s_name, s_node] : manifest["stages"].items()) {
                    eng->stages.push_back(parse_stage_node(s_name, s_node));
                }
            } else if (manifest["stages"].is_array()) {
                for (const auto& s_node : manifest["stages"]) {
                    eng->stages.push_back(parse_stage_node("", s_node));
                }
            }

            // Ensure stages are sorted strictly by stage index (0..8)
            std::sort(eng->stages.begin(), eng->stages.end(), [](const StageResource& a, const StageResource& b) {
                return a.stage_idx < b.stage_idx;
            });
        }

        // 5b. Unified Single-Dispatch Monolithic ERT Instruction Stream
        std::string mono_blob = manifest.value("monolithic_exec_blob", "exec_monolithic.bin");
        if (blobs.count(mono_blob)) {
            const auto& b = blobs.at(mono_blob);
            eng->ninstr_monolithic = static_cast<uint32_t>(b.size);
            eng->bo_monolithic = xrt::bo(*eng->device, eng->ninstr_monolithic, xrt::bo::flags::cacheable, eng->kernel->group_id(1));
            eng->bo_monolithic.write(eng->mmap_base + b.offset, eng->ninstr_monolithic, 0);
            eng->bo_monolithic.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            eng->single_dispatch = true;
        }

        // A container with neither a monolithic stream nor any stage exec
        // stream would run nothing per frame and return stale detections.
        bool any_stage_exec = false;
        for (const auto& s : eng->stages) {
            any_stage_exec = any_stage_exec || (s.ninstr_exec > 0 && s.bo_exec);
        }
        if (!eng->single_dispatch && !any_stage_exec) {
            throw std::runtime_error("container has no executable transaction stream (no '" + mono_blob
                                     + "' and no stage exec blob)");
        }

        // 6. Stationary Parameter Programming & Double-Buffer Warmup.
        // Every dispatch state is checked: an init that times out would leave
        // the cores unprogrammed while the engine reports success.
        for (const auto& s : eng->stages) {
            if (s.ninstr_init > 0 && s.bo_init) {
                xrt::run r = (*eng->kernel)(3, s.bo_init, s.ninstr_init, eng->bo_in[0], eng->bo_out[0]);
                require_completed(r.wait(3000), ("init stream of stage " + s.name).c_str());
            }
        }
        for (int s_idx = 0; s_idx < ignite_engine::NUM_SLOTS; ++s_idx) {
            if (eng->single_dispatch && eng->bo_monolithic && eng->ninstr_monolithic > 0) {
                xrt::run r = (*eng->kernel)(3, eng->bo_monolithic, eng->ninstr_monolithic, eng->bo_in[s_idx], eng->bo_out[s_idx]);
                require_completed(r.wait(2000), "monolithic warm-up dispatch");
            } else {
                for (const auto& s : eng->stages) {
                    if (s.ninstr_exec > 0 && s.bo_exec) {
                        xrt::run r = (*eng->kernel)(3, s.bo_exec, s.ninstr_exec, eng->bo_in[s_idx], eng->bo_out[s_idx]);
                        require_completed(r.wait(2000), ("warm-up dispatch of stage " + s.name).c_str());
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        set_error("Native XRT hardware initialization error: " + std::string(e.what()));
        return nullptr;
    }

    // 7. Scratch vector allocations for DFL decode
    eng->scratch_max_logits.resize(6400, 0.0f);
    eng->scratch_best_cls.resize(6400, 0);
    eng->scratch_candidates.reserve(512);
    eng->scratch_suppressed.reserve(512);

    // 8. Try loading reference heads if available
    try_load_default_heads(eng.get(), mp);

    // 9. Start background NPU worker thread for async ping-pong processing
    eng->start_worker();

    return eng.release();
}

int ignite_run_async(
    ignite_engine_t* engine,
    const uint8_t* bgr_data,
    int width,
    int height,
    int stride,
    uint64_t* out_ticket
) {
    if (!engine) {
        set_error("Engine handle is NULL");
        return -1;
    }
    if (!bgr_data || width <= 0 || height <= 0 || stride <= 0) {
        set_error("Invalid input image buffer arguments");
        return -2;
    }
    if (!out_ticket) {
        set_error("out_ticket pointer cannot be NULL");
        return -3;
    }

    uint64_t ticket = engine->next_ticket.fetch_add(1);
    int slot = static_cast<int>(ticket % ignite_engine::NUM_SLOTS);

    // Bounded pipeline: wait until the previous occupant of this slot (ticket
    // - NUM_SLOTS) has been post-processed. Keyed on that ticket's own ring
    // entry, so out-of-order completion by concurrent producers cannot free
    // a slot that is still in flight.
    {
        const uint64_t predecessor = ticket > ignite_engine::NUM_SLOTS ? ticket - ignite_engine::NUM_SLOTS : 0;
        const size_t predecessor_idx = static_cast<size_t>(predecessor % ignite_engine::RING_SIZE);
        std::unique_lock<std::mutex> lock(engine->result_mutex);
        engine->slot_cv.wait(lock, [engine, predecessor, predecessor_idx]() {
            return predecessor == 0
                || engine->result_ring[predecessor_idx].ticket == predecessor
                || engine->worker_stop.load();
        });
        if (engine->worker_stop.load()) {
            set_error("Worker thread stopped");
            return -4;
        }
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    engine->slot_t_start[slot] = t_start;

    // Stage 1: C-SIMD Letterbox + Bilinear Interpolation + BGR->RGB + Int8 Quantize
    int pad_top = 0, pad_left = 0;
    float scale = 1.0f;
    int prep_rc = fused_preprocess_bgr_to_chw_int8(
        bgr_data, width, height, stride,
        engine->chw_buffer[slot].data(), 640, 640,
        &pad_top, &pad_left, &scale
    );
    if (prep_rc != 0) {
        set_error("Ingress preprocessor failed with code: " + std::to_string(prep_rc));
        return -5;
    }
    engine->slot_pad_top[slot] = pad_top;
    engine->slot_pad_left[slot] = pad_left;
    engine->slot_scale[slot] = scale;

    // DMA Write and Sync to Device
    engine->bo_in[slot].write(engine->chw_buffer[slot].data(), engine->in_bytes, 0);
    engine->bo_in[slot].sync(XCL_BO_SYNC_BO_TO_DEVICE);

    engine->slot_t_prep[slot] = std::chrono::high_resolution_clock::now();

    // Enqueue work item for concurrent NPU worker thread
    {
        std::lock_guard<std::mutex> lock(engine->npu_queue_mutex);
        engine->npu_work_queue.push({ticket, slot});
    }
    engine->npu_queue_cv.notify_one();

    *out_ticket = ticket;
    return 0;
}

int ignite_wait(
    ignite_engine_t* engine,
    uint64_t ticket,
    ignite_detection_t* out_detections,
    int max_detections
) {
    if (!engine || ticket == 0) {
        set_error("Invalid argument to ignite_wait");
        return -1;
    }

    // Wait for this ticket's own ring entry (tickets may complete out of
    // order); give up when the entry has been overwritten by a ticket a full
    // ring later, or when the workers stop.
    {
        const size_t ring_idx = static_cast<size_t>(ticket % ignite_engine::RING_SIZE);
        std::unique_lock<std::mutex> lock(engine->result_mutex);
        engine->result_cv.wait(lock, [engine, ticket, ring_idx]() {
            return engine->result_ring[ring_idx].ticket == ticket
                || engine->last_completed_ticket.load() >= ticket + ignite_engine::RING_SIZE
                || engine->worker_stop.load();
        });

        const auto& res = engine->result_ring[ring_idx];
        if (res.ticket != ticket) {
            if (engine->worker_stop.load()) {
                set_error("Worker stopped before ticket completion");
                return -2;
            }
            set_error("Ticket sequence out of sync or expired in ring buffer");
            return -3;
        }

        engine->last_timings = res.timings;

        int num_dets = 0;
        if (out_detections && max_detections > 0) {
            num_dets = static_cast<int>(std::min(static_cast<size_t>(max_detections), res.detections.size()));
            for (int i = 0; i < num_dets; ++i) {
                out_detections[i].x0 = res.detections[i].x0;
                out_detections[i].y0 = res.detections[i].y0;
                out_detections[i].w = res.detections[i].w;
                out_detections[i].h = res.detections[i].h;
                out_detections[i].score = res.detections[i].score;
                out_detections[i].class_id = res.detections[i].class_id;
                const char* cname = (res.detections[i].class_id >= 0 && res.detections[i].class_id < 80)
                    ? COCO_CLASSES[res.detections[i].class_id] : "unknown";
                strncpy_s(out_detections[i].class_name, sizeof(out_detections[i].class_name), cname, _TRUNCATE);
            }
        }
        return num_dets;
    }
}

int ignite_run(
    ignite_engine_t* engine,
    const uint8_t* bgr_data,
    int width,
    int height,
    int stride,
    ignite_detection_t* out_detections,
    int max_detections
) {
    uint64_t ticket = 0;
    int rc = ignite_run_async(engine, bgr_data, width, height, stride, &ticket);
    if (rc != 0) {
        return rc;
    }
    return ignite_wait(engine, ticket, out_detections, max_detections);
}

void ignite_free(ignite_engine_t* engine) {
    if (engine) {
        delete engine;
    }
}

void ignite_set_thresholds(ignite_engine_t* engine, float conf_thres, float iou_thres) {
    if (engine) {
        engine->conf_thres = conf_thres;
        engine->iou_thres = iou_thres;
    }
}

void ignite_get_last_timings(ignite_engine_t* engine, ignite_timings_t* out_timings) {
    if (engine && out_timings) {
        // last_timings is written by the post-processing thread under result_mutex.
        std::lock_guard<std::mutex> lock(engine->result_mutex);
        *out_timings = engine->last_timings;
    }
}

int ignite_load_reference_heads(ignite_engine_t* engine, const char* heads_bin_path) {
    if (!engine || !heads_bin_path) {
        set_error("Invalid arguments to ignite_load_reference_heads");
        return -1;
    }

    std::ifstream f(heads_bin_path, std::ios::binary);
    if (!f.is_open()) {
        set_error("Could not open heads binary at: " + std::string(heads_bin_path));
        return -2;
    }

    engine->ref_p3_box.resize(64 * 80 * 80);
    engine->ref_p4_box.resize(64 * 40 * 40);
    engine->ref_p5_box.resize(64 * 20 * 20);
    engine->ref_p3_cls.resize(80 * 80 * 80);
    engine->ref_p4_cls.resize(80 * 40 * 40);
    engine->ref_p5_cls.resize(80 * 20 * 20);

    f.read(reinterpret_cast<char*>(engine->ref_p3_box.data()), engine->ref_p3_box.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(engine->ref_p4_box.data()), engine->ref_p4_box.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(engine->ref_p5_box.data()), engine->ref_p5_box.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(engine->ref_p3_cls.data()), engine->ref_p3_cls.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(engine->ref_p4_cls.data()), engine->ref_p4_cls.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(engine->ref_p5_cls.data()), engine->ref_p5_cls.size() * sizeof(float));

    if (!f.good()) {
        set_error("Failed to read all 6 head tensors from: " + std::string(heads_bin_path));
        return -3;
    }

    engine->has_reference_heads = true;
    return 0;
}

const char* ignite_get_last_error(void) {
    return g_last_error.c_str();
}
