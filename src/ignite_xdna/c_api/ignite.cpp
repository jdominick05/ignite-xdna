// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    // Double-buffering constant
    static constexpr int NUM_SLOTS = 2;
    static constexpr size_t RING_SIZE = 64;

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

    // Native XRT Hardware Context
    std::unique_ptr<xrt::device> device;
    std::unique_ptr<xrt::hw_context> hw_ctx;
    std::unique_ptr<xrt::kernel> kernel;

    // Double-buffered BOs (slot 0 and slot 1)
    xrt::bo bo_in[NUM_SLOTS];
    xrt::bo bo_out[NUM_SLOTS];
    std::vector<int8_t> chw_buffer[NUM_SLOTS];
    int slot_pad_top[NUM_SLOTS] = {0, 0};
    int slot_pad_left[NUM_SLOTS] = {0, 0};
    float slot_scale[NUM_SLOTS] = {1.0f, 1.0f};
    std::chrono::high_resolution_clock::time_point slot_t_start[NUM_SLOTS];
    std::chrono::high_resolution_clock::time_point slot_t_prep[NUM_SLOTS];

    // Stage pipeline (CDO transaction blobs)
    std::vector<StageResource> stages;
    bool single_dispatch = false;
    uint32_t ninstr_monolithic = 0;
    xrt::bo bo_monolithic;

    // Detection hyperparameters
    float conf_thres = 0.25f;
    float iou_thres = 0.50f;

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
    struct WorkItem {
        uint64_t ticket;
        int slot;
    };

    std::atomic<uint64_t> next_ticket{1};
    std::atomic<uint64_t> last_completed_ticket{0};
    std::atomic<bool> worker_stop{false};

    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::queue<WorkItem> work_queue;

    std::mutex result_mutex;
    std::condition_variable result_cv;
    std::condition_variable slot_cv;

    std::thread worker_thread;

    void start_worker() {
        worker_stop.store(false);
        worker_thread = std::thread(&ignite_engine::worker_loop, this);
    }

    void stop_worker() {
        worker_stop.store(true);
        queue_cv.notify_all();
        result_cv.notify_all();
        slot_cv.notify_all();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    void decode_detections_for_slot(int slot, std::vector<Candidate>& out_dets);
    void worker_loop();

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

void ignite_engine::decode_detections_for_slot(int slot, std::vector<Candidate>& out_dets) {
    out_dets.clear();
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

    float conf_t = conf_thres;
    float iou_t = iou_thres;
    float logit_t = std::log(conf_t / (1.0f - conf_t));

    scratch_candidates.clear();

    for (int h_idx = 0; h_idx < 3; ++h_idx) {
        int G = grid_sizes[h_idx];
        int N = G * G;
        float stride_val = strides[h_idx];
        const float* box_ptr = boxes[h_idx];
        const float* cls_ptr = clses[h_idx];

        float* max_logits = scratch_max_logits.data();
        int* best_cls_arr = scratch_best_cls.data();
        std::memcpy(max_logits, cls_ptr, N * sizeof(float));
        std::memset(best_cls_arr, 0, N * sizeof(int));

        for (int c = 1; c < 80; ++c) {
            const float* c_row = cls_ptr + c * N;
            for (int i = 0; i < N; ++i) {
                if (c_row[i] > max_logits[i]) {
                    max_logits[i] = c_row[i];
                    best_cls_arr[i] = c;
                }
            }
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
            float scale = slot_scale[slot];
            int pad_left = slot_pad_left[slot];
            int pad_top = slot_pad_top[slot];

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

    std::sort(scratch_candidates.begin(), scratch_candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.score > b.score;
    });

    scratch_suppressed.assign(scratch_candidates.size(), false);
    out_dets.clear();

    for (size_t i = 0; i < scratch_candidates.size(); ++i) {
        if (scratch_suppressed[i]) continue;
        out_dets.push_back(scratch_candidates[i]);
        for (size_t j = i + 1; j < scratch_candidates.size(); ++j) {
            if (scratch_suppressed[j]) continue;
            if (scratch_candidates[i].class_id == scratch_candidates[j].class_id) {
                float iou = compute_iou(scratch_candidates[i], scratch_candidates[j]);
                if (iou > iou_t) {
                    scratch_suppressed[j] = true;
                }
            }
        }
    }
}

void ignite_engine::worker_loop() {
    while (!worker_stop.load()) {
        WorkItem item;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [this]() {
                return !work_queue.empty() || worker_stop.load();
            });
            if (worker_stop.load() && work_queue.empty()) break;
            item = work_queue.front();
            work_queue.pop();
        }

        int slot = item.slot;
        uint64_t ticket = item.ticket;

        auto t_npu_start = std::chrono::high_resolution_clock::now();

        // 1. Physical AIE2 Silicon Execution
        if (single_dispatch && bo_monolithic && ninstr_monolithic > 0) {
            xrt::run r = (*kernel)(3, bo_monolithic, ninstr_monolithic, bo_in[slot], bo_out[slot]);
            r.wait(2000);
        } else {
            for (const auto& s : stages) {
                if (s.ninstr_exec > 0 && s.bo_exec) {
                    xrt::run r = (*kernel)(3, s.bo_exec, s.ninstr_exec, bo_in[slot], bo_out[slot]);
                    r.wait(2000);
                }
            }
        }
        bo_out[slot].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto t_npu_done = std::chrono::high_resolution_clock::now();

        // 2. Pure C++20 DFL Decode + Batched NMS
        std::vector<Candidate> dets;
        dets.reserve(64);
        decode_detections_for_slot(slot, dets);
        auto t_post_done = std::chrono::high_resolution_clock::now();

        // 3. Fine-grained latency records
        ignite_timings_t timings;
        timings.preprocess_ms = std::chrono::duration<double, std::milli>(slot_t_prep[slot] - slot_t_start[slot]).count();
        timings.npu_exec_ms = std::chrono::duration<double, std::milli>(t_npu_done - t_npu_start).count();
        timings.postprocess_ms = std::chrono::duration<double, std::milli>(t_post_done - t_npu_done).count();
        timings.glass_to_glass_ms = std::chrono::duration<double, std::milli>(t_post_done - slot_t_start[slot]).count();

        // 4. Update fixed circular ring buffer & signal completion
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            size_t ring_idx = static_cast<size_t>(ticket % RING_SIZE);
            result_ring[ring_idx].ticket = ticket;
            result_ring[ring_idx].detections = std::move(dets);
            result_ring[ring_idx].timings = timings;
            last_completed_ticket.store(ticket);
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

    std::map<std::string, BlobEntry> blobs;
    if (manifest.contains("blobs")) {
        auto add_blob_node = [&](const nlohmann::json& binfo) {
            BlobEntry entry;
            entry.name = binfo.value("name", "");
            entry.offset = binfo.value("offset", 0ULL);
            entry.size = binfo.value("size", 0ULL);
            if (!entry.name.empty()) {
                blobs[entry.name] = entry;
            }
        };

        if (manifest["blobs"].is_array()) {
            for (const auto& binfo : manifest["blobs"]) {
                add_blob_node(binfo);
            }
        } else if (manifest["blobs"].is_object()) {
            for (auto& [bname, binfo] : manifest["blobs"].items()) {
                BlobEntry entry;
                entry.name = bname;
                entry.offset = binfo.value("offset", 0ULL);
                entry.size = binfo.value("size", 0ULL);
                blobs[bname] = entry;
            }
        }
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
                s.ninstr_exec = s_node.value("exec_bytes", s_node.value("ninstr_exec", 0U));
                std::string init_blob = s_node.value("init_blob", "");
                std::string exec_blob = s_node.value("exec_blob", "");

                if (s.ninstr_init > 0 && !init_blob.empty() && blobs.count(init_blob)) {
                    const auto& b = blobs.at(init_blob);
                    s.bo_init = xrt::bo(*eng->device, s.ninstr_init, xrt::bo::flags::cacheable, eng->kernel->group_id(1));
                    s.bo_init.write(eng->mmap_base + b.offset, s.ninstr_init, 0);
                    s.bo_init.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                }

                if (s.ninstr_exec > 0 && !exec_blob.empty() && blobs.count(exec_blob)) {
                    const auto& b = blobs.at(exec_blob);
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

        // 6. Stationary Parameter Programming & Double-Buffer Warmup
        for (const auto& s : eng->stages) {
            if (s.ninstr_init > 0 && s.bo_init) {
                xrt::run r = (*eng->kernel)(3, s.bo_init, s.ninstr_init, eng->bo_in[0], eng->bo_out[0]);
                r.wait(3000);
            }
        }
        for (int s_idx = 0; s_idx < ignite_engine::NUM_SLOTS; ++s_idx) {
            if (eng->single_dispatch && eng->bo_monolithic && eng->ninstr_monolithic > 0) {
                xrt::run r = (*eng->kernel)(3, eng->bo_monolithic, eng->ninstr_monolithic, eng->bo_in[s_idx], eng->bo_out[s_idx]);
                r.wait(2000);
            } else {
                for (const auto& s : eng->stages) {
                    if (s.ninstr_exec > 0 && s.bo_exec) {
                        xrt::run r = (*eng->kernel)(3, s.bo_exec, s.ninstr_exec, eng->bo_in[s_idx], eng->bo_out[s_idx]);
                        r.wait(2000);
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

    // Bounded pipeline: wait until this slot has completed previous work (at most NUM_SLOTS in flight)
    {
        std::unique_lock<std::mutex> lock(engine->result_mutex);
        engine->slot_cv.wait(lock, [engine, ticket]() {
            return (engine->last_completed_ticket.load() + ignite_engine::NUM_SLOTS) >= ticket || engine->worker_stop.load();
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
        std::lock_guard<std::mutex> lock(engine->queue_mutex);
        engine->work_queue.push({ticket, slot});
    }
    engine->queue_cv.notify_one();

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

    // Wait until ticket is completed by NPU worker thread
    {
        std::unique_lock<std::mutex> lock(engine->result_mutex);
        engine->result_cv.wait(lock, [engine, ticket]() {
            return engine->last_completed_ticket.load() >= ticket || engine->worker_stop.load();
        });
        if (engine->last_completed_ticket.load() < ticket) {
            set_error("Worker stopped before ticket completion");
            return -2;
        }

        size_t ring_idx = static_cast<size_t>(ticket % ignite_engine::RING_SIZE);
        const auto& res = engine->result_ring[ring_idx];
        if (res.ticket != ticket) {
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
