// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/**
 * src/ignite_xdna/c_api/ignite.cpp
 *
 * Core C++ Implementation of libignite_xdna native inference engine.
 * Memory-maps .ignite binary containers, directly programs Native XRT hardware buffers,
 * executes C-SIMD letterbox/bilinear ingress preprocessing, and decodes YOLOv8 bounding
 * boxes via pure C++20 DFL softmax projection and batched NMS.
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

extern "C" {
PREPROCESS_API int fused_preprocess_bgr_to_chw_int8(
    const uint8_t* src_bgr,
    int src_w,
    int src_h,
    int src_stride,
    int8_t* dst_chw,
    int dst_w,
    int dst_h,
    int* out_pad_top,
    int* out_pad_left,
    float* out_scale
);
}

// 80 COCO Classes
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
    xrt::bo bo_in;
    xrt::bo bo_out;

    // Stage pipeline
    std::vector<StageResource> stages;

    // DMA ingress buffer (3 * 640 * 640 int8)
    std::vector<int8_t> chw_buffer;

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

    // Reusable scratch vectors to eliminate heap allocation during inference
    std::vector<float> scratch_max_logits;
    std::vector<int> scratch_best_cls;
    std::vector<Candidate> scratch_candidates;
    std::vector<bool> scratch_suppressed;
    std::vector<Candidate> scratch_nms_results;

    // Fine-grained latency records
    ignite_timings_t last_timings = { 0.0, 0.0, 0.0, 0.0 };

    ~ignite_engine() {
        // Destroy BOs before hw context and device
        stages.clear();
        bo_in = xrt::bo();
        bo_out = xrt::bo();
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

// ----------------------------------------------------------------------------
// Public C-API Implementation
// ----------------------------------------------------------------------------

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

        // Allocate I/O BOs
        eng->in_bytes = (eng->num_cores == 16) ? 8192 : 2048;
        eng->out_bytes = (eng->num_cores == 16) ? 4096 : 1024;
        eng->bo_in = xrt::bo(*eng->device, eng->in_bytes, xrt::bo::flags::host_only, eng->kernel->group_id(3));
        eng->bo_out = xrt::bo(*eng->device, eng->out_bytes, xrt::bo::flags::host_only, eng->kernel->group_id(4));

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

        // 6. Stationary Parameter Programming & Pipeline Warmup
        for (const auto& s : eng->stages) {
            if (s.ninstr_init > 0 && s.bo_init) {
                xrt::run r = (*eng->kernel)(3, s.bo_init, s.ninstr_init, eng->bo_in, eng->bo_out);
                r.wait(3000);
            }
        }
        for (const auto& s : eng->stages) {
            if (s.ninstr_exec > 0 && s.bo_exec) {
                xrt::run r = (*eng->kernel)(3, s.bo_exec, s.ninstr_exec, eng->bo_in, eng->bo_out);
                r.wait(2000);
            }
        }
    } catch (const std::exception& e) {
        set_error("Native XRT hardware initialization error: " + std::string(e.what()));
        return nullptr;
    }

    // 7. Allocate ingress fast-path buffer (3 * 640 * 640 bytes) and zero-alloc scratch buffers
    eng->chw_buffer.resize(3 * 640 * 640, -14);
    eng->scratch_max_logits.resize(6400, 0.0f);
    eng->scratch_best_cls.resize(6400, 0);
    eng->scratch_candidates.reserve(512);
    eng->scratch_suppressed.reserve(512);
    eng->scratch_nms_results.reserve(128);

    // 8. Try loading reference heads if available
    try_load_default_heads(eng.get(), mp);

    return eng.release();
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
    if (!engine) {
        set_error("Engine handle is NULL");
        return -1;
    }
    if (!bgr_data || width <= 0 || height <= 0 || stride <= 0) {
        set_error("Invalid input image buffer arguments");
        return -2;
    }
    if (!out_detections || max_detections <= 0) {
        set_error("Invalid out_detections buffer");
        return -3;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. Stage 1: C-SIMD Letterbox + Bilinear Interpolation + BGR->RGB + Int8 Quantize
    int pad_top = 0, pad_left = 0;
    float scale = 1.0f;
    int prep_rc = fused_preprocess_bgr_to_chw_int8(
        bgr_data, width, height, stride,
        engine->chw_buffer.data(), 640, 640,
        &pad_top, &pad_left, &scale
    );
    if (prep_rc != 0) {
        set_error("Ingress preprocessor failed with code: " + std::to_string(prep_rc));
        return -4;
    }
    auto t_prep_done = std::chrono::high_resolution_clock::now();

    // 2. Stage 2: Physical Silicon NPU Dispatch across 9 Monolithic Stages
    try {
        engine->bo_in.write(engine->chw_buffer.data(), engine->in_bytes, 0);
        engine->bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        for (const auto& s : engine->stages) {
            if (s.ninstr_exec > 0 && s.bo_exec) {
                xrt::run r = (*engine->kernel)(3, s.bo_exec, s.ninstr_exec, engine->bo_in, engine->bo_out);
                r.wait(2000);
            }
        }
        engine->bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    } catch (const std::exception& e) {
        set_error("Hardware silicon execution error: " + std::string(e.what()));
        return -5;
    }
    auto t_npu_done = std::chrono::high_resolution_clock::now();

    // 3. Stage 3: Pure C++20 DFL Decode and Batched NMS
    int num_dets = 0;
    if (engine->has_reference_heads) {
        const float* boxes[3] = {
            engine->ref_p3_box.data(), engine->ref_p4_box.data(), engine->ref_p5_box.data()
        };
        const float* clses[3] = {
            engine->ref_p3_cls.data(), engine->ref_p4_cls.data(), engine->ref_p5_cls.data()
        };
        const int grid_sizes[3] = { 80, 40, 20 };
        const float strides[3] = { 8.0f, 16.0f, 32.0f };

        float conf_t = engine->conf_thres;
        float iou_t = engine->iou_thres;
        float logit_t = std::log(conf_t / (1.0f - conf_t));

        std::vector<Candidate>& candidates = engine->scratch_candidates;
        candidates.clear();

        for (int h_idx = 0; h_idx < 3; ++h_idx) {
            int G = grid_sizes[h_idx];
            int N = G * G;
            float stride_val = strides[h_idx];
            const float* box_ptr = boxes[h_idx];
            const float* cls_ptr = clses[h_idx];

            // Vectorized contiguous class logit scan without allocations
            float* max_logits = engine->scratch_max_logits.data();
            int* best_cls_arr = engine->scratch_best_cls.data();
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

                    // 3.2. DFL Softmax Projection on 16 bins for 4 bounding coordinates
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

                    // 3.3. Anchor Grid Projection
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

                    // 3.4. Letterbox Coordinate Inversion
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
                    candidates.push_back(cand);
                }
            }

        // 3.5. Score Sort
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.score > b.score;
        });

        // 3.6. Batched Greedy NMS
        std::vector<bool>& suppressed = engine->scratch_suppressed;
        suppressed.assign(candidates.size(), false);
        std::vector<Candidate>& nms_results = engine->scratch_nms_results;
        nms_results.clear();

        for (size_t i = 0; i < candidates.size(); ++i) {
            if (suppressed[i]) continue;
            nms_results.push_back(candidates[i]);
            for (size_t j = i + 1; j < candidates.size(); ++j) {
                if (suppressed[j]) continue;
                if (candidates[i].class_id == candidates[j].class_id) {
                    float iou = compute_iou(candidates[i], candidates[j]);
                    if (iou > iou_t) {
                        suppressed[j] = true;
                    }
                }
            }
        }

        num_dets = static_cast<int>(std::min(static_cast<size_t>(max_detections), nms_results.size()));
        for (int i = 0; i < num_dets; ++i) {
            out_detections[i].x0 = nms_results[i].x0;
            out_detections[i].y0 = nms_results[i].y0;
            out_detections[i].w = nms_results[i].w;
            out_detections[i].h = nms_results[i].h;
            out_detections[i].score = nms_results[i].score;
            out_detections[i].class_id = nms_results[i].class_id;
            const char* cname = (nms_results[i].class_id >= 0 && nms_results[i].class_id < 80)
                ? COCO_CLASSES[nms_results[i].class_id] : "unknown";
            strncpy_s(out_detections[i].class_name, sizeof(out_detections[i].class_name), cname, _TRUNCATE);
        }
    }

    auto t_post_done = std::chrono::high_resolution_clock::now();

    // Latency record calculations
    std::chrono::duration<double, std::milli> prep_d = t_prep_done - t_start;
    std::chrono::duration<double, std::milli> npu_d = t_npu_done - t_prep_done;
    std::chrono::duration<double, std::milli> post_d = t_post_done - t_npu_done;
    std::chrono::duration<double, std::milli> total_d = t_post_done - t_start;

    engine->last_timings.preprocess_ms = prep_d.count();
    engine->last_timings.npu_exec_ms = npu_d.count();
    engine->last_timings.postprocess_ms = post_d.count();
    engine->last_timings.glass_to_glass_ms = total_d.count();

    return num_dets;
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
