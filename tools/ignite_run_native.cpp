// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/**
 * tools/ignite_run_native.cpp
 *
 * Standalone High-Performance C++ Inference Runner for AMD Phoenix NPU Silicon.
 * Links directly against libignite_xdna and native XRT runtime.
 * Eliminates all Python runtime overhead (GIL, ctypes, PyXRT).
 *
 * Usage:
 *   ./ignite-run --model build/yolov8n.ignite --image assets/bus.jpg --benchmark 1000
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <filesystem>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <objidl.h>
#include <gdiplus.h>

#include "ignite_xdna/c_api/ignite.h"

#pragma comment(lib, "gdiplus.lib")

namespace fs = std::filesystem;

struct LoadedImage {
    int width = 0;
    int height = 0;
    int stride = 0;
    std::vector<uint8_t> bgr_data;
};

static bool load_image_gdiplus(const std::wstring& path, LoadedImage& out) {
    Gdiplus::Bitmap bmp(path.c_str());
    if (bmp.GetLastStatus() != Gdiplus::Ok) {
        return false;
    }

    out.width = bmp.GetWidth();
    out.height = bmp.GetHeight();

    Gdiplus::Rect rect(0, 0, out.width, out.height);
    Gdiplus::BitmapData bmp_data;
    if (bmp.LockBits(&rect, Gdiplus::ImageLockModeRead, PixelFormat24bppRGB, &bmp_data) != Gdiplus::Ok) {
        return false;
    }

    out.stride = std::abs(bmp_data.Stride);
    size_t total_bytes = static_cast<size_t>(out.stride) * out.height;
    out.bgr_data.resize(total_bytes);
    std::memcpy(out.bgr_data.data(), bmp_data.Scan0, total_bytes);

    bmp.UnlockBits(&bmp_data);
    return true;
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --model <path>       Path to .ignite model file (default: build/yolov8n.ignite)\n"
              << "  --image <path>       Path to input image (default: assets/bus.jpg)\n"
              << "  --device <int>       XRT device index (default: 0)\n"
              << "  --benchmark <N>      Run benchmark for N steady-state iterations\n"
              << "  --warmup <N>         Number of warmup iterations (default: 10)\n"
              << "  --conf <float>       Confidence threshold (default: 0.25)\n"
              << "  --iou <float>        NMS IoU threshold (default: 0.50)\n"
              << "  --heads <path>       Path to reference heads binary dump\n"
              << "  -h, --help           Show this help message\n";
}

int main(int argc, char** argv) {
    std::string model_path = "build/yolov8n.ignite";
    std::string image_path = "assets/bus.jpg";
    std::string heads_path = "";
    int device_id = 0;
    int benchmark_runs = 0;
    int warmup_runs = 10;
    float conf_thres = 0.25f;
    float iou_thres = 0.50f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--heads" && i + 1 < argc) {
            heads_path = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else if (arg == "--benchmark" && i + 1 < argc) {
            benchmark_runs = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_runs = std::stoi(argv[++i]);
        } else if (arg == "--conf" && i + 1 < argc) {
            conf_thres = std::stof(argv[++i]);
        } else if (arg == "--iou" && i + 1 < argc) {
            iou_thres = std::stof(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Initialize GDI+ for native Windows image loading
    Gdiplus::GdiplusStartupInput gdi_input;
    ULONG_PTR gdi_token;
    Gdiplus::GdiplusStartup(&gdi_token, &gdi_input, NULL);

    LoadedImage img;
    std::wstring w_img_path(image_path.begin(), image_path.end());
    if (!load_image_gdiplus(w_img_path, img)) {
        std::cerr << "[ERROR] Failed to load input image: " << image_path << std::endl;
        Gdiplus::GdiplusShutdown(gdi_token);
        return 1;
    }
    std::cout << "[INFO] Loaded image: " << image_path << " (" << img.width << "x" << img.height 
              << ", stride=" << img.stride << " bytes)\n";

    // Initialize Native Engine
    std::cout << "[INFO] Initializing libignite_xdna engine on Device " << device_id << " from: " << model_path << "\n";
    auto t_load_start = std::chrono::high_resolution_clock::now();
    ignite_engine_t* engine = ignite_load(model_path.c_str(), device_id);
    auto t_load_end = std::chrono::high_resolution_clock::now();

    if (!engine) {
        std::cerr << "[ERROR] ignite_load failed: " << ignite_get_last_error() << std::endl;
        Gdiplus::GdiplusShutdown(gdi_token);
        return 1;
    }

    std::chrono::duration<double, std::milli> load_d = t_load_end - t_load_start;
    std::cout << "[INFO] Model loaded and stationary hardware initialized in " 
              << std::fixed << std::setprecision(2) << load_d.count() << " ms\n";

    ignite_set_thresholds(engine, conf_thres, iou_thres);

    if (!heads_path.empty()) {
        if (ignite_load_reference_heads(engine, heads_path.c_str()) == 0) {
            std::cout << "[INFO] Loaded reference head activations from: " << heads_path << "\n";
        } else {
            std::cerr << "[WARN] Failed to load reference heads: " << ignite_get_last_error() << "\n";
        }
    }

    // Single-frame execution and verification
    const int max_dets = 100;
    ignite_detection_t detections[max_dets];

    int num_dets = ignite_run(
        engine, img.bgr_data.data(), img.width, img.height, img.stride, detections, max_dets
    );

    if (num_dets < 0) {
        std::cerr << "[ERROR] ignite_run failed with code " << num_dets << ": " 
                  << ignite_get_last_error() << std::endl;
        ignite_free(engine);
        Gdiplus::GdiplusShutdown(gdi_token);
        return 1;
    }

    ignite_timings_t timings;
    ignite_get_last_timings(engine, &timings);

    std::cout << "\n=============================================================\n";
    std::cout << "  Single-Frame Native Inference Results\n";
    std::cout << "=============================================================\n";
    std::cout << "Detected Objects: " << num_dets << "\n";
    for (int i = 0; i < num_dets; ++i) {
        std::cout << "  [" << (i + 1) << "] " << std::left << std::setw(15) << detections[i].class_name
                  << " score=" << std::fixed << std::setprecision(3) << detections[i].score
                  << "  bbox=(" << std::fixed << std::setprecision(1) << detections[i].x0 << ", "
                  << detections[i].y0 << ", " << detections[i].w << ", " << detections[i].h << ")\n";
    }

    std::cout << "\nTiming Breakdown:\n";
    std::cout << "  Ingress SIMD Preprocess: " << std::fixed << std::setprecision(3) << timings.preprocess_ms << " ms\n";
    std::cout << "  Physical Silicon NPU:    " << std::fixed << std::setprecision(3) << timings.npu_exec_ms << " ms\n";
    std::cout << "  Pure C++20 DFL + NMS:    " << std::fixed << std::setprecision(3) << timings.postprocess_ms << " ms\n";
    std::cout << "  Glass-to-Glass Latency:  " << std::fixed << std::setprecision(3) << timings.glass_to_glass_ms << " ms\n";
    std::cout << "=============================================================\n";

    // Benchmark loop if requested
    if (benchmark_runs > 0) {
        std::cout << "\n[INFO] Starting benchmark: " << warmup_runs << " warmup runs, " 
                  << benchmark_runs << " steady-state runs...\n";

        for (int i = 0; i < warmup_runs; ++i) {
            ignite_run(engine, img.bgr_data.data(), img.width, img.height, img.stride, detections, max_dets);
        }

        std::vector<double> latencies;
        latencies.reserve(benchmark_runs);

        auto t_bench_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < benchmark_runs; ++i) {
            auto t_iter_start = std::chrono::high_resolution_clock::now();
            ignite_run(engine, img.bgr_data.data(), img.width, img.height, img.stride, detections, max_dets);
            auto t_iter_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> iter_d = t_iter_end - t_iter_start;
            latencies.push_back(iter_d.count());
        }
        auto t_bench_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_bench_d = t_bench_end - t_bench_start;

        std::sort(latencies.begin(), latencies.end());
        double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        double mean = sum / benchmark_runs;
        double median = latencies[benchmark_runs / 2];
        double p90 = latencies[static_cast<size_t>(benchmark_runs * 0.90)];
        double p95 = latencies[static_cast<size_t>(benchmark_runs * 0.95)];
        double p99 = latencies[static_cast<size_t>(benchmark_runs * 0.99)];
        double fps = (benchmark_runs * 1000.0) / total_bench_d.count();

        std::cout << "\n=============================================================\n";
        std::cout << "  Native C++ (libignite_xdna) Silicon Benchmark Summary\n";
        std::cout << "=============================================================\n";
        std::cout << "Iterations:          " << benchmark_runs << "\n";
        std::cout << "Mean Latency:        " << std::fixed << std::setprecision(3) << mean << " ms\n";
        std::cout << "Median Latency:      " << std::fixed << std::setprecision(3) << median << " ms\n";
        std::cout << "P90 Latency:         " << std::fixed << std::setprecision(3) << p90 << " ms\n";
        std::cout << "P95 Latency:         " << std::fixed << std::setprecision(3) << p95 << " ms\n";
        std::cout << "P99 Latency:         " << std::fixed << std::setprecision(3) << p99 << " ms\n";
        std::cout << "Throughput:          " << std::fixed << std::setprecision(2) << fps << " FPS\n";
        std::cout << "=============================================================\n";
    }

    ignite_free(engine);
    Gdiplus::GdiplusShutdown(gdi_token);
    return 0;
}
