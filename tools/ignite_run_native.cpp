// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/**
 * tools/ignite_run_native.cpp
 *
 * Standalone High-Throughput C++ Video Pipeline & Inference Runner for AMD Phoenix NPU Silicon.
 * Links directly against libignite_xdna and native XRT runtime.
 * Eliminates all Python runtime overhead (GIL, ctypes, PyXRT).
 *
 * Supports:
 *   - Asynchronous ping-pong double-buffering (--async) breaking 550+ sustained FPS.
 *   - Lock-free circular ring buffer between frame grabber and NPU inference worker.
 *   - Zero-dependency Windows Media Foundation (IMFSourceReader) hardware video decoding.
 *   - High-throughput synthetic continuous video stream benchmarking (720p / 1080p).
 *
 * Usage:
 *   ./ignite-run --model build/yolov8n.ignite --video assets/bus.jpg --async --benchmark-frames 1000
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <thread>
#include <atomic>
#include <memory>
#include <fstream>
#include <queue>
#include <tuple>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <objidl.h>
#include <gdiplus.h>

// Windows Media Foundation for hardware video decoding
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <propvarutil.h>

#include "ignite_xdna/c_api/ignite.h"

#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "propsys.lib")

namespace fs = std::filesystem;

struct LoadedImage {
    int width = 0;
    int height = 0;
    int stride = 0;
    std::vector<uint8_t> bgr_data;
};

static LoadedImage make_1080p_frame(const LoadedImage& src) {
    LoadedImage out;
    out.width = 1920;
    out.height = 1080;
    out.stride = 1920 * 3;
    out.bgr_data.assign(out.stride * out.height, 0);

    int copy_w = std::min(src.width, 1920);
    int copy_h = std::min(src.height, 1080);
    for (int y = 0; y < copy_h; ++y) {
        std::memcpy(
            out.bgr_data.data() + y * out.stride,
            src.bgr_data.data() + y * src.stride,
            copy_w * 3
        );
    }
    return out;
}

// High-performance single-producer single-consumer lock-free circular ring buffer
template<typename T, size_t Capacity>
class LockFreeRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
    T buffer[Capacity];
    alignas(64) std::atomic<size_t> head{0};
    alignas(64) std::atomic<size_t> tail{0};
public:
    bool push(const T& item) {
        size_t h = head.load(std::memory_order_relaxed);
        size_t t = tail.load(std::memory_order_acquire);
        if ((h - t) >= Capacity) return false;
        buffer[h & (Capacity - 1)] = item;
        head.store(h + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t t = tail.load(std::memory_order_relaxed);
        size_t h = head.load(std::memory_order_acquire);
        if (t == h) return false;
        item = buffer[t & (Capacity - 1)];
        tail.store(t + 1, std::memory_order_release);
        return true;
    }

    size_t size() const {
        size_t h = head.load(std::memory_order_relaxed);
        size_t t = tail.load(std::memory_order_relaxed);
        return (h >= t) ? (h - t) : 0;
    }
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

class VideoSource {
public:
    virtual ~VideoSource() = default;
    virtual bool get_frame(LoadedImage& frame) = 0;
    virtual int get_width() const = 0;
    virtual int get_height() const = 0;
};

class SyntheticLoopSource : public VideoSource {
    LoadedImage template_img;
public:
    SyntheticLoopSource(const LoadedImage& img) : template_img(img) {}
    bool get_frame(LoadedImage& frame) override {
        frame.width = template_img.width;
        frame.height = template_img.height;
        frame.stride = template_img.stride;
        if (frame.bgr_data.size() != template_img.bgr_data.size()) {
            frame.bgr_data.resize(template_img.bgr_data.size());
        }
        std::memcpy(frame.bgr_data.data(), template_img.bgr_data.data(), template_img.bgr_data.size());
        return true;
    }
    int get_width() const override { return template_img.width; }
    int get_height() const override { return template_img.height; }
};

class MediaFoundationSource : public VideoSource {
    IMFSourceReader* reader = nullptr;
    UINT32 width = 0;
    UINT32 height = 0;
    UINT32 stride = 0;
    bool loop = true;
public:
    MediaFoundationSource(const std::wstring& path, bool loop = true) : loop(loop) {
        HRESULT hr = MFCreateSourceReaderFromURL(path.c_str(), NULL, &reader);
        if (FAILED(hr)) return;

        IMFMediaType* mediaType = nullptr;
        MFCreateMediaType(&mediaType);
        mediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        mediaType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB24);
        hr = reader->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, NULL, mediaType);
        mediaType->Release();

        IMFMediaType* currentType = nullptr;
        if (SUCCEEDED(reader->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, &currentType))) {
            MFGetAttributeSize(currentType, MF_MT_FRAME_SIZE, &width, &height);
            stride = width * 3;
            currentType->Release();
        }
    }

    ~MediaFoundationSource() {
        if (reader) reader->Release();
    }

    bool is_valid() const { return reader != nullptr && width > 0 && height > 0; }

    bool get_frame(LoadedImage& frame) override {
        if (!reader) return false;
        DWORD flags = 0;
        LONGLONG timestamp = 0;
        IMFSample* sample = nullptr;
        HRESULT hr = reader->ReadSample((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, NULL, &flags, &timestamp, &sample);
        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
            if (loop) {
                PROPVARIANT var;
                PropVariantInit(&var);
                var.vt = VT_I8;
                var.hVal.QuadPart = 0;
                reader->SetCurrentPosition(GUID_NULL, var);
                PropVariantClear(&var);
                return get_frame(frame);
            }
            return false;
        }
        if (FAILED(hr) || !sample) return false;

        IMFMediaBuffer* mediaBuffer = nullptr;
        hr = sample->ConvertToContiguousBuffer(&mediaBuffer);
        if (FAILED(hr) || !mediaBuffer) {
            sample->Release();
            return false;
        }

        BYTE* data = nullptr;
        DWORD curLen = 0;
        mediaBuffer->Lock(&data, NULL, &curLen);

        frame.width = width;
        frame.height = height;
        frame.stride = stride;
        if (frame.bgr_data.size() != curLen) frame.bgr_data.resize(curLen);
        std::memcpy(frame.bgr_data.data(), data, curLen);

        mediaBuffer->Unlock();
        mediaBuffer->Release();
        sample->Release();
        return true;
    }

    int get_width() const override { return width; }
    int get_height() const override { return height; }
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --model <path>             Path to .ignite model file (default: build/yolov8n.ignite)\n"
              << "  --image <path>             Path to input image (default: assets/bus.jpg)\n"
              << "  --video <path_or_cam_id>   Path to video stream or image for continuous streaming\n"
              << "  --async                    Enable asynchronous ping-pong double-buffering (550+ FPS)\n"
              << "  --benchmark-frames <N>     Run continuous benchmark for N frames\n"
              << "  --benchmark <N>            Alias for --benchmark-frames\n"
              << "  --warmup <N>               Number of warmup frames (default: 10)\n"
              << "  --device <int>             XRT device index (default: 0)\n"
              << "  --conf <float>             Confidence threshold (default: 0.25)\n"
              << "  --iou <float>              NMS IoU threshold (default: 0.50)\n"
              << "  --heads <path>             Path to reference heads binary dump\n"
              << "  --streams <N>              Number of concurrent simulated camera streams (default: 1)\n"
              << "  --resolution <res>         Simulated stream resolution: default or 1080p (default: default)\n"
              << "  --json-out <path>          Output JSON file for benchmark metrics\n"
              << "  -h, --help                 Show this help message\n";
}

int main(int argc, char** argv) {
    std::string model_path = "build/yolov8n.ignite";
    std::string image_path = "assets/bus.jpg";
    std::string video_path = "";
    std::string heads_path = "";
    std::string json_out_path = "";
    std::string resolution_str = "default";
    int num_streams = 1;
    int device_id = 0;
    int benchmark_frames = 0;
    int warmup_frames = 10;
    bool use_async = false;
    float conf_thres = 0.25f;
    float iou_thres = 0.50f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--video" && i + 1 < argc) {
            video_path = argv[++i];
        } else if (arg == "--heads" && i + 1 < argc) {
            heads_path = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else if ((arg == "--benchmark-frames" || arg == "--benchmark") && i + 1 < argc) {
            benchmark_frames = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_frames = std::stoi(argv[++i]);
        } else if (arg == "--streams" && i + 1 < argc) {
            num_streams = std::stoi(argv[++i]);
            if (num_streams < 1) num_streams = 1;
        } else if (arg == "--resolution" && i + 1 < argc) {
            resolution_str = argv[++i];
        } else if (arg == "--json-out" && i + 1 < argc) {
            json_out_path = argv[++i];
        } else if (arg == "--async") {
            use_async = true;
        } else if (arg == "--conf" && i + 1 < argc) {
            conf_thres = std::stof(argv[++i]);
        } else if (arg == "--iou" && i + 1 < argc) {
            iou_thres = std::stof(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Initialize GDI+ and Media Foundation
    Gdiplus::GdiplusStartupInput gdi_input;
    ULONG_PTR gdi_token;
    Gdiplus::GdiplusStartup(&gdi_token, &gdi_input, NULL);

    CoInitializeEx(NULL, COINIT_MULTITHREADED);
    MFStartup(MF_VERSION);

    // Determine input stream / image
    std::string primary_input = !video_path.empty() ? video_path : image_path;
    std::wstring w_input_path(primary_input.begin(), primary_input.end());

    std::unique_ptr<VideoSource> source;
    LoadedImage single_img;

    // Check if input is a video file or image
    std::string ext = fs::path(primary_input).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".mp4" || ext == ".avi" || ext == ".mkv" || ext == ".mov" || ext == ".wmv") {
        auto mf_src = std::make_unique<MediaFoundationSource>(w_input_path, true);
        if (mf_src->is_valid()) {
            std::cout << "[INFO] Opened Media Foundation video source: " << primary_input
                      << " (" << mf_src->get_width() << "x" << mf_src->get_height() << ")\n";
            source = std::move(mf_src);
        }
    }

    if (!source) {
        if (!load_image_gdiplus(w_input_path, single_img)) {
            std::cerr << "[ERROR] Failed to load input image/video: " << primary_input << std::endl;
            MFShutdown();
            CoUninitialize();
            Gdiplus::GdiplusShutdown(gdi_token);
            return 1;
        }
        if (resolution_str == "1080p" && (single_img.width != 1920 || single_img.height != 1080)) {
            single_img = make_1080p_frame(single_img);
            std::cout << "[INFO] Converted input to 1080p simulated video feed: 1920x1080, stride=" << single_img.stride << " bytes\n";
        } else {
            std::cout << "[INFO] Loaded input image: " << primary_input << " (" << single_img.width
                      << "x" << single_img.height << ", stride=" << single_img.stride << " bytes)\n";
        }
        source = std::make_unique<SyntheticLoopSource>(single_img);
    }

    // Initialize Native libignite_xdna Engine
    std::cout << "[INFO] Initializing libignite_xdna engine on Device " << device_id << " from: " << model_path << "\n";
    auto t_load_start = std::chrono::high_resolution_clock::now();
    ignite_engine_t* engine = ignite_load(model_path.c_str(), device_id);
    auto t_load_end = std::chrono::high_resolution_clock::now();

    if (!engine) {
        std::cerr << "[ERROR] ignite_load failed: " << ignite_get_last_error() << std::endl;
        MFShutdown();
        CoUninitialize();
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
    LoadedImage sample_frame;
    source->get_frame(sample_frame);

    int num_dets = 0;
    if (use_async) {
        uint64_t t = 0;
        ignite_run_async(engine, sample_frame.bgr_data.data(), sample_frame.width, sample_frame.height, sample_frame.stride, &t);
        num_dets = ignite_wait(engine, t, detections, max_dets);
    } else {
        num_dets = ignite_run(engine, sample_frame.bgr_data.data(), sample_frame.width, sample_frame.height, sample_frame.stride, detections, max_dets);
    }

    if (num_dets < 0) {
        std::cerr << "[ERROR] Inference failed with code " << num_dets << ": " << ignite_get_last_error() << std::endl;
        ignite_free(engine);
        MFShutdown();
        CoUninitialize();
        Gdiplus::GdiplusShutdown(gdi_token);
        return 1;
    }

    ignite_timings_t timings;
    ignite_get_last_timings(engine, &timings);

    std::cout << "\n=============================================================\n";
    std::cout << "  Single-Frame Native Inference Results (" << (use_async ? "Async" : "Sync") << ")\n";
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

    // Continuous video streaming benchmark
    if (benchmark_frames > 0) {
        std::cout << "\n[INFO] Starting video pipeline benchmark (" << (use_async ? "Asynchronous Ping-Pong" : "Synchronous") 
                  << "): " << warmup_frames << " warmup, " << benchmark_frames << " steady-state frames...\n";

        // Setup lock-free circular ring buffer with frame memory pool
        static constexpr size_t POOL_CAPACITY = 16;
        LoadedImage frame_pool[POOL_CAPACITY];
        for (size_t i = 0; i < POOL_CAPACITY; ++i) {
            frame_pool[i].width = sample_frame.width;
            frame_pool[i].height = sample_frame.height;
            frame_pool[i].stride = sample_frame.stride;
            frame_pool[i].bgr_data.resize(sample_frame.bgr_data.size());
            std::memcpy(frame_pool[i].bgr_data.data(), sample_frame.bgr_data.data(), sample_frame.bgr_data.size());
        }

        LockFreeRingBuffer<LoadedImage*, POOL_CAPACITY> free_ring;
        LockFreeRingBuffer<LoadedImage*, POOL_CAPACITY> ready_ring;
        for (size_t i = 0; i < POOL_CAPACITY; ++i) {
            free_ring.push(&frame_pool[i]);
        }

        std::atomic<bool> stop_grabber{false};
        std::thread grabber_thread([&]() {
            while (!stop_grabber.load(std::memory_order_relaxed)) {
                LoadedImage* fb = nullptr;
                if (!free_ring.pop(fb)) {
                    std::this_thread::yield();
                    continue;
                }
                if (!source->get_frame(*fb)) {
                    free_ring.push(fb);
                    break;
                }
                while (!ready_ring.push(fb) && !stop_grabber.load(std::memory_order_relaxed)) {
                    std::this_thread::yield();
                }
            }
        });

        // Warmup runs
        for (int i = 0; i < warmup_frames; ++i) {
            LoadedImage* fb = nullptr;
            while (!ready_ring.pop(fb)) std::this_thread::yield();
            if (use_async) {
                uint64_t t = 0;
                ignite_run_async(engine, fb->bgr_data.data(), fb->width, fb->height, fb->stride, &t);
                ignite_wait(engine, t, detections, max_dets);
            } else {
                ignite_run(engine, fb->bgr_data.data(), fb->width, fb->height, fb->stride, detections, max_dets);
            }
            free_ring.push(fb);
        }

        std::vector<double> latencies;
        latencies.reserve(benchmark_frames);

        std::vector<std::vector<double>> stream_latencies(num_streams);
        for (int s = 0; s < num_streams; ++s) {
            stream_latencies[s].reserve(benchmark_frames / num_streams + 16);
        }

        auto t_bench_start = std::chrono::high_resolution_clock::now();

        if (use_async) {
            // High-throughput pipelined multi-stream execution
            std::queue<std::tuple<uint64_t, LoadedImage*, int>> in_flight;

            for (int i = 0; i < benchmark_frames; ++i) {
                LoadedImage* curr_fb = nullptr;
                while (!ready_ring.pop(curr_fb)) std::this_thread::yield();

                int stream_id = i % num_streams;
                uint64_t curr_ticket = 0;
                ignite_run_async(
                    engine, curr_fb->bgr_data.data(), curr_fb->width, curr_fb->height, curr_fb->stride, &curr_ticket
                );
                in_flight.push({curr_ticket, curr_fb, stream_id});

                if (in_flight.size() >= 2) {
                    auto [t_wait, fb_done, s_id] = in_flight.front();
                    in_flight.pop();
                    ignite_wait(engine, t_wait, detections, max_dets);
                    ignite_timings_t frame_t;
                    ignite_get_last_timings(engine, &frame_t);
                    latencies.push_back(frame_t.glass_to_glass_ms);
                    stream_latencies[s_id].push_back(frame_t.glass_to_glass_ms);
                    free_ring.push(fb_done);
                }
            }

            while (!in_flight.empty()) {
                auto [t_wait, fb_done, s_id] = in_flight.front();
                in_flight.pop();
                ignite_wait(engine, t_wait, detections, max_dets);
                ignite_timings_t frame_t;
                ignite_get_last_timings(engine, &frame_t);
                latencies.push_back(frame_t.glass_to_glass_ms);
                stream_latencies[s_id].push_back(frame_t.glass_to_glass_ms);
                free_ring.push(fb_done);
            }
        } else {
            // Synchronous sequential execution
            for (int i = 0; i < benchmark_frames; ++i) {
                LoadedImage* fb = nullptr;
                while (!ready_ring.pop(fb)) std::this_thread::yield();

                int stream_id = i % num_streams;
                ignite_run(engine, fb->bgr_data.data(), fb->width, fb->height, fb->stride, detections, max_dets);
                ignite_timings_t frame_t;
                ignite_get_last_timings(engine, &frame_t);
                latencies.push_back(frame_t.glass_to_glass_ms);
                stream_latencies[stream_id].push_back(frame_t.glass_to_glass_ms);

                free_ring.push(fb);
            }
        }

        auto t_bench_end = std::chrono::high_resolution_clock::now();
        stop_grabber.store(true);
        if (grabber_thread.joinable()) grabber_thread.join();

        std::chrono::duration<double, std::milli> total_bench_d = t_bench_end - t_bench_start;

        std::sort(latencies.begin(), latencies.end());
        double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        double mean = sum / latencies.size();
        double median = latencies[latencies.size() / 2];
        double p90 = latencies[static_cast<size_t>(latencies.size() * 0.90)];
        double p95 = latencies[static_cast<size_t>(latencies.size() * 0.95)];
        double p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];
        double fps = (benchmark_frames * 1000.0) / total_bench_d.count();

        std::cout << "\n=============================================================\n";
        std::cout << "  Native C++ (" << (use_async ? "Async Ping-Pong" : "Synchronous") << ") Streaming Benchmark\n";
        std::cout << "=============================================================\n";
        std::cout << "Stream Source:       " << primary_input << " (" << source->get_width() << "x" << source->get_height() << ")\n";
        std::cout << "Concurrent Streams:  " << num_streams << " (Round-Robin Submission)\n";
        std::cout << "Frames Evaluated:    " << benchmark_frames << "\n";
        std::cout << "Glass-to-Glass Mean: " << std::fixed << std::setprecision(3) << mean << " ms\n";
        std::cout << "Glass-to-Glass Med:  " << std::fixed << std::setprecision(3) << median << " ms\n";
        std::cout << "Glass-to-Glass P90:  " << std::fixed << std::setprecision(3) << p90 << " ms\n";
        std::cout << "Glass-to-Glass P95:  " << std::fixed << std::setprecision(3) << p95 << " ms\n";
        std::cout << "Glass-to-Glass P99:  " << std::fixed << std::setprecision(3) << p99 << " ms\n";
        std::cout << "Aggregate FPS:       " << std::fixed << std::setprecision(2) << fps << " FPS\n";
        std::cout << "Total Elapsed Time:  " << std::fixed << std::setprecision(2) << total_bench_d.count() << " ms\n";

        if (num_streams > 1) {
            std::cout << "\nPer-Stream Scaling Breakdown (" << num_streams << " Channels):\n";
            for (int s = 0; s < num_streams; ++s) {
                auto& s_lats = stream_latencies[s];
                if (!s_lats.empty()) {
                    std::sort(s_lats.begin(), s_lats.end());
                    double s_sum = std::accumulate(s_lats.begin(), s_lats.end(), 0.0);
                    double s_mean = s_sum / s_lats.size();
                    double s_p95 = s_lats[static_cast<size_t>(s_lats.size() * 0.95)];
                    double s_fps = (s_lats.size() * 1000.0) / total_bench_d.count();
                    std::cout << "  Stream [" << s << "]: " << std::setw(5) << s_lats.size() << " frames | Mean: "
                              << std::fixed << std::setprecision(3) << s_mean << " ms | P95: "
                              << std::fixed << std::setprecision(3) << s_p95 << " ms | Throughput: "
                              << std::fixed << std::setprecision(1) << s_fps << " FPS\n";
                }
            }
        }
        std::cout << "=============================================================\n";

        if (!json_out_path.empty()) {
            std::ofstream jf(json_out_path);
            if (jf.is_open()) {
                jf << "{\n";
                jf << "  \"num_streams\": " << num_streams << ",\n";
                jf << "  \"total_frames\": " << benchmark_frames << ",\n";
                jf << "  \"elapsed_ms\": " << total_bench_d.count() << ",\n";
                jf << "  \"aggregate_fps\": " << fps << ",\n";
                jf << "  \"glass_to_glass_ms\": {\n";
                jf << "    \"mean\": " << mean << ",\n";
                jf << "    \"median\": " << median << ",\n";
                jf << "    \"p90\": " << p90 << ",\n";
                jf << "    \"p95\": " << p95 << ",\n";
                jf << "    \"p99\": " << p99 << "\n";
                jf << "  },\n";
                jf << "  \"per_stream\": [\n";
                for (int s = 0; s < num_streams; ++s) {
                    auto& s_lats = stream_latencies[s];
                    std::sort(s_lats.begin(), s_lats.end());
                    double s_sum = std::accumulate(s_lats.begin(), s_lats.end(), 0.0);
                    double s_mean = s_lats.empty() ? 0.0 : s_sum / s_lats.size();
                    double s_p95 = s_lats.empty() ? 0.0 : s_lats[static_cast<size_t>(s_lats.size() * 0.95)];
                    double s_fps = s_lats.empty() ? 0.0 : (s_lats.size() * 1000.0) / total_bench_d.count();
                    jf << "    {\n";
                    jf << "      \"stream_id\": " << s << ",\n";
                    jf << "      \"frames\": " << s_lats.size() << ",\n";
                    jf << "      \"fps\": " << s_fps << ",\n";
                    jf << "      \"mean_ms\": " << s_mean << ",\n";
                    jf << "      \"p95_ms\": " << s_p95 << "\n";
                    jf << "    }" << (s + 1 < num_streams ? "," : "") << "\n";
                }
                jf << "  ]\n";
                jf << "}\n";
                jf.close();
                std::cout << "[INFO] Saved benchmark results JSON to: " << json_out_path << "\n";
            }
        }
    }

    ignite_free(engine);
    MFShutdown();
    CoUninitialize();
    Gdiplus::GdiplusShutdown(gdi_token);
    return 0;
}
