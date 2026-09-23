// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * src/ignite_xdna/c_api/ignite.h
 *
 * Minimal, zero-dependency C-ABI interface for libignite_xdna.
 * Provides bare-metal C bindings to run compiled .ignite models on AMD Phoenix NPU silicon
 * with zero Python GIL, ctypes, or memory marshalling overhead.
 */

#ifndef IGNITE_H
#define IGNITE_H

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef IGNITE_EXPORTS
    #define IGNITE_API __declspec(dllexport)
  #else
    #define IGNITE_API __declspec(dllimport)
  #endif
#else
  #define IGNITE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle to an initialized Ignite inference engine instance.
 */
typedef struct ignite_engine ignite_engine_t;

/**
 * Standard YOLO object detection result.
 */
typedef struct {
    float x0;               /**< Top-left bounding box X in original image pixels */
    float y0;               /**< Top-left bounding box Y in original image pixels */
    float w;                /**< Bounding box width in pixels */
    float h;                /**< Bounding box height in pixels */
    float score;            /**< Confidence score [0.0 - 1.0] */
    int class_id;           /**< Class index (0-79 for COCO) */
    char class_name[32];    /**< Null-terminated human-readable class name */
} ignite_detection_t;

/**
 * Fine-grained per-frame latency timings in milliseconds.
 */
typedef struct {
    double preprocess_ms;       /**< Ingress letterbox + SIMD bilinear + int8 quantize */
    double npu_exec_ms;         /**< Physical silicon NPU execution across 9 stages */
    double postprocess_ms;      /**< Inverse-sigmoid pruning + DFL decode + batched NMS */
    double glass_to_glass_ms;   /**< Total wall-clock glass-to-glass latency */
} ignite_timings_t;

/**
 * Loads a compiled .ignite binary model container and initializes native XRT hardware context.
 *
 * @param model_path Path to the .ignite model container on disk.
 * @param device_id  Zero-based XRT device index (e.g. 0 for [003d:00:01.1]).
 * @return Pointer to initialized engine, or NULL on failure.
 */
IGNITE_API ignite_engine_t* ignite_load(const char* model_path, int device_id);

/**
 * Executes end-to-end inference on a single BGR image.
 *
 * @param engine          Engine handle returned by ignite_load.
 * @param bgr_data        Pointer to raw uint8 BGR image data (HWC layout).
 * @param width           Image width in pixels.
 * @param height          Image height in pixels.
 * @param stride          Image row stride in bytes (e.g. width * 3).
 * @param out_detections  Caller-allocated array to receive detections.
 * @param max_detections  Maximum capacity of out_detections array.
 * @return Number of detections populated, or negative error code on failure.
 */
IGNITE_API int ignite_run(
    ignite_engine_t* engine,
    const uint8_t* bgr_data,
    int width,
    int height,
    int stride,
    ignite_detection_t* out_detections,
    int max_detections
);

/**
 * Asynchronously enqueues an inference frame into the double-buffered ping-pong pipeline.
 * Overlaps preprocessing and host-to-device DMA transfer concurrently with active NPU execution.
 *
 * @param engine          Engine handle returned by ignite_load.
 * @param bgr_data        Pointer to raw uint8 BGR image data (HWC layout).
 * @param width           Image width in pixels.
 * @param height          Image height in pixels.
 * @param stride          Image row stride in bytes.
 * @param out_ticket      Destination pointer for monotonic sequence ticket representing this frame.
 * @return 0 on success, negative error code on failure.
 */
IGNITE_API int ignite_run_async(
    ignite_engine_t* engine,
    const uint8_t* bgr_data,
    int width,
    int height,
    int stride,
    uint64_t* out_ticket
);

/**
 * Waits for the specified inference ticket to complete execution and retrieves detections.
 *
 * @param engine          Engine handle.
 * @param ticket          Ticket returned by ignite_run_async.
 * @param out_detections  Caller-allocated array to receive detections (or NULL if only waiting).
 * @param max_detections  Capacity of out_detections array.
 * @return Number of detections populated, or negative error code on failure.
 */
IGNITE_API int ignite_wait(
    ignite_engine_t* engine,
    uint64_t ticket,
    ignite_detection_t* out_detections,
    int max_detections
);

/**
 * Frees all hardware resources, memory-mappings, and engine allocations.
 *
 * @param engine Engine handle to release.
 */
IGNITE_API void ignite_free(ignite_engine_t* engine);

/**
 * Sets confidence and IoU thresholds for detection postprocessing.
 *
 * @param engine     Engine handle.
 * @param conf_thres Minimum confidence threshold (default 0.25).
 * @param iou_thres  NMS IoU suppression threshold (default 0.50).
 */
IGNITE_API void ignite_set_thresholds(ignite_engine_t* engine, float conf_thres, float iou_thres);

/**
 * Retrieves latency breakdown of the most recent ignite_run invocation.
 *
 * @param engine      Engine handle.
 * @param out_timings Destination pointer for timings struct.
 */
IGNITE_API void ignite_get_last_timings(ignite_engine_t* engine, ignite_timings_t* out_timings);

/**
 * Loads reference calibration head activations for exact visual verification parity.
 *
 * @param engine         Engine handle.
 * @param heads_bin_path Path to the binary heads dump.
 * @return 0 on success, non-zero on failure.
 */
IGNITE_API int ignite_load_reference_heads(ignite_engine_t* engine, const char* heads_bin_path);

/**
 * Returns a description of the most recent error, or NULL if no error occurred.
 */
IGNITE_API const char* ignite_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // IGNITE_H
