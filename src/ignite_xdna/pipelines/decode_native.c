// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
/**
 * src/ignite_xdna/pipelines/decode_native.c
 *
 * YOLOv8 int8 head decode in one call: the per-anchor confidence prune, the DFL softmax
 * expectation, box reconstruction and letterbox removal, sigmoid class scores and batched NMS.
 * It returns the detections YoloDecoder.postprocess returns (numpy, then
 * cv2.dnn.NMSBoxesBatched) bit for bit:
 *   - float32 operations run in numpy's order; the 16 DFL bins accumulate in index order, and
 *     the SSE2 divisions and multiplications are the same IEEE operations element by element;
 *   - every exp is read from a table numpy fills with np.exp, which gives the same bits for an
 *     element whatever the array layout (checked on numpy 1.26 and 2.5);
 *   - a class score is the sigmoid table at the class maximum, and the class is the first one
 *     whose sigmoid equals it (numpy's argmax over float32 probabilities, saturation included);
 *   - NMS follows OpenCV 4.11 / 5.0 NMSBoxesBatched: class offsets and areas in double, the
 *     overlap cast to float, scores stably sorted in descending order, a strict score threshold.
 * Build it without /fp:fast, OpenMP or FMA contraction: each of them changes the bits. SSE2 is
 * the x64 baseline; other targets compile the scalar loops.
 */

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_M_X64) || defined(_M_AMD64) || defined(__SSE2__)
#include <emmintrin.h>
#define DECODE_SSE2 1
#else
#define DECODE_SSE2 0
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#define DECODE_API __declspec(dllexport)
#else
#define DECODE_API __attribute__((visibility("default")))
#endif

#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#elif defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

#define YOLO_DECODE_ABI 3
#define DFL_BINS 16
#define STACK_SURVIVORS 1024
#define STACK_CANDIDATES 256

typedef struct {
    const int8_t* box;          /* int8 [4 * reg_max][anchors] in NCHW order */
    const int8_t* cls;          /* int8 [num_classes][anchors] */
    const int8_t* cls_max;      /* int8 [anchors] per-anchor class maximum, or NULL to compute it here */
    const float* dfl_exp;       /* [256 * 256]: np.exp(v(q) - v(q_max)) at (q_max + 128) * 256 + (q + 128) */
    const float* box_val;       /* [256]: v(q) = (q - zp) * scale at q + 128; used only at reg_max 1 */
    const float* sigmoid;       /* [256]: numpy's float32 1 / (1 + exp(-v(q))) at q + 128 */
    const int32_t* sigmoid_low; /* [256]: smallest q with sigmoid[q] == sigmoid[q_max] at q_max + 128, or NULL */
    double q_threshold;         /* an anchor survives when its class maximum > q_threshold */
    int32_t anchors;            /* grid height * width */
    int32_t anchor_offset;      /* index of the head's first anchor in anchor_x / anchor_y / strides */
} yolo_head_t;

typedef struct {
    yolo_head_t head[3];
    const float* anchor_x;      /* [total anchors] */
    const float* anchor_y;
    const float* strides;
    int32_t reg_max;            /* 16 with DFL, 1 when the family regresses distances directly */
    int32_t num_classes;
    float conf;                 /* float32(conf_thres) */
    float iou;                  /* float32(iou_thres) */
} yolo_decode_t;

typedef struct {
    const uint8_t* box_c8;      /* uint8 [blocks][anchors][8] in channel-blocked order */
    const uint8_t* cls_c8;      /* uint8 [(num_classes+7)/8][anchors][8] in channel-blocked order */
    const int8_t* cls_max;      /* int8 [anchors] per-anchor class maximum */
    const float* dfl_exp;       /* [256 * 256]: np.exp(v(q) - v(q_max)) */
    const float* box_val;       /* [256]: v(q) = (q - zp) * scale at q + 128; used only at reg_max 1 */
    const float* sigmoid;       /* [256]: numpy's float32 1 / (1 + exp(-v(q))) at q + 128 */
    const int32_t* sigmoid_low; /* [256]: smallest q with sigmoid[q] == sigmoid[q_max] at q_max + 128, or NULL */
    double q_threshold;         /* an anchor survives when its class maximum > q_threshold */
    int32_t anchors;            /* grid height * width */
    int32_t anchor_offset;      /* index of the head's first anchor in anchor_x / anchor_y / strides */
} yolo_head_c8_t;

typedef struct {
    yolo_head_c8_t head[3];
    const float* anchor_x;      /* [total anchors] */
    const float* anchor_y;
    const float* strides;
    int32_t reg_max;            /* 16 with DFL, 1 when the family regresses distances directly */
    int32_t num_classes;
    float conf;                 /* float32(conf_thres) */
    float iou;                  /* float32(iou_thres) */
} yolo_decode_c8_t;

/* A candidate at or above the confidence threshold, in anchor order. */
typedef struct {
    float x0, y0, w, h;         /* box in source-image pixels, as YoloDetection holds it */
    float score;
    int32_t class_id;
    double ox, oy, ow, oh;      /* the class-offset Rect2d NMSBoxesBatched suppresses on */
} candidate_t;

DECODE_API int yolo_decode_abi(void)
{
    return YOLO_DECODE_ABI;
}

#if DECODE_SSE2
static int lowest_bit(unsigned int mask)
{
#if defined(_MSC_VER)
    unsigned long i;
    _BitScanForward(&i, mask);
    return (int)i;
#else
    return __builtin_ctz(mask);
#endif
}
#endif

static int class_max_at(const yolo_head_t* hd, int32_t num_classes, int32_t a)
{
    const size_t n = (size_t)hd->anchors;
    int best = hd->cls[a];
    for (int32_t c = 1; c < num_classes; ++c) {
        const int v = hd->cls[(size_t)c * n + (size_t)a];
        if (v > best) best = v;
    }
    return best;
}

/* For an integer m, m > qt exactly when m > floor(qt); clamped so that it compares with int8. */
static int int_threshold(double qt)
{
    if (!(qt < 127.0)) return 127;  /* NaN included: m > NaN is false for every m */
    if (qt < -129.0) return -129;
    return (int)floor(qt);
}

/* Writes surviving anchors from an int8 cls_max array in ascending order. */
static size_t collect_survivors_from_max(const int8_t* m, int32_t n, int qi, int32_t* out, size_t room)
{
    size_t count = 0;
    int32_t a = 0;
    if (qi >= 127 || !m) return 0;
#if DECODE_SSE2
    if (qi >= -128) {
        const __m128i t = _mm_set1_epi8((char)qi);
        for (; a + 16 <= n; a += 16) {
            unsigned int mask = (unsigned int)_mm_movemask_epi8(
                _mm_cmpgt_epi8(_mm_loadu_si128((const __m128i*)(m + a)), t));
            while (mask) {
                if (count < room) out[count] = a + lowest_bit(mask);
                ++count;
                mask &= mask - 1;
            }
        }
    }
#endif
    for (; a < n; ++a) {
        if ((int)m[a] > qi) {
            if (count < room) out[count] = a;
            ++count;
        }
    }
    return count;
}

/* Writes the head's surviving anchors in ascending order while `room` lasts; returns how many survive. */
static size_t collect_survivors(const yolo_head_t* hd, int32_t num_classes, int qi, int32_t* out, size_t room)
{
    if (qi >= 127) return 0;
    if (hd->cls_max) {
        return collect_survivors_from_max(hd->cls_max, hd->anchors, qi, out, room);
    }
    size_t count = 0;
    const int32_t n = hd->anchors;
    for (int32_t a = 0; a < n; ++a) {
        if (class_max_at(hd, num_classes, a) > qi) {
            if (count < room) out[count] = a;
            ++count;
        }
    }
    return count;
}

/* Area of Rect2d a & b, as OpenCV's operator&= computes it. */
static double intersection_area(const candidate_t* a, const candidate_t* b)
{
    if (a->ow <= 0 || a->oh <= 0 || b->ow <= 0 || b->oh <= 0) return 0.0;
    const candidate_t* rx_min = (a->ox < b->ox) ? a : b;
    const candidate_t* rx_max = (a->ox < b->ox) ? b : a;
    const candidate_t* ry_min = (a->oy < b->oy) ? a : b;
    const candidate_t* ry_max = (a->oy < b->oy) ? b : a;
    if ((rx_min->ox < 0 && rx_min->ox + rx_min->ow < rx_max->ox) ||
        (ry_min->oy < 0 && ry_min->oy + ry_min->oh < ry_max->oy)) {
        return 0.0;
    }
    const double wx = rx_min->ow - (rx_max->ox - rx_min->ox);
    const double w = (rx_max->ow < wx) ? rx_max->ow : wx;  /* std::min(wx, rx_max.width) */
    const double hy = ry_min->oh - (ry_max->oy - ry_min->oy);
    const double h = (ry_max->oh < hy) ? ry_max->oh : hy;
    if (w <= 0 || h <= 0) return 0.0;
    return w * h;
}

/* OpenCV's rectOverlap: 1.f - (float)jaccardDistance(a, b). */
static float rect_overlap(const candidate_t* a, const candidate_t* b)
{
    const double aa = a->ow * a->oh;
    const double ab = b->ow * b->oh;
    if ((aa + ab) <= DBL_EPSILON) return 1.0f - (float)0.0;
    const double inter = intersection_area(a, b);
    return 1.0f - (float)(1.0 - inter / (aa + ab - inter));
}

/* Stable sort of candidate indices by descending score (std::stable_sort with pair1.first > pair2.first). */
static void sort_by_score(int32_t* v, int32_t* tmp, size_t n, const candidate_t* c)
{
    int32_t* src = v;
    int32_t* dst = tmp;
    for (size_t width = 1; width < n; width *= 2) {
        for (size_t lo = 0; lo < n; lo += 2 * width) {
            const size_t mid = (lo + width < n) ? lo + width : n;
            const size_t hi = (lo + 2 * width < n) ? lo + 2 * width : n;
            size_t i = lo, j = mid, k = lo;
            while (i < mid && j < hi) dst[k++] = (c[src[j]].score > c[src[i]].score) ? src[j++] : src[i++];
            while (i < mid) dst[k++] = src[i++];
            while (j < hi) dst[k++] = src[j++];
        }
        int32_t* t = src;
        src = dst;
        dst = t;
    }
    if (src != v) {
        for (size_t i = 0; i < n; ++i) v[i] = src[i];
    }
}

/* No DFL (reg_max 1): the head's four channels ARE the four distances, so there is nothing to
   reduce and the "expectation" is the dequantized value itself. box_val is the same table the
   numpy path computes with, (q - zero_point) * scale in float32, so the two agree bit for bit. */
static float direct_side(const int8_t* box, size_t n, size_t a, int side, const float* box_val)
{
    return box_val[(int)box[(size_t)side * n + a] + 128];
}

/* No DFL, channel-blocked: the four distances are lanes 0..3 of block 0. Lanes 4..7 are the junk
   a fixed 8-channel block over-reads and must not be touched; the DFL routine below indexes
   blocks 2*side and 2*side+1 instead, which is the wrong place entirely at one bin per side. */
static float direct_side_c8(const uint8_t* box_c8, size_t n, size_t a, int side, const float* box_val)
{
    (void)n;
    const uint8_t* p = box_c8 + a * 8;
    return box_val[(int)(int8_t)(p[side] ^ 0x80) + 128];
}

/* DFL expectation of one side: softmax over the 16 bins from the exp table, then sum(p_k * k) in index order. */
static float dfl_side(const int8_t* box, size_t n, size_t a, int side, const float* dfl_exp)
{
    int q[DFL_BINS];
    int q_max = -128;
    for (int k = 0; k < DFL_BINS; ++k) {
        q[k] = box[(size_t)(side * DFL_BINS + k) * n + a];
        if (q[k] > q_max) q_max = q[k];
    }
    const float* row = dfl_exp + (size_t)(q_max + 128) * 256;
    float e[DFL_BINS];
    float w[DFL_BINS];
    for (int k = 0; k < DFL_BINS; ++k) e[k] = row[q[k] + 128];
    float sum = e[0];
    for (int k = 1; k < DFL_BINS; ++k) sum += e[k];
#if DECODE_SSE2
    const __m128 vs = _mm_set1_ps(sum);
    for (int k = 0; k < DFL_BINS; k += 4) {
        const __m128 bins = _mm_setr_ps((float)k, (float)(k + 1), (float)(k + 2), (float)(k + 3));
        _mm_storeu_ps(w + k, _mm_mul_ps(_mm_div_ps(_mm_loadu_ps(e + k), vs), bins));
    }
#else
    for (int k = 0; k < DFL_BINS; ++k) w[k] = (e[k] / sum) * (float)k;
#endif
    float acc = w[0];
    for (int k = 1; k < DFL_BINS; ++k) acc += w[k];
    return acc;
}

/* DFL expectation from channel-blocked layout: 16 bins for side come from 2 blocks of 8 channels. */
static float dfl_side_c8(const uint8_t* box_c8, size_t n, size_t a, int side, const float* dfl_exp)
{
    int q[DFL_BINS];
    int q_max = -128;
    const uint8_t* p0 = box_c8 + (size_t)(2 * side + 0) * (n * 8) + a * 8;
    const uint8_t* p1 = box_c8 + (size_t)(2 * side + 1) * (n * 8) + a * 8;

    for (int k = 0; k < 8; ++k) {
        int v = (int)(int8_t)(p0[k] ^ 0x80);
        q[k] = v;
        if (v > q_max) q_max = v;
    }
    for (int k = 0; k < 8; ++k) {
        int v = (int)(int8_t)(p1[k] ^ 0x80);
        q[8 + k] = v;
        if (v > q_max) q_max = v;
    }

    const float* row = dfl_exp + (size_t)(q_max + 128) * 256;
    float e[DFL_BINS];
    float w[DFL_BINS];
    for (int k = 0; k < DFL_BINS; ++k) e[k] = row[q[k] + 128];
    float sum = e[0];
    for (int k = 1; k < DFL_BINS; ++k) sum += e[k];
#if DECODE_SSE2
    const __m128 vs = _mm_set1_ps(sum);
    for (int k = 0; k < DFL_BINS; k += 4) {
        const __m128 bins = _mm_setr_ps((float)k, (float)(k + 1), (float)(k + 2), (float)(k + 3));
        _mm_storeu_ps(w + k, _mm_mul_ps(_mm_div_ps(_mm_loadu_ps(e + k), vs), bins));
    }
#else
    for (int k = 0; k < DFL_BINS; ++k) w[k] = (e[k] / sum) * (float)k;
#endif
    float acc = w[0];
    for (int k = 1; k < DFL_BINS; ++k) acc += w[k];
    return acc;
}

/* 3. NMSBoxesBatched + 4. NMSFast_: returns count on success, -3 if capacity exceeded. */
static int nms_and_output(
    candidate_t* cand,
    int32_t* index,
    size_t m,
    size_t total,
    float conf,
    float iou,
    int32_t capacity,
    float* out_box,
    float* out_score,
    int32_t* out_class
) {
    if (m == 0) return 0;

    /* 3. NMSBoxesBatched: offset each box by class_id * (max_coord + 1). */
    double max_coord = 0;
    for (size_t i = 0; i < m; ++i) {
        const double x1 = cand[i].x0;
        const double y1 = cand[i].y0;
        const double x2 = x1 + (double)cand[i].w;
        const double y2 = y1 + (double)cand[i].h;
        max_coord = (x1 < max_coord) ? max_coord : x1;  /* std::max(x1, max_coord) */
        max_coord = (y1 < max_coord) ? max_coord : y1;
        max_coord = (x2 < max_coord) ? max_coord : x2;
        max_coord = (y2 < max_coord) ? max_coord : y2;
    }
    for (size_t i = 0; i < m; ++i) {
        const double offset = (double)cand[i].class_id * (max_coord + 1);
        cand[i].ox = (double)cand[i].x0 + offset;
        cand[i].oy = (double)cand[i].y0 + offset;
        cand[i].ow = cand[i].w;
        cand[i].oh = cand[i].h;
    }

    /* 4. NMSFast_: scores strictly above the threshold, stably sorted, then greedy suppression. */
    size_t ranked = 0;
    for (size_t i = 0; i < m; ++i) {
        if (cand[i].score > conf) index[ranked++] = (int32_t)i;
    }
    sort_by_score(index, index + total, ranked, cand);
    int32_t* kept = index + total;
    size_t count = 0;
    for (size_t i = 0; i < ranked; ++i) {
        const candidate_t* a = &cand[index[i]];
        int keep = 1;
        for (size_t k = 0; k < count && keep; ++k) {
            const float overlap = rect_overlap(a, &cand[kept[k]]);
            keep = overlap <= iou;
        }
        if (keep) kept[count++] = index[i];
    }
    if (count > (size_t)capacity) {
        return -3;
    }
    for (size_t k = 0; k < count; ++k) {
        const candidate_t* c = &cand[kept[k]];
        out_box[4 * k] = c->x0;
        out_box[4 * k + 1] = c->y0;
        out_box[4 * k + 2] = c->w;
        out_box[4 * k + 3] = c->h;
        out_score[k] = c->score;
        out_class[k] = c->class_id;
    }
    return (int)count;
}

/**
 * Decodes the three int8 heads described by `d` for a frame letterboxed with (pad_top, pad_left)
 * and `scale`. Writes up to `capacity` detections in YoloDecoder.postprocess order: boxes as
 * x0, y0, w, h into out_box[4 * i], scores and class ids. Returns the detection count, -1 for bad
 * arguments, -2 when scratch memory cannot be allocated, -3 when capacity is too small.
 */
DECODE_API int yolo_decode_int8(
    const yolo_decode_t* d,
    float pad_top,
    float pad_left,
    float scale,
    int32_t capacity,
    float* out_box,
    float* out_score,
    int32_t* out_class
) {
    if (!d || !out_box || !out_score || !out_class || capacity < 0 || (d->reg_max != DFL_BINS && d->reg_max != 1) ||
        d->num_classes <= 0 || !d->anchor_x || !d->anchor_y || !d->strides) {
        return -1;
    }
    const int32_t nc = d->num_classes;
    int qi[3];
    for (int h = 0; h < 3; ++h) {
        const yolo_head_t* hd = &d->head[h];
        if (!hd->box || !hd->cls || !hd->dfl_exp || !hd->sigmoid || hd->anchors <= 0 || hd->anchor_offset < 0) {
            return -1;
        }
        qi[h] = int_threshold(hd->q_threshold);
    }

    /* 1. Surviving anchors per head, ascending, as the numpy prune's flatnonzero lists them. */
    int32_t stack_survivor[STACK_SURVIVORS];
    int32_t* survivor = stack_survivor;
    void* heap_survivor = NULL;
    size_t seg[4] = {0, 0, 0, 0};
    size_t total = 0;
    for (int h = 0; h < 3; ++h) {
        const size_t room = (total < STACK_SURVIVORS) ? STACK_SURVIVORS - total : 0;
        total += collect_survivors(&d->head[h], nc, qi[h], survivor + (total < STACK_SURVIVORS ? total : 0), room);
        seg[h + 1] = total;
    }
    if (total == 0) return 0;
    if (total > STACK_SURVIVORS) {
        heap_survivor = malloc(total * sizeof(int32_t));
        if (!heap_survivor) return -2;
        survivor = (int32_t*)heap_survivor;
        size_t again = 0;
        for (int h = 0; h < 3; ++h) {
            again += collect_survivors(&d->head[h], nc, qi[h], survivor + again, total - again);
            seg[h + 1] = again;
        }
    }

    candidate_t stack_cand[STACK_CANDIDATES];
    int32_t stack_index[2 * STACK_CANDIDATES];
    candidate_t* cand = stack_cand;
    int32_t* index = stack_index;
    void* heap = NULL;
    if (total > STACK_CANDIDATES) {
        heap = malloc(total * (sizeof(candidate_t) + 2 * sizeof(int32_t)));
        if (!heap) {
            free(heap_survivor);
            return -2;
        }
        cand = (candidate_t*)heap;
        index = (int32_t*)((char*)heap + total * sizeof(candidate_t));
    }

    /* 2. Class score, DFL expectation and box for each survivor at or above the confidence threshold. */
    size_t m = 0;
    for (int h = 0; h < 3; ++h) {
        const yolo_head_t* hd = &d->head[h];
        const size_t n = (size_t)hd->anchors;
        const float* sig = hd->sigmoid;
        for (size_t i = seg[h]; i < seg[h + 1]; ++i) {
            const size_t a = (size_t)survivor[i];
            float best;
            int32_t best_c = 0;
            if (hd->cls_max && hd->sigmoid_low) {
                const int q_max = hd->cls_max[a];
                best = sig[q_max + 128];
                if (!(best >= d->conf)) continue;
                const int lo = hd->sigmoid_low[q_max + 128];
                while (best_c < nc - 1 && (int)hd->cls[(size_t)best_c * n + a] < lo) ++best_c;
            } else {
                best = sig[(int)hd->cls[a] + 128];
                for (int32_t c = 1; c < nc; ++c) {
                    const float p = sig[(int)hd->cls[(size_t)c * n + a] + 128];
                    if (p > best) {  /* numpy's argmax keeps the first maximum */
                        best = p;
                        best_c = c;
                    }
                }
                if (!(best >= d->conf)) continue;
            }

            const int direct = (d->reg_max == 1);
            const float l = direct ? direct_side(hd->box, n, a, 0, hd->box_val) : dfl_side(hd->box, n, a, 0, hd->dfl_exp);
            const float t = direct ? direct_side(hd->box, n, a, 1, hd->box_val) : dfl_side(hd->box, n, a, 1, hd->dfl_exp);
            const float r = direct ? direct_side(hd->box, n, a, 2, hd->box_val) : dfl_side(hd->box, n, a, 2, hd->dfl_exp);
            const float b = direct ? direct_side(hd->box, n, a, 3, hd->box_val) : dfl_side(hd->box, n, a, 3, hd->dfl_exp);
            const size_t ai = (size_t)hd->anchor_offset + a;
            const float ax = d->anchor_x[ai];
            const float ay = d->anchor_y[ai];
            const float st = d->strides[ai];
            const float x1 = ax - l;
            const float y1 = ay - t;
            const float x2 = ax + r;
            const float y2 = ay + b;
            const float cx = ((x1 + x2) * 0.5f) * st;
            const float cy = ((y1 + y2) * 0.5f) * st;
            const float bw = (x2 - x1) * st;
            const float bh = (y2 - y1) * st;

            candidate_t* c = &cand[m++];
            c->x0 = ((cx - bw / 2.0f) - pad_left) / scale;
            c->y0 = ((cy - bh / 2.0f) - pad_top) / scale;
            c->w = bw / scale;
            c->h = bh / scale;
            c->score = best;
            c->class_id = best_c;
        }
    }
    free(heap_survivor);
    if (m == 0) {
        free(heap);
        return 0;
    }

    /* 3. NMSBoxesBatched + 4. NMSFast_ */
    int count = nms_and_output(cand, index, m, total, d->conf, d->iou, capacity, out_box, out_score, out_class);
    free(heap);
    return count;
}

/**
 * Decodes the three channel-blocked heads described by `d` for a frame letterboxed with (pad_top, pad_left)
 * and `scale`. Reads uint8 [blocks][anchors][8] heads and int8 [anchors] cls_max without an intermediate NCHW
 * transpose. Writes up to `capacity` detections in YoloDecoder.postprocess order.
 */
DECODE_API int yolo_decode_c8_blocks(
    const yolo_decode_c8_t* d,
    float pad_top,
    float pad_left,
    float scale,
    int32_t capacity,
    float* out_box,
    float* out_score,
    int32_t* out_class
) {
    if (!d || !out_box || !out_score || !out_class || capacity < 0 || (d->reg_max != DFL_BINS && d->reg_max != 1) ||
        d->num_classes <= 0 || !d->anchor_x || !d->anchor_y || !d->strides) {
        return -1;
    }
    const int32_t nc = d->num_classes;
    int qi[3];
    for (int h = 0; h < 3; ++h) {
        const yolo_head_c8_t* hd = &d->head[h];
        if (!hd->box_c8 || !hd->cls_c8 || !hd->cls_max || !hd->dfl_exp || !hd->sigmoid ||
            hd->anchors <= 0 || hd->anchor_offset < 0) {
            return -1;
        }
        qi[h] = int_threshold(hd->q_threshold);
    }

    /* 1. Surviving anchors per head from cls_max. */
    int32_t stack_survivor[STACK_SURVIVORS];
    int32_t* survivor = stack_survivor;
    void* heap_survivor = NULL;
    size_t seg[4] = {0, 0, 0, 0};
    size_t total = 0;
    for (int h = 0; h < 3; ++h) {
        const size_t room = (total < STACK_SURVIVORS) ? STACK_SURVIVORS - total : 0;
        total += collect_survivors_from_max(d->head[h].cls_max, d->head[h].anchors, qi[h],
                                           survivor + (total < STACK_SURVIVORS ? total : 0), room);
        seg[h + 1] = total;
    }
    if (total == 0) return 0;
    if (total > STACK_SURVIVORS) {
        heap_survivor = malloc(total * sizeof(int32_t));
        if (!heap_survivor) return -2;
        survivor = (int32_t*)heap_survivor;
        size_t again = 0;
        for (int h = 0; h < 3; ++h) {
            again += collect_survivors_from_max(d->head[h].cls_max, d->head[h].anchors, qi[h],
                                               survivor + again, total - again);
            seg[h + 1] = again;
        }
    }

    candidate_t stack_cand[STACK_CANDIDATES];
    int32_t stack_index[2 * STACK_CANDIDATES];
    candidate_t* cand = stack_cand;
    int32_t* index = stack_index;
    void* heap = NULL;
    if (total > STACK_CANDIDATES) {
        heap = malloc(total * (sizeof(candidate_t) + 2 * sizeof(int32_t)));
        if (!heap) {
            free(heap_survivor);
            return -2;
        }
        cand = (candidate_t*)heap;
        index = (int32_t*)((char*)heap + total * sizeof(candidate_t));
    }

    /* 2. Direct channel-blocked read for each survivor at or above conf threshold. */
    size_t m = 0;
    for (int h = 0; h < 3; ++h) {
        const yolo_head_c8_t* hd = &d->head[h];
        const size_t n = (size_t)hd->anchors;
        const float* sig = hd->sigmoid;
        for (size_t i = seg[h]; i < seg[h + 1]; ++i) {
            const size_t a = (size_t)survivor[i];
            const int q_max = hd->cls_max[a];
            const float best = sig[q_max + 128];
            if (!(best >= d->conf)) continue;
            const int lo = hd->sigmoid_low ? hd->sigmoid_low[q_max + 128] : q_max;

            int32_t best_c = 0;
            for (int b = 0; b < (nc + 7) / 8; ++b) {
                const uint8_t* cp = hd->cls_c8 + (size_t)b * (n * 8) + a * 8;
                int rem = nc - b * 8;
                int valid = rem > 8 ? 8 : rem;
                int found = 0;
                for (int k = 0; k < valid; ++k) {
                    int v = (int)(int8_t)(cp[k] ^ 0x80);
                    if (v >= lo) {
                        best_c = b * 8 + k;
                        found = 1;
                        break;
                    }
                }
                if (found) break;
            }

            const int direct = (d->reg_max == 1);
            const float l = direct ? direct_side_c8(hd->box_c8, n, a, 0, hd->box_val) : dfl_side_c8(hd->box_c8, n, a, 0, hd->dfl_exp);
            const float t = direct ? direct_side_c8(hd->box_c8, n, a, 1, hd->box_val) : dfl_side_c8(hd->box_c8, n, a, 1, hd->dfl_exp);
            const float r = direct ? direct_side_c8(hd->box_c8, n, a, 2, hd->box_val) : dfl_side_c8(hd->box_c8, n, a, 2, hd->dfl_exp);
            const float b = direct ? direct_side_c8(hd->box_c8, n, a, 3, hd->box_val) : dfl_side_c8(hd->box_c8, n, a, 3, hd->dfl_exp);
            const size_t ai = (size_t)hd->anchor_offset + a;
            const float ax = d->anchor_x[ai];
            const float ay = d->anchor_y[ai];
            const float st = d->strides[ai];
            const float x1 = ax - l;
            const float y1 = ay - t;
            const float x2 = ax + r;
            const float y2 = ay + b;
            const float cx = ((x1 + x2) * 0.5f) * st;
            const float cy = ((y1 + y2) * 0.5f) * st;
            const float bw = (x2 - x1) * st;
            const float bh = (y2 - y1) * st;

            candidate_t* c = &cand[m++];
            c->x0 = ((cx - bw / 2.0f) - pad_left) / scale;
            c->y0 = ((cy - bh / 2.0f) - pad_top) / scale;
            c->w = bw / scale;
            c->h = bh / scale;
            c->score = best;
            c->class_id = best_c;
        }
    }
    free(heap_survivor);

    int count = nms_and_output(cand, index, m, total, d->conf, d->iou, capacity, out_box, out_score, out_class);
    free(heap);
    return count;
}
