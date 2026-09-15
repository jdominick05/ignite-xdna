#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
tests/test_npu_inference.py

The oracle-free YOLOv8n path, checked without a device where that is possible
and on Device 0 where it is not.

    conda activate resnet_env17 && python tests/test_npu_inference.py   # offline half; silicon tests skip
    bash scripts/research-iron.sh tests/test_npu_inference.py           # both halves (pyxrt loads there)

Offline (no NPU):
  * the shipped container declares six heads (1,209,600 int8 values) but its
    egress is 4,096 bytes and it declares no head_layout, so the runtime reports
    the heads absent — and keeps reporting them absent even if the byte count
    matched, because it never guesses a packing order;
  * pack -> unpack round trip of a synthetic head_layout is exact and zero-copy;
  * int8-domain pruning in YoloDecoder.postprocess gives the same detections as
    the float path on synthetic heads (>= 4 objects, person and bus among them);
  * the ONNX Runtime CPU oracle on assets/bus.jpg decodes >= 4 detections with
    person and bus, which pins the preprocess -> postprocess chain the NPU path
    would feed;
  * CameraManager on a webcam index that does not exist degrades without raising.

On silicon (skipped without pyxrt, the container or the xclbin):
  * head status of the shipped container, and the >= 4 detection / IoU >= 0.70
    checks against the oracle — skipped with the reason when no heads exist;
  * 100 consecutive predict_sync frames on synthetic buffers: no buffer object
    is allocated after warm-up, working-set growth is bounded, and the
    glass-to-glass latency is reported (mean, p95, p99);
  * tools/live_camera_ignition.py --source assets/bus.jpg --headless runs to
    completion and prints HUD lines without a camera error;
  * YOLOv8n-pose (build/yolov8n_pose.ignite, PoseOnSilicon): on npu.yolo's
    letterbox the nine heads equal ONNX Runtime CPU's and the people (boxes,
    scores, 17 keypoints) equal its numpy tail's bit for bit, with no
    InferenceSession.run call; POSE_FRAMES native-ingress frames allocate no
    buffer objects and keep the working set within 5 MB.
"""
import os

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import ctypes
import gc
import importlib.util
import platform
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2  # noqa: E402

from ignite_xdna.runtime.heads import (  # noqa: E402
    HEAD_NAMES,
    DetectHeadLayout,
    contiguous_head_layout,
    declared_head_bytes,
    resolve_head_layout,
)
from ignite_xdna.pipelines.yolo_pipeline import (  # noqa: E402
    COCO_CLASSES,
    NUM_CLASSES,
    REG_MAX,
    STRIDES,
    YoloDecoder,
    YoloDetection,
)

def _default_container() -> Path:
    """IGNITE_MODEL, else the graph-engine container when compiled, else the legacy container."""
    override = os.environ.get("IGNITE_MODEL")
    if override:
        return Path(override)
    graph = REPO_ROOT / "build" / "yolov8n_full.ignite"
    return graph if graph.exists() else REPO_ROOT / "build" / "yolov8n.ignite"


MODEL_IGNITE = _default_container()
N_FRAMES = int(os.environ.get("IGNITE_TEST_FRAMES", "500"))  # continuous frames in the stability/latency test
# Glass-to-glass budgets of the stability/latency test (yolov8n_full defaults; IGNITE_MODEL=build/yolov8s.ignite
# runs with IGNITE_G2G_MEAN_MS=25 and IGNITE_G2G_P99_MS=27).
G2G_MEAN_BUDGET_MS = float(os.environ.get("IGNITE_G2G_MEAN_MS", "8.0"))
G2G_P99_BUDGET_MS = float(os.environ.get("IGNITE_G2G_P99_MS", "9.5"))
CUT_ONNX = REPO_ROOT / "models" / "yolov8n_cut_xint8.onnx"
XCLBIN = REPO_ROOT / "build" / "im2col_4d_16core.xclbin"
BUS_JPG = REPO_ROOT / "assets" / "bus.jpg"
CAMERA_TOOL = REPO_ROOT / "tools" / "live_camera_ignition.py"

PERSON = COCO_CLASSES.index("person")
BUS = COCO_CLASSES.index("bus")
CAR = COCO_CLASSES.index("car")

HEAD_SHAPES = {
    "p3_box": (1, 4 * REG_MAX, 80, 80), "p3_cls": (1, NUM_CLASSES, 80, 80),
    "p4_box": (1, 4 * REG_MAX, 40, 40), "p4_cls": (1, NUM_CLASSES, 40, 40),
    "p5_box": (1, 4 * REG_MAX, 20, 20), "p5_cls": (1, NUM_CLASSES, 20, 20),
}
DECLARED_HEAD_BYTES = 1_209_600
SHIPPED_EGRESS_BYTES = 4096


def _hardware_skip_reason() -> Optional[str]:
    try:
        import pyxrt  # noqa: F401
    except Exception as ex:  # noqa: BLE001
        return f"pyxrt is not importable in this interpreter ({type(ex).__name__}); use scripts/research-iron.sh"
    if not MODEL_IGNITE.exists():
        return f"missing {MODEL_IGNITE}"
    # A graph-engine container carries its own xclbin; the legacy one needs the conv0 template's.
    from ignite_xdna.compiler.serializer import IgniteModelReader
    with IgniteModelReader(MODEL_IGNITE) as reader:
        graph_engine = reader.manifest.get("engine") == "conv_engine_v1"
    if not graph_engine and not XCLBIN.exists():
        return f"missing {XCLBIN}"
    return None


HW_SKIP = _hardware_skip_reason()


def _load_camera_tool():
    """Import tools/live_camera_ignition.py by path (a site-packages 'tools' shadows the repo dir)."""
    spec = importlib.util.spec_from_file_location("live_camera_ignition", CAMERA_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rss_bytes() -> int:
    if platform.system() == "Windows":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        # HANDLE is pointer-sized: without these, the (HANDLE)-1 pseudo-handle is truncated to 32 bits
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        pmc = Counters()
        pmc.cb = ctypes.sizeof(Counters)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise OSError(f"GetProcessMemoryInfo failed (error {ctypes.get_last_error()})")
        return int(pmc.WorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _iou(a: YoloDetection, b: YoloDetection) -> float:
    ax2, ay2, bx2, by2 = a.x0 + a.w, a.y0 + a.h, b.x0 + b.w, b.y0 + b.h
    iw = max(0.0, min(ax2, bx2) - max(a.x0, b.x0))
    ih = max(0.0, min(ay2, by2) - max(a.y0, b.y0))
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _best_iou_per_detection(dets: List[YoloDetection], refs: List[YoloDetection]) -> List[float]:
    return [max((_iou(d, r) for r in refs if r.class_id == d.class_id), default=0.0) for d in dets]


def _synthetic_layout(shapes=HEAD_SHAPES) -> Tuple[Dict, DetectHeadLayout]:
    """A manifest with a contiguous head_layout and one dequantization per head."""
    scales = {"p3_box": (0.0625, 0), "p3_cls": (0.0625, -5), "p4_box": (0.0625, 0),
              "p4_cls": (0.0625, 3), "p5_box": (0.0625, 0), "p5_cls": (0.0625, 0)}
    manifest = {"output_shapes": {k: list(v) for k, v in shapes.items()},
                "head_layout": contiguous_head_layout(shapes, scales)}
    status = resolve_head_layout(manifest, DECLARED_HEAD_BYTES)
    assert status.present, status.reason
    return manifest, status.layout


def _synthetic_float_heads(objects) -> Dict[str, np.ndarray]:
    """Float heads with every logit representable at scale 1/16, carrying ``objects``.

    objects: (cx, cy, w, h, class_id) in letterboxed 640x640 pixels. Each object
    is written at the anchor nearest its centre on the level whose stride fits
    its size, with class logit +6 (sigmoid 0.9975) and one-hot DFL bins at the
    integer left/top/right/bottom distances in stride units (max 15).
    """
    heads = {}
    for name, shape in HEAD_SHAPES.items():
        heads[name] = np.full(shape, 0.0 if name.endswith("box") else -8.0, dtype=np.float32)
    for cx, cy, w, h, cls in objects:
        level = 0 if max(w, h) < 96 else (1 if max(w, h) < 224 else 2)
        stride = STRIDES[level]
        grid = 640 // stride
        gx, gy = min(grid - 1, int(cx // stride)), min(grid - 1, int(cy // stride))
        ax, ay = (gx + 0.5) * stride, (gy + 0.5) * stride
        ltrb = [(ax - (cx - w / 2)) / stride, (ay - (cy - h / 2)) / stride,
                ((cx + w / 2) - ax) / stride, ((cy + h / 2) - ay) / stride]
        box = heads[f"p{level + 3}_box"]
        for side, dist in enumerate(ltrb):
            b = int(np.clip(round(dist), 0, REG_MAX - 1))
            box[0, side * REG_MAX + b, gy, gx] = 6.0
        heads[f"p{level + 3}_cls"][0, cls, gy, gx] = 6.0
    return heads


SYNTHETIC_OBJECTS = [
    (100.0, 120.0, 40.0, 72.0, PERSON),   # P3
    (300.0, 200.0, 48.0, 88.0, PERSON),   # P3
    (500.0, 420.0, 56.0, 80.0, PERSON),   # P3
    (200.0, 400.0, 160.0, 96.0, CAR),     # P4
    (360.0, 300.0, 320.0, 256.0, BUS),    # P5
]


class DetectHeadLayoutOffline(unittest.TestCase):
    """No device: manifest resolution, zero-copy unpack, int8 vs float decode."""

    def test_01_shipped_container_reports_heads_absent(self):
        if not MODEL_IGNITE.exists():
            self.skipTest(f"{MODEL_IGNITE} not built")
        from ignite_xdna.compiler.serializer import IgniteModelReader
        reader = IgniteModelReader(MODEL_IGNITE)
        try:
            manifest = reader.manifest
        finally:
            reader.close()
        self.assertEqual(declared_head_bytes(manifest), DECLARED_HEAD_BYTES)
        if manifest.get("engine") == "conv_engine_v1":
            # The graph-engine container carries the six heads: present at its egress size,
            # still absent at the legacy template's 4,096-byte egress.
            self.assertTrue(resolve_head_layout(manifest, int(manifest["egress_bytes"])).present)
            self.assertFalse(resolve_head_layout(manifest, SHIPPED_EGRESS_BYTES).present)
            return
        status = resolve_head_layout(manifest, SHIPPED_EGRESS_BYTES)
        self.assertFalse(status.present)
        self.assertIn("head_layout", status.reason)
        self.assertEqual((status.declared_bytes, status.egress_bytes), (DECLARED_HEAD_BYTES, SHIPPED_EGRESS_BYTES))
        # Fail closed: a large enough egress without a declared layout is still not a head buffer
        self.assertFalse(resolve_head_layout(manifest, DECLARED_HEAD_BYTES).present)
        print(f"\n[heads] shipped container: {status.reason}")

    def test_02_layout_rejects_malformed_declarations(self):
        manifest, layout = _synthetic_layout()
        self.assertEqual(layout.required_bytes, DECLARED_HEAD_BYTES)
        self.assertFalse(resolve_head_layout(manifest, DECLARED_HEAD_BYTES - 1).present)
        bad = {**manifest, "head_layout": {**manifest["head_layout"],
                                           "p4_box": {**manifest["head_layout"]["p4_box"], "offset": 0}}}
        self.assertIn("overlap", resolve_head_layout(bad, DECLARED_HEAD_BYTES).reason)
        bad = {**manifest, "head_layout": {**manifest["head_layout"],
                                           "p5_cls": {**manifest["head_layout"]["p5_cls"], "scale": 0.0}}}
        self.assertIn("scale", resolve_head_layout(bad, DECLARED_HEAD_BYTES).reason)
        self.assertIn("fused-DFL", resolve_head_layout({**manifest, "fused_dfl": True}, DECLARED_HEAD_BYTES).reason)
        self.assertFalse(resolve_head_layout({}, DECLARED_HEAD_BYTES).present)
        self.assertFalse(resolve_head_layout(None, DECLARED_HEAD_BYTES).present)

    def test_03_pack_unpack_round_trip_is_exact_and_zero_copy(self):
        _, layout = _synthetic_layout()
        rng = np.random.default_rng(7)
        heads = {name: rng.integers(-128, 128, size=shape, dtype=np.int8) for name, shape in HEAD_SHAPES.items()}
        raw = layout.pack(heads, egress_bytes=DECLARED_HEAD_BYTES + 64)
        views = layout.unpack(raw)
        for name in HEAD_NAMES:
            self.assertEqual(views[name].dtype, np.int8)
            self.assertEqual(views[name].shape, HEAD_SHAPES[name])
            self.assertTrue(np.array_equal(views[name], heads[name]))
            self.assertTrue(np.shares_memory(views[name], raw), f"{name} is a copy")
        deq = layout.dequantize(views)
        spec = layout.spec("p3_cls")
        expect = (heads["p3_cls"].astype(np.float32) - spec.zero_point) * np.float32(spec.scale)
        self.assertTrue(np.array_equal(deq["p3_cls"], expect))
        self.assertTrue(np.array_equal(layout.quantize(deq)["p3_cls"], heads["p3_cls"]))
        with self.assertRaises(ValueError):
            layout.unpack(raw[: DECLARED_HEAD_BYTES - 1])
        with self.assertRaises(TypeError):
            layout.unpack(raw.astype(np.uint8))

    def test_04_int8_pruning_matches_float_decode(self):
        _, layout = _synthetic_layout()
        decoder = YoloDecoder(imgsz=640, conf_thres=0.25, iou_thres=0.5)
        float_heads = _synthetic_float_heads(SYNTHETIC_OBJECTS)
        pad, scale = (0, 0), 1.0

        ref = decoder.postprocess(float_heads, pad, scale)
        raw = layout.pack(layout.quantize(float_heads))
        views = layout.unpack(raw)
        got = decoder.postprocess({**views, "scales": layout.scales()}, pad, scale)

        self.assertGreaterEqual(len(ref), 4)
        self.assertEqual(len(got), len(ref))
        self.assertEqual([d.class_id for d in got], [d.class_id for d in ref])
        for g, r in zip(got, ref):
            self.assertAlmostEqual(g.score, r.score, places=5)
            for attr in ("x0", "y0", "w", "h"):
                self.assertAlmostEqual(getattr(g, attr), getattr(r, attr), places=3)
        classes = {d.class_id for d in got}
        self.assertIn(PERSON, classes)
        self.assertIn(BUS, classes)
        self.assertTrue(all(v >= 0.70 for v in _best_iou_per_detection(got, ref)))
        # The six list-form heads and a dict with a missing head are both accepted
        as_list = [float_heads[k] for k in ("p3_box", "p4_box", "p5_box", "p3_cls", "p4_cls", "p5_cls")]
        self.assertEqual(len(decoder.postprocess(as_list, pad, scale)), len(ref))
        self.assertEqual(decoder.postprocess({**float_heads, "p3_box": None}, pad, scale), [])
        with self.assertRaises(ValueError):
            decoder.postprocess(dict(views), pad, scale)  # int8 without scales
        print(f"\n[decode] synthetic heads: {len(got)} detections, classes "
              f"{sorted(COCO_CLASSES[c] for c in classes)}, int8 path == float path")

    def test_05_cpu_oracle_on_bus_jpg_decodes_person_and_bus(self):
        if not CUT_ONNX.exists():
            self.skipTest(f"{CUT_ONNX} not present (models/ is generated)")
        if not BUS_JPG.exists():
            self.skipTest(f"{BUS_JPG} missing")
        try:
            import onnxruntime as ort
        except Exception as ex:  # noqa: BLE001
            self.skipTest(f"onnxruntime unavailable: {ex}")
        from ignite_xdna.pipelines.preprocess import FusedPreprocessor

        img = cv2.imread(str(BUS_JPG))
        self.assertIsNotNone(img)
        quant, pad, scale = FusedPreprocessor(imgsz=640).preprocess(img)
        self.assertEqual(quant.shape, (1, 3, 640, 640))
        self.assertEqual(quant.dtype, np.int8)

        sess = ort.InferenceSession(str(CUT_ONNX), providers=["CPUExecutionProvider"])
        x_float = (quant.astype(np.float32) + 128.0) / 255.0  # the same feed predict_sync uses
        outs = sess.run(None, {sess.get_inputs()[0].name: x_float})
        self.assertEqual(len(outs), 6)
        # The pipeline assumes box P3/P4/P5 then cls P3/P4/P5
        self.assertEqual(tuple(outs[0].shape), HEAD_SHAPES["p3_box"])
        self.assertEqual(tuple(outs[3].shape), HEAD_SHAPES["p3_cls"])

        dets = YoloDecoder(imgsz=640, conf_thres=0.25, iou_thres=0.5).postprocess(outs, pad, scale)
        classes = [d.class_name for d in dets]
        print(f"\n[oracle] bus.jpg: {len(dets)} detections: " +
              ", ".join(f"{d.class_name} {d.score:.2f}" for d in dets))
        self.assertGreaterEqual(len(dets), 4)
        self.assertIn("person", classes)
        self.assertIn("bus", classes)


class CameraCaptureOffline(unittest.TestCase):
    """No device: the camera tool's probing and file sources."""

    def test_06_camera_manager_degrades_without_a_camera(self):
        tool = _load_camera_tool()
        messages = []
        mgr = tool.CameraManager(indices=(99,), open_timeout_s=3.0, log=messages.append)
        t0 = time.perf_counter()
        cap = mgr.open()
        elapsed = time.perf_counter() - t0
        self.assertIsNone(cap)
        self.assertEqual(len(mgr.attempts), len(mgr.backends))
        self.assertTrue(all(a.index == 99 for a in mgr.attempts))
        self.assertTrue(all("opened, first frame read" != a.outcome for a in mgr.attempts))
        self.assertLess(elapsed, 3.0 * len(mgr.backends) + 5.0)
        self.assertEqual(len(messages), len(mgr.attempts))
        with self.assertRaises(RuntimeError):
            tool.FrameSource.from_arg("99", manager=mgr, log=messages.append)
        print("\n[camera] index 99: " + "; ".join(f"{a.backend} {a.outcome} {a.seconds:.2f}s" for a in mgr.attempts))

    def test_07_image_source_and_property_readback(self):
        if not BUS_JPG.exists():
            self.skipTest(f"{BUS_JPG} missing")
        tool = _load_camera_tool()
        src = tool.FrameSource.from_arg(str(BUS_JPG), log=lambda *_: None)
        self.assertEqual(src.kind, "image")
        ok, frame = src.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, cv2.imread(str(BUS_JPG)).shape)
        frame[:] = 0
        ok2, frame2 = src.read()
        self.assertTrue(ok2 and frame2.max() > 0, "image source must hand out a fresh copy")
        src.release()
        with self.assertRaises(FileNotFoundError):
            tool.FrameSource.from_arg(str(REPO_ROOT / "assets" / "does_not_exist.mp4"))

        # A refused property is reported, not raised: a VideoCapture over a still image accepts nothing
        mgr = tool.CameraManager(indices=(), log=lambda *_: None)
        with tool.quiet_opencv():
            cap = cv2.VideoCapture(str(BUS_JPG))
        try:
            res = mgr.set_property(cap, cv2.CAP_PROP_FPS, 60.0, "FPS")
        finally:
            cap.release()
        self.assertIsInstance(res.accepted, bool)
        self.assertEqual(res.requested, 60.0)
        dets = [YoloDetection(x0=10.0, y0=20.0, w=30.0, h=40.0, score=0.9, class_id=0, class_name="person")]
        self.assertEqual(tool.extract_box(dets[0], 640, 480), (10, 20, 40, 60))
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(tool.draw_detections(canvas, dets, 0.25), 1)
        self.assertGreater(int(canvas.max()), 0)


@unittest.skipIf(HW_SKIP is not None, HW_SKIP or "")
class NpuInferenceOnSilicon(unittest.TestCase):
    """Device 0: head status, buffer stability over 100 frames, latency, the camera tool."""

    @classmethod
    def setUpClass(cls):
        if not BUS_JPG.exists():
            raise unittest.SkipTest(f"{BUS_JPG} missing")
        cls.bus = cv2.imread(str(BUS_JPG))

    def tearDown(self):
        gc.collect()
        time.sleep(0.05)

    def _pipeline(self):
        from ignite_xdna.pipelines.yolo_pipeline import YoloPipeline
        return YoloPipeline(model_path_or_bundle=str(MODEL_IGNITE), device_index=0, conf_thres=0.25, iou_thres=0.5)

    def test_10_head_status_and_oracle_free_detections(self):
        with self._pipeline() as pipe:
            status = pipe.session.head_status
            dets, timings = pipe.predict_sync(self.bus, use_oracle_for_boxes=False)
            self.assertEqual(timings.head_source, "npu" if status.present else "none")
            self.assertEqual(pipe.last_head_status, status.reason)
            heads = pipe.session.run_yolo_monolithic(np.zeros((1, 3, 640, 640), dtype=np.int8), unswizzle=False)
            self.assertEqual(heads["heads_present"], status.present)
            self.assertEqual(heads["raw_output"].dtype, np.int8)
            self.assertEqual(heads["raw_output"].size, pipe.session.out_bytes)
            print(f"\n[silicon] heads {'present' if status.present else 'absent'}: {status.reason}")
            if not status.present:
                self.assertEqual(dets, [])
                self.assertTrue(all(heads[name] is None for name in HEAD_NAMES))
                self.skipTest(f"no detect heads in the device egress, so the >= 4 detection and IoU >= 0.70 "
                              f"checks cannot run: {status.reason}")
            classes = [d.class_name for d in dets]
            self.assertGreaterEqual(len(dets), 4)
            self.assertIn("person", classes)
            self.assertIn("bus", classes)
            ref, _ = pipe.predict_sync(self.bus, use_oracle_for_boxes=True)
            ious = _best_iou_per_detection(dets, ref)
            print(f"[silicon] IoU vs oracle per detection: {np.round(ious, 3).tolist()}")
            self.assertGreaterEqual(float(np.mean(ious)), 0.70)

    def test_11_hundred_frames_no_buffer_growth_and_latency(self):
        """N_FRAMES continuous frames: no buffer objects allocated, < 5 MB working-set drift, G2G within budget."""
        rng = np.random.default_rng(1234)
        frames = [rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8) for _ in range(4)]
        with self._pipeline() as pipe:
            harness = pipe.session.harness
            allocations = {"host_bo": 0, "instr_bo": 0}
            real_host, real_instr = harness.create_host_bo, harness.create_instruction_bo_from_bytes

            def counted_host(*a, **k):
                allocations["host_bo"] += 1
                return real_host(*a, **k)

            def counted_instr(*a, **k):
                allocations["instr_bo"] += 1
                return real_instr(*a, **k)

            harness.create_host_bo = counted_host
            harness.create_instruction_bo_from_bytes = counted_instr
            try:
                for i in range(10):
                    pipe.predict_sync(frames[i % 4], use_oracle_for_boxes=False)
                gc.collect()
                rss_before = _rss_bytes()
                g2g, npu, prep, post = [], [], [], []
                sources = set()
                for i in range(N_FRAMES):
                    _, t = pipe.predict_sync(frames[i % 4], use_oracle_for_boxes=False)
                    g2g.append(t.glass_to_glass_ms)
                    npu.append(t.npu_forward_ms)
                    prep.append(t.preprocess_ms)
                    post.append(t.postprocess_ms)
                    sources.add(t.head_source)
                gc.collect()
                rss_after = _rss_bytes()
            finally:
                harness.create_host_bo = real_host
                harness.create_instruction_bo_from_bytes = real_instr
            status = pipe.session.head_status

        arr = np.asarray(g2g)
        growth_mb = (rss_after - rss_before) / (1024 * 1024)
        print(f"\n[silicon] {N_FRAMES} frames (1280x720 synthetic): G2G mean {arr.mean():.3f} ms median "
              f"{np.median(arr):.3f} p95 {np.percentile(arr, 95):.3f} p99 {np.percentile(arr, 99):.3f} "
              f"max {arr.max():.3f} | preprocess {np.mean(prep):.3f} | NPU {np.mean(npu):.3f} | "
              f"postprocess {np.mean(post):.3f} ms | boxes from {sorted(sources)} | "
              f"buffer objects allocated after warm-up: {allocations} | working set "
              f"{rss_before / 2**20:.1f} -> {rss_after / 2**20:.1f} MB ({growth_mb:+.2f} MB)")
        if not status.present:
            print("[silicon] note: the postprocess stage saw no head tensors, so this is the latency of "
                  "preprocess + NPU dispatch + an empty decode, not of a detection pipeline")
        self.assertEqual(allocations, {"host_bo": 0, "instr_bo": 0})
        self.assertLess(growth_mb, 5.0, f"working set grew by {growth_mb:.2f} MB over {N_FRAMES} frames")
        self.assertLessEqual(float(arr.mean()), G2G_MEAN_BUDGET_MS,
                             f"mean glass-to-glass {arr.mean():.3f} ms > {G2G_MEAN_BUDGET_MS} ms")
        p99 = float(np.percentile(arr, 99))
        self.assertLessEqual(p99, G2G_P99_BUDGET_MS, f"p99 glass-to-glass {p99:.3f} ms > {G2G_P99_BUDGET_MS} ms")

    def test_12_camera_tool_headless_on_bus_jpg(self):
        cmd = [sys.executable, str(CAMERA_TOOL), "--source", str(BUS_JPG), "--headless", "--frames", "5",
               "--boxes", "npu", "--model", str(MODEL_IGNITE)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=str(REPO_ROOT))
        out = res.stdout + res.stderr
        print("\n[camera-tool] " + " | ".join(line for line in res.stdout.splitlines() if line.startswith("[")))
        self.assertEqual(res.returncode, 0, out[-2000:])
        self.assertEqual(sum(1 for line in res.stdout.splitlines() if line.startswith("[hud] frame ")), 5)
        self.assertIn("[summary] frames 5", res.stdout)
        self.assertIn("source: image", res.stdout)
        self.assertNotIn("could not open source", out)
        self.assertNotIn("-2147023832", out)


SR_IGNITE = Path(os.environ.get("IGNITE_SR_MODEL", str(REPO_ROOT / "build" / "sesr_m7.ignite")))
SR_ONNX = REPO_ROOT / "models" / "sesr_m7_xint8.onnx"
SR_FRAMES = int(os.environ.get("IGNITE_SR_FRAMES", "500"))
SR_DISPATCH_BUDGET_MS = float(os.environ.get("IGNITE_SR_DISPATCH_MS", "1.5"))


def _sr_skip_reason() -> Optional[str]:
    try:
        import pyxrt  # noqa: F401
    except Exception as ex:  # noqa: BLE001
        return f"pyxrt is not importable in this interpreter ({type(ex).__name__}); use scripts/research-iron.sh"
    if not SR_IGNITE.exists():
        return f"missing {SR_IGNITE} (ignite-compile --engine graph --input models/sesr_m7_xint8.onnx)"
    return None


SR_SKIP = _sr_skip_reason()


@unittest.skipIf(SR_SKIP is not None, SR_SKIP or "")
class SuperResolutionOnSilicon(unittest.TestCase):
    """Device 0, SESR M7 2x (build/sesr_m7.ignite): image parity with ONNX Runtime, buffer stability, dispatch."""

    @classmethod
    def setUpClass(cls):
        if not BUS_JPG.exists():
            raise unittest.SkipTest(f"{BUS_JPG} missing")
        cls.bus = cv2.imread(str(BUS_JPG))

    def tearDown(self):
        gc.collect()
        time.sleep(0.05)

    def test_20_image_matches_onnx_runtime(self):
        """Every pixel of the NPU image equals ONNX Runtime's (graph optimizations off) SESR output."""
        import onnxruntime as ort
        from ignite_xdna.pipelines.sr_pipeline import SuperResolutionPipeline
        with SuperResolutionPipeline(SR_IGNITE) as sr:
            got, t = sr.predict_sync(self.bus)
        if not SR_ONNX.exists():
            self.skipTest(f"{SR_ONNX} missing: image checked for shape only")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(str(SR_ONNX), options, providers=["CPUExecutionProvider"])
        small = cv2.resize(self.bus, (256, 256), interpolation=cv2.INTER_LINEAR)
        x = np.transpose(cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) - 128.0, (2, 0, 1))[None]
        y = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
        ref = cv2.cvtColor(np.clip(np.transpose(y, (1, 2, 0)) + 128.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        diff = int(np.count_nonzero(got != ref))
        print(f"\n[silicon-sr] bus.jpg -> {got.shape}: {diff} of {ref.size} values differ from ONNX Runtime | "
              f"dispatch {t.dispatch_ms:.3f} ms, readback {t.readback_ms:.3f}, G2G {t.glass_to_glass_ms:.3f} ms")
        self.assertEqual(got.shape, (512, 512, 3))
        self.assertEqual(diff, 0)

    def _run_frames(self):
        from ignite_xdna.pipelines.sr_pipeline import SuperResolutionPipeline
        rng = np.random.default_rng(4321)
        frames = [rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for _ in range(4)]
        stats = {k: [] for k in ("g2g", "dispatch", "readback", "pre", "post")}
        with SuperResolutionPipeline(SR_IGNITE) as sr:
            harness = sr.session.harness
            allocations = {"host_bo": 0, "instr_bo": 0}
            real_host, real_instr = harness.create_host_bo, harness.create_instruction_bo_from_bytes

            def counted_host(*a, **k):
                allocations["host_bo"] += 1
                return real_host(*a, **k)

            def counted_instr(*a, **k):
                allocations["instr_bo"] += 1
                return real_instr(*a, **k)

            harness.create_host_bo = counted_host
            harness.create_instruction_bo_from_bytes = counted_instr
            try:
                for i in range(10):
                    sr.predict_sync(frames[i % 4])
                gc.collect()
                rss_before = _rss_bytes()
                for i in range(SR_FRAMES):
                    _, t = sr.predict_sync(frames[i % 4])
                    stats["g2g"].append(t.glass_to_glass_ms)
                    stats["dispatch"].append(t.dispatch_ms)
                    stats["readback"].append(t.readback_ms)
                    stats["pre"].append(t.preprocess_ms)
                    stats["post"].append(t.postprocess_ms)
                gc.collect()
                rss_after = _rss_bytes()
            finally:
                harness.create_host_bo = real_host
                harness.create_instruction_bo_from_bytes = real_instr
        return {k: np.asarray(v) for k, v in stats.items()}, allocations, (rss_after - rss_before) / 2 ** 20

    def test_21_frames_without_buffer_growth(self):
        """SR_FRAMES continuous frames: no buffer objects allocated, < 5 MB working-set drift, no dispatch errors."""
        stats, allocations, growth_mb = self._run_frames()
        type(self).stats = stats
        g, d = stats["g2g"], stats["dispatch"]
        print(f"\n[silicon-sr] {SR_FRAMES} frames (640x480 synthetic): G2G mean {g.mean():.3f} ms p50 "
              f"{np.median(g):.3f} p95 {np.percentile(g, 95):.3f} p99 {np.percentile(g, 99):.3f} | dispatch mean "
              f"{d.mean():.3f} p50 {np.median(d):.3f} p99 {np.percentile(d, 99):.3f} min {d.min():.3f} | preprocess "
              f"{stats['pre'].mean():.3f} | readback {stats['readback'].mean():.3f} | postprocess "
              f"{stats['post'].mean():.3f} | buffer objects allocated after warm-up: {allocations} | working set "
              f"{growth_mb:+.2f} MB")
        self.assertEqual(allocations, {"host_bo": 0, "instr_bo": 0})
        self.assertLess(growth_mb, 5.0)

    def test_22_dispatch_budget(self):
        """Mean raw NPU dispatch within IGNITE_SR_DISPATCH_MS (default 1.5 ms)."""
        stats = getattr(type(self), "stats", None)
        if stats is None:
            stats, _, _ = self._run_frames()
        mean = float(stats["dispatch"].mean())
        self.assertLessEqual(mean, SR_DISPATCH_BUDGET_MS,
                             f"mean dispatch {mean:.3f} ms > {SR_DISPATCH_BUDGET_MS} ms")


POSE_IGNITE = Path(os.environ.get("IGNITE_POSE_MODEL", str(REPO_ROOT / "build" / "yolov8n_pose.ignite")))
POSE_ONNX = REPO_ROOT / "models" / "yolov8n-pose_cut_xint8.onnx"
POSE_FRAMES = int(os.environ.get("IGNITE_POSE_FRAMES", "500"))


def _pose_skip_reason() -> Optional[str]:
    try:
        import pyxrt  # noqa: F401
    except Exception as ex:  # noqa: BLE001
        return f"pyxrt is not importable in this interpreter ({type(ex).__name__}); use scripts/research-iron.sh"
    if not POSE_IGNITE.exists():
        return (f"missing {POSE_IGNITE} (ignite-compile --engine graph --input models/yolov8n-pose_cut_xint8.onnx "
                f"--output build/yolov8n_pose.ignite)")
    return None


POSE_SKIP = _pose_skip_reason()


@unittest.skipIf(POSE_SKIP is not None, POSE_SKIP or "")
class PoseOnSilicon(unittest.TestCase):
    """Device 0, YOLOv8n-pose (build/yolov8n_pose.ignite): heads and people equal ONNX Runtime's, buffer stability."""

    @classmethod
    def setUpClass(cls):
        if not BUS_JPG.exists():
            raise unittest.SkipTest(f"{BUS_JPG} missing")
        cls.bus = cv2.imread(str(BUS_JPG))

    def tearDown(self):
        gc.collect()
        time.sleep(0.05)

    def test_30_heads_and_people_match_onnx_runtime(self):
        """numpy ingress: the nine heads equal ONNX Runtime CPU's, people equal its numpy tail's bit for bit, no ORT call."""
        if not POSE_ONNX.exists():
            self.skipTest(f"{POSE_ONNX} missing")
        import onnxruntime as ort
        import npu.yolo as yolo
        import npu.yolo_pose as yp
        from npu.yolo_pose_decode import decode_heads
        from ignite_xdna.pipelines.pose_pipeline import HEAD_ORDER, PosePipeline
        settings = ((0.25, 0.5), (0.001, 0.7))
        sess = ort.InferenceSession(str(POSE_ONNX), providers=["CPUExecutionProvider"])
        x, pad, scale = yolo.letterbox(self.bus)
        float_heads = sess.run(yp.HEAD_OUTS, {sess.get_inputs()[0].name: x})
        calls = []
        real_run = ort.InferenceSession.run

        def counting_run(session, *args, **kwargs):
            calls.append(1)
            return real_run(session, *args, **kwargs)

        with PosePipeline(POSE_IGNITE, ingress="numpy") as pose:
            ort.InferenceSession.run = counting_run
            try:
                got = {conf: pose.predict_sync(self.bus, conf_thres=conf, iou_thres=iou)[0] for conf, iou in settings}
                heads = pose.session.run_yolo_monolithic(None)
            finally:
                ort.InferenceSession.run = real_run
            scales = heads["scales"]
            bad_heads = [name for name, ref in zip(HEAD_ORDER, float_heads)
                         if not np.array_equal((heads[name].astype(np.float32) - np.float32(scales[name][1]))
                                               * np.float32(scales[name][0]), ref)]
        print(f"\n[silicon-pose] bus.jpg, numpy ingress: heads {9 - len(bad_heads)}/9 equal ONNX Runtime | people "
              f"{len(got[0.25])} at conf 0.25, {len(got[0.001])} at conf 0.001 | InferenceSession.run calls {len(calls)}")
        self.assertEqual(bad_heads, [])
        self.assertEqual(calls, [])
        for conf, iou in settings:
            ref = yp.postprocess(decode_heads(float_heads, conf_thres=conf), pad, scale, conf, iou, 300)
            self.assertEqual(len(got[conf]), len(ref), f"conf {conf}")
            for d, (x0, y0, w, h, s, kpts) in zip(got[conf], ref):
                self.assertEqual((np.float32(d.x0), np.float32(d.y0), np.float32(d.w), np.float32(d.h),
                                  np.float32(d.score)), (x0, y0, w, h, s))
                self.assertTrue(np.array_equal(d.keypoints, kpts))
        self.assertGreaterEqual(len(got[0.25]), 3)

    def test_31_native_ingress_frames_without_buffer_growth(self):
        """POSE_FRAMES frames of bus.jpg through native ingress: no buffer objects allocated, < 5 MB working-set drift."""
        from ignite_xdna.pipelines.pose_pipeline import PosePipeline
        stats = {k: [] for k in ("g2g", "dispatch", "readback", "pre", "post")}
        counts = []
        with PosePipeline(POSE_IGNITE) as pose:
            harness = pose.session.harness
            allocations = {"host_bo": 0, "instr_bo": 0}
            real_host, real_instr = harness.create_host_bo, harness.create_instruction_bo_from_bytes

            def counted_host(*a, **k):
                allocations["host_bo"] += 1
                return real_host(*a, **k)

            def counted_instr(*a, **k):
                allocations["instr_bo"] += 1
                return real_instr(*a, **k)

            harness.create_host_bo = counted_host
            harness.create_instruction_bo_from_bytes = counted_instr
            try:
                for _ in range(10):
                    pose.predict_sync(self.bus)
                gc.collect()
                rss_before = _rss_bytes()
                for _ in range(POSE_FRAMES):
                    people, t = pose.predict_sync(self.bus)
                    counts.append(len(people))
                    stats["g2g"].append(t.glass_to_glass_ms)
                    stats["dispatch"].append(t.dispatch_ms)
                    stats["readback"].append(t.readback_ms)
                    stats["pre"].append(t.preprocess_ms)
                    stats["post"].append(t.postprocess_ms)
                gc.collect()
                rss_after = _rss_bytes()
            finally:
                harness.create_host_bo = real_host
                harness.create_instruction_bo_from_bytes = real_instr
        s = {k: np.asarray(v) for k, v in stats.items()}
        growth_mb = (rss_after - rss_before) / 2 ** 20
        print(f"\n[silicon-pose] {POSE_FRAMES} frames of bus.jpg (native ingress): G2G mean {s['g2g'].mean():.3f} ms "
              f"p50 {np.median(s['g2g']):.3f} p99 {np.percentile(s['g2g'], 99):.3f} | dispatch mean {s['dispatch'].mean():.3f} "
              f"| preprocess {s['pre'].mean():.3f} | readback {s['readback'].mean():.3f} | decode and NMS "
              f"{s['post'].mean():.3f} | people per frame {sorted(set(counts))} | buffer objects allocated after "
              f"warm-up: {allocations} | working set {growth_mb:+.2f} MB")
        self.assertEqual(allocations, {"host_bo": 0, "instr_bo": 0})
        self.assertLess(growth_mb, 5.0)
        self.assertEqual(len(set(counts)), 1)
        self.assertGreaterEqual(counts[0], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
