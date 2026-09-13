#!/usr/bin/env python3
"""Live YOLOv8n on the Phoenix NPU from a webcam, a video file or a still image.

    conda activate resnet_env17         # pyxrt loads only in the mlir-aie ironenv:
    bash scripts/research-iron.sh tools/live_camera_ignition.py             # probe webcams 0 and 1
    bash scripts/research-iron.sh tools/live_camera_ignition.py --source 1  # one webcam index
    bash scripts/research-iron.sh tools/live_camera_ignition.py --source clip.mp4
    bash scripts/research-iron.sh tools/live_camera_ignition.py --source assets/bus.jpg --headless --frames 5

Camera opening is the part that used to fail. OpenCV's MSMF backend takes ~90 s
per open unless OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0 is in the environment
before cv2 is imported (docs/BENCHMARKS.md, "The ~90s camera open was OpenCV's
backend"); asking a sensor for MJPG or 60 fps it does not have makes MSMF log
error -2147023832 and can drop the stream; and DirectShow can block for good on
an IR / Windows Hello sensor. CameraManager therefore probes (index, backend)
pairs in the order MSMF -> DSHOW -> ANY, each in a worker thread with a timeout,
applies every requested property through a read-back check, and never raises
on a property the sensor refuses (the sensor default is kept).

Boxes: --boxes npu decodes the heads the container's egress carries. The shipped
build/yolov8n.ignite carries none (its stage streams are the single-layer conv0
template, docs/LOW_LEVEL_AUDIT.md section 1.5), so that mode draws nothing and
the HUD says why. --boxes oracle runs the ONNX Runtime CPU pass over the cut
model (~43 ms/frame). --boxes auto (default) uses the NPU heads when present and
otherwise the oracle, labelled as such on the HUD and in the summary.
"""
import os

# Must precede every cv2 import, including the transitive one through
# ignite_xdna.pipelines: OpenCV reads it at videoio init (BENCHMARKS.md,
# results/cam_probe_late_set.log measured that a later assignment is a no-op).
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import argparse
import contextlib
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
WINDOW_NAME = "Ignition YOLOv8n - Bare-Metal XDNA1 Silicon"


def backend_table() -> List[Tuple[str, int]]:
    """(name, api) in probe order; a constant missing from this OpenCV build is skipped."""
    table = []
    for attr in ("CAP_MSMF", "CAP_DSHOW", "CAP_ANY"):
        api = getattr(cv2, attr, None)
        if api is not None:
            table.append((attr[4:], int(api)))
    return table


@contextlib.contextmanager
def quiet_opencv():
    """Silence OpenCV's own log lines (the MSMF -2147023832 report on a refused property).

    OpenCV 4.x exposes ``cv2.setLogLevel`` (0 = silent); 5.x moved it to
    ``cv2.utils.logging``. Both were checked on this machine's two builds
    (4.11 in resnet_env17, 5.0 in the ironenv); a build with neither just logs.
    """
    restore = None
    try:
        if hasattr(cv2, "setLogLevel"):
            prev = cv2.getLogLevel() if hasattr(cv2, "getLogLevel") else 1
            cv2.setLogLevel(0)
            restore = lambda: cv2.setLogLevel(prev)  # noqa: E731
        else:
            from cv2.utils import logging as cvlog  # type: ignore
            prev = cvlog.getLogLevel()
            cvlog.setLogLevel(cvlog.LOG_LEVEL_SILENT)
            restore = lambda: cvlog.setLogLevel(prev)  # noqa: E731
    except Exception:
        restore = None
    try:
        yield
    finally:
        if restore is not None:
            try:
                restore()
            except Exception:
                pass


@dataclass
class OpenAttempt:
    index: int
    backend: str
    outcome: str
    seconds: float


@dataclass
class PropertyResult:
    name: str
    requested: float
    readback: float
    accepted: bool


class CameraManager:
    """Open the first webcam that yields a frame, trying (index, backend) pairs in order.

    Every attempt runs the constructor, ``isOpened`` and the first ``read`` in a
    daemon thread joined with ``open_timeout_s``; a backend that blocks (DSHOW on
    an IR sensor) is abandoned rather than waited for. ``attempts`` records every
    pair tried and ``properties`` every requested setting with its read-back.
    """

    def __init__(
        self,
        indices: Sequence[int] = (0, 1),
        backends: Optional[Sequence[Tuple[str, int]]] = None,
        open_timeout_s: float = 8.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        fourcc: Optional[str] = None,
        log: Callable[[str], None] = print,
    ):
        self.indices = tuple(int(i) for i in indices)
        self.backends = list(backends) if backends is not None else backend_table()
        self.open_timeout_s = float(open_timeout_s)
        self.width, self.height, self.fps, self.fourcc = width, height, fps, fourcc
        self.log = log
        self.attempts: List[OpenAttempt] = []
        self.properties: List[PropertyResult] = []
        self.backend: Optional[str] = None
        self.index: Optional[int] = None

    # -- opening -----------------------------------------------------------------
    def _try_open(self, index: int, api: int, name: str) -> Optional["cv2.VideoCapture"]:
        box = {}

        def worker():
            try:
                with quiet_opencv():
                    cap = cv2.VideoCapture(index, api)
                    opened = cap.isOpened()
                    frame_ok = False
                    if opened:
                        frame_ok, _ = cap.read()
                box["cap"], box["opened"], box["frame"] = cap, opened, frame_ok
            except Exception as ex:  # a backend that raises is just another failed attempt
                box["error"] = ex

        t = threading.Thread(target=worker, daemon=True, name=f"cam-open-{name}-{index}")
        t0 = time.perf_counter()
        t.start()
        t.join(self.open_timeout_s)
        dt = time.perf_counter() - t0

        if t.is_alive():
            outcome = f"no answer after {self.open_timeout_s:.0f}s, abandoned"
            cap = None
        elif "error" in box:
            outcome = f"raised {type(box['error']).__name__}: {box['error']}"
            cap = None
        elif box.get("frame"):
            outcome = "opened, first frame read"
            cap = box["cap"]
        else:
            outcome = "opened but no frame" if box.get("opened") else "not opened"
            cap = None
            stale = box.get("cap")
            if stale is not None:
                try:
                    stale.release()
                except Exception:
                    pass
        self.attempts.append(OpenAttempt(index, name, outcome, dt))
        self.log(f"[camera] index {index} via {name}: {outcome} ({dt:.2f}s)")
        return cap

    def open(self) -> Optional["cv2.VideoCapture"]:
        for index in self.indices:
            for name, api in self.backends:
                cap = self._try_open(index, api, name)
                if cap is None:
                    continue
                self.backend, self.index = name, index
                cap = self.configure(cap)
                if cap is not None:
                    return cap
        return None

    # -- properties --------------------------------------------------------------
    def set_property(self, cap, prop: int, value: float, name: str, tolerance: float = 0.5) -> PropertyResult:
        """``cap.set`` never raises and its return value is not trustworthy; the read-back decides."""
        with quiet_opencv():
            try:
                cap.set(prop, float(value))
            except Exception:
                pass
            try:
                readback = float(cap.get(prop))
            except Exception:
                readback = float("nan")
        accepted = readback == readback and abs(readback - float(value)) <= tolerance
        res = PropertyResult(name, float(value), readback, accepted)
        self.properties.append(res)
        verdict = "ok" if accepted else "refused, keeping the sensor default"
        self.log(f"[camera] {name}: requested {value:g} -> reports {readback:g} ({verdict})")
        return res

    def _read_with_timeout(self, cap, timeout_s: float) -> bool:
        box = {}

        def worker():
            try:
                ok, _ = cap.read()
                box["ok"] = bool(ok)
            except Exception:
                box["ok"] = False

        t = threading.Thread(target=worker, daemon=True, name="cam-verify-read")
        t.start()
        t.join(timeout_s)
        return bool(box.get("ok", False))

    def configure(self, cap):
        """Apply the requested properties; if frames stop afterwards, reopen with sensor defaults."""
        requested = any(v is not None for v in (self.fourcc, self.width, self.height, self.fps))
        if not requested:
            return cap
        if self.fourcc:
            code = cv2.VideoWriter_fourcc(*self.fourcc)
            self.set_property(cap, cv2.CAP_PROP_FOURCC, code, f"FOURCC {self.fourcc}", tolerance=0.0)
        if self.width:
            self.set_property(cap, cv2.CAP_PROP_FRAME_WIDTH, self.width, "FRAME_WIDTH")
        if self.height:
            self.set_property(cap, cv2.CAP_PROP_FRAME_HEIGHT, self.height, "FRAME_HEIGHT")
        if self.fps:
            self.set_property(cap, cv2.CAP_PROP_FPS, self.fps, "FPS")
        if self._read_with_timeout(cap, self.open_timeout_s):
            return cap
        self.log("[camera] frames stopped after the property changes; reopening with sensor defaults")
        try:
            cap.release()
        except Exception:
            pass
        api = dict(self.backends)[self.backend]
        return self._try_open(self.index, api, self.backend)

    def describe(self) -> str:
        return f"index {self.index} via {self.backend}" if self.backend is not None else "no camera"


class ThreadedGrabber:
    """Keeps reading the newest frame in the background so USB sensors never back up."""

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.failures = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name="cam-grab")
        self.thread.start()

    def _loop(self):
        while self.running:
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
            if ok and frame is not None:
                with self.lock:
                    self.frame, self.ok = frame, True
                self.failures = 0
            else:
                self.failures += 1
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if not self.ok or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass


class FrameSource:
    """One interface over a webcam (``CameraManager``), a video file or a still image."""

    def __init__(self, kind: str, label: str, reader, releaser, loop_video: bool = False):
        self.kind = kind
        self.label = label
        self._reader = reader
        self._releaser = releaser
        self.loop_video = loop_video

    @classmethod
    def from_arg(cls, source: str, manager: Optional[CameraManager] = None, loop_video: bool = False,
                 log: Callable[[str], None] = print) -> "FrameSource":
        text = str(source).strip()
        if text.replace(",", "").replace(" ", "").isdigit():
            indices = [int(t) for t in text.replace(" ", "").split(",") if t]
            mgr = manager if manager is not None else CameraManager(indices=indices, log=log)
            if manager is not None:
                mgr.indices = tuple(indices)
            cap = mgr.open()
            if cap is None:
                tried = "; ".join(f"{a.index}/{a.backend}: {a.outcome}" for a in mgr.attempts)
                raise RuntimeError(f"no webcam delivered a frame ({tried})")
            grabber = ThreadedGrabber(cap)
            return cls("camera", mgr.describe(), grabber.read, grabber.release)

        path = Path(text)
        if not path.exists():
            raise FileNotFoundError(f"source {path} does not exist")
        if path.suffix.lower() in IMAGE_SUFFIXES:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"OpenCV could not decode image {path}")
            return cls("image", str(path), lambda: (True, image.copy()), lambda: None)

        with quiet_opencv():
            cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video {path}")

        def read_video():
            ok, frame = cap.read()
            if not ok and loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            return ok, frame

        return cls("video", str(path), read_video, cap.release, loop_video=loop_video)

    def read(self):
        return self._reader()

    def release(self):
        try:
            self._releaser()
        except Exception:
            pass


# -- drawing ---------------------------------------------------------------------
def extract_box(det, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """Pixel x1, y1, x2, y2 for a YoloDetection (x0, y0, w, h in source pixels)."""
    if all(hasattr(det, a) for a in ("x0", "y0", "w", "h")):
        x1, y1 = float(det.x0), float(det.y0)
        x2, y2 = x1 + float(det.w), y1 + float(det.h)
    elif all(hasattr(det, a) for a in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = float(det.x1), float(det.y1), float(det.x2), float(det.y2)
    elif isinstance(det, (list, tuple)) and len(det) >= 4:
        x1, y1, x2, y2 = (float(v) for v in det[:4])
    else:
        return 0, 0, 0, 0
    if 0.0 <= max(x1, y1, x2, y2) <= 1.0:
        x1, x2, y1, y2 = x1 * frame_w, x2 * frame_w, y1 * frame_h, y2 * frame_h
    return (int(max(0, x1)), int(max(0, y1)), int(min(frame_w, x2)), int(min(frame_h, y2)))


def draw_detections(frame, detections, conf_thresh: float = 0.25) -> int:
    h, w = frame.shape[:2]
    count = 0
    for det in detections or ():
        score = float(getattr(det, "score", 0.0))
        if score < conf_thresh:
            continue
        count += 1
        x1, y1, x2, y2 = extract_box(det, w, h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 64), 2)
        tag = f"{getattr(det, 'class_name', 'object')} {score:.2f}"
        cv2.putText(frame, tag, (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 64), 1, cv2.LINE_AA)
    return count


def hud_line(g2g_ms: float, npu_ms: float, fps_loop: float, count: int, boxes_from: str) -> str:
    return (f"Ignition YOLOv8n | Device 0 | G2G {g2g_ms:5.2f} ms | NPU {npu_ms:5.2f} ms | "
            f"loop {fps_loop:6.1f} FPS | objects {count} | boxes: {boxes_from}")


def draw_hud(frame, line1: str, line2: str):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(frame, line1, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "q/ESC: quit   s: snapshot", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


# -- main ------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default="0,1",
                    help="webcam index or comma list to probe (default 0,1), a video file, or a still image")
    ap.add_argument("--model", default=str(ROOT / "build" / "yolov8n.ignite"))
    ap.add_argument("--boxes", choices=("auto", "npu", "oracle"), default="auto",
                    help="where boxes come from: NPU heads, the ONNX CPU oracle, or auto (NPU if present)")
    ap.add_argument("--headless", action="store_true", help="no window; print one HUD line per frame")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = until quit / end of file)")
    ap.add_argument("--loop", action="store_true", help="restart a video file at its end")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=None, help="request a capture width (verified by read-back)")
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--fps", type=float, default=None, help="request a capture rate (verified by read-back)")
    ap.add_argument("--fourcc", default=None, help="request a pixel format such as MJPG (verified by read-back)")
    ap.add_argument("--open-timeout", type=float, default=8.0, help="seconds per (index, backend) attempt")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from ignite_xdna.pipelines.yolo_pipeline import YoloPipeline

    print(f"[Ignition] Initializing bare-metal pipeline on Device 0 from {args.model} ...", flush=True)
    pipeline = YoloPipeline(model_path_or_bundle=args.model, device_index=0,
                            conf_thres=args.conf, iou_thres=args.iou)
    status = pipeline.session.head_status
    if args.boxes == "auto":
        boxes_mode = "npu" if status.present else "oracle"
    else:
        boxes_mode = args.boxes
    use_oracle = boxes_mode == "oracle"
    if use_oracle and pipeline._ort_cut_sess is None:
        print("[Ignition] the ONNX cut model is not available, so boxes: oracle is unavailable; using NPU heads",
              flush=True)
        use_oracle = False
        boxes_mode = "npu"
    print(f"[Ignition] NPU heads {'present' if status.present else 'absent'}: {status.reason}", flush=True)
    print(f"[Ignition] boxes: {boxes_mode}", flush=True)

    manager = CameraManager(open_timeout_s=args.open_timeout, width=args.width, height=args.height,
                            fps=args.fps, fourcc=args.fourcc)
    try:
        source = FrameSource.from_arg(args.source, manager=manager, loop_video=args.loop)
    except Exception as ex:
        pipeline.close()
        print(f"[Ignition] could not open source {args.source!r}: {ex}", flush=True)
        return 2
    print(f"[Ignition] source: {source.kind} {source.label}", flush=True)

    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    g2g, npu, loop_dt = [], [], []
    t_prev = time.perf_counter()
    frames_done = 0
    status_line = (f"NPU heads: {'present' if status.present else 'absent'}"
                   + ("" if status.present else f" ({status.egress_bytes} B egress, {status.declared_bytes} B declared)"))
    try:
        while True:
            if args.frames and frames_done >= args.frames:
                break
            ok, frame = source.read()
            if not ok or frame is None:
                if source.kind == "video":
                    break
                time.sleep(0.002)
                continue

            dets, timings = pipeline.predict_sync(frame, use_oracle_for_boxes=use_oracle)
            now = time.perf_counter()
            loop_dt.append(now - t_prev)
            t_prev = now
            fps = 1.0 / max(float(np.median(loop_dt[-30:])), 1e-6)
            g2g.append(timings.glass_to_glass_ms)
            npu.append(timings.npu_forward_ms)
            frames_done += 1

            count = draw_detections(frame, dets, args.conf)
            line1 = hud_line(timings.glass_to_glass_ms, timings.npu_forward_ms, fps, count, timings.head_source)
            if args.headless:
                print(f"[hud] frame {frames_done}{'/' + str(args.frames) if args.frames else ''} | {line1} | {status_line}",
                      flush=True)
                continue
            draw_hud(frame, line1, status_line)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                snap = ROOT / "outputs" / f"snapshot_{int(time.time())}.png"
                snap.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snap), frame)
                print(f"[Ignition] Snapshot saved: {snap}", flush=True)
    finally:
        source.release()
        if not args.headless:
            cv2.destroyAllWindows()
        pipeline.close()

    if manager.attempts:
        print("[summary] camera attempts: " + "; ".join(
            f"{a.index}/{a.backend} {a.outcome} {a.seconds:.2f}s" for a in manager.attempts), flush=True)
    if manager.properties:
        print("[summary] camera properties: " + "; ".join(
            f"{p.name} {p.requested:g}->{p.readback:g} {'ok' if p.accepted else 'refused'}" for p in manager.properties),
            flush=True)
    if g2g:
        arr = np.asarray(g2g)
        print(f"[summary] frames {frames_done} | source {source.kind} | boxes {boxes_mode} | "
              f"G2G mean {arr.mean():.3f} ms median {np.median(arr):.3f} p95 {np.percentile(arr, 95):.3f} | "
              f"NPU mean {np.mean(npu):.3f} ms | NPU heads {'present' if status.present else 'absent'}", flush=True)
    else:
        print(f"[summary] frames 0 | source {source.kind} delivered no frames", flush=True)
    print("[Ignition] Hardware context released.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
