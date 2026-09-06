"""Live webcam demo of the round-robin idea from RESEARCH.md's multi-partition
finding: does splitting 1x4.xclbin into N independent single-column HW
partitions (measured by tools/multi_partition_bench.py at 67.2 -> 245.2 fps,
3.65x at N=4, yolov8n) turn into a faster *live* demo than the single
4x4.xclbin session pipelines/yolov8n/4_detect.py uses?

Every worker is a separate OS process holding its own NPU session -- per
multi_partition_bench.py's finding, a separate process is what actually
claims a separate HW partition/column; threads on one process share the
same partition and do not parallelize this way.

yolov8n is the model here on purpose, not a bigger one: it is the model this
repo's 3.65x combined-throughput number was measured on, and at ~13 ms/call
solo it is fast enough that a webcam's own capture rate -- not NPU
throughput -- may be the thing actually capping the on-screen fps. That's a
real caveat this demo can surface, not a reason to pick a slower model to
hide it.

Frames are dropped, not queued, when a worker is still busy on its last one
(each worker's input queue is maxsize=1) -- keeps this a real-time demo
instead of a growing backlog, so the on-screen combined-fps figure is
genuine live throughput, not an average over a queue that will eventually
stall.

    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'
    python tools/webcam_multipartition_demo.py --workers 4 --source 0
        # q quits; defaults to yolov8n_cut_xint8.onnx on yolocut1x4cachekey

--workers must be <= 4 on this 4-column Phoenix chip. 1x4.xclbin is
deprecated by AMD since Ryzen AI 1.5 (still functions under the 1.7.1 install
this project depends on -- see multi_partition_bench.py's docstring for the
same caveat, not repeated here).
"""
import argparse
import collections
import multiprocessing as mp
import os
import queue
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

import npu.yolo as yc
from npu.session import build_session, clear_cache

XRT_SMI = r"C:\Windows\System32\AMD\xrt-smi.exe"
XCLBIN_1X4_SUBPATH = os.path.join("voe-4.0-win_amd64", "xclbins", "phoenix", "1x4.xclbin")
_PS_HINT = r"  $env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.7.1'"


def resolve_1x4_xclbin(explicit):
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"xclbin not found: {explicit}")
        return explicit
    install = os.environ.get("RYZEN_AI_INSTALLATION_PATH", "")
    if not install:
        raise SystemExit("RYZEN_AI_INSTALLATION_PATH is not set. In PowerShell:\n"
                         + _PS_HINT + "\nor pass --xclbin explicitly.")
    candidate = os.path.join(install, XCLBIN_1X4_SUBPATH)
    if not os.path.isfile(candidate):
        raise SystemExit(f"no xclbin at {candidate}\n" + _PS_HINT)
    return candidate


def raw_partitions():
    try:
        return subprocess.run([XRT_SMI, "examine", "-r", "aie-partitions"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        return str(e)


def active_count(raw):
    return raw.count("|Active")


def worker(rank, model, cache_key, xclbin, in_q, out_q, ready_flag, stop_event):
    """Runs in a fresh interpreter (spawn start method) -- own OS process,
    own NPU HW context/column. Pulls (seq, frame) off its own queue, infers,
    posts (seq, rank, dets, infer_ms) to the shared output queue."""
    import npu.yolo as yc
    from npu.yolo_decode import decode_heads, head_order

    sess = build_session(model, "npu", cache_key, xclbin, log_severity=3)
    inp = sess.get_inputs()[0].name
    imgsz = yc.input_size(sess.get_inputs()[0].shape, model)
    order = head_order(sess, imgsz)

    warm = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    x, _pad, _scale = yc.letterbox(warm, imgsz)
    sess.run(None, {inp: x})  # first-call warmup, not counted
    ready_flag.value = 1

    while not stop_event.is_set():
        try:
            seq, frame = in_q.get(timeout=0.5)
        except queue.Empty:
            continue
        t0 = time.perf_counter()
        x, pad, scale = yc.letterbox(frame, imgsz)
        r = sess.run(None, {inp: x})
        out = decode_heads([r[i] for i in order], imgsz=imgsz, conf_thres=0.25)
        dets = yc.postprocess(out, pad, scale, conf_thres=0.25, iou_thres=0.5)
        infer_ms = (time.perf_counter() - t0) * 1000
        try:
            out_q.put_nowait((seq, rank, dets, infer_ms))
        except queue.Full:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yolov8n_cut_xint8.onnx")
    ap.add_argument("--cache-key", default="yolocut1x4cachekey")
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--source", default="0")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--ready-timeout", type=float, default=60.0)
    args = ap.parse_args()

    xclbin = resolve_1x4_xclbin(args.xclbin)
    if args.fresh:
        clear_cache(args.cache_key)

    print(f"model: {args.model}\nxclbin: {xclbin}\nworkers: {args.workers}")
    print("pre-building one session to force the compile before any worker forks "
          "-- avoids N processes racing the same cache directory on first run.")
    _sess = build_session(args.model, "npu", args.cache_key, xclbin, log_severity=2)
    del _sess  # release this process's HW context before spawning workers
    print("compile cache warm.\n")

    n = args.workers
    ctx = mp.get_context("spawn")
    in_qs = [ctx.Queue(maxsize=1) for _ in range(n)]
    out_q = ctx.Queue(maxsize=n * 4)
    ready_flags = [ctx.Value("i", 0) for _ in range(n)]
    stop_event = ctx.Event()

    procs = [ctx.Process(target=worker, args=(
        i, args.model, args.cache_key, xclbin, in_qs[i], out_q, ready_flags[i], stop_event))
        for i in range(n)]
    for p in procs:
        p.start()

    print(f"waiting for {n} workers to build sessions and report ready...")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.ready_timeout:
        if all(f.value for f in ready_flags):
            break
        time.sleep(0.5)
    else:
        for p in procs:
            p.terminate()
        raise SystemExit(f"only {sum(f.value for f in ready_flags)}/{n} workers came up "
                         f"after {args.ready_timeout:.0f}s")

    active = active_count(raw_partitions())
    print(f"{n} workers ready. xrt-smi reports {active} active HW context(s) "
          f"(expect {n} for genuine separate-column parallelism).\n")

    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"could not open source {args.source}")
    # No explicit CAP_PROP_FRAME_WIDTH/HEIGHT request: on this machine, changing
    # resolution away from the camera's default measured at 178s on top of the
    # ~90s open itself (cam_probe.py) -- a driver-level renegotiation cost, not
    # anything NPU- or code-related. Native resolution avoids paying it; letterbox()
    # handles whatever size comes back.
    print(f"camera native size {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
          f"{cap.get(cv2.CAP_PROP_FPS):.1f} fps capture rate "
          "-- that, not the NPU, may be the real ceiling here.", flush=True)
    print("running - press q to quit", flush=True)

    frame_count = 0
    last_dets = []
    infer_ms_by_rank = {}
    completion_times = collections.deque()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            target = frame_count % n
            try:
                in_qs[target].put_nowait((frame_count, frame.copy()))
            except queue.Full:
                pass  # that worker is still busy on its last frame -- drop this one
            frame_count += 1

            while True:
                try:
                    _seq, rank, dets, infer_ms = out_q.get_nowait()
                except queue.Empty:
                    break
                last_dets = dets
                infer_ms_by_rank[rank] = infer_ms
                completion_times.append(time.perf_counter())

            now = time.perf_counter()
            while completion_times and now - completion_times[0] > 2.0:
                completion_times.popleft()
            combined_fps = len(completion_times) / 2.0

            yc.draw(frame, last_dets)
            avg_infer = (sum(infer_ms_by_rank.values()) / len(infer_ms_by_rank)
                        if infer_ms_by_rank else 0.0)
            cv2.putText(frame,
                        f"NPU x{n} round-robin  combined {combined_fps:5.1f} fps  "
                        f"per-worker ~{avg_infer:4.1f} ms",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow(f"yolov8n npu x{n} round-robin", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_event.set()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
