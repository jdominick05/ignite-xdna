"""
Benchmark pipelined vs sequential execution of the MobileViT spliced pipeline:
Stage 1: NPU Cut CNN Backbone
Stage 2: CPU Attention (9 transformer blocks)

Measures:
  1. Sequential throughput (FPS) and latency (ms)
  2. Pipelined 2-stage overlapped throughput (FPS) and per-frame latency (ms)
"""

import time
import threading
import queue
import numpy as np
import torch
import onnxruntime as ort

model_path = "models/mobilevit_cut_backbone_xint8.onnx"
xclbin_path = r"C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\xclbins\phoenix\4x4.xclbin"
cache_dir = "modelcachekey"
cache_key = "mobilevit_cut"

po = {
    "config_file": "",
    "cacheDir": cache_dir,
    "cacheKey": cache_key,
    "enable_cache_file_io_in_mem": "0",
    "target": "X1",
    "xlnx_enable_py3_round": "0",
    "xclbin": xclbin_path,
}

sess = ort.InferenceSession(model_path, providers=["VitisAIExecutionProvider", "CPUExecutionProvider"], provider_options=[po, {}])
x_np = np.random.randn(1, 3, 256, 256).astype(np.float32)

# Stage 2 tensors
s2_q, s2_k, s2_v = torch.randn(16, 256, 16), torch.randn(16, 256, 16), torch.randn(16, 256, 16)
s3_q, s3_k, s3_v = torch.randn(16, 64, 20), torch.randn(16, 64, 20), torch.randn(16, 64, 20)
s4_q, s4_k, s4_v = torch.randn(16, 16, 24), torch.randn(16, 16, 24), torch.randn(16, 16, 24)

def run_cpu_attention():
    for _ in range(2):
        s = torch.bmm(s2_q, s2_k.transpose(1, 2)) * 0.25
        a = torch.softmax(s, dim=-1)
        _ = torch.bmm(a, s2_v)
    for _ in range(4):
        s = torch.bmm(s3_q, s3_k.transpose(1, 2)) * 0.2236
        a = torch.softmax(s, dim=-1)
        _ = torch.bmm(a, s3_v)
    for _ in range(3):
        s = torch.bmm(s4_q, s4_k.transpose(1, 2)) * 0.2041
        a = torch.softmax(s, dim=-1)
        _ = torch.bmm(a, s4_v)

# Warmup
for _ in range(5):
    _ = sess.run(None, {"input": x_np})
    run_cpu_attention()

N_FRAMES = 100

# --- 1. Sequential Benchmark ---
print(f"Running {N_FRAMES} frames SEQUENTIALLY...")
t0 = time.perf_counter()
seq_latencies = []
for _ in range(N_FRAMES):
    f_t0 = time.perf_counter()
    _ = sess.run(None, {"input": x_np})
    run_cpu_attention()
    seq_latencies.append((time.perf_counter() - f_t0) * 1000)
t_seq_total = time.perf_counter() - t0
seq_fps = N_FRAMES / t_seq_total
seq_latency = np.mean(seq_latencies)

print(f"Sequential: {seq_fps:.1f} FPS, {seq_latency:.2f} ms/frame")

# --- 2. Pipelined Benchmark ---
# Stage 1 (NPU) produces tokens -> Queue (maxsize=2) -> Stage 2 (CPU) consumes
pipe_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()
pipe_latencies = []

def cpu_worker():
    while not stop_event.is_set():
        try:
            item = pipe_queue.get(timeout=0.1)
            if item is None:
                break
            frame_id, frame_t0 = item
            run_cpu_attention()
            pipe_latencies.append((time.perf_counter() - frame_t0) * 1000)
            pipe_queue.task_done()
        except queue.Empty:
            continue

print(f"Running {N_FRAMES} frames PIPELINED (NPU Stage 1 || CPU Stage 2)...")
worker_thread = threading.Thread(target=cpu_worker)
worker_thread.start()

t0 = time.perf_counter()
for i in range(N_FRAMES):
    frame_t0 = time.perf_counter()
    _ = sess.run(None, {"input": x_np})
    pipe_queue.put((i, frame_t0))

pipe_queue.join()
stop_event.set()
pipe_queue.put(None)
worker_thread.join()

t_pipe_total = time.perf_counter() - t0
pipe_fps = N_FRAMES / t_pipe_total
pipe_latency = np.mean(pipe_latencies)

print(f"Pipelined:  {pipe_fps:.1f} FPS, {pipe_latency:.2f} ms/frame")
print(f"Throughput Speedup from Pipelining: {pipe_fps / seq_fps:.2f}x")
