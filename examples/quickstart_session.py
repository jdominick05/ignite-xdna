"""High-Level ignite-xdna InferenceSession Quickstart."""
import numpy as np
from ignite_xdna import InferenceSession

# 1. Initialize hardware session on AMD Phoenix NPU (Device 0)
with InferenceSession("build/layer_conv0_exec.bin", device_index=0) as session:
    # 2. Ingress activations (NumPy array or PyTorch CPU tensor)
    input_data = np.random.randint(-128, 127, size=(64, 32), dtype=np.int8)

    # 3. Synchronous inference execution returning unswizzled channels
    output = session.run(input_data)
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")

    # 4. Measure sustained hardware throughput and latency
    metrics = session.benchmark(warmup=50, iterations=500)
    print(f"Sustained NPU Latency: {metrics['median_us']:.2f} us ({metrics['fps']:.1f} FPS)")
