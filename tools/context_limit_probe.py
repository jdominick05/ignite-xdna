"""How many hardware contexts does Linux amdxdna grant one process? (Windows: 5, the sixth fails, DECISIONS.md.)

Opens hw_contexts on one xclbin until creation fails or MAX is reached, printing xrt-smi's partition table at the peak.
"""
import subprocess
import sys

sys.path.append("/opt/xilinx/xrt/python")
import pyxrt

MAX = 12
xclbin_path = sys.argv[1]
dev = pyxrt.device(0)
xb = pyxrt.xclbin(xclbin_path)
dev.register_xclbin(xb)
uuid = xb.get_uuid()
ctxs = []
err = None
for i in range(MAX):
    try:
        ctxs.append(pyxrt.hw_context(dev, uuid))
        print(f"context {i + 1}: ok", flush=True)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"context {i + 1}: FAILED {err}", flush=True)
        break
print(f"CONTEXTS_GRANTED {len(ctxs)} (cap probed {MAX}); first failure: {err}")
print(subprocess.run(["/opt/xilinx/xrt/bin/xrt-smi", "examine", "-r", "aie-partitions"],
                     capture_output=True, text=True).stdout)
del ctxs
