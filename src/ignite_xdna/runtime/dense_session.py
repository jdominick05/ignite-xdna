# SPDX-License-Identifier: AGPL-3.0-or-later
"""Named dense outputs from hybrid graph containers, with separate stage timings."""
import time
import numpy as np

from .graph_session import EngineSession, read_boundary, write_boundary


class DenseTensorSession(EngineSession):
    def __init__(self, container_path, device_index=0, **kwargs):
        # Validate the task before acquiring a device context.
        from ignite_xdna.compiler.serializer import IgniteModelReader
        with IgniteModelReader(container_path) as reader:
            if reader.manifest.get("task") not in ("segment", "matte"):
                raise ValueError("DenseTensorSession requires a segment or matte container")
        super().__init__(container_path, device_index=device_index, **kwargs)
        m = self.ignite_manifest
        self.output = m["dense_output"]
        self.input = {"name": m.get("input_name", self.ge["input_tensor"]),
                      "tensor": self.ge["input_tensor"], "shape": m["input_shape"],
                      "dtype": m["input_dtype"], "zero_point": m["quant_scales"]["input_zero_point"],
                      "storage": self.input_placement.get("storage", "workspace")}

    def run(self, x):
        if self._closed:
            raise RuntimeError("DenseTensorSession is closed")
        t0 = time.perf_counter()
        self._host_values.clear()
        if self.input["dtype"] == "uint8" and np.asarray(x).dtype == np.float32:
            from ignite_xdna.compiler.graph_reference import quantize_input
            q = self.ignite_manifest["quant_scales"]
            x = quantize_input(x, q["input_scale"], q["input_zero_point"])
        write_boundary(self, self.input, x)
        t1 = time.perf_counter()
        self.dispatch()
        t2 = time.perf_counter()
        y = read_boundary(self, self.output)
        if self.output["dtype"] == "uint8":
            y = (y.astype(np.float32) - self.output["zero_point"]) * self.output["scale"]
        t3 = time.perf_counter()
        return y, {"stage_ms": (t1-t0)*1e3, "npu_ms": self.last_dispatch_ms,
                   "host_ms": sum(s.last_cpu_ms for s in self._host_steps),
                   "transfer_ms": (t1-t0+t3-t2)*1e3 + sum(s.last_transfer_ms for s in self._host_steps),
                   "readback_ms": (t3-t2)*1e3, "total_ms": (t3-t0)*1e3}
