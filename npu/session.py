"""ONNX Runtime session construction for CPU and the Ryzen AI NPU.

One implementation for both pipelines. The provider options below are the
verified working set for Phoenix / Hawk Point (XDNA1); see docs/DECISIONS.md
"LOCKED DECISIONS" before changing any of them.
"""
import os
import shutil

import onnxruntime as ort

from .paths import CACHE_DIR

# Ryzen AI 1.8.0 ships no Phoenix xclbin. 1.7.1 does, at this path.
XCLBIN_SUBPATH = os.path.join("voe-4.0-win_amd64", "xclbins", "phoenix", "4x4.xclbin")

_PS_HINT = r"  $env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.7.1'"


def resolve_xclbin(explicit=None):
    """Find the Phoenix 4x4 xclbin, or raise.

    This never returns None. Letting the driver resolve firmware does not work
    on Phoenix: with no xclbin the compiler targets AMD_AIE2P_4x4_Overlay
    (Strix) and the run either DPU-timeouts or quietly falls back to CPU. Such
    a run has no diagnostic value, so refusing beats warning-and-continuing.
    """
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"xclbin not found: {explicit}")
        return explicit
    install = os.environ.get("RYZEN_AI_INSTALLATION_PATH", "")
    if not install:
        raise SystemExit(
            "RYZEN_AI_INSTALLATION_PATH is not set, so the Phoenix xclbin cannot be "
            "found. In PowerShell:\n" + _PS_HINT + "\nor pass --xclbin explicitly.")
    candidate = os.path.join(install, XCLBIN_SUBPATH)
    if not os.path.isfile(candidate):
        raise SystemExit(
            f"no xclbin at {candidate}\n"
            "1.8.0 ships none; only the 1.7.1 install has the Phoenix 4x4 xclbin. "
            "Point RYZEN_AI_INSTALLATION_PATH at 1.7.1:\n" + _PS_HINT +
            "\nor pass --xclbin.")
    return candidate


def clear_cache(cache_key):
    """Delete a stale compile cache. Do this whenever the model or xclbin
    changes: the cache is keyed by cache_key alone, so a stale entry is reused
    silently and produces wrong-architecture artifacts."""
    path = os.path.join(str(CACHE_DIR), cache_key)
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"deleted stale cache {path}")


def build_session(model, ep, cache_key, xclbin=None, log_severity=1):
    """Build an InferenceSession on the CPU EP or the VitisAI (NPU) EP.

    log_severity is passed straight to ORT: 0 = verbose (per-node EP
    assignment), 1 = info (compile log), 2 = warning only.
    """
    so = ort.SessionOptions()
    so.log_severity_level = log_severity

    if ep == "cpu":
        return ort.InferenceSession(str(model), sess_options=so,
                                    providers=["CPUExecutionProvider"])

    # The EP writes its report only when this is set. scripts/lib.sh exports it,
    # but a bare `python pipelines/.../4_detect.py --ep npu` from the shell does
    # not -- and then diag.sh silently skips that cache, because the report it
    # keys off never got written. setdefault, so an explicit export still wins.
    os.environ.setdefault("XLNX_ONNX_EP_REPORT_FILE", "vitisai_ep_report.json")

    opts = {
        "cacheDir": str(CACHE_DIR),
        "cacheKey": cache_key,
        "enable_cache_file_io_in_mem": "0",  # INT8 disk cache + op report
        "target": "X1",                      # PHX/HPT legacy INT8 backend
        "xlnx_enable_py3_round": "0",        # from AMD's 1.7 PHX example
        "xclbin": resolve_xclbin(xclbin),
    }

    print(f"RYZEN_AI_INSTALLATION_PATH = {os.environ.get('RYZEN_AI_INSTALLATION_PATH')}")
    print(f"xclbin: {opts['xclbin']}")
    print(f"onnxruntime {ort.__version__}  providers: {ort.get_available_providers()}")
    print(f"cache:  {os.path.join(str(CACHE_DIR), cache_key)}")
    print("First run compiles the model. It takes a while and looks frozen. It isn't.")

    sess = ort.InferenceSession(str(model), sess_options=so,
                                providers=["VitisAIExecutionProvider"],
                                provider_options=[opts])
    # Registration is necessary but not sufficient: the EP can register and
    # still claim zero nodes (that is exactly the yolov8n blocker). Check
    # <cacheKey>/vitisai_ep_report.json with tools/diag_ep.py for what it took.
    if "VitisAIExecutionProvider" not in sess.get_providers():
        raise SystemExit("VitisAI EP did not register; session providers are "
                         f"{sess.get_providers()}. This run would be pure CPU.")
    return sess
