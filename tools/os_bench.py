#!/usr/bin/env python3
"""Linux vs Windows on one laptop: the same binaries, through the same harness, on either OS.

    python tools/os_bench.py bundle --out bench_bundle                     # Linux, in .venv-iron: needs IRON
    python tools/os_bench.py run --bundle bench_bundle --round linux1      # either OS: needs only pyxrt
    python tools/os_bench.py run --bundle bench_bundle --round win2 --reverse
    python tools/os_bench.py run --bundle bench_bundle --round dev --checks-only   # short counts, no timing claim
    python tools/os_bench.py compare results/os_compare

WHY ONE BUNDLE. A gap between two operating systems means something only if both ran the same
bytes. `bundle` compiles the replay designs once (mlir-aie 1.4.2, Peano a36c62b9) and copies the
containers beside them with a sha256 manifest; `run` refuses a bundle that does not match its
manifest, and `compare` refuses rounds made from different bundles. Every kernel is replayed
through plain pyxrt -- xclbin, hw_context, kernel(3, insts, n, bos...) -- so the Windows side needs
the XRT SDK's pyxrt and this repo, not IRON.

WHAT IS TIMED. Each workload runs in its own process, after a preflight that refuses to start
unless the NPU reports power mode default and holds no hardware context (and, outside
--checks-only, the host-load guard says CLEAR):
  passthrough   no-compute 32 KB shim->memtile->shim design: raw submit+wait, then pyxrt.runlist
                batches 1..64. Output checked before and after the timed calls.
  cpp_runner    kernels/dispatch_floor/dispatch_runner(.exe), built per OS, on the same xclbin/insts.
  clock         kernels/clock_probe's untraced scalar loop: submit+wait time fitted against loop
                length. The slope is the loop's cost per iteration; with the 9.000 cycles per
                iteration the Linux trace measured (results/linux/clock_probe_default_*.log) it is
                the core clock, the intercept being the per-dispatch cost. Checksum verified per call.
  e2e:<name>    Ignition's live_ignition.py --headless on assets/bus.jpg, one process per container,
                --npu-power off (the NPU mode stays default) and --power-mode balanced (the host's
                OpenMP settings), which records g2g and per-stage means.

`compare` calls an OS faster on a metric only if its WORST round beats the other OS's BEST round.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

CONTAINERS = ["yolov8n_full", "yolo11n", "sesr_m7", "yolov8s", "yolov8n_pose"]
WORKLOADS = ["passthrough", "cpp_runner", "clock"] + [f"e2e:{c}" for c in CONTAINERS]
PASSTHROUGH_N = 4096                       # int32 per buffer, the 32 KB payload of every dispatch-floor log
BATCHES = [1, 2, 4, 8, 16, 32, 64]
CLOCK_TARGETS = [1 << k for k in range(20, 26)]
# Scalar loop, trace-measured on Linux with the bundle's toolchain (cycles = 9 * iters + 4 at every length):
# results/linux/clock_probe_default_a36c62b9_laptop-linux_20261028T181724Z.log
CLOCK_CYCLES_PER_ITER = 9.0
COUNTS = {   # (timed, warmup) per workload; --checks-only uses the second set
    "timed": {"raw": (1000, 50), "runlist": (100, 10), "cpp": (1000, 50), "clock": (20, 3), "e2e": (500, 50)},
    "checks": {"raw": (20, 2), "runlist": (5, 1), "cpp": (20, 2), "clock": (3, 1), "e2e": (30, 5)},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    return {"n": int(v.size), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)), "min": float(v.min()), "max": float(v.max()),
            "stdev": float(v.std(ddof=1)) if v.size > 1 else 0.0}


# ----------------------------------------------------------------------------------------- bundle (Linux, IRON)

def cmd_bundle(a) -> int:
    sys.path.insert(0, str(ROOT / "kernels" / "dispatch_floor"))
    sys.path.insert(0, str(ROOT / "kernels" / "clock_probe"))
    import aie.iron as iron
    import measure_runlist as mr
    import clock_probe as cp

    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists; bundles are never rewritten -- choose a new --out")
    files = {}

    def put(src: Path, rel: str):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files[rel] = sha256(dst)

    xclbin, insts, _ = mr.compile_and_find_cache(PASSTHROUGH_N)       # verifies the passthrough on the device
    put(Path(xclbin), "passthrough/final.xclbin")
    put(Path(insts), "passthrough/insts.bin")

    a_buf = iron.zeros(cp.N_IO, dtype=np.int32, device="npu")
    c_buf = iron.zeros_like(a_buf)
    a_buf[0], a_buf[1] = cp.MODES["scalar"], CLOCK_TARGETS[0]
    with mr._capture_compiled_paths() as captured:
        cp.clock_probe(a_buf, c_buf)
    reason = cp.verify(cp.decode(c_buf.numpy()), cp.MODES["scalar"], CLOCK_TARGETS[0], None, False)
    if reason or "xclbin" not in captured:
        sys.exit(f"clock design failed on the device or its paths were not captured: {reason} {captured}")
    put(captured["xclbin"], "clock/final.xclbin")
    put(captured["insts"], "clock/insts.bin")

    for c in CONTAINERS:
        put(ROOT / "build" / f"{c}.ignite", f"containers/{c}.ignite")
    put(ROOT / "assets" / "bus.jpg", "assets/bus.jpg")

    from importlib.metadata import version
    manifest = {"schema": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "commit": git_commit(), "files": files,
                "toolchain": {"mlir_aie": version("mlir-aie"), "llvm_aie": version("llvm-aie")},
                "clock_cycles_per_iter": CLOCK_CYCLES_PER_ITER,
                "containers_source": "Hugging Face jdominick05/ignite-xdna, revision of tools/hf_manifest.json"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[bundle] {out}: {len(files)} files, manifest sha256 {sha256(out / 'manifest.json')}")
    return 0


def git_commit(repo: Path = ROOT) -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=repo).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def check_bundle(bundle: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    bad = [rel for rel, h in manifest["files"].items() if not (bundle / rel).is_file() or sha256(bundle / rel) != h]
    if bad:
        sys.exit(f"bundle {bundle} does not match its manifest: {bad}")
    return {"manifest_sha256": sha256(bundle / "manifest.json"), "commit": manifest["commit"],
            "toolchain": manifest["toolchain"], "cycles_per_iter": manifest["clock_cycles_per_iter"]}


# ----------------------------------------------------------------------------------------- witnesses

def xrt_smi(*args) -> str:
    from ignite_xdna.pipelines.npu_power import run_xrt_smi
    return run_xrt_smi(list(args))[1]


def power_state() -> dict:
    """AC and the host power policy, on either OS: what the pre-registration's conditions are checked against."""
    import psutil
    battery = psutil.sensors_battery()
    state = {"ac_online": None if battery is None else bool(battery.power_plugged)}
    if os.name == "nt":
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True).stdout.strip()
        state["power_scheme"] = out
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes") as k:
                # Settings' Power mode slider: an overlay GUID (Best performance is ded574b5-45a0-4f42-8737-46345c09c238).
                state["power_mode_overlay_ac"] = winreg.QueryValueEx(k, "ActiveOverlayAcPowerScheme")[0]
        except OSError as exc:
            state["power_mode_overlay_ac"] = f"unreadable: {exc}"
    else:
        import research_process as rp
        stack = rp.linux_stack()
        state.update({k: stack[k] for k in ("platform_profile", "cpu_epp", "cpu_governor")})
    return state


def power_refusals(state: dict) -> list:
    """Pre-registered conditions a timed workload must meet; empty when it may run."""
    refuse = []
    if state.get("ac_online") is False:
        refuse.append("on battery")
    if os.name != "nt" and state.get("platform_profile") != "performance":
        refuse.append(f"platform_profile {state.get('platform_profile')!r}, not performance (powerprofilesctl set performance)")
    if os.name == "nt" and state.get("power_mode_overlay_ac") != "ded574b5-45a0-4f42-8737-46345c09c238":
        refuse.append(f"Windows power mode overlay {state.get('power_mode_overlay_ac')!r}, not Best performance")
    return refuse


def witness() -> dict:
    import research_process as rp
    w = rp.machine()
    w["python"] = sys.version.split()[0]
    w["power"] = power_state()
    w["xrt_smi_version"] = xrt_smi("--version")
    w["xrt_smi_examine"] = xrt_smi("examine")
    w["xrt_smi_platform"] = xrt_smi("examine", "-r", "platform")
    try:
        import onnxruntime
        w["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass
    return json.loads(rp.scrub(json.dumps(w)))


def host_load(wait_s: float) -> dict:
    # -ExecutionPolicy Bypass: a machine left at the default policy refuses the script, and the verdict would read UNKNOWN.
    guard = (["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "tools/host_load.ps1")]
             if os.name == "nt" else [sys.executable, str(ROOT / "tools/host_load.py")])
    deadline = time.monotonic() + wait_s
    while True:
        out = subprocess.run(guard, capture_output=True, text=True, timeout=60).stdout
        verdict = next((l.split()[1] for l in out.splitlines() if l.startswith("HOST_LOAD_VERDICT")), "UNKNOWN")
        if verdict == "CLEAR" or time.monotonic() >= deadline:
            import research_process as rp
            return {"verdict": verdict, "lines": rp.scrub(out).splitlines()}
        time.sleep(2)


def npu_state() -> dict:
    from ignite_xdna.pipelines.npu_power import context_pids, read_mode
    return {"pmode": read_mode(), "context_pids": context_pids()}


# ----------------------------------------------------------------------------------------- pyxrt replay

def open_design(xclbin_path: Path, insts_path: Path):
    from ignite_xdna.runtime.driver import setup_xrt_environment
    setup_xrt_environment()
    import pyxrt
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(str(xclbin_path))
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kernel = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    insts = np.fromfile(insts_path, dtype=np.uint32)
    insts_bo = pyxrt.bo(dev, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
    insts_bo.write(insts.tobytes(), 0)
    insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    # Instruction count in uint32 words, as IRON's own host path and measure_runlist.py pass it.
    return pyxrt, dev, ctx, kernel, insts_bo, len(insts)


def run_passthrough(bundle: Path, counts) -> dict:
    pyxrt, dev, ctx, kernel, insts_bo, n_words = open_design(bundle / "passthrough/final.xclbin",
                                                            bundle / "passthrough/insts.bin")
    to_dev, from_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    expected = np.arange(1, PASSTHROUGH_N + 1, dtype=np.int32)

    def buffers():
        bos = [pyxrt.bo(dev, PASSTHROUGH_N * 4, pyxrt.bo.host_only, kernel.group_id(g)) for g in (3, 4, 5)]
        bos[0].write(expected.tobytes(), 0)
        bos[0].sync(to_dev)
        bos[2].write(np.zeros_like(expected).tobytes(), 0)
        bos[2].sync(to_dev)
        return bos

    def ok(bo_c):
        bo_c.sync(from_dev)
        return bool(np.array_equal(np.frombuffer(bo_c.read(PASSTHROUGH_N * 4, 0), dtype=np.int32), expected))

    iters, warmup = counts["raw"]
    a, b, c = buffers()
    for _ in range(warmup):
        kernel(3, insts_bo, n_words, a, b, c).wait()
    verified_before = ok(c)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        kernel(3, insts_bo, n_words, a, b, c).wait()
        times.append((time.perf_counter() - t0) * 1e6)
    result = {"raw_us": stats(times), "raw_verified": verified_before and ok(c), "runlist": {}}

    iters, warmup = counts["runlist"]
    for n in BATCHES:
        sets = [buffers() for _ in range(n)]
        for _ in range(warmup):
            kernel(3, insts_bo, n_words, *sets[0]).wait()
        times = []
        for _ in range(iters):
            rl = pyxrt.runlist(ctx)
            runs = []
            for s in sets:
                run = pyxrt.run(kernel)     # unstarted: kernel(...) would start it before the runlist does
                for i, arg in enumerate((3, insts_bo, n_words, *s)):
                    run.set_arg(i, arg)
                rl.add(run)
                runs.append(run)
            t0 = time.perf_counter()
            rl.execute()
            rl.wait()
            times.append((time.perf_counter() - t0) * 1e6)
            del runs, rl                     # freed before the next list, never at interpreter exit
        result["runlist"][str(n)] = {"total_us": stats(times), "per_dispatch_us_mean": statistics.mean(times) / n,
                                     "verified": all(ok(s[2]) for s in sets)}
        del sets
    del kernel, ctx
    return result


def run_clock(bundle: Path, counts, cycles_per_iter: float) -> dict:
    pyxrt, dev, ctx, kernel, insts_bo, n_words = open_design(bundle / "clock/final.xclbin", bundle / "clock/insts.bin")
    to_dev, from_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    n_io = 64
    bo_a = pyxrt.bo(dev, n_io * 4, pyxrt.bo.host_only, kernel.group_id(3))
    bo_c = pyxrt.bo(dev, n_io * 4, pyxrt.bo.host_only, kernel.group_id(4))
    iters, warmup = counts["clock"]
    points, failures = [], []
    for target in CLOCK_TARGETS:
        params = np.zeros(n_io, dtype=np.int32)
        params[0], params[1] = 0, target                       # mode 0 = scalar loop
        bo_a.write(params.tobytes(), 0)
        bo_a.sync(to_dev)
        times = []
        for i in range(warmup + iters):
            bo_c.write(np.zeros(n_io, dtype=np.int32).tobytes(), 0)
            bo_c.sync(to_dev)
            t0 = time.perf_counter()
            kernel(3, insts_bo, n_words, bo_a, bo_c).wait()
            dt = (time.perf_counter() - t0) * 1e3
            bo_c.sync(from_dev)
            out = np.frombuffer(bo_c.read(n_io * 4, 0), dtype=np.int32)
            want = (target * (target - 1) // 2) & 0xFFFFFFFF
            if not (int(out[0]) == 0 and int(out[1]) & 0xFFFFFFFF == target and int(out[4]) & 0xFFFFFFFF == want):
                failures.append({"target": target, "call": i, "out": out[:5].tolist()})
            elif i >= warmup:
                times.append(dt)
        points.append({"iterations": target, "ms": stats(times) if times else None})
    good = [(p["iterations"], p["ms"]["mean"]) for p in points if p["ms"]]
    xs, ys = np.array([g[0] for g in good], float), np.array([g[1] for g in good], float)
    slope, intercept = np.polyfit(xs, ys, 1)
    r2 = 1 - ((ys - (slope * xs + intercept)) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    del kernel, ctx
    return {"points": points, "failures": failures, "slope_ms_per_iter": float(slope), "intercept_ms": float(intercept),
            "r2": float(r2), "cycles_per_iter": cycles_per_iter,
            "clock_ghz": float(cycles_per_iter / (slope * 1e6)) if slope > 0 else float("nan")}


def run_cpp(bundle: Path, counts) -> dict:
    exe = ROOT / "kernels" / "dispatch_floor" / ("dispatch_runner.exe" if os.name == "nt" else "dispatch_runner")
    if not exe.is_file():
        return {"skipped": f"{exe.name} not built (scripts/build_dispatch_runner.{'bat' if os.name == 'nt' else 'sh'})"}
    iters, warmup = counts["cpp"]
    p = subprocess.run([str(exe), "--xclbin", str(bundle / "passthrough/final.xclbin"),
                        "--insts", str(bundle / "passthrough/insts.bin"), "--payload", str(PASSTHROUGH_N),
                        "--iters", str(iters), "--warmup", str(warmup), "--batch-sizes", ",".join(map(str, BATCHES))],
                       capture_output=True, text=True, timeout=900)
    rows = []
    for line in p.stdout.splitlines():
        f = line.split()
        if len(f) == 6 and f[1].isdigit() and f[5] in ("yes", "NO"):
            rows.append({"mode": f[0], "batch": int(f[1]), "mean_us": float(f[2]), "min_us": float(f[3]),
                         "median_us": float(f[4]), "verified": f[5] == "yes"})
    return {"exit_code": p.returncode, "rows": rows, "stdout": p.stdout[-6000:], "stderr": p.stderr[-2000:]}


def run_e2e(bundle: Path, name: str, counts, ignition: Path, out_dir: Path) -> dict:
    frames, warmup = counts["e2e"]
    js = out_dir / f"live_{name}.json"
    cmd = [sys.executable, str(ignition), "--model", str(bundle / "containers" / f"{name}.ignite"),
           "--source", str(bundle / "assets/bus.jpg"), "--headless", "--warmup", str(warmup), "--frames", str(frames),
           "--npu-power", "off", "--power-mode", "balanced", "--json", str(js)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=ROOT)
    record = json.loads(js.read_text()) if js.is_file() else None
    summary = [l for l in p.stdout.splitlines() if l.startswith("[summary]")]
    return {"exit_code": p.returncode, "record": record, "summary": summary, "stderr": p.stderr[-2000:]}


# ----------------------------------------------------------------------------------------- run

def cmd_one(a) -> int:
    """One workload in this process; writes its result JSON. Called by `run`, one process per workload."""
    bundle = Path(a.bundle)
    counts = COUNTS["checks" if a.checks_only else "timed"]
    info = check_bundle(bundle)
    if a.workload == "passthrough":
        result = run_passthrough(bundle, counts)
    elif a.workload == "clock":
        result = run_clock(bundle, counts, info["cycles_per_iter"])
    elif a.workload == "cpp_runner":
        result = run_cpp(bundle, counts)
    else:
        result = run_e2e(bundle, a.workload.split(":", 1)[1], counts, Path(a.ignition), Path(a.out).parent)
    Path(a.out).write_text(json.dumps(result, indent=1, allow_nan=True) + "\n")
    return 0


def cmd_run(a) -> int:
    bundle = Path(a.bundle)
    info = check_bundle(bundle)
    out_dir = Path(a.results) / a.round
    if out_dir.exists():
        sys.exit(f"{out_dir} exists; rounds are never rewritten -- choose a new --round")
    out_dir.mkdir(parents=True)
    order = list(reversed(WORKLOADS)) if a.reverse else list(WORKLOADS)
    if a.only:
        order = [w for w in order if w in a.only.split(",")]
    meta = {"schema": 1, "round": a.round, "os": platform.system(), "checks_only": a.checks_only,
            "order": order, "bundle": info, "code_commit": git_commit(),
            "ignition_commit": git_commit(Path(a.ignition).resolve().parent), "witness": witness(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "round.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[os_bench] round {a.round} on {meta['witness']['machine']}: {', '.join(order)}", flush=True)
    for w in order:
        pre = {"npu": npu_state(), "host_load": host_load(0 if a.checks_only else 60), "power": power_state()}
        refuse = []
        if pre["host_load"]["verdict"] not in ("CLEAR", "BUSY", "PEER"):
            refuse.append(f"host-load guard gave no verdict ({pre['host_load']['verdict']})")   # a broken guard, even in a smoke
        for reason in power_refusals(pre["power"]):
            if a.checks_only:
                print(f"[os_bench] WARNING (a timed round would refuse): {reason}", flush=True)
            else:
                refuse.append(reason)
        if pre["npu"]["pmode"] != "default":
            refuse.append(f"NPU power mode {pre['npu']['pmode']!r}, not default")
        if pre["npu"]["context_pids"] != []:
            refuse.append(f"NPU not idle: contexts {pre['npu']['context_pids']}")
        if not a.checks_only and pre["host_load"]["verdict"] != "CLEAR" and not any("guard" in r for r in refuse):
            refuse.append(f"host load {pre['host_load']['verdict']}")
        tag = w.replace(":", "_")
        record = {"round": a.round, "workload": w, "os": meta["os"], "bundle_manifest_sha256": info["manifest_sha256"],
                  "preflight": pre}
        if refuse:
            record.update(refused=refuse, timing_eligible=False)
        else:
            res_file = out_dir / f"{tag}.result.json"
            t0 = time.monotonic()
            p = subprocess.run([sys.executable, __file__, "_one", "--bundle", str(bundle), "--workload", w,
                                "--out", str(res_file), "--ignition", a.ignition] + (["--checks-only"] if a.checks_only else []),
                               cwd=ROOT)
            record["elapsed_s"] = time.monotonic() - t0
            record["exit_code"] = p.returncode
            record["result"] = json.loads(res_file.read_text()) if res_file.is_file() else None
            if res_file.is_file():
                res_file.unlink()
            record["postflight"] = {"npu": npu_state()}
            record["timing_eligible"] = (not a.checks_only and p.returncode == 0 and record["result"] is not None
                                         and record["postflight"]["npu"]["pmode"] == "default")
        (out_dir / f"{tag}.json").write_text(json.dumps(record, indent=1, allow_nan=True) + "\n")
        print(f"[os_bench] {w}: [host {pre['host_load']['verdict']}] {headline(record)}", flush=True)
    return 0


def headline(record: dict) -> str:
    if record.get("refused"):
        return "REFUSED " + "; ".join(record["refused"])
    r, w = record.get("result") or {}, record["workload"]
    try:
        if w == "passthrough":
            rl = r["runlist"]
            return (f"raw {r['raw_us']['mean']:.1f} us (verified {r['raw_verified']}) | runlist N=64 "
                    f"{rl['64']['per_dispatch_us_mean']:.1f} us/dispatch (verified {rl['64']['verified']})")
        if w == "clock":
            return (f"{r['clock_ghz']:.4f} GHz, intercept {r['intercept_ms'] * 1e3:.1f} us, R^2 {r['r2']:.6f}, "
                    f"{len(r['failures'])} failed calls")
        if w == "cpp_runner":
            if "skipped" in r:
                return "skipped: " + r["skipped"]
            single = next((x for x in r["rows"] if x["batch"] == 1), None)
            return f"exit {r['exit_code']}, {len(r['rows'])} rows, first batch-1 mean {single['mean_us'] if single else '?'} us"
        g = r["record"]["g2g_ms"]
        s = r["record"]["stages_ms"]
        return (f"g2g mean {g['mean']:.3f} ms p50 {g['p50']:.3f} p95 {g['p95']:.3f} | pre {s['preprocess']:.3f} "
                f"net {s['network']:.3f} post {s['postprocess']:.3f}")
    except (KeyError, TypeError, StopIteration) as exc:
        return f"exit {record.get('exit_code')}, no headline ({type(exc).__name__}: {exc})"


# ----------------------------------------------------------------------------------------- compare

def metrics(record: dict) -> dict:
    """Flat {metric: value} of one workload record, lower is better except the clock."""
    r, w, m = record.get("result") or {}, record["workload"], {}
    if w == "passthrough":
        m["passthrough raw mean (us)"] = r["raw_us"]["mean"]
        for n, x in r["runlist"].items():
            m[f"passthrough runlist N={n} (us/dispatch)"] = x["per_dispatch_us_mean"]
    elif w == "cpp_runner":
        for x in r.get("rows", []):
            m[f"C++ {x['mode']} N={x['batch']} mean (us)"] = x["mean_us"]
    elif w == "clock":
        m["NPU clock (GHz)"] = r["clock_ghz"]
        m["clock-design dispatch intercept (us)"] = r["intercept_ms"] * 1e3
    elif r.get("record") and "g2g_ms" in r["record"]:
        name, rec = w.split(":", 1)[1], r["record"]
        m[f"{name} g2g mean (ms)"] = rec["g2g_ms"]["mean"]
        m[f"{name} g2g p95 (ms)"] = rec["g2g_ms"]["p95"]
        for stage, v in rec["stages_ms"].items():
            m[f"{name} {stage} (ms)"] = v
    return m


def cmd_compare(a) -> int:
    rounds = {}
    for meta_file in sorted(Path(a.results).glob("*/round.json")):
        meta = json.loads(meta_file.read_text())
        if meta["checks_only"] and not a.include_checks:
            continue
        # Workload records only: live_<name>.json beside them is Ignition's own record, kept for the evidence.
        recs = [json.loads(p.read_text()) for p in meta_file.parent.glob("*.json")
                if p.name != "round.json" and not p.name.startswith("live_")]
        rounds[meta["round"]] = (meta, [r for r in recs if r.get("timing_eligible") or a.include_checks])
    bundles = {m["bundle"]["manifest_sha256"] for m, _ in rounds.values()}
    if len(bundles) > 1:
        sys.exit(f"rounds come from {len(bundles)} different bundles; refusing to compare: {sorted(bundles)}")
    code = {(m["code_commit"], m.get("ignition_commit")) for m, _ in rounds.values()}
    if len(code) > 1 and not a.include_checks:
        sys.exit(f"rounds ran different code (ignite-xdna, Ignition): {sorted(code)}; refusing to compare")
    table = {}   # metric -> os -> [(round, value)]
    for name, (meta, recs) in rounds.items():
        for rec in recs:
            for k, v in metrics(rec).items():
                table.setdefault(k, {}).setdefault(meta["os"], []).append((name, v))
    oses = sorted({m["os"] for m, _ in rounds.values()})
    print(f"rounds: {', '.join(f'{n} ({m['os']})' for n, (m, _) in rounds.items())} | bundle {next(iter(bundles), '-')[:16]}")
    print(f"| metric | {' | '.join(oses)} | Linux / Windows | verdict |")
    print(f"|---|{'---|' * len(oses)}---|---|")
    for k in sorted(table):
        cells, means = [], {}
        for o in oses:
            vals = table[k].get(o, [])
            means[o] = [v for _, v in vals]
            cells.append(" / ".join(f"{v:.4g}" for _, v in vals) or "-")
        verdict, ratio = "-", "-"
        if "Linux" in means and "Windows" in means and means["Linux"] and means["Windows"]:
            lo_is_better = "GHz" not in k
            L, W = means["Linux"], means["Windows"]
            ratio = f"{statistics.mean(L) / statistics.mean(W):.3f}"
            if lo_is_better:
                verdict = "Linux" if max(L) < min(W) else "Windows" if max(W) < min(L) else "not separated"
            else:
                verdict = "Linux higher" if min(L) > max(W) else "Windows higher" if min(W) > max(L) else "not separated"
        print(f"| {k} | {' | '.join(cells)} | {ratio} | {verdict} |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True)
    r.add_argument("--round", required=True)
    r.add_argument("--results", default=str(ROOT / "results" / "os_compare"))
    r.add_argument("--reverse", action="store_true", help="run the workloads in reverse order (the second round)")
    r.add_argument("--only", default=None, help="comma-separated subset of " + ",".join(WORKLOADS))
    r.add_argument("--checks-only", action="store_true", help="short counts, no host-load gate, never timing-eligible")
    r.add_argument("--ignition", default=str(ROOT.parent / "Ignition" / "live_ignition.py"))
    o = sub.add_parser("_one")
    o.add_argument("--bundle", required=True)
    o.add_argument("--workload", required=True, choices=WORKLOADS)
    o.add_argument("--out", required=True)
    o.add_argument("--ignition", required=True)
    o.add_argument("--checks-only", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("results", nargs="?", default=str(ROOT / "results" / "os_compare"))
    c.add_argument("--include-checks", action="store_true", help="include --checks-only rounds (development only)")
    a = ap.parse_args()
    return {"bundle": cmd_bundle, "run": cmd_run, "_one": cmd_one, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
