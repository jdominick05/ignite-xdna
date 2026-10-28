# The Windows half of the Linux vs Windows comparison

The runbook for rounds `win1` and `win2` of
[the pre-registered comparison](../results/os_compare/prereg_os_compare_laptop_20261028.md). Run it on the
**laptop's** Windows install (Ryzen 5 8645HS), not on Desktop 2. Everything here is PowerShell. Linux rounds
`linux1` and `linux2` bracket these two.

## What you need

| | Where it comes from | Check |
|---|---|---|
| The `testing` branch at the pre-registration commit | `git fetch` + `git checkout testing` once it is pushed, or `git clone ignite-xdna.gitbundle -b testing ignite-xdna` from the transfer drive | `git log -1` shows the commit the pre-registration names |
| Ignition at `ebfc077` | `git clone https://github.com/jdominick05/Ignition` beside `ignite-xdna`, then `git checkout ebfc077` | `..\Ignition\live_ignition.py` exists |
| The bundle | copy `os_bench_bundle_20261028\` from the transfer drive, e.g. to `C:\os_bench_bundle_20261028` | `os_bench run` checks every file against `manifest.json` and refuses on a mismatch |
| pyxrt | the XRT SDK 2.21.75 at `C:\Xilinx\XRT\xrt_sdk\xrt` (the version Desktop 2 uses). `runtime\driver.py` finds `python\pyxrt*.pyd` there | `python -c "from ignite_xdna.runtime.driver import setup_xrt_environment as s; s(); import pyxrt; print(pyxrt.device(0))"` |
| Python | the version the SDK's `pyxrt` was built for (3.13 on Desktop 2). Make a conda env from `environment.yml`'s runtime pins: numpy 2.5.3, onnx 1.22.0, onnxruntime 1.30.0, opencv-python 5.0.0.93, ml-dtypes, pillow, psutil, click. mlir-aie is **not** needed. Then `pip install --no-deps -e .` in `ignite-xdna` and in `Ignition` | the pyxrt check above |
| MSVC Build Tools (optional) | `scripts\build_dispatch_runner.bat` builds `kernels\dispatch_floor\dispatch_runner.exe` | without it, the `cpp_runner` workload records `skipped` and the C++ rows have no Windows side |

## Before the timed rounds
1. **Smoke round**, into a scratch folder. This is not part of the sitting:
   ```powershell
   conda activate <env>
   cd ignite-xdna
   python tools\os_bench.py run --bundle C:\os_bench_bundle_20261028 --round smoke --checks-only --results $env:TEMP\os_smoke
   ```
   Every workload should print a headline, not `REFUSED` or `no headline`. The `clock` line should read about
   1.80 GHz. If it reads well away from that, stop: the comparison would be measuring the clock.
2. Record the stack:
   ```powershell
   C:\Windows\System32\AMD\xrt-smi.exe examine
   ```
   The round also stores this output, including the NPU firmware.

## The two timed rounds
1. **Fresh boot**, on AC power. Set Settings → System → Power → Power mode to **Best performance**. Close browsers,
   IDEs and anything else that is open.
2. Run the rounds back to back, and **touch nothing while they run**. Each takes about 10 minutes.
   ```powershell
   python tools\os_bench.py run --bundle C:\os_bench_bundle_20261028 --round win1
   python tools\os_bench.py run --bundle C:\os_bench_bundle_20261028 --round win2 --reverse
   ```
3. If a workload prints `REFUSED`, or a round stops early:
   - re-sit that round in full under a new name (`win1b`);
   - keep the incomplete folder.
4. Copy `results\os_compare\win1\` and `win2\` (and any re-sits) back to the transfer drive. On Linux they go into
   `results/os_compare/`, then `linux2`, then `python tools/os_bench.py compare`.

## Footguns
- **The NPU power mode must stay `default`.** `os_bench` runs Ignition with `--npu-power off`, and refuses a workload
  if `xrt-smi` reports any other mode. If an earlier tool left it lowered, run
  `xrt-smi configure --pmode default` first.
- **Don't run the rounds from Git Bash.** PowerShell's `host_load.ps1` is the host-load guard. Git Bash's mintty
  works, but keep one shell for both rounds.
- **PowerShell's `*>` writes UTF-16.** The rounds write their own JSON, so no redirection is needed.
