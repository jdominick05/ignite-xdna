#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hybrid stack, S0: the GPU runtimes at their latest supported versions, pinned. The light steps only: nothing
here installs, runs a downloaded binary or computes on a chip.

    python tools/hybrid_s0.py baseline          # re-read the drivers, Vulkan, the HIP runtime, the NPU driver,
                                                # the toolchains and the disk, against the plan's v1 read
    python tools/hybrid_s0.py assets            # the asset list, read from the publishers' APIs: URL, size,
                                                # checksum (or "pinned only"), the source and the read time
    python tools/hybrid_s0.py fetch COMPONENT   # download one component's assets into scratch/rt/, verified

Standard library only. Sizes are printed in bytes and in GB = 1e9 bytes. The plan is
scratch/llm/hybrid_s0_plan_draft.md (v2, approved 2026-09-24).
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import winreg
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / "scratch/rt"
RESULTS = ROOT / "results/llm"
SYSTEM32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
FLOOR_GB = 100.0
DRIVE = "C:\\"

# The plan's v1 read (2026-09-24, before anything changed). v1's disk figures were PowerShell's /1GB, which is
# GiB: 573.40 GiB free is 615.68 GB.
V1 = {
    "radeon_driver_version": "32.0.31041.1004",
    "radeon_software_version": "26.8.1",
    "radeon_release_version_prefix": "26.10.41.01-260811a-203304C",
    "vulkan_api_version": "1.4.349",
    "vulkan_driver_info": "26.8.1 (LLPC)",
    "vulkan_conformance": "1.4.3.3",
    "amdhip64_6.dll": "10.0.3652.0",
    "amdhip64_7.dll": "10.0.3679.0",
    "npu_driver_version": "32.0.20101.3760",
    "disk_free_gib": 573.40,
}

VK_EXTENSIONS = ("VK_KHR_shader_integer_dot_product", "VK_KHR_cooperative_matrix", "VK_NV_cooperative_matrix2",
                 "VK_KHR_shader_float16_int8", "VK_KHR_16bit_storage", "VK_KHR_8bit_storage",
                 "VK_EXT_subgroup_size_control", "VK_KHR_shader_bfloat16", "VK_EXT_shader_float8")


def say(tag: str, obj) -> None:
    print(f"{tag} " + json.dumps(obj), flush=True)


def gb(n: int) -> float:
    return round(n / 1e9, 2)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ps_json(cmd: str):
    """Run one PowerShell query whose output is ConvertTo-Json; a list, a dict or None."""
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                        f"{cmd} | ConvertTo-Json -Depth 3 -Compress"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    s = r.stdout.strip()
    if r.returncode != 0 or not s:
        return None
    return json.loads(s)


def as_list(x) -> list:
    return [] if x is None else (x if isinstance(x, list) else [x])


def run_text(args: list, timeout: int = 120) -> tuple:
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return r.returncode, r.stdout


def disk() -> dict:
    u = shutil.disk_usage(DRIVE)
    return {"drive": DRIVE[:2], "total_bytes": u.total, "used_bytes": u.used, "free_bytes": u.free,
            "total_gb": gb(u.total), "used_gb": gb(u.used), "free_gb": gb(u.free), "floor_gb": FLOOR_GB}


def display_registry() -> list:
    out = []
    names = ("DriverDesc", "DriverVersion", "DriverDate", "ProviderName", "RadeonSoftwareVersion",
             "RadeonSoftwareEdition", "ReleaseVersion", "MatchingDeviceId")
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, DISPLAY_CLASS) as cls:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(cls, i)
            except OSError:
                break
            i += 1
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(cls, sub) as k:
                    row = {"subkey": sub}
                    for n in names:
                        try:
                            row[n] = winreg.QueryValueEx(k, n)[0]
                        except OSError:
                            pass
                    out.append(row)
            except OSError as e:
                out.append({"subkey": sub, "error": type(e).__name__})
    return out


def vulkan() -> dict:
    exe = shutil.which("vulkaninfo") or str(SYSTEM32 / "vulkaninfo.exe")
    if not Path(exe).exists():
        return {"vulkaninfo": "absent"}
    rc, s = run_text([exe, "--summary"])
    keep, section = [], None
    for line in s.splitlines():
        t = line.strip()
        if t.startswith("Vulkan Instance Version"):
            keep.append(t)
        if t in ("Devices:",):
            section = "devices"
            continue
        if section == "devices" and "=" in t:
            keep.append(t)
    rc2, full = run_text([exe])
    ext = {}
    for e in VK_EXTENSIONS:
        hits = sorted({ln.strip() for ln in full.splitlines() if ln.strip().startswith(e + " ")})
        ext[e] = hits[0] if hits else "absent"
    feats = sorted({" ".join(ln.split()) for ln in full.splitlines()
                    if ln.strip().startswith(("integerDotProduct8BitSignedAccelerated ",
                                              "integerDotProduct4x8BitPackedSignedAccelerated ",
                                              "cooperativeMatrix ", "shaderFloat16 ", "shaderInt8 ",
                                              "shaderBFloat16Type "))})
    return {"rc_summary": rc, "rc_full": rc2, "summary": keep, "extensions": ext, "features": feats}


def hip_dlls() -> list:
    rows = []
    for pat in ("amdhip64*.dll", "amd_comgr*.dll", "hiprtc*.dll", "amdhiprtc*.dll"):
        for p in sorted(glob.glob(str(SYSTEM32 / pat))):
            q = Path(p)
            v = ps_json(f"(Get-Item -LiteralPath '{p}').VersionInfo | Select-Object FileVersion, ProductVersion")
            rows.append({"name": q.name, "bytes": q.stat().st_size, "sha256": sha256_file(q),
                         "file_version": (v or {}).get("FileVersion"),
                         "product_version": (v or {}).get("ProductVersion")})
    return rows


def signed_drivers() -> list:
    q = ("Get-CimInstance Win32_PnPSignedDriver | Where-Object { $_.DeviceName -cmatch '\\bNPU\\b|\\bIPU\\b|Radeon' } "
         "| Select-Object DeviceName, DeviceClass, DriverVersion, "
         "@{n='DriverDate';e={if ($_.DriverDate) { $_.DriverDate.ToString('yyyy-MM-dd') }}}, "
         "Manufacturer, InfName")
    return as_list(ps_json(q))


def video_controllers() -> list:
    q = ("Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, "
         "@{n='DriverDate';e={if ($_.DriverDate) { $_.DriverDate.ToString('yyyy-MM-dd') }}}, VideoProcessor, Status")
    return as_list(ps_json(q))


def host() -> dict:
    os_ = ps_json("Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber")
    cpu = ps_json("Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors")
    mem = as_list(ps_json("Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity, Speed, "
                          "ConfiguredClockSpeed, DeviceLocator"))
    for m in mem:
        m["capacity_gb"] = gb(int(m.get("Capacity") or 0))
    return {"os": os_, "cpu": cpu, "memory_modules": mem,
            "memory_total_gb": gb(sum(int(m.get("Capacity") or 0) for m in mem))}


def toolchains() -> dict:
    env = {k: ("set" if os.environ.get(k) else "unset")
           for k in ("HIP_PATH", "ROCM_PATH", "CUDA_PATH", "VULKAN_SDK")}
    env.update({k: "set" for k in os.environ if k.startswith(("HIP_PATH_", "CUDA_PATH_V"))})
    dirs = {}
    for d in (r"C:\Program Files\AMD\ROCm", r"C:\Program Files\NVIDIA GPU Computing Toolkit", r"C:\VulkanSDK"):
        p = Path(d)
        dirs[d] = sorted(x.name for x in p.iterdir()) if p.is_dir() else "absent"
    on_path = {t: ("found" if shutil.which(t) else "absent") for t in ("nvcc", "hipcc", "hipconfig", "cmake", "ninja")}
    vers = {}
    for t, args in (("cmake", ["cmake", "--version"]), ("ninja", ["ninja", "--version"])):
        if shutil.which(t):
            rc, s = run_text(args)
            vers[t] = s.splitlines()[0].strip() if s else f"rc {rc}"
    vs = []
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio/Installer/vswhere.exe"
    if vswhere.exists():
        rc, s = run_text([str(vswhere), "-all", "-products", "*", "-format", "json", "-utf8"])
        rc2, s2 = run_text([str(vswhere), "-all", "-products", "*", "-requires",
                            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-format", "json", "-utf8"])
        with_vc = {x.get("instanceId") for x in json.loads(s2 or "[]")}
        for x in json.loads(s or "[]"):
            vs.append({"displayName": x.get("displayName"), "installationVersion": x.get("installationVersion"),
                       "vc_tools_x64": x.get("instanceId") in with_vc})
    conda = {}
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda.exe") or shutil.which("conda")
    if exe and Path(exe).exists():
        rc, s = run_text([exe, "--version"])
        conda["version"] = s.strip()
        rc, s = run_text([exe, "env", "list", "--json"])
        envs = json.loads(s or "{}").get("envs", [])
        names = sorted("base" if i == 0 else Path(e).name for i, e in enumerate(envs))
        conda["envs"] = names
        conda["planned_envs_absent"] = {n: n not in names for n in ("genai016", "rocm10")}
    return {"env": env, "dirs": dirs, "on_path": on_path, "versions": vers, "visual_studio": vs, "conda": conda}


def baseline() -> int:
    print("S0 BASELINE (light: a re-read only; nothing is installed, and no chip computes). GB = 1e9 bytes.",
          flush=True)
    say("TIME_JSON", {"utc": utc_now()})
    say("HOST_JSON", host())
    reg = display_registry()
    say("DISPLAY_REGISTRY_JSON", reg)
    say("VIDEO_CONTROLLER_JSON", video_controllers())
    drv = signed_drivers()
    say("SIGNED_DRIVERS_JSON", drv)
    vk = vulkan()
    say("VULKAN_JSON", vk)
    dlls = hip_dlls()
    say("HIP_DLLS_JSON", dlls)
    say("TOOLCHAINS_JSON", toolchains())
    rt = sorted(x.name for x in RT.iterdir()) if RT.is_dir() else "absent"
    say("SCRATCH_RT_JSON", {"path": rel(RT.parent) + "/rt", "entries": rt})
    d = disk()
    say("DISK_JSON", d)

    radeon = next((r for r in reg if "Radeon" in str(r.get("DriverDesc", ""))), {})
    npu = next((r for r in drv if str(r.get("DeviceClass", "")) == "COMPUTEACCELERATOR"), {})
    summ = {ln.split("=", 1)[0].strip(): ln.split("=", 1)[1].strip() for ln in vk.get("summary", []) if "=" in ln}
    dll = {r["name"]: r["file_version"] for r in dlls}
    now = {
        "radeon_driver_version": radeon.get("DriverVersion"),
        "radeon_software_version": radeon.get("RadeonSoftwareVersion"),
        "radeon_release_version_prefix": str(radeon.get("ReleaseVersion", ""))[:27],
        "vulkan_api_version": summ.get("apiVersion"),
        "vulkan_driver_info": summ.get("driverInfo"),
        "vulkan_conformance": summ.get("conformanceVersion"),
        "amdhip64_6.dll": dll.get("amdhip64_6.dll"),
        "amdhip64_7.dll": dll.get("amdhip64_7.dll"),
        "npu_driver_version": npu.get("DriverVersion"),
        "disk_free_gib": round(d["free_bytes"] / 2**30, 2),
    }
    cmp_ = {}
    for k, v1 in V1.items():
        v = now[k]
        if k == "disk_free_gib":
            cmp_[k] = {"v1": v1, "now": v, "delta_gib": round(v - v1, 2)}
        else:
            cmp_[k] = {"v1": v1, "now": v, "same": v == v1}
    say("V1_COMPARE_JSON", cmp_)
    changed = [k for k, c in cmp_.items() if k != "disk_free_gib" and not c["same"]]
    above = d["free_gb"] >= FLOOR_GB
    print(f"BASELINE {'OK' if not changed else 'CHANGED'}: "
          f"{'no version differs from v1' if not changed else 'differs from v1: ' + ', '.join(changed)}; "
          f"{d['drive']} {d['free_gb']} GB free ({'above' if above else 'BELOW'} the {FLOOR_GB:g} GB floor)",
          flush=True)
    return 0 if above else 3


# ---------------------------------------------------------------- the asset list (the publishers' own records)

UA = "ignite-xdna-s0/1 (urllib; one read per record)"
GH = "https://api.github.com/repos/"
PYPI = "https://pypi.org/pypi/"
ANACONDA = "https://api.anaconda.org/package/"
THEROCK = "https://stable.repo.amd.com/rocm/whl-next/"
PY_TAG = "cp312"                                          # the genai016 env's Python (3.12)
NVCC_VERSION = "13.1.115"                                 # the newest CUDA ZLUDA v6's notes name ("CUDA 13.1")
ROCM_VERSION = "10.0.0"
ROCM_PKGS = ("rocm", "rocm-sdk-core", "rocm-sdk-libraries", "rocm-sdk-devel", "rocm-sdk-device-gfx1103")
NVCC_PKGS = ("cuda-nvcc", "cuda-nvcc_win-64", "cuda-nvcc-dev_win-64", "cuda-nvcc-impl", "cuda-nvcc-tools",
             "cuda-nvvm-tools", "cuda-nvvm-impl", "cuda-crt-tools", "cuda-crt-dev_win-64")
LLAMA_WIN = re.compile(r"^(llama-b\d+-bin-win-(cpu|vulkan|rocm-[\d.]+|hip-[\w.-]+|cuda-[\d.]+)-x64\.zip|"
                       r"cudart-llama-bin-win-cuda-[\d.]+-x64\.zip)$")
LLAMA_FETCH = re.compile(r"-(cpu|vulkan|rocm-[\d.]+)-x64\.zip$|cuda-12\.[\d.]+-x64\.zip$")
_GH_CALLS = []


def http(url: str, method: str = "GET", accept: str = None):
    h = {"User-Agent": UA}
    if accept:
        h["Accept"] = accept
    r = urllib.request.urlopen(urllib.request.Request(url, headers=h, method=method), timeout=120)
    return r


def get_json(url: str) -> dict:
    with http(url, accept="application/vnd.github+json" if url.startswith(GH) else "application/json") as r:
        if url.startswith(GH):
            _GH_CALLS.append(r.headers.get("X-RateLimit-Remaining"))
        return json.loads(r.read().decode("utf-8"))


def get_text(url: str) -> tuple:
    with http(url) as r:
        return r.read().decode("utf-8", errors="replace"), r.geturl()


def head_bytes(url: str):
    try:
        with http(url, method="HEAD") as r:
            n = r.headers.get("Content-Length")
            return int(n) if n is not None else None
    except Exception as e:                                # a HEAD refusal leaves the size to the download
        return f"HEAD failed: {type(e).__name__}"


def row(component: str, version: str, name: str, url: str, size, sha256, source: str, api: str, fetch: bool,
        note: str = "") -> dict:
    return {"component": component, "version": version, "name": name, "url": url, "bytes": size,
            "gb": gb(size) if isinstance(size, int) else None, "sha256": sha256,
            "checksum": f"publisher sha256 ({source})" if sha256 else "none published: pinned only (hashed at download)",
            "api": api, "read_utc": utc_now(), "fetch": fetch, "note": note}


def gh_digest(a: dict):
    d = a.get("digest") or ""
    return d.split(":", 1)[1] if d.startswith("sha256:") else None


def gh_release_info(rel: dict) -> dict:
    return {"tag": rel.get("tag_name"), "name": rel.get("name"), "published_at": rel.get("published_at"),
            "prerelease": rel.get("prerelease"), "draft": rel.get("draft"),
            "assets": [a["name"] for a in rel.get("assets", [])]}


def assets_llamacpp() -> tuple:
    repo = "ggml-org/llama.cpp"
    api = f"{GH}{repo}/releases/latest"
    latest = get_json(api)
    info = {"releases_latest": gh_release_info(latest), "api": api}
    tag = None
    nt = next((a for a in latest.get("assets", []) if a["name"] == "nightly-tag.txt"), None)
    if nt is not None:
        tag = get_text(nt["browser_download_url"])[0].strip()
        info["nightly_tag_txt"] = {"url": nt["browser_download_url"], "bytes": nt["size"], "sha256": gh_digest(nt),
                                   "reads": tag}
    if not tag or not re.fullmatch(r"b\d+", tag):
        tag = latest.get("tag_name")
    api_b = f"{GH}{repo}/releases/tags/{tag}"
    rel = get_json(api_b)
    info["binaries_release"] = gh_release_info(rel)
    info["binaries_commit"] = get_json(f"{GH}{repo}/commits/{tag}").get("sha")
    recent = get_json(f"{GH}{repo}/releases?per_page=5")
    info["newest_releases"] = [{"tag": r["tag_name"], "published_at": r["published_at"], "prerelease": r["prerelease"]}
                               for r in recent]
    rows = []
    for a in rel.get("assets", []):
        if LLAMA_WIN.match(a["name"]):
            rows.append(row("llamacpp", tag, a["name"], a["browser_download_url"], a["size"], gh_digest(a),
                            "GitHub release digest", api_b, bool(LLAMA_FETCH.search(a["name"])),
                            "" if LLAMA_FETCH.search(a["name"]) else "listed, not fetched by default"))
    src = f"https://github.com/{repo}/archive/refs/tags/{tag}.zip"
    rows.append(row("llamacpp", tag, f"llama.cpp-{tag}-source.zip", src, None, None, "", api_b, False,
                    f"GitHub's generated source archive; the tag's commit {info['binaries_commit']} is the pin"))
    return info, rows


def assets_zluda() -> tuple:
    repo = "vosen/ZLUDA"
    api = f"{GH}{repo}/releases/latest"
    latest = get_json(api)
    recent = get_json(f"{GH}{repo}/releases?per_page=10")
    pre = next((r for r in recent if r.get("prerelease")), None)
    info = {"stable": gh_release_info(latest), "newest_prerelease": gh_release_info(pre) if pre else None,
            "api": api}
    rows = []
    for rel, stable in ((latest, True), (pre, False)):
        if rel is None:
            continue
        for a in rel.get("assets", []):
            if "windows" in a["name"].lower():
                rows.append(row("zluda", rel["tag_name"], a["name"], a["browser_download_url"], a["size"], gh_digest(a),
                                "GitHub release digest", f"{GH}{repo}/releases/tags/{rel['tag_name']}", stable,
                                "" if stable else "PREVIEW: only if the stable release fails"))
    return info, rows


def pypi_files(pkg: str, version: str = None) -> tuple:
    api = f"{PYPI}{pkg}/{version}/json" if version else f"{PYPI}{pkg}/json"
    j = get_json(api)
    return j["info"], j.get("urls", []), api


def pypi_rows(component: str, pkg: str, version, fetch: bool, note: str = "") -> tuple:
    info, files, api = pypi_files(pkg, version)
    rows = [row(component, info["version"], f["filename"], f["url"], f["size"], f["digests"].get("sha256"),
                "PyPI digests", api, fetch, note)
            for f in files if f["filename"].endswith(f"-{PY_TAG}-{PY_TAG}-win_amd64.whl")
            or (f["filename"].endswith("-py3-none-win_amd64.whl"))]
    meta = {"package": pkg, "version": info["version"], "requires_dist": info.get("requires_dist"),
            "cp312_win_amd64": [r["name"] for r in rows], "api": api}
    if not rows:
        rows = [row(component, info["version"], f"{pkg}: no {PY_TAG} win_amd64 wheel", None, None, None, "", api,
                    False, "absent on PyPI")]
    return meta, rows


def assets_genai() -> tuple:
    info, rows = {}, []
    for pkg, ver, fetch, note in (
            ("onnxruntime-genai", "0.16.0", True, "GenAI 0.16, the CPU package"),
            ("onnxruntime", None, True, "the newest onnxruntime (onnxruntime-genai 0.16 requires it)"),
            ("onnxruntime-genai-directml", None, False, "the newest DirectML package; see its requires_dist"),
            ("onnxruntime-directml", None, False, "the newest onnxruntime-directml on PyPI"),
            ("onnxruntime-genai-winml", "0.16.0", False, "the WinML package of 0.16; for the gate"),
            ("onnxruntime-ep-amdgpu", None, False, "the AMDGPU EP's PyPI name, if any")):
        try:
            meta, rr = pypi_rows("genai", pkg, ver, fetch, note)
        except urllib.error.HTTPError as e:
            meta, rr = {"package": pkg, "version": ver, "error": f"HTTP {e.code}"}, []
        info[pkg] = meta
        rows += rr
    api = f"{GH}microsoft/onnxruntime-genai/releases/tags/v0.16.0"
    rel = get_json(api)
    info["github_v0.16.0"] = gh_release_info(rel) | {"api": api}
    api = f"{GH}ROCm/hip-ep/releases/latest"
    try:
        rel = get_json(api)
        body_shas = sorted(set(re.findall(r"\b[0-9a-f]{64}\b", rel.get("body") or "")))
        info["hip_ep_latest"] = gh_release_info(rel) | {"api": api, "sha256_in_body": body_shas}
        for a in rel.get("assets", []):
            if "windows" in a["name"].lower():
                d = gh_digest(a)
                rows.append(row("hip-ep", rel["tag_name"], a["name"], a["browser_download_url"], a["size"], d,
                                "GitHub release digest", api, False,
                                f"the AMDGPU EP plugin; the release body lists {body_shas}; "
                                f"{'the body agrees with the digest' if d in body_shas else 'the body DISAGREES with the digest'}"))
    except urllib.error.HTTPError as e:
        info["hip_ep_latest"] = {"api": api, "error": f"HTTP {e.code}"}
    return info, rows


def link_rows(html: str) -> list:
    return re.findall(r'<a\s+[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', html, re.I)


def assets_rocm() -> tuple:
    info, rows = {"index": THEROCK, "version": ROCM_VERSION, "packages": {}}, []
    for pkg in ROCM_PKGS:
        page = f"{THEROCK}{pkg}/"
        try:
            html, final = get_text(page)
        except urllib.error.HTTPError as e:
            info["packages"][pkg] = {"page": page, "error": f"HTTP {e.code}"}
            continue
        files = []
        for href, text in link_rows(html):
            fn = text.strip()
            if not (fn.endswith(".whl") or fn.endswith(".tar.gz")):
                continue
            files.append(fn)
            v = re.search(r"-(\d+\.\d+\.\d+[\w.]*?)(?:-|\.tar\.gz)", fn)
            if v and v.group(1) == ROCM_VERSION and (fn.endswith("win_amd64.whl") or fn.endswith(".tar.gz")):
                url, frag = urllib.parse.urldefrag(urllib.parse.urljoin(final, href))
                sha = frag.split("=", 1)[1] if frag.startswith("sha256=") else None
                rows.append(row("rocm", ROCM_VERSION, fn, url, head_bytes(url), sha, "the index's #sha256", final, True))
        info["packages"][pkg] = {"page": page, "final": final, "files_listed": len(files),
                                 "versions": sorted({m.group(1) for f in files
                                                     for m in [re.search(r"-(\d+\.\d+\.\d+[\w.]*?)(?:-|\.tar\.gz)", f)] if m})}
    return info, rows


def assets_nvcc() -> tuple:
    info, rows = {"channel": "conda-forge", "version": NVCC_VERSION, "packages": {}}, []
    for pkg in NVCC_PKGS + ("vs2019_win-64", "vs2022_win-64"):
        api = f"{ANACONDA}conda-forge/{pkg}"
        try:
            j = get_json(api)
        except urllib.error.HTTPError as e:
            info["packages"][pkg] = {"api": api, "error": f"HTTP {e.code}"}
            continue
        files = j.get("files", [])
        want = NVCC_VERSION if pkg.startswith("cuda-") else j.get("latest_version")
        hits = [f for f in files if f.get("version") == want and f.get("attrs", {}).get("subdir") in ("win-64", "noarch")]
        hits.sort(key=lambda f: (f.get("attrs", {}).get("build_number", 0), f.get("upload_time", "")))
        info["packages"][pkg] = {"api": api, "latest_version": j.get("latest_version"), "wanted": want,
                                 "builds_at_wanted": [f["basename"] for f in hits]}
        if hits:
            f = hits[-1]
            url = "https:" + f["download_url"] if f["download_url"].startswith("//") else f["download_url"]
            rows.append(row("nvcc", want, f["basename"].split("/")[-1], url, f.get("size"), f.get("sha256"),
                            "anaconda.org file sha256", api, False,
                            "conda resolves and verifies it at install (the closure is pinned by a logged dry-run)"))
    return info, rows


COMPONENTS = {"llamacpp": assets_llamacpp, "zluda": assets_zluda, "genai": assets_genai, "rocm": assets_rocm,
              "nvcc": assets_nvcc}


def assets() -> int:
    print("S0 ASSETS (light: publisher records only; nothing is downloaded but llama.cpp's 7-byte nightly-tag.txt). "
          "GB = 1e9 bytes.", flush=True)
    say("TIME_JSON", {"utc": utc_now()})
    total, allrows, bad = 0, [], []
    for comp, fn in COMPONENTS.items():
        try:
            info, rows = fn()
        except Exception as e:
            say("COMPONENT_ERROR_JSON", {"component": comp, "error": f"{type(e).__name__}: {e}"})
            bad.append(comp)
            continue
        say("RELEASE_JSON", {"component": comp, **info})
        for r in rows:
            say("ASSET_JSON", r)
        allrows += rows
    fetch = [r for r in allrows if r["fetch"]]
    total = sum(r["bytes"] for r in fetch if isinstance(r["bytes"], int))
    d = disk()
    say("ASSETS_DONE_JSON", {"rows": len(allrows), "fetch_rows": len(fetch), "fetch_bytes": total,
                             "fetch_gb": gb(total), "pinned_only_fetch_rows": sum(1 for r in fetch if not r["sha256"]),
                             "free_gb": d["free_gb"], "free_after_gb": gb(d["free_bytes"] - total),
                             "floor_gb": FLOOR_GB, "github_rate_remaining": _GH_CALLS[-1] if _GH_CALLS else None,
                             "errors": bad})
    print(f"\n  {'component':9s} {'version':10s} {'fetch':5s} {'bytes':>13s}  checksum  name")
    for r in allrows:
        b = f"{r['bytes']:,}" if isinstance(r["bytes"], int) else str(r["bytes"])
        print(f"  {r['component']:9s} {r['version']:10s} {str(r['fetch']):5s} {b:>13s}  "
              f"{'sha256' if r['sha256'] else 'pinned only':11s} {r['name']}")
    print(f"ASSETS {'OK' if not bad else 'INCOMPLETE: ' + ', '.join(bad)}: {len(allrows)} assets, {len(fetch)} to "
          f"fetch by default ({gb(total)} GB); {d['free_gb']} GB free, the floor {FLOOR_GB:g} GB", flush=True)
    return 0 if not bad else 2


# ---------------------------------------------------------------- fetch (verified downloads into scratch/rt)

def fetch(component: str) -> int:
    logs = sorted(RESULTS.glob("hybrid_s0_assets_*.log"))
    if not logs:
        print("FETCH REFUSED: no assets log; run the assets stage and send the list to the gate first", flush=True)
        return 2
    lines = logs[-1].read_text(encoding="utf-8").splitlines()
    rows = [json.loads(s.split(" ", 1)[1]) for s in lines if s.startswith("ASSET_JSON ")]
    rows = [r for r in rows if r["component"] == component and r["fetch"]]
    lf = hashlib.sha256(logs[-1].read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    say("FETCH_PLAN_JSON", {"assets_log": logs[-1].name, "assets_log_lf_sha256": lf, "component": component,
                            "rows": [r["name"] for r in rows], "utc": utc_now()})
    if not rows:
        print(f"FETCH REFUSED: the assets log marks nothing to fetch for {component}", flush=True)
        return 2
    rc = 0
    for r in rows:
        dest = RT / f"{component}-{r['version']}"
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / r["name"]
        before = disk()
        need = (r["bytes"] if isinstance(r["bytes"], int) else 0) / 1e9
        if before["free_gb"] - need < FLOOR_GB:
            say("FETCH_REFUSED_JSON", {"name": r["name"], "free_gb": before["free_gb"], "need_gb": round(need, 2),
                                       "floor_gb": FLOOR_GB})
            print(f"FETCH STOP: {r['name']} would leave less than {FLOOR_GB:g} GB free", flush=True)
            return 3
        if out.exists():
            got = sha256_file(out)
            same = got == r["sha256"] if r["sha256"] else None
            say("FETCH_EXISTS_JSON", {"name": r["name"], "path": rel(out), "sha256": got, "publisher_match": same})
            if same is not False:
                continue
            out.unlink()
        part = out.with_name(out.name + ".part")
        h, n, t0 = hashlib.sha256(), 0, time.time()
        with http(r["url"]) as resp, open(part, "wb") as f:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                h.update(chunk)
                f.write(chunk)
                n += len(chunk)
        got = h.hexdigest()
        size_ok = not isinstance(r["bytes"], int) or n == r["bytes"]
        match = (got == r["sha256"]) if r["sha256"] else None
        if match is False or not size_ok:
            part.unlink()
            rc = 3
        else:
            part.replace(out)
        after = disk()
        say("FETCH_JSON", {"name": r["name"], "url": r["url"], "path": rel(out) if rc == 0 else None, "bytes": n,
                           "bytes_listed": r["bytes"], "sha256_publisher": r["sha256"], "sha256_got": got,
                           "verified": "MATCH" if match else ("pinned only" if match is None else "MISMATCH"),
                           "size_ok": size_ok, "seconds": round(time.time() - t0, 1),
                           "free_gb_before": before["free_gb"], "free_gb_after": after["free_gb"]})
        if rc:
            print(f"FETCH STOP: {r['name']} failed verification; the partial file is deleted", flush=True)
            return rc
    print(f"FETCH OK: {component}, {len(rows)} assets in {rel(RT)}/{component}-*", flush=True)
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("baseline", "assets", "fetch"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.mode == "fetch":
        if len(a.args) != 1 or a.args[0] not in COMPONENTS:
            ap.error(f"fetch takes one COMPONENT: {', '.join(COMPONENTS)}")
        return fetch(a.args[0])
    return {"baseline": baseline, "assets": assets}[a.mode]()


if __name__ == "__main__":
    sys.exit(main())
