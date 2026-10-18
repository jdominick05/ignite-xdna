"""Gate 0 of the dynamic-witness plan: does XRT's AIE trace plugin attach to the VitisAI EP process?

The question is only whether XRT's own trace plugin (xdp_aie_trace_plugin.dll) loads into an EP process and
writes non-empty trace data, configured purely through environment variables and an xrt.ini, without changing
the outputs. Trace content is not decoded here.

Two arms, each its own process (XRT reads xrt.ini and its env at load time):
  --arm untraced --out DIR
      Builds the 32-channel shift_cut 8 / shift_bias 0 fixture of tools/xint8_arithmetic_probe.py, clears and
      recompiles the worktree's own EP cache, runs every case twice (must repeat exactly), and saves the
      outputs, the cached xmodel's sha256 and mtime, the EP report summary, and the XRT/XDP/EP DLLs the process
      loaded.
  --arm traced --ref DIR --out DIR2 [--expect trace|profile]
      Run with cwd, XRT_INI_PATH and AMD_NPU_SDK_PATH set by the caller. It does NOT clear the cache: the
      cached xmodel must keep the reference's sha256 and mtime (no recompile). Then it runs the same cases and
      records the DLLs and every file that appeared in the working directory. Exit 0 iff these all hold:
      - the xdp_aie_<expect>_plugin.dll is loaded;
      - at least one non-empty aie_<expect>* file exists;
      - the outputs are bit-equal to the reference;
      - the xmodel is unchanged;
      - the device was idle before.
      XDP writes some files only at process exit (the profile CSV does), so the in-process file check is
      advisory: a caller that needs those files lists the directory after the process ends. Idle after exit
      is also the caller's check.

Nothing here copies or replaces a DLL anywhere. Run in resnet_env17.
What has been run: results/aie/xdp_trace_attach_desktop2_20261012T213156Z.log (Gate 0, trace),
xdp_trace_metadata_desktop2_20261012T214206Z.log (Gate 0b, trace, metadata supplied) and
xdp_profile_attach_desktop2_20261012T215835Z.log (B-0, profile).
"""
import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from npu.ep_report import read_report  # noqa: E402
from npu.paths import CACHE_DIR, RESNET_CACHE_KEY  # noqa: E402
from npu.session import build_session, clear_cache, resolve_xclbin  # noqa: E402
import xint8_arithmetic_probe as probe  # noqa: E402

XRT_SMI = 'C:/Windows/System32/AMD/xrt-smi.exe'
SC, SB, CHANNELS = 8, 0, 32
MODULE_RE = re.compile(r'(xrt|xdp|aie|vaip|onnxruntime|amd)', re.I)


def idle():
    out = subprocess.run([XRT_SMI, 'examine', '-r', 'aie-partitions'], capture_output=True, text=True,
                         timeout=60).stdout
    return 'No hardware contexts running' in out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def file_version(path):
    """The fixed-file version of a DLL as 'a.b.c.d', or None."""
    v = ctypes.windll.version
    n = v.GetFileVersionInfoSizeW(str(path), None)
    if not n:
        return None
    buf = ctypes.create_string_buffer(n)
    if not v.GetFileVersionInfoW(str(path), 0, n, buf):
        return None
    ptr, size = ctypes.c_void_p(), ctypes.c_uint()
    if not v.VerQueryValueW(buf, '\\', ctypes.byref(ptr), ctypes.byref(size)) or size.value < 52:
        return None
    ms, ls = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32))[2:4]
    return '%d.%d.%d.%d' % (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)


def modules():
    rows = {}
    for m in psutil.Process().memory_maps(grouped=True):
        p = m.path
        if p.lower().endswith('.dll') and MODULE_RE.search(Path(p).name):
            rows[p] = {'path': p, 'version': file_version(p), 'size': Path(p).stat().st_size}
    return sorted(rows.values(), key=lambda r: r['path'].lower())


def cached_xmodel():
    xms = sorted((CACHE_DIR / RESNET_CACHE_KEY).glob('*.xmodel'))
    if len(xms) != 1:
        raise SystemExit('expected one cached xmodel, found %d' % len(xms))
    return xms[0]


def run_cases(session, cases):
    outs = {}
    for name, x in cases.items():
        first = probe.output_codes(session.run(None, {'input': x})[0], SC)
        again = probe.output_codes(session.run(None, {'input': x})[0], SC)
        if not np.array_equal(first, again):
            raise SystemExit('nondeterministic repeated output on ' + name)
        outs[name] = first
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', choices=('untraced', 'traced'), required=True)
    ap.add_argument('--out', type=Path, required=True, help='new directory for this arm\'s results')
    ap.add_argument('--ref', type=Path, help='the untraced arm\'s --out (traced arm only)')
    ap.add_argument('--expect', choices=('trace', 'profile'), default='trace',
                    help='which XDP plugin\'s output the traced arm must find: aie_trace* or aie_profile* files')
    a = ap.parse_args()
    if a.out.exists():
        ap.error('--out must be new')
    if a.arm == 'traced' and not a.ref:
        ap.error('--arm traced needs --ref')
    cache = (CACHE_DIR / RESNET_CACHE_KEY).resolve()
    if cache.parent != ROOT.resolve():
        raise SystemExit('EP cache escaped this checkout: %s' % cache)
    report = {'arm': a.arm, 'cwd': os.getcwd(), 'idle_before': idle(),
              'env': {k: os.environ.get(k) for k in ('AMD_NPU_SDK_PATH', 'XRT_INI_PATH', 'XILINX_XRT')}}
    if not report['idle_before']:
        raise SystemExit('NPU not idle: refusing (a foreign context is contention, not a finding)')
    a.out.mkdir(parents=True)
    g, w, b = probe.model(CHANNELS, SC, SB, False)
    onnx_path = a.out / 'fixture.onnx'
    g.save(onnx_path)
    cases = probe.fixtures(CHANNELS, False)
    xclbin = str(resolve_xclbin())
    cwd_before = {p.name for p in Path.cwd().iterdir()}

    if a.arm == 'untraced':
        clear_cache(RESNET_CACHE_KEY)
    else:
        ref = json.loads((a.ref / 'report.json').read_text())
        xm = cached_xmodel()
        if sha(xm) != ref['xmodel_sha256']:
            raise SystemExit('cached xmodel differs from the reference before the traced build')
    session = build_session(onnx_path, 'npu', RESNET_CACHE_KEY, xclbin, log_severity=3)
    ep = read_report(cache / 'vitisai_ep_report.json').summary()
    report['ep'] = {'npu': ep['npu'], 'total': ep['total'], 'conv_on_npu': ep['ops_by_device'].get('NPU:Conv', 0)}
    outs = run_cases(session, cases)
    report['modules'] = modules()
    del session
    xm = cached_xmodel()
    report['xmodel_sha256'], report['xmodel_mtime_ns'] = sha(xm), xm.stat().st_mtime_ns
    np.savez(a.out / 'outputs.npz', **outs)
    report['cases'] = sorted(outs)
    names = [Path(m['path']).name.lower() for m in report['modules']]
    report['coreutil'] = [m for m in report['modules'] if Path(m['path']).name.lower() == 'xrt_coreutil.dll']

    ok = True
    if a.arm == 'traced':
        new = sorted(set(p.name for p in Path.cwd().iterdir()) - cwd_before)
        report['new_files'] = [{'name': n, 'size': (Path.cwd() / n).stat().st_size if (Path.cwd() / n).is_file()
                                else None} for n in new]
        ref_outs = np.load(a.ref / 'outputs.npz')
        checks = {
            'plugin_loaded': 'xdp_aie_%s_plugin.dll' % a.expect in names,
            'plugin_file_nonempty': any(f['name'].lower().startswith('aie_' + a.expect) and (f['size'] or 0) > 0
                                       for f in report['new_files']),
            'outputs_bit_equal': sorted(ref_outs.files) == sorted(outs)
                                 and all(np.array_equal(ref_outs[k], outs[k]) for k in outs),
            'xmodel_unchanged': report['xmodel_sha256'] == ref['xmodel_sha256']
                                and report['xmodel_mtime_ns'] == ref['xmodel_mtime_ns'],
            'idle_before': report['idle_before'],
        }
        report['gate0_checks'] = checks
        ok = all(checks.values())
        report['gate0'] = 'PASS' if ok else 'FAIL'
    else:
        ok = report['ep']['conv_on_npu'] == 1
    (a.out / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in report if k not in ('modules',)}, indent=1))
    for m in report['modules']:
        print('MODULE %-16s %s' % (m['version'], m['path']))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
