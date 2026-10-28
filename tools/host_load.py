"""Host contention witness for Linux: tools/host_load.ps1's one-shot output, from psutil.

    python tools/host_load.py                  # one-shot, parseable
    python tools/host_load.py --sample-sec 2

Same records and the same verdict rule as host_load.ps1, so research_process.py and
the logs read one format on either OS:

    HOST_LOAD busy_cores=<f> logical_cores=<n> free_ram_gb=<f> sample_sec=<f>
    HOST_LOAD_PEER <cores> <name>:<pid> <reason> <command line, truncated>
    HOST_LOAD_TOP <cores> <name>:<pid> ws_mb=<n>
    HOST_LOAD_VERDICT CLEAR|BUSY|PEER

PEER outranks BUSY. A peer is a build tool, one of this repo's own heavy producers, or
any single process holding CORES_PEER cores or more. ws_mb is the resident set.
"""
import argparse
import os
import re
import time

import psutil

CORES_PEER = 2.0
CMD_CHARS = 140
PEER_NAMES = re.compile(r'^(cl|link|lib|ninja|cmake|make|nmake|msbuild|devenv|gcc|g\+\+|cc1|cc1plus|'
                        r'clang|clang\+\+|lld|lld-link|ld|rustc|cargo|go|javac|node|tsc|'
                        r'xchesscc|xchesscc_wrapper|aie-opt|aie-translate|llc|opt|nvcc|hipcc)$')
PEER_CMDS = re.compile(r'(3d_quantize_compare\.py|3_quantize\.py|3b_quantize_cut\.py|'
                       r'1_export\.py|-m\s+quant\b|quant\.__main__|quant\.adaround|'
                       r'build_hwinfo_bridge|setup\.py\s+build)')


def redact(s):
    return re.sub(r'(/home/)[^/\s"]+', r'\1<user>', s or '')


def cpu_times():
    t = {}
    for p in psutil.process_iter(['cpu_times']):
        ct = p.info['cpu_times']
        if ct is not None:
            t[p.pid] = ct.user + ct.system
    return t


def sample(seconds):
    before = cpu_times()
    t0 = time.monotonic()
    time.sleep(seconds)
    dt = time.monotonic() - t0 or seconds
    rows = []
    for p in psutil.process_iter(['name', 'cmdline', 'cpu_times', 'memory_info']):
        ct = p.info['cpu_times']
        if ct is None:
            continue
        d = ct.user + ct.system - before.get(p.pid, 0)
        if d <= 0:
            continue
        mem = p.info['memory_info']
        rows.append({'name': p.info['name'] or '?', 'pid': p.pid, 'cores': round(d / dt, 2),
                     'ws_mb': round(mem.rss / 2**20) if mem else 0,
                     'cmd': ' '.join(p.info['cmdline'] or [])})
    return sorted(rows, key=lambda r: -r['cores'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sample-sec', type=float, default=1.0)
    a = ap.parse_args()
    rows = sample(a.sample_sec)
    total = sum(r['cores'] for r in rows)
    cores = os.cpu_count() or 1
    print(f'HOST_LOAD busy_cores={round(total, 2)} logical_cores={cores} '
          f'free_ram_gb={round(psutil.virtual_memory().available / 2**30, 1)} sample_sec={a.sample_sec}')
    peers = 0
    for r in rows:
        cmd = redact(re.sub(r'\s+', ' ', r['cmd']))
        # Never report the guard itself.
        if r['pid'] == os.getpid() or 'host_load.py' in cmd:
            continue
        reason = ('build-tool' if PEER_NAMES.match(r['name']) else
                  'repo-producer' if PEER_CMDS.search(cmd) else
                  'cpu-heavy' if r['cores'] >= CORES_PEER else '')
        if not reason:
            continue
        if len(cmd) > CMD_CHARS:
            cmd = cmd[:CMD_CHARS] + '...'
        print(f"HOST_LOAD_PEER {r['cores']} {r['name']}:{r['pid']} {reason} {cmd or '(command line unavailable)'}")
        peers += 1
    for r in rows[:5]:
        print(f"HOST_LOAD_TOP {r['cores']} {r['name']}:{r['pid']} ws_mb={r['ws_mb']}")
    limit = float(os.environ.get('HOST_LOAD_MAX_CORES') or 0) or max(2.0, cores * 0.25)
    print('HOST_LOAD_VERDICT ' + ('PEER' if peers else 'BUSY' if total > limit else 'CLEAR'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
