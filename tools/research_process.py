"""Bound one research child and record host/device witnesses; never rewrite a log.

The shell wrapper owns UTF-8 logging. This parent scrubs stdout before it reaches
run_logged and bounds the whole descendant tree, including native compiler jobs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import platform
import re
import subprocess
import sys
import threading
import time
import winreg

import psutil

ROOT = Path(__file__).resolve().parents[1]
SMI = Path('C:/Windows/System32/AMD/xrt-smi.exe')
PRODUCER = re.compile(r'3\w*_quantize|1_export\.py|-m\s+quant\b|quant_adaround|acc_spill_probe|aie_disasm', re.I)
BUILD = re.compile(r'^(cl|link|ninja|cmake|make|clang\+\+|clang|llc|opt|aie-opt|aie-translate|xchesscc)(\.exe)?$', re.I)


def scrub(value):
    value = re.sub(r'(?i)([a-z]:[\\/]+Users[\\/]+)[^\\/\s"\'<>]+', r'\1<user>', str(value))
    return re.sub(r'(?i)(/[a-z]/Users/)[^/\s"\'<>]+', r'\1<user>', value)


def emit(label, value):
    print(scrub(label + ' ' + json.dumps(value, allow_nan=False)), flush=True)


def machine():
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    role = ('Desktop 2' if '8700G' in cpu else 'Laptop' if '8645HS' in cpu else
            'Desktop 1 (no NPU)' if '7800X3D' in cpu else 'unclassified')
    return {'machine': role, 'cpu': cpu, 'os': platform.platform(),
            'ram_bytes': psutil.virtual_memory().total}


def source_manifest():
    paths = [ROOT / p for p in ('tools/research_process.py', 'tools/quant_calib_alphabet.py',
             'tools/xint8_arithmetic_probe.py', 'scripts/research-lowlevel.sh',
             'scripts/research-iron.sh', 'kernels/memory_placement/probe.py',
             'kernels/memory_placement/memory_kernels.cc', 'scripts/research-matrix.sh',
             'kernels/memory_placement/gemm.py', 'kernels/memory_placement/gemm_placement.cc',
             'tools/research_summary.py', 'tools/research_toolchain.py')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def context_pids(witness):
    pids = {int(x) for x in re.findall(r'(?:PID|Process ID)\s*:\s*(\d+)', witness, re.I)}
    pids.update(int(x) for x in re.findall(r'^\s*\|(\d+)\s*\|\s*\d+\s*\|', witness, re.M))
    if not pids and 'No hardware contexts running' not in witness:
        raise RuntimeError('cannot establish ownership of the active context report')
    return pids


def context_report():
    result = subprocess.run([str(SMI), 'examine', '-r', 'aie-partitions'],
                            capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError('xrt-smi failed: ' + result.stderr)
    return result.stdout


def cpu_snapshot():
    result = {}
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_times']):
        try:
            d = p.info
            result[p.pid] = (sum(d['cpu_times'][:2]), d['name'], ' '.join(d['cmdline'] or []))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return time.monotonic(), result


def peers(before, after, own):
    dt = after[0] - before[0]
    found = []
    for pid, (cpu, name, command) in after[1].items():
        if pid == 0 or pid in own or pid not in before[1]:
            continue
        cores = max(0, (cpu - before[1][pid][0]) / dt)
        if PRODUCER.search(command) or BUILD.fullmatch(name) or cores >= 2:
            found.append({'pid': pid, 'name': name, 'cores': round(cores, 3), 'command': command[:240]})
    return found


def descendants(process):
    try:
        return [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def kill_owned(process):
    children = descendants(process)
    for p in reversed(children):
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(children, timeout=5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=300)
    ap.add_argument('--rss-gib', type=float, default=8)
    ap.add_argument('--npu', action='store_true')
    ap.add_argument('--wait-clear', type=float, default=0, help='Wait up to this many seconds for a clear host')
    ap.add_argument('--checks-only', action='store_true', help='No performance claim; skip load gating for self-checks')
    ap.add_argument('command', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command or a.seconds <= 0 or a.rss_gib <= 0 or not 0 <= a.wait_clear <= 60:
        ap.error('need a command and positive resource limits')
    # IDE and agent ancestors can perform unrelated work in the same process.
    # Exclude only this monitor and the descendants it actually launches.
    own = {os.getpid()}
    emit('RESEARCH_PLATFORM', {**machine(), 'source_sha256': source_manifest(),
         'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
         'command': command, 'seconds_limit': a.seconds, 'rss_gib_limit': a.rss_gib,
         'checks_only': a.checks_only, 'free_ram_bytes': psutil.virtual_memory().available})
    if not a.checks_only:
        deadline = time.monotonic() + a.wait_clear
        while True:
            result = subprocess.run(['powershell', '-NoProfile', '-File', str(ROOT / 'tools/host_load.ps1')],
                                    capture_output=True, text=True, timeout=30)
            print(scrub(result.stdout + result.stderr), flush=True)
            if not result.returncode and 'HOST_LOAD_VERDICT CLEAR' in result.stdout:
                break
            if result.returncode or time.monotonic() >= deadline:
                emit('RESEARCH_STOP', {'reason': 'host_not_clear'})
                return 3
            time.sleep(1)
    if a.npu:
        witness = context_report()
        emit('PRE_NPU_CONTEXT_WITNESS', witness)
        if 'No hardware contexts running' not in witness:
            return 3
    child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding='utf-8', errors='replace', bufsize=1)
    proc = psutil.Process(child.pid)
    lines = queue.Queue()
    def read_stdout():
        for line in child.stdout:
            lines.put(line)
    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()
    start = time.monotonic()
    previous = cpu_snapshot()
    peak = 0
    contended = False
    stop = None
    last_smi = 0
    try:
        while child.poll() is None or reader.is_alive() or not lines.empty():
            try:
                print(scrub(lines.get(timeout=0.1)), end='', flush=True)
                while not lines.empty():
                    print(scrub(lines.get_nowait()), end='', flush=True)
            except queue.Empty:
                pass
            now = time.monotonic()
            tree = descendants(proc)
            own.update(p.pid for p in tree)
            rss = 0
            for p in tree:
                try:
                    rss += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak = max(peak, rss)
            if child.poll() is None and (now - start > a.seconds or rss > a.rss_gib * 1024**3):
                stop = 'time_limit' if now - start > a.seconds else 'tree_rss_limit'
                kill_owned(proc)
            if now - previous[0] >= 2:
                current = cpu_snapshot()
                foreign = peers(previous, current, own)
                contended |= bool(foreign)
                emit('HOST_TIMELINE', {'elapsed_seconds': round(now-start, 3), 'tree_rss_bytes': rss,
                                      'foreign_peers': foreign})
                previous = current
            if a.npu and now-last_smi >= 5 and child.poll() is None:
                witness = context_report()
                emit('NPU_TIMELINE', witness)
                own.update(p.pid for p in descendants(proc))
                pids = context_pids(witness)
                emit('NPU_OWNERSHIP', {'context_pids': sorted(pids), 'owned_pids': sorted(own),
                                      'foreign_pids': sorted(pids-own)})
                contended |= bool(pids - own)
                last_smi = now
        rc = child.wait()
    finally:
        if child.poll() is None:
            kill_owned(proc)
            child.wait()
    emit('RESEARCH_PROCESS_RESULT', {'exit_code': rc, 'stop': stop, 'peak_tree_rss_bytes': peak or None,
         'rss_measurement': 'maximum observed across sampled descendant working sets; zero observations reported as null',
         'elapsed_seconds': time.monotonic()-start, 'foreign_contention_observed': contended,
         'timing_eligible': not (contended or stop or a.checks_only or rc)})
    return 4 if stop else 5 if contended and not a.checks_only and rc == 0 else rc


if __name__ == '__main__':
    raise SystemExit(main())
