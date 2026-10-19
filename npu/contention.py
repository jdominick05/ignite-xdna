"""Read-only Windows CPU/GPU/NPU contention snapshot without a PowerShell child.

CPU and PDH GPU rate counters share the same sampling interval. A fresh XRT
idle witness follows the sample. Missing/invalid telemetry never implies idle.
PDH ABI: https://learn.microsoft.com/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import ctypes as ct
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import psutil

INFERENCE=re.compile(r'^(llama.*|python.*|onnx.*|ollama.*|vllm.*)\.exe$',re.I)


def owned_process_ids(study_root,gpu_pid=None):
    """This process, its worker and only the exact research-supervisor ancestor."""
    owned={os.getpid()}
    if gpu_pid is not None:owned.add(gpu_pid)
    supervisor=Path(study_root).resolve()/'tools/research_process.py'
    for parent in psutil.Process().parents():
        try:
            args=parent.cmdline()
            if len(args)>1 and (Path(parent.cwd())/args[1]).resolve()==supervisor:
                owned.add(parent.pid)
        except (psutil.Error,OSError):
            continue
    return owned


class _Value(ct.Union):
    _fields_=[('doubleValue',ct.c_double),('largeValue',ct.c_int64),('pointer',ct.c_void_p)]


class _Formatted(ct.Structure):
    _fields_=[('status',ct.c_uint32),('value',_Value)]


class _Item(ct.Structure):
    _fields_=[('name',ct.c_wchar_p),('formatted',_Formatted)]


class _GpuQuery:
    def __init__(self):
        self.api=ct.WinDLL('pdh.dll')
        self.query=ct.c_void_p();self.counter=ct.c_void_p()
        signatures={
            'PdhOpenQueryW':[ct.c_wchar_p,ct.c_size_t,ct.POINTER(ct.c_void_p)],
            'PdhAddEnglishCounterW':[ct.c_void_p,ct.c_wchar_p,ct.c_size_t,ct.POINTER(ct.c_void_p)],
            'PdhCollectQueryData':[ct.c_void_p],
            'PdhGetFormattedCounterArrayW':[ct.c_void_p,ct.c_uint32,ct.POINTER(ct.c_uint32),ct.POINTER(ct.c_uint32),ct.c_void_p],
            'PdhCloseQuery':[ct.c_void_p],
        }
        for name,args in signatures.items():
            fn=getattr(self.api,name);fn.argtypes=args;fn.restype=ct.c_uint32
        try:
            self._check(self.api.PdhOpenQueryW(None,0,ct.byref(self.query)))
            self._check(self.api.PdhAddEnglishCounterW(self.query,r'\GPU Engine(*)\Utilization Percentage',0,ct.byref(self.counter)))
            self._check(self.api.PdhCollectQueryData(self.query))
        except BaseException:
            self.close();raise

    @staticmethod
    def _check(status):
        if status:raise OSError(f'PDH telemetry failed: 0x{status:08x}')

    def sample(self):
        self._check(self.api.PdhCollectQueryData(self.query))
        size=ct.c_uint32();count=ct.c_uint32()
        more=self.api.PdhGetFormattedCounterArrayW(self.counter,0x200,ct.byref(size),ct.byref(count),None)
        if more!=0x800007D2:raise OSError(f'PDH sizing failed: 0x{more:08x}')
        if not 0<size.value<=64*1024*1024:raise ValueError('PDH counter array bound')
        buffer=ct.create_string_buffer(size.value)
        self._check(self.api.PdhGetFormattedCounterArrayW(self.counter,0x200,ct.byref(size),ct.byref(count),buffer))
        if not 0<count.value<=size.value//ct.sizeof(_Item):raise ValueError('PDH item count bound')
        items=ct.cast(buffer,ct.POINTER(_Item));busy=[];invalid=0
        for index in range(count.value):
            item=items[index];value=item.formatted.value.doubleValue
            if item.formatted.status not in (0,1) or not math.isfinite(value) or value<0:
                invalid+=1;continue
            if value>=1:
                name=item.name or '';match=re.search(r'pid_(\d+)',name)
                busy.append(dict(pid=int(match[1]) if match else None,engine=name,utilization_percent=value))
        return not invalid,busy,dict(instances=count.value,invalid_instances=invalid)

    def close(self):
        if self.query.value:
            self.api.PdhCloseQuery(self.query);self.query=ct.c_void_p()


def _cpu_times():
    result={};known=True
    for process in psutil.process_iter(['pid','name'],ad_value=None):
        info=process.info;name=info['name'] or ''
        if not INFERENCE.match(name):continue
        try:
            with process.oneshot():
                cpu=process.cpu_times();key=(info['pid'],process.create_time())
            result[key]=(name,cpu.user+cpu.system)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            known=False
    return result,known


def snapshot(sample_seconds=.5):
    if not .25<=sample_seconds<=5:raise ValueError('Use a 0.25..5 second sample')
    with ExitStack() as resources:
        return _snapshot(sample_seconds,resources)


def _snapshot(sample_seconds,resources):
    blockers=[];gpu=None;gpu_error=None;gpu_known=False;engines=[];gpu_details={}
    start=time.perf_counter()
    try:
        gpu=_GpuQuery()
        resources.callback(gpu.close)
    except (OSError,ValueError,AttributeError) as error:
        gpu_error=str(error)
    before,cpu_known=_cpu_times();sample_start=time.perf_counter();wall_start=time.time()
    time.sleep(sample_seconds)
    after,after_known=_cpu_times();elapsed=time.perf_counter()-sample_start
    cpu_known=cpu_known and after_known
    cpu=[]
    for (pid,creation),(name,value) in after.items():
        old=before.get((pid,creation))
        # A process created during the interval has a valid zero-time origin.
        if old is None and creation<wall_start:
            cpu_known=False;continue
        cores=max(0,(value-(old[1] if old else 0))/elapsed)
        cpu.append(dict(pid=pid,name=name,active_cpu_cores=round(cores,3)))
        if cores>=.1:blockers.append(f'CPU inference work: PID {pid}')
    if not cpu_known:blockers.append('CPU inference telemetry unavailable')
    if gpu is not None:
        try:
            gpu_known,engines,gpu_details=gpu.sample()
        except (OSError,ValueError) as error:
            gpu_error=str(error)
        finally:
            gpu.close()
    if engines:blockers.append('GPU engine activity observed')
    if not gpu_known:blockers.append('GPU engine telemetry unavailable')
    npu_known=npu_idle=False
    try:
        smi=Path(os.environ.get('WINDIR','C:/Windows'))/'System32/AMD/xrt-smi.exe'
        witness=subprocess.run([str(smi),'examine','-r','aie-partitions'],capture_output=True,text=True,timeout=10,check=False)
        npu_known=witness.returncode==0 and 'AIE Partitions' in witness.stdout
        npu_idle=npu_known and 'No hardware contexts running' in witness.stdout
    except (OSError,subprocess.TimeoutExpired):
        pass
    if not npu_known:blockers.append('NPU context telemetry unavailable')
    elif not npu_idle:blockers.append('NPU contexts active')
    return dict(utc=datetime.now(timezone.utc).isoformat(),sample_seconds=elapsed,
                snapshot_seconds=time.perf_counter()-start,inference_processes=cpu,
                cpu_telemetry_known=cpu_known,gpu_telemetry_known=gpu_known,
                busy_gpu_engines=engines,gpu_details=gpu_details,gpu_error=gpu_error,
                npu_telemetry_known=npu_known,npu_idle=npu_idle,idle_verified=not blockers,
                blockers=blockers,scope='Read-only preflight snapshot; no process is stopped')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-idle',action='store_true')
    args=parser.parse_args();report=snapshot();print(json.dumps(report,indent=2))
    if args.require_idle and not report['idle_verified']:raise SystemExit(2)
