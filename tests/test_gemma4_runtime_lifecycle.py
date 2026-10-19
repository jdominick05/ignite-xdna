"""Hardware-free checks for request admission and abandoned streams."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import threading
from types import SimpleNamespace

import pytest

from npu.gemma4.runtime import HardwareBusy, PersistentAssistant
from npu.gemma4 import runtime
from npu import contention


def unloaded_assistant():
    assistant = object.__new__(PersistentAssistant)
    assistant.closed = False
    assistant.sources = {}
    assistant._lock = threading.Lock()
    assistant._cancel = threading.Event()
    assistant._stream = None
    assistant.gpu = assistant._temp = None
    assistant._activity = assistant.activity_path = None
    assistant.collect_trace = False
    assistant.input_depth = 1
    assistant.vector_copy = False
    assistant.context_tokens = 512
    assistant.max_prompt_tokens = 256
    return assistant


def test_unread_stream_close_releases_admission():
    assistant = unloaded_assistant()
    for _ in range(2):
        stream = assistant.generate([1])
        assert assistant._lock.locked()
        with pytest.raises(RuntimeError, match='already active'):
            assistant.generate([1])
        stream.close()
        assert not assistant._lock.locked()
        assert assistant._stream is None
        assert assistant.last_result['cancelled']
    assistant.close()
    assistant.close()
    with pytest.raises(RuntimeError, match='closed'):
        assistant.generate([1])


def test_preflight_failure_releases_admission():
    assistant = unloaded_assistant()
    def busy():
        raise HardwareBusy('test workload')
    assistant._preflight = busy
    stream = assistant.generate([1])
    with pytest.raises(HardwareBusy):
        next(stream)
    assert not assistant._lock.locked()
    assert assistant.last_result['rejected']
    assistant.close()


def test_busy_admission_preserves_loaded_worker():
    assistant = unloaded_assistant()
    worker = object()
    assistant.gpu = worker
    def busy():
        raise HardwareBusy('test workload')
    assistant._preflight = busy
    with pytest.raises(HardwareBusy):
        next(assistant.generate([1]))
    assert assistant.gpu is worker
    assert not assistant._lock.locked()
    assistant.gpu = None
    assistant.close()


@pytest.mark.parametrize('supervisor', [True, False])
def test_only_exact_supervisor_ancestor_is_owned(monkeypatch, supervisor):
    assistant = unloaded_assistant()
    parent = SimpleNamespace(pid=12345,
                             cmdline=lambda: ['python.exe', 'tools/research_process.py'],
                             cwd=lambda: str(runtime.ROOT if supervisor else runtime.ROOT/'unrelated'))
    monkeypatch.setattr(contention.psutil, 'Process', lambda: SimpleNamespace(parents=lambda: [parent]))
    report = dict(inference_processes=[dict(pid=12345, active_cpu_cores=1)],
                  busy_gpu_engines=[], npu_idle=True,npu_telemetry_known=True,
                  cpu_telemetry_known=True,gpu_telemetry_known=True)
    monkeypatch.setattr(runtime, 'contention_snapshot',lambda: dict(report))
    if supervisor:
        assert 12345 in assistant._preflight()['owned_pids']
    else:
        with pytest.raises(HardwareBusy) as caught:
            assistant._preflight()
        assert 12345 not in caught.value.report['owned_pids']


@pytest.mark.parametrize('missing',['cpu_telemetry_known','gpu_telemetry_known','npu_telemetry_known','npu_idle'])
def test_missing_sensor_blocks_admission(monkeypatch,missing):
    assistant=unloaded_assistant()
    report=dict(inference_processes=[],busy_gpu_engines=[],npu_idle=True,
                cpu_telemetry_known=True,gpu_telemetry_known=True,npu_telemetry_known=True)
    report[missing]=False
    monkeypatch.setattr(runtime,'contention_snapshot',lambda: report)
    monkeypatch.setattr(runtime,'owned_process_ids',lambda *a:set())
    with pytest.raises(HardwareBusy):assistant._preflight()


@pytest.mark.parametrize('pid,busy',[(100,False),(101,True),(None,True)])
def test_gpu_ownership_filter(monkeypatch,pid,busy):
    assistant=unloaded_assistant()
    report=dict(inference_processes=[],busy_gpu_engines=[dict(pid=pid,utilization_percent=10)],
                npu_idle=True,cpu_telemetry_known=True,gpu_telemetry_known=True,npu_telemetry_known=True)
    monkeypatch.setattr(runtime,'contention_snapshot',lambda: report)
    monkeypatch.setattr(runtime,'owned_process_ids',lambda *a:{100})
    if busy:
        with pytest.raises(HardwareBusy):assistant._preflight()
    else:
        assert assistant._preflight() is report


def test_snapshot_interrupt_closes_query(monkeypatch):
    closed=[]
    query=SimpleNamespace(close=lambda:closed.append(True))
    monkeypatch.setattr(contention,'_GpuQuery',lambda:query)
    def interrupted():raise KeyboardInterrupt
    monkeypatch.setattr(contention,'_cpu_times',interrupted)
    with pytest.raises(KeyboardInterrupt):contention.snapshot()
    assert closed


@pytest.mark.parametrize('alive',[True,False])
def test_only_live_activity_handle_is_excluded_from_contention(monkeypatch,alive):
    assistant=unloaded_assistant()
    assistant._activity=SimpleNamespace(process=SimpleNamespace(pid=202,poll=lambda:None if alive else 0))
    report=dict(inference_processes=[dict(pid=202,active_cpu_cores=2)],
                busy_gpu_engines=[dict(pid=202,utilization_percent=40)],
                npu_idle=True,npu_telemetry_known=True,cpu_telemetry_known=True,gpu_telemetry_known=True)
    monkeypatch.setattr(runtime,'contention_snapshot',lambda:dict(report))
    monkeypatch.setattr(runtime,'owned_process_ids',lambda *a:set())
    if alive:assert 202 in assistant._preflight()['owned_pids']
    else:
        with pytest.raises(HardwareBusy):assistant._preflight()


@pytest.mark.parametrize('failure',[False,True])
def test_activity_joins_before_leaving_draft_scope(monkeypatch,failure):
    assistant=unloaded_assistant();assistant.activity_path='explicit-helper';assistant.llama_runtime='runtime'
    calls=[]
    class Activity:
        identity=dict(hardware=True)
        process=SimpleNamespace(pid=202)
        active=False
        def __init__(self,*a):calls.append('create')
        def start(self,duration):
            assert duration==100;self.active=True;calls.append('start')
        def wait(self,*,stop):
            assert stop;self.active=False;calls.append('join')
            return dict(completed=True,gpu_validated=True,dispatches=2,elapsed_ms=20)
    monkeypatch.setattr(runtime,'ActivityHelper',Activity)
    result=dict(operation_ms={},gpu_activity=dict(windows=[]))
    try:
        with assistant._draft_activity(result):
            assert assistant._activity.active
            if failure:raise ValueError('cancelled native operation')
    except ValueError:
        assert failure
    assert calls==['create','start','join'] and not assistant._activity.active
    assert result['gpu_activity']['windows'][0]['gpu_validated']
    with assistant._draft_activity(result):pass
    assert calls==['create','start','join','start','join']


def test_stream_verifies_only_after_activity_has_joined(monkeypatch):
    assistant=unloaded_assistant();assistant.activity_path='helper';assistant.model_loads=1
    events=[]
    class Activity:
        active=False;identity=dict(hardware=True);process=SimpleNamespace(pid=202)
        def start(self,duration):self.active=True;events.append('start')
        def wait(self,*,stop):
            assert stop;self.active=False;events.append('join')
            return dict(completed=True,gpu_validated=True,dispatches=2,elapsed_ms=20)
        def close(self):pass
    assistant._activity=Activity()
    class Gpu:
        process=SimpleNamespace(pid=100,poll=lambda:None)
        def render(self,tokens):return dict(text_bytes=b'x'*len(tokens),eog=[False]*len(tokens))
        def decode(self,tokens):
            assert not assistant._activity.active;events.append('verify')
            return dict(predictions=[11,12])
        def close(self):pass
    assistant.gpu=Gpu();assistant._preflight=lambda:{}
    assistant._prepare=lambda tokens:dict(predictions=[7])
    class State:
        hidden=None;position=1
        def __init__(self,*args):pass
        def append(self,response,keep):self.position+=keep
    class Session:
        active=False;images={};contexts_created=0;evictions=0
        def __init__(self,**kwargs):pass
        def __enter__(self):self.active=True;return self
        def __exit__(self,*args):self.active=False;self.post_witness='No hardware contexts running'
    def draft(*args):
        assert assistant._activity.active;events.append('draft');return 11,None,{}
    assistant.drafter=SimpleNamespace(invalidate_prefix=lambda:None,step=draft)
    monkeypatch.setattr(runtime,'SharedState',State);monkeypatch.setattr(runtime,'PacketSession',Session)
    chunks=list(assistant.generate([1],max_new_tokens=2,depth=1))
    assert [t for c in chunks for t in c.token_ids]==[7,11]
    assert events==['start','draft','join','verify']
    assert assistant.last_result['npu_released'] and assistant.last_result['gpu_activity']['idle']
    assistant.close()


@pytest.mark.parametrize('limit,context',[(256,512),(2048,2560)])
def test_prompt_bound_before_hardware(limit,context):
    assistant=unloaded_assistant()
    assistant.max_prompt_tokens=limit;assistant.context_tokens=context
    stream=assistant.generate([1]*limit)
    stream.close()
    with pytest.raises(ValueError,match='Prompt requires'):
        assistant.generate([1]*(limit+1))
    assert assistant.gpu is None and not assistant._lock.locked()
    assistant.close()
