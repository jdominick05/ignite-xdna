"""Experimental persistent Gemma4 E2B GPU target and BF16 NPU assistant.

Weights stay loaded between requests. Each request prefills its complete prompt
and owns a bounded NPU session; no GPU-only reference generation runs here.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import codecs
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from kernels.transformer_bf16.common import save, sha
from kernels.transformer_bf16.persistent import PacketSession
from tools.ggml_gpu_target.client import Target
from tools.ggml_gpu_target.activity import ActivityHelper
from npu.contention import snapshot as contention_snapshot,owned_process_ids
from .draft import Draft, SharedState

ROOT=Path(__file__).resolve().parents[2]


@contextmanager
def _measure(phases,name):
    """Accumulate host wall time only inside an operation, including failures."""
    start=time.perf_counter()
    try:yield
    finally:phases[name]=phases.get(name,0.0)+(time.perf_counter()-start)*1000


class HardwareBusy(RuntimeError):
    """Another workload or missing telemetry prevents admission."""
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report=report


@dataclass(frozen=True)
class TokenChunk:
    token_ids: tuple
    text: str
    finished: bool = False


def admit_hardware(gpu=None,*,auxiliary_processes=()):
    """Apply the serving admission policy without loading model assets."""
    report=contention_snapshot()
    gpu_pid=gpu.process.pid if gpu is not None and gpu.process.poll() is None else None
    owned=owned_process_ids(ROOT,gpu_pid)
    owned.update(p.pid for p in auxiliary_processes if p.poll() is None)
    report['owned_pids']=sorted(owned)
    foreign_cpu=[p for p in report['inference_processes'] if p['pid'] not in owned and p['active_cpu_cores']>=.1]
    foreign_gpu=[p for p in report['busy_gpu_engines'] if p['pid'] not in owned and p['utilization_percent']>2]
    if (not report['npu_idle'] or not report['npu_telemetry_known'] or not report['cpu_telemetry_known']
            or not report['gpu_telemetry_known'] or foreign_cpu or foreign_gpu):
        raise HardwareBusy('Inference hardware is busy or telemetry is unavailable',report)
    return report


class PersistentAssistant:
    def __init__(self, artifacts, llama_runtime, *, collect_trace=False,input_depth=1,vector_copy=False,context_tokens=512,gpu_activity_helper=None):
        if type(context_tokens) is not int or context_tokens not in (512,2560):raise ValueError('Use context 512 or 2560')
        self.context_tokens=context_tokens
        self.max_prompt_tokens=256 if context_tokens==512 else 2048
        self.artifacts=Path(artifacts).resolve()
        self.llama_runtime=Path(llama_runtime).resolve()
        self.model=self.artifacts/'models/gemma-4-E2B-it-Q5_K_M-pleQ4-imatrix.gguf'
        self.executable=self.artifacts/'bin/ggml_gpu_target.exe'
        weights_path=self.artifacts/'npu/gemma-4-E2B-it-assistant.weights.npz'
        metadata_path=weights_path.with_suffix('.json')
        metadata=json.loads(metadata_path.read_text())
        if sha(weights_path)!=metadata['sha256']:raise ValueError('Assistant weight export changed')
        self.config=metadata['config']
        if self.config['backbone_hidden_size']!=1536:raise ValueError('E2B assistant required')
        with np.load(weights_path,allow_pickle=False) as f:weights={k:f[k].copy() for k in f.files}
        rope=self.artifacts/f'npu/rope-e2b-live-{context_tokens}-20261018.npz'
        lut=self.artifacts/'npu/gelu-bf16-tanh-v1.npz'
        rm,lm=[json.loads(p.with_suffix('.json').read_text()) for p in (rope,lut)]
        local_config=self.artifacts/'npu/local-checkpoints/gemma-4-E2B-it-assistant/config.json'
        if (sha(rope)!=rm['sha256'] or sha(lut)!=lm['sha256'] or rm['maximum']<context_tokens
                or sha(local_config)!=rm['config_sha256'] or json.loads(local_config.read_text())!=self.config):
            raise ValueError('Rotary/GELU/configuration identity mismatch')
        with np.load(rope,allow_pickle=False) as f:rotations={k:f[k].copy() for k in f.files}
        with np.load(lut,allow_pickle=False) as f:table=f['table'].copy()
        for array in [*weights.values(),*rotations.values(),table]:array.flags.writeable=False
        self.drafter=Draft(self.artifacts,self.config,weights,rotations,table,
                           prepared=True,compact=True,kv_capacity=256,small_packets=True,input_depth=input_depth,vector_copy=vector_copy)
        self.input_depth=input_depth
        self.vector_copy=vector_copy
        files=[Path(__file__),Path(__file__).with_name('arena.py'),Path(__file__).with_name('draft.py'),
               ROOT/'tools/ggml_gpu_target/client.py',ROOT/'tools/ggml_gpu_target/worker.cpp',
               ROOT/'tools/ggml_gpu_embedding/embedding.cpp',
               ROOT/'tools/check-inference-contention.ps1',ROOT/'tools/research_process.py',ROOT/'npu/contention.py',weights_path,metadata_path,rope,lut,
               rope.with_suffix('.json'),lut.with_suffix('.json'),local_config,self.model,self.executable]
        files += [self.llama_runtime/name for name in ('llama.dll','ggml.dll','ggml-base.dll','ggml-hip.dll')]
        self.activity_path=Path(gpu_activity_helper).resolve() if gpu_activity_helper is not None else None
        if self.activity_path is not None:
            files += [self.activity_path,ROOT/'tools/ggml_gpu_target/activity.py',ROOT/'tools/ggml_gpu_target/gpu_activity.cpp',
                      ROOT/'tools/ggml_gpu_target/build_activity.cmd',self.llama_runtime/'amdhip64_7.dll',
                      Path(os.environ.get('WINDIR','C:/Windows'))/'System32/d3dcompiler_47.dll']
        self.sources={str(p):sha(p) for p in files}
        self.collect_trace=collect_trace
        self.gpu=None
        self._activity=None
        self._temp=None
        self._stream=None
        self._lock=threading.Lock()
        self._cancel=threading.Event()
        self.closed=False
        self.model_loads=0
        self.last_result=None

    def _preflight(self):
        # Desktop/system activity up to 2% is recorded, not treated as a timing
        # qualification. Foreign inference CPU work and occupied NPU block use.
        children=() if self._activity is None else (self._activity.process,)
        return admit_hardware(self.gpu,auxiliary_processes=children)

    @contextmanager
    def _draft_activity(self,result):
        if self.activity_path is None:
            yield;return
        phases=result['operation_ms'];record=result['gpu_activity']
        if self._activity is None:
            with _measure(phases,'activity_setup'):
                self._activity=ActivityHelper(self.activity_path,self.llama_runtime)
        record.update(identity=self._activity.identity,pid=self._activity.process.pid,stop_phase='before_verification')
        with _measure(phases,'activity_start'):self._activity.start(100)
        try:yield
        finally:
            with _measure(phases,'activity_join'):
                record['windows'].append(self._activity.wait(stop=True))

    def _prepare(self, tokens):
        if self.gpu is None:
            self._temp=tempfile.TemporaryDirectory(prefix='ignition-gemma4-')
            prompt=Path(self._temp.name)/'prompt.i32'
            prompt.write_bytes(np.asarray(tokens,np.int32).tobytes())
            env=os.environ.copy();env['GGML_CUDA_DISABLE_FUSION']='1'
            self.gpu=Target(self.executable,self.llama_runtime,self.model,prompt,env=env,context_tokens=self.context_tokens)
            self.model_loads+=1
            return self.gpu.initial
        if self.gpu.process.poll() is not None:raise RuntimeError('GPU worker exited; close and recreate the assistant')
        return self.gpu.prepare(tokens)

    def generate(self, tokens, *, max_new_tokens=64, depth=2):
        if self.closed:raise RuntimeError('Assistant is closed')
        if (not 1<=len(tokens)<=self.max_prompt_tokens or any(isinstance(t,bool) or not isinstance(t,(int,np.integer)) or not 0<=int(t)<262144 for t in tokens)):
            raise ValueError(f'Prompt requires 1..{self.max_prompt_tokens} valid integer token IDs')
        if type(max_new_tokens) is not int or not 1<=max_new_tokens<=64 or type(depth) is not int or not 1<=depth<=4:
            raise ValueError('Use 1..64 output tokens and 1..4 drafts')
        if not self._lock.acquire(blocking=False):raise RuntimeError('A generation is already active')
        self._cancel.clear()
        self._stream=self._generate([int(t) for t in tokens],max_new_tokens,depth)
        # Enter the generator's cleanup scope without touching hardware. A
        # caller may close it before asking for the first token.
        next(self._stream)
        return self._stream

    def _generate(self, prompt, limit, depth):
        emitted=[];trace=[];accepted_total=proposed_total=0
        decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
        pieces=[];terminal=None;complete=False;session=None;finish_reason=None
        compiler_output=io.StringIO()
        result=dict(scope=__doc__,prompt_tokens=len(prompt),prompt_sha256=hashlib.sha256(np.asarray(prompt,np.int32).tobytes()).hexdigest(),
                    completed=False,cancelled=False,input_depth=self.input_depth,vector_copy=self.vector_copy,
                    context_tokens=self.context_tokens,source_sha256=dict(self.sources))
        phases={}
        result['operation_ms']=phases
        result['gpu_activity']=dict(enabled=self.activity_path is not None,windows=[])
        self.last_result=result
        try:
            yield None  # consumed internally by generate(), never exposed
            with _measure(phases,'admission'):
                result['preflight']=self._preflight()
            with _measure(phases,'target_prepare'):
                current=self._prepare(prompt)
            result['worker_pid']=self.gpu.process.pid
            self.drafter.invalidate_prefix()
            with _measure(phases,'target_state'):
                state=SharedState(current);pending=current['predictions'][-1]
            session=PacketSession(share_program=True,resident_inputs=True)
            with session:
                while len(emitted)<limit and not self._cancel.is_set():
                    bonus=pending
                    with _measure(phases,'render'):
                        rendered=self.gpu.render([bonus])
                    emitted.append(bonus)
                    pieces.append(rendered['text_bytes'])
                    done=rendered['eog'][0] or len(emitted)==limit
                    text=decoder.decode(rendered['text_bytes'],final=done)
                    if done:
                        finish_reason='eos' if rendered['eog'][0] else 'length'
                        terminal=TokenChunk((bonus,),text,True);complete=True;break
                    # A target bonus is already verified. Show it before NPU
                    # drafting instead of delaying first text until verification.
                    yield TokenChunk((bonus,),text)
                    if self._cancel.is_set():break
                    proposals=[];hidden=state.hidden;dispatches=[]
                    with self._draft_activity(result):
                        for _ in range(min(depth,limit-len(emitted))):
                            with _measure(phases,'draft'),redirect_stdout(compiler_output):
                                token,hidden,evidence=self.drafter.step(self.gpu,session,state,
                                                          proposals[-1] if proposals else bonus,hidden)
                            proposals.append(token)
                            if self.collect_trace:dispatches.append(evidence)
                            if self._cancel.is_set():break
                        if self._cancel.is_set():break
                    if self._activity is not None and self._activity.active:
                        raise RuntimeError('GPU activity remains before target verification')
                    with _measure(phases,'target_verify'):
                        verified=self.gpu.decode([bonus,*proposals])
                    accepted=0
                    for token,prediction in zip(proposals,verified['predictions']):
                        if token!=prediction:break
                        accepted+=1
                    output=proposals[:accepted]
                    with _measure(phases,'render'):
                        rendered=self.gpu.render(output) if output else None
                    done=False
                    if output and any(rendered['eog']):
                        accepted=rendered['eog'].index(True)+1;output=output[:accepted]
                        with _measure(phases,'render'):
                            rendered=self.gpu.render(output)
                        done=True
                    keep=1+accepted
                    pending=verified['predictions'][keep-1]
                    with _measure(phases,'target_rollback'):
                        if keep<1+len(proposals):self.gpu.rollback(state.position+keep)
                    with _measure(phases,'target_state'):
                        state.append(verified,keep)
                    emitted.extend(output)
                    if state.position!=len(prompt)+len(emitted):raise ValueError('Accepted target prefix diverged')
                    accepted_total+=accepted;proposed_total+=len(proposals)
                    if self.collect_trace:trace.append(dict(proposals=proposals,accepted=accepted,dispatches=dispatches))
                    if output:
                        done=done or len(emitted)==limit
                        pieces.append(rendered['text_bytes'])
                        text=decoder.decode(rendered['text_bytes'],final=done)
                        chunk=TokenChunk(tuple(output),text,done)
                        if done:
                            finish_reason='eos' if any(rendered['eog']) else 'length'
                            terminal=chunk;complete=True;break
                        yield chunk
            if complete:
                with _measure(phases,'render'):
                    whole=self.gpu.render(emitted)['text_bytes']
                if b''.join(pieces)!=whole:raise ValueError('Chunk rendering differs from complete output')
                result['generated_text']=whole.decode('utf-8',errors='replace')
            result.update(completed=complete,cancelled=not complete,finish_reason=finish_reason,tokens=emitted,accepted=accepted_total,
                          proposed=proposed_total,model_loads=self.model_loads,worker_pid=self.gpu.process.pid)
            result['npu_released']=not session.active and not session.images
            if terminal is not None:yield terminal
        except GeneratorExit:
            result.update(cancelled=not complete,tokens=emitted)
            raise
        except HardwareBusy as error:
            # Admission did not touch model state. Preserve an already loaded
            # worker so a foreign workload does not force a model reload.
            result.update(rejected=True,tokens=emitted,preflight=error.report)
            raise
        except BaseException:
            result.update(failed=True,tokens=emitted)
            self._close_gpu()
            raise
        finally:
            result['gpu_activity']['idle']=self._activity is None or not self._activity.active
            result['compiler_log_tail']=compiler_output.getvalue()[-65536:]
            compiler_output.close()
            if session is not None:
                result.update(npu_released=not session.active and not session.images,
                              contexts_created=session.contexts_created,evictions=session.evictions,
                              npu_idle='No hardware contexts running' in getattr(session,'post_witness',''))
            if self.collect_trace:result['rounds']=trace
            self._stream=None
            self._lock.release()

    def cancel(self):
        self._cancel.set()

    def _close_gpu(self):
        try:
            if self._activity is not None:self._activity.close();self._activity=None
        finally:
            if self.gpu is not None:self.gpu.close();self.gpu=None
            if self._temp is not None:self._temp.cleanup();self._temp=None

    def close(self):
        if self.closed:return
        if self._stream is not None:
            stream=self._stream
            stream.close()
            if self._stream is stream:
                self._stream=None;self._lock.release()
        self._close_gpu();self.closed=True
        if any(sha(Path(p))!=h for p,h in self.sources.items()):raise ValueError('Assistant source or artifact changed')

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
