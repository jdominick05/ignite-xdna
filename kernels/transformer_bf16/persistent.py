"""Synchronous packet transport with bounded, explicitly owned XRT contexts.

The caller must keep exclusive NPU ownership for this session. Idle witnesses
bracket the session; bounded research runs additionally monitor foreign contexts.
This transport does not establish fused execution or serving performance.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
from collections import OrderedDict
import gc
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

from .common import STUDY,command,sha

_LEASE=threading.Lock()


class PacketSession:
    """Reuse at most two contexts, optionally sharing byte-identical programs."""
    def __init__(self,max_contexts=2,share_program=False):
        if max_contexts not in (1,2):raise ValueError('Use one or two cached contexts')
        self.max_contexts=max_contexts
        self.share_program=share_program
        self.images=OrderedDict()
        self.active=False
        self.thread=None
        self.contexts_created=0
        self.cache_hits=0
        self.evictions=0
        self._verified={}
        self._registered_files={}

    def __enter__(self):
        if self.active or not _LEASE.acquire(blocking=False):
            raise RuntimeError('Another packet session is active in this process')
        try:
            witness=command(['C:/Windows/System32/AMD/xrt-smi.exe','examine','-r','aie-partitions'],STUDY)
            if 'No hardware contexts running' not in witness:
                raise RuntimeError('NPU is busy; did not open packet session')
            self.active=True;self.thread=threading.get_ident()
            self._verified={}
            self._driver_sha256=sha(STUDY/'src/ignite_xdna/runtime/driver.py')
            self._registered_files={str(STUDY/'src/ignite_xdna/runtime/driver.py'):self._driver_sha256}
            self.pre_witness=witness
            return self
        except BaseException:
            _LEASE.release();raise

    @staticmethod
    def _release(image):
        # Drop all BOs before their kernel context; no process-global XRT handles.
        while image['streams']:
            _,stream=image['streams'].popitem(last=False)
            stream['input']=stream['output']=stream['instructions']=None
        image['harness'].close()

    def close(self,verify=True):
        if not self.active:return
        if threading.get_ident()!=self.thread:raise RuntimeError('Close on the owning thread')
        try:
            while self.images:
                _,image=self.images.popitem(last=False)
                self._release(image)
            gc.collect()
            if verify:
                for path,expected in self._registered_files.items():
                    if sha(path)!=expected:raise ValueError(f'Admitted packet image changed: {path}')
        finally:
            self.active=False;self.thread=None;_LEASE.release()

    def __exit__(self,exc_type,exc,tb):
        self.close(verify=exc_type is None)
        try:
            self.post_witness=command(['C:/Windows/System32/AMD/xrt-smi.exe','examine','-r','aie-partitions'],STUDY)
        except Exception:
            if exc_type is None:raise
            return False
        if exc_type is None and 'No hardware contexts running' not in self.post_witness:
            raise RuntimeError('Contexts remain after packet session cleanup')
        return False

    def execute(self,folder,packets):
        if not self.active or threading.get_ident()!=self.thread:
            raise RuntimeError('Use packet transport on its owning thread inside a context manager')
        folder=Path(folder).resolve()
        # Existing streams dispatch only already-loaded BOs. Admit each image
        # once and verify all admitted disk identities again at successful exit.
        # A new Bf16Ops/session is required to adopt edited sources or binaries.
        if str(folder) not in self._verified:
            meta=json.loads((folder/'build.json').read_text())
            manifest_sha=sha(folder/'build.json')
            identities=dict(meta['sources'])
            for collection in ('artifacts','program_artifacts'):
                identities.update({str(folder/name):value for name,value in meta.get(collection,{}).items()})
            identities[str(folder/'build.json')]=manifest_sha
            for path,expected in identities.items():
                if path in self._registered_files and self._registered_files[path]!=expected:
                    raise ValueError('Conflicting source/artifact identities in session')
                if sha(path)!=expected:raise ValueError(f'Source or artifact changed: {path}')
            self._registered_files.update(identities)
            self._verified[str(folder)]=(meta,manifest_sha)
        meta,manifest_sha=self._verified[str(folder)]
        if packets.dtype!=np.uint32 or packets.shape!=(meta['blocks'],meta['packet_words']):
            raise ValueError('Packet dimensions differ from compiled stream')
        key=str(folder)
        if self.share_program:
            program=meta.get('program_artifacts')
            required={'prj/main.pdi','prj/kernels_main.json','prj/memTopology_main.json'}
            if not program or set(program)!=required:
                raise ValueError('Missing exact program/interface identities')
            key=hashlib.sha256(json.dumps(program,sort_keys=True).encode()).hexdigest()
        context_hit=key in self.images
        if context_hit:
            image=self.images.pop(key);self.images[key]=image
        else:
            if len(self.images)>=self.max_contexts:
                _,old=self.images.popitem(last=False);self._release(old)
                self.evictions+=1
            sys.path.insert(0,str(STUDY/'src'))
            import ignite_xdna
            if not Path(ignite_xdna.__file__).resolve().is_relative_to(STUDY/'src'):
                raise RuntimeError('Runtime import resolved outside this checkout')
            from ignite_xdna.runtime.driver import XrtSiliconHarness
            h=XrtSiliconHarness()
            image=dict(harness=h,streams=OrderedDict())
            try:
                h.load_xclbin(str(folder/'final.xclbin'))
            except BaseException:
                self._release(image);raise
            self.images[key]=image;self.contexts_created+=1
        stream_key=str(folder)
        hit=stream_key in image['streams']
        if hit:
            stream=image['streams'].pop(stream_key);image['streams'][stream_key]=stream
            if stream['manifest_sha256']!=manifest_sha:raise ValueError('Cached image manifest changed')
            self.cache_hits+=1
        else:
            if len(image['streams'])>=32:
                _,old=image['streams'].popitem(last=False)
                old['input']=old['output']=old['instructions']=None
            h=image['harness']
            stream=dict(input=None,output=None,instructions=None,manifest_sha256=manifest_sha)
            try:
                stream['instructions'],stream['instruction_bytes']=h.create_instruction_bo(str(folder/'insts.bin'))
                stream['input']=h.create_host_bo(packets.nbytes,3)
                stream['output_bytes']=meta['blocks']*meta['output_words']*4
                stream['output']=h.create_host_bo(stream['output_bytes'],4)
            except BaseException:
                stream['input']=stream['output']=stream['instructions']=None
                self.close(verify=False);raise
            image['streams'][stream_key]=stream
        h=image['harness'];bi=stream['input'];bo=stream['output']
        to=h.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        fr=h.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        handle=None
        try:
            bi.write(packets.tobytes(),0);bi.sync(to)
            bo.write(np.full(stream['output_bytes']//4,0x7fc98765,np.uint32).tobytes(),0);bo.sync(to)
            start=time.perf_counter()
            handle,status=h.dispatch_kernel(stream['instructions'],stream['instruction_bytes'],bi,bo,timeout_ms=5000)
            elapsed=(time.perf_counter()-start)*1000
            if 'COMPLETED' not in str(status):raise RuntimeError(f'Dispatch {status}; stop without retry')
            bo.sync(fr)
            result=np.frombuffer(bo.read(stream['output_bytes'],0),np.uint32).copy().reshape(meta['blocks'],meta['output_words'])
            bi.sync(fr)
            if not np.array_equal(np.frombuffer(bi.read(packets.nbytes,0),np.uint32),packets.reshape(-1)):
                raise ValueError('Input BO modified')
            return result,dict(dispatch_ms=elapsed,status=str(status),input_unchanged=True,
                persistent_transport=True,image_cache_hit=hit,contexts_created=self.contexts_created,
                context_cache_hit=context_hit,exact_program_sharing=self.share_program,
                context_cache_size=len(self.images),runtime_driver_sha256=self._driver_sha256)
        except BaseException:
            handle=bi=bo=h=None
            self.close(verify=False)
            raise
        finally:
            handle=None
