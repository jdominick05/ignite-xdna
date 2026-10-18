"""Native drafter operations using the ignite-xdna XRT harness.

All model arithmetic in this module executes on the NPU. Python packs BF16
bits, selects buffers, and dispatches. Context creation and host round trips
make this a composition baseline; it is not the optimized serving backend.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .common import sha
from . import stream as npu_stream

HERE=Path(__file__).resolve().parent


class Bf16Ops:
    def __init__(self,cache_dir,poison=0x7fc12345,precise_projection=False,transport=None,single_image=False):
        self.cache_dir=Path(cache_dir).resolve()
        self.poison=poison
        self.precise_projection=precise_projection
        self.transport=transport
        if single_image and not precise_projection:
            raise ValueError("Single-image dispatch requires compensated projections")
        self.single_image=single_image
        self.dispatches=[]
        # Bind this instance to source bytes once. Loaded packet images retain
        # their own identity; PacketSession verifies admitted files at exit.
        paths=list(HERE.glob("*.cc"))+[HERE/"stream.py",HERE/"common.py",HERE/"ops.py"]
        self._source_hashes={str(path):sha(path) for path in paths}
        self._build_meta={}

    def dispatch(self,function,packets,output_words,state_words=0):
        filenames={"draft_projection":"gemv.cc","draft_projection_precise":"gemv_precise.cc","draft_norm":"rmsnorm.cc","draft_rope":"rope.cc",
                   "draft_softmax":"softmax.cc","draft_geglu":"geglu.cc","draft_residual":"residual.cc","draft_topk":"topk.cc"}
        source=HERE/filenames[function]
        paths=[source,HERE/"stream.py",HERE/"common.py",HERE/"ops.py"]
        entry=function;optimization="-O3"
        if self.single_image:
            opcodes={"draft_projection_precise":1,"draft_norm":2,"draft_rope":3,"draft_softmax":4,
                     "draft_geglu":5,"draft_residual":6,"draft_topk":7}
            if function not in opcodes:raise ValueError("Operation absent from single image")
            padded=np.zeros((packets.shape[0],4112),np.uint32)
            padded[:,:packets.shape[1]]=packets
            padded[:,15]=opcodes[function];packets=padded
            entry="draft_transformer";source=HERE/"dispatch.cc";optimization="-Os"
            paths=list(HERE.glob("*.cc"))+[HERE/"stream.py",HERE/"common.py",HERE/"ops.py"]
            output_words,state_words=1056,64
        sources={str(path):self._source_hashes[str(path)] for path in paths}
        spec=dict(function=entry,optimization=optimization,packet_words=packets.shape[1],output_words=output_words,
                  blocks=packets.shape[0],state_words=state_words,sources=sources)
        identity=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
        folder=self.cache_dir/(entry+"-"+identity)
        contract=dict(scope="Native operation composition; separate qualification required",spec_sha256=identity)
        if not folder.exists():
            npu_stream.build(folder,source,entry,packets.shape[1],output_words,packets.shape[0],sources,contract,state_words=state_words,optimization=optimization)
        if folder not in self._build_meta:
            self._build_meta[folder]=json.loads((folder/"build.json").read_text())
        meta=self._build_meta[folder]
        if meta["contract"]!=contract or meta["sources"]!=sources:
            raise ValueError("Cached operation image identity mismatch")
        start=time.perf_counter()
        execute=self.transport.execute if self.transport is not None else npu_stream.execute
        result,evidence=execute(folder,np.ascontiguousarray(packets))
        self.dispatches.append(dict(function=function,build=str(folder/"build.json"),
                                    wall_ms=(time.perf_counter()-start)*1000,**evidence))
        if not np.all(result[:,:16]==self.poison):
            raise ValueError("Native output prefix guard changed")
        return result

    def projection(self,x,weight):
        x=np.asarray(x).reshape(-1)
        if x.dtype!=np.uint16 or weight.dtype!=np.uint16 or weight.ndim!=2 or weight.shape[1]!=x.size:
            raise ValueError("BF16 projection shape or dtype mismatch")
        n,k=weight.shape
        pn,pk=((n+15)//16)*16,((k+255)//256)*256
        padded=np.zeros((pn,pk),np.uint16);padded[:n,:k]=weight
        inputs=np.zeros(pk,np.uint16);inputs[:k]=x
        packets=np.zeros(((pn//16)*(pk//256),2192),np.uint32)
        cursor=0;final=[]
        for row in range(0,pn,16):
            for column in range(0,pk,256):
                packet=packets[cursor]
                packet[:3]=[self.poison,int(column==0),int(column+256==pk)]
                packet[16:144].view(np.uint16)[:]=inputs[column:column+256]
                block=padded[row:row+16,column:column+256]
                packed=block.T if self.precise_projection else block.reshape(4,4,32,8).transpose(2,0,3,1)
                packet[144:].view(np.uint16)[:]=packed.reshape(-1)
                if column+256==pk:final.append(cursor)
                cursor+=1
        function="draft_projection_precise" if self.precise_projection else "draft_projection"
        out=self.dispatch(function,packets,48,32 if self.precise_projection else 64)
        if not np.all(out[:,24:]==self.poison):raise ValueError("Projection suffix guard changed")
        return out[np.array(final),16:24].copy().view(np.uint16).reshape(-1)[:n].copy()

    def norm(self,x,weight):
        shape=x.shape;x=np.asarray(x).reshape(-1)
        if x.dtype!=np.uint16 or weight.dtype!=np.uint16 or weight.size not in (256,512) or x.size>2048 or x.size%weight.size:
            raise ValueError("BF16 normalization shape or dtype mismatch")
        packet=np.zeros((1,1296),np.uint32)
        packet[0,:3]=[self.poison,weight.size,x.size//weight.size]
        packet[0,3:4].view(np.float32)[0]=1.0/weight.size
        packet[0,16:1040].view(np.uint16)[:x.size]=x
        packet[0,1040:].view(np.uint16)[:weight.size]=weight.reshape(-1)
        out=self.dispatch("draft_norm",packet,1056)
        if not np.all(out[0,16+x.size//2:]==self.poison):raise ValueError("Normalization suffix guard changed")
        return out[0,16:16+x.size//2].copy().view(np.uint16).reshape(shape)

    def rope(self,q,cosine,sine):
        heads,width=q.shape
        if q.dtype!=np.uint16 or cosine.dtype!=np.uint16 or sine.dtype!=np.uint16 or heads!=4 or width not in (256,512):
            raise ValueError("BF16 rotary shape or dtype mismatch")
        packet=np.zeros((1,1552),np.uint32)
        packet[0,:3]=[self.poison,width,heads]
        packet[0,16:1040].view(np.uint16)[:q.size]=q.reshape(-1)
        packet[0,1040:1296].view(np.uint16)[:width]=cosine.reshape(-1)
        packet[0,1296:].view(np.uint16)[:width]=sine.reshape(-1)
        out=self.dispatch("draft_rope",packet,1056)
        if not np.all(out[0,16+q.size//2:]==self.poison):raise ValueError("Rotary suffix guard changed")
        return out[0,16:16+q.size//2].copy().view(np.uint16).reshape(q.shape)

    def softmax(self,scores):
        if scores.dtype!=np.uint16 or scores.ndim!=2 or not 0<scores.shape[1]<=32768:
            raise ValueError("BF16 softmax shape or dtype mismatch")
        rows,length=scores.shape
        count=(length+511)//512
        packets=np.zeros((rows*2*count,272),np.uint32)
        emit=[];cursor=0
        for head in range(rows):
            indices=[]
            for phase in (0,1):
                for offset in range(0,length,512):
                    valid=min(512,length-offset)
                    packets[cursor,:4]=[self.poison,int(phase==0 and offset==0),phase,valid]
                    packets[cursor,16:].view(np.uint16)[:valid]=scores[head,offset:offset+valid]
                    if phase:indices.append((cursor,valid))
                    cursor+=1
            emit.append(indices)
        out=self.dispatch("draft_softmax",packets,288,32)
        if not np.all(out[:,272:]==self.poison):raise ValueError("Softmax suffix guard changed")
        return np.stack([np.concatenate([out[index,16:272].copy().view(np.uint16)[:valid] for index,valid in indices]) for indices in emit])

    def geglu(self,gate,up,table):
        if any(x.dtype!=np.uint16 for x in (gate,up,table)) or gate.size!=2048 or up.size!=2048 or table.size!=4096:
            raise ValueError("BF16 GeGLU input or table mismatch")
        packet=np.zeros((1,4112),np.uint32);packet[0,0]=self.poison
        packet[0,16:1040].view(np.uint16)[:]=gate.reshape(-1)
        packet[0,1040:2064].view(np.uint16)[:]=up.reshape(-1)
        packet[0,2064:].view(np.uint16)[:]=table.reshape(-1)
        out=self.dispatch("draft_geglu",packet,1056)
        if not np.all(out[0,1040:]==self.poison):raise ValueError("GeGLU suffix guard changed")
        return out[0,16:1040].copy().view(np.uint16)

    def residual(self,x,residual,scale=np.uint16(0x3f80)):
        if x.dtype!=np.uint16 or residual.dtype!=np.uint16 or x.shape!=residual.shape or not 0<x.size<=2048 or x.size%16:
            raise ValueError("BF16 residual input mismatch")
        if not isinstance(scale,(np.uint16,int)) or not 0<=int(scale)<=65535:
            raise ValueError("Scale must be raw BF16 bits")
        packet=np.zeros((1,2064),np.uint32);packet[0,:3]=[self.poison,x.size,int(scale)]
        packet[0,16:1040].view(np.uint16)[:x.size]=x.reshape(-1)
        packet[0,1040:].view(np.uint16)[:x.size]=residual.reshape(-1)
        out=self.dispatch("draft_residual",packet,1056)
        if not np.all(out[0,16+x.size//2:]==self.poison):raise ValueError("Residual suffix guard changed")
        return out[0,16:16+x.size//2].copy().view(np.uint16).reshape(x.shape)

    def topk(self,scores,k):
        if scores.dtype!=np.uint16 or not 0<scores.size<=4096 or not 0<k<=min(32,scores.size):
            raise ValueError("BF16 top-k input mismatch")
        if np.any((scores&0x7f80)==0x7f80):raise ValueError("Top-k requires finite BF16 scores")
        packet=np.zeros((1,2064),np.uint32);packet[0,:3]=[self.poison,scores.size,k]
        packet[0,16:].view(np.uint16)[:scores.size]=scores.reshape(-1)
        out=self.dispatch("draft_topk",packet,64)
        if not np.all(out[0,16+k:]==self.poison):raise ValueError("Top-k suffix guard changed")
        return out[0,16:16+k].copy().view(np.int32)

