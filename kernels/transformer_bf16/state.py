"""Compile packet commands whose BF16 intermediate values stay on the NPU.

Host code packs weights and metadata only. A static initialization map rejects
reads of unwritten arena locations before compilation or device execution.
This is a research command stream, outside the ONNX graph compiler.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .common import sha
from . import stream

HERE=Path(__file__).resolve().parent
WORDS,OUTPUT,STATE_WORDS,ARENA=4112,1056,8256,16384


class ArenaCompiler:
    """Bind prepared programs to one source snapshot, checked by PacketSession."""
    def __init__(self,exact_norm=False,compact=False,small_packets=False,input_depth=1,vector_copy=False):
        if small_packets and not exact_norm:raise ValueError('Small packets currently require exact-norm matrix mode')
        if input_depth not in (1,2) or input_depth==2 and not (small_packets and compact):
            raise ValueError('Double-buffered arena requires compact small packets')
        if vector_copy and not small_packets:raise ValueError('Vector copies currently require small packets')
        self.exact_norm=exact_norm
        self.compact=compact
        self.small_packets=small_packets
        self.input_depth=input_depth
        self.vector_copy=vector_copy
        self.packet_words=2192 if small_packets else WORDS
        self.upload_limit=(self.packet_words-16)*2
        self.sources={str(p):sha(p) for p in HERE.iterdir() if p.suffix in ('.cc','.hpp')}
        self.sources.update({str(p):sha(p) for p in (HERE/'state.py',HERE/'stream.py',HERE/'common.py')})
        if compact:self.sources[str(HERE/'compact.py')]=sha(HERE/'compact.py')
        self._builds={}

    def build(self,cache,blocks,output_packets=None):
        selected=tuple(output_packets) if self.compact and output_packets is not None else None
        if self.compact and not selected:raise ValueError('Compact arena requires declared exports')
        key=(str(Path(cache).resolve()),blocks,selected)
        if key in self._builds:return self._builds[key]
        spec=dict(function='draft_state_transformer',blocks=blocks,packet_words=self.packet_words,
                  output_words=OUTPUT,state_words=STATE_WORDS,sources=self.sources,
                  optimization='-Oz' if self.exact_norm else '-Os',
                  exact_norm=self.exact_norm,compact=self.compact,output_packets=selected,small_packets=self.small_packets,
                  input_depth=self.input_depth,vector_copy=self.vector_copy)
        identity=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
        folder=Path(key[0])/('draft_state_transformer-'+identity)
        contract=dict(scope=__doc__,spec_sha256=identity)
        if not folder.exists():
            source=HERE/('state_dispatch_vector.cc' if self.vector_copy else 'state_dispatch_small.cc' if self.small_packets else 'state_dispatch_exact.cc' if self.exact_norm else 'state_dispatch.cc')
            stream.build(folder,source,spec['function'],self.packet_words,OUTPUT,blocks,
                         self.sources,contract,state_words=STATE_WORDS,optimization=spec['optimization'],output_packets=selected,
                         input_depth=self.input_depth)
        meta=json.loads((folder/'build.json').read_text())
        if meta['contract']!=contract or meta['sources']!=self.sources:raise ValueError('Arena program identity mismatch')
        self._builds[key]=folder
        return folder


class ArenaProgram:
    def __init__(self,poison=0x7fc12345,compiler=None):
        self.poison=poison;self.packets=[];self.exports={}
        self.compiler=compiler if compiler is not None else ArenaCompiler()
        self.uploads={}
        self.matrices={}
        self.rotations={}
        self.softmax_lengths={}
        self.initialized=np.zeros(ARENA,dtype=bool)
        self._gelu_table=None

    def _range(self,slot,count,read=False):
        if not isinstance(slot,int) or not isinstance(count,int) or slot<0 or count<0 or slot+count>ARENA:
            raise ValueError('Activation arena span out of bounds')
        if read and not self.initialized[slot:slot+count].all():raise ValueError('Read of unwritten activation slot')
        if not read and self._gelu_table is not None and slot<12288 and 8192<slot+count:
            raise ValueError('Write overlaps resident GELU table')

    def _packet(self,opcode):
        packet=np.zeros(self.compiler.packet_words,np.uint32);packet[0]=self.poison;packet[15]=opcode
        return packet

    def _emit(self,packet,source=None,second=None,destination=None):
        flags=0
        for span,offset,bit in ((source,5,1),(second,6,2)):
            if span is not None:
                slot,count=span;self._range(slot,count,read=True)
                packet[offset]=slot;packet[offset+3]=count;flags|=bit
        if destination is not None:
            slot,count=destination;self._range(slot,count)
            packet[7]=slot;packet[10]=count;flags|=4
            self.initialized[slot:slot+count]=True
        packet[4]=flags
        self.packets.append(packet)
        return len(self.packets)-1

    def upload(self,slot,bits,label=None):
        bits=np.asarray(bits)
        if bits.dtype!=np.uint16 or not 0<bits.size<=self.compiler.upload_limit:raise ValueError('BF16 upload shape or dtype')
        self._range(slot,int(bits.size))
        packet=self._packet(9);packet[7]=slot;packet[8]=bits.size
        packet[16:].view(np.uint16)[:bits.size]=bits.reshape(-1)
        if label is not None:
            if label in self.uploads:raise ValueError('Duplicate upload label')
            self.uploads[label]=dict(packet=len(self.packets),count=int(bits.size))
        self.initialized[slot:slot+bits.size]=True;self.packets.append(packet)

    def download(self,label,slot,count):
        self._range(slot,count,read=True)
        if label in self.exports or not 0<count<=2048:raise ValueError('Duplicate/oversized export')
        packet=self._packet(10);packet[5]=slot;packet[10]=count
        self.exports[label]=dict(packet=len(self.packets),count=count,dtype='bf16')
        self.packets.append(packet)

    def projection(self,source,weight,destination,precise=False,label=None):
        if precise and self.compiler.exact_norm:
            raise ValueError('Exact-norm arena supports matrix projections only')
        if weight.dtype!=np.uint16 or weight.ndim!=2:raise ValueError('BF16 matrix expected')
        n,k=map(int,weight.shape);self._range(source,k,read=True);self._range(destination,n)
        if not n or not k:raise ValueError('Empty projection dimensions')
        if source<destination+n and destination<source+k:
            raise ValueError('Projection input and output must not overlap')
        if label is not None:
            if label in self.matrices:raise ValueError('Duplicate matrix label')
            self.matrices[label]=dict(packet=len(self.packets),shape=(n,k),precise=precise)
        pn,pk=((n+15)//16)*16,((k+255)//256)*256
        padded=np.zeros((pn,pk),np.uint16);padded[:n,:k]=weight
        for row in range(0,pn,16):
            for column in range(0,pk,256):
                packet=self._packet(1 if precise else 8)
                last=column+256==pk;packet[1:3]=[int(column==0),int(last)]
                block=padded[row:row+16,column:column+256]
                packed=block.T if precise else block.reshape(4,4,32,8).transpose(2,0,3,1)
                packet[144:2192].view(np.uint16)[:]=packed.reshape(-1)
                self._emit(packet,source=(source+column,min(256,k-column)),
                    destination=(destination+row,min(16,n-row)) if last else None)

    def norm(self,source,weight,destination,rows=1):
        width=int(weight.size);count=rows*width
        if weight.dtype!=np.uint16 or width not in (256,512) or not 0<count<=2048:
            raise ValueError('Unsupported BF16 RMSNorm')
        packet=self._packet(2);packet[1:3]=[width,rows]
        packet[3:4].view(np.float32)[0]=1.0/width
        packet[1040:1296].view(np.uint16)[:width]=weight.reshape(-1)
        self._emit(packet,source=(source,count),destination=(destination,count))

    def rope(self,slot,cosine,sine,width,label=None):
        if width not in (256,512) or any(v.dtype!=np.uint16 or v.size!=width for v in (cosine,sine)):
            raise ValueError('Supplied BF16 rotary factors required')
        packet=self._packet(3);packet[1:3]=[width,4]
        if label is not None:
            if label in self.rotations:raise ValueError('Duplicate rotary label')
            self.rotations[label]=dict(packet=len(self.packets),width=width)
        packet[1040:1296].view(np.uint16)[:width]=cosine.reshape(-1)
        packet[1296:1552].view(np.uint16)[:width]=sine.reshape(-1)
        self._emit(packet,source=(slot,4*width),destination=(slot,4*width))

    def softmax(self,slot,length,capacity=None,label=None):
        if not 0<length<=32768:raise ValueError('Unsupported softmax length')
        if label is not None:
            if label in self.softmax_lengths or type(capacity) is not int or not length<=capacity<=ARENA:
                raise ValueError('Named dynamic softmax requires a bounded arena capacity')
            self._range(slot,capacity,read=True)
            self.softmax_lengths[label]=dict(packets=[],offsets=[],phases=[],capacity=capacity)
        elif capacity is not None:
            raise ValueError('Softmax capacity requires a label')
        self._range(slot,length,read=True)
        for phase in (0,1):
            for offset in range(0,capacity if label is not None else length,512):
                valid=max(0,min(512,length-offset));packet=self._packet(4)
                packet[1:4]=[int(phase==0 and offset==0),phase,valid]
                if label is not None:
                    span=self.softmax_lengths[label]
                    span['packets'].append(len(self.packets));span['offsets'].append(offset);span['phases'].append(phase)
                self._emit(packet,source=(slot+offset,valid),
                    destination=(slot+offset,valid) if phase else None)

    def geglu(self,gate,up,table,destination):
        if table.dtype!=np.uint16 or table.size!=4096:raise ValueError('BF16 GELU table expected')
        if self.compiler.small_packets:
            if self._gelu_table is None:
                if self.initialized[8192:12288].any():raise ValueError('Resident GELU table overlaps initialized activations')
                self.upload(8192,table)
                self._gelu_table=table.copy()
                self._gelu_table.flags.writeable=False
            elif not np.array_equal(table,self._gelu_table):raise ValueError('One resident GELU table per program required')
            packet=self._packet(11);packet[12]=8192
        else:
            packet=self._packet(5);packet[2064:].view(np.uint16)[:]=table.reshape(-1)
        self._emit(packet,source=(gate,2048),second=(up,2048),destination=(destination,2048))

    def residual(self,source,second,destination,count=256,scale=np.uint16(0x3f80)):
        if not 0<count<=2048 or count%16:raise ValueError('Unsupported BF16 residual span')
        packet=self._packet(6);packet[1:3]=[count,int(scale)]
        self._emit(packet,source=(source,count),second=(second,count),destination=(destination,count))

    def topk(self,label,source,count,k):
        if not 0<count<=4096 or not 0<k<=min(32,count) or label in self.exports:
            raise ValueError('Unsupported top-k shape or export')
        packet=self._packet(7);packet[1:3]=[count,k]
        index=self._emit(packet,source=(source,count))
        self.exports[label]=dict(packet=index,count=k,dtype='i32')

    def build(self,cache):
        if not self.packets:raise ValueError('Empty arena program')
        return self.compiler.build(cache,len(self.packets),sorted({v['packet'] for v in self.exports.values()}))

    def output_exports(self):
        if not self.compiler.compact:return self.exports
        indices={p:i for i,p in enumerate(sorted({v['packet'] for v in self.exports.values()}))}
        return {k:dict(v,packet=indices[v['packet']]) for k,v in self.exports.items()}

    def packet_array(self):
        packets=np.stack(self.packets)
        if self.compiler.compact:
            if np.any(packets[:,11]):raise ValueError('Reserved compact command marker is already set')
            packets[sorted({v['packet'] for v in self.exports.values()}),11]=1
            packets[-1,11]|=2
        return packets

    def prepare(self,cache):
        return PreparedArena(self.build(cache),self.packet_array(),self.output_exports(),self.uploads,
                             self.matrices,self.rotations,self.poison,self.softmax_lengths)

    def execute(self,cache,transport):
        start=time.perf_counter()
        folder=self.build(cache)
        build_ms=(time.perf_counter()-start)*1000;start=time.perf_counter()
        packets=self.packet_array()
        pack_ms=(time.perf_counter()-start)*1000
        result,evidence=transport.execute(folder,packets)
        if not np.all(result[:,:16]==self.poison):raise ValueError('Arena command failed or output prefix guard changed')
        values={}
        for label,export in self.output_exports().items():
            frame=result[export['packet'],16:]
            values[label]=(frame.copy().view(np.uint16) if export['dtype']=='bf16' else frame.copy().view(np.int32))[:export['count']].copy()
        return values,dict(build=str(folder/'build.json'),commands=len(self.packets),
                           host_phases_ms=dict(build_lookup=build_ms,packet_stack=pack_ms),**evidence)


def pack_matrix(weight,precise=False):
    """Pack BF16 tiles in command order without a Python loop per tile."""
    if weight.dtype!=np.uint16 or weight.ndim!=2 or not all(weight.shape):
        raise ValueError('Nonempty BF16 matrix required')
    n,k=weight.shape
    pn,pk=((n+15)//16)*16,((k+255)//256)*256
    padded=np.zeros((pn,pk),np.uint16);padded[:n,:k]=weight
    tiles=padded.reshape(pn//16,16,pk//256,256).transpose(0,2,1,3)
    if precise:
        packed=tiles.transpose(0,1,3,2)
    else:
        packed=tiles.reshape(pn//16,pk//256,4,4,32,8).transpose(0,1,4,2,5,3)
    return np.ascontiguousarray(packed).reshape(-1,4096).view(np.uint32)


class PreparedArena:
    """Owned immutable packet template with explicitly named dynamic data.

    Data patching copies the template before editing. Every dispatch verifies
    its complete input BO. The optional resident-input transport uploads only
    pages differing from its last verified buffer; the default uploads it all.
    """
    def __init__(self,folder,packets,exports,uploads,matrices,rotations,poison,softmax_lengths=None):
        self.folder=folder;self.poison=poison
        self._packets=packets;self._packets.flags.writeable=False
        self.exports={k:dict(v) for k,v in exports.items()}
        self.uploads={k:dict(v) for k,v in uploads.items()}
        self.matrices={k:dict(v) for k,v in matrices.items()}
        self.rotations={k:dict(v) for k,v in rotations.items()}
        self.softmax_lengths={k:dict(v) for k,v in (softmax_lengths or {}).items()}

    def execute(self,transport,uploads=None,matrices=None,rotations=None,lengths=None):
        start=time.perf_counter()
        packets=self._packets
        if uploads or matrices or rotations or lengths:
            packets=packets.copy()
            for label,bits in (uploads or {}).items():
                if label not in self.uploads:raise ValueError('Unknown prepared upload')
                span=self.uploads[label];bits=np.asarray(bits)
                if bits.dtype!=np.uint16 or bits.size!=span['count']:
                    raise ValueError('Prepared upload shape or dtype changed')
                packets[span['packet'],16:].view(np.uint16)[:span['count']]=bits.reshape(-1)
            for label,weight in (matrices or {}).items():
                if label not in self.matrices:raise ValueError('Unknown prepared matrix')
                span=self.matrices[label];weight=np.asarray(weight)
                n,k=span['shape']
                if weight.dtype!=np.uint16 or weight.shape!=(n,k):
                    raise ValueError('Prepared matrix shape or dtype changed; rebuild program')
                index=span['packet']
                packed=pack_matrix(weight,span['precise'])
                packets[index:index+len(packed),144:2192]=packed
            for label,factors in (rotations or {}).items():
                if label not in self.rotations:raise ValueError('Unknown prepared rotary factors')
                span=self.rotations[label]
                if len(factors)!=2:raise ValueError('Two rotary factors required')
                for bits,offset in zip(factors,(1040,1296)):
                    bits=np.asarray(bits)
                    if bits.dtype!=np.uint16 or bits.size!=span['width']:
                        raise ValueError('Prepared rotary shape or dtype changed')
                    packets[span['packet'],offset:offset+256].view(np.uint16)[:span['width']]=bits.reshape(-1)
            for label,length in (lengths or {}).items():
                if label not in self.softmax_lengths:raise ValueError('Unknown prepared softmax length')
                span=self.softmax_lengths[label]
                if type(length) is not int or not 1<=length<=span['capacity']:
                    raise ValueError('Prepared softmax length exceeds capacity')
                for index,offset,phase in zip(span['packets'],span['offsets'],span['phases']):
                    valid=max(0,min(512,length-offset))
                    packets[index,3]=valid
                    packets[index,8]=valid
                    if phase:packets[index,10]=valid
        pack_ms=(time.perf_counter()-start)*1000
        result,evidence=transport.execute(self.folder,packets,reuse_input=True)
        if not np.all(result[:,:16]==self.poison):raise ValueError('Prepared arena output prefix guard changed')
        values={}
        for label,export in self.exports.items():
            frame=result[export['packet'],16:]
            values[label]=(frame.copy().view(np.uint16) if export['dtype']=='bf16' else frame.copy().view(np.int32))[:export['count']].copy()
        return values,dict(build=str(self.folder/'build.json'),commands=len(packets),prepared=True,
                           host_phases_ms=dict(build_lookup=0.0,packet_stack=pack_ms),**evidence)
