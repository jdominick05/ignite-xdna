#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-core streamed attention: inspect, build, preregister, check, benchmark.

Run builds through scripts/research-iron.sh and device work through
scripts/research-lowlevel.sh --npu. No model weights or user text are saved.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
IRON = Path.home() / "mlir-aie" / "ironenv"
PEANO = IRON / "Lib/site-packages/llvm-aie"
INSTALL = IRON / "Lib/site-packages/mlir_aie"
UPSTREAM = Path.home() / "mlir-aie"
PACKET = 2832  # i32 words; 16 header + bf16 Q(1024), Kt(2048), V(2048), mask(512)
OUT = 1056     # i32 words; 16-word guards either side of fp32 O[16,64]
STATE = 1056
SCRATCH = 1024
STACK = 4096
ATOL, RTOL, L2 = .01, .02, .02
SEEDS = (11, 29, 47)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def command(args, cwd):
    p = subprocess.run([str(x) for x in args], cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-5000:] + p.stderr[-7000:], flush=True)
        raise RuntimeError(f"subprocess failed: {Path(args[0]).name} ({p.returncode})")
    return p.stdout


def software_float_helpers(symbols):
    return sorted(set(re.findall(r"\b__(?:(?:add|sub|mul|div|eq|ne|lt|le|gt|ge|cmp)sf[23]|fix(?:uns)?sfsi|float(?:un)?sisf)\b",symbols)))


def rounded(x):
    x = np.asarray(x, np.float32)
    u = x.view(np.uint32)
    bits = ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def bits(x):
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def tiled(x, r, c):
    rows, cols = x.shape
    return x.reshape(rows//r, r, cols//c, c).transpose(0, 2, 1, 3).reshape(-1)


def oracle(q, k, v, mask):
    s = q.astype(np.float64) @ k.astype(np.float64).swapaxes(-1, -2) * .125
    s = np.where(mask == 2, 0.0, s)
    s = np.where(mask, s, -np.inf)
    if not np.all(mask.any(axis=-1)):
        raise ValueError("effective mask has a fully masked row")
    p = np.exp(s - s.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    return p @ v.astype(np.float64)


def inspect(model, out):
    """Execute only the exported mask ancestors, with no external weight data loaded."""
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper
    m = onnx.load(str(model), load_external_data=False)
    byout = {o: n for n in m.graph.node for o in n.output}
    soft = [n for n in m.graph.node if n.op_type == "Softmax"][:2]
    masks, scales = [], []
    initial = {t.name: t for t in m.graph.initializer}
    for n in soft:
        add = byout[n.input[0]]
        assert add.op_type == "Add"
        where = next(byout[i] for i in add.input if i in byout and byout[i].op_type == "Where")
        masks.append(where.output[0])
        mm = next(byout[i] for i in add.input if i in byout and byout[i].op_type == "MatMul")
        scale = []
        for inp in mm.input:
            mul = byout[inp]
            assert mul.op_type == "Mul"
            t = next(initial[i] for i in mul.input if i in initial)
            scale.append(float(numpy_helper.to_array(t)))
        scales.append(scale)
    needed, names = set(), set(masks)
    def visit(name):
        if name in byout and name not in needed:
            needed.add(name)
            for i in byout[name].input:
                names.add(i)
                visit(i)
    for name in masks:
        visit(name)
    nodes = [n for n in m.graph.node if any(o in needed for o in n.output)]
    ins = [i for i in m.graph.input if i.name in names]
    inits = [t for t in m.graph.initializer if t.name in names]
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in inits):
        raise ValueError("mask extraction unexpectedly needs external model weights")
    g = helper.make_graph(nodes, "attention_masks", ins,
                          [helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, None) for n in masks], inits)
    tiny = helper.make_model(g, opset_imports=m.opset_import, ir_version=m.ir_version)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(tiny.SerializeToString(), so, providers=["CPUExecutionProvider"])
    cases = {}
    for valid in (512, 193, 1, 0):
        inputs = {}
        for i in sess.get_inputs():
            if i.name == "attention_mask":
                inputs[i.name] = np.array([[1]*valid+[0]*(512-valid)], np.int64)
            elif i.name == "qtype":
                inputs[i.name] = np.array([0], np.int64)
            else:
                raise ValueError(f"unexpected mask input: {i.name}")
        values = sess.run(None, inputs)
        cases[str(valid)] = []
        for value in values:
            value = np.broadcast_to(value, (1, 1, 512, 512))[0, 0]
            allow = (value == 0).astype(np.uint8)
            # The export uses finite -FLT_MAX. An all-masked row has equal scores,
            # hence uniform probabilities, not zero output.
            empty = ~allow.any(axis=1)
            allow[empty] = 2
            cases[str(valid)].append(allow.astype(np.uint8))
    expected = np.abs(np.arange(512)[:, None] - np.arange(512)[None, :]) <= 64
    assert np.array_equal(cases["512"][0], np.ones((512, 512), np.uint8))
    assert np.array_equal(cases["512"][1], expected)
    assert all(abs(a*b-.125) < 1e-7 for a, b in scales)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "masks.npz", **{f"{mode}_{valid}": masks_[j]
             for valid, masks_ in cases.items() for j, mode in enumerate(("global", "local"))})
    # Validate finite-mask equivalence on oracle inputs small enough that adding
    # them to -FLT_MAX rounds back to -FLT_MAX, as in the source graph.
    info = dict(model_sha256=sha(model), graph_mask_outputs=masks, scale_factors=scales,
                scale=.125, local_inclusive_radius=64, local_interior_keys=129,
                padding="keys only; padded query rows still produce outputs",
                all_masked="finite -FLT_MAX yields uniform over all 512 keys",
                masks_sha256=sha(out / "masks.npz"))
    dump(out / "semantics.json", info)
    print(json.dumps(info), flush=True)


def harness(packets, outputs, resident="none"):
    ni, no = packets * PACKET, outputs * OUT
    fn = "attention_empty" if resident == "empty" else "attention_block"
    finish = "attention_zero" if resident == "empty" else "attention_finish"
    return f'''module {{
  aie.device(npu1) {{
    %s = aie.tile(0, 0)
    %t = aie.tile(0, 2)
    %state = aie.buffer(%t) {{sym_name = "attn_state"}} : memref<{STATE}xi32>
    %scratch = aie.buffer(%t) {{sym_name = "attn_scratch"}} : memref<{SCRATCH}xi32>
    func.func private @{fn}(memref<{PACKET}xi32>, memref<{STATE}xi32>, memref<{SCRATCH}xi32>) -> ()
    func.func private @{finish}(memref<{STATE}xi32>, memref<{OUT}xi32>) -> ()
    aie.objectfifo @inF(%s, {{%t}}, 1 : i32) : !aie.objectfifo<memref<{PACKET}xi32>>
    aie.objectfifo @outF(%t, {{%s}}, 1 : i32) : !aie.objectfifo<memref<{OUT}xi32>>
    %core = aie.core(%t) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c16 = arith.constant 16 : index
      %c1040 = arith.constant 1040 : index
      %c1056 = arith.constant {OUT} : index
      %forever = arith.constant 4294967295 : index
      %z = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      scf.for %it = %c0 to %forever step %c1 {{
        %si = aie.objectfifo.acquire @inF(Consume, 1) : !aie.objectfifosubview<memref<{PACKET}xi32>>
        %p = aie.objectfifo.subview.access %si[0] : !aie.objectfifosubview<memref<{PACKET}xi32>> -> memref<{PACKET}xi32>
        %calls = memref.load %p[%c3] : memref<{PACKET}xi32>
        scf.for %repeat = %z to %calls step %one : i32 {{
          func.call @{fn}(%p, %state, %scratch) : (memref<{PACKET}xi32>, memref<{STATE}xi32>, memref<{SCRATCH}xi32>) -> ()
        }}
        %last = memref.load %p[%c1] : memref<{PACKET}xi32>
        %done = arith.cmpi ne, %last, %z : i32
        scf.if %done {{
          %so = aie.objectfifo.acquire @outF(Produce, 1) : !aie.objectfifosubview<memref<{OUT}xi32>>
          %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<{OUT}xi32>> -> memref<{OUT}xi32>
          %poison = memref.load %p[%c2] : memref<{PACKET}xi32>
          scf.for %j = %c0 to %c1056 step %c1 {{
            memref.store %poison, %o[%j] : memref<{OUT}xi32>
          }}
          func.call @{finish}(%state, %o) : (memref<{STATE}xi32>, memref<{OUT}xi32>) -> ()
          aie.objectfifo.release @outF(Produce, 1)
        }}
        aie.objectfifo.release @inF(Consume, 1)
      }}
      aie.end
    }} {{link_with = "kernel.o", stack_size = {STACK} : i32}}
    aie.runtime_sequence(%a: memref<{ni}xi32>, %o: memref<{no}xi32>) {{
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, {ni//16}, 16][0, 0, 16, 1]) {{id = 0 : i64, metadata = @inF}} : memref<{ni}xi32>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][1, 1, {no//16}, 16][0, 0, 16, 1]) {{id = 1 : i64, metadata = @outF, issue_token = true}} : memref<{no}xi32>
      aiex.npu.dma_wait {{symbol = @outF}}
    }}
  }}
}}
'''


def packet_geometry(mask, heads):
    return int(heads * sum(sum(mask[i:i+16, j:j+32].any() for j in range(0,512,32))
                       for i in range(0,512,16))), heads * 32


def build(out, masks_path, heads, mode, valid, resident="none", implementation="vector"):
    mask = np.load(masks_path)[f"{mode}_{valid}"]
    count, outputs = packet_geometry(mask, heads)
    if resident != "none": count, outputs = 1, 1
    out.mkdir(parents=True, exist_ok=True)
    flags = ["-O2", "-std=c++20", "--target=aie2-none-unknown-elf", "-DNDEBUG", "-fstack-size-section",
             "-D__AIE_API_AIE_ADF_HPP__", f"-I{INSTALL / 'include'}",
             f"-I{UPSTREAM / 'aie_runtime_lib/AIE2'}", "-Wno-attributes", "-Wno-parentheses"]
    clang = PEANO / "bin/clang++.exe"
    # Compile the upstream helpers with the actual runtime include directory;
    # this distinguishes missing include flags from an unsupported operation.
    for name in ("bf16_exp", "softmax"):
        command([clang, *flags, "-c", UPSTREAM / f"aie_kernels/aie2/{name}.cc", "-o", out / f"{name}.o"], out)
    source=HERE / {"vector":"kernel_vector.cc", "scalar":"kernel.cc", "lut":"kernel_lut.cc"}[implementation]
    command([clang, *flags, "-c", source, "-o", out / "kernel_core.o"], out)
    undefined=command([PEANO / "bin/llvm-nm.exe","-u",out/"kernel_core.o"],out)
    allowed={"exp_ilut_ab","exp_ilut_cd","exp_flut_ab","exp_flut_cd","m_inv_lut"} if implementation=="lut" else set()
    unknown={line.split()[-1] for line in undefined.splitlines() if line.strip()}-allowed
    if implementation!="scalar" and unknown:
        raise ValueError(f"vector object contains unresolved runtime helpers: {sorted(unknown)}")
    command([clang, *flags, "-c", UPSTREAM / "aie_runtime_lib/AIE2/lut_based_ops.cpp", "-o", out / "lut.o"], out)
    command([clang, "--target=aie2-none-unknown-elf", "-r", "-nostdlib", out / "kernel_core.o", out / "lut.o", "-o", out / "kernel.o"], out)
    (out / "harness.mlir").write_text(harness(count, outputs,resident), encoding="utf-8")
    command([IRON / "Scripts/aiecc.exe", out / "harness.mlir", f"--peano={PEANO}",
             "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-xclbin", "--xclbin-name=final.xclbin",
             f"--tmpdir={out / 'prj'}"], out)
    ld = (out / "prj/ldScripts_main_core_0_2.ld.script").read_text()
    places = {m[1]: int(m[0], 16)-0x70000 for m in re.findall(
        r"\. = (0x[0-9A-Fa-f]+);\s*\n\s*([A-Za-z_]\w*) = \.;", ld)}
    sizes = dict(attn_state=STATE*4, attn_scratch=SCRATCH*4,
                 inF_cons_buff_0=PACKET*4, outF_buff_0=OUT*4)
    if not sizes.keys() <= places.keys():
        raise ValueError(f"missing linker symbols: {sizes.keys()-places.keys()}")
    spans = sorted((places[n], places[n]+s, n) for n,s in sizes.items())
    for lo, hi, name in spans:
        if lo < STACK or hi > 65536:
            raise ValueError(f"buffer outside tile memory: {name} {lo}:{hi}")
    if any(a[1] > b[0] for a,b in zip(spans,spans[1:])):
        raise ValueError("overlapping tile buffers")
    elfs = list((out / "prj").rglob("*.elf"))
    if len(elfs) != 1:
        raise ValueError(f"expected one linked core ELF, found {len(elfs)}")
    elf = elfs[0]
    symbols=command([PEANO / "bin/llvm-nm.exe",elf],out)
    if implementation == "lut":
        for name in ("exp_ilut_ab","exp_ilut_cd","exp_flut_ab","exp_flut_cd"):
            match=re.search(r"^([0-9a-fA-F]+) [A-Za-z] "+name+r"$",symbols,re.M)
            if not match:
                raise ValueError(f"missing linked LUT: {name}")
            lo=int(match[1],16)-0x70000
            spans.append((lo,lo+1024,name))
        spans.sort()
        if any(lo<STACK or hi>65536 for lo,hi,_ in spans) or any(a[1]>b[0] for a,b in zip(spans,spans[1:])):
            raise ValueError("LUT memory overlaps stack or tile buffers")
    helpers=software_float_helpers(symbols)
    if implementation!="scalar" and helpers:
        raise ValueError(f"linked vector program contains software float helpers: {helpers}")
    sections = command([PEANO / "bin/llvm-size.exe", "-A", elf], out)
    print(sections, flush=True)
    code = sum(int(m[1]) for m in re.findall(r"^(\.(?:text|text\.[^\s]+))\s+(\d+)\s+", sections, re.M))
    if code == 0 or code > 16384:
        raise ValueError(f"invalid program-memory size: {code}")
    stack_report=command([PEANO / "bin/llvm-readobj.exe","--stack-sizes",elf],out)
    stack_sizes=[int(s,16) for s in re.findall(r"Size: (0x[0-9A-Fa-f]+)",stack_report)]
    # Conservative sum of all recorded function frames, plus room for the MLIR
    # harness/call chain. The generated core has no large local arrays.
    if not stack_sizes or sum(stack_sizes)+256 > STACK:
        raise ValueError(f"stack exceeds reservation: {stack_report}")
    if f". += 0x{STACK:X}; /* stack */".lower() not in ld.lower():
        raise ValueError("linker did not reserve the requested stack")
    meta = dict(heads=heads, mode=mode, valid=valid, resident=resident, implementation=implementation, packets=count, outputs=outputs,
                placement=spans, code_bytes=code, kernel_sha256=sha(source), kernel_source=source.name,
                unresolved_helpers=undefined.splitlines(),
                software_float_helpers=helpers,
                stack_sizes=stack_sizes, stack_allowance_bytes=STACK,
                upstream_sha256={name:sha(UPSTREAM/name) for name in (
                    "aie_runtime_lib/AIE2/lut_based_ops.h","aie_runtime_lib/AIE2/lut_based_ops.cpp",
                    "aie_kernels/aie2/bf16_exp.cc","aie_kernels/aie2/softmax.cc")},
                masks_sha256=sha(masks_path), xclbin_sha256=sha(out / "final.xclbin"),
                instructions_sha256=sha(out / "insts.bin"))
    dump(out / "build.json", meta)
    print(json.dumps(meta), flush=True)


def case(heads, seed, kind):
    rng = np.random.default_rng(seed)
    q, k, v = [rounded(rng.normal(size=(heads,512,64))) for _ in range(3)]
    if kind in ("uniform", "zero"):
        q.fill(0)
    if kind == "zero":
        k.fill(0); v.fill(0)
    if kind == "peaked":
        q = rounded(q*4); k = rounded(k*4)
    return q,k,v


def pack_reference(q, k, v, mask, poison):
    rows = []
    for h in range(len(q)):
        for i in range(0,512,16):
            js = [j for j in range(0,512,32) if mask[i:i+16,j:j+32].any()]
            for index,j in enumerate(js):
                p = np.zeros(PACKET,np.uint32)
                p[:4] = [int(index==0), int(index==len(js)-1), poison, 1]
                qt = q[h,i:i+16].copy()
                qt[np.all(mask[i:i+16] == 2, axis=1)] = 0
                data = np.concatenate((bits(tiled(qt,4,8)),
                                       bits(tiled(k[h,j:j+32].T,8,4)),
                                       bits(tiled(v[h,j:j+32],8,4)),
                                       bits(mask[i:i+16,j:j+32].astype(np.float32).reshape(-1))))
                p[16:] = data.view(np.uint32)
                rows.append(p)
    return np.stack(rows)


def pack(q, k, v, mask, poison):
    """Tile each unique operand once, then assemble the legacy packet stream."""
    heads=len(q)
    qt=bits(q.copy()).reshape(heads,32,16,64)
    qt[:,np.all(mask==2,axis=1).reshape(32,16)]=0
    qt=qt.reshape(heads,32,4,4,8,8).transpose(0,1,2,4,3,5).reshape(heads,32,1024)
    kt=bits(k).reshape(heads,16,32,64).transpose(0,1,3,2).reshape(heads,16,8,8,8,4)
    kt=kt.transpose(0,1,2,4,3,5).reshape(heads,16,2048)
    vt=bits(v).reshape(heads,16,4,8,16,4).transpose(0,1,2,4,3,5).reshape(heads,16,2048)
    mt=mask.reshape(32,16,16,32).transpose(0,2,1,3).reshape(32,16,512)
    qi,ki=np.nonzero(mt.any(axis=-1))
    out=np.zeros((heads,len(qi),PACKET),np.uint32)
    out[:,:,0]=np.r_[True,qi[1:]!=qi[:-1]]
    out[:,:,1]=np.r_[qi[1:]!=qi[:-1],True]
    out[:,:,2]=poison; out[:,:,3]=1
    raw=out[:,:,16:].view(np.uint16)
    raw[:,:,:1024]=qt[:,qi]
    raw[:,:,1024:3072]=kt[:,ki]
    raw[:,:,3072:5120]=vt[:,ki]
    raw[:,:,5120:]=bits(mt[qi,ki].astype(np.float32))
    return out.reshape(-1,PACKET)


def preregister(out, builds, semantics):
    contract = dict(machine=platform.node(), shape=[512,16,64], atol=ATOL, rtol=RTOL, relative_l2=L2,
                    seeds=SEEDS, cases=["random","uniform","zero","peaked"],
                    poisons=[0x7fc12345,0x7fc54321], valid_lengths=[512,193,1,0],
                    correctness="all outputs including padded queries; all heads; two poisons; guards and inputs",
                    performance="full-head and 16-head dispatch, packing, write/sync, read/sync; CPU same sitting",
                    alternatives=["local block skipping reduces arithmetic and transfers",
                                  "vector arithmetic removes scalar software-float cost but packing remains",
                                  "resident work or transport limits the vector kernel", "unexplained"],
                    comparison="unchanged numerical gate, no scalar float helpers; same-sitting one-core vector controls and CPU; competitive attention requires full 16-head latency including required packing and transfers below CPU, not just faster dispatch",
                    semantics=json.loads(Path(semantics).read_text()),
                    sources={p.name:sha(p) for p in (HERE / "kernel.cc", HERE / "kernel_vector.cc", HERE / "kernel_lut.cc", HERE / "probe.py", HERE / "array.py", HERE / "shared.py", HERE / "kernel_shared.cc", HERE / "pack.c")},
                    resident_calls=[0,25,50,100], resident_reps=5,
                    builds={str(Path(b).name):json.loads((Path(b)/"build.json").read_text()) for b in builds})
    if out.exists():
        raise FileExistsError("refusing to overwrite preregistration")
    dump(out,contract)
    print("preregistration_sha256",sha(out),flush=True)


def numeric(got, ref):
    err = np.abs(got-ref)
    norm = np.linalg.norm(ref)
    rel = float(np.linalg.norm(got-ref)/norm) if norm else float(np.linalg.norm(got-ref))
    ok = bool(np.isfinite(got).all() and np.all(err <= ATOL+RTOL*np.abs(ref)) and rel <= L2)
    return dict(ok=ok,max_abs=float(np.max(err)),relative_l2=rel,
                bad_elements=int(np.sum(~(err<=ATOL+RTOL*np.abs(ref)))))


def cpu_session(heads,mask):
    import onnx
    import onnxruntime as ort
    from onnx import helper as oh, numpy_helper as nh
    shape=[heads,512,64]
    nodes=[oh.make_node("Transpose",["k"],["kt"],perm=[0,2,1]),
           oh.make_node("MatMul",["q","kt"],["scores"]),
           oh.make_node("Mul",["scores","scale"],["scaled"]),
           oh.make_node("Add",["scaled","mask"],["masked"]),
           oh.make_node("Softmax",["masked"],["p"],axis=-1),
           oh.make_node("MatMul",["p","v"],["out"])]
    initial=[nh.from_array(np.float32(.125),"scale"),
             nh.from_array(np.where(mask==1,np.float32(0),np.float32(-np.finfo(np.float32).max)),"mask")]
    graph=oh.make_graph(nodes,"attention_cpu",[oh.make_tensor_value_info(n,onnx.TensorProto.FLOAT,shape) for n in ("q","k","v")],
                        [oh.make_tensor_value_info("out",onnx.TensorProto.FLOAT,shape)],initial)
    model=oh.make_model(graph,opset_imports=[oh.make_opsetid("",17)],ir_version=9)
    options=ort.SessionOptions(); options.intra_op_num_threads=8; options.inter_op_num_threads=1
    options.add_session_config_entry("session.intra_op.allow_spinning","0")
    return ort.InferenceSession(model.SerializeToString(),sess_options=options,providers=["CPUExecutionProvider"])


def run(out, masks_path, registration, benchmark, reps):
    contract = json.loads(registration.read_text())
    for path in (HERE / name for name in contract["sources"]):
        if contract["sources"][path.name] != sha(path):
            raise ValueError("source changed after preregistration")
    meta = json.loads((out/"build.json").read_text())
    if meta.get("kernel_sha256") != contract["sources"].get(meta.get("kernel_source")):
        raise ValueError("build source differs from registered source")
    for name,digest in meta.get("dependencies",{}).items():
        if contract["sources"].get(name)!=digest: raise ValueError("build dependency changed: "+name)
    if meta.get("pack_dll_sha256") and sha(out/"attention_pack.dll")!=meta["pack_dll_sha256"]:
        raise ValueError("native pack artifact changed")
    if meta not in contract["builds"].values():
        raise ValueError("build not registered")
    for filename,key in (("final.xclbin","xclbin_sha256"),("insts.bin","instructions_sha256")):
        if sha(out/filename) != meta[key]:
            raise ValueError("artifact changed after preregistration")
    if sha(masks_path) != meta["masks_sha256"]:
        raise ValueError("masks changed")
    from ignite_xdna.runtime.driver import XrtSiliconHarness
    import ignite_xdna
    assert Path(ignite_xdna.__file__).resolve().is_relative_to(ROOT / "src")
    witness = command(["C:/Windows/System32/AMD/xrt-smi.exe", "examine", "-r", "aie-partitions"], ROOT)
    print(witness,flush=True)
    if "No hardware contexts running" not in witness:
        raise RuntimeError("NPU busy")
    mask = np.load(masks_path)[f"{meta['mode']}_{meta['valid']}"]
    # Validate CPU timing dependencies before allocating any device handles.
    if benchmark: cpu_session(meta["heads"],mask)
    print("preregistration_sha256",sha(registration), "build",out.name,flush=True)
    all_ok, samples = True, []
    with XrtSiliconHarness() as h:
        ib = bi = bo = None
        try:
            h.load_xclbin(str(out/"final.xclbin"))
            ib, nbytes = h.create_instruction_bo(str(out/"insts.bin"))
            bi = h.create_host_bo(meta.get("input_words",meta["packets"]*PACKET)*4,3)
            bo = h.create_host_bo(meta["outputs"]*OUT*4,4)
            to = h.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
            fr = h.pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
            def dispatch():
                start = time.perf_counter()
                r, status = h.dispatch_kernel(ib,nbytes,bi,bo,timeout_ms=5000)
                elapsed = (time.perf_counter()-start)*1e3
                if "COMPLETED" not in str(status):
                    raise RuntimeError(f"dispatch failed {status}; stop without retry")
                return elapsed
            if meta.get("resident","none") != "none":
                q,k,v=case(1,11,"random")
                packet=pack(q,k,v,np.ones((512,512),np.uint8),contract["poisons"][0])[:1].copy()
                packet[0,1]=1
                ref=oracle(q[:,:16],k[:,:32],v[:,:32],np.ones((16,32),np.uint8))
                if meta["resident"] == "kernel":
                    for calls in (1,3):
                        packet[0,3]=calls; bi.write(packet.tobytes(),0); bi.sync(to)
                        dispatch(); bo.sync(fr)
                        result=np.frombuffer(bo.read(OUT*4,0),np.uint32).copy()
                        got=result[16:-16].view(np.float32).reshape(1,16,64)
                        verdict=numeric(got,ref)
                        guard=bool(np.all(result[:16]==packet[0,2]) and np.all(result[-16:]==packet[0,2]))
                        print(json.dumps(dict(resident_correctness_calls=calls,guards=guard,**verdict)),flush=True)
                        if not verdict["ok"] or not guard:
                            raise ValueError("resident correctness failed")
                xs,ys=[],[]
                for calls in contract["resident_calls"]:
                    packet[0,3]=calls; bi.write(packet.tobytes(),0); bi.sync(to)
                    dispatch()
                    ts=[dispatch() for _ in range(contract["resident_reps"])]
                    xs += [calls]*len(ts); ys += ts
                    print(json.dumps(dict(resident=meta["resident"],calls=calls,dispatch_ms=ts)),flush=True)
                slope,intercept=np.polyfit(xs,ys,1)
                residual=np.array(ys)-(intercept+slope*np.array(xs))
                r2=1-float(residual@residual)/float(np.sum((ys-np.mean(ys))**2))
                print(json.dumps(dict(resident=meta["resident"],slope_ms=float(slope),intercept_ms=float(intercept),r_squared=r2,
                                      scope="one resident 16x32 attention block, reset every call; not a full head")),flush=True)
                del bi,bo,ib
            else:
                result=run_full(h,ib,nbytes,bi,bo,meta,mask,contract,benchmark,reps,dispatch,to,fr,out)
                del bi,bo,ib
                if result: return result
        finally:
            bi = bo = ib = None
            gc.collect()
    gc.collect()
    print(command(["C:/Windows/System32/AMD/xrt-smi.exe","examine","-r","aie-partitions"],ROOT),flush=True)
    return 0


def run_full(h,ib,nbytes,bi,bo,meta,mask,contract,benchmark,reps,dispatch,to,fr,out):
        all_ok=True
        mapped=meta.get("transport")=="shared"
        if mapped:
            from shared import pack_native
            imap=np.frombuffer(bi.map(),np.uint32)
            omap=np.frombuffer(bo.map(),np.uint32)
            dll=out/"attention_pack.dll"
            pack_options={"mask_params":(meta["valid"],meta["mode"]=="local" and meta["valid"]>0)} if meta.get("fast_packing") else {}
        for seed in SEEDS:
            for kind in contract["cases"]:
                q,k,v = case(meta["heads"],seed,kind)
                ref = oracle(q,k,v,mask)
                prior = None
                for poison in contract["poisons"]:
                    t0 = time.perf_counter()
                    if meta.get("transport")=="shared":
                        packet=pack_native(q,k,v,mask,poison,dll,imap,**pack_options)
                    else:
                        packet = pack(q,k,v,mask,poison)
                    packing = (time.perf_counter()-t0)*1e3
                    if meta.get("transport")=="shared":
                        assert packet.size==meta["input_words"]
                    else:
                        assert packet.shape == (meta["packets"],PACKET)
                    if meta.get("cores",1)==16 and meta.get("transport")!="shared":
                        packet=packet.reshape(4,4,-1,PACKET).transpose(0,2,1,3).reshape(-1,PACKET).copy()
                    raw = packet.tobytes()
                    t0 = time.perf_counter()
                    if mapped:
                        omap.fill(poison)
                    else:
                        bi.write(raw,0)
                        bo.write(np.full(meta["outputs"]*OUT,poison,np.uint32).tobytes(),0)
                    bi.sync(to); bo.sync(to)
                    writing = (time.perf_counter()-t0)*1e3
                    ms = dispatch()
                    t0 = time.perf_counter()
                    bo.sync(fr)
                    result = (omap.copy() if mapped else np.frombuffer(bo.read(meta["outputs"]*OUT*4,0),np.uint32).copy()).reshape(-1,OUT)
                    reading = (time.perf_counter()-t0)*1e3
                    bi.sync(fr)
                    immutable = bytes(bi.read(len(raw),0)) == raw
                    guards = bool(np.all(result[:,:16]==poison) and np.all(result[:,-16:]==poison))
                    if meta.get("cores",1)==16 and meta.get("transport")!="shared":
                        result=result.reshape(4,32,4,OUT).transpose(0,2,1,3).reshape(-1,OUT)
                    body = result[:,16:-16].copy()
                    written = bool(np.all(body != poison))
                    got = body.view(np.float32).reshape(meta["heads"],512,64)
                    verdict = numeric(got,ref)
                    independent = prior is None or np.array_equal(got,prior)
                    ok = verdict["ok"] and guards and immutable and written and independent
                    all_ok &= ok
                    print(json.dumps(dict(seed=seed,kind=kind,poison=hex(poison),**verdict,
                          guards=guards,inputs_immutable=immutable,written=written,independent=independent,
                          packing_ms=packing,write_sync_ms=writing,dispatch_ms=ms,read_sync_ms=reading,
                          verdict="PASS" if ok else "FAIL")),flush=True)
                    prior = got.copy()
                if not all_ok:
                    print("correctness FAIL; timing withheld",flush=True)
                    return 1
        if benchmark:
            for _ in range(3): dispatch()
            samples = [dispatch() for _ in range(reps)]
            # Same operations, fp32 on the same bf16 input values. Full float64
            # oracle is correctness only, never used as the CPU speed baseline.
            cpu=cpu_session(meta["heads"],mask)
            feed=dict(q=q,k=k,v=v)
            cpu_result=cpu.run(None,feed)[0]
            if not numeric(cpu_result,ref)["ok"]:
                raise ValueError("CPU baseline disagrees with oracle")
            times=[]
            for _ in range(reps):
                t=time.perf_counter(); cpu.run(None,feed); times.append((time.perf_counter()-t)*1e3)
            e2e=[]
            if mapped:
                for _ in range(reps):
                    t=time.perf_counter()
                    pack_native(q,k,v,mask,contract["poisons"][0],dll,imap,**pack_options)
                    bi.sync(to); dispatch(); bo.sync(fr)
                    detached=omap.reshape(-1,OUT)[:,16:-16].copy().view(np.float32).reshape(16,512,64)
                    e2e.append((time.perf_counter()-t)*1e3)
                    if not numeric(detached,ref)["ok"]: raise ValueError("mapped benchmark output incorrect")
            print(json.dumps(dict(dispatch_ms=samples,cpu_ort_8threads_ms=times,mapped_end_to_end_ms=e2e,
                                  end_to_end_scope="native pack into mapped BO, input sync, dispatch, output sync, detached output; no RoPE",
                                  cpu_scope="attention only, no RoPE; dense masked local product",
                                  timing_scope=f"{meta.get('cores',1)} cores, includes streaming/dispatch; excludes packing and BO copies")),flush=True)
        print("correctness PASS",flush=True)
        return 0


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("inspect"); p.add_argument("--model",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p=sub.add_parser("build"); p.add_argument("--out",type=Path,required=True); p.add_argument("--masks",type=Path,required=True)
    p.add_argument("--heads",type=int,choices=(1,16),default=1); p.add_argument("--mode",choices=("global","local"),required=True)
    p.add_argument("--valid",type=int,choices=(512,193,1,0),default=512)
    p.add_argument("--resident",choices=("none","kernel","empty"),default="none")
    p.add_argument("--implementation",choices=("vector","scalar","lut"),default="vector")
    p=sub.add_parser("build-suite"); p.add_argument("--out",type=Path,required=True); p.add_argument("--masks",type=Path,required=True)
    p.add_argument("--implementation",choices=("vector","scalar","lut"),default="vector")
    p.add_argument("--with-scalar-control",action="store_true")
    p=sub.add_parser("suite"); p.add_argument("--out",type=Path,required=True); p.add_argument("--masks",type=Path,required=True); p.add_argument("--prereg",type=Path,required=True)
    p=sub.add_parser("preregister"); p.add_argument("--out",type=Path,required=True); p.add_argument("--builds",type=Path,nargs="+",required=True); p.add_argument("--semantics",type=Path,required=True)
    for name in ("check","benchmark"):
        p=sub.add_parser(name); p.add_argument("--out",type=Path,required=True); p.add_argument("--masks",type=Path,required=True); p.add_argument("--prereg",type=Path,required=True); p.add_argument("--reps",type=int,default=10)
    a=ap.parse_args()
    if a.cmd=="inspect": inspect(a.model,a.out)
    elif a.cmd=="build": build(a.out.resolve(),a.masks,a.heads,a.mode,a.valid,a.resident,a.implementation)
    elif a.cmd=="build-suite":
        for mode in ("global","local"):
            for heads,valid in ((1,512),(16,512),(1,193),(1,1),(1,0)):
                build((a.out/f"{mode}_h{heads}_v{valid}").resolve(),a.masks,heads,mode,valid,implementation=a.implementation)
        for resident in ("kernel","empty"):
            build((a.out/f"resident_{resident}").resolve(),a.masks,1,"global",512,resident,a.implementation)
        if a.with_scalar_control:
            for mode in ("global","local"):
                build((a.out/f"scalar_{mode}_h16_v512").resolve(),a.masks,16,mode,512,implementation="scalar")
            build((a.out/"scalar_resident_kernel").resolve(),a.masks,1,"global",512,"kernel","scalar")
    elif a.cmd=="suite":
        print("PREREGISTRATION",a.prereg.read_text(),flush=True)
        failed=[]
        for directory in sorted(a.out.glob("*/build.json")):
            meta=json.loads(directory.read_text())
            print("SUITE_BEGIN",directory.parent.name,flush=True)
            result=run(directory.parent.resolve(),a.masks,a.prereg,True,10)
            if result: failed.append(directory.parent.name)
        print("SUITE_RESULT",json.dumps(dict(verdict="FAIL" if failed else "PASS",failed=failed)),flush=True)
        return int(bool(failed))
    elif a.cmd=="preregister": preregister(a.out,a.builds,a.semantics)
    else: return run(a.out.resolve(),a.masks,a.prereg,a.cmd=="benchmark",a.reps)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
