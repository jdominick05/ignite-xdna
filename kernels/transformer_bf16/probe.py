"""Prepare CPU references and check the reusable BF16 operations on Phoenix.

Preparation needs torch; device checks need only NumPy and the native XRT env.
No checkpoint, network access, model prompt, or CPU arithmetic fallback is used
by the device operations. This is correctness qualification, not timing.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16 import Bf16Ops
from kernels.transformer_bf16.common import bf16_bits,bf16_values,save,sha

GATES=dict(atol=0.002,rtol=0.008,relative_l2=0.004,
           scope="Synthetic individual operations; no whole-model accuracy, graph compiler support, or speedup")


def prepare(folder,seeds):
    import torch
    if folder.exists():raise FileExistsError("Refusing to replace probe fixtures")
    folder.mkdir(parents=True)
    source_bits=np.arange(65536,dtype=np.uint16)
    with torch.inference_mode():
        gelu=torch.nn.functional.gelu(torch.from_numpy(source_bits).view(torch.bfloat16),approximate="tanh").view(torch.uint16).numpy().copy()
    table=np.concatenate((gelu[0x3980:0x4180],gelu[0xb980:0xc180]))
    magnitude=source_bits.astype(np.uint32)&0x7fff;sign=source_bits.astype(np.uint32)&0x8000
    half=np.where(magnitude>=0x100,magnitude-0x80,(magnitude>>1)+((magnitude&3)==3))
    expected=(sign|half).astype(np.uint16)
    middle=(magnitude>=0x3980)&(magnitude<0x4180)
    expected[middle]=table[np.where(sign[middle]!=0,2048,0)+magnitude[middle]-0x3980]
    high=magnitude>=0x4180
    expected[high]=np.where(sign[high]!=0,0x8000,source_bits[high])
    finite=(source_bits&0x7f80)!=0x7f80
    if not np.array_equal(expected[finite],gelu[finite]):raise ValueError("GELU table semantics differ")
    cases=[]
    for seed in seeds:
        rng=np.random.default_rng(seed);a={"table":table}
        def tensor(name,shape,scale=1.0):
            bits=bf16_bits(rng.normal(size=shape)*scale);a[name]=bits
            return torch.from_numpy(bits).view(torch.bfloat16)
        def store(name,t):a[name]=t.contiguous().view(torch.uint16).numpy().copy()
        with torch.inference_mode():
            x=tensor("projection.x",(768,));w=tensor("projection.w",(384,768),0.05)
            store("projection.expected",w@x)
            x=tensor("norm.x",(4,512));w=tensor("norm.w",(512,),0.5)
            store("norm.expected",(x.float()*torch.pow(x.float().square().mean(-1,keepdim=True)+1e-6,-0.5)*w.float()).to(torch.bfloat16))
            q=tensor("rope.q",(4,512));phase=torch.from_numpy(rng.uniform(-100,100,size=512).astype(np.float32))
            cosine=phase.cos().to(torch.bfloat16);sine=phase.sin().to(torch.bfloat16)
            store("rope.cosine",cosine);store("rope.sine",sine)
            store("rope.expected",q*cosine+torch.cat((-q[:,256:],q[:,:256]),dim=-1)*sine)
            scores=tensor("softmax.x",(4,513),12)
            store("softmax.expected",torch.softmax(scores,dim=-1,dtype=torch.float32).to(torch.bfloat16))
            gate=tensor("geglu.gate",(2048,),3);up=tensor("geglu.up",(2048,))
            store("geglu.expected",torch.nn.functional.gelu(gate,approximate="tanh")*up)
            x=tensor("residual.x",(2048,));r=tensor("residual.r",(2048,))
            store("residual.expected",(x+r)*torch.tensor(.875,dtype=torch.bfloat16))
            top=tensor("topk.x",(2048,))
            a["topk.expected"]=np.lexsort((np.arange(2048),-top.float().numpy()))[:32].astype(np.int32)
        path=folder/f"seed-{seed}.npz"
        with path.open("xb") as f:np.savez(f,**a)
        cases.append(dict(file=path.name,sha256=sha(path),seed=seed))
    save(folder/"manifest.json",dict(cases=cases,seeds=seeds,gates=GATES,gelu_finite_inputs_checked=int(finite.sum()),
        gelu_bitwise_equal=True,torch_version=torch.__version__,prepare_source_sha256=sha(__file__)))
    print(json.dumps(dict(fixtures=len(cases),gelu_finite_inputs_checked=int(finite.sum()))),flush=True)


def run(folder,cache,output):
    if output.exists():raise FileExistsError("Refusing to replace probe evidence")
    manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["gates"]!=GATES:raise ValueError("Probe criteria changed")
    ops=Bf16Ops(cache);rows=[]
    for case in manifest["cases"]:
        path=folder/case["file"]
        if sha(path)!=case["sha256"]:raise ValueError("Fixture changed")
        with np.load(path,allow_pickle=False) as a:
            got={"projection":ops.projection(a["projection.x"],a["projection.w"]),
                 "norm":ops.norm(a["norm.x"],a["norm.w"]),
                 "rope":ops.rope(a["rope.q"],a["rope.cosine"],a["rope.sine"]),
                 "softmax":ops.softmax(a["softmax.x"]),
                 "geglu":ops.geglu(a["geglu.gate"],a["geglu.up"],a["table"]),
                 "residual":ops.residual(a["residual.x"],a["residual.r"],np.uint16(0x3f60)),
                 "topk":ops.topk(a["topk.x"],32)}
            for name,value in got.items():
                reference=a[name+".expected"]
                if name=="topk":
                    metric=dict(passed=bool(np.array_equal(value,reference)))
                else:
                    x,y=bf16_values(value).astype(np.float64),bf16_values(reference).astype(np.float64)
                    error=np.abs(x-y);relative=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
                    metric=dict(passed=bool(np.isfinite(x).all() and np.all(error<=GATES["atol"]+GATES["rtol"]*np.abs(y)) and relative<=GATES["relative_l2"]),
                                relative_l2=relative,max_abs=float(error.max()))
                row=dict(seed=case["seed"],operation=name,**metric);rows.append(row)
                print(json.dumps(row),flush=True)
    sources={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in Path(__file__).parent.iterdir() if p.suffix in (".py",".cc")}
    report=dict(passed=all(r["passed"] for r in rows),cases=len(rows),gates=GATES,sources=sources,
                fixture_manifest_sha256=sha(folder/"manifest.json"),results=rows,
                dispatches=[dict(function=r["function"],status=r["status"],input_unchanged=r["input_unchanged"]) for r in ops.dispatches])
    save(output,report)
    print(json.dumps(dict(passed=report["passed"],cases=len(rows),output=str(output))),flush=True)
    if not report["passed"]:raise ValueError("Reusable BF16 operation check failed")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","run"])
    parser.add_argument("--fixtures",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[11,29,83])
    parser.add_argument("--cache",type=Path)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.action=="prepare":prepare(args.fixtures,args.seeds)
    else:
        if not args.cache or not args.out:parser.error("run needs --cache and --out")
        run(args.fixtures,args.cache,args.out)
