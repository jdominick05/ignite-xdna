"""Toolchain paths and bit-preserving helpers for the standalone BF16 kernels."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

STUDY=Path(__file__).resolve().parents[2]
IRON=Path(os.environ.get("MLIR_AIE_IRON",Path.home()/"mlir-aie/ironenv"))
PEANO=Path(os.environ.get("PEANO_INSTALL_DIR",IRON/"Lib/site-packages/llvm-aie"))
INSTALL=Path(os.environ.get("MLIR_AIE_INSTALL_DIR",IRON/"Lib/site-packages/mlir_aie"))
XRT=Path(os.environ.get("XRT_ROOT","C:/Xilinx/XRT/xrt_sdk/xrt"))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda:file.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def save(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    temporary.replace(path)


def command(args,cwd):
    env=dict(os.environ)
    env["PATH"]=os.pathsep.join([str(INSTALL/"bin"),str(INSTALL/"lib"),str(XRT),str(XRT/"lib"),env.get("PATH","")])
    result=subprocess.run([str(arg) for arg in args],cwd=cwd,env=env,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-4000:]+result.stderr[-6000:])
    return result.stdout


def bf16_values(bits):
    bits=np.asarray(bits)
    if bits.dtype!=np.uint16:
        raise TypeError("BF16 storage must be uint16")
    return (bits.astype(np.uint32)<<16).view(np.float32)


def bf16_bits(values):
    bits=np.asarray(values,np.float32).view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)
