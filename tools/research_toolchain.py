"""Read-only provenance for the existing AIE toolchain and its local source patches."""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from aie.utils import config


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


root = Path(os.environ.get('MLIR_AIE_ROOT', Path.home() / 'mlir-aie')).resolve()
peano = Path(config.peano_install_dir())
patch = subprocess.check_output(['git', '-C', str(root), 'diff', '--binary', 'HEAD'])
names = subprocess.check_output(['git', '-C', str(root), 'diff', '--name-only', 'HEAD'], text=True).splitlines()
report = {'python': sys.version, 'versions': {p: metadata.version(p) for p in ('numpy', 'mlir_aie', 'llvm_aie')},
          'mlir_source_head': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
          'mlir_source_tag': subprocess.check_output(['git', '-C', str(root), 'describe', '--tags', '--always'], text=True).strip(),
          'source_diff_sha256': hashlib.sha256(patch).hexdigest(),
          'modified_source_sha256': {n: digest(root/n) for n in names},
          'peano_executable_sha256': {n: digest(peano/'bin'/n) for n in
              ('clang++.exe', 'llvm-objdump.exe', 'llvm-readelf.exe')},
          'thread_settings': {n: os.environ.get(n) for n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
          'scope': 'read-only snapshot; local source patches were not changed by this research'}
print('RESEARCH_TOOLCHAIN', json.dumps(report), flush=True)
