"""Run the documented study syntax, shell syntax and import checks.

Run in resnet_env17. This does not replace engine pytest or hardware validation.
Import names come from CONTRIBUTING.md rather than a second maintained list.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import compileall
import importlib
from pathlib import Path
import re
import subprocess
import sys


def main(root,bash):
    sys.path.insert(0,str(root))
    for name in ('npu','pipelines','tools','kernels','quant','tui','demos'):
        if not compileall.compile_dir(root/name,quiet=1):raise RuntimeError(f'Syntax failure in {name}')
    for path in sorted((root/'scripts').glob('*.sh')):
        subprocess.run([bash,'-n',str(path)],check=True)
    text=(root/'CONTRIBUTING.md').read_text(encoding='utf-8')
    modules=set()
    for imports in re.findall(r'python -c "import ([^"\n]+)"',text):
        modules.update(item.strip() for item in imports.split(','))
    if not modules:raise ValueError('Documented import gate not found')
    for name in sorted(modules):importlib.import_module(name)
    print(f'PIPELINE CHECKS PASS: syntax, shell syntax, {len(modules)} imports',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--bash',default='bash')
    args=parser.parse_args();main(args.root.resolve(),args.bash)
