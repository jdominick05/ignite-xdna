#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Commit a source/artifact registration before the L1 slope diagnostic."""
import argparse
import json
from pathlib import Path
import subprocess

import laya_l1_gemm as probe


def qualify(path,build_sha):
    sections=[]
    for line in path.read_text().splitlines():
        if line.startswith('RAW_ARCHIVE '):sections.append([])
        if sections:sections[-1].append(line)
    successful=[]
    for lines in sections:
        results=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('LAYA_L1_RESULT ')]
        if not results:continue
        starts=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('LAYA_L1_START ')]
        terminals=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('RESEARCH_PROCESS_RESULT ')]
        profiles=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('LAYA_L1_PROFILE ')]
        if (len(starts)!=1 or starts[0]['build_sha256']!=build_sha or starts[0]['calls']!=[0,1,3] or
            len(terminals)!=1 or terminals[0]['exit_code']!=0 or terminals[0]['stop'] is not None or
            len(results)!=1 or not results[0]['all_outputs_inputs_guards_exact'] or not results[0]['idle_after_close'] or
            not results[0]['registered_files_unchanged'] or results[0]['profiles']!=3 or len(profiles)!=3 or
            sorted(row['variant'] for row in profiles)!=['empty','generic','specialized'] or
            any(not row['all_output_input_guard_checks_pass'] or [s['calls'] for s in row['samples']]!=[0,1,3] for row in profiles)):
            raise ValueError('incomplete L1 qualification')
        successful.append(results[0])
    if len(successful)!=1:raise ValueError('one complete L1 qualification required')
    return probe.sha(path)


def identity(folder):
    meta=probe.validate(folder)
    files=dict(meta['sources']);files['tools/laya_l1_gemm_run.py']=probe.sha(Path(__file__))
    for file in [folder/'build.json',*[folder/name/file for name in meta['variants'] for file in ('kernel.o','insts.bin','final.xclbin')]]:
        files[str(file.absolute().relative_to(probe.ROOT)).replace('\\','/')]=probe.sha(file)
    evidence=probe.ROOT/'results/laya/l1_gemm_v3_qualification_desktop2_20261016.log'
    files[str(evidence.relative_to(probe.ROOT)).replace('\\','/')]=qualify(evidence,probe.sha(folder/'build.json'))
    return files


def protocol():
    return dict(calls=[0,32768,65536,131072],reps=3,
        order=['empty','generic','specialized','specialized','generic','empty'],reset=0,
        scope='L1 resident one-core 32x64x64 accumulated GEMM; DMA outside repeated call loop; no serving speed claim',
        arithmetic='integer-valued BF16 operands in [-1,1]; partial sums below 2^24; every output bit and guard checked',
        fit='wall seconds against call count; slope in ns/call; empty-call subtraction; no clock or peak conversion')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('register','run'))
    ap.add_argument('--folder',type=Path,required=True);ap.add_argument('--prereg',type=Path,required=True)
    args=ap.parse_args();folder=args.folder.absolute();files=identity(folder)
    registered=dict(version=1,folder=str(folder.relative_to(probe.ROOT)).replace('\\','/'),files=files,protocol=protocol())
    if args.command=='register':
        with args.prereg.open('x',encoding='utf-8',newline='\n') as file:file.write(json.dumps(registered,indent=2)+'\n')
        probe.emit('LAYA_L1_REGISTER',dict(sha256=probe.sha(args.prereg),files=len(files),protocol=protocol()))
    else:
        relative=str(args.prereg.absolute().relative_to(probe.ROOT)).replace('\\','/')
        committed=json.loads(subprocess.check_output(['git','show','HEAD:'+relative],cwd=probe.ROOT,text=True))
        if committed!=registered or json.loads(args.prereg.read_text())!=registered:raise ValueError('L1 registration changed or uncommitted')
        probe.emit('LAYA_L1_REGISTERED_RUN',dict(prereg_sha256=probe.sha(args.prereg),files=len(files)))
        probe.run(folder,False)
        if identity(folder)!=files:raise ValueError('L1 registered files changed during timing')


if __name__=='__main__':main()
