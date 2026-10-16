# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline full-buffer parity for the native layout converter; no device access."""
import argparse
import hashlib
import json
from pathlib import Path

import probe as p
from shared import INPUT_WORDS, pack, pack_native


def check(dll,masks,prevalidated=False):
    count=0
    with p.np.load(masks) as archive:
        for seed in p.SEEDS:
            q,k,v=p.case(16,seed,'random')
            before=[hashlib.sha256(x.tobytes()).hexdigest() for x in (q,k,v)]
            for key in archive.files:
                for poison in (0x7fc12345,0x7fc54321):
                    expected=pack(q,k,v,archive[key],poison)
                    mode,valid=key.rsplit('_',1); valid=int(valid)
                    options=[None,(valid,mode=='local' and valid>0)] if prevalidated else [None]
                    for params in options:
                        buffer=p.np.full(INPUT_WORDS+32,poison,p.np.uint32)
                        got=pack_native(q,k,v,archive[key],poison,dll,buffer[16:-16],mask_params=params)
                        assert p.np.array_equal(got,expected),(seed,key,poison)
                        assert p.np.all(buffer[:16]==poison) and p.np.all(buffer[-16:]==poison)
                        assert before==[hashlib.sha256(x.tobytes()).hexdigest() for x in (q,k,v)]
                        count+=1
    print(json.dumps(dict(native_pack='PASS',full_buffer_comparisons=count,guards=True,
                          inputs_immutable=True,dll_sha256=p.sha(dll),source_sha256=p.sha(p.HERE/'pack.c'))))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dll',type=Path,required=True); ap.add_argument('--masks',type=Path,required=True)
    ap.add_argument('--prevalidated',action='store_true')
    a=ap.parse_args(); check(a.dll.resolve(),a.masks,a.prevalidated)
