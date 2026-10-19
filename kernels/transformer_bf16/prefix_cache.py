"""Experimental bounded cache of prepared matrix/rotary/length patches.

Keeps one immutable patched template. Uploads always retain PreparedArena's
fresh-base semantics; no transport, numerical, or input-readback gate changes.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import hashlib
import time

import numpy as np

from .state import PreparedArena


def patch_key(matrices, rotations, lengths):
    """Content identities detect mutation of a caller-owned patch array."""
    digest = hashlib.sha256()
    contents = {}
    for group, patches in (('matrices', matrices), ('rotations', rotations), ('lengths', lengths)):
        digest.update(group.encode())
        for label, value in sorted((patches or {}).items()):
            digest.update(repr(label).encode())
            if group == 'lengths':
                if type(value) is not int:
                    raise ValueError('Prepared softmax length must be an integer')
                digest.update(str(value).encode())
            else:
                arrays = value if group == 'rotations' else (value,)
                if group == 'rotations' and len(arrays) != 2:
                    raise ValueError('Two rotary factors required')
                for array in arrays:
                    array = np.asarray(array)
                    if array.dtype != np.uint16:
                        raise ValueError('BF16 patch arrays required')
                    digest.update(repr(array.shape).encode())
                    # Heads alias the same KV arrays. Hash each distinct view
                    # once per call; rehash on every later call to detect edits.
                    view = (array.__array_interface__['data'][0], array.shape, array.strides)
                    if view not in contents:
                        contents[view] = hashlib.sha256(np.ascontiguousarray(array)).digest()
                    digest.update(contents[view])
    return digest.digest()


class CachedPrefixArena:
    """PreparedArena proxy with a single content-keyed immutable cache entry."""
    def __init__(self, base):
        self.base = base
        self.enabled = True
        self.cached = None
        self.key = None
        self.hits = self.misses = 0
        # Resetting uploads on a miss must not undo another kind of patch.
        # Compiler-produced commands have distinct packets for these roles.
        patched = {s['packet'] for s in base.rotations.values()}
        patched.update(i for s in base.softmax_lengths.values() for i in s['packets'])
        for span in base.matrices.values():
            n, k = span['shape']
            count = ((n+15)//16)*((k+255)//256)
            patched.update(range(span['packet'], span['packet']+count))
        if any(span['packet'] in patched for span in base.uploads.values()):
            raise ValueError('Upload and prefix patch packets overlap')
        upload_packets = [span['packet'] for span in base.uploads.values()]
        if len(upload_packets) != len(set(upload_packets)):
            raise ValueError('Upload packets must be distinct')
        self.upload_defaults = {label: base._packets[span['packet'], 16:].view(np.uint16)[:span['count']].copy()
                                for label, span in base.uploads.items()}
        for value in self.upload_defaults.values():
            value.flags.writeable = False

    def execute(self, transport, uploads=None, matrices=None, rotations=None, lengths=None):
        if not self.enabled or not (matrices or rotations or lengths):
            return self.base.execute(transport, uploads, matrices, rotations, lengths)
        start = time.perf_counter()
        key = patch_key(matrices, rotations, lengths)
        key_ms = (time.perf_counter()-start)*1000
        hit = self.cached is not None and key == self.key
        if hit:
            self.hits += 1
            # The cache may contain the miss's upload payload. Explicit defaults
            # restore fresh-base semantics when a later caller omits an upload.
            values, evidence = self.cached.execute(transport, uploads={**self.upload_defaults, **(uploads or {})})
        else:
            self.misses += 1
            owner = self

            class BindAndRun:
                def execute(self, folder, packets, **kwargs):
                    # PreparedArena already validated and applied all patches.
                    # The patched array is already exclusively owned. Retain
                    # it immutably instead of copying the full command buffer.
                    owner.cached = PreparedArena(folder, packets, owner.base.exports,
                        owner.base.uploads, owner.base.matrices, owner.base.rotations,
                        owner.base.poison, owner.base.softmax_lengths)
                    owner.key = key
                    return transport.execute(folder, packets, **kwargs)

            try:
                values, evidence = self.base.execute(BindAndRun(), uploads, matrices, rotations, lengths)
            except BaseException:
                self.cached = self.key = None
                raise
        evidence['prefix_cache'] = dict(hit=hit, key_ms=key_ms,
            hits=self.hits, misses=self.misses, retained_bytes=self.cached._packets.nbytes)
        return values, evidence
