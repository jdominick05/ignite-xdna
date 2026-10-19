"""Experimental owned prefix binding with complete replacement of dynamic uploads.

Prefix data is materialized by the existing PreparedArena path once. A bound
owner updates only its upload payloads for subsequent steps in the same chain.
The transport still reads back and verifies the entire input buffer every time.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import threading
import time

import numpy as np

from .state import PreparedArena


class BoundUploadsArena:
    """Exclusively owned packet buffer; every upload must be supplied each time."""
    def __init__(self, base, packets):
        if packets is base._packets or not packets.flags.owndata:
            raise ValueError('An exclusively materialized packet buffer is required')
        prefix_packets = {span['packet'] for span in base.rotations.values()}
        prefix_packets.update(i for span in base.softmax_lengths.values() for i in span['packets'])
        for span in base.matrices.values():
            n, k = span['shape']
            count = ((n+15)//16)*((k+255)//256)
            prefix_packets.update(range(span['packet'], span['packet']+count))
        if any(span['packet'] in prefix_packets for span in base.uploads.values()):
            raise ValueError('Bound uploads must have separate packets from prefix patches')
        self.base = PreparedArena(base.folder, packets, base.exports, base.uploads,
                                  base.matrices, base.rotations, base.poison, base.softmax_lengths)
        self.thread = threading.get_ident()
        self.failed = False

    def execute(self, transport, uploads=None, matrices=None, rotations=None, lengths=None):
        if self.failed or threading.get_ident() != self.thread:
            raise RuntimeError('Use a live binding on its owning thread')
        if matrices or rotations or lengths or set(uploads or {}) != set(self.base.uploads):
            raise ValueError('A bound prefix requires every upload and no prefix changes')
        start = time.perf_counter()
        values = {}
        for label, bits in uploads.items():
            bits = np.asarray(bits)
            span = self.base.uploads[label]
            if bits.dtype != np.uint16 or bits.size != span['count']:
                raise ValueError('Bound upload shape or dtype changed')
            values[label] = bits
        packets = self.base._packets
        packets.flags.writeable = True
        try:
            for label, bits in values.items():
                span = self.base.uploads[label]
                packets[span['packet'], 16:].view(np.uint16)[:span['count']] = bits.reshape(-1)
        finally:
            packets.flags.writeable = False
        patch_ms = (time.perf_counter()-start)*1000
        try:
            output, evidence = self.base.execute(transport)
        except BaseException:
            self.failed = True
            raise
        evidence['prefix_bound_uploads'] = dict(patch_ms=patch_ms, retained_bytes=packets.nbytes)
        return output, evidence


class PrefixBinder:
    """Pass-through template that can transfer one validated materialized input."""
    def __init__(self, base):
        self.base = base
        self.enabled = True
        self.bound = None

    def take_bound(self):
        result = self.bound
        self.bound = None
        return result

    def execute(self, transport, uploads=None, matrices=None, rotations=None, lengths=None):
        self.bound = None
        if not self.enabled or not (matrices or rotations or lengths):
            return self.base.execute(transport, uploads, matrices, rotations, lengths)
        owner = self

        class BindAndRun:
            def execute(self, folder, packets, **kwargs):
                result = transport.execute(folder, packets, **kwargs)
                owner.bound = BoundUploadsArena(owner.base, packets)
                return result

        try:
            values, evidence = self.base.execute(BindAndRun(), uploads, matrices, rotations, lengths)
        except BaseException:
            self.bound = None
            raise
        evidence['prefix_bound'] = True
        return values, evidence
