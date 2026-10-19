"""Exact changed-page ranges for a previously verified packet input buffer."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np


def changed_ranges(current, previous, page_bytes=4096):
    if (current.dtype!=np.uint32 or previous.dtype!=np.uint32 or current.shape!=previous.shape
            or not current.flags.c_contiguous or not previous.flags.c_contiguous):
        raise ValueError('Matching contiguous uint32 packet arrays required')
    if type(page_bytes) is not int or page_bytes<4 or page_bytes%4:
        raise ValueError('Page size must be a positive multiple of four bytes')
    words=page_bytes//4
    difference=current.reshape(-1)!=previous.reshape(-1)
    whole=difference.size//words
    changed=difference[:whole*words].reshape(whole,words).any(axis=1)
    if difference.size%words:
        changed=np.append(changed,difference[whole*words:].any())
    pages=np.flatnonzero(changed)
    spans=[]
    for page in pages:
        start=int(page)*page_bytes
        end=min(start+page_bytes,current.nbytes)
        if spans and spans[-1][1]==start:
            spans[-1]=(spans[-1][0],end)
        else:
            spans.append((start,end))
    return spans
