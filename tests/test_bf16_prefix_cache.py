"""Compare complete candidate packets with the independent prepared control."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path
import threading

import numpy as np
import pytest

from kernels.transformer_bf16.prefix_cache import CachedPrefixArena
from kernels.transformer_bf16.prefix_binding import PrefixBinder
from kernels.transformer_bf16.state import PreparedArena


class Capture:
    def __init__(self, fail=False):
        self.packets = []
        self.fail = fail

    def execute(self, folder, packets, **kwargs):
        self.packets.append(packets.copy())
        if self.fail:
            raise RuntimeError('Injected transport failure')
        return np.full((1, 32), 7, np.uint32), dict(input_unchanged=True)


def template():
    return PreparedArena(Path('not-a-build'), np.arange(6*2200, dtype=np.uint32).reshape(6, 2200),
        {}, {'x': dict(packet=0, count=4)}, {'w': dict(packet=1, shape=(16, 256), precise=False)},
        {'r': dict(packet=2, width=8)}, 7,
        {'s': dict(packets=[3, 4], offsets=[0, 0], phases=[0, 1], capacity=32)})


def patches():
    return dict(matrices={'w': np.arange(4096, dtype=np.uint16).reshape(16, 256)},
                rotations={'r': (np.arange(8, dtype=np.uint16), np.arange(8, dtype=np.uint16)+4)},
                lengths={'s': 17})


def test_cache_preserves_complete_packets_across_uploads_omission_mutation_and_restore():
    base = template()
    original = base._packets.copy()
    cache = CachedPrefixArena(base)
    data = patches()
    actual, control = Capture(), Capture()
    for i, upload in enumerate((1, 2, None, 3, 4, None)):
        if i == 3:
            data['matrices']['w'][0, 0] ^= np.uint16(1)
        if i == 4:
            data['lengths']['s'] = 9
        kwargs = dict(data)
        if upload is not None:
            kwargs['uploads'] = {'x': np.full(4, upload, np.uint16)}
        base.execute(control, **kwargs)
        cache.execute(actual, **kwargs)
        assert np.array_equal(actual.packets[-1], control.packets[-1])
    assert cache.hits == 3 and cache.misses == 3
    assert np.array_equal(base._packets, original)
    assert not cache.cached._packets.flags.writeable
    # Omitting all prefix patches must return to the original base semantics.
    cache.execute(actual, uploads={'x': np.full(4, 2, np.uint16)})
    base.execute(control, uploads={'x': np.full(4, 2, np.uint16)})
    assert np.array_equal(actual.packets[-1], control.packets[-1])


def test_failed_dispatch_does_not_retain_a_new_cache_entry():
    cache = CachedPrefixArena(template())
    with pytest.raises(RuntimeError, match='Injected'):
        cache.execute(Capture(fail=True), **patches())
    assert cache.cached is None and cache.key is None
    cache.execute(Capture(), **patches())
    assert cache.misses == 2 and cache.hits == 0


@pytest.mark.parametrize('fault', ['matrix_shape', 'rotary_type', 'length', 'upload'])
def test_invalid_patches_still_reject_before_transport(fault):
    cache = CachedPrefixArena(template())
    data = patches()
    if fault == 'matrix_shape':
        data['matrices']['w'] = np.zeros((1, 256), np.uint16)
    elif fault == 'rotary_type':
        data['rotations']['r'] = (np.zeros(8, np.float32), np.zeros(8, np.uint16))
    elif fault == 'length':
        data['lengths']['s'] = 33
    else:
        cache.execute(Capture(), **data)
        data['uploads'] = {'x': np.zeros(5, np.uint16)}
    capture = Capture()
    with pytest.raises(ValueError):
        cache.execute(capture, **data)
    assert not capture.packets


def test_bound_prefix_matches_control_without_copying_its_owned_command_buffer():
    base = template()
    binder = PrefixBinder(base)
    actual, control = Capture(), Capture()
    data = patches()
    binder.execute(actual, uploads={'x': np.full(4, 1, np.uint16)}, **data)
    bound = binder.take_bound()
    assert binder.take_bound() is None
    owned = bound.base._packets
    for value in (2, 3, 4, 1):
        upload = {'x': np.full(4, value, np.uint16)}
        bound.execute(actual, uploads=upload)
        base.execute(control, uploads=upload, **data)
        assert np.array_equal(actual.packets[-1], control.packets[-1])
        assert bound.base._packets is owned and not owned.flags.writeable
    # Prefix growth gets a fresh owner and cannot mutate the previous binding.
    previous = owned.copy()
    data['lengths']['s'] = 9
    binder.execute(actual, uploads={'x': np.full(4, 3, np.uint16)}, **data)
    assert binder.take_bound().base._packets is not owned
    assert np.array_equal(owned, previous)


def test_binding_rejects_missing_uploads_prefix_changes_and_failed_reuse():
    binder = PrefixBinder(template())
    binder.execute(Capture(), **patches())
    bound = binder.take_bound()
    before = bound.base._packets.copy()
    for kwargs in ({}, {'uploads': {'x': np.zeros(5, np.uint16)}},
                   {'uploads': {'x': np.zeros(4, np.uint16)}, 'lengths': {'s': 3}}):
        with pytest.raises(ValueError):
            bound.execute(Capture(), **kwargs)
        assert np.array_equal(before, bound.base._packets)
    with pytest.raises(RuntimeError, match='Injected'):
        bound.execute(Capture(fail=True), uploads={'x': np.zeros(4, np.uint16)})
    with pytest.raises(RuntimeError, match='live binding'):
        bound.execute(Capture(), uploads={'x': np.zeros(4, np.uint16)})


def test_bound_prefix_is_thread_owned_and_rejects_overlapping_patch_roles():
    binder = PrefixBinder(template())
    binder.execute(Capture(), **patches())
    bound = binder.take_bound()
    errors = []

    def other_thread():
        try:
            bound.execute(Capture(), uploads={'x': np.zeros(4, np.uint16)})
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join()
    assert len(errors) == 1 and 'owning thread' in errors[0]
    base = template()
    base.uploads['x']['packet'] = 1
    with pytest.raises(ValueError, match='separate packets'):
        PrefixBinder(base).execute(Capture(), **patches())
