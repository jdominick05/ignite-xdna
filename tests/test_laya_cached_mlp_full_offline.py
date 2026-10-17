# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cold selection advances only on authoritative completed waits."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import laya_cached_mlp_full as full


def kernel(state):
    calls=[]
    def delegate(*args):
        calls.append(args)
        return SimpleNamespace(wait=lambda timeout:state)
    cold=SimpleNamespace(bo=object(),length=123);first=object()
    return full.CachedKernel(delegate,cold,first,'completed'),calls,cold,first


def test_cold_once_after_completed_wait_then_warm():
    proxy,calls,cold,first=kernel('completed')
    handle=proxy(3,first,456,'source','target')
    assert calls[-1][1:3]==(cold.bo,123)
    assert not proxy.started
    assert handle.wait(5000)=='completed'
    assert proxy.cold_calls==1
    with pytest.raises(RuntimeError,match='already consumed'):handle.wait(5000)
    warm=object();proxy(3,warm,789,'source','target').wait(5000)
    assert calls[-1][1:3]==(warm,789)
    assert proxy.warm_calls==1


def test_timeout_never_advances_and_never_allows_retry():
    proxy,calls,_,first=kernel('timeout')
    assert proxy(3,first,456,'source','target').wait(5000)=='timeout'
    assert not proxy.started and proxy.cold_calls==0
    with pytest.raises(RuntimeError,match='no retry'):proxy(3,first,456,'source','target')
    assert len(calls)==1


def test_cold_layer_must_be_zero():
    proxy,calls,_,_=kernel('completed')
    with pytest.raises(ValueError,match='layer0'):proxy(3,object(),456,'source','target')
    assert not calls


def test_actual_complete_layer_gate():
    path=Path.cwd()/'results/laya/cached_mlp_v3_layer_desktop2_20261017.log'
    if not path.exists():pytest.skip('cached layer evidence unavailable')
    assert full.layer_gate(path)==full.c.sha(path)


def test_failed_context_integration_is_disabled_before_hardware(monkeypatch):
    def forbidden():raise AssertionError('must not initialize hardware or load model')
    monkeypatch.setattr(full,'identity',forbidden)
    with pytest.raises(RuntimeError,match='context switches is unqualified'):full.qualify()
