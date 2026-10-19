"""Keep bounded GPU activity fail-closed at the client command boundary."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import pytest

from tools.ggml_gpu_target.activity import ActivityHelper


def helper(reply,active=False):
    value=object.__new__(ActivityHelper);value.active=active;value.sent=[]
    value._send=value.sent.append;value._receive=lambda:reply
    return value


@pytest.mark.parametrize('duration',[0,101,-1,True,1.5,None,'10'])
def test_invalid_window_never_reaches_child(duration):
    activity=helper({})
    with pytest.raises(ValueError):activity.start(duration)
    assert activity.sent==[]


def test_overlapping_window_never_reaches_child():
    activity=helper({},active=True)
    with pytest.raises(RuntimeError):activity.start(40)
    assert activity.sent==[] and activity.active


@pytest.mark.parametrize('reply',[{'started':1,'window_ms':40},{'started':True,'window_ms':40.0},
                                  {'started':True,'window_ms':100},{'started':True,'window_ms':40,'other':0}])
def test_invalid_start_acknowledgement_is_rejected(reply):
    activity=helper(reply)
    with pytest.raises(ValueError):activity.start(40)


def test_early_stop_joins_and_checks_work():
    reply=dict(completed=True,gpu_validated=True,dispatches=2,elapsed_ms=20.0)
    activity=helper(reply,active=True)
    assert activity.wait(stop=True)==reply
    assert activity.sent==['stop'] and not activity.active


@pytest.mark.parametrize('changes',[{'gpu_validated':False},{'dispatches':0},{'dispatches':True},
                                    {'elapsed_ms':float('nan')},{'elapsed_ms':-1}])
def test_unverified_completion_retains_active_state_for_cleanup(changes):
    reply=dict(completed=True,gpu_validated=True,dispatches=2,elapsed_ms=20.0);reply.update(changes)
    activity=helper(reply,active=True)
    with pytest.raises(ValueError):activity.wait(stop=True)
    assert activity.active
