"""Admission retries preserve output, ownership and bounded waiting."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
from types import SimpleNamespace

import pytest

from npu.gemma4.runtime import HardwareBusy
from tools import gemma4_request as request


def test_rejected_stream_closes_before_wait_and_same_assistant_retries(monkeypatch):
    events = []
    attempts = []
    clock = [0.]
    monkeypatch.setattr(request.time, 'monotonic', lambda: clock[0])
    def sleep(seconds):
        assert events[-1] == 'closed'
        events.append('wait')
        clock[0] += seconds
    monkeypatch.setattr(request.time, 'sleep', sleep)
    def stream(prompt, **kwargs):
        attempts.append((prompt, kwargs))
        try:
            if len(attempts) == 1:
                raise HardwareBusy('busy', {'pid': 123})
            yield 'first'
            yield 'last'
        finally:
            events.append('closed')
    assistant = SimpleNamespace(stream_tokens=stream)
    rejected = []
    assert list(request.stream_with_admission(assistant, [1, 2], max_new_tokens=2,
        depth=1, wait_seconds=3, on_rejection=rejected.append)) == ['first', 'last']
    assert attempts[0] == attempts[1]
    assert rejected == [{'pid': 123}]
    assert events == ['closed', 'wait', 'closed']


@pytest.mark.parametrize('error,partial', [(HardwareBusy('busy'), True), (ValueError('numerical'), False)])
def test_partial_or_fatal_failure_is_not_replayed(error, partial):
    calls = []
    def stream(*args, **kwargs):
        calls.append(1)
        if partial:
            yield 'partial'
        raise error
    outer = request.stream_with_admission(SimpleNamespace(stream_tokens=stream), [1],
        max_new_tokens=2, depth=1, wait_seconds=60)
    if partial:
        assert next(outer) == 'partial'
    with pytest.raises(type(error), match=str(error)):
        next(outer)
    assert len(calls) == 1


def test_deadline_prevents_another_attempt(monkeypatch):
    clock = [0.]
    calls = []
    monkeypatch.setattr(request.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(request.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    def stream(*args, **kwargs):
        calls.append(1)
        raise HardwareBusy('busy')
    with pytest.raises(HardwareBusy):
        list(request.stream_with_admission(SimpleNamespace(stream_tokens=stream), [1],
            max_new_tokens=1, depth=1, wait_seconds=2))
    assert len(calls) == 2 and clock[0] == 2


def test_default_rejects_immediately_and_closing_releases_active_stream():
    closed = []
    def stream(*args, **kwargs):
        try:
            yield 'first'
            yield 'last'
        finally:
            closed.append(True)
    outer = request.stream_with_admission(SimpleNamespace(stream_tokens=stream), [1],
        max_new_tokens=2, depth=1)
    assert next(outer) == 'first'
    outer.close()
    assert closed == [True]
    def busy(*args, **kwargs):
        raise HardwareBusy('busy')
    with pytest.raises(HardwareBusy):
        list(request.stream_with_admission(SimpleNamespace(stream_tokens=busy), [1],
            max_new_tokens=1, depth=1))


def test_one_shot_rejection_writes_report_after_cleanup(monkeypatch, tmp_path):
    import json
    import sys
    from tools import gemma4_chat as chat

    events = []
    class Assistant:
        max_prompt_tokens = 256
        last_result = None
        _backend = SimpleNamespace(gpu=None)
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            events.append('assistant closed')
        def stream_tokens(self, *args, **kwargs):
            self.last_result = dict(completed=False, rejected=True)
            raise HardwareBusy('foreign work', {'npu_idle': False})
    class Tokenizer:
        identity = {'fixture': True}
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            events.append('tokenizer closed')
        def encode(self, messages):
            return [1, 2]
    out = tmp_path/'rejected.json'
    monkeypatch.setattr(chat, 'HybridAssistant', Assistant)
    monkeypatch.setattr(chat, 'ChatTokenizer', Tokenizer)
    monkeypatch.setattr(sys, 'argv', ['chat', '--artifacts', str(tmp_path),
        '--llama-runtime', str(tmp_path), '--tokenizer-python', 'unused',
        '--snapshot', str(tmp_path), '--prompt', 'hello', '--report', str(out)])
    assert chat.main() == 2
    assert events == ['tokenizer closed', 'assistant closed']
    report = json.loads(out.read_text())
    assert report['admission_rejections'] == [{'npu_idle': False}]
    assert report['result'] == {'completed': False, 'rejected': True}
    assert report['worker_released'] is None
    assert report['request_error'] == 'foreign work'
