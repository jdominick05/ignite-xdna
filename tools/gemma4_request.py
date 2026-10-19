"""Retry rejected chat admission before any output, within a bounded deadline."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import time

from npu.gemma4.runtime import HardwareBusy


def stream_with_admission(assistant, prompt, *, max_new_tokens, depth,
                          wait_seconds=0, on_rejection=None):
    """Keep the same assistant; close rejected streams before waiting.

    The deadline bounds retry scheduling, not admission telemetry or generation.
    HardwareBusy after any chunk is never replayed. Numerical errors propagate.
    """
    if type(wait_seconds) is not int or not 0 <= wait_seconds <= 60:
        raise ValueError('Admission wait must be an integer from 0 to 60 seconds')
    deadline = time.monotonic() + wait_seconds
    while True:
        stream = None
        emitted = False
        rejection = None
        try:
            stream = assistant.stream_tokens(prompt, max_new_tokens=max_new_tokens, depth=depth)
            for chunk in stream:
                emitted = True
                yield chunk
            return
        except HardwareBusy as error:
            if emitted:
                raise
            rejection = error
        finally:
            if stream is not None:
                stream.close()
        if on_rejection is not None:
            on_rejection(rejection.report)
        remaining = deadline - time.monotonic()
        if wait_seconds == 0 or remaining <= 0:
            raise rejection
        time.sleep(min(1, remaining))
        if time.monotonic() >= deadline:
            raise rejection
