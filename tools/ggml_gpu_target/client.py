"""Binary pipe client for the bounded GPU target worker (no CPU inference)."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import struct
import json
import math
import subprocess
import threading

import numpy as np


class Target:
    def __init__(self, executable, runtime, model, prompt, log=None, timeout=90, env=None,context_tokens=512):
        if context_tokens not in (512,2560):raise ValueError('Use context 512 or 2560')
        self.context_tokens=context_tokens
        self.timeout = timeout
        self._stderr_bytes = bytearray()
        self._stderr_lock = threading.Lock()
        self._stderr_thread = None
        arguments=[str(p) for p in (executable,runtime,model,prompt)]
        if context_tokens!=512:arguments.append(str(context_tokens))
        self.process = subprocess.Popen(arguments,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=log if log is not None else subprocess.PIPE, env=env)
        if log is None:
            self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self._stderr_thread.start()
        try:
            self.initial = self.receive()
        except BaseException:
            self.close()
            raise

    def _read_stderr(self):
        try:
            with self.process.stderr as source:
                while chunk := source.read(4096):
                    with self._stderr_lock:
                        self._stderr_bytes.extend(chunk)
                        del self._stderr_bytes[:-65536]
        except (OSError, ValueError):
            pass

    @property
    def stderr_tail(self):
        with self._stderr_lock:
            return bytes(self._stderr_bytes).decode('utf-8', errors='replace')

    def receive(self):
        timer = threading.Timer(self.timeout, self.process.kill)
        timer.daemon = True
        timer.start()
        try:
            def exact(n):
                value = bytearray()
                while len(value) < n:
                    part = self.process.stdout.read(n-len(value))
                    if not part:
                        raise RuntimeError('GPU target pipe closed; inspect target log')
                    value.extend(part)
                return value
            length, = struct.unpack('<I', exact(4))
            if not 4 <= length <= 32*1024*1024:
                raise ValueError('Target response byte bound')
            data = exact(length)
        finally:
            timer.cancel()
        cursor = 0
        def word():
            nonlocal cursor
            if cursor+4 > len(data):
                raise ValueError('Truncated target response')
            value, = struct.unpack_from('<I', data, cursor)
            cursor += 4
            return value
        def raw(n):
            nonlocal cursor
            if n > len(data)-cursor:
                raise ValueError('Target field exceeds frame')
            result = data[cursor:cursor+n]
            cursor += n
            return result
        if word():
            raise RuntimeError(raw(word()).decode('utf-8', errors='replace'))
        opcode, position, count = word(), word(), word()
        if count > 256 or position > self.context_tokens:
            raise ValueError('Target response bounds')
        predictions = [word() for _ in range(count)]
        tensors = {}
        tensor_count = word()
        if tensor_count > 6:
            raise ValueError('Unexpected target tensor count')
        for _ in range(tensor_count):
            name = raw(word()).decode('ascii')
            dtype = word()
            shape = tuple(word() for _ in range(4))
            strides = tuple(word() for _ in range(4))
            payload = raw(word())
            if dtype != 0 or name in tensors or not all(0 < n <= 65536 for n in shape):
                raise ValueError('Unsupported or duplicate target tensor')
            extent = 4+sum((n-1)*s for n, s in zip(shape, strides))
            if extent > len(payload):
                raise ValueError('Target tensor strides exceed buffer')
            array = np.ndarray(shape, np.float32, buffer=payload, strides=strides).copy()
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite target state')
            tensors[name] = array
        extra = {}
        if opcode == 5:
            text = bytes(raw(word()))
            eog_count = word()
            if not 1 <= eog_count <= 256:
                raise ValueError('Invalid token piece count')
            eog = [word() for _ in range(eog_count)]
            if any(flag not in (0, 1) for flag in eog):
                raise ValueError('Invalid end-of-generation flag')
            extra = dict(text_bytes=text, eog=[bool(flag) for flag in eog])
        if opcode == 8:
            if count or tensor_count:raise ValueError('Timing response contains model output')
            length=word()
            if not 1<=length<=4096:raise ValueError('Timing payload byte bound')
            timing=json.loads(bytes(raw(length)).decode('ascii'))
            counters={'calls','tokens','capture_calls','captured_bytes','response_bytes'}
            durations={'process_ms','capture_ms','logits_ms','tensors_pack_ms','send_ms'}
            if (not isinstance(timing,dict) or set(timing)!={'version','groups'} or type(timing['version']) is not int
                    or timing['version']!=1 or not isinstance(timing['groups'],dict)
                    or not set(timing['groups'])<= {'1','4','7'}):
                raise ValueError('Unsupported timing schema')
            for values in timing['groups'].values():
                if not isinstance(values,dict) or set(values)!=counters|durations:raise ValueError('Invalid timing fields')
                if any(type(values[k]) is not int or not 0<=values[k]<2**64 for k in counters):
                    raise ValueError('Invalid timing counter')
                if any(type(values[k]) not in (float,int) or not math.isfinite(values[k]) or values[k]<0 for k in durations):
                    raise ValueError('Invalid timing duration')
            extra=dict(timing=timing)
        if cursor != len(data):
            raise ValueError('Trailing target response bytes')
        self.position = position
        return dict(opcode=opcode, position=position, predictions=predictions, tensors=tensors, **extra)

    def request(self, opcode, values=()):
        words = [opcode, len(values), *values]
        data = struct.pack('<'+'I'*len(words), *words)
        self.process.stdin.write(struct.pack('<I', len(data))+data)
        self.process.stdin.flush()
        result = self.receive()
        if result['opcode'] != opcode:
            raise ValueError('Target operation response mismatch')
        return result

    def decode(self, tokens):
        return self.request(1, tokens)

    def timing(self):
        """Read cumulative completed decode diagnostics from a supporting worker."""
        return self.request(8)['timing']

    def rollback(self, position):
        return self.request(2, [position])

    def embedding(self, token):
        return self.request(3, [token])['tensors']['embedding'][:, 0, 0, 0]

    def reset(self):
        return self.request(4)

    def prepare(self, tokens):
        """Prefill a new complete prompt while retaining model/backend weights."""
        return self.request(7, tokens)

    def render(self, tokens):
        result = self.request(5, tokens)
        if len(result['eog']) != len(tokens):
            raise ValueError('Token rendering count mismatch')
        return result

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(struct.pack('<III', 8, 0, 0))
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=10)
        self.process.stdin.close()
        self.process.stdout.close()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
