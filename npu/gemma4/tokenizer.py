"""Local-only chat tokenization in a separate Transformers environment.

The worker loads tokenizer assets only; no model is instantiated. Some versions
of Transformers import Torch even when the worker requests tokenizer-only use.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

REVISION = '3e22461f65e89153144f8adb70e3b8c2cc9845a7'


class ChatTokenizer:
    def __init__(self, python, snapshot, timeout=60,max_prompt_tokens=256):
        if type(max_prompt_tokens) is not int or max_prompt_tokens not in (256,2048):
            raise ValueError('Use a 256 or 2048 token prompt bound')
        self.max_prompt_tokens=max_prompt_tokens
        self.timeout = timeout
        self._responses = queue.Queue()
        self._lock = threading.Lock()
        env = os.environ.copy()
        env.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', PYTHONIOENCODING='utf-8')
        self.process = subprocess.Popen([str(python), '-I', str(Path(__file__).resolve()),
                                         '--worker', str(snapshot),str(max_prompt_tokens)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding='utf-8', env=env)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        try:
            self.identity = self._receive()
            if self.identity.get('revision') != REVISION:
                raise ValueError('Unexpected tokenizer revision')
            if self.identity.get('max_prompt_tokens')!=max_prompt_tokens:raise ValueError('Tokenizer prompt bound mismatch')
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                if len(line) > 131072:
                    self._responses.put(ValueError('Tokenizer frame exceeds bound'))
                    return
                self._responses.put(line)
        finally:
            self._responses.put(EOFError('Tokenizer worker exited'))

    def _receive(self):
        try:
            value = self._responses.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError('Tokenizer response timed out') from None
        if isinstance(value, BaseException):
            raise value
        result = json.loads(value)
        if 'error' in result:
            raise ValueError(result['error'])
        return result

    def _request(self, value):
        payload = json.dumps(value, ensure_ascii=False)
        if len(payload.encode('utf-8')) > 65536:
            raise ValueError('Chat text exceeds tokenizer bound')
        with self._lock:
            if self.process.poll() is not None:
                raise RuntimeError('Tokenizer is closed')
            self.process.stdin.write(payload + '\n')
            self.process.stdin.flush()
            return self._receive()

    def encode(self, messages, *, tools=None):
        return self._request(dict(messages=messages,tools=tools))['tokens']

    def decode(self,tokens):
        """Preserve native control tokens for protocol parsing, not display."""
        return self._request(dict(operation='decode',tokens=list(tokens)))['text']

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self._reader.join(timeout=2)
        self.process.stdout.close()

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


def worker(snapshot,max_prompt_tokens=256):
    # Isolated Python ignores PYTHONIOENCODING. Set pipe encoding explicitly
    # so Windows' legacy code page cannot corrupt accented text or emoji.
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    from collections.abc import Mapping
    from transformers import AutoTokenizer
    from jinja2.exceptions import TemplateError
    snapshot = Path(snapshot).resolve()
    if snapshot.name != REVISION or snapshot.parent.parent.name != 'models--google--gemma-4-E2B-it':
        raise ValueError('Use the pinned Google E2B-it snapshot')
    files = [snapshot / name for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')]
    identities = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    print(json.dumps(dict(revision=REVISION, source_sha256=identities,max_prompt_tokens=max_prompt_tokens,
                          torch_imported='torch' in sys.modules)), flush=True)
    while line := sys.stdin.readline(65538):
        if len(line.encode('utf-8')) > 65536 or not line.endswith('\n'):
            raise ValueError('Tokenizer request exceeds frame bound')
        try:
            request=json.loads(line)
            if request.get('operation')=='decode':
                ids=request['tokens']
                if not isinstance(ids,list) or not 1<=len(ids)<=256 or any(type(t) is not int or not 0<=t<262144 for t in ids):
                    raise ValueError('Decode requires 1..256 valid token IDs')
                result=dict(text=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False))
                print(json.dumps(result),flush=True)
                continue
            messages = request['messages']
            if not isinstance(messages,list) or not 1 <= len(messages) <= 64:
                raise ValueError('Chat requires 1..64 messages')
            if any(not isinstance(m,dict) or m.get('role') not in ('system','user','assistant','tool')
                   or not isinstance(m.get('content'),str) for m in messages):
                raise ValueError('Chat messages require supported roles and string content')
            tools=request.get('tools')
            if tools is not None and (not isinstance(tools,list) or len(tools)>8):raise ValueError('Use at most eight tool declarations')
            ids = tokenizer.apply_chat_template(messages, tools=tools,enable_thinking=False,
                                                tokenize=True, add_generation_prompt=True)
            if isinstance(ids,Mapping):ids = ids['input_ids']
            if not 1 <= len(ids) <= max_prompt_tokens:
                raise ValueError(f'Full conversation exceeds the current {max_prompt_tokens}-token prompt bound; use /reset')
            result = dict(tokens=list(ids))
        except (ValueError,KeyError,TypeError,TemplateError) as error:
            result = dict(error=str(error))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--worker' or sys.argv[3] not in ('256','2048'):
        raise SystemExit('This module is a tokenizer worker; use tools/gemma4_chat.py')
    worker(sys.argv[2],int(sys.argv[3]))
