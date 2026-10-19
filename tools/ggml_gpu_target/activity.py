"""Owned client for the research-only bounded Radeon activity helper."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import math
import subprocess
import threading


class ActivityHelper:
    def __init__(self,executable,runtime):
        self.process=subprocess.Popen([str(executable),str(runtime)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        self.active=False
        try:
            self.identity=self._receive(timeout=15)
            if (self.identity.get('ready') is not True or self.identity.get('hardware') is not True
                    or self.identity.get('vendor')!=0x1002 or self.identity.get('device')!=0x15bf
                    or self.identity.get('hip_luid_matched') is not True):
                raise RuntimeError('Activity helper did not select Phoenix hardware')
        except BaseException:
            self.close();raise

    def _receive(self,timeout=5):
        timer=threading.Timer(timeout,self.process.kill);timer.daemon=True;timer.start()
        try:line=self.process.stdout.readline(4097)
        finally:timer.cancel()
        if not line or len(line)>4096:raise RuntimeError('GPU activity reply missing or oversized')
        try:value=json.loads(line)
        except (ValueError,UnicodeError) as error:
            raise RuntimeError('GPU activity helper failed: '+line[:512].decode('utf-8',errors='replace').strip()) from error
        if not isinstance(value,dict):raise ValueError('GPU activity reply must be an object')
        return value

    def _send(self,command):
        self.process.stdin.write((command+'\n').encode('ascii'));self.process.stdin.flush()

    def start(self,milliseconds):
        if type(milliseconds) is not int or not 1<=milliseconds<=100:
            raise ValueError('Use a bounded 1..100 ms GPU activity window')
        if self.active:raise RuntimeError('An activity window is already active')
        self._send('start '+str(milliseconds));reply=self._receive()
        if (set(reply)!={'started','window_ms'} or reply['started'] is not True
                or type(reply['window_ms']) is not int or reply['window_ms']!=milliseconds):
            raise ValueError('Unexpected GPU activity acknowledgement')
        self.active=True

    def wait(self,*,stop=False):
        if not self.active:raise RuntimeError('No GPU activity window to join')
        self._send('stop' if stop else 'wait');reply=self._receive()
        if (reply.get('completed') is not True or reply.get('gpu_validated') is not True
                or type(reply.get('dispatches')) is not int or reply['dispatches']<1
                or type(reply.get('elapsed_ms')) not in (int,float)
                or not math.isfinite(reply['elapsed_ms']) or reply['elapsed_ms']<=0):
            raise ValueError('GPU activity completion not verified')
        self.active=False
        return reply

    def close(self):
        if self.process.poll() is None:
            try:self._send('quit');self.process.wait(timeout=5)
            except (OSError,subprocess.TimeoutExpired):self.process.kill();self.process.wait(timeout=5)
        self.process.stdin.close();self.process.stdout.close();self.active=False

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
