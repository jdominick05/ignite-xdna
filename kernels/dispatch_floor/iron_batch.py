"""Batched submission for ordinary `@iron.jit` designs, without editing the toolchain.

WHY THIS EXISTS
---------------
`results/aie/dispatch_runlist_npu.log` measured a dispatch amortising to 36.3 us when 64 runs
are submitted as one `pyxrt.runlist`, against 617.0 us through IRON's own path -- a 17x drop,
and a quarter of the 169.8 us that log had attributed to hardware. But that measurement drove
raw pyxrt directly against a cached xclbin. Nothing under mlir-aie's `aie/utils/hostruntime/`
references `runlist`, so the 17x was unreachable from any design anyone would actually write.

This module makes it reachable, from inside IRON's own host path, with no file outside this
repository modified.

HOW IT WORKS
------------
`XRTHostRuntime.run()` prepares arguments, validates them against the xclbin ABI, builds the
instruction buffer, and then submits with

    h = kernel_handle.kernel(3, insts_bo, insts_bytes, *buffers)
    r = h.wait()

`kernel(...)` in pyxrt creates AND STARTS a run, which is exactly the thing that cannot go
into a runlist. So inside the context manager this module swaps `kernel_handle.kernel` for a
proxy whose `__call__` builds the run *unstarted* -- `pyxrt.run(kernel)` plus `set_arg` -- adds
it to a runlist, and hands back a deferred stand-in. Everything else in IRON's `run()` executes
unchanged: the same ABI validation, the same instruction buffer, the same argument order.

    with iron_batch.batched(max_in_flight=64):
        for i in range(64):
            design(a[i], b[i], c[i])     # queued, not executed
    # one execute(), one wait(), on leaving the block

TWO THINGS THIS GETS RIGHT ON PURPOSE, BECAUSE BOTH FAIL SILENTLY
-----------------------------------------------------------------
1. **Buffers must not be shared between queued calls.** Concurrent buffer access within a
   runlist is undefined, and the symptom is wrong data, not an error. Every queued call's
   buffers are checked against the rest of the batch and a repeat raises. Pass
   `allow_shared_buffers=True` only if you genuinely mean it.

2. **Batching gives up per-call status entirely, so the caller MUST verify its outputs.**
   IRON checks each call's `.wait()` against `ERT_CMD_STATE_COMPLETED`, and at queue time that
   answer does not exist, so the proxy returns a placeholder to let IRON's check pass. It
   cannot be made good afterwards either: a run inside a runlist cannot be polled on this
   binding -- `run.state()` raises *"Cannot poll a command that has not been submitted"*,
   because the runlist owns submission. `runlist.wait()` is the only completion signal there
   is. **Verifying output buffers is therefore the only correctness gate under batching**, and
   a batch that silently did nothing would otherwise look extremely fast. Per-call timings go
   the same way: `XRTKernelResult` carries a `perf_counter_ns` bracket around a submit that no
   longer waits for anything. Read the batch bracket this module reports, never the per-call
   one.

SCOPE: THE TRANSACTION SUBMIT PATH ONLY
---------------------------------------
`XRTHostRuntime.run()` has two submit paths and this module batches one of them. The
transaction path -- the default for `@iron.jit` designs, and what both measured designs use --
is the one described above. The full-ELF path (`is_full_elf`) calls `pyxrt.run(...)` on the
kernel itself, so it would hand the C++ binding this module's proxy; it is refused with a
clear message rather than allowed to fail as a pybind type error.

Usage requires the ironenv with the XRT SDK on PATH and pyxrt importable.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

_state = None


class _DeferredRun:
    """Stands in for a started run that has not been submitted yet."""

    def __init__(self, batch, index):
        self._batch = batch
        self._index = index

    def wait(self):
        # IRON compares this against ERT_CMD_STATE_COMPLETED, so a value has to be
        # returned here. It is a PLACEHOLDER and nothing more: the run has not been
        # submitted yet, and once the batch is flushed its per-run state still cannot
        # be recovered -- run.state() raises on a run the runlist owns. Nothing
        # downstream ever makes this good. Verify your output buffers.
        import pyxrt

        return pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED

    def wait2(self):
        return None


class _QueueingKernel:
    """Wraps a `pyxrt.kernel` so calling it queues an unstarted run."""

    def __init__(self, kernel, batch):
        self._kernel = kernel
        self._batch = batch

    def __call__(self, *args):
        return self._batch.enqueue(self._kernel, args)

    def __getattr__(self, name):
        # group_id() and anything else IRON asks of the kernel passes straight through.
        return getattr(self._kernel, name)


class _Batch:
    def __init__(self, max_in_flight, allow_shared_buffers):
        self.max_in_flight = max_in_flight
        self.allow_shared_buffers = allow_shared_buffers
        self.context = None
        self.runlist = None
        self.runs = []
        self.seen_buffers = set()
        self.flushes = []          # (n_runs, wall_ns) per flush
        self.states = []

    # -- queueing ---------------------------------------------------------- #

    def enqueue(self, kernel, args):
        import pyxrt

        if self.runlist is None:
            if self.context is None:
                raise RuntimeError(
                    "iron_batch: no hardware context captured; the patch did not see a "
                    "kernel handle. This is a bug in the patch, not in your design."
                )
            self.runlist = pyxrt.runlist(self.context)

        if not self.allow_shared_buffers:
            for a in args[3:]:            # args = (opcode, insts_bo, insts_bytes, *buffers)
                key = id(a)
                if key in self.seen_buffers:
                    raise RuntimeError(
                        "iron_batch: a buffer object is reused by two calls in the same "
                        "batch. Concurrent buffer access within a runlist is undefined and "
                        "produces wrong data with no error. Give each queued call its own "
                        "buffers, or pass allow_shared_buffers=True if you are certain."
                    )
                self.seen_buffers.add(key)

        run = pyxrt.run(kernel)
        for i, a in enumerate(args):
            run.set_arg(i, a)
        self.runlist.add(run)
        self.runs.append(run)

        if self.max_in_flight and len(self.runs) >= self.max_in_flight:
            self.flush()
        return _DeferredRun(self, len(self.runs) - 1)

    # -- flushing ---------------------------------------------------------- #

    def flush(self):
        import pyxrt

        if not self.runs:
            return
        n = len(self.runs)
        t0 = time.perf_counter_ns()
        self.runlist.execute()
        self.runlist.wait()
        wall = time.perf_counter_ns() - t0

        # There is deliberately no per-run status check here, because none is
        # available. A run inside a runlist CANNOT be polled individually on this
        # binding: `run.state()` raises "Cannot poll a command that has not been
        # submitted". The runlist owns submission, so `runlist.wait()` above is the
        # only completion signal there is. That is why every caller of this module
        # must verify its OUTPUT BUFFERS -- with batching there is no per-call status
        # to check, and a batch that silently did nothing would otherwise look
        # extremely fast.
        self.flushes.append((n, wall))

        # Drop the runlist before the next one is built; leaving these to
        # interpreter-exit collection faults on this driver.
        self.runs = []
        self.seen_buffers = set()
        del self.runlist
        self.runlist = None

    # -- reporting --------------------------------------------------------- #

    @property
    def total_runs(self) -> int:
        return sum(n for n, _ in self.flushes)

    @property
    def total_ns(self) -> int:
        return sum(w for _, w in self.flushes)

    @property
    def per_run_us(self) -> float:
        return (self.total_ns / 1e3 / self.total_runs) if self.total_runs else float("nan")


@contextmanager
def batched(max_in_flight: int = 64, allow_shared_buffers: bool = False):
    """Queue every `@iron.jit` call made inside the block and submit them as one runlist.

    Yields the batch object, whose `per_run_us` is the amortised cost per dispatch and whose
    `flushes` lists (n_runs, wall_ns) for each submission actually made.
    """
    global _state
    if _state is not None:
        raise RuntimeError("iron_batch: batched() does not nest")

    from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime

    batch = _Batch(max_in_flight, allow_shared_buffers)
    original_run = XRTHostRuntime.run

    def patched_run(self, kernel_handle, args, *a, **kw):
        # Only the transaction submit path is batched. The full-ELF path calls
        # `pyxrt.run(kernel_handle.kernel)` itself, which would hand the C++ binding
        # this module's proxy and fail with an opaque pybind type error. Say so.
        if getattr(kernel_handle, "is_full_elf", False):
            raise RuntimeError(
                "iron_batch: this design uses IRON's full-ELF submit path, which "
                "builds its own pyxrt.run() and cannot be batched by this module. "
                "Only the transaction path (the default for @iron.jit designs) is "
                "supported."
            )
        batch.context = getattr(kernel_handle, "context", None) or batch.context
        real_kernel = kernel_handle.kernel
        kernel_handle.kernel = _QueueingKernel(real_kernel, batch)
        try:
            return original_run(self, kernel_handle, args, *a, **kw)
        finally:
            kernel_handle.kernel = real_kernel

    XRTHostRuntime.run = patched_run
    _state = batch
    try:
        yield batch
        batch.flush()                      # anything still queued
    finally:
        XRTHostRuntime.run = original_run
        _state = None
