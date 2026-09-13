"""Direct BO handoff for native MLIR_AIE graph stages on Phoenix.

The downstream transaction's DDR_PATCH relocations use the *same XRT BO* as
the producer. Firmware translates its device address into Shim DMA addresses;
never pass bo.address() as a scalar (that loses XRT's BO dependency tracking).
This avoids host copies/cache syncs between stages, but still traverses DDR.
It is not an adapter for ONNX Runtime's proprietary VitisAI allocations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
import threading
import time
from typing import Any


# Retain resources after an indeterminate completion. Releasing a BO while DMA
# may still reference it is unsafe. A failed chain cannot be reused.
_quarantined: list[Any] = []


def memory_bank(group_id: int) -> int:
    """XRT xcl_bo_flags: bank[15:0], context slot[23:16], flags[31:24].

    XRT's validate_bo_at_index checks grp.bank, not the entire encoded ID.
    Keeping the original group ID for allocation is essential; masking is only
    for connectivity comparison across contexts, never for allocating BOs.
    """
    if type(group_id) is not int or not 0 <= group_id <= 0xFFFFFFFF:
        raise ValueError("Invalid XRT memory group")
    return group_id & 0xFFFF


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str
    layout: str

    def __post_init__(self):
        if not isinstance(self.shape, tuple) or not self.shape or any(
            type(n) is not int or n <= 0 for n in self.shape
        ):
            raise ValueError("shape must be a nonempty tuple of positive integers")
        if self.dtype not in {"int8", "uint8", "bf16", "int16", "int32", "float32"}:
            raise ValueError(f"Unsupported storage dtype: {self.dtype}")
        if not self.layout:
            raise ValueError("An explicit physical layout is required")

    @property
    def nbytes(self) -> int:
        width = {"int8": 1, "uint8": 1, "bf16": 2, "int16": 2,
                 "int32": 4, "float32": 4}[self.dtype]
        return math.prod(self.shape) * width


@dataclass(frozen=True)
class ShimPatch:
    register: int
    argument: int  # zero-based tensor argument, i.e. XRT kernel argument - 3
    offset: int


def shim_patches(transaction: bytes) -> tuple[ShimPatch, ...]:
    """Strictly decode the v0.1 Phoenix TXN subset used by these native stages.

    Unknown encodings fail closed. This audits relocations, not arbitrary DMA
    strides: the caller's TensorSpec must describe the compiled kernel's ABI.
    """
    if len(transaction) < 16 or len(transaction) % 4:
        raise ValueError("Truncated or unaligned transaction")
    major, minor, generation, rows, cols, memrows = struct.unpack_from("<6B", transaction)
    count, size = struct.unpack_from("<II", transaction, 8)
    if (major, minor, generation, rows, memrows) != (0, 1, 3, 6, 1) or not 1 <= cols <= 5:
        raise ValueError("Expected a v0.1 Phoenix/AIE2 transaction")
    if size != len(transaction):
        raise ValueError("Transaction header length mismatch")
    cursor, seen, patches = 16, 0, []
    while cursor < size:
        op = struct.unpack_from("<I", transaction, cursor)[0]
        size_index = {0: 5, 1: 3, 3: 6, 4: 6, 0x80: 1, 0x81: 1}.get(op)
        if size_index is None or cursor + (size_index + 1) * 4 > size:
            raise ValueError(f"Unsupported/truncated TXN opcode {op:#x} at {cursor}")
        length = struct.unpack_from("<I", transaction, cursor + size_index * 4)[0]
        fixed = {0: 24, 3: 28, 4: 28, 0x80: 16, 0x81: 48}.get(op)
        if length < (size_index + 1) * 4 or length % 4 or cursor + length > size:
            raise ValueError("Invalid TXN operation length")
        if fixed is not None and length != fixed:
            raise ValueError("Invalid fixed-size TXN operation")
        if op == 0x81:
            words = struct.unpack_from("<12I", transaction, cursor)
            if any(words[i] for i in (2, 3, 4, 5, 7, 9, 11)):
                raise ValueError("Unsupported DDR_PATCH action or high bits")
            reg, arg, offset = words[6], words[8], words[10]
            # Row zero Shim BD address-low word: base 0x1d000, stride 0x20.
            local = reg & 0xFFFFF
            row, col = (reg >> 20) & 0x1F, (reg >> 25) & 0x7F
            if row or col >= cols or not 0x1D004 <= local <= 0x1D1E4 or (local - 0x1D004) % 32:
                raise ValueError(f"DDR_PATCH does not target a Shim BD address: {reg:#x}")
            # Firmware translates the first five tensor BO arguments. Later
            # arguments require another ABI; deliberately reject them for now.
            if arg >= 5:
                raise ValueError("Only the first five firmware-translated BO arguments are supported")
            patches.append(ShimPatch(reg, arg, offset))
        cursor += length
        seen += 1
    if seen != count:
        raise ValueError("Transaction operation count mismatch")
    if not patches:
        raise ValueError("Transaction has no Shim DDR relocations")
    return tuple(patches)


class DeviceBuffer:
    """Owned BO with explicit ABI and observable host-transfer operations.

    No host mapping is created. Host access is forbidden throughout a splice.
    The counters measure this API's calls, not bus traffic or driver internals.
    """

    def __init__(self, bo, spec: TensorSpec, device_index: int, group_id: int, pyxrt):
        if bo.size() != spec.nbytes:
            raise ValueError("BO allocation and TensorSpec byte counts differ")
        self._bo, self.spec = bo, spec
        self.device_index, self.group_id = device_index, group_id
        self._xrt = pyxrt
        self._access_lock = threading.Lock()
        self.host_read_bytes = self.host_write_bytes = self.host_sync_calls = 0

    @property
    def address(self) -> int:
        return int(self._bo.address())

    def write(self, data: bytes):
        if len(data) != self.spec.nbytes:
            raise ValueError("Upload must cover the entire tensor")
        if not self._access_lock.acquire(blocking=False):
            raise RuntimeError("Buffer is in use; host access is forbidden")
        try:
            self._bo.write(data, 0)
            self.host_write_bytes += len(data)
            self._bo.sync(self._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self.host_sync_calls += 1
        finally:
            self._access_lock.release()

    def read(self) -> bytes:
        if not self._access_lock.acquire(blocking=False):
            raise RuntimeError("Buffer is in use; host access is forbidden")
        try:
            self._bo.sync(self._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            self.host_sync_calls += 1
            result = bytes(self._bo.read(self.spec.nbytes, 0))
            self.host_read_bytes += len(result)
            return result
        finally:
            self._access_lock.release()


class KernelStage:
    """A native transaction and its declared tensor BO arguments.

    Argument keys use the XRT ABI (3=input, 4=params or output, ...).
    Bundles must have balanced DMA locks and wait for all output DMA completion.
    init_path, if supplied, is submitted once before the first execution. Never
    invent semaphore primes from an unrelated tile design.
    """

    def __init__(self, name: str, harness, instruction_bo, instruction_bytes: bytes,
                 specs: dict[int, TensorSpec], *, input_arg: int, output_arg: int,
                 init: tuple[Any, int] | None = None, owner=None):
        self.name, self.harness, self.owner = name, harness, owner
        self._device = harness.dev  # retain even if a borrowed session is closed
        self.kernel, self.context = harness.kernel, harness.context
        self.instruction_bo = instruction_bo
        self.instruction_size = len(instruction_bytes)  # firmware ABI is BYTES
        self.transaction_sha256 = hashlib.sha256(instruction_bytes).hexdigest()
        self.patches = shim_patches(instruction_bytes)
        self.specs = dict(specs)
        if not 2 <= len(specs) <= 5 or sorted(specs) != list(range(3, 3 + len(specs))):
            raise ValueError("Declare two to five contiguous tensor arguments from XRT argument 3")
        if input_arg == output_arg or input_arg not in specs or output_arg not in specs:
            raise ValueError("Distinct input and output arguments are required")
        for patch in self.patches:
            if patch.argument + 3 not in specs or patch.offset >= specs[patch.argument + 3].nbytes:
                raise ValueError("DDR_PATCH points outside a declared tensor argument")
        for arg in (input_arg, output_arg):
            if not any(p.argument + 3 == arg for p in self.patches):
                raise ValueError(f"No Shim relocation for tensor argument {arg}")
        self.input_arg, self.output_arg = input_arg, output_arg
        self.buffers: dict[int, DeviceBuffer] = {}
        for arg, spec in specs.items():
            gid = self.kernel.group_id(arg)
            bo = harness.pyxrt.bo(harness.dev, spec.nbytes, harness.pyxrt.bo.host_only, gid)
            self.buffers[arg] = DeviceBuffer(bo, spec, harness.device_idx, gid, harness.pyxrt)
        self._init = init
        self._primed = init is None
        self._chain = None
        self._closed = False

    def close(self):
        """Release owned BOs before context references; never close the owner session."""
        if self._chain is not None:
            raise RuntimeError("Close the owning splicer before its stages")
        self.buffers.clear()
        self.instruction_bo = self._init = None
        self.kernel = self.context = None
        self.owner = self.harness = self._device = None
        self._closed = True

    @classmethod
    def from_files(cls, name, harness, instruction_path, specs, *, input_arg, output_arg,
                   init_path=None):
        data = Path(instruction_path).read_bytes()
        shim_patches(data)  # validate before upload
        instruction_bo, _ = harness.create_instruction_bo(str(Path(instruction_path).resolve()))
        init = None
        if init_path is not None:
            init = harness.create_instruction_bo(str(Path(init_path).resolve()))
        return cls(name, harness, instruction_bo, data, specs,
                   input_arg=input_arg, output_arg=output_arg, init=init)

    @classmethod
    def from_session(cls, name, session, input_spec: TensorSpec, output_spec: TensorSpec):
        """Borrow an idle native InferenceSession's transaction and context.

        Uses separate BOs, leaving the session ring intact. The caller must keep
        the session open and use it exclusively until the splicer is released.
        Physical layouts must already agree; no CPU unswizzle is inserted.
        """
        if session._closed or any(b["in_flight_run"] is not None for b in session.buffers):
            raise ValueError("Session must be open and idle")
        if session.partitioned_graph is not None and session.partitioned_graph.cpu_partitions:
            raise ValueError("CPU fallback partitions do not expose native BO handoff")
        if (input_spec.nbytes, output_spec.nbytes) != (session.in_bytes, session.out_bytes):
            raise ValueError("Session byte counts disagree with tensor ABI")
        return cls(name, session.harness, session.bo_instr_exec,
                   Path(session.exec_txn_path).read_bytes(), {3: input_spec, 4: output_spec},
                   input_arg=3, output_arg=4, owner=session)

    @property
    def input(self) -> DeviceBuffer:
        return self.buffers[self.input_arg]

    @property
    def output(self) -> DeviceBuffer:
        return self.buffers[self.output_arg]

    def _submit(self, *, prime=False):
        if self.owner is not None and self.owner._closed:
            raise RuntimeError("Borrowed InferenceSession was closed")
        bo, size = self._init if prime else (self.instruction_bo, self.instruction_size)
        return self.kernel(3, bo, size, *(self.buffers[a]._bo for a in sorted(self.buffers)))

    def _prepare(self):
        run = self.harness.pyxrt.run(self.kernel)
        run.set_arg(0, 3)
        run.set_arg(1, self.instruction_bo)
        run.set_arg(2, self.instruction_size)
        for a, buffer in self.buffers.items():
            run.set_arg(a, buffer._bo)
        return run


@dataclass(frozen=True)
class SpliceResult:
    output: DeviceBuffer
    stage_us: tuple[float, ...]  # submit+wait, includes context scheduling
    handoff_us: tuple[float, ...]  # producer wait returned -> consumer submit begins
    wall_us: float
    completed_stages: tuple[str, ...]
    host_read_bytes: int
    host_write_bytes: int
    host_sync_calls: int


class KernelSplicer:
    """Single-flight ERT completion-ordered chain; no sleeps, IPC, or host sync.

    Construction binds producer outputs into consumer Shim relocations.
    Multiple contexts are ordered by run.wait(timeout_ms), not by an assumed
    cross-context lock namespace. An error/timeout poisons the entire chain.
    """

    def __init__(self, stages: list[KernelStage] | tuple[KernelStage, ...]):
        if not stages or len({id(s) for s in stages}) != len(stages):
            raise ValueError("A chain needs distinct stages")
        self.stages = tuple(stages)
        self._token = object()  # no stage -> splicer reference cycle at XRT teardown
        self._mutex = threading.Lock()
        self._failed = False
        self._closed = False
        for stage in self.stages:
            if stage._chain is not None or stage._closed:
                raise ValueError("Stage must be open and not belong to another splicer")
        # Validate every edge before mutating bindings.
        for producer, consumer in zip(self.stages, self.stages[1:]):
            if producer.output.spec != consumer.input.spec:
                raise ValueError("Splice shape/dtype/physical layout mismatch")
            if producer.output.device_index != consumer.input.device_index:
                raise ValueError("Cannot splice across devices")
            if memory_bank(producer.output.group_id) != memory_bank(consumer.kernel.group_id(consumer.input_arg)):
                raise ValueError("Memory-group mismatch would require a driver copy")
        original_inputs = [stage.input for stage in self.stages]
        try:
            for producer, consumer in zip(self.stages, self.stages[1:]):
                consumer.buffers[consumer.input_arg] = producer.output
            self._runs = tuple(stage._prepare() for stage in self.stages)
        except BaseException:
            for stage, original in zip(self.stages, original_inputs):
                stage.buffers[stage.input_arg] = original
            raise
        self._bindings = tuple(dict(stage.buffers) for stage in self.stages)
        for stage in self.stages:
            stage._chain = self._token

    def close(self):
        """Release the exclusive claim after successful completion, retaining stages."""
        if not self._mutex.acquire(blocking=False):
            raise RuntimeError("Cannot close an in-flight splice")
        try:
            if self._closed:
                return
            if self._failed:
                raise RuntimeError("Failed splice resources must remain retained")
            self._runs = ()
            for stage in self.stages:
                stage._chain = None
            self._closed = True
        finally:
            self._mutex.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if not self._failed:
            self.close()

    def bindings(self) -> list[dict[str, Any]]:
        """Audit BO identity and downstream relocation targets before dispatch."""
        result = []
        for producer, consumer in zip(self.stages, self.stages[1:]):
            if producer.output is not consumer.input:
                raise RuntimeError("A linked BO binding was changed")
            result.append({
                "producer": producer.name, "consumer": consumer.name,
                "same_bo": producer.output._bo is consumer.input._bo,
                "dpa": hex(producer.output.address), "nbytes": producer.output.spec.nbytes,
                "group_id": producer.output.group_id,
                "consumer_group_id": consumer.kernel.group_id(consumer.input_arg),
                "memory_bank": memory_bank(producer.output.group_id),
                "consumer_patches": [
                    {"register": hex(p.register), "offset": p.offset,
                     "bo_address_plus_offset": hex(producer.output.address + p.offset)}
                    for p in consumer.patches if p.argument + 3 == consumer.input_arg
                ],
            })
        return result

    def run(self, timeout_ms: int = 2000) -> SpliceResult:
        if type(timeout_ms) is not int or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        if not self._mutex.acquire(blocking=False):
            raise RuntimeError("Splice is already in flight")
        buffers = {id(b): b for s in self.stages for b in s.buffers.values()}.values()
        before = tuple(sum(getattr(b, k) for b in buffers) for k in
                       ("host_read_bytes", "host_write_bytes", "host_sync_calls"))
        run, submitted = None, False
        locked = []
        try:
            if self._closed:
                raise RuntimeError("Splice is closed")
            if self._failed:
                raise RuntimeError("Splice failed previously; retained BOs cannot be reused")
            for stage, bound in zip(self.stages, self._bindings):
                if stage.buffers != bound:
                    raise RuntimeError("A prepared BO binding was changed")
                if stage.owner is not None and stage.owner._closed:
                    raise RuntimeError("Borrowed InferenceSession was closed")
            for b in buffers:
                if not b._access_lock.acquire(blocking=False):
                    raise RuntimeError("Buffer is already in use")
                locked.append(b)
            times, gaps, completed = [], [], []
            t0 = time.perf_counter_ns()
            last_end = None
            for stage, command in zip(self.stages, self._runs):
                if not stage._primed:
                    submitted = True
                    run = stage._submit(prime=True)
                    self._wait(run, stage, timeout_ms)
                    stage._primed = True
                begin = time.perf_counter_ns()
                if last_end is not None:
                    gaps.append((begin - last_end) / 1000)
                submitted = True
                run = command
                run.start()
                self._wait(run, stage, timeout_ms)
                last_end = time.perf_counter_ns()
                times.append((last_end - begin) / 1000)
                completed.append(stage.name)
                run = None
            wall = (time.perf_counter_ns() - t0) / 1000
            after = tuple(sum(getattr(b, k) for b in buffers) for k in
                          ("host_read_bytes", "host_write_bytes", "host_sync_calls"))
            traffic = tuple(a - b for a, b in zip(after, before))
            if any(traffic):
                raise RuntimeError(f"Unexpected host traffic during splice: {traffic}")
            return SpliceResult(self.stages[-1].output, tuple(times), tuple(gaps), wall,
                                tuple(completed), *traffic)
        except BaseException:
            if submitted:
                self._failed = True
                _quarantined.append((self, run))
            raise
        finally:
            if not self._failed:
                for b in locked:
                    b._access_lock.release()
            self._mutex.release()

    @staticmethod
    def _wait(run, stage, timeout_ms):
        state = run.wait(timeout_ms)
        expected = stage.harness.pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
        if state != expected:
            raise RuntimeError(f"{stage.name}: ERT completion failed: {state}")
