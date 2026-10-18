"""Four-column, sixteen-core DFL transport for Phoenix XDNA1.

The host input is anchor-major and remains contiguous.  Each shim DMA writes
one column's 2,100 anchors into a 75,600-byte MemTile input slot.  Four MemTile
MM2S channels feed the four cores in that column.  A core consumes 35 x 15
anchor chunks through local ping/pong buffers.

Each core emits one anchor-major wire record containing four boxes followed by
80 class scores.  The four core streams use the four remaining MemTile S2MM
channels.  The MemTile then uses two strided MM2S descriptors to deinterleave
the shared output slot into contiguous boxes and scores on the two shim S2MM
channels.  This preserves separate zero-copy host BOs without requiring eight
core-to-MemTile routes from a six-channel MemTile.

The shared output slot is handed from core to core by explicit turn locks, one
per core, in anchor order.  ``module(cols, cores_per_col)`` instantiates a
reduced design (the first ``cores_per_col`` cores of the first ``cols``
columns) with the same anchor mapping, lock numbering and host BO layout as the
full design, so a probe differs from production only in how many cores exist.
"""
from pathlib import Path

INPUT_ANCHOR_BYTES = 144
ANCHORS = 8400
CORES = 16
COLS = 4
CORES_PER_COL = 4
ANCHORS_PER_CORE = ANCHORS // CORES
CHUNKS_PER_CORE = ANCHORS_PER_CORE // 15
CORE_INPUT_BYTES = ANCHORS_PER_CORE * INPUT_ANCHOR_BYTES
CORE_BOX_BYTES = ANCHORS_PER_CORE * 4 * 4
CORE_SCORE_BYTES = ANCHORS_PER_CORE * 80 * 4
CORE_WIRE_BYTES = CORE_BOX_BYTES + CORE_SCORE_BYTES
COL_INPUT_BYTES = CORES_PER_COL * CORE_INPUT_BYTES
COL_BOX_BYTES = CORES_PER_COL * CORE_BOX_BYTES
COL_SCORE_BYTES = CORES_PER_COL * CORE_SCORE_BYTES
INPUT_BYTES = ANCHORS * INPUT_ANCHOR_BYTES
BOXES_BYTES = ANCHORS * 4 * 4
SCORES_BYTES = ANCHORS * 80 * 4
CHUNK_INPUT_BYTES = 15 * INPUT_ANCHOR_BYTES
CHUNK_BOX_BYTES = 15 * 4 * 4
CHUNK_SCORE_BYTES = 15 * 80 * 4
CHUNK_WIRE_BYTES = CHUNK_BOX_BYTES + CHUNK_SCORE_BYTES
# One anchor's wire record: four box floats, then eighty score floats.
RECORD_BOX_BYTES = 4 * 4
RECORD_SCORE_BYTES = 80 * 4
RECORD_BYTES = RECORD_BOX_BYTES + RECORD_SCORE_BYTES
OUTPUT_DMA_PART_BYTES = CHUNK_WIRE_BYTES * 5
OUTPUT_DMA_PARTS = CORE_WIRE_BYTES // OUTPUT_DMA_PART_BYTES
MEM_IN_BYTES = CORE_INPUT_BYTES
MEM_OUT_BYTES = CORE_WIRE_BYTES

# MemTile lock numbers are fixed per core row, so a reduced probe uses the same
# lock identities as the full design.
LOCK_IN_SLOT = 0
LOCK_IN_READY = 1
LOCK_OUT_TURN = 5
LOCK_BOX_READY = 9
LOCK_SCORE_READY = 13
LOCK_PART_READY = 17

assert CORE_WIRE_BYTES % OUTPUT_DMA_PART_BYTES == 0


def check_shape(cols, cores_per_col):
    if not (1 <= cols <= COLS and 1 <= cores_per_col <= CORES_PER_COL):
        raise ValueError(
            f"cols must be 1..{COLS} and cores_per_col 1..{CORES_PER_COL}, "
            f"got {cols} x {cores_per_col}"
        )


def active_cores(cols=COLS, cores_per_col=CORES_PER_COL):
    """Global core indices instantiated by a (possibly reduced) design."""
    check_shape(cols, cores_per_col)
    return [col * CORES_PER_COL + row for col in range(cols) for row in range(cores_per_col)]


def _emit(lines, text):
    lines.append("    " + text)


def _buffer(lines, name, tile, size):
    _emit(lines, f'%{name} = aie.buffer(%{tile}) {{sym_name = "{name}"}} : memref<{size}xi8>')


def _core_dma(lines, tile, names):
    """Emit local input S2MM and one interleaved output MM2S engine."""
    inp, out = names
    one = "%one"
    _emit(lines, f"aie.mem(%{tile}) {{")
    _emit(lines, f"  %one = arith.constant 1 : i32")

    _emit(lines, "  aie.dma_start(S2MM, 0, ^in_ping, ^out_wire)")
    _emit(lines, "^in_ping:")
    _emit(lines, f"  aie.use_lock(%{inp['free_ping']}, AcquireGreaterEqual, {one})")
    _emit(lines, f"  aie.dma_bd(%{inp['ping']} : memref<{CHUNK_INPUT_BYTES}xi8> offset = 0 len = {CHUNK_INPUT_BYTES})")
    _emit(lines, f"  aie.use_lock(%{inp['ready_ping']}, Release, {one})")
    _emit(lines, "  aie.next_bd ^in_pong")
    _emit(lines, "^in_pong:")
    _emit(lines, f"  aie.use_lock(%{inp['free_pong']}, AcquireGreaterEqual, {one})")
    _emit(lines, f"  aie.dma_bd(%{inp['pong']} : memref<{CHUNK_INPUT_BYTES}xi8> offset = 0 len = {CHUNK_INPUT_BYTES})")
    _emit(lines, f"  aie.use_lock(%{inp['ready_pong']}, Release, {one})")
    _emit(lines, "  aie.next_bd ^in_ping")

    _emit(lines, "^out_wire:")
    _emit(lines, "  aie.dma_start(MM2S, 0, ^wire_ping, ^end)")
    _emit(lines, "^wire_ping:")
    _emit(lines, f"  aie.use_lock(%{out['ready_ping']}, AcquireGreaterEqual, {one})")
    _emit(lines, f"  aie.dma_bd(%{out['ping']} : memref<{CHUNK_WIRE_BYTES}xi8> offset = 0 len = {CHUNK_WIRE_BYTES})")
    _emit(lines, f"  aie.use_lock(%{out['free_ping']}, Release, {one})")
    _emit(lines, "  aie.next_bd ^wire_pong")
    _emit(lines, "^wire_pong:")
    _emit(lines, f"  aie.use_lock(%{out['ready_pong']}, AcquireGreaterEqual, {one})")
    _emit(lines, f"  aie.dma_bd(%{out['pong']} : memref<{CHUNK_WIRE_BYTES}xi8> offset = 0 len = {CHUNK_WIRE_BYTES})")
    _emit(lines, f"  aie.use_lock(%{out['free_pong']}, Release, {one})")
    _emit(lines, "  aie.next_bd ^wire_ping")
    _emit(lines, "^end:")
    _emit(lines, "  aie.end")
    _emit(lines, "}")


def _lock_set(prefix, initial_free):
    names = {
        "free_ping": f"{prefix}_free_ping",
        "ready_ping": f"{prefix}_ready_ping",
        "free_pong": f"{prefix}_free_pong",
        "ready_pong": f"{prefix}_ready_pong",
    }
    locks = [
        (names["free_ping"], initial_free),
        (names["ready_ping"], 0),
        (names["free_pong"], initial_free),
        (names["ready_pong"], 0),
    ]
    return names, locks


def module(cols=COLS, cores_per_col=CORES_PER_COL):
    check_shape(cols, cores_per_col)
    rows = range(cores_per_col)
    last = cores_per_col - 1
    lines = ["module {", "  aie.device(npu1) {"]
    emit = lambda s: _emit(lines, s)

    emit(f"func.func private @dfl_decode_chunk(memref<{CHUNK_INPUT_BYTES}xi8>, memref<{CHUNK_WIRE_BYTES}xi8>, i32) -> ()")

    for col in range(cols):
        tiles = [f"t{col}_{row}" for row in range(2 + cores_per_col)]
        for row, tile in enumerate(tiles):
            emit(f"%{tile} = aie.tile({col}, {row})")
        shim = tiles[0]
        mem = tiles[1]
        cores = tiles[2:]

        emit(f"%mem_in_{col} = aie.buffer(%{mem}) {{sym_name = \"mem_in_{col}\", address = 0 : i32}} : memref<{MEM_IN_BYTES}xi8>")
        emit(f"%mem_out_{col} = aie.buffer(%{mem}) {{sym_name = \"mem_out_{col}\", address = 131072 : i32}} : memref<{MEM_OUT_BYTES}xi8>")

        emit(f"%in_slot_free_{col} = aie.lock(%{mem}, {LOCK_IN_SLOT}) {{init = 1 : i32}}")
        for row in rows:
            emit(f"%in_ready_{col}_{row} = aie.lock(%{mem}, {LOCK_IN_READY + row}) {{init = 0 : i32}}")
        # The output slot is a single-token sequence per core: core output
        # releases box_ready, boxes releases score_ready, and scores hands the
        # slot to the next core's turn lock.  One shared slot lock acquired by
        # all core S2MM channels let a channel whose core had no input yet win
        # the slot; core 0's output then backed up, core 0 stopped draining its
        # input, and the input slot never reached the next core.
        for row in rows:
            emit(f"%out_turn_{col}_{row} = aie.lock(%{mem}, {LOCK_OUT_TURN + row}) {{init = {1 if row == 0 else 0} : i32}}")
        for row in rows:
            emit(f"%box_ready_{col}_{row} = aie.lock(%{mem}, {LOCK_BOX_READY + row}) {{init = 0 : i32}}")
        for row in rows:
            emit(f"%score_ready_{col}_{row} = aie.lock(%{mem}, {LOCK_SCORE_READY + row}) {{init = 0 : i32}}")
        for row in rows:
            emit(f"%part_ready_{col}_{row} = aie.lock(%{mem}, {LOCK_PART_READY + row}) {{init = 0 : i32}}")

        emit(f"aie.flow(%{shim}, DMA : 0, %{mem}, DMA : 0)")
        emit(f"aie.flow(%{mem}, DMA : 4, %{shim}, DMA : 0)")
        emit(f"aie.flow(%{mem}, DMA : 5, %{shim}, DMA : 1)")
        for row, core in enumerate(cores):
            emit(f"aie.flow(%{mem}, DMA : {row}, %{core}, DMA : 0)")
            # One output stream per core.  Channels 1..4 are the four
            # remaining MemTile S2MM routes; the MemTile deinterleaves the
            # wire records for the two shim output streams below.
            emit(f"aie.flow(%{core}, DMA : 0, %{mem}, DMA : {row + 1})")

        emit(f"aie.shim_dma_allocation @in_{col}(%{shim}, MM2S, 0)")
        emit(f"aie.shim_dma_allocation @boxes_{col}(%{shim}, S2MM, 0)")
        emit(f"aie.shim_dma_allocation @scores_{col}(%{shim}, S2MM, 1)")

        emit(f"aie.memtile_dma(%{mem}) {{")
        emit("  %one = arith.constant 1 : i32")
        emit("  aie.dma_start(S2MM, 0, ^host_input_0, ^in_core_0)")
        for row in rows:
            emit(f"^host_input_{row}:")
            next_label = f"^host_input_{row + 1}" if row < last else "^mem_end"
            emit(f"  aie.use_lock(%in_slot_free_{col}, AcquireGreaterEqual, %one)")
            emit(f"  aie.dma_bd(%mem_in_{col} : memref<{MEM_IN_BYTES}xi8> offset = 0 len = {CORE_INPUT_BYTES})")
            emit(f"  aie.use_lock(%in_ready_{col}_{row}, Release, %one)")
            emit(f"  aie.next_bd {next_label}")
        emit("^in_core_0:")
        # The input MM2S channels start independently and wait for their
        # corresponding host segment.  The labels are separate DMA programs.
        for row in rows:
            if row:
                emit(f"^in_core_{row}:")
            next_channel = f"^in_core_{row + 1}" if row < last else "^out_core_0"
            emit(f"  aie.dma_start(MM2S, {row}, ^in_core_bd_{row}, {next_channel})")
            emit(f"^in_core_bd_{row}:")
            emit(f"  aie.use_lock(%in_ready_{col}_{row}, AcquireGreaterEqual, %one)")
            emit(f"  aie.dma_bd(%mem_in_{col} : memref<{MEM_IN_BYTES}xi8> offset = 0 len = {CORE_INPUT_BYTES})")
            emit(f"  aie.use_lock(%in_slot_free_{col}, Release, %one)")
            emit("  aie.next_bd ^mem_end")
        # Each core emits one interleaved wire stream.  The output slot is
        # consumed in order: core output, boxes egress, then scores egress.
        for row in rows:
            emit(f"^out_core_{row}:")
            next_channel = f"^out_core_{row + 1}" if row < last else "^box_stream"
            emit(f"  aie.dma_start(S2MM, {row + 1}, ^out_core_bd_{row}_0, {next_channel})")
            for part in range(OUTPUT_DMA_PARTS):
                emit(f"^out_core_bd_{row}_{part}:")
                if part == 0:
                    emit(f"  aie.use_lock(%out_turn_{col}_{row}, AcquireGreaterEqual, %one)")
                else:
                    emit(f"  aie.use_lock(%part_ready_{col}_{row}, AcquireGreaterEqual, %one)")
                offset = part * OUTPUT_DMA_PART_BYTES
                emit(f"  aie.dma_bd(%mem_out_{col} : memref<{MEM_OUT_BYTES}xi8> offset = {offset} len = {OUTPUT_DMA_PART_BYTES})")
                if part + 1 == OUTPUT_DMA_PARTS:
                    emit(f"  aie.use_lock(%box_ready_{col}_{row}, Release, %one)")
                else:
                    emit(f"  aie.use_lock(%part_ready_{col}_{row}, Release, %one)")
                if part + 1 == OUTPUT_DMA_PARTS:
                    emit("  aie.next_bd ^mem_end")
                else:
                    emit(f"  aie.next_bd ^out_core_bd_{row}_{part + 1}")
        emit("^box_stream:")
        emit("  aie.dma_start(MM2S, 4, ^box_stream_0, ^score_stream)")
        for row in rows:
            emit(f"^box_stream_{row}:")
            emit(f"  aie.use_lock(%box_ready_{col}_{row}, AcquireGreaterEqual, %one)")
            emit(f"  aie.dma_bd(%mem_out_{col} : memref<{MEM_OUT_BYTES}xi8> offset = 0 len = {CORE_BOX_BYTES} sizes = [{ANCHORS_PER_CORE}, {RECORD_BOX_BYTES}] strides = [{RECORD_BYTES}, 1])")
            emit(f"  aie.use_lock(%score_ready_{col}_{row}, Release, %one)")
            emit(f"  aie.next_bd ^box_stream_{row + 1}" if row < last else "  aie.next_bd ^mem_end")
        emit("^score_stream:")
        emit("  aie.dma_start(MM2S, 5, ^score_stream_0, ^mem_end)")
        for row in rows:
            emit(f"^score_stream_{row}:")
            emit(f"  aie.use_lock(%score_ready_{col}_{row}, AcquireGreaterEqual, %one)")
            # Scores follow the four box floats inside each record.  Reading
            # from record offset 0 (the original descriptor) returned the box
            # floats as scores[0:4] and dropped scores[76:80] on silicon.
            emit(f"  aie.dma_bd(%mem_out_{col} : memref<{MEM_OUT_BYTES}xi8> offset = {RECORD_BOX_BYTES} len = {CORE_SCORE_BYTES} sizes = [{ANCHORS_PER_CORE}, {RECORD_SCORE_BYTES}] strides = [{RECORD_BYTES}, 1])")
            emit(f"  aie.use_lock(%out_turn_{col}_{(row + 1) % cores_per_col}, Release, %one)")
            emit(f"  aie.next_bd ^score_stream_{row + 1}" if row < last else "  aie.next_bd ^mem_end")
        emit("^mem_end:")
        emit("  aie.end")
        emit("}")

        for row, core in enumerate(cores):
            prefix = f"c{col}_{row}"
            in_names, in_locks = _lock_set(f"{prefix}_in", 1)
            wire_names, wire_locks = _lock_set(f"{prefix}_wire", 1)
            for number, (name, value) in enumerate(in_locks + wire_locks):
                # The lock namespace is local to the tile; preserve the
                # explicit order rather than depending on allocator ordering.
                emit(f"%{name} = aie.lock(%{core}, {number}) {{init = {value} : i32}}")

            _buffer(lines, f"{prefix}_in_ping", core, CHUNK_INPUT_BYTES)
            _buffer(lines, f"{prefix}_in_pong", core, CHUNK_INPUT_BYTES)
            _buffer(lines, f"{prefix}_wire_ping", core, CHUNK_WIRE_BYTES)
            _buffer(lines, f"{prefix}_wire_pong", core, CHUNK_WIRE_BYTES)
            _core_dma(lines, core, [
                {"ping": f"{prefix}_in_ping", "pong": f"{prefix}_in_pong",
                 "free_ping": in_names["free_ping"], "ready_ping": in_names["ready_ping"],
                 "free_pong": in_names["free_pong"], "ready_pong": in_names["ready_pong"]},
                {"ping": f"{prefix}_wire_ping", "pong": f"{prefix}_wire_pong",
                 "free_ping": wire_names["free_ping"], "ready_ping": wire_names["ready_ping"],
                 "free_pong": wire_names["free_pong"], "ready_pong": wire_names["ready_pong"]},
            ])

            emit(f"aie.core(%{core}) {{")
            emit("  %one = arith.constant 1 : i32")
            global_core = col * CORES_PER_COL + row
            core_base = global_core * ANCHORS_PER_CORE
            for chunk in range(CHUNKS_PER_CORE):
                inp = f"%{prefix}_in_{'ping' if chunk % 2 == 0 else 'pong'}"
                wire = f"%{prefix}_wire_{'ping' if chunk % 2 == 0 else 'pong'}"
                suffix = 'ping' if chunk % 2 == 0 else 'pong'
                emit(f"  aie.use_lock(%{in_names[f'ready_{suffix}']}, AcquireGreaterEqual, %one)")
                emit(f"  aie.use_lock(%{wire_names[f'free_{suffix}']}, AcquireGreaterEqual, %one)")
                emit(f"  %anchor_{chunk} = arith.constant {core_base + chunk * 15} : i32")
                emit(f"  func.call @dfl_decode_chunk({inp}, {wire}, %anchor_{chunk}) : (memref<{CHUNK_INPUT_BYTES}xi8>, memref<{CHUNK_WIRE_BYTES}xi8>, i32) -> ()")
                emit(f"  aie.use_lock(%{in_names[f'free_{suffix}']}, Release, %one)")
                emit(f"  aie.use_lock(%{wire_names[f'ready_{suffix}']}, Release, %one)")
            emit("  aie.end")
            emit("} {link_files = [\"dfl_decode.o\"]}")

    # Host BOs are byte buffers.  Keeping this ABI byte typed makes DMA
    # offsets and lengths directly match the MemTile wire layout; the host
    # test interprets the resulting bytes as float32 arrays.  A reduced design
    # keeps the full BO sizes and each column's full-design offsets, so the
    # anchors it does not instantiate stay untouched in the host BOs.
    emit(f"aie.runtime_sequence(%x: memref<{INPUT_BYTES}xi8>, %boxes: memref<{BOXES_BYTES}xi8>, %scores: memref<{SCORES_BYTES}xi8>) {{")
    for col in range(cols):
        emit(f"  %in_{col} = aiex.dma_configure_task_for @in_{col} {{")
        emit(f"    aie.dma_bd(%x : memref<{INPUT_BYTES}xi8> offset = {col * COL_INPUT_BYTES} len = {cores_per_col * CORE_INPUT_BYTES})")
        emit("    aie.end")
        emit("  }")
        emit(f"  %box_{col} = aiex.dma_configure_task_for @boxes_{col} {{")
        emit(f"    aie.dma_bd(%boxes : memref<{BOXES_BYTES}xi8> offset = {col * COL_BOX_BYTES} len = {cores_per_col * CORE_BOX_BYTES})")
        emit("    aie.end")
        emit("  } {issue_token = true}")
        emit(f"  %score_{col} = aiex.dma_configure_task_for @scores_{col} {{")
        emit(f"    aie.dma_bd(%scores : memref<{SCORES_BYTES}xi8> offset = {col * COL_SCORE_BYTES} len = {cores_per_col * CORE_SCORE_BYTES})")
        emit("    aie.end")
        emit("  } {issue_token = true}")
        emit(f"  aiex.dma_start_task(%box_{col})")
        emit(f"  aiex.dma_start_task(%score_{col})")
        emit(f"  aiex.dma_start_task(%in_{col})")
    for col in range(cols):
        emit(f"  aiex.dma_await_task(%box_{col})")
        emit(f"  aiex.dma_await_task(%score_{col})")
        emit(f"  aiex.dma_free_task(%in_{col})")
        emit(f"  aiex.dma_free_task(%box_{col})")
        emit(f"  aiex.dma_free_task(%score_{col})")
    emit("}")
    lines.extend(["  }", "}"])
    return "\n".join(lines) + "\n"


def compile_design(directory: Path, cols=COLS, cores_per_col=CORES_PER_COL):
    from aie.utils.compile.utils import compile_cxx_core_function, compile_mlir_module

    directory.mkdir(parents=True, exist_ok=True)
    work = directory / "design.prj"
    work.mkdir(exist_ok=True)
    compile_cxx_core_function(
        str(Path(__file__).with_name("dfl_decode.cc")),
        "aie2",
        str(work / "dfl_decode.o"),
        compile_args=["-O3"],
    )
    ir = module(cols, cores_per_col)
    (directory / "dfl_stage.mlir").write_text(ir, encoding="utf-8")
    compile_mlir_module(
        ir,
        insts_path=directory / "insts.bin",
        xclbin_path=directory / "design.xclbin",
        work_dir=work,
    )
