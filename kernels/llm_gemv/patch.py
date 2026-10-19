"""Move a compiled program's DDR addresses between dispatches: the per-position half of a decode step.

A decode step at position p appends k and v at cache row p, reads RoPE's row p and embeds one token. Each of those is
a DDR address in the instruction stream, not data, so the step patches it in the cached instruction buffer, the way
kernels/laya_transformer/native.py patches token addresses. A runtime-sequence transfer lowers to a ``DDR_PATCH``
operation whose word 10 is the byte offset into its argument; rewriting that word moves the transfer and nothing else.

Operations are found by their place in the program, never by searching values: ``calls`` splits the stream's
DDR_PATCH operations call by call in the order design.harness emits them, and a patch names (call, role, column,
channel). The counts per call are checked against the program, so a stream that does not match its program is refused.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Sequence

from ignite_xdna.compiler import scheduler
from kernels.llm_gemv import pack

ARG = {"w": 0, "x": 1, "o": 2}
VALUE_WORD = 10                      # DDR_PATCH word holding the argument offset in bytes


@dataclass(frozen=True)
class Site:
    value_offset: int    # byte offset of the arg_offset field in the instruction stream
    arg: str
    arg_offset: int      # bytes, as compiled
    role: str            # "x", "tail", "w" or "out"
    column: int
    channel: int         # 0 or 1 for MM2S roles; -1 for "out"
    copy: int            # which core of the pair an x/tail transfer feeds (0 or 1); 0 otherwise


def _expected(call) -> List[tuple]:
    """(arg, role, column, channel, copy, word offset) in design.harness's emission order for one call."""
    rows = []
    sw = pack.stream_elems(call.n, call.k) // 2
    ow = call.n // 2 // pack.COLS
    for c in range(pack.COLS):
        for h in range(2):
            if call.x_out is None:
                rows.append(("x", "x", c, h, 0, call.x_offset))
            else:
                for e in range(2):
                    rows += [("o", "x", c, h, e, call.x_out), ("x", "tail", c, h, e, call.x_offset)]
            rows.append(("w", "w", c, h, 0, call.w_offset + (c * 2 + h) * sw))
        rows.append(("o", "out", c, -1, 0, call.out_offset + c * ow))
    return rows


def calls(insts: bytes, program: Sequence) -> List[List[Site]]:
    """Every DDR_PATCH site of a compiled program, grouped by call; refused unless it matches the program exactly."""
    ops = [op for op in scheduler.parse_transaction_stream(insts) if op["op"] == "DDR_PATCH"]
    out, i = [], 0
    for n, call in enumerate(program):
        sites = []
        for arg, role, c, h, e, words in _expected(call):
            if i >= len(ops):
                raise ValueError(f"call {n}: the stream has fewer transfers than the program")
            op = ops[i]
            i += 1
            if op["arg_idx"] != ARG[arg] or op["arg_offset"] != words * 4:
                raise ValueError(f"call {n} {role} col {c} ch {h}: stream has arg {op['arg_idx']} offset "
                                 f"{op['arg_offset']}, program says {ARG[arg]} offset {words * 4}")
            off = op["offset"] + 4 * VALUE_WORD
            if struct.unpack_from("<I", insts, off)[0] != op["arg_offset"]:
                raise ValueError("decoder and value field disagree")
            sites.append(Site(off, arg, op["arg_offset"], role, c, h, e))
        out.append(sites)
    if i != len(ops):
        raise ValueError(f"the stream has {len(ops) - i} transfers past the program")
    return out


def apply(insts: bytes, changes: Dict[int, int]) -> bytes:
    """A copy of the stream with each {value_offset: new byte offset} written; re-parsed before it is returned."""
    out = bytearray(insts)
    for off, value in changes.items():
        if value < 0 or value % 4:
            raise ValueError(f"offset {value} is not a non-negative multiple of 4")
        struct.pack_into("<I", out, off, value)
    scheduler.parse_transaction_stream(bytes(out))
    return bytes(out)


def move_output(sites: List[Site], call, out_words: int) -> Dict[int, int]:
    """Changes that make a call write its output at ``out_words`` instead of where it was compiled to."""
    ow = call.n // 2 // pack.COLS
    return {s.value_offset: (out_words + s.column * ow) * 4 for s in sites if s.role == "out"}


def move_x(sites: List[Site], from_out_words: int) -> Dict[int, int]:
    """Changes that make a chained call read its x from ``from_out_words`` of the out argument."""
    found = [s for s in sites if s.role == "x" and s.arg == "o"]
    if len(found) != 2 * pack.CHANNELS:
        raise ValueError("not a chained call: its x does not come from the out argument")
    return {s.value_offset: from_out_words * 4 for s in found}
