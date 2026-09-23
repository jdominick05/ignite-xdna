"""The bf16 engine's reference: every output tile computed by the core emulator on windows cut with numpy.

WHY IT IS BIT-EXACT AND WHAT THAT PROVES. A tile here runs the same packets through the same
``engine_bf16_emulator.run_packet`` as a scheduled layer does, chunk by chunk with the same psum, so
the summation order is identical and the result is identical to the bit. What it does NOT share with
the schedule is everything in between: no workspace, no placement, no DMA pattern, no halo fill, no
stream of objects and no drain. Each activation window is sliced from a zero-padded numpy tensor, and
tiles are laid on their own strip grid rather than the schedule's quads. So a schedule that disagrees
with this by a single 16-bit pattern has moved a byte somewhere between DDR and the core - the class
of bug the int8 byte counts invite when they are carried into bf16 (right addresses, wrong lengths).

It does NOT check the packets, which both sides share. The packet layout is checked separately,
against a float64 convolution, in tests/test_engine_schedule_bf16_offline.py - and the numerics
against the model through tools/w8a16_oracle.py.

VALUE CONVENTION. Tensors here are float32 arrays holding bf16-representable values, channel-blocked
to ``blocks * 8`` channels, ``[C][H][W]``. Compare them with ``bf16_bits``, never as floats: -0.0 and
+0.0 compare equal and are different bytes.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ignite_xdna.compiler import engine_bf16_emulator as em
from ignite_xdna.compiler import engine_schedule_bf16 as eb
from ignite_xdna.compiler.engine_schedule import layer_chunks, tile_origins
from ignite_xdna.compiler.graph_ir import ConvLayer, GraphIR

# Zero border around a padded tensor. A window reaches at most its layer's padding into it on the
# rows and columns it USES; a fixed-size activation window reaches further (k5s1 takes 16 x 50 for a
# 9 x 24 footprint) and those positions are never multiplied, so zeros are as good as anything.
MARGIN = 64


def _window(xp: np.ndarray, block0: int, ncin: int, oy: int, ox: int, rows: int, cols: int) -> np.ndarray:
    """``ncin`` blocks of a padded ``[C][H][W]`` tensor as one activation packet, ``[block][row][col][8]``."""
    win = xp[block0 * 8:(block0 + ncin) * 8, MARGIN + oy:MARGIN + oy + rows, MARGIN + ox:MARGIN + ox + cols]
    if win.shape != (ncin * 8, rows, cols):
        raise ValueError(f"window {win.shape} leaves the padded tensor")
    act = np.zeros(em.A_ELEMS, np.float32)
    flat = win.reshape(ncin, 8, rows, cols).transpose(0, 2, 3, 1).reshape(-1)
    act[:flat.size] = flat
    return act


def direct_layer(ir: GraphIR, layer: ConvLayer, tensors: Dict[str, np.ndarray],
                 mac_model: Optional[str] = None) -> np.ndarray:
    """float32 ``[blocks * 8][Ho][Wo]`` of bf16 values: the layer's output, tile by tile."""
    t = ir.tensors[layer.output]
    chunks = layer_chunks(ir, layer)
    padded = {}
    for si, seg in enumerate(layer.inputs):
        src = np.asarray(tensors[seg.tensor], np.float32)[seg.block_offset * 8:(seg.block_offset + seg.blocks) * 8]
        xp = np.zeros((seg.blocks * 8, src.shape[1] + 2 * MARGIN, src.shape[2] + 2 * MARGIN), np.float32)
        xp[:src.shape[0], MARGIN:MARGIN + src.shape[1], MARGIN:MARGIN + src.shape[2]] = src
        padded[si] = xp
    out = np.zeros((t.blocks * 8, t.height, t.width), np.float32)
    for g in range(eb.output_groups(t.blocks)):
        packets = [eb.unpack_w(eb.chunk_packet(layer, g, ch)) for ch in chunks]
        c0 = g * eb.GROUP_CHANNELS
        n = min(eb.GROUP_CHANNELS, t.blocks * 8 - c0)
        for y0 in tile_origins(t.height, em.TILE_ROWS):
            for x0 in tile_origins(t.width, em.TILE_COLS):
                psum = np.zeros(em.PSUM_FLOATS, np.float32)
                scratch = np.zeros(em.SCRATCH_ELEMS, np.float32)
                emitted = None
                for ch, (hdr, bias, wts) in zip(chunks, packets):
                    if ch.kind == "res":
                        seg = layer.residual
                        res = np.asarray(tensors[seg.tensor], np.float32)
                        r0 = (seg.block_offset + g * eb.OUT_BLOCKS_8) * 8
                        tile = np.zeros((eb.GROUP_CHANNELS, em.TILE_ROWS, em.TILE_COLS), np.float32)
                        part = res[r0:r0 + eb.GROUP_CHANNELS, y0:y0 + em.TILE_ROWS, x0:x0 + em.TILE_COLS]
                        tile[:part.shape[0]] = part
                        act = np.zeros(em.A_ELEMS, np.float32)
                        act[:em.OUT_ELEMS] = (tile.reshape(eb.OUT_BLOCKS_8, 8, em.TILE_ROWS, em.TILE_COLS)
                                              .transpose(0, 2, 3, 1).reshape(-1))
                    else:
                        act = _window(padded[ch.seg_index], ch.block_start, int(hdr[em.H_NCIN]),
                                      layer.stride * y0 - layer.pad, layer.stride * x0 - layer.pad,
                                      ch.rows_in, ch.cols_in)
                    o = np.zeros(em.OUT_ELEMS, np.float32)
                    em.run_packet(hdr, act, wts, bias, psum, o, scratch, mac_model=mac_model)
                    if eb.emits(hdr):
                        emitted = o
                if emitted is None:
                    raise AssertionError(f"{layer.name}: no chunk of the round emitted")
                tile = emitted.reshape(eb.OUT_BLOCKS_8, em.TILE_ROWS, em.TILE_COLS, 8).transpose(0, 3, 1, 2)
                out[c0:c0 + n, y0:y0 + em.TILE_ROWS, x0:x0 + em.TILE_COLS] = \
                    tile.reshape(eb.GROUP_CHANNELS, em.TILE_ROWS, em.TILE_COLS)[:n]
    return out


def run_direct(ir: GraphIR, x: np.ndarray, mac_model: Optional[str] = None,
               stop_after: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Every tensor of the graph from the input ``x`` (``[C][H][W]``, real units), chained layer by layer.

    ``x`` is rounded to bf16 and padded to whole blocks with zeros, which is what the runtime must write
    into the input's unused channels (see engine_schedule_bf16's docstring on 0 x NaN).
    """
    eb.check_ir(ir)
    t_in = ir.tensors[ir.input]
    x = np.asarray(x, np.float32)
    xin = np.zeros((t_in.blocks * 8, t_in.height, t_in.width), np.float32)
    xin[:x.shape[0]] = em.to_bf16(x)
    tensors = {ir.input: xin}
    for i, L in enumerate(ir.layers):
        tensors[L.output] = direct_layer(ir, L, tensors, mac_model=mac_model)
        if stop_after is not None and i >= stop_after:
            break
    return tensors
