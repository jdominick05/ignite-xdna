"""SmolLM2-135M greedy decode with every neural op on the XDNA1 NPU: one dispatch per token.

The token is one instruction stream of 181 calls of the whole-array kernel (design.py): per layer the six calls
tools/llm_layer_probe.py checks (qkv+norm+RoPE with the KV append, scores, softmax V, o+residual,
gate/up+norm+SiLU, down+residual), then the tied head with the final norm. Every input after the token's embedding
is DMA'd from an earlier call's output. The KV cache is the device's: CAP positions per layer, kept in the out
argument between tokens.

Per token the host writes the embedding row (h), the position's cos/sin row and p into the shared tails, patches
the 30 KV appends' destinations in the cached instruction stream (patch.py), dispatches once and reads the 49,152
logits. Greedy selection is np.argmax on those logits (the device's bf16 values, first index on ties).

Regions are shared by every layer: a layer reads h from H and writes its output back to H once nothing reads the
old h any more (the o call took it as its residual before the down call runs).

With LLM_DECODE_W8=1 in the environment when this module is imported, the projections and the head take W8A16
weights (reference.Q8 of the gamma-folded matrices, pack.pack8): qkv pads to 1,152 rows (the padding to a junk
region), o and down to 768 on 3 columns; the head stays one call, its scales overflowing into two extra lead
objects (pack.xobj8). Each W8 call's scales and header sit in the weight arena; the x argument holds only the cos/sin
row and the attention tails. The decoder then expects reference.quantize_weights(w, "w8a16-dev") as its weights.
181 calls a token, as in bf16.
"""
from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from kernels.llm_gemv import pack
from kernels.llm_gemv import reference as R

CAP = 576
HIDDEN, MLP, KV, QK, VOCAB, LAYERS = R.HIDDEN, R.MLP, 192, 768, R.VOCAB, R.LAYERS
F = pack
SPEC = {"qkv_r": (960, HIDDEN, F.F_NORM | F.F_ROPE), "o": (HIDDEN, HIDDEN, F.F_INIT),
        "gu_s": (MLP, HIDDEN, F.F_NORM | F.F_SILU), "down": (HIDDEN, MLP, F.F_INIT)}
HEAD = (VOCAB, HIDDEN, F.F_NORM)
CACHE_WORDS = CAP * KV // 2
W8 = os.environ.get("LLM_DECODE_W8") == "1"
# W8: name -> (padded N, columns)
W8_SHAPE = {"qkv_r": (1152, 4), "o": (768, 3), "gu_s": (MLP, 4), "down": (768, 3)}
JUNK = 192                                   # W8: qkv's padding rows land here


def _w8_call(w_off, x_off, out_off, name, **kw):
    """A W8 Call of block ``name`` (W8_SHAPE, or "head")."""
    from kernels.llm_gemv.design import Call
    if name == "head":
        (n, k, f), cols = HEAD, 4
    else:
        (n, cols), (_, k, f) = W8_SHAPE[name], SPEC[name]
    return Call(w_off, x_off, out_off, n, k, flags=f, w8=True, cols=cols, **kw)


def _w8_words(name):
    kw = dict(x_out=0, h_out=0 if (name in SPEC and SPEC[name][2] & F.F_INIT) else None,
              rope_end=QK if name == "qkv_r" else 0)
    return _w8_call(0, 0, 0, name, **kw).weight_words()


def _layout():
    w, at = {}, 0
    for i in range(LAYERS):
        for s, (n, k, f) in SPEC.items():
            w[(i, s)] = at
            at += _w8_words(s) if W8 else pack.weight_words(n, k, f)
    w["head"] = at
    at += _w8_words("head") if W8 else pack.weight_words(*HEAD)
    o, ot = {}, 0
    for name, n in [("h", HIDDEN), ("q", HIDDEN), ("s", pack.KV_COLS * CAP * pack.GQA), ("a", HIDDEN),
                    ("h1", HIDDEN), ("m", MLP), ("logits", VOCAB)] + ([("junk", JUNK)] if W8 else []):
        o[name] = ot
        ot += n // 2
    for i in range(LAYERS):
        o[("kc", i)], o[("vc", i)] = ot, ot + CACHE_WORDS
        ot += 2 * CACHE_WORDS
    return w, at, o, ot


W_OFF, W_WORDS, OUT, OUT_WORDS = _layout()


def program():
    """(calls, x_words): the token's calls with the shared tails at their x offsets."""
    from kernels.llm_gemv.design import Attn, Call
    tails = {}
    xw = 0

    def tail(key, size):
        nonlocal xw
        if key not in tails:
            tails[key] = xw
            xw += size
        return tails[key]

    calls = []
    if W8:
        cs_at = tail("cs", pack.CS_ELEMS // 2)
        for i in range(LAYERS):
            kv0 = ((0, HIDDEN, OUT["q"]), (HIDDEN, KV, OUT[("kc", i)]), (QK, KV, OUT[("vc", i)]),
                   (960, JUNK, OUT["junk"]))
            layer = [_w8_call(W_OFF[(i, "qkv_r")], cs_at, 0, "qkv_r", x_out=OUT["h"], rope_end=QK, out_map=kv0),
                     Attn(F.F_SCORE, CAP, OUT["q"], 0, OUT[("kc", i)], OUT["s"]),
                     Attn(F.F_PV, CAP, OUT["s"], 0, OUT[("vc", i)], OUT["a"]),
                     _w8_call(W_OFF[(i, "o")], 0, OUT["h1"], "o", x_out=OUT["a"], h_out=OUT["h"]),
                     _w8_call(W_OFF[(i, "gu_s")], 0, OUT["m"], "gu_s", x_out=OUT["h1"]),
                     _w8_call(W_OFF[(i, "down")], 0, OUT["h"], "down", x_out=OUT["m"], h_out=OUT["h1"])]
            for j, c in enumerate(layer):
                if isinstance(c, Attn):
                    c = dataclasses.replace(c, x_offset=tail(j, pack.OBJ_WORDS - c.lead_words()))
                calls.append(c)
        calls.append(_w8_call(W_OFF["head"], 0, OUT["logits"], "head", x_out=OUT["h"]))
        return calls, xw, tails
    for i in range(LAYERS):
        kv0 = ((0, HIDDEN, OUT["q"]), (HIDDEN, KV, OUT[("kc", i)]), (QK, KV, OUT[("vc", i)]))
        layer = [Call(W_OFF[(i, "qkv_r")], 0, 0, 960, HIDDEN, x_out=OUT["h"], flags=SPEC["qkv_r"][2], rope_end=QK,
                      out_map=kv0),
                 Attn(F.F_SCORE, CAP, OUT["q"], 0, OUT[("kc", i)], OUT["s"]),
                 Attn(F.F_PV, CAP, OUT["s"], 0, OUT[("vc", i)], OUT["a"]),
                 Call(W_OFF[(i, "o")], 0, OUT["h1"], HIDDEN, HIDDEN, x_out=OUT["a"], flags=F.F_INIT, h_out=OUT["h"]),
                 Call(W_OFF[(i, "gu_s")], 0, OUT["m"], MLP, HIDDEN, x_out=OUT["h1"], flags=SPEC["gu_s"][2]),
                 Call(W_OFF[(i, "down")], 0, OUT["h"], HIDDEN, MLP, x_out=OUT["m"], flags=F.F_INIT,
                      h_out=OUT["h1"])]
        for j, c in enumerate(layer):
            size = (pack.OBJ_WORDS - c.lead_words()) if isinstance(c, Attn) else c.header().size
            calls.append(dataclasses.replace(c, x_offset=tail(j, size)))
    head = Call(W_OFF["head"], 0, OUT["logits"], *HEAD[:2], x_out=OUT["h"], flags=HEAD[2])
    calls.append(dataclasses.replace(head, x_offset=tail("head", head.header().size)))
    return calls, xw, tails


CALLS, X_WORDS, TAILS = program()


def x_arg(p: int) -> np.ndarray:
    """The shared tails for position p: cos/sin row p in qkv_r's, p in pv's."""
    from kernels.llm_gemv.design import Attn
    cos, sin = R.rope_tables(p + 1)
    cs = R.em.bf16_bits(np.concatenate([cos[p, :32], sin[p, :32]]).astype(np.float32))
    out = np.zeros(X_WORDS, np.uint32)
    for c in CALLS[:6] + CALLS[-1:]:
        if isinstance(c, Attn):
            t = pack.attn_tail(c.flags, CAP, p if c.flags == F.F_PV else 0)
        elif c.w8:
            if not c.flags & F.F_ROPE:
                continue
            t = cs.view(np.uint32)
        else:
            t = c.header(cs if c.flags & F.F_ROPE else None)
        out[c.x_offset:c.x_offset + t.size] = t
    return out


def _w8_block(name, *qs) -> np.ndarray:
    """Q8 matrices (gate, up for gu_s) -> the call's block and channel tails, rows zero-padded to its N."""
    call = _w8_call(0, 0, 0, name, x_out=0, h_out=0 if (name in SPEC and SPEC[name][2] & F.F_INIT) else None,
                    rope_end=QK if name == "qkv_r" else 0)
    codes, scales = [], []
    for q in qs:
        c = np.zeros((call.n, call.k), np.int8)
        c[:q.codes.shape[0]] = q.codes
        sc = np.zeros(call.n, np.uint16)
        sc[:q.s.shape[0]] = R.em.bf16_bits(np.asarray(q.s, np.float32))
        codes.append(c), scales.append(sc)
    out = np.concatenate([pack.pack8(*codes, cols=call.cols),
                          pack.tails8(call.n, call.k, call.flags, *scales, rope_end=call.rope_end, cols=call.cols)])
    assert out.size == call.weight_words()
    return out


def _rows(q: "R.Q8", rows) -> "R.Q8":
    return R.Q8(q.codes[rows], q.s[rows])


def arena(w: "R.Weights") -> np.ndarray:
    """uint32 [W_WORDS]: every layer's four blocks and the head, packed once (Q8 matrices under W8)."""
    bits = lambda a: R.em.bf16_bits(np.asarray(a, np.float32))  # noqa: E731
    out = np.zeros(W_WORDS, np.uint32)
    if W8:
        perm_qkv, perm_o = pack.rope_perm(960, QK), pack.attn_out_perm()
        for i, L in enumerate(w.layers):
            gu = L["gu_g"]
            blocks = {"qkv_r": _w8_block("qkv_r", _rows(L["qkv_g"], perm_qkv)), "o": _w8_block("o", L["o"][:, perm_o]),
                      "gu_s": _w8_block("gu_s", _rows(gu, slice(0, MLP)), _rows(gu, slice(MLP, 2 * MLP))),
                      "down": _w8_block("down", L["down"])}
            for s, b in blocks.items():
                out[W_OFF[(i, s)]:W_OFF[(i, s)] + b.size] = b
        b = _w8_block("head", w.head_g)
        out[W_OFF["head"]:W_OFF["head"] + b.size] = b
        return out
    perm_qkv, perm_o = pack.rope_perm(960, QK), pack.attn_out_perm()
    for i, L in enumerate(w.layers):
        blocks = {"qkv_r": pack.pack(bits(L["qkv_g"][perm_qkv])), "o": pack.pack(bits(L["o"][:, perm_o])),
                  "gu_s": pack.pack_silu(bits(L["gu_g"][:MLP]), bits(L["gu_g"][MLP:])),
                  "down": pack.pack(bits(L["down"]))}
        for s, b in blocks.items():
            out[W_OFF[(i, s)]:W_OFF[(i, s)] + b.size] = b
    head = pack.pack(bits(w.head_g))
    out[W_OFF["head"]:W_OFF["head"] + head.size] = head
    return out


class Patcher:
    """The cached stream for any position: the 30 KV appends moved to cache row p."""

    def __init__(self, raw: bytes):
        from kernels.llm_gemv import patch
        self.raw, self.patch = raw, patch
        sites = patch.calls(raw, CALLS)
        self.row0 = {}                     # {byte offset in the stream: argument offset in bytes at row 0}
        for n in range(0, LAYERS * 6, 6):            # the head's calls follow the 30 layers
            for piece, region in ((1, OUT[("kc", n // 6)]), (2, OUT[("vc", n // 6)])):
                self.row0.update(patch.move_piece(sites[n], CALLS[n], piece, region))

    def changes(self, p: int) -> dict:
        """{byte offset in the stream: new argument offset} that move the KV-append transfers to row p."""
        if not 0 <= p < CAP:
            raise ValueError(f"position {p} is past the cache's {CAP}")
        return {off: v + p * KV * 2 for off, v in self.row0.items()}

    def stream(self, p: int) -> bytes:
        """The whole stream for position p, re-parsed (a check; the decoder rewrites only the changed words)."""
        return self.patch.apply(self.raw, self.changes(p))


class NpuDecoder:
    """Greedy decode on the device. Open with a loaded harness; step() is one dispatch."""

    WAIT_MS = 10000

    def __init__(self, harness, folder: Path, w: "R.Weights"):
        h = self.h = harness
        h.load_xclbin(str(Path(folder) / "final.xclbin"))
        xrt = self.xrt = h.pyxrt
        self.to = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.fr = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.embed = R.em.bf16_bits(np.asarray(w.embed, np.float32))
        self.wb = h.create_host_bo(W_WORDS * 4, 3)
        np.frombuffer(self.wb.map(), np.uint32)[:] = arena(w)
        self.wb.sync(self.to)
        self.xb, self.ob = h.create_host_bo(X_WORDS * 4, 4), h.create_host_bo(OUT_WORDS * 4, 5)
        self.patcher = Patcher((Path(folder) / "insts.bin").read_bytes())
        self.patcher.stream(CAP - 1)                      # the patch sites parse at the far end of the cache
        self.insts = h.create_instruction_bo_from_bytes(self.patcher.raw)
        self.at = None
        # x tails per position: only the cos/sin row and p change
        self.tails = {}

    def close(self):
        self.wb = self.xb = self.ob = self.insts = None

    def _move(self, p: int):
        """Rewrite the 60 KV-append offsets of the cached instruction buffer in place for position p."""
        if self.at == p:
            return
        bo, _ = self.insts
        for off, value in self.patcher.changes(p).items():
            bo.write(int(value).to_bytes(4, "little"), off)
        bo.sync(self.to)
        self.at = p

    def step(self, token: int, p: int) -> np.ndarray:
        """Embed ``token`` at position p, run the 30 layers and the head: uint16 bf16 logits [VOCAB]."""
        if p not in self.tails:
            self.tails[p] = x_arg(p).tobytes()
        self.xb.write(self.tails[p], 0)
        self.xb.sync(self.to)
        self.ob.write(self.embed[token].view(np.uint32).tobytes(), OUT["h"] * 4)
        self.ob.sync(self.to, HIDDEN * 2, OUT["h"] * 4)    # only h: the cache and the rest are the device's
        self._move(p)
        bo, count = self.insts
        r = self.xrt.run(self.h.kernel)
        for i, v in enumerate((3, bo, count, self.wb, self.xb, self.ob)):
            r.set_arg(i, v)
        t0 = time.perf_counter()
        r.start()
        state = r.wait(self.WAIT_MS)
        self.device_s = time.perf_counter() - t0          # start to completion, as the host sees it
        if state != self.xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"position {p} did not complete: {state}")
        r = None
        self.ob.sync(self.fr, VOCAB * 2, OUT["logits"] * 4)
        return np.frombuffer(self.ob.read(VOCAB * 2, OUT["logits"] * 4), np.uint16).copy()

    def generate(self, prompt: List[int], new_tokens: int, eos: Optional[int] = None, on_token=None) -> List[int]:
        logits, out = None, []
        for p, t in enumerate(prompt):
            logits = self.step(t, p)
        for _ in range(new_tokens):
            t = int(np.argmax(R.em.from_bf16_bits(logits)))
            out.append(t)
            if on_token is not None:
                on_token(t)
            if (eos is not None and t == eos) or len(prompt) + len(out) > CAP:
                break
            logits = self.step(t, len(prompt) + len(out) - 1)
        return out
