"""Offline checks for the whole-array M=1 GEMV (kernels/llm_gemv): layout, oracle and program text. No device."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ignite_xdna.compiler import engine_bf16_emulator as em  # noqa: E402
from kernels.llm_gemv import oracle, pack  # noqa: E402


def bf16(rng, shape, scale=1.0):
    return em.bf16_bits(em.to_bf16(rng.normal(0, scale, shape).astype(np.float32)))


def run_streams(block: np.ndarray, x: np.ndarray, n: int, k: int) -> np.ndarray:
    """Read the packed block the way the 16 cores do (gemv.cc), with the emulator's MAC: y as uint16 bits."""
    halves = block.view(np.uint16)
    sw = pack.stream_elems(n, k)
    xf = em.from_bf16_bits(x)
    y = np.zeros(n, np.uint16)
    for c in range(pack.COLS):
        for h in range(2):
            stream = halves[(c * 2 + h) * sw:(c * 2 + h + 1) * sw].reshape(-1, pack.OBJ_ELEMS)
            acc = {e: None for e in range(2)}
            pos = 0
            for g in range(pack.groups(n)):
                for q in range(pack.chunks(k)):
                    for e in range(2):
                        obj = em.from_bf16_bits(stream[pos]).reshape(pack.KCHUNK // 8, 3, 8, 4)
                        pos += 1
                        a = np.zeros((3, 1, 1, 4), np.float32) if q == 0 else acc[e]
                        for s in range(pack.KCHUNK // 8):
                            row = xf[q * pack.KCHUNK + s * 8:q * pack.KCHUNK + s * 8 + 8].reshape(1, 1, 8)
                            a = em.mac(a, row, obj[s], em.MAC_MODEL)
                        acc[e] = a
                for e in range(2):
                    y[pack.core_rows(n, c, 2 * h + e, g)] = em.bf16_bits(em.to_bf16(acc[e].reshape(12)))
    return y


@pytest.mark.parametrize("n,k", [(192, 192), (576, 576), (384, 1536), (960, 576)])
def test_packed_streams_equal_the_oracle(n, k):
    rng = np.random.default_rng(n * 7 + k)
    w, x = bf16(rng, (n, k), 0.05), bf16(rng, k)
    block = pack.pack(w)
    assert block.shape == (pack.block_words(n, k),)
    assert np.array_equal(run_streams(block, x, n, k), oracle.gemv(w, x))


def test_oracle_sits_near_float64():
    rng = np.random.default_rng(3)
    w, x = bf16(rng, (576, 576), 0.05), bf16(rng, 576)
    y = em.from_bf16_bits(oracle.gemv(w, x)).astype(np.float64)
    ref = oracle.reference_fp64(w, x)
    assert np.max(np.abs(y - ref) / (np.abs(ref) + 1e-3)) < 2 ** -7


def test_x_slot_header_and_duplicate():
    x = np.arange(576, dtype=np.uint16)
    slot = pack.x_slot(x, 960)
    assert slot.shape == (pack.X_SLOT_WORDS,)
    obj = slot[:pack.OBJ_WORDS]
    assert list(obj[:3]) == [pack.groups(960), pack.chunks(576), 576]
    assert np.array_equal(obj.view(np.uint16)[pack.X_OFFSET_ELEMS:pack.X_OFFSET_ELEMS + 576], x)
    assert np.array_equal(slot[pack.OBJ_WORDS:], obj)


@pytest.mark.parametrize("n,k", [(100, 576), (576, 100), (576, 2304), (0, 576)])
def test_shapes_outside_the_geometry_are_refused(n, k):
    with pytest.raises(ValueError):
        pack.check_shape(n, k)


def test_smollm2_shapes_fit_without_padding():
    for n, k in [(960, 576), (576, 576), (3072, 576), (576, 1536), (49152, 576)]:
        pack.check_shape(n, k)
        assert pack.block_words(n, k) * 2 == n * k


def test_program_text_covers_every_call():
    from kernels.llm_gemv import design
    calls = [design.Call(0, 0, 0, 960, 576), design.Call(pack.block_words(960, 576), pack.X_SLOT_WORDS, 480, 576, 1536)]
    w_words = sum(pack.block_words(c.n, c.k) for c in calls)
    text = design.harness(calls, w_words, 2 * pack.X_SLOT_WORDS, 480 + 288)
    assert text.count("aiex.npu.dma_wait") == 4 * len(calls)
    assert text.count("aie.core(") == pack.CORES
    with pytest.raises(ValueError):
        design.harness(calls, w_words - 1, 2 * pack.X_SLOT_WORDS, 768)
