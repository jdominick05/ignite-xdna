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


def test_x_slot_is_x_then_the_tail_twice():
    x = np.arange(576, dtype=np.uint16)
    slot = pack.x_slot(x, 960)
    assert slot.shape == (pack.X_SLOT_WORDS,)
    obj = slot[:pack.OBJ_WORDS]
    hdr = obj[-pack.HDR_WORDS:]
    assert list(hdr[:5]) == [pack.groups(960), pack.chunks(576), 576, 960, 0]
    assert np.array_equal(obj.view(np.uint16)[:576], x)
    assert np.array_equal(obj[288:], pack.tail(960, 576))       # what a chained call reads from the constant region
    assert not obj[288:-pack.HDR_WORDS].any()
    assert np.array_equal(slot[pack.OBJ_WORDS:], obj)


def test_a_residual_slot_is_x_then_h_and_its_header_names_both():
    x, h = np.arange(1536, dtype=np.uint16), np.arange(576, dtype=np.uint16) + 7
    obj = pack.x_slot(x, 576, h, pack.F_INIT)[:pack.OBJ_WORDS]
    hdr = obj[-pack.HDR_WORDS:]
    assert [hdr[pack.H_G], hdr[pack.H_Q], hdr[pack.H_K], hdr[pack.H_N]] == [3, 8, 1536, 576]
    assert hdr[pack.H_FLAGS] == pack.F_INIT and hdr[pack.H_AUX_OFF] == 1536 and hdr[pack.H_COPY] == 2112
    e = obj.view(np.uint16)
    assert np.array_equal(e[:1536], x) and np.array_equal(e[1536:2112], h)
    with pytest.raises(ValueError):
        pack.x_slot(x, 576, None, pack.F_INIT)                 # F_INIT without h
    with pytest.raises(ValueError):
        pack.tail(1152, 1536, pack.F_INIT)                     # x and h past the core's buffer
    with pytest.raises(ValueError):
        pack.tail(576, 576, 4)                                 # an unknown flag


def test_the_cores_residual_rows_are_pack_core_rows():
    # gemv.cc: base = (core / ROWS) * (N / 4) + g * (ROWS * GROUP) + (core % ROWS) * GROUP, core = c * ROWS + j
    n = 576
    for core in range(pack.CORES):
        for g in range(pack.groups(n)):
            base = (core // pack.ROWS) * (n // 4) + g * (pack.ROWS * 12) + (core % pack.ROWS) * 12
            assert pack.core_rows(n, core // pack.ROWS, core % pack.ROWS, g).start == base


def test_the_norm_scale_is_the_table_at_the_sums_bits():
    from kernels.llm_gemv import reference as R
    assert R.RSQ.dtype == np.uint16 and R.RSQ.size == R.RSQ_SIZE
    r = em.from_bf16_bits(R.RSQ).astype(np.float64)
    assert np.all(np.diff(r) <= 0)                             # 1/sqrt is non-increasing in ss
    rng = np.random.default_rng(5)
    h = em.from_bf16_bits(bf16(rng, 576, 3.0)).astype(np.float32)
    ss = em.bf16_bits(em.from_bf16_bits(oracle.gemv(np.tile(em.bf16_bits(h), (4, 1)), em.bf16_bits(h))[:1]))[0]
    assert R.rscale(h, True) == em.from_bf16_bits(np.uint16(R.RSQ[int(ss) - R.RSQ_BASE]))
    assert R.rscale(np.zeros(576, np.float32), True) == em.from_bf16_bits(np.uint16(R.RSQ[0]))   # clamped low


def _all_finite_bf16():
    u = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    return u[(u & 0x7F80) != 0x7F80]


def test_device_silu_departs_from_correct_rounding_only_at_its_rules():
    from kernels.llm_gemv import reference as R
    u = _all_finite_bf16()
    x = em.from_bf16_bits(u).astype(np.float64)
    got, full = R.silu_bits(u), em.bf16_bits(R.SILU)[u]
    off = got != full
    assert not np.any(off & (np.abs(x) >= 2.0 ** -8) & (x > -32))       # inside the table and x >= 8: correct
    assert np.all(np.abs(em.from_bf16_bits(got[off]).astype(np.float64)
                         - em.from_bf16_bits(full[off]).astype(np.float64)) < 1e-12)
    assert not np.any((got & 0x7F80) == 0) or np.all(got[(got & 0x7F80) == 0] & 0x7FFF == 0)   # no subnormal


def test_device_exp_and_recip_are_correctly_rounded_in_range():
    from kernels.llm_gemv import reference as R
    u = _all_finite_bf16()
    d = u[(u & 0x8000) != 0]
    got, full = R.exp_bits(d), em.bf16_bits(R.EXP)[d]
    normal = (full & 0x7F80) != 0
    xd = em.from_bf16_bits(d).astype(np.float64)
    assert np.array_equal(got[normal & (xd > -128)], full[normal & (xd > -128)])
    assert not np.any(got[~normal] & 0x7FFF)                               # underflow is +0, never subnormal
    z = u[(u >= 0x3F80) & (u < 0x5F80)]
    assert np.array_equal(R.recip_bits(z), em.bf16_bits(R.RECIP)[z])     # the mantissa table loses nothing


def test_exact_rope_matches_the_fast_rope_but_for_rare_last_bits():
    from kernels.llm_gemv import reference as R
    rng = np.random.default_rng(8)
    x = R.bf16(rng.normal(0, 2, (12, 64)))
    cos, sin = (t[37] for t in R.rope_tables(64))
    a, b = R.rope(x, cos, sin, True), R.rope(x, cos, sin, False)
    assert np.count_nonzero(a != b) <= 8 and np.max(np.abs(a - b)) <= 2 ** -6 * np.max(np.abs(b))


def test_a_residual_add_is_the_accumulators_start():
    from kernels.llm_gemv import reference as R
    rng = np.random.default_rng(6)
    w, x, h = bf16(rng, (576, 576), 0.05), bf16(rng, 576), bf16(rng, 576)
    f = lambda b: em.from_bf16_bits(b).astype(np.float32)  # noqa: E731
    y = em.bf16_bits(R.matvec(f(w), f(x), True, init=f(h)))
    assert np.array_equal(y, oracle.gemv(w, x, init=h))
    assert not np.array_equal(y, oracle.gemv(w, x))


def test_a_chained_residual_call_reads_x_and_h_from_out_within_16_bds():
    from kernels.llm_gemv import design, patch
    n, k = 576, 1536
    call = design.Call(0, 0, 1000, n, k, x_out=0, flags=pack.F_INIT, h_out=768)
    tail = call.header().size
    text = design.harness([call], pack.block_words(n, k), tail, 2000)
    assert text.count("%o[0,0,0,0][1,1,1,768]") == 16 and text.count("%o[0,0,0,768][1,1,1,288]") == 16
    assert text.count(f"%x[0,0,0,0][1,1,1,{tail}]") == 16
    ids = [int(i) for i in __import__("re").findall(r"id = (\d+) : i64", text)]
    assert max(ids) == 15 and len(set(ids)) == 15
    roles = [(e[0], e[1]) for e in patch._expected(call)[:7]]
    assert roles == [("o", "x"), ("o", "h"), ("x", "tail")] * 2 + [("w", "w")]
    with pytest.raises(ValueError):
        design.Call(0, 0, 1000, n, k, x_out=0, flags=pack.F_INIT)            # no h source


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


def test_patch_finds_every_transfer_of_the_compiled_chain_and_moves_only_what_it_names():
    sys.path.insert(0, str(ROOT / "tools"))
    import llm_gemv_probe as P
    from kernels.llm_gemv import design, patch
    insts = P.BUILD / "chain" / "insts.bin"
    if not insts.exists():
        pytest.skip("build/llm_gemv_v2/chain is not compiled (tools/llm_gemv_probe.py compile)")
    calls = P.chain_calls(design.Call)
    raw = insts.read_bytes()
    sites = patch.calls(raw, calls)
    assert [len(s) for s in sites] == [20] + [44] * (len(calls) - 1)
    changes = {**patch.move_output(sites[-2], calls[-2], P.OUT["o"]), **patch.move_x(sites[-1], P.OUT["o"])}
    assert len(changes) == 4 + 16
    new = patch.apply(raw, changes)
    moved = patch.calls(new, calls[:-2] + [design.Call(calls[-2].w_offset, calls[-2].x_offset, P.OUT["o"],
                                                       calls[-2].n, calls[-2].k, calls[-2].x_out),
                                           design.Call(calls[-1].w_offset, calls[-1].x_offset, calls[-1].out_offset,
                                                       calls[-1].n, calls[-1].k, P.OUT["o"])])
    assert len(moved) == len(calls)                       # the patched stream is exactly the moved program
    with pytest.raises(ValueError):
        patch.calls(raw, calls[:-1])                      # a stream with transfers past its program is refused


def test_a_chained_call_reads_x_from_the_out_argument_and_stays_within_16_bds():
    from kernels.llm_gemv import design
    n = 576
    calls = [design.Call(0, 0, 0, n, n),                                     # host x
             design.Call(pack.block_words(n, n), pack.X_SLOT_WORDS, 288, n, n, x_out=0)]   # x = call 0's output
    text = design.harness(calls, 2 * pack.block_words(n, n), pack.X_SLOT_WORDS + pack.OBJ_WORDS, 576)
    second = text.split("aiex.npu.dma_wait")[4]
    assert second.count("%o[0,0,0,0][1,1,1,288]") == 16         # x from out: 8 channels x the 2 cores each feeds
    assert second.count(f"%x[0,0,0,{pack.X_SLOT_WORDS}][1,1,1,{pack.OBJ_WORDS - 288}]") == 16
    ids = {int(i) for i in __import__("re").findall(r"id = (\d+) : i64", text)}
    assert max(ids) < 16
    with pytest.raises(ValueError):                              # chained x past the out argument
        design.harness([design.Call(0, 0, 0, n, n, x_out=400)], pack.block_words(n, n), pack.OBJ_WORDS, 576)
