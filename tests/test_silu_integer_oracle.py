"""Offline checks of tools/silu_integer_oracle.py: its HardSigmoid mode is the compiler's epilogue, and its K-line form
is the same integer function whether written as t * qh (the ONNX table) or as the core expression (64 t + |t| g)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ignite_xdna.compiler.graph_ir import fit_hardswish, hardswish_float_table

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("silu_integer_oracle", ROOT / "tools" / "silu_integer_oracle.py")
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)

K_DPU = 1.0001220703125
PAIRS = [(1 / 16, 1 / 32), (1 / 16, 1 / 16), (1 / 8, 1 / 16), (1 / 4, 1 / 4), (1 / 2, 1 / 2)]


@pytest.mark.parametrize("s1,s2", PAIRS)
def test_hardsigmoid_mode_is_the_compilers_epilogue(s1, s2):
    assert fit_hardswish(s1, s2, K_DPU).max_error == 0
    y = oracle.y_from_qh(oracle.qh_hardsigmoid(s1, s2, K_DPU), s1, s2)
    assert np.array_equal(y + 128, hardswish_float_table(s1, s2, K_DPU).astype(np.int64))


@pytest.mark.parametrize("s1,s2", PAIRS[:3])
@pytest.mark.parametrize("K", [3, 4])
def test_pl_table_equals_core_expression_and_beats_hardsigmoid(s1, s2, K):
    lines, S, (max_err, mean_err) = oracle.fit_pl(s1, s2, K)
    # int16 multipliers; a slope may round to 0 (a flat line above the cap, e.g. line 4 at s1 1/8)
    assert len(lines) == K and all(0 <= a < 1 << 15 for a, _ in lines)
    qh = oracle.qh_pl(lines, S)
    assert qh.min() >= 0 and qh.max() <= 128
    y_core = oracle.y_pl(lines, S, s1, s2)
    assert np.array_equal(oracle.y_from_qh(qh, s1, s2), y_core)
    x = (oracle.T.astype(np.float32) * np.float32(s1)).astype(np.float32)
    y_float = np.clip(np.round((x * (qh.astype(np.float32) / np.float32(128))).astype(np.float64) / s2), -128, 127)
    assert np.array_equal(y_float.astype(np.int64), y_core)
    exact = oracle.silu_exact_y(s1, s2)
    hs_max, hs_mean = oracle.score(oracle.y_from_qh(oracle.qh_hardsigmoid(s1, s2, K_DPU), s1, s2), exact)
    assert max_err <= hs_max and mean_err < hs_mean
