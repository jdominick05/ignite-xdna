"""Identify observable XINT8 Conv arithmetic with exact, discriminating fixtures.

This is a compiler/runtime/NPU-stack probe, not a new production quantizer preset.
All model changes require --fresh. Output codes, independent references and the
EP report are retained alongside each synthetic model in a private run directory.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npu.ep_report import read_report
from npu.paths import CACHE_DIR, RESNET_CACHE_KEY
from npu.session import build_session, clear_cache, resolve_xclbin
from quant.graph import Graph
from quant.pow2 import TensorQ
from quant.qdq import emit
from quant.quantize import file_hash

ROUNDERS = {
    'even': np.rint,
    'half_up': lambda x: np.floor(x + .5),
    'away': lambda x: np.sign(x) * np.floor(np.abs(x)+.5),
    'trunc': np.trunc,
    'floor': np.floor,
}


def parameters(channels, reverse):
    w = np.zeros((32, channels), dtype=np.int64)
    for out in range(32):
        case = out % 8
        if case == 0:
            w[out, 0] = 1
        elif case == 1:
            w[out, 0] = -1
        elif case == 2:
            w[out] = 1
        elif case == 3:
            w[out] = -1
        elif case == 4:
            w[out] = np.where(np.arange(channels) % 2, -3, 3)
        elif case == 5:
            w[out] = 127
        elif case == 6:
            w[out] = -127
        else:
            w[out, -1] = 1
    b = np.repeat(np.array([0, 1, -1, 127], dtype=np.int64), 8)
    if reverse:
        w = w[:, ::-1].copy()
    return w, b


def model(channels, sc, sb, reverse):
    weights, bias = parameters(channels, reverse)
    inputs = [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, channels, 16, 16])]
    outputs = [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 32, 16, 16])]
    node = helper.make_node('Conv', ['input', 'weight', 'bias'], ['output'], name='probe_conv',
                            kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])
    graph = helper.make_graph([node], 'exact_arithmetic_probe', inputs, outputs,
        [numpy_helper.from_array(weights.astype(np.float32).reshape(32, channels, 1, 1), 'weight'),
         numpy_helper.from_array((bias * 2.**sb).astype(np.float32), 'bias')])
    g = Graph(helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 17)]))
    g.infer_shapes()
    positions = {name: TensorQ(name, dtype, pos, zp, 'exact_fixture') for name, dtype, pos, zp in
                 [('input', 'uint8', 0, 128), ('weight', 'int8', 0, 0),
                  ('bias', 'int8', -sb, 0), ('output', 'uint8', -sc, 128)]}
    emit(g, positions)
    g.model.producer_name = 'Ignition'
    g.model.producer_version = '0.1.0a1'
    return g, weights, bias


def fixtures(channels, reverse):
    ramp = np.arange(-128, 128, dtype=np.float32).reshape(1, 1, 16, 16)
    cases = {'fit_ramp': np.repeat(ramp, channels, axis=1),
             'fit_zero_input_bias': np.zeros((1, channels, 16, 16), np.float32)}
    # Exact half ties and the adjacent float32 values isolate input Q rounding.
    ties = np.arange(-128, 128, dtype=np.float32) + np.float32(.5)
    for label, vals in [('fit_input_ties', ties),
                        ('holdout_below_ties', np.nextafter(ties, np.float32(-np.inf))),
                        ('holdout_above_ties', np.nextafter(ties, np.float32(np.inf)))]:
        cases[label] = np.repeat(vals.reshape(1, 1, 16, 16), channels, axis=1)
    phase = (np.arange(channels)[:, None] * 17 + np.arange(256)[None, :]) % 256 - 128
    cases['holdout_channel_phase'] = phase.astype(np.float32).reshape(1, channels, 16, 16)
    cancel = np.repeat(ramp, channels, axis=1)
    cancel[:, 1::2] = -cancel[:, 1::2] - 1
    cases['holdout_cancellation'] = cancel
    if reverse:
        cases = {k: v[:, ::-1].copy() for k, v in cases.items()}
    return cases


def predict(x, weights, bias, sc, sb, output_round='even', bias_round='exact', input_round='even'):
    codes = np.clip(ROUNDERS[input_round](x.astype(np.float64)), -128, 127).astype(np.int64)
    accumulator = weights @ codes.reshape(weights.shape[1], -1)
    b = bias.astype(np.float64) * 2.**sb
    if bias_round != 'exact':
        b = ROUNDERS[bias_round](b)
    scaled = (accumulator + b[:, None]) / 2.**sc
    return np.clip(ROUNDERS[output_round](scaled)+128, 0, 255).astype(np.uint8).reshape(1, 32, 16, 16)


def output_codes(y, sc):
    scaled = y.astype(np.float64) / 2.**sc + 128
    if not np.all(np.isfinite(scaled)) or np.any(scaled != np.rint(scaled)) or np.any((scaled < 0) | (scaled > 255)):
        raise ValueError('output is not on the declared UINT8 grid')
    return scaled.astype(np.uint8)


def compare(actual, expected):
    indices = np.argwhere(actual != expected)
    return {'mismatches': len(indices), 'elements': actual.size,
            'max_lsb': int(np.abs(actual.astype(np.int16)-expected.astype(np.int16)).max()),
            'first': [{'index': p.tolist(), 'actual': int(actual[tuple(p)]),
                       'expected': int(expected[tuple(p)])} for p in indices[:12]]}


def self_check():
    # Independent scalar rational arithmetic checks the vectorized oracle itself.
    w, b = parameters(33, False)
    sample = fixtures(33, False)['holdout_channel_phase']
    tested = 0
    for sc, sb in [(0, 0), (1, -1), (4, 3), (16, 15)]:
        result = predict(sample, w, b, sc, sb)
        for out in range(32):
            for pixel in (0, 1, 31, 127, 255):
                a = sum(int(w[out, c]) * int(sample.reshape(33, -1)[c, pixel]) for c in range(33))
                val = (Fraction(a) + Fraction(int(b[out])) * Fraction(2)**sb) / Fraction(2)**sc
                expected = min(255, max(0, round(val)+128))
                assert expected == int(result.reshape(32, -1)[out, pixel])
                tested += 1
    reversed_w, reversed_b = parameters(33, True)
    for sc in (0, 1, 16):
        assert np.array_equal(predict(sample, w, b, sc, 0),
                              predict(sample[:, ::-1], reversed_w, reversed_b, sc, 0))
    print('ARITHMETIC_ORACLE_CHECKS_PASS', json.dumps({'rational_cases': tested, 'permutation_checks': 3}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-check', action='store_true')
    ap.add_argument('--channels', type=int, default=32)
    ap.add_argument('--shift-cut', type=int, default=1)
    ap.add_argument('--shift-bias', type=int, default=0)
    ap.add_argument('--reverse-channels', action='store_true')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--cpu-only', action='store_true')
    ap.add_argument('--fresh', action='store_true')
    a = ap.parse_args()
    if a.self_check:
        self_check()
        return
    if not a.out or a.out.exists() or not 1 <= a.channels <= 64:
        ap.error('need a new --out directory and 1..64 channels')
    if not 0 <= a.shift_cut <= 16 or not min(0, a.shift_cut-16) <= a.shift_bias <= 15:
        ap.error('shift parameters must stay inside the current XINT8 contract')
    if not a.cpu_only and not a.fresh:
        ap.error('NPU fixtures require --fresh')
    a.out.mkdir(parents=True)
    path = a.out / 'fixture.onnx'
    g, w, b = model(a.channels, a.shift_cut, a.shift_bias, a.reverse_channels)
    g.save(path)
    cases = fixtures(a.channels, a.reverse_channels)
    refs = {name: predict(x, w, b, a.shift_cut, a.shift_bias) for name, x in cases.items()}
    report = {'channels': a.channels, 'shift_cut': a.shift_cut, 'shift_bias': a.shift_bias,
              'reverse_channels': a.reverse_channels, 'model_sha256': file_hash(path),
              'ort_version': ort.__version__, 'fresh': a.fresh,
              'scope': 'observable compiled-stack arithmetic; no accuracy or latency claim',
              'formula': 'clip(zp_out + R((sum((qx-zpx)*qw) + qb*2^shift_bias)/2^shift_cut))',
              'weights': w.tolist(), 'bias_codes': b.tolist(), 'comparisons': {}}
    arrays = {'weights': w, 'bias_codes': b}
    for name, x in cases.items():
        arrays['input_'+name] = x
        arrays['reference_'+name] = refs[name]
    observed = {}
    for optimized in (False, True):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimized
                                         else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
        session = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
        label = 'cpu_optimized' if optimized else 'cpu_unoptimized'
        observed[label] = {name: output_codes(session.run(None, {'input': x})[0], a.shift_cut)
                           for name, x in cases.items()}
        del session
    if any(not np.array_equal(refs[n], observed['cpu_unoptimized'][n]) for n in refs):
        raise AssertionError('unoptimized CPU disagrees with independent exact reference')
    if not a.cpu_only:
        witness = subprocess.check_output(['C:/Windows/System32/AMD/xrt-smi.exe', 'examine', '-r', 'aie-partitions'], text=True)
        print('ARITHMETIC_CONTEXT', witness, flush=True)
        if 'No hardware contexts running' not in witness:
            raise SystemExit(3)
        xclbin = Path(resolve_xclbin())
        report['xclbin_sha256'] = file_hash(xclbin)
        cache = (CACHE_DIR / RESNET_CACHE_KEY).resolve()
        if cache.parent != ROOT.resolve():
            raise ValueError('EP cache escaped isolated worktree')
        clear_cache(RESNET_CACHE_KEY)
        session = build_session(path, 'npu', RESNET_CACHE_KEY, str(xclbin), log_severity=3)
        ep = read_report(cache / 'vitisai_ep_report.json')
        (a.out / 'ep.json').write_bytes(ep.path.read_bytes())
        report['ep'] = ep.summary()
        report['tested_conv_on_npu'] = report['ep']['ops_by_device'].get('NPU:Conv', 0) == 1
        print('ARITHMETIC_EP', json.dumps(report['ep']), flush=True)
        observed['npu'] = {}
        for name, x in cases.items():
            first = output_codes(session.run(None, {'input': x})[0], a.shift_cut)
            repeat = output_codes(session.run(None, {'input': x})[0], a.shift_cut)
            if not np.array_equal(first, repeat):
                raise AssertionError('nondeterministic repeated output')
            observed['npu'][name] = first
        del session
    for provider, outputs in observed.items():
        report['comparisons'][provider] = {}
        for name, actual in outputs.items():
            arrays[provider+'_'+name] = actual
            difference = compare(actual, refs[name])
            report['comparisons'][provider][name] = difference
            print('ARITHMETIC_COMPARISON', json.dumps({'provider': provider, 'fixture': name, **difference}), flush=True)
    candidates = []
    target = observed.get('npu', observed['cpu_unoptimized'])
    for ir in ROUNDERS:
        for br in ['exact', *ROUNDERS]:
            for final_round in ROUNDERS:
                fit = holdout = 0
                for name, x in cases.items():
                    mismatches = int(np.count_nonzero(target[name] != predict(x, w, b, a.shift_cut, a.shift_bias, final_round, br, ir)))
                    if name.startswith('fit_'):
                        fit += mismatches
                    else:
                        holdout += mismatches
                if fit == 0:
                    candidates.append({'input_round': ir, 'bias_round': br, 'output_round': final_round,
                                       'fit_mismatches': fit, 'holdout_mismatches': holdout})
    report['fit_survivors'] = candidates
    report['heldout_survivors'] = [c for c in candidates if c['holdout_mismatches'] == 0]
    report['status'] = 'cpu_only' if a.cpu_only else 'placed' if report['tested_conv_on_npu'] else 'fallback'
    np.savez_compressed(a.out / 'raw_outputs.npz', **arrays)
    report['raw_outputs_sha256'] = file_hash(a.out / 'raw_outputs.npz')
    (a.out / 'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('ARITHMETIC_RESULT', json.dumps({k: v for k, v in report.items() if k not in ('weights', 'bias_codes', 'comparisons', 'ep')}), flush=True)
    if report['status'] == 'fallback':
        raise SystemExit(6)


if __name__ == '__main__':
    main()
