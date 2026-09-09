"""Research exact float16 value counts versus Ignition's ordered sample spool.

No production option is changed. A temporary, process-local collector replacement
lets the existing producer perform preparation, emission and refinement verbatim.
The certificate uses a conservative bound valid for ANY float32 addition tree with
N-1 additions, rather than assuming a particular SIMD reduction implementation.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import quant.calib as calibration
from quant.graph import Graph
from quant.pow2 import TensorQ, quantize as qvalue, dequantize
from quant.qdq import prunable_tensors, quantizable_tensors
from quant.quantize import graph_family, quantize, file_hash
from quant.sources import ImageFolderSource, CocoSource, ModnetSource


def error_values(values, position):
    restored = dequantize(qvalue(values, position, 128, 'uint8'), position, 128)
    return (restored - values) ** 2


def certified_choice(counts):
    """Bound the legacy float32 sum of the SAME per-element float32 errors.

All errors are nonnegative. gamma_(N-1) bounds any addition tree's relative
error, provided (N-1)*u < 1 and intermediate sums do not overflow. Float16
samples and the five locally chosen positions give zero or normal float32
squared errors; the implementation explicitly checks that prerequisite.
The float64 histogram dot product gets its own conservative 2*K-operation bound.
Strictly separated intervals certify a unique first-minimum equivalence class.
Pointwise-identical candidate error arrays have identical legacy sums, including
ties, regardless of input order. No assertion is made for merely equal totals.
"""
    if counts.shape != (65536,) or counts.dtype != np.uint64:
        raise ValueError('need a uint64 count for every float16 bit pattern')
    occupied = np.flatnonzero(counts)
    if not occupied.size:
        raise ValueError('empty count table')
    values = occupied.astype(np.uint16).view(np.float16).astype(np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError('nonfinite float16 samples')
    n = sum(int(c) for c in counts[occupied])
    if n > 2**53:
        return None, {'reason': 'count_exceeds_exact_float64_integer', 'count': n}
    _, seed = calibration.choose_pow2_minmse(values, 'uint8')
    candidates = seed['candidate_positions']
    errors = np.stack([error_values(values, p) for p in candidates])
    if not np.all(np.isfinite(errors)):
        return None, {'reason': 'nonfinite_element_error', 'count': n}
    groups = []
    for i in range(5):
        if not any(np.array_equal(errors[i], errors[j]) for j in groups):
            groups.append(i)
    estimates = errors.astype(np.float64) @ counts[occupied].astype(np.float64)
    stats = {'count': n, 'unique_values': int(occupied.size), 'vmin': float(values.min()),
             'vmax': float(values.max()), 'minmax_pos': seed['minmax_pos'],
             'candidate_positions': candidates, 'histogram_loss_float64': estimates.tolist(),
             'equivalence_representatives': groups, 'certificate': 'nonnegative_gamma_n_minus_1'}
    if np.any((errors != 0) & (errors < np.finfo(np.float32).tiny)):
        return None, {**stats, 'reason': 'subnormal_element_error'}
    if len(groups) == 1:
        return candidates[0], {**stats, 'reason': 'pointwise_identical_errors', 'certified': True}
    u = 2.0**-24
    nu = max(0, n-1) * u
    if nu >= 1:
        return None, {**stats, 'reason': 'unbounded_universal_reduction_error', 'certified': False}
    gamma = np.nextafter(nu / (1-nu), np.inf)
    # Each weighted product plus each summation is covered. All operands here
    # are exactly representable in float64; multiply/add rounding is bounded.
    ku = 2 * occupied.size * 2.0**-53
    beta = np.nextafter(ku / (1-ku), np.inf)
    # Round every interval operation outwards, including the bound constants.
    lower = np.nextafter(estimates / np.nextafter(1+beta, np.inf), -np.inf)
    lower = np.nextafter(lower * max(0, np.nextafter(1-gamma, -np.inf)), -np.inf)
    upper = np.nextafter(estimates / np.nextafter(1-beta, -np.inf), np.inf)
    upper = np.nextafter(upper * np.nextafter(1+gamma, np.inf), np.inf)
    stats.update(loss_intervals=[[max(0, float(l)), float(h)] for l, h in zip(lower, upper)],
                 reduction_gamma=gamma, histogram_gamma=beta)
    if np.any(upper > np.finfo(np.float32).max):
        return None, {**stats, 'reason': 'possible_legacy_overflow', 'certified': False}
    for j in groups:
        if all(upper[j] < lower[k] for k in groups if k != j):
            return candidates[j], {**stats, 'reason': 'separated_intervals', 'certified': True}
    return None, {**stats, 'reason': 'overlapping_intervals', 'certified': False}


def add_counts(table, sample):
    delta = np.bincount(sample.view(np.uint16).ravel(), minlength=65536).astype(np.uint64)
    if np.any(table > np.iinfo(np.uint64).max - delta):
        raise OverflowError('uint64 frequency counter overflow')
    table += delta


def collect(g, source, scratch, method):
    all_acts, weights, sharing = quantizable_tensors(g)
    pruned = prunable_tensors(g)
    acts = [name for name in all_acts if name not in pruned]
    g.infer_shapes()
    shapes = {name: g.value_shape(name) for name in acts}
    if any(s is None or any(d <= 0 for d in s) for s in shapes.values()):
        raise ValueError('fully inferred static shapes required')
    if any(g.value_dtype(n) != onnx.TensorProto.FLOAT for n in acts):
        raise ValueError('float32 activations required')
    sizes = {n: math.prod(shapes[n]) * len(source) * 2 for n in acts}
    estimated = sum(sizes.values())
    scratch = Path(scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    # Reserve the worst-case fallback before starting, even in alphabet mode.
    if shutil.disk_usage(scratch).free < estimated + 2*1024**3:
        raise OSError('insufficient disk for worst-case exact fallback plus 2 GiB')
    augmented = onnx.ModelProto()
    augmented.CopyFrom(g.model)
    infos = {v.name: v for v in (*g.model.graph.input, *g.model.graph.value_info, *g.model.graph.output)}
    del augmented.graph.output[:]
    augmented.graph.output.extend(infos[n] for n in acts)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(augmented.SerializeToString(), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    tables = {name: np.zeros(65536, dtype=np.uint64) for name in acts}
    hashes = {name: hashlib.sha256() for name in acts}
    positions, stats = {}, {}
    timings = {'inference_seconds': 0., 'count_seconds': 0., 'write_seconds': 0., 'hash_seconds': 0.}
    report = {'method': method, 'store': 'float16_bit_pattern_counts_uint64',
              'images': len(source), 'listing': [str(p) for p in source.listing()],
              'input_file_hashes': [file_hash(p) for p in source.listing()],
              'augmented_graph_sha256': hashlib.sha256(augmented.SerializeToString()).hexdigest(),
              'graph_optimization': 'ORT_DISABLE_ALL', 'ort_version': ort.__version__,
              'numpy_version': np.__version__, 'spooled_tensors': 0,
              'skipped_prunable_tensors': len(pruned), 'legacy_sample_bytes': estimated,
              'count_table_bytes': len(acts)*65536*8, 'tensors': stats, 'sharing': sharing}
    print('ALPHABET_COLLECTION', json.dumps({k: v for k, v in report.items()
          if k not in ('listing', 'input_file_hashes', 'tensors', 'sharing')}), flush=True)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='alphabet-', dir=scratch) as directory:
        root = Path(directory).resolve()
        if root.parent != scratch:
            raise ValueError('temporary directory escaped scratch')
        paths = {n: root / f'{i}.f16' for i, n in enumerate(acts)}
        count = 0
        for count, image in enumerate(source, 1):
            if image.dtype != np.float32 or image.shape != shapes[source.input_name]:
                raise ValueError('preprocessing shape/dtype mismatch')
            t = time.perf_counter()
            outputs = session.run(acts, {source.input_name: image})
            timings['inference_seconds'] += time.perf_counter()-t
            for name, value in zip(acts, outputs, strict=True):
                with np.errstate(over='ignore'):
                    sample = value.astype(np.float16)
                if sample.shape != shapes[name] or not np.all(np.isfinite(sample)):
                    raise ValueError('invalid sample at ' + name)
                t = time.perf_counter()
                hashes[name].update(sample.tobytes())
                timings['hash_seconds'] += time.perf_counter()-t
                t = time.perf_counter()
                add_counts(tables[name], sample)
                timings['count_seconds'] += time.perf_counter()-t
                if method == 'dual':
                    t = time.perf_counter()
                    with paths[name].open('ab') as f:
                        sample.tofile(f)
                    timings['write_seconds'] += time.perf_counter()-t
            if count % 16 == 0 or count == len(source):
                print('ALPHABET_COLLECT', count, len(source), flush=True)
        if count != len(source):
            raise ValueError('source length changed')
        del outputs, sample
        ambiguous = []
        t = time.perf_counter()
        for name in acts:
            p, row = certified_choice(tables[name])
            row['samples_sha256'] = hashes[name].hexdigest()
            if row['count'] != sizes[name] // 2:
                raise ValueError('sample count mismatch')
            stats[name] = row
            if p is None:
                ambiguous.append(name)
            else:
                positions[name] = TensorQ(name, 'uint8', p, 128, 'alphabet_certified')
        timings['certificate_seconds'] = time.perf_counter()-t
        del tables
        report['fallback_tensors'] = len(ambiguous)
        report['fallback_sample_bytes'] = sum(sizes[n] for n in ambiguous)
        report['spooled_tensors'] = len(acts) if method == 'dual' else len(ambiguous)
        report['sample_bytes'] = estimated if method == 'dual' else report['fallback_sample_bytes']
        print('ALPHABET_FALLBACK', json.dumps({k: report[k] for k in
              ('fallback_tensors', 'fallback_sample_bytes', 'sample_bytes')}), flush=True)
        if ambiguous and method == 'alphabet':
            replay_hashes = {n: hashlib.sha256() for n in ambiguous}
            t = time.perf_counter()
            for image in source:
                # Identical output list and session, preserving the original execution graph.
                outputs = session.run(acts, {source.input_name: image})
                for name, value in zip(acts, outputs, strict=True):
                    if name not in replay_hashes:
                        continue
                    sample = value.astype(np.float16)
                    replay_hashes[name].update(sample.tobytes())
                    with paths[name].open('ab') as f:
                        sample.tofile(f)
            for name in ambiguous:
                if replay_hashes[name].hexdigest() != stats[name]['samples_sha256']:
                    raise ValueError('nondeterministic calibration replay: ' + name)
            timings['replay_seconds'] = time.perf_counter()-t
            del outputs, sample
        del session
        t = time.perf_counter()
        for name in acts if method == 'dual' else ambiguous:
            values = np.fromfile(paths[name], dtype=np.float16).astype(np.float32)
            if values.size != sizes[name] // 2:
                raise ValueError('incomplete spool')
            p, legacy = calibration.choose_pow2_minmse(values, 'uint8')
            if name in positions and positions[name].pos != p:
                raise AssertionError('UNSOUND CERTIFICATE: ' + name)
            stats[name]['legacy_minmse_pos'] = p
            stats[name]['legacy_candidate_sqerr'] = legacy['candidate_sqerr']
            if name in ambiguous:
                positions[name] = TensorQ(name, 'uint8', p, 128, 'alphabet_exact_fallback')
            del values
        timings['legacy_reduction_seconds'] = time.perf_counter()-t
    for name in weights:
        p, row = calibration.choose_pow2_minmse(g.initializer(name), 'int8')
        positions[name] = TensorQ(name, 'int8', p, 0, 'initializer_minmse')
    for name, provider in sharing.items():
        positions[name] = replace(positions[provider], name=name, source='shared:'+provider)
    report['timings'] = timings
    report['collector_seconds'] = time.perf_counter()-start
    report['certified_tensors'] = len(acts)-len(ambiguous)
    report['certified_positions_checked_against_legacy'] = method == 'dual'
    print('ALPHABET_RESULT', json.dumps({k: v for k, v in report.items()
          if k not in ('listing', 'input_file_hashes', 'tensors', 'sharing')}), flush=True)
    for name, row in stats.items():
        print('ALPHABET_TENSOR', json.dumps({'name': name, 'position': positions[name].pos, **row}), flush=True)
    return positions, report


def self_check():
    rng = np.random.default_rng(1729)
    cases = [np.array([-2., -1., -.5, -0., 0., .5, 1., 2.], np.float16),
             np.array([np.nextafter(np.float16(0), np.float16(1)), -1., 1.], np.float16),
             np.array([-65504., 65504., .1, -.1], np.float16)]
    for length in (7, 8, 127, 128, 129, 8191, 8192, 8193, 65537):
        cases.append(rng.normal(size=length).astype(np.float16))
        cases.append(rng.choice(np.array([-.5, .5, 1., 8.], np.float16), size=length))
    certified = ambiguous = 0
    for sample in cases:
        for ordered in (sample, sample[::-1], np.sort(sample)):
            table = np.zeros(65536, np.uint64)
            add_counts(table, ordered.copy())
            pos, stat = certified_choice(table)
            legacy, _ = calibration.choose_pow2_minmse(ordered.astype(np.float32), 'uint8')
            if pos is None:
                ambiguous += 1
            else:
                assert pos == legacy, (pos, legacy, stat)
                certified += 1
    fallback_reasons = []
    # Force both conservative-bound exits without constructing huge samples.
    table = np.zeros(65536, np.uint64)
    add_counts(table, np.array([.1, .2, 10.], np.float16))
    for repeats, reason in ((4000000, 'overlapping_intervals'),
                            (8000000, 'unbounded_universal_reduction_error'),
                            (2**53, 'count_exceeds_exact_float64_integer')):
        pos, stat = certified_choice(table * np.uint64(repeats))
        assert pos is None and stat['reason'] == reason, stat
        fallback_reasons.append(reason)
    for bad in (np.array([np.inf], np.float16), np.array([np.nan], np.float16), np.zeros(8, np.float16)):
        table = np.zeros(65536, np.uint64)
        add_counts(table, bad)
        try:
            certified_choice(table)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid activation was accepted')
    table = np.zeros(65536, np.uint64)
    table[np.array([1.], np.float16).view(np.uint16)[0]] = np.iinfo(np.uint64).max
    try:
        add_counts(table, np.array([1.], np.float16))
    except OverflowError:
        pass
    else:
        raise AssertionError('counter overflow accepted')
    print('ALPHABET_CHECKS_PASS', json.dumps({'certified_cases': certified, 'fallback_cases': ambiguous,
          'forced_fallback_reasons': fallback_reasons,
          'invalid_inputs_rejected': 3, 'counter_overflow_rejected': True}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-check', action='store_true')
    ap.add_argument('--in-model', type=Path)
    ap.add_argument('--calib-dir', type=Path)
    ap.add_argument('--cfg-path', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--reference', type=Path)
    ap.add_argument('--method', choices=['legacy', 'dual', 'alphabet'], default='dual')
    ap.add_argument('--limit', type=int, default=64)
    ap.add_argument('--scratch', type=Path, default=ROOT / 'scratch/alphabet')
    cle = ap.add_mutually_exclusive_group()
    cle.add_argument('--cle', action='store_true')
    cle.add_argument('--no-cle', action='store_true')
    a = ap.parse_args()
    if a.self_check:
        self_check()
        return
    if not a.in_model or not a.calib_dir or not a.out or a.limit <= 0 or not (a.cle or a.no_cle):
        ap.error('need --in-model, --calib-dir, --out, positive --limit and --cle/--no-cle')
    g = Graph.load(a.in_model)
    family = graph_family(g)
    name = g.model.graph.input[0].name
    shape = g.value_shape(name)
    if family == 'yolo_cut':
        from npu.yolo import input_size
        source = CocoSource(a.calib_dir, a.limit, input_size(list(shape), str(a.in_model)), name)
    elif family == 'modnet':
        cfg = json.loads(a.cfg_path.read_text(encoding='utf-8'))
        if (cfg.get('height'), cfg.get('width')) != shape[-2:] or shape[-2] != shape[-1]:
            raise ValueError('MODNet config/shape mismatch')
        source = ModnetSource(a.calib_dir, a.limit, shape[-1], name)
    else:
        cfg = json.loads(a.cfg_path.read_text(encoding='utf-8'))
        if shape != (1, *cfg['input_size']):
            raise ValueError('ResNet config/shape mismatch')
        source = ImageFolderSource(a.calib_dir, cfg, a.limit, name)
    del g
    original = calibration.collect_and_choose
    if a.method != 'legacy':
        calibration.collect_and_choose = lambda g, s, scratch: collect(g, s, scratch, a.method)
    try:
        report = quantize(a.in_model, a.out, source=source, preprocess=source.preprocess(),
                          scratch=a.scratch, cle=a.cle)
    finally:
        calibration.collect_and_choose = original
    report['research'] = {'method': a.method, 'production_default_changed': False,
                          'calibration_error_report': 'legacy_float32' if a.method == 'legacy' else 'certified_intervals_and_exact_fallback'}
    if a.reference:
        report['research']['reference_sha256'] = file_hash(a.reference)
        report['research']['onnx_bytes_identical'] = file_hash(a.reference) == report['output_sha256']
    Path(str(a.out)+'.quant.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('ALPHABET_MODEL_RESULT', json.dumps({'method': a.method, 'family': family,
          'wall_seconds': report['wall_seconds'], 'output_sha256': report['output_sha256'], **report['research']}), flush=True)
    if a.reference and not report['research']['onnx_bytes_identical']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
