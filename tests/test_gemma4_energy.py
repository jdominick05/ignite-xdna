"""Energy accounting gates: interval alignment, complete work and rejection."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import copy
import os
from types import SimpleNamespace

import pytest

from tools.profile_gemma4_energy import arm_metrics, evaluate, summarize_window, PROTOCOL
from tools import profile_gemma4_energy as measurement


def power_rows(start, count, watts):
    return [[start+i, watts*1000, 5000, 2, 0, 0] for i in range(count)]


def test_power_intervals_exclude_row_straddling_start():
    rows = power_rows(100, 61, 40)
    # A large row ending at 100.5 starts outside this window.
    rows.insert(1, [100.5, 999000, 0, 2, 0, 0])
    result = summarize_window(rows, 100, 160)
    assert result['rows'] == 60
    assert result['package_w'] == 40


def test_energy_counts_only_complete_interior_requests():
    requests = [dict(start=100+4*i, end=104+4*i, tokens=list(range(32))) for i in range(20)]
    idle = dict(start=0, end=60, rows=power_rows(0, 61, 30))
    result = arm_metrics(idle, power_rows(100, 81, 40), requests)
    assert result['complete_requests'] == 18
    assert result['tokens'] == 576
    assert result['active']['seconds'] == 72
    assert result['tokens_per_second'] == 8
    assert result['package_j_per_token'] == 5
    assert result['above_idle_j_per_token'] == 1.25


@pytest.mark.parametrize('fault', ['sparse', 'gap', 'nan', 'negative', 'duplicate'])
def test_invalid_counter_windows_are_rejected(fault):
    rows = power_rows(0, 61, 40)
    if fault == 'sparse':
        rows = rows[:20]
    elif fault == 'gap':
        del rows[30:34]
    elif fault == 'nan':
        rows[30][1] = float('nan')
    elif fault == 'negative':
        rows[30][2] = -1
    else:
        rows.insert(30, rows[30].copy())
    with pytest.raises(ValueError):
        summarize_window(rows, 0, 60)


def mirrored_arms():
    return [dict(mode=mode, metrics=dict(idle=dict(package_w=30),
                package_j_per_token=5, above_idle_j_per_token=1.25)) for mode in PROTOCOL['order']]


def test_mirrored_comparison_rejects_missing_arm_drift_or_repeat_disagreement():
    arms = mirrored_arms()
    assert evaluate(arms)['baseline_median_w'] == 30
    with pytest.raises(ValueError, match='Incomplete'):
        evaluate(arms[:-1])
    drift = copy.deepcopy(arms)
    drift[0]['metrics']['idle']['package_w'] = 33
    with pytest.raises(ValueError, match='drift'):
        evaluate(drift)
    mismatch = copy.deepcopy(arms)
    mismatch[0]['metrics']['above_idle_j_per_token'] = 1.6
    with pytest.raises(ValueError, match='disagreement'):
        evaluate(mismatch)


def test_no_energy_claim_from_nonpositive_delta_or_disturbed_idle():
    requests = [dict(start=100+4*i, end=104+4*i, tokens=list(range(32))) for i in range(20)]
    idle = dict(start=0, end=60, rows=power_rows(0, 61, 40))
    with pytest.raises(ValueError, match='Nonpositive'):
        arm_metrics(idle, power_rows(100, 81, 30), requests)
    for row in idle['rows']:
        row[3] = 20
    with pytest.raises(ValueError, match='Disturbed'):
        arm_metrics(idle, power_rows(100, 81, 50), requests)


@pytest.mark.parametrize('fault', ['none', 'cpu', 'gpu', 'npu', 'unknown'])
def test_monitor_rejects_foreign_work_and_unknown_telemetry_and_closes_query(monkeypatch, fault):
    closed = []
    own = os.getpid()
    foreign = own+100000

    class Query:
        def sample(self):
            return fault != 'unknown', ([dict(pid=foreign, utilization_percent=30)] if fault == 'gpu' else []), {}

        def close(self):
            closed.append(True)

    samples = iter([({(own, 0): ('python.exe', 0), (foreign, 0): ('python.exe', 0)}, True),
                    ({(own, 0): ('python.exe', 1), (foreign, 0): ('python.exe', 1 if fault == 'cpu' else 0)}, True)])
    monkeypatch.setattr(measurement, '_GpuQuery', Query)
    monkeypatch.setattr(measurement, '_cpu_times', lambda: next(samples))
    monkeypatch.setattr(measurement, 'context_report', lambda: f'PID: {foreign}' if fault == 'npu' else 'No hardware contexts running')
    monitor = measurement.ContentionMonitor(SimpleNamespace(gpu=None, _activity=None))
    waits = iter([False, True])
    monitor.stop = SimpleNamespace(wait=lambda _: next(waits))
    monitor.run()
    assert monitor.failed.is_set() == (fault != 'none')
    assert closed == [True]
