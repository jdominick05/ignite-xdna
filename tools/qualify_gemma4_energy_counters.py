"""Check existing package-power counters; this is not an energy or idle result.

Reuse the study's measured milliwatt Power scale. Do not use the unresolved
Energy counter or infer whole-system power completeness from availability.
"""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kernels.transformer_bf16.common import sha,save
from npu.contention import snapshot
from tools import power_probe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    assert Path(power_probe.__file__).resolve()==ROOT/'tools/power_probe.py'
    sources={str(path.resolve()):sha(path) for path in (Path(__file__).resolve(),ROOT/'tools/power_probe.py',ROOT/'npu/contention.py')}
    report=dict(passed=False,scope=__doc__,source_sha256=sources,timing_eligible=False,
                energy_eligible=False,unit='milliwatts, from the prior registered power-counter study',
                completeness='Not shown; package counters exclude DRAM and are not wall power')
    try:
        report['contention']=snapshot()
        series=power_probe.sample_counters(3,1)
        report.update(counters_mw=series,counter_count=len(series),requested_samples=3,
                      valid_sample_counts={path:len(values) for path,values in series.items()})
        assert set(series)=={power_probe.PKG,*power_probe.CORES}
        assert all(len(values)>=2 and all(math.isfinite(v) and v>=0 for v in values) for values in series.values())
        assert all(v>0 for v in series[power_probe.PKG])
        assert all(sha(path)==expected for path,expected in sources.items())
        report.update(passed=True,counters_mw=series,counter_count=len(series))
        print('PASS: package and eight core counters available; no energy comparison performed',flush=True)
    finally:save(a.out,report)


if __name__=='__main__':main()
