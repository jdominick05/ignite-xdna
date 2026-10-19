"""FP64 host diagnostics on GPU logits; never part of model serving."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import math

import numpy as np

GATES = dict(mean_kl_max=1e-4,max_kl_max=1e-3,mean_abs_delta_nll_max=.01,
             max_abs_delta_nll_max=.1,top1_agreement_min=.99)


def log_probs(logits):
    value = np.asarray(logits,dtype=np.float64)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError('Finite nonempty vocabulary vector required')
    value = value-value.max()
    return value-math.log(float(np.exp(value).sum()))


def compare_distribution(actual,control,target):
    if actual.shape != control.shape or not 0 <= target < control.size:
        raise ValueError('Matching vocabularies and a valid target required')
    a,c = log_probs(actual),log_probs(control)
    kl = float(np.sum(np.exp(c)*(c-a)))
    if kl < -1e-10:
        raise ValueError('Negative KL exceeds rounding noise')
    return dict(kl=max(0.,kl),delta_nll=float(c[target]-a[target]),
        control_nll=float(-c[target]),actual_nll=float(-a[target]),
        control_top1=int(np.argmax(control)),actual_top1=int(np.argmax(actual)),
        top1_equal=int(np.argmax(control))==int(np.argmax(actual)),
        max_logit_abs=float(np.max(np.abs(np.asarray(actual,np.float64)-np.asarray(control,np.float64)))))


def summarize(rows,expected_count):
    if len(rows) != expected_count or not rows:
        raise ValueError('Incomplete distribution study')
    result = dict(count=len(rows),mean_kl=float(np.mean([r['kl'] for r in rows])),
        max_kl=max(r['kl'] for r in rows),
        mean_abs_delta_nll=float(np.mean([abs(r['delta_nll']) for r in rows])),
        max_abs_delta_nll=max(abs(r['delta_nll']) for r in rows),
        top1_agreement=float(np.mean([r['top1_equal'] for r in rows])),
        mean_delta_nll=float(np.mean([r['delta_nll'] for r in rows])))
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError('Nonfinite distribution metric')
    result['diagnostic_passed'] = all(result[key.removesuffix('_max').removesuffix('_min')] <= value
        if key.endswith('_max') else result[key.removesuffix('_min')] >= value for key,value in GATES.items())
    return result
