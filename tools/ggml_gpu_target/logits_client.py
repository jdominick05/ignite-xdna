"""Bounded read-only full-vocabulary output from research workers."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np

from tools.ggml_gpu_target.client import Target


class LogitTarget(Target):
    def logits(self, count=1):
        if type(count) is not int or not 1 <= count <= 16:
            raise ValueError('Use 1..16 current output rows')
        response = self.request(11,[count])
        if response['predictions'] or set(response['tensors']) != {'logits'}:
            raise ValueError('Unexpected logits response')
        value = response['tensors']['logits']
        if value.shape != (32768,8,count,1):
            raise ValueError('Unexpected logits shape')
        return np.ascontiguousarray(value[:,:,:,0].transpose(2,1,0).reshape(count,262144))
