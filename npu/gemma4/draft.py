"""NPU E2B draft steps and shared target state, without CPU model inference."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import time

import numpy as np

from kernels.transformer_bf16.common import bf16_bits
from kernels.transformer_bf16.state import ArenaCompiler, ArenaProgram
from .arena import body, state_patches


class SharedState:
    def __init__(self, response):
        self.shared = {}
        self.append(response, response['position'], initial=True)

    def append(self, response, keep, initial=False):
        if not 0 < keep <= response['position']:
            raise ValueError('Invalid accepted target state length')
        for kind, layer, width in [('sliding_attention', 13, 256), ('full_attention', 14, 512)]:
            pair = []
            for prefix in ('Kcur_pos-', 'Vcur_normed-'):
                value = response['tensors'][prefix+str(layer)]
                if value.shape[0] != width or value.shape[1] != 1 or value.shape[3] != 1 or value.shape[2] < keep:
                    raise ValueError('Invalid live target KV shape')
                pair.append(bf16_bits(value[:, 0, :keep, 0].T[None]))
            if not initial:
                pair = [np.concatenate((old, new), axis=1) for old, new in zip(self.shared[kind], pair)]
            self.shared[kind] = tuple(pair)
        hidden = response['tensors'].get('h_nextn', response['tensors']['result_norm'])
        index = hidden.shape[1]-1 if initial else keep-1
        if hidden.shape[0] != 1536 or index >= hidden.shape[1]:
            raise ValueError('Missing accepted target hidden row')
        self.hidden = bf16_bits(hidden[:, index, 0, 0])
        self.position = self.shared['full_attention'][0].shape[1]


class Draft:
    def __init__(self, artifacts, config, weights, rotations, table, prepared=False, compare_prepared=False, compact=False, kv_capacity=0, small_packets=False,input_depth=1,vector_copy=False):
        self.artifacts, self.config, self.weights = artifacts, config, weights
        self.rotations, self.table = rotations, table
        self.compiler = ArenaCompiler(exact_norm=True,compact=compact,small_packets=small_packets,input_depth=input_depth,vector_copy=vector_copy)
        self.control_compiler = ArenaCompiler(exact_norm=True) if (compact or small_packets) and compare_prepared else self.compiler
        self.cache = artifacts/'npu/arena-images'
        self.prepared = prepared
        self.compare_prepared = compare_prepared
        self.body_template = self.head_template = None
        self.body_position = None
        self.kv_capacity = kv_capacity
        self.body_templates = {}
        self.body_state_patches = None
        self.counts = dict(body_templates=0, body_reuses=0, head_templates=0, head_reuses=0,
                           compared_bodies=0, compared_heads=0)

    def invalidate_prefix(self):
        """A new prompt can have the same length but different KV contents."""
        self.body_position = None
        self.body_state_patches = None

    @staticmethod
    def equal(actual, expected, stage):
        if actual.keys() != expected.keys() or any(not np.array_equal(actual[k], expected[k]) for k in actual):
            detail={k:dict(different=int(np.count_nonzero(actual[k]!=expected[k])),
                           indices=np.flatnonzero(actual[k]!=expected[k])[:8].tolist())
                    for k in actual if not np.array_equal(actual[k],expected[k])}
            raise ValueError('Prepared '+stage+' differs bitwise from rebuilt command stream: '+json.dumps(detail))

    def step(self, gpu, session, state, token, hidden):
        step_start = time.perf_counter()
        factors = {kind: tuple(self.rotations[kind+'.'+label][state.position] for label in ('cosine', 'sine'))
                   for kind in self.config['text_config']['layer_types']}
        inputs = np.concatenate((bf16_bits(gpu.embedding(token)), hidden))
        embedding_ms = (time.perf_counter()-step_start)*1000
        body_start = time.perf_counter()
        program = None
        capacity = ((state.position+self.kv_capacity-1)//self.kv_capacity)*self.kv_capacity if self.kv_capacity else None
        needs_template = capacity not in self.body_templates if capacity else self.body_position != state.position
        if not self.prepared or needs_template or self.compare_prepared:
            program = body(inputs, self.weights, self.config, factors, state.shared, self.table, self.compiler,capacity)
        if self.prepared:
            if needs_template:
                # KV length/content and rotary position change only between
                # verification rounds. Retain just this prefix's body template.
                self.body_template = program.prepare(self.cache)
                if capacity:self.body_templates[capacity] = self.body_template
                self.body_state_patches = None
                self.body_position = state.position
                self.counts['body_templates'] += 1
                actual, evidence = self.body_template.execute(session)
            elif capacity and self.body_position != state.position:
                self.body_template = self.body_templates[capacity]
                self.body_state_patches = state_patches(inputs,factors,state.shared,self.config,capacity)
                actual,evidence = self.body_template.execute(session,
                    **self.body_state_patches)
                self.counts['body_reuses'] += 1
            else:
                # A draft chain keeps KV and position fixed. Only its token
                # embedding and post-projected hidden state need replacement.
                patches = dict(self.body_state_patches or {})
                patches['uploads'] = {'target_states': inputs}
                actual, evidence = self.body_template.execute(session, **patches)
                self.counts['body_reuses'] += 1
            self.body_position = state.position
            if self.compare_prepared:
                control = body(inputs, self.weights, self.config, factors, state.shared, self.table, self.control_compiler)
                expected, comparison = control.execute(self.cache, session)
                self.equal(actual, expected, 'body')
                evidence['comparison'] = comparison
                self.counts['compared_bodies'] += 1
        else:
            actual, evidence = program.execute(self.cache, session)
        body_ms = (time.perf_counter()-body_start)*1000
        head_start = time.perf_counter()
        ordering = self.weights['masked_embedding.token_ordering'].reshape(self.config['num_centroids'], -1)
        selected = np.sort(ordering[actual['clusters']].reshape(-1))
        if selected.size != 4096 or np.unique(selected).size != 4096:
            raise ValueError('Drafter head candidate set invalid')
        selected_weights = self.weights['model.embed_tokens.weight'][selected]
        head = None
        if not self.prepared or self.head_template is None or self.compare_prepared:
            head = ArenaProgram(compiler=self.compiler)
            head.upload(0, actual['hidden'], label='hidden')
            head.projection(0, selected_weights, 256, label='selected')
            if self.compare_prepared:
                head.download('logits_low', 256, 2048)
                head.download('logits_high', 2304, 2048)
            head.topk('index', 256, 4096, 1)
        if self.prepared:
            if self.head_template is None:
                self.head_template = head.prepare(self.cache)
                self.counts['head_templates'] += 1
                output, head_evidence = self.head_template.execute(session)
            else:
                output, head_evidence = self.head_template.execute(session,
                                uploads={'hidden': actual['hidden']}, matrices={'selected': selected_weights})
                self.counts['head_reuses'] += 1
            if self.compare_prepared:
                control = ArenaProgram(compiler=self.control_compiler)
                control.upload(0, actual['hidden'])
                control.projection(0, selected_weights, 256)
                control.download('logits_low', 256, 2048)
                control.download('logits_high', 2304, 2048)
                control.topk('index', 256, 4096, 1)
                expected, comparison = control.execute(self.cache, session)
                self.equal(output, expected, 'head')
                head_evidence['comparison'] = comparison
                self.counts['compared_heads'] += 1
        else:
            output, head_evidence = head.execute(self.cache, session)
        index = int(output['index'][0])
        if not 0 <= index < selected.size:
            raise ValueError('Drafter selected head index out of bounds')
        evidence['step_wall_ms'] = (time.perf_counter()-step_start)*1000
        evidence['step_phases_ms'] = dict(gpu_embedding=embedding_ms, body=body_ms,
                                         head=(time.perf_counter()-head_start)*1000)
        return int(selected[index]), actual['projected'], dict(body=evidence, head=head_evidence)
