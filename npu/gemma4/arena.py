"""Gemma4 E2B assistant command composition over the reusable BF16 arena."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np

from kernels.transformer_bf16.state import ArenaProgram


def padded_pair(key,value,capacity):
    if key.dtype!=np.uint16 or value.dtype!=np.uint16 or key.ndim!=3 or key.shape!=value.shape:
        raise ValueError('Matching BF16 shared KV arrays required')
    if capacity is None:return key,value
    if type(capacity) is not int or not key.shape[1]<=capacity<=2560:
        raise ValueError('Padded live KV capacity must cover the cache and be at most 2560')
    padded=[]
    for a in (key,value):
        result=np.zeros((a.shape[0],capacity,a.shape[2]),np.uint16)
        result[:,:a.shape[1],:]=a
        padded.append(result)
    return tuple(padded)


def layer_capacity(capacity,kind):
    # A 512-wide inclusive sliding window needs at most 513 keys. Round that
    # up to the existing 256-key schedule unit instead of padding it globally.
    return min(capacity,768) if capacity is not None and kind=='sliding_attention' else capacity


def body(inputs, weights, config, rotations, shared, table, compiler=None,capacity=None):
    text = config['text_config']
    if config['backbone_hidden_size'] != 1536 or text['num_hidden_layers'] != 4 or text['num_kv_shared_layers'] != 4:
        raise ValueError('This prototype supports the E2B assistant only')
    if inputs.dtype != np.uint16 or inputs.size != 3072 or not config['use_ordered_embeddings']:
        raise ValueError('Expected concatenated E2B BF16 target states')
    x, residual, norm, tmp = 0, 256, 512, 768
    query, context, scores = 1536, 3584, 5632
    p = ArenaProgram(compiler=compiler)
    p.upload(query, inputs.reshape(-1), label='target_states')
    p.projection(query, weights['pre_projection.weight'], x)
    p.download('pre_projection', x, 256)
    for layer, kind in enumerate(text['layer_types']):
        prefix = f'model.layers.{layer}'
        w = lambda name: weights[prefix+'.'+name]
        key, value = shared[kind]
        if key.shape != value.shape or key.dtype != np.uint16 or value.dtype != np.uint16:
            raise ValueError('Target KV shape or dtype mismatch')
        if kind == 'sliding_attention':
            key, value = key[:, -513:, :], value[:, -513:, :]
        heads, length, width = key.shape
        if heads != 1 or length > 8192 or width not in (256, 512):
            raise ValueError('Unsupported E2B KV dimensions')
        local_capacity=layer_capacity(capacity,kind)
        key,value=padded_pair(key,value,local_capacity)
        p.norm(x, w('input_layernorm.weight'), norm)
        p.projection(norm, w('self_attn.q_proj.weight'), query)
        p.norm(query, w('self_attn.q_norm.weight'), query, rows=4)
        p.rope(query, *rotations[kind], width, label=f'layer_{layer}')
        for head in range(4):
            p.projection(query+head*width, key[0], scores, label=f'layer_{layer}.head_{head}.key')
            p.softmax(scores, length,capacity=local_capacity,
                      label=f'layer_{layer}.head_{head}' if capacity is not None else None)
            p.projection(scores, value[0].T, context+head*width, label=f'layer_{layer}.head_{head}.value')
        p.projection(context, w('self_attn.o_proj.weight'), tmp)
        p.norm(tmp, w('post_attention_layernorm.weight'), tmp)
        p.residual(x, tmp, residual)
        p.norm(residual, w('pre_feedforward_layernorm.weight'), norm)
        p.projection(norm, w('mlp.gate_proj.weight'), query)
        p.projection(norm, w('mlp.up_proj.weight'), context)
        p.geglu(query, context, table, scores)
        p.projection(scores, w('mlp.down_proj.weight'), tmp)
        p.norm(tmp, w('post_feedforward_layernorm.weight'), tmp)
        p.residual(residual, tmp, x, scale=np.uint16(w('layer_scalar').reshape(-1)[0]))
        p.download(f'layer_{layer}', x, 256)
    p.norm(x, weights['model.norm.weight'], norm)
    p.download('hidden', norm, 256)
    p.projection(norm, weights['post_projection.weight'], context)
    p.download('projected', context, 1536)
    p.projection(norm, weights['masked_embedding.centroids.weight'], query)
    p.download('centroids', query, 2048)
    p.topk('clusters', query, 2048, config['centroid_intermediate_top_k'])
    return p


def state_patches(inputs, rotations, shared, config,capacity=None):
    """Replace state data; growing KV requires declared padding or a new schedule."""
    matrices, factors, lengths = {}, {}, {}
    for layer, kind in enumerate(config['text_config']['layer_types']):
        key, value = shared[kind]
        if kind == 'sliding_attention':
            key, value = key[:, -513:, :], value[:, -513:, :]
        if key.shape != value.shape or key.shape[0] != 1:
            raise ValueError('Unsupported shared E2B KV')
        length=int(key.shape[1])
        key,value=padded_pair(key,value,layer_capacity(capacity,kind))
        factors[f'layer_{layer}'] = rotations[kind]
        for head in range(4):
            matrices[f'layer_{layer}.head_{head}.key'] = key[0]
            matrices[f'layer_{layer}.head_{head}.value'] = value[0].T
            if capacity is not None:lengths[f'layer_{layer}.head_{head}']=length
    patches=dict(uploads={'target_states': inputs.reshape(-1)}, matrices=matrices, rotations=factors)
    if capacity is not None:patches['lengths']=lengths
    return patches
