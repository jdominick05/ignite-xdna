"""Separate research candidate; the serving Draft implementation is untouched."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
from kernels.transformer_bf16.prefix_cache import CachedPrefixArena
from kernels.transformer_bf16.prefix_binding import PrefixBinder
from npu.gemma4.draft import Draft


class PrefixCachedDraft(Draft):
    @classmethod
    def from_draft(cls, draft):
        result = object.__new__(cls)
        result.__dict__ = draft.__dict__.copy()
        result.body_templates = dict(draft.body_templates)
        result.counts = dict(draft.counts)
        result.prefix_cache_enabled = True
        result.bind_prefix = False
        return result

    def _wrap(self):
        wrappers = {}
        for capacity, template in self.body_templates.items():
            wrapper_type = PrefixBinder if self.bind_prefix else CachedPrefixArena
            wrapper = template if isinstance(template, wrapper_type) else wrapper_type(template)
            wrapper.enabled = self.prefix_cache_enabled
            wrappers[id(template)] = wrapper
            self.body_templates[capacity] = wrapper
        if self.body_template is not None:
            template = self.body_template
            self.body_template = wrappers.get(id(template), template)

    def step(self, *args, **kwargs):
        self._wrap()
        result = super().step(*args, **kwargs)
        self._wrap()
        if self.bind_prefix and self.prefix_cache_enabled and self.body_state_patches:
            bound = self.body_template.take_bound()
            if bound is None:
                raise RuntimeError('Prefix materialization did not produce a binding')
            self.body_template = bound
            self.body_state_patches = None
        return result
