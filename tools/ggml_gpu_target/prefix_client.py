"""Client for the isolated prefix worker; the regular Target stays unchanged."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
from .client import Target


class PrefixTarget(Target):
    def prepare_cached(self, tokens):
        """Reuse a matching retained prefix and return complete target KV state."""
        return self.request(9, tokens)
