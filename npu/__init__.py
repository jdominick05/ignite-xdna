"""Shared, Quark-free helpers for the Ryzen AI XDNA1 pipelines.

Nothing in this package may import quark: the inference scripts import it on
every run, and importing quark triggers a custom-op build attempt each time.
"""
