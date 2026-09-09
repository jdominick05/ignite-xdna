"""A terminal launcher for the models in this repo.

Two lanes: `task` runs a model on your own input in-process and writes a usable
result under outputs/; `demo` subprocess-launches one of demos/*.py.

This is not a measurement tool. results/ is the tracked evidence base with a
naming contract, so nothing here writes there and every latency printed is
labelled indicative. See tui/registry.py for the catalogue.
"""
__version__ = "0.1.0a1"
