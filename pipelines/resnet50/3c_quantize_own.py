"""Compatibility entry point for Ignition: python -m quant quantize."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quant.cli import main

if __name__ == "__main__":
    main(["quantize", *sys.argv[1:]])
