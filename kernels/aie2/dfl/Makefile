# Build and qualify the Phoenix 16-core DFL decode transport.
# Run from an activated mlir-aie ironenv.
PYTHON ?= python
OUT ?= build/dfl_decode
ITERS ?= 20
WARMUP ?= 5
.NOTPARALLEL:
.PHONY: all check offline compile hardware

all: offline compile

check:
	$(PYTHON) tests/test_dfl_kernel.py --offline

offline:
	$(PYTHON) tests/test_dfl_kernel.py --offline

compile:
	$(PYTHON) tests/test_dfl_kernel.py --compile --build-dir "$(OUT)"

hardware:
	$(PYTHON) tests/test_dfl_kernel.py --hardware --iters $(ITERS) --warmup $(WARMUP) --build-dir "$(OUT)"
