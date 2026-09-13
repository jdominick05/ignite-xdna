# Build and qualify the Phoenix 16-core DFL decode transport.
# Run from an activated mlir-aie ironenv.  COLS/CORES_PER_COL build a reduced
# probe with the full design's anchor mapping; the latency gate is full 4x4 only.
PYTHON ?= python
OUT ?= build/dfl_decode
COLS ?= 4
CORES_PER_COL ?= 4
ITERS ?= 100
WARMUP ?= 10
SHAPE = --cols $(COLS) --cores-per-col $(CORES_PER_COL)
.NOTPARALLEL:
.PHONY: all check offline compile hardware

all: offline compile

check:
	$(PYTHON) tests/test_dfl_kernel.py --offline

offline:
	$(PYTHON) tests/test_dfl_kernel.py --offline

compile:
	$(PYTHON) tests/test_dfl_kernel.py --compile --skip-offline $(SHAPE) --build-dir "$(OUT)"

hardware:
	$(PYTHON) tests/test_dfl_kernel.py --hardware --skip-offline $(SHAPE) --iters $(ITERS) --warmup $(WARMUP) --build-dir "$(OUT)"
