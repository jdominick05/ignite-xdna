# Invoke from the repository root in the activated mlir-aie ironenv.
# make -f kernels/aie2/fused_conv_epilogue/Makefile all
# The Python build uses the installed Peano --target=aie2 at -O3 and writes
# explicit xclbin/transaction/ELF artifacts plus disassembly and a hash manifest.
PYTHON ?= python
OUT ?= build/fused_conv_epilogue
CIN ?= 512
ITERS ?= 10
.NOTPARALLEL:
.PHONY: all check hardware
all:
	$(PYTHON) tests/test_fused_epilogue_kernel.py --compile --cin $(CIN) --build-dir "$(OUT)"
check:
	$(PYTHON) tests/test_fused_epilogue_kernel.py --offline --cin $(CIN)
hardware:
	$(PYTHON) tests/test_fused_epilogue_kernel.py --hardware --cin $(CIN) --iters $(ITERS) --build-dir "$(OUT)"
