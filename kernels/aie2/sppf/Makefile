# From repository root; wrapper activates the installed Windows Peano/IRON environment.
ITERS ?= 100
.NOTPARALLEL:
.PHONY: all hardware check
all:
	bash scripts/sppf.sh --compile
hardware:
	bash scripts/sppf.sh --hardware --iters $(ITERS)
check:
	bash scripts/sppf.sh --offline
