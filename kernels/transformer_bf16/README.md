# Standalone BF16 transformer operations

Experimental single-token Phoenix kernels and a reusable `Bf16Ops` host wrapper.
Model arithmetic executes on the NPU through `ignite_xdna.runtime.driver`.
The host packs raw BF16 bits and dispatches each operation. Host round trips
remain in this baseline. These kernels are outside the ONNX
compiler and do not add operations or precision modes to `.ignite` containers.

| Method | Supported layout and semantics |
| --- | --- |
| `projection` | One BF16 vector times BF16 weights `[N,K]`; pads K to 256 and N to 16; FP32 chunk carry, BF16 output |
| `norm` | RMSNorm, epsilon `1e-6`; width 256 or 512, at most 2048 elements; FP32 normalization, BF16 output |
| `rope` | Four query heads, width 256 or 512; supplied BF16 cosine/sine; both product roundings preserved |
| `softmax` | BF16 score rows, each 1 to 32768 elements; streamed statistics, BF16 probabilities; exp tails below -80 become zero |
| `geglu` | 2048-element BF16 gate and up vectors; supplied lookup table for tanh GELU; rounds activation before multiplication |
| `residual` | BF16 addition followed by BF16 scalar multiplication; at most 2048 elements, multiple of 16 |
| `topk` | At most 4096 finite BF16 scores, k from 1 to 32; descending value, ascending index on ties |

Inputs and outputs use NumPy `uint16` arrays containing BF16 bits, not numerically
converted integers. `common.py` provides explicit bit conversions for reference
generation. Kernel entry symbols retain their original `draft_` ABI names; the
operations have no dependency on a model checkpoint or the external chat workspace.

```python
from kernels.transformer_bf16 import Bf16Ops

ops = Bf16Ops(cache_dir="scratch/bf16-cache")
y_bits = ops.projection(x_bits, weight_bits)
```

`precise_projection=True` selects a second GEMV implementation with compensated
FP32 accumulation and BF16 operands/output. It is an accuracy experiment; it
does not guarantee bit agreement with a CPU library's accumulation order.
The original matrix-unit implementation remains the default.

To reuse instructions and buffers, pass an explicitly owned session:

```python
from kernels.transformer_bf16.persistent import PacketSession

with PacketSession(max_contexts=2) as transport:
    ops = Bf16Ops("scratch/bf16-cache", transport=transport)
    y_bits = ops.projection(x_bits, weight_bits)
```

The caller must reserve exclusive NPU use for the session. Idle witnesses bracket
it; the research wrapper monitors foreign contexts during bounded checks. An LRU
cache retains at most two contexts with instruction/data BOs. Eviction and
context-manager exit release owned BOs before contexts. Image admission verifies
source/artifact hashes; successful exit checks that all admitted files remain
unchanged. Replays use already-loaded BOs. Each operation checks output guards
and input preservation. Calls are
synchronous and restricted to the thread that opened the session.

Each image records source and compiler hashes, transaction and xclbin hashes,
code size, stack frames and SRAM spans. Standalone dispatch checks hashes and
idle state each time; cached sessions check these at the boundaries described
above. Construct a new ops/session instance to adopt edited sources or images.
Build artifacts belong in `scratch/`, not alongside these sources. Toolchain paths
can be set with `MLIR_AIE_IRON`, `PEANO_INSTALL_DIR`, `MLIR_AIE_INSTALL_DIR` and
`XRT_ROOT`; defaults match the existing Windows research environment.

Prepare independent references in a torch environment:

```powershell
python kernels/transformer_bf16/probe.py prepare --fixtures scratch/bf16-fixtures
```

Run checks in the PyXRT environment, through the bounded research wrapper:

```bash
bash scripts/research-lowlevel.sh --npu --checks-only --seconds 180 --rss-gib 6 \
  --log results/aie/transformer_bf16_checks.log -- \
  python kernels/transformer_bf16/probe.py run --fixtures scratch/bf16-fixtures \
  --cache scratch/bf16-cache --out scratch/bf16-checks.json
```

The preparation checks the GELU table against every finite BF16 input. Device
checks cover each operation against independent CPU references. Passing these
checks establishes individual-operation correctness within the recorded criteria;
it does not establish whole-model accuracy, fused execution, serving latency or
hybrid GPU/NPU benefit. Model orchestration and checkpoint conversion stay with
their respective applications.

The [bounded device check](../../results/aie/transformer_bf16_checks_20261018_v2.log)
and [structured results](../../results/aie/transformer_bf16_checks_20261018.json)
are indexed with [method and limits](../../docs/BENCHMARKS.md#standalone-bf16-transformer-primitives-2026-10-18-desktop-2).

The cached transport has a separate replay/eviction check:

```bash
bash scripts/research-lowlevel.sh --npu --checks-only --seconds 180 --rss-gib 6 \
  --log results/aie/transformer_bf16_cached_checks.log -- \
  python kernels/transformer_bf16/session_probe.py --fixtures scratch/bf16-fixtures \
  --cache scratch/bf16-cached --out scratch/bf16-cached-checks.json --precise-projection
```

Its [28 passing checks](../../results/aie/transformer_bf16_cached_transport_20261018.log)
include three distinct inputs and an exact repeat of the first for each operation.
The consuming application's complete draft still fails its broader numerical
qualification; see the [full record](../../docs/BENCHMARKS.md#complete-draft-composition-and-cached-transport).

An experimental opcode dispatcher includes the seven operations in one core
program. Enable it with `single_image=True, precise_projection=True` on `Bf16Ops`
and `share_program=True` on `PacketSession`. Context reuse across packet counts
requires byte-identical PDI, kernel argument metadata and memory topology. A
matching ELF alone is insufficient. Each context retains at most 32 distinct
transaction/data-BO sets; allocation uses that context's full XRT group IDs.
This is a shared program, with host round trips between operations, not a fused
decoder graph. Its size-optimized build stays within the 16 KiB program limit.

Add `--single-image --check-drift` to the cached probe to test opcode routing and
successful-exit rejection of a changed cached manifest. Only the probe's own
cache manifest is temporarily changed and then restored.
