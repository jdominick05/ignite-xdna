---
license: agpl-3.0
pretty_name: ignite-xdna measurement logs
tags:
- ryzen-ai
- xdna
- npu
- benchmark
- logs
---

# ignite-xdna measurement logs

This dataset is the `results/` directory of [ignite-xdna](https://github.com/jdominick05/ignite-xdna) at commit
`{{SHA}}`, exactly as git tracks it. ignite-xdna is a hardware-characterization study of AMD's XDNA1 NPU
(Phoenix / Hawk Point, Windows) and a bare-metal AIE2 engine that grew out of it. Every figure in its documents
is backed by a log here.

The logs are evidence, not a benchmark suite you can rerun from this dataset alone.

- **What each number means, and its method and caveats:** `docs/BENCHMARKS.md` and
  `docs/MODEL_ZOO_BENCHMARKS.md` in the code repository. Read a figure there before quoting it from a log.
- **Index of logs:** `results/README.md` (and `results/aie/README.md` for the AIE-kernel logs). Each index entry
  says which document section the log backs, and which logs were retracted or superseded. Superseded logs are
  kept on purpose.
- **Machine:** each log names the machine it ran on (Desktop 2 is a Ryzen 7 8700G with a Phoenix NPU; the laptop
  is a Ryzen 5 8645HS with a Hawk Point NPU). Do not compare latencies across machines or across days without
  saying so. NPU latency on these machines drifts between sessions with no code change.
- **Paths:** the local profile path is scrubbed to `C:\Users\<user>`.

The published models and compiled `.ignite` containers are at
[jdominick05/ignite-xdna](https://huggingface.co/jdominick05/ignite-xdna). Their device verification is
`results/hf/verify_published_containers_desktop2_20261013.log`.

Licence: AGPL-3.0, the same licence as the code repository these logs come from.
