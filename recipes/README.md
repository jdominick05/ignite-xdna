# Build recipes

`models/` and `build/` are not tracked: they cost hours and gigabytes to make, and Syncthing carries `models/` in from
Desktop 1. So until now, how a model or a container was made lived only in old commit bodies. Rebuilding one meant
finding its flags there first. Every model here has a recipe instead: one TOML file per model, from the source weights
to what the NPU runs, step by step.

```bash
python tools/recipe.py list                    # every recipe, its status and steps
python tools/recipe.py show yolov8n            # the exact commands
python tools/recipe.py build yolov8n --dry-run # what would run
python tools/recipe.py build yolov8n --step compile-overlap
python tools/recipe.py check                   # offline: recipes parse, scripts exist, compile flags parse
python tools/recipe.py status                  # outputs present, and containers in build/ no recipe makes
```

A graph-engine container also carries its own build record. `manifest["build"]` holds the command, the compiler's
`git describe`, the Python version, the UTC time and the recipe step that ran it (`yolov8n:compile-overlap`). Beside
it, `source_model_sha256` holds the input ONNX's hash. Containers built before 2026-10-19 have neither the record nor,
before mid-September, the hash.

## The format

```toml
name = "yolov8n"                 # must match the file name
description = "YOLOv8n detect, head cut, XINT8; the Ignition golden model"
status = "verified"              # verified | reconstructed | blocked
evidence = ["c16daca8", "results/aie/..."]   # where each command was found

[verified]                       # required when status = "verified"
date = "2026-10-19"
machine = "Desktop 2"
check = "verify_engine_container.py: every layer EXACT"

[[step]]
id = "compile-overlap"
env = "iron"                     # shell | engine | iron | resnet_env | resnet_env17
run = ["-m", "ignite_xdna.compiler.cli", "--engine", "graph", "--input", "models/yolov8n_cut_xint8.onnx",
       "--output", "build/df_yolov8n_silu_def.ignite", "--silu-sigmoid"]
inputs = ["models/yolov8n_cut_xint8.onnx"]
outputs = ["build/df_yolov8n_silu_def.ignite"]
npu = false                      # true makes build refuse while the NPU is busy
note = "the layer-overlap default (2026-10-15)"
```

`run` is the argument list after `python`, except in `shell` steps. `iron` runs it through
`scripts/research-iron.sh`, which the graph-engine compiler needs. `resnet_env` and `resnet_env17` are activated
through `scripts/lib.sh`, because Quark needs the env's `Scripts\` on PATH. Every step runs from the checkout root
with `MSYS_NO_PATHCONV=1`, so a node name such as `/model.10/...` reaches the compiler unmangled.

**`status` is load-bearing.**
- `verified`: the recipe was run as written and its output checked. The `[verified]` table says when, where and how.
- `reconstructed`: assembled from history and not yet re-run. Such a recipe can be wrong; re-run it and promote it.
- `blocked`: a step is known not to work today, and `blocked_by` says why.

A step whose outputs all exist is skipped. A step with only some of its outputs refuses to run, and `--force`
rebuilds. Quantization is not bit-reproducible across Quark versions or machines, so a rebuilt `models/*.onnx` can
differ from the one a published number was measured on. Compare `source_model_sha256` before calling two
containers the same model.

When a model or container changes how it is made, change its recipe in the same commit.
`tests/test_recipes_offline.py` keeps every recipe parseable and its compile flags valid.
