"""Build a ring container from layers CHOSEN BY SHAPE, to hold the ring's width constant.

Why this exists. The ring is byte-exact and re-runnable when its width never changes (a one-layer container at
width 1 survives twenty repeat dispatches), and it breaks when the width changes (two layers at widths 1 then 2
are exact on the first dispatch and time out on a repeat). Testing that properly needs several layers at ONE
width, and no prefix of the graph gives you that - the first eight layers run widths 1, 2, 1, 1, 2, 2, 4, 1.

What this does NOT do, deliberately: it does not take layers off the ring. ``ring_plan`` returns a plan for
every layer that moves packets, because the ring REPLACES the split ObjectFifo rather than sitting beside it -
a layer left on the per-group schedule would fetch activations that nothing ever serves, which hangs by
construction and would look exactly like the fault under investigation. So selection happens at the schedule,
by dropping whole layers from the build.

THE LIMITATION, stated so a result is not over-read: the graph is sequential, so dropping an interior layer
leaves the next one reading a workspace region this frame never wrote. Layers downstream of a gap compute from
stale bytes and CANNOT be expected to match the reference. Byte-exactness is only meaningful for a selected
layer whose predecessor was also built; for everything else the signal is hang versus no-hang, which is the
question this is asked to answer.

The guard matters as much as the selection. A filter that silently matched everything would emit an ordinary
ring build and its clean run would read as a pass, so this refuses to write a container unless the selection
actually dropped layers, and prints exactly which ones it kept.
"""
import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="checkout whose compiler to use")
    ap.add_argument("--model", required=True, help="quantized ONNX, relative to --root or absolute")
    ap.add_argument("--out", required=True, help="container to write")
    ap.add_argument("--activation-ring", type=int, required=True, help="ring slots (the compiled window)")
    ap.add_argument("--width", type=int, required=True,
                    help="keep only layers whose ring pass width (RingPlan.capacity) is exactly this")
    ap.add_argument("--serves", type=int, default=None,
                    help="also require this replay count, so the lock values are constant too")
    ap.add_argument("--max-layers", type=int, default=None, help="keep at most this many of the matches")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    sys.path[0:0] = [str(root / "src"), str(root)]
    import ignite_xdna
    where = Path(ignite_xdna.__file__).resolve()
    if root not in where.parents:
        raise SystemExit(f"ignite_xdna resolved to {where}, not under {root}")
    print(f"compiler: {where}")

    from ignite_xdna.compiler.engine_compile import compile_graph_container
    from ignite_xdna.compiler import engine_schedule as es
    from ignite_xdna.compiler.graph_ir import lower_yolov8n

    model = Path(args.model)
    model = model if model.is_absolute() else root / model

    # Decide the selection from a lowering of our own, then let the compiler do its own lowering: the model is
    # the same, so layer indices agree.
    ir = lower_yolov8n(str(model))
    shapes = {}
    for layer in ir.layers:
        plan = es.ring_plan(ir, layer, args.activation_ring)
        if plan is not None:
            shapes[layer.index] = plan

    selected = [i for i, p in sorted(shapes.items())
                if p.capacity == args.width and (args.serves is None or p.serves == args.serves)]
    if args.max_layers is not None:
        selected = selected[:args.max_layers]

    if not selected:
        widths = sorted({(p.capacity, p.serves) for p in shapes.values()})
        raise SystemExit(f"no layer has width {args.width}"
                         + (f" and serves {args.serves}" if args.serves is not None else "")
                         + f"; available (capacity, serves) pairs: {widths}")
    # The guard: a selection that keeps everything is not a selection, and its container would be an ordinary
    # ring build whose clean run would be read as a result about constant width.
    if len(selected) == len(shapes):
        raise SystemExit(f"selection kept all {len(shapes)} ring layers - that is not a fixed-width test")

    keep = set(selected)
    print(f"ring layers in the graph: {len(shapes)}; kept {len(keep)} at width {args.width}"
          + (f", serves {args.serves}" if args.serves is not None else ""))
    contiguous_from_start = selected[0] == 0 and selected == list(range(len(selected)))
    for i in selected:
        p = shapes[i]
        gap = "" if (i == 0 or (i - 1) in keep) else "   <- predecessor not built: data is stale here"
        print(f"  L{i:2d} {ir.layers[i].name[:48]:<50} chunks {p.chunks:>3}  width {p.capacity:>2}  "
              f"serves {p.serves:>2}  passes {p.passes}{gap}")
    if not contiguous_from_start:
        print("  NOTE: this build has gaps, so only the hang/no-hang signal is valid - mismatches are expected.")

    # Filter at the schedule. compile_graph_container looks es.schedule_graph up on the module at call time,
    # so replacing it here reaches the real scheduler; `layers` stays None so nothing is truncated as a prefix.
    real_schedule_graph = es.schedule_graph
    stats = {"before": 0, "after": 0}

    def schedule_graph_selected(ir_arg, ws_arg, activation_ring: int = 0):
        scheds, store = real_schedule_graph(ir_arg, ws_arg, activation_ring=activation_ring)
        stats["before"] = len(scheds)
        kept = [s for s in scheds if s.layer_index in keep]
        stats["after"] = len(kept)
        return kept, store

    es.schedule_graph = schedule_graph_selected
    try:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest = compile_graph_container(str(model), str(out), layers=None,
                                           activation_ring=args.activation_ring)
    finally:
        es.schedule_graph = real_schedule_graph

    # Second half of the guard: prove the wrapper actually ran and actually dropped layers.
    if stats["after"] == 0 or stats["after"] == stats["before"]:
        raise SystemExit(f"the schedule filter did not fire as intended "
                         f"(before {stats['before']}, after {stats['after']}); the container is not a "
                         f"fixed-width build - do not dispatch it")
    print(f"schedule filtered: {stats['before']} -> {stats['after']} layers")
    graph = manifest.get("graph_engine", {})
    n = graph.get("layers")
    print(f"wrote {out} ({out.stat().st_size:,} B); {len(n) if isinstance(n, list) else n} layers")


if __name__ == "__main__":
    main()
