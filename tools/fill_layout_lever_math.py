#!/usr/bin/env python3
"""Price a descriptor reduction against a container's measured floor, dispatch and G2G.

    python tools/fill_layout_lever_math.py            # defaults are SESR M7's logged figures
    python tools/fill_layout_lever_math.py --saved 158 --floor-ms 2.629 ...

Every default below is a figure from a results/ log (see docs/BENCHMARKS.md: the cadence sweep's
floor and dispatch, the shim-channel audit's descriptor count and bytes, the stream sitting's
per-column rate, the head-to-head's G2G). Nothing here is measured on a device: it is arithmetic
that says whether a proposed compiler change could be worth doing before anyone builds it.

The model is deliberately simple and its assumptions are printed with the result:
  * floor = transfer + per-task, transfer being this column's activation bytes over the measured
    rate, so the per-task unit is the residue divided by the descriptor count;
  * per-task cost is proportional to descriptor count, which the retirement-cadence measurement
    supports (identical traffic, 0.599 ms of floor between rb=4 and rb=2) but does not prove;
  * compute and the host stages (preprocess, readback, image output) are unchanged;
  * a layout that moves more bytes pays for them at the same measured rate.
So the numbers are an upper bound on the gain of any descriptor-count lever, not a forecast.
"""
import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floor-ms", type=float, default=2.629, help="measured non-compute floor")
    ap.add_argument("--dispatch-ms", type=float, default=4.311, help="measured dispatch mean")
    ap.add_argument("--g2g-ms", type=float, default=4.883, help="measured end-to-end time")
    ap.add_argument("--target-ms", type=float, default=3.820, help="the number to beat")
    ap.add_argument("--descriptors", type=int, default=1007, help="shim descriptors per dispatch")
    ap.add_argument("--act-bytes-per-col", type=int, default=2982400,
                    help="host->device activation bytes per column")
    ap.add_argument("--rate-gbps", type=float, default=6.898931, help="measured per-column rate")
    ap.add_argument("--saved", type=int, action="append", default=[], metavar="N",
                    help="descriptors removed in the scenario, repeatable")
    ap.add_argument("--extra-bytes-per-col", type=int, action="append", default=[], metavar="B",
                    help="extra bytes per column each scenario moves, same length as --saved")
    args = ap.parse_args()

    compute = args.dispatch_ms - args.floor_ms
    host = args.g2g_ms - args.dispatch_ms
    transfer = args.act_bytes_per_col / (args.rate_gbps * 1e9) * 1e3
    per_task = args.floor_ms - transfer
    unit = per_task / args.descriptors
    print(f"[model] floor {args.floor_ms:.3f} = transfer {transfer:.3f} + per-task "
          f"{per_task:.3f} over {args.descriptors} descriptors -> {unit * 1e3:.3f} us/descriptor; "
          f"compute {compute:.3f} ms, host stages {host:.3f} ms")
    if not args.saved:
        need = args.target_ms - host - compute
        print(f"[break-even] to reach {args.target_ms:.3f} ms G2G the floor must fall to "
              f"{need:.3f} ms: remove {args.floor_ms - need:.3f} ms = "
              f"{(args.floor_ms - need) / unit:.0f} descriptors = "
              f"{(args.floor_ms - need) / unit / args.descriptors * 100:.0f}% of the stream")
    for i, saved in enumerate(args.saved):
        extra = args.extra_bytes_per_col[i] if i < len(args.extra_bytes_per_col) else 0
        d_extra = extra / (args.rate_gbps * 1e9) * 1e3
        floor = args.floor_ms - saved * unit + d_extra
        disp = floor + compute
        g2g = disp + host
        verdict = (f"BEATS {args.target_ms:.3f} by {args.target_ms - g2g:.3f} ms" if g2g < args.target_ms
                   else f"still short by {g2g - args.target_ms:.3f} ms")
        print(f"[scenario] -{saved} descriptors ({saved / args.descriptors * 100:.1f}% of "
              f"{args.descriptors}), +{extra:,} B/col: floor {args.floor_ms:.3f} -> {floor:.3f}, "
              f"dispatch -> {disp:.3f}, G2G {args.g2g_ms:.3f} -> {g2g:.3f}; {verdict}")
    print("[caveat] upper bound: assumes per-task cost scales with descriptor count and that no "
          "other stage regains the time; ignores lock/barrier structure and workspace capacity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
