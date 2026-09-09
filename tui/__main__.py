"""Entry point: python -m tui

Two ways in, and the wrapper is the worse UI:

  ./scripts/tui.sh                 Git Bash. lib.sh sets the env for you, but mintty
                                   has no msvcrt, so you get numbered menus.
  python -m tui                    PowerShell, after:
                                     conda activate resnet_env17
                                     $env:RYZEN_AI_INSTALLATION_PATH = 'C:\\Program Files\\RyzenAI\\1.7.1'
                                   Arrow keys work here. The startup checks are what
                                   catch a wrong env or a 1.8.0 path before any run.
"""
# ---------------------------------------------------------------------------
# This block must stay above every other import in the process.
#
# results/cam_probe_backends.log: OpenCV's default Windows backend (MSMF) takes
# 90.02 s to construct a VideoCapture on this machine; with this variable set it
# takes 0.22 s. results/cam_probe_late_set.log: setting it *after* cv2 is imported
# is a silent no-op -- 89.99 s, indistinguishable from unset. OpenCV reads it once
# at videoio init, and every npu/ module imports cv2, so this has to happen before
# `from tui import ...` pulls any of them in.
import os as _os
_os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
# Demo stdout is heavily non-ASCII and results/ logs must be UTF-8.
_os.environ.setdefault("PYTHONUTF8", "1")
_os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# ---------------------------------------------------------------------------

import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m tui",
        description="Run this repo's models -- real tasks, or the demos behind the findings.",
        epilog="Nothing here writes to results/. Latencies shown are indicative, "
               "not doc-quotable: read docs/BENCHMARKS.md for measured numbers.")
    p.add_argument("--selftest", action="store_true",
                   help="validate the whole catalogue -- every model path, cache key and "
                        "demo command line -- and exit. Read-only: opens no hardware context.")
    p.add_argument("--plain", action="store_true",
                   help="numbered menus instead of arrow keys (forced under Git Bash anyway)")
    p.add_argument("--lane", choices=["task", "demo"], default=None,
                   help="skip the lane picker and go straight to one")
    p.add_argument("--no-guards", action="store_true",
                   help="skip the startup checks. You are on your own: a stale cache or a "
                        "foreign hardware context will not be reported.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest:
        from tui import selftest
        return selftest.run()
    from tui import app
    return app.run(plain=args.plain, lane=args.lane, guards=not args.no_guards)


if __name__ == "__main__":
    sys.exit(main())
