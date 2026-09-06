"""Times cv2.VideoCapture open cost per OpenCV videoio backend, to attribute the
~90s camera-open README.md reports for tools/webcam_multipartition_demo.py.

The open cost was originally blamed on the camera ("a driver-level renegotiation
cost specific to this camera"). That was measured with OpenCV's *default* Windows
backend only -- which is MSMF -- so it never separated "this camera is slow" from
"this backend is slow with this camera". This probe does separate them:

  msmf        default backend, i.e. what the demo does today
  msmf_nohw   MSMF with OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0, set in the
              environment *before* cv2 is imported (OpenCV reads it at videoio
              init; setting it after the import is a silent no-op, so a null
              result from that ordering would not be evidence)
  dshow       cv2.CAP_DSHOW, the older DirectShow path
  msmf_late   the same variable, but set *after* cv2 is imported -- decides
              whether the fix can live in one shared helper (order-independent)
              or has to be repeated at the top of every entry point

Each case runs in its own subprocess so its cv2 import is fresh and its env is
its own. Within a case the camera is opened twice: slow-then-fast points at a
one-off Windows Frame Server / device-enumeration warmup, slow-both at a real
per-open cost. --set-res adds the cap.set(WIDTH/HEIGHT) renegotiation the demo
deliberately avoids, so that decision can be re-checked per backend too.

    conda activate resnet_env17
    python tools/cam_probe.py                    # all three backends, native res
    python tools/cam_probe.py --set-res 1280x720 # + the resolution-change cost
    python tools/cam_probe.py --case dshow       # one backend only

Nothing here touches the NPU: it is a pure videoio measurement, and the numbers
are only comparable within one machine + camera (name both when logging).
"""
import argparse
import os
import subprocess
import sys
import time

CASES = ("msmf", "msmf_nohw", "dshow", "msmf_late")
# per-case subprocess ceiling: the known-bad path is ~90s to open and ~178s more
# to renegotiate resolution, twice over for the second open -- so allow well past that
CASE_TIMEOUT = 900.0


def case_env(case):
    env = dict(os.environ)
    if case == "msmf_nohw":
        env["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
    else:
        env.pop("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", None)
    return env


def child(case, source, set_res):
    """One backend, in a fresh interpreter: open twice, timing every step."""
    import cv2  # imported here so case_env's variable is already in place

    if case == "msmf_late":
        # deliberately after the import above -- that is the whole point of this case
        os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

    api = {"msmf": cv2.CAP_MSMF, "msmf_nohw": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW,
           "msmf_late": cv2.CAP_MSMF}[case]
    hw = os.environ.get("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "<unset>")
    print(f"case {case}: api={api} OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS={hw}",
          flush=True)

    for attempt in (1, 2):
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source, api)
        t_ctor = time.perf_counter() - t0

        t1 = time.perf_counter()
        opened = cap.isOpened()
        t_open = time.perf_counter() - t1
        if not opened:
            print(f"  open {attempt}: FAILED after {t_ctor:.2f}s", flush=True)
            cap.release()
            return 1

        t_set = 0.0
        if set_res:
            w, h = (int(v) for v in set_res.lower().split("x"))
            t2 = time.perf_counter()
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            t_set = time.perf_counter() - t2

        t3 = time.perf_counter()
        ok, frame = cap.read()
        t_read = time.perf_counter() - t3

        size = (f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        actual = f"{frame.shape[1]}x{frame.shape[0]}" if ok else "no frame"

        t4 = time.perf_counter()
        cap.release()
        t_rel = time.perf_counter() - t4

        print(f"  open {attempt}: ctor {t_ctor:7.2f}s  isOpened {t_open:5.2f}s"
              + (f"  set({set_res}) {t_set:7.2f}s" if set_res else "")
              + f"  first read {t_read:6.2f}s  release {t_rel:5.2f}s"
              f"  -> total {t_ctor + t_open + t_set + t_read:7.2f}s", flush=True)
        print(f"           reports {size} @ {fps:.1f} fps, frame {actual}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--case", choices=CASES, default=None,
                    help="run one backend only (default: all three, in order)")
    ap.add_argument("--set-res", default=None, metavar="WxH",
                    help="also time cap.set(FRAME_WIDTH/HEIGHT) to this size")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        raise SystemExit(child(args.case, args.source, args.set_res))

    cases = [args.case] if args.case else list(CASES)
    print(f"camera source {args.source}, backends {', '.join(cases)}"
          + (f", requesting {args.set_res}" if args.set_res else ", native resolution"))
    print("first case pays any cold Windows Frame Server cost -- read the two opens "
          "within a case, not just the across-case order.\n", flush=True)

    for case in cases:
        cmd = [sys.executable, "-u", __file__, "--child", "--case", case,
               "--source", args.source]
        if args.set_res:
            cmd += ["--set-res", args.set_res]
        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, env=case_env(case), timeout=CASE_TIMEOUT, check=False)
        except subprocess.TimeoutExpired:
            print(f"  case {case} exceeded {CASE_TIMEOUT:.0f}s and was killed", flush=True)
        print(f"  case {case} wall clock {time.perf_counter() - t0:.2f}s\n", flush=True)


if __name__ == "__main__":
    main()
