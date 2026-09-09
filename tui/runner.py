"""Launching demos as subprocesses, and building the command line that does it.

demos/*.py are standalone: they own their cv2 windows, and the numbered pipeline
entry points start with a digit precisely so they cannot be imported. So they get
run, not called -- and the argv is shown to the user first, because a launcher that
hides --fresh and --ep is worse than the CLI it replaces.

build_argv() is pure and takes no I/O, which is what lets --selftest check every
demo's command line without launching anything.
"""
import os
import subprocess
import sys

from npu.paths import ROOT

from tui.registry import Entry


def demo_out_dir() -> str:
    """Absolute, and under outputs/ rather than results/.

    Every demo defaults --out-dir to results/, and width_ladder_demo.py resolves it
    against the current working directory instead of the repo root. Passing an
    absolute path under outputs/ fixes both: results/ stays the evidence base, and
    the CWD quirk stops mattering.
    """
    return str(ROOT / "outputs" / "demos")


def build_argv(entry: Entry, ep=None, source=None, fresh=False, extra=()) -> list:
    """The exact command line for a demo, honouring that script's own flag dialect."""
    if entry.lane != "demo":
        raise ValueError(f"{entry.key} is not a demo")
    d = entry.dialect
    argv = [sys.executable, str(ROOT / entry.script)]

    if d.ep_flag and ep:
        argv += [d.ep_flag, ep]
    if d.source_flag and source:
        argv += [d.source_flag, str(source)]
    if d.out_dir_flag:
        argv += [d.out_dir_flag, demo_out_dir()]
    if d.fresh_flag and fresh:
        argv.append(d.fresh_flag)
    argv += list(d.extra)
    argv += list(extra)
    return argv


def child_env() -> dict:
    """The environment a demo subprocess needs.

    OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS has to be in place before the child
    imports cv2, which means putting it in the child's environment -- inheriting it
    is fine, setting it inside the child after import is not (a 90 s camera open
    versus 0.22 s; results/cam_probe_late_set.log).
    """
    env = dict(os.environ)
    env.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # The EP writes its report only when this is set. build_session setdefaults it
    # too, but a demo that builds a session another way would silently skip it.
    env.setdefault("XLNX_ONNX_EP_REPORT_FILE", "vitisai_ep_report.json")
    return env


def run(entry: Entry, argv: list, echo=print) -> int:
    """Run a demo, streaming its output. Returns the child's exit status.

    Output is merged (stderr into stdout) and decoded as UTF-8, matching what
    run_logged does in scripts/lib.sh -- the demos print box-drawing characters and
    arrows, and a mis-decoded stream is the kind of thing that silently breaks a
    later grep rather than raising.
    """
    os.makedirs(demo_out_dir(), exist_ok=True)
    proc = subprocess.Popen(
        argv, cwd=str(ROOT), env=child_env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in proc.stdout:
            echo(line.rstrip("\n"))
    except KeyboardInterrupt:
        proc.terminate()
        echo("\ninterrupted -- terminating the demo")
    finally:
        proc.wait()
    return proc.returncode
