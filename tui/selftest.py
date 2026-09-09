"""Validate the whole catalogue without touching the hardware.

There is no unit-test suite in this repo and there cannot be one -- the hardware
cannot be faked and a mocked NPU session would test nothing. This is the closest
honest substitute: it resolves every model path, every cache key, every sample
input and every demo command line, and says which of them are actually present on
this machine.

It opens no session, so it never holds a hardware context and can be run while
something else is using the NPU.
"""
import importlib
from pathlib import Path

from npu.paths import ROOT

from tui import registry, runner

# Which npu/ module backs each task family. The adapters import these; the
# selftest only checks they exist, so a typo here fails loudly and early rather
# than at the moment someone picks that task.
FAMILY_MODULES = {
    "modnet": ("npu.modnet",),
    "yolov8": ("npu.yolo", "npu.yolo_decode"),
    "yolov6": ("npu.yolov6", "npu.yolov6_decode"),
    "yolo_pose": ("npu.yolo_pose", "npu.yolo_pose_decode"),
    "fastdepth": ("npu.fastdepth",),
    "midas": ("npu.midas",),
    "sesr": ("npu.sesr",),
    "realesrgan": ("npu.realesrgan",),
    "bisenetv2": ("npu.bisenetv2",),
}

DOT = "--"


def _task_families():
    """Families tui/task.py actually has an adapter for.

    Importing tui.task pulls in cv2 and onnxruntime, which is slow but opens no
    hardware context -- and it doubles as a check that the adapters import at all.
    """
    try:
        from tui.task import DISPATCH
        return set(DISPATCH)
    except Exception as exc:                              # noqa: BLE001
        print(f"warning: tui/task.py does not import ({exc}); "
              "adapter coverage unchecked")
        return None


def _check_outputs_never_touch_results() -> list:
    """No path the launcher hands a child may resolve inside results/.

    results/ is the tracked evidence base -- 69 of its images are cited from the
    docs, and every demo defaults --out-dir to it. adaround_diff_demo.py writes
    exactly results/adaround_diff_yolov8n_npu.jpg, which demos/README.md links.
    Launching that through the TUI without an explicit --out-dir would overwrite
    cited evidence, so this asserts the redirect is actually in place.
    """
    from npu.paths import RESULTS
    out = Path(runner.demo_out_dir()).resolve()
    problems = []
    try:
        if out.is_relative_to(RESULTS.resolve()):
            problems.append(f"demo output dir {out} is inside results/")
    except AttributeError:                                # py<3.9
        if str(out).startswith(str(RESULTS.resolve())):
            problems.append(f"demo output dir {out} is inside results/")
    if not out.is_relative_to((ROOT / "outputs").resolve()):
        problems.append(f"demo output dir {out} is not under outputs/")
    return problems


def _check_task(entry, families=None) -> list:
    problems = []
    mods = FAMILY_MODULES.get(entry.family)
    if not mods:
        problems.append(f"no npu module mapped for family {entry.family!r}")
    else:
        for m in mods:
            try:
                importlib.import_module(m)
            except Exception as exc:                      # noqa: BLE001 - report, don't crash
                problems.append(f"{m} does not import: {exc}")

    if not entry.models:
        problems.append("no models listed")

    for mc in entry.models:
        if not mc.path.is_file():
            problems.append(f"model absent: models/{mc.relpath}")
        try:
            key = registry.cache_key_for(entry, mc)
        except Exception as exc:                          # noqa: BLE001
            problems.append(f"cache key unresolvable for {mc.relpath}: {exc}")
            continue
        if not key:
            problems.append(f"empty cache key for {mc.relpath}")

    if families is not None and entry.family not in families:
        problems.append(f"no adapter in tui/task.py for family {entry.family!r}")
    if entry.sample and not (ROOT / entry.sample).exists():
        problems.append(f"sample input absent: {entry.sample}")
    if entry.default_ep and entry.default_ep not in entry.eps:
        problems.append(f"default_ep {entry.default_ep!r} not in eps {entry.eps}")
    return problems


def shared_cache_keys(entry) -> dict:
    """Which of an entry's models share a compile cache, and so cannot both be
    cached at once.

    This is normal and not a bug -- yolocutcachekey serves every yolov8 cut
    variant, exactly as the pipelines use it. What makes it safe is that switching
    model forces a recompile: tui/task.py compares the model against the cache's
    own context.json and passes --fresh when they differ. So this is reported as a
    cost ("switching model here costs a full recompile"), not as a failure. It
    would only be a defect if something let a switch reuse the cache silently.
    """
    groups = {}
    for mc in entry.models:
        try:
            groups.setdefault(registry.cache_key_for(entry, mc), []).append(mc.relpath)
        except Exception:                                 # noqa: BLE001
            continue
    return {k: v for k, v in groups.items() if len(v) > 1}


def _check_demo(entry) -> list:
    problems = []
    script = ROOT / entry.script
    if not script.is_file():
        problems.append(f"script absent: {entry.script}")
    if entry.sample and entry.sample != "0" and not (ROOT / entry.sample).exists():
        problems.append(f"sample input absent: {entry.sample}")
    try:
        argv = runner.build_argv(entry, ep=entry.default_ep or None,
                                 source=entry.sample or None)
        if len(argv) < 2:
            problems.append("built an empty command line")
    except Exception as exc:                              # noqa: BLE001
        problems.append(f"command line does not build: {exc}")
    if entry.default_ep and entry.default_ep not in entry.eps:
        problems.append(f"default_ep {entry.default_ep!r} not in eps {entry.eps}")
    return problems


def run() -> int:
    """Print the catalogue with a verdict per entry. Returns a shell exit status."""
    from tui.ui import console, rule
    from rich.table import Table

    rule("Catalogue self-test")
    console.print("[dim]Read-only. No session is built and no hardware context is "
                  "opened.[/dim]\n")

    failed = []
    families = _task_families()

    safety = _check_outputs_never_touch_results()
    if safety:
        failed.append(("output paths", safety))
        console.print("[red]output-path safety: " + "; ".join(safety) + "[/red]\n")
    else:
        console.print(f"[green]output paths safe[/green] -- demos are redirected to "
                      f"{runner.demo_out_dir()}, nothing lands in results/\n")

    for lane_name, title in (("task", "Tasks"), ("demo", "Demos")):
        table = Table(title=title, title_justify="left", header_style="bold",
                      expand=False, pad_edge=False)
        table.add_column("entry", style="cyan", no_wrap=True)
        table.add_column("what")
        table.add_column("models", justify="right")
        table.add_column("cost of switching")
        table.add_column("status")

        for entry in registry.lane(lane_name):
            problems = (_check_task(entry, families) if entry.is_task
                        else _check_demo(entry))
            if entry.is_task:
                have = sum(1 for m in entry.models if m.path.is_file())
                count = f"{have}/{len(entry.models)}"
                shared = shared_cache_keys(entry)
                note = (f"{max(len(v) for v in shared.values())} models share "
                        f"{', '.join(shared)} -- switching recompiles"
                        if shared else "")
            else:
                count = DOT
                note = (f"{entry.recompiles} recompile(s)" if entry.recompiles
                        else "")
            status = ("[green]OK[/green]" if not problems
                      else "[red]" + "; ".join(problems) + "[/red]")
            if problems:
                failed.append((entry.key, problems))
            table.add_row(entry.key, entry.title, count, f"[dim]{note}[/dim]", status)
        console.print(table)
        console.print()

    total = len(registry.ALL)
    if failed:
        console.print(f"[red]{len(failed)} of {total} entries have problems.[/red] "
                      "A missing model is normal on a machine that has not built it "
                      "(models/ is a Syncthing folder); a missing script or an "
                      "unresolvable cache key is a bug in tui/registry.py.")
        return 1
    console.print(f"[green]all {total} entries resolve[/green] -- "
                  f"{len(registry.TASKS)} tasks, {len(registry.DEMOS)} demos")
    return 0
