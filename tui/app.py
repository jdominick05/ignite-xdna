"""The menu loop.

Everything that touches the NPU is a subprocess: demos are demos, and a task is
`python -m tui.task`. This process never builds a session, so it never holds a
hardware context of its own -- which matters because the contention check would
otherwise flag the launcher itself, and because a DPU timeout would take the UI
down with it.
"""
import os
import subprocess
import sys
from pathlib import Path

from npu.paths import ROOT

from tui import guards, registry, runner, ui

LOCK = ROOT / "outputs" / ".npu.lock"


def _acquire_lock() -> bool:
    """One run at a time. The device is single-tenant and the demos share compile
    cache keys they each delete on entry, so two at once is not slow -- it is wrong."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip())
        except (OSError, ValueError):
            pid = -1
        if pid > 0 and _pid_alive(pid):
            ui.console.print(f"[red]another TUI run is active (pid {pid}).[/red] "
                             "The NPU is single-tenant; wait for it to finish.")
            return False
        LOCK.unlink(missing_ok=True)          # stale lock from a killed run
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _pid_alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True).stdout
    return str(pid) in out


def _release_lock():
    LOCK.unlink(missing_ok=True)


def preflight(entry, ep, model_choice=None) -> bool:
    """Everything the user should know before this run starts. Returns False to abort."""
    lines = []

    if entry.recompiles:
        lines.append(f"[yellow]{entry.recompiles} full NPU recompile(s)[/yellow] -- "
                     f"that, not inference, is most of the wall clock here")

    # Six demos call clear_cache() unconditionally, so even --ep cpu destroys the
    # NPU compile and the next NPU run pays for it. Only a launcher can say this.
    if entry.clears_cache_on_cpu and ep == "cpu":
        keys = ", ".join(entry.touches_cache_keys)
        lines.append(f"[yellow]this CPU run still wipes the NPU cache[/yellow] ({keys}) -- "
                     "the demo clears it unconditionally, so your next NPU run recompiles")
    elif entry.touches_cache_keys and ep == "npu":
        lines.append("wipes and rebuilds: " + ", ".join(entry.touches_cache_keys))

    if entry.needs_person:
        lines.append("[yellow]needs a webcam and a person at the machine[/yellow]")
    for where in entry.writes_camera_frames:
        lines.append(f"[red]writes camera frames to[/red] {where}")

    if model_choice is not None and ep == "npu":
        key = registry.cache_key_for(entry, model_choice)
        check = guards.cache_status(key, model_choice.path)
        style = ui.STATUS_STYLE.get(check.status, "")
        lines.append(f"[{style}]{check.detail}[/{style}]")

    if ep == "npu":
        for check in (guards.check_contention(), guards.check_host_load()):
            style = ui.STATUS_STYLE.get(check.status, "")
            lines.append(f"{check.name}: [{style}]{check.detail}[/{style}]")
    elif ep in ("cpu", "dml"):
        lines.append(f"[dim]{ep}: no EP report exists for this provider, so "
                     "'did it engage' cannot be answered the way it can for the NPU[/dim]")

    ui.panel("\n".join(lines) if lines else "nothing to flag", title="Before this runs")
    return ui.confirm("Run it?", default=True)


def _pick_ep(entry):
    if not entry.eps:
        return None                      # the script chooses its own providers
    if len(entry.eps) == 1:
        return entry.eps[0]
    labels = {"npu": "NPU (XDNA1)", "dml": "iGPU (DirectML)", "cpu": "CPU"}
    ep = ui.choose("Where should it run?", list(entry.eps),
                   describe=lambda e: (labels.get(e, e),
                                       "default" if e == entry.default_ep else ""))
    return ep


def _pick_model(entry):
    if len(entry.models) == 1:
        return entry.models[0]
    return ui.choose("Which model?", list(entry.models),
                     describe=lambda m: (m.label,
                                         m.note + ("" if m.path.is_file()
                                                   else "  [not built here]")))


def _pick_input(entry):
    live = bool({"webcam", "video"} & set(entry.inputs))
    if "image" not in entry.inputs and not live:
        return entry.sample or None
    default = entry.sample
    if live:
        # The registry has declared "webcam" on six task entries since the launcher
        # landed, and this prompt never said so -- a camera index was a legal answer
        # nothing told the user about. Say it, and say which camera 0 is.
        kinds = "path, camera index (0 = default camera)"
        if "video" in entry.inputs:
            kinds = "path, video file, camera index (0 = default camera)"
        raw = ui.ask(f"Input ({kinds}, or Enter for the sample):", default)
    else:
        raw = ui.ask("Input image (path, or Enter for the sample):", default)
    return raw


def run_task(entry):
    choice = _pick_model(entry)
    if choice is None:
        return
    if not choice.path.is_file():
        ui.console.print(f"[red]{choice.relpath} is not on this machine.[/red] "
                         "models/ is a Syncthing folder -- build it, or wait for it to sync.")
        return
    ep = _pick_ep(entry)
    if ep is None:
        return
    src = _pick_input(entry)

    argv = [sys.executable, "-m", "tui.task", entry.key,
            "--model", choice.relpath, "--ep", ep]
    if src:
        argv += ["--input", str(src)]

    ui.clear_screen()
    ui.rule(entry.title)
    ui.console.print(f"\n[dim]$ {' '.join(argv[1:])}[/dim]\n")
    if not preflight(entry, ep, choice):
        return
    subprocess.run(argv, cwd=str(ROOT), env=runner.child_env())


def run_demo(entry):
    ep = _pick_ep(entry)
    if entry.eps and ep is None:
        return
    src = entry.sample if "image" in entry.inputs else None
    if "webcam" in entry.inputs:
        src = ui.ask("Camera index or video path:", entry.sample or "0")

    argv = runner.build_argv(entry, ep=ep, source=src)

    ui.clear_screen()
    ui.rule(entry.title)
    ui.console.print(f"\n[dim]$ {' '.join(argv[1:])}[/dim]\n")
    if not preflight(entry, ep or "npu"):
        return
    # Nothing clears from here on: the demo's own output, and the EP verdict after
    # it, stay on screen until the user presses Enter to go back.
    rc = runner.run(entry, argv, echo=ui.console.print)

    if (ep or entry.default_ep) == "npu":
        for key in entry.touches_cache_keys:
            ui.console.print(f"[bold]{key}[/bold]: {guards.ep_verdict(key)}")
    ui.console.print(f"\n[dim]exit {rc}; anything written is under "
                     f"{runner.demo_out_dir()}[/dim]")


def device_screen():
    ui.clear_screen()
    ui.rule("Device")
    ui.render_checks(guards.startup_checks(need_1x4=True))
    from rich.table import Table
    rows = guards.all_caches()
    if not rows:
        ui.console.print("\n[dim]no compile caches at the repo root yet[/dim]")
        return
    table = Table(title=f"\nCompile caches ({len(rows)})", title_justify="left",
                  header_style="bold", box=None, pad_edge=False)
    table.add_column("cache key", style="cyan", no_wrap=True)
    table.add_column("compiled from")
    table.add_column("EP verdict")
    for name, src, verdict in rows:
        table.add_row(name, src or "[dim]unrecorded[/dim]", str(verdict))
    ui.console.print(table)
    ui.console.print("\n[dim]'compiled from' is the cache's own context.json. A run whose "
                     "model differs from that line reuses the wrong compile unless "
                     "--fresh is passed.[/dim]")


def run(plain=False, lane=None, guards_on=True, **kw):
    guards_on = kw.get("guards", guards_on)
    if guards_on:
        ui.clear_screen()
        ui.rule("Startup")
        checks = guards.startup_checks()
        ui.render_checks(checks)
        if any(c.blocks_npu for c in checks):
            ui.console.print("\n[yellow]NPU runs will not work until the above is "
                             "fixed. CPU and iGPU still will.[/yellow]")
        # The first menu clears this panel, so hold only when there is something
        # worth reading. An all-clear startup goes straight through -- nothing is
        # lost, and it saves a keystroke on every launch.
        if any(c.status != guards.OK for c in checks):
            ui.ask("\nEnter to continue", "")
        else:
            ui.console.print()

    if not _acquire_lock():
        return 1
    try:
        while True:
            if lane:
                entries = list(registry.lane(lane))
                title = "Tasks" if lane == "task" else "Demos"
            else:
                pick = ui.choose(
                    "What do you want to do?",
                    ["task", "demo", "device"],
                    describe=lambda k: {
                        "task": ("Run a model on my own input",
                                 "matte, detect, pose, depth, upscale, segment"),
                        "demo": ("Run a demo", "the ten findings in demos/"),
                        "device": ("Device status", "what the EP took, per cache"),
                    }[k], plain=plain, allow_back=True)
                if pick is None:
                    return 0
                if pick == "device":
                    device_screen()
                    ui.ask("\nEnter to go back", "")
                    continue
                entries = list(registry.lane(pick))
                title = "Tasks" if pick == "task" else "Demos"

            entry = ui.choose(title, entries,
                              describe=lambda e: (e.title, e.blurb), plain=plain)
            if entry is None:
                if lane:
                    return 0
                continue
            if entry.is_task:
                run_task(entry)
            else:
                run_demo(entry)
            ui.ask("\nEnter to go back", "")
    except KeyboardInterrupt:
        ui.console.print("\ninterrupted")
        return 130
    finally:
        _release_lock()
