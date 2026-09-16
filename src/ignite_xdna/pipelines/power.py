"""Power modes for the host side of the engine: how the native preprocessor's OpenMP workers use the CPU.

A graph-engine frame is a few short parallel regions on the CPU around a 7 ms NPU dispatch. How the workers wait
between those regions decides most of what a frame costs, and thread count decides the rest. Three modes trade one
against the other, and every thread count is a fraction of the host's own cores, never a fixed number:

=================  ==============  =====================================
mode               workers wait    workers
=================  ==============  =====================================
``performance``    spinning        every logical processor
``balanced``       sleeping        one per physical core
``efficiency``     sleeping        a quarter of the physical cores, >= 1
=================  ==============  =====================================

``balanced`` is the default. Set ``IGNITE_XDNA_POWER_MODE`` to choose, BEFORE ``ignite_xdna`` is first imported:
importing any part of the package loads the native preprocessor, and the OpenMP runtime reads these settings once,
when it initialises. An explicit ``OMP_WAIT_POLICY`` or ``OMP_NUM_THREADS`` overrides that part of the mode.
``current_settings()`` reports what was applied.

Why the waits matter, measured on Desktop 2 (Ryzen 7 8700G, 8 cores / 16 threads) with ``tools/energy_sitting.py``
on YOLOv8n through Ignition: with spinning workers on every thread, 99.9 % CPU and about 370 mJ per frame above idle;
with sleeping workers, about 14 % CPU and 134 mJ, for about 5 % fewer frames per second.

The settings have to reach the C runtime, not only Python: MSVC's OpenMP runtime (``vcomp140``) reads the UCRT's
environment table, which ``os.environ`` has not written since CPython 3.9. Measured: a passive policy set with
``os.environ`` as the first statement of the process still spun every thread (100.0 % CPU, 372.9 mJ per frame);
also written with ``_putenv_s``, it did not (14.0 % CPU, 133.8 mJ, 118.8 fps). So ``apply_openmp_settings`` writes
both.
"""
from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import asdict, dataclass
from typing import Dict, Optional

MODES = ("efficiency", "balanced", "performance")
DEFAULT_MODE = "balanced"
ENV_MODE = "IGNITE_XDNA_POWER_MODE"

_applied: Optional["PowerSettings"] = None


@dataclass(frozen=True)
class PowerSettings:
    mode: str
    wait_policy: str          # "ACTIVE" or "PASSIVE", as OMP_WAIT_POLICY spells it
    threads: int
    physical_cores: int
    logical_processors: int
    wait_policy_source: str   # "mode" or "environment"
    threads_source: str       # "mode" or "environment"

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

    def describe(self) -> str:
        wait = "spinning" if self.wait_policy == "ACTIVE" else "sleeping"
        extra = [f"{k} from the environment" for k, src in (("wait policy", self.wait_policy_source),
                                                            ("thread count", self.threads_source)) if src != "mode"]
        return (f"{self.mode}: {self.threads} worker thread{'s' if self.threads != 1 else ''}, {wait} between "
                f"regions ({self.physical_cores} cores / {self.logical_processors} threads on this host"
                f"{'; ' + ', '.join(extra) if extra else ''})")


def host_cores() -> tuple:
    """(physical cores, logical processors) of this host, from the OS where it says so."""
    logical = os.cpu_count() or 1
    physical = None
    if platform.system() == "Windows":
        physical = _windows_physical_cores()
    if not physical:
        physical = max(1, logical // 2) if logical > 1 else 1
    return max(1, min(physical, logical)), logical


def _windows_physical_cores() -> Optional[int]:
    """Count RelationProcessorCore records from GetLogicalProcessorInformationEx; None if the call fails."""
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = kernel32.GetLogicalProcessorInformationEx
        fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        fn.restype = ctypes.c_int
        relation_processor_core = 0
        size = ctypes.c_uint32(0)
        fn(relation_processor_core, None, ctypes.byref(size))
        if size.value == 0:
            return None
        buf = (ctypes.c_ubyte * size.value)()
        if not fn(relation_processor_core, buf, ctypes.byref(size)):
            return None
        count, offset = 0, 0
        while offset < size.value:
            # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: DWORD Relationship, DWORD Size, then the union.
            record_size = int.from_bytes(bytes(buf[offset + 4:offset + 8]), "little")
            if record_size <= 0:
                return None
            count += 1
            offset += record_size
        return count or None
    except (OSError, AttributeError, ValueError):
        return None


def resolve(mode: Optional[str], env: Dict[str, str], physical: int, logical: int) -> PowerSettings:
    """The settings a mode means on a host with this many cores, after the environment's explicit overrides."""
    mode = (mode or DEFAULT_MODE).strip().lower()
    if mode not in MODES:
        raise ValueError(f"unknown power mode {mode!r}; choose one of {', '.join(MODES)}")
    physical = max(1, min(int(physical), int(logical)))
    logical = max(1, int(logical))
    wait = "ACTIVE" if mode == "performance" else "PASSIVE"
    threads = {"performance": logical, "balanced": physical, "efficiency": max(1, physical // 4)}[mode]
    wait_src = threads_src = "mode"
    if env.get("OMP_WAIT_POLICY"):
        wait, wait_src = env["OMP_WAIT_POLICY"].strip().upper(), "environment"
    if env.get("OMP_NUM_THREADS"):
        try:
            threads, threads_src = max(1, int(env["OMP_NUM_THREADS"].split(",")[0])), "environment"
        except ValueError:
            pass
    return PowerSettings(mode, wait, threads, physical, logical, wait_src, threads_src)


def _put_env(name: str, value: str) -> None:
    os.environ[name] = value
    if platform.system() == "Windows":
        try:
            ctypes.cdll.ucrtbase._putenv_s(name.encode(), value.encode())
        except (OSError, AttributeError):
            pass


def apply_openmp_settings() -> PowerSettings:
    """Resolve the mode for this host and write it where the OpenMP runtime will read it. Call before the DLL loads."""
    global _applied
    if _applied is not None:
        return _applied
    physical, logical = host_cores()
    settings = resolve(os.environ.get(ENV_MODE), dict(os.environ), physical, logical)
    _put_env("OMP_WAIT_POLICY", settings.wait_policy)
    _put_env("OMP_NUM_THREADS", str(settings.threads))
    _applied = settings
    return settings


def current_settings() -> Optional[PowerSettings]:
    """The settings the native preprocessor loaded with, or None if it has not loaded."""
    return _applied
