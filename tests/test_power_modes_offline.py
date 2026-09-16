"""Offline (no device): power modes resolve to thread counts relative to the host, and reach the C runtime.

* ``resolve`` maps each mode to a wait policy and a thread count that is a fraction of the host's own cores, on
  hosts shaped like the parts this engine can meet (6 cores / 12 threads, 8 / 16, 16 / 32, a 4-core part without
  SMT, a single core), defaults to balanced, rejects unknown names, and lets an explicit OMP_WAIT_POLICY or
  OMP_NUM_THREADS override its part;
* ``host_cores`` returns 1 <= physical <= logical on this machine;
* ``apply_openmp_settings`` writes the settings where MSVC's OpenMP runtime reads them - the UCRT table, which
  ``os.environ`` alone does not reach - checked in a fresh interpreter through ucrtbase's own ``getenv``.

    python -m pytest tests/test_power_modes_offline.py
"""
import os
import platform
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import the module file directly: importing ignite_xdna.pipelines would load the native preprocessor.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("ignite_power", ROOT / "src" / "ignite_xdna" / "pipelines" / "power.py")
power = importlib.util.module_from_spec(_spec)
sys.modules["ignite_power"] = power  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(power)


class ResolveTest(unittest.TestCase):
    HOSTS = {
        # (physical, logical): (performance, balanced, efficiency) thread counts
        (6, 12): (12, 6, 1),
        (8, 16): (16, 8, 2),
        (16, 32): (32, 16, 4),
        (4, 4): (4, 4, 1),
        (1, 1): (1, 1, 1),
    }

    def test_threads_scale_with_the_host(self):
        for (physical, logical), expected in self.HOSTS.items():
            for mode, threads in zip(("performance", "balanced", "efficiency"), expected):
                s = power.resolve(mode, {}, physical, logical)
                self.assertEqual(s.threads, threads, f"{mode} on {physical}C/{logical}T")
                self.assertEqual(s.wait_policy, "ACTIVE" if mode == "performance" else "PASSIVE")
                self.assertEqual((s.wait_policy_source, s.threads_source), ("mode", "mode"))

    def test_default_is_balanced(self):
        self.assertEqual(power.resolve(None, {}, 8, 16).mode, "balanced")
        self.assertEqual(power.resolve("  Balanced ", {}, 8, 16).threads, 8)

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            power.resolve("turbo", {}, 8, 16)

    def test_environment_overrides_its_part(self):
        s = power.resolve("efficiency", {"OMP_WAIT_POLICY": "active"}, 8, 16)
        self.assertEqual((s.wait_policy, s.wait_policy_source, s.threads, s.threads_source),
                         ("ACTIVE", "environment", 2, "mode"))
        s = power.resolve("balanced", {"OMP_NUM_THREADS": "3"}, 8, 16)
        self.assertEqual((s.wait_policy, s.threads, s.threads_source), ("PASSIVE", 3, "environment"))

    def test_physical_never_exceeds_logical(self):
        self.assertEqual(power.resolve("balanced", {}, 32, 8).threads, 8)


class HostTest(unittest.TestCase):
    def test_host_cores_are_sane(self):
        physical, logical = power.host_cores()
        self.assertGreaterEqual(physical, 1)
        self.assertLessEqual(physical, logical)
        self.assertEqual(logical, os.cpu_count() or 1)


@unittest.skipUnless(platform.system() == "Windows", "the UCRT table is a Windows concern")
class ReachesTheCRuntimeTest(unittest.TestCase):
    def test_ucrt_getenv_sees_the_settings(self):
        script = textwrap.dedent(f"""
            import ctypes, importlib.util, os, sys
            for k in ("OMP_WAIT_POLICY", "OMP_NUM_THREADS", "IGNITE_XDNA_POWER_MODE"):
                os.environ.pop(k, None)
            os.environ["IGNITE_XDNA_POWER_MODE"] = "efficiency"
            spec = importlib.util.spec_from_file_location("p", {str(ROOT / "src" / "ignite_xdna" / "pipelines" / "power.py")!r})
            p = importlib.util.module_from_spec(spec); sys.modules["p"] = p; spec.loader.exec_module(p)
            s = p.apply_openmp_settings()
            getenv = ctypes.cdll.ucrtbase.getenv
            getenv.restype = ctypes.c_char_p
            print(s.wait_policy, s.threads, getenv(b"OMP_WAIT_POLICY").decode(), getenv(b"OMP_NUM_THREADS").decode())
        """)
        env = {k: v for k, v in os.environ.items() if k not in ("OMP_WAIT_POLICY", "OMP_NUM_THREADS")}
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        policy, threads, crt_policy, crt_threads = out.stdout.split()
        self.assertEqual((policy, crt_policy), ("PASSIVE", "PASSIVE"))
        self.assertEqual(threads, crt_threads)


if __name__ == "__main__":
    unittest.main()
