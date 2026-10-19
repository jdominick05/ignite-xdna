"""Offline checks for the build recipes (recipes/*.toml, tools/recipe.py) and the build record a graph-engine
container carries. No device, no IRON, nothing is built.

Every recipe must parse, name real scripts and an env tools/recipe.py can run, and give each ignite-compile step
flags the real CLI accepts; no two steps may write the same file. The build record is what lets a container in the
untracked build/ be traced back to its recipe, so its fields are pinned here too.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import recipe  # noqa: E402


class TestRecipes(unittest.TestCase):
    def test_every_recipe_is_well_formed(self):
        found = recipe.recipes()
        self.assertTrue(found, "recipes/ holds no recipe")
        for name, r in found.items():
            with self.subTest(recipe=name):
                self.assertEqual(recipe.problems(name, r), [])

    def test_no_two_steps_write_one_file(self):
        writers = {}
        for name, r in recipe.recipes().items():
            for s in r["step"]:
                for o in s.get("outputs", []):
                    self.assertNotIn(o, writers, f"{name}:{s['id']} and {writers.get(o)} both write {o}")
                    writers[o] = f"{name}:{s['id']}"

    def test_compile_steps_use_the_graph_engine_through_iron(self):
        for name, r in recipe.recipes().items():
            for s in r["step"]:
                argv = recipe.compile_argv(s)
                if argv is None or "--engine" not in argv or argv[argv.index("--engine") + 1] != "graph":
                    continue
                with self.subTest(step=f"{name}:{s['id']}"):
                    self.assertEqual(s["env"], "iron", "the graph engine compiles only in the IRON environment")

    def test_a_bad_recipe_is_caught(self):
        bad = {"name": "x", "status": "verified", "step": [
            {"id": "a", "env": "nowhere", "run": ["tools/does_not_exist.py"]},
            {"id": "a", "env": "iron", "run": ["-m", "ignite_xdna.compiler.cli", "--no-such-flag"]}]}
        found = recipe.problems("x", bad)
        for fragment in ("[verified]", "env 'nowhere'", "does not exist", "repeated", "rejects its flags"):
            self.assertTrue(any(fragment in p for p in found), f"{fragment!r} not in {found}")


class TestBuildRecord(unittest.TestCase):
    def test_build_record_carries_the_command_commit_and_recipe(self):
        import os
        from ignite_xdna.compiler.cli import build_record
        os.environ["IGNITE_RECIPE"] = "yolov8n:compile"
        try:
            rec = build_record(["--engine", "graph", "--input", "m.onnx"])
        finally:
            del os.environ["IGNITE_RECIPE"]
        self.assertEqual(rec["command"], ["ignite-compile", "--engine", "graph", "--input", "m.onnx"])
        self.assertEqual(rec["recipe"], "yolov8n:compile")
        self.assertIn("compiler_commit", rec)
        self.assertRegex(rec["built_utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")

    def test_compile_graph_container_takes_a_build_record(self):
        import inspect
        from ignite_xdna.compiler.engine_compile import compile_graph_container
        self.assertIn("build", inspect.signature(compile_graph_container).parameters)


if __name__ == "__main__":
    unittest.main()
