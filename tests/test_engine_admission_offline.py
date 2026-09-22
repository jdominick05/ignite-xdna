"""Admission: a container a session cannot read must be refused before it is read.

WHY THIS EXISTS. Every region length in ``graph_session.py`` is an element count used as a byte
count. That is correct at one byte an element and exactly half the region at two, and the halved byte
count is numerically equal to the element count - so a reshape of a bf16 workspace SUCCEEDS, a sync
returns a plausible tensor of interleaved high and low bytes, and nothing raises. Silence is the
default failure mode here, which is why admission is checked at the top rather than trusted to blow
up somewhere useful.

No device and no build artifacts: the manifest guard runs before any blob or xclbin is touched, so
these containers are written with nothing in them but a manifest.
"""
import json
import tempfile
import unittest
from pathlib import Path

from ignite_xdna.compiler.serializer import (ELEM_BF16, ELEM_INT, ENGINE_CONV_BF16, ENGINE_CONV_INT8,
                                             GRAPH_ENGINES, IgniteModelWriter)
from ignite_xdna.runtime.graph_session import (ClassificationSession, DenseGraphSession, GraphSession,
                                               container_elem, is_graph_container)


def manifest(engine=ENGINE_CONV_BF16, task="super_resolution", elem=ELEM_BF16, dtype="uint16"):
    """A manifest with just enough in it to reach the admission check."""
    placement = {"base": 0, "halo": 1, "height": 8, "width": 8, "blocks": 1, "planes": 1,
                 "band_rows": 0, "channels": 3, "dtype": dtype, "storage": "workspace"}
    if elem is not None:
        placement["elem"] = elem
    return {"engine": engine, "task": task,
            "graph_engine": {"placements": {"input": placement}, "input_tensor": "input"}}


def container(tmp: Path, name="probe.ignite", **kw) -> str:
    path = tmp / name
    IgniteModelWriter(manifest_meta=manifest(**kw)).write(path)
    return str(path)


class TestGraphContainerIdentity(unittest.TestCase):
    """``is_graph_container`` is a membership test now, not a single-engine equality."""

    def test_both_graph_engines_are_admitted_as_graph_containers(self):
        for engine in GRAPH_ENGINES:
            self.assertTrue(is_graph_container({"engine": engine}), engine)

    def test_a_foreign_engine_is_not_a_graph_container(self):
        self.assertFalse(is_graph_container({"engine": "some_other_engine_v9"}))
        self.assertFalse(is_graph_container({}))
        self.assertFalse(is_graph_container(None))

    def test_the_two_engine_names_are_distinct(self):
        self.assertNotEqual(ENGINE_CONV_INT8, ENGINE_CONV_BF16)


class TestContainerElem(unittest.TestCase):
    """``elem`` absent means integer, so no container built before bf16 needs rebuilding."""

    def test_absent_elem_reads_as_integer(self):
        self.assertEqual(container_elem(manifest(elem=None)), ELEM_INT)

    def test_no_placements_at_all_reads_as_integer(self):
        self.assertEqual(container_elem({"engine": ENGINE_CONV_INT8, "graph_engine": {}}), ELEM_INT)
        self.assertEqual(container_elem({}), ELEM_INT)

    def test_declared_bf16_is_reported(self):
        self.assertEqual(container_elem(manifest(elem=ELEM_BF16)), ELEM_BF16)

    def test_mixed_widths_in_one_container_are_refused(self):
        m = manifest(elem=ELEM_BF16)
        wide = dict(m["graph_engine"]["placements"]["input"])
        narrow = dict(wide, elem=ELEM_INT)
        m["graph_engine"]["placements"] = {"a": wide, "b": narrow}
        with self.assertRaises(ValueError) as cm:
            container_elem(m)
        self.assertIn("mixes activation element widths", str(cm.exception))


class TestSessionsRefuseAWidthTheyCannotRead(unittest.TestCase):
    """Every session derives from ``EngineSession``, so one guard covers all four."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _assert_refused_naming_the_width(self, fn, path):
        with self.assertRaises(ValueError) as cm:
            fn(path)
        message = str(cm.exception)
        self.assertIn(ELEM_BF16, message,
                      "the refusal has to name the width, or it reads as an unrelated bug")
        return message

    def test_graph_session_refuses_bf16(self):
        path = container(self.tmp, task="detect")
        self._assert_refused_naming_the_width(GraphSession, path)

    def test_dense_graph_session_refuses_bf16(self):
        path = container(self.tmp, task="super_resolution")
        self._assert_refused_naming_the_width(DenseGraphSession, path)

    def test_classification_session_refuses_bf16(self):
        path = container(self.tmp, task="classify")
        self._assert_refused_naming_the_width(ClassificationSession, path)

    def test_dense_tensor_session_refuses_bf16(self):
        # This one validates its task before ``super().__init__``, so it needs a task it accepts
        # for the width check to be what fires.
        from ignite_xdna.runtime.dense_session import DenseTensorSession
        path = container(self.tmp, task="segment")
        self._assert_refused_naming_the_width(DenseTensorSession, path)

    def test_inference_session_from_file_refuses_bf16(self):
        from ignite_xdna.runtime.session import InferenceSession
        path = container(self.tmp, task="super_resolution")
        self._assert_refused_naming_the_width(InferenceSession.from_file, path)

    def test_the_refusal_names_the_session_that_could_not_read_it(self):
        path = container(self.tmp, task="detect")
        self.assertIn("GraphSession", self._assert_refused_naming_the_width(GraphSession, path))

    def test_a_foreign_engine_refusal_names_the_engine_it_found(self):
        path = container(self.tmp, engine="some_other_engine_v9", task="detect")
        with self.assertRaises(ValueError) as cm:
            GraphSession(path)
        message = str(cm.exception)
        self.assertIn("some_other_engine_v9", message)
        self.assertIn("not a graph-engine container", message)

    def test_an_integer_container_is_not_refused_by_the_width_guard(self):
        # It will fail later for want of a real kernel and blobs; what must NOT happen is a refusal
        # on width, which would mean the guard had started rejecting every shipping container.
        path = container(self.tmp, engine=ENGINE_CONV_INT8, task="detect", elem=None, dtype="uint8")
        try:
            GraphSession(path)
        except Exception as exc:  # noqa: BLE001 - any later failure is fine, this one is not
            self.assertNotIn("carries", str(exc))
            self.assertNotIn("activations but", str(exc))


class TestManifestFixtureIsHonest(unittest.TestCase):
    """If the fixture stopped round-tripping, every refusal above would pass for the wrong reason."""

    def test_the_written_manifest_survives_the_container(self):
        from ignite_xdna.compiler.serializer import IgniteModelReader
        with tempfile.TemporaryDirectory() as d:
            path = container(Path(d), task="detect")
            with IgniteModelReader(path) as reader:
                got = reader.manifest
        self.assertEqual(got["engine"], ENGINE_CONV_BF16)
        self.assertEqual(got["graph_engine"]["placements"]["input"]["elem"], ELEM_BF16)
        self.assertEqual(json.loads(json.dumps(got))["task"], "detect")


if __name__ == "__main__":
    unittest.main()
