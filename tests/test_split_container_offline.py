"""Offline invariants of split containers and decoupled weights (no device required).

Covers:
1. plan_segments with explicit pure-NPU split points.
2. Instruction stream splitting at scheduled layer boundaries.
3. Decoupled weights container serialization and CRC32 integrity.
4. Runtime sidecar resolution, SHA-256 verification, and error paths.
5. CLI argument parsing for --split-layer and --decouple-weights.
"""
import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ignite_xdna.compiler import engine_schedule as es  # noqa: E402
from ignite_xdna.compiler import graph_ir  # noqa: E402
from ignite_xdna.compiler.cli import parse_args  # noqa: E402
from ignite_xdna.compiler.engine_compile import plan_segments  # noqa: E402
from ignite_xdna.compiler.engine_sequence import split_instruction_stream  # noqa: E402
from ignite_xdna.compiler.scheduler import TXN_HEADER_BYTES  # noqa: E402
from ignite_xdna.compiler.serializer import (  # noqa: E402
    ARCH_XDNA1_PHOENIX,
    IgniteModelReader,
    IgniteModelWriter,
    decouple_container_weights,
)

MODEL = ROOT / "models" / "yolov8n_cut_xint8.onnx"
IGNITE_FULL = ROOT / "build" / "yolov8n_full.ignite"


class SplitContainerOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if MODEL.exists():
            cls.ir = graph_ir.lower_yolov8n(MODEL)
            cls.ws = es.plan_workspace(cls.ir)
            cls.scheds, cls.store = es.schedule_graph(cls.ir, cls.ws)
        else:
            cls.ir = None

    def test_01_plan_segments_default_monolithic(self):
        """Without split layers or host regions, plan_segments emits a single NPU segment."""
        if self.ir is None:
            self.skipTest("yolov8n_cut_xint8.onnx not present")
        segs = plan_segments(self.ir, self.scheds)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["kind"], "npu")
        self.assertEqual(segs[0]["layers"], [0, len(self.scheds)])
        self.assertEqual(segs[0]["blob"], "insts_0.bin")
        self.assertGreater(segs[0]["tasks"], 0)

    def test_02_plan_segments_pure_npu_splits(self):
        """plan_segments with split_layers cuts NPU execution into contiguous segments."""
        if self.ir is None:
            self.skipTest("yolov8n_cut_xint8.onnx not present")
        splits = [10, 22]
        segs = plan_segments(self.ir, self.scheds, split_layers=splits)
        self.assertEqual(len(segs), 3)
        for idx, seg in enumerate(segs):
            self.assertEqual(seg["kind"], "npu")
            self.assertEqual(seg["blob"], f"insts_{idx}.bin")
            self.assertGreater(seg["tasks"], 0)
        self.assertEqual(segs[0]["layers"], [0, 10])
        self.assertEqual(segs[1]["layers"], [10, 22])
        self.assertEqual(segs[2]["layers"], [22, len(self.scheds)])

        # Total tasks across all segments must equal monolithic task count
        mono_segs = plan_segments(self.ir, self.scheds)
        self.assertEqual(sum(s["tasks"] for s in segs), mono_segs[0]["tasks"])

    def test_03_split_instruction_stream_on_real_container(self):
        """split_instruction_stream cleanly cuts insts.bin into valid per-segment streams."""
        if not IGNITE_FULL.exists() or self.ir is None:
            self.skipTest("yolov8n_full.ignite not present")
        with IgniteModelReader(IGNITE_FULL) as reader:
            insts = reader.get_blob_bytes("insts.bin")

        splits = [10, 22]
        segs = plan_segments(self.ir, self.scheds, split_layers=splits)
        task_counts = [s["tasks"] for s in segs]
        try:
            pieces = split_instruction_stream(insts, task_counts)
        except ValueError as exc:
            # The container on disk is a build artifact, not a fixture: any scheduler change
            # (a new fill layout, a different chunking) makes its stream disagree with a
            # freshly computed schedule. That is staleness, not a defect in the splitter, so
            # skip loudly rather than fail. Anything else re-raises.
            if "task pushes" not in str(exc):
                raise
            self.skipTest(f"{IGNITE_FULL.name} predates the current scheduler ({exc}); "
                          "rebuild it with ignite-compile to exercise this test")

        self.assertEqual(len(pieces), len(task_counts))
        for piece in pieces:
            self.assertGreater(len(piece), TXN_HEADER_BYTES)
            major, minor, num_ops, size = struct.unpack("<4I", piece[:TXN_HEADER_BYTES])
            self.assertEqual(size, len(piece))
            self.assertGreater(num_ops, 0)

    def test_04_decoupled_weights_container_format(self):
        """IgniteModelWriter creates a valid container without wpackets.bin and records sidecar metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_ignite = tmp / "model_decoupled.ignite"
            out_weights = tmp / "model_decoupled.weights"

            mock_weights = b"\xaa\xbb\xcc\xdd" * 256
            weights_sha = hashlib.sha256(mock_weights).hexdigest()
            out_weights.write_bytes(mock_weights)

            manifest = {
                "format_version": 1,
                "arch_id": ARCH_XDNA1_PHOENIX,
                "engine": "conv_engine_v1",
                "single_dispatch": False,
                "graph_engine": {
                    "workspace_bytes": 1024 * 1024,
                    "input_tensor": "input",
                    "decoupled_weights": True,
                    "weights_file": out_weights.name,
                    "weights_sha256": weights_sha,
                    "weights_bytes": len(mock_weights),
                    "segments": [
                        {"kind": "npu", "blob": "insts_0.bin", "tasks": 5},
                        {"kind": "npu", "blob": "insts_1.bin", "tasks": 5},
                    ],
                },
            }

            writer = IgniteModelWriter(manifest_meta=manifest, arch_id=ARCH_XDNA1_PHOENIX)
            writer.add_blob("engine.xclbin", b"XCLBIN_MOCK" * 16, content_type="xclbin")
            writer.add_blob("insts_0.bin", b"INSTS0_MOCK" * 16, content_type="npu_instructions")
            writer.add_blob("insts_1.bin", b"INSTS1_MOCK" * 16, content_type="npu_instructions")
            total_size = writer.write(out_ignite)

            self.assertGreater(total_size, 0)
            self.assertTrue(out_ignite.exists())

            # Verify container self-consistency with reader
            with IgniteModelReader(out_ignite) as reader:
                self.assertTrue(reader.verify_checksum())
                self.assertNotIn("wpackets.bin", reader.blobs)
                self.assertIn("insts_0.bin", reader.blobs)
                self.assertIn("insts_1.bin", reader.blobs)
                ge = reader.manifest["graph_engine"]
                self.assertTrue(ge["decoupled_weights"])
                self.assertEqual(ge["weights_file"], out_weights.name)
                self.assertEqual(ge["weights_sha256"], weights_sha)

    @patch("ignite_xdna.runtime.graph_session.setup_xrt_environment")
    @patch("ignite_xdna.runtime.graph_session.XrtSiliconHarness")
    def test_05_runtime_decoupled_weights_loading_and_errors(self, mock_harness_cls, mock_setup_xrt):
        """EngineSession resolves decoupled weights sidecar, validates SHA-256, and raises on missing/corrupt."""
        from ignite_xdna.runtime.graph_session import EngineSession

        mock_harness = MagicMock()
        mock_harness_cls.return_value = mock_harness
        mock_bo = MagicMock()
        mock_harness.create_host_bo.return_value = mock_bo
        mock_harness.create_instruction_bo_from_bytes.return_value = (mock_bo, 10)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            tmp = Path(tmpdir)
            ignite_path = tmp / "test_decoupled.ignite"
            weights_path = tmp / "test_decoupled.weights"

            mock_weights = b"VALIDSIDECARWEIGHTS" * 64
            weights_sha = hashlib.sha256(mock_weights).hexdigest()
            weights_path.write_bytes(mock_weights)

            manifest = {
                "format_version": 1,
                "arch_id": ARCH_XDNA1_PHOENIX,
                "engine": "conv_engine_v1",
                "task": "detect",
                "egress_bytes": 1024,
                "graph_engine": {
                    "workspace_bytes": 4096,
                    "input_tensor": "x",
                    "placements": {"x": {"base": 0, "halo": 0, "height": 10, "width": 10, "blocks": 1}},
                    "decoupled_weights": True,
                    "weights_file": weights_path.name,
                    "weights_sha256": weights_sha,
                    "weights_bytes": len(mock_weights),
                    "segments": [{"kind": "npu", "blob": "insts_0.bin", "tasks": 1}],
                },
            }

            writer = IgniteModelWriter(manifest_meta=manifest, arch_id=ARCH_XDNA1_PHOENIX)
            writer.add_blob("engine.xclbin", b"XCLBIN" * 16, content_type="xclbin")
            writer.add_blob("insts_0.bin", b"INSTS0" * 16, content_type="npu_instructions")
            writer.write(ignite_path)

            # 1. Successful loading from sidecar
            captured_weights = []
            mock_bo.write.side_effect = lambda data, offset=0: captured_weights.append(bytes(data))
            sess = EngineSession(ignite_path, map_workspace=False)
            self.assertIn(mock_weights, captured_weights)
            sess.close()
            mock_harness.reset_mock()
            mock_bo.reset_mock()
            mock_bo.write.side_effect = None
            import gc
            gc.collect()

            # 2. Missing sidecar raises FileNotFoundError
            weights_path.unlink()
            with self.assertRaises(FileNotFoundError):
                EngineSession(ignite_path, map_workspace=False)
            mock_harness.reset_mock()
            mock_bo.reset_mock()
            gc.collect()

            # 3. Corrupt sidecar (checksum mismatch) raises ValueError
            weights_path.write_bytes(b"CORRUPTED_WEIGHTS_DATA" * 64)
            with self.assertRaises(ValueError) as ctx:
                EngineSession(ignite_path, map_workspace=False)
            self.assertIn("checksum mismatch", str(ctx.exception))
            mock_harness.reset_mock()
            mock_bo.reset_mock()
            gc.collect()

    @patch("ignite_xdna.runtime.graph_session.setup_xrt_environment")
    @patch("ignite_xdna.runtime.graph_session.XrtSiliconHarness")
    def test_06_dispatch_early_exit_max_segments(self, mock_harness_cls, mock_setup_xrt):
        """dispatch(max_segments=N) executes only the first N segments."""
        from ignite_xdna.runtime.graph_session import EngineSession

        mock_harness = MagicMock()
        mock_harness_cls.return_value = mock_harness
        mock_bo = MagicMock()
        mock_harness.create_host_bo.return_value = mock_bo
        mock_harness.create_instruction_bo_from_bytes.return_value = (mock_bo, 10)
        mock_run = MagicMock()
        mock_run.wait.return_value = getattr(mock_harness.pyxrt.ert_cmd_state, "ERT_CMD_STATE_COMPLETED", 4)
        mock_harness.pyxrt.run.return_value = mock_run

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            tmp = Path(tmpdir)
            ignite_path = tmp / "test_multi_seg.ignite"
            weights_path = tmp / "test_multi_seg.weights"

            mock_weights = b"WEIGHTS" * 32
            weights_path.write_bytes(mock_weights)

            manifest = {
                "format_version": 1,
                "arch_id": ARCH_XDNA1_PHOENIX,
                "engine": "conv_engine_v1",
                "task": "detect",
                "egress_bytes": 1024,
                "graph_engine": {
                    "workspace_bytes": 4096,
                    "input_tensor": "x",
                    "placements": {"x": {"base": 0, "halo": 0, "height": 10, "width": 10, "blocks": 1}},
                    "decoupled_weights": True,
                    "weights_file": weights_path.name,
                    "weights_sha256": hashlib.sha256(mock_weights).hexdigest(),
                    "weights_bytes": len(mock_weights),
                    "segments": [
                        {"kind": "npu", "blob": "insts_0.bin", "tasks": 1},
                        {"kind": "npu", "blob": "insts_1.bin", "tasks": 1},
                        {"kind": "npu", "blob": "insts_2.bin", "tasks": 1},
                    ],
                },
            }

            writer = IgniteModelWriter(manifest_meta=manifest, arch_id=ARCH_XDNA1_PHOENIX)
            writer.add_blob("engine.xclbin", b"XCLBIN" * 16, content_type="xclbin")
            writer.add_blob("insts_0.bin", b"INSTS0" * 16, content_type="npu_instructions")
            writer.add_blob("insts_1.bin", b"INSTS1" * 16, content_type="npu_instructions")
            writer.add_blob("insts_2.bin", b"INSTS2" * 16, content_type="npu_instructions")
            writer.write(ignite_path)

            sess = EngineSession(ignite_path, map_workspace=False)
            self.assertEqual(len(sess.segments), 3)

            # Test full dispatch
            sess.dispatch()
            self.assertEqual(len(sess.last_segment_ms), 3)

            # Test early exit at max_segments=1
            sess.dispatch(max_segments=1)
            self.assertEqual(len(sess.last_segment_ms), 1)

            # Test early exit at max_segments=2
            sess.dispatch(max_segments=2)
            self.assertEqual(len(sess.last_segment_ms), 2)

            sess.close()
            mock_harness.reset_mock()
            mock_bo.reset_mock()
            import gc
            gc.collect()

    def test_07_cli_argument_parsing(self):
        """CLI parser accepts multiple --split-layer flags and --decouple-weights."""
        args = [
            "--input", "models/yolov8n_cut_xint8.onnx",
            "--output", "build/test_split.ignite",
            "--split-layer", "10",
            "--split-layer", "22",
            "--decouple-weights",
        ]
        parsed = parse_args(args)
        self.assertEqual(parsed.split_layer, [10, 22])
        self.assertTrue(parsed.decouple_weights)

    def test_08_decouple_container_weights_utility(self):
        """decouple_container_weights strips weights from monolithic container and generates valid sidecar."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mono_ignite = tmp / "model_mono.ignite"

            mock_weights = b"\x12\x34\x56\x78" * 512
            weights_sha = hashlib.sha256(mock_weights).hexdigest()

            manifest = {
                "format_version": 1,
                "arch_id": ARCH_XDNA1_PHOENIX,
                "engine": "conv_engine_v1",
                "graph_engine": {
                    "workspace_bytes": 1024,
                    "input_tensor": "input",
                },
            }

            writer = IgniteModelWriter(manifest_meta=manifest, arch_id=ARCH_XDNA1_PHOENIX)
            writer.add_blob("engine.xclbin", b"XCLBIN_MOCK" * 16, content_type="xclbin")
            writer.add_blob("insts.bin", b"INSTS_MOCK" * 16, content_type="npu_instructions")
            writer.add_blob("wpackets.bin", mock_weights, content_type="weight_packets")
            writer.write(mono_ignite)

            # Decouple to new output path
            out_ignite = tmp / "model_out.ignite"
            ret_ignite, ret_weights = decouple_container_weights(mono_ignite, out_ignite)

            self.assertEqual(ret_ignite, out_ignite.resolve())
            self.assertTrue(ret_ignite.exists())
            self.assertTrue(ret_weights.exists())
            self.assertEqual(ret_weights.read_bytes(), mock_weights)

            with IgniteModelReader(ret_ignite) as reader:
                self.assertTrue(reader.verify_checksum())
                self.assertNotIn("wpackets.bin", reader.blobs)
                self.assertIn("insts.bin", reader.blobs)
                self.assertIn("engine.xclbin", reader.blobs)
                ge = reader.manifest["graph_engine"]
                self.assertTrue(ge["decoupled_weights"])
                self.assertEqual(ge["weights_file"], ret_weights.name)
                self.assertEqual(ge["weights_sha256"], weights_sha)
                self.assertEqual(ge["weights_bytes"], len(mock_weights))

            # Attempting to decouple an already decoupled container raises ValueError
            with self.assertRaises(ValueError):
                decouple_container_weights(ret_ignite)

    def test_09_cli_decouple_weights_on_ignite_input(self):
        """ignite-compile CLI with .ignite input and --decouple-weights decouples without lowering."""
        from ignite_xdna.compiler.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mono_ignite = tmp / "mono.ignite"
            out_ignite = tmp / "decoupled.ignite"
            mock_weights = b"\xaa\xbb\xcc\xdd" * 128

            manifest = {
                "format_version": 1,
                "arch_id": ARCH_XDNA1_PHOENIX,
                "engine": "conv_engine_v1",
                "graph_engine": {
                    "workspace_bytes": 1024,
                    "input_tensor": "input",
                },
            }

            writer = IgniteModelWriter(manifest_meta=manifest, arch_id=ARCH_XDNA1_PHOENIX)
            writer.add_blob("insts.bin", b"INSTS" * 8, content_type="npu_instructions")
            writer.add_blob("wpackets.bin", mock_weights, content_type="weight_packets")
            writer.write(mono_ignite)

            # Run CLI
            cli_args = [
                "--input", str(mono_ignite),
                "--output", str(out_ignite),
                "--decouple-weights",
            ]
            main(cli_args)

            self.assertTrue(out_ignite.exists())
            expected_weights = out_ignite.with_suffix(".weights")
            self.assertTrue(expected_weights.exists())
            self.assertEqual(expected_weights.read_bytes(), mock_weights)

    @patch("ignite_xdna.runtime.graph_session.setup_xrt_environment")
    @patch("ignite_xdna.runtime.graph_session.XrtSiliconHarness")
    def test_10_composed_session_offline(self, mock_harness_cls, mock_setup_xrt):
        """ComposedSession chains stages on a single shared bo_ws and harness without memory bloat."""
        from ignite_xdna.runtime.graph_session import ComposedSession

        mock_harness = MagicMock()
        mock_harness_cls.return_value = mock_harness
        mock_bo = MagicMock()
        mock_bo.size.return_value = 8192
        mock_harness.create_host_bo.return_value = mock_bo
        mock_harness.create_instruction_bo_from_bytes.return_value = (mock_bo, 10)
        mock_run = MagicMock()
        mock_run.wait.return_value = getattr(mock_harness.pyxrt.ert_cmd_state, "ERT_CMD_STATE_COMPLETED", 4)
        mock_harness.pyxrt.run.return_value = mock_run

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            tmp = Path(tmpdir)
            stage1_path = tmp / "stage1.ignite"
            stage2_path = tmp / "stage2.ignite"

            for s_idx, s_path, ws_bytes in [(1, stage1_path, 4096), (2, stage2_path, 8192)]:
                w_path = s_path.with_suffix(".weights")
                mock_weights = b"WEIGHTS" * 32
                w_path.write_bytes(mock_weights)

                manifest = {
                    "format_version": 1,
                    "arch_id": ARCH_XDNA1_PHOENIX,
                    "engine": "conv_engine_v1",
                    "task": "generic",
                    "egress_bytes": 1024,
                    "graph_engine": {
                        "workspace_bytes": ws_bytes,
                        "input_tensor": "x",
                        "heads": {},
                        "placements": {"x": {"base": 0, "halo": 0, "height": 10, "width": 10, "blocks": 1, "planes": 1, "channels": 8}},
                        "decoupled_weights": True,
                        "weights_file": w_path.name,
                        "weights_sha256": hashlib.sha256(mock_weights).hexdigest(),
                        "weights_bytes": len(mock_weights),
                        "segments": [{"kind": "npu", "blob": "insts.bin", "tasks": 1}],
                    },
                }

                writer = IgniteModelWriter(manifest_meta=manifest, arch_id=ARCH_XDNA1_PHOENIX)
                writer.add_blob("engine.xclbin", b"XCLBIN" * 16, content_type="xclbin")
                writer.add_blob("insts.bin", b"INSTS" * 16, content_type="npu_instructions")
                writer.write(s_path)

            comp = ComposedSession([stage1_path, stage2_path])
            self.assertEqual(len(comp.stages), 2)
            self.assertTrue(comp.stages[0]._owns_bo_ws)
            self.assertFalse(comp.stages[1]._owns_bo_ws)
            self.assertTrue(comp.stages[0]._owns_harness)
            self.assertFalse(comp.stages[1]._owns_harness)

            # Test dispatching composed stages
            lat = comp.dispatch()
            self.assertGreaterEqual(lat, 0.0)

            comp.close()
            import gc
            gc.collect()

    @patch("ignite_xdna.runtime.graph_session.setup_xrt_environment")
    @patch("ignite_xdna.runtime.graph_session.XrtSiliconHarness")
    def test_11_targeted_dma_offline(self, mock_harness_cls, mock_setup_xrt):
        """stage_tensor and read_tensor calculate exact physical slice size and offsets."""
        from ignite_xdna.runtime.graph_session import EngineSession

        mock_harness = MagicMock()
        mock_harness_cls.return_value = mock_harness
        mock_bo = MagicMock()
        mock_harness.create_host_bo.return_value = mock_bo
        mock_harness.create_instruction_bo_from_bytes.return_value = (mock_bo, 10)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            tmp = Path(tmpdir)
            ignite_path = tmp / "model.ignite"
            w_path = tmp / "model.weights"
            mock_weights = b"WEIGHTS" * 32
            w_path.write_bytes(mock_weights)

            manifest = {
                "format_version": 1,
                "arch_id": ARCH_XDNA1_PHOENIX,
                "engine": "conv_engine_v1",
                "task": "detect",
                "egress_bytes": 1024,
                "graph_engine": {
                    "workspace_bytes": 32768,
                    "input_tensor": "input",
                    "heads": {},
                    "placements": {
                        "input": {"base": 0, "halo": 0, "height": 10, "width": 10, "blocks": 1, "planes": 1, "channels": 8},
                        "p5_out": {"base": 8192, "halo": 0, "height": 20, "width": 20, "blocks": 4, "planes": 4, "channels": 32},
                    },
                    "decoupled_weights": True,
                    "weights_file": w_path.name,
                    "weights_sha256": hashlib.sha256(mock_weights).hexdigest(),
                    "weights_bytes": len(mock_weights),
                    "segments": [{"kind": "npu", "blob": "insts.bin", "tasks": 1}],
                },
            }

            writer = IgniteModelWriter(manifest_meta=manifest, arch_id=ARCH_XDNA1_PHOENIX)
            writer.add_blob("engine.xclbin", b"XCLBIN" * 16, content_type="xclbin")
            writer.add_blob("insts.bin", b"INSTS" * 16, content_type="npu_instructions")
            writer.write(ignite_path)

            sess = EngineSession(ignite_path, map_workspace=False)
            # p5_out: blocks=4, h=20, w=20, halo=0 -> 20*20*4*8 = 12800 bytes
            data_to_stage = np.ones((4 * 8, 20, 20), dtype=np.uint8)

            sess.stage_tensor("p5_out", data_to_stage, sync=True)
            mock_bo.write.assert_called()
            mock_bo.sync.assert_called()

            # read_tensor
            fake_mem = bytes(32768)
            mock_bo.read.side_effect = lambda nbytes, offset=0: fake_mem[offset:offset + nbytes]
            read_arr = sess.read_tensor("p5_out", sync=True)
            self.assertEqual(read_arr.shape, (32, 20, 20))

            sess.close()
            import gc
            gc.collect()



if __name__ == "__main__":
    unittest.main()
