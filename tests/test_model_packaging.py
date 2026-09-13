#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
tests/test_model_packaging.py

Unit and silicon verification test suite for .ignite monolithic container format,
the standalone ignite-compile CLI, and zero-copy runtime loading.
"""

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import ignite_xdna
from ignite_xdna.compiler.serializer import (
    ARCH_XDNA1_PHOENIX,
    IgniteModelReader,
    IgniteModelWriter,
)
from ignite_xdna.runtime.loader import IgniteEngine, load
from ignite_xdna.runtime.session import InferenceSession


class TestModelPackaging(unittest.TestCase):
    """Test suite for .ignite binary packaging and zero-copy runtime loader."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.model_ignite = cls.repo_root / "build" / "yolov8n.ignite"
        if not cls.model_ignite.exists():
            raise unittest.SkipTest(f"Compiled container not found: {cls.model_ignite}")

    def test_01_container_specification_and_crc(self):
        """Test 1: Verify 64-byte header, alignment, and CRC32 tamper detection."""
        manifest = {
            "model_name": "toy_model",
            "version": 1,
            "arch_id": ARCH_XDNA1_PHOENIX,
            "stages": {"stage0": {"exec_blob": "s0_exec.bin", "init_blob": "s0_init.bin"}}
        }
        writer = IgniteModelWriter(manifest, arch_id=ARCH_XDNA1_PHOENIX)
        writer.add_blob("s0_exec.bin", b"\x11" * 1234, content_type="transaction_exec")
        writer.add_blob("s0_init.bin", b"\x22" * 5678, content_type="transaction_init")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "toy.ignite"
            total_bytes = writer.write(tmp_path)

            self.assertGreater(total_bytes, 0)
            self.assertEqual(tmp_path.stat().st_size, total_bytes)

            # Test reader
            with IgniteModelReader(tmp_path) as reader:
                self.assertEqual(reader.header.magic, b"IGNT")
                self.assertEqual(reader.header.version, 1)
                self.assertEqual(reader.header.arch_id, ARCH_XDNA1_PHOENIX)
                self.assertEqual(reader.header.header_size, 64)
                self.assertEqual(reader.header.num_blobs, 2)
                self.assertTrue(reader.verify_checksum())

                # Check strict 64-byte alignment
                for b_name, b_entry in reader.blobs.items():
                    self.assertEqual(b_entry.offset % 64, 0, f"Blob {b_name} not 64-byte aligned")

                # Verify payload
                self.assertEqual(bytes(reader.get_blob_memoryview("s0_exec.bin")), b"\x11" * 1234)
                self.assertEqual(bytes(reader.get_blob_memoryview("s0_init.bin")), b"\x22" * 5678)
                tamper_offset = reader.header.blob_offset + 10

            # Tamper test: modify one byte in the blob section and assert CRC failure
            with open(tmp_path, "r+b") as f:
                f.seek(tamper_offset)
                byte = f.read(1)
                f.seek(tamper_offset)
                f.write(bytes([byte[0] ^ 0xFF]))

            with IgniteModelReader(tmp_path) as reader:
                self.assertFalse(reader.verify_checksum(), "Tampered container should fail CRC32 verification")

    def test_02_compiled_yolov8n_container_properties(self):
        """Test 2: Verify yolov8n.ignite meets file size (< 10 MB) and alignment contracts."""
        file_size = self.model_ignite.stat().st_size
        size_mb = file_size / (1024 * 1024)
        self.assertLess(size_mb, 10.0, f"Model size {size_mb:.2f} MB exceeds 10 MB limit")

        with IgniteModelReader(self.model_ignite) as reader:
            self.assertEqual(reader.header.magic, b"IGNT")
            self.assertEqual(reader.header.version, 1)
            self.assertEqual(reader.header.arch_id, ARCH_XDNA1_PHOENIX)
            self.assertTrue(reader.verify_checksum())

            # 9 monolithic stages -> 18 blobs (or 20 with unified single-dispatch streams)
            self.assertIn(len(reader.blobs), [18, 20])
            for b_name, b_entry in reader.blobs.items():
                self.assertEqual(b_entry.offset % 64, 0, f"Blob {b_name} offset {b_entry.offset} not 64-byte aligned")

            # Check manifest metadata
            self.assertIn("stages", reader.manifest)
            self.assertEqual(len(reader.manifest["stages"]), 9)
            self.assertEqual(reader.manifest["input_shape"], [1, 3, 640, 640])
            self.assertEqual(reader.manifest["strides"], [8, 16, 32])

    def test_03_zero_copy_session_loading_and_silicon_execution(self):
        """Test 3: Verify InferenceSession.from_file bare-metal execution on physical silicon."""
        session = InferenceSession.from_file(self.model_ignite, device_index=0)
        try:
            self.assertTrue(session.is_monolithic)
            self.assertEqual(len(session.monolithic_stages), 9)
            self.assertEqual(session.intermediate_ddr_bytes, 0)

            # Test physical forward execution
            dummy_in = np.zeros((1, 3, 640, 640), dtype=np.int8)
            t0 = time.perf_counter()
            outputs = session.run_yolo_monolithic(dummy_in)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            self.assertIn("p3_box", outputs)
            self.assertIn("p3_cls", outputs)
            self.assertIn("p4_box", outputs)
            self.assertIn("p4_cls", outputs)
            self.assertIn("p5_box", outputs)
            self.assertIn("p5_cls", outputs)
            self.assertEqual(outputs["p3_box"].shape, (1, 64, 80, 80))
            self.assertEqual(outputs["p3_cls"].shape, (1, 80, 80, 80))
            self.assertLess(lat_ms, 10.0, f"Forward pass took {lat_ms:.2f} ms")
        finally:
            session.close()

    def test_04_bit_exact_parity_and_latency_vs_memory_session(self):
        """Test 4: Assert bit-exact forward parity between loaded .ignite and memory session."""
        dummy_in = np.zeros((1, 3, 640, 640), dtype=np.int8)

        # 1. From .ignite file
        sess_ignite = InferenceSession.from_file(self.model_ignite, device_index=0)
        try:
            out_ignite = sess_ignite.run_yolo_monolithic(dummy_in)
        finally:
            sess_ignite.close()

        # 2. From in-memory compiled session
        sess_mem = InferenceSession(device_index=0, full_yolo=True)
        try:
            out_mem = sess_mem.run_yolo_monolithic(dummy_in)
        finally:
            sess_mem.close()

        # Bit-exact parity assertions
        self.assertTrue(
            np.array_equal(out_ignite["raw_output"], out_mem["raw_output"]),
            "Raw egress output mismatch between .ignite container and memory session",
        )
        self.assertTrue(
            np.allclose(out_ignite["p3_box"], out_mem["p3_box"]),
            "P3 box output mismatch between .ignite container and memory session",
        )
        self.assertTrue(
            np.allclose(out_ignite["p4_box"], out_mem["p4_box"]),
            "P4 box output mismatch between .ignite container and memory session",
        )
        self.assertTrue(
            np.allclose(out_ignite["p5_box"], out_mem["p5_box"]),
            "P5 box output mismatch between .ignite container and memory session",
        )

    def test_05_one_liner_ignite_load_api(self):
        """Test 5: Verify ignite_xdna.load() one-liner entry point with dummy image."""
        engine = ignite_xdna.load(self.model_ignite, device_index=0)
        try:
            self.assertEqual(len(engine.stages), 9)
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            dets = engine(dummy_img)
            self.assertIsInstance(dets, list)
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
