"""The bf16 super-resolution session's host half, offline: ingress, egress, the input plane, routing.

WHAT THIS CHECKS. Everything ``Bf16DenseGraphSession`` does on the host, each against something it does
not share code with:

* the staged input plane against the COMPILER's workspace layout (``engine_schedule_bf16.input_plane``) of
  the float pipeline's own input (``npu/sesr.py``) - the ingress half of the device == emulator link;
* the image against ``npu/sesr.py``'s own postprocess, fed by a DepthToSpace written out from its
  definition rather than as the session's transpose;
* the lengths handed to the buffer-object syncs, through a stand-in that records them. At two bytes an
  element a half-length sync is the failure that raises nothing;
* routing and admission.

The manifests carry the int8 leftovers a real bf16 manifest does (``quant_scales``, the dense output's
scale and zero point), set to values that would change every result if anything read them.

WHAT IT CANNOT CHECK. No device is opened. The session is made with ``object.__new__`` and handed the
state ``EngineSession.__init__`` leaves behind, the int8 zero-point fill of the input plane included,
because that fill is the trap. If that constructor changes, this stand-in does not follow it. Whether
pyxrt, the syncs and the kernel then do what these calls ask is for a sitting on the device.
"""
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from ignite_xdna.compiler import engine_schedule_bf16 as eb
from ignite_xdna.compiler.engine_bf16_emulator import bf16_bits, from_bf16_bits, to_bf16
from ignite_xdna.compiler.engine_schedule import Placement
from ignite_xdna.compiler.serializer import (ELEM_BF16, ELEM_INT, ENGINE_CONV_BF16, ENGINE_CONV_INT8,
                                             IgniteModelReader, IgniteModelWriter)
from ignite_xdna.runtime.graph_session import (ZP, Bf16DenseGraphSession, DenseGraphSession, EngineSession,
                                               bf16_input_lut, container_elem, halo_fill_image, placement_dtype,
                                               sr_session_class)

ROOT = Path(__file__).resolve().parents[1]
BF16_SESR = ROOT / "build" / "sesr_m7_bf16.ignite"
H_IN, W_IN = 12, 10      # not square, so a swapped height and width shows
NAN, INF = 0x7FC0, 0x7F80


def sr_manifest(h=H_IN, w=W_IN, halo=2):
    """A bf16 super-resolution manifest, int8 leftovers included and set where they would show."""
    in_bytes = (h + 2 * halo) * (w + 2 * halo) * 8 * 2
    out_base = (in_bytes + 63) // 64 * 64 + 128
    placement = {"halo_value": 0, "height": h, "width": w, "band_rows": 0, "dtype": "uint16",
                 "storage": "workspace", "elem": ELEM_BF16}
    return {"engine": ENGINE_CONV_BF16, "task": "super_resolution", "input_shape": [1, 3, h, w],
            "input_dtype": "uint8", "egress_bytes": 12 * h * w * 2, "upscale": 2,
            "input_normalization": {"mean": 128.0, "divisor": 1.0},
            "quant_scales": {"input_scale": 0.5, "input_zero_point": 7, "input_dtype": "uint8"},
            "dense_output": {"tensor": "tail", "channels": 12, "height": h, "width": w, "scale": 3.0,
                             "zero_point": 9, "layout": "blocks_hw8",
                             "transform": {"op": "depth_to_space", "blocksize": 2, "mode": "CRD"}},
            "graph_engine": {"input_tensor": "image", "workspace_bytes": out_base + 2 * h * w * 8 * 2 + 64,
                             "placements": {
                                 "image": dict(placement, base=0, halo=halo, blocks=1, planes=1, channels=3,
                                               scale=0.5, zero_point=7),
                                 "tail": dict(placement, base=out_base, halo=0, blocks=2, planes=2, channels=12,
                                              scale=3.0, zero_point=9)}}}


class FakeBO:
    """A buffer object over a host array. It records every sync, and writes only bytes: what pyxrt does
    with a wider array is not known here, and a write that counted elements as bytes would send half."""

    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.syncs = []

    def sync(self, direction, nbytes, offset):
        self.syncs.append((direction, int(nbytes), int(offset)))

    def write(self, data, offset):
        data = np.asarray(data)
        if data.dtype != np.uint8:
            raise TypeError(f"write takes bytes, got {data.dtype}")
        flat = np.ascontiguousarray(data).reshape(-1)
        self.arr[offset:offset + flat.size] = flat

    def read(self, nbytes, offset):
        return self.arr[offset:offset + nbytes].tobytes()


def offline_session(m, mapped=True) -> Bf16DenseGraphSession:
    """A session as ``EngineSession.__init__`` would leave it, with no device, then ``_init_dense``."""
    s = object.__new__(Bf16DenseGraphSession)
    s.path = Path("offline.ignite")
    s.ignite_manifest, s.ge, s.task = m, m["graph_engine"], m["task"]
    s.elem = container_elem(m)
    ws = halo_fill_image(s.ge)            # the runtime's own initial image
    s.bo_ws = FakeBO(ws)
    s.harness = SimpleNamespace(pyxrt=SimpleNamespace(xclBOSyncDirection=SimpleNamespace(
        XCL_BO_SYNC_BO_TO_DEVICE="to", XCL_BO_SYNC_BO_FROM_DEVICE="from")))
    p = s.input_placement = s.ge["placements"][s.ge["input_tensor"]]
    shape = (p["height"] + 2 * p["halo"], p["width"] + 2 * p["halo"], 8)
    s._input_dtype = placement_dtype(p)
    s._input_bytes = int(np.prod(shape)) * s._input_dtype.itemsize
    if mapped:
        s._ws_map = ws
        s._input_plane = ws[p["base"]:p["base"] + s._input_bytes].view(s._input_dtype).reshape(shape)
        s._input_plane[:] = ZP            # what EngineSession.__init__ does, at every width
    else:
        s._ws_map = None
        s._input_plane = np.full(shape, ZP, dtype=s._input_dtype)
    s._init_dense()
    return s


def compiler_workspace(m) -> eb.Bf16Workspace:
    """The compiler's view of a manifest's workspace, to plant and read tensors the way the DMA sees them."""
    ge = m["graph_engine"]
    return eb.Bf16Workspace(
        placements={name: Placement(name=name, base=p["base"], halo=p["halo"], height=p["height"],
                                    width=p["width"], blocks=p["blocks"], planes=p["planes"],
                                    halo_value=p["halo_value"], dtype=p["dtype"], band_rows=p["band_rows"])
                    for name, p in ge["placements"].items()},
        nbytes=int(ge["workspace_bytes"]), input=ge["input_tensor"])


def float_input(img_bgr, h, w) -> np.ndarray:
    """The float pipeline's input at any size, as ``npu.sesr.preprocess`` makes it at a square one."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.transpose(rgb.astype(np.float32) - 128.0, (2, 0, 1))


def depth_to_space_crd(x: np.ndarray, bs: int) -> np.ndarray:
    """DepthToSpace, CRD, from its definition: channel k*bs*bs + i*bs + j is pixel (y*bs + i, x*bs + j) of k."""
    c, h, w = x.shape
    out = np.empty((c // (bs * bs), h * bs, w * bs), x.dtype)
    for k in range(c // (bs * bs)):
        for i in range(bs):
            for j in range(bs):
                out[k, i::bs, j::bs] = x[k * bs * bs + i * bs + j]
    return out


def tail_values(h, w, seed=3) -> np.ndarray:
    """12 channels of bf16 values across the clip edges, with fractions for the truncation to act on."""
    v = np.random.default_rng(seed).normal(scale=90.0, size=(12, h, w)).astype(np.float32)
    v.flat[:12] = [-0.0, 0.0, -0.4, 0.99, -128.5, -129.0, 127.5, 128.0, 300.0, -300.0, 1e-30, -1e-30]
    return to_bf16(v)


class TestIngressTable(unittest.TestCase):

    def test_sesrs_table_is_the_centred_pixel_exactly(self):
        lut = bf16_input_lut({"mean": 128.0, "divisor": 1.0})
        self.assertEqual(lut.dtype, np.uint16)
        np.testing.assert_array_equal(from_bf16_bits(lut), np.arange(256, dtype=np.float32) - 128.0)

    def test_a_divisor_other_than_one_is_refused_because_the_output_units_are_not_known(self):
        m = sr_manifest()
        m["input_normalization"] = {"mean": 128.0, "divisor": 255.0}
        with self.assertRaises(ValueError) as cm:
            offline_session(m)
        self.assertIn("divisor", str(cm.exception))


class TestTheInputPlane(unittest.TestCase):

    def setUp(self):
        self.img = np.random.default_rng(1).integers(0, 256, size=(17, 23, 3), dtype=np.uint8)

    def test_the_stand_in_copies_the_fill_engine_session_really_does(self):
        # offline_session replicates this line of EngineSession.__init__. If it changes, the stand-in no
        # longer models the real constructor and the zeroing test below proves nothing about it.
        self.assertIn("self._input_plane[:] = ZP", inspect.getsource(EngineSession.__init__))

    def test_the_int8_zero_point_fill_is_gone_before_anything_is_synced(self):
        # 0x0080 is a bf16 denormal: finite, so no 0 x NaN fires, and wrong in the halo and lanes 3..7.
        for mapped in (True, False):
            s = offline_session(sr_manifest(), mapped=mapped)
            self.assertFalse(np.any(s._input_plane), f"mapped={mapped}")
            self.assertEqual(s.bo_ws.syncs, [])

    def test_a_staged_image_is_the_compilers_layout_of_the_float_pipelines_input(self):
        m = sr_manifest()
        want = eb.input_plane(m["graph_engine"]["placements"]["image"], float_input(self.img, H_IN, W_IN))
        for mapped in (True, False):
            s = offline_session(m, mapped=mapped)
            s.stage_image(self.img)
            np.testing.assert_array_equal(s._input_plane, want, err_msg=f"mapped={mapped}")
            # What reached the buffer object, which is what the device would sync.
            np.testing.assert_array_equal(s.bo_ws.arr[:want.nbytes].view(np.uint16).reshape(want.shape), want)

    def test_the_upload_syncs_the_whole_plane_at_two_bytes_an_element(self):
        s = offline_session(sr_manifest())
        s.stage_image(self.img)
        self.assertEqual(s.bo_ws.syncs, [("to", (H_IN + 4) * (W_IN + 4) * 8 * 2, 0)])

    def test_an_image_at_the_network_size_is_staged_without_a_resize(self):
        img = self.img[:H_IN, :W_IN].copy()
        s = offline_session(sr_manifest())
        s.stage_image(img)
        want = eb.input_plane(s.input_placement, np.transpose(img[:, :, ::-1].astype(np.float32) - 128, (2, 0, 1)))
        np.testing.assert_array_equal(s._input_plane, want)

    def test_the_halo_and_the_lanes_past_the_image_stay_zero_across_frames(self):
        s = offline_session(sr_manifest())
        for seed in (4, 5):
            s.stage_image(np.random.default_rng(seed).integers(0, 256, size=(30, 9, 3), dtype=np.uint8))
        plane = s._input_plane
        self.assertFalse(np.any(plane[2:-2, 2:-2, 3:]))
        ring = np.ones(plane.shape[:2], bool)
        ring[2:-2, 2:-2] = False
        self.assertFalse(np.any(plane[ring]))

    def test_stage_quantized_takes_patterns_and_zeroes_the_lanes_it_does_not_write(self):
        s = offline_session(sr_manifest())
        with self.assertRaises(TypeError):
            s.stage_quantized(np.zeros((3, H_IN, W_IN), np.float32))
        rng = np.random.default_rng(6)
        s.stage_quantized(bf16_bits(to_bf16(rng.normal(size=(5, H_IN, W_IN)).astype(np.float32))))
        three = bf16_bits(to_bf16(rng.normal(size=(3, H_IN, W_IN)).astype(np.float32)))
        s.stage_quantized(three)
        np.testing.assert_array_equal(s._input_plane, eb.input_plane(s.input_placement, from_bf16_bits(three)))
        with self.assertRaises(ValueError):
            s.stage_quantized(three[:, :, :-1])

    def test_stage_tensor_is_refused_because_it_pads_with_the_int8_zero_point(self):
        with self.assertRaises(NotImplementedError):
            offline_session(sr_manifest()).stage_tensor("image", np.zeros((3, H_IN, W_IN), np.uint16))

    def test_a_frame_that_is_not_uint8_bgr_is_refused(self):
        s = offline_session(sr_manifest())
        for bad in (self.img.astype(np.float32), self.img[:, :, :2]):
            with self.assertRaises(ValueError):
                s.stage_image(bad)


class TestEgress(unittest.TestCase):

    def _planted(self, m, mapped=True, values=None):
        s = offline_session(m, mapped=mapped)
        dense = m["dense_output"]
        v = tail_values(dense["height"], dense["width"]) if values is None else values
        compiler_workspace(m).write_values(s.bo_ws.arr, dense["tensor"], v)
        return s, v

    def test_read_output_returns_the_planted_tail_and_syncs_all_of_it(self):
        m = sr_manifest()
        tail = m["graph_engine"]["placements"]["tail"]
        for mapped in (True, False):
            s, v = self._planted(m, mapped)
            blocks = s.read_output()
            self.assertEqual(blocks.dtype, np.uint16)
            got = np.moveaxis(blocks, 3, 1).reshape(16, H_IN, W_IN)
            np.testing.assert_array_equal(got[:12], bf16_bits(v))
            self.assertFalse(np.any(got[12:]))
            self.assertEqual(s.bo_ws.syncs, [("from", 2 * H_IN * W_IN * 8 * 2, tail["base"])])

    def test_the_image_is_the_float_pipelines_own_postprocess(self):
        from npu.sesr import postprocess
        s, v = self._planted(sr_manifest())
        image = s.postprocess(s.read_output())
        self.assertEqual((image.shape, image.dtype), ((2 * H_IN, 2 * W_IN, 3), np.dtype(np.uint8)))
        np.testing.assert_array_equal(image, postprocess(depth_to_space_crd(v, 2)))

    def test_a_non_finite_value_in_a_real_channel_raises(self):
        for pattern in (NAN, INF):
            s, _ = self._planted(sr_manifest())
            blocks = s.read_output().copy()
            blocks[1, 3, 4, 2] = pattern          # channel 10
            with self.assertRaises(RuntimeError) as cm:
                s.postprocess(blocks)
            self.assertIn("non-finite", str(cm.exception))

    def test_a_non_finite_value_in_a_padding_lane_is_not_part_of_the_image(self):
        s, v = self._planted(sr_manifest())
        blocks = s.read_output().copy()
        blocks[1, 3, 4, 6] = NAN                  # lane 14 of a 12-channel tensor
        from npu.sesr import postprocess
        np.testing.assert_array_equal(s.postprocess(blocks), postprocess(depth_to_space_crd(v, 2)))


class _Opened(Exception):
    pass


def _opened(self, *args, **kwargs):
    raise _Opened(type(self).__name__)


class TestRouting(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _container(self, m, name="sr.ignite") -> str:
        path = self.tmp / name
        IgniteModelWriter(manifest_meta=m).write(path)
        return str(path)

    @staticmethod
    def _int8(m):
        m = dict(m, engine=ENGINE_CONV_INT8)
        m["graph_engine"] = dict(m["graph_engine"], placements={
            k: {kk: vv for kk, vv in dict(p, dtype="uint8").items() if kk != "elem"}
            for k, p in m["graph_engine"]["placements"].items()})
        return m

    def test_the_class_follows_the_declared_width(self):
        self.assertIs(sr_session_class(sr_manifest()), Bf16DenseGraphSession)
        self.assertIs(sr_session_class(self._int8(sr_manifest())), DenseGraphSession)

    def test_from_file_and_the_pipeline_open_the_bf16_session_for_a_bf16_container(self):
        from ignite_xdna.pipelines.sr_pipeline import SuperResolutionPipeline
        from ignite_xdna.runtime.session import InferenceSession
        bf16_path = self._container(sr_manifest(), "bf16.ignite")
        int8_path = self._container(self._int8(sr_manifest()), "int8.ignite")
        with mock.patch.object(DenseGraphSession, "__init__", _opened):
            for opener in (InferenceSession.from_file, SuperResolutionPipeline):
                for path, want in ((bf16_path, "Bf16DenseGraphSession"), (int8_path, "DenseGraphSession")):
                    with self.assertRaises(_Opened) as cm:
                        opener(path)
                    self.assertEqual(str(cm.exception), want, f"{opener} on {Path(path).name}")

    def test_the_bf16_session_refuses_an_int8_container_naming_the_width(self):
        path = self._container(self._int8(sr_manifest()))
        with self.assertRaises(ValueError) as cm:
            Bf16DenseGraphSession(path)
        self.assertIn(repr(ELEM_INT), str(cm.exception))
        self.assertIn("Bf16DenseGraphSession", str(cm.exception))

    def test_the_bf16_session_is_not_refused_on_width_by_a_bf16_container(self):
        # It fails later for want of an xclbin, before any device is opened; what must not happen is
        # the width refusal.
        path = self._container(sr_manifest())
        with self.assertRaises(Exception) as cm:
            Bf16DenseGraphSession(path)
        self.assertNotIn("activations but", str(cm.exception))


@unittest.skipUnless(BF16_SESR.exists(), "build/sesr_m7_bf16.ignite not built")
class TestTheFirstBf16Container(unittest.TestCase):
    """The session's host half on SESR-M7's real manifest. No device is opened."""

    @classmethod
    def setUpClass(cls):
        with IgniteModelReader(str(BF16_SESR)) as reader:
            cls.m = reader.manifest

    def test_it_reads_the_real_geometry_and_zeroes_the_plane(self):
        s = offline_session(self.m)
        self.assertEqual((s.input_hw, s._out_blocks, s._bs, s._image_channels), ((256, 256), 2, 2, 3))
        self.assertEqual(s._out_region, 2 * 256 * 256 * 8 * 2)
        self.assertFalse(np.any(s._input_plane))

    def test_a_staged_image_is_the_compilers_layout_of_npu_sesr_preprocess(self):
        from npu.sesr import preprocess
        img = np.random.default_rng(7).integers(0, 256, size=(141, 203, 3), dtype=np.uint8)
        s = offline_session(self.m)
        s.stage_image(img)
        want = eb.input_plane(s.input_placement, preprocess(img)[0][0])
        np.testing.assert_array_equal(s._input_plane, want)
        self.assertEqual(s.bo_ws.syncs, [("to", 260 * 260 * 8 * 2, 0)])

    def test_a_planted_tail_comes_back_as_npu_sesr_postprocess(self):
        from npu.sesr import postprocess
        s = offline_session(self.m)
        v = tail_values(256, 256)
        compiler_workspace(self.m).write_values(s.bo_ws.arr, self.m["dense_output"]["tensor"], v)
        np.testing.assert_array_equal(s.postprocess(s.read_output()), postprocess(depth_to_space_crd(v, 2)))


if __name__ == "__main__":
    unittest.main()
