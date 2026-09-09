"""The catalogue: every task and demo the TUI can run, as data.

Nothing outside this module knows a model path or a flag name. Two reasons it is
declarative rather than a pile of if-statements:

1. The run-stage entry points do not agree with each other. yolov8n-pose has no
   `dml` choice; `--ep` defaults to cpu in resnet50/4_run.py and every yolo
   4_detect but npu everywhere else; `--fresh` is NPU-gated in bisenetv2 /
   fastdepth / midas / realesrgan / sesr but unconditional in resnet50 / modnet /
   all-yolo; verbosity is `--log {0,1,2}` in yolo and boolean `--verbose`
   elsewhere. A launcher that assumes one dialect builds a command line argparse
   rejects, or worse, one it silently mis-reads. FlagDialect records the
   differences per entry instead.

2. The compile cache is keyed by NAME, not by model hash (npu/session.py), so
   every entry has to be able to say which cache key its model lands in. Those
   come from npu.paths -- never a hand-rolled substring test, which is the exact
   bug npu/paths.modnet_cache_key exists to prevent.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from npu import paths


# --------------------------------------------------------------------------
# Cache-key resolution
# --------------------------------------------------------------------------
# A cache key is either a constant or a function of the model path. Both forms
# come from npu.paths so this module never re-implements the marker ordering
# that modnet_cache_key/realesrgan_cache_key get right.
CacheKey = Union[str, Callable[[str], str]]


def resolve_cache_key(key: CacheKey, model_relpath: str) -> str:
    """Turn a cache_key field into the actual directory name."""
    return key(model_relpath) if callable(key) else key


def cache_key_for(entry, choice) -> str:
    """The compile cache a given (entry, model) lands in.

    A ModelChoice's own key wins, so a family whose variants need separate caches
    can say so without npu.paths growing a resolver it does not have.
    """
    return resolve_cache_key(choice.cache_key or entry.cache_key, choice.relpath)


@dataclass(frozen=True)
class ModelChoice:
    """One selectable model within an entry.

    `relpath` is relative to models/ so it reads the same as the docs do.
    `note` is shown in the picker -- use it for the accuracy/speed trade-off,
    not for provenance.

    `cache_key` overrides the entry's when a family has no resolver in npu.paths
    but its variants still need separate caches. SESR is the case that forced
    this: sesr_m7_xint8 and sesr_m7_xint8_adaround are different graphs, and
    pipelines/sesr/4_upscale.py picks between SESR_CACHE_KEY and
    SESR_ADAROUND_CACHE_KEY with an inline `"adaround" in path` test. Naming the
    key per model here is explicit and cannot drift; a substring test copied into
    this file could.
    """
    label: str
    relpath: str
    note: str = ""
    cache_key: Optional[CacheKey] = None

    @property
    def path(self):
        return paths.MODELS / self.relpath


@dataclass(frozen=True)
class FlagDialect:
    """How one demo script spells the flags every demo roughly shares.

    Defaults describe the yolo-family demos, which are the majority. Every field
    that differs for a given script is set explicitly in DEMOS below, with the
    source line that proves it.
    """
    ep_flag: Optional[str] = "--ep"       # None: the script has no --ep at all
    source_flag: Optional[str] = "--source"
    out_dir_flag: Optional[str] = "--out-dir"
    verbosity_flag: Optional[str] = "--log"   # or "--verbose" (boolean), or None
    runs_flag: Optional[str] = "--runs"
    fresh_flag: Optional[str] = None      # demos that expose --fresh
    extra: tuple = ()                     # fixed args always appended


@dataclass(frozen=True)
class Entry:
    key: str
    lane: str                 # "task" | "demo"
    title: str
    blurb: str
    family: str               # selects the npu/<family> adapter, or "" for demos
    models: tuple = ()
    cache_key: CacheKey = ""
    eps: tuple = ("npu", "cpu")
    default_ep: str = "npu"
    inputs: tuple = ("sample",)          # image | video | webcam | sample | none
    sample: str = ""                     # repo-relative default input

    # Demo-only
    script: str = ""                     # repo-relative path to the .py
    dialect: FlagDialect = field(default_factory=FlagDialect)
    recompiles: int = 0                  # dominates wall clock; shown as an ETA
    clears_cache_on_cpu: bool = False    # --ep cpu still rmtree's the NPU cache
    touches_cache_keys: tuple = ()       # every key this entry may clear

    # Safety / privacy
    needs_person: bool = False
    writes_camera_frames: tuple = ()     # committed dirs it writes frames into
    needs_1x4_xclbin: bool = False

    @property
    def is_task(self) -> bool:
        return self.lane == "task"


# --------------------------------------------------------------------------
# Lane 1: real tasks, run in-process over npu/<family>
# --------------------------------------------------------------------------
# Each of these maps onto a module in npu/ that already exposes
# preprocess()/postprocess_*()/a renderer. The adapters in tui/tasks.py call
# those shared functions -- they never inline a second preprocess, which is what
# went wrong when MODNet's transform existed in five places and two disagreed.

TASKS = (
    Entry(
        key="matte", lane="task", family="modnet",
        title="Remove background",
        blurb="Portrait matting -- cut a person out, blur or replace the background.",
        models=(
            ModelChoice("Zero-Concat (fast)", "modnet/modnet_zero_concat_xint8_calibfix.onnx",
                        "~17.8 ms; 1.85x the alpha error of Cut"),
            ModelChoice("Cut (accurate)", "modnet/modnet_cut_xint8_calibfix.onnx",
                        "~26.4 ms; the lower-error graph"),
        ),
        cache_key=paths.modnet_cache_key,
        eps=("npu", "dml", "cpu"), default_ep="npu",
        inputs=("image", "webcam", "sample"),
        sample="data/modnet_val",
    ),
    Entry(
        key="detect", lane="task", family="yolov8",
        title="Detect objects",
        blurb="COCO detection, 80 classes, boxes drawn on the image.",
        models=(
            ModelChoice("YOLOv8n", "yolov8n_cut_xint8.onnx", "fastest"),
            ModelChoice("YOLOv8n AdaRound", "yolov8n_cut_xint8_adaround.onnx",
                        "same speed, recovered accuracy"),
            ModelChoice("YOLOv8s", "yolov8s_cut_xint8.onnx", ""),
            ModelChoice("YOLOv8m", "yolov8m_cut_xint8.onnx", "most accurate, slowest"),
        ),
        cache_key=paths.YOLO_CUT_CACHE_KEY,
        eps=("npu", "dml", "cpu"), default_ep="npu",
        inputs=("image", "video", "webcam", "sample"),
        sample="assets/test_image.jpg",
    ),
    Entry(
        key="detect-v6", lane="task", family="yolov6",
        title="Detect objects (YOLOv6n)",
        blurb="RepVGG backbone, no DFL -- a different detector shape on the same fabric.",
        models=(
            ModelChoice("YOLOv6n", "yolov6n_cut_xint8.onnx", ""),
            ModelChoice("YOLOv6n AdaRound", "yolov6n_cut_xint8_adaround.onnx",
                        "33.57 vs 22.92 mAP -- AdaRound matters a lot here"),
        ),
        cache_key=paths.YOLOV6_CUT_CACHE_KEY,
        eps=("npu", "dml", "cpu"), default_ep="npu",
        inputs=("image", "video", "webcam", "sample"),
        sample="assets/test_image.jpg",
    ),
    Entry(
        key="pose", lane="task", family="yolo_pose",
        title="Estimate pose",
        blurb="17-point COCO keypoints with the skeleton drawn on.",
        models=(
            ModelChoice("YOLOv8n-pose", "yolov8n-pose_cut_xint8.onnx", ""),
            ModelChoice("YOLOv8n-pose AdaRound", "yolov8n-pose_cut_xint8_adaround.onnx",
                        "34.32 vs 32.64 OKS mAP"),
        ),
        cache_key=paths.YOLO_POSE_CUT_CACHE_KEY,
        # The pose pipeline offers only {cpu,npu}; dml is untested for this graph,
        # so the TUI does not invent support the repo has never measured.
        eps=("npu", "cpu"), default_ep="npu",
        inputs=("image", "video", "webcam", "sample"),
        sample="assets/test_image.jpg",
    ),
    Entry(
        key="depth", lane="task", family="fastdepth",
        title="Depth map (FastDepth)",
        blurb="Monocular depth, colourised. 2.87 ms -- beats the iGPU.",
        models=(ModelChoice("FastDepth", "fastdepth_fp32_xint8.onnx", ""),),
        cache_key=paths.FASTDEPTH_CACHE_KEY,
        eps=("npu", "dml", "cpu"), default_ep="npu",
        inputs=("image", "webcam", "sample"),
        sample="data/fastdepth_val",
    ),
    Entry(
        key="depth-midas", lane="task", family="midas",
        title="Depth map (MiDaS)",
        blurb="Heavier depth model, 10.81 ms, higher correlation with ground truth.",
        models=(ModelChoice("MiDaS small (nearest-resize cut)",
                            "midas_small_nearest_cut_xint8.onnx",
                            "the single-subgraph variant"),),
        cache_key=paths.MIDAS_NEAREST_CACHE_KEY,
        eps=("npu", "dml", "cpu"), default_ep="npu",
        inputs=("image", "webcam", "sample"),
        sample="data/midas_val",
    ),
    Entry(
        key="upscale", lane="task", family="sesr",
        title="Upscale image 2x",
        blurb="SESR-M7 super-resolution, 1.48 ms a tile. Tiles a full-size photo.",
        models=(
            ModelChoice("SESR-M7", "sesr_m7_xint8.onnx", "",
                        cache_key=paths.SESR_CACHE_KEY),
            ModelChoice("SESR-M7 AdaRound", "sesr_m7_xint8_adaround.onnx", "",
                        cache_key=paths.SESR_ADAROUND_CACHE_KEY),
        ),
        cache_key=paths.SESR_CACHE_KEY,
        eps=("npu", "dml", "cpu"), default_ep="npu",
        inputs=("image", "sample"),
        sample="data/sesr_val/Set5_LR_x2",
    ),
    Entry(
        key="upscale-4x", lane="task", family="realesrgan",
        title="Upscale image 4x",
        blurb="Real-ESRGAN, 14.02 ms a tile -- beats the iGPU. Tiles a full-size photo.",
        models=(
            ModelChoice("Compact (fast)", "realesrgan_compact_r64_xint8.onnx", ""),
            ModelChoice("RRDB (heavier)", "realesrgan_rrdb_r64_xint8.onnx", ""),
            ModelChoice("RRDB AdaRound", "realesrgan_rrdb_r64_xint8_adaround.onnx", ""),
        ),
        cache_key=paths.realesrgan_cache_key,
        eps=("npu", "dml", "cpu"), default_ep="npu",
        inputs=("image", "sample"),
        sample="data/realesrgan_val/Set5_LR_x4",
    ),
    Entry(
        key="segment", lane="task", family="bisenetv2",
        title="Segment a street scene",
        blurb="BiSeNetV2, 19 Cityscapes classes, 13.12 ms -- beats the iGPU.",
        models=(ModelChoice("BiSeNetV2", "bisenetv2_fp32_xint8.onnx",
                            "INT8 attenuates minority classes -- see BENCHMARKS"),),
        cache_key=paths.BISENETV2_CACHE_KEY,
        eps=("npu", "dml", "cpu"), default_ep="npu",
        inputs=("image", "sample"),
        sample="data/bisenetv2_val",
    ),
    # No `classify` task. ResNet50 is the repo's best-characterised model, but there
    # is no ImageNet index->name mapping anywhere here -- data/eval/labels.json is
    # filename->index, and DECISIONS' locked entry records the label column as an
    # "int class index (sorted-wnid order)". A task that answers "285" is not a task
    # anyone can use, and shipping a 1000-name list is a new tracked asset with its
    # own licensing question, not a launcher decision. ResNet stays represented
    # through the demo lane (classifier_width, resolution_ladder, batch_failure).
)


# --------------------------------------------------------------------------
# Lane 2: demos, launched as subprocesses
# --------------------------------------------------------------------------
# These are standalone scripts that own their own cv2 windows, so they are run
# rather than imported. `recompiles` is the number of clear_cache() calls each
# makes -- that, not inference, is what the wall clock is made of.
#
# `clears_cache_on_cpu` records the trap: six of eight call clear_cache()
# unconditionally, so a "quick CPU preview" still destroys the NPU compile and
# the next NPU run pays a full recompile. Only tri_hardware_showdown guards it.

_YOLO_CUT = (paths.YOLO_CUT_CACHE_KEY,)
_RESNET = (paths.RESNET_CACHE_KEY,)

DEMOS = (
    Entry(
        key="demo-conf-sweep", lane="demo", family="",
        title="Confidence threshold ladder",
        blurb="The long tail mAP integrates over that a 0.25 demo window never shows.",
        script="demos/conf_sweep_demo.py",
        dialect=FlagDialect(runs_flag=None),
        recompiles=1, clears_cache_on_cpu=True, touches_cache_keys=_YOLO_CUT,
        eps=("npu", "cpu"), default_ep="npu",
        inputs=("image", "sample"), sample="assets/test_image.jpg",
    ),
    Entry(
        key="demo-batch-failure", lane="demo", family="",
        title="Batch > 1 returns a stale buffer",
        blurb="Slot 1 comes back byte-identical -- unwritten RAM, plausible latency, no error.",
        script="demos/batch_failure_demo.py",
        # No --ep and no --source: it runs cpu and npu itself, on fixed images.
        dialect=FlagDialect(ep_flag=None, source_flag=None, runs_flag=None),
        recompiles=1, clears_cache_on_cpu=False, touches_cache_keys=_RESNET,
        eps=(), default_ep="", inputs=("none",),
    ),
    Entry(
        key="demo-adaround-diff", lane="demo", family="",
        title="AdaRound vs plain XINT8, box by box",
        blurb="Bipartite IoU matching: which boxes AdaRound recovers and which XINT8 invents.",
        script="demos/adaround_diff_demo.py",
        recompiles=2, clears_cache_on_cpu=True, touches_cache_keys=_YOLO_CUT,
        eps=("npu", "cpu"), default_ep="npu",
        inputs=("image", "sample"), sample="assets/test_image.jpg",
    ),
    Entry(
        key="demo-pose-adaround", lane="demo", family="",
        title="Keypoint jitter and AdaRound",
        blurb="8.82 px mean joint drift, drawn as displacement vectors.",
        script="demos/pose_adaround_demo.py",
        recompiles=2, clears_cache_on_cpu=True,
        touches_cache_keys=(paths.YOLO_POSE_CUT_CACHE_KEY,),
        eps=("npu", "cpu"), default_ep="npu",
        inputs=("image", "sample"), sample="assets/test_image.jpg",
    ),
    Entry(
        key="demo-classifier-width", lane="demo", family="",
        title="Width scaling across classifiers",
        blurb="ResNet50 vs Wide-ResNet50-2 vs Wide-ResNet101-2: width is nearly free.",
        script="demos/classifier_width_demo.py",
        recompiles=3, clears_cache_on_cpu=True, touches_cache_keys=_RESNET,
        eps=("npu", "cpu"), default_ep="npu",
        inputs=("image", "sample"), sample="data/calib/000000.jpg",
    ),
    Entry(
        key="demo-width-ladder", lane="demo", family="",
        title="YOLO width ladder",
        blurb="n/s/m/x in one sitting -- 9.1x the FLOPs for 4.19x the latency.",
        script="demos/width_ladder_demo.py",
        # --out-dir here resolves against the CWD, not ROOT (width_ladder:126).
        # runner.py always passes an absolute path, which makes that moot.
        recompiles=4, clears_cache_on_cpu=True, touches_cache_keys=_YOLO_CUT,
        eps=("npu", "cpu"), default_ep="npu",
        inputs=("image", "sample"), sample="assets/test_image.jpg",
    ),
    Entry(
        key="demo-resolution-ladder", lane="demo", family="",
        title="Dispatch floor and compute knee",
        blurb="ResNet50 128 to 384 sq. Below 224, 3.07x the FLOPs costs 1.63x the latency.",
        script="demos/resolution_ladder_demo.py",
        recompiles=5, clears_cache_on_cpu=True, touches_cache_keys=_RESNET,
        eps=("npu", "cpu"), default_ep="npu",
        inputs=("image", "sample"), sample="data/calib/000000.jpg",
    ),
    Entry(
        key="demo-tri-hardware", lane="demo", family="",
        title="CPU vs iGPU vs NPU, same sitting",
        blurb="The only demo that touches all three providers in one run.",
        script="demos/tri_hardware_showdown_demo.py",
        # No --ep: the four pathways are hardcoded. This is also the only demo
        # that guards clear_cache with `if ep == npu` (tri_hardware:92).
        dialect=FlagDialect(ep_flag=None),
        recompiles=1, clears_cache_on_cpu=False, touches_cache_keys=_YOLO_CUT,
        eps=(), default_ep="", inputs=("image", "sample"),
        sample="assets/test_image.jpg",
    ),
    Entry(
        key="demo-portrait-matting", lane="demo", family="",
        title="Live portrait matting (webcam)",
        blurb="Bokeh, green screen, live EP cycling. Needs a person at the machine.",
        script="demos/portrait_matting_demo.py",
        dialect=FlagDialect(out_dir_flag=None, verbosity_flag=None,
                            runs_flag=None, fresh_flag="--fresh"),
        recompiles=0, touches_cache_keys=(paths.MODNET_ZERO_CONCAT_CACHE_KEY,),
        eps=("npu", "dml", "cpu"), default_ep="npu",
        inputs=("webcam", "video"), sample="0",
        needs_person=True,
        # Two places, and they are not equally bad. data/ is wholesale git-ignored
        # (.gitignore:2, zero tracked files), so the 't' key is a consent question --
        # 25 photos of you land on disk without asking -- not a commit hazard. The
        # 'p' key is the sharp one: results/ IS tracked, 69 of its images are cited
        # from the docs, and a snapshot there is one `git add -A` from being public.
        # --snapshot-dir was added to the demo so the TUI can redirect it.
        writes_camera_frames=("data/user_calib/ -- 25 frames, the 't' key, git-ignored",
                              "results/modnet/ -- the 'p' key, TRACKED unless redirected"),
    ),
    Entry(
        key="demo-multipartition", lane="demo", family="",
        title="Live round-robin across 4 NPU partitions (webcam)",
        blurb="Four single-column partitions serving one camera. Needs the 1x4 xclbin.",
        script="demos/webcam_multipartition_demo.py",
        dialect=FlagDialect(ep_flag=None, out_dir_flag=None, verbosity_flag=None,
                            runs_flag=None, fresh_flag="--fresh"),
        recompiles=0, touches_cache_keys=(paths.YOLO_CUT_1X4_CACHE_KEY,),
        eps=(), default_ep="", inputs=("webcam", "video"), sample="0",
        needs_1x4_xclbin=True,
    ),
)


ALL = TASKS + DEMOS


def by_key(key: str) -> Entry:
    for e in ALL:
        if e.key == key:
            return e
    raise KeyError(f"no registry entry {key!r}")


def lane(name: str) -> tuple:
    return tuple(e for e in ALL if e.lane == name)
