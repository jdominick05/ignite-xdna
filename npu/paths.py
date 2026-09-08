"""Repo-relative paths, so scripts behave the same from any working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"

# Ryzen AI compile caches. These stay at the repo root: the cache is keyed by
# the hardcoded cacheKey below rather than by a model hash, and the existing
# compiled artifacts were built against this location.
CACHE_DIR = ROOT
RESNET_CACHE_KEY = "modelcachekey"
YOLO_CACHE_KEY = "yolocachekey"
# The head-cut YOLO model gets its own key so it can never collide with a
# compile cached from the full-graph model.
YOLO_CUT_CACHE_KEY = "yolocutcachekey"
YOLO_POSE_CUT_CACHE_KEY = "yoloposecutcachekey"
# Same model, the 1-column overlay instead of 4x4 (tools/multi_partition_bench.py)
# -- its own key so a 1x4 compile can never collide with the 4x4 one above.
YOLO_CUT_1X4_CACHE_KEY = "yolocut1x4cachekey"
# yolov6n: a different architecture (RepVGG backbone, no DFL), own key so its
# compile can never collide with the yolov8-family caches above.
YOLOV6_CUT_CACHE_KEY = "yolov6cutcachekey"
# MobileViT: the attention-free CNN (mobilevit_cut_backbone_xint8.onnx) and the
# stock 49-subgraph graph. Both sit at the repo root like every key above.
# tools/demo_attention.py and tools/pipeline_splice_bench.py once hardcoded
# cacheDir=ROOT/modelcachekey with cacheKey=mobilevit_cut, nesting a second copy
# inside resnet50's cache -- use these constants, not a hand-rolled dict.
MOBILEVIT_CUT_CACHE_KEY = "mobilevit_cut"
MOBILEVIT_STOCK_CACHE_KEY = "mobilevit_stock"
# MODNet ships four quantized variants and each needs its own key. The two
# underscore-named dirs already hold compiled artifacts under those names, so they keep
# them rather than being renamed into the `*cachekey` shape used above.
MODNET_CACHE_KEY = "modnetcachekey"
MODNET_CUT_CACHE_KEY = "modnetcutcachekey"
MODNET_WEBCAM_CUT_CACHE_KEY = "modnet_webcam_cache"
MODNET_ZERO_CONCAT_CACHE_KEY = "modnet_zero_concat_cache"
MODNET_USER_CACHE_KEY = "modnet_user_cache"


def modnet_cache_key(model_path):
    """Pick a MODNet variant's compile-cache key from its filename.

    Markers are checked most-specific first: `modnet_webcam_cut_xint8` contains both
    "webcam" and "cut", and the plain `"cut" in path` test this replaces sent it to the
    cut model's cache -- two different graphs (522 vs 507 nodes) sharing one name-keyed
    compile, which is reused silently rather than recompiled.
    """
    stem = Path(model_path).stem
    for marker, key in (
        ("zero_concat", MODNET_ZERO_CONCAT_CACHE_KEY),
        ("webcam", MODNET_WEBCAM_CUT_CACHE_KEY),
        ("user", MODNET_USER_CACHE_KEY),
        ("cut", MODNET_CUT_CACHE_KEY),
    ):
        if marker in stem:
            return key
    return MODNET_CACHE_KEY

