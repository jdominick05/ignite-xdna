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
# yolo11n: C2PSA spatial attention block + decoupled DWConv detect head.
YOLO11_CACHE_KEY = "yolo11cachekey"
YOLO11_CUT_CACHE_KEY = "yolo11cutcachekey"
YOLO11_NO_C2PSA_CACHE_KEY = "yolo11noc2psacachekey"
# YOLO-World v2: open-vocabulary detector with text-guided cross-attention (MaxSigmoidAttnBlock)
YOLOW_CACHE_KEY = "yolowcachekey"
YOLOW_CUT_CACHE_KEY = "yolowcutcachekey"
YOLOW_NO_ATTN_CACHE_KEY = "yolownoattncachekey"
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
# MiDaS monocular depth estimation: stock bilinear resize (5 DPU subgraphs) vs
# NPU-optimized nearest resize (single monolithic DPU subgraph).
MIDAS_CACHE_KEY = "midascachekey"
MIDAS_NEAREST_CACHE_KEY = "midasnearestcache"
# SESR super-resolution: single monolithic DPU subgraph upscaling via DepthToSpace.
SESR_CACHE_KEY = "sesrcachekey"
SESR_ADAROUND_CACHE_KEY = "sesradaroundcachekey"
# RegNetX regular channel capacity classification.
REGNETX_CACHE_KEY = "regnetxcachekey"


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


# Real-ESRGAN 4x super-resolution: AMD 10-RRDB vs SRVGGNet-v3 Compact across resolutions.
REALESRGAN_RRDB_64_CACHE_KEY = "realesrgan_rrdb_64_cache"
REALESRGAN_RRDB_64_ADAROUND_CACHE_KEY = "realesrgan_rrdb_64_adaround_cache"
REALESRGAN_RRDB_128_CACHE_KEY = "realesrgan_rrdb_128_cache"
REALESRGAN_RRDB_256_CACHE_KEY = "realesrgan_rrdb_256_cache"
REALESRGAN_COMPACT_64_CACHE_KEY = "realesrgan_compact_64_cache"
REALESRGAN_COMPACT_128_CACHE_KEY = "realesrgan_compact_128_cache"
REALESRGAN_COMPACT_256_CACHE_KEY = "realesrgan_compact_256_cache"
REALESRGAN_CACHE_KEY = "realesrgancachekey"


def realesrgan_cache_key(model_path):
    """Pick a Real-ESRGAN variant's compile-cache key from its filename.

    Ensures models of different architectures (compact vs rrdb) and spatial
    resolutions (r64, r128, r256) never share a compile cache key.
    """
    stem = Path(model_path).stem.lower()
    is_compact = "compact" in stem or "srvgg" in stem or "v3" in stem
    is_rrdb = "rrdb" in stem or "amd" in stem

    res = "64"
    for r in ("256", "128", "64"):
        if f"r{r}" in stem or f"_{r}" in stem or f"x{r}" in stem:
            res = r
            break

    if is_compact:
        if res == "256":
            return REALESRGAN_COMPACT_256_CACHE_KEY
        if res == "128":
            return REALESRGAN_COMPACT_128_CACHE_KEY
        return REALESRGAN_COMPACT_64_CACHE_KEY
    if is_rrdb:
        if "adaround" in stem and res == "64":
            return REALESRGAN_RRDB_64_ADAROUND_CACHE_KEY
        if res == "256":
            return REALESRGAN_RRDB_256_CACHE_KEY
        if res == "128":
            return REALESRGAN_RRDB_128_CACHE_KEY
        return REALESRGAN_RRDB_64_CACHE_KEY
    return REALESRGAN_CACHE_KEY


def yolo11_cache_key(model_path):
    """Pick a YOLO11 variant's compile-cache key from its filename."""
    stem = Path(model_path).stem.lower()
    if "no_c2psa" in stem or "noc2psa" in stem:
        return YOLO11_NO_C2PSA_CACHE_KEY
    if "cut" in stem:
        return YOLO11_CUT_CACHE_KEY
    return YOLO11_CACHE_KEY


def yolow_cache_key(model_path):
    """Pick a YOLO-World variant's compile-cache key from its filename."""
    stem = Path(model_path).stem.lower()
    if "no_attn" in stem or "noattn" in stem:
        return YOLOW_NO_ATTN_CACHE_KEY
    if "cut" in stem:
        return YOLOW_CUT_CACHE_KEY
    return YOLOW_CACHE_KEY


