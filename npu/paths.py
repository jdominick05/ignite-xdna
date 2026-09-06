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
