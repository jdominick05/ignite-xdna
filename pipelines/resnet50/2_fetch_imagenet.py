"""
Step 1.5 (v4): Pull calibration + eval slices from ImageNet-1k val (parquet).

The HF repo stores validation as 14 parquet shards (~480MB each, ~3570
images per shard). This downloads only the shards you ask for and streams
through them in small batches with pyarrow directly - no `datasets`, no
Arrow tables held in memory. Image bytes are written straight to disk
without decoding.

Writes two disjoint sets:
    <out>/calib/   images + labels.json
    <out>/eval/    images + labels.json

Prereqs:
    pip install huggingface_hub pyarrow
    hf auth login        (the account approved for imagenet-1k)

Usage:
    python pipelines/resnet50/2_fetch_imagenet.py --n-calib 300 --n-eval 1000
    python pipelines/resnet50/2_fetch_imagenet.py --shards 0 4 9 13 ...  (more classes)
"""

import argparse
import json
import os

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on sys.path

from npu.paths import DATA

REPO = "ILSVRC/imagenet-1k"
N_SHARDS = 14
BATCH = 64  # rows per batch pulled from parquet - bounds memory


def shard_name(k):
    return f"data/validation-{k:05d}-of-{N_SHARDS:05d}.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 7],
                    help="which validation shards to pull (0-13)")
    ap.add_argument("--local", nargs="+", default=None,
                    help="paths to already-downloaded parquet files (skips download)")
    args = ap.parse_args()

    calib_dir = os.path.join(args.out, "calib")
    eval_dir = os.path.join(args.out, "eval")
    os.makedirs(calib_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # --- Get shards ------------------------------------------------------
    if args.local:
        paths = args.local
    else:
        paths = []
        for k in args.shards:
            print(f"downloading {shard_name(k)} (~480MB, cached after first run)...")
            paths.append(hf_hub_download(repo_id=REPO, repo_type="dataset",
                                         filename=shard_name(k)))

    # --- Cheap pre-pass: labels only, to report class coverage -----------
    total_rows = 0
    all_labels = set()
    for p in paths:
        t = pq.read_table(p, columns=["label"])
        labs = t.column("label").to_pylist()
        total_rows += len(labs)
        all_labels.update(labs)
        print(f"  {os.path.basename(p)}: {len(labs)} rows, {len(set(labs))} classes")
    print(f"total: {total_rows} rows, {len(all_labels)} distinct classes across chosen shards")
    if len(all_labels) < 200:
        print("  (low coverage - consider more shards, e.g. --shards 0 4 9 13)")

    # --- Stride plan -----------------------------------------------------
    # Even global row index -> calib pool, odd -> eval pool. Disjoint by
    # construction. Stride within each pool to spread selections out.
    half = total_rows // 2
    stride_c = max(1, half // args.n_calib)
    stride_e = max(1, half // args.n_eval)

    calib_labels, eval_labels = {}, {}
    kc = ke = 0
    gi = 0  # global row index across all shards

    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=BATCH, columns=["image", "label"]):
            imgs = batch.column("image").to_pylist()   # list of {'bytes','path'}
            labs = batch.column("label").to_pylist()
            for img, lab in zip(imgs, labs):
                i = gi
                gi += 1
                if i % 2 == 0:
                    if kc >= args.n_calib or (i // 2) % stride_c != 0:
                        continue
                    d, labels, fname = calib_dir, calib_labels, f"{kc:06d}.jpg"
                    kc += 1
                else:
                    if ke >= args.n_eval or (i // 2) % stride_e != 0:
                        continue
                    d, labels, fname = eval_dir, eval_labels, f"{ke:06d}.jpg"
                    ke += 1
                with open(os.path.join(d, fname), "wb") as f:
                    f.write(img["bytes"])
                labels[fname] = int(lab)
                if (kc + ke) % 100 == 0:
                    print(f"  calib {kc}/{args.n_calib}   eval {ke}/{args.n_eval}")
            if kc >= args.n_calib and ke >= args.n_eval:
                break
        if kc >= args.n_calib and ke >= args.n_eval:
            break

    with open(os.path.join(calib_dir, "labels.json"), "w") as f:
        json.dump(calib_labels, f, indent=2)
    with open(os.path.join(eval_dir, "labels.json"), "w") as f:
        json.dump(eval_labels, f, indent=2)

    print(f"\ncalib: {kc} images, {len(set(calib_labels.values()))} classes -> {calib_dir}")
    print(f"eval:  {ke} images, {len(set(eval_labels.values()))} classes -> {eval_dir}")


if __name__ == "__main__":
    main()
