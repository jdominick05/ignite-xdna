"""Profile segmentation postprocessing on a saved, pinned output; no NPU context."""
import argparse
from pathlib import Path
import sys
import time
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from npu.bisenetv2 import postprocess_mask


def scan(x):
    best = x[0].copy()
    labels = np.zeros(best.shape,np.uint8)
    for c in range(1,x.shape[0]):
        take = x[c] > best
        np.copyto(best,x[c],where=take)
        np.copyto(labels,c,where=take)
    return labels


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('--frames',type=int,default=50)
    args = ap.parse_args()
    x = np.load(args.output).squeeze(0)
    ref = np.argmax(x,axis=0).astype(np.uint8)
    for name,fn in [('canonical',lambda:postprocess_mask(x,(480,640))),
                    ('argmax',lambda:np.argmax(x,axis=0).astype(np.uint8)),
                    ('scan',lambda:scan(x)),
                    ('opencv',lambda:cv2.reduceArgMax(x.reshape(x.shape[0],-1),0).reshape(x.shape[1:]).astype(np.uint8))]:
        actual = fn()
        if name != 'canonical':
            np.testing.assert_array_equal(actual,ref)
        times = []
        for _ in range(args.frames):
            t = time.perf_counter()
            fn()
            times.append((time.perf_counter()-t)*1e3)
        print(json.dumps(dict(name=name,shape=x.shape,strides=x.strides,mean_ms=float(np.mean(times)),
                              p95_ms=float(np.percentile(times,95)))),flush=True)


if __name__ == '__main__':
    main()
