"""Time SG-2 on whatever machine you're on (meant for the Jetson Orin Nano, works on a laptop too).

Needs only OpenCV + numpy + the YuNet onnx (python module_sg2/scripts/download_assets.py --models).
No torch, no SG-1 needed: it uses our committed sample frame + its V1 face record.

    python module_sg2/scripts/bench_sg2.py              # 300 runs each
    python module_sg2/scripts/bench_sg2.py --runs 1000

Prints ms per face for: landmarks given by SG-1 / landmarks missing (YuNet ROI fallback),
plus the platform info, so we can paste the numbers straight into the README.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SG2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SG2))
import sg2_quality  # noqa: E402


def time_it(fn, runs):
    fn()                                   # warm-up (model load, first allocation)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(ts)), 3), round(float(np.percentile(ts, 95)), 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=300)
    a = ap.parse_args()

    msg = json.loads((SG2.parent / "interfaces" / "examples" / "sg2_example_input.json").read_text())
    frame = cv2.imread(str(SG2.parent / msg["image"]))
    face = msg["faces"][0]
    cfg = sg2_quality.load_config(SG2 / "config" / "sg1_yolov12_v1.json")

    with_lmk = time_it(lambda: sg2_quality.assess_face(frame, face, cfg), a.runs)
    fallback = time_it(lambda: sg2_quality.assess_face(frame, {**face, "landmarks": None}, cfg), a.runs)
    r = sg2_quality.assess_face(frame, {**face, "landmarks": None}, cfg)
    assert r["accept"] and r["face"].shape == (112, 112, 3), r["reason"]   # fallback actually worked

    try:
        cuda = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception:
        cuda = 0
    print(json.dumps({
        "machine": platform.machine(), "system": platform.platform(), "python": platform.python_version(),
        "opencv": cv2.__version__, "opencv_cuda_devices": cuda, "cpu_threads": cv2.getNumThreads(),
        "ms_per_face_landmarks_given (median, p95)": with_lmk,
        "ms_per_face_yunet_fallback (median, p95)": fallback,
        "runs": a.runs}, indent=2))
