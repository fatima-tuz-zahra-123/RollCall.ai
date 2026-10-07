"""INFERENCE: recognizes the face in one image (or every image in a folder).

Input : a raw image (or folder of images). Use --aligned if the input is already a face crop from SG2.
Output: SG3 V1 JSON, one object per image, e.g.
  {"image": "test1.jpg", "method": "sface", "decision": "KNOWN", "student_id": "468915",
   "best_match": "468915", "score": 0.6123, "score_type": "cosine_similarity",
   "threshold": 0.363, "bbox": [210, 95, 180, 230], "time_ms": 41.2}

Usage:
    python recognize.py --method sface --input data/test/known/468915/1.jpg
    python recognize.py --method lbph  --input data/test/unknown
    python recognize.py --method sface --input some_folder --threshold 0.45 --save results/out.json
"""

import argparse
import json
import os

from sg3_core import METHODS, FaceDetector, list_images, load_config, make_recognizer, recognize_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="sface", choices=METHODS)
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--threshold", type=float, default=None, help="override config threshold")
    ap.add_argument("--aligned", action="store_true", help="input is already a cropped face")
    ap.add_argument("--save", default=None, help="optional path to write JSON results")
    args = ap.parse_args()

    cfg = load_config()
    rec = make_recognizer(args.method, cfg)
    rec.load()
    det = None if args.aligned else FaceDetector(cfg)

    paths = list_images(args.input) if os.path.isdir(args.input) else [args.input]
    results = [recognize_image(p, rec, det, args.threshold, args.aligned) for p in paths]

    text = json.dumps(results if len(results) > 1 else results[0], indent=2)
    print(text)
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, "w") as f:
            f.write(text)
        print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()
