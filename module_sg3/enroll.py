"""ENROLLMENT: builds the gallery of known students.

Input : data/enroll/<student_id>/*.jpg     (2-5 raw photos per student, one face per photo)
Output: gallery/sface_gallery.npz          (for --method sface)
        gallery/lbph_model.yml + lbph_labels.json   (for --method lbph)

Usage:
    python enroll.py --method all       # build both galleries
    python enroll.py --method sface
    python enroll.py --method lbph
    python enroll.py --method all --aligned   # photos are already face crops (from SG2)
"""

import argparse
import os

from sg3_core import METHODS, FaceDetector, list_images, list_people, load_config, make_recognizer, read_image


def collect_samples(enroll_dir, detector, aligned):
    samples, skipped = [], []
    for sid in list_people(enroll_dir):
        imgs = list_images(os.path.join(enroll_dir, sid))
        n_ok = 0
        for p in imgs:
            img = read_image(p)
            if aligned:
                row = None
            else:
                img, row, _ = detector.detect(img)
                if row is None:
                    skipped.append(p)
                    continue
            samples.append((sid, img, row))
            n_ok += 1
        print(f"  {sid}: {n_ok}/{len(imgs)} photos usable")
    return samples, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="all", choices=list(METHODS) + ["all"])
    ap.add_argument("--aligned", action="store_true", help="input photos are already cropped faces")
    args = ap.parse_args()

    cfg = load_config()
    enroll_dir = os.path.join(cfg["data_dir"], "enroll")
    if not list_people(enroll_dir):
        raise SystemExit(f"No student folders found in {enroll_dir}\n"
                         f"Create data/enroll/<student_id>/ and put photos inside.")

    detector = None if args.aligned else FaceDetector(cfg)
    print(f"Reading enrollment photos from {enroll_dir}")
    samples, skipped = collect_samples(enroll_dir, detector, args.aligned)
    for p in skipped:
        print(f"  [skip] no face detected: {p}")
    if not samples:
        raise SystemExit("No usable faces found.")

    methods = METHODS if args.method == "all" else [args.method]
    for m in methods:
        rec = make_recognizer(m, cfg)
        rec.fit(samples)
        path = rec.save()
        print(f"[{m}] enrolled {len({s for s, _, _ in samples})} students "
              f"({len(samples)} photos) -> {path}")


if __name__ == "__main__":
    main()
