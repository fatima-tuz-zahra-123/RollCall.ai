"""EXP-7  enrollment photos (Class Dataset/<person>/*.HEIC) through real SG-1 -> SG-2.

SG-3 builds its gallery from these photos, and gallery faces have to be aligned exactly like the
video faces, otherwise the embeddings don't line up. So we push every enrollment shot through the
same SG-1 -> SG-2 path and check which views (front / left / right / top) actually survive.

    python module_sg2/experiments/exp7_enrollment_gallery.py --dataset "path/to/Class Dataset"

Aligned gallery faces are written to module_sg2/data/gallery_aligned/ (gitignored, they're
classmates' faces). Only per-view counts go to results/.
"""
import argparse
import sys
from collections import defaultdict

from common import *
from exp3_edc import EDifFIQA

sys.path.insert(0, str(SG2.parent / "integration"))


def read_heic(path):
    import pillow_heif
    from PIL import Image, ImageOps
    pillow_heif.register_heif_opener()
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")   # iPhone stores rotation in EXIF
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def view_of(name):
    name = name.lower()
    for v in ("front", "left", "right", "top"):
        if name.startswith(v):
            return v
    return "unnamed"   # IMG_xxxx files, mostly front-ish


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="cctv_v2.json")
    a = ap.parse_args()
    from sg1_adapter import SG1Detector
    det, fiqa = SG1Detector(), EDifFIQA()
    cfg = sg2_quality.load_config(SG2 / "config" / a.config)
    out_dir = DATA / "gallery_aligned"
    people = sorted(p for p in Path(a.dataset).iterdir() if p.is_dir())
    per_view = defaultdict(lambda: {"shots": 0, "accepted": 0, "reasons": defaultdict(int)})
    per_person = []
    for k, person in enumerate(people):
        ok = 0
        for f in sorted(person.glob("*.HEIC")):
            img = read_heic(f)
            faces = det(img)
            v = view_of(f.stem)
            per_view[v]["shots"] += 1
            if not faces:
                per_view[v]["reasons"]["sg1_no_face"] += 1
                continue
            face = max(faces, key=lambda x: x["box"][2] * x["box"][3])   # the person being enrolled
            r = sg2_quality.assess_face(img, face, cfg)
            if r["accept"]:
                ok += 1
                per_view[v]["accepted"] += 1
                (out_dir / person.name).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / person.name / f"{f.stem}.png"), r["face"])
            else:
                per_view[v]["reasons"][r["reason"]] += 1
            print(f"{person.name}/{f.name}: {v:7s} {'OK' if r['accept'] else r['reason']:12s} "
                  f"yaw={r['metrics'].get('yaw') or 0:+.2f} px={r['metrics']['face_px']:.0f} "
                  f"pitch_dev={abs((r['metrics'].get('pitch') or 0.495) - 0.495):.2f} "
                  f"fiqa={fiqa(r['face']) if r['face'] is not None else float('nan'):.2f}")
        per_person.append({"person": f"person_{k + 1}", "shots": len(list(person.glob('*.HEIC'))), "accepted": ok})
    res = {"data": f"Class Dataset enrollment HEIC photos, real SG-1 (YOLOv12n-face) + SG-2 {a.config}",
           "per_view": {v: {**d, "reasons": dict(d["reasons"])} for v, d in per_view.items()},
           "people": len(people), "people_with_no_usable_shot": sum(p["accepted"] == 0 for p in per_person),
           "per_person": per_person}
    print(json.dumps(res, indent=1))
    save(f"exp7_enrollment_gallery_{Path(a.config).stem}", res)
