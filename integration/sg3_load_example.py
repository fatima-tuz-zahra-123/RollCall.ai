"""SG-3 side: read SG-2's handoff and run SFace on it (compatibility check + starting point for SG-3).

    python integration/sg3_load_example.py runs/sg1_sg2/sg3 --sface path/to/face_recognition_sface_2021dec.onnx

Live pipeline (no files): call SG-2 in-process and use the same items directly
    out   = sg2_quality.process_frame(frame, sg1_faces, cfg, frame_id=n)
    items = sg2_quality.sg3_items(out, sg1_faces)        # accepted faces only
    for it in items:
        sid, score = recognizer.match(it["face"], None)   # SG-3 SFaceRecognizer: face is already aligned
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def load_handoff(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    faces = d["faces"]
    assert faces.dtype == np.uint8 and faces.ndim == 4 and faces.shape[1:] == (112, 112, 3), faces.shape
    return [{"frame_id": int(d["frame_id"]), "face_id": str(fid), "face_index": int(i),
             "face": f, "quality": float(q), "box": b.tolist()}
            for fid, i, f, q, b in zip(d["face_ids"], d["face_index"], faces, d["quality"], d["boxes"])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="the sg3/ folder written by sg1_sg2_pipeline.py")
    ap.add_argument("--sface", help="SFace ONNX model; omit to only check the format")
    a = ap.parse_args()
    net = cv2.FaceRecognizerSF.create(a.sface, "") if a.sface else None
    for p in sorted(Path(a.folder).glob("*.npz")):
        items = load_handoff(p)
        print(f"{p.name}: frame {items[0]['frame_id'] if items else '-'}, {len(items)} accepted faces")
        for it in items[:5]:
            line = f"  {it['face_id']:>8}  q={it['quality']:.2f}  box={[round(v) for v in it['box']]}"
            if net is not None:
                emb = net.feature(it["face"]).flatten()  # SFace takes the 112x112 BGR face as is
                line += f"  embedding {emb.shape[0]}-d, norm {np.linalg.norm(emb):.1f}"
            print(line)
