"""SG-2 Face Quality, Alignment & Preprocessing (V1 interface: see interfaces/sg2_v1_interface.md).

    from sg2_quality import load_config, process_frame
    out = process_frame(frame_bgr, faces, load_config("config/baseline_v1.json"), frame_id=7)

Input  faces: [{"box": [x, y, w, h], "det_conf": 0.93, "landmarks": [[x, y] * 5] or None}, ...]
Output: {"frame_id": 7, "results": [{"face_index", "accept", "reason", "quality", "metrics", "face"}, ...]}
Never raises on bad input: a bad face becomes accept=False with a reason code.
"""
import json
from pathlib import Path

import cv2
import numpy as np

# ArcFace / InsightFace 5-point template for a 112x112 face (same one OpenCV Zoo + insightface use).
# Order = image-left eye, image-right eye, nose tip, image-left mouth corner, image-right mouth corner.
TEMPLATE_112 = np.float32([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                           [41.5493, 92.3655], [70.7299, 92.2041]])
REASONS = ("bad_input", "low_det_conf", "too_small", "no_landmarks", "pose", "align_failed", "blurry", "illumination")


def load_config(path):
    return json.loads(Path(path).read_text())


def _crop_resize(frame, box, margin, size):
    """No-alignment path: square crop around the box (+margin), resized. Out-of-frame pixels are edge-replicated."""
    x, y, w, h = box
    side = max(w, h) * (1 + 2 * margin)
    cx, cy = x + w / 2, y + h / 2
    s = size / side  # scale + translate only (no rotation) -> same warpAffine path as alignment
    M = np.float32([[s, 0, size / 2 - s * cx], [0, s, size / 2 - s * cy]])
    return cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def align(frame, landmarks, size=112):
    """Similarity transform (rotation + uniform scale + shift) of 5 landmarks onto the template."""
    M, _ = cv2.estimateAffinePartial2D(np.float32(landmarks), TEMPLATE_112 * (size / 112), method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def yaw_proxy(lmk):
    """Nose offset from the eye midpoint, measured ALONG the eye axis, / eye distance.
    ~0 frontal, grows as the head turns; sign = direction. Along-axis makes it roll-invariant
    (a tilted but frontal face must not look 'turned')."""
    lmk = np.float32(lmk)
    eye_vec = lmk[1] - lmk[0]
    eye_d = float(np.linalg.norm(eye_vec)) + 1e-6
    return float(np.dot(lmk[2] - (lmk[0] + lmk[1]) / 2, eye_vec / eye_d) / eye_d)


def image_metrics(face112):
    g = cv2.cvtColor(face112, cv2.COLOR_BGR2GRAY)
    return {"blur": float(cv2.Laplacian(g, cv2.CV_64F).var()),     # sharpness: variance of 2nd derivative
            "brightness": float(g.mean()),
            "sat_frac": float(((g < 10) | (g > 245)).mean())}       # share of crushed/blown pixels


def heuristic_quality(m, cfg):
    """Baseline soft score in [0,1]: mean of normalised sharpness, size, pose, exposure terms. Heuristic, not learned."""
    q = cfg["quality_ref"]
    terms = [min(1.0, m["blur"] / q["blur"]),
             min(1.0, m["face_px"] / q["face_px"]),
             max(0.0, 1 - abs(m["yaw"]) / q["yaw"]) if m["yaw"] is not None else 0.5,
             max(0.0, 1 - abs(m["brightness"] - 128) / 128) * (1 - m["sat_frac"])]
    return round(float(np.mean(terms)), 4)


def assess_face(frame, face, cfg):
    g = cfg["gates"]
    r = {"accept": False, "reason": None, "quality": 0.0, "metrics": {}, "face": None}
    try:
        box = [float(v) for v in face["box"]]
        assert len(box) == 4 and box[2] > 0 and box[3] > 0 and frame is not None and frame.ndim == 3
    except Exception:
        r["reason"] = "bad_input"
        return r
    lmk = face.get("landmarks")
    if lmk is not None and np.shape(lmk) != (5, 2):
        lmk = None
    m = r["metrics"]
    m["face_px"] = float(min(box[2], box[3]))
    m["det_conf"] = float(face.get("det_conf", 1.0))
    m["yaw"] = yaw_proxy(lmk) if lmk is not None else None

    def reject(reason):
        r["reason"] = reason
        return r

    if g.get("min_det_conf") is not None and m["det_conf"] < g["min_det_conf"]:
        return reject("low_det_conf")
    if g.get("min_face_px") is not None and m["face_px"] < g["min_face_px"]:
        return reject("too_small")
    if cfg["align"] == "similarity5" and lmk is None:
        return reject("no_landmarks")
    if g.get("max_abs_yaw") is not None and m["yaw"] is not None and abs(m["yaw"]) > g["max_abs_yaw"]:
        return reject("pose")
    size = cfg["output_size"]
    out = align(frame, lmk, size) if cfg["align"] == "similarity5" else _crop_resize(frame, box, cfg["crop_margin"], size)
    if out is None:
        return reject("align_failed")
    m.update(image_metrics(out))
    if g.get("min_blur") is not None and m["blur"] < g["min_blur"]:
        return reject("blurry")
    lo, hi = g.get("brightness_range") or (None, None)
    if lo is not None and not lo <= m["brightness"] <= hi:
        return reject("illumination")
    if g.get("max_sat_frac") is not None and m["sat_frac"] > g["max_sat_frac"]:
        return reject("illumination")
    r.update(accept=True, face=out, quality=heuristic_quality(m, cfg))
    return r


def process_frame(frame, faces, cfg, frame_id=None):
    """One result per input face, same order. Empty faces -> empty results."""
    results = []
    for i, f in enumerate(faces or []):
        res = assess_face(frame, f, cfg)
        res["face_index"] = i
        results.append(res)
    return {"frame_id": frame_id, "results": results}


def to_json(out):
    """Same dict without the image arrays (for logging / handing metadata to SG-4/SG-5)."""
    return {"frame_id": out["frame_id"],
            "results": [{k: v for k, v in r.items() if k != "face"} for r in out["results"]]}


if __name__ == "__main__":  # self-check on synthetic data: python module_sg2/sg2_quality.py
    rng = np.random.default_rng(0)
    frame = cv2.GaussianBlur(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), (0, 0), 1.2)
    th = np.deg2rad(20)
    R = np.float32([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    lmk = ((TEMPLATE_112 - 56) @ R.T * 1.6 + [320, 240]).tolist()
    good = {"box": [230, 150, 180, 180], "det_conf": 0.9, "landmarks": lmk}
    cfg = load_config(Path(__file__).parent / "config" / "baseline_v1.json")
    r = assess_face(frame, good, cfg)
    assert r["accept"] and r["face"].shape == (112, 112, 3), r
    # alignment check: warping the landmarks with the fitted transform lands them on the template
    M, _ = cv2.estimateAffinePartial2D(np.float32(lmk), TEMPLATE_112, method=cv2.LMEDS)
    assert np.abs(np.float32(lmk) @ M[:, :2].T + M[:, 2] - TEMPLATE_112).max() < 1e-3
    assert assess_face(cv2.GaussianBlur(frame, (0, 0), 5), good, cfg)["reason"] == "blurry"
    assert assess_face(frame, {**good, "box": [0, 0, 20, 20]}, cfg)["reason"] == "too_small"
    assert assess_face(frame, {**good, "landmarks": None}, cfg)["reason"] == "no_landmarks"
    assert assess_face(frame, {**good, "det_conf": 0.1}, cfg)["reason"] == "low_det_conf"
    turned = [p[:] for p in lmk]
    turned[2] = [lmk[2][0] + 25, lmk[2][1]]
    assert assess_face(frame, {**good, "landmarks": turned}, cfg)["reason"] == "pose"
    assert assess_face((frame * 0.1).astype(np.uint8), good, cfg)["reason"] in ("illumination", "blurry")
    assert assess_face(None, good, cfg)["reason"] == "bad_input"
    assert assess_face(frame, {"box": "junk"}, cfg)["reason"] == "bad_input"
    assert process_frame(frame, [], cfg, 1) == {"frame_id": 1, "results": []}
    crop_cfg = {**cfg, "align": "crop"}
    assert assess_face(frame, {**good, "landmarks": None}, crop_cfg)["face"].shape == (112, 112, 3)
    print("sg2_quality self-check: 12/12 passed")
