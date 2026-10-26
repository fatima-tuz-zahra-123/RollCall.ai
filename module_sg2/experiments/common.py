"""Shared test bench for SG-2 experiments.

Test data : LFW (13,233 web photos, 250x250) + the official pairs.txt (10 folds x 300 genuine + 300 impostor pairs).
Mock SG-1 : YuNet (OpenCV Zoo) -> box + det_conf + 5 landmarks, i.e. exactly the V1 SG-1 -> SG-2 message.
Stand-in SG-3 : SFace (OpenCV Zoo, 128-d embedding, cosine similarity) so we can measure DOWNSTREAM impact.
Everything is cached in module_sg2/data/ (gitignored) so re-runs are fast.
"""
import json, sys, time
from pathlib import Path

import cv2
import numpy as np

SG2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SG2))
import sg2_quality  # noqa: E402

DATA, MODELS, RESULTS = SG2 / "data", SG2 / "models", SG2 / "results"
LFW = DATA / "lfw"
FMR_TARGET = 0.01   # operating point: 1% false-match rate


def load_pairs(folds=range(10)):
    """-> list of (path_a, path_b, is_genuine, fold)."""
    lines = (DATA / "pairs.txt").read_text().split("\n")
    n_folds, n = map(int, lines[0].split())
    p = lambda name, i: str(LFW / name / f"{name}_{int(i):04d}.jpg")
    out, k = [], 1
    for fold in range(n_folds):
        for genuine in (True, False):
            for _ in range(n):
                t = lines[k].split("\t")
                k += 1
                if fold in folds:
                    out.append((p(t[0], t[1]), p(t[0], t[2]), True, fold) if genuine
                               else (p(t[0], t[1]), p(t[2], t[3]), False, fold))
    return out


def yunet(size=(250, 250)):
    return cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2023mar.onnx"), "", size, 0.5, 0.3, 50)


def sface():
    return cv2.FaceRecognizerSF.create(str(MODELS / "face_recognition_sface_2021dec.onnx"), "")


def detect_central(det, img):
    """Mock SG-1: YuNet on the image, keep the face nearest the centre (LFW's labelled person). V1 face dict or None."""
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    c = np.array([w / 2, h / 2])
    f = min(faces, key=lambda r: np.linalg.norm(r[:2] + r[2:4] / 2 - c))
    return {"box": [float(v) for v in f[:4]], "det_conf": float(f[14]),
            "landmarks": f[4:14].reshape(5, 2).astype(float).tolist()}


def mock_sg1(paths):
    """Cached YuNet detections for every image = the mock SG-1 output we develop against."""
    cache = DATA / "mock_sg1_lfw.json"
    db = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [p for p in paths if str(Path(p).relative_to(LFW)) not in db]
    if todo:
        det = yunet()
        for p in todo:
            db[str(Path(p).relative_to(LFW))] = detect_central(det, cv2.imread(p))
        cache.write_text(json.dumps(db))
    return {p: db[str(Path(p).relative_to(LFW))] for p in paths}


def embed(rec, face112):
    f = rec.feature(face112).flatten()
    return f / (np.linalg.norm(f) + 1e-12)


# ---------------- verification metrics ----------------
def threshold_at_fmr(scores, labels, fmr=FMR_TARGET):
    imp = np.sort(scores[~labels])[::-1]
    return float(imp[int(np.floor(fmr * len(imp)))])  # accept if score > t


def fnmr_at(scores, labels, t):
    gen = scores[labels]
    return float((gen <= t).mean()) if len(gen) else float("nan")


def lfw_accuracy(scores, labels, folds):
    """Standard LFW protocol: best threshold on 9 folds, accuracy on the held-out fold; mean +- std over 10."""
    accs = []
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        cand = np.unique(scores[tr])
        best = max(cand, key=lambda t: ((scores[tr] > t) == labels[tr]).mean())
        accs.append(((scores[te] > best) == labels[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def eer(scores, labels):
    ts = np.unique(scores)
    fmr = np.array([(scores[~labels] > t).mean() for t in ts])
    fnmr = np.array([(scores[labels] <= t).mean() for t in ts])
    i = np.argmin(np.abs(fmr - fnmr))
    return float((fmr[i] + fnmr[i]) / 2)


def save(name, obj):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=2))


def timer():
    t0 = time.perf_counter()
    return lambda: time.perf_counter() - t0
