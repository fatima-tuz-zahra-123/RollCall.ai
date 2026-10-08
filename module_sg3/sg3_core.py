"""
SG3 - Student Face Recognition : shared code.

Two recognition methods are compared:
  * "lbph"  : Baseline. Classical OpenCV LBPH (Local Binary Pattern Histograms).
              Score = LBPH distance  (LOWER = more similar). KNOWN if distance <= threshold.
  * "sface" : Alternative. Deep-learning SFace embeddings (128-D, OpenCV Zoo ONNX).
              Score = cosine similarity (HIGHER = more similar). KNOWN if similarity >= threshold.

Both use the same YuNet face detector so the comparison is fair.
Only needs:  opencv-contrib-python, numpy  (runs on Windows / Linux / Jetson).
"""

import json
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
METHODS = ("lbph", "sface")


# ----------------------------------------------------------------- config / files
def load_config(path=None):
    path = path or os.path.join(HERE, "config.json")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k in ("data_dir", "models_dir", "gallery_dir", "results_dir"):
        if not os.path.isabs(cfg[k]):
            cfg[k] = os.path.join(HERE, cfg[k])
    return cfg


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)
    )


def list_people(folder):
    """Sub-folders of `folder`; each sub-folder name = student ID."""
    if not os.path.isdir(folder):
        return []
    return sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))


def read_image(path):
    # np.fromfile + imdecode works with Windows paths that contain spaces/unicode
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def model_path(cfg, name):
    p = os.path.join(cfg["models_dir"], name)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"Model file missing: {p}\nRun:  python download_models.py"
        )
    return p


# ----------------------------------------------------------------- face detection
class FaceDetector:
    """YuNet detector. Returns the largest face in a raw image."""

    def __init__(self, cfg):
        d = cfg["detector"]
        self.max_side = d["max_side"]
        self.det = cv2.FaceDetectorYN.create(
            model_path(cfg, d["model"]), "", (320, 320), d["score_threshold"], d["nms_threshold"], 5000
        )

    def detect(self, img):
        """Returns (img_used, face_row, scale). face_row is None if no face.
        face_row = [x, y, w, h, 10 landmark coords, score] in img_used coordinates;
        divide by `scale` to get original-image coordinates."""
        h, w = img.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        if faces is None or len(faces) == 0:
            return img, None, scale
        largest = max(faces, key=lambda f: f[2] * f[3])
        return img, largest, scale


def bbox_of(face_row):
    return [int(round(v)) for v in face_row[:4]]


# ----------------------------------------------------------------- SFace (alternative)
class SFaceRecognizer:
    method = "sface"
    score_type = "cosine_similarity"
    higher_is_better = True

    def __init__(self, cfg):
        self.cfg = cfg
        self.threshold = cfg["sface"]["threshold"]
        self.net = cv2.FaceRecognizerSF.create(model_path(cfg, cfg["sface"]["model"]), "")
        self.gallery = {}  # student_id -> (N,128) array of L2-normalised embeddings

    def embed(self, img, face_row=None):
        """face_row given -> align+crop using landmarks. None -> img is already a face crop."""
        if face_row is not None:
            crop = self.net.alignCrop(img, face_row)
        else:
            crop = cv2.resize(img, (112, 112))
        feat = self.net.feature(crop).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-12)

    # gallery -------------------------------------------------------------
    def fit(self, samples):
        """samples: list of (student_id, img, face_row)."""
        g = {}
        for sid, img, row in samples:
            g.setdefault(sid, []).append(self.embed(img, row))
        self.gallery = {k: np.stack(v) for k, v in g.items()}

    def save(self):
        os.makedirs(self.cfg["gallery_dir"], exist_ok=True)
        path = os.path.join(self.cfg["gallery_dir"], "sface_gallery.npz")
        np.savez(path, **self.gallery)
        return path

    def load(self):
        path = os.path.join(self.cfg["gallery_dir"], "sface_gallery.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not found. Run:  python enroll.py --method sface")
        data = np.load(path)
        self.gallery = {k: data[k] for k in data.files}

    # matching ------------------------------------------------------------
    def match(self, img, face_row=None):
        """Returns (best_student_id, best_score). Score = max cosine over that student's photos."""
        q = self.embed(img, face_row)
        best_id, best = None, -1.0
        for sid, embs in self.gallery.items():
            s = float(np.max(embs @ q))
            if s > best:
                best_id, best = sid, s
        return best_id, best

    def accept(self, score, threshold=None):
        t = self.threshold if threshold is None else threshold
        return score >= t


# ----------------------------------------------------------------- LBPH (baseline)
class LBPHRecognizer:
    method = "lbph"
    score_type = "lbph_distance"
    higher_is_better = False

    def __init__(self, cfg):
        self.cfg = cfg
        c = cfg["lbph"]
        self.size = c["face_size"]
        self.threshold = c["threshold"]
        if not hasattr(cv2, "face"):
            raise ImportError(
                "cv2.face missing. Install the contrib build:\n"
                "  pip uninstall -y opencv-python\n  pip install opencv-contrib-python"
            )
        self.model = cv2.face.LBPHFaceRecognizer_create(
            radius=c["radius"], neighbors=c["neighbors"], grid_x=c["grid_x"], grid_y=c["grid_y"]
        )
        self.labels = []  # index -> student_id

    def prep(self, img, face_row=None):
        if face_row is not None:
            x, y, w, h = bbox_of(face_row)
            H, W = img.shape[:2]
            x, y = max(0, x), max(0, y)
            img = img[y : min(H, y + h), x : min(W, x + w)]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray = cv2.resize(gray, (self.size, self.size))
        return cv2.equalizeHist(gray)

    def fit(self, samples):
        self.labels = sorted({sid for sid, _, _ in samples})
        idx = {sid: i for i, sid in enumerate(self.labels)}
        X = [self.prep(img, row) for _, img, row in samples]
        y = np.array([idx[sid] for sid, _, _ in samples], dtype=np.int32)
        self.model.train(X, y)

    def save(self):
        os.makedirs(self.cfg["gallery_dir"], exist_ok=True)
        path = os.path.join(self.cfg["gallery_dir"], "lbph_model.yml")
        self.model.write(path)
        with open(os.path.join(self.cfg["gallery_dir"], "lbph_labels.json"), "w") as f:
            json.dump(self.labels, f, indent=2)
        return path

    def load(self):
        path = os.path.join(self.cfg["gallery_dir"], "lbph_model.yml")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not found. Run:  python enroll.py --method lbph")
        self.model.read(path)
        with open(os.path.join(self.cfg["gallery_dir"], "lbph_labels.json")) as f:
            self.labels = json.load(f)

    def match(self, img, face_row=None):
        label, dist = self.model.predict(self.prep(img, face_row))
        return self.labels[label], float(dist)

    def accept(self, score, threshold=None):
        t = self.threshold if threshold is None else threshold
        return score <= t


def make_recognizer(method, cfg):
    if method == "sface":
        return SFaceRecognizer(cfg)
    if method == "lbph":
        return LBPHRecognizer(cfg)
    raise ValueError(f"Unknown method '{method}'. Use one of {METHODS}")


# ----------------------------------------------------------------- V1 output interface
def recognize_image(path, recognizer, detector, threshold=None, aligned=False):
    """Runs detection + recognition on one image and returns the SG3 V1 output dict."""
    t0 = time.perf_counter()
    img = read_image(path)
    out = {
        "image": os.path.basename(path),
        "method": recognizer.method,
        "decision": "NO_FACE",
        "student_id": None,
        "best_match": None,
        "score": None,
        "score_type": recognizer.score_type,
        "threshold": recognizer.threshold if threshold is None else threshold,
        "bbox": None,
    }
    if aligned:  # input is already a face crop (e.g. from SG2)
        row = None
        out["bbox"] = [0, 0, img.shape[1], img.shape[0]]
    else:
        img, row, scale = detector.detect(img)
        if row is None:
            out["time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return out
        out["bbox"] = [int(round(v / scale)) for v in row[:4]]  # original-image pixels
    sid, score = recognizer.match(img, row)
    known = recognizer.accept(score, threshold)
    out.update(
        decision="KNOWN" if known else "UNKNOWN",
        student_id=sid if known else "UNKNOWN",
        best_match=sid,
        score=round(score, 4),
    )
    out["time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out
