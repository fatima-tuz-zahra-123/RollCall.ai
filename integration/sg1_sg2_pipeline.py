"""SG-1 -> SG-2 pipeline (first multi-module integration): YOLOv12n-face boxes -> SG-2 quality gate + alignment.

    # image, folder of images, video file, or camera index
    python integration/sg1_sg2_pipeline.py --source path/to/classroom.jpg
    python integration/sg1_sg2_pipeline.py --source path/to/clip.mp4 --every 5
    python integration/sg1_sg2_pipeline.py --source 0 --show                # live webcam
    # SG-1 unavailable: replay saved/mock SG-1 output (V1 JSON, see interfaces/examples/)
    python integration/sg1_sg2_pipeline.py --mock-sg1 interfaces/examples/sg2_example_input.json

Outputs in --out (default runs/sg1_sg2/):
    annotated/   frame with green = accepted (quality), red = rejected (reason), dots = landmarks used
    faces/       accepted 112x112 aligned faces as PNG (for looking at)
    sg3/         SG-3 handoff per frame (.npz): faces (N,112,112,3) uint8 BGR, face_ids, face_index, quality,
                 boxes, frame_id - see module_sg2/sg2_quality.py save_sg3_npz and integration/sg3_load_example.py
    sheets/      contact sheet per frame: every detected face, accepted first, with its decision
    results.jsonl  per frame: SG-1 faces (V1) + SG-2 results (no image arrays) + timings
    summary.json   totals, reject-reason histogram, landmark source, ms per frame / per face
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "module_sg2"))
sys.path.insert(0, str(REPO / "integration"))
import sg2_quality  # noqa: E402
from sg1_adapter import DEFAULT_WEIGHTS  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GREEN, RED, YELLOW, CYAN = (60, 200, 60), (40, 40, 230), (0, 220, 255), (255, 200, 0)


def frames_from(source, every=1, max_frames=None):
    """Yield (frame_id, name, timestamp_s, frame_bgr) for an image, a folder, a video file or a camera index."""
    src = Path(source)
    if src.is_dir() or src.suffix.lower() in IMG_EXT:
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
        for i, p in enumerate(files[:max_frames]):
            img = cv2.imread(str(p))
            if img is None:
                print(f"cannot read {p}, skipped")
                continue
            yield i, p.stem, None, img
        return
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video/camera {source}")
    i = n = 0
    while max_frames is None or n < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        if i % every == 0:
            yield i, f"frame_{i:06d}", cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, img
            n += 1
        i += 1
    cap.release()


def mock_frames(json_path):
    """Replay V1 SG-1 messages: a single dict or a list of {frame_id, timestamp, image, faces}."""
    msgs = json.loads(Path(json_path).read_text())
    for msg in msgs if isinstance(msgs, list) else [msgs]:
        img = cv2.imread(str(REPO / msg["image"]))
        if img is None:
            raise SystemExit(f"mock image not found: {msg['image']}")
        yield msg.get("frame_id", 0), Path(msg["image"]).stem, msg.get("timestamp"), img, msg["faces"]


def annotate(frame, faces, results):
    out = frame.copy()
    t = max(1, round(max(frame.shape[:2]) / 900))
    for f, r in zip(faces, results):
        x, y, w, h = (int(round(v)) for v in f["box"])
        col = GREEN if r["accept"] else RED
        cv2.rectangle(out, (x, y), (x + w, y + h), col, t)
        label = f"{r['face_index']}:{r['quality']:.2f}" if r["accept"] else f"{r['face_index']}:{r['reason']}"
        cv2.putText(out, label, (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * t, col, t)
        lmk = r.get("landmarks")
        if lmk is not None:
            dot = YELLOW if r["metrics"].get("lmk_source") == "sg1" else CYAN
            for px, py in lmk:
                cv2.circle(out, (int(px), int(py)), t, dot, -1)
    return out


def contact_sheet(frame, faces, results, cols=10):
    tiles = []
    for f, r in sorted(zip(faces, results), key=lambda fr: (not fr[1]["accept"], -fr[1]["quality"])):
        img = r["face"] if r["accept"] else sg2_quality._crop_resize(frame, f["box"], 0.15, 112)
        col = GREEN if r["accept"] else RED
        tile = cv2.copyMakeBorder(img, 3, 21, 3, 3, cv2.BORDER_CONSTANT, value=col)
        txt = f"#{r['face_index']} q={r['quality']:.2f}" if r["accept"] else f"#{r['face_index']} {r['reason']}"
        cv2.putText(tile, txt, (5, 131), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return None
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % cols)
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", help="image, folder, video file or camera index")
    ap.add_argument("--mock-sg1", help="V1 JSON with saved SG-1 output (replaces the live detector)")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--sg1-conf", type=float, default=0.25, help="SG-1 detection threshold (run.py default)")
    ap.add_argument("--imgsz", type=int, default=1280, help="SG-1 inference size (run.py predict default)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--config", default=str(REPO / "module_sg2" / "config" / "cctv_v2.json"))  # tuned on the class video (EXP-6)
    ap.add_argument("--out", default=str(REPO / "runs" / "sg1_sg2"))
    ap.add_argument("--every", type=int, default=1, help="video: process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--show", action="store_true", help="display annotated frames live (q to quit)")
    a = ap.parse_args()
    if not a.source and not a.mock_sg1:
        ap.error("give --source or --mock-sg1")

    cfg = sg2_quality.load_config(a.config)
    out = Path(a.out)
    for d in ("annotated", "faces", "sheets", "sg3"):
        (out / d).mkdir(parents=True, exist_ok=True)

    if a.mock_sg1:
        stream = mock_frames(a.mock_sg1)
        detector = None
    else:
        from sg1_adapter import SG1Detector
        detector = SG1Detector(a.weights, a.sg1_conf, a.imgsz, a.device)
        stream = ((fid, name, ts, img, None) for fid, name, ts, img in frames_from(a.source, a.every, a.max_frames))

    reasons, sources = Counter(), Counter()
    sg1_ms, sg2_ms_face, n_faces, n_frames = [], [], 0, 0
    with open(out / "results.jsonl", "w") as log:
        for fid, name, ts, frame, faces in stream:
            t0 = time.perf_counter()
            if detector is not None:
                faces = detector(frame)
                sg1_ms.append((time.perf_counter() - t0) * 1000)
            t1 = time.perf_counter()
            res = sg2_quality.process_frame(frame, faces, cfg, frame_id=fid)
            dt2 = (time.perf_counter() - t1) * 1000
            if faces:
                sg2_ms_face.append(dt2 / len(faces))
            n_frames += 1
            n_faces += len(faces)
            for r in res["results"]:
                reasons["accepted" if r["accept"] else r["reason"]] += 1
                sources[r["metrics"].get("lmk_source") or "none"] += 1
                if r["accept"]:
                    cv2.imwrite(str(out / "faces" / f"{name}_f{r['face_index']:03d}.png"), r["face"])
            sg2_quality.save_sg3_npz(out / "sg3" / f"{name}.npz", res, faces)
            ann = annotate(frame, faces, res["results"])
            cv2.imwrite(str(out / "annotated" / f"{name}.jpg"), ann)
            sheet = contact_sheet(frame, faces, res["results"])
            if sheet is not None:
                cv2.imwrite(str(out / "sheets" / f"{name}.jpg"), sheet)
            acc = sum(r["accept"] for r in res["results"])
            log.write(json.dumps({"frame_id": fid, "name": name, "timestamp": ts, "sg1_faces": faces,
                                  "sg2": sg2_quality.to_json(res),
                                  "sg1_ms": round(sg1_ms[-1], 1) if detector else None,
                                  "sg2_ms": round(dt2, 2)}) + "\n")
            print(f"{name}: SG-1 {len(faces)} faces -> SG-2 accepted {acc}, rejected {len(faces) - acc}"
                  + (f" | SG-1 {sg1_ms[-1]:.0f} ms" if detector else "") + f" | SG-2 {dt2:.1f} ms")
            if a.show:
                cv2.imshow("SG-1 -> SG-2", ann)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    summary = {"source": a.source or a.mock_sg1, "config": Path(a.config).name, "frames": n_frames,
               "faces_from_sg1": n_faces, "decisions": dict(reasons.most_common()),
               "accept_rate": round(reasons["accepted"] / n_faces, 3) if n_faces else None,
               "landmark_source": dict(sources),
               "sg1_ms_per_frame": round(float(np.mean(sg1_ms[1:] or sg1_ms)), 1) if sg1_ms else None,
               "sg2_ms_per_face": round(float(np.mean(sg2_ms_face)), 2) if sg2_ms_face else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if a.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
