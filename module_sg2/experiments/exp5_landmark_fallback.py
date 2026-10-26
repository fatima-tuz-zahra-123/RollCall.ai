"""EXP-5  landmark fallback (fix for the SG1 -> SG2 blocker)

SG1's YOLOv12n-face only outputs boxes, so V1 rejected every face with no_landmarks.
With landmark_fallback on, SG2 runs YuNet on the box crop to get the 5 points itself.

To not cheat, the boxes are jittered (random +-8% shift, 0.9-1.15x size) so they look like
some other detector's boxes, not YuNet's own. Then we compare:
  A  landmarks given by SG1 (what we had in week 5)
  B  boxes only, fallback on
on all 6000 LFW pairs, plus small faces (32 / 24 px) on folds 1-5.

    python module_sg2/experiments/exp5_landmark_fallback.py
"""
from common import *
from exp2_thresholds import degrade

rng = np.random.default_rng(5)


def jitter(box):
    x, y, w, h = box
    k = rng.uniform(0.9, 1.15)
    dx, dy = rng.uniform(-0.08, 0.08, 2) * w
    nw, nh = w * k, h * k
    return [x + w / 2 - nw / 2 + dx, y + h / 2 - nh / 2 + dy, nw, nh]


def run(pairs, faces, imgs, rec, cfg):
    """faces: path -> face dict. Returns metrics + per-image landmark error."""
    emb, found, err, t_total = {}, 0, [], 0.0
    paths = sorted({p for a, b, _, _ in pairs for p in (a, b)})
    for p in paths:
        t = timer()
        r = sg2_quality.assess_face(imgs[p], faces[p], cfg)
        t_total += t()
        if r["face"] is None:
            continue
        found += 1
        emb[p] = embed(rec, r["face"])
    ok = [(a, b, g) for a, b, g, _ in pairs if a in emb and b in emb]
    if not ok:
        return {"faces_ok_pct": 0.0, "pairs_scored": 0}
    s =np.array([float(emb[a] @ emb[b]) for a, b, _ in ok])
    lab = np.array([g for _, _, g in ok])
    return {"faces_ok_pct": round(100 * found / len(paths), 2),
            "pairs_scored": len(ok),
            "fnmr_at_fmr1": round(fnmr_at(s, lab, threshold_at_fmr(s, lab)), 4),
            "eer": round(eer(s, lab), 4),
            "ms_per_face": round(1000 * t_total / len(paths), 2)}


if __name__ == "__main__":
    pairs = load_pairs()
    paths = sorted({p for a, b, _, _ in pairs for p in (a, b)})
    sg1, rec = mock_sg1(paths), sface()
    imgs = {p: cv2.imread(p) for p in paths}
    cfg = sg2_quality.load_config(SG2 / "config" / "week6_candidate.json")
    nogate = {**cfg, "gates": {}}          # gates off, so we only see the landmark effect
    no_fb = {**nogate, "landmark_fallback": None}

    boxes_only = {p: {"box": jitter(sg1[p]["box"]), "det_conf": sg1[p]["det_conf"], "landmarks": None}
                  for p in paths}

    out = {"data": "LFW 6000 pairs, SFace stand-in for SG3, gates off, jittered boxes"}
    out["A_sg1_landmarks"] = run(pairs, sg1, imgs, rec, no_fb)
    out["B_boxes_only_no_fallback"] = run(pairs, boxes_only, imgs, rec, no_fb)  # = V1 today: all no_landmarks
    out["C_boxes_only_fallback"] = run(pairs, boxes_only, imgs, rec, nogate)

    # how close are the fallback points to YuNet-on-the-whole-image points?
    errs = []
    for p in paths:
        got = sg2_quality.find_landmarks(imgs[p], boxes_only[p]["box"], cfg["landmark_fallback"])
        if got is not None:
            ref = np.float32(sg1[p]["landmarks"])
            eye_d = np.linalg.norm(ref[1] - ref[0])
            errs.append(float(np.linalg.norm(np.float32(got) - ref, axis=1).mean() / eye_d))
    out["landmark_error_vs_full_image"] = {"mean_pct_of_eye_distance": round(100 * np.mean(errs), 2),
                                           "p95_pct": round(100 * np.percentile(errs, 95), 2)}

    # small back-row faces: does the fallback still find the points?
    small = load_pairs(folds=range(5))
    sp = sorted({p for a, b, _, _ in small for p in (a, b)})
    for px in (32, 24):
        f_ref, f_box, im = {}, {}, {}
        for p in sp:
            im[p], f_ref[p] = degrade(imgs[p], sg1[p], "size", px)
            f_box[p] = {**f_ref[p], "box": jitter(f_ref[p]["box"]), "landmarks": None}
        out[f"small_{px}px"] = {"sg1_landmarks": run(small, f_ref, im, rec, no_fb),
                                "fallback": run(small, f_box, im, rec, nogate)}
        print(px, "px done")
    print(json.dumps(out, indent=1))
    save("exp5_landmark_fallback", out)
