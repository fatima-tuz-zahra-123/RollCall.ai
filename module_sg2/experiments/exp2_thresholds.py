"""EXP-2  Quality-gate thresholds: do our gates reject exactly the faces recognition fails on?

Controlled degradation of the PROBE image of each pair (gallery stays clean), on LFW folds 1-5 (3,000 pairs):
  blur      Gaussian sigma           (camera focus / motion)
  size      face width in pixels     (back-row students): whole frame downscaled, SG-1 boxes scaled with it
  exposure  intensity x alpha        (dark classroom / blown-out window light)
SG-1 = YuNet on the clean image (oracle detection) so only SG-2 + SG-3 are being measured.
For every level we record: recognition FNMR at the clean FMR=1% threshold (does SG-3 break?) and the
baseline gate accept-rate (does SG-2 catch it?). Then each threshold is swept on the pooled data.
Part D sweeps the pose gate on the REAL head-pose variation of all 6,000 clean pairs.
                                   Run: python module_sg2/experiments/exp2_thresholds.py
"""
from common import *

LEVELS = {"blur": [0, 1, 2, 3, 4, 6],
          "size": [None, 64, 48, 32, 24, 16],          # None = original (~93 px median)
          "exposure": [1.0, 0.6, 0.35, 0.2, 0.1, 1.6, 2.2, 3.0]}


def degrade(img, face, kind, lvl):
    if kind == "blur":
        return (cv2.GaussianBlur(img, (0, 0), lvl) if lvl else img), face
    if kind == "exposure":
        return cv2.convertScaleAbs(img, alpha=lvl), face
    if lvl is None:
        return img, face
    s = lvl / min(face["box"][2:])
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return small, {**face, "box": [v * s for v in face["box"]],
                   "landmarks": [[x * s, y * s] for x, y in face["landmarks"]]}


def sweep(rows, key, thresholds, keep, t, clean_rows):
    """For each threshold: % clean probes rejected (false rejects) + FNMR of genuine pairs that are ACCEPTED."""
    out = []
    for th in thresholds:
        acc = [r for r in rows if keep(r["m"], th)]
        gen = np.array([r["s"] for r in acc if r["g"]])
        out.append({key: th,
                    "clean_rejected_pct": round(100 * np.mean([not keep(r["m"], th) for r in clean_rows]), 2),
                    "all_rejected_pct": round(100 * (1 - len(acc) / len(rows)), 2),
                    "fnmr_of_accepted": round(float((gen <= t).mean()), 4) if len(gen) else None})
    return out


if __name__ == "__main__":
    pairs = load_pairs(folds=range(5))
    paths = sorted({p for a, b, _, _ in pairs for p in (a, b)})
    sg1, rec = mock_sg1(paths), sface()
    base = sg2_quality.load_config(SG2 / "config" / "baseline_v1.json")
    nogate = {**base, "gates": {}}
    imgs = {p: cv2.imread(p) for p in paths}
    gal = {a: embed(rec, sg2_quality.assess_face(imgs[a], sg1[a], nogate)["face"]) for a, _, _, _ in pairs}
    labels = np.array([g for _, _, g, _ in pairs])

    per = {}  # kind -> level -> list of {s, g, m, accept, reason}
    for kind, levels in LEVELS.items():
        per[kind] = {}
        for lvl in levels:
            rs = []
            for a, b, g, _ in pairs:
                img, face = degrade(imgs[b], sg1[b], kind, lvl)
                raw = sg2_quality.assess_face(img, face, nogate)          # always produce the face -> SG-3 score
                gated = sg2_quality.assess_face(img, face, base)          # what the baseline V1 gates decide
                rs.append({"s": float(gal[a] @ embed(rec, raw["face"])), "g": g, "m": raw["metrics"],
                           "accept": gated["accept"], "reason": gated["reason"]})
            per[kind][str(lvl)] = rs
            print(kind, lvl, "done")

    clean = per["blur"]["0"]
    t = threshold_at_fmr(np.array([r["s"] for r in clean]), labels)
    level_table = []
    for kind, d in per.items():
        for lvl, rs in d.items():
            s = np.array([r["s"] for r in rs])
            reasons = {}
            for r in rs:
                if not r["accept"]:
                    reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
            gen_acc = np.array([r["s"] for r in rs if r["g"] and r["accept"]])
            level_table.append({
                "kind": kind, "level": lvl,
                "fnmr_no_gate": round(fnmr_at(s, labels, t), 4),
                "fmr_no_gate": round(float((s[~labels] > t).mean()), 4),
                "baseline_accept_pct": round(100 * np.mean([r["accept"] for r in rs]), 1),
                "fnmr_of_accepted": round(float((gen_acc <= t).mean()), 4) if len(gen_acc) else None,
                "median_blur": round(float(np.median([r["m"]["blur"] for r in rs])), 1),
                "median_brightness": round(float(np.median([r["m"]["brightness"] for r in rs])), 1),
                "median_face_px": round(float(np.median([r["m"]["face_px"] for r in rs])), 1),
                "reject_reasons": reasons})
            print(level_table[-1])

    pool = lambda kind: [r for rs in per[kind].values() for r in rs]
    sweeps = {
        "min_blur": sweep(pool("blur"), "min_blur", [0, 10, 20, 30, 50, 75, 100, 150],
                          lambda m, th: m["blur"] >= th, t, clean),
        "min_face_px": sweep(pool("size"), "min_face_px", [0, 16, 24, 32, 40, 48, 64],
                             lambda m, th: m["face_px"] >= th, t, per["size"]["None"]),
        "brightness_low": sweep(pool("exposure"), "min_brightness", [0, 20, 35, 50, 70],
                                lambda m, th: m["brightness"] >= th, t, clean),
        "brightness_high": sweep(pool("exposure"), "max_brightness", [255, 230, 220, 200, 180],
                                 lambda m, th: m["brightness"] <= th, t, clean),
    }
    # Part D: pose gate on real LFW pose variation (all 6,000 clean pairs; pair is kept if BOTH faces pass)
    allp = load_pairs()
    sg1_all = mock_sg1(sorted({p for a, b, _, _ in allp for p in (a, b)}))
    embs = {}
    for a, b, _, _ in allp:
        for p in (a, b):
            if p not in embs:
                embs[p] = embed(rec, sg2_quality.assess_face(cv2.imread(p), sg1_all[p], nogate)["face"])
    s_all = np.array([float(embs[a] @ embs[b]) for a, b, _, _ in allp])
    lab_all = np.array([g for _, _, g, _ in allp])
    t_all = threshold_at_fmr(s_all, lab_all)
    pyaw = np.array([max(abs(sg2_quality.yaw_proxy(sg1_all[a]["landmarks"])),
                         abs(sg2_quality.yaw_proxy(sg1_all[b]["landmarks"]))) for a, b, _, _ in allp])
    pose = []
    for th in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 9.9]:
        keep = pyaw <= th
        pose.append({"max_abs_yaw": th, "pairs_dropped_pct": round(100 * (1 - keep.mean()), 2),
                     "fnmr_of_kept": round(fnmr_at(s_all[keep], lab_all[keep], t_all), 4)})
    sweeps["max_abs_yaw (real LFW pose, 6000 pairs)"] = pose
    print(json.dumps(sweeps, indent=1))
    save("exp2_thresholds", {"data": "LFW folds 1-5 (3000 pairs), probe degraded, gallery clean; oracle SG-1",
                             "threshold_fmr1_clean": round(t, 4), "levels": level_table, "sweeps": sweeps})
