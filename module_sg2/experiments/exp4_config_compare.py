"""EXP-4  Baseline V1 config vs Week-6 candidate config, END-TO-END with a REAL detector on degraded images.

Removes EXP-2's oracle-detection shortcut: YuNet (mock SG-1) is re-run on every degraded probe, so missed
detections and noisier landmarks on small/dark faces count. Same 3,000 pairs (LFW folds 1-5), same 19
degradation levels as EXP-2, same FMR=1% threshold from the clean set. For each config we report:
  accept %       share of probes SG-2 forwards to SG-3 (clean accept % = 100 - false-reject %)
  harmful pass % genuine pairs that were ACCEPTED but still not matched (would cost a student attendance)
  safe drop %    genuine pairs REJECTED that SG-3 would have failed anyway (useful rejections)
                                   Run: python module_sg2/experiments/exp4_config_compare.py
"""
from common import *
from exp2_thresholds import LEVELS, degrade

CFGS = {name: sg2_quality.load_config(SG2 / "config" / f"{name}.json") for name in ("baseline_v1", "week6_candidate")}

if __name__ == "__main__":
    pairs = load_pairs(folds=range(5))
    paths = sorted({p for a, b, _, _ in pairs for p in (a, b)})
    sg1, rec, det = mock_sg1(paths), sface(), yunet()
    nogate = {**CFGS["baseline_v1"], "gates": {}}
    imgs = {p: cv2.imread(p) for p in paths}
    gal = {a: embed(rec, sg2_quality.assess_face(imgs[a], sg1[a], nogate)["face"]) for a, _, _, _ in pairs}
    labels = np.array([g for _, _, g, _ in pairs])
    clean_s = np.array([float(gal[a] @ embed(rec, sg2_quality.assess_face(imgs[b], sg1[b], nogate)["face"]))
                        for a, b, _, _ in pairs])
    t = threshold_at_fmr(clean_s, labels)

    rows = []
    for kind, levels in LEVELS.items():
        for lvl in levels:
            stats = {n: {"acc": 0, "harm": 0, "safe": 0} for n in CFGS}
            missed, fnmr_hits = 0, []
            for (a, b, g, _) in pairs:
                img, _ = degrade(imgs[b], sg1[b], kind, lvl)
                face = detect_central(det, img)                     # REAL SG-1 on the degraded image
                if face is None:
                    missed += 1
                    continue
                s = float(gal[a] @ embed(rec, sg2_quality.assess_face(img, face, nogate)["face"]))
                if g:
                    fnmr_hits.append(s <= t)
                for n, cfg in CFGS.items():
                    ok = sg2_quality.assess_face(img, face, cfg)["accept"]
                    stats[n]["acc"] += ok
                    if g and ok and s <= t:
                        stats[n]["harm"] += 1
                    if g and not ok and s <= t:
                        stats[n]["safe"] += 1
            n_gen = int(labels.sum())
            row = {"kind": kind, "level": str(lvl), "sg1_missed_pct": round(100 * missed / len(pairs), 2),
                   "fnmr_no_gate_detected": round(float(np.mean(fnmr_hits)), 4) if fnmr_hits else None}
            for n in CFGS:
                row[f"{n}_accept_pct"] = round(100 * stats[n]["acc"] / len(pairs), 1)
                row[f"{n}_harmful_pass_pct"] = round(100 * stats[n]["harm"] / n_gen, 2)
                row[f"{n}_safe_drop_pct"] = round(100 * stats[n]["safe"] / n_gen, 2)
            rows.append(row)
            print(row)
    summary = {}
    for n in CFGS:
        summary[n] = {"clean_accept_pct": rows[0][f"{n}_accept_pct"],
                      "mean_accept_pct_all_levels": round(float(np.mean([r[f"{n}_accept_pct"] for r in rows])), 1),
                      "total_harmful_pass_pct": round(float(np.sum([r[f"{n}_harmful_pass_pct"] for r in rows])), 2),
                      "total_safe_drop_pct": round(float(np.sum([r[f"{n}_safe_drop_pct"] for r in rows])), 2)}
    print(json.dumps(summary, indent=1))
    save("exp4_config_compare", {"data": "LFW folds 1-5 (3000 pairs), probe degraded, REAL YuNet on degraded probe",
                                 "threshold_fmr1_clean": round(t, 4), "summary": summary, "levels": rows})
