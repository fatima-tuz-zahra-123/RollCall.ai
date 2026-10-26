"""EXP-3b  Diagnostic for EXP-3: WHY does the baseline heuristic score rank worse than random on the mix?
Mean quality score per degradation type (same 8 types as EXP-3's mix) next to SFace's FNMR for that type.
A useful score must be LOW where FNMR is HIGH.      Run: python module_sg2/experiments/exp3b_score_by_degradation.py
"""
from common import *
from exp2_thresholds import degrade
from exp3_edc import EDifFIQA

TYPES = [("blur", 0), ("blur", 3), ("blur", 5), ("size", 24), ("size", 16), ("exposure", 0.2), ("exposure", 2.2)]

if __name__ == "__main__":
    pairs = [p for p in load_pairs(folds=range(5)) if p[2]][:1000]          # 1,000 genuine pairs
    paths = sorted({p for a, b, _, _ in pairs for p in (a, b)})
    sg1, rec, fiqa = mock_sg1(paths), sface(), EDifFIQA()
    cfg = {**sg2_quality.load_config(SG2 / "config" / "baseline_v1.json"), "gates": {}}
    all_pairs = load_pairs()
    s_all, lab_all = [], []
    # threshold at FMR=1% from clean exp-1 style scores on all 6000 pairs is in results/exp1; recompute cheaply here
    t = json.loads((RESULTS / "exp2_thresholds.json").read_text())["threshold_fmr1_clean"]
    rows = []
    for kind, lvl in TYPES:
        h, e, miss = [], [], []
        for a, b, _, _ in pairs:
            ga = sg2_quality.assess_face(cv2.imread(a), sg1[a], cfg)["face"]
            img, face = degrade(cv2.imread(b), sg1[b], kind, lvl)
            r = sg2_quality.assess_face(img, face, cfg)
            h.append(r["quality"])
            e.append(fiqa(r["face"]))
            miss.append(float(embed(rec, ga) @ embed(rec, r["face"])) <= t)
        rows.append({"type": f"{kind}={lvl}", "fnmr": round(float(np.mean(miss)), 4),
                     "mean_heuristic": round(float(np.mean(h)), 3), "mean_ediffiqa": round(float(np.mean(e)), 3)})
        print(rows[-1])
    fn = np.array([r["fnmr"] for r in rows])
    corr = {k: round(float(np.corrcoef(fn, [r[k] for r in rows])[0, 1]), 3) for k in ("mean_heuristic", "mean_ediffiqa")}
    print("Pearson corr(score, FNMR) across types (want strongly NEGATIVE):", corr)
    save("exp3b_score_by_degradation", {"data": "1000 genuine LFW pairs, probe degraded, oracle SG-1",
                                         "threshold_fmr1_clean": t, "rows": rows, "corr_with_fnmr": corr})
