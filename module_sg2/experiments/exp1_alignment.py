"""EXP-1  Alignment ON vs OFF (and crop margin): which SG-2 output gives SG-3 the best recognition?

Same data for every config: all 6,000 LFW pairs, faces from mock SG-1 (YuNet). Gates are OFF here so we
measure only the preprocessing. Stand-in SG-3 = SFace. Metrics: LFW 10-fold accuracy, FNMR @ FMR=1%, EER,
SG-2 time per face.            Run: python module_sg2/experiments/exp1_alignment.py
"""
from common import *

CONFIGS = {  # name -> (align mode, crop margin)
    "A0 baseline: 5-pt similarity align": ("similarity5", 0.0),
    "A1 no align: tight box crop": ("crop", 0.0),
    "A2 no align: box crop + 20% margin": ("crop", 0.2),
    "A3 no align: box crop + 40% margin": ("crop", 0.4),
    "REF OpenCV FaceRecognizerSF.alignCrop": ("opencv", None),
}

if __name__ == "__main__":
    pairs = load_pairs()
    paths = sorted({p for a, b, _, _ in pairs for p in (a, b)})
    sg1 = mock_sg1(paths)
    rec = sface()
    base = sg2_quality.load_config(SG2 / "config" / "baseline_v1.json")
    labels = np.array([g for _, _, g, _ in pairs])
    folds = np.array([f for _, _, _, f in pairs])
    rows, imgs = [], {p: cv2.imread(p) for p in paths}
    for name, (mode, margin) in CONFIGS.items():
        cfg = {**base, "align": mode, "crop_margin": margin, "gates": {}}
        emb, t_sg2 = {}, 0.0
        for p in paths:
            face = sg1[p]
            t = timer()
            if mode == "opencv":
                row = np.array(face["box"] + sum(face["landmarks"], []) + [face["det_conf"]], np.float32)
                out = rec.alignCrop(imgs[p], row)
            else:
                out = sg2_quality.assess_face(imgs[p], face, cfg)["face"]
            t_sg2 += t()
            emb[p] = embed(rec, out)
        s = np.array([float(emb[a] @ emb[b]) for a, b, _, _ in pairs])
        acc, sd = lfw_accuracy(s, labels, folds)
        thr = threshold_at_fmr(s, labels)
        rows.append({"config": name, "lfw_acc": round(acc, 4), "acc_std": round(sd, 4),
                     "fnmr_at_fmr1": round(fnmr_at(s, labels, thr), 4), "eer": round(eer(s, labels), 4),
                     "sg2_ms_per_face": round(1000 * t_sg2 / len(paths), 3)})
        print(rows[-1])
    save("exp1_alignment", {"data": "LFW 6000 pairs (10 folds), mock SG-1 = YuNet, SG-3 stand-in = SFace",
                            "n_images": len(paths), "rows": rows})
