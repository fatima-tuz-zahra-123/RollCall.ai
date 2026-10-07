"""EXP-3  Quality SCORE: does our score rank faces the way recognition errors do?  (EDC, ISO/IEC 29794-1)

Error-versus-Discard Characteristic: drop the x% lowest-quality pairs (pair quality = min of its two faces),
re-measure FNMR at the FIXED threshold set for FMR=1% before discarding. A good quality score makes FNMR fall
fast. Summary number = pAUC of the EDC over 0-20% discard, normalised by FNMR at 0% (lower = better;
random discarding ~ 1.0).
Scores compared:  random | blur only (Laplacian var) | baseline heuristic (sg2_quality) | eDifFIQA(T) (learned,
OpenCV Zoo, 7 MB).  Two test sets: clean LFW (6,000 pairs) and a "classroom-like" mix where each probe is
randomly clean / blurred / small / dark (seeded).      Run: python module_sg2/experiments/exp3_edc.py
"""
from common import *
from exp2_thresholds import degrade

MAX_DISCARD = 0.20


class EDifFIQA:  # same preprocessing as OpenCV Zoo's ediffiqa.py: RGB, [-1,1], NCHW, on the ALIGNED 112x112 face
    def __init__(self):
        self.net = cv2.dnn.readNetFromONNX(str(MODELS / "ediffiqa_tiny_jun2024.onnx"))

    def __call__(self, face112):
        x = (cv2.cvtColor(face112, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - 0.5) / 0.5
        self.net.setInput(np.moveaxis(x[None], -1, 1))
        return float(np.asarray(self.net.forward()).flatten()[0])


def edc(scores, labels, pair_q, t, steps=41):
    order = np.argsort(pair_q)                      # lowest quality first
    xs = np.linspace(0, MAX_DISCARD, steps)
    ys = []
    for x in xs:
        keep = np.ones(len(scores), bool)
        keep[order[:int(round(x * len(scores)))]] = False
        ys.append(fnmr_at(scores[keep], labels[keep], t))
    ys = np.array(ys)
    # stepwise (not trapezoid) integration, as recommended by Schlett et al. (arXiv 2303.13294)
    area = float(np.sum(ys[:-1] * np.diff(xs)))
    pauc = area / (ys[0] * MAX_DISCARD) if ys[0] > 0 else float("nan")
    return xs, ys, pauc


def run(pairs, faces, imgs, rec, cfg, fiqa, rng):
    paths = sorted({p for a, b, _, _ in pairs for p in (a, b)})
    q = {k: {} for k in ("random", "blur only", "baseline heuristic", "eDifFIQA(T)")}
    emb, ms = {}, {"baseline heuristic": 0.0, "eDifFIQA(T)": 0.0}
    for p in paths:
        r = sg2_quality.assess_face(imgs[p], faces[p], {**cfg, "gates": {}})
        emb[p] = embed(rec, r["face"])
        q["random"][p] = rng.random()
        q["blur only"][p] = r["metrics"]["blur"]
        q["baseline heuristic"][p] = r["quality"]
        t0 = timer()
        q["eDifFIQA(T)"][p] = fiqa(r["face"])
        ms["eDifFIQA(T)"] += t0()
    s = np.array([float(emb[a] @ emb[b]) for a, b, _, _ in pairs])
    lab = np.array([g for _, _, g, _ in pairs])
    t = threshold_at_fmr(s, lab)
    out = {}
    for name, d in q.items():
        pq = np.array([min(d[a], d[b]) for a, b, _, _ in pairs])
        xs, ys, pauc = edc(s, lab, pq, t)
        out[name] = {"pauc_0_20": round(pauc, 4), "fnmr_at_0": round(ys[0], 4),
                     "fnmr_at_10pct": round(ys[np.argmin(abs(xs - 0.10))], 4),
                     "fnmr_at_20pct": round(ys[-1], 4), "curve": [round(v, 4) for v in ys]}
        print(name, {k: v for k, v in out[name].items() if k != "curve"})
    out["_eDifFIQA_ms_per_face"] = round(1000 * ms["eDifFIQA(T)"] / len(paths), 2)
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    pairs = load_pairs()
    paths = sorted({p for a, b, _, _ in pairs for p in (a, b)})
    sg1, rec, fiqa = mock_sg1(paths), sface(), EDifFIQA()
    cfg = sg2_quality.load_config(SG2 / "config" / "baseline_v1.json")
    imgs = {p: cv2.imread(p) for p in paths}

    res = {"clean LFW": run(pairs, sg1, imgs, rec, cfg, fiqa, rng)}

    # classroom-like mix: degrade every PROBE (b) image independently, keyed so a b-image reused as an
    # a-image elsewhere stays clean there
    mix_imgs, mix_faces, mix_pairs = dict(imgs), dict(sg1), []
    for i, (a, b, g, f) in enumerate(pairs):
        kind, lvl = [("blur", 0), ("blur", 3), ("blur", 5), ("size", 24), ("size", 16),
                     ("exposure", 0.2), ("exposure", 2.2), ("blur", 0)][rng.integers(8)]
        key = f"{b}#probe{i}"
        mix_imgs[key], mix_faces[key] = degrade(imgs[b], sg1[b], kind, lvl)
        mix_pairs.append((a, key, g, f))
    res["classroom-like mix"] = run(mix_pairs, mix_faces, mix_imgs, rec, cfg, fiqa, rng)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    xs = np.linspace(0, MAX_DISCARD, 41) * 100
    for ax, (name, d) in zip(axes, res.items()):
        for k, v in d.items():
            if not k.startswith("_"):
                ax.step(xs, v["curve"], where="post", label=f"{k} (pAUC {v['pauc_0_20']:.2f})")
        ax.set(title=f"EDC - {name}", xlabel="lowest-quality pairs discarded (%)",
               ylabel="FNMR @ FMR=1% threshold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp3_edc.png", dpi=120)
    save("exp3_edc", {"data": "LFW 6000 pairs; SG-3 stand-in SFace; threshold fixed at FMR=1% before discard",
                      "results": res})
