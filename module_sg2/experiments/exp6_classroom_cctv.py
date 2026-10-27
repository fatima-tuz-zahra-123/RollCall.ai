"""EXP-6  real classroom CCTV video (CR #28, 8 Oct): how do our gates behave on real data?

Real SG-1 (YOLOv12n-face via integration/sg1_adapter.py) -> SG-2 -> SFace as SG-3 stand-in.

Problem: nobody has labelled who sits where, so how do we know if a face is recognisable?
Trick: students stay in their seats during a lecture. So
  same seat, frames >= 60 s apart        -> same person   (genuine pair)
  different seats, same frame             -> different people (impostor pair)
That gives us thousands of real genuine/impostor pairs without labelling anything, and we can
measure where recognition breaks as a function of pose, face size and blur, on the actual camera.
(I spot-checked pairs by eye, see README.)

    python module_sg2/experiments/exp6_classroom_cctv.py --video "path/to/CR # 28 ... .mp4"
    python module_sg2/experiments/exp6_classroom_cctv.py --analyse-only     # reuse the cached faces
    # another room: --tag keeps its cache/results separate, and "transfer" checks the CR #28 config on it
    python module_sg2/experiments/exp6_classroom_cctv.py --video "path/to/CR # 29 ... .mp4" --tag cr29

Only numbers are saved to results/ (the repo is public, so no classmate faces get committed).
"""
import argparse
import sys
import time

from common import *
from exp3_edc import EDifFIQA

sys.path.insert(0, str(SG2.parent / "integration"))
CACHE = DATA / "exp6_faces.npz"
TAG = ""


def collect(video, start_s, end_s, step_s):
    from sg1_adapter import SG1Detector
    det = SG1Detector()
    rec, fiqa = sface(), EDifFIQA()
    cfg = sg2_quality.load_config(SG2 / "config" / "sg1_yolov12_v1.json")
    nogate = {**cfg, "gates": {}}
    cap = cv2.VideoCapture(video)
    rows, emb, lmks, sg1_total, no_lmk = [], [], [], 0, 0
    t0 = time.time()
    for t in range(start_s, end_s, step_s):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        faces = det(frame)
        sg1_total += len(faces)
        for f in faces:
            r = sg2_quality.assess_face(frame, f, nogate)      # gates off: we want every face aligned
            if r["face"] is None:
                no_lmk += 1
                continue
            m = r["metrics"]
            g = sg2_quality.assess_face(frame, {**f, "landmarks": r["landmarks"]}, cfg)  # what V1.1 decides
            x, y, w, h = f["box"]
            rows.append([t, x + w / 2, y + h / 2, w, h, m["det_conf"], m["face_px"], m["yaw"], m["blur"],
                         m["brightness"], m["sat_frac"], r["quality"], fiqa(r["face"]), g["accept"],
                         REASONS.index(g["reason"]) if g["reason"] else -1])
            emb.append(embed(rec, r["face"]))
            lmks.append(r["landmarks"])
        print(f"t={t}s  {len(faces)} faces  ({time.time() - t0:.0f}s elapsed)")
    np.savez_compressed(CACHE, rows=np.array(rows, np.float32), emb=np.array(emb, np.float32),
                        lmk=np.array(lmks, np.float32),
                        sg1_total=sg1_total, no_lmk=no_lmk)


COLS = ["t", "cx", "cy", "w", "h", "det_conf", "face_px", "yaw", "blur", "brightness", "sat_frac",
        "heuristic", "ediffiqa", "accept_v11", "reason_v11"]
REASONS = sg2_quality.REASONS


def make_pairs(R, min_gap=60, max_imp_per_frame=40, seed=0):
    rng = np.random.default_rng(seed)
    t, cx, cy, w = R["t"], R["cx"], R["cy"], R["w"]
    gen, imp = [], []
    for i in range(len(t)):
        # same seat later in the lecture: centre within a quarter face width, similar size
        j = np.where((t > t[i] + min_gap) & (np.hypot(cx - cx[i], cy - cy[i]) < 0.25 * w[i])
                     & (w / w[i] > 0.75) & (w / w[i] < 1.33))[0]
        if len(j) > 5:                                  # cap per face, else ~millions of pairs
            j = rng.choice(j, 5, replace=False)
        gen += [(i, k) for k in j]
    for ft in np.unique(t):
        idx = np.where(t == ft)[0]
        cand = [(a, b) for ai, a in enumerate(idx) for b in idx[ai + 1:]
                if np.hypot(cx[a] - cx[b], cy[a] - cy[b]) > 1.5 * max(w[a], w[b])]
        if len(cand) > max_imp_per_frame:
            cand = [cand[k] for k in rng.choice(len(cand), max_imp_per_frame, replace=False)]
        imp += cand
    return np.array(gen), np.array(imp)


def config_search(R, pairs, lab, s, split_t=35 * 60):
    """Pick CCTV gate values on the first half of the lecture, then report them on the second half.
    Tuning and reporting on the same frames would just overfit, so the halves never mix."""
    import itertools
    first = lambda i: R["t"][pairs[:, i]] < split_t
    tune, test = first(0) & first(1), ~first(0) & ~first(1)
    lit = (R["brightness"] >= 20) & (R["brightness"] <= 180)

    def evaluate(mask_face, sel):
        keep = sel & mask_face[pairs[:, 0]] & mask_face[pairs[:, 1]]
        if (keep & lab).sum() < 200 or (keep & ~lab).sum() < 200:
            return None
        t_k = threshold_at_fmr(s[keep], lab[keep])
        faces_in_half = np.zeros(len(R["t"]), bool)
        faces_in_half[np.unique(pairs[sel].ravel())] = True
        return {"faces_accepted_pct": round(100 * mask_face[faces_in_half].mean(), 1),
                "genuine_pairs": int((keep & lab).sum()),
                "fnmr_kept": round(fnmr_at(s[keep], lab[keep], t_k), 4)}

    def gate(r):
        return ((np.abs(R["yaw"]) <= r["max_abs_yaw"]) & (R["pitch_dev"] <= r["max_pitch_dev"])
                & (R["blur"] >= r["min_blur"]) & (R["face_px"] >= r["min_face_px"])
                & (R["det_conf"] >= r["min_det_conf"]) & lit)

    rows = []
    for yaw, pitch, blur, px, conf in itertools.product([0.45, 0.6, 0.75], [0.1, 0.15, 0.2, 9],
                                                       [20, 35, 50, 75], [24, 28], [0.25, 0.5]):
        r = {"max_abs_yaw": yaw, "max_pitch_dev": pitch, "min_blur": blur, "min_face_px": px, "min_det_conf": conf}
        e = evaluate(gate(r), tune)
        if e:
            rows.append({**r, **e})
    # rule fixed BEFORE looking at the test half: lowest tune-half FNMR among configs that still accept
    # >= 60% of faces (attendance needs coverage: a student whose faces always get rejected = absent)
    best = min((r for r in rows if r["faces_accepted_pct"] >= 60), key=lambda r: r["fnmr_kept"])
    chosen = {k: best[k] for k in ("max_abs_yaw", "max_pitch_dev", "min_blur", "min_face_px", "min_det_conf")}
    v11 = R["accept_v11"] > 0.5
    allf = np.ones(len(R["t"]), bool)
    return {"rule": "lowest tune-half FNMR with >=60% faces accepted, then reported on the unseen second half",
            "split": "tune = minutes 12-35, test = minutes 35-58", "configs_tried": len(rows), "chosen": chosen,
            "tune_half": {"no_gates": evaluate(allf, tune), "v1.1": evaluate(v11, tune),
                          "chosen": evaluate(gate(chosen), tune)},
            "test_half": {"no_gates": evaluate(allf, test), "v1.1": evaluate(v11, test),
                          "chosen": evaluate(gate(chosen), test)}}


def imgsz_check(video, times=(1200, 1800, 2700)):
    """Does SG-1 find more back-row faces at a bigger inference size? (it doesn't)"""
    from sg1_adapter import SG1Detector
    cap = cv2.VideoCapture(video)
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        frames.append(cap.read()[1])
    out = {}
    for imgsz in (1280, 1920, 2560):
        det = SG1Detector(imgsz=imgsz)
        det(frames[0])                                   # warm-up
        n, ms = [], []
        for f in frames:
            t0 = time.perf_counter()
            n.append(len(det(f)))
            ms.append((time.perf_counter() - t0) * 1000)
        out[imgsz] = {"faces_per_frame": n, "ms_per_frame": round(float(np.mean(ms)))}
        print(imgsz, out[imgsz])
    save("exp6_sg1_imgsz_check", {"frames_at_seconds": list(times), "cpu": "dev laptop", "results": out})


def transfer(R, pairs, lab, s):
    """cctv_v2's gates (picked on CR #28) applied as-is to the whole video, no re-tuning."""
    g = sg2_quality.load_config(SG2 / "config" / "cctv_v2.json")["gates"]
    lo, hi = g["brightness_range"]
    m = ((np.abs(R["yaw"]) <= g["max_abs_yaw"]) & (R["pitch_dev"] <= g["max_pitch_dev"]) & (R["blur"] >= g["min_blur"])
         & (R["face_px"] >= g["min_face_px"]) & (R["det_conf"] >= g["min_det_conf"])
         & (R["brightness"] >= lo) & (R["brightness"] <= hi))
    out = {}
    for name, mask in [("no_gates", np.ones(len(m), bool)), ("v1.1", R["accept_v11"] > 0.5), ("cctv_v2", m)]:
        keep = mask[pairs[:, 0]] & mask[pairs[:, 1]]
        t = threshold_at_fmr(s[keep], lab[keep])
        out[name] = {"faces_accepted_pct": round(100 * mask.mean(), 1), "genuine_pairs": int((keep & lab).sum()),
                     "fnmr_kept": round(fnmr_at(s[keep], lab[keep], t), 4)}
    return out


def analyse():
    z = np.load(CACHE)
    R = {c: z["rows"][:, k] for k, c in enumerate(COLS)}
    E = z["emb"]
    R["pitch_dev"] = np.array([abs(sg2_quality.pitch_proxy(l) - 0.495) for l in z["lmk"]], np.float32)
    gen, imp = make_pairs(R)
    pairs = np.vstack([gen, imp])
    lab = np.r_[np.ones(len(gen), bool), np.zeros(len(imp), bool)]
    s = np.einsum("ij,ij->i", E[pairs[:, 0]], E[pairs[:, 1]])
    thr = threshold_at_fmr(s, lab)
    pmax = lambda k: np.maximum(np.abs(R[k][pairs[:, 0]]), np.abs(R[k][pairs[:, 1]]))
    pmin = lambda k: np.minimum(R[k][pairs[:, 0]], R[k][pairs[:, 1]])

    out = {"data": "CR #28 CCTV 2560x1440, real SG-1 YOLOv12n-face (imgsz 1280) + SG-2 V1.1 fallback, "
                   "SFace stand-in for SG-3; seat-consistency pseudo labels",
           "faces_from_sg1": int(z["sg1_total"]), "faces_aligned": len(E), "no_landmarks": int(z["no_lmk"]),
           "frames": int(len(np.unique(R["t"]))), "genuine_pairs": int(len(gen)), "impostor_pairs": int(len(imp)),
           "threshold_fmr1": round(thr, 4), "fnmr_all_faces": round(fnmr_at(s, lab, thr), 4),
           "eer_all_faces": round(eer(s, lab), 4)}

    def by_bins(vals, edges, name):
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (vals >= lo) & (vals < hi) & lab
            rows.append({name: f"{lo}-{hi}", "genuine_pairs": int(m.sum()),
                         "fnmr": round(float((s[m] <= thr).mean()), 4) if m.sum() >= 30 else None})
        return rows

    out["fnmr_by_pair_max_yaw"] = by_bins(pmax("yaw"), [0, 0.15, 0.3, 0.45, 0.6, 0.8, 9], "max_abs_yaw")
    out["fnmr_by_pair_min_face_px"] = by_bins(pmin("face_px"), [0, 16, 20, 24, 28, 32, 40, 999], "min_face_px")
    out["fnmr_by_pair_max_pitch_dev"] = by_bins(pmax("pitch_dev"), [0, 0.1, 0.2, 0.3, 0.4, 9], "max_pitch_dev")
    out["fnmr_by_pair_min_blur"] = by_bins(pmin("blur"), [0, 10, 20, 35, 50, 100, 1e9], "min_blur")

    # face-level distributions (what the camera actually gives us)
    out["face_stats"] = {k: {"p10": round(float(np.percentile(np.abs(R[k]) if k == "yaw" else R[k], 10)), 3),
                             "median": round(float(np.median(np.abs(R[k]) if k == "yaw" else R[k])), 3),
                             "p90": round(float(np.percentile(np.abs(R[k]) if k == "yaw" else R[k], 90)), 3)}
                         for k in ("face_px", "yaw", "pitch_dev", "blur", "brightness", "det_conf")}
    rr = R["reason_v11"].astype(int)
    out["v11_decisions"] = {"accepted": int(R["accept_v11"].sum()),
                            **{REASONS[k]: int((rr == k).sum()) for k in np.unique(rr) if k >= 0}}

    # pose gate sweep on real data: a pair counts only if both faces pass
    sweep = []
    for th in [0.3, 0.45, 0.6, 0.75, 9]:
        ok = np.abs(R["yaw"]) <= th
        keep = ok[pairs[:, 0]] & ok[pairs[:, 1]]
        t_k = threshold_at_fmr(s[keep], lab[keep])
        sweep.append({"max_abs_yaw": th, "faces_accepted_pct": round(100 * ok.mean(), 1),
                      "genuine_pairs_kept": int((keep & lab).sum()),
                      "fnmr_kept": round(fnmr_at(s[keep], lab[keep], t_k), 4)})
    out["pose_gate_sweep"] = sweep

    # EDC on real data: which quality score ranks faces the way recognition errors do?
    edc = {}
    rng = np.random.default_rng(1)
    scores = {"random": rng.random(len(E)), "heuristic (V1)": R["heuristic"], "eDifFIQA(T)": R["ediffiqa"],
              "blur only": R["blur"], "frontal-ness (-|yaw|)": -np.abs(R["yaw"]), "level head (-|pitch dev|)": -R["pitch_dev"],
              "face size": R["face_px"]}
    xs = np.linspace(0, 0.3, 31)
    for name, q in scores.items():
        pq = np.minimum(q[pairs[:, 0]], q[pairs[:, 1]])
        order = np.argsort(pq)
        ys = []
        for x in xs:
            keep = np.ones(len(s), bool)
            keep[order[:int(round(x * len(s)))]] = False
            ys.append(fnmr_at(s[keep], lab[keep], thr))
        ys = np.array(ys)
        edc[name] = {"pauc_0_30": round(float(np.sum(ys[:-1] * np.diff(xs)) / (ys[0] * 0.3)), 3),
                     "fnmr_0": round(float(ys[0]), 4), "fnmr_after_30pct": round(float(ys[-1]), 4)}
    out["edc_real"] = edc
    out["config_search"] = config_search(R, pairs, lab, s)
    out["transfer_cctv_v2"] = transfer(R, pairs, lab, s)
    print(json.dumps(out, indent=1))
    save("exp6_classroom_cctv" + (f"_{TAG}" if TAG else ""), out)
    return R, E, gen, imp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--start", type=int, default=12 * 60, help="seconds; class is seated from ~min 12")
    ap.add_argument("--end", type=int, default=58 * 60)
    ap.add_argument("--step", type=int, default=5, help="one frame every N seconds")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--imgsz-check", action="store_true", help="only compare SG-1 inference sizes")
    ap.add_argument("--tag", default="", help="name for another video's cache/results, e.g. cr29")
    a = ap.parse_args()
    if a.tag:
        TAG, CACHE = a.tag, DATA / f"exp6_faces_{a.tag}.npz"
    if a.imgsz_check:
        imgsz_check(a.video)
        sys.exit()
    if not a.analyse_only:
        collect(a.video, a.start, a.end, a.step)
    analyse()
