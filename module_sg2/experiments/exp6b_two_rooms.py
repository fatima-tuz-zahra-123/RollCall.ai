"""EXP-6b  one gate config for BOTH classrooms (CR #28 + CR #29).

cctv_v2 was tuned on CR #28 only. On CR #29 (different camera angle, faces turned more) pitch and
blur behaved the same, but yaw didn't predict failure at all, so the 0.45 yaw limit just threw faces
away there. The Face-Off may be in yet another room, so we want settings that hold in both.

Same idea as EXP-6: pick on the first half (min 12-35) of BOTH videos, report on the second half
(min 35-58) of both. Rule fixed beforehand: lowest worst-room FNMR among configs that keep >= 50% of
faces in each room. Needs both caches (run exp6 on each video first).

    python module_sg2/experiments/exp6b_two_rooms.py
"""
import itertools

from common import *
import exp6_classroom_cctv as e6

ROOMS = {"CR28": DATA / "exp6_faces.npz", "CR29": DATA / "exp6_faces_cr29.npz"}


def load(path):
    z = np.load(path)
    R = {c: z["rows"][:, k] for k, c in enumerate(e6.COLS)}
    R["pitch_dev"] = np.array([abs(sg2_quality.pitch_proxy(l) - 0.495) for l in z["lmk"]], np.float32)
    gen, imp = e6.make_pairs(R)
    pairs = np.vstack([gen, imp])
    lab = np.r_[np.ones(len(gen), bool), np.zeros(len(imp), bool)]
    s = np.einsum("ij,ij->i", z["emb"][pairs[:, 0]], z["emb"][pairs[:, 1]])
    first = (R["t"][pairs[:, 0]] < 35 * 60) & (R["t"][pairs[:, 1]] < 35 * 60)
    second = (R["t"][pairs[:, 0]] >= 35 * 60) & (R["t"][pairs[:, 1]] >= 35 * 60)
    return R, pairs, lab, s, first, second


def mask(R, c):
    if c is None:
        return np.ones(len(R["t"]), bool)
    return ((np.abs(R["yaw"]) <= c["max_abs_yaw"]) & (R["pitch_dev"] <= c["max_pitch_dev"])
            & (R["blur"] >= c["min_blur"]) & (R["face_px"] >= c["min_face_px"]) & (R["det_conf"] >= c["min_det_conf"])
            & (R["brightness"] >= 20) & (R["brightness"] <= 180))


def score(room, m, half):
    R, pairs, lab, s, first, second = room
    sel = first if half == "first" else second
    keep = sel & m[pairs[:, 0]] & m[pairs[:, 1]]
    if (keep & lab).sum() < 200:
        return None
    faces = np.zeros(len(m), bool)
    faces[np.unique(pairs[sel].ravel())] = True
    t = threshold_at_fmr(s[keep], lab[keep])
    return {"faces_pct": round(100 * m[faces].mean(), 1), "fnmr": round(fnmr_at(s[keep], lab[keep], t), 4)}


if __name__ == "__main__":
    rooms = {k: load(p) for k, p in ROOMS.items()}
    cands = []
    for yaw, pitch, blur, px in itertools.product([0.45, 0.6, 0.75, 9], [0.1, 0.15, 0.2], [35, 50, 75, 100], [24, 28]):
        c = {"max_abs_yaw": yaw, "max_pitch_dev": pitch, "min_blur": blur, "min_face_px": px, "min_det_conf": 0.5}
        res = {k: score(r, mask(r[0], c), "first") for k, r in rooms.items()}
        if all(res.values()) and all(v["faces_pct"] >= 50 for v in res.values()):
            cands.append((max(v["fnmr"] for v in res.values()), c, res))
    worst, best, tune = min(cands, key=lambda x: x[0])

    v2 = sg2_quality.load_config(SG2 / "config" / "cctv_v2.json")["gates"]
    v2c = {k: v2[k] for k in ("max_abs_yaw", "max_pitch_dev", "min_blur", "min_face_px", "min_det_conf")}
    report = {}
    for name, c in [("no_gates", None), ("cctv_v2 (CR28 only)", v2c), ("chosen (both rooms)", best)]:
        report[name] = {k: score(r, mask(r[0], c), "second") for k, r in rooms.items()}
    # v1.1 = what sg1_yolov12_v1 actually decided, stored in the caches
    report["v1.1"] = {k: score(r, r[0]["accept_v11"] > 0.5, "second") for k, r in rooms.items()}
    out = {"rule": "lowest worst-room FNMR on first halves, >= 50% faces kept in each room; reported on second halves",
           "configs_tried": len(cands), "chosen": best, "tune_first_halves": tune, "test_second_halves": report}
    print(json.dumps(out, indent=1))
    save("exp6b_two_rooms", out)
