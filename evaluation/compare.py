"""Run:  python -m evaluation.compare   (from the repo root)
Same seeds + same data + same metrics for A and B."""
import time, itertools, csv, statistics as st
from module_sg5.mock_gen import generate
from module_sg5.algo_a import NofM
from module_sg5.algo_b import EvidenceAccum

SEEDS = range(20)

def run_one(make, seed):
    roster, frames, truth = generate(seed)
    dec = make(roster)
    t0 = time.perf_counter()
    for t, obs in frames:
        dec.update(t, obs)
    cpu_ms = (time.perf_counter() - t0) * 1000 / len(frames)
    out = dec.finalize(frames[-1][0])
    tp = fp = fn = tn = unc_p = 0
    ttd = []
    for s, g in truth.items():
        pred = out[s].status
        if pred == "Uncertain":
            pred = "Absent"            # Uncertain counts as not marked present
        if g == "Present" and pred == "Present": tp += 1; ttd.append(out[s].decision_time - out[s].first_seen)
        elif g == "Absent" and pred == "Present": fp += 1
        elif g == "Present": fn += 1
        else: tn += 1
    return tp, fp, fn, tn, ttd, cpu_ms

def evaluate(make):
    TP = FP = FN = TN = 0; ttd = []; cpu = []
    for sd in SEEDS:
        tp, fp, fn, tn, t, c = run_one(make, sd)
        TP += tp; FP += fp; FN += fn; TN += tn; ttd += t; cpu.append(c)
    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return dict(false_present_rate=FP / (FP + TN), missed_rate=FN / (FN + TP),
                precision=prec, recall=rec, f1=f1,
                mean_time_to_decision_s=st.mean(ttd) if ttd else float("nan"),
                cpu_ms_per_frame=st.mean(cpu))

if __name__ == "__main__":
    rows = []
    for T, N, W in itertools.product([0.6, 0.7, 0.8], [10, 20, 30, 40], [30, 60]):
        r = evaluate(lambda ro: NofM(ro, T=T, N=N, window=W))
        rows.append(dict(algo="A_NofM", params=f"T={T},N={N},W={W}", **r))
    for Tm, tau, on in itertools.product([0.5, 0.6, 0.7], [10, 20], [8, 12, 16, 24]):
        r = evaluate(lambda ro: EvidenceAccum(ro, T_min=Tm, tau=tau, on_thr=on, off_thr=on / 2))
        rows.append(dict(algo="B_Evidence", params=f"Tmin={Tm},tau={tau},on={on}", **r))
    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"{'algo':11}{'params':26}{'FPR':>7}{'miss':>7}{'F1':>7}{'TTD(s)':>8}{'ms/fr':>8}")
    for algo in ("A_NofM", "B_Evidence"):
        best = sorted([r for r in rows if r["algo"] == algo], key=lambda r: -r["f1"])[:5]
        print(f"-- top 5 {algo} by F1")
        for r in best:
            print(f"{r['algo']:11}{r['params']:26}{r['false_present_rate']:7.3f}{r['missed_rate']:7.3f}"
                  f"{r['f1']:7.3f}{r['mean_time_to_decision_s']:8.1f}{r['cpu_ms_per_frame']:8.3f}")
