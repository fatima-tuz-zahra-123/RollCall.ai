"""WEEK 5 EXPERIMENT: compares LBPH (baseline) vs SFace (alternative) on the SAME test set
and sweeps the KNOWN/UNKNOWN threshold for each.

Input : data/test/known/<student_id>/*.jpg   enrolled students, DIFFERENT photos from enrollment
        data/test/unknown/*.jpg              people who are NOT enrolled
        (run enroll.py first)
Output (results/ folder):
        raw_scores_<method>.csv   one row per test image (best match + score)
        sweep_<method>.csv        metrics at every threshold in config.json
        errors_<method>.csv       wrong decisions at the operating point (for "representative errors")
        far_frr_plot.png          FAR / FRR / accuracy vs threshold for both methods
        summary.md                comparison table + recommended operating point

Usage:
    python evaluate.py
    python evaluate.py --methods sface        # only one method
"""

import argparse
import csv
import os
import time

from sg3_core import METHODS, FaceDetector, list_images, list_people, load_config, make_recognizer, recognize_image

# ------------------------------------------------------------------ data
def load_test_set(data_dir):
    items = []  # (path, true_id or None for unknown)
    known_dir = os.path.join(data_dir, "test", "known")
    for sid in list_people(known_dir):
        items += [(p, sid) for p in list_images(os.path.join(known_dir, sid))]
    items += [(p, None) for p in list_images(os.path.join(data_dir, "test", "unknown"))]
    return items


def run_method(method, cfg, items, detector):
    rec = make_recognizer(method, cfg)
    rec.load()
    rows = []
    for path, true_id in items:
        r = recognize_image(path, rec, detector)
        rows.append({
            "image": os.path.relpath(path, cfg["data_dir"]),
            "true_id": true_id or "UNKNOWN",
            "enrolled": true_id is not None,
            "best_match": r["best_match"],
            "score": r["score"],
            "face_found": r["decision"] != "NO_FACE",
            "time_ms": r["time_ms"],
        })
    return rec, rows


# ------------------------------------------------------------------ metrics
def decide(rec, row, thr):
    """Returns predicted id ('UNKNOWN' if rejected or no face)."""
    if not row["face_found"]:
        return "UNKNOWN"
    return row["best_match"] if rec.accept(row["score"], thr) else "UNKNOWN"


def metrics_at(rec, rows, thr):
    known = [r for r in rows if r["enrolled"]]
    unknown = [r for r in rows if not r["enrolled"]]
    ta = misid = fr = fa = tr = 0
    for r in known:
        p = decide(rec, r, thr)
        if p == "UNKNOWN":
            fr += 1
        elif p == r["true_id"]:
            ta += 1
        else:
            misid += 1
    for r in unknown:
        if decide(rec, r, thr) == "UNKNOWN":
            tr += 1
        else:
            fa += 1
    nk, nu = max(len(known), 1), max(len(unknown), 1)
    n = max(len(rows), 1)
    return {
        "threshold": thr,
        "correct_accept": ta, "misidentified": misid, "false_reject": fr,
        "false_accept": fa, "correct_reject": tr,
        "accuracy": round((ta + tr) / n, 4),
        "FAR": round(fa / nu, 4),                 # unknown people wrongly accepted
        "FRR": round(fr / nk, 4),                 # enrolled students wrongly rejected
        "misID_rate": round(misid / nk, 4),       # enrolled student given someone else's ID
    }


def rank1(rows):
    known = [r for r in rows if r["enrolled"]]
    if not known:
        return 0.0
    return round(sum(r["face_found"] and r["best_match"] == r["true_id"] for r in known) / len(known), 4)


def pick_operating_point(sweep):
    # highest accuracy; ties -> lowest FAR (wrongly marking a stranger present is worse)
    return max(sweep, key=lambda m: (m["accuracy"], -m["FAR"], -m["FRR"]))


# ------------------------------------------------------------------ outputs
def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot(sweeps, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot")
        return
    fig, axes = plt.subplots(1, len(sweeps), figsize=(6 * len(sweeps), 4.2), squeeze=False)
    for ax, (method, (rec, sweep)) in zip(axes[0], sweeps.items()):
        t = [m["threshold"] for m in sweep]
        ax.plot(t, [m["FAR"] for m in sweep], "o-", label="FAR (stranger accepted)")
        ax.plot(t, [m["FRR"] for m in sweep], "s-", label="FRR (student rejected)")
        ax.plot(t, [m["accuracy"] for m in sweep], "^-", label="Accuracy")
        ax.plot(t, [m["misID_rate"] for m in sweep], "x--", label="Mis-ID rate")
        ax.axvline(rec.threshold, color="gray", ls=":", label=f"config thr = {rec.threshold}")
        ax.set_title(f"{method.upper()}  ({rec.score_type})")
        ax.set_xlabel("threshold" + ("  (higher = stricter)" if rec.higher_is_better else "  (lower = stricter)"))
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"plot -> {out_png}")


def write_summary(path, items, results):
    nk = sum(1 for _, s in items if s is not None)
    nu = len(items) - nk
    L = [
        "# SG3 Face Recognition - Week 5 Results",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        f"Test set: {nk} enrolled-student images, {nu} non-enrolled (unknown) images.",
        "",
        "## Method comparison at each method's best threshold",
        "",
        "| Method | Score type | Best threshold | Accuracy | FAR | FRR | Mis-ID | Rank-1 (closed set) | No-face images | Avg time / image (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m, r in results.items():
        op = r["op"]
        L.append(
            f"| {m} | {r['rec'].score_type} | {op['threshold']} | {op['accuracy']:.3f} | {op['FAR']:.3f} | "
            f"{op['FRR']:.3f} | {op['misID_rate']:.3f} | {r['rank1']:.3f} | {r['noface']} | {r['avg_ms']:.1f} |"
        )
    L += ["", "## Full threshold sweeps", ""]
    for m, r in results.items():
        L += [f"### {m}", "", "| Threshold | Accuracy | FAR | FRR | Mis-ID |", "|---|---|---|---|---|"]
        for s in r["sweep"]:
            mark = " **<- best**" if s is r["op"] else ""
            L.append(f"| {s['threshold']} | {s['accuracy']:.3f} | {s['FAR']:.3f} | {s['FRR']:.3f} | {s['misID_rate']:.3f} |{mark}")
        L.append("")
    if results:
        best = max(results.items(), key=lambda kv: (kv[1]["op"]["accuracy"], -kv[1]["op"]["FAR"], kv[1]["rank1"], -kv[1]["op"]["misID_rate"]))
        L += [
            "## Preliminary recommended operating point",
            "",
            f"**{best[0]} with threshold {best[1]['op']['threshold']}** "
            f"(accuracy {best[1]['op']['accuracy']:.3f}, FAR {best[1]['op']['FAR']:.3f}, FRR {best[1]['op']['FRR']:.3f}).",
            "",
            "Definitions: FAR = non-enrolled faces accepted as a student; FRR = enrolled students rejected as UNKNOWN; "
            "Mis-ID = enrolled student accepted but given the wrong ID; Rank-1 = best match correct ignoring threshold.",
        ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"summary -> {path}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    args = ap.parse_args()

    cfg = load_config()
    items = load_test_set(cfg["data_dir"])
    if not items:
        raise SystemExit("No test images. Put photos in data/test/known/<student_id>/ and data/test/unknown/")
    os.makedirs(cfg["results_dir"], exist_ok=True)
    detector = FaceDetector(cfg)

    results, sweeps = {}, {}
    for m in args.methods:
        print(f"\n=== {m} ===")
        rec, rows = run_method(m, cfg, items, detector)
        sweep = [metrics_at(rec, rows, t) for t in cfg[m]["sweep"]]
        op = pick_operating_point(sweep)
        errors = []
        for r in rows:
            pred = decide(rec, r, op["threshold"])
            if pred != r["true_id"]:
                kind = ("no_face" if not r["face_found"] else
                        "false_accept" if not r["enrolled"] else
                        "false_reject" if pred == "UNKNOWN" else "misidentified")
                errors.append({**r, "predicted": pred, "error_type": kind, "threshold": op["threshold"]})
        rd = cfg["results_dir"]
        write_csv(os.path.join(rd, f"raw_scores_{m}.csv"), rows)
        write_csv(os.path.join(rd, f"sweep_{m}.csv"), sweep)
        write_csv(os.path.join(rd, f"errors_{m}.csv"), errors or [{"note": "no errors"}])
        timed = [r["time_ms"] for r in rows]
        results[m] = {
            "rec": rec, "sweep": sweep, "op": op, "rank1": rank1(rows),
            "noface": sum(not r["face_found"] for r in rows),
            "avg_ms": sum(timed) / len(timed),
        }
        sweeps[m] = (rec, sweep)
        print(f"rank-1 = {results[m]['rank1']:.3f}   best thr = {op['threshold']}   "
              f"acc = {op['accuracy']:.3f}  FAR = {op['FAR']:.3f}  FRR = {op['FRR']:.3f}   errors = {len(errors)}")

    plot(sweeps, os.path.join(cfg["results_dir"], "far_frr_plot.png"))
    write_summary(os.path.join(cfg["results_dir"], "summary.md"), items, results)


if __name__ == "__main__":
    main()
