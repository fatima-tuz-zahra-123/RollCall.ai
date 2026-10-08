#!/usr/bin/env python3
"""
Face enrolment + identification demo on the LFW dataset (CS-477, SG4 stand-in).

Pipeline
    LFW photos -> MTCNN (find faces) -> InceptionResnetV1 (face -> 512-d embedding)
    -> average per person = fingerprint -> cosine similarity -> ID or UNKNOWN

Steps (all figures and tables are saved to --out-dir):
    1. show known / unknown images
    2. embedding demo + embeddings of all enrolment photos
    3. fingerprints (gallery) + fingerprint similarity matrix
    4. single-image tests (known and unknown people)
    5. multi-person image tests (collage with ground truth)
    6. threshold sweep (CSV) over held-out photos

Usage
    python face_id_lfw.py                       # default settings, saves to results/
    python face_id_lfw.py --threshold 0.7 --n-enrolled 8 --seed 1
    python face_id_lfw.py --show                # also open plot windows
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# --------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-enrolled", type=int, default=6, help="known people to enrol")
    p.add_argument("--n-unknown", type=int, default=3, help="people never enrolled")
    p.add_argument("--enroll-imgs", type=int, default=5, help="photos per known person for enrolment")
    p.add_argument("--threshold", type=float, default=0.60,
                   help="cosine similarity; below this -> UNKNOWN")
    p.add_argument("--seed", type=int, default=0, help="random seed (which people are picked)")
    p.add_argument("--out-dir", default="results", help="folder for figures / tables")
    p.add_argument("--show", action="store_true", help="also display plots interactively")
    return p.parse_args()


# ------------------------------------------------------------------- models
class FaceEngine:
    """MTCNN detector + InceptionResnetV1 embedder."""

    def __init__(self):
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=True, device=self.device, min_face_size=20)
        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        print("Device:", self.device)

    def _embed(self, face_tensors):
        with self.torch.no_grad():
            e = self.embedder(face_tensors.to(self.device))
        return self.torch.nn.functional.normalize(e, p=2, dim=1).cpu().numpy()

    def detect_and_embed(self, img):
        """image -> (boxes N x 4, det probs N, embeddings N x 512) or (None, None, None)"""
        boxes, probs = self.mtcnn.detect(img)
        if boxes is None:
            return None, None, None
        faces = self.mtcnn.extract(img, boxes, None)          # N x 3 x 160 x 160
        return boxes, probs, self._embed(faces)

    def largest_face(self, img):
        """box and embedding of the biggest face, or (None, None)"""
        boxes, _, e = self.detect_and_embed(img)
        if boxes is None:
            return None, None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        return boxes[i], e[i]


# --------------------------------------------------------------------- data
def load_lfw(n_enrolled, n_unknown, enroll_imgs, seed):
    """Download LFW (people with >=10 photos) and split into known / unknown."""
    from sklearn.datasets import fetch_lfw_people
    lfw = fetch_lfw_people(min_faces_per_person=10, resize=1.0, color=True,
                           slice_=(slice(0, 250), slice(0, 250)))
    imgs = lfw.images
    if imgs.max() <= 1.5:                       # some versions return 0-1 floats
        imgs = imgs * 255.0
    imgs = np.clip(imgs, 0, 255).astype("uint8")
    print(f"LFW images used: {imgs.shape[0]} | people: {len(lfw.target_names)}")

    by_person = {}
    for im, t in zip(imgs, lfw.target):
        by_person.setdefault(lfw.target_names[t], []).append(Image.fromarray(im))

    rng = random.Random(seed)
    people = sorted(by_person)
    rng.shuffle(people)
    enrolled = people[:n_enrolled]
    unknown = people[n_enrolled:n_enrolled + n_unknown]
    return {
        "enrolled": enrolled,
        "unknown": unknown,
        "known_ids": {n: f"S{i:03d}" for i, n in enumerate(enrolled, 1)},
        "enroll_set": {n: by_person[n][:enroll_imgs] for n in enrolled},
        "test_set": {n: by_person[n][enroll_imgs:] for n in enrolled},   # held out
        "unknown_set": {n: by_person[n] for n in unknown},
        "n_total": int(imgs.shape[0]),
        "n_people": len(lfw.target_names),
    }


# ------------------------------------------------------------------ gallery
class Gallery:
    def __init__(self, ids, names, mat):
        self.ids, self.names, self.mat = ids, names, mat

    def scores(self, emb):
        return self.mat @ emb

    def identify(self, emb, threshold):
        s = self.scores(emb)
        j = int(np.argmax(s))
        known = s[j] >= threshold
        return (self.ids[j] if known else "UNKNOWN",
                self.names[j] if known else "UNKNOWN",
                float(s[j]), j)


def expected_label(true_name, enrolled):
    return true_name if true_name in enrolled else "UNKNOWN"


# ------------------------------------------------------------------ figures
class Saver:
    def __init__(self, out_dir, show):
        self.out_dir, self.show = out_dir, show
        os.makedirs(out_dir, exist_ok=True)

    def __call__(self, fig, name):
        path = os.path.join(self.out_dir, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        if self.show:
            plt.show()
        plt.close(fig)
        print("  saved", path)


def fig_rows(names, source, n_imgs, title, label_fn):
    fig, axes = plt.subplots(len(names), n_imgs, figsize=(2.3 * n_imgs, 2.5 * len(names)))
    axes = np.array(axes).reshape(len(names), n_imgs)
    for r, name in enumerate(names):
        for c in range(n_imgs):
            axes[r, c].axis("off")
            if c < len(source[name]):
                axes[r, c].imshow(source[name][c])
        axes[r, 0].set_title(label_fn(name), fontsize=10, loc="left", fontweight="bold")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def step1_images(data, save):
    print("\nSTEP 1: known / unknown images")
    ids = data["known_ids"]
    save(fig_rows(data["enrolled"], data["enroll_set"], len(next(iter(data["enroll_set"].values()))),
                  "KNOWN people - photos used for ENROLMENT",
                  lambda n: f"{ids[n]}  {n}"), "01_known_images.png")
    save(fig_rows(data["unknown"], data["unknown_set"], 5,
                  "UNKNOWN people - never enrolled",
                  lambda n: f"UNKNOWN  {n}"), "02_unknown_images.png")


def step2_embeddings(engine, data, save):
    print("\nSTEP 2: embeddings")
    name = data["enrolled"][0]
    demo = data["enroll_set"][name][0]
    box, emb = engine.largest_face(demo)
    if box is None:
        raise RuntimeError("No face found in the demo image; try another --seed")

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1, 1, 3]})
    d = demo.copy()
    ImageDraw.Draw(d).rectangle(box.tolist(), outline=(0, 200, 0), width=3)
    ax[0].imshow(d); ax[0].axis("off"); ax[0].set_title("1. MTCNN finds the face")
    ax[1].imshow(demo.crop(tuple(box))); ax[1].axis("off"); ax[1].set_title("2. cropped face")
    ax[2].bar(range(512), emb, width=1.0)
    ax[2].set_title("3. embedding = 512 numbers"); ax[2].set_xlabel("dimension")
    fig.tight_layout()
    save(fig, "03_embedding_demo.png")

    print("  embedding shape :", emb.shape)
    print("  first 10 numbers:", np.round(emb[:10], 4))
    print("  vector length   :", round(float(np.linalg.norm(emb)), 4))

    _, e1 = engine.largest_face(data["enroll_set"][name][0])
    _, e2 = engine.largest_face(data["enroll_set"][name][1])
    _, e3 = engine.largest_face(data["unknown_set"][data["unknown"][0]][0])
    sim = {"same_person": float(e1 @ e2), "different_people": float(e1 @ e3)}
    print(f"  same person, 2 photos : {sim['same_person']:.3f} (should be high)")
    print(f"  different people      : {sim['different_people']:.3f} (should be low)")

    embs_by_person = {}
    for n in data["enrolled"]:
        es = [engine.largest_face(im)[1] for im in data["enroll_set"][n]]
        es = [e for e in es if e is not None]
        if not es:
            raise RuntimeError(f"No usable enrolment photo for {n}; try another --seed")
        embs_by_person[n] = np.stack(es)
        print(f"  {data['known_ids'][n]} {n:22s} -> {embs_by_person[n].shape}")

    allv = np.concatenate([embs_by_person[n] for n in data["enrolled"]])
    fig = plt.figure(figsize=(14, 4))
    plt.imshow(allv, aspect="auto", cmap="coolwarm"); plt.colorbar(label="value")
    plt.title("All enrolment embeddings (row = photo, grouped by person)")
    plt.xlabel("512 dimensions"); plt.ylabel("photo")
    save(fig, "04_enrolment_embeddings.png")
    return embs_by_person, sim


def step3_gallery(data, embs_by_person, save, out_dir):
    print("\nSTEP 3: fingerprints / gallery")
    ids, names, rows = [], [], []
    for n in data["enrolled"]:
        fp = embs_by_person[n].mean(axis=0)
        fp = fp / np.linalg.norm(fp)
        ids.append(data["known_ids"][n]); names.append(n); rows.append(fp)
        print(f"  {ids[-1]}  {n:24s} photos={len(embs_by_person[n])}  "
              f"first 6: {np.round(fp[:6], 3)}")
    mat = np.stack(rows)

    fig, ax = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [2, 1]})
    im0 = ax[0].imshow(mat, aspect="auto", cmap="coolwarm")
    ax[0].set_yticks(range(len(ids)))
    ax[0].set_yticklabels([f"{i} {n[:12]}" for i, n in zip(ids, names)])
    ax[0].set_title("Fingerprints (one row per known person)"); plt.colorbar(im0, ax=ax[0])
    S = mat @ mat.T
    ax[1].imshow(S, vmin=0, vmax=1, cmap="viridis")
    ax[1].set_xticks(range(len(ids))); ax[1].set_yticks(range(len(ids)))
    ax[1].set_xticklabels(ids); ax[1].set_yticklabels(ids)
    for a in range(len(ids)):
        for b in range(len(ids)):
            ax[1].text(b, a, f"{S[a, b]:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax[1].set_title("Similarity between fingerprints")
    fig.tight_layout()
    save(fig, "05_fingerprints.png")

    np.savez(os.path.join(out_dir, "gallery.npz"), ids=ids, names=names, embs=mat)
    with open(os.path.join(out_dir, "id_table.json"), "w") as f:
        json.dump(dict(zip(ids, names)), f, indent=2)
    return Gallery(ids, names, mat)


def step4_single_tests(engine, data, gal, thr, save):
    print("\nSTEP 4: single-image tests")
    results = []

    def test_image(img, truth, fname):
        box, emb = engine.largest_face(img)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.9),
                                     gridspec_kw={"width_ratios": [1, 1.6]})
        if box is None:
            a1.imshow(img); a1.axis("off"); a2.axis("off")
            a1.set_title(f"truth: {truth}\nNO FACE DETECTED", color="red")
            save(fig, fname)
            results.append({"truth": truth, "pred": "NO FACE", "score": None, "correct": False})
            return
        pid, pname, best, j = gal.identify(emb, thr)
        sims = gal.scores(emb)
        ok = pname == expected_label(truth, data["enrolled"])
        color = (0, 200, 0) if pid != "UNKNOWN" else (220, 0, 0)
        d = img.copy(); dr = ImageDraw.Draw(d)
        dr.rectangle(box.tolist(), outline=color, width=3)
        dr.text((box[0], max(0, box[1] - 14)), f"{pid} {pname[:12]} ({best:.2f})", fill=color)
        a1.imshow(d); a1.axis("off"); a1.set_title(f"truth: {truth}", fontsize=10)
        a2.barh([f"{i} {n[:14]}" for i, n in zip(gal.ids, gal.names)], sims,
                color=["tab:green" if k == j else "lightgray" for k in range(len(sims))])
        a2.axvline(thr, color="red", linestyle="--", label=f"threshold {thr}")
        a2.set_xlim(-0.2, 1.0); a2.legend(loc="lower right")
        a2.set_xlabel("similarity to each fingerprint")
        a2.set_title(f"Output: {pid} {pname if pid != 'UNKNOWN' else ''} -> "
                     f"{'CORRECT' if ok else 'WRONG'}",
                     color=("green" if ok else "red"), fontsize=12)
        fig.tight_layout()
        save(fig, fname)
        print(f"  truth={truth} | best={gal.ids[j]} {gal.names[j]} ({best:.3f}) "
              f"| decision={pid} | {'CORRECT' if ok else 'WRONG'}")
        results.append({"truth": truth, "pred": pname, "score": best, "correct": bool(ok)})

    for k, n in enumerate(data["enrolled"][:3], 1):
        test_image(data["test_set"][n][0], n, f"06_test_known_{k}.png")
    for k, n in enumerate(data["unknown"][:3], 1):
        test_image(data["unknown_set"][n][0], n, f"07_test_unknown_{k}.png")
    return results


def make_collage(items, cols, size=250):
    rows = int(np.ceil(len(items) / cols))
    canvas = Image.new("RGB", (cols * size, rows * size), (255, 255, 255))
    for k, (im, _) in enumerate(items):
        canvas.paste(im.resize((size, size)), ((k % cols) * size, (k // cols) * size))
    return canvas


def step5_multi(engine, data, gal, thr, save, seed):
    print("\nSTEP 5: multi-person image (collage with ground truth)")
    rng = random.Random(seed)
    items = [(data["test_set"][n][1], n) for n in data["enrolled"][:4]]
    items += [(data["unknown_set"][n][1], n) for n in data["unknown"][:2]]
    rng.shuffle(items)
    cols, size = 3, 250
    collage = make_collage(items, cols, size)

    boxes, probs, embs = engine.detect_and_embed(collage)
    out = collage.copy(); dr = ImageDraw.Draw(out)
    cell_pred = {}
    if boxes is not None:
        for b, e in zip(boxes, embs):
            pid, pname, score, _ = gal.identify(e, thr)
            x1, y1, x2, y2 = b
            color = (0, 200, 0) if pid != "UNKNOWN" else (220, 0, 0)
            dr.rectangle([x1, y1, x2, y2], outline=color, width=3)
            dr.text((x1, max(0, y1 - 14)), f"{pid} {pname[:12]} {score:.2f}", fill=color)
            k = int((y1 + y2) / 2 // size) * cols + int((x1 + x2) / 2 // size)
            cell_pred[k] = (pname, score)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    ax[0].imshow(collage); ax[0].axis("off"); ax[0].set_title(f"INPUT: {len(items)} people")
    ax[1].imshow(out); ax[1].axis("off"); ax[1].set_title("OUTPUT: every face identified")
    fig.tight_layout()
    save(fig, "08_multi_face.png")

    correct = 0
    for k, (_, truth) in enumerate(items):
        pname = cell_pred.get(k, ("NO FACE", 0))[0]
        ok = pname == expected_label(truth, data["enrolled"]); correct += ok
        print(f"  face {k + 1}: truth={truth:22s} -> {pname:22s} {'OK' if ok else 'WRONG'}")
    n_det = 0 if boxes is None else len(boxes)
    print(f"  faces detected: {n_det}/{len(items)} | correct: {correct}/{len(items)}")
    return {"faces": len(items), "detected": n_det, "correct": int(correct)}


def step6_sweep(engine, data, gal, out_dir, max_per_person=10):
    print("\nSTEP 6: threshold sweep over held-out photos")
    records = []                                   # (truth, best_idx|None, best_score)
    for n in data["enrolled"]:
        for im in data["test_set"][n][:max_per_person]:
            _, e = engine.largest_face(im)
            records.append((n, None, -1.0) if e is None
                           else (n, int(np.argmax(gal.scores(e))), float(gal.scores(e).max())))
    for n in data["unknown"]:
        for im in data["unknown_set"][n][:max_per_person]:
            _, e = engine.largest_face(im)
            records.append(("UNKNOWN", None, -1.0) if e is None
                           else ("UNKNOWN", int(np.argmax(gal.scores(e))), float(gal.scores(e).max())))

    kn = [r for r in records if r[0] != "UNKNOWN"]
    un = [r for r in records if r[0] == "UNKNOWN"]
    rows = []
    print(f"  known photos: {len(kn)} | unknown photos: {len(un)}")
    print(f"  {'thr':>4} {'known_acc':>10} {'false_rej':>10} {'wrong_id':>9} "
          f"{'unk_rejected':>13} {'false_acc':>10}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        correct = sum(1 for t, j, s in kn if j is not None and s >= thr and gal.names[j] == t)
        f_rej = sum(1 for t, j, s in kn if j is None or s < thr)
        wrong = len(kn) - correct - f_rej
        rejected = sum(1 for t, j, s in un if j is None or s < thr)
        row = {"threshold": thr,
               "known_accuracy": correct / len(kn), "false_reject": f_rej / len(kn),
               "wrong_id": wrong / len(kn),
               "unknown_rejected": rejected / len(un),
               "false_accept": (len(un) - rejected) / len(un)}
        rows.append(row)
        print(f"  {thr:4.1f} {row['known_accuracy']:10.2%} {row['false_reject']:10.2%} "
              f"{row['wrong_id']:9.2%} {row['unknown_rejected']:13.2%} {row['false_accept']:10.2%}")
    path = os.path.join(out_dir, "threshold_sweep.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("  saved", path)
    return rows


# --------------------------------------------------------------------- main
def main():
    args = parse_args()
    if not args.show:
        plt.switch_backend("Agg")
    random.seed(args.seed); np.random.seed(args.seed)

    engine = FaceEngine()
    data = load_lfw(args.n_enrolled, args.n_unknown, args.enroll_imgs, args.seed)
    print("\nKNOWN :", [(data["known_ids"][n], n) for n in data["enrolled"]])
    print("UNKNOWN:", data["unknown"])

    save = Saver(args.out_dir, args.show)
    step1_images(data, save)
    embs_by_person, sim = step2_embeddings(engine, data, save)
    gal = step3_gallery(data, embs_by_person, save, args.out_dir)
    single = step4_single_tests(engine, data, gal, args.threshold, save)
    multi = step5_multi(engine, data, gal, args.threshold, save, args.seed)
    sweep = step6_sweep(engine, data, gal, args.out_dir)

    metrics = {"settings": vars(args), "lfw_images": data["n_total"],
               "lfw_people": data["n_people"], "similarity_demo": sim,
               "single_image_tests": single, "multi_face_test": multi,
               "threshold_sweep": sweep}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nDone. Outputs are in '{args.out_dir}/'")


if __name__ == "__main__":
    main()
