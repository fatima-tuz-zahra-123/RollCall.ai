"""Download the models + test data SG-2 needs. Nothing here is committed (see module_sg2/.gitignore).

    python module_sg2/scripts/download_assets.py            # models + LFW (~250 MB total)
    python module_sg2/scripts/download_assets.py --models   # models only (~46 MB)
"""
import hashlib, sys, tarfile, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # module_sg2/
MODELS, DATA = ROOT / "models", ROOT / "data"
HF = "https://huggingface.co/opencv/{}/resolve/main/{}"

ASSETS = [  # (folder, filename, url, sha256 or None)
    (MODELS, "face_detection_yunet_2023mar.onnx", HF.format("face_detection_yunet", "face_detection_yunet_2023mar.onnx"), None),
    (MODELS, "face_recognition_sface_2021dec.onnx", HF.format("face_recognition_sface", "face_recognition_sface_2021dec.onnx"), None),
    (MODELS, "ediffiqa_tiny_jun2024.onnx", HF.format("face_image_quality_assessment_ediffiqa", "ediffiqa_tiny_jun2024.onnx"), None),
    # LFW: same files + checksums scikit-learn's fetch_lfw_people uses
    (DATA, "pairs.txt", "https://ndownloader.figshare.com/files/5976006",
     "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"),
    (DATA, "lfw.tgz", "https://ndownloader.figshare.com/files/5976018",
     "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0"),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(folder, name, url, digest):
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / name
    if not dst.exists():
        print("downloading", name)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req) as r, open(dst.with_suffix(".part"), "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except urllib.error.HTTPError as e:
            # figshare blocks scripted downloads (403) -> download in a browser, then re-run to verify + extract
            dst.with_suffix(".part").unlink(missing_ok=True)
            print(f"  HTTP {e.code}: open {url} in a browser, save it as {dst}, then re-run this script")
            return None
        dst.with_suffix(".part").rename(dst)
    if digest and sha256(dst) != digest:
        dst.unlink()
        sys.exit(f"checksum mismatch for {name} - deleted, re-run")
    print("ok", name, f"{dst.stat().st_size / 1e6:.1f} MB")
    return dst


if __name__ == "__main__":
    models_only = "--models" in sys.argv
    for a in ASSETS:
        if models_only and a[0] is DATA:
            continue
        p = fetch(*a)
        if p and p.name == "lfw.tgz" and not (DATA / "lfw").exists():
            print("extracting lfw.tgz")
            with tarfile.open(p) as t:
                t.extractall(DATA, filter="data")
