"""Download public OpenCV Zoo models with content pinned by upstream SHA-256.

Only weights are downloaded; student data is never read or transmitted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

try:
    from .sg3_core import load_config
except ImportError:
    from sg3_core import load_config


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download(cfg):
    manifest = json.loads(Path(__file__).with_name("models_manifest.json").read_text())
    destination = Path(cfg["models_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    for name, entry in manifest.items():
        target = destination / name
        if target.is_file() and sha256(target) == entry["sha256"]:
            print(f"[verified] {name}")
            continue
        temporary = target.with_suffix(".onnx.part")
        try:
            print(f"[download] {name}", flush=True)
            with urllib.request.urlopen(entry["url"], timeout=60) as response, temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if temporary.stat().st_size != entry["size"] or sha256(temporary) != entry["sha256"]:
                raise ValueError(f"Model integrity check failed: {name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"[verified] {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    download(load_config(args.config))


if __name__ == "__main__":
    main()
