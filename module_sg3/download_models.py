"""Downloads the two ONNX models (YuNet detector + SFace recognizer) into ./models.
Run once:   python download_models.py
(If the download is blocked, open the URLs in a browser and save the files into module_sg3/models/.)"""

import os
import urllib.request

from sg3_core import load_config

BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
FILES = {
    "face_detection_yunet_2023mar.onnx": BASE + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": BASE + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def main():
    cfg = load_config()
    os.makedirs(cfg["models_dir"], exist_ok=True)
    for name, url in FILES.items():
        dst = os.path.join(cfg["models_dir"], name)
        if os.path.isfile(dst) and os.path.getsize(dst) > 100_000:
            print(f"[ok] already have {name}")
            continue
        print(f"[..] downloading {name}")
        urllib.request.urlretrieve(url, dst)
        size = os.path.getsize(dst)
        if size < 100_000:
            os.remove(dst)
            raise RuntimeError(f"Download of {name} failed (got {size} bytes). Download manually:\n{url}")
        print(f"[ok] {name}  ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
