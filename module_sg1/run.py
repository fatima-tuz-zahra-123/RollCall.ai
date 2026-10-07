from __future__ import annotations

import argparse
import sys
from importlib.metadata import distribution
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
DATA_YAML = DATA_DIR / "widerface.yaml"
MODEL_PATH = BASE_DIR / "weights" / "yolov12n-face.pt"
DEFAULT_IMAGE = BASE_DIR / "images (1).jpg"
ANNOTATIONS_DIR = DATA_DIR / "wider_face_split" / "wider_face_split"


def load_yolo():
    site_packages = str(distribution("ultralytics").locate_file(""))
    sys.path.insert(0, site_packages)
    try:
        from ultralytics import YOLO

        return YOLO
    finally:
        sys.path.remove(site_packages)


def next_non_empty_line(lines, index):
    while index < len(lines) and not lines[index].strip():
        index += 1

    if index >= len(lines):
        raise ValueError("Unexpected end of WIDER annotation file.")

    return lines[index].strip(), index + 1


def convert_wider_split(split):
    annotation_file = ANNOTATIONS_DIR / f"wider_face_{split}_bbx_gt.txt"
    images_dir = DATA_DIR / f"WIDER_{split}" / "images"
    labels_dir = DATA_DIR / f"WIDER_{split}" / "labels"

    if not annotation_file.is_file():
        raise FileNotFoundError(f"Missing WIDER annotations: {annotation_file}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing WIDER image folder: {images_dir}")

    lines = annotation_file.read_text(encoding="utf-8").splitlines()
    index = 0
    image_count = 0
    valid_face_count = 0

    while index < len(lines):
        image_name, index = next_non_empty_line(lines, index)
        face_count_text, index = next_non_empty_line(lines, index)

        try:
            face_count = int(face_count_text)
        except ValueError as exc:
            raise ValueError(
                f"Expected face count after {image_name!r}, got {face_count_text!r}."
            ) from exc

        if face_count == 0 and index < len(lines):
            empty_line = lines[index].split()
            if len(empty_line) == 10 and all(value == "0" for value in empty_line):
                index += 1

        image_rel_path = Path(image_name.replace("/", "\\"))
        image_path = images_dir / image_rel_path

        if not image_path.is_file():
            raise FileNotFoundError(f"Image listed in annotations is missing: {image_path}")

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        yolo_boxes = []

        for _ in range(face_count):
            box_text, index = next_non_empty_line(lines, index)
            values = box_text.split()

            if len(values) < 8:
                raise ValueError(f"Malformed face annotation for {image_name!r}: {box_text!r}")

            x, y, width, height = map(float, values[:4])
            is_invalid = int(values[7])

            if is_invalid or width <= 0 or height <= 0:
                continue

            x1 = min(max(x, 0.0), float(image_width))
            y1 = min(max(y, 0.0), float(image_height))
            x2 = min(max(x + width, 0.0), float(image_width))
            y2 = min(max(y + height, 0.0), float(image_height))

            clipped_width = x2 - x1
            clipped_height = y2 - y1

            if clipped_width <= 0 or clipped_height <= 0:
                continue

            center_x = (x1 + x2) / (2 * image_width)
            center_y = (y1 + y2) / (2 * image_height)
            normalized_width = clipped_width / image_width
            normalized_height = clipped_height / image_height

            yolo_boxes.append(
                f"0 {center_x:.6f} {center_y:.6f} "
                f"{normalized_width:.6f} {normalized_height:.6f}"
            )

        label_path = labels_dir / image_rel_path.with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            "\n".join(yolo_boxes) + ("\n" if yolo_boxes else ""),
            encoding="utf-8",
        )

        image_count += 1
        valid_face_count += len(yolo_boxes)

    return image_count, valid_face_count


def prepare_dataset():
    for split in ("train", "val"):
        image_count, face_count = convert_wider_split(split)
        print(f"WIDER_{split}: {image_count} images, {face_count} valid faces")


def build_parser():
    parser = argparse.ArgumentParser(description="Train, validate, or run YOLOv12n-face.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", help="Convert WIDER annotations to YOLO labels.")

    train = subparsers.add_parser("train", help="Train the face detector.")
    train.add_argument("--model", type=Path, default=MODEL_PATH)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device", default=None)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--name", default="face")

    val = subparsers.add_parser("val", help="Validate the model on the WIDER validation set.")
    val.add_argument("--model", type=Path, default=MODEL_PATH)
    val.add_argument("--imgsz", type=int, default=640)
    val.add_argument("--device", default=None)

    predict = subparsers.add_parser("predict", help="Run face detection on an image or media source.")
    predict.add_argument("--model", type=Path, default=MODEL_PATH)
    predict.add_argument("--source", type=Path, default=DEFAULT_IMAGE)
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--imgsz", type=int, default=1280)
    predict.add_argument("--device", default=None)
    predict.add_argument("--name", default="face")

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "prepare":
        prepare_dataset()
        return

    if not args.model.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    YOLO = load_yolo()
    model = YOLO(str(args.model))

    if args.command == "train":
        model.train(
            data=str(DATA_YAML),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(BASE_DIR / "runs" / "train"),
            name=args.name,
        )
    elif args.command == "val":
        model.val(
            data=str(DATA_YAML),
            split="val",
            imgsz=args.imgsz,
            device=args.device,
            project=str(BASE_DIR / "runs" / "val"),
        )
    elif args.command == "predict":
        model.predict(
            source=str(args.source),
            conf=args.conf,
            imgsz=args.imgsz,
            max_det=1000,
            device=args.device,
            save=True,
            show_labels=False,
            show_conf=False,
            project=str(BASE_DIR / "runs" / "predict"),
            name=args.name,
            exist_ok=True,
        )


if __name__ == "__main__":
    main()
