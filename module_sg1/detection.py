import csv
import os
import time

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_opencv")
CSV_PATH = os.path.join(BASE_DIR, "opencv_yunet_results.csv")

CONFIDENCE_THRESHOLD = 0.5
MIN_FACE_SIZE = 8


def build_detector():
    return cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (320, 320),
        CONFIDENCE_THRESHOLD,
    )


def get_image_files():
    return sorted(
        name for name in os.listdir(IMAGE_DIR)
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )


def detect_faces(image, detector):
    height, width = image.shape[:2]
    detector.setInputSize((width, height))

    start = time.perf_counter()
    _, detections = detector.detect(image)
    elapsed_ms = (time.perf_counter() - start) * 1000

    annotated = image.copy()
    rows = []
    face_count = 0

    if detections is not None:
        for detection in detections:
            x, y, w, h = detection[:4]
            score = float(detection[14])

            x1 = max(int(round(x)), 0)
            y1 = max(int(round(y)), 0)
            x2 = min(int(round(x + w)), width)
            y2 = min(int(round(y + h)), height)
            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < MIN_FACE_SIZE or box_h < MIN_FACE_SIZE:
                continue

            face_count += 1
            rows.append([x1, y1, box_w, box_h, round(score, 4)])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{score:.2f}",
                (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    return annotated, rows, face_count, elapsed_ms


def save_results(rows):
    with open(CSV_PATH, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image", "x", "y", "w", "h", "confidence"])
        writer.writerows(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detector = build_detector()
    image_names = get_image_files()

    results = []
    timings = []

    for index, image_name in enumerate(image_names):
        image_path = os.path.join(IMAGE_DIR, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read {image_name}, skipping.")
            continue

        annotated, detections, face_count, elapsed_ms = detect_faces(image, detector)

        if index > 0:
            timings.append(elapsed_ms)

        for x, y, w, h, score in detections:
            results.append([image_name, x, y, w, h, score])

        output_path = os.path.join(OUTPUT_DIR, image_name)
        cv2.imwrite(output_path, annotated)
        print(f"{image_name}: {face_count} face(s), {elapsed_ms:.1f} ms")

    save_results(results)

    if timings:
        average_time = sum(timings) / len(timings)
        print(f"\nImages processed: {len(image_names)}")
        print(f"Total faces detected: {len(results)}")
        print(f"Average time: {average_time:.1f} ms per image ({1000 / average_time:.1f} FPS)")


if __name__ == "__main__":
    main()