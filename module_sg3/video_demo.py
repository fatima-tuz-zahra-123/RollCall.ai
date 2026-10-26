"""Local sampled CCTV demonstration: per-frame recognition, never tracking/attendance.

Example (run from the repository root)::

    python -m module_sg3.video_demo --input "../Class Dataset/classroom.mp4" \
        --output-dir private_runs/video --interval 30 --max-frames 12

The AVI is a time-compressed preview of sampled frames. JSONL timestamps and boxes
refer to the original video; no biometric images or templates leave this machine.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np


STATUS_COLORS = {
    "RECOGNIZED": (60, 220, 60),
    "UNKNOWN": (0, 190, 255),
    "QUALITY_REJECTED": (180, 180, 180),
    "ERROR": (70, 70, 255),
}


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _timing_stats(values):
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "total_ms": 0.0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "total_ms": float(a.sum()),
    }


def annotate_frame(frame, results, frame_id, timestamp, method):
    """Draw without changing the source frame or original-frame metadata."""
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.4, min(1.0, w / 1500.0))
    thickness = max(1, int(round(scale * 2)))
    banner_h = max(45, int(round(68 * scale)))
    cv2.rectangle(out, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.putText(out, "FRAME-LEVEL RECOGNITION | no tracking or attendance", (10, int(25 * scale) + 4),
                cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(out, f"{method.upper()} | source frame {frame_id} | t={timestamp:.3f}s | SAMPLED PREVIEW",
                (10, int(50 * scale) + 4), cv2.FONT_HERSHEY_SIMPLEX,
                scale * 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    for result in results:
        box = result.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x, y, bw, bh = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x, y, bw, bh)) or bw <= 0 or bh <= 0:
            continue
        x1, y1 = max(0, min(w - 1, round(x))), max(0, min(h - 1, round(y)))
        x2, y2 = max(0, min(w - 1, round(x + bw))), max(0, min(h - 1, round(y + bh)))
        status = result.get("status", "ERROR")
        color = STATUS_COLORS.get(status, STATUS_COLORS["ERROR"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = str(result.get("student_id") or status)
        score = result.get("score")
        if score is not None:
            label += f" {float(score):.3f}"
        if status == "QUALITY_REJECTED":
            label = f"QUALITY: {result.get('reason') or 'rejected'}"
        elif status == "ERROR":
            label = "RECOGNITION ERROR"
        text_scale = max(0.38, scale * 0.55)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
        label_y = max(banner_h + th + 2, y1 - 4)
        label_y = min(h - baseline - 2, label_y)
        cv2.rectangle(out, (x1, label_y - th - 2), (min(w - 1, x1 + tw + 4), label_y + baseline), (20, 20, 20), -1)
        cv2.putText(out, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale, color, 1, cv2.LINE_AA)
    # Some video encoders silently truncate an odd final row/column; pad instead.
    return cv2.copyMakeBorder(out, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def verify_video(path, expected_frames):
    """Decode every saved preview frame; a writer opening alone is insufficient."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Saved preview cannot be opened: {path}")
    count, size = 0, None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise RuntimeError("Saved preview contains an empty decoded frame")
            count += 1
            size = [int(frame.shape[1]), int(frame.shape[0])]
    finally:
        capture.release()
    if count != expected_frames:
        raise RuntimeError(f"Saved preview decoded {count} frames; expected {expected_frames}")
    return {"decoded_frames": count, "decoded_size": size, "verified": True}


def run_video(input_path, output_dir, cfg=None, *, method="sface", interval=30.0,
              start=0.0, duration=None, max_frames=12, max_side=1280,
              sg2_config_path=None, preview_fps=2.0, pipeline=None, service=None,
              overwrite=False):
    """Run a bounded local demonstration and return its JSON-safe summary.

    ``pipeline`` and ``service`` may be injected for tests or an existing loaded
    application. Their interfaces are YuNetSG2Pipeline.process and
    RecognitionService.recognize_face. A frame is sampled every ``interval``
    original-video seconds. The duration bound is exclusive at its endpoint.
    """
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Video input not found: {input_path}")
    if output_dir == input_path:
        raise ValueError("Output directory must not be the input file")
    interval = _number(interval, "interval", positive=True)
    start = _number(start, "start", nonnegative=True)
    preview_fps = _number(preview_fps, "preview_fps", positive=True)
    if duration is not None:
        duration = _number(duration, "duration", positive=True)
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")
    if method not in ("sface", "lbph"):
        raise ValueError("method must be sface or lbph")
    paths = {
        "preview": output_dir / "annotated_samples.avi",
        "frame_results": output_dir / "frame_results.jsonl",
        "summary": output_dir / "summary.json",
    }
    for path in paths.values():
        # resolve catches symlink aliases; samefile also catches hard links.
        if path.resolve() == input_path or (path.exists() and path.samefile(input_path)):
            raise ValueError("An output path aliases the original input video")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite output: {path}; select another output directory or --overwrite")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video input: {input_path}")
    writer = None
    frame_count = 0
    statuses = Counter()
    rejection_reasons = Counter()
    accepted = 0
    detected = 0
    frame_timestamps = []
    stage_times = {stage: [] for stage in ("decode_seek", "detection", "preprocessing", "recognition", "annotation", "write", "sample_total")}
    recognition_per_face = []
    started = time.perf_counter()
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid FPS; timestamps cannot be reconstructed safely")
        n_frames_raw = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        n_frames = int(n_frames_raw) if math.isfinite(n_frames_raw) and n_frames_raw > 0 else None
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_duration = n_frames / fps if n_frames is not None else None
        if source_duration is not None and start >= source_duration:
            raise ValueError(f"start ({start}s) is beyond video duration ({source_duration:.3f}s)")
        start_frame = int(round(start * fps))
        step_frames = max(1, int(round(interval * fps)))
        end_seconds = start + duration if duration is not None else None
        if cfg is None and (pipeline is None or service is None):
            from .sg3_core import load_config
            cfg = load_config()
        if pipeline is None:
            from .pipeline import YuNetSG2Pipeline
            pipeline = YuNetSG2Pipeline(cfg, sg2_config_path=sg2_config_path, max_side=max_side)
        if service is None:
            from .api import RecognitionService
            from .sg3_core import make_recognizer
            recognizer = make_recognizer(method, cfg)
            recognizer.load()
            service = RecognitionService(recognizer)
        initialization_ms = (time.perf_counter() - started) * 1000
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_start = time.perf_counter()
        with paths["frame_results"].open("w", encoding="utf-8", newline="\n") as log:
            for sample_id in range(max_frames):
                target_frame = start_frame + sample_id * step_frames
                if n_frames is not None and target_frame >= n_frames:
                    break
                if end_seconds is not None and target_frame / fps >= end_seconds:
                    break
                t_sample = time.perf_counter()
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame):
                    raise RuntimeError(f"Video backend could not seek to source frame {target_frame}")
                ok, frame = capture.read()
                decode_ms = (time.perf_counter() - t_sample) * 1000
                if not ok or frame is None or frame.size == 0:
                    if sample_id == 0 or (n_frames is not None and target_frame < n_frames):
                        raise RuntimeError(f"Video decode failed at source frame {target_frame}")
                    break
                pos = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
                actual_frame = int(round(pos)) - 1 if math.isfinite(pos) and pos > 0 else target_frame
                if abs(actual_frame - target_frame) > 1:
                    raise RuntimeError(f"Inaccurate video seek: requested {target_frame}, decoded {actual_frame}")
                timestamp = actual_frame / fps
                packet = pipeline.process(frame, frame_id=actual_frame, timestamp=timestamp)
                faces = packet.get("results", [])
                results = []
                t_recognition = time.perf_counter()
                for face in faces:
                    result = service.recognize_face(face)
                    results.append(result)
                    status = result.get("status", "ERROR")
                    statuses[status] += 1
                    if face.get("accept"):
                        accepted += 1
                        recognition_per_face.append(float(result.get("recognition_ms", 0.0)))
                    if status == "QUALITY_REJECTED":
                        rejection_reasons[str(result.get("reason") or "unspecified")] += 1
                recognition_ms = (time.perf_counter() - t_recognition) * 1000
                detected += len(faces)
                t_annotation = time.perf_counter()
                annotated = annotate_frame(frame, results, actual_frame, timestamp, method)
                annotation_ms = (time.perf_counter() - t_annotation) * 1000
                if writer is None:
                    output_size = (annotated.shape[1], annotated.shape[0])
                    writer = cv2.VideoWriter(str(paths["preview"]), cv2.VideoWriter_fourcc(*"MJPG"), preview_fps, output_size)
                    if not writer.isOpened():
                        raise RuntimeError("VideoWriter could not open MJPG AVI output")
                elif (annotated.shape[1], annotated.shape[0]) != output_size:
                    raise RuntimeError("Video frame size changed during processing")
                t_write = time.perf_counter()
                writer.write(annotated)
                write_ms = (time.perf_counter() - t_write) * 1000
                timing = {
                    "decode_seek": decode_ms,
                    "detection": float(packet.get("detection_ms", 0.0)),
                    "preprocessing": float(packet.get("preprocessing_ms", 0.0)),
                    "recognition": recognition_ms,
                    "annotation": annotation_ms,
                    "write": write_ms,
                    "sample_total": (time.perf_counter() - t_sample) * 1000,
                }
                record = {
                    "sample_index": frame_count,
                    "frame_id": actual_frame,
                    "timestamp": timestamp,
                    "bbox_format": "xywh_original_frame_pixels",
                    "mode": "frame_level_recognition",
                    "detected_faces": len(faces),
                    "sg2_accepted_faces": sum(bool(f.get("accept")) for f in faces),
                    "results": results,
                    "timing_ms": timing,
                }
                if packet.get("pipeline_error"):
                    record["pipeline_error"] = packet["pipeline_error"]
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                for stage, elapsed in timing.items():
                    stage_times[stage].append(elapsed)
                frame_count += 1
                frame_timestamps.append(timestamp)
        processing_seconds = time.perf_counter() - processed_start
        if frame_count == 0:
            raise RuntimeError("No sampled frames were decoded; no demonstration was produced")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    verification = verify_video(paths["preview"], frame_count)
    # Re-read saved records to verify complete JSON serialization and frame count.
    with paths["frame_results"].open(encoding="utf-8") as log:
        logged_frames = sum(1 for line in log if json.loads(line))
    if logged_frames != frame_count:
        raise RuntimeError("Saved JSONL frame count differs from preview frame count")
    summary = {
        "mode": "frame_level_recognition",
        "tracking": False,
        "attendance_records": False,
        "accuracy": None,
        "accuracy_note": "No CCTV identity ground truth supplied; counts and runtime are observations, not accuracy.",
        "detector": "SG3 temporary YuNet all-faces adapter; SG1 pipeline is not claimed integrated",
        "preprocessing": "actual module_sg2 quality/alignment implementation",
        "method": method,
        "source": {"filename": input_path.name, "fps": fps, "frame_count": n_frames,
                   "width": source_width, "height": source_height, "duration_seconds": source_duration},
        "sampling": {"requested_start_seconds": start, "requested_duration_seconds": duration,
                     "requested_interval_seconds": interval, "effective_interval_seconds": step_frames / fps,
                     "maximum_sampled_frames": max_frames, "sampled_frames": frame_count,
                     "first_timestamp": frame_timestamps[0], "last_timestamp": frame_timestamps[-1]},
        "preview": {"filename": paths["preview"].name, "codec": "MJPG", "fps": preview_fps,
                    "duration_seconds": frame_count / preview_fps,
                    "time_compressed_sampled_preview": True, **verification},
        "counts": {"detected_faces": detected, "sg2_accepted_faces": accepted,
                   "sg2_rejected_faces": detected - accepted,
                   "statuses": {key: statuses.get(key, 0) for key in STATUS_COLORS},
                   "quality_rejection_reasons": dict(rejection_reasons)},
        "timing": {"initialization_ms": initialization_ms, "processing_wall_seconds": processing_seconds,
                   "sampled_frames_per_wall_second": frame_count / processing_seconds,
                   "detected_faces_per_wall_second": detected / processing_seconds,
                   "stages": {stage: _timing_stats(values) for stage, values in stage_times.items()},
                   "recognition_per_sg2_accepted_face": _timing_stats(recognition_per_face),
                   "note": "CPU wall-clock timings on this PC include seeks/decoding and cold first inference; not Jetson benchmarks."},
        "runtime": {"python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__,
                    "platform": platform.platform(), "processor": platform.processor(),
                    "opencv_threads": cv2.getNumThreads()},
        "files": {key: path.name for key, path in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    json.loads(paths["summary"].read_text(encoding="utf-8"))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="local original CCTV video (never uploaded)")
    parser.add_argument("--output-dir", required=True, help="private local directory for annotated preview and logs")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--method", choices=("sface", "lbph"), default="sface")
    parser.add_argument("--interval", type=float, default=30.0, help="original-video seconds between sampled frames")
    parser.add_argument("--start", type=float, default=0.0, help="start time in original-video seconds")
    parser.add_argument("--duration", type=float, help="exclusive source-window duration in seconds")
    parser.add_argument("--max-frames", type=int, default=12, help="maximum number of sampled frames")
    parser.add_argument("--max-side", type=int, default=1280, help="YuNet inference image maximum side; coordinates remain original")
    parser.add_argument("--sg2-config", help="actual SG2 preprocessing configuration JSON")
    parser.add_argument("--preview-fps", type=float, default=2.0, help="playback FPS of time-compressed sampled preview")
    parser.add_argument("--overwrite", action="store_true", help="replace previous generated outputs (never the source video)")
    args = parser.parse_args(argv)
    from .sg3_core import load_config
    try:
        summary = run_video(args.input, args.output_dir, load_config(args.config), method=args.method,
                            interval=args.interval, start=args.start, duration=args.duration,
                            max_frames=args.max_frames, max_side=args.max_side,
                            sg2_config_path=args.sg2_config, preview_fps=args.preview_fps,
                            overwrite=args.overwrite)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f"video_demo: {exc}\n")
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()),
                      "sampled_frames": summary["sampling"]["sampled_frames"],
                      "counts": summary["counts"], "saved_preview_verified": summary["preview"]["verified"]}, indent=2))


if __name__ == "__main__":
    main()
