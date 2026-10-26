# Pre-implementation technical assessment — 26 October 2026

This is a historical assessment of the initial ZIP snapshot, before the SG3
changes. The gaps below describe that starting point, not the current module.
See the [current README](README.md) and [class enrollment report](results/CLASS_ENROLLMENT.md)
for the implemented API, tests, reproducible commands and recorded results.

The supplied ZIP was inspected and extracted without changing original sources. It is a source snapshot, not a Git checkout. All Python files, READMEs, interface/configuration files and existing result files were inventoried. `integration/`, SG5 and `evaluation/` contain placeholders.

## Reusable implementation

- SG3 already implements YuNet largest-face detection, SFace 128-dimensional L2-normalized embeddings with multi-template maximum-cosine matching and NPZ persistence, and OpenCV LBPH with YAML/JSON persistence. Preserve these implementations. Existing thresholds are cosine 0.363 and LBPH distance 80, neither a calibrated probability.
- SG2 has a callable `process_frame(frame_bgr, detections, cfg, frame_id)` and tests. Its baseline produces aligned 112×112×3 `uint8` BGR images using five landmarks; rejected results have `face=None`. It forwards quality measurements and `face_index`, but not original boxes or timestamps. SG3 must join the original detection metadata by index.
- SG4's `face_id_lfw.py` is an LFW identification stand-in using MTCNN and 512-D InceptionResnet embeddings. It has no persistent tracking, SG3 consumer, track IDs or timestamp association. Its gallery cannot consume SFace vectors.
- SG1 has a YuNet image wrapper that discards landmarks and a YOLO CLI/weight, without a complete callable SG1→SG2 packet. A temporary SG3-local YuNet adapter must retain all faces and landmarks.

## Gaps and compatibility problems

There are no SG3 tests, real SG2→SG3 adapter, CCTV runner or leakage-aware preparation scripts. Supplied photos are HEIC, unsupported by existing SG3 image loading. Root `evaluate.py` imports `sg3_core` from the wrong location. Empty input directories crash inference. Model downloads use a mutable URL and only a size check. Gallery loading does not validate embeddings. Existing algorithm comparison aligns SFace but only crops LBPH, so new comparison must share the actual SG2 output. Existing evaluation conflates detection failure with UNKNOWN and replaces missing denominators with zero. Historical nine-image result files lack their source dataset and cannot verify current classroom performance.

## Data discovered before implementation

The workspace contains `RollCall.ai-main.zip` and `Class Dataset-20261026T043822Z-1-001.zip`. The latter contains one approximately 1.68 GB MP4 and 52 HEIC images: 23 labeled images across ten student subdirectories and 29 unlabeled root images. All photos were captured on 15 October 2026 in one session. Full metadata, corruption, duplicate and session audit is performed locally before selecting evaluation records. Root unlabeled photos cannot supply identity ground truth and must not be relabeled by appearance. Original filenames/data are preserved. Only stable pseudonyms appear in new public-facing reports.

## Minimum implementation scope

Reuse recognizers and SG2 unchanged. Add SG3-only strict callable API, all-faces detector/SG2 adapter, private dataset manifest and grouped splits, persistent gallery provenance, identical-preprocessing SFace/LBPH experiments, threshold diagnostics, local annotated CCTV output and a mock downstream consumer. Distinguish quality rejection, UNKNOWN and errors. Test malformed/empty inputs and the actual SG2 boundary. Keep biometric artifacts ignored and local. Record measured PC results only; SG4 tracking, attendance decisions, shared GitHub evidence and Jetson/camera readiness require their owning groups and are not completed by this prototype.

For a single capture session, reserve whole identities for UNKNOWN tests; do not claim independent known-student accuracy from neighboring enrollment captures. Enrollment replay may demonstrate mechanics but must be reported separately from evaluation. Threshold sweeps are exploratory and do not select a validated production threshold.
