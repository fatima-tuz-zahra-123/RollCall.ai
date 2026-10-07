# SG-2 — Face Quality, Alignment & Preprocessing

**Owners:** Syed Ali Mehdi Jaffari (454054) · Fatima Tuz Zahra (478087) · branch `SG2`

## Purpose
SG-2 sits between face detection (SG-1) and recognition (SG-3). For every detected face it decides:
1. **Is this face usable?** It rejects faces that are too small, turned away, badly lit, blurry or low-confidence, and gives a reason code.
2. **If usable, what exactly does SG-3 receive?** A 112×112 face aligned to the standard ArcFace 5-point template, so eyes, nose and mouth always land on the same pixels.

A rejected face is cheap, because a classroom video shows every student in many frames. A wrong recognition marks the wrong student present.

## Dependencies
Python ≥ 3.10, `opencv-python` (tested 5.0), `numpy`, `matplotlib` (plots only). No PyTorch: everything runs on OpenCV DNN + ONNX, which also runs on the Jetson.
```bash
pip install opencv-python numpy matplotlib
```

## Setup and run (from the repo root)
```bash
python module_sg2/scripts/download_assets.py      # ONNX models (46 MB) + LFW test data (181 MB), sha256-checked, gitignored
python module_sg2/sg2_quality.py                  # baseline self-check (12 tests, no downloads needed)
python -m unittest discover -s module_sg2/tests -v  # V1 contract/regression tests
python module_sg2/experiments/exp1_alignment.py   # EXP-1  ~3 min
python module_sg2/experiments/exp2_thresholds.py  # EXP-2  ~10 min
python module_sg2/experiments/exp3_edc.py         # EXP-3  ~6 min
python module_sg2/experiments/exp3b_score_by_degradation.py  # EXP-3b ~2 min
python module_sg2/experiments/exp4_config_compare.py         # EXP-4  ~12 min
python module_sg2/scripts/make_samples.py         # sample I/O for SG-3 + results/before_after.png
```


Use from code:
```python
from sg2_quality import load_config, process_frame
out = process_frame(frame_bgr, faces_from_sg1, load_config("module_sg2/config/baseline_v1.json"), frame_id=7)
```

## Input / output (V1): full spec in [`interfaces/sg2_v1_interface.md`](../interfaces/sg2_v1_interface.md)
- **In (from SG-1):** BGR frame + `faces: [{box: [x,y,w,h], det_conf, landmarks: 5×[x,y]}]`, in full-frame pixels. Landmark order: image-left eye, image-right eye, nose, image-left mouth corner, image-right mouth corner.
- **Out (to SG-3):** one result per face: `{face_index, accept, reason, quality 0–1, metrics, face}`. `face` = **112×112×3 uint8 BGR, aligned**, or `null` if rejected.
- Reason codes: `bad_input, low_det_conf, too_small, no_landmarks, pose, align_failed, illumination, blurry`.
- Sample in/out for SG-3: [`interfaces/examples/`](../interfaces/examples/) + [`samples/`](samples/).

## Processing (baseline = `config/baseline_v1.json`)
validate → det-conf ≥ 0.5 → face ≥ 40 px → landmarks present → |yaw| ≤ 0.30 → **5-point similarity alignment** (`estimateAffinePartial2D` + LMEDS → `warpAffine` 112×112) → illumination (mean 50–220, ≤30% crushed/blown) → blur (Laplacian variance ≥ 50 on the aligned face) → heuristic quality score.

## Test data and metrics (same for every configuration)
- **LFW** official `pairs.txt`: 6,000 face pairs, 10 folds, 7,701 images. Public, labelled, standard verification protocol.
- **Mock SG-1 = YuNet** (OpenCV Zoo): box + 5 landmarks, exactly our V1 input. Detected 7,701 / 7,701.
- **Stand-in SG-3 = SFace** (OpenCV Zoo) to measure **downstream recognition impact**.
- **Metrics:**
  - LFW 10-fold accuracy;
  - **FNMR at FMR = 1%** (how many genuine students are missed when only 1% of impostors get through);
  - EER;
  - accept/reject %;
  - **EDC / pAUC** for the quality score (the ISO/IEC 29794-1 evaluation method);
  - ms per face.

## Week-5 experiments and results
Raw numbers: [`results/`](results/) (JSON, versioned). Before/after examples: [`results/before_after.png`](results/before_after.png).

### EXP-1: alignment ON vs OFF / crop margin (`results/exp1_alignment.json`)
| Config (gates off, same 6,000 pairs) | LFW acc | FNMR @ FMR 1% | EER | SG-2 ms/face |
|---|---|---|---|---|
| **A0 baseline: 5-pt similarity alignment** | **99.33% ± 0.37** | **0.83%** | **0.87%** | 0.74 |
| A1 no alignment, tight box | 87.38% | 37.13% | 12.60% | 0.57 |
| A2 no alignment, box + 20% margin | 92.12% | 21.37% | 8.20% | 0.58 |
| A3 no alignment, box + 40% margin | 87.45% | 38.00% | 12.63% | 0.57 |
| REF: OpenCV `FaceRecognizerSF.alignCrop` | 99.35% | 0.73% | 0.80% | 0.12 |

**Result:** alignment cuts missed matches about 26×, from 21.4% to 0.83%. Our alignment equals OpenCV's reference implementation, and the bench reproduces SFace's published 99.40% LFW accuracy. → **Keep alignment ON.**

### EXP-2: gate thresholds vs where recognition really breaks (`results/exp2_thresholds.json`)
Each probe face was degraded in a controlled way (gallery clean; 3,000 pairs; oracle SG-1 landmarks).

| Degradation | SFace FNMR (no gate) | Baseline accepts | Reading |
|---|---|---|---|
| clean | 1.13% | 93.1% | baseline already drops 6.9% of good faces |
| blur σ=1 / 2 / 3 / 4 / 6 | 1.20 / 1.40 / 3.07 / 9.8 / 57% | 58.9 / 2.0 / 0.2 / 0.1 / 0% | gate fires far too early (σ=2 is still fine) |
| face 64 / 48 / 32 / 24 / 16 px | 1.13 / 1.13 / 1.40 / 1.67 / 6.2% | 88 / 79 / 0 / 0 / 0% | 40 px gate throws away recognisable back-row faces |
| exposure ×0.6 / 0.35 / 0.2 / 0.1 (dark) | 1.20 / 1.27 / 1.47 / 3.27% | 63 / 1.2 / 0 / 0% | darkness barely hurts SFace (noise-free simulation) |
| exposure ×1.6 / 2.2 / 3.0 (blown out) | 1.67 / 7.9 / 30.6% (FMR ↑ to 1.4%) | 70.5 / 16.8 / 2.2% | over-exposure is the real danger, including false matches |

Threshold sweeps (pooled): `min_blur` 20 → accepted FNMR 1.23% (vs 12.3% ungated) at only 0.17% clean false-rejects (50 → 3.1%). `min_face_px` 24 → 1.28%. `max_brightness` 180 → 2.08% (vs 6.07%). Pose on real LFW pose: `|yaw| ≤ 0.30` gives the lowest FNMR, 0.72% vs 0.83% ungated (difference is within noise).

### EXP-3: is the quality SCORE any good? EDC curve (`results/exp3_edc.json`, `exp3_edc.png`)
Error-versus-Discard Characteristic (ISO/IEC 29794-1): drop the lowest-quality x% of pairs and re-measure FNMR at the fixed FMR=1% threshold. pAUC over 0–20%, normalised: **lower = better, random ≈ 1.0**.

| Quality score | pAUC clean LFW | pAUC classroom-like mix | FNMR after dropping worst 20% (mix) |
|---|---|---|---|
| random | 0.97 | 0.99 | 5.45% |
| blur only (Laplacian) | 0.99 | 0.67 | 2.43% |
| **baseline heuristic** | 0.94 | **1.05 (worse than random)** | 6.17% |
| **eDifFIQA(T)** (learned, 7 MB ONNX, ~3 ms/face CPU) | **0.53** | **0.60** | **1.83%** (from 5.63%) |

EXP-3b (`exp3b_score_by_degradation.json`) explains the failure. Across degradation types, the correlation between mean score and FNMR is **+0.11 for the heuristic** (wrong direction) and **−0.74 for eDifFIQA**. The heuristic *averages* four terms, so a destroyed σ=5 face still scores 0.63 while harmless dark or 24 px faces score about 0.47. eDifFIQA does not penalise over-exposure (0.58), which is why the brightness gate stays.

### EXP-4: baseline vs Week-6 candidate, end-to-end with a REAL detector (`results/exp4_config_compare.json`)
YuNet is re-run on every degraded image (no oracle). Same 3,000 pairs, same 19 levels.

| | baseline_v1 | **week6_candidate** |
|---|---|---|
| clean faces accepted | 93.1% | **95.5%** |
| mean accepted over all 19 levels | 33.1% | **49.5%** |
| 32 px / 24 px faces accepted | 0% / 0% | **92.8% / 67.6%** |
| missed matches among accepted (≈) | 1.18% | 1.33% |

The real detector is worse than the oracle on degraded images (σ=4 FNMR 13.5% vs 9.8%; 24 px 3.1% vs 1.7%; at ×3 exposure 15.6% of faces are not even detected).

## Week-6 recommendation
1. **Alignment ON** (5-point similarity → 112×112): EXP-1, a 26× reduction in missed matches.
2. **Thresholds = `config/week6_candidate.json`** (blur 20, size 24 px, brightness 20–180, yaw 0.30): EXP-2 + EXP-4, about 50% more usable faces at nearly the same reliability. Back-row students stop being permanently rejected.
3. **Quality score → eDifFIQA(T)**: EXP-3/3b. Keep the hand-made gates for explainable reject reasons. Use eDifFIQA for ranking, i.e. picking the best frame per track for SG-4.
4. **Week-6 rigorous checks:**
   - real classroom clip through SG-1;
   - Jetson latency of the full SG-2 step with eDifFIQA (with SG-6);
   - low light **with sensor noise** (our darkening is noise-free);
   - `max_brightness` 180 vs 200 (180 drops some ×1.6 faces that are still recognisable).

## Current limitations
- LFW faces are web photos: large (median 93 px), mostly sharp and frontal. Classroom problems are **simulated** (EXP-2/3), not yet measured on real classroom video.
- EXP-2/3 use oracle detection (landmarks from the clean image). EXP-4 removes this and shows real detection on degraded images is worse.
- Darkening is simulated without sensor noise, so real low light will be harder than EXP-2 suggests.
- The baseline heuristic quality score ranks faces poorly (EXP-3); it is replaced by eDifFIQA(T) for Week 6.
- FNMR differences below ~0.2 points are within noise (3,000 genuine pairs → 0.1% ≈ 3 pairs).
- The pose gate uses only yaw from 5 landmarks. Pitch and occlusion (masks, hands) have no gate yet.
- Jetson timing not measured yet (needs SG-6 / the board). OpenCV Zoo publishes Orin Nano numbers for YuNet (2.59 ms CPU) and SFace (20.05 ms CPU).
- SG-1 has not confirmed it will send landmarks. Without them V1 rejects the face (`no_landmarks`).

## Reference implementations
Ranked screening of 6 candidates: [`docs/reference_screening.md`](docs/reference_screening.md). Primary = OpenCV Zoo (YuNet + 5-pt alignment + eDifFIQA), Backup = InsightFace.
