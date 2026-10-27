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
python module_sg2/experiments/exp5_landmark_fallback.py      # EXP-5  ~8 min (landmark fallback)
python module_sg2/scripts/make_samples.py         # sample I/O for SG-3 + results/before_after.png
```

## V1 contract tests
The new [`tests/test_sg2_quality.py`](tests/test_sg2_quality.py) file adds four
deterministic regression tests for the SG2 module. These tests use a synthetic
SG1-style frame and face record to verify that an accepted face is returned in
the required 112x112x3 `uint8` format, that multiple results preserve input
order and `face_index`, and that `to_json()` removes image data before the
metadata is handed to downstream modules. They also check that invalid input,
low-confidence faces, small faces, and missing landmarks are rejected with
clear reason codes, while an empty face list produces an empty result. This is
the individual integration contribution for SG2: it protects the frozen V1
interface and catches regressions in the SG1-to-SG2-to-SG3 handoff while the
larger experiments measure recognition and preprocessing performance. Run it
from the repository root with:

```bash
python -m unittest discover -s module_sg2/tests -v
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
- `landmarks` can now be `null`. SG-1's YOLOv12n-face only gives boxes, so with `landmark_fallback` on we find the points ourselves (see below). `metrics.lmk_source` says where they came from: `sg1` or `yunet_roi`.

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

## Landmark fallback (fix for the SG1 → SG2 blocker)

The problem: our alignment needs 5 points on the face (eyes, nose, mouth corners), but SG-1 moved to YOLOv12n-face, which only gives a box. So V1 rejected every single face as `no_landmarks` and nothing reached SG-3.

The fix (`roi_landmarks()` in `sg2_quality.py`): if a face shows up without points, we cut a padded square around the box, resize it to 160 px, run YuNet on just that patch and take the 5 points of the face nearest the centre. If that face's centre isn't inside SG-1's box (we caught the student next to them) we still reject with `no_landmarks`. If SG-1 does send points, those always win. `metrics.lmk_source` says where the points came from (`sg1` / `yunet_roi`).

Fatima and I both wrote a version of this on separate branches. We kept hers (it's the one wired into `integration/sg1_sg2_pipeline.py` and the SG-3 handoff) and re-ran my LFW check against it. Settings live in the config (`sg1_yolov12_v1.json`, same block in `week6_candidate.json`):
```json
"landmark_fallback": {"method": "yunet_roi", "model": "models/face_detection_yunet_2023mar.onnx",
                      "pad": 0.3, "roi_size": 160, "min_score": 0.3}
```
Set it to `null` to get the old V1 behaviour. `baseline_v1.json` is left alone since it's our reference.

EXP-5 (`experiments/exp5_landmark_fallback.py`, `results/exp5_landmark_fallback.json`): boxes shifted and resized randomly a bit so they don't look like YuNet's own, gates off, all 6000 LFW pairs:

| | faces through | FNMR @ FMR 1% | time / face (PC) |
|---|---|---|---|
| SG-1 gives landmarks (week 5) | 100% | 0.83% | 0.8 ms |
| boxes only, no fallback (V1) | **0%** | – | – |
| boxes only + fallback | **100%** | 0.53% | 2.9 ms |
| 32 px faces: SG-1 points / fallback | 100% / 100% | 1.60% / 1.20% | |
| 24 px faces: SG-1 points / fallback | 100% / 100% | 2.73% / 2.53% | |

Recognition is the same either way (the differences are noise) and the cost is ~2 ms extra per face. The fallback points sit on average 8.5% of the eye distance from YuNet-on-the-full-image points, which doesn't bother SFace.

Things to keep an eye on:
- On the Jetson this is one extra YuNet run per face (OpenCV Zoo lists ~2.6 ms on Orin CPU). 40 faces would add ~100 ms per frame, so we may only run it once per track (SG-4) instead of every frame. `scripts/bench_sg2.py` measures it.
- If SG-1 later switches to a detector with landmarks, nothing changes on their side.

## Week-6 recommendation
1. **Alignment ON** (5-point similarity → 112×112): EXP-1, a 26× reduction in missed matches.
2. **Thresholds = `config/week6_candidate.json`** (blur 20, size 24 px, brightness 20–180, yaw 0.30): EXP-2 + EXP-4, about 50% more usable faces at nearly the same reliability. Back-row students stop being permanently rejected.
3. **Quality score → eDifFIQA(T)**: EXP-3/3b. Keep the hand-made gates for explainable reject reasons. Use eDifFIQA for ranking, i.e. picking the best frame per track for SG-4.
4. **Week-6 rigorous checks:**
   - real classroom clip through SG-1;
   - Jetson latency of the full SG-2 step with eDifFIQA (with SG-6);
   - low light **with sensor noise** (our darkening is noise-free);
   - `max_brightness` 180 vs 200 (180 drops some ×1.6 faces that are still recognisable).

## Week-8 integration with the real SG-1 (YOLOv12n-face)
SG-1's checkpoint is a box-only `DetectionModel` (no keypoint head), so it sends `landmarks: null`. Under V1 every face would be rejected `no_landmarks`. **V1.1 fix:** when landmarks are missing, SG-2 runs YuNet on the padded SG-1 box ROI and maps the 5 points back to the frame (`landmark_fallback` in [`config/sg1_yolov12_v1.json`](config/sg1_yolov12_v1.json); spec in the interface doc, *V1.1 amendment*).

```bash
python module_sg2/scripts/download_assets.py --models        # YuNet for the fallback
pip install -r module_sg1/requirements.txt                    # SG-1's ultralytics
python integration/sg1_sg2_pipeline.py --source path/to/classroom.jpg     # image / folder / video
python integration/sg1_sg2_pipeline.py --source 0 --show                  # live webcam
python integration/sg1_sg2_pipeline.py --mock-sg1 interfaces/examples/sg2_example_input.json  # SG-1 absent
python -m unittest integration/test_sg1_adapter.py -v
```
Outputs go to `runs/sg1_sg2/`: `annotated/` (green = accepted + quality, red = reason, dots = landmarks used), `faces/` (aligned 112×112 = what SG-3 gets), `sheets/` (every face with its decision), `results.jsonl`, `summary.json`.

**First results (real SG-1 → SG-2, `sg1_yolov12_v1.json`, dev PC CPU):**

| Image | SG-1 faces | Fallback found landmarks | Accepted | Rejected |
|---|---|---|---|---|
| Class photo, 57 people, 1280×1024, faces 27–42 px | 57 / 57 people | 57 / 57 | 19 | 38 `blurry` |
| Group still, 610×395, faces ~46 px | 6 | 6 / 6 | 6 | 0 |
| Single portrait, 512×512 | 1 | 1 / 1 | 1 | 0 |

- Fallback landmark agreement with full-frame YuNet: mean error 9% of eye distance on the class photo (7% group, 6% portrait); 0 misses over 64 faces.
- SG-2 cost: about 6 ms/face including the fallback (CPU, dev PC). Jetson timing is still pending.

**Finding (blur gate is size-biased):** the 38 `blurry` rejects on the class photo look as sharp as the accepted faces. They are only slightly smaller (median 33 vs 36 px). Laplacian variance is measured after upscaling to 112 px, which smooths small faces: corr(face_px, blur) = +0.58 on this image. `min_blur = 20` came from LFW (~93 px faces) and does not transfer to back-row faces. Measuring blur at native resolution over-corrects (corr −0.53 on the 19 accepted faces), so it is not a drop-in fix. → Week-9 experiment: blur × face-size grid on the EXP-2 bench, then a size-aware blur threshold.

## Week-8: real class data (CCTV video + enrollment photos)

Data: the CR #28 ceiling camera recording of the 8 Oct lecture (2560×1440, 70 min) and the enrollment photos (`Class Dataset/<name>/*.HEIC`, iPhone, front/left/right/top). The data is **not** in the repo (it's classmates' faces and the repo is public); ask the team for the Drive folder. Results below are numbers only.

### EXP-6: the real SG-1 → SG-2 → SFace on the lecture (`experiments/exp6_classroom_cctv.py`)
552 frames (minutes 12–58, one every 5 s), SG-1's real YOLOv12n-face at imgsz 1280, our landmark fallback, SFace as SG-3 stand-in.

Nobody labelled who sits where, so we used the seats as labels: **same seat ≥ 60 s apart = same person, different seats in the same frame = different people**. Students don't swap seats mid-lecture, and we checked a sheet of pairs by eye (including the lowest-scoring "same person" pairs: they really are the same person, just looking down or with a hand on the face). That gave 89,981 same-person and 22,080 different-person pairs from real footage.

What the camera gives us: 22,242 faces, median **33 px** wide, median yaw 0.28. Fallback found landmarks for 97.9%.

Where recognition actually breaks (FNMR at FMR 1%, pair-level):

| | | | | |
|---|---|---|---|---|
| yaw (turned) | 0–0.15: 58%* / 0.15–0.3: 45% | 0.3–0.45: 42% | 0.45–0.6: 46% | 0.6–0.8: 55%, >0.8: 69% |
| pitch dev (head down/up, new) | <0.1: 44% | 0.1–0.2: 68% | 0.2–0.3: 82% | >0.3: 92% |
| blur | <50: 75–90% | 50–100: 60% | >100: 38% | |
| face size | 20–24 px: 77% | 24–28: 59% | 28–40: 46–48% | >40: 41% |

*The near-frontal bin looks worse because 20% of those faces are heads bent over the desk (vs 5–9% in the other bins): looking down lines the nose up between the eyes, so yaw reads ~0 even though the face is bad. That's what the pitch gate below catches.

So on this camera: turning the head barely matters until ~0.6 (our 0.30 limit was throwing away faces for nothing), **looking down matters a lot** (we had no check for it), and blur matters much more than on LFW (our LFW-tuned 20 was too loose).

We added a pitch measure (`pitch_proxy()`: where the nose sits between the eye line and the mouth line, 0.495 = frontal) and picked new thresholds **on the first half of the lecture only**, then checked them on the second half, which they never saw:

| second half (min 35–58) | faces sent to SG-3 | FNMR @ FMR 1% |
|---|---|---|
| no gates | 100% | 46.2% |
| V1.1 (`sg1_yolov12_v1.json`) | 46.4% | 40.7% |
| **`cctv_v2.json`** (yaw 0.45, pitch 0.10, blur 75) | **62.7%** | **33.8%** |

More faces *and* fewer mistakes. On 56 frames (one per minute) the real pipeline accepts 57.1% vs 43.8% before, at ~2.8 ms per face.

### EXP-6 on a second room: our own class, CR #29 (6 Oct), and EXP-6b
Same run on the CR #29 recording (different camera angle, brighter, students seen more side-on: median yaw 0.48 vs 0.28). 552 frames, 14,147 faces, 54,588 same-person / 22,080 different-person pairs. Every frame is new to the config.

- **Pitch and blur behaved the same as in CR #28:** pitch dev <0.1: 54% FNMR, 0.1–0.2: 76%, 0.2–0.3: 82%. Blur <50: 80–90%, >100: 53%. So those two gates carry over.
- **Yaw didn't predict failure at all here** (55–60% FNMR in every bin), so `cctv_v2`'s 0.45 limit just threw faces away: on CR #29 it kept only 32.5% of faces (still better than V1.1's 23.5%, with fewer misses: 52.3% vs 58.0%).

The Face-Off might be in yet another room, so we picked one config on the **first half of both** lectures and checked it on the **second half of both** (`experiments/exp6b_two_rooms.py`):

| second halves | CR #28 faces / FNMR | CR #29 faces / FNMR |
|---|---|---|
| no gates | 100% / 46.2% | 100% / 54.9% |
| V1.1 (`sg1_yolov12_v1.json`) | 46.4% / 40.7% | 25.6% / 58.0% |
| `cctv_v2` (CR #28 only) | 62.7% / 33.8% | 33.9% / 54.5% |
| **`cctv_v3`** (both rooms) | **73.2% / 34.7%** | **54.6% / 47.3%** |

`cctv_v3` = `cctv_v2` with yaw 0.45 → 0.75 (pitch 0.10, blur 75 unchanged). It's now the default in `integration/sg1_sg2_pipeline.py`. The real pipeline on one frame per minute accepts **69.1%** of faces in CR #28 (V1.1: 43.8%) and **52.2%** in CR #29 (V1.1: 21.6%), ~2.8 ms per face.

CR #29 is harder overall (58.6% FNMR with no gates vs 47.9%): faces are more side-on to the camera. Camera placement matters as much as our thresholds, worth telling SG-6 for the Face-Off setup.

Honest part: even with good gates, SFace misses about a third of same-person pairs on 33 px CCTV faces at FMR 1%. That's CCTV-vs-CCTV though; SG-3 will match against high-res enrollment photos, and SG-4/SG-5 combine many frames per student, so per-frame misses don't mean missed attendance. Quality scores help only a little here (EDC pAUC: eDifFIQA 0.91, blur 0.91, heuristic 0.93, random 1.0).

### EXP-7: enrollment photos (`experiments/exp7_enrollment_gallery.py`)
SG-3's gallery must go through the **same** SG-2 alignment as the video faces. Pushing all 35 shots through SG-1 → SG-2:

| config | shots accepted | people with no usable shot (of 15) |
|---|---|---|
| `sg1_yolov12_v1.json` | 7 / 35 | 8 |
| `cctv_v2.json` | 7 / 35 | 10 |
| **`enroll_v2.json`** (yaw 0.60, pitch 0.15) | **16 / 35** | **4** |

The left/right/top shots are near-profiles (yaw 0.5–1.3) and EXP-6 shows recognition falls apart past ~0.6, so most of them can't be used no matter what. **4 people only have profile shots and need one front photo retaken.** The 17 unsorted `IMG_xxxx` photos in the dataset root also need to go into name folders.

### Running it
```bash
python integration/sg1_sg2_pipeline.py --source path/to/frames_or_video     # uses cctv_v3.json now
python module_sg2/experiments/exp6_classroom_cctv.py --video "path/to/CR # 28 ... .mp4"   # ~20 min CPU
python module_sg2/experiments/exp6_classroom_cctv.py --analyse-only                        # re-use cached faces
python module_sg2/experiments/exp6_classroom_cctv.py --video "path/to/CR # 29 ... .mp4" --tag cr29
python module_sg2/experiments/exp6b_two_rooms.py                                           # needs both caches
python module_sg2/experiments/exp7_enrollment_gallery.py --dataset "path/to/Class Dataset" --config enroll_v2.json
python module_sg2/scripts/bench_sg2.py           # SG-2 timing on this machine (for the Jetson)
```
Needs `pip install ultralytics==8.4.173 pillow-heif` on top of our normal deps. Our review of SG-1 is in [`docs/sg1_review.md`](docs/sg1_review.md).

## Current limitations
- EXP-1 to EXP-5 are on LFW (web photos, ~93 px faces). EXP-6/6b/7 are on real class data: two lectures in two rooms. A third room or very different lighting could still shift the thresholds (yaw especially).
- EXP-6 labels come from seats, not from names, so it measures CCTV-vs-CCTV matching, not gallery-vs-CCTV.
- EXP-2/3 use oracle detection (landmarks from the clean image). EXP-4 removes this and shows real detection on degraded images is worse.
- Darkening is simulated without sensor noise, so real low light will be harder than EXP-2 suggests.
- The baseline heuristic quality score ranks faces poorly (EXP-3); it is replaced by eDifFIQA(T) for Week 6.
- FNMR differences below ~0.2 points are within noise (3,000 genuine pairs → 0.1% ≈ 3 pairs).
- The pose gate uses only yaw from 5 landmarks. Pitch and occlusion (masks, hands) have no gate yet.
- Jetson timing not measured yet (needs SG-6 / the board). OpenCV Zoo publishes Orin Nano numbers for YuNet (2.59 ms CPU) and SFace (20.05 ms CPU).
- ~~SG-1 has not confirmed it will send landmarks.~~ SG-1's YOLOv12n-face is box-only. Resolved in V1.1 with the YuNet ROI landmark fallback (see *Week-8 integration* below). This adds a second model per face, so its Jetson cost must be measured.
- The YOLO `det_conf` scale differs from YuNet's, and `min_det_conf = 0.5` was tuned on YuNet. It needs re-tuning on classroom data.

## Reference implementations
Ranked screening of 6 candidates: [`docs/reference_screening.md`](docs/reference_screening.md). Primary = OpenCV Zoo (YuNet + 5-pt alignment + eDifFIQA), Backup = InsightFace.
