# SG-2's review of SG-1 (code + algorithm + compatibility)

Asked by the group lead: every subgroup goes through the code of the subgroup right before it. We're SG-2, so this is SG-1 (face detection). Reviewed on main as of 27 Oct, tested on the real class video (CR #28, 8 Oct lecture) and the enrollment photos.

**Short version:** SG-1's detector works on the real camera and plugs into SG-2 fine. We found nothing that blocks integration. The main gaps are reproducibility from a fresh clone and the missing V1 output in `module_sg1/` itself.

## What SG-1 gives us

- Model: YOLOv12n-face (`module_sg1/weights/yolov12n-face.pt`, 5.5 MB), run through `ultralytics==8.4.173`, `conf=0.25`, `imgsz=1280`.
- Output per face: box + confidence. **No landmarks**: it's a plain detection head, no keypoint head.
- Conversion to our V1 format happens in `integration/sg1_adapter.py` (written by Fatima, SG-2), not in `module_sg1/`.

## Tested on real data

| Check | Result |
|---|---|
| Detections over the lecture (552 frames, min 12–58, every 5 s) | 22,242 faces, ~40 per frame |
| Face sizes it gives us | median 33 px, 10% of faces under 28 px |
| SG-2 could find landmarks for (YuNet fallback) | 97.9% of detections |
| Confidence | median 0.77, 6% below 0.5 |
| `imgsz` 1280 vs 1920 vs 2560 (3 frames, `exp6 --imgsz-check`) | 41.3 / 42.3 / 41.0 faces per frame, 183 / 658 / 1928 ms per frame on CPU |
| Enrollment photos (35 shots, 3024×4032) | found the person in every shot |

The `imgsz` test matters: we first thought 1280 was too small for a 2560 px CCTV frame and was missing back-row students. It isn't. Going bigger finds the same faces (±1) and is 3.5–10× slower, so **1280 is the right setting**.

## Compatibility issues (and what happened)

| # | Issue | Impact | Status |
|---|---|---|---|
| 1 | No landmarks in the output | SG-2 rejected every face (`no_landmarks`) | **Fixed on SG-2's side**: YuNet runs on the box ROI (`roi_landmarks`). Costs ~2–3 ms per face. If SG-1 ever adds landmarks, we use theirs automatically. |
| 2 | Confidence scale differs from YuNet's, and our 0.5 cut-off was tuned on YuNet | could reject real faces | Re-checked on the class video: 0.25 vs 0.5 was part of the EXP-6 search and 0.5 gave lower error, so it stays. We did spot a clearly visible face rejected for low confidence in a sample frame, so it's worth SG-1 checking which faces score under 0.5. |
| 3 | `module_sg1/` has no V1 output, only annotated pictures (`run.py predict`) | other SGs depend on `integration/sg1_adapter.py` | Open. Suggest SG-1 moves `xyxy_to_v1_faces` into `module_sg1/` and owns it. |

## Reproducibility (things a fresh clone can't run)

- `run.py prepare / train / val` need `datasets/` (WIDER FACE) and `datasets/widerface.yaml`, which aren't in the repo, and the README doesn't say where to get them.
- `run.py predict` with no `--source` points at `images (1).jpg`, which isn't in the repo either, so the default command fails.
- `load_yolo()` adds site-packages to `sys.path` by hand. That was needed while the copied ultralytics folder was in the repo. It's gone now, so a plain `from ultralytics import YOLO` is enough.
- No WIDER val numbers (mAP / recall) in the README yet, and the midterm rubric asks for preliminary evidence.

## Suggestions for SG-1 (in order of usefulness)

1. Add a `--json` option (or a tiny `detect.py`) that writes our V1 format, so SG-1 owns its output.
2. Put the WIDER download steps + `widerface.yaml` in the README, and point the default `predict` image at something in the repo.
3. Run `run.py val` once and paste the numbers in the README.
4. Optional: a face-with-landmarks checkpoint (YOLO-face pose, SCRFD or YuNet). It would save SG-2's ~2–3 ms per face on the Jetson.

Numbers above: `module_sg2/results/exp6_classroom_cctv.json` and `exp6_sg1_imgsz_check.json` (video) and `exp7_enrollment_gallery_*.json` (photos). Scripts: `module_sg2/experiments/exp6_classroom_cctv.py`, `exp7_enrollment_gallery.py`.
