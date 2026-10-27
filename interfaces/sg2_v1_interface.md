# SG-2 V1 Interface — Face Quality, Alignment & Preprocessing

Status: **V1 proposal from SG-2** (Syed Ali Mehdi Jaffari, Fatima Tuz Zahra). Any change after freeze needs agreement from SG-1 and SG-3 (and SG-4/SG-5 for the metadata fields).
Reference implementation: [`module_sg2/sg2_quality.py`](../module_sg2/sg2_quality.py) · examples: [`interfaces/examples/`](examples/)

## Conventions (apply to every field below)
| Item | Convention |
|---|---|
| Image | `numpy.ndarray`, `uint8`, **BGR** (OpenCV order), shape `H×W×3`, the full camera frame |
| Coordinates | pixels, origin **top-left**, x → right, y → down, floats allowed, **full-frame** coordinates (not crop-relative) |
| Box | `[x, y, w, h]` (top-left corner + width/height) |
| Landmarks | 5 points `[[x,y]×5]` in this order: **image-left eye, image-right eye, nose tip, image-left mouth corner, image-right mouth corner** (= YuNet / RetinaFace / SCRFD order = ArcFace template order) |
| Confidence / quality | float in **[0, 1]** |

## Input: SG-1 → SG-2 (per frame)
```json
{
  "frame_id": 1042,
  "timestamp": 12.37,
  "faces": [
    {"box": [x, y, w, h], "det_conf": 0.93,
     "landmarks": [[x, y], [x, y], [x, y], [x, y], [x, y]]}
  ]
}
```
+ the frame itself (`image`) passed alongside (in-process reference, not copied/serialised).

### V1.1 amendment (Week 8): SG-1 is box-only → `landmarks: null` is now a supported input
SG-1 selected **YOLOv12n-face** (`module_sg1/weights/yolov12n-face.pt`). Its checkpoint is an Ultralytics `DetectionModel` with a `Detect` head and **no keypoint head**, so it cannot output landmarks. Agreed handling:

| Item | Rule |
|---|---|
| SG-1 native output | Ultralytics `boxes.xyxy` `[x1,y1,x2,y2]` (original-frame pixels, letterbox already undone) + `boxes.conf` |
| Conversion to V1 | [`integration/sg1_adapter.py`](../integration/sg1_adapter.py) `xyxy_to_v1_faces`: → `box: [x, y, w, h]`, clipped to the frame; degenerate boxes dropped; `landmarks: null` |
| SG-2 behaviour | config [`sg1_yolov12_v1.json`](../module_sg2/config/sg1_yolov12_v1.json) enables `landmark_fallback`: YuNet runs on the box ROI (padded 30 %, resized to 160 px), the detection nearest the ROI centre is kept only if its centre lies inside SG-1's box, its 5 landmarks are mapped back to full-frame pixels, then the normal alignment runs |
| No face found in the ROI | `accept:false, reason:"no_landmarks"` (unchanged code — SG-1 false positives end here) |
| Traceability | `metrics.lmk_source` = `"sg1"` / `"yunet_roi"`, `metrics.lmk_score` = YuNet score; result field `landmarks` = the 5 points actually used |
| Order of gates | `det_conf` and `too_small` run **before** the fallback, so YuNet never runs on faces that would be dropped anyway |
| If SG-1 later adds landmarks | Sent landmarks always take priority; the fallback simply stops running. No interface change |

Backup (documented, not implemented): SG-1 switches to a YOLO-face *pose* checkpoint with 5 keypoints, which removes the second model. Decision to revisit after Jetson profiling (Lab 10).

## Output: SG-2 → SG-3 (and metadata for SG-4/SG-5)
One result **per input face, same order** (`face_index` = position in `faces`).
```json
{
  "frame_id": 1042,
  "results": [
    {"face_index": 0,
     "accept": true,
     "reason": null,
     "quality": 0.81,
     "metrics": {"face_px": 92.8, "det_conf": 0.93, "lmk_source": "sg1", "yaw": 0.04, "pitch": 0.51, "blur": 412.0, "brightness": 118.3, "sat_frac": 0.01},
     "face": "<112x112x3 uint8 BGR, aligned>"}
  ]
}
```
| Field | Meaning |
|---|---|
| `accept` | `true` → send `face` to SG-3. `false` → do **not** recognise this face in this frame |
| `reason` | `null` if accepted, else one of: `bad_input`, `low_det_conf`, `too_small`, `no_landmarks`, `pose`, `align_failed`, `blurry`, `illumination` |
| `quality` | soft score 0–1 (higher = better). SG-4 can keep the best-quality crop per track; SG-5 can weight evidence |
| `metrics` | raw measurements behind the decision (for debugging + failure analysis). `pitch` (Week 8) = nose position between eye line and mouth line, 0.495 = level head |
| `face` | **112×112×3, uint8, BGR, aligned** to the ArcFace 5-point template (eyes at ≈(38,52) and (74,52)); `null` when rejected. **No mean/std normalisation is applied** — SG-3 applies its model's own normalisation (e.g. SFace takes this image directly; ArcFace-ONNX models expect `(x−127.5)/127.5`, RGB). |

### SG-3 handoff (agreed with SG-3, Week 8): NumPy arrays
SG-3 (SFace) takes only **accepted** faces. Two equivalent forms:
- **In-process (live pipeline):** `sg2_quality.sg3_items(out, sg1_faces)` → list of
  `{frame_id, face_index, face_id, face, quality, box}`, where `face` is the 112×112×3 uint8 BGR array.
  SG-3 calls `recognizer.match(item["face"], None)` (no re-detection, no re-alignment).
- **File (offline testing, CCTV clips):** one `.npz` per frame from `sg2_quality.save_sg3_npz`:

| Key | Shape / type | Meaning |
|---|---|---|
| `faces` | `(N,112,112,3)` uint8 | aligned BGR faces, no normalisation |
| `face_ids` | `(N,)` str | `"<frame_id>_<face_index>"`, unique per detection within a run, **not** a person or track ID (that is SG-4's job) |
| `face_index` | `(N,)` int32 | position in SG-1's face list for this frame |
| `quality` | `(N,)` float32 | SG-2 soft quality score 0–1 |
| `boxes` | `(N,4)` float32 | SG-1 box `[x, y, w, h]` in frame pixels: use this as SG-3's bbox |
| `frame_id` | scalar | frame number |

`N = 0` is valid (no usable face in that frame). Rejected faces and their reasons are not in the `.npz`; they are in `results.jsonl` (for SG-4/SG-5). Reader: [`integration/sg3_load_example.py`](../integration/sg3_load_example.py).

## Behaviour rules
- Never raises on bad input: malformed face → `accept:false, reason:"bad_input"`.
- `faces: []` → `results: []`.
- All faces rejected → results present, every `accept:false`.
- Thresholds live in a versioned config file (`module_sg2/config/*.json`), not in code.
- Week 8: `cctv_v2.json` is the config for the classroom camera (tuned on the 8 Oct lecture, EXP-6) and `enroll_v2.json` is for SG-3's gallery photos (EXP-7). Gallery and video faces must both go through SG-2 so they're aligned the same way.
- A new gate `max_pitch_dev` (head tilted down/up) reuses the existing `pose` reason code, so no new reason values.

## Open points to agree with neighbours
1. SG-1: ~~landmarks in output~~ resolved by V1.1 (SG-2 fallback); still open: confidence calibration (YOLO `conf` ≠ YuNet score, so `min_det_conf` 0.5 needs re-tuning on classroom data), minimum face size they can detect, and SG-1 emitting V1 directly (move `sg1_adapter.py` into `module_sg1/`).
2. SG-3: confirm 112×112 ArcFace-template input; who normalises pixels (proposal: SG-3).
3. SG-4: do you want `quality` to select the best crop per track?
