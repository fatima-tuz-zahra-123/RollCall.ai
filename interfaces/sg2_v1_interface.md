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

**Request to SG-1:** please include the 5 landmarks. Every candidate detector we looked at (YuNet, RetinaFace, SCRFD) outputs them for free; without them SG-2 must run a second landmark model on every face (slower on the Jetson). If `landmarks` is `null`, SG-2 rejects with `no_landmarks` in V1 (alignment is required — see EXP-1: no alignment costs ~7 points of recognition accuracy).

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
     "metrics": {"face_px": 92.8, "det_conf": 0.93, "yaw": 0.04, "blur": 412.0, "brightness": 118.3, "sat_frac": 0.01},
     "face": "<112x112x3 uint8 BGR, aligned>"}
  ]
}
```
| Field | Meaning |
|---|---|
| `accept` | `true` → send `face` to SG-3. `false` → do **not** recognise this face in this frame |
| `reason` | `null` if accepted, else one of: `bad_input`, `low_det_conf`, `too_small`, `no_landmarks`, `pose`, `align_failed`, `blurry`, `illumination` |
| `quality` | soft score 0–1 (higher = better). SG-4 can keep the best-quality crop per track; SG-5 can weight evidence |
| `metrics` | raw measurements behind the decision (for debugging + failure analysis) |
| `face` | **112×112×3, uint8, BGR, aligned** to the ArcFace 5-point template (eyes at ≈(38,52) and (74,52)); `null` when rejected. **No mean/std normalisation is applied** — SG-3 applies its model's own normalisation (e.g. SFace takes this image directly; ArcFace-ONNX models expect `(x−127.5)/127.5`, RGB). |

## Behaviour rules
- Never raises on bad input: malformed face → `accept:false, reason:"bad_input"`.
- `faces: []` → `results: []`.
- All faces rejected → results present, every `accept:false`.
- Thresholds live in a versioned config file (`module_sg2/config/*.json`), not in code.

## Open points to agree with neighbours
1. SG-1: landmarks in output (requested above); minimum face size they can detect.
2. SG-3: confirm 112×112 ArcFace-template input; who normalises pixels (proposal: SG-3).
3. SG-4: do you want `quality` to select the best crop per track?
