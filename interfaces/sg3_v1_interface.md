# SG3 V1 callable recognition interface

Status: implemented SG3 boundary, **proposed downstream contract pending SG4 agreement**.
SG4 currently contains an LFW identification stand-in, with no tracker or SG3
ingestion API. The mock described here validates evidence only; it is not SG4
integration, persistent identity tracking, or an attendance decision.

## Python API

Run from the repository root:

```python
from module_sg3.sg3_core import load_config, make_recognizer
from module_sg3.api import RecognitionService
from module_sg3.pipeline import YuNetSG2Pipeline
from module_sg3.mock_sg4 import consume_result

cfg = load_config()
recognizer = make_recognizer("sface", cfg)
recognizer.load()                 # requires an enrolled local gallery
service = RecognitionService(recognizer)

# Direct incoming SG2 packet: these four fields are sufficient.
result = service.recognize_face({
    "frame_id": 42,
    "face_id": 2,                 # detection identifier within frame 42
    "aligned_face": face112_bgr,  # numpy uint8, shape (112,112,3), BGR
    "quality_pass": True,
})
# result["face_id"] == result["detection_id"] == 2
# result["bbox"] and result["timestamp"] are None unless supplied.

# Existing repository SG2 pipeline results also remain supported.
pipeline = YuNetSG2Pipeline(cfg)  # actual unchanged SG2 baseline; temporary YuNet detector
packet = pipeline.process(frame_bgr, frame_id=42, timestamp=1.4)
evidence = [service.recognize_face(face) for face in packet["results"]]
mock_records = [consume_result(result) for result in evidence]
```

`RecognitionService(recognizer).recognize_face(sg2_result, metadata=None)` consumes
one SG2 packet or legacy SG2 result. The minimal four-field incoming packet
`{frame_id, face_id, aligned_face, quality_pass}` is valid directly. The optional
metadata mapping can provide source boxes and timestamps separately; an incoming
`face_id` cannot be changed by metadata. Existing SFace and LBPH algorithms are
reused, with validation and persistence hardening in SG3.
The service captures its recognition threshold at construction; reconstruct the
service after changing the configuration/threshold.

## SG2 input contract

| Incoming packet field | Equivalent legacy SG2 field | Meaning |
|---|---|---|
| `quality_pass` | `accept` | Boolean quality decision |
| `aligned_face` | `face` | Accepted image: **112×112×3 uint8 BGR**, already aligned |
| `face_id` | `detection_id` | Frame-local detection identifier |
| `frame_id` | `frame_id` | Source frame identifier |

The incoming packet needs no `quality`, `reason`, `metrics`, `bbox` or
`timestamp` fields. Missing optional quality/geometry/time becomes `None`, and
missing metrics becomes `{}`. No coordinates or timestamps are inferred from
the 112px face crop. The older `accept`/`face` payload used by the repository
pipeline continues to work.

Duplicate aliases must agree: contradictory `quality_pass`/`accept`,
`aligned_face`/`face`, or `face_id`/`detection_id` returns `ERROR`. Accepted
payloads need a complete quality/image pair (`quality_pass`+`aligned_face` or
`accept`+`face`); crossed partial pairs are malformed. When both image aliases
are present, their shape, dtype and pixel values must match.

The legacy fields and optional diagnostics are:

| Field | Type / meaning |
|---|---|
| `accept` | Boolean; only accepted images reach recognition |
| `face` | Accepted: `numpy.ndarray`, exactly **112×112×3 uint8 BGR**; rejected: `None` |
| `reason` | Accepted: `None`; rejected: SG2 reason string |
| `quality` | Finite float in [0,1], heuristic quality; optional `None` |
| `metrics` | JSON-shaped dictionary of finite values, such as blur, brightness, yaw |

Pixels are aligned to SG2's five-point ArcFace template, without external
mean/std or RGB conversion. SFace's OpenCV wrapper performs its own model input
handling. Do not realign an already accepted SG2 face.

The adapter calls `module_sg2.sg2_quality.process_frame` with the original BGR
frame and original-coordinate detections. SG2 does not retain boxes/timestamps;
the adapter joins those using each detection's position. The temporary all-faces
YuNet detector retains all five landmarks. Its row order is already the order of
SG2's template: OpenCV's [alignment source](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/face_recognize.cpp)
consumes `row[4:14]` directly with the same template.

## Association metadata

| Field | Type / convention |
|---|---|
| `frame_id` | Integer, string, or `None`; source frame identifier |
| `detection_id` | Integer, string, or `None`; unique **within this frame**, never a track ID |
| `face_id` | Same value as `detection_id`; preserved incoming packet identifier, not an enrolled identity or track ID |
| `timestamp` | Finite nonnegative float, seconds from source video start, or `None` |
| `bbox` | `[x,y,width,height]`, finite original-frame pixels, positive dimensions, or `None` |

Origin is top-left; x increases right and y increases down. Boxes may cross frame
boundaries and retain subpixel precision; consumers should clip only for drawing.
Metadata is optional for recognition. An integrator may send geometry separately
to SG4 and correlate it with SG3 evidence using `(frame_id, face_id)` (equivalently
`(frame_id, detection_id)`). SG4 still needs geometry/timing for whichever tracking
algorithm it eventually implements; the minimal packet alone does not supply it.

## SG3 output

All results are JSON-safe dictionaries with `schema_version="sg3.v1"` and the
association fields above. `face_id` always equals `detection_id`. Neither image
arrays nor embeddings are returned.

| Field | Type / meaning |
|---|---|
| `status` | `RECOGNIZED`, `UNKNOWN`, `QUALITY_REJECTED`, or `ERROR` |
| `method` | `sface` or `lbph` |
| `student_id` | Enrolled pseudonymous ID when recognized; literal `UNKNOWN` for identity rejection; `None` for quality rejection/error |
| `best_match` | Closest enrolled pseudonym for recognized/UNKNOWN results; otherwise `None` |
| `score` | Full-precision finite raw matching measurement; otherwise `None` |
| `score_type` | `cosine_similarity` or `lbph_distance` |
| `threshold` | Finite threshold actually used by this service |
| `quality` | Preserved SG2 quality or `None` |
| `quality_metrics` | Preserved finite JSON-safe SG2 metrics |
| `reason` | `None` on recognition; SG2 quality reason, threshold reason, or failure code |
| `recognition_ms` | Nonnegative elapsed API time in milliseconds; excludes detection and SG2 |

SFace cosine similarity has range **[-1,1]**, higher is more similar, and identity
is accepted at `score >= threshold`. LBPH distance is nonnegative, lower is more
similar, and accepted at `score <= threshold`. Neither score is a calibrated
probability or verified confidence. A cosine threshold adopted from an external
benchmark is provisional until independently validated on the intended data.

## Errors, rejection and empty input

- `quality_pass=False` (or legacy `accept=False`): `QUALITY_REJECTED`;
  recognition is never called. Image data can be `None` or omitted on rejection.
  Missing optional reason becomes `quality_rejected`.
- Malformed SG2 fields, invalid image shape/type, nonfinite metadata/metrics:
  `ERROR`, reason `invalid_input`; recognition is never called.
- SG2 adapter exception: `ERROR`, reason `preprocessing_failure`.
- Empty/invalid gallery: `ERROR`, reason `empty_or_invalid_gallery` (or
  `recognition_failure` if the underlying implementation raises).
- Inference exceptions or invalid model scores: `ERROR`, reason
  `recognition_failure`. Errors never become evidence of UNKNOWN identity.
- Adequate input, insufficient match: `UNKNOWN`, reason `below_threshold`
  (SFace) or `above_threshold` (LBPH).
- No detections: pipeline `results=[]`. No fabricated face/identity packet.
- SG2 may omit rejected detections from its incoming packet stream. An omitted
  face produces no SG3 result; omission is **not** an UNKNOWN identity decision.
  If rejected packets are sent, their identifiers survive in rejection results.
  An integrator can preserve all SG1 detection geometry separately so SG4 can
  associate detections even when there is no fresh SG3 identity evidence.
- Bad full frame, malformed detector output or invalid frame metadata: the
  pipeline raises `ValueError`; callers should record this as a frame failure.
- Constructor configuration/model-loading errors raise normally.

`pipeline.process` returns raw SG2 dictionaries (including private image arrays),
`detections`, `frame_id`, `timestamp`, `detector="temporary_sg3_yunet"`,
`detection_ms`, and `preprocessing_ms`. Only pass `RecognitionService` outputs to
JSON logging or downstream consumers. `pipeline.detect(frame)` returns SG2
detection dictionaries with full-frame `box`, `det_conf` and `landmarks`.

## Mock downstream consumer and validation

`module_sg3.mock_sg4.consume_result(result)` checks schema, finite values, score
range, threshold/identity consistency and association geometry. It returns
`bbox_xyxy=[x,y,x+width,y+height]`, `id`, score evidence and `tracking=False`.
`consume_frame([])` returns `[]`; malformed evidence raises `ValueError`.

Run boundary tests with:

```bash
python -m unittest discover -s module_sg3/tests -v
python -m unittest discover -s module_sg2/tests -v
```

Tests use synthetic images and recognizer stubs to check both packet shapes,
alias contradictions and control flow, along with
the actual SG2 alignment and quality implementation. Their passing is contract
evidence, not recognition accuracy. Real-data evaluation is a separate experiment.
See [example output](examples/sg3_example_output.json).
