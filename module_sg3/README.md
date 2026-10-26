# SG3 — Student Face Recognition midterm prototype

SG3 consumes accepted SG2 face crops and returns an enrolled pseudonymous student
ID or `UNKNOWN`. SFace is the primary embedding baseline; the existing LBPH
implementation provides a comparison. The implementation preserves SG2 and SG4
ownership: it invokes the actual SG2 quality/alignment code and validates SG3
evidence with a small mock downstream consumer. Persistent tracking, attendance
decisions, and Jetson deployment remain separate integration work.

The active classroom gallery now contains **19 students and 21 SFace templates**
from the newly supplied named dataset. All 93 photos were processed: all 19
frontal photos and two side views passed SG2; the other 72 failed its unchanged
pose gate. The default `config.json` loads `gallery_class/`. See the
[current enrollment report](results/CLASS_ENROLLMENT.md). No students were
withheld for UNKNOWN testing in this gallery.

Read the [initial repository assessment](ASSESSMENT.md),
[callable V1 contract](../interfaces/sg3_v1_interface.md), and
[executed experiment report](results/REPORT.md). The report records actual runs,
limitations, and the development PC environment. Existing root-level evaluation
files are historical snapshot results and are not measurements of this dataset.

## Rebuild the complete class gallery

The new archive was extracted without changing original files into
`../private_data/Cv_Dataset`. From the repository root:

```bash
python -m module_sg3.prepare_dataset --source "../private_data/Cv_Dataset" --output module_sg3/private_data/cv_prepared --mode enrollment_only --first-image-front
python -m module_sg3.enroll_dataset --manifest module_sg3/private_data/cv_prepared/manifest.json --gallery-dir module_sg3/gallery_class --output-dir module_sg3/private_runs/class_enrollment --methods sface lbph --require-all-students
```

The gallery already exists in this workspace. To intentionally regenerate it,
add `--replace` to the enrollment command; prior gallery/report directories are
preserved as timestamped backups. New outputs are staged, every saved template
is verified after reload, and only then published. `--require-all-students`
refuses to publish a gallery missing any declared student.

`enrollment_only` includes every labeled student and capture session; it does not
create a test split. Exact duplicates are excluded; perceptual-similarity groups
are flagged for review rather than automatically removing students with similar
backgrounds. The first sorted filename per student is tagged frontal using the
user's capture-order declaration. Specific left/right labels are not inferred.

Private outputs include `gallery_class/sface_gallery.npz`, the LBPH files,
`student_lookup.private.json`, `roster.private.json`, and
`provenance.private.json`. Detailed image decisions and a sanitized aggregate
summary are under `private_runs/class_enrollment/`. The old four-student gallery
is preserved separately in `gallery/`.

The agreed minimal SG2 packet works directly:

```python
from module_sg3.sg3_core import load_config, make_recognizer
from module_sg3.api import RecognitionService

recognizer = make_recognizer("sface", load_config())
recognizer.load()  # current complete class gallery
service = RecognitionService(recognizer)
result = service.recognize_face({
    "frame_id": 150,
    "face_id": 1,
    "aligned_face": face_image,  # 112x112x3 uint8 BGR
    "quality_pass": True,
})
```

`face_id` is preserved alongside the canonical `detection_id`. Missing
`timestamp`, `bbox` and quality measurements stay `None`/empty; none is invented.
If SG2 sends a failed-quality packet, it is rejected without running the model.
If SG2 omits rejected detections, SG3 creates no observation for them. The older
SG2 `accept`/`face` shape remains supported.

The sections below describe the original dataset's evaluation protocol as well
as general setup. New experimental runs save galleries inside their own output
directory's `gallery/`, so experiments do not replace the active class gallery.
The earlier UNKNOWN test photos overlap this new enrollment dataset, and their
old FAR does not validate the new gallery.

## Install from the repository root

This workspace already has a prepared `.venv`, downloaded models, private
galleries and executed outputs. From the parent workspace, enter
`RollCall.ai-main` and use `.\.venv\Scripts\python.exe` for the commands below.
The installation steps are for reproducing the environment on another machine.

Use a normal CPython **3.10–3.12** environment. Install one OpenCV distribution:
`opencv-contrib-python`, which supplies both SFace/YuNet and `cv2.face` for LBPH.
Installing `opencv-python` or a second OpenCV wheel in the same environment can
replace the shared `cv2` package. Use a fresh virtual environment if one already
contains conflicting OpenCV packages.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r module_sg3/requirements-dev.txt
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
python --version
```

The PATH change lasts for that shell. If the Python launcher is unavailable,
create the environment with an installed Python 3.10–3.12 executable instead.

Ubuntu 22.04:

```bash
sudo apt-get update
sudo apt-get install python3-venv libgl1 libglib2.0-0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r module_sg3/requirements-dev.txt
```

`requirements.txt` contains runtime dependencies; `requirements-dev.txt` adds
pytest. The exact tested Windows resolution is saved in
[requirements-lock-windows.txt](requirements-lock-windows.txt); it is not an
ARM/Ubuntu lock file. HEIC decoding uses Pillow and pillow-heif. All recognition and image
processing run locally in OpenCV; there is no recognition service or dataset
upload. The downloader below retrieves only public model weights.

```bash
python -m module_sg3.download_models
```

The downloader uses [models_manifest.json](models_manifest.json) to verify the
expected byte count and SHA-256 of both YuNet and SFace. Existing files are also
verified; partial or mismatched downloads are not accepted. The manifest records
the documented OpenCV Zoo download sources and content hashes. Models live in
`module_sg3/models/` and are excluded from Git.

## Prepare the private dataset

The supplied dataset ZIP was extracted locally into
`../private_data/Class Dataset` beside the repository. Originals keep their
names and contents. To repeat the preparation after extraction:

```bash
python -m module_sg3.prepare_dataset --source "../private_data/Class Dataset" --output module_sg3/private_data/prepared --unknown-students 2
```

The supplied data contains **52 HEIC photographs**, including **23 photos across
10 labeled student folders** and **29 unlabeled photos**. All decoded at
3024×4032 pixels after orientation. Each student has 1–4 photos; two students
have only one. All photos have the same EXIF capture day and camera session.

Preparation assigns stable folder-label hash pseudonyms, applies EXIF
orientation, saves private JPEG copies, and writes a reproducible manifest.
It never assigns a label by facial appearance. Unlabeled photographs remain
unlabeled and provide no verified UNKNOWN ground truth.

The split holds out two complete labeled identities, each with at least two
usable photos, for UNKNOWN evaluation. For remaining identities, only another
capture session can become a known-person evaluation set. Missing EXIF is
conservatively treated as insufficient evidence of session separation. With
this supplied dataset there are **no independent known-person evaluation
images**. Enrollment replay shows that the enrolled system runs; it does not
measure generalization accuracy or false rejection on new captures.

The executed preparation produced:

| Role | Images |
|---|---:|
| Enrollment, eight identities before SG2 quality gates | 16 |
| UNKNOWN evaluation, two withheld identities | 5 |
| Independent known evaluation | 0 |
| Unlabeled, usable for inspection only | 26 |
| Conservatively excluded similarity candidates | 5 |

SHA-256, decoded-pixel hashes, and a 64-bit dHash screen group duplicate or
near-duplicate candidates before splitting. A dHash Hamming distance ≤5 is a
conservative heuristic, not proof of the same person or capture. It can produce
false positives. A same-label group retains one canonical image; groups that
span different labels or labeled/unlabeled images are excluded for review.
No original is deleted. The prepared data had zero exact-byte duplicates and
three conservative similarity candidate groups.

The private manifest contains source paths, the label-to-pseudonym mapping,
hashes, capture sessions, pose hints from supplied filenames, conversion errors,
and split roles. `image_path` is relative to the manifest directory; `source` is
relative to private `source_root`. `audit_summary.json` omits names and source
paths. Both files are reproducible outputs of [prepare_dataset.py](prepare_dataset.py).

## Enroll, reload, and compare both methods

```bash
python -m module_sg3.experiment --manifest module_sg3/private_data/prepared/manifest.json --output-dir module_sg3/private_runs/experiment --max-side 1280
```

This command runs temporary YuNet detection and the unchanged SG2 baseline,
selects the dominant face in each labeled photograph, enrolls both methods, saves
their galleries, reloads them, and verifies persistence with a repeated match.
Ambiguous photographs with similarly sized faces are recorded as failures rather
than assigned the folder label automatically. SG2-rejected images never enter
enrollment or matching. Both methods receive exactly the same accepted
112×112×3 `uint8` BGR aligned crops.

SFace reuses the existing 128-dimensional L2-normalized embedding implementation
and maximum cosine similarity over each student's templates. LBPH reuses the
existing grayscale/equalized local texture implementation. The initial
thresholds remain **SFace 0.363** and **LBPH 80**, from [config.json](config.json).
Cosine similarity lies in [-1,1] and higher is better; LBPH distance is
nonnegative and lower is better. Neither score is a probability or calibrated
recognition confidence. The small threshold sweep is descriptive and does not
establish a production threshold.

Outputs:

| Location | Contents |
|---|---|
| `module_sg3/private_runs/experiment/gallery/sface_gallery.npz` | Separate experimental student embeddings |
| `module_sg3/private_runs/experiment/gallery/lbph_model.yml`, `lbph_labels.json` | Separate experimental LBPH model and pseudonyms |
| `module_sg3/private_runs/experiment/gallery/provenance.private.json` | Experimental input/model/config hashes, sample IDs, timings, reload verification |
| `module_sg3/private_runs/experiment/aggregate_results.json` | Genuine aggregate evaluation, enrollment, latency, runtime and limitation data |
| `module_sg3/private_runs/experiment/*_records.private.json` | Per-sample evidence and statuses |
| `module_sg3/private_runs/experiment/thresholds.png` | Held-out UNKNOWN FAR and explicitly labeled enrollment replay curves |

Undefined metrics are JSON `null`, not zero. Recognition metrics condition on
scored faces, while quality/failure counts and pipeline acceptance are reported
separately. Independent known-person accuracy and FRR are unavailable for this
dataset. Synthetic blur, darkening and lower-face occlusion are diagnostic
transformations of enrollment data, not independent test samples. Pose hints
describe supplied filenames and are not inferred viewing-angle ground truth.

For a shorter single-method run, add `--methods sface`. `--config` selects an SG3
configuration; relative model/gallery paths resolve from that configuration
file's directory (the default is `module_sg3`). `--sg2-config` selects an existing SG2 config
without editing SG2. Preserve the selected config and provenance with each run.

## CCTV frame-level demonstration

The supplied MP4 is **2560×1440, 25 fps, approximately 69 min 57.4 sec**, encoded
with HEVC. The bounded demonstration samples six frames at five-minute intervals
starting ten minutes into the video:

```bash
python -m module_sg3.video_demo --input "../private_data/Class Dataset/CR # 28 - Thursday 8 Oct 2026 - 1155 to 1305.mp4" --start 600 --interval 300 --max-frames 6 --max-side 1280 --output-dir module_sg3/private_runs/video
```

YuNet is an explicitly temporary SG3 all-faces detector because SG1 does not yet
provide the required complete reusable landmark contract. The adapter passes
original-frame boxes and five landmarks to the actual SG2 implementation, then
preserves boxes, frame IDs, detection IDs and source timestamps through SG3.
The maximum side limits detector inference resolution; coordinates remain in
the original frame. A detection ID applies within one frame and is never a track
ID.

The private output directory contains `annotated_samples.avi`,
`frame_results.jsonl`, and `summary.json`. The script reopens the generated AVI
and checks saved frame counts and JSON records. The AVI is a time-compressed
sampled preview, not continuous processed CCTV. Labels distinguish recognized,
UNKNOWN, quality-rejected, and failed faces. Stage timings include decode/seek,
detection, SG2 preprocessing, recognition, annotation and writing.

Timestamp reconstruction assumes constant frame rate. All six sampled source
timestamps in this supplied MP4 were checked against decoder timestamps and
matched exactly. Variable-frame-rate input has not been validated.

Use another output directory for another run, or explicitly add `--overwrite`
to replace generated demonstration outputs. The original video is not changed.
Without labeled CCTV identity ground truth, counts and runtime are observable
results; CCTV identification accuracy is unavailable. These results are
**frame-level recognition**, with no persistent tracking or attendance output.

## Callable interface and tests

Applications can load the gallery once and consume SG2 results directly:

```python
from module_sg3.sg3_core import load_config, make_recognizer
from module_sg3.api import RecognitionService

recognizer = make_recognizer("sface", load_config())
recognizer.load()
service = RecognitionService(recognizer)
result = service.recognize_face(sg2_result, {
    "frame_id": 42,
    "detection_id": 0,
    "timestamp": 1.68,
    "bbox": [210, 95, 180, 230],
})
```

`sg2_result` is the actual SG2 dictionary containing `accept`, `reason`,
`quality`, `metrics`, and `face`. An accepted face must be exactly
112×112×3 `uint8` BGR. Do not convert it to RGB, divide pixels by 255, or align
it again. See the [V1 contract](../interfaces/sg3_v1_interface.md) and
[example output](../interfaces/examples/sg3_example_output.json) for complete
types, score conventions and error behavior.

Statuses are `RECOGNIZED`, `UNKNOWN`, `QUALITY_REJECTED`, and `ERROR`.
Quality rejection skips recognition; malformed input or inference failure
becomes `ERROR`, never a forced identity or evidence of an unknown person.
An empty detection list produces an empty results list. Boxes use
`[x,y,width,height]` in original-frame pixels, with top-left origin and seconds
from source-video start for timestamps.

```bash
python -m pytest module_sg3/tests module_sg2/tests -q --basetemp module_sg3/private_data/pytest
```

For restricted Windows environments, also set the standard-library temporary
directory before running the tests:

```powershell
$env:TEMP = Join-Path $PWD "module_sg3/private_data/test_temp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
python -m pytest module_sg3/tests module_sg2/tests -q --basetemp module_sg3/private_data/pytest
```

These settings keep both pytest and standard-library synthetic test files within
the workspace. Tests exercise malformed
and rejected inputs, empty galleries, metadata preservation, JSON output,
duplicate/session split integrity, model persistence, actual SG2 behavior, and
the mock downstream consumer. Test stubs establish control-flow/contract
behavior; real recognition measurements are in the experiment outputs.

Legacy `enroll.py` and `recognize.py` remain available for their original folder
layout. The manifest-based experiment above is the reproducible main path for
this supplied HEIC dataset and SG2-compatible comparison. The historical root
`evaluate.py` is retained; its outputs should not be mixed with the new run.

## Five-minute midterm sequence

Complete installation, model downloads and preparation before presenting. In
the first shell, run the experiment and show gallery reload verification,
recognized enrollment replay, rejection of a whole held-out identity, and the
actual threshold/timing results. Explain that replay is a functional check and
that independent known-person accuracy is unavailable.

In a second shell using the same virtual environment, run the bounded CCTV
command and open the private AVI locally. Show frame IDs/timestamps, the SG2
accept/reject counts, and a saved SG3 JSON record. Finish with the V1 contract,
passing tests, [executed report](results/REPORT.md), and integration blockers.
Keep the gallery consistent between shells; wait for enrollment to finish
before loading it in the video process.

## SG4, SG6 and project evidence

The current SG4 module contains an identification stand-in rather than a stable
persistent tracker or SG3 ingestion API. [mock_sg4.py](mock_sg4.py) checks SG3
evidence and converts box conventions; its use does not establish real SG4
integration. SG4 still needs to agree the contract, associate detections across
frames, and decide how to aggregate quality and identity evidence. SG5 owns the
attendance decision; SG3 produces no attendance records.

SG6 can import the same callable service and preload models/gallery once, but
must test this on the actual Jetson Orin Nano 8GB and camera. Development-PC CPU
timings are not Jetson measurements. The code does not assume CUDA or TensorRT;
installing OpenCV alone does not establish either acceleration path. On Ubuntu
22.04/aarch64, wheel availability may require a compatible OpenCV 4.8+ build
with contrib `face`, ONNX DNN, `FaceDetectorYN`, `FaceRecognizerSF`, and video
decoding support. Verify imports and rerun tests/benchmarks on the device before
claiming deployment readiness.

The provided repository is a ZIP snapshot without `.git` history. This work
does not create shared GitHub commits, branch/review history, project-board
evidence, camera validation, or Jetson readiness evidence. Those shared rubric
items need the team repository and actual hardware workflow.

Student photographs, originals, label mappings, derived crops, embeddings,
per-face logs and annotated CCTV are private biometric data. Keep them local and
out of Git; `.gitignore` excludes the data/gallery/model/video run paths. Do not
publish or upload them with the demonstration. Only reviewed aggregate reports
and source/tests should be considered for the shared repository. No cloud model
inference or dataset transmission is part of these commands.
