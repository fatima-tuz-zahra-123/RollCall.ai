# Complete class enrollment — 26 October 2026

The new named dataset has been processed locally and the default SG3 gallery now
contains **all 19 students**, with **21 normalized 128-dimensional SFace
embeddings**. The matching LBPH comparison model was also regenerated. No
students were withheld as UNKNOWN in this operating gallery.

## Executed enrollment

| Measurement | Actual result |
|---|---:|
| Supplied HEIC photos / named student folders | 93 / 19 |
| Folder sizes | 18 folders with five photos; one with three |
| Fully decoded photos | 93 |
| Exact duplicate files | 0 |
| Perceptual-similarity candidate groups, flagged only | 7 |
| Images submitted to actual SG2 baseline | 93 |
| Accepted frontal / side images | 19 / 2 |
| Pose-rejected images | 72 |
| Students enrolled / missing | 19 / 0 |
| SFace templates | 21 |
| Templates per student | 17 students with one; two students with two |
| SFace gallery size, excluding ONNX model and metadata | 14,790 bytes |
| SFace fitting / embedding extraction, including first inference | 348.65 ms total |
| SFace extraction mean / median | 16.59 / 13.69 ms per accepted face |
| Saved-template reload checks | All 21 passed for each method |

The first filename in each student folder was marked frontal from the user's
declared capture order; this also agrees with file-number/capture-time ordering.
Other images were marked `side_unspecified`, without guessing exact left/right
labels from facial appearance. All images are 3024×4032 after orientation and
come from the same capture session on 15 October 2026.

**Why 21 embeddings instead of 93:** the existing SG2 baseline rejects strong
side views using its yaw-proxy limit of 0.30. All 19 frontal images pass, but only
two of 74 side images pass. No quality gates were weakened to include additional
templates. Rejected files remain preserved and their measurements are recorded.
This establishes roster coverage, not robust recognition at all head angles.

## Saved artifacts

The active path is `module_sg3/gallery_class/`, selected by `config.json`:

- `sface_gallery.npz`: student pseudonym → N×128 float32, unit-normalized vectors.
- `lbph_model.yml` and `lbph_labels.json`: comparison model on the same 21 crops.
- `student_lookup.private.json`: private pseudonym-to-supplied-name mapping.
- `roster.private.json`: per-student image/template counts and rejection reasons.
- `provenance.private.json`: source/model/config hashes and each template's
  source image, quality measurements and gallery-row linkage.

Private detailed reports are in `module_sg3/private_runs/class_enrollment/`:
`image_decisions.private.json`, `enrollment_summary.json`,
`sg2_packet_smoke.private.json`, and the previous default configuration backup.
The earlier four-student gallery and its experimental outputs were preserved.
New gallery/report replacement requires `--replace` and retains backups.

## Code changes and SG2 packet validation

- Preparation now has an explicit `enrollment_only` mode: every labeled student
  and session is enrolled, with no UNKNOWN holdout. Near-duplicate candidates are
  reported without automatic exclusion; exact duplicates remain checked.
- New `enroll_dataset.py` stages files, verifies every saved template and match
  after reload, and publishes only after successful verification. Optional
  `--require-all-students` protects complete-roster enrollment.
- `RecognitionService.recognize_face` directly accepts
  `{frame_id, face_id, aligned_face, quality_pass}`. Legacy SG2 packets also work.
  Missing geometry/time remains absent; conflicting aliases fail as ERROR.
- Experimental galleries now live under their experiment output directory,
  protecting the active class gallery from evaluation enrollment.

All 19 frontal images were replayed through actual SFace recognition using only
the agreed four-field packet. IDs were preserved, all returned their enrolled
identity, and no bounding box/timestamp was fabricated. A failed-quality packet
returned QUALITY_REJECTED without matching. **These are enrollment replay and
interface checks, not independent recognition accuracy.**

**83 automated tests passed**, including the unchanged SG2 tests, packet aliases,
quality-failure behavior, all-student preparation, staging/rollback, multiple
templates, and separation of experimental and operating galleries.

## CCTV regression and limitations

The new default gallery was used for the same six CCTV timestamps, 600 through
2100 seconds at 300-second intervals. There were 215 face detections: 18 accepted
by SG2 and 197 quality-rejected. Outputs were one RECOGNIZED, 17 UNKNOWN and zero
recognition errors. All 215 result packets passed the mock downstream contract,
including matching face_id/detection_id; the saved six-frame AVI decodes fully.
Outputs remain private under `module_sg3/private_runs/class_video/`.

The sampled processing loop took 3.43 seconds including seeks/output, and mean
recognition was 17.49 ms per accepted face. These are PC measurements, not live
camera throughput or Jetson benchmarks. CCTV identities have no ground truth,
so the one matching decision is unverified and supplies no accuracy claim.

Forty-eight photos overlap the previous archive, including all five old UNKNOWN
test photos. The previous UNKNOWN FAR is therefore **not applicable** to this
fully enrolled gallery. The raw cosine threshold stays at the provisional
0.363; no new threshold calibration, independent accuracy or FAR/FRR estimate was
performed. Separate capture sessions and independently held-out probes are
needed for those measurements. SG4 tracking and SG5 attendance remain outside
this enrollment task.

All original images and the ZIP were preserved. Templates, label mappings,
converted copies, detailed logs, staging/backups and annotated videos are ignored
by Git. Only the aggregate report contains shareable enrollment counts.

See [run instructions](../README.md) and [the SG2/SG3 contract](../../interfaces/sg3_v1_interface.md).
