# SG3 — Student Face Recognition (Automatic Class Attendance, Group 2)

**Members:** Muhammad Fouzan Yasin (468915), Fatima Tuz Zahra (478087)  — swapped in from SG2 with Huzyepha Javaid
**Branch:** `SG3` **Folder:** `module_sg3/`

## 1. Purpose
Given a face, decide **which enrolled student it is**, or **UNKNOWN** if the person is not enrolled.
Pipeline position: `SG1 Detection → SG2 Quality/Alignment → **SG3 Recognition** → SG4 Tracking → SG5 Attendance`.

## 2. Methods compared (Week 5 experiment)
| | Method | How it works | Score | KNOWN if |
|---|---|---|---|---|
| **Baseline** | **LBPH** (OpenCV `cv2.face`) | Classical texture histograms of a 112×112 grey face | LBPH distance (lower = more similar) | distance ≤ threshold |
| **Alternative** | **SFace** (OpenCV Zoo, ONNX, 128-D deep embedding) | Face is aligned with 5 landmarks, CNN gives an embedding, compared with cosine similarity | cosine similarity (higher = more similar) | similarity ≥ threshold |

Both use the same **YuNet** detector and the same test set, so the comparison is fair.
Parameter study: the KNOWN/UNKNOWN threshold is swept for both methods (values in `config.json`).

## 3. Dependencies
Python 3.9–3.12, and:
```
pip install -r requirements.txt
python download_models.py      # downloads YuNet (0.2 MB) + SFace (37 MB) into models/
```
> If `cv2.face` is missing: `pip uninstall -y opencv-python` then `pip install opencv-contrib-python`.

## 4. Test data layout (photos are NOT committed — see `.gitignore`)
```
data/
  enroll/<student_id>/*.jpg        2–5 clear photos per enrolled student
  test/known/<student_id>/*.jpg    DIFFERENT photos of the same enrolled students
  test/unknown/*.jpg               photos of people who are NOT enrolled
```
Folder name = student ID (e.g. `468915`). Raw photos are fine; the code detects and crops the face itself.
Use photos with consent; keep at least ~5 students enrolled and ~5 unknown people for meaningful numbers.

## 5. How to run
```
python enroll.py --method all                     # builds gallery/ for both methods
python recognize.py --method sface --input data/test/known/468915/1.jpg
python recognize.py --method lbph  --input data/test/unknown
python evaluate.py                                # Week 5 comparison -> results/
```
Add `--aligned` to `enroll.py` / `recognize.py` if inputs are already face crops from SG2.

## 6. I/O interface (V1)
**Input:** BGR image — raw frame/photo, or a face crop from SG2 (`--aligned`).
**Output (one JSON object per face):**
```json
{
  "image": "1.jpg",
  "method": "sface",
  "decision": "KNOWN",              // KNOWN | UNKNOWN | NO_FACE
  "student_id": "468915",           // "UNKNOWN" if rejected, null if NO_FACE
  "best_match": "468915",           // closest student even if rejected
  "score": 0.6123,
  "score_type": "cosine_similarity",// or "lbph_distance"
  "threshold": 0.363,
  "bbox": [210, 95, 180, 230],      // x, y, w, h in original-image pixels
  "time_ms": 41.2
}
```
SG4/SG5 should use `student_id` + `score` (+ `bbox` for association).

## 7. Metrics (from Week 3)
- **Accuracy** = (correct accepts + correct rejects) / all test images
- **FAR** = unknown people accepted as a student / all unknown images
- **FRR** = enrolled students rejected as UNKNOWN / all known images
- **Mis-ID rate** = enrolled student accepted with the wrong ID / all known images
- **Rank-1** = best match correct ignoring the threshold (closed-set accuracy)
- Avg time per image (ms)

## 8. Results
See `results/summary.md`, `results/far_frr_plot.png`, `results/sweep_*.csv`, `results/errors_*.csv`
(regenerate any time with `python evaluate.py`).

## 9. Current limitations
- Small, self-collected test set; numbers are preliminary.
- One face per image (largest face) in this version; classroom frames with many faces come via SG1/SG4.
- LBPH is sensitive to lighting and pose; SFace is heavier (~37 MB model) but still CPU-real-time.
- Thresholds must be re-tuned on the final class data and on the Jetson camera.
