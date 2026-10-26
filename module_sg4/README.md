# SG4 stand-in: face enrolment + identification (LFW)

## Purpose
Gives SG4 a working source of **face boxes + identity evidence** (known ID or UNKNOWN)
so the tracker can be developed before the real SG1 (detection) and SG3 (recognition)
modules are ready. It is a stand-in for those modules, not a replacement.

Pipeline: LFW photo -> MTCNN (face boxes) -> InceptionResnetV1 / VGGFace2 (512-d embedding)
-> per-person average "fingerprint" -> cosine similarity -> `S001..` ID or `UNKNOWN`.

## Dependencies
See `requirements.txt`. Tested design target: Google Colab (GPU runtime). On other machines
install `torch`/`torchvision` first, then `pip install facenet-pytorch --no-deps`.

## How to run
```bash
python face_id_lfw.py                      # defaults: 6 known, 3 unknown, threshold 0.60
python face_id_lfw.py --threshold 0.7 --n-enrolled 8 --seed 1
python face_id_lfw.py --show               # also open plot windows
```
In Colab: `!pip install facenet-pytorch --no-deps -q` then `!python face_id_lfw.py`.

## Input / output interface
- Input: an image (PIL).
- Per detected face: `box [x1,y1,x2,y2]` (pixels), detection confidence, `id`
  (`S001`... or `UNKNOWN`), `name`, similarity `score` (cosine, 0-1).
- Empty-output behaviour: no face detected -> no results (reported as `NO FACE` in tests).

## Test data
LFW (Labeled Faces in the Wild), downloaded automatically by scikit-learn
(people with >=10 photos). It is **not** committed. Known people use the first
`--enroll-imgs` photos for enrolment; their remaining photos are held out for testing;
`--n-unknown` other people are never enrolled.

## Outputs (written to `results/`, git-ignored)
Figures `01`-`08`, `gallery.npz`, `id_table.json`, `threshold_sweep.csv`, `metrics.json`.

## Current performance
Run the script and paste the numbers from `results/threshold_sweep.csv` here.

## Limitations
- LFW is mostly single, frontal, well-lit faces: easier than a real classroom.
- The multi-person test is a collage of separate photos, not a real group photo.
- MTCNN may be too slow for Jetson Orin Nano; SG1's detector should replace it at integration.
- This module does no tracking yet (tracking is the next SG4 step).
- Check licences of the pretrained weights before reuse outside coursework.
