# SG3 Face Recognition - Week 5 Results

Generated: 2026-10-07 16:13
Test set: 6 enrolled-student images, 3 non-enrolled (unknown) images.

## Method comparison at each method's best threshold

| Method | Score type | Best threshold | Accuracy | FAR | FRR | Mis-ID | Rank-1 (closed set) | No-face images | Avg time / image (ms) |
|---|---|---|---|---|---|---|---|---|---|
| lbph | lbph_distance | 90 | 0.778 | 0.000 | 0.000 | 0.333 | 0.667 | 0 | 61.1 |
| sface | cosine_similarity | 0.363 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 70.7 |

## Full threshold sweeps

### lbph

| Threshold | Accuracy | FAR | FRR | Mis-ID |
|---|---|---|---|---|
| 40 | 0.333 | 0.000 | 1.000 | 0.000 |
| 50 | 0.333 | 0.000 | 1.000 | 0.000 |
| 60 | 0.333 | 0.000 | 1.000 | 0.000 |
| 70 | 0.333 | 0.000 | 1.000 | 0.000 |
| 80 | 0.333 | 0.000 | 1.000 | 0.000 |
| 90 | 0.778 | 0.000 | 0.000 | 0.333 | **<- best**
| 100 | 0.667 | 0.333 | 0.000 | 0.333 |
| 110 | 0.444 | 1.000 | 0.000 | 0.333 |
| 120 | 0.444 | 1.000 | 0.000 | 0.333 |

### sface

| Threshold | Accuracy | FAR | FRR | Mis-ID |
|---|---|---|---|---|
| 0.2 | 0.889 | 0.333 | 0.000 | 0.000 |
| 0.25 | 0.889 | 0.333 | 0.000 | 0.000 |
| 0.3 | 0.889 | 0.333 | 0.000 | 0.000 |
| 0.363 | 1.000 | 0.000 | 0.000 | 0.000 | **<- best**
| 0.4 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.45 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.5 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.55 | 1.000 | 0.000 | 0.000 | 0.000 |
| 0.6 | 1.000 | 0.000 | 0.000 | 0.000 |

## Preliminary recommended operating point

**sface with threshold 0.363** (accuracy 1.000, FAR 0.000, FRR 0.000).

Definitions: FAR = non-enrolled faces accepted as a student; FRR = enrolled students rejected as UNKNOWN; Mis-ID = enrolled student accepted but given the wrong ID; Rank-1 = best match correct ignoring threshold.
