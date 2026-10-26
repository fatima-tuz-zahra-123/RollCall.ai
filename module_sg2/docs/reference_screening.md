# SG-2 Reference Implementation Feasibility Screening (Week 5)

> Course PDF Week 5 §3.4–3.6. Screened by SG-2 (Syed Ali Mehdi Jaffari, Fatima Tuz Zahra), 2026-10-06.
> Rule from the PDF: *an LLM is a search assistant, not the evidence source; every claim must be opened and verified; unknowns are marked **Not Verified**.* Every ✅ below was opened on 2026-10-06 (repo page / model card / GitHub API). "Executed" means **we ran it**.

## 1. Weights used (PDF §3.4, 100 points)
End-to-end 15 · Python/source 15 · Jetson/edge evidence 15 · Accuracy evidence 10 · Reproducibility 10 · Pretrained weights 10 · Real-time feasibility 10 · Dataset 5 · V1 interface fit 5 · Repo maturity/licence 5

## 2. Candidate screening table (PDF §3.6 template)
| Item | **A. OpenCV Zoo pipeline** (YuNet 5-pt → similarity align → eDifFIQA(T)) | **B. InsightFace python-package** (SCRFD + `norm_crop`) | **C. CR-FIQA** (CVPR 2023) | **D. MagFace** (CVPR 2021) | **E. OFIQ** (ISO/IEC 29794-5 ref.) | **F. SER-FIQ** (CVPR 2020) |
|---|---|---|---|---|---|---|
| Repository | github.com/opencv/opencv_zoo, huggingface.co/opencv | github.com/deepinsight/insightface | github.com/fdbtrs/CR-FIQA | github.com/IrvingMeng/MagFace | github.com/BSI-OFIQ/OFIQ-Project | github.com/pterhoer/FaceImageQuality |
| Supporting paper | YuNet (Mach. Intell. Research 2023); eDifFIQA (IEEE TBIOM) | SCRFD/ArcFace papers | Boutros et al., CVPR 2023 | Meng et al., CVPR 2021 (oral) | ISO/IEC 29794-5; BSI report v1.2 (2024-11-06) | Terhörst et al., CVPR 2020 |
| End-to-end for SG-2? | ✅ detect→landmarks→align→quality score (demo does all) | Partial: detect + align ✅, **no quality score** | Quality only, needs a separate detector/aligner | Quality = embedding norm, needs a separate detector/aligner | ✅ full quality pipeline (C++ sample app) | Quality only |
| Python/source | ✅ Python + OpenCV DNN, ONNX | ✅ Python, ONNX Runtime, `pip install -U insightface` (Py ≥3.10) | ✅ PyTorch **1.7.1** (old) | ✅ PyTorch | ❌ **C/C++** (cmake ≥3.26 + conan 2.18.1); Python adapter mentioned, **Not Verified** | MXNet, **Python 3.7/3.8 only** |
| Jetson/edge evidence | ✅ YuNet + SFace benchmarked on "Jetson Nano Orin" in the zoo benchmark (YuNet 2.59 ms CPU / 5.23 ms GPU). eDifFIQA: **Not Verified** on Jetson | Generic CUDA via onnxruntime-gpu; **Jetson Not Verified** | **Not Verified** | **Not Verified** | Build doc is Ubuntu 22.04 **x86_64**; aarch64/Jetson **Not Verified** | **Not Verified** |
| Reported accuracy | YuNet WIDER val AP 0.884/0.866/0.750; SFace LFW 0.9940; eDifFIQA: SOTA-level EDC in paper | model-zoo tables (Not re-checked) | EDC/pAUC in paper | LFW/CFP/AgeDB/IJB in paper | ISO conformance tests report | EDC in paper |
| Reported speed | YuNet 0.69 ms (i7-12700K), SFace 5.09 ms | **Not Verified** | **Not Verified** (iResNet50/100 backbones = heavy) | **Not Verified** (iResNet100) | **Not Verified** | **Not Verified** |
| Pretrained weights | ✅ HF: YuNet 233 kB, SFace 38.7 MB, eDifFIQA(T) 7.27 MB | ✅ model packs (buffalo_l/s…), **non-commercial research only** | ✅ Google Drive (S = iResNet50, L = iResNet100) | ✅ Google/Baidu Drive | ✅ shipped via conan/models | ✅ Google Drive |
| Dataset/test data | ✅ LFW (we used it) | LFW etc. | LFW/IJB etc. | LFW/IJB etc. | ISO test set | LFW etc. |
| **Successfully cloned/downloaded?** | ✅ **Yes**: all 3 ONNX models downloaded | Not attempted | No (weights on Google Drive) | No | No | No |
| **Successfully executed?** | ✅ **Yes**: EXP-1/2/3 on 6,000 LFW pairs; our align = OpenCV `alignCrop` (99.33% vs 99.35%) | No | No | No | No | No |
| Major risk | eDifFIQA Jetson speed unknown; eDifFIQA licence CC-BY-4.0 (attribution) | non-commercial model licence; a second framework (onnxruntime) to install on the Jetson | CC BY-NC 4.0; PyTorch 1.7.1 pin; large backbone | heavy backbone; really an SG-3 model | passport/enrolment-oriented measures (head size, crop, background) don't fit classroom faces; C++ build on ARM | **obsolete stack** (MXNet, Py3.7/3.8) → hard-reject criterion |
| V1 interface fit | ✅ exact: 5 landmarks in, 112×112 aligned out | ✅ same template (`arcface_dst`) | ✅ takes our 112×112 output (RGB, [-1,1]) | ✅ takes our 112×112 output | ⚠ whole-portrait input; a different interface | ✅ |
| Licence | MIT (YuNet), Apache-2.0 (SFace), CC-BY-4.0 (eDifFIQA) | code MIT, models non-commercial | CC BY-NC 4.0 | Apache-2.0 | "Other" (see LICENSE.md) | CC BY-NC-SA 4.0 |

## 3. Weighted scores (our assessment; justification is in the row above)
| Criterion (weight) | A Zoo | B InsightFace | C CR-FIQA | D MagFace | E OFIQ | F SER-FIQ |
|---|---|---|---|---|---|---|
| End-to-end (15) | 15 | 10 | 7 | 7 | 12 | 6 |
| Python/source (15) | 15 | 15 | 11 | 13 | 4 | 6 |
| Jetson evidence (15) | 11 | 5 | 0 | 0 | 0 | 0 |
| Accuracy evidence (10) | 9 | 9 | 10 | 10 | 8 | 8 |
| Reproducibility (10) | 10 | 8 | 6 | 6 | 7 | 3 |
| Pretrained weights (10) | 10 | 8 | 8 | 8 | 9 | 7 |
| Real-time feasibility (10) | 9 | 6 | 3 | 3 | 3 | 2 |
| Dataset (5) | 5 | 5 | 5 | 5 | 3 | 5 |
| V1 interface (5) | 5 | 5 | 5 | 5 | 1 | 4 |
| Maturity/licence (5) | 5 | 4 | 3 | 4 | 5 | 1 |
| **Total /100** | **94** | **75** | **58** | **61** | **52** | **42** |
| **Decision** | **PRIMARY** | **BACKUP** (alignment/detector) | research reference | → hand to SG-3 as a recognition+quality option | rejected for V1 (interface mismatch, C++/ARM build risk) | **hard reject** (obsolete deps) |

**Primary = A (OpenCV Zoo).** It is the only candidate that is end-to-end for SG-2, Python/OpenCV-only, has published Jetson Orin timings, and was cloned **and executed** by us with a result matching the published accuracy.

**Backup = B (InsightFace).** Same alignment template and a strong SCRFD detector. Not executed yet. Its pretrained-model licence is non-commercial, which is fine for coursework.

## 4. Evidence progression (PDF: "Found → code → clone → runs on PC → reproduces sample → Jetson → speed → integrates V1")
| Step | A OpenCV Zoo |
|---|---|
| Found online | ✅ |
| Code available | ✅ |
| Clone/download succeeds | ✅ |
| Runs on development PC | ✅ |
| Reproduces sample result | ✅ SFace LFW: published 0.9940, ours **0.9933** |
| Runs on Jetson | ⏳ needs SG-6 / the Jetson. Zoo publishes Orin numbers for YuNet/SFace |
| Meets speed/memory | ⏳ Week 6 on the Jetson |
| Integrates with V1 | ✅ our V1 is built on its I/O |

## 5. For the team lead
The +5 bonus needs **≥3 ranked candidates for each relevant subgroup** (or a justified project-level shortlist). SG-2's part is above. Other SGs can reuse the method and this template; candidate A also covers SG-1 (YuNet) and SG-3 (SFace).
