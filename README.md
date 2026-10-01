AutoAttend-Orin: Real-Time Embedded Classroom Attendance System
📌 Executive Summary
AutoAttend-Orin is an end-to-end, edge-deployed computer vision pipeline designed for automated classroom attendance tracking. Engineered specifically for the NVIDIA Jetson Orin Nano 8GB, the system replaces manual roll calls with continuous multi-face detection, quality filtering, open-set identification, multi-person tracking, and temporal decision fusion.
Rather than relying on isolated single-frame classifications, AutoAttend-Orin aggregates visual evidence across video frames to eliminate false positives and ensure high-confidence attendance logging under variable lighting, poses, and occlusions.
🏛 System Architecture & Sub-Group Pipeline
The system is structured as a progressive multi-module pipeline, divided across 6 dedicated technical sub-groups (SG1–SG6):



                        [ Live Camera Feed ]
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  SG-1: Face Detection │  (Frame -> Face Bounding Boxes)
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ SG-2: Alignment & QA  │  (Face ROI -> Normalized Face / Reject)
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  SG-3: Recognition    │  (Normalized Face -> Student ID / Unknown)
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  SG-4: Multi-Tracking │  (Detections -> Persistent Track IDs)
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  SG-5: Decision Logic │  (Tracks/IDs -> Attendance Record)
                     └───────────┬───────────┘
                                 │
                                 ▼
                    [ Final Attendance Report ]

  ═══════════════════════════════════════════════════════════════════════
   SG-6: Embedded Deployment, Optimization & System Integration (Jetson)
  ═══════════════════════════════════════════════════════════════════════


Module Responsibilities
Sub-Group
Technical Focus
Core Inputs & Outputs
Key Evaluation Metrics
SG-1
Classroom Face Detection
Frame  Face Bounding Boxes + Confidence
mAP@0.5, Precision, Recall, Latency (ms)
SG-2
Quality, Alignment & Filtering
Face ROI  Quality Score + Aligned Face
Blur/Pose/Lighting Rejection Rate
SG-3
Open-Set Face Recognition
Aligned Face  Student ID / UNKNOWN
Rank-1 Accuracy, Open-Set FAR/FRR
SG-4
Multi-Person Tracking & Association
Detections + IDs  Persistent Tracks
MOTA, MOTP, ID Switches, FPS
SG-5
Attendance Decision Logic
Tracks + IDs + Time  Attendance Record
Precision/Recall against Ground Truth
SG-6
Jetson Integration & Profiling
All Modules  Optimized Pipeline
End-to-End FPS, Memory, Power (W)

📁 Repository Structure



AutoAttend-Orin/
├── interfaces/             # Frozen JSON/Pydantic schemas, contract specs, mock inputs/outputs
├── datasets/               # Dataset acquisition scripts, manifests, calibration targets (No raw images)
├── module_sg1/             # SG-1: Face Detection engine & scripts
├── module_sg2/             # SG-2: Quality assessment, blur check, face alignment
├── module_sg3/             # SG-3: Open-set identity matcher & enrollment DB interface
├── module_sg4/             # SG-4: Multi-object tracker (DeepSORT/ByteTrack adapters)
├── module_sg5/             # SG-5: Temporal decision fusion engine & report generator
├── embedded_sg6/           # SG-6: TensorRT export scripts, CUDA pipelines, GStreamer capture
├── integration/            # Multi-module pipeline wrappers & inter-module tests
├── evaluation/             # Benchmarking scripts (FPS, Latency, Accuracy, Confusion Matrix)
├── documentation/          # Architecture diagrams, lab reports, API specs
├── .gitignore
├── README.md
└── requirements.txt


⚡ Quick Start & Development Setup
1. Clone the Repository



Bash
git clone https://github.com/YOUR_TEAM/AutoAttend-Orin.git
cd AutoAttend-Orin


2. Set Up Virtual Environment (Host / Development PC)



Bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


3. Initialize Empty Directory Hooks (.gitkeep)



Bash
find . -type d -exec touch {}/.gitkeep \;


🛠 Interface Contracts & Testing
To ensure concurrent development, each downstream sub-group can run against mock interfaces before upstream modules are fully integrated.
Schemas are located in interfaces/schemas.py.
Mock data generators are available in interfaces/mock_data/.
To test module compliance against agreed contracts:



Bash
python3 -m unittest discover -s interfaces/tests


🎯 Target Platform & Constraints
Hardware: NVIDIA Jetson Orin Nano 8GB
Target OS: JetPack 5.x / 6.x (Ubuntu 22.04 LTS)
Target Performance: 15FPS end-to-end pipeline speed at 1080p video resolution.
Precision Target: TensorRT FP16/INT8 optimized engines.
🗓 Project Roadmap (14-Lab Execution Model)
[x] Lab 1–3 (Phase-I Exit Gate): System definition, literature review, frozen V1 interfaces.
[ ] Lab 4–7: Baseline algorithm implementations & quantitative parameter selection.
[ ] Lab 8 (Integration Gate 1): Inter-module integration with neighboring real code.
[ ] Lab 9–10: Jetson TensorRT conversion, GStreamer integration, performance profiling.
[ ] Lab 11 (Integration Gate 2): Complete camera-to-decision pipeline operational on Jetson.
[ ] Lab 12–13: End-to-end optimization, threshold tuning, and blind evaluation.
[ ] Lab 14: SEECS CV Face-Off 2026 Final Demonstration.
👥 Team & Governance
Role / Sub-Group
Lead Engineers
Primary Responsibility
Module Status
System Lead
TBD
Overall Coordination & Integration
🟩 GREEN
SG-1 (Detection)
Student A & Student B
Face Detection Engine
🟩 GREEN
SG-2 (Quality/Align)
Student C & Student D
Quality Filter & Alignment
🟩 GREEN
SG-3 (Recognition)
Student E & Student F
Open-Set Identification
🟩 GREEN
SG-4 (Tracking)
Student G & Student H
Multi-Person Tracking
🟩 GREEN
SG-5 (Decision)
Student I & Student J
Temporal Attendance Logic
🟩 GREEN
SG-6 (Deployment)
Student K & Student L
Jetson TensorRT & Integration
🟩 GREEN

📄 License & Course Information
Developed for CS-477 Computer Vision at the School of Electrical Engineering and Computer Science (SEECS), NUST.
SEECS CV Face-Off 2026 Project Submission.
