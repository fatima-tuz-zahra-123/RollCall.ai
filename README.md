Real-Time Embedded Classroom Attendance System

AutoAttend-Orin is an end-to-end, edge-deployed computer vision pipeline designed for automated classroom attendance tracking. Engineered specifically for the NVIDIA Jetson Orin Nano 8GB, the system replaces manual roll calls with continuous multi-face detection, quality filtering, open-set identification, multi-person tracking, and temporal decision fusion.
Rather than relying on isolated single-frame classifications, AutoAttend-Orin aggregates visual evidence across video frames to eliminate false positives and ensure high-confidence attendance logging under variable lighting, poses, and occlusions.

System Architecture & Sub-Group Pipeline

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



