# SG-5 Attendance Decision (module_sg5)
Purpose: turn SG-4 identity observations into Present/Absent/Uncertain per student.
Interface: interfaces/sg5_interface_v1.md (frozen V1).
Run: `python -m evaluation.compare` (synthetic data, 20 seeds, results in evaluation/results.csv)
Algorithms: A = N-of-M window (baseline), B = evidence accumulation with decay.
Test data: module_sg5/mock_gen.py (seeded, with ground truth). Swap in SG-4 output via parse_observation().
