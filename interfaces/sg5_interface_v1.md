# SG-5 Interface V1 (FROZEN)

## Input from SG-4: one record per tracked face per processed frame
| field       | type   | rule |
|-------------|--------|------|
| timestamp   | float  | seconds since session start (monotonic) |
| track_id    | int    | persistent track ID from SG-4 |
| student_id  | string | enrolled ID, or "UNKNOWN" |
| confidence  | float  | 0.0 to 1.0 (identity confidence) |
| quality     | float  | optional, 0.0 to 1.0, default 1.0 |

Empty frame = empty list []. Missing/invalid records are skipped, never crash.

## Output to SG-6: one record per enrolled student
| field                 | type   | rule |
|-----------------------|--------|------|
| student_id            | string | |
| status                | string | "Present" / "Absent" / "Uncertain" |
| first_seen            | float? | seconds, null if never seen |
| last_seen             | float? | seconds, null if never seen |
| total_confident_time  | float  | seconds with confidence >= T |
| decision_confidence   | float  | 0.0 to 1.0 |
| decision_time         | float? | when status became Present, else null |
| evidence_log          | list   | max 5 short strings |

Change after Lab 4 needs team agreement + documentation.
