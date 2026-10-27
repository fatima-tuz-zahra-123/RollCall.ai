"""V1 interface. Both algorithms implement Decider. Do not change after Lab 4."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

UNKNOWN = "UNKNOWN"

@dataclass
class Observation:
    timestamp: float
    track_id: int
    student_id: str
    confidence: float
    quality: float = 1.0

def parse_observation(d: dict) -> Optional[Observation]:
    """Safe parser: returns None for bad records instead of crashing."""
    try:
        o = Observation(float(d["timestamp"]), int(d["track_id"]), str(d["student_id"]),
                        float(d["confidence"]), float(d.get("quality", 1.0)))
        if not (0.0 <= o.confidence <= 1.0):
            return None
        return o
    except (KeyError, ValueError, TypeError):
        return None

@dataclass
class Decision:
    student_id: str
    status: str = "Absent"                 # Present / Absent / Uncertain
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None
    total_confident_time: float = 0.0
    decision_confidence: float = 0.0
    decision_time: Optional[float] = None
    evidence_log: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

class Decider:
    """update() once per frame with a list of observations; finalize() at end."""
    def __init__(self, roster: List[str]):
        self.roster = list(roster)
    def update(self, t: float, observations: List[Observation]) -> None:
        raise NotImplementedError
    def finalize(self, t_end: float) -> Dict[str, Decision]:
        raise NotImplementedError
