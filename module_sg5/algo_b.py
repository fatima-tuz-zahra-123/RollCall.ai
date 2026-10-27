"""Algorithm B (ALTERNATE): evidence accumulation with decay.
score = score*exp(-dt/tau) + confidence (only if confidence >= T_min).
Present when score >= on_thr (latched). Uncertain if peak score >= off_thr."""
import math
from .interface import Decider, Decision, UNKNOWN

class EvidenceAccum(Decider):
    def __init__(self, roster, T_min=0.4, tau=15.0, on_thr=4.0, off_thr=2.0):
        super().__init__(roster)
        self.T_min, self.tau, self.on_thr, self.off_thr = T_min, tau, on_thr, off_thr
        self.score = {s: 0.0 for s in roster}
        self.peak = {s: 0.0 for s in roster}
        self.last_t = {s: None for s in roster}
        self.d = {s: Decision(s) for s in roster}
        self.prev_t = None

    def update(self, t, observations):
        dt = 0.0 if self.prev_t is None else t - self.prev_t
        self.prev_t = t
        decay = math.exp(-dt / self.tau)
        for s in self.roster:
            self.score[s] *= decay
        seen = set()
        for o in observations:
            s = o.student_id
            if s == UNKNOWN or s not in self.d:
                continue
            d = self.d[s]
            if o.confidence < self.T_min:
                continue          # weak sightings are not evidence of presence
            if d.first_seen is None:
                d.first_seen = t
            d.last_seen = t
            if o.confidence >= self.T_min:
                self.score[s] += o.confidence * o.quality
                seen.add(s)
        for s in seen:
            self.d[s].total_confident_time += dt
        for s in self.roster:
            self.peak[s] = max(self.peak[s], self.score[s])
            d = self.d[s]
            if d.status != "Present" and self.score[s] >= self.on_thr:
                d.status, d.decision_time = "Present", t
                d.evidence_log.append(f"Present at t={t:.1f}: score {self.score[s]:.1f} >= {self.on_thr}")

    def finalize(self, t_end):
        for s, d in self.d.items():
            if d.status != "Present" and self.peak[s] >= self.off_thr:
                d.status = "Uncertain"
                d.evidence_log.append(f"Peak score {self.peak[s]:.1f} below {self.on_thr}")
            d.decision_confidence = min(1.0, self.peak[s] / (2 * self.on_thr))
        return self.d
