"""Algorithm A (BASELINE): N-of-M sliding window rule.
Present if, within any window of `window` seconds, the student has >= N
observations with confidence >= T. Once Present, stays Present (latched)."""
from collections import deque
from .interface import Decider, Decision, UNKNOWN

class NofM(Decider):
    def __init__(self, roster, T=0.6, N=6, window=30.0):
        super().__init__(roster)
        self.T, self.N, self.window = T, N, window
        self.hits = {s: deque() for s in roster}
        self.d = {s: Decision(s) for s in roster}
        self.prev_t = None
        self.best = {s: 0 for s in roster}

    def update(self, t, observations):
        dt = 0.0 if self.prev_t is None else t - self.prev_t
        self.prev_t = t
        seen = set()
        for o in observations:
            s = o.student_id
            if s == UNKNOWN or s not in self.d:
                continue
            d = self.d[s]
            if o.confidence < self.T:
                continue          # weak sightings are not evidence of presence
            if d.first_seen is None:
                d.first_seen = t
            d.last_seen = t
            if o.confidence >= self.T:
                self.hits[s].append(t)
                seen.add(s)
        for s in seen:
            self.d[s].total_confident_time += dt
        for s in self.roster:
            q = self.hits[s]
            while q and t - q[0] > self.window:
                q.popleft()
            self.best[s] = max(self.best[s], len(q))
            d = self.d[s]
            if d.status != "Present" and len(q) >= self.N:
                d.status, d.decision_time = "Present", t
                d.evidence_log.append(f"Present at t={t:.1f}: {len(q)} confident hits in {self.window:.0f}s window")

    def finalize(self, t_end):
        for s, d in self.d.items():
            if d.status != "Present":
                if self.best[s] >= max(1, self.N // 2):
                    d.status = "Uncertain"
                    d.evidence_log.append(f"Only {self.best[s]}/{self.N} hits in best window")
                d.decision_confidence = min(1.0, self.best[s] / self.N)
            else:
                d.decision_confidence = min(1.0, 0.5 + 0.5 * self.best[s] / (2 * self.N))
        return self.d
