"""Synthetic classroom generator. Produces what SG-4 would send, plus ground truth.
Ground truth: a student is truly Present if visible >= min_presence seconds.
Brief visitors (5-20 s) are truly Absent; they test false-present control."""
import random
from .interface import Observation, UNKNOWN

def generate(seed=0, n_students=30, session=600.0, fps=2.0, p_present=0.75,
             p_detect=0.8, p_correct=0.85, n_unknown=3, n_brief=3, min_presence=30.0):
    rng = random.Random(seed)
    roster = [f"S{i:03d}" for i in range(n_students)]
    chosen = rng.sample(roster, n_students)
    present = chosen[:int(n_students * p_present)]
    rest = chosen[int(n_students * p_present):]
    brief = rest[:n_brief]
    # presence intervals per person: list of (start, end)
    people = {}
    for s in present:
        start = rng.uniform(0, session * 0.5)          # includes late arrivals
        end = min(session, start + rng.uniform(90, session))
        gaps = []
        for _ in range(rng.randint(0, 2)):              # occlusion / leaving and re-entry
            g0 = rng.uniform(start, end)
            gaps.append((g0, g0 + rng.uniform(5, 40)))
        people[s] = ("enrolled", start, end, gaps)
    for s in brief:
        start = rng.uniform(0, session - 25)
        people[s] = ("enrolled", start, start + rng.uniform(5, 20), [])
    for k in range(n_unknown):
        start = rng.uniform(0, session * 0.5)
        people[f"_unk{k}"] = ("unknown", start, start + rng.uniform(60, 200), [])

    truth = {}
    for s in roster:
        vis = 0.0
        if s in people:
            _, a, b, gaps = people[s]
            vis = (b - a) - sum(g1 - g0 for g0, g1 in gaps)
        truth[s] = "Present" if (s in people and vis >= min_presence) else "Absent"

    frames, track = [], {}
    nxt = 0
    steps = int(session * fps)
    for i in range(steps):
        t = i / fps
        obs = []
        for pid, (kind, a, b, gaps) in people.items():
            if not (a <= t <= b) or any(g0 <= t <= g1 for g0, g1 in gaps):
                continue
            if rng.random() > p_detect:
                continue
            # track fragmentation: occasionally new track id
            if pid not in track or rng.random() < 0.003:
                track[pid] = nxt; nxt += 1
            if kind == "unknown":
                sid = rng.choice(roster) if rng.random() < 0.5 else UNKNOWN
                conf = min(1, max(0, rng.betavariate(2, 3) + (0.3 if rng.random() < 0.08 else 0)))
            elif rng.random() < p_correct:
                sid, conf = pid, rng.betavariate(6, 2)
            else:
                sid = rng.choice(roster)           # wrong ID (ID switch / misrecognition)
                conf = rng.betavariate(2, 4) + (0.35 if rng.random() < 0.08 else 0)
                conf = min(1.0, conf)
            obs.append(Observation(t, track[pid], sid, round(conf, 3)))
        frames.append((t, obs))
    return roster, frames, truth
