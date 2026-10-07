"""Generate the sample I/O for SG-3 + the before/after figure (needs models + LFW: run download_assets.py first).

    python module_sg2/scripts/make_samples.py
Writes: interfaces/examples/sg2_example_{input,output}.json, module_sg2/samples/*.jpg, results/before_after.png
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np

SG2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SG2 / "experiments"))
from common import LFW, mock_sg1, sg2_quality  # noqa: E402
from exp2_thresholds import degrade  # noqa: E402

EX, SAMPLES = SG2.parent / "interfaces" / "examples", SG2 / "samples"
PEOPLE = ["Abel_Pacheco/Abel_Pacheco_0001.jpg", "Akhmed_Zakayev/Akhmed_Zakayev_0001.jpg",
          "Tony_Blair/Tony_Blair_0010.jpg", "Serena_Williams/Serena_Williams_0003.jpg"]

if __name__ == "__main__":
    EX.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(exist_ok=True)
    cfg = sg2_quality.load_config(SG2 / "config" / "baseline_v1.json")
    paths = [str(LFW / p) for p in PEOPLE]
    sg1 = mock_sg1(paths)

    # 1) one frame, exact V1 messages
    frame = cv2.imread(paths[0])
    faces = [sg1[paths[0]]]
    cv2.imwrite(str(SAMPLES / "example_input_frame.jpg"), frame)
    (EX / "sg2_example_input.json").write_text(json.dumps(
        {"frame_id": 1, "timestamp": 0.0, "image": "module_sg2/samples/example_input_frame.jpg", "faces": faces}, indent=2))
    out = sg2_quality.process_frame(frame, faces, cfg, frame_id=1)
    cv2.imwrite(str(SAMPLES / "example_output_face_112.png"), out["results"][0]["face"])  # png = lossless
    meta = sg2_quality.to_json(out)
    meta["results"][0]["face"] = "module_sg2/samples/example_output_face_112.png (112x112x3 uint8 BGR, aligned)"
    (EX / "sg2_example_output.json").write_text(json.dumps(meta, indent=2))

    # 2) before/after figure: accepted faces + controlled rejects with their reason codes
    tiles = []
    cases = [(p, "blur", 0) for p in paths] + [(paths[2], "blur", 4), (paths[2], "size", 16),
                                                (paths[2], "exposure", 0.1), (paths[2], "exposure", 3.0)]
    for p, kind, lvl in cases:
        img, face = degrade(cv2.imread(p), sg1[p], kind, lvl)
        r = sg2_quality.assess_face(img, face, cfg)
        before = img.copy()
        x, y, w, h = map(int, face["box"])
        cv2.rectangle(before, (x, y), (x + w, y + h), (0, 255, 0), 1)
        for lx, ly in face["landmarks"]:
            cv2.circle(before, (int(lx), int(ly)), 2, (0, 0, 255), -1)
        before = cv2.resize(before, (224, 224), interpolation=cv2.INTER_NEAREST)
        after = cv2.resize(r["face"] if r["accept"] else sg2_quality.assess_face(img, face, {**cfg, "gates": {}})["face"],
                           (224, 224), interpolation=cv2.INTER_NEAREST)
        label = f"ACCEPT q={r['quality']:.2f}" if r["accept"] else f"REJECT: {r['reason']}"
        bar = np.full((30, 448, 3), (40, 120, 40) if r["accept"] else (40, 40, 160), np.uint8)
        cv2.putText(bar, f"{kind}={lvl}  {label}", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(np.vstack([np.hstack([before, after]), bar]))
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    cv2.imwrite(str(SG2 / "results" / "before_after.png"), grid)
    print("wrote samples, examples and results/before_after.png")
