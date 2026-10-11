#!/usr/bin/env python3
"""seam_check.py — is there an audible gap where two segments of a film are joined?

For every finished film in the studio library (job kinds "chain" = guided segments, "keyframes" = keyframe films) it decodes the film's
audio (mono, 16 kHz), computes the level in 50 ms windows, and at each join reports how far the quietest 50 ms window near the seam
(from 200 ms before to 1.25 s after) falls below the median level of the surrounding context (2 s .. 0.5 s before and 1.5 s .. 3 s after).
The same measure at up to 6 random non-seam points of the same film gives a yardstick for ordinary level wobble.
Seam positions follow the join rules: guided segments overlap by 22 frames (dropped on join), keyframe segments by 1 frame.
Reads only; renders nothing.  -> $BENCH_OUT/seams.jsonl
env: STUDIO_HOME (library/ and state/jobs.json live there)"""
import json, os, subprocess, sys
import numpy as np
import benchlib as B

sys.path.insert(0, B.APP)
import stitch                                   # app/stitch.py: probe()
GUIDE_FRAMES, FPS = 22, 24

J = json.load(open(os.path.join(B.STUDIO_HOME, "state", "jobs.json")))
J = J if isinstance(J, list) else J.get("jobs", J); J = list(J.values()) if isinstance(J, dict) else J


def rms_db(path):
    a = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True).stdout
    x = np.frombuffer(a, np.int16).astype(np.float32) / 32768; w = 800                      # 800 samples = 50 ms
    n = len(x) // w; r = np.sqrt((x[:n * w].reshape(n, w) ** 2).mean(1) + 1e-12); return 20 * np.log10(r)


for j in J:
    if j.get("kind") not in ("chain", "keyframes") or j.get("status") != "done": continue
    d = os.path.join(B.STUDIO_HOME, "library", j["id"])
    if not os.path.isdir(d): continue
    segs = sorted(f for f in os.listdir(d) if f.startswith("seg") and f.endswith(".mp4") and ".v" not in f and "retake" not in f)
    film = os.path.join(d, "film.mp4")
    if len(segs) < 2 or not os.path.exists(film): continue
    db = rms_db(film); t = 0.0; eng = j.get("progress", {}).get("engine", "?")
    for k, sname in enumerate(segs[:-1]):
        dur = stitch.probe(os.path.join(d, sname))["dur"]
        t += dur if k == 0 else dur - (GUIDE_FRAMES / FPS if j["kind"] == "chain" else 1 / FPS)
        c = int(t / 0.05); a0 = max(0, c - 4); win = db[a0: c + 25]
        ctx = np.concatenate([db[max(0, c - 40): max(0, c - 10)], db[c + 30: c + 60]])
        dip = float(np.median(ctx) - win.min()); at = (int(np.argmin(win)) + a0 - c) * 50
        rng = np.random.default_rng(k); pts = [int(x) for x in rng.integers(60, len(db) - 60, 6) if abs(int(x) - c) > 60]
        base = [float(np.median(np.concatenate([db[p - 40: p - 10], db[p + 30: p + 60]])) - db[p - 4: p + 25].min()) for p in pts]
        B.append("seams.jsonl", {"job": j["id"], "kind": j["kind"], "engine": eng, "seam": k + 1, "at_s": round(t, 2), "dip_db": round(dip, 1),
                                 "dip_at_ms": at, "ordinary_points_median_db": round(float(np.median(base)), 1) if base else None})
