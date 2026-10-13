#!/usr/bin/env python3
"""bench_ming.py — Ming-Image 0.1 Design the way the studio runs it (studio_server.run_ming): optional director layout, ComfyUI /free,
a memory check, then ONE ming_runner.py process per job that loads the model, renders every seed and exits.

  layout  the director rewrites the poster brief into Ming's own Figma-style JSON layout (director.ming_layout: Ming's released rewriter
          prompt on the studio director). Timed on its own; the director start-up (if it was not running) is timed separately.
          Skipped with MING_LAYOUT=0. The layout is saved but NOT rendered here: the two Ming runs below take plain prompts.
  plain   the poster brief from bench_qwen_image.py, 1024x1024, seeds 77,78 in one process -> load, first image (cold), second (warm)
  rgba    a badge on a transparent background (the studio's "transparent" option = Ming's fixed phrase at the very start of the
          prompt), 1024x1024, seeds 77,78 in one process

Per process: wall time of the whole process, the runner's own load_s and per-image gen_s, peak memory in use on the machine and the highest
GPU temperature (benchlib.Sampler). The studio needs about 58 GiB + 5 GiB per megapixel free before it starts Ming (workflows.ming_need_gib).
-> $BENCH_OUT/ming.jsonl and PNGs  (results: benchmarks/results/images_ming_design.csv)
env: MING_PY (Ming's venv python), MING_RUNNER (app/ming_runner.py), STUDIO_APP (for director.py)"""
import json, os, subprocess, sys, time, urllib.request
import benchlib as B
from bench_qwen_image import P

OUT = os.path.join(B.BENCH_OUT, "ming"); os.makedirs(OUT, exist_ok=True)
MING_PY = os.path.expanduser(os.environ.get("MING_PY", "~/lion-studio/envs/ming/bin/python"))
MING_RUNNER = os.path.expanduser(os.environ.get("MING_RUNNER", os.path.join(B.APP, "ming_runner.py")))
RGBA = "RGBA, 4-channel, transparent background\n"           # the phrase studio_server.run_ming puts first when "transparent" is ticked
BADGE = ('A round badge logo for "LIGHTHOUSE COVE": a small red and white lighthouse inside a navy ring, the name in bold art-deco '
         'lettering around the ring, flat vector style, crisp edges')
W_, H_ = 1024, 1024


def need_gib(w, h): return 58 + 5 * w * h / 1048576           # = workflows.ming_need_gib


def comfy_free():
    try: B.post("/free", {"unload_models": True, "free_memory": True}); time.sleep(4)
    except Exception: pass


def layout():
    if B.APP not in sys.path: sys.path.insert(0, B.APP)
    import director as D
    t = time.time(); up = D.status().get("director"); ok = D.ensure(); start_s = 0.0 if up else round(time.time() - t, 1)
    if not ok: raise RuntimeError("the director did not start")
    t = time.time(); text, secs = D.ming_layout(P["poster"], W_, H_); wall = round(time.time() - t, 1)
    open(f"{OUT}/layout.json", "w").write(text)
    B.append("ming.jsonl", {"run": "layout", "director_was_running": up, "director_start_s": start_s, "layout_s": secs, "wall_s": wall,
                            "layout_chars": len(text)})


def ming(tag, prompt, seeds="77,78"):
    B.studio_idle(); comfy_free()
    pf = f"{OUT}/{tag}_prompt.txt"; open(pf, "w").write(prompt)
    need = need_gib(W_, H_); free = B.avail_gib(); paused = False
    if free < need:                                            # as the studio does: pause the director only when Ming would not fit
        if B.APP not in sys.path: sys.path.insert(0, B.APP)
        import director as D
        if D.status().get("director"): D.stop(); time.sleep(8); free = B.avail_gib(); paused = True
    if free < need: raise RuntimeError(f"{tag}: Ming needs ~{need:.0f} GiB free, {free:.0f} GiB is")
    with B.Sampler() as s:
        t = time.time()
        r = subprocess.run([MING_PY, MING_RUNNER, "--prompt-file", pf, "--width", str(W_), "--height", str(H_), "--seeds", seeds, "--out", f"{OUT}/{tag}"],
                           capture_output=True, text=True, timeout=3600)
        wall = time.time() - t
    lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode or not lines: raise RuntimeError(f"{tag}: Ming failed: {(r.stderr or r.stdout)[-400:]}")
    res = json.loads(lines[-1])
    B.append("ming.jsonl", {"run": tag, "w": W_, "h": H_, "seeds": seeds, "free_before_gib": round(free, 1), "need_gib": round(need, 1), "director_paused": paused,
                            "wall_s": round(wall, 1), "load_s": res.get("load_s"), "gen_s": res.get("gen_s"), "steps": res.get("steps"),
                            **s.stats(), "files": [os.path.basename(f) for f in res.get("outputs", [])],
                            "torch": res.get("torch"), "transformers": res.get("transformers")})


if __name__ == "__main__":
    if os.environ.get("MING_LAYOUT", "1") != "0":
        try: layout()
        except Exception as e: B.append("ming.jsonl", {"run": "layout", "error": f"{type(e).__name__}: {str(e)[:300]}"})
    ming("plain", P["poster"])
    ming("rgba", RGBA + BADGE)
