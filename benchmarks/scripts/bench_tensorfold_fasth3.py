#!/usr/bin/env python3
# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""bench_tensorfold_fasth3.py — FastH3 text-to-video on ComfyUI vs on TensorFold's CUDA engine, 5 s clip with sound, same prompt and seed.

Per run: wall time (see benchlib), output size/frames, decoder (fp16 or ComfyUI's 8-bit video decoder) and the GPU temperature at the end.
Runs one at a time and stops if ComfyUI is busy. ComfyUI's runs go first, then the engine's, with /free in between so the two FastH3
copies are never resident together.  -> $BENCH_OUT/tensorfold_fasth3.jsonl
(results: benchmarks/results/video_fasth3_comfyui_vs_tensorfold.csv)

Read the results with care:
  * D (ComfyUI + 8-bit decoder) repeats A's prompt and seed, so ComfyUI serves the sampler output from its cache: it times the decode only.
  * B1 is the first TensorFold run on a machine: it includes the one-time bf16 -> int8 conversion of the checkpoint.
  * B2 uses seed+1 for the same cache reason; C and E use the same seed as A and F.
  * The two engines do not produce the same frames from the same seed (different weights and sparse-attention schedule)."""
import json, os, sys, time
import benchlib as B
import bench_graphs as W

PROMPT = W.compose_prompt("A weathered fisherman mends a red net on a wooden dock at golden hour, gulls overhead, harbour boats bobbing behind him.",
                          "He pulls the net taut, looks up at the camera and smiles; the camera slowly pushes in.",
                          "Water lapping, gulls calling, the creak of the dock.")
SEED = 1009
RUNS = [  # name, preset, engine, 8-bit decoder[, seed]
    ("A_comfy_480", "draft", "comfy", False),
    ("D_comfy_480_vae8", "draft", "comfy", True),
    ("F_comfy_768", "fasthd", "comfy", False),
    ("B1_tf_480_first", "draft", "tensorfold", False),
    ("B2_tf_480", "draft", "tensorfold", False, SEED + 1),
    ("C_tf_480_vae8", "draft", "tensorfold", True),
    ("E_tf_768", "fasthd", "tensorfold", False),
]


def run(name, preset, engine, vae8, seed=SEED):
    q = B.get("/queue")
    if q["queue_running"] or q["queue_pending"]: sys.exit(f"ComfyUI is busy — stopping before {name}")
    g, meta = W.video_graph(preset, PROMPT, 5, seed, f"bench_tf/{name}", engine=engine, vae8=vae8)
    if engine == "tensorfold" and "TensorFold" not in meta["engine"]: sys.exit(f"{name}: TensorFold not ready (engine library or bf16 checkpoint missing)")
    files, took = B.render(g, name, client="bench_tf")
    return {"name": name, "preset": preset, "engine": meta["engine"], "vae": meta["vae"], "size": f"{meta['w']}x{meta['h']}", "frames": meta["frames"],
            "seed": seed, "wall_s": round(took, 1), "status": "success", "file": os.path.basename(files[0]) if files else None, "gpu_temp_end_c": B.gpu_temp()}


if __name__ == "__main__":
    only = sys.argv[1:]; done_tf = False
    for spec in RUNS:
        if only and spec[0] not in only: continue
        if spec[2] == "tensorfold" and not done_tf:
            B.post("/free", {"unload_models": True, "free_memory": True}); time.sleep(5); done_tf = True
        B.append("tensorfold_fasth3.jsonl", run(*spec))
