#!/usr/bin/env python3
"""bench_qwen_image.py — Qwen-Image 2.1 Turbo (bf16, 8 steps on the checkpoint's own sigma schedule) vs Qwen-Image 2.1 (int8, 25 steps),
two prompts (a text-heavy poster and a photographic portrait), seed 77, one image per run, through the studio's image node.

Order per prompt: Turbo 1344x768, Turbo 2752x1536, base 1344x768. Per run: wall time (see benchlib; up to 2 s high from polling), which
includes loading a model when the previous run used the other one.  -> $BENCH_OUT/qwen_image.jsonl and PNGs
(results: benchmarks/results/images_qwen_image_21.csv)
env: IMAGE_NODE (default LionStudioImageGen — comfyui_nodes/lion_studio_image_gen.py)"""
import os, shutil
import benchlib as B
import bench_graphs as W

OUT = os.path.join(B.BENCH_OUT, "qwen_image"); os.makedirs(OUT, exist_ok=True)
P = {"poster": 'A vintage travel poster for "LIGHTHOUSE COVE" in bold art-deco lettering, a red and white lighthouse on a rocky point at sunset, '
               'sailboats on a teal sea, small text at the bottom reading "Ferries daily · Summer 2026", flat screen-print style, crisp typography',
     "portrait": "A candid photograph of an elderly fisherman mending a red net on a wooden dock at golden hour, weathered skin, detailed hands, "
                 "shallow depth of field, 85mm lens, natural light, photorealistic"}
RUNS = [("turbo", "Qwen-Image 2.1 Turbo (8 steps, 2K, best text)", 1344, 768), ("turbo2k", "Qwen-Image 2.1 Turbo (8 steps, 2K, best text)", 2752, 1536),
        ("base", "Qwen-Image 2.1 (newest, best text)", 1344, 768)]

if __name__ == "__main__":
    for name, prompt in P.items():
        for tag, model, w, h in RUNS:
            files, took = B.render(W.image_graph(model, prompt, w, h, 1, 77, f"bench_qi/{name}_{tag}"), f"{name}_{tag}", client="bench_qi")
            out = f"{OUT}/{name}_{tag}.png"; shutil.copy(files[0], out)
            B.append("qwen_image.jsonl", {"prompt": name, "variant": tag, "model": model, "w": w, "h": h, "seed": 77, "wall_s": round(took, 1),
                                          "file": os.path.basename(out)})
