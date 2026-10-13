#!/usr/bin/env python3
# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""bench_image_edit.py — the studio's image edit (Qwen-Image 2.1 Turbo, bf16, 8 steps on the checkpoint's own sigmas, euler, CFG 1;
bench_graphs.edit_graph = app/workflows.py's edit_graph) with 1, 2 and 4 reference images. The output takes the first reference's size,
which is 1344x768 in every run here.

References (all synthetic, made by bench_images_all.py — run that first): R1 the Qwen-Image 2.1 Turbo portrait, R2 its poster,
R3 the Z-Image Turbo portrait, R4 the FLUX.2 klein poster. Copied into ComfyUI's input folder as bx_edit_r1..r4.png.

Order: e1 (1 reference, seed 77) right after the image models, so it loads the edit model; e1 again with seed 78 (warm; seed 78 so ComfyUI
cannot serve it from its cache); e2 (2 references, seed 77); e4 (4 references, seed 77). Per run: wall time, peak memory, GPU temperature
(see benchlib).
Then a control: every model unloaded (ComfyUI /free), and the three seed-77 edits again in reverse order (c_e4, c_e2, c_e1), to check that the
run order and earlier runs do not change the pictures or the times.  -> $BENCH_OUT/image_edit.jsonl, image_edit_control.jsonl and PNGs  (results: benchmarks/results/images_edit_qwen_image_21.csv)"""
import os, shutil, time
import benchlib as B
import bench_graphs as W

SRC = os.path.join(B.BENCH_OUT, "images_all")
OUT = os.path.join(B.BENCH_OUT, "image_edit"); os.makedirs(OUT, exist_ok=True)
REFS = ["qwen_image_2_1_turbo_warm1_portrait_1344.png", "qwen_image_2_1_turbo_cold_poster_1344.png",
        "z_image_turbo_warm1_portrait_1344.png", "flux_2_klein_cold_poster_1344.png"]
P = {1: "Give the fisherman a red knit beanie and a navy waterproof jacket; keep everything else exactly the same.",
     2: "Put the fisherman from image 1 on a wooden pier in front of the lighthouse from image 2 at sunset, photorealistic; keep his face and "
        "clothes exactly the same.",
     4: 'A new travel poster in the flat screen-print style of image 2 and image 4: the fisherman from image 1 and the fisherman from image 3 '
        'stand side by side on the dock mending one red net, with the title "LIGHTHOUSE COVE" in bold art-deco lettering.'}
RUNS = [("e1_cold", 1, 77), ("e1_warm", 1, 78), ("e2", 2, 77), ("e4", 4, 77)]
CONTROL = [("c_e4", 4, 77), ("c_e2", 2, 77), ("c_e1", 1, 77)]


def run(runs, log):
    for order, (tag, n, seed) in enumerate(runs, 1):
        files, took, st = B.render_measured(W.edit_graph(names[:n], P[n], seed, f"bench_edit/{tag}"), tag, client="bench_edit", node="8")
        out = f"{OUT}/{tag}.png"; shutil.copy(files[0], out)
        B.append(log, {"order": order, "run": tag, "refs": n, "seed": seed, "wall_s": round(took, 1), **st, "file": os.path.basename(out)})


if __name__ == "__main__":
    names = []
    for i, f in enumerate(REFS):
        dst = f"bx_edit_r{i + 1}.png"; shutil.copy(os.path.join(SRC, f), os.path.join(B.CI, dst)); names.append(dst)
    run(RUNS, "image_edit.jsonl")
    B.post("/free", {"unload_models": True, "free_memory": True}); time.sleep(5)
    run(CONTROL, "image_edit_control.jsonl")
