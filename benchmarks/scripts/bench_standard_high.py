#!/usr/bin/env python3
# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""bench_standard_high.py — Standard (H3 + Turbo 8-step) and High (H3 base, 20 steps) on ComfyUI vs an EXPERIMENTAL dense TensorFold path,
then the shipped Fast HD preset (FastH3 on TensorFold, 1344x768) on the same prompts and seed.

5 s with sound, seed 9090, 8-bit decoder on every run. Per run: wall time (see benchlib); for the image-start prompt the SSIM of frame 0
against the given first frame; for TensorFold runs the mean SSIM of the first 120 frames against the ComfyUI clip (how far the same seed
drifts between engines — a low number is expected and is NOT a quality score); the director's review (AI judge, picture only).
-> $BENCH_OUT/standard_high.jsonl  (results: benchmarks/results/video_standard_high_fasthd.csv)

The TensorFold Standard/High rows need merged dense bf16 checkpoints (H3 fl2va with the Turbo LoRA merged in using stochastic rounding,
and with zeroed sparse-gate tensors) named as in bench_graphs.H3_TF. They are not distributed and the studio does not use them: the gain
was 5-8 % end to end. Without those files the TensorFold rows are refused ("routed to ...") and only the ComfyUI and Fast HD rows run.

usage: bench_standard_high.py [run names, e.g. std_txt_comfy fhd_img_tensorfold]"""
import os, shutil, sys
import benchlib as B
import bench_graphs as W
from bench_image_conditioning import P_IMG2          # the chef prompt; its first frame is bx_img2.png (made by that script's prep())
W.ALLOW_FASTH3_ANY_INPUT = False                     # that import switches it on; this bench uses the studio's own routing

OUT = os.path.join(B.BENCH_OUT, "full"); os.makedirs(OUT, exist_ok=True); SEED = 9090
P_TXT = ("A glassblower in a dim workshop shapes a glowing orange vase at the end of a long pipe, sparks and heat shimmer around the furnace.",
         "He turns the pipe slowly, blows gently and the vase swells, then he lifts it toward the light to inspect it.",
         "Roar of the furnace, the hiss of hot glass, a soft breath through the pipe.")
RUNS = [  # name, preset, engine, first image — ComfyUI first, then TensorFold, then Fast HD
    ("std_txt", "standard", "comfy", None), ("std_img", "standard", "comfy", "bx_img2.png"), ("high_txt", "high", "comfy", None),
    ("std_txt", "standard", "tensorfold", None), ("std_img", "standard", "tensorfold", "bx_img2.png"), ("high_txt", "high", "tensorfold", None),
    ("fhd_txt", "fasthd", "tensorfold", None), ("fhd_img", "fasthd", "tensorfold", "bx_img2.png")]


def run(name, preset, engine, fi):
    tag = f"{name}_{engine}"; pr = P_IMG2 if fi else P_TXT
    g, meta = W.video_graph(preset, W.compose_prompt(*pr), 5, SEED, f"bench_full/{tag}", first_image=fi, engine=engine, vae8=True)
    if engine == "tensorfold" and "TensorFold" not in meta["engine"]: raise RuntimeError(f"{tag}: routed to {meta['engine']}")
    files, took = B.render(g, tag, client="bench_full")
    out = f"{OUT}/{tag}.mp4"; shutil.copy(next(f for f in files if f.endswith(".mp4")), out)
    r = {"name": name, "preset": preset, "engine": meta["engine"], "size": f"{meta['w']}x{meta['h']}", "steps": meta["steps"], "wall_s": round(took, 1),
         "file": os.path.basename(out)}
    B.WW, B.HH = meta["w"], meta["h"]
    if fi: r["first_ssim"] = B.ssim_frame(out, 0, f"{B.CI}/{fi}")
    other = f"{OUT}/{name}_comfy.mp4"
    if engine == "tensorfold" and os.path.exists(other): r["ssim_vs_comfy"] = B.ssim_span(out, 0, other, 0, 120)
    try: r["director"] = B.review(out, W.compose_prompt(*pr))
    except Exception as e: r["director"] = {"error": str(e)[:200]}
    B.append("standard_high.jsonl", r)


if __name__ == "__main__":
    only = sys.argv[1:]
    for spec in RUNS:
        if only and f"{spec[0]}_{spec[2]}" not in only: continue
        try: run(*spec)
        except Exception as e: B.append("standard_high.jsonl", {"run": f"{spec[0]}_{spec[2]}", "error": str(e)[:400]})
