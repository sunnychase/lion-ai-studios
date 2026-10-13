#!/usr/bin/env python3
"""bench_cast.py — Cast (MiniMax H3 reference-to-video with sound, MiniMax's official recipe: 20 steps, res_multistep, no Turbo LoRA,
no gate patch; bench_graphs.cast_graph = app/workflows.py's cast_graph), 5 s (124 frames), straight on ComfyUI.

Same inputs as the studio Cast job in studio_jobs_end_to_end.csv (1010-134808-af1f, Fast HD, 979 s): the SYNTHETIC chef face
bx_img2.png (a Z-Image Turbo render, made by bench_image_conditioning.py), the same prompt and spoken line, seed 4242, 8-bit video decoder,
no reference voice. The prompt is built the way the studio builds it (studio_server.cast_prompt: H3's <Picture 1> / <Subject 1> naming).

usage: bench_cast.py [draft] [fasthd]      (default: draft = 864x480; fasthd = 1344x768)
The 8-bit decoder is used only if bench_graphs finds it under COMFY_MODELS (vae/), so set COMFY_MODELS to the folder that really holds the
models. In the published Draft row it did not, and that run used the fp16 decoder.
Per run: wall time, peak memory, GPU temperature (see benchlib).  -> $BENCH_OUT/cast.jsonl and MP4s  (results: benchmarks/results/cast_ref2va.csv)"""
import os, shutil, sys
import benchlib as B
import bench_graphs as W

OUT = os.path.join(B.BENCH_OUT, "cast"); os.makedirs(OUT, exist_ok=True)
FACE = "bx_img2.png"
PROMPT = ("The chef from the reference image stands in a busy restaurant kitchen at night, medium close-up, warm light.\n\n"
          "He turns to the camera, wipes his hands on a towel and speaks proudly.\n\nSound: Kitchen clatter in the background.")
LINE = "Tonight the special is slow roasted lamb with rosemary, and it is the best thing on the menu."


def studio_prompt(prompt, n_images=1, line=LINE):
    """= studio_server.cast_prompt for images and a spoken line, no reference voice"""
    pics = " and ".join(f"<Picture {i + 1}>" for i in range(n_images))
    subj = f"subject_definitions:\n<Subject 1> is the person in {pics}; keep their face and identity exactly.\n\n"
    for a in ("The person from the reference image", "the person from the reference image"): prompt = prompt.replace(a, "<Subject 1>")
    return subj + prompt + (f'\n\n<Subject 1> says, "{line}"' if line else "")


if __name__ == "__main__":
    for preset in (sys.argv[1:] or ["draft"]):
        g, info = W.cast_graph(preset, studio_prompt(PROMPT), 5, 4242, f"bench_cast/{preset}", [FACE], None, vae8=True)
        files, took, st = B.render_measured(g, preset, client="bench_cast", node="14")
        out = f"{OUT}/cast_{preset}.mp4"; shutil.copy(next(f for f in files if f.endswith(".mp4")), out)
        B.append("cast.jsonl", {"preset": preset, "w": info["w"], "h": info["h"], "steps": info["steps"], "frames": info["frames"], "vae": info["vae"],
                                "seed": 4242, "wall_s": round(took, 1), **st, "file": os.path.basename(out)})
