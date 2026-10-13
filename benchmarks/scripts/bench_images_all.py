#!/usr/bin/env python3
# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""bench_images_all.py — every image model the studio offers, at the studio's defaults for that model (the image node's own steps,
sampler and CFG: `steps` = 0), one image per run, through the studio's image node (bench_graphs.image_graph).

Models: the studio's own list (GET $STUDIO_URL/api/state -> image_models, i.e. what is installed), minus Ming-Image (it runs outside
ComfyUI: bench_ming.py). Or set IMAGE_MODELS="name|name|...". Same two prompts as bench_qwen_image.py (a text-heavy poster, a portrait).

Per model, back to back:
  cold   poster   1344x768  seed 77   the first run after switching from the previous model (the node keeps one model family loaded)
  warm1  portrait 1344x768  seed 77
  warm2  poster   1344x768  seed 78   seed 78, not 77: ComfyUI keeps earlier results and would serve an identical graph from its cache
Then, for the models whose 2K sizes the studio offers (TWO_K, Qwen-Image 2.1 Turbo), poster and portrait at 2752x1536, seed 77 (warm).

"Cold" here means the model was not in ComfyUI's memory. Its files may still be in the operating system's file cache from earlier use,
so a load straight from disk can take longer. Per run: wall time (see benchlib), peak memory in use on the machine and the highest GPU
temperature (benchlib.Sampler, 1 s samples), and whether ComfyUI served the image node from its cache (it must not).
-> $BENCH_OUT/images_all.jsonl and PNGs  (results: benchmarks/results/images_all_models.csv)
env: IMAGE_NODE (default LionStudioImageGen), IMAGE_MODELS, STUDIO_URL"""
import json, os, shutil, urllib.request
import benchlib as B
import bench_graphs as W
from bench_qwen_image import P

OUT = os.path.join(B.BENCH_OUT, "images_all"); os.makedirs(OUT, exist_ok=True)
TWO_K = ("Qwen-Image 2.1 Turbo",)          # the studio: "2K sizes are for Qwen-Image 2.1 Turbo"
RUNS = [("cold", "poster", 1344, 768, 77), ("warm1", "portrait", 1344, 768, 77), ("warm2", "poster", 1344, 768, 78)]
RUNS_2K = [("warm_2k", "poster", 2752, 1536, 77), ("warm_2k", "portrait", 2752, 1536, 77)]


def models():
    if os.environ.get("IMAGE_MODELS"): return os.environ["IMAGE_MODELS"].split("|")
    names = json.load(urllib.request.urlopen(B.STUDIO_URL + "/api/state", timeout=30))["image_models"]
    return [n for n in names if not n.startswith("Ming-Image")]


def slug(name): return "".join(c if c.isalnum() else "_" for c in name.split(" (")[0]).strip("_").lower()


if __name__ == "__main__":
    order = 0
    for model in models():
        runs = RUNS + (RUNS_2K if model.startswith(TWO_K) else [])
        for kind, prompt, w, h, seed in runs:
            order += 1; tag = f"{slug(model)}_{kind}_{prompt}_{w}"
            files, took, st = B.render_measured(W.image_graph(model, P[prompt], w, h, 1, seed, f"bench_all/{tag}"), tag, client="bench_all", node="1")
            out = f"{OUT}/{tag}.png"; shutil.copy(files[0], out)
            B.append("images_all.jsonl", {"order": order, "model": model, "run": kind, "prompt": prompt, "w": w, "h": h, "seed": seed,
                                          "wall_s": round(took, 1), **st, "file": os.path.basename(out)})
