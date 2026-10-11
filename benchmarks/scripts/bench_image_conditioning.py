#!/usr/bin/env python3
"""bench_image_conditioning.py — can FastH3 (on ComfyUI and on TensorFold) take a first frame, a first+last frame pair and a guide
(the previous clip's tail), so image starts, films and keyframe films can leave the slower H3 + Turbo path?

Draft size (864x480), 5 s with sound, 8 steps, same inputs and seed (4242) for all three configurations, 8-bit decoder on all:
  H3T  MiniMax H3 fl2va int8 + Turbo 8-step LoRA on ComfyUI (the old path)
  FC   FastH3 on ComfyUI            (bench_graphs.ALLOW_FASTH3_ANY_INPUT — the studio itself never routes this)
  FTF  FastH3 on TensorFold
Per run: wall time (see benchlib); SSIM of the anchored frames against what was given (first frame / last frame vs the input image,
or the first 22 frames vs the guide's last 22 frames — ffmpeg ssim, 'All' average, at 864x480); the director's review (an AI judge that
watches the picture only, not told which engine made the clip).  -> $BENCH_OUT/image_conditioning.jsonl
(results: benchmarks/results/video_image_conditioning_480p.csv)

Inputs (created once in ComfyUI's input folder as bx_*):
  bx_img1/2.png   two Z-Image Turbo renders of the prompts below (1344x768, cropped to 864x480)
  bx_key1/2.png   frames 12 and 230 of KEYFRAME_VIDEO (env; the run used a 10 s Fast HD lighthouse clip made from P_KEYS)
  bx_guide.mp4    GUIDE_VIDEO (env; default = the C_tf_480_vae8 clip from bench_tensorfold_fasth3.py)"""
import os, shutil, sys
import benchlib as B
import bench_graphs as W

W.ALLOW_FASTH3_ANY_INPUT = True
OUT = os.path.join(B.BENCH_OUT, "img"); os.makedirs(OUT, exist_ok=True)
SEED = 4242; B.WW, B.HH = 864, 480
CONF = {"H3T": dict(engine="comfy", force_h3=True), "FC": dict(engine="comfy"), "FTF": dict(engine="tensorfold")}
P_IMG1 = ("A young woman in a red wool coat stands on a snowy railway platform at dawn beside a vintage steam locomotive, soft golden light, "
          "cinematic photo", "She turns toward the camera, pulls her scarf tighter and walks toward the train as steam drifts past.",
          "Steam hissing, distant train whistle, boots crunching on snow.")
P_IMG2 = ("A chef in a white jacket stands at a stainless steel counter in a busy restaurant kitchen, plating a chocolate dessert, warm light, cinematic photo",
          "He drizzles raspberry sauce over the dessert, adds a mint leaf, looks up and nods with a satisfied smile.",
          "Kitchen clatter, a pan sizzling, someone calling an order.")
P_GUIDE = ("A weathered fisherman on a wooden dock at golden hour with a red net, harbour boats behind him.",
           "He coils the red net over his shoulder, steps onto his boat and waves as the boat pulls away from the dock.",
           "Water lapping, gulls calling, a boat engine starting.")
P_KEYS = ("A lighthouse keeper in a yellow oilskin inside a stone lighthouse at dusk, storm clouds over a dark sea.",
          "She climbs the spiral stairs with her lantern and reaches the great lens at the top.",
          "Wind howling, rain on glass, footsteps on iron stairs.")
KEYFRAME_VIDEO = os.environ.get("KEYFRAME_VIDEO", "")
GUIDE_VIDEO = os.environ.get("GUIDE_VIDEO", os.path.join(B.CO, "bench_tf", "C_tf_480_vae8_00001_.mp4"))


def prep():
    for i, pic in ((1, P_IMG1[0]), (2, P_IMG2[0])):
        dst = f"{B.CI}/bx_img{i}.png"
        if os.path.exists(dst): continue
        files, _ = B.render(W.image_graph("Z-Image-Turbo (photoreal, best)", pic, 1344, 768, 1, SEED + i, f"bench_img/src{i}"), f"img{i}")
        B.ff("-i", files[0], "-vf", f"scale={B.WW}:{B.HH}:force_original_aspect_ratio=increase,crop={B.WW}:{B.HH}", dst)
    for i, n in ((1, 12), (2, 230)):
        dst = f"{B.CI}/bx_key{i}.png"
        if os.path.exists(dst): continue
        if not KEYFRAME_VIDEO: sys.exit("set KEYFRAME_VIDEO to a >= 10 s clip (e.g. a Fast HD render of P_KEYS) or provide bx_key1.png / bx_key2.png")
        B.ff("-i", KEYFRAME_VIDEO, "-vf", f"select=eq(n\\,{n}),scale={B.WW}:{B.HH}:force_original_aspect_ratio=increase,crop={B.WW}:{B.HH}", "-frames:v", "1", dst)
    if not os.path.exists(f"{B.CI}/bx_guide.mp4"): shutil.copy(GUIDE_VIDEO, f"{B.CI}/bx_guide.mp4")


def run(test, conf):
    tag = f"{test}_{conf}"; c = CONF[conf]
    if test.startswith("first"):
        k = test[-1]; pr = P_IMG1 if k == "1" else P_IMG2
        g, meta = W.video_graph("draft", W.compose_prompt(*pr), 5, SEED, f"bench_img/{tag}", first_image=f"bx_img{k}.png", vae8=True, **c)
    elif test == "keys":
        g, meta = W.video_graph("draft", W.compose_prompt(*P_KEYS), 5, SEED, f"bench_img/{tag}", first_image="bx_key1.png", last_image="bx_key2.png", vae8=True, **c)
    else:
        gf = B.nframes(f"{B.CI}/bx_guide.mp4")
        g, meta = W.video_graph("draft", W.compose_prompt(*P_GUIDE), 5, SEED, f"bench_img/{tag}", guide_video="bx_guide.mp4", vae8=True, **c)
        g["33"]["inputs"]["batch_index"] = gf - W.GUIDE_FRAMES; g["34"]["inputs"]["start_index"] = (gf - W.GUIDE_FRAMES) / W.FPS
    if conf != "H3T" and "FastH3" not in meta["engine"]: raise RuntimeError(f"{tag}: routed to {meta['engine']}")
    files, took = B.render(g, tag, client="bench_img")
    vid = next(f for f in files if f.endswith(".mp4")); out = f"{OUT}/{tag}.mp4"; shutil.copy(vid, out)
    r = {"test": test, "conf": conf, "engine": meta["engine"], "wall_s": round(took, 1), "file": os.path.basename(out), "frames": B.nframes(out)}
    if test.startswith("first"): r["first_ssim"] = B.ssim_frame(out, 0, f"{B.CI}/bx_img{test[-1]}.png")
    if test == "keys":
        r["first_ssim"] = B.ssim_frame(out, 0, f"{B.CI}/bx_key1.png"); r["last_ssim"] = B.ssim_frame(out, r["frames"] - 1, f"{B.CI}/bx_key2.png")
    if test == "guide":
        gf = B.nframes(f"{B.CI}/bx_guide.mp4"); r["anchor_ssim"] = B.ssim_span(out, 0, f"{B.CI}/bx_guide.mp4", gf - W.GUIDE_FRAMES, W.GUIDE_FRAMES)
    pr = {"first1": P_IMG1, "first2": P_IMG2, "keys": P_KEYS, "guide": P_GUIDE}[test]
    try: r["director"] = B.review(out, W.compose_prompt(*pr))
    except Exception as e: r["director"] = {"error": str(e)[:200]}
    B.append("image_conditioning.jsonl", r)


if __name__ == "__main__":
    prep()
    plan = [(t, c) for c in ("H3T", "FC", "FTF") for t in ("first1", "first2", "guide", "keys")]   # grouped by engine: fewer model swaps
    only = sys.argv[1:]
    for t, c in plan:
        if only and f"{t}_{c}" not in only: continue
        try: run(t, c)
        except Exception as e: B.append("image_conditioning.jsonl", {"test": t, "conf": c, "error": str(e)[:400]})
