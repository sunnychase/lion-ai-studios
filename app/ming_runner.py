#!/usr/bin/env python3
"""Lion AI Studios — Ming-Image-0.1-Design runner. One process per studio job: load once, render N seeds, save, exit (so all of its
memory comes back). Runs in Ming's own venv (config.MING_PY), called by studio_server.run_ming():

  ming_runner.py --prompt-file P --width W --height H --seeds 1,2,3 --out DIR [--model DIR] [--code DIR] [--steps 12] [--cfg 1.0]

Uses the official inclusionAI/Ming-Image code (infer.load_model_and_processor / run_generation) unchanged. --model and --code default
to config.MING_MODEL and config.MING_CODE (studio.env). Prints one JSON line:
{"outputs": [...png paths...], "load_s": .., "gen_s": [..], "width": .., "height": .., "torch": .., "transformers": .., "steps": ..}.
Prompts that start with one of Ming's fixed RGBA phrases keep the alpha channel in the PNG."""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402  (standard library only, so it imports from Ming's venv too)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", required=True); ap.add_argument("--width", type=int, required=True); ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--seeds", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=C.MING_MODEL); ap.add_argument("--code", default=C.MING_CODE)
    ap.add_argument("--steps", type=int); ap.add_argument("--cfg", type=float)
    a = ap.parse_args()
    model_dir = Path(a.model); sys.path.insert(0, str(Path(a.code)))
    import infer                                                       # the official CLI module; its argparse builds the load options
    prompt = Path(a.prompt_file).read_text().strip()
    sys.argv = ["infer.py", "--model", str(model_dir), "--task", "text-to-image", "--prompt", prompt, "--width", str(a.width), "--height", str(a.height),
                "--local-files-only"] + (["--steps", str(a.steps)] if a.steps else []) + (["--cfg", str(a.cfg)] if a.cfg else [])
    args = infer.parse_args()
    profile = infer.load_checkpoint_capabilities(model_dir)
    sampling = profile.resolve_sampling_parameters(steps=args.steps, cfg=args.cfg)
    _, _, eff = infer.resolve_t2i_size("text-to-image", None, args.width, args.height)
    t0 = time.time(); model, processor = infer.load_model_and_processor(model_dir, args); load_s = time.time() - t0
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); files, gen_s = [], []
    for i, seed in enumerate(int(s) for s in a.seeds.split(",")):
        t1 = time.time()
        imgs = infer.run_generation(model, processor, profile, task="text-to-image", prompt=prompt, input_image=None, resolution=eff,
                                    width=args.width, height=args.height, sampling=sampling, seed=seed, num_layers=1, dtype=infer._dtype(args.dtype))
        gen_s.append(round(time.time() - t1, 1))
        for k, im in enumerate(imgs):
            p = out / f"ming_{i + 1:02d}{'' if k == 0 else f'_{k}'}.png"; im.save(p); files.append(str(p))
    import torch, transformers                                         # every job records the runtime it ran on (it may share another venv's torch)
    print(json.dumps({"outputs": files, "load_s": round(load_s, 1), "gen_s": gen_s, "width": args.width, "height": args.height,
                      "torch": torch.__version__, "transformers": transformers.__version__,
                      "steps": sampling.steps if hasattr(sampling, "steps") else None}))


if __name__ == "__main__":
    main()
