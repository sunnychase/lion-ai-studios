#!/usr/bin/env python3
# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""bench_presets_comfyui.py — baseline: the three MiniMax H3 presets on stock ComfyUI (no TensorFold), one 5 s clip with sound each.

Measures: wall time (POST /prompt -> /history done, polled every 3 s), the per-step time ComfyUI prints at the end of each sampler
progress bar (read from the ComfyUI service journal), GPU utilisation / power / temperature and free unified memory sampled every second.
Same prompt and seed for every preset.  -> $BENCH_OUT/presets_comfyui.jsonl  (results: benchmarks/results/video_presets_comfyui_baseline.csv)

  draft     FastH3 V2 int8 + VSA sparse attention, 864x480, 8 steps
  standard  H3 fl2va int8 + Turbo 8-step LoRA, 1280x720, 8 steps   (the studio later moved Standard to 1248x704: H3 needs sides divisible by 32)
  high      H3 fl2va int8 base, 1344x768, 20 steps

usage: bench_presets_comfyui.py PRESET [SECONDS]
env:   COMFY_URL, BENCH_OUT, COMFY_SERVICE (systemd --user unit whose journal holds ComfyUI's log; default comfyui)"""
import json, os, re, subprocess, sys, threading, time, urllib.request
from benchlib import COMFY as API, BENCH_OUT
from bench_graphs import VSA                      # FastH3's sparse-attention settings (keep 10 %)

SERVICE = os.environ.get("COMFY_SERVICE", "comfyui").removesuffix(".service")
PROMPT = ("Cinematic live-action, golden hour. An elderly street musician in a grey flat cap plays a worn accordion on a wet cobblestone "
          "square, warm street lamps glowing behind him with soft bokeh, people walking past out of focus. Slow push-in on a 50mm lens, "
          "shallow depth of field, natural skin texture, film grain. Sound: the accordion plays a gentle waltz, footsteps on cobblestones, "
          "distant chatter and a church bell far away.")
SEED = 20261009


def frames(sec): n = max(5, round(sec * 24)); return n + (5 - n % 17) % 17          # the model's 17k+5 grid


def graph(preset, sec):
    W, H, unet, steps, lora, attn, shift, sparse = {
        "draft":    (864, 480, "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors", 8, None, "comfy kitchen attention", (10, 3), True),
        "standard": (1280, 720, "minimax_h3_fl2va_pruned_int8_convrot.safetensors", 8, "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors", "pytorch attention", None, False),
        "high":     (1344, 768, "minimax_h3_fl2va_pruned_int8_convrot.safetensors", 20, None, "pytorch attention", None, False),
    }[preset]
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "5": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": PROMPT, "width": W, "height": H, "length": frames(sec)}},
    }
    m = ["1", 0]
    if lora: g["6"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": m, "lora_name": lora, "strength_model": 1.0}}; m = ["6", 0]
    if shift: g["7"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": m, "shift_video": float(shift[0]), "shift_audio": float(shift[1])}}; m = ["7", 0]
    g["8"] = {"class_type": "ModelAttentionBackend", "inputs": {"model": m, "attention": attn}}; m = ["8", 0]
    if sparse:
        g["9"] = {"class_type": "BlockSparseAttention", "inputs": dict(VSA, model=m)}
        m = ["9", 0]
    g.update({
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "12": {"class_type": "BasicScheduler", "inputs": {"model": m, "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "13": {"class_type": "BasicGuider", "inputs": {"model": m, "conditioning": ["5", 0]}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["13", 0], "sampler": ["11", 0], "sigmas": ["12", 0], "latent_image": ["5", 1]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["3", 0]}},
        "16": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["4", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "audio": ["16", 0], "fps": 24.0}},
        "18": {"class_type": "SaveVideo", "inputs": {"video": ["17", 0], "filename_prefix": f"bench_presets/{preset}_{int(sec)}s", "format": "auto", "format.codec": "auto"}},
    })
    return g, (W, H, steps)


def post(path, body):
    return json.load(urllib.request.urlopen(urllib.request.Request(API + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=30))


def main():
    preset = sys.argv[1]; sec = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    g, (W, H, steps) = graph(preset, sec)
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout
                u, p, t = [x.strip() for x in o.split(",")]
                mem = [l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]
                samples.append((time.time(), float(u), float(p) if p not in ("[N/A]", "N/A") else None, int(t), int(mem) / 1048576))
            except Exception: pass
            time.sleep(1)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    since = time.strftime("%Y-%m-%d %H:%M:%S"); t0 = time.time()
    pid = post("/prompt", {"prompt": g, "client_id": "bench-presets"})["prompt_id"]
    print(f"{preset}: queued {pid} ({W}x{H}, {frames(sec)} frames, {steps} steps)", flush=True)
    while True:
        time.sleep(3)
        h = json.load(urllib.request.urlopen(f"{API}/history/{pid}", timeout=30))
        if pid in h: status = h[pid].get("status", {}); break
        if time.time() - t0 > 4 * 3600: status = {"status_str": "timeout"}; break
    wall = time.time() - t0; stop.set(); th.join(timeout=3)
    log = subprocess.run(["journalctl", "--user", "-u", SERVICE, "--since", since, "--no-pager", "-q", "-o", "cat"], capture_output=True, text=True).stdout
    its = re.findall(r"(\d+)/(\d+) \[[\d:]+<[\d:?]+,\s*([\d.]+)(s/it|it/s)\]", log)
    step_s = [float(x[2]) if x[3] == "s/it" else 1 / float(x[2]) for x in its if x[0] == x[1]]   # the final line of each progress bar
    errs = [l for l in log.splitlines() if re.search(r"Error|Traceback|out of memory|OOM", l)][-5:]
    busy = [s for s in samples if s[1] > 0]
    res = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "preset": preset, "seconds": sec, "frames": frames(sec), "w": W, "h": H, "steps": steps,
           "status": status.get("status_str"), "wall_s": round(wall, 1), "sampler_s_per_step": step_s,
           "gpu_util_mean": round(sum(s[1] for s in samples) / len(samples), 1) if samples else None,
           "gpu_util_busy_mean": round(sum(s[1] for s in busy) / len(busy), 1) if busy else None,
           "power_max_w": max((s[2] for s in samples if s[2] is not None), default=None), "temp_max_c": max((s[3] for s in samples), default=None),
           "mem_avail_min_gib": round(min((s[4] for s in samples), default=0), 1), "errors": errs}
    os.makedirs(BENCH_OUT, exist_ok=True)
    with open(os.path.join(BENCH_OUT, "presets_comfyui.jsonl"), "a") as f: f.write(json.dumps(res) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
