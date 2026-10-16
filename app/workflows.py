# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""workflows.py — ComfyUI API graphs for the studio (ComfyUI 0.39 native MiniMax-H3 nodes + the LionStudioImageGen image node).
Presets were benchmarked on a DGX Spark (5 s clip with sound): draft 129 s · standard 415 s · high 1,052 s.
(same 5 s clip, TensorFold engine): draft 76 s (70 s with the 8-bit decoder) · fast HD 192 s (ComfyUI 272 s)."""
import os
import config as C

H3_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
FASTH3_UNET = "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"
TURBO8 = "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
H3_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_AVAE = "minimax_h3_audio_vae_fp32.safetensors"
# TensorFold's CUDA engine runs FastH3's 50 blocks in int8 on the tensor cores (reads the bf16 checkpoint,
# converts it once and saves <file>.tf-int8 beside it); ComfyUI keeps the text encoder, sampler and decoders.
FASTH3_BF16 = "fastvideo_fasth3_8step_v2_pruned_bf16.safetensors"
TF_LIB = C.TF_LIB
TF_SPARSITY = 0.8                # what FastH3 was trained with: each video tile attends to the top 20% of tiles
H3_VAE_Q8 = "minimax_h3_video_vae_int8_convrot.safetensors"    # ComfyUI's 8-bit video decoder: ~2x faster decode, 42-44 dB of the fp16 one
GUIDE_FRAMES = 22                # the previous clip's tail handed to the next one (frames AND audio); a 17k+5 length the node keeps whole
FPS = 24

# preset -> (width, height, steps, s/step measured at 5 s, label)
VIDEO = {
    "draft":    {"w": 864,  "h": 480, "steps": 8,  "sps5": 11.8, "sps5_tf": 4.0, "label": "Draft · FastH3 · 480p",       "fast": True},
    # bench (5 s, same seed): ComfyUI 272 s · TensorFold 192 s. With an image or in a film it runs on H3 + Turbo at 768p
    "fasthd":   {"w": 1344, "h": 768, "steps": 8,  "sps5": 28.4, "sps5_tf": 18.4, "label": "Fast HD · FastH3 · 768p",  "fast": True},
    "standard": {"w": 1248, "h": 704, "steps": 8,  "sps5": 43.8, "label": "Standard · H3 Turbo · 704p",  "fast": False},   # /32: 1280x720 breaks first frames
    "high":     {"w": 1344, "h": 768, "steps": 20, "sps5": 51.3, "label": "High · H3 base · 768p",       "fast": False},
}
# image models — the LionStudioImageGen node's names, with the files each needs (the studio offers only what is installed)
IMAGE = {
    "Z-Image-Turbo (photoreal, best)":          ["diffusion_models/z_image_turbo_bf16.safetensors", "text_encoders/qwen_3_4b.safetensors"],
    "Krea 2 Turbo + realism LoRA":              ["diffusion_models/krea2_turbo_bf16.safetensors", "loras/Krea2-realism-V2.safetensors", "text_encoders/qwen3vl_4b_fp8_scaled.safetensors"],
    "Krea 2 Turbo (photoreal)":                 ["diffusion_models/krea2_turbo_bf16.safetensors", "text_encoders/qwen3vl_4b_fp8_scaled.safetensors"],
    "Qwen-Image 2.1 Turbo (8 steps, 2K, best text)": ["diffusion_models/qwen_image_2.1_turbo_bf16.safetensors", "text_encoders/qwen3vl_8b_int8_convrot.safetensors"],
    "Qwen-Image 2.1 (newest, best text)":       ["diffusion_models/qwen_image_2.1_int8_convrot.safetensors", "text_encoders/qwen3vl_8b_int8_convrot.safetensors"],
    "Qwen-Image 2512 (fast, best text)":        ["diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors", "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"],
    "Chroma1-HD (FLUX-class)":      ["diffusion_models/Chroma1-HD.safetensors", "text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors"],
    "HiDream-O1 (best quality)":                ["checkpoints/hidream_o1_image_fp8_scaled.safetensors"],
    "Juggernaut Z Fast (4 steps)":              ["diffusion_models/Juggernaut_Z_V1_Fast_FP16.safetensors", "text_encoders/qwen_3_4b.safetensors"],
    "Juggernaut Z (Z-Image photoreal finetune)": ["diffusion_models/Juggernaut_Z_V1_by_RunDiffusion_fp16.safetensors", "text_encoders/qwen_3_4b.safetensors"],
    "FLUX.2 klein (fastest)":                   ["diffusion_models/flux-2-klein-4b.safetensors", "text_encoders/qwen_3_4b.safetensors"],
    # Black Forest Labs' FLUX 3 Image Klein (distilled bf16) + its Qwen3-VL-4B encoder + BFL's prompt upsampler (Qwen3.5-9B int8) + its VAE
    "FLUX 3 Image Klein (newest, edits, 16 steps)": ["diffusion_models/flux3_image_klein_sd_bf16.safetensors", "text_encoders/qwen3vl_4b_bf16.safetensors",
        "text_encoders/qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors", "vae/flux3_klein_vae.safetensors"],
}
IMAGE_PRESET = {"fasthd": "Z-Image-Turbo (photoreal, best)", "draft": "Z-Image-Turbo (photoreal, best)", "standard": "Krea 2 Turbo + realism LoRA", "high": "Qwen-Image 2.1 Turbo (8 steps, 2K, best text)"}
MODELS_DIR = C.COMFY_MODELS
# the optional "Fast (LocalRouter)" image engine. LocalRouter's qwen-image-2.1-turbo (FP8, its checkpoint's own 8 steps, CFG 1, as the ComfyUI
# node runs it) is the same Qwen-Image 2.1 Turbo as this ComfyUI entry. Plain text to image only, and only what was measured: 1024x1024, one
# image per job. Everything else (other sizes, variations, 2K, edits, other models, Ming) stays on ComfyUI.
LR_IMAGE = {"Qwen-Image 2.1 Turbo (8 steps, 2K, best text)": C.LR_MODEL}; LR_RESERVE_GIB = C.LR_RESERVE_GIB
LR_GO_SIZES = ((1024, 1024),); LR_GO_N = 1                 # the measured envelope
# what LocalRouter itself accepts (not enabled; widen LR_GO_SIZES / LR_GO_N only after measuring each size and count):
LR_MAX_SIDE = 1664; LR_MIN_SIDE = 256; LR_SIDE_STEP = 16; LR_MAX_N = 4   # tools.json max_side · API sizes 256+ in steps of 16 · n 1-4


def lr_fits(model, w, h, count=1):
    """'' when an image job can take the fast path, else why it stays on ComfyUI"""
    try: w, h, count = int(w), int(h), int(count)
    except (TypeError, ValueError): return "no usable size"
    if model not in LR_IMAGE: return f"Fast runs Qwen-Image 2.1 Turbo only, not {model}"
    if (w, h) not in LR_GO_SIZES: return f"Fast takes {' / '.join(f'{a}x{b}' for a, b in LR_GO_SIZES)} only for now, not {w}x{h}"
    if count != LR_GO_N: return f"Fast makes {LR_GO_N} image per job for now, not {count}"
    if max(w, h) > LR_MAX_SIDE or min(w, h) < LR_MIN_SIDE or w % LR_SIDE_STEP or h % LR_SIDE_STEP: return f"{w}x{h} is outside what LocalRouter takes"   # for a wider envelope
    if not 1 <= count <= LR_MAX_N: return f"LocalRouter makes 1 to {LR_MAX_N} images"
    return ""


def lr_need_gib(w, h, warm=False):
    """MemAvailable LocalRouter needs to admit a job: its own estimate (36 000 MiB resident for fp8s, counted only when the model is not
    loaded yet, + w*h*8 bytes working) + its reserve (it refuses unless MemAvailable - reserve >= that) + 0.5 GiB margin
    -> ~50.7 GiB cold, ~15.5 GiB warm with a 15 GiB reserve. Measured peak at 1024x1024 on a DGX Spark: 42.6 GB."""
    return (0 if warm else 36000 / 1024) + int(w) * int(h) * 8 / 2 ** 30 + LR_RESERVE_GIB + 0.5


def tf_ready():
    return os.path.exists(TF_LIB) and os.path.exists(os.path.join(MODELS_DIR, "diffusion_models", FASTH3_BF16))


def vae8_ready():
    return os.path.exists(os.path.join(MODELS_DIR, "vae", H3_VAE_Q8))


def image_models():
    return [n for n, files in IMAGE.items() if all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in files)] + ([MING] if ming_ready() else [])


# Ming-Image-0.1-Design runs outside ComfyUI: its own venv + the official inclusionAI/Ming-Image code, one process per job (app/ming_runner.py)
MING = "Ming-Image 0.1 Design (posters, UI, text, transparent)"
MING_MODEL, MING_CODE, MING_PY, MING_RUNNER = C.MING_MODEL, C.MING_CODE, C.MING_PY, C.MING_RUNNER


def ming_need_gib(w, h):
    """free memory a Ming job waits for. Measured on a DGX Spark (MemAvailable drop during one run): ~55 GiB peak at 1024x1024,
    ~70 GiB at 2048x2048 -> about 50 + 5 per megapixel, plus 8 GiB headroom"""
    return 58 + 5 * int(w) * int(h) / 1048576


def ming_ready():
    """the weights (both halves of the checkpoint), the official code and its venv are all there"""
    need = (os.path.join(MING_MODEL, "transformer", "config.json"), os.path.join(MING_MODEL, "mllm", "config.json"),
            os.path.join(MING_CODE, "infer.py"), MING_PY, MING_RUNNER)
    return all(os.path.exists(p) for p in need)


def frames(sec):
    n = max(5, round(float(sec) * FPS)); return n + (5 - n % 17) % 17          # the model's 17k+5 grid


def estimate_s(preset, sec, engine="comfy"):
    """rough: per-step time grows a little faster than the frame count (attention). Measured at 5 s; longer is extrapolated."""
    p = VIDEO[preset]; k = frames(sec) / 124
    sps = p["sps5_tf"] if engine == "tensorfold" and p.get("sps5_tf") and tf_ready() else p["sps5"]
    return round(sps * p["steps"] * k ** 1.35 + 45 * k)


def compose_prompt(picture, action, sound):
    parts = [picture.strip()] if picture.strip() else []
    if action.strip(): parts.append(action.strip())
    if sound.strip(): parts.append("Sound: " + sound.strip())
    return "\n\n".join(parts)


def video_graph(preset, prompt, sec, seed, prefix, first_image=None, guide_video=None, force_h3=False, last_image=None, engine="comfy", vae8=False,
                dims=None):
    """preset: draft|standard|high. first_image: a file in ComfyUI's input dir (first frame). guide_video: the previous segment
    (input dir) whose last GUIDE_FRAMES frames AND audio are anchored at frame 0, so picture and sound carry on (ComfyUI PR #15439).
    FastH3 is text-to-video only, so a draft with an image or a guide runs on H3 + Turbo at 480p instead."""
    p = VIDEO[preset]; W, H = dims or (p["w"], p["h"]); steps = p["steps"]             # dims = (w, h) override (e.g. vertical)
    if W % 32 or H % 32: raise ValueError(f"{W}x{H}: H3 needs both sides divisible by 32 (its latent is patched 2x2 at 1/16 scale)")
    # on TensorFold, FastH3 takes first/last frames and guides too (benched 10-09: same quality, ~1.75x faster than H3+Turbo);
    # the ComfyUI engine keeps the old rule (FastH3 text-only; force_h3 = one engine for a whole film)
    tf_all = p["fast"] and engine == "tensorfold" and tf_ready()
    fast = tf_all or (p["fast"] and not first_image and not guide_video and not force_h3 and not last_image)
    tf = fast and engine == "tensorfold" and tf_ready()                    # the engine covers FastH3 only; anything else stays on ComfyUI
    unet = FASTH3_BF16 if tf else FASTH3_UNET if fast else H3_UNET
    lora = TURBO8 if (preset == "standard" or (preset in ("draft", "fasthd") and not fast)) else None
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_CLIP, "type": "minimax"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AVAE}},
        "5": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "width": W, "height": H,
                                                                "length": frames(sec)}},
    }
    if first_image:
        g["30"] = {"class_type": "LoadImage", "inputs": {"image": first_image}}
        g["5"]["inputs"]["first_frame"] = ["30", 0]
    if last_image:                  # first-and-last-frame: the clip travels from one keyframe to the next
        g["36"] = {"class_type": "LoadImage", "inputs": {"image": last_image}}
        g["5"]["inputs"]["last_frame"] = ["36", 0]
    pos = ["5", 0]
    if guide_video:
        g["31"] = {"class_type": "LoadVideo", "inputs": {"file": guide_video}}
        g["32"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["31", 0]}}
        g["33"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["32", 0], "batch_index": 0, "length": GUIDE_FRAMES}}  # batch_index set by caller
        g["34"] = {"class_type": "TrimAudioDuration", "inputs": {"audio": ["32", 1], "start_index": 0.0, "duration": GUIDE_FRAMES / FPS}}
        g["35"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": pos, "latent": ["5", 1], "frame_idx": 0, "vae": ["3", 0],
                                                                 "audio_vae": ["4", 0], "image": ["33", 0], "audio": ["34", 0]}}
        pos = ["35", 0]
    m = ["1", 0]
    if lora: g["6"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": m, "lora_name": lora, "strength_model": 1.0}}; m = ["6", 0]
    if fast:
        g["7"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": m, "shift_video": 10.0, "shift_audio": 3.0}}; m = ["7", 0]
        g["8"] = {"class_type": "ModelAttentionBackend", "inputs": {"model": m, "attention": "comfy kitchen attention"}}; m = ["8", 0]
    if tf:
        g["9"] = {"class_type": "TensorFoldH3Blocks", "inputs": {"model": m, "unet_name": FASTH3_BF16, "sparsity": TF_SPARSITY}}; m = ["9", 0]
    elif fast:
        g["9"] = {"class_type": "BlockSparseAttention", "inputs": {"model": m, "selection": "vsa", "selection.keep_percent": 10.0, "start_percent": 0.2,
                  "end_percent": 1.0, "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows", "verbose": False}}
        m = ["9", 0]
    g.update({
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "12": {"class_type": "BasicScheduler", "inputs": {"model": m, "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "13": {"class_type": "BasicGuider", "inputs": {"model": m, "conditioning": pos}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["13", 0], "sampler": ["11", 0], "sigmas": ["12", 0],
                                                                 "latent_image": ["5", 1]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["19", 0] if vae8 and vae8_ready() else ["3", 0]}},
        "16": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["4", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "audio": ["16", 0], "fps": float(FPS)}},
        "18": {"class_type": "SaveVideo", "inputs": {"video": ["17", 0], "filename_prefix": prefix, "format": "mp4", "format.codec": "h264",
                                                     "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 14.0}},
    })
    if vae8 and vae8_ready(): g["19"] = {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE_Q8}}    # decode only; encoding keeps fp16
    return g, {"w": W, "h": H, "steps": steps, "frames": frames(sec), "vae": "8-bit" if vae8 and vae8_ready() else "fp16",
               "engine": ("FastH3 · TensorFold" if tf else "FastH3") if fast else ("H3 Turbo" if lora else "H3 base")}


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
REF2VA = {"unet": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "turbo4": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
          "gate": "fasth3_vsa_gate.safetensors"}
CAST_SPS5 = 51.3                 # s per dense step at 1344x768, 5 s (as High); 20 steps — re-measured on the first official run
CAST_STEPS = 20


def cast_ready():
    return os.path.exists(os.path.join(MODELS_DIR, "diffusion_models", REF2VA["unet"]))


def cast_graph(preset, prompt, sec, seed, prefix, images, audio=None, vae8=False, dims=None):
    """images: up to 3 file names in ComfyUI's input dir (faces/characters); audio: a voice clip there (or None)"""
    p = VIDEO[preset]; W_, H_ = dims or (p["w"], p["h"])
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": REF2VA["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_CLIP, "type": "minimax"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AVAE}},
        "5": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0], "prompt": prompt, "width": W_, "height": H_,
                                                                    "length": frames(sec), "ref_image_size": "match"}},
        # MiniMax H3's own recipe (template video_minimax_h3_r2v.json) — no Turbo LoRA, no VSA gate patch, no shift
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "12": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": CAST_STEPS, "denoise": 1.0}},
        "13": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["13", 0], "sampler": ["11", 0], "sigmas": ["12", 0], "latent_image": ["5", 1]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["19", 0] if vae8 and vae8_ready() else ["3", 0]}},
        "16": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["4", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "audio": ["16", 0], "fps": float(FPS)}},
        "18": {"class_type": "SaveVideo", "inputs": {"video": ["17", 0], "filename_prefix": prefix, "format": "mp4", "format.codec": "h264",
                                                     "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 14.0}},
    }
    for i, im in enumerate(images[:3]):
        g[f"3{i}"] = {"class_type": "LoadImage", "inputs": {"image": im}}; g["5"]["inputs"][f"ref_images.ref_image_{i}"] = [f"3{i}", 0]
    if audio:
        g["40"] = {"class_type": "LoadAudio", "inputs": {"audio": audio}}; g["5"]["inputs"]["ref_audios.ref_audio_0"] = ["40", 0]
    if vae8 and vae8_ready(): g["19"] = {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE_Q8}}
    return g, {"w": W_, "h": H_, "steps": CAST_STEPS, "frames": frames(sec), "engine": "MiniMax H3 ref2va" + (" + voice" if audio else ""), "vae": "8-bit" if vae8 and vae8_ready() else "fp16"}


def cast_estimate_s(preset, sec):
    p = VIDEO[preset]; k = frames(sec) / 124; area = p["w"] * p["h"] / (1344 * 768)
    return round(CAST_SPS5 * CAST_STEPS * area * k ** 1.35 + 45 * k)


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────
EDIT_MODEL = "Qwen-Image 2.1 Turbo (8 steps, 2K, best text)"
TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0"   # Qwen/Qwen-Image-2.1-Turbo model_index.json sample_sigmas


def edit_ready():
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in IMAGE[EDIT_MODEL])


# FLUX 3 Image Klein edits too (BFL's template image_flux3_klein_distilled_image_edit). Qwen-Image 2.1 Turbo stays the default.
FLUX3K = "FLUX 3 Image Klein (newest, edits, 16 steps)"
EDIT_MODELS = (EDIT_MODEL, FLUX3K)
F3_SYS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comfyui_nodes", "flux3_klein_system_prompts.json")


def edit_models():
    return [m for m in EDIT_MODELS if all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in IMAGE[m])]


def flux3_edit_graph(images, prompt, seed, prefix):
    """BFL's edit template: the upsampler sees the references and writes the instruction → ReferenceMultiLatent (each reference ~1 MP;
    the output takes the first reference's shape at ~1 MP) → KSampler 16 steps, CFG 1, euler / simple"""
    import json
    sysp = json.load(open(F3_SYS_FILE))["edit"]
    g = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux3_image_klein_sd_bf16.safetensors", "weight_dtype": "default"}},
         "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_bf16.safetensors", "type": "flux3", "device": "default"}},
         "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux3_klein_vae.safetensors"}},
         "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors", "type": "flux3", "device": "default"}},
         "5": {"class_type": "BatchImagesNode", "inputs": {}},
         "6": {"class_type": "TextGenerate", "inputs": {"clip": ["4", 0], "prompt": prompt, "image": ["5", 0], "max_length": 3584, "sampling_mode": "on",
               "sampling_mode.temperature": 1.0, "sampling_mode.top_k": 1, "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.0,
               "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": int(seed) & 0xFFFFFFFF, "sampling_mode.presence_penalty": 0.0,
               "thinking": False, "use_default_template": True, "mtp": "off", "system_prompt": sysp}},
         "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ["6", 0]}},
         "8": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7", 0]}},
         "9": {"class_type": "ReferenceMultiLatent", "inputs": {"positive": ["7", 0], "negative": ["8", 0], "vae": ["3", 0], "resolution": "resize",
               "resolution.megapixels": 1.0}},
         "10": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": int(seed), "steps": 16, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                "positive": ["9", 0], "negative": ["9", 1], "latent_image": ["9", 2], "denoise": 1.0}},
         "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
         "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": prefix}}}
    for i, im in enumerate(images[:4]):
        g[f"2{i}"] = {"class_type": "LoadImage", "inputs": {"image": im}}
        g["5"]["inputs"][f"images.image{i}"] = [f"2{i}", 0]; g["9"]["inputs"][f"images.image{i}"] = [f"2{i}", 0]
    return g


def shape_dims(preset, src_w, src_h, shape="auto"):
    """the preset's size in the source's shape — a vertical Short renders vertical (both sides stay /32)"""
    w, h = VIDEO[preset]["w"], VIDEO[preset]["h"]
    return (h, w) if shape == "auto" and src_w and src_h and src_h > src_w else (w, h)


def edit_graph(images, prompt, seed, prefix, negative="", model=EDIT_MODEL):
    """images: 1-4 file names in ComfyUI's input dir; the output takes the first reference's size (rounded to /32)"""
    if model == FLUX3K: return flux3_edit_graph(images, prompt, seed, prefix)
    g = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_turbo_bf16.safetensors", "weight_dtype": "default"}},
         "2": {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"}},
         "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
         "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
         "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": prompt, "negative_prompt": negative, "resolution": 0}},
         "6": {"class_type": "ManualSigmas", "inputs": {"sigmas": TURBO_SIGMAS}},
         "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
         "8": {"class_type": "SamplerCustom", "inputs": {"model": ["2", 0], "add_noise": True, "noise_seed": int(seed), "cfg": 1.0, "positive": ["5", 0],
                                                          "negative": ["5", 1], "sampler": ["7", 0], "sigmas": ["6", 0], "latent_image": ["5", 2]}},
         "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
         "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}}}
    for i, im in enumerate(images[:4]):
        g[f"2{i}"] = {"class_type": "LoadImage", "inputs": {"image": im}}; g["5"]["inputs"][f"images.image_{i + 1}"] = [f"2{i}", 0]
    return g


def image_graph(model, prompt, w, h, count, seed, prefix, negative=""):
    return {
        "1": {"class_type": "LionStudioImageGen", "inputs": {"model_name": model, "prompt": prompt, "negative_prompt": negative, "width": int(w),
                                                            "height": int(h), "batch_size": int(count), "seed": int(seed), "steps": 0}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": prefix}},
    }
