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
    "Juggernaut Z (Z-Image photoreal finetune)": ["diffusion_models/Juggernaut_Z_V1_by_RunDiffusion_fp16.safetensors", "text_encoders/qwen_3_4b.safetensors"],
    "FLUX.2 klein (fastest)":                   ["diffusion_models/flux-2-klein-4b.safetensors", "text_encoders/qwen_3_4b.safetensors"],
}
IMAGE_PRESET = {"fasthd": "Z-Image-Turbo (photoreal, best)", "draft": "Z-Image-Turbo (photoreal, best)", "standard": "Krea 2 Turbo + realism LoRA", "high": "Qwen-Image 2.1 Turbo (8 steps, 2K, best text)"}
MODELS_DIR = C.COMFY_MODELS


def tf_ready():
    return os.path.exists(TF_LIB) and os.path.exists(os.path.join(MODELS_DIR, "diffusion_models", FASTH3_BF16))


def vae8_ready():
    return os.path.exists(os.path.join(MODELS_DIR, "vae", H3_VAE_Q8))


def image_models():
    return [n for n, files in IMAGE.items() if all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in files)]


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


def video_graph(preset, prompt, sec, seed, prefix, first_image=None, guide_video=None, force_h3=False, last_image=None, engine="comfy", vae8=False):
    """preset: draft|standard|high. first_image: a file in ComfyUI's input dir (first frame). guide_video: the previous segment
    (input dir) whose last GUIDE_FRAMES frames AND audio are anchored at frame 0, so picture and sound carry on (ComfyUI PR #15439).
    FastH3 is text-to-video only, so a draft with an image or a guide runs on H3 + Turbo at 480p instead."""
    p = VIDEO[preset]; W, H, steps = p["w"], p["h"], p["steps"]
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


def cast_graph(preset, prompt, sec, seed, prefix, images, audio=None, vae8=False):
    """images: up to 3 file names in ComfyUI's input dir (faces/characters); audio: a voice clip there (or None)"""
    p = VIDEO[preset]; W_, H_ = p["w"], p["h"]
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


def edit_graph(images, prompt, seed, prefix, negative=""):
    """images: 1-4 file names in ComfyUI's input dir; the output takes the first reference's size (rounded to /32)"""
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
