"""bench_graphs.py — the ComfyUI API graphs the video benchmarks used (ComfyUI v0.39 native MiniMax-H3 nodes + the TensorFold node).

This is the benchmark copy of app/workflows.py as it stood when the numbers were taken (2026-10-09/10), plus two switches the app
does not have:
  ALLOW_FASTH3_ANY_INPUT  let the ComfyUI engine run FastH3 with first/last frames and guides (to compare it with TensorFold)
  H3_TF                   the experimental dense TensorFold path for Standard/High (merged bf16 files, not shipped — see BENCHMARKS.md)

Presets (5 s clip = 124 frames at 24 fps, sound included):
  draft     FastH3 8 steps 864x480
  fasthd    FastH3 8 steps 1344x768
  standard  H3 fl2va + Turbo 8-step LoRA 1248x704   (H3 needs both sides divisible by 32)
  high      H3 fl2va base 20 steps 1344x768"""
import os
from benchlib import COMFY_MODELS, COMFY_DIR

H3_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
FASTH3_UNET = "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"
TURBO8 = "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
H3_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_AVAE = "minimax_h3_audio_vae_fp32.safetensors"
# TensorFold's CUDA engine runs FastH3's transformer blocks in int8 on the tensor cores. It reads the bf16 checkpoint, converts it once
# and saves <file>.tf-int8 beside it; ComfyUI keeps the text encoder, sampler and decoders.
FASTH3_BF16 = "fastvideo_fasth3_8step_v2_pruned_bf16.safetensors"
TF_LIB = os.path.expanduser(os.environ.get("TENSORFOLD_LIB", os.path.join(COMFY_DIR, "custom_nodes/tensorfold_h3/libtf_h3.so")))
TF_SPARSITY = 0.8                # what FastH3 was trained with: each video tile attends to the top 20 % of tiles
H3_VAE_Q8 = "minimax_h3_video_vae_int8_convrot.safetensors"     # ComfyUI's 8-bit video decoder (decode only; encoding stays fp16)
# EXPERIMENTAL (not shipped): full H3 for the engine in dense mode — Turbo 8-step merged in for Standard, plain for High.
H3_TF = {"turbo": "minimax_h3_fl2va_turbo8sr_tf_bf16.safetensors", "base": "minimax_h3_fl2va_tf_bf16.safetensors"}
GUIDE_FRAMES = 22                # the previous clip's tail handed to the next one (frames AND audio)
FPS = 24
ALLOW_FASTH3_ANY_INPUT = False   # set True by bench_image_conditioning.py only
IMAGE_NODE = os.environ.get("IMAGE_NODE", "LionStudioImageGen")   # comfyui_nodes/lion_studio_image_gen.py
VSA = {"selection": "vsa", "selection.keep_percent": 10.0, "start_percent": 0.2, "end_percent": 1.0, "dense_blocks": "",
       "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows", "verbose": False}   # ComfyUI's FastH3 sparse attention

VIDEO = {
    "draft":    {"w": 864,  "h": 480, "steps": 8,  "fast": True},
    "fasthd":   {"w": 1344, "h": 768, "steps": 8,  "fast": True},
    "standard": {"w": 1248, "h": 704, "steps": 8,  "fast": False},
    "high":     {"w": 1344, "h": 768, "steps": 20, "fast": False},
}


def tf_ready():
    return os.path.exists(TF_LIB) and os.path.exists(os.path.join(COMFY_MODELS, "diffusion_models", FASTH3_BF16))


def tf_full_ready(kind):
    return os.path.exists(TF_LIB) and os.path.exists(os.path.join(COMFY_MODELS, "diffusion_models", H3_TF[kind]))


def vae8_ready():
    return os.path.exists(os.path.join(COMFY_MODELS, "vae", H3_VAE_Q8))


def frames(sec):
    n = max(5, round(float(sec) * FPS)); return n + (5 - n % 17) % 17          # the model's 17k+5 grid


def compose_prompt(picture, action, sound):
    parts = [picture.strip()] if picture.strip() else []
    if action.strip(): parts.append(action.strip())
    if sound.strip(): parts.append("Sound: " + sound.strip())
    return "\n\n".join(parts)


def video_graph(preset, prompt, sec, seed, prefix, first_image=None, guide_video=None, force_h3=False, last_image=None, engine="comfy", vae8=False):
    """first_image / last_image: files in ComfyUI's input dir. guide_video: the previous segment (input dir) whose last GUIDE_FRAMES frames
    AND audio are anchored at frame 0 via MiniMaxH3AddGuide (ComfyUI PR #15439); the caller sets the slice (nodes 33/34)."""
    p = VIDEO[preset]; W, H, steps = p["w"], p["h"], p["steps"]
    if W % 32 or H % 32: raise ValueError(f"{W}x{H}: H3 needs both sides divisible by 32")
    if ALLOW_FASTH3_ANY_INPUT:
        fast = p["fast"] and not force_h3
    else:   # the studio's rule: on TensorFold FastH3 takes any input; the ComfyUI engine uses FastH3 for text only
        tf_all = p["fast"] and engine == "tensorfold" and tf_ready()
        fast = tf_all or (p["fast"] and not first_image and not guide_video and not force_h3 and not last_image)
    tf = fast and engine == "tensorfold" and tf_ready()
    unet = FASTH3_BF16 if tf else FASTH3_UNET if fast else H3_UNET
    lora = TURBO8 if (preset == "standard" or (preset in ("draft", "fasthd") and not fast)) else None
    tf_full = not fast and engine == "tensorfold" and tf_full_ready("turbo" if lora else "base")
    if tf_full: unet = H3_TF["turbo" if lora else "base"]
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
    if last_image:
        g["36"] = {"class_type": "LoadImage", "inputs": {"image": last_image}}
        g["5"]["inputs"]["last_frame"] = ["36", 0]
    pos = ["5", 0]
    if guide_video:
        g["31"] = {"class_type": "LoadVideo", "inputs": {"file": guide_video}}
        g["32"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["31", 0]}}
        g["33"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["32", 0], "batch_index": 0, "length": GUIDE_FRAMES}}
        g["34"] = {"class_type": "TrimAudioDuration", "inputs": {"audio": ["32", 1], "start_index": 0.0, "duration": GUIDE_FRAMES / FPS}}
        g["35"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": pos, "latent": ["5", 1], "frame_idx": 0, "vae": ["3", 0],
                                                                 "audio_vae": ["4", 0], "image": ["33", 0], "audio": ["34", 0]}}
        pos = ["35", 0]
    m = ["1", 0]
    if lora and not tf_full: g["6"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": m, "lora_name": lora, "strength_model": 1.0}}; m = ["6", 0]
    if tf_full: g["9"] = {"class_type": "TensorFoldH3Blocks", "inputs": {"model": m, "unet_name": unet, "sparsity": 0.0}}; m = ["9", 0]
    if fast:
        g["7"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": m, "shift_video": 10.0, "shift_audio": 3.0}}; m = ["7", 0]
        g["8"] = {"class_type": "ModelAttentionBackend", "inputs": {"model": m, "attention": "comfy kitchen attention"}}; m = ["8", 0]
    if tf:
        g["9"] = {"class_type": "TensorFoldH3Blocks", "inputs": {"model": m, "unet_name": FASTH3_BF16, "sparsity": TF_SPARSITY}}; m = ["9", 0]
    elif fast:
        g["9"] = {"class_type": "BlockSparseAttention", "inputs": dict(VSA, model=m)}
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
    if vae8 and vae8_ready(): g["19"] = {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE_Q8}}
    return g, {"w": W, "h": H, "steps": steps, "frames": frames(sec), "vae": "8-bit" if vae8 and vae8_ready() else "fp16",
               "engine": ("FastH3 · TensorFold" if tf else "FastH3") if fast else ("H3 Turbo" if lora else "H3 base") + (" · TensorFold" if tf_full else "")}


def image_graph(model, prompt, w, h, count, seed, prefix, negative=""):
    return {
        "1": {"class_type": IMAGE_NODE, "inputs": {"model_name": model, "prompt": prompt, "negative_prompt": negative, "width": int(w),
                                                   "height": int(h), "batch_size": int(count), "seed": int(seed), "steps": 0}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": prefix}},
    }


# ── added 2026-10-12 for bench_cast.py and bench_image_edit.py: copies of app/workflows.py's cast_graph and edit_graph ──────────────────
REF2VA = {"unet": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "turbo4": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
          "gate": "fasth3_vsa_gate.safetensors"}
CAST_STEPS = 20
TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0"   # Qwen/Qwen-Image-2.1-Turbo model_index.json sample_sigmas


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
