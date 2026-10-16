# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""Lion AI Studios — all-in-one image generator node for ComfyUI.

One `model_name` dropdown covers every supported image model; choosing one switches the whole pipeline (loader, text encoder, VAE,
sampler recipe), which a plain checkpoint-filename swap cannot do across architectures. Only one model family stays resident.
Install: copy this file and flux3_klein_system_prompts.json into ComfyUI/custom_nodes/ (ComfyUI >= 0.39 for Qwen-Image 2.1;
FLUX 3 Image Klein needs ComfyUI's FLUX 3 support, see scripts/install_comfyui.sh)."""

import gc
import json
import math
import os

import torch
import inspect

import comfy.model_management
import nodes


def _call(node_name, **kwargs):
    """Invoke a registered node's entry function; works for v1 and v3 API nodes."""
    cls = nodes.NODE_CLASS_MAPPINGS[node_name]
    fn = getattr(cls, cls.FUNCTION)
    if inspect.ismethod(fn):  # v3 classmethod
        return fn(**kwargs)
    return fn(cls(), **kwargs)


MODELS = {
    "Z-Image-Turbo (photoreal, best)": {"snap": 16, "steps": 8},   # Tongyi Z-Image-Turbo 6B, Apache-2.0
    "Krea 2 Turbo (photoreal)": {"snap": 16, "steps": 8},
    "Krea 2 Turbo + realism LoRA": {"snap": 16, "steps": 8},
    "Juggernaut Z Fast (4 steps)": {"snap": 16, "steps": 4},               # RunDiffusion's Fast release: 4 steps, Turbo-style workflow
    "Juggernaut Z (Z-Image photoreal finetune)": {"snap": 16, "steps": 35},  # a Z-Image BASE fine-tune: its card says 35 steps, CFG 6
    "Chroma1-HD (FLUX-class)": {"snap": 16, "steps": 26},
    # Qwen-Image 2.1 needs ComfyUI >= 0.39 (QwenImage21Cache + TextEncodeQwenImage21); check the Qwen licence before commercial use
    "Qwen-Image 2.1 Turbo (8 steps, 2K, best text)": {"snap": 32, "steps": 8},  # bf16, the checkpoint's own sigmas
    "Qwen-Image 2.1 (newest, best text)": {"snap": 32, "steps": 25},
    "Qwen-Image 2512 (fast, best text)": {
        "snap": 16,
        "steps": 8,
    },
    "HiDream-O1 (best quality)": {
        "snap": 32,
        "steps": 40,
    },
    "FLUX.2 klein (fastest)": {
        "snap": 32,
        "steps": 4,
    },
    # Black Forest Labs' FLUX 3 Image Klein, distilled 5.6B, bf16. Recipe = BFL's ComfyUI template image_flux3_klein_distilled_t2i:
    # 16 steps, CFG 1, euler / simple, the prompt through BFL's own upsampler first. FLUX Kommunity License (see docs/MODELS.md).
    "FLUX 3 Image Klein (newest, edits, 16 steps)": {"snap": 16, "steps": 16},
}

F3_UNET = "flux3_image_klein_sd_bf16.safetensors"
F3_TE = "qwen3vl_4b_bf16.safetensors"
F3_PUP = "qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors"
F3_VAE = "flux3_klein_vae.safetensors"   # BFL's flux2-vae from the Klein repo, saved under its own name (not the same file as FLUX.2's)
_F3_SYS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flux3_klein_system_prompts.json")   # BFL's upsampler prompts, verbatim
_pup = {"clip": None}


def flux3_upsample(prompt, seed, width=0, height=0, image=None, edit=False):
    """BFL's card: do not run Klein without its prompt upsampler — a short request becomes a full scene caption.
    Settings = the templates' TextGenerate: 3584 tokens, temperature 1, top_k 1, top_p 0.95, repetition 1.05, presence 1.5 (t2i) / 0 (edit), MTP off."""
    if _pup["clip"] is None:
        _pup["clip"] = _call("CLIPLoader", clip_name=F3_PUP, type="flux3", device="default")[0]
    req = prompt
    if width and height and not edit:   # the t2i system prompt reads an optional "TARGET CANVAS: W:H" first line
        g = math.gcd(int(width), int(height)); req = f"TARGET CANVAS: {width // g}:{height // g}\n{prompt}"
    mode = {"sampling_mode": "on", "temperature": 1.0, "top_k": 1, "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.05,
            "seed": int(seed) & 0xFFFFFFFF, "presence_penalty": 0.0 if edit else 1.5}
    sysp = json.load(open(_F3_SYS_FILE))["edit" if edit else "t2i"]
    out = _call("TextGenerate", clip=_pup["clip"], prompt=req, max_length=3584, sampling_mode=mode, image=image, thinking=False,
                use_default_template=True, mtp="off", system_prompt=sysp)[0]
    out = (out or "").strip()
    print(f"[Lion AI Studios] FLUX 3 Klein prompt upsampled ({len(prompt)} -> {len(out)} chars): {out[:400]}")
    return out or prompt

_cache = {"key": None, "loaded": None}


def _load(choice):
    if _cache["key"] == choice:
        return _cache["loaded"]
    # keep at most one model family resident; free the previous one first
    _cache["key"], _cache["loaded"] = None, None
    _pup["clip"] = None   # the FLUX 3 upsampler goes with its family
    gc.collect()
    comfy.model_management.soft_empty_cache()

    if choice.startswith("Z-Image"):
        unet = _call("UNETLoader", unet_name="z_image_turbo_bf16.safetensors", weight_dtype="default")[0]
        unet = _call("ModelSamplingAuraFlow", model=unet, shift=3.0)[0]
        clip = _call("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="lumina2", device="default")[0]
        vae = _call("VAELoader", vae_name="ae.safetensors")[0]
    elif choice.startswith("Krea 2"):
        unet = _call("UNETLoader", unet_name="krea2_turbo_bf16.safetensors", weight_dtype="default")[0]
        if "realism" in choice:
            unet = _call("LoraLoaderModelOnly", model=unet, lora_name="Krea2-realism-V2.safetensors", strength_model=0.8)[0]
        clip = _call("CLIPLoader", clip_name="qwen3vl_4b_fp8_scaled.safetensors", type="krea2", device="default")[0]
        vae = _call("VAELoader", vae_name="qwen_image_vae.safetensors")[0]
    elif choice.startswith("Juggernaut Z Fast"):   # before the full model: same prefix
        unet = _call("UNETLoader", unet_name="Juggernaut_Z_V1_Fast_FP16.safetensors", weight_dtype="default")[0]
        unet = _call("ModelSamplingAuraFlow", model=unet, shift=3.0)[0]
        clip = _call("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="lumina2", device="default")[0]
        vae = _call("VAELoader", vae_name="ae.safetensors")[0]
    elif choice.startswith("Juggernaut Z"):
        unet = _call("UNETLoader", unet_name="Juggernaut_Z_V1_by_RunDiffusion_fp16.safetensors", weight_dtype="default")[0]
        unet = _call("ModelSamplingAuraFlow", model=unet, shift=3.0)[0]
        clip = _call("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="lumina2", device="default")[0]
        vae = _call("VAELoader", vae_name="ae.safetensors")[0]
    elif choice.startswith("Chroma"):
        unet = _call("UNETLoader", unet_name="Chroma1-HD.safetensors", weight_dtype="default")[0]
        clip = _call("CLIPLoader", clip_name="t5xxl_fp8_e4m3fn_scaled.safetensors", type="chroma", device="default")[0]
        vae = _call("VAELoader", vae_name="ae.safetensors")[0]
    elif choice.startswith("Qwen-Image 2.1"):   # before the 2512 branch: startswith("Qwen") would catch it
        un = "qwen_image_2.1_turbo_bf16.safetensors" if "Turbo" in choice else "qwen_image_2.1_int8_convrot.safetensors"
        unet = _call("UNETLoader", unet_name=un, weight_dtype="default")[0]
        unet = _call("QwenImage21Cache", model=unet, device="auto", dtype="default")[0]
        clip = _call("CLIPLoader", clip_name="qwen3vl_8b_int8_convrot.safetensors", type="qwen_image", device="default")[0]
        vae = _call("VAELoader", vae_name="qwen_image_2.1_vae_bf16.safetensors")[0]
    elif choice.startswith("Qwen"):
        unet = _call("UNETLoader", unet_name="qwen_image_2512_fp8_e4m3fn.safetensors", weight_dtype="default")[0]
        unet = _call("LoraLoaderModelOnly", model=unet,
                     lora_name="Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors", strength_model=1.0)[0]
        unet = _call("ModelSamplingAuraFlow", model=unet, shift=3.1)[0]
        clip = _call("CLIPLoader", clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors", type="qwen_image", device="default")[0]
        vae = _call("VAELoader", vae_name="qwen_image_vae.safetensors")[0]
    elif choice.startswith("HiDream"):
        model, clip, vae = _call("CheckpointLoaderSimple", ckpt_name="hidream_o1_image_fp8_scaled.safetensors")[:3]
        base = _call("ModelNoiseScale", model=model, noise_scale=8.0)[0]
        patched = _call("HiDreamO1PatchSeamSmoothing", model=base, start_percent=0.8, end_percent=1.0,
                        pattern="single_shift", passes="ramp_2_4", blend="median", strength=1.0)[0]
        unet = {"base": base, "patched": patched}
    elif choice.startswith("FLUX 3"):
        unet = _call("UNETLoader", unet_name=F3_UNET, weight_dtype="default")[0]
        clip = _call("CLIPLoader", clip_name=F3_TE, type="flux3", device="default")[0]
        vae = _call("VAELoader", vae_name=F3_VAE)[0]
    else:  # FLUX.2 klein
        unet = _call("UNETLoader", unet_name="flux-2-klein-4b.safetensors", weight_dtype="default")[0]
        clip = _call("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="flux2", device="default")[0]
        vae = _call("VAELoader", vae_name="flux2-vae.safetensors")[0]

    loaded = (unet, clip, vae)
    _cache["key"], _cache["loaded"] = choice, loaded
    return loaded


class LionStudioImageGen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(MODELS.keys()),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2816, "step": 8}),  # Qwen-Image 2.1 2K presets
                "height": ("INT", {"default": 1024, "min": 256, "max": 2816, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "0 = model's tuned default"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "image"

    def generate(self, model_name, prompt, negative_prompt, width, height, batch_size, seed, steps):
        spec = MODELS[model_name]
        snap = spec["snap"]
        width, height = max(snap, width // snap * snap), max(snap, height // snap * snap)
        if steps <= 0:
            steps = spec["steps"]

        unet, clip, vae = _load(model_name)
        if model_name.startswith("Qwen-Image 2.1"):
            positive, negative = _call("TextEncodeQwenImage21", clip=clip, prompt=prompt, negative_prompt=negative_prompt, resolution=1024)[:2]
        elif model_name.startswith("FLUX 3"):   # distilled, CFG 1: the negative is never used
            positive = _call("CLIPTextEncode", clip=clip, text=flux3_upsample(prompt, seed, width, height))[0]
            negative = _call("ConditioningZeroOut", conditioning=positive)[0]
        else:
            positive = _call("CLIPTextEncode", clip=clip, text=prompt)[0]
            negative = _call("CLIPTextEncode", clip=clip, text=negative_prompt)[0]

        if model_name.startswith("Qwen-Image 2.1"):   # 64-ch latent at 1/16, as TextEncodeQwenImage21 builds it; recipe = factory template (25 steps, cfg 1, euler/simple)
            latent = {"samples": torch.zeros([batch_size, 64, height // 16, width // 16], device=comfy.model_management.intermediate_device())}
            if "Turbo" in model_name:  # Qwen/Qwen-Image-2.1-Turbo model_index.json sample_sigmas (8 steps) + the final 0
                sigmas = _call("ManualSigmas", sigmas="1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0")[0]
                sampler = _call("KSamplerSelect", sampler_name="euler")[0]
                samples = _call("SamplerCustom", model=unet, add_noise=True, noise_seed=seed, cfg=1.0, positive=positive, negative=negative,
                                sampler=sampler, sigmas=sigmas, latent_image=latent)[0]
            else:
                samples = _call("KSampler", model=unet, seed=seed, steps=steps, cfg=1.0, sampler_name="euler", scheduler="simple",
                                positive=positive, negative=negative, latent_image=latent, denoise=1.0)[0]
        elif model_name.startswith("Chroma"):
            latent = _call("EmptySD3LatentImage", width=width, height=height, batch_size=batch_size)[0]
            samples = _call("KSampler", model=unet, seed=seed, steps=steps, cfg=4.0,
                            sampler_name="euler", scheduler="beta",
                            positive=positive, negative=negative, latent_image=latent, denoise=1.0)[0]
        elif model_name.startswith(("Z-Image", "Qwen", "Krea 2", "Juggernaut Z")):
            latent = _call("EmptySD3LatentImage", width=width, height=height, batch_size=batch_size)[0]
            samples = _call("KSampler", model=unet, seed=seed, steps=steps, cfg=6.0 if model_name.startswith("Juggernaut Z (") else 1.0,
                            sampler_name="euler", scheduler="simple",
                            positive=positive, negative=negative, latent_image=latent, denoise=1.0)[0]
        elif model_name.startswith("FLUX 3"):   # BFL's template: EmptyFlux2LatentImage, KSampler 16 steps, CFG 1, euler / simple
            latent = _call("EmptyFlux2LatentImage", width=width, height=height, batch_size=batch_size)[0]
            samples = _call("KSampler", model=unet, seed=seed, steps=steps, cfg=1.0, sampler_name="euler", scheduler="simple",
                            positive=positive, negative=negative, latent_image=latent, denoise=1.0)[0]
        elif model_name.startswith("HiDream"):
            latent = _call("EmptyHiDreamO1LatentImage", width=width, height=height, batch_size=batch_size)[0]
            sampler = _call("KSamplerSelect", sampler_name="dpmpp_2m_sde_gpu")[0]
            sigmas = _call("BasicScheduler", model=unet["base"], scheduler="normal", steps=steps, denoise=1.0)[0]
            samples = _call("SamplerCustom", model=unet["patched"], add_noise=True, noise_seed=seed, cfg=5.0,
                            positive=positive, negative=negative, sampler=sampler,
                            sigmas=sigmas, latent_image=latent)[0]
        else:  # FLUX.2 klein (distilled: cfg 1, zeroed negative)
            latent = _call("EmptyFlux2LatentImage", width=width, height=height, batch_size=batch_size)[0]
            negative = _call("ConditioningZeroOut", conditioning=positive)[0]
            guider = _call("CFGGuider", model=unet, positive=positive, negative=negative, cfg=1.0)[0]
            sampler = _call("KSamplerSelect", sampler_name="euler")[0]
            sigmas = _call("Flux2Scheduler", steps=steps, width=width, height=height)[0]
            noise = _call("RandomNoise", noise_seed=seed)[0]
            samples = _call("SamplerCustomAdvanced", noise=noise, guider=guider,
                            sampler=sampler, sigmas=sigmas, latent_image=latent)[0]

        image = _call("VAEDecode", samples=samples, vae=vae)[0]
        return (image,)


NODE_CLASS_MAPPINGS = {"LionStudioImageGen": LionStudioImageGen}
NODE_DISPLAY_NAME_MAPPINGS = {"LionStudioImageGen": "Lion AI Studios · Image Generator"}
