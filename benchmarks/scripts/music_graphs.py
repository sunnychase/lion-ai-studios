# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""music_graphs.py — the music graphs the benchmark used (ComfyUI v0.39 native nodes; settings from ComfyUI's own templates
audio_ace_step_1_5_checkpoint.json and audio_minimax_music_3.json). Same as app/music.py's graphs at benchmark time.
  fast  ACE-Step 1.5 Turbo (one 10 GB checkpoint): style tags + lyrics, BPM, key; an LM writes audio codes first, then 8 steps
  best  MiniMax Music 3 (int8 DiT + int8 text model + DAV decoder): a structured brief + lyrics; the text model plans the song
        (and its length, up to max_duration), then 30 steps"""
import os
from benchlib import COMFY_MODELS as MODELS_DIR

ACE_CKPT = "ace_step_1.5_turbo_aio.safetensors"
MM3 = {"unet": "minimax_music3_dit_int8_convrot.safetensors", "clip": "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
       "vae": "minimax_music3_dav.safetensors"}
ENGINES = {"fast": "ACE-Step 1.5 Turbo", "best": "MiniMax Music 3"}
KEYS = [f"{n} {m}" for m in ("major", "minor") for n in ("C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B")]


def ready(engine):
    if engine == "fast": return os.path.exists(f"{MODELS_DIR}/checkpoints/{ACE_CKPT}")
    return all(os.path.exists(f"{MODELS_DIR}/{d}/{f}") for d, f in (("diffusion_models", MM3["unet"]), ("text_encoders", MM3["clip"]), ("vae", MM3["vae"])))


def instrumental(lyrics):
    return lyrics.strip() or "[Instrumental]"


def ace_graph(style, lyrics, seconds, seed, prefix, bpm=120, key="C major", timesig="4", language="en", codes=True):
    seconds = float(seconds)
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ACE_CKPT}},
        "2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
        "3": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {"clip": ["1", 1], "tags": style, "lyrics": instrumental(lyrics), "seed": int(seed), "bpm": int(bpm),
              "duration": seconds, "timesignature": str(timesig), "language": language, "keyscale": key if key in KEYS else "C major",
              "generate_audio_codes": bool(codes), "cfg_scale": 2.0, "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0}},
        "4": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}},
        "5": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {"seconds": seconds, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "seed": int(seed), "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
              "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0], "denoise": 1.0}},
        "7": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveAudio", "inputs": {"audio": ["7", 0], "filename_prefix": prefix}},
    }


def mm3_graph(brief, lyrics, seconds, seed, prefix):
    return {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": MM3["clip"], "type": "minimax", "device": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": MM3["unet"], "weight_dtype": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MM3["vae"]}},
        "4": {"class_type": "MiniMaxMusic3TextEncode", "inputs": {"clip": ["1", 0], "caption": brief, "lyrics": instrumental(lyrics), "seed": int(seed),
              "max_duration": float(seconds), "cfg_scale": 1.7, "top_k": 50}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyMiniMaxMusic3LatentAudio", "inputs": {"seconds": ["4", 1], "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "seed": int(seed), "steps": 30, "cfg": 1.7, "sampler_name": "euler", "scheduler": "simple",
              "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "denoise": 1.0}},
        "8": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveAudio", "inputs": {"audio": ["8", 0], "filename_prefix": prefix}},
    }


def graph(engine, style, lyrics, seconds, seed, prefix, **kw):
    return ace_graph(style, lyrics, seconds, seed, prefix, **kw) if engine == "fast" else mm3_graph(style, lyrics, seconds, seed, prefix)
