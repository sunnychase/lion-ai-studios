#!/usr/bin/env python3
"""bench_lipsync.py — MiniMax H3 reference-to-video speaking a quoted line in a reference voice. Every input is synthetic:
the face is a Z-Image Turbo render (bx_img1.png from bench_image_conditioning.py) and the voice is the audio of a Draft clip of a
fictional speaker made here first. No real person's face or voice is used — keep it that way if you change the inputs.

  step 0  voice source: Draft 5 s clip (FastH3 on TensorFold, 8-bit decoder, seed 31); its audio -> bx_voice.wav
  step 1  ref_voice: ref2va int8 + ref2v Turbo 4-step LoRA v0.1 + VSA gate patch (sparsity 0.75) + sigma shift 12/3, 1344x768, 124 frames,
          seed 2024 — face image + reference voice + the quoted line
  step 2  no_voice: the same without the reference audio (control)
Per run: wall time (see benchlib). Afterwards: transcribe each clip (e.g. Whisper) to check the line, and compare the median pitch of the
speech with the reference (pitch() below, librosa pYIN 70-400 Hz).  -> $BENCH_OUT/lipsync/render.json  (results: benchmarks/results/lipsync_ref2va.csv)

Note: this 4-step recipe is the fast community one. The studio's Cast feature now uses MiniMax's official ref2va recipe (20 steps,
res_multistep, no Turbo LoRA, no gate patch) because the 4-step picture was visibly worse; see BENCHMARKS.md."""
import json, os, shutil, subprocess
import benchlib as B
import bench_graphs as W

OUT = os.path.join(B.BENCH_OUT, "lipsync"); os.makedirs(OUT, exist_ok=True)
LINE = "The lighthouse has been dark for twenty years, but tonight I am going to light it again."


def voice_clip():
    ref = f"{OUT}/voice_src.mp4"
    if not os.path.exists(ref):
        g, _ = W.video_graph("draft", W.compose_prompt("A woman in her thirties with short dark hair sits at a kitchen table, medium close-up, soft window light.",
                             'She looks at the camera and says clearly, "I grew up by the sea, and every morning I walked along the harbour wall before school."',
                             "Her voice only, quiet room."), 5, 31, "bench_ls/voice", engine="tensorfold", vae8=True)
        f, t = B.render(g, "voice", client="bench_ls"); shutil.copy(next(x for x in f if x.endswith(".mp4")), ref); print(f"voice source clip {t:.0f} s", flush=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", ref, "-vn", "-ac", "1", "-ar", "48000", f"{B.CI}/bx_voice.wav"], check=True)


def ref2va(tag, with_audio):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": W.H3_CLIP, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": W.H3_VAE}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": W.H3_AVAE}},
        "50": {"class_type": "LoadImage", "inputs": {"image": "bx_img1.png"}},
        "6": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["2", 0], "vae": ["4", 0], "audio_vae": ["5", 0], "width": 1344, "height": 768, "length": 124,
              "ref_image_size": "match", "ref_images.ref_image_0": ["50", 0],
              "prompt": f'The woman from the reference image stands on a windy harbour at dusk, medium close-up, cinematic. She looks at the camera and says, "{LINE}" '
                        "Her voice is the reference voice. Waves and wind in the background."}},
        "143": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "strength_model": 1.0}},
        "144": {"class_type": "Ref2VAVSAGatePatch", "inputs": {"model": ["143", 0], "gate_file": "fasth3_vsa_gate.safetensors", "sparsity": 0.75}},
        "142": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["144", 0], "shift_video": 12, "shift_audio": 3}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": 2024}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "10": {"class_type": "BasicScheduler", "inputs": {"model": ["142", 0], "scheduler": "simple", "steps": 4, "denoise": 1.0}},
        "11": {"class_type": "BasicGuider", "inputs": {"model": ["142", 0], "conditioning": ["6", 0]}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["8", 0], "guider": ["11", 0], "sampler": ["9", 0], "sigmas": ["10", 0], "latent_image": ["6", 1]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "audio": ["14", 0], "fps": 24.0}},
        "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": f"bench_ls/{tag}", "format": "mp4", "format.codec": "h264",
                                                     "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 14.0}},
    }
    if with_audio:
        g["51"] = {"class_type": "LoadAudio", "inputs": {"audio": "bx_voice.wav"}}
        g["6"]["inputs"]["ref_audios.ref_audio_0"] = ["51", 0]
    f, t = B.render(g, tag, client="bench_ls"); out = f"{OUT}/{tag}.mp4"; shutil.copy(next(x for x in f if x.endswith(".mp4")), out)
    return out, t


def pitch(path):
    """median F0 of the voiced frames (Hz)"""
    import numpy as np, librosa
    y, sr = librosa.load(path, sr=16000, mono=True); f0, v, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr)
    f0 = f0[v & ~np.isnan(f0)]; return float(np.median(f0)) if len(f0) else None


if __name__ == "__main__":
    voice_clip()
    res = {}
    for tag, wa in (("ref_voice", True), ("no_voice", False)):
        out, t = ref2va(tag, wa); res[tag] = {"file": os.path.basename(out), "wall_s": round(t, 1)}; print(tag, f"{t:.0f} s", flush=True)
    json.dump(res, open(f"{OUT}/render.json", "w"), indent=1)
