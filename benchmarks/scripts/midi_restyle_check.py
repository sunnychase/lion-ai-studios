#!/usr/bin/env python3
"""midi_restyle_check.py — how much of a MIDI arrangement survives an ACE-Step "restyle" at different strengths.

Step 1 (render): the MIDI render (a WAV made by app/midi_tool.py, copied into ComfyUI's input folder as bx_midi.wav) is encoded with
ACE-Step's VAE and re-sampled with KSampler at denoise = 0.4, 0.6 and 0.8 (8 steps, seed 5, lo-fi jazz style tags). Wall time per render.
Step 2 (score): frame-by-frame chroma correlation (librosa chroma_cqt, hop 2048, first 36 s, mono 22.05 kHz) between the MIDI render and
each restyle; an unrelated ACE-Step piece (score_fast.mp3 from bench_music.py) is scored the same way as a floor.
Higher = more of the original harmony/melody kept. One render per strength; this is a sanity check, not a listening test.

env: MIDI_RENDER (default $BENCH_OUT/arrangement_render.wav), UNRELATED (default $BENCH_OUT/music/score_fast.mp3)
usage: midi_restyle_check.py [--score-only]"""
import os, shutil, subprocess, sys
import benchlib as B

OUT = os.path.join(B.BENCH_OUT, "music"); os.makedirs(OUT, exist_ok=True)
MIDI_RENDER = os.environ.get("MIDI_RENDER", os.path.join(B.BENCH_OUT, "arrangement_render.wav"))
UNRELATED = os.environ.get("UNRELATED", os.path.join(OUT, "score_fast.mp3"))
STYLE = "lo-fi jazz hop, mellow Rhodes piano, warm upright bass, dusty brushed drums, vinyl crackle, relaxed late-night mood"


def graph(denoise, seconds):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ace_step_1.5_turbo_aio.safetensors"}},
        "2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
        "3": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {"clip": ["1", 1], "tags": STYLE, "lyrics": "[Instrumental]", "seed": 5, "bpm": 110, "duration": seconds,
              "timesignature": "4", "language": "en", "keyscale": "C major", "generate_audio_codes": False, "cfg_scale": 2.0, "temperature": 0.85,
              "top_p": 0.9, "top_k": 0, "min_p": 0.0}},
        "4": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}},
        "20": {"class_type": "LoadAudio", "inputs": {"audio": "bx_midi.wav"}},
        "21": {"class_type": "VAEEncodeAudio", "inputs": {"audio": ["20", 0], "vae": ["1", 2]}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "seed": 5, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
              "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["21", 0], "denoise": denoise}},
        "7": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveAudio", "inputs": {"audio": ["7", 0], "filename_prefix": f"bench_music/restyle_{int(denoise * 100)}"}},
    }


def chroma(path):
    import librosa
    y, sr = librosa.load(path, sr=22050, mono=True, duration=36)
    return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048), len(y) / sr


def score():
    import numpy as np
    ref, _ = chroma(MIDI_RENDER)
    for f in [f"{OUT}/restyle_{d}.wav" for d in (40, 60, 80)] + [UNRELATED]:
        if not os.path.exists(f): continue
        c, dur = chroma(f); n = min(ref.shape[1], c.shape[1]); a, b = ref[:, :n], c[:, :n]
        m = float(np.mean([np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(n) if a[:, i].std() > 0 and b[:, i].std() > 0]))
        B.append("midi_restyle.jsonl", {"file": os.path.basename(f), "seconds": round(dur, 1), "chroma_match": round(m, 2), "floor": f == UNRELATED})


if __name__ == "__main__":
    if "--score-only" not in sys.argv:
        shutil.copy(MIDI_RENDER, os.path.join(B.CI, "bx_midi.wav"))
        secs = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", MIDI_RENDER], capture_output=True, text=True).stdout)
        for d in (0.4, 0.6, 0.8):
            files, took = B.render(graph(d, round(secs, 1)), f"restyle{d}", client="bench_music")
            f = next(x for x in files if x.endswith(".flac")); out = f"{OUT}/restyle_{int(d * 100)}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f, out], check=True)
            B.append("midi_restyle.jsonl", {"denoise": d, "wall_s": round(took, 1), "file": os.path.basename(out)})
    score()
