#!/usr/bin/env python3
# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""bench_voice.py — voice cloning and narration with Qwen3-TTS-12Hz-1.7B-Base, through the studio's own voice worker (app/voice.py).
Run it with the tts venv's python (the one in studio.env's STUDIO_TTS_PY), e.g.  $STUDIO_TTS_PY bench_voice.py

The reference voice is SYNTHETIC: the audio of bench_lipsync.py's Draft clip of a fictional speaker (VOICE_REF, default
$BENCH_OUT/lipsync/voice_src.mp4; about 5 s of speech). Never use a real person's voice without their consent.

The studio clones a voice from a YouTube Short: download, Demucs (music out), Whisper (timestamps), the director picks a stretch of the
wanted speaker, cut, and a spoken preview. Here the clip is a local file, so there is no download, and the whole of its speech is the
reference, so there is no director pick. The rest is the studio's code:
  prep       the capture steps after the download: ffmpeg to 44.1 kHz WAV, voice.separate (Demucs htdemucs), Whisper large-v3-turbo with
             word timestamps (as voice.capture does it; model load included)
  cut        voice.cut: the speech span -> ref.wav (24 kHz mono, loudness-normalised) + the words spoken in it
  preview    `voice.py say` with the studio's preview line, as its own process (as the studio runs it: model load included)
  -> clone time = prep + cut + preview
  narration  `voice.py say` with a ~30 s script (126 words), as its own process (load included): realtime factor = seconds of speech / wall seconds
  gen x2     the same script twice in this process with the model loaded once: load_s, then generation only -> realtime factor
Per stage: wall time, peak memory in use on the machine, highest GPU temperature (benchlib.Sampler).
-> $BENCH_OUT/voice.jsonl and WAVs  (results: benchmarks/results/voice_qwen3_tts.csv)"""
import json, os, subprocess, sys, time
import benchlib as B

OUT = os.path.join(B.BENCH_OUT, "voice"); os.makedirs(OUT, exist_ok=True)
REF = os.path.expanduser(os.environ.get("VOICE_REF", os.path.join(B.BENCH_OUT, "lipsync", "voice_src.mp4")))
VOICE_PY = os.path.join(B.APP, "voice.py")
PREVIEW = "Hello. This is a preview of the matched voice, ready for your next film."      # the studio's default preview line
SCRIPT = ("The lighthouse on the point was built in eighteen ninety, from granite carried across the bay by boat. For a hundred years a keeper "
          "climbed its stairs every evening to light the lamp. Today the light runs by itself, but the village still gathers on the harbour "
          "wall each autumn to watch the beam come on, and the oldest fishermen still tell the story of the winter storm when the keeper "
          "kept it burning for three nights without sleep. Now the old stairs are open to visitors on summer weekends, and from the gallery at the "
          "top you can see the ferries crossing to the islands, the fishing boats coming home, and on a clear evening the lights of the "
          "mainland far away across the water.")


def audio_s(path):
    import soundfile as sf
    i = sf.info(path); return round(i.frames / i.samplerate, 2)


def stage(name, fn):
    with B.Sampler() as s:
        t = time.time(); extra = fn() or {}; wall = time.time() - t
    row = {"stage": name, "wall_s": round(wall, 1), **extra, **s.stats()}; B.append("voice.jsonl", row); return row


def prep():
    import torch
    from transformers import pipeline
    B.ff("-i", REF, "-vn", "-ac", "2", "-ar", "44100", f"{OUT}/source.wav")
    t = time.time(); V.separate(f"{OUT}/source.wav", f"{OUT}/vocals.wav"); demucs = time.time() - t
    t = time.time()
    asr = pipeline("automatic-speech-recognition", model=V.WHISPER, dtype=torch.float16, device="cuda:0")
    out = asr(f"{OUT}/vocals.wav", return_timestamps="word", chunk_length_s=30, batch_size=4, generate_kwargs={"task": "transcribe"})
    words = [{"t0": round(c["timestamp"][0] or 0, 2), "t1": round(c["timestamp"][1] or (c["timestamp"][0] or 0) + 0.3, 2), "w": c["text"]}
             for c in out.get("chunks", []) if c.get("text", "").strip()]
    whisper = time.time() - t
    json.dump({"segments": [{"start": words[0]["t0"], "end": words[-1]["t1"], "text": out.get("text", "").strip()}], "words": words,
               "text": out.get("text", "").strip()}, open(f"{OUT}/transcript.json", "w"), indent=1)
    del asr; torch.cuda.empty_cache()
    return {"demucs_s": round(demucs, 1), "whisper_s": round(whisper, 1), "clip_s": audio_s(f"{OUT}/source.wav"), "transcript": out.get("text", "").strip()}


def cut():
    w = json.load(open(f"{OUT}/transcript.json"))["words"]
    start, end = w[0]["t0"], max(w[-1]["t1"], w[0]["t0"] + 3.0)
    r = subprocess.run([sys.executable, VOICE_PY, "cut", OUT, str(start), str(end)], capture_output=True, text=True)
    d = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1])
    if not d.get("ok"): raise RuntimeError(d.get("error"))
    return {"ref_s": d["seconds"], "ref_text": d["ref_text"]}


def say(text, out):
    r = subprocess.run([sys.executable, VOICE_PY, "say", OUT, text, out], capture_output=True, text=True, timeout=900)
    d = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1])
    if not d.get("ok"): raise RuntimeError(d.get("error"))
    return {"audio_s": d["seconds"], "file": os.path.basename(out)}


def gen_twice():
    import torch, soundfile as sf
    from qwen_tts import Qwen3TTSModel
    ref = json.load(open(f"{OUT}/ref.json"))
    t = time.time(); m = Qwen3TTSModel.from_pretrained(V.QTTS, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"); load = time.time() - t
    B.append("voice.jsonl", {"stage": "load", "wall_s": round(load, 1)})
    for k in (1, 2):
        def one(k=k):
            wavs, sr = m.generate_voice_clone(text=SCRIPT, language="Auto", ref_audio=f"{OUT}/ref.wav", ref_text=ref["ref_text"])
            f = f"{OUT}/narration_gen{k}.wav"; sf.write(f, wavs[0], sr); return {"audio_s": round(len(wavs[0]) / sr, 2), "file": os.path.basename(f)}
        stage(f"gen{k}", one)


if __name__ == "__main__":
    if B.APP not in sys.path: sys.path.insert(0, B.APP)
    import voice as V
    B.studio_idle()
    stage("prep", prep)
    stage("cut", cut)
    stage("preview", lambda: say(PREVIEW, f"{OUT}/preview.wav"))
    stage("narration", lambda: say(SCRIPT, f"{OUT}/narration.wav"))
    gen_twice()
