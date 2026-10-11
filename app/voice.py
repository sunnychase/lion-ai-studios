#!/usr/bin/env python3
"""voice.py — the studio's voice worker (runs in the tts venv: torch 2.10+cu130, qwen-tts, demucs, transformers Whisper).
  capture URL DIR        Short -> DIR/source.mp4 (<=480p, for the director to watch), source.wav, vocals.wav (Demucs: music removed),
                         transcript.json (Whisper large-v3-turbo, timestamped chunks)
  cut DIR START END      the chosen window of vocals.wav -> DIR/ref.wav (24 kHz mono, level-normalised) + ref_text (the words inside it)
  say DIR TEXT OUT.wav   Qwen3-TTS-12Hz-1.7B-Base clones the voice in DIR (ref.wav + its transcript) and speaks TEXT
Prints one JSON line on success; exits non-zero with the reason on failure."""
import json, os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
M = C.MODELS
QTTS = f"{M}/Qwen3-TTS-12Hz-1.7B-Base"; WHISPER = f"{M}/whisper-large-v3-turbo"


def sh(cmd, t=900):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=t)
    if r.returncode: raise RuntimeError(f"{os.path.basename(cmd[0])} failed: {(r.stderr or r.stdout)[-400:]}")
    return r.stdout


def separate(src, out):
    """htdemucs through its Python API, reading and writing with soundfile (torchaudio 2.10's own save needs torchcodec, which is not
    built for this aarch64 box)"""
    import numpy as np, soundfile as sf, torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    model = get_model("htdemucs"); model.eval().cuda()
    wav, sr = sf.read(src, dtype="float32", always_2d=True)                 # (samples, channels) at 44.1 kHz = the model's rate
    if sr != model.samplerate: raise RuntimeError(f"expected {model.samplerate} Hz audio, got {sr}")
    x = torch.from_numpy(wav.T.copy()); ref = x.mean(0); x = (x - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad(): stems = apply_model(model, x[None].cuda(), shifts=1, split=True, overlap=0.25, progress=False)[0]
    voc = stems[model.sources.index("vocals")].cpu() * (ref.std() + 1e-8) + ref.mean()
    sf.write(out, voc.numpy().T.astype(np.float32), sr)


def capture(url, d):
    os.makedirs(d, exist_ok=True)
    if not url.startswith(("https://", "http://")): raise RuntimeError("paste a full https:// link")
    ytdlp = os.path.join(os.path.dirname(sys.executable), "yt-dlp")      # the studio's own, current yt-dlp (the system one is from 2024)
    sh([ytdlp if os.path.exists(ytdlp) else "yt-dlp", "--no-playlist", "--max-filesize", "300M", "-f", "b[height<=480][ext=mp4]/bv*[height<=480]+ba/b", "--merge-output-format", "mp4",
        "-o", f"{d}/source.%(ext)s", "--print-to-file", "%(title)s\t%(duration)s\t%(uploader)s", f"{d}/info.txt", url], 600)
    src = next((f"{d}/{f}" for f in os.listdir(d) if f.startswith("source.")), None)
    if not src: raise RuntimeError("yt-dlp produced no file")
    if not src.endswith(".mp4"): sh(["ffmpeg", "-v", "error", "-y", "-i", src, "-c:v", "libx264", "-c:a", "aac", f"{d}/source.mp4"]); os.remove(src)
    title, dur, who = (open(f"{d}/info.txt").read().strip().split("\t") + ["", "", ""])[:3]
    if float(dur or 0) > 600: raise RuntimeError(f"the video is {dur} s long — use a Short or a clip under 10 minutes")
    sh(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/source.mp4", "-vn", "-ac", "2", "-ar", "44100", f"{d}/source.wav"])
    separate(f"{d}/source.wav", f"{d}/vocals.wav")                          # Demucs: the music out, the voice kept
    import torch
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=WHISPER, dtype=torch.float16, device="cuda:0")
    out = asr(f"{d}/vocals.wav", return_timestamps="word", chunk_length_s=30, batch_size=4, generate_kwargs={"task": "transcribe"})
    words = [{"t0": round(c["timestamp"][0] or 0, 2), "t1": round(c["timestamp"][1] or (c["timestamp"][0] or 0) + 0.3, 2), "w": c["text"]}
             for c in out.get("chunks", []) if c.get("text", "").strip()]
    segs, cur = [], None                                                      # phrases: split at pauses > 0.5 s, sentence ends, or 8 s
    for w in words:
        if cur and (w["t0"] - cur["end"] > 0.5 or cur["end"] - cur["start"] > 8 or cur["text"].rstrip().endswith((".", "?", "!"))):
            segs.append(cur); cur = None
        if not cur: cur = {"start": w["t0"], "end": w["t1"], "text": w["w"].strip()}
        else: cur["end"] = w["t1"]; cur["text"] += w["w"] if w["w"].startswith(" ") else " " + w["w"]
    if cur: segs.append(cur)
    meta = {"title": title, "duration": float(dur or 0), "uploader": who, "url": url, "segments": segs, "words": words, "text": out.get("text", "").strip()}
    json.dump(meta, open(f"{d}/transcript.json", "w"), indent=1)
    print(json.dumps({"ok": True, "title": title, "duration": meta["duration"], "segments": len(segs)}))


def cut(d, start, end):
    start, end = float(start), float(end)
    if not 3 <= end - start <= 20: raise RuntimeError("the reference window must be 3 to 20 s")
    meta = json.load(open(f"{d}/transcript.json"))
    ws = meta.get("words") or [{"t0": s["start"], "t1": s["end"], "w": " " + s["text"]} for s in meta["segments"]]
    inside = [w for w in ws if w["t0"] >= start - 0.05 and w["t1"] <= end + 0.05]          # the exact words spoken inside the window
    if inside: start, end = max(start, inside[0]["t0"] - 0.15), min(end, inside[-1]["t1"] + 0.25)   # trim to speech, keep a breath
    ref_text = "".join(w["w"] for w in inside).strip()
    sh(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", f"{d}/vocals.wav", "-ac", "1", "-ar", "24000",
        "-af", "highpass=f=70,loudnorm=I=-20:TP=-2:LRA=7", f"{d}/ref.wav"])
    if not ref_text: raise RuntimeError("no words were transcribed inside that window")
    json.dump({"start": start, "end": end, "ref_text": ref_text}, open(f"{d}/ref.json", "w"), indent=1)
    print(json.dumps({"ok": True, "ref_text": ref_text, "seconds": round(end - start, 2)}))


def say(d, text, out):
    import torch, soundfile as sf
    from qwen_tts import Qwen3TTSModel
    ref = json.load(open(f"{d}/ref.json"))
    m = Qwen3TTSModel.from_pretrained(QTTS, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    wavs, sr = m.generate_voice_clone(text=text, language="Auto", ref_audio=f"{d}/ref.wav", ref_text=ref["ref_text"])
    sf.write(out, wavs[0], sr)
    print(json.dumps({"ok": True, "out": out, "sr": sr, "seconds": round(len(wavs[0]) / sr, 2)}))


if __name__ == "__main__":
    try:
        c = sys.argv[1]
        if c == "capture": capture(sys.argv[2], sys.argv[3])
        elif c == "cut": cut(sys.argv[2], sys.argv[3], sys.argv[4])
        elif c == "say": say(sys.argv[2], sys.argv[3], sys.argv[4])
        else: raise RuntimeError(__doc__)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e)[:500]}"})); sys.exit(1)
