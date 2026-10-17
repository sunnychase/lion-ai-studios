#!/usr/bin/env python3
# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""voice.py — the studio's voice worker (runs in the tts venv: torch 2.10+cu130, qwen-tts, demucs, transformers Whisper).
  capture URL DIR        Short -> DIR/source.mp4 (<=480p, for the director to watch), source.wav, vocals.wav (Demucs: music removed),
                         transcript.json (Whisper large-v3-turbo, timestamped chunks)
  speakers DIR [MAX] [K] who speaks when (Qwen3-TTS speaker encoder + clustering) -> DIR/speakers.json
  cutparts DIR OUT JSON  several [start, end] parts of one speaker joined into one reference
  cut DIR START END [OUT]  the chosen window of vocals.wav -> OUT/ref.wav (24 kHz mono, level-normalised) + ref_text (the words inside it)
  say DIR TEXT OUT.wav   Qwen3-TTS-12Hz-1.7B-Base clones the voice in DIR (ref.wav + its transcript) and speaks TEXT; Whisper checks each take
  say3 DIR TEXT OUT.wav  optional: Fish Audio Drama 3 (TensorFold CUDA port) in the voice of DIR, word-checked — Built with Fish Audio
  voice3 DIR VID         optional: DIR's ref.wav + its transcript -> a Drama 3 voice file (drama-tensorfold's tools/zig/drama_voice.py, the
                         pinned sglang-omni encoder, in DRAMA3_VOICE_ENV_PY); cached by audio sha256 + transcript
  say3s DIR TEXT OUT.wav VID  optional: Fish Audio Drama 3 through the running drama-tensorfold speech server (POST /v1/audio/speech,
                         references inline), word-checked and re-drawn like say/say3 — Built with Fish Audio
Prints one JSON line on success; exits non-zero with the reason on failure."""
import json, os, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
M = C.MODELS
QTTS = f"{M}/Qwen3-TTS-12Hz-1.7B-Base"; WHISPER = f"{M}/whisper-large-v3-turbo"


def sh(cmd, t=900):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=t)
    if r.returncode: raise RuntimeError(f"{os.path.basename(cmd[0])} failed: {(r.stderr or r.stdout)[-400:]}")
    return r.stdout


def separate(src, out, music_out=None):
    """htdemucs through its Python API, reading and writing with soundfile (torchaudio 2.10's own save needs torchcodec, which is not
    built for this aarch64 box)"""
    import numpy as np, soundfile as sf, torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    model = get_model("htdemucs"); model.eval()
    wav, sr = sf.read(src, dtype="float32", always_2d=True)                 # (samples, channels) at 44.1 kHz = the model's rate
    if sr != model.samplerate: raise RuntimeError(f"expected {model.samplerate} Hz audio, got {sr}")
    x = torch.from_numpy(wav.T.copy()); ref = x.mean(0); x = (x - ref.mean()) / (ref.std() + 1e-8)
    def run(dev):
        with torch.no_grad(): return apply_model(model.to(dev), x[None].to(dev), shifts=1, split=True, overlap=0.25, progress=False)[0]
    try: stems = run("cuda")
    except Exception as e:  # short of GPU memory -> the CPU (slower, same result)
        if "out of memory" not in str(e).lower(): raise
        torch.cuda.empty_cache(); stems = run("cpu")
    vi = model.sources.index("vocals"); voc = stems[vi].cpu() * (ref.std() + 1e-8) + ref.mean()
    sf.write(out, voc.numpy().T.astype(np.float32), sr)
    if music_out:  # everything but the voice
        mus = (stems.sum(0) - stems[vi]).cpu() * (ref.std() + 1e-8) + ref.mean()
        sf.write(music_out, mus.numpy().T.astype(np.float32), sr)


def stems(src, d):
    """any audio/video -> d/vocals.wav (speech and singing) + d/music.wav (drums + bass + other), 44.1 kHz stereo"""
    os.makedirs(d, exist_ok=True)
    sh(["ffmpeg", "-v", "error", "-y", "-i", src, "-vn", "-ac", "2", "-ar", "44100", f"{d}/mix.wav"])
    separate(f"{d}/mix.wav", f"{d}/vocals.wav", f"{d}/music.wav"); os.remove(f"{d}/mix.wav")
    print(json.dumps({"ok": True, "vocals": f"{d}/vocals.wav", "music": f"{d}/music.wav"}))


def capture(url, d, maxh=480):
    os.makedirs(d, exist_ok=True)
    if not url.startswith(("https://", "http://")): raise RuntimeError("paste a full https:// link")
    ytdlp = os.path.join(os.path.dirname(sys.executable), "yt-dlp")      # the studio's own, current yt-dlp (the system one is from 2024)
    deno = C.DENO_BIN or shutil.which("deno") or ""                       # YouTube now needs a JS runtime (else 403 on some videos); yt-dlp-ejs
    js = ["--js-runtimes", f"deno:{deno}"] if deno and os.path.exists(deno) else []   # supplies the solver scripts
    cmd = [ytdlp if os.path.exists(ytdlp) else "yt-dlp", *js, "--no-playlist", "--max-filesize", "300M", "-f", f"b[height<={maxh}][ext=mp4]/bv*[height<={maxh}]+ba/b", "--merge-output-format", "mp4",
           "--retries", "5", "--fragment-retries", "5", "-o", f"{d}/source.%(ext)s", "--print-to-file", "%(title)s\t%(duration)s\t%(uploader)s", f"{d}/info.txt", url]
    for tries in range(3):                                                 # YouTube 403s come and go — fresh URLs each try
        try: sh(cmd, 600); break
        except RuntimeError as e:
            if "403" not in str(e) or tries == 2: raise
            for f in os.listdir(d):
                if f.startswith("source.") or f.endswith(".part"): os.remove(os.path.join(d, f))
            time.sleep(6 * (tries + 1))
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


def speakers(d, max_spk=6, force_k=0):
    """who speaks when. Whisper's WORD timings (sorted, overlaps from its 30 s chunks dropped) are cut into
    1.2-4.5 s windows (cut at 0.6 s pauses); each window is fingerprinted with the clone model's OWN speaker encoder (Qwen3-TTS extract_speaker_embedding, 24 kHz);
    the common "speech" component is removed and the windows grouped (agglomerative, cosine). The number of speakers = the most that each hold
    >= 6 % / >= 12 s of the speech, sit >= 0.6 apart and speak in turns (consecutive windows keep their label clearly more often than random labels would: Σp² + ¼ of the
    rest — rapid dialogue alternates, a single voice split in two looks random), else 1.
    Each speaker's reference = their longest solo run if it is >= 6 s, else up to 4 of their own runs joined (>= 5 s, <= 15 s): rapid
    dialogue (a skit) rarely gives anyone 6 s alone. -> DIR/speakers.json"""
    import numpy as np, soundfile as sf, torch
    from qwen_tts import Qwen3TTSModel
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    meta = json.load(open(f"{d}/transcript.json"))
    words, last = [], -1.0
    for w in sorted(meta.get("words") or [], key=lambda w: w["t0"]):
        if w["t0"] < last - 0.3 or not w["w"].strip(): continue              # Whisper chunk overlap: the same words twice
        words.append(w); last = max(last, w["t1"])
    if not words: raise RuntimeError("no transcribed words")
    wins, cur = [], []
    for w in words:
        if cur and (w["t0"] - cur[-1]["t1"] > 0.6 or w["t1"] - cur[0]["t0"] > 4.5):       # 4.5 s / 0.6 s / 1.2 s windows
            wins.append(cur); cur = []
        cur.append(w)
    if cur: wins.append(cur)
    wins = [{"t0": c[0]["t0"], "t1": c[-1]["t1"], "text": "".join(x["w"] if x["w"].startswith(" ") else " " + x["w"] for x in c).strip()} for c in wins]
    wins = [x for x in wins if x["t1"] - x["t0"] >= 1.2]
    sh(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/vocals.wav", "-ac", "1", "-ar", "24000", f"{d}/v24.wav"])
    wav, sr = sf.read(f"{d}/v24.wav", dtype="float32"); os.remove(f"{d}/v24.wav")
    enc = Qwen3TTSModel.from_pretrained(QTTS, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa").model
    keep, emb = [], []
    for x in wins:
        a = wav[max(0, int((x["t0"] - 0.05) * sr)):int((x["t1"] + 0.05) * sr)]
        if len(a) < int(1.2 * sr) or float(np.sqrt((a ** 2).mean())) < 1e-3: continue
        e = enc.extract_speaker_embedding(np.ascontiguousarray(a), 24000).float().cpu().numpy(); emb.append(e / (np.linalg.norm(e) + 1e-9)); keep.append(x)
    if len(keep) < 2: raise RuntimeError("not enough speech to tell voices apart")
    E = np.stack(emb); C = E - E.mean(0); C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    dur = np.array([x["t1"] - x["t0"] for x in keep]); best_k, best_s, labels, why = 1, 0.0, np.zeros(len(keep), int), {}
    force_k = int(force_k or 0)
    for k in range(2, min(max(max_spk, force_k), len(keep) - 1) + 1):
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(C)
        share = np.bincount(lab, minlength=k) / len(lab); secs = np.array([dur[lab == l].sum() for l in range(k)])
        cents = [C[lab == l].mean(0) for l in range(k)]; cents = [c / (np.linalg.norm(c) + 1e-9) for c in cents]
        dist = min(1 - float(a @ b) for i, a in enumerate(cents) for b in cents[i + 1:])
        coh = float(np.mean(lab[1:] == lab[:-1])); rnd = float((share ** 2).sum()); need = rnd + 0.25 * (1 - rnd)   # turns vs random labels
        why[k] = {"min_share": round(float(share.min()), 3), "min_seconds": round(float(secs.min()), 1), "centroid_dist": round(dist, 3), "turns": round(coh, 3), "turns_needed": round(need, 3)}
        if (k == force_k) or (not force_k and share.min() >= 0.06 and secs.min() >= 12 and dist >= 0.6 and coh >= need):
            best_k, labels = k, lab; best_s = float(silhouette_score(C, lab, metric="cosine"))
    if force_k == 1: best_k, labels, best_s = 1, np.zeros(len(keep), int), 0.0          # the user said: one person
    out = []
    for l in range(best_k):
        idx = [i for i in range(len(keep)) if labels[i] == l]
        if not idx: continue
        runs, cur = [], [idx[0]]
        for i in idx[1:]:
            if i == cur[-1] + 1 and keep[i]["t0"] - keep[cur[-1]]["t1"] < 1.0: cur.append(i)
            else: runs.append(cur); cur = [i]
        runs.append(cur)
        span = lambda r: keep[r[-1]]["t1"] - keep[r[0]]["t0"]
        runs.sort(key=span, reverse=True)
        if span(runs[0]) >= 6: parts = [runs[0]]
        else:
            parts, tot = [], 0.0
            for r in runs:
                if len(parts) == 4 or tot >= 10: break
                if span(r) < 1.0: continue
                parts.append(r); tot += span(r)
            parts.sort(key=lambda r: keep[r[0]]["t0"])
        segs, tot = [], 0.0
        for r in parts:
            a, b = keep[r[0]]["t0"], keep[r[-1]]["t1"]; b = min(b, a + 15 - tot)
            if b - a < 0.8: break
            segs.append([round(a, 2), round(b, 2)]); tot += b - a
        c = E[idx].mean(0)
        out.append({"n": 0, "seconds": round(float(dur[idx].sum()), 1), "windows": len(idx), "first": round(keep[idx[0]]["t0"], 1),
                    "best": {"parts": segs, "start": segs[0][0] if segs else 0, "end": segs[-1][1] if segs else 0, "seconds": round(tot, 2),
                             "text": " ".join(keep[i]["text"] for r in parts for i in r)[:400]},
                    "enough": tot >= 5, "joined": len(segs) > 1, "sample": keep[idx[0]]["text"][:120],
                    "centroid": [round(float(v), 5) for v in c / (np.linalg.norm(c) + 1e-9)]})
    out.sort(key=lambda s: -s["seconds"])
    for n, s in enumerate(out, 1): s["n"] = n
    res = {"k": len(out), "silhouette": round(best_s, 3), "windows": len(keep), "forced": force_k or None, "tested": why, "speakers": out}
    json.dump(res, open(f"{d}/speakers.json", "w"), indent=1)
    print(json.dumps({"ok": True, "k": len(out), "silhouette": res["silhouette"], "seconds": [s["seconds"] for s in out]}))


def cutparts(d, outdir, parts_json):
    """a reference made of several of one speaker's own turns, in time order, 0.2 s of silence between, -> ref.wav +
    ref.json (ref_text = the words inside each part, joined) in outdir"""
    parts = [(float(a), float(b)) for a, b in json.loads(parts_json)]; os.makedirs(outdir, exist_ok=True)
    meta = json.load(open(f"{d}/transcript.json"))
    ws = meta.get("words") or []
    texts, files = [], []
    for k, (a, b) in enumerate(parts):
        inside = [w for w in ws if w["t0"] >= a - 0.05 and w["t1"] <= b + 0.05]
        if inside: a, b = max(a, inside[0]["t0"] - 0.1), min(b, inside[-1]["t1"] + 0.15)
        texts.append("".join(w["w"] for w in inside).strip())
        f = f"{outdir}/part{k}.wav"; files.append(f)
        sh(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.2f}", "-to", f"{b:.2f}", "-i", f"{d}/vocals.wav", "-ac", "1", "-ar", "24000", "-af", "highpass=f=70", f])
    gap = f"{outdir}/gap.wav"; sh(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "0.2", gap])
    seq = [x for f in files for x in (f, gap)][:-1]
    sh(["ffmpeg", "-v", "error", "-y", *sum([["-i", f] for f in seq], []), "-filter_complex", f"concat=n={len(seq)}:v=0:a=1,loudnorm=I=-20:TP=-2:LRA=7",
        "-ar", "24000", "-ac", "1", f"{outdir}/ref.wav"])
    for f in files + [gap]: os.remove(f)
    ref_text = heard(f"{outdir}/ref.wav") or " ".join(t for t in texts if t)   # the joined clip's own words (see heard)
    if not ref_text: raise RuntimeError("no words were transcribed inside those parts")
    secs = sum(b - a for a, b in parts)
    if not 3 <= secs <= 20: raise RuntimeError(f"the joined reference is {secs:.1f} s; it must be 3 to 20 s")
    json.dump({"parts": parts, "start": parts[0][0], "end": parts[-1][1], "ref_text": ref_text}, open(f"{outdir}/ref.json", "w"), indent=1)
    print(json.dumps({"ok": True, "ref_text": ref_text, "seconds": round(secs, 2), "parts": len(parts)}))


def heard(path):
    """Whisper on one short clip -> its words. A reference clip's ref_text must match ITS audio exactly, or Qwen3-TTS
    reads the unmatched words aloud before/after the line (slices of a whole-video transcript leaked into previews)."""
    import torch, soundfile as sf, numpy as np
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=WHISPER, dtype=torch.float16, device="cuda:0")
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1: w = w.mean(1)
    x = w if sr == 16000 else np.interp(np.arange(0, len(w), sr / 16000), np.arange(len(w)), w).astype(np.float32)
    return (asr({"raw": x, "sampling_rate": 16000}, generate_kwargs={"task": "transcribe"}).get("text") or "").strip()


def cut(d, start, end, outdir=None):
    """the reference window -> ref.wav + ref.json in outdir (default d; one voice dir per speaker)"""
    start, end = float(start), float(end); o = outdir or d; os.makedirs(o, exist_ok=True)
    if not 3 <= end - start <= 20: raise RuntimeError("the reference window must be 3 to 20 s")
    meta = json.load(open(f"{d}/transcript.json"))
    ws = meta.get("words") or [{"t0": s["start"], "t1": s["end"], "w": " " + s["text"]} for s in meta["segments"]]
    inside = [w for w in ws if w["t0"] >= start - 0.05 and w["t1"] <= end + 0.05]          # the exact words spoken inside the window
    if inside: start, end = max(start, inside[0]["t0"] - 0.15), min(end, inside[-1]["t1"] + 0.25)   # trim to speech, keep a breath
    ref_text = "".join(w["w"] for w in inside).strip()
    sh(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", f"{d}/vocals.wav", "-ac", "1", "-ar", "24000",
        "-af", "highpass=f=70,loudnorm=I=-20:TP=-2:LRA=7", f"{o}/ref.wav"])
    if not ref_text: raise RuntimeError("no words were transcribed inside that window")
    ref_text = heard(f"{o}/ref.wav") or ref_text                          # the clip's own words
    json.dump({"start": start, "end": end, "ref_text": ref_text}, open(f"{o}/ref.json", "w"), indent=1)
    print(json.dumps({"ok": True, "ref_text": ref_text, "seconds": round(end - start, 2)}))


def say(d, text, out, takes=3):
    """speak TEXT in the voice of DIR. Whisper listens to each take and wordcheck.check_text decides; a take that drops
    words, runs 3+ missing in a row or loses its ending is re-drawn (new seed), up to TAKES; the best take is kept and its check reported."""
    import torch, soundfile as sf, numpy as np
    from qwen_tts import Qwen3TTSModel
    from transformers import pipeline
    import wordcheck as WC
    ref = json.load(open(f"{d}/ref.json"))
    m = Qwen3TTSModel.from_pretrained(QTTS, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    asr = pipeline("automatic-speech-recognition", model=WHISPER, dtype=torch.float16, device="cuda:0")
    best = None; n = max(1, int(takes))
    for k in range(n + 1):                                                   # the extra pass = x-vector only (cannot read the reference)
        xv = k == n
        if xv and best and best[3]["action"] != "redraw": break
        torch.manual_seed(1009 + 7919 * k)
        wavs, sr = m.generate_voice_clone(text=text, language="Auto", ref_audio=f"{d}/ref.wav", ref_text=None if xv else ref["ref_text"], x_vector_only_mode=xv)
        w = np.asarray(wavs[0], dtype=np.float32)
        x = w if sr == 16000 else np.interp(np.arange(0, len(w), sr / 16000), np.arange(len(w)), w).astype(np.float32)
        said = asr({"raw": x, "sampling_rate": 16000}, generate_kwargs={"task": "transcribe"}).get("text", "")
        chk = WC.check_text(text, said); chk["take"] = k + 1; chk["x_vector_only"] = xv
        lead = _leading_extra(text, said); chk["leading_extra"] = lead           # reference words read BEFORE the line
        if lead > 1 or len(chk.get("tail_words") or []) > 2:
            chk["action"], chk["reason"] = "redraw", f"{lead} extra word(s) before the line, {len(chk.get('tail_words') or [])} after (reference text leaked)"
        score = (chk["action"] != "redraw", not xv, chk["coverage"], -chk["longest_missing_run"])
        if best is None or score > best[0]: best = (score, w, sr, chk)
        if chk["action"] != "redraw": break
    _, w, sr, chk = best
    sf.write(out, w, sr)
    print(json.dumps({"ok": True, "out": out, "sr": sr, "seconds": round(len(w) / sr, 2), "takes": chk["take"],
                      "wordcheck": {k: chk.get(k) for k in ("action", "words", "matched", "coverage", "missing", "reason", "heard", "leading_extra", "x_vector_only")}}))


import drama3 as DR3  # noqa: E402  (every Drama 3 path, from config.py)
D3 = DR3.OLD_DIR; D3_MODEL = DR3.CKPT


def drama3_ready():
    return DR3.old_built() and DR3.local()


def say3(d, text, out, takes=3):
    """optional: speak TEXT in the voice of DIR with Fish Audio Drama 3 (TensorFold's native CUDA port, drama3-generate; Built with
    Fish Audio; Fish Audio Community License v1.0 — clone only with consent). One process per take (~18 s load, ~42 GB peak: the studio only
    runs it when the GPU is free). Same Whisper word check and re-draws as `say`; the reference is converted to 44.1 kHz mono WAV first."""
    import torch, numpy as np, soundfile as sf
    from transformers import pipeline
    import wordcheck as WC
    if not drama3_ready(): raise RuntimeError("Fish Audio Drama 3 is not installed (see docs/DRAMA3.md; DRAMA3_DIR / DRAMA3_MODEL in studio.env)")
    ref = json.load(open(f"{d}/ref.json")); r44 = f"{os.path.dirname(out) or '.'}/.ref44.wav"
    sh(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/ref.wav", "-ac", "1", "-ar", "44100", r44])
    asr = None; best = None
    for k in range(max(1, int(takes))):
        tmp = f"{out}.take{k + 1}.wav"
        sh([f"{D3}/bin/drama3-generate", "--model", D3_MODEL, "--codec", f"{D3}/codec-v1.safetensors", "--text", text, "--wav", tmp,
            "--ref-audio", r44, "--ref-text", ref["ref_text"], "--seed", str(1009 + 7919 * k)], 900)
        w, sr = sf.read(tmp, dtype="float32"); w = w.mean(1) if w.ndim > 1 else w
        if asr is None: asr = pipeline("automatic-speech-recognition", model=WHISPER, dtype=torch.float16, device="cuda:0")
        x = np.interp(np.arange(0, len(w), sr / 16000), np.arange(len(w)), w).astype(np.float32)
        said = asr({"raw": x, "sampling_rate": 16000}, generate_kwargs={"task": "transcribe"}).get("text", "")
        chk = WC.check_text(text, said); chk["take"] = k + 1; lead = _leading_extra(text, said); chk["leading_extra"] = lead
        if lead > 1 or len(chk.get("tail_words") or []) > 2: chk["action"], chk["reason"] = "redraw", "reference text leaked"
        score = (chk["action"] != "redraw", chk["coverage"], -chk["longest_missing_run"])
        if best is None or score > best[0]: best = (score, tmp, chk)
        if chk["action"] != "redraw": break
    _, keep, chk = best; os.replace(keep, out)
    for f in os.listdir(os.path.dirname(out) or "."):
        if f.startswith(os.path.basename(out) + ".take") or f == ".ref44.wav": os.remove(os.path.join(os.path.dirname(out) or ".", f))
    print(json.dumps({"ok": True, "out": out, "engine": "drama3", "takes": chk["take"], "credit": "Built with Fish Audio",
                      "wordcheck": {k: chk.get(k) for k in ("action", "words", "matched", "coverage", "missing", "reason", "heard", "leading_extra")}}))


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def _voice_ok(v):
    """a drama3-voice/1 file the server would load: [10, frames] codes in [0, 4096) matching codes_sha256 (little-endian int32, codebook-major)"""
    import hashlib
    rows = v.get("vq_codes")
    if v.get("format") != "drama3-voice/1" or not isinstance(v.get("text"), str) or not v["text"].strip(): return False
    if not isinstance(rows, list) or len(rows) != 10 or not rows[0] or any(not isinstance(r, list) or len(r) != len(rows[0]) for r in rows): return False
    if any(not isinstance(c, int) or c < 0 or c >= 4096 for r in rows for c in r): return False
    return hashlib.sha256(b"".join(c.to_bytes(4, "little", signed=True) for r in rows for c in r)).hexdigest() == v.get("codes_sha256")


def voice3(d, vid):
    """optional: the Drama 3 voice file for DIR (ref.wav + its ref.json transcript), made once with drama-tensorfold's drama_voice.py (the
    pinned sglang-omni reference encoder, bf16 on CUDA, ~2.6 GB / 10-20 s) and reused while ref.wav and the transcript are unchanged. Written
    under a hidden temp name and renamed, so a server reading its voices directory never sees half a file."""
    if not DR3.voice_tool_ready(): raise RuntimeError("the Drama 3 voice tool is not installed (drama-tensorfold build / DRAMA3_VOICE_ENV_PY / DRAMA3_SGLANG_OMNI_DIR; see docs/DRAMA3.md)")
    if not DR3.local(): raise RuntimeError("the Drama 3 weights are not on this machine (DRAMA3_MODEL)")
    ref = json.load(open(f"{d}/ref.json")); text = (ref.get("ref_text") or "").strip(); wav = f"{d}/ref.wav"
    if not text: raise RuntimeError("this voice has no reference transcript")
    os.makedirs(DR3.VOICES, exist_ok=True)
    name = DR3.voice_name(vid); out = f"{DR3.VOICES}/{name}.json"; asha = _sha256_file(wav)
    try:
        v = json.load(open(out))
        if v.get("audio_sha256") == asha and v.get("text") == text and _voice_ok(v):
            print(json.dumps({"ok": True, "voice_file": out, "cached": True, "frames": v.get("frames")})); return
    except (OSError, ValueError): pass
    tmp_txt, tmp_out = f"{DR3.VOICES}/.{name}.txt.tmp", f"{DR3.VOICES}/.{name}.json.tmp"
    open(tmp_txt, "w", encoding="utf-8").write(text + "\n")
    try:
        sh([DR3.VOICE_ENV_PY, DR3.VOICE_TOOL, "--reference", DR3.SGL_REF, "--checkpoint", DR3.CKPT, "--audio", wav, "--text", tmp_txt,
            "--out", tmp_out, "--device", "cuda"], 600)
        v = json.load(open(tmp_out))
        if not _voice_ok(v) or v.get("audio_sha256") != asha: raise RuntimeError("the voice tool wrote a file that does not check out")
        os.replace(tmp_out, out)
    finally:
        for f in (tmp_txt, tmp_out):
            if os.path.exists(f): os.remove(f)
    print(json.dumps({"ok": True, "voice_file": out, "cached": False, "frames": v.get("frames"), "encoder": v.get("encoder")}))


def _chunks(text, limit=1500):
    """split at sentence ends into pieces <= limit chars (the server takes <= 4096; --max-frames 4096 ~ 190 s ~ 2700 chars)"""
    import re
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?…])\s+", text.strip()):
        while len(s) > limit:                                                # one huge sentence: cut at the last comma/space before the limit
            k = max(s.rfind(", ", 0, limit), s.rfind(" ", 0, limit)); k = k if k > limit // 3 else limit
            if cur: out.append(cur); cur = ""
            out.append(s[:k + 1].strip()); s = s[k + 1:].strip()
        if cur and len(cur) + 1 + len(s) > limit: out.append(cur); cur = s
        else: cur = f"{cur} {s}".strip()
    if cur: out.append(cur)
    return [c for c in out if c]


def _d3_post(body, t=900):
    """POST /v1/audio/speech -> (wav bytes, X-Finish-Reason). 503 = the single slot is busy: wait and retry; 400 = refused (its message)."""
    import urllib.request, urllib.error
    data = json.dumps(body, ensure_ascii=False).encode()
    for _ in range(60):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{DR3.URL}/v1/audio/speech", data=data, headers={"Content-Type": "application/json"}), timeout=t) as r:
                return r.read(), (r.headers.get("X-Finish-Reason") or "stop")
        except urllib.error.HTTPError as e:
            if e.code == 503: time.sleep(3); continue
            raise RuntimeError(f"Drama 3 server HTTP {e.code}: {(e.read() or b'').decode('utf-8', 'replace')[:300]}")
    raise RuntimeError("the Drama 3 server stayed busy (503) for 3 minutes")


def say3s(d, text, out, vid, takes=3):
    """optional: speak TEXT in the voice of DIR through the drama-tensorfold speech server (Built with Fish Audio). The voice file's transcript
    + vq_codes ride INLINE as `references` (no server restart per voice). Same Whisper word check and re-draws as say/say3. The server seeds
    each synthesis from its prompt rows (no seed field), so a re-draw trims 3 more frames (~0.14 s, the breath after the last word) off the
    end of the reference codes — a different prompt, hence a new draw. A reply that ends in `length` or `silence` is re-drawn."""
    import io, torch, numpy as np, soundfile as sf
    from transformers import pipeline
    import wordcheck as WC
    v = json.load(open(f"{DR3.VOICES}/{DR3.voice_name(vid)}.json"))
    if not _voice_ok(v): raise RuntimeError("the Drama 3 voice file does not check out — make it again (voice3)")
    codes, vtext = v["vq_codes"], v["text"]
    asr = None; parts, checks, fins, sr0 = [], [], [], None
    for chunk in _chunks(text):
        best = None
        for k in range(max(1, int(takes))):
            trim = 3 * k if len(codes[0]) > 3 * k + 40 else 0
            ref = [r[:len(r) - trim] for r in codes] if trim else codes
            wavb, fin = _d3_post({"model": DR3.MODEL, "input": chunk, "references": [{"text": vtext, "vq_codes": ref}], "response_format": "wav"})
            w, sr = sf.read(io.BytesIO(wavb), dtype="float32"); w = w.mean(1) if w.ndim > 1 else w
            if asr is None: asr = pipeline("automatic-speech-recognition", model=WHISPER, dtype=torch.float16, device="cuda:0")
            x = np.interp(np.arange(0, len(w), sr / 16000), np.arange(len(w)), w).astype(np.float32)
            said = asr({"raw": x, "sampling_rate": 16000}, generate_kwargs={"task": "transcribe"}).get("text", "")
            chk = WC.check_text(chunk, said); chk["take"] = k + 1; chk["finish"] = fin; lead = _leading_extra(chunk, said); chk["leading_extra"] = lead
            if lead > 1 or len(chk.get("tail_words") or []) > 2: chk["action"], chk["reason"] = "redraw", "reference text leaked"
            if fin != "stop": chk["action"], chk["reason"] = "redraw", f"the reply ended in '{fin}' (clipped or stalled)"
            score = (chk["action"] != "redraw", fin == "stop", chk["coverage"], -chk["longest_missing_run"])
            if best is None or score > best[0]: best = (score, w, sr, chk)
            if chk["action"] != "redraw": break
        _, w, sr, chk = best
        if sr0 and sr != sr0: raise RuntimeError("the server changed sample rate between pieces")
        sr0 = sr; parts.append(w); checks.append(chk); fins.append(chk["finish"])
    gap = np.zeros(int(0.35 * sr0), dtype=np.float32)
    w = np.concatenate([y for i, p in enumerate(parts) for y in ((gap, p) if i else (p,))])
    sf.write(out, w, sr0)
    worst = min(checks, key=lambda c: (c["action"] != "redraw", c["coverage"]))
    print(json.dumps({"ok": True, "out": out, "engine": "drama3-server", "path": "say3s", "sr": sr0, "seconds": round(len(w) / sr0, 2),
                      "pieces": len(parts), "finish": fins, "takes": [c["take"] for c in checks], "credit": "Built with Fish Audio",
                      "wordcheck": {k: worst.get(k) for k in ("action", "words", "matched", "coverage", "missing", "reason", "heard", "leading_extra", "finish")}}))


def _leading_extra(text, said):
    """words heard before the first word of the text (normalised as wordcheck does)"""
    import wordcheck as WC, difflib
    ref, hyp = WC.words_of(text), WC.words_of(said)
    if not ref or not hyp: return 0
    for j, h in enumerate(hyp[:12]):
        if difflib.SequenceMatcher(None, h, ref[0]).ratio() >= 0.75 or (len(ref) > 1 and j + 1 < len(hyp) and hyp[j + 1] == ref[1]): return j
    return 0


if __name__ == "__main__":
    try:
        c = sys.argv[1]
        if c == "capture": capture(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 480)   # optional max height (Faithful Recreate asks for 1080)
        elif c == "cut": cut(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
        elif c == "speakers": speakers(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6, int(sys.argv[4]) if len(sys.argv) > 4 else 0)
        elif c == "cutparts": cutparts(sys.argv[2], sys.argv[3], sys.argv[4])
        elif c == "say": say(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 3)
        elif c == "say3": say3(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 3)
        elif c == "voice3": voice3(sys.argv[2], sys.argv[3])
        elif c == "say3s": say3s(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]) if len(sys.argv) > 6 else 3)
        elif c == "stems": stems(sys.argv[2], sys.argv[3])
        else: raise RuntimeError(__doc__)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e)[:500]}"})); sys.exit(1)
