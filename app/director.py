# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""director.py — the studio's director: a vision-language model behind an OpenAI-compatible chat API (config.DIRECTOR_URL).
It WRITES (one idea -> picture, action, sound, one beat per segment) and REVIEWS (watches each rendered segment and scores it against
its prompt). It must accept video input ("video_url" content parts), e.g. vLLM serving a Qwen3-VL-class model. If config.WRITER_URL
answers, writing uses that model instead (reviews always use the director)."""
import base64, json, os, re, subprocess, tempfile, time, urllib.request
import config as C

DIRECTOR = C.DIRECTOR_URL; FLASH = C.WRITER_URL


def _up(base):
    """True when an OpenAI-compatible server answers at base (GET /v1/models)"""
    if not base: return False
    try: urllib.request.urlopen(base + "/v1/models", timeout=3).read(); return True
    except Exception: return False


def ensure(timeout=420):
    """start the director's service (config.DIRECTOR_SERVICE) if it is not answering; returns True when it answers"""
    if _up(DIRECTOR): return True
    if not C.DIRECTOR_SERVICE: return False
    subprocess.run(["systemctl", "--user", "start", C.DIRECTOR_SERVICE], capture_output=True, timeout=30)
    t = time.time()
    while time.time() - t < timeout:
        if _up(DIRECTOR): return True
        if subprocess.run(["systemctl", "--user", "is-active", "-q", C.DIRECTOR_SERVICE]).returncode: return False
        time.sleep(5)
    return False


def stop():
    if C.DIRECTOR_SERVICE: subprocess.run(["systemctl", "--user", "stop", C.DIRECTOR_SERVICE], capture_output=True, timeout=90)


def status():
    return {"director": _up(DIRECTOR), "flash": _up(FLASH), "label": C.DIRECTOR_LABEL, "writer_label": C.WRITER_LABEL}


def _chat(base, model, messages, max_tokens=900, temperature=0.6):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                                                headers={"Content-Type": "application/json"}), timeout=900))
    return (r["choices"][0]["message"].get("content") or "").strip()


def _json(txt):
    """the director's reply -> dict. Strict JSON first; then the slips a small model makes (a ```json fence, missing braces, missing
    commas between "key": "value" lines); last, the expected fields pulled out one by one. A reply with no fields still raises."""
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try: return json.loads(m.group(0))
        except ValueError: pass
    body = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", txt.strip())
    if not body.startswith("{"):
        try: return json.loads("{" + re.sub(r'"\s*\n\s*"', '",\n"', body).rstrip(", \n") + "}")
        except ValueError: pass
    d = {k: json.loads('"' + v + '"') for k, v in re.findall(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"', txt)}
    d.update({k: json.loads(v) for k, v in re.findall(r'"(\w+)"\s*:\s*(-?\d+(?:\.\d+)?|true|false)\b', txt)})
    if d: return d
    raise ValueError("the director did not return JSON")


def _ask(base, model, messages, **kw):
    """_chat + _json, with one retry when the reply cannot be read as JSON"""
    try: return _json(_chat(base, model, messages, **kw))
    except ValueError:
        return _json(_chat(base, model, messages + [{"role": "user", "content": "Reply again with only one valid JSON object, in braces, nothing else."}], **kw))


WRITE_SYS = ("You are a film director writing prompts for MiniMax-H3, a text-to-video model that makes picture AND sound together. "
             "Write concrete, filmable prose: subject, setting, light, lens and camera move; dialogue in quotes; sounds named plainly. "
             "Reply with JSON only.")


def write(idea, seconds, n_segments, mode="video", n_images=0):
    """one idea -> {picture, action, sound, beats[]}; uses the writer when it answers, else the director"""
    if mode == "keyframes": shape = f"{n_images} keyframe images joined by {max(0, n_images - 1)} clips; one beat per clip saying what happens between images"
    else: shape = f"{seconds:g} s in {n_segments} segment(s) of up to 15 s; one beat per segment" if n_segments > 1 else f"{seconds:g} s, one shot"
    nb = max(1, n_segments if mode != "keyframes" else n_images - 1); beats = ", ".join(['"..."'] * nb)
    msg = [{"role": "system", "content": WRITE_SYS},
           {"role": "user", "content": f"Idea: {idea}\nFormat: {shape}.\nReturn {{\"picture\": \"40-90 words: who/what is in frame, where, the light, the lens\", "
                                       f"\"action\": \"what happens and how the camera moves\", \"sound\": \"music, ambience, any spoken or sung lines\", "
                                       f"\"beats\": [{beats}]}}"}]
    if _up(FLASH): base, model, who = FLASH, C.WRITER_MODEL, C.WRITER_LABEL
    else:
        if not ensure(): raise RuntimeError(f"the director did not answer at {DIRECTOR} — is the server running? (see DIRECTOR_URL in studio.env)")
        base, model, who = DIRECTOR, C.DIRECTOR_MODEL, C.DIRECTOR_LABEL
    t = time.time(); d = _ask(base, model, msg); d["by"] = who; d["secs"] = round(time.time() - t, 1)
    d["beats"] = [str(b) for b in (d.get("beats") or [])]
    return d


def review(video_path, prompt):
    """the director watches a rendered segment (downscaled, picture only — it does not hear) -> {score, matches_prompt, issues[], fix}"""
    if not ensure(): raise RuntimeError("the director is not running")
    tmp = tempfile.mktemp(suffix=".mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video_path, "-an", "-vf", "scale=640:-2,fps=8", "-c:v", "libx264", "-crf", "30", tmp], capture_output=True, timeout=120)
    try: v = base64.b64encode(open(tmp, "rb").read()).decode()
    finally:
        try: os.remove(tmp)
        except OSError: pass
    msg = [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + v}},
            {"type": "text", "text": "You review AI-generated clips for a film. The clip was made from this prompt:\n\n" + prompt[:3000] +
             "\n\nJudge only what you can see. Reply JSON only: {\"score\": 1-10, \"matches_prompt\": true|false, \"issues\": [\"...\"], "
             "\"fix\": \"one sentence a re-take prompt should add\"}"}]}]
    t = time.time(); d = _ask(DIRECTOR, C.DIRECTOR_MODEL, msg, max_tokens=400, temperature=0.2); d["secs"] = round(time.time() - t, 1)
    try: d["score"] = int(d.get("score") or 0)
    except (TypeError, ValueError): d["score"] = 0
    return d


def pick_voice(video_path, segments, who, duration):
    """the director WATCHES the Short with its timestamped transcript and picks one clean 6-15 s stretch of the wanted speaker.
    -> {start, end, speaker, reason, by}. Falls back to the longest transcribed run when it cannot decide."""
    tx = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments)[:6000]
    tmp = tempfile.mktemp(suffix=".mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-t", "120", "-i", video_path, "-an", "-vf", "scale=480:-2,fps=4", "-c:v", "libx264", "-crf", "32", tmp],
                   capture_output=True, timeout=180)
    try: v = base64.b64encode(open(tmp, "rb").read()).decode()
    finally:
        try: os.remove(tmp)
        except OSError: pass
    target = who.strip() or "the main speaker"
    msg = [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + v}},
            {"type": "text", "text": f"This is a short video with its speech transcribed (seconds):\n{tx}\n\nFind ONE continuous stretch of 6 to 15 seconds "
             f"where {target} is speaking clearly on their own (nobody talking over them, few long pauses). Use the video to tell who is talking. "
             "Reply JSON only: {\"start\": seconds, \"end\": seconds, \"speaker\": \"short description of who is speaking\", \"reason\": \"one sentence\"}"}]}]
    if _up(FLASH): base, model, by = FLASH, C.WRITER_MODEL, C.WRITER_LABEL
    else:
        if not ensure(): raise RuntimeError("the director is not running")
        base, model, by = DIRECTOR, C.DIRECTOR_MODEL, C.DIRECTOR_LABEL
    try:
        d = _ask(base, model, msg, max_tokens=300, temperature=0.2); s, e = float(d["start"]), float(d["end"])
        if e - s < 3: e = s + 6
        if e - s > 15: e = s + 15
        s, e = max(0.0, s), min(duration or e, e)
        if e - s < 3: raise ValueError("window too short")
        return {"start": round(s, 2), "end": round(e, 2), "speaker": d.get("speaker", ""), "reason": d.get("reason", ""), "by": by}
    except Exception as ex:
        best, cur = (0, 0), None                                             # fallback: the longest run of speech with small gaps
        for sg in segments:
            if cur and sg["start"] - cur[1] < 0.8: cur = (cur[0], sg["end"])
            else: cur = (sg["start"], sg["end"])
            if cur[1] - cur[0] > best[1] - best[0]: best = cur
        s, e = best[0], min(best[1], best[0] + 15)
        if e - s < 3: raise RuntimeError(f"no clear stretch of speech found ({type(ex).__name__}: {ex})")
        return {"start": round(s, 2), "end": round(e, 2), "speaker": "", "reason": f"longest run of speech (director could not decide: {str(ex)[:80]})", "by": "fallback"}


# music briefs ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
MUSIC_SYS = ("You are a film composer writing a brief for a music model. Be concrete: genre, mood arc, tempo, key, instruments, production. "
             "Reply with JSON only.")


def music_brief(idea, seconds, engine="fast"):
    """an idea or a film prompt -> {style, bpm, key, by}. fast: 25-50 words of comma-separated tags for ACE-Step 1.5;
    best: MiniMax Music 3's structured brief (Global Metadata / Vocal Details / Arrangement). Instrumental unless the idea asks for singing."""
    if engine == "best":
        form = ('"style": "Global Metadata: <genre>. <BPM> BPM, <key>. <mood arc over the piece>. <use>. <production sound>.\\n\\n'
                'Vocal Details: No vocals.\\n\\nArrangement: <instruments; intro / build / peak / outro in a few sentences>"')
    else:
        form = '"style": "25-50 words of comma-separated tags: genre, mood, instruments, production, tempo feel"'
    msg = [{"role": "system", "content": MUSIC_SYS},
           {"role": "user", "content": f"Music for: {idea}\nLength: {float(seconds):g} s. Instrumental, no vocals, unless the idea explicitly asks for singing.\nReturn {{{form}, \"bpm\": <40-200>, \"key\": \"<e.g. D minor>\"}}"}]
    if _up(FLASH): base, model, who = FLASH, C.WRITER_MODEL, C.WRITER_LABEL
    else:
        if not ensure(): raise RuntimeError("the director did not start")
        base, model, who = DIRECTOR, C.DIRECTOR_MODEL, C.DIRECTOR_LABEL
    d = _ask(base, model, msg, max_tokens=700); d["by"] = who
    try: d["bpm"] = max(40, min(200, int(d.get("bpm") or 100)))
    except (TypeError, ValueError): d["bpm"] = 100
    d["style"] = str(d.get("style") or "").strip(); d["key"] = str(d.get("key") or "C major").strip()
    if not d["style"]: raise ValueError("the director returned no style")
    return d


# Recreate a Short ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def describe_shot(clip_path, words, notes="", has_cast=True):
    """watch one shot (picture only) -> {picture, action, sound, by}: a prompt that re-shoots it with the cast from the reference image.
    The original performer's looks are deliberately NOT described — the reference image supplies the new face."""
    if not ensure(): raise RuntimeError("the director is not running")
    tmp = tempfile.mktemp(suffix=".mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip_path, "-an", "-vf", "scale=640:-2,fps=8", "-c:v", "libx264", "-crf", "30", tmp], capture_output=True, timeout=120)
    try: v = base64.b64encode(open(tmp, "rb").read()).decode()
    finally:
        try: os.remove(tmp)
        except OSError: pass
    who = ("Write the main person as \"the person from the reference image\". Do NOT describe their natural face, hair, skin colour, age or "
           "ethnicity — the reference image gives their identity. DO describe the scene's look as applied to the person from the reference image: "
           "any makeup, face paint, special-effects or monster/zombie makeup, skin tint from makeup, prosthetics, wounds, masks, unusual eye colour "
           "or contact lenses, styled or messy hair, and their costume, pose and facial expression.") if has_cast else "Describe any people neutrally."
    said = f'In this shot someone says: "{words}". Do not repeat the words; they are added separately.' if words else "Nobody speaks in this shot."
    change = f"Changes the user wants in the remake: {notes}" if notes.strip() else "Keep everything as close to the original as you can."
    msg = [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + v}},
            {"type": "text", "text": "You are remaking this shot of a short video with a new actor, for a text-to-video model that makes picture and sound. "
             f"{who} Keep the setting, framing, lens, light, camera movement, action, timing and mood. {said} {change}\n\n"
             "Reply JSON only: {\"picture\": \"40-90 words: who/what is in frame, where, light, lens\", \"action\": \"what happens and how the camera moves\", "
             "\"sound\": \"ambience and music only, not the spoken words\"}"}]}]
    t = time.time(); d = _ask(DIRECTOR, C.DIRECTOR_MODEL, msg, max_tokens=500, temperature=0.3)
    for k in ("picture", "action", "sound"): d[k] = str(d.get(k) or "").strip()
    if not d["picture"]: raise ValueError("the director returned no picture description")
    named = lambda x: "reference image" in (str(x.get("picture", "")) + " " + str(x.get("action", ""))).lower()
    if has_cast and not named(d):                                  # the cast must be named, or the shot has no <Subject 1> to tie it to the face
        try:
            d2 = _ask(DIRECTOR, C.DIRECTOR_MODEL, msg + [{"role": "assistant", "content": json.dumps({k: d[k] for k in ("picture", "action", "sound")})},
                      {"role": "user", "content": 'Rewrite it: call the main person only "the person from the reference image", and do not describe their '
                                                  "natural face, hair, skin colour, age or ethnicity (keep any makeup, face paint, costume and expression). "
                                                  "Same JSON keys."}], max_tokens=500, temperature=0.3)
            if named(d2): d.update({k: str(d2.get(k) or d[k]).strip() for k in ("picture", "action", "sound")})
        except Exception: pass
        if not named(d): d["picture"] = "The person from the reference image is the main person in this shot. " + d["picture"]
    d["by"] = C.DIRECTOR_LABEL; d["secs"] = round(time.time() - t, 1)
    return d


# Faithful Recreate ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def who_is_performer(video_path):
    """watch the Short (picture only, small) -> a visual description that picks out the main performer in any single frame"""
    if not ensure(): raise RuntimeError("the director is not running")
    tmp = tempfile.mktemp(suffix=".mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-t", "60", "-i", video_path, "-an", "-vf", "scale=360:-2,fps=4", "-c:v", "libx264", "-crf", "32", tmp],
                   capture_output=True, timeout=180)
    try: v = base64.b64encode(open(tmp, "rb").read()).decode()
    finally:
        try: os.remove(tmp)
        except OSError: pass
    d = _ask(DIRECTOR, C.DIRECTOR_MODEL, [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + v}},
             {"type": "text", "text": "Who is the main performer of this video (the star a remake would recast)? Reply JSON only: {\"who\": \"a short visual "
              "description that picks them out in any single frame: clothing, hair, distinctive features — never their name\"}"}]}], max_tokens=120, temperature=0.1)
    return str(d.get("who") or "the main person").strip()


def check_frame(image_path, who):
    """one still -> {present, face_visible, where, expression, makeup} for the performer"""
    b = base64.b64encode(open(image_path, "rb").read()).decode()
    d = _ask(DIRECTOR, C.DIRECTOR_MODEL, [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}},
             {"type": "text", "text": f"The performer is: {who}. Look at this frame. Reply JSON only: {{\"present\": true|false, \"face_visible\": true|false "
              "(their face is clearly visible and turned at least partly to the camera, not blurred or tiny), \"where\": \"where they are in the frame\", "
              "\"expression\": \"their facial expression in a few words\", \"makeup\": \"any makeup, face paint or special-effects makeup on their face, or none\"}"}]}],
             max_tokens=200, temperature=0.1)
    return {"present": d.get("present") is True, "face_visible": d.get("face_visible") is True, "where": str(d.get("where") or ""),
            "expression": str(d.get("expression") or ""), "makeup": str(d.get("makeup") or "none")}


# MIDI arranging ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def midi_arrange(parts, request, bpm, seconds, gm_names, kit_names):
    """the director reads a MIDI file's parts and the user's request -> {parts: {ch: GM name | kit name | "mute"}, style, restyle, strength, notes}"""
    if not ensure(): raise RuntimeError("the director is not running")
    rows = []
    for p in parts:
        lo, hi = (p.get("range") or [0, 0])[:2]
        role = "drums" if p.get("drums") else "bass register" if hi and hi < 60 else "melody / high register" if lo >= 64 else "middle register (chords or melody)"
        rows.append(f'channel {p["channel"]}: {"DRUM part" if p.get("drums") else "melodic part"}, track name "{p.get("name") or ""}", now plays as '
                    f'{p.get("instrument")}, {p.get("notes")} notes, MIDI range {lo}-{hi} ({role})')
    msg = [{"role": "system", "content": "You are a music arranger. You read MIDI parts and choose instruments that realise the requested sound. "
                                         "Reply with JSON only."},
           {"role": "user", "content": f"A MIDI file, {float(bpm):g} BPM, {float(seconds):g} s, with these parts:\n" + "\n".join(rows) +
            f"\n\nThe user wants: {request}\n\nChoose, for EVERY channel above, one instrument from the General MIDI list (exact names below), "
            f"or \"mute\" to drop a part. Drum parts take one of these kits instead: {', '.join(kit_names)}. Keep the parts' roles sensible "
            "(a low part stays a bass instrument unless asked otherwise).\n"
            f"General MIDI names: {'; '.join(gm_names)}.\n\nAlso write a 20-40 word style description of the requested sound for an AI music "
            "model that will re-render the arrangement (genre, instruments, playing feel, production), and say whether that AI restyle is needed: "
            "true when the request asks for something a General MIDI soundfont cannot produce well (a genre or production feel, a live band, "
            "analogue synths, a specific era), false when plain instrument swaps are enough. strength 0.4 = stay very close to the notes, 0.7 = freer.\n"
            'Return {"parts": {"<channel>": "<GM name | kit name | mute>"}, "style": "...", "restyle": true|false, "strength": 0.4-0.7, '
            '"notes": "one sentence on the choices"}'}]
    d = _ask(DIRECTOR, C.DIRECTOR_MODEL, msg, max_tokens=700, temperature=0.3)
    return d


# Ming-Image layout ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def ming_layout(request, width, height):
    """Ming-Image's own released text-to-image rewriter (a Figma-style JSON layout, layer by layer) run on the director.
    Returns (layout_json_text, seconds). aspect_ratio is forced to the chosen canvas, as Ming's makers ask."""
    from math import gcd
    rewriter = os.path.join(C.MING_CODE, "assets", "t2i_rewriter_system_prompt.txt")
    if not ensure(): raise RuntimeError(f"the director did not answer at {DIRECTOR} — is the server running? (see DIRECTOR_URL in studio.env)")
    g = gcd(int(width), int(height)); ar = f"{int(width) // g}:{int(height) // g}"
    msg = [{"role": "system", "content": open(rewriter).read()},
           {"role": "user", "content": f"{request}\n\nCanvas: {int(width)}x{int(height)} pixels, aspect_ratio {ar}."}]
    t = time.time(); d = _ask(DIRECTOR, C.DIRECTOR_MODEL, msg, max_tokens=3000, temperature=0.3)
    if not isinstance(d, dict) or "layers" not in d: raise ValueError("the director's layout had no layers")
    d.setdefault("canvas_settings", {})["aspect_ratio"] = ar
    return json.dumps(d, ensure_ascii=False), round(time.time() - t, 1)
