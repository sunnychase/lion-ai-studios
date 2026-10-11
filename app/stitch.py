"""stitch.py — join chained segments into one film. Segment k+1 was rendered with segment k's last GUIDE frames (and their audio) anchored
at its frame 0, so its first GUIDE frames repeat k's tail: they are dropped, and the sound crossfades over a short window across the seam."""
import json, os, subprocess

FPS = 24


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path], capture_output=True, text=True)
    d = json.loads(r.stdout or "{}"); v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), {})
    return {"dur": float(d.get("format", {}).get("duration") or 0), "w": v.get("width"), "h": v.get("height"), "frames": int(v.get("nb_frames") or 0),
            "audio": bool(a), "sr": int(a.get("sample_rate") or 0)}


def join(segments, out, guide_frames=22, xfade=0.12, crf=14):
    """segments: rendered mp4 paths in order (segment 1 unguided, the rest guided by their predecessor). Returns ffmpeg rc and a log tail."""
    if len(segments) == 1:
        subprocess.run(["cp", segments[0], out]); return 0, "single segment"
    cut = guide_frames / FPS
    inputs, fv, fa = [], [], []
    for i, s in enumerate(segments):
        inputs += ["-i", s]
        if i == 0:
            fv.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]"); fa.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")
        else:   # drop the repeated tail of the previous segment; keep a little extra sound for the crossfade
            fv.append(f"[{i}:v]trim=start_frame={guide_frames},setpts=PTS-STARTPTS[v{i}]")
            fa.append(f"[{i}:a]atrim=start={max(0.0, cut - xfade):.4f},asetpts=PTS-STARTPTS[a{i}]")
    vcat = "".join(f"[v{i}]" for i in range(len(segments))) + f"concat=n={len(segments)}:v=1:a=0[v]"
    achain, prev = [], "a0"
    for i in range(1, len(segments)):
        nxt = f"ax{i}"; achain.append(f"[{prev}][a{i}]acrossfade=d={xfade}:c1=qsin:c2=qsin[{nxt}]"); prev = nxt
    graph = ";".join(fv + fa + [vcat] + achain)
    cmd = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[v]", "-map", f"[{prev}]",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stderr or "")[-600:]


def join_keyframes(segments, out, fade=0.06, crf=14):
    """keyframe films: segment i ends on keyframe i+1 and segment i+1 starts on it, so the repeated boundary frame is dropped; each
    segment's sound was made on its own, so the joins get a very short fade out/in (no overlap: picture and sound stay in sync)."""
    if len(segments) == 1:
        subprocess.run(["cp", segments[0], out]); return 0, "single segment"
    # the dropped boundary frame's sound (1/FPS) overlaps the previous clip's end as an equal-power crossfade — no hole, same sync
    inputs, parts = [], []
    for i, s in enumerate(segments):
        inputs += ["-i", s]
        vt = "setpts=PTS-STARTPTS" if i == 0 else "trim=start_frame=1,setpts=PTS-STARTPTS"
        parts.append(f"[{i}:v]{vt}[v{i}];[{i}:a]asetpts=PTS-STARTPTS[a{i}]")
    vcat = "".join(f"[v{i}]" for i in range(len(segments))) + f"concat=n={len(segments)}:v=1:a=0[v]"
    achain, prev = [], "a0"
    for i in range(1, len(segments)):
        achain.append(f"[{prev}][a{i}]acrossfade=d={1 / FPS:.5f}:c1=qsin:c2=qsin[ax{i}]"); prev = f"ax{i}"
    cmd = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(parts + [vcat] + achain), "-map", "[v]", "-map", f"[{prev}]",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stderr or "")[-600:]


def poster(video, jpg, at=1.0):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(at), "-i", video, "-frames:v", "1", "-vf", "scale=480:-2", jpg], capture_output=True)
    return os.path.exists(jpg)
