"""shot.py — is a rendered clip the one shot it was asked to be?

    check(clip, first=None, last=None, canvas=None, single=True)
        -> {passed, note, cuts, first, last, window, frames, fps, seconds}
    one_shot(prompt)   -> True unless the prompt itself asks for a cut
    longest_take(clip) -> {start, end, seconds, frames}: the longest stretch with no cut

How it looks: ffmpeg decodes the clip small and grey into a raw pipe (W = 128 px, numpy reads it; no PyAV). Held frames (a
frame that repeats the one before while its neighbours move) are skipped. Between consecutive pictures two numbers: the mean
absolute difference (MAD, 0..255) and the correlation of the two pictures after each has its mean removed, the best over
shifts of up to 3 px (1 = same picture, 0 = unrelated; a pan or shake is not a new picture). A hard cut is a change whose MAD
is an outlier against the median of its neighbours AND whose correlation is low AND that stands alone (the changes just
before and after are small next to it); a cut through one blended frame counts when both steps are sharp. A burst of 3 or
more flagged frames within 6 frames is something crossing the lens (petals, a hand, a flash), not a cut, and is ignored.

Keyframes: with FIRST / LAST (image paths) the clip's opening / closing frames are compared with the image fitted to the
clip's frame. canvas says how the image was fitted ("crop": scaled to cover and centre-cropped, "pad": letter/pillar-boxed,
"stretch"); None tries all three and keeps the best. A keyframe is held when its correlation is high and its MAD low.

Needs ffmpeg, ffprobe and numpy when it runs.

Calibration (2026-10-16, 36 rendered studio clips + 8 synthetic cuts on a DGX Spark): every cut found (4 clips joined from two
different renders, one joined through a fade, 3 jump cuts inside one render, and 4 real cuts in 3 rendered clips) except one soft,
two-frame cut into a fast dance shot; no false cut in the other clips, including petals sweeping the lens (2.75-2.88 s), a
spinning wheel, sparks and beam effects. The held-frame skip matters: several engines repeat every other frame. Keyframes on
21 first+last-frame takes: held at correlation >= 0.94 and MAD <= 6.3; the wrong images scored correlation <= 0.44, MAD >= 23.6.
"""
import json
import re
import subprocess
from pathlib import Path

W = 128                       # analysis width in pixels; height follows the clip's aspect
HOLD_MIN, HOLD_FRAC = 1.0, 0.15   # a frame within this of the one before (absolute, or as a share of its neighbours' change) is held
CUT_RATIO = 3.0               # a cut's MAD is at least this many times the local median MAD
CUT_FLOOR = 12.0              # ... and at least this much on the 0..255 scale
CUT_CORR = 0.80               # ... and the two pictures correlate below this even when shifted a few pixels
ISOLATED = 0.5                # ... and the changes just before and after it are at most this share of it (a cut is a spike)
LOCAL = 8                     # distinct frames on each side for the local median
BURST_N, BURST_SPAN = 3, 6    # this many flagged frames within this many frames = something crossing the lens
HELD_CORR = 0.85              # a keyframe is held at or above this correlation ...
HELD_MAD = 20.0               # ... and at or below this MAD
EDGE = 3                      # frames at each end searched for the best keyframe match
FITS = ("crop", "pad", "stretch")

CUT_WORDS = re.compile(r"\b(?:cut(?:s|ting)?\s+(?:back\s+)?to|smash[\s-]cuts?|jump[\s-]cuts?|hard\s+cuts?\s+to|match[\s-]cuts?|"
                       r"montage|intercut|cutaway)\b|\[\s*shot\s*[2-9]\d*\s*\]", re.I)


def one_shot(prompt: str) -> bool:
    """True when the prompt describes one continuous shot, False when it asks for a cut ("cut to", "smash cut", "jump cut",
    "montage", a second "[Shot 2]" heading...)"""
    return not CUT_WORDS.search(prompt or "")


def _probe(clip) -> dict:
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
                        "-show_entries", "format=duration", "-of", "json", str(clip)], capture_output=True, text=True, timeout=120)
    if o.returncode:
        raise RuntimeError(f"ffprobe failed: {(o.stderr or '').strip()[-200:]}")
    j = json.loads(o.stdout)
    s = j["streams"][0]

    def rate(x):
        a, _, b = (x or "0/1").partition("/")
        return float(a) / float(b or 1) if float(b or 1) else 0.0
    fps = rate(s.get("avg_frame_rate")) or rate(s.get("r_frame_rate")) or 24.0
    return {"w": int(s["width"]), "h": int(s["height"]), "fps": fps, "dur": float(j.get("format", {}).get("duration") or 0)}


def _size(info) -> tuple:
    h = max(2, int(round(W * info["h"] / info["w"] / 2)) * 2)
    return W, h


def frames(clip, info=None):
    """the clip as a float32 array (n, h, w) of grey frames at the analysis size"""
    import numpy as np
    info = info or _probe(clip)
    w, h = _size(info)
    o = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(clip), "-an", "-vf", f"scale={w}:{h}:flags=area,format=gray",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, timeout=600)
    if o.returncode:
        raise RuntimeError(f"ffmpeg failed: {o.stderr.decode(errors='replace').strip()[-200:]}")
    n = len(o.stdout) // (w * h)
    return np.frombuffer(o.stdout[:n * w * h], np.uint8).reshape(n, h, w).astype(np.float32)


def _image(path, w, h, fit):
    """an image fitted onto a w x h frame the way the renderer may have fitted it, grey, float32 (h, w)"""
    import numpy as np
    vf = {"crop": f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=area,crop={w}:{h}",
          "pad": f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=area,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
          "stretch": f"scale={w}:{h}:flags=area"}[fit]
    o = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-frames:v", "1", "-vf", vf + ",format=gray",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, timeout=120)
    if o.returncode or len(o.stdout) < w * h:
        raise RuntimeError(f"could not read {Path(path).name}: {o.stderr.decode(errors='replace').strip()[-160:]}")
    return np.frombuffer(o.stdout[:w * h], np.uint8).reshape(h, w).astype(np.float32)


def _corr(a, b) -> float:
    import numpy as np
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-6 else (1.0 if np.abs(a - b).max() < 1 else 0.0)


def _shift_corr(a, b, reach=3) -> float:
    """the best correlation of B with A moved by up to REACH pixels each way: a pan or a shake is not a new picture"""
    h, w = a.shape
    best = -1.0
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            pa = a[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)]
            pb = b[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
            best = max(best, _corr(pa, pb))
    return best


def distinct(fr) -> list:
    """indices of the frames that show something new. A frame that barely differs from the one before while its neighbours
    change a lot is a held frame (12 fps motion doubled to 24, an engine repeating frames) and is skipped, so that the
    change between two pictures is measured once, not spread over a repeat."""
    import numpy as np
    raw = [0.0] + [float(np.abs(fr[i] - fr[i - 1]).mean()) for i in range(1, len(fr))]
    keep = [0]
    for i in range(1, len(fr)):
        around = max(raw[i - 1] if i > 1 else 0.0, raw[i + 1] if i + 1 < len(fr) else 0.0)
        if raw[i] >= max(HOLD_MIN, HOLD_FRAC * around):
            keep.append(i)
    return keep


def find_cuts(fr, fps) -> tuple:
    """(cuts, crossings): cuts = [{frame, t, mad, corr, frames}], ``frame`` the first frame of the new shot and ``frames`` 1, or 2
    when the change took two steps (a blended frame between the shots); crossings = frame numbers where the picture changed
    sharply but not as a cut (a burst of changes, or a change that builds up over several frames)"""
    import numpy as np
    keep = distinct(fr)
    m = len(keep)
    mad = np.zeros(m)
    for k in range(1, m):
        mad[k] = float(np.abs(fr[keep[k]] - fr[keep[k - 1]]).mean())
    sharp = []                                                    # (k, span, change, corr, isolated)
    k = 1
    while k < m:
        lo, hi = max(1, k - LOCAL), min(m, k + LOCAL + 2)
        around = np.concatenate([mad[lo:k], mad[k + 2:hi]])
        base = float(np.median(around)) if len(around) else 0.0
        found = None
        for span in (1, 2):
            if k + span - 1 >= m:
                break
            a, b = fr[keep[k - 1]], fr[keep[k + span - 1]]
            change = mad[k] if span == 1 else float(np.abs(b - a).mean())
            if change < max(CUT_FLOOR, CUT_RATIO * base):
                continue
            if span == 2 and not (min(mad[k], mad[k + 1]) >= 0.35 * change
                                  and max(_shift_corr(fr[keep[k - 1]], fr[keep[k]]), _shift_corr(fr[keep[k]], b)) < CUT_CORR):
                continue                                          # two steps: each must be a sharp change of its own
            c = _shift_corr(a, b)
            if c >= CUT_CORR:
                continue
            outside = [mad[j] for j in (k - 1, k + span) if 1 <= j < m]
            found = (k, span, change, c, (max(outside) if outside else 0.0) <= ISOLATED * change)
            break
        if found:
            sharp.append(found)
        k += found[1] if found and found[4] else 1
    burst = set()
    for x in sharp:
        near = [y for y in sharp if abs(keep[y[0]] - keep[x[0]]) < BURST_SPAN]
        if len(near) >= BURST_N:
            burst.update(y[0] for y in near)
    cuts, crossing = [], []
    for k, span, change, c, isolated in sharp:
        i = keep[k]
        if isolated and k not in burst:
            cuts.append({"frame": i, "t": round(i / fps, 3), "mad": round(float(change), 1), "corr": round(c, 3), "frames": span})
        else:
            crossing.append(i)
    return cuts, crossing


def _window(cuts, n, fps) -> dict:
    edges = [0] + [c["frame"] for c in cuts] + [n]
    a, b = max(zip(edges, edges[1:]), key=lambda p: p[1] - p[0])
    return {"start": round(a / fps, 3), "end": round(b / fps, 3), "seconds": round((b - a) / fps, 3), "frames": b - a,
            "whole": not cuts}


def _held(img_path, ends, w, h, canvas) -> dict:
    """compare the image with the clip's frames ``ends`` (a few at one end); the best fit and frame win"""
    import numpy as np
    best = None
    for fit in ([canvas] if canvas in FITS else FITS):
        img = _image(img_path, w, h, fit)
        for k, f in ends:
            m = float(np.abs(f - img).mean())
            c = _corr(f, img)
            if best is None or (c, -m) > (best["corr"], -best["mad"]):
                best = {"corr": round(c, 3), "mad": round(m, 1), "frame": k, "fit": fit}
    best["held"] = best["corr"] >= HELD_CORR and best["mad"] <= HELD_MAD
    return best


def longest_take(clip) -> dict:
    """the longest stretch of CLIP with no hard cut"""
    info = _probe(clip)
    fr = frames(clip, info)
    cuts, _ = find_cuts(fr, info["fps"])
    return _window(cuts, len(fr), info["fps"])


def check(clip, first=None, last=None, canvas=None, single=True) -> dict:
    """the frame check of one take. ``single``: the prompt asked for one continuous shot, so any cut fails it."""
    info = _probe(clip)
    fr = frames(clip, info)
    n, fps = len(fr), info["fps"]
    if n < 2:
        return {"passed": True, "note": "too few frames to check", "cuts": [], "first": None, "last": None,
                "window": None, "frames": n, "fps": round(fps, 3), "seconds": round(n / fps, 3)}
    w, h = fr.shape[2], fr.shape[1]
    cuts, crossing = find_cuts(fr, fps)
    head = [(k, fr[k]) for k in range(min(EDGE, n))]
    tail = [(k, fr[k]) for k in range(max(0, n - EDGE), n)]
    fh = _held(first, head, w, h, canvas) if first else None
    lh = _held(last, tail, w, h, canvas) if last else None

    said, passed = [], True
    if cuts and single:
        passed = False
        said.append(f"{len(cuts)} hard cut{'s' if len(cuts) > 1 else ''} inside a one-shot clip at "
                    + ", ".join(f"{c['t']:.2f} s" for c in cuts))
    elif cuts:
        said.append(f"{len(cuts)} cut{'s' if len(cuts) > 1 else ''} (the prompt asks for them)")
    for name, r in (("first", fh), ("last", lh)):
        if r and not r["held"]:
            passed = False
            said.append(f"{name} frame not held (correlation {r['corr']:.2f}, difference {r['mad']:.0f})")
    if passed:
        bits = ["one continuous shot" if not cuts else "cuts as asked"]
        held = [nm for nm, r in (("first", fh), ("last", lh)) if r]
        if held:
            bits.append(" and ".join(held) + " frame" + ("s" if len(held) > 1 else "") + " held")
        said = ["; ".join(bits)]
    if crossing:
        said.append(f"fast change at {crossing[0] / fps:.2f}-{crossing[-1] / fps:.2f} s, not a cut (something crossing the lens)")
    return {"passed": passed, "note": "; ".join(said), "cuts": cuts, "first": fh, "last": lh, "window": _window(cuts, n, fps),
            "frames": n, "fps": round(fps, 3), "seconds": round(n / fps, 3)}


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    print(json.dumps(check(a[0], a[1] if len(a) > 1 and a[1] else None, a[2] if len(a) > 2 and a[2] else None), indent=1))
