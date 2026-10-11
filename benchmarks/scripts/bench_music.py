#!/usr/bin/env python3
"""bench_music.py — the same three briefs on ACE-Step 1.5 Turbo ("fast") and MiniMax Music 3 ("best"), one render each.

Per run: wall time (POST /prompt -> done, polled every 2 s, so a short ACE-Step time can be up to 2 s high), the length asked for and the
length delivered (ffprobe of the MP3), the realtime factor = delivered seconds / wall seconds, and the mean volume (ffmpeg volumedetect).
ACE-Step renders exactly the asked length; MiniMax Music 3 treats the length as a maximum and plans its own.
Seeds 1010, 1011, 1012. Engines run in order fast then best, so the first run of each includes loading that model if it was not resident.
-> $BENCH_OUT/music.jsonl and MP3s  (results: benchmarks/results/music_ace_step_vs_minimax_music3.csv)"""
import os, re, subprocess
import benchlib as B
import music_graphs as M

OUT = os.path.join(B.BENCH_OUT, "music"); os.makedirs(OUT, exist_ok=True)
LYR = """[Verse]
Paper boats on a silver stream
Racing down to the end of the dream
You said hold on, the tide is high
We were laughing at the falling sky

[Chorus]
Run, run, into the light
Every window is open tonight
We don't need a map to find our way
Hold my hand until the break of day

[Outro]
Until the break of day"""
BRIEFS = [  # name, seconds, lyrics, ACE-Step style tags, MiniMax Music 3 brief, ACE-Step bpm/key
    ("score", 60, "",
     "cinematic orchestral film score, instrumental, 90 BPM, D minor, soft piano and strings building to a hopeful swelling climax with French horns and timpani, then settling",
     "Global Metadata: Cinematic orchestral film score, instrumental. 90 BPM, D minor. Begins intimate and tender, grows steadily into a hopeful, "
     "soaring climax, then settles into calm resolution. Film trailer and end-credits use. Clean, wide, modern orchestral recording.\n\n"
     "Vocal Details: No vocals.\n\nArrangement: Intro: solo felt piano with soft sustained strings. Build: cellos and violas add a rising ostinato, "
     "light percussion enters. Climax: full strings, French horns carry the theme, timpani rolls and cymbal swells. Outro: piano and high strings alone, a final held chord.", dict(bpm=90, key="D minor")),
    ("song", 60, LYR,
     "upbeat indie pop, bright female vocal, jangly electric guitars, driving drums, warm bass, catchy chorus, 118 BPM, G major",
     "Global Metadata: Upbeat indie pop. 118 BPM, G major. Bright, nostalgic and euphoric, a summer-night feeling that lifts into the chorus. "
     "Polished modern indie production, warm and punchy.\n\nVocal Details: Bright, clear female lead vocal, close and intimate in the verses, "
     "open and soaring in the chorus, light doubled harmonies on the chorus.\n\nArrangement: Jangly clean electric guitars, driving four-on-the-floor drums, "
     "warm melodic bass, shimmering synth pad in the chorus. Verses sparse, chorus full and wide, short outro repeating the last line.", dict(bpm=118, key="G major")),
    ("lofi", 30, "",
     "lo-fi hip-hop, chillhop, instrumental, 78 BPM, mellow Rhodes chords, dusty boom-bap drums, vinyl crackle, warm bass",
     "Global Metadata: Lo-fi hip-hop, chillhop, instrumental. 78 BPM, D flat major. Laid-back and dreamy, late-night study mood. "
     "Dusty bedroom production with vinyl crackle and tape hiss.\n\nVocal Details: No vocals.\n\nArrangement: Dusty boom-bap drums, soft kick, lazy snare, "
     "warm round bass, mellow Rhodes chords with slow chorus wobble, a jazzy guitar lick answering every few bars.", dict(bpm=78, key="Db major")),
]

if __name__ == "__main__":
    for eng in ("fast", "best"):
        if not M.ready(eng): B.append("music.jsonl", {"engine": eng, "error": "model files missing"}); continue
        for i, (name, sec, lyr, tags, brief, kw) in enumerate(BRIEFS):
            tag = f"{name}_{eng}"
            try:
                g = M.graph(eng, tags if eng == "fast" else brief, lyr, sec, 1010 + i, f"bench_music/{tag}", **(kw if eng == "fast" else {}))
                files, took = B.render(g, tag, client="bench_music"); mp3 = f"{OUT}/{tag}.mp3"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", files[0], "-c:a", "libmp3lame", "-q:a", "2", mp3], check=True)
                dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", mp3], capture_output=True, text=True).stdout)
                lv = re.search(r"mean_volume: ([-0-9.]+)", subprocess.run(["ffmpeg", "-i", mp3, "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True).stderr)
                r = {"name": name, "engine": M.ENGINES[eng], "seed": 1010 + i, "asked_s": sec, "got_s": round(dur, 1), "wall_s": round(took, 1),
                     "x_realtime": round(dur / took, 2), "mean_db": float(lv.group(1)) if lv else None, "file": os.path.basename(mp3)}
            except Exception as e: r = {"name": name, "engine": M.ENGINES[eng], "error": str(e)[:400]}
            B.append("music.jsonl", r)
