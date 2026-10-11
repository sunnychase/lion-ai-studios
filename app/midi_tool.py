#!/usr/bin/env python3
"""midi_tool.py — the studio's MIDI worker (runs in the midi venv: mido + tinysoundfont, MuseScore General soundfont, MIT).
  inspect FILE              -> JSON: the parts (one per MIDI channel that plays notes), their GM instrument, notes, range, drums; tempo; length
  render FILE OUT MAP.json  -> renders with the user's instruments. MAP: {"parts": {"<ch>": <program 0-127> | "mute" | "kit:<n>"},
                               "transpose": semitones (melodic parts), "speed": 0.5-2.0, "gain": dB}
Prints one JSON line on success; exits non-zero with the reason on failure."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
SF2 = C.SF2
SR = 44100; DRUMS = 9
GM = ["Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
      "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba", "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ", "Percussive Organ", "Rock Organ",
      "Church Organ", "Reed Organ", "Accordion", "Harmonica", "Tango Accordion", "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)",
      "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics", "Acoustic Bass", "Electric Bass (finger)",
      "Electric Bass (pick)", "Fretless Bass", "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2", "Violin", "Viola", "Cello", "Contrabass", "Tremolo Strings",
      "Pizzicato Strings", "Orchestral Harp", "Timpani", "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2", "Choir Aahs", "Voice Oohs",
      "Synth Choir", "Orchestra Hit", "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2", "Soprano Sax",
      "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe", "English Horn", "Bassoon", "Clarinet", "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi",
      "Whistle", "Ocarina", "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)",
      "Lead 8 (bass + lead)", "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)",
      "Pad 8 (sweep)", "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)", "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)",
      "FX 8 (sci-fi)", "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bagpipe", "Fiddle", "Shanai", "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum",
      "Melodic Tom", "Synth Drum", "Reverse Cymbal", "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet", "Telephone Ring", "Helicopter", "Applause", "Gunshot"]
KITS = {0: "Standard Kit", 8: "Room Kit", 16: "Power Kit", 24: "Electronic Kit", 25: "TR-808 Kit", 32: "Jazz Kit", 40: "Brush Kit", 48: "Orchestra Kit"}


def inspect(path):
    import mido
    m = mido.MidiFile(path); parts = {}; tempo = None
    for ti, tr in enumerate(m.tracks):
        tname = next((x.name for x in tr if x.type == "track_name"), "") or f"Track {ti + 1}"
        for msg in tr:
            if msg.type == "set_tempo" and tempo is None: tempo = round(mido.tempo2bpm(msg.tempo), 1)
            if not hasattr(msg, "channel"): continue
            p = parts.setdefault(msg.channel, {"channel": msg.channel, "program": None, "notes": 0, "low": 127, "high": 0, "tracks": set()})
            if msg.type == "program_change" and p["program"] is None: p["program"] = msg.program
            if msg.type == "note_on" and msg.velocity > 0:
                p["notes"] += 1; p["low"] = min(p["low"], msg.note); p["high"] = max(p["high"], msg.note); p["tracks"].add(tname)
    out = []
    for ch, p in sorted(parts.items()):
        if not p["notes"]: continue
        drums = ch == DRUMS; prog = p["program"] or 0
        out.append({"channel": ch, "drums": drums, "program": prog, "instrument": KITS.get(prog, "Standard Kit") if drums else GM[prog], "notes": p["notes"],
                    "range": [p["low"], p["high"]], "name": ", ".join(sorted(p["tracks"]))[:60]})
    if not out: raise RuntimeError("that MIDI file has no notes")
    return {"ok": True, "parts": out, "bpm": tempo or 120.0, "seconds": round(m.length, 2), "tracks": len(m.tracks)}


def render(path, out, mapping):
    import numpy as np, soundfile as sf, tinysoundfont as tsf
    from tinysoundfont.midi import NoteOn, NoteOff, ProgramChange
    M = json.load(open(mapping)) if isinstance(mapping, str) else mapping
    parts = {int(k): v for k, v in (M.get("parts") or {}).items()}
    tr = int(M.get("transpose") or 0); speed = max(0.5, min(2.0, float(M.get("speed") or 1.0)))
    synth = tsf.Synth(samplerate=SR); sfid = synth.sfload(SF2)
    for ch in range(16):                                   # defaults first; the file's own program changes are remapped below
        v = parts.get(ch)
        if ch == DRUMS: synth.program_select(ch, sfid, 128, int(str(v)[4:]) if isinstance(v, str) and v.startswith("kit:") else 0, True)
        else: synth.program_select(ch, sfid, 0, v if isinstance(v, int) else 0)

    def edit(ev):
        ch = ev.channel; v = parts.get(ch); a = ev.action
        if v == "mute" and isinstance(a, (NoteOn, NoteOff)): return True
        if isinstance(a, ProgramChange):
            if ch == DRUMS and isinstance(v, str) and v.startswith("kit:"): a.program = int(v[4:])
            elif isinstance(v, int): a.program = v
        if tr and ch != DRUMS and isinstance(a, (NoteOn, NoteOff)): a.key = max(0, min(127, a.key + tr))
        ev.t = ev.t / speed
        return None

    events = tsf.midi.load(path, filter=edit, persistent=False)
    if not events: raise RuntimeError("nothing to play (every part muted?)")
    end = max(e.t for e in events) + 2.5                     # let the last notes ring
    seq = tsf.Sequencer(synth); seq.add(events)
    chunks, n, block = [], int(end * SR), 4096
    for _ in range(0, n, block): chunks.append(np.frombuffer(synth.generate(block), dtype=np.float32).copy())
    audio = np.concatenate(chunks)[: n * 2].reshape(-1, 2)
    peak = float(np.abs(audio).max() or 1.0); audio *= (10 ** (-1.0 / 20)) / peak * (10 ** (float(M.get("gain") or 0) / 20))
    sf.write(out, np.clip(audio, -1, 1), SR, subtype="PCM_16")
    return {"ok": True, "out": out, "seconds": round(len(audio) / SR, 2)}


if __name__ == "__main__":
    try:
        c = sys.argv[1]
        if c == "inspect": r = inspect(sys.argv[2])
        elif c == "render": r = render(sys.argv[2], sys.argv[3], sys.argv[4])
        elif c == "instruments": r = {"ok": True, "gm": GM, "kits": KITS}
        else: raise RuntimeError(__doc__)
        print(json.dumps(r))
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}"})); sys.exit(1)
