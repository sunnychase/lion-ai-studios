#!/usr/bin/env python3
# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""download_models.py — fetch the models Lion AI Studios uses (Python standard library + curl; resumable; sizes checked).

  python3 scripts/download_models.py --list                      # what exists, sizes, licences
  python3 scripts/download_models.py                             # the core set (~122 GB): video, images, fast music
  python3 scripts/download_models.py --all                       # everything (~331 GB)
  python3 scripts/download_models.py --group video --group music # whole groups (core + optional)
  python3 scripts/download_models.py --file Chroma1-HD.safetensors

Destinations come from studio.env (COMFY_MODELS, STUDIO_HOME) or the environment. Files already present with the right size are skipped;
an interrupted download resumes from its .part file. Gated repos need HF_TOKEN in the environment (read by curl, never printed).
Check each model's licence before you use its output commercially (see --list and docs/MODELS.md)."""
import argparse, json, os, subprocess, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MANIFEST = json.load(open(os.path.join(HERE, "models.json")))


def env_file():
    p = os.path.join(ROOT, "studio.env")
    if not os.path.exists(p): return
    for line in open(p):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); v = v.split(" #")[0].strip().strip('"').strip("'")
            os.environ.setdefault(k.strip(), os.path.expandvars(v))


def dest_dir(e):
    kind, sub = e["dest"].split(":", 1)
    if kind == "comfy":
        base = os.path.expanduser(os.environ.get("COMFY_MODELS") or os.path.join(os.environ.get("COMFY_DIR", "~/ComfyUI"), "models"))
    else:
        base = os.path.join(os.path.expanduser(os.environ.get("STUDIO_HOME", "~/lion-studio")), "models")
    return os.path.join(base, sub)


def curl(url, out, size):
    if os.path.exists(out) and (size is None or os.path.getsize(out) == size): return "have"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    hdr = ["-H", f"Authorization: Bearer {os.environ['HF_TOKEN']}"] if os.environ.get("HF_TOKEN") and "huggingface.co" in url else []
    rc = subprocess.run(["curl", "-fSL", "-C", "-", "--retry", "5", "--retry-delay", "5", *hdr, "-o", out + ".part", url]).returncode
    got = os.path.getsize(out + ".part") if os.path.exists(out + ".part") else 0
    if rc == 0 and (size is None or got == size): os.replace(out + ".part", out); return "ok"
    return f"FAILED (curl {rc}, {got} of {size} bytes) — run again to resume"


def repo_files(repo):
    t = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1", timeout=60))
    return [(x["path"], x.get("size")) for x in t if x.get("type") == "file"]


def fetch(e):
    d = dest_dir(e)
    if e.get("url"): return [(e["file"], curl(e["url"], os.path.join(d, e["file"]), e.get("bytes")))]
    if e.get("path") == "*":                                   # a whole repo (the voice models)
        return [(p, curl(f"https://huggingface.co/{e['repo']}/resolve/main/{p}", os.path.join(d, p), s)) for p, s in repo_files(e["repo"])]
    return [(e["file"], curl(f"https://huggingface.co/{e['repo']}/resolve/main/{e['path']}", os.path.join(d, e["file"]), e.get("bytes")))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--group", action="append", choices=sorted({e["group"] for e in MANIFEST})); ap.add_argument("--file", action="append")
    a = ap.parse_args(); env_file()
    if a.list:
        for e in MANIFEST:
            print(f"{e['group']:6} {e['tier']:8} {e['bytes'] / 1e9:6.1f} GB  {e.get('file') or e['repo']:58} {e.get('license') or '?':30} {e['what']}")
        print(f"core {sum(e['bytes'] for e in MANIFEST if e['tier'] == 'core') / 1e9:.0f} GB · all {sum(e['bytes'] for e in MANIFEST) / 1e9:.0f} GB"); return
    pick = [e for e in MANIFEST if a.all or (a.group and e["group"] in a.group) or (a.file and e.get("file") in a.file)
            or (not (a.all or a.group or a.file) and e["tier"] == "core")]
    if not pick: sys.exit("nothing matched")
    print(f"{len(pick)} item(s), {sum(e['bytes'] for e in pick) / 1e9:.0f} GB"); bad = 0
    for e in pick:
        for name, res in fetch(e):
            print(f"  {res:6} {name}"); bad += not res.startswith(("ok", "have"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
