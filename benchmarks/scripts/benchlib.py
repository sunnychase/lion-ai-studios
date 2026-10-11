"""benchlib.py — shared settings and helpers for the benchmark scripts in this folder.

Everything machine-specific comes from the environment (same names as studio.env):
  COMFY_URL     ComfyUI API                       default http://127.0.0.1:8188
  COMFY_DIR     the ComfyUI checkout              default ~/ComfyUI       (its input/ and output/ folders are used)
  COMFY_MODELS  ComfyUI's models folder           default $COMFY_DIR/models
  STUDIO_URL    the studio web app                default http://127.0.0.1:8482   (only for jobs submitted through the studio)
  STUDIO_HOME   the studio's data folder          default ~/lion-studio   (library/, state/)
  BENCH_OUT     where results and copies go       default ./bench-out
  TENSORFOLD_LIB  libtf_h3.so for the TensorFold engine (optional)

Timing: every wall time below is measured from POST /prompt until ComfyUI's /history reports the prompt complete, polled every
2 s — so each number can be up to 2 s longer than the true time (never shorter) and includes text encoding, sampling, video + audio decode and the MP4 encode.
Model loading is included whenever the model was not already resident (the scripts do not force a warm or cold start).
The director review (where used) runs AFTER the timed render and is not part of wall_s."""
import json, os, re, subprocess, sys, time, urllib.request

E = os.environ.get
COMFY = E("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = os.path.expanduser(E("COMFY_DIR", "~/ComfyUI"))
COMFY_MODELS = os.path.expanduser(E("COMFY_MODELS", os.path.join(COMFY_DIR, "models")))
CI = os.path.join(COMFY_DIR, "input")
CO = os.path.join(COMFY_DIR, "output")
STUDIO_URL = E("STUDIO_URL", "http://127.0.0.1:8482").rstrip("/")
STUDIO_HOME = os.path.expanduser(E("STUDIO_HOME", "~/lion-studio"))
BENCH_OUT = os.path.abspath(os.path.expanduser(E("BENCH_OUT", "./bench-out")))
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.abspath(os.path.join(HERE, "..", "..", "app"))     # the studio's own modules (director.py, stitch.py, workflows.py)
WW, HH = 864, 480                                                 # frame size used for SSIM comparisons (callers may change it)

MIN_FREE_GIB = 6      # unified memory: interrupt a render rather than let the machine run dry
HOT_C = 85            # wait for the GPU to cool below this before starting a render


def get(p): return json.load(urllib.request.urlopen(COMFY + p, timeout=30))


def post(p, b):
    raw = urllib.request.urlopen(urllib.request.Request(COMFY + p, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=60).read()
    return json.loads(raw) if raw.strip() else {}


def avail_gib():
    return int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 2**20


def gpu_temp():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def ff(*a): subprocess.run(["ffmpeg", "-v", "error", "-y", *a], check=True)


def render(g, tag, client="bench"):
    """queue one graph when ComfyUI is idle and the GPU is cool; return (output files, wall seconds)"""
    while gpu_temp() >= HOT_C: time.sleep(30)
    while True:                                                    # never benchmark on top of someone else's job
        q = get("/queue")
        if not q["queue_running"] and not q["queue_pending"]: break
        time.sleep(20)
    t0 = time.time(); pid = post("/prompt", {"prompt": g, "client_id": client})["prompt_id"]
    while True:
        time.sleep(2)
        if avail_gib() < MIN_FREE_GIB:
            post("/interrupt", {}); time.sleep(3); post("/free", {"unload_models": True, "free_memory": True})
            raise RuntimeError(f"{tag}: available memory fell below {MIN_FREE_GIB} GiB — interrupted and freed")
        h = get(f"/history/{pid}")
        if pid in h and h[pid].get("status", {}).get("completed") is not None: break
    st = h[pid]["status"]
    if st.get("status_str") != "success":
        msg = {m[0]: m[1] for m in st.get("messages", [])}.get("execution_error", {})
        raise RuntimeError(f"{tag}: {msg.get('exception_type')}: {str(msg.get('exception_message'))[:300]}")
    files = [os.path.join(CO, x.get("subfolder", ""), x["filename"]) for o in h[pid]["outputs"].values() for v in o.values() if isinstance(v, list)
             for x in v if isinstance(x, dict) and x.get("filename") and x.get("type", "output") == "output"]
    return files, time.time() - t0


def ssim_frame(video, n, ref):
    """SSIM (ffmpeg's ssim filter, 'All' channel average) of frame n of `video` against an image, both scaled to WW x HH"""
    os.makedirs(BENCH_OUT, exist_ok=True)
    fr = os.path.join(BENCH_OUT, "_frame.png")
    ff("-i", video, "-vf", f"select=eq(n\\,{n}),scale={WW}:{HH}", "-frames:v", "1", fr)
    r = subprocess.run(["ffmpeg", "-i", fr, "-i", ref, "-lavfi", f"[1:v]scale={WW}:{HH}[r];[0:v][r]ssim", "-f", "null", "-"], capture_output=True, text=True).stderr
    return float(re.search(r"All:([0-9.]+)", r).group(1))


def ssim_span(a, a0, b, b0, n):
    """mean SSIM over n frames: a[a0:a0+n] against b[b0:b0+n], both scaled to WW x HH"""
    r = subprocess.run(["ffmpeg", "-i", a, "-i", b, "-lavfi",
                        f"[0:v]trim=start_frame={a0}:end_frame={a0 + n},setpts=PTS-STARTPTS,scale={WW}:{HH}[x];"
                        f"[1:v]trim=start_frame={b0}:end_frame={b0 + n},setpts=PTS-STARTPTS,scale={WW}:{HH}[y];[x][y]ssim", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    return float(re.search(r"All:([0-9.]+)", r).group(1))


def nframes(v):
    return int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", v],
                              capture_output=True, text=True).stdout.strip())


def review(video, prompt):
    """the studio director's review (app/director.py): a vision-language model watches the clip downscaled to 640 px wide at 8 fps,
    picture only (it does not hear the audio), and returns {score 1-10, matches_prompt, issues[]}. It is an AI judge, not a human rating."""
    if APP not in sys.path: sys.path.insert(0, APP)
    import director
    rv = director.review(video, prompt)
    return {k: rv.get(k) for k in ("score", "matches_prompt", "issues")}


def append(name, row):
    os.makedirs(BENCH_OUT, exist_ok=True)
    print(json.dumps(row), flush=True)
    with open(os.path.join(BENCH_OUT, name), "a") as f: f.write(json.dumps(row) + "\n")
