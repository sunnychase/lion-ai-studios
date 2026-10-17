#!/usr/bin/env python3
# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""Lion AI Studios — a professional media suite for the NVIDIA DGX Spark: video with sound (MiniMax H3 / FastH3), images (the ComfyUI
image models), music (ACE-Step 1.5, MiniMax Music 3, MIDI), cloned voices, a director that writes and reviews every shot, and long films
chained from 15 s segments whose seams carry the previous segment's frames AND sound. Renders run in ComfyUI.
Listens on config.HOST:PORT (127.0.0.1 by default) — put an authenticating reverse proxy in front before exposing it anywhere.
GPU rule: if another GPU service is configured (GPU_TENANT_*), a render that needs the GPU names it, waits for the click, stops it,
and offers to start it again afterwards. Thermal guard: before every segment, waits while the hottest board zone is >= HOT_C."""
import json, math, os, re, shutil, socket, base64, struct, subprocess, sys, threading, time, urllib.request, urllib.error, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import config as C                                             # noqa: E402
import workflows as W, stitch, director as DIR                 # noqa: E402
import music as MU                                              # noqa: E402  
import studio_broker as SB                                      # noqa: E402  (the memory broker)
import midi_tool as MT                                          # noqa: E402  (GM names only; mido/tinysoundfont run in midi-venv)
import negation as NG                                           # noqa: E402  (prompt words the models would show)
import shot as SHOT                                             # noqa: E402  (cuts / keyframes held)
import drama3 as DR3                                            # noqa: E402  (optional Fish Audio Drama 3: paths, server unit, fetch hook)

ROOT = C.STUDIO_HOME; LIB = f"{ROOT}/library"; STATE = f"{ROOT}/state"; UP = f"{ROOT}/uploads"
VOICES = f"{ROOT}/voices"; TTS_PY = C.TTS_PY; VOICE_PY = f"{HERE}/voice.py"
TRASH = C.TRASH                                                 # deletes move here; Restore brings them back
for d in (LIB, STATE, UP, VOICES, TRASH): os.makedirs(d, exist_ok=True)
COMFY = C.COMFY_URL; COMFY_IN = f"{C.COMFY_DIR}/input"; COMFY_OUT = f"{C.COMFY_DIR}/output"
JOBS_F = f"{STATE}/jobs.json"; PORT = C.PORT
NEED_GIB = C.NEED_GIB; HOT_C = C.HOT_C; COOL_C = C.COOL_C
LOCK = threading.Lock(); JOBS = {}; GPU = {"freed_tf": False, "busy": "", "note": ""}; PROGRESS = {}


def now(): return time.strftime("%Y-%m-%dT%H:%M:%S")
def save():
    with LOCK:
        tmp = JOBS_F + ".tmp"; json.dump(JOBS, open(tmp, "w"), indent=1); os.replace(tmp, JOBS_F)
def sh(cmd, t=20):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=t).stdout.strip()
    except Exception: return ""


# ── the machine: GPU owner, memory, heat ─────────────────────────────────────────────────────────────────────────────────────────────
def meminfo_gib():
    try: return int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 1048576
    except Exception: return 0.0
def heat():
    zs = []
    for z in os.listdir("/sys/class/thermal"):
        try: zs.append(int(open(f"/sys/class/thermal/{z}/temp").read()) / 1000)
        except Exception: pass
    q = sh(["nvidia-smi", "--query-gpu=temperature.gpu,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown", "--format=csv,noheader"], 8)
    parts = [x.strip() for x in q.split(",")] if q else []
    return {"board": max(zs) if zs else None, "gpu": float(parts[0]) if parts and parts[0].replace(".", "").isdigit() else None,
            "throttle": any(p == "Active" for p in parts[1:])}
_OWN = {"t": 0, "v": None}
def gpu_owner(fresh=False):
    """what holds the GPU besides ComfyUI: {'tf': bool, 'other': [names], 'mem': GiB free} (cached 5 s for the page's polls)"""
    if not fresh and _OWN["v"] and time.time() - _OWN["t"] < 5: return _OWN["v"]
    _OWN.update(t=time.time(), v=_gpu_owner()); return _OWN["v"]
def _gpu_owner():
    """the optional other GPU service (GPU_TENANT_CHECK exits 0 while it runs)"""
    up = bool(C.TENANT_CHECK) and subprocess.run(C.TENANT_CHECK, shell=True, capture_output=True, timeout=20).returncode == 0
    return {"tf": up, "other": [], "mem": round(meminfo_gib(), 1), "tenant": C.TENANT_NAME}
def comfy_up():
    try: urllib.request.urlopen(f"{COMFY}/system_stats", timeout=3).read(); return True
    except Exception: return False
def ensure_comfy():
    if comfy_up(): return True
    if not C.COMFY_SERVICE: return False
    sh(["systemctl", "--user", "start", C.COMFY_SERVICE], 30)
    for _ in range(90):
        if comfy_up(): WS.kick(); comfy_oom_first(); return True
        time.sleep(2)
    return False


# ── the memory guard: the floor under a render ──────────────────────────────────────────────────────────────────────────────────────
# The Spark's GPU and system share one pool: a render that overruns it makes the whole box stop answering. While a studio render runs,
# MemAvailable is read 4x/s: under MEM_FLOOR_GIB, or falling fast enough to cross MEM_CRIT_GIB within a second -> ComfyUI is interrupted and the job fails with the reason; under MEM_CRIT_GIB ComfyUI is also stopped
# outright (COMFY_SERVICE; the next job starts it again). ComfyUI is also marked the kernel's first choice if memory runs out all the same.
MEM_FLOOR_GIB, MEM_CRIT_GIB = C.MEM_FLOOR_GIB, C.MEM_CRIT_GIB; RUNNING = {"pid": None, "stop": ""}


def comfy_oom_first():
    if not C.COMFY_SERVICE: return
    try:
        mp = sh(["systemctl", "--user", "show", "-p", "MainPID", "--value", C.COMFY_SERVICE], 10).strip()
        if mp and mp != "0": open(f"/proc/{mp}/oom_score_adj", "w").write("800")
    except Exception: pass


def mem_guard():
    prev = None
    while True:
        time.sleep(0.25)
        if not RUNNING["pid"]: prev = None; continue
        m = meminfo_gib(); fall = (prev - m) * 4 if prev is not None else 0.0; prev = m
        if m <= 0: continue
        if m < MEM_FLOOR_GIB or (fall > 0 and m < 2 * MEM_FLOOR_GIB and m - fall < MEM_CRIT_GIB):   # a fast fall only counts under twice the floor (a model loading mid-render is a step, not a runaway)
            RUNNING["stop"] = (f"stopped to protect the Spark: free memory fell to {m:.1f} GiB (floor {MEM_FLOOR_GIB:g} GiB) — try a shorter clip, "
                               "a smaller size, or stop other GPU services first")
            try: urllib.request.urlopen(urllib.request.Request(f"{COMFY}/interrupt", data=b"{}", headers={"Content-Type": "application/json"}), timeout=5)
            except Exception: pass
            if m < MEM_CRIT_GIB and C.COMFY_SERVICE:
                try: subprocess.run(["systemctl", "--user", "stop", C.COMFY_SERVICE], timeout=60)
                except Exception: pass
            time.sleep(3); prev = None


# ── the optional Fast image engine (LocalRouter) + the memory broker ─────────────────────────────────────────────────────────────────
# LocalRouter (config.LR_*) is off unless LOCALROUTER_ENABLE is set. The studio starts it for a Fast job (LOCALROUTER_SERVICE or
# LOCALROUTER_START), LocalRouter's own idle_ttl_s unloads the model, and lr_idle() stops it after LR_IDLE_S without a Fast job.
LR = C.LR_URL; LR_IDLE_S = C.LR_IDLE_S; LRF = {"last": 0.0, "busy": False}
def cur_kind():
    j = JOBS.get(GPU.get("busy") or "")
    return "" if not j else "ming" if j["kind"] == "image" and j["params"].get("model") == W.MING else j["kind"]
def lr_can_stop(): return bool(C.LR_SERVICE or C.LR_STOP)
def lr_stop_now():
    """stop LocalRouter (its unit, or LOCALROUTER_STOP); returns a short note"""
    try:
        if C.LR_SERVICE: return f"systemctl --user stop {C.LR_SERVICE} → rc {subprocess.run(['systemctl', '--user', 'stop', C.LR_SERVICE], capture_output=True, timeout=90).returncode}"
        if C.LR_STOP: return f"LOCALROUTER_STOP → rc {subprocess.run(C.LR_STOP, shell=True, capture_output=True, timeout=120).returncode}"
    except Exception as e: return f"stop failed ({type(e).__name__})"
    return "no stop command is configured"
BR = SB.studio_registry(SB.Broker(), COMFY, LR if C.LR_ENABLE else None, lr_stop_now if lr_can_stop() else None, DIR, cur_kind,
                        C.DIRECTOR_LABEL, bool(C.DIRECTOR_SERVICE), C.LR_RESERVE_GIB)
# a director call in flight marks the director busy, so the broker never stops it mid-request
DIR._chat = BR.track("director", DIR._chat); DIR.ensure = BR.track("director", DIR.ensure)
def lr_ready(): return C.LR_ENABLE
def lr_up():
    try: return json.load(urllib.request.urlopen(f"{LR}/health", timeout=3)).get("ok") is True
    except Exception: return False
def lr_active():
    if not C.LR_SERVICE: return lr_up()
    try: return subprocess.run(["systemctl", "--user", "is-active", "-q", C.LR_SERVICE], capture_output=True, timeout=10).returncode == 0
    except Exception: return False
def lr_ensure():
    if lr_up(): return True
    if not lr_ready() or not (C.LR_SERVICE or C.LR_START): return False
    try:
        if C.LR_SERVICE: sh(["systemctl", "--user", "start", C.LR_SERVICE], 30)
        else: subprocess.run(C.LR_START, shell=True, capture_output=True, timeout=120)   # must return once it is started (e.g. docker start / run -d)
    except Exception: return False
    for _ in range(45):
        if lr_up(): return True
        if C.LR_SERVICE and not lr_active(): return False
        time.sleep(2)
    return False
def lr_job(j):
    p = j.get("params") or {}
    return lr_ready() and j.get("kind") == "image" and p.get("iengine") == "localrouter" and not W.lr_fits(p.get("model"), p.get("w"), p.get("h"), p.get("count", 1))
def lr_yield(job):
    """a warm LocalRouter (~36 GiB) gives way before any other job: its models unload (its own POST /v1/tools/release); never fails the job"""
    if not lr_ready(): return
    try:
        if not lr_job(job) and lr_up() and SB.lr_held(LR): BR.release("localrouter", f"before {job['kind']} {job['id']}")
    except Exception: pass
def lr_idle():
    """stop LocalRouter after LR_IDLE_S without a Fast job — never while one runs or is queued, or while LocalRouter reports a model
    loading/busy. One found running (a studio restart, or started by hand) gets the full window from when it was first seen."""
    while True:
        time.sleep(30)
        try:
            if LRF["busy"] or not lr_active(): continue
            if not LRF["last"]: LRF["last"] = time.time(); continue
            if time.time() - LRF["last"] < LR_IDLE_S: continue
            if any(j["status"] in ("queued", "waiting_gpu", "running") and lr_job(j) for j in list(JOBS.values())): continue
            t = SB.lr_tools(LR)
            if t and any(x.get("state") in ("loading", "busy") for x in t.get("data") or []): continue
            note = lr_stop_now(); LRF["last"] = 0.0
            BR._remember({"ok": True, "log": [f"LocalRouter stopped after {LR_IDLE_S // 60} min idle ({note})"]}, "idle stop")
        except Exception: pass


# ── ComfyUI websocket (progress) — a minimal RFC 6455 client; text frames only ─────────────────────────────────────────────────────
CLIENT_ID = "lion-studio-" + uuid.uuid4().hex[:8]


class ComfyWS(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.sock = None
    def kick(self):
        try: self.sock and self.sock.close()
        except Exception: pass
    def _connect(self):
        u = urlparse(COMFY); hp = (u.hostname or "127.0.0.1", u.port or 80)
        s = socket.create_connection(hp, timeout=10); key = base64.b64encode(os.urandom(16)).decode()
        s.sendall((f"GET /ws?clientId={CLIENT_ID} HTTP/1.1\r\nHost: {hp[0]}:{hp[1]}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                   f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        hdr = b""
        while b"\r\n\r\n" not in hdr: hdr += s.recv(1)
        if b" 101 " not in hdr.split(b"\r\n")[0]: raise OSError("no upgrade")
        s.settimeout(None); return s
    def _recv(self, n):
        b = b""
        while len(b) < n:
            c = self.sock.recv(n - len(b))
            if not c: raise OSError("closed")
            b += c
        return b
    def run(self):
        while True:
            try:
                self.sock = self._connect()
                while True:
                    h = self._recv(2); op = h[0] & 0x0F; ln = h[1] & 0x7F
                    if ln == 126: ln = struct.unpack(">H", self._recv(2))[0]
                    elif ln == 127: ln = struct.unpack(">Q", self._recv(8))[0]
                    data = self._recv(ln)
                    if op == 8: raise OSError("close")
                    if op != 1: continue
                    try: m = json.loads(data)
                    except ValueError: continue
                    d = m.get("data") or {}; pid = d.get("prompt_id")
                    if m.get("type") == "progress" and pid: PROGRESS[pid] = {"value": d.get("value"), "max": d.get("max"), "node": d.get("node"), "t": time.time()}
                    elif m.get("type") == "executing" and pid: PROGRESS.setdefault(pid, {})["node"] = d.get("node")
            except Exception:
                time.sleep(3)
WS = ComfyWS()


def comfy_post(graph):
    req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": graph, "client_id": "studio"}).encode(), headers={"Content-Type": "application/json"})
    try: return json.load(urllib.request.urlopen(req, timeout=30))["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError("ComfyUI refused the graph: " + e.read().decode("utf-8", "replace")[:600])
def comfy_wait(pid, job):
    RUNNING.update(pid=pid, stop="")                                          # the memory guard watches while this runs
    try: return _comfy_wait(pid, job)
    finally: RUNNING.update(pid=None)


def _comfy_wait(pid, job):
    while True:
        if RUNNING["stop"]: raise RuntimeError(RUNNING["stop"])              # the memory guard stopped it
        if job.get("cancel"):
            try: urllib.request.urlopen(urllib.request.Request(f"{COMFY}/interrupt", data=b"{}", headers={"Content-Type": "application/json"}), timeout=10)
            except Exception: pass
            raise RuntimeError("cancelled")
        try: h = json.load(urllib.request.urlopen(f"{COMFY}/history/{pid}", timeout=30))
        except Exception: h = {}
        if pid in h:
            st = h[pid].get("status") or {}
            if st.get("status_str") == "error":
                msg = next((m[1].get("exception_message") for m in st.get("messages", []) if m[0] == "execution_error"), "render failed")
                raise RuntimeError(str(msg)[:400])
            files = []
            for out in (h[pid].get("outputs") or {}).values():
                for key in ("images", "videos", "gifs", "audio"):
                    for f in out.get(key) or []:    # only what the graph SAVED (a LoadVideo/LoadImage also reports its input file here)
                        if isinstance(f, dict) and f.get("filename") and f.get("type", "output") == "output":
                            files.append(os.path.join(COMFY_OUT, f.get("subfolder") or "", f["filename"]))
            return files
        p = PROGRESS.get(pid) or {}
        if p.get("max"): job["progress"].update(step=p.get("value"), steps=p.get("max"))
        time.sleep(2)


# ── jobs ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def new_job(kind, params):
    jid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    est = 0
    segs = plan_segments(params) if kind == "chain" else params["gaps"] if kind == "keyframes" else [params.get("seconds", 5)]
    if kind in ("video", "chain", "keyframes"):
        est = sum(W.estimate_s(params["preset"], s, params.get("engine", "comfy")) for s in segs)   # engine-aware
        if is_cast(params): est = sum(W.cast_estimate_s(params["preset"], s) for s in segs)
        if params.get("music") in ("fast", "best"): est += MU.estimate_s(params["music"], sum(segs))
    elif kind == "music": est = MU.estimate_s(params["engine"], params["seconds"])
    elif kind == "retake": est = W.estimate_s(params["preset"], params["seconds"], params.get("engine", "comfy")) + 20
    elif kind == "export": est = 30
    elif kind == "image" and params.get("model") == W.MING:     # measured on a DGX Spark: load 309-363 s, ~22 s per image at 1 MP, ~122 s at 4 MP
        est = round(340 + 30 * params["w"] * params["h"] / 1048576 * params.get("count", 1) + (60 if params.get("layout") else 0))
    elif kind == "recreate": est = W.cast_estimate_s(params["preset"], params["max_seconds"]) + 60 + 25 * math.ceil(params["max_seconds"] / 5)
    j = {"id": jid, "kind": kind, "status": "queued", "created": now(), "params": params, "progress": {"seg": 0, "segs": len(segs) if kind != "image" else 1,
         "step": 0, "steps": 0, "phase": "queued"}, "outputs": [], "error": "", "est_s": est}
    with LOCK: JOBS[jid] = j
    save(); return j


def plan_segments(p):
    """a chain of 15 s segments; each continuation repeats GUIDE_FRAMES of its predecessor, so it adds 15 s minus that overlap"""
    total = float(p["seconds"]); first = min(15.0, total); segs = [first]; have = first; step = 15.0 - W.GUIDE_FRAMES / W.FPS
    while have + 0.5 < total:
        nxt = max(5.0, min(15.0, total - have + W.GUIDE_FRAMES / W.FPS)); segs.append(round(nxt, 2)); have += nxt - W.GUIDE_FRAMES / W.FPS
    return segs


def wait_gpu(job):
    """returns when the GPU may be used; otherwise holds the job in 'waiting_gpu' with the reason (the page shows the button)"""
    while True:
        if job.get("cancel"): raise RuntimeError("cancelled")
        o = gpu_owner(fresh=True)
        if o["tf"]: GPU["note"] = f"needs the GPU — {C.TENANT_NAME or 'another GPU service'} is running"; GPU["ask"] = "tf"
        elif o["other"]: GPU["note"] = f"needs the GPU — {', '.join(o['other'])} is running"; GPU["ask"] = ""
        elif o["mem"] < NEED_GIB and not comfy_up(): GPU["note"] = f"needs ~{NEED_GIB} GiB free memory; {o['mem']} GiB free now"; GPU["ask"] = ""
        else:
            GPU["note"] = ""; GPU["ask"] = ""; return
        job["status"] = "waiting_gpu"; job["progress"]["phase"] = GPU["note"]; save(); time.sleep(4)


def wait_cool(job):
    while True:
        h = heat()
        if not ((h["board"] or 0) >= HOT_C or h["throttle"]): return
        job["progress"]["phase"] = f"cooling: board {h['board']:.0f} °C, GPU {h['gpu']} °C — resumes below {COOL_C:.0f} °C"; save()
        while True:
            if job.get("cancel"): raise RuntimeError("cancelled")
            time.sleep(15); h = heat()
            if (h["board"] or 0) < COOL_C and not h["throttle"]: break


def to_input(src, name):
    dst = os.path.join(COMFY_IN, name); shutil.copy2(src, dst); return name


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def is_cast(p): return bool(p.get("cast_images") or p.get("cast_voice"))


def subjects(p, voice):
    """H3's own way of naming references — <Picture N> are the images, <Audio 1> the voice"""
    ims = p.get("cast_images") or []
    if not ims and not voice: return ""
    pics = " and ".join(f"<Picture {i + 1}>" for i in range(len(ims)))
    return ("subject_definitions:\n" + (f"<Subject 1> is the person in {pics}; keep their face and identity exactly." if ims else "<Subject 1> is the speaker.")
            + (" <Subject 1> speaks with the voice in <Audio 1>, exactly as it sounds." if voice else "") + "\n\n")


def official(text):
    for a in ("The person from the reference image", "the person from the reference image"): text = text.replace(a, "<Subject 1>")
    return text


def cast_prompt(p, prompt):
    lines = (p.get("cast_lines") or "").strip()
    return subjects(p, p.get("cast_voice")) + official(prompt) + (f'\n\n<Subject 1> says, "{lines}"' if lines else "")


def cast_refs(p, tag):
    """the cast's images and voice copied into ComfyUI's input dir -> (image names, audio name or None)"""
    ims = [to_input(f"{ROOT}/{im}", f"studio_{tag}_face{i}{os.path.splitext(im)[1]}") for i, im in enumerate(p.get("cast_images") or [])]
    au = to_input(f"{VOICES}/{p['cast_voice']}/ref.wav", f"studio_{tag}_voice.wav") if p.get("cast_voice") else None
    return ims, au


def run_cast(job):
    """cast clips: each segment from text + the faces + the voice (no first frame, no guides — reference-to-video has neither); segments are cut together"""
    p = job["params"]; d = f"{LIB}/{job['id']}"; os.makedirs(d, exist_ok=True)
    segs = plan_segments(p) if job["kind"] == "chain" else [float(p["seconds"])]
    beats = [b.strip() for b in (p.get("beats") or "").splitlines() if b.strip()]
    ims, au = cast_refs(p, job["id"]); done = []
    for k, sec in enumerate(segs):
        wait_cool(job)
        job["progress"].update(seg=k + 1, segs=len(segs), step=0, steps=0, phase=f"cast segment {k + 1} of {len(segs)} · {sec:g} s"); save()
        prompt = cast_prompt(p, p["prompt"] if not beats else (p["prompt"] + "\n\nThis part: " + beats[min(k, len(beats) - 1)]))
        g, info = W.cast_graph(p["preset"], prompt, sec, int(p["seed"]) + k, f"studio/{job['id']}_s{k + 1}", ims, au, vae8=bool(p.get("vae8")))
        pid = comfy_post(g); job["progress"]["engine"] = info["engine"]; save()
        files = [f for f in comfy_wait(pid, job) if f.endswith(".mp4") and os.path.basename(f).startswith(f"{job['id']}_s{k + 1}_")]
        if not files: raise RuntimeError(f"segment {k + 1} produced no video")
        seg = f"{d}/seg{k + 1:02d}.mp4"; shutil.move(files[0], seg)
        def rerender(pr, bump, k=k, sec=sec):
            g2, _ = W.cast_graph(p["preset"], pr, sec, int(p["seed"]) + k + bump, f"studio/{job['id']}_s{k + 1}r", ims, au, vae8=bool(p.get("vae8")))
            fs = [f for f in comfy_wait(comfy_post(g2), job) if f.endswith(".mp4") and os.path.basename(f).startswith(f"{job['id']}_s{k + 1}r_")]
            if not fs: raise RuntimeError(f"segment {k + 1} re-take produced no video")
            out = f"{d}/seg{k + 1:02d}.retake.mp4"; shutil.move(fs[0], out); return out
        seg = directed(job, k, seg, prompt, rerender); done.append(seg)
        job["outputs"] = [{"file": os.path.basename(x), "kind": "segment"} for x in done]; save()
    if len(done) > 1:
        job["progress"]["phase"] = "cutting the segments together"; save()
        rc, log = stitch.join_keyframes(done, f"{d}/film.mp4")
        if rc: raise RuntimeError("join failed: " + log)
        main = f"{d}/film.mp4"
    else:
        main = done[0]
    main = add_score(job, main)
    vo = add_voiceover(job, main)
    if vo != main: job["outputs"].insert(0, {"file": os.path.basename(main), "kind": "without voice-over"}); main = vo
    stitch.poster(main, f"{d}/poster.jpg", at=min(1.5, max(0.2, stitch.probe(main)["dur"] / 3)))
    job["outputs"].insert(0, {"file": os.path.basename(main), "kind": "film" if len(done) > 1 else "video", "poster": "poster.jpg"})
    for f in os.listdir(COMFY_IN):
        if f.startswith(f"studio_{job['id']}_"): os.remove(os.path.join(COMFY_IN, f))


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def find_shots(src, limit, min_len=2.5, max_len=8.0, max_shots=12):      # 10-10: scene > 0.2 and an 8 s cap follow the original's rhythm
    """picture cuts (ffmpeg scene score > 0.3) within the first `limit` seconds -> [(start, end)]; short shots merged, long ones split"""
    dur = min(stitch.probe(src)["dur"], limit)
    r = subprocess.run(["ffmpeg", "-i", src, "-t", f"{dur:.2f}", "-an", "-filter:v", "select='gt(scene,0.2)',showinfo", "-f", "null", "-"], capture_output=True, text=True)
    cuts = sorted({round(float(m), 2) for m in re.findall(r"pts_time:([0-9.]+)", r.stderr) if 0.5 < float(m) < dur - 0.5})
    b = [0.0] + cuts + [round(dur, 2)]; shots = []
    for a, e in zip(b, b[1:]):
        if shots and (e - a < min_len or shots[-1][1] - shots[-1][0] < min_len): shots[-1] = (shots[-1][0], e)
        else: shots.append((a, e))
    out = []
    for a, e in shots:
        n = max(1, math.ceil((e - a) / max_len)); step = (e - a) / n
        out += [(round(a + i * step, 2), round(a + (i + 1) * step, 2)) for i in range(n)]
    return out[:max_shots]


def run_recreate(job):
    p = job["params"]; d = f"{LIB}/{job['id']}"; src_dir = f"{d}/source"; os.makedirs(src_dir, exist_ok=True)
    job["progress"].update(phase="downloading the Short, separating its speech, transcribing it"); save()
    vworker("capture", p["url"], src_dir, *(["1080"] if p.get("style") == "faithful" else []), t=2400)    # Faithful: capture up to 1080p (sharper anchor frames)
    meta = json.load(open(f"{src_dir}/transcript.json")); src = f"{src_dir}/source.mp4"
    _pr = stitch.probe(src); dims = W.shape_dims(p["preset"], _pr["w"], _pr["h"], p.get("shape", "auto")); p["dims"] = list(dims)
    shots = find_shots(src, float(p["max_seconds"]))
    if not shots: raise RuntimeError("no shots found in that video")
    words = meta.get("words") or []
    plan = []
    for i, (a, e) in enumerate(shots):
        clip = f"{src_dir}/shot{i + 1:02d}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.2f}", "-to", f"{e:.2f}", "-i", src, "-c:v", "libx264", "-crf", "20", "-c:a", "aac", clip], capture_output=True)
        said = "".join(w["w"] for w in words if a - 0.05 <= w["t0"] < e).strip() if p.get("keep_words", True) else ""
        job["progress"].update(seg=i + 1, segs=len(shots), phase=f"the director is watching shot {i + 1} of {len(shots)} ({e - a:.1f} s)"); save()
        dsc = DIR.describe_shot(clip, said, p.get("notes", ""), bool(p.get("cast_images")))
        plan.append({"start": a, "end": e, "dur": round(max(3.0, e - a), 2), "words": said, "prompt": W.compose_prompt(dsc["picture"], dsc["action"], dsc["sound"])})
    p.update(shots=plan, title=meta.get("title", ""), uploader=meta.get("uploader", ""), prompt=f"Remake of \"{meta.get('title', '')[:80]}\"" + (f" — {p['notes']}" if p.get("notes") else ""),
             seconds=round(sum(x["dur"] for x in plan), 2)); save()
    ims, au = cast_refs(p, job["id"]); done = []
    if p.get("style") == "faithful": done = faithful_shots(job, plan, src, d, dims, ims)
    for k, sh in enumerate(plan if p.get("style") != "faithful" else []):
        wait_cool(job)
        job["progress"].update(seg=k + 1, segs=len(plan), step=0, steps=0, phase=f"shot {k + 1} of {len(plan)} · {sh['dur']:g} s with your cast"); save()
        prompt = subjects(p, au and sh["words"]) + official(sh["prompt"]) + (f'\n\n<Subject 1> says, "{sh["words"]}"' if sh["words"] else "")
        g, info = W.cast_graph(p["preset"], prompt, sh["dur"], int(p["seed"]) + k, f"studio/{job['id']}_s{k + 1}", ims, au, vae8=bool(p.get("vae8")), dims=dims)
        pid = comfy_post(g); job["progress"]["engine"] = info["engine"]; save()
        files = [f for f in comfy_wait(pid, job) if f.endswith(".mp4") and os.path.basename(f).startswith(f"{job['id']}_s{k + 1}_")]
        if not files: raise RuntimeError(f"shot {k + 1} produced no video")
        seg = f"{d}/seg{k + 1:02d}.mp4"; shutil.move(files[0], seg)
        def rerender(pr, bump, k=k, sh=sh):
            g2, _ = W.cast_graph(p["preset"], pr, sh["dur"], int(p["seed"]) + k + bump, f"studio/{job['id']}_s{k + 1}r", ims, au, vae8=bool(p.get("vae8")), dims=dims)
            fs = [f for f in comfy_wait(comfy_post(g2), job) if f.endswith(".mp4") and os.path.basename(f).startswith(f"{job['id']}_s{k + 1}r_")]
            if not fs: raise RuntimeError(f"shot {k + 1} re-take produced no video")
            out = f"{d}/seg{k + 1:02d}.retake.mp4"; shutil.move(fs[0], out); return out
        seg = directed(job, k, seg, prompt, rerender); done.append(seg)
        job["outputs"] = [{"file": os.path.basename(x), "kind": "segment"} for x in done]; save()
    if p.get("soundtrack", "new") != "new" or p.get("style") == "faithful":          # faithful shots are already cut to the original's timing
        main = original_sound(job, plan, done, src_dir, d)
    else:
        if len(done) > 1:
            job["progress"]["phase"] = "cutting the shots together"; save()
            rc, log = stitch.join_keyframes(done, f"{d}/film.mp4")
            if rc: raise RuntimeError("join failed: " + log)
            main = f"{d}/film.mp4"
        else:
            main = done[0]
        main = add_score(job, main)
    stitch.poster(main, f"{d}/poster.jpg", at=min(1.5, max(0.2, stitch.probe(main)["dur"] / 3)))
    job["outputs"].insert(0, {"file": os.path.basename(main), "kind": "film" if len(done) > 1 else "video", "poster": "poster.jpg"})
    job["outputs"].append({"file": "source/source.mp4", "kind": "original Short"})


def faithful_shots(job, plan, src, d, dims, ims):
    """the remake follows the original frame by frame. The director picks out the performer once, and a clean studio portrait of the cast is
    made once. Inside every shot, anchor frames about 1.2 s apart are taken from the Short (the last one 3 frames before the cut); the director
    checks each one, and only where the performer's face is visible is the cast swapped in with the image editor (references: that frame + the
    portrait, never an earlier swap). H3 animates between neighbouring anchors (first + last frame) with a neutral one-continuous-shot prompt;
    each clip is retimed to the exact time between its anchors and nothing is animated across a cut.
    -> one video per shot (segNN.mp4), each the shot's own length"""
    p = job["params"]; Wd, Hd = dims; fd = f"{d}/faithful"; os.makedirs(fd, exist_ok=True)
    anchors = []
    for sh in plan:
        S = sh["end"] - sh["start"]; g = max(1, round(S / 1.2))               # ~1.2 s apart: less room for H3 to invent a face between anchors
        anchors.append([sh["start"] + j * S / g for j in range(g)] + [sh["end"] - 3.0 / W.FPS])     # 3 frames in: a seek at the very end can return nothing
    total = sum(len(t) for t in anchors); n = 0; edited = []
    job["progress"].update(phase="the director is picking out the performer; making a clean portrait of your cast"); save()
    who = DIR.who_is_performer(src); p["performer"] = who
    pid = comfy_post(W.edit_graph(ims[:3], "Make a clean studio portrait of this exact person: the same face, facial features, skin tone, hair and beard, "
                                  "unchanged. Head and shoulders, looking straight at the camera, neutral expression, plain light grey background, soft even "
                                  "light, bare head, an empty plain background around them.", int(p["seed"]) + 3, f"studio/{job['id']}_card"))
    fs = [f for f in comfy_wait(pid, job) if f.lower().endswith(".png") and os.path.basename(f).startswith(f"{job['id']}_card")]
    if not fs: raise RuntimeError("could not make the cast portrait")
    card_png = f"{fd}/cast_portrait.png"; shutil.move(fs[0], card_png); card = to_input(card_png, f"studio_{job['id']}_card.png"); swapped = 0
    for i, ts in enumerate(anchors):                                            # 1) the cast into every anchor frame
        row = []
        for j, t in enumerate(ts):
            n += 1; wait_cool(job)
            job["progress"].update(seg=n, segs=total, step=0, steps=0, phase=f"putting your cast into frame {n} of {total}"); save()
            raw = f"{fd}/a{i + 1:02d}_{j:02d}_orig.png"                         # twice the video size: the editor gets 4x the pixels for the face
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1", "-vf",
                            f"scale={2 * Wd}:{2 * Hd}:force_original_aspect_ratio=increase:flags=lanczos,crop={2 * Wd}:{2 * Hd}", raw], capture_output=True)
            if not os.path.exists(raw):                                         # exact (slower) seek as the fallback
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ss", f"{t:.3f}", "-frames:v", "1", "-vf",
                                f"scale={2 * Wd}:{2 * Hd}:force_original_aspect_ratio=increase:flags=lanczos,crop={2 * Wd}:{2 * Hd}", raw], capture_output=True)
            if not os.path.exists(raw): raise RuntimeError(f"could not read the Short's frame at {t:.2f} s")
            out = f"{fd}/a{i + 1:02d}_{j:02d}.png"
            try: c = DIR.check_frame(raw, who)
            except Exception: c = {"present": False, "face_visible": False}
            if not (c["present"] and c["face_visible"]):                      # the performer's face is not in this frame: keep it as it was
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-vf", f"scale={Wd}:{Hd}:flags=lanczos", out], capture_output=True); row.append(out); continue
            mk = c["makeup"]; mk = "" if mk.lower().startswith(("none", "no ")) else f" Keep the {mk} on the new face, exactly as in image 1."
            pr = (f"In image 1, replace only {who} ({c['where']}) with the person in image 2: give them that person's face, facial features, face shape, "
                  f"beard and identity. Keep their facial expression ({c['expression']}), head angle and gaze exactly as in image 1.{mk} Everyone else, "
                  "their costume, pose, hands, framing, lighting, colours and background stay exactly as in image 1.")   # said positively: no negative prompt
            pid = comfy_post(W.edit_graph([to_input(raw, f"studio_{job['id']}_f{n:03d}.png"), card], pr, int(p["seed"]) + 7, f"studio/{job['id']}_f{n:03d}"))
            files = [f for f in comfy_wait(pid, job) if f.lower().endswith(".png") and os.path.basename(f).startswith(f"{job['id']}_f{n:03d}")]
            if not files: raise RuntimeError(f"frame {n}: the cast swap produced no image")
            big = f"{fd}/a{i + 1:02d}_{j:02d}_swap2x.png"; shutil.move(files[0], big)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", big, "-vf", f"scale={Wd}:{Hd}:flags=lanczos", out], capture_output=True)
            if not os.path.exists(out): raise RuntimeError(f"frame {n}: could not scale the swapped frame")
            row.append(out); swapped += 1
        edited.append(row)
    p["faithful_frames"] = {"anchors": total, "swapped": swapped}
    shots = []; m = 0; tot = sum(len(t) - 1 for t in anchors)
    for i, (ts, row, sh) in enumerate(zip(anchors, edited, plan)):             # 2) H3 animates between neighbouring anchors
        clips = []
        for j in range(len(ts) - 1):
            m += 1; wait_cool(job); G = ts[j + 1] - ts[j]
            job["progress"].update(seg=m, segs=tot, step=0, steps=0, phase=f"animating {m} of {tot} · shot {i + 1} of {len(plan)}"); save()
            a = to_input(row[j], f"studio_{job['id']}_k{m:03d}a.png"); b = to_input(row[j + 1], f"studio_{job['id']}_k{m:03d}b.png")
            cp = ("One continuous, unbroken shot. Keep exactly the scene, setting, people, costumes, makeup, framing and lighting of the first and last "   # said positively:
                  "frames; everything moves naturally and smoothly from the first frame to the last. Only the people, places and objects of the first "  # the video model has no
                  "and last frames appear.")                                                                                                            # negative prompt
            if p.get("soundtrack") == "new": cp += "\n\n" + sh["prompt"]                # only the remake's own sound needs the director's sound line
            g, info = W.video_graph(p["preset"], cp, G, int(p["seed"]) + m, f"studio/{job['id']}_c{m:03d}", first_image=a, last_image=b,
                                    engine=p.get("engine", "comfy"), vae8=bool(p.get("vae8")), dims=dims)
            pid = comfy_post(g); job["progress"]["engine"] = info["engine"]; save()
            fs = [f for f in comfy_wait(pid, job) if f.endswith(".mp4") and os.path.basename(f).startswith(f"{job['id']}_c{m:03d}_")]
            if not fs: raise RuntimeError(f"clip {m} produced no video")
            raw = f"{fd}/c{m:03d}_raw.mp4"; shutil.move(fs[0], raw)
            L = stitch.probe(raw)["frames"] or W.frames(G); k = G * W.FPS / max(1, L - 1)        # its (L-1) frame steps -> exactly G seconds
            clip = f"{fd}/c{m:03d}.mp4"
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-filter_complex",
                                f"[0:v]setpts=PTS*{k:.6f},fps={W.FPS},trim=0:{G:.4f},setpts=PTS-STARTPTS[v];"
                                f"[0:a]atempo={max(0.5, min(2.0, 1 / k)):.6f},atrim=0:{G:.4f},asetpts=PTS-STARTPTS[a]",
                                "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", clip],
                               capture_output=True, text=True)
            if r.returncode: raise RuntimeError("retiming a clip failed: " + (r.stderr or "")[-300:])
            clips.append(clip)
        shot = f"{d}/seg{i + 1:02d}.mp4"; ins = sum((["-i", c] for c in clips), []); nc = len(clips)
        S = sh["end"] - sh["start"]                                              # the shot's exact length: hold the last frame for the few missing frames
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", "".join(f"[{x}:v][{x}:a]" for x in range(nc)) + f"concat=n={nc}:v=1:a=1[cv][ca];"
                            f"[cv]tpad=stop_mode=clone:stop_duration=0.5,trim=0:{S:.4f},setpts=PTS-STARTPTS[v];[ca]apad,atrim=0:{S:.4f}[a]",
                            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", shot],
                           capture_output=True, text=True)
        if r.returncode: raise RuntimeError(f"joining shot {i + 1} failed: " + (r.stderr or "")[-300:])
        shots.append(shot); job["outputs"] = [{"file": os.path.basename(x), "kind": "segment"} for x in shots]; save()
    return shots


def original_sound(job, plan, done, src_dir, d):
    """the remake on the Short's own sound. Every shot is cut to its original length (hard cuts on the original's cut
    points, so picture and music stay in step); 'original' lays the Short's audio under it as it was, 'music' lays the Short's music
    (Demucs: speech and singing removed) under the remake's own voices (Demucs: H3's music and effects removed).
    -> film.mp4; the same cut with the remake's own sound is kept as film_newsound.mp4"""
    p = job["params"]; mode = p["soundtrack"]; src = f"{src_dir}/source.mp4"; n = len(done)
    job["progress"].update(phase="cutting the shots to the original's timing"); save()
    lens = [min(sh["end"] - sh["start"], stitch.probe(seg)["dur"]) for seg, sh in zip(done, plan)]
    ins, parts = [], []
    for i, seg in enumerate(done):
        ins += ["-i", seg]
        parts.append(f"[{i}:v]trim=0:{lens[i]:.4f},setpts=PTS-STARTPTS[v{i}];[{i}:a]atrim=0:{lens[i]:.4f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d=0.01,afade=t=out:st={max(0, lens[i] - 0.01):.4f}:d=0.01[a{i}]")
        parts.append(f"[{n}:a]atrim={plan[i]['start']:.4f}:{plan[i]['start'] + lens[i]:.4f},asetpts=PTS-STARTPTS[o{i}]")
    graph = ";".join(parts + ["".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]",
                              "".join(f"[o{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[o]"])
    cut, orig = f"{d}/film_newsound.mp4", f"{d}/original_sound.wav"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-i", src, "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
                        "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", cut,
                        "-map", "[o]", "-ac", "2", "-ar", "44100", orig], capture_output=True, text=True)
    if r.returncode: raise RuntimeError("cutting to the original's timing failed: " + (r.stderr or "")[-300:])
    out = f"{d}/film.mp4"
    if mode == "new":                                                           # faithful + the remake's own sound
        os.replace(cut, out); return out
    if mode == "original":
        graph, extra = "[1:a]apad[a]", ["-i", orig]
    else:
        job["progress"].update(phase="separating the original's music from its voices, and the remake's voices from its music"); save()
        try:                                                                    # the renders are done: ComfyUI gives its memory back for Demucs
            urllib.request.urlopen(urllib.request.Request(f"{COMFY}/free", data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                                          headers={"Content-Type": "application/json"}), timeout=60).read(); time.sleep(3)
        except Exception: pass
        vworker("stems", orig, f"{d}/stems_original", t=900); vworker("stems", cut, f"{d}/stems_remake", t=900)
        extra = ["-i", f"{d}/stems_original/music.wav", "-i", f"{d}/stems_remake/vocals.wav"]
        graph = "[1:a]volume=0.9[m];[2:a]volume=1.0[s];[m][s]amix=inputs=2:duration=first:normalize=0,apad[a]"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", cut, *extra, "-filter_complex", graph, "-map", "0:v", "-map", "[a]", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", out], capture_output=True, text=True)
    if r.returncode: raise RuntimeError("laying the original sound failed: " + (r.stderr or "")[-300:])
    job["outputs"].insert(0, {"file": "film_newsound.mp4", "kind": "with the remake's own sound"})
    return out
    for f in os.listdir(COMFY_IN):
        if f.startswith(f"studio_{job['id']}_"): os.remove(os.path.join(COMFY_IN, f))


def run_video(job):
    p = job["params"]; d = f"{LIB}/{job['id']}"; os.makedirs(d, exist_ok=True)
    if is_cast(p): return run_cast(job)
    segs = plan_segments(p) if job["kind"] == "chain" else [float(p["seconds"])]
    beats = [b.strip() for b in (p.get("beats") or "").splitlines() if b.strip()]
    first_image = to_input(p["first_image_path"], f"studio_{job['id']}_first{os.path.splitext(p['first_image_path'])[1]}") if p.get("first_image_path") else None
    done = []; prev_in = None; prev_frames = 0
    for k, sec in enumerate(segs):
        wait_cool(job)
        job["progress"].update(seg=k + 1, segs=len(segs), step=0, steps=0, phase=f"segment {k + 1} of {len(segs)} · {sec:g} s"); save()
        prompt = p["prompt"] if not beats else (p["prompt"] + "\n\nThis part: " + beats[min(k, len(beats) - 1)])
        graph, info = vgraph(p, prompt, sec, int(p["seed"]) + k, f"studio/{job['id']}_s{k + 1}",
                                    first_image=first_image if k == 0 else None, guide_video=prev_in, force_h3=job["kind"] == "chain")
        if prev_in:
            graph["33"]["inputs"]["batch_index"] = max(0, prev_frames - W.GUIDE_FRAMES)
            graph["34"]["inputs"]["start_index"] = max(0.0, (prev_frames - W.GUIDE_FRAMES) / W.FPS)
        pid = comfy_post(graph); job["progress"]["engine"] = info["engine"]; save()
        files = [f for f in comfy_wait(pid, job) if f.endswith(".mp4") and os.path.basename(f).startswith(f"{job['id']}_s{k + 1}_")]
        if not files: raise RuntimeError(f"segment {k + 1} produced no video")
        seg = f"{d}/seg{k + 1:02d}.mp4"; shutil.move(files[0], seg)
        def rerender(pr, bump, k=k, sec=sec, prev=prev_in, pf=prev_frames):
            g2, _ = vgraph(p, pr, sec, int(p["seed"]) + k + bump, f"studio/{job['id']}_s{k + 1}r",
                                  first_image=first_image if k == 0 else None, guide_video=prev, force_h3=job["kind"] == "chain")
            if prev:
                g2["33"]["inputs"]["batch_index"] = max(0, pf - W.GUIDE_FRAMES); g2["34"]["inputs"]["start_index"] = max(0.0, (pf - W.GUIDE_FRAMES) / W.FPS)
            fs = [f for f in comfy_wait(comfy_post(g2), job) if f.endswith(".mp4") and os.path.basename(f).startswith(f"{job['id']}_s{k + 1}r_")]
            if not fs: raise RuntimeError(f"segment {k + 1} re-take produced no video")
            out = f"{d}/seg{k + 1:02d}.retake.mp4"; shutil.move(fs[0], out); return out
        seg = directed(job, k, seg, prompt, rerender); done.append(seg)
        prev_in = to_input(seg, f"studio_{job['id']}_seg{k + 1:02d}.mp4"); prev_frames = stitch.probe(seg)["frames"] or info["frames"]
        job["outputs"] = [{"file": os.path.basename(x), "kind": "segment"} for x in done]; save()
    if len(done) > 1:
        job["progress"]["phase"] = "joining the segments"; save()
        rc, log = stitch.join(done, f"{d}/film.mp4", guide_frames=W.GUIDE_FRAMES)
        if rc: raise RuntimeError("join failed: " + log)
        main = f"{d}/film.mp4"
    else:
        main = done[0]
    main = add_score(job, main)
    vo = add_voiceover(job, main)
    if vo != main: job["outputs"].insert(0, {"file": os.path.basename(main), "kind": "without voice-over"}); main = vo
    stitch.poster(main, f"{d}/poster.jpg", at=min(1.5, max(0.2, stitch.probe(main)["dur"] / 3)))
    job["outputs"].insert(0, {"file": os.path.basename(main), "kind": "film" if len(done) > 1 else "video", "poster": "poster.jpg"})
    for f in os.listdir(COMFY_IN):
        if f.startswith(f"studio_{job['id']}_"): os.remove(os.path.join(COMFY_IN, f))


def shot_check(job, k, seg, prompt, first=None, last=None):
    """frame-level check (shot.py) of one take — a hard cut inside a one-shot clip, or a keyframe the clip did not open/close on.
    Never fails a job; a check that cannot run says so. The web app itself needs only the standard library: when this python has no
    numpy/Pillow, the check runs in the voice environment (which has both)."""
    single = SHOT.one_shot(prompt)
    try:
        try:
            import numpy, PIL  # noqa: F401
            r = SHOT.check(seg, first, last, single=single)
        except ImportError:
            code = ("import json, sys; sys.path.insert(0, sys.argv[1]); import shot; "
                    "print(json.dumps(shot.check(sys.argv[2], sys.argv[3] or None, sys.argv[4] or None, single=sys.argv[5] == '1')))")
            o = subprocess.run([TTS_PY, "-c", code, HERE, seg, first or "", last or "", "1" if single else "0"], capture_output=True, text=True, timeout=300)
            if o.returncode: raise RuntimeError((o.stderr or o.stdout or "no output").strip()[-200:])
            r = json.loads(o.stdout.strip().splitlines()[-1])
    except Exception as e: r = {"passed": True, "note": f"shot check not run: {type(e).__name__}: {str(e)[:120]}"}
    job.setdefault("shots", {})[str(k + 1)] = {x: r.get(x) for x in ("passed", "note", "cuts", "first", "last", "window")}; save(); return r


def directed(job, k, seg, prompt, rerender, first=None, last=None):
    """the director watches segment k; with review='retake' a weak take (score <= 5 or off-prompt) is re-rendered once with its fix, and the
    better-scored take is kept. rerender(prompt, seed_bump) -> path of the new take.
    Every take is also frame-checked (cut inside a one-shot clip / keyframe not held); a failed check triggers the same
    single re-take, and the take that passes the frame check wins before the director's score is compared."""
    mode = job["params"].get("review", "retake")
    sc = shot_check(job, k, seg, prompt, first, last)
    if mode == "off":
        return seg
    job["progress"]["phase"] = f"director is watching segment {k + 1}"; save()
    try: rv = DIR.review(seg, prompt)
    except Exception as e:
        job.setdefault("reviews", {})[str(k + 1)] = {"error": f"review skipped: {type(e).__name__}: {str(e)[:160]}"}; save(); return seg
    rv["take"] = 1; job.setdefault("reviews", {})[str(k + 1)] = rv; save()
    weak = (rv["score"] <= 5 or not rv.get("matches_prompt", True)) and rv.get("fix")
    if mode != "retake" or not (weak or not sc["passed"]): return seg
    fix = rv.get("fix") if weak else ""
    if not sc["passed"]: fix = (fix + " " if fix else "") + "One continuous, unbroken shot that holds its first and last frames."
    job["progress"]["phase"] = f"segment {k + 1}: re-take (" + (f"director scored {rv['score']}/10" if weak else sc.get("note", "frame check")[:60]) + ")"; save()
    new = rerender(prompt + "\n\nDirector's note for this take: " + fix, 100)
    try: rv2 = DIR.review(new, prompt)
    except Exception: rv2 = {"score": 0}
    sc2 = shot_check(job, k, new, prompt, first, last)
    rv2["take"] = 2; rv2["first"] = rv
    if (sc2["passed"], rv2.get("score", 0)) > (sc["passed"], rv["score"]):
        os.replace(new, seg); rv2["kept"] = "re-take"; job["shots"][str(k + 1)]["kept"] = "re-take"
    else:
        os.remove(new); rv2["kept"] = "first take"; job["shots"][str(k + 1)] = {**{x: sc.get(x) for x in ("passed", "note", "cuts", "first", "last", "window")}, "kept": "first take"}
    job["reviews"][str(k + 1)] = rv2; save(); return seg


def run_keyframes(job):
    """images 1..n in order; clip i runs from image i to image i+1 (H3 first-and-last frame, with sound); the clips are joined."""
    p = job["params"]; d = f"{LIB}/{job['id']}"; os.makedirs(d, exist_ok=True)
    names = [to_input(f"{ROOT}/{im}", f"studio_{job['id']}_k{i + 1:02d}{os.path.splitext(im)[1]}") for i, im in enumerate(p["images"])]
    lines = [x.strip() for x in (p.get("beats") or "").splitlines() if x.strip()]
    done = []
    for k, sec in enumerate(p["gaps"]):
        wait_cool(job)
        job["progress"].update(seg=k + 1, segs=len(p["gaps"]), step=0, steps=0, phase=f"clip {k + 1} of {len(p['gaps'])}: image {k + 1} → {k + 2} · {sec:g} s"); save()
        prompt = p["prompt"] + (("\n\nThis part: " + lines[min(k, len(lines) - 1)]) if lines else "")
        graph, info = vgraph(p, prompt, sec, int(p["seed"]) + k, f"studio/{job['id']}_s{k + 1}", first_image=names[k], last_image=names[k + 1])
        pid = comfy_post(graph); job["progress"]["engine"] = info["engine"]; save()
        files = [f for f in comfy_wait(pid, job) if f.endswith(".mp4") and os.path.basename(f).startswith(f"{job['id']}_s{k + 1}_")]
        if not files: raise RuntimeError(f"clip {k + 1} produced no video")
        seg = f"{d}/seg{k + 1:02d}.mp4"; shutil.move(files[0], seg)
        def rerender(pr, bump, k=k, sec=sec):
            g2, _ = vgraph(p, pr, sec, int(p["seed"]) + k + bump, f"studio/{job['id']}_s{k + 1}r", first_image=names[k], last_image=names[k + 1])
            fs = [f for f in comfy_wait(comfy_post(g2), job) if f.endswith(".mp4") and os.path.basename(f).startswith(f"{job['id']}_s{k + 1}r_")]
            if not fs: raise RuntimeError(f"clip {k + 1} re-take produced no video")
            out = f"{d}/seg{k + 1:02d}.retake.mp4"; shutil.move(fs[0], out); return out
        seg = directed(job, k, seg, prompt, rerender, first=f"{ROOT}/{p['images'][k]}", last=f"{ROOT}/{p['images'][k + 1]}"); done.append(seg)   # keyframes for the shot check
        job["outputs"] = [{"file": os.path.basename(x), "kind": "segment"} for x in done]; save()
    job["progress"]["phase"] = "joining the clips"; save()
    main = f"{d}/film.mp4"; rc, log = stitch.join_keyframes(done, main)
    if rc: raise RuntimeError("join failed: " + log)
    main = add_score(job, main)
    vo = add_voiceover(job, main)
    if vo != main: job["outputs"].insert(0, {"file": os.path.basename(main), "kind": "without voice-over"}); main = vo
    stitch.poster(main, f"{d}/poster.jpg", at=min(1.5, max(0.2, stitch.probe(main)["dur"] / 3)))
    job["outputs"].insert(0, {"file": os.path.basename(main), "kind": "film", "poster": "poster.jpg"})
    for f in os.listdir(COMFY_IN):
        if f.startswith(f"studio_{job['id']}_"): os.remove(os.path.join(COMFY_IN, f))


def vworker(*args, t=1800):
    """run voice.py in the voice environment; returns its JSON line or raises with its reason"""
    r = subprocess.run([TTS_PY, VOICE_PY, *args], capture_output=True, text=True, timeout=t)
    line = next((l for l in reversed((r.stdout or "").splitlines()) if l.startswith("{")), "")
    try: d = json.loads(line)
    except ValueError: d = {"ok": False, "error": (r.stderr or r.stdout or "no output")[-400:]}
    if not d.get("ok"): raise RuntimeError(d.get("error") or "voice worker failed")
    return d


def run_voice(job):
    """a voice from a Short: download, remove music, transcribe; the director watches it and picks a clean stretch of the wanted
    speaker; that stretch becomes the reference, and a preview line is spoken in the cloned voice"""
    p = job["params"]; vid = job["id"]; d = f"{VOICES}/{vid}"; os.makedirs(d, exist_ok=True)
    job["progress"].update(phase="downloading the video and removing the music (Demucs), then transcribing (Whisper)"); save()
    cap = vworker("capture", p["url"], d, t=2400)
    meta = json.load(open(f"{d}/transcript.json"))
    if not meta["segments"]: raise RuntimeError("no speech was found in that video")
    if p.get("multi"): return run_voice_multi(job, d, cap, meta)                    # everyone who speaks
    job["progress"].update(phase=f"the director is watching \"{cap.get('title', '')[:60]}\" to find {p.get('who') or 'the main speaker'}"); save()
    pick = DIR.pick_voice(f"{d}/source.mp4", meta["segments"], p.get("who", ""), meta["duration"])
    job["progress"].update(phase=f"cutting {pick['start']:.1f}–{pick['end']:.1f} s and cloning a preview"); save()
    cut = vworker("cut", d, str(pick["start"]), str(pick["end"]))
    vworker("say", d, p.get("preview") or "Hello. This is a preview of the matched voice, ready for your next film.", f"{d}/preview.wav", t=900)
    v = {"id": vid, "label": p.get("label") or meta["title"][:60] or "voice", "url": p["url"], "title": meta["title"], "uploader": meta.get("uploader"),
         "who": p.get("who", ""), "pick": pick, "ref_text": cut["ref_text"], "consent": True, "created": now()}
    json.dump(v, open(f"{d}/voice.json", "w"), indent=1)
    job["outputs"] = [{"file": "preview.wav", "kind": "voice"}, {"file": "ref.wav", "kind": "reference"}]; job["voice"] = v


# ── the Trash: deleted renders and voices wait here until restored or emptied ──
def _trash_put(kind, iid, src, record):
    tid = f"{kind}-{iid}"; d = f"{TRASH}/{tid}"
    if os.path.exists(d): d = f"{d}-{int(time.time())}"; tid = os.path.basename(d)
    os.makedirs(d); json.dump({"kind": kind, "id": iid, "trashed": now(), "record": record}, open(f"{d}/trash.json", "w"), indent=1)
    if os.path.exists(src): shutil.move(src, f"{d}/data")
    return {"ok": True, "trash_id": tid}


def trash_job(j):
    with LOCK: JOBS.pop(j["id"], None)
    save(); return _trash_put("job", j["id"], f"{LIB}/{j['id']}", j)


def trash_voice(vid):
    v = {}
    try: v = json.load(open(f"{VOICES}/{vid}/voice.json"))
    except (OSError, ValueError): pass
    return _trash_put("voice", vid, f"{VOICES}/{vid}", v)


def trash_list():
    out = []
    for tid in sorted(os.listdir(TRASH), reverse=True):
        try: t = json.load(open(f"{TRASH}/{tid}/trash.json"))
        except (OSError, ValueError): continue
        r = t.get("record") or {}; p = r.get("params") or {}
        label = r.get("label") if t["kind"] == "voice" else (p.get("prompt") or p.get("style") or r.get("kind") or "")
        out.append({"trash_id": tid, "kind": t["kind"], "id": t["id"], "trashed": t["trashed"], "label": str(label)[:90], "job_kind": r.get("kind")})
    return out


def trash_restore(tid):
    if not re.match(r"^[\w-]+$", tid) or not os.path.isfile(f"{TRASH}/{tid}/trash.json"): raise ValueError("no such item in the trash")
    t = json.load(open(f"{TRASH}/{tid}/trash.json")); dst = f"{LIB}/{t['id']}" if t["kind"] == "job" else f"{VOICES}/{t['id']}"
    if os.path.exists(dst): raise ValueError("something with that id already exists")
    if os.path.exists(f"{TRASH}/{tid}/data"): shutil.move(f"{TRASH}/{tid}/data", dst)
    if t["kind"] == "job":
        with LOCK: JOBS[t["id"]] = t["record"]
        save()
    shutil.rmtree(f"{TRASH}/{tid}"); return {"ok": True, "kind": t["kind"], "id": t["id"]}


def trash_empty(tid=None):
    """permanent: the one place the studio deletes files, and only on the user's explicit Empty (one item or all)"""
    ids = [tid] if tid else os.listdir(TRASH); n = 0
    for x in ids:
        if re.match(r"^[\w-]+$", str(x)) and os.path.isdir(f"{TRASH}/{x}"): shutil.rmtree(f"{TRASH}/{x}"); n += 1
    return {"ok": True, "removed": n}


def run_voice_multi(job, d, cap, meta):
    """everyone who speaks in the video becomes a voice of their own. voice.py `speakers` groups the phrases by the
    clone model's own speaker fingerprints and picks each person's longest clean solo stretch; the director names each from frames taken
    while only they talk; each gets voices/<job>-sN/ (ref.wav, ref.json, preview.wav, voice.json). The capture stays in voices/<job>/
    (source, stems, transcript, speakers.json) and is not itself a voice."""
    p = job["params"]; vid = job["id"]
    job["progress"].update(phase="telling the voices apart (Qwen3-TTS speaker fingerprints)"); save()
    vworker("speakers", d, str(int(p.get("max_speakers") or 6)), str(int(p.get("speakers") or 0)), t=1200)
    sp = json.load(open(f"{d}/speakers.json")); made, skipped = [], []
    for s in sp["speakers"]:
        n = s["n"]; tag = f"speaker {n} of {sp['k']}"
        if not s["enough"]: skipped.append(f"{tag}: only {s['best']['seconds']:.0f} s of clean speech (needs 5 s)"); continue
        vd = f"{VOICES}/{vid}-s{n}"; os.makedirs(vd, exist_ok=True); b = s["best"]
        job["progress"].update(seg=n, segs=sp["k"], phase=f"{tag}: the director looks at who is talking"); save()
        frames = []
        for q, (pa, pb) in zip((0.2, 0.5, 0.8), (b["parts"] * 3)[:3]):          # frames while only this speaker talks
            fp = f"{vd}/look{int(q * 10)}.jpg"; t = pa + q * (pb - pa)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", f"{d}/source.mp4", "-frames:v", "1", "-vf", "scale=640:-2", fp], capture_output=True)
            if os.path.exists(fp): frames.append(fp)
        try: who = DIR.name_speaker(frames, b["text"], meta.get("title", ""))
        except Exception as e: who = {"label": "", "name": "", "looks": "", "error": f"{type(e).__name__}: {str(e)[:100]}"}
        job["progress"].update(phase=f"{tag}: cutting {b['seconds']:.1f} s of their speech ({len(b['parts'])} part(s)) and cloning a preview"); save()
        cut = vworker("cutparts", d, vd, json.dumps(b["parts"]))
        vworker("say", vd, p.get("preview") or "Hello. This is a preview of the matched voice, ready for your next film.", f"{vd}/preview.wav", "2", t=900)
        label = who.get("name") or who.get("label") or f"Speaker {n}"
        if p.get("label"): label = f"{p['label']} — {label}"
        v = {"id": f"{vid}-s{n}", "label": label[:60], "url": p["url"], "title": meta["title"], "uploader": meta.get("uploader"), "who": who.get("looks", ""),
             "pick": {"start": b["start"], "end": b["end"], "speaker": who.get("looks") or tag, "by": "voice fingerprints (Qwen3-TTS) + director",
                      "reason": f"{tag} · {s['seconds']:.0f} s of speech · reference {b['seconds']:.1f} s" + (f" joined from {len(b['parts'])} turns" if len(b['parts']) > 1 else "")}, "ref_text": cut["ref_text"], "consent": True,
             "created": now(), "group": vid, "speaker": n}
        json.dump(v, open(f"{vd}/voice.json", "w"), indent=1); made.append(v)
    if not made: raise RuntimeError("no speaker had a clean 6 s stretch on their own — " + "; ".join(skipped))
    job["voices"] = made; job["skipped"] = skipped
    job["outputs"] = []                                                       # the voices live on the Voices tab (voices/<job>-sN/)
    job["notes"] = f"{len(made)} voice(s) from {sp['k']} speaker(s)" + (" · skipped: " + "; ".join(skipped) if skipped else "")


def voices():
    out = []
    for vid in sorted(os.listdir(VOICES), reverse=True):
        try: out.append(json.load(open(f"{VOICES}/{vid}/voice.json")))
        except Exception: pass
    return out


def add_voiceover(job, main):
    """speak the job's voice-over lines in its cloned voice and lay them over the film: the film's own sound ducks under the voice;
    a voice-over longer than the film holds the last frame"""
    p = job["params"]; vid = p.get("voice"); text = (p.get("vo") or "").strip()
    if not vid or not text: return main
    vd = f"{VOICES}/{vid}"
    if not os.path.exists(f"{vd}/ref.json"): raise RuntimeError("that voice is gone from the Voices tab")
    d = os.path.dirname(main); vo = f"{d}/voiceover.wav"
    if p.get("vo_engine") == "drama3" and drama3_voiceover(job, vd, vid, text, vo): pass   # optional Drama 3: server → per-take → Qwen3-TTS
    else:
        job["progress"].update(phase="speaking the voice-over in the cloned voice (Qwen3-TTS)"); save()
        r = vworker("say", vd, text, vo, t=900); job["vo_check"] = r.get("wordcheck")
        if p.get("vo_engine") == "drama3": job["vo_engine"] = "Qwen3-TTS (Drama 3 fallback — reason in the notes)"
    vdur, fdur = stitch.probe(vo)["dur"] or 0, stitch.probe(main)["dur"]; lead = 0.4; extra = max(0.0, lead + vdur + 0.3 - fdur)
    out = f"{d}/film_vo.mp4"
    graph = (f"[0:v]tpad=stop_mode=clone:stop_duration={extra:.2f}[v];[1:a]adelay={int(lead * 1000)}|{int(lead * 1000)},apad[vo];"
             f"[vo]asplit=2[vo1][vo2];[0:a]apad=pad_dur={extra:.2f}[bed];[bed][vo1]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[duck];"
             f"[duck][vo2]amix=inputs=2:duration=first:normalize=0[a]")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", main, "-i", vo, "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "14",
                        "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", out], capture_output=True, text=True)
    if r.returncode: raise RuntimeError("voice-over mix failed: " + (r.stderr or "")[-300:])
    return out


# ── optional: Fish Audio Drama 3 — the fast speech server, then the per-take build ───────────────────────────────────────────────────
# Order for a Drama 3 voice-over: weights (the optional fetch hook DRAMA3_FETCH_CMD, if they are missing) → ComfyUI /free → voice file
# (voice3, ~2.6 GB, before the server loads) → the speech server (DRAMA3_SERVICE: drama-tensorfold's guarded pilot.py, needs ~88 GiB free
# to load, capped at 54 GiB) → say3s. Fallbacks never fail the film: server path fails → the per-take say3 build; that fails too (or no
# weights) → Qwen3-TTS, with the reason in the job notes. The unit is stopped DRAMA3_IDLE_S after the last voice-over.
# Built with Fish Audio (Fish Audio Community License v1.0: loopback only; clone only voices with consent).
D3S = {"last": 0.0, "busy": False}


def _d3_note(job, msg):
    job["vo_note"] = (job.get("vo_note", "") + " · " if job.get("vo_note") else "") + msg
    job["notes"] = ((job.get("notes") or "") + " · " if job.get("notes") else "") + msg


def d3_fetch(job):
    """the checkpoint, fetched by the optional fetch hook (DRAMA3_FETCH_CMD) when it is missing -> (ok, note). Never raises except on cancel."""
    if DR3.local(): return True, ""
    if not DR3.FETCH_CMD: return False, "not installed (no weights in DRAMA3_MODEL; see docs/DRAMA3.md)"
    ph = f"fetching Fish Audio Drama 3 ({DR3.GIB_ON_DISK} GB, the optional fetch hook)"; job["progress"].update(phase=ph); save(); t0 = time.time()
    os.makedirs(DR3.WORK, exist_ok=True); log = f"{DR3.WORK}/fetch.log"
    try:
        with open(log, "a") as lf:
            lf.write(f"\n== {now()} DRAMA3_FETCH_CMD\n"); lf.flush()
            pr = subprocess.Popen(["/bin/bash", "-c", DR3.FETCH_CMD], stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  env={**os.environ, "DRAMA3_MODEL": DR3.CKPT})
            while pr.poll() is None:
                if job.get("cancel"): pr.terminate(); raise RuntimeError("cancelled")
                if time.time() - t0 > 3 * 3600: pr.terminate(); return False, "the fetch hook did not finish in 3 hours"
                time.sleep(5)
    except RuntimeError: raise
    except Exception as e: return False, f"the fetch hook could not run ({type(e).__name__}: {str(e)[:120]})"
    if DR3.local(): return True, "fetched by the optional fetch hook"
    return False, f"the fetch hook exited {pr.returncode} without a complete checkpoint (see {log})"


def d3_health():
    try:
        with urllib.request.urlopen(f"{DR3.URL}/health", timeout=3) as r: return r.status == 200
    except Exception: return False
def d3_active():
    if not DR3.UNIT: return d3_health()
    return sh(["systemctl", "--user", "is-active", DR3.UNIT], 8) in ("active", "activating", "reloading")


def d3_server_up(job):
    """start the speech server unit if needed and wait until /health answers -> (ok, note). It needs ~88 GiB free; the broker frees
    the studio's own holders only (ComfyUI, LocalRouter, the director) — never other services."""
    if d3_health(): D3S["last"] = time.time(); return True, ""
    if not DR3.UNIT: return False, f"no speech server answers on port {DR3.PORT} and DRAMA3_SERVICE is not set"
    if not d3_active():
        rm = BR.make_room(DR3.NEED_GIB + 1, "Fish Audio Drama 3 speech server", may_release=("comfyui", "localrouter", "director"))
        if not rm["ok"]: return False, f"only {rm['have_gib']:.0f} GiB free — the Drama 3 server needs {DR3.NEED_GIB} GiB"
        try:
            os.makedirs(os.path.dirname(DR3.GPU_LOCK), exist_ok=True); os.close(os.open(DR3.GPU_LOCK, os.O_CREAT | os.O_RDWR, 0o644))   # pilot.py refuses a missing lock file
        except OSError: pass
        sh(["systemctl", "--user", "reset-failed", DR3.UNIT], 10); sh(["systemctl", "--user", "start", "--no-block", DR3.UNIT], 30)
    job["progress"].update(phase="loading Fish Audio Drama 3 (speech server, ~10 s to a minute)"); save(); t0 = time.time()
    while time.time() - t0 < 900:
        if job.get("cancel"): raise RuntimeError("cancelled")
        if d3_health(): D3S["last"] = time.time(); return True, ""
        if time.time() - t0 > 10 and not d3_active():
            why = sh(["journalctl", "--user", "-u", DR3.UNIT, "-n", "4", "--no-pager", "-o", "cat"], 10).replace("\n", " ")[-240:]
            return False, f"the Drama 3 server stopped while loading ({why or 'see journalctl --user -u ' + DR3.UNIT})"
        time.sleep(2)
    return False, "the Drama 3 server did not become ready in 15 minutes"


def d3_stop():
    if DR3.UNIT: sh(["systemctl", "--user", "stop", DR3.UNIT], 120)


def d3_idle():
    """stop the speech server unit DRAMA3_IDLE_S after its last voice-over so its ~54 GiB comes back — never while a voice-over is
    speaking; at once when another job is running (a render). A unit found running (studio restart, started by hand) gets the window
    from when it was first seen."""
    while True:
        time.sleep(15)
        try:
            if D3S["busy"] or not d3_active(): continue
            if not D3S["last"]: D3S["last"] = time.time(); continue
            other = any(j["status"] == "running" for j in list(JOBS.values()))   # a render now: the server's ~54 GiB would only starve it
            if not other and time.time() - D3S["last"] < DR3.IDLE_STOP_S: continue
            d3_stop(); D3S["last"] = 0.0
            BR._remember({"ok": True, "log": [f"Drama 3 speech server stopped after {DR3.IDLE_STOP_S} s idle"]}, "idle stop")
        except Exception: pass


def drama3_voiceover(job, vd, vid, text, vo):
    """-> True when Drama 3 (server or per-take) spoke the voice-over; False = the caller speaks it with Qwen3-TTS (reason in job notes).
    job["vo_engine"] / job["vo_path"] record which path was used."""
    D3S["busy"] = True
    try:
        ok, note = d3_fetch(job)
        if note and ok: _d3_note(job, f"Drama 3 {note}")
        if not ok:
            _d3_note(job, f"Drama 3 unavailable — {note}; spoken with Qwen3-TTS instead"); job["vo_path"] = "say (fallback: no weights)"; return False
        try: urllib.request.urlopen(urllib.request.Request(f"{COMFY}/free", data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                                           headers={"Content-Type": "application/json"}), timeout=30)
        except Exception: pass
        why = ""
        if DR3.voice_tool_ready() and ((DR3.UNIT and DR3.server_built()) or d3_health()):
            try:
                job["progress"].update(phase="preparing the Drama 3 voice file (first use of this voice: ~20 s)"); save()
                vworker("voice3", vd, vid, t=900)
                up, why = d3_server_up(job)
                if up:
                    job["progress"].update(phase="speaking the voice-over with Fish Audio Drama 3 (speech server)"); save()
                    r = vworker("say3s", vd, text, vo, vid, t=3600); D3S["last"] = time.time()
                    job["vo_check"] = r.get("wordcheck"); job["vo_path"] = "say3s"
                    job["vo_engine"] = "Fish Audio Drama 3 · speech server (drama-tensorfold) — Built with Fish Audio"; return True
            except RuntimeError as e:
                if str(e) == "cancelled": raise
                why = str(e)[:240]
        elif DR3.old_built(): why = ""                                       # only the per-take build is set up: no note needed
        else: why = "the drama-tensorfold server is not set up"
        if why: _d3_note(job, f"Drama 3 server path failed ({why}); trying the per-take build")
        if DR3.old_built():
            if DR3.UNIT and d3_active() and not d3_health(): d3_stop()       # a half-started server would starve the per-take run
            try:
                job["progress"].update(phase="speaking the voice-over with Fish Audio Drama 3 (per-take, loads ~18 s)"); save()
                r = vworker("say3", vd, text, vo, t=1800)
                job["vo_check"] = r.get("wordcheck"); job["vo_path"] = "say3"
                job["vo_engine"] = "Fish Audio Drama 3 · per-take (drama3-generate) — Built with Fish Audio"; return True
            except RuntimeError as e:
                _d3_note(job, f"Drama 3 per-take failed ({str(e)[:200]})")
        _d3_note(job, "spoken with Qwen3-TTS instead"); job["vo_path"] = "say (fallback: Drama 3 failed)"; return False
    finally:
        D3S["busy"] = False


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
LEVELS = {"low": 0.25, "medium": 0.4, "high": 0.6}


def make_music(job, engine, style, lyrics, seconds, seed, tag, bpm=None, key=None):
    """render one piece in ComfyUI -> (flac path, seconds it came out)"""
    kw = {"bpm": bpm or 100, "key": key or "C major"} if engine == "fast" else {}
    pid = comfy_post(MU.graph(engine, style, lyrics, seconds, seed, f"studio/{job['id']}_{tag}", **kw))
    files = [f for f in comfy_wait(pid, job) if f.endswith(".flac") and os.path.basename(f).startswith(f"{job['id']}_{tag}")]
    if not files: raise RuntimeError("no music came back")
    return files[0], stitch.probe(files[0])["dur"] or 0


def gm_pick(name, drums):
    """the director's instrument name -> a GM program (0-127), "kit:<n>" for drums, or "mute"; None if nothing is close"""
    import difflib
    n = str(name or "").strip().lower()
    if not n: return None
    if n in ("mute", "none", "drop", "silent", "off"): return "mute"
    if drums:
        kits = {v.lower(): k for k, v in MT.KITS.items()}
        hit = kits.get(n, kits.get(n + " kit"))                                   # (kit 0 is falsy: compare with None)
        if hit is None: hit = next((v for k, v in kits.items() if n in k or k.split()[0] in n), None)
        if hit is None:
            m = difflib.get_close_matches(n, list(kits), n=1, cutoff=0.4); hit = kits[m[0]] if m else 0
        return f"kit:{hit}"
    gm = [x.lower() for x in MT.GM]
    if n in gm: return gm.index(n)
    m = difflib.get_close_matches(n, gm, n=1, cutoff=0.55)
    if m: return gm.index(m[0])
    hit = next((i for i, x in enumerate(gm) if n in x or x in n), None)
    return hit


def midi_arrange_api(b):
    """{file, request} -> {parts: {ch: program | "kit:n" | "mute"}, names, style, restyle, strength, notes, by}"""
    mid = str(b.get("file") or ""); path = os.path.realpath(os.path.join(ROOT, mid)); req = str(b.get("request") or "").strip()[:800]
    if not path.startswith(ROOT + "/uploads/") or not path.endswith(".mid") or not os.path.isfile(path): raise ValueError("attach a MIDI file first")
    if not req: raise ValueError("describe the sound you want, e.g. \"jazz trio with brushed drums\"")
    info = midi_worker("inspect", path, t=60)
    d = DIR.midi_arrange(info["parts"], req, info.get("bpm") or 120, info.get("seconds") or 30, MT.GM, list(MT.KITS.values()))
    raw = {str(k): v for k, v in (d.get("parts") or {}).items()}; parts, names = {}, {}
    for p in info["parts"]:
        ch = str(p["channel"]); v = gm_pick(raw.get(ch), p.get("drums"))
        if v is None: v = ("kit:0" if p.get("drums") else p.get("program", 0))      # unknown name: keep what the file had
        parts[ch] = v; names[ch] = "mute" if v == "mute" else MT.KITS[int(v[4:])] if isinstance(v, str) else MT.GM[v]
    try: strength = max(0.3, min(0.8, float(d.get("strength") or 0.55)))
    except (TypeError, ValueError): strength = 0.55
    return {"parts": parts, "names": names, "style": str(d.get("style") or "").strip(), "restyle": d.get("restyle") is True,
            "strength": strength, "notes": str(d.get("notes") or "").strip(), "by": C.DIRECTOR_LABEL}


def midi_worker(*args, t=600):
    r = subprocess.run([MU.MIDI_PY, MU.MIDI_TOOL, *args], capture_output=True, text=True, timeout=t)
    line = next((l for l in reversed((r.stdout or "").splitlines()) if l.startswith("{")), "")
    try: d = json.loads(line)
    except ValueError: d = {"ok": False, "error": (r.stderr or r.stdout or "no output")[-300:]}
    if not d.get("ok"): raise RuntimeError(d.get("error") or "MIDI worker failed")
    return d


def run_midi(job):
    """the MIDI with the chosen instruments (MuseScore General soundfont), then optionally restyled by ACE-Step"""
    p = job["params"]; d = f"{LIB}/{job['id']}"; os.makedirs(d, exist_ok=True)
    job["progress"].update(phase="playing the MIDI with your instruments"); save()
    json.dump({"parts": p["parts"], "transpose": p["transpose"], "speed": p["speed"]}, open(f"{d}/instruments.json", "w"))
    shutil.copy(f"{ROOT}/{p['midi']}", f"{d}/source.mid")
    r = midi_worker("render", f"{d}/source.mid", f"{d}/midi_render.wav", f"{d}/instruments.json")
    outs = []
    if p.get("restyle"):
        job["progress"].update(phase=f"restyling it with ACE-Step (strength {p['strength']:.2f})"); save()
        name = to_input(f"{d}/midi_render.wav", f"studio_{job['id']}_midi.wav")
        pid = comfy_post(MU.restyle_graph(p["style"], name, r["seconds"], p["seed"], f"studio/{job['id']}_mus", p["strength"], p.get("bpm") or 120))
        files = [f for f in comfy_wait(pid, job) if f.endswith(".flac") and os.path.basename(f).startswith(f"{job['id']}_mus")]
        if not files: raise RuntimeError("the restyle produced no audio")
        shutil.move(files[0], f"{d}/music.flac"); os.remove(os.path.join(COMFY_IN, name)); outs.append({"file": "midi_render.wav", "kind": "MIDI render"})
    else:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/midi_render.wav", f"{d}/music.flac"], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/music.flac", "-c:a", "libmp3lame", "-b:a", "256k", f"{d}/music.mp3"], check=True)
    job["outputs"] = [{"file": "music.mp3", "kind": "music"}, {"file": "music.flac", "kind": "lossless"}] + outs + [{"file": "source.mid", "kind": "MIDI"}]
    job["music"] = {"engine": "MIDI" + (" + ACE-Step restyle" if p.get("restyle") else ""), "seconds": round(stitch.probe(f"{d}/music.flac")["dur"] or r["seconds"], 1)}


def run_music(job):
    p = job["params"]; d = f"{LIB}/{job['id']}"; os.makedirs(d, exist_ok=True); eng = p["engine"]
    if not MU.ready(eng): raise RuntimeError(f"{MU.ENGINES[eng]} is not installed")
    if eng == "midi": return run_midi(job)
    if not p.get("style"):
        job["progress"].update(phase="the director is writing the music brief"); save()
        b = DIR.music_brief(p["idea"], p["seconds"], eng); p.update(style=b["style"], bpm=b["bpm"], key=b["key"], brief_by=b["by"])
    job["progress"].update(phase=f"{MU.ENGINES[eng]} · {p['seconds']:g} s" + (" with lyrics" if p.get("lyrics", "").strip() else " · instrumental")); save()
    f, dur = make_music(job, eng, p["style"], p.get("lyrics", ""), p["seconds"], p["seed"], "mus", p.get("bpm"), p.get("key"))
    shutil.move(f, f"{d}/music.flac")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/music.flac", "-c:a", "libmp3lame", "-b:a", "256k", f"{d}/music.mp3"], capture_output=True, text=True)
    if r.returncode: raise RuntimeError("mp3 encode failed: " + r.stderr[-300:])
    job["outputs"] = [{"file": "music.mp3", "kind": "music"}, {"file": "music.flac", "kind": "lossless"}]; job["music"] = {"engine": MU.ENGINES[eng], "seconds": round(dur, 1)}


def add_score(job, main):
    """score the film: a track its exact length (Best may end early -> looped), 1 s fade in, 2.5 s fade out, under the film's own sound"""
    p = job["params"]; eng = p.get("music")
    if eng not in ("fast", "best") or not MU.ready(eng): return main
    d = os.path.dirname(main); dur = stitch.probe(main)["dur"]; style = (p.get("music_style") or "").strip(); bpm = key = None
    if not style:
        job["progress"].update(phase="the director is writing the score brief"); save()
        try: b = DIR.music_brief(p.get("prompt", ""), dur, eng); style, bpm, key = b["style"], b["bpm"], b["key"]; p["music_brief"] = b
        except Exception: style = "cinematic instrumental film score, emotive, matching this scene: " + (p.get("picture") or p.get("prompt", ""))[:300]
    job["progress"].update(phase=f"scoring the film · {MU.ENGINES[eng]} · {dur:.0f} s"); save()
    f, got = make_music(job, eng, style, "", math.ceil(dur) + 1, int(p["seed"]), "score", bpm, key)
    track = f"{d}/score.flac"; shutil.move(f, track)
    lv = LEVELS.get(p.get("music_level"), 0.4); out = f"{d}/film_music.mp4"
    loop = ["-stream_loop", "-1"] if got < dur - 0.5 else []
    graph = (f"[1:a]atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=1,afade=t=out:st={max(0, dur - 2.5):.3f}:d=2.5,volume={lv}[m];"
             f"[0:a][m]amix=inputs=2:duration=first:normalize=0[a]")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", main, *loop, "-i", track, "-filter_complex", graph, "-map", "0:v", "-map", "[a]", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", out], capture_output=True, text=True)
    if r.returncode: raise RuntimeError("music mix failed: " + (r.stderr or "")[-300:])
    job["outputs"].insert(0, {"file": os.path.basename(main), "kind": "without music"}); job["outputs"].insert(0, {"file": "score.flac", "kind": "score"})
    return out


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
SEG_RE = re.compile(r"^seg(\d\d)(?:\.v(\d+))?\.mp4$")
FILM_KINDS = ("video", "chain", "keyframes", "recreate")


def film_takes(jid):
    """segment number -> its takes in the film's folder: segNN.mp4 (as rendered) then segNN.vK.mp4 (editor re-renders)"""
    takes = {}
    for f in os.listdir(f"{LIB}/{jid}"):
        m = SEG_RE.match(f)
        if m: takes.setdefault(int(m.group(1)), []).append((int(m.group(2) or 0), f))
    return {k: [f for _, f in sorted(v)] for k, v in sorted(takes.items())}


def seg_prompt(p, k):
    if p.get("shots"): return official(p["shots"][k]["prompt"]) + (f'\n\n<Subject 1> says, "{p["shots"][k]["words"]}"' if p["shots"][k]["words"] else "")
    lines = [x.strip() for x in (p.get("beats") or "").splitlines() if x.strip()]
    return p["prompt"] + (("\n\nThis part: " + lines[min(k, len(lines) - 1)]) if lines else "")


def seg_seconds(j, k):
    p = j["params"]
    if j["kind"] == "recreate": return float(p["shots"][k]["dur"])
    return float(p["gaps"][k]) if j["kind"] == "keyframes" else plan_segments(p)[k] if j["kind"] == "chain" else float(p["seconds"])


def film_job(fid):
    j = JOBS.get(fid)
    if not j or j["kind"] not in FILM_KINDS or j["status"] != "done": raise ValueError("open a finished video or film")
    return j


def edit_state(j):
    takes = film_takes(j["id"]); p = j["params"]
    e = j.get("edit") or {"clips": [{"seg": k, "src": v[0], "in": 0.0, "out": None, "on": True} for k, v in takes.items()],
                          "music": p.get("music", ""), "music_style": p.get("music_style", ""), "music_level": p.get("music_level", "medium"),
                          "voice": p.get("voice", ""), "vo": p.get("vo", ""), "versions": []}
    info = {}
    for v in takes.values():
        for f in v: pr = stitch.probe(f"{LIB}/{j['id']}/{f}"); info[f] = {"dur": round(pr["dur"], 3), "frames": pr["frames"]}
    return {"id": j["id"], "kind": j["kind"], "preset": p.get("preset"), "seconds": p.get("seconds"), "prompt": p.get("prompt", ""), "edit": e,
            "takes": {str(k): v for k, v in takes.items()}, "info": info, "prompts": {str(k): seg_prompt(p, k - 1) for k in takes},
            "busy": [x["id"] for x in JOBS.values() if x["kind"] in ("retake", "export") and x["params"].get("film") == j["id"] and x["status"] in ("queued", "waiting_gpu", "running")]}


def edit_save(j, b):
    takes = film_takes(j["id"]); known = {f for v in takes.values() for f in v}; clips = []
    for c in (b.get("clips") or [])[:60]:
        src = str(c.get("src") or ""); m = SEG_RE.match(src)
        if src not in known or not m: raise ValueError(f"unknown clip {src}")
        dur = stitch.probe(f"{LIB}/{j['id']}/{src}")["dur"]
        i = max(0.0, min(dur - 0.25, float(c.get("in") or 0))); o = c.get("out")
        o = None if o in (None, "", 0, "0") or float(o) >= dur - 0.01 else max(i + 0.25, float(o))
        clips.append({"seg": int(m.group(1)), "src": src, "in": round(i, 3), "out": None if o is None else round(o, 3), "on": bool(c.get("on", True))})
    if not any(c["on"] for c in clips): raise ValueError("keep at least one clip")
    e = j.get("edit") or {"versions": []}
    e.update(clips=clips, voice=str(b.get("voice") or ""), vo=str(b.get("vo") or "")[:3000], **scorep(b)); j["edit"] = e; save(); return e


def run_retake(job):
    """one segment again, with a new prompt/seed, guided exactly like the original (previous segment's tail / first image / its keyframes)"""
    q = job["params"]; fj = JOBS[q["film"]]; p = fj["params"]; d = f"{LIB}/{fj['id']}"; k = int(q["seg"]) - 1; tag = f"{fj['id']}_e{job['id'][-4:]}"
    sec = seg_seconds(fj, k); first = last = guide = None; gframes = 0
    if fj["kind"] == "keyframes":
        first = to_input(f"{ROOT}/{p['images'][k]}", f"studio_{tag}_a{os.path.splitext(p['images'][k])[1]}")
        last = to_input(f"{ROOT}/{p['images'][k + 1]}", f"studio_{tag}_b{os.path.splitext(p['images'][k + 1])[1]}")
    elif k == 0 and p.get("first_image_path"):
        first = to_input(p["first_image_path"], f"studio_{tag}_first{os.path.splitext(p['first_image_path'])[1]}")
    elif k > 0 and fj["kind"] == "chain":
        prev = f"{d}/seg{k:02d}.mp4"; guide = to_input(prev, f"studio_{tag}_guide.mp4"); gframes = stitch.probe(prev)["frames"]
    job["progress"].update(seg=1, segs=1, phase=f"re-rendering segment {k + 1} of the film · {sec:g} s"); save()
    if is_cast(p):
        ims, au = cast_refs(p, tag); guide = None
        g, info = W.cast_graph(p["preset"], cast_prompt(p, q["prompt"]), sec, int(q["seed"]), f"studio/{tag}", ims, au, vae8=bool(p.get("vae8")))
    else:
        g, info = vgraph(p, q["prompt"], sec, int(q["seed"]), f"studio/{tag}", first_image=first, last_image=last, guide_video=guide, force_h3=fj["kind"] == "chain")
    if guide:
        g["33"]["inputs"]["batch_index"] = max(0, gframes - W.GUIDE_FRAMES); g["34"]["inputs"]["start_index"] = max(0.0, (gframes - W.GUIDE_FRAMES) / W.FPS)
    pid = comfy_post(g); job["progress"]["engine"] = info["engine"]; save()
    files = [f for f in comfy_wait(pid, job) if f.endswith(".mp4") and os.path.basename(f).startswith(f"{tag}_")]
    if not files: raise RuntimeError("the re-render produced no video")
    n = max([0] + [int(SEG_RE.match(f).group(2) or 0) for f in film_takes(fj["id"]).get(k + 1, [])]) + 1
    out = f"{d}/seg{k + 1:02d}.v{n}.mp4"; shutil.move(files[0], out)
    if q.get("review", True):
        try: rv = DIR.review(out, q["prompt"]); job["reviews"] = {"1": rv}
        except Exception as e: job["reviews"] = {"1": {"error": f"review skipped: {str(e)[:120]}"}}
    job["outputs"] = [{"file": os.path.basename(out), "kind": "take"}]; job["take"] = os.path.basename(out)
    for f in os.listdir(COMFY_IN):
        if f.startswith(f"studio_{tag}_"): os.remove(os.path.join(COMFY_IN, f))


def run_export(job):
    """the saved edit -> a new version in <film>/vN/: clips cut in order (a chain segment's 22 guide frames always dropped, a keyframe clip's repeated
    boundary frame dropped after its own predecessor), 60 ms sound fades at every cut, then score and voice-over if set"""
    fj = JOBS[job["params"]["film"]]; e = fj["edit"]; d = f"{LIB}/{fj['id']}"
    clips = [c for c in e["clips"] if c["on"]]
    n = len(e.get("versions") or []) + 1; vd = f"{d}/v{n}"; os.makedirs(vd, exist_ok=True)
    job["progress"].update(seg=1, segs=1, phase=f"cutting {len(clips)} clip(s) into version {n}"); save()
    inputs, parts, prev, xs, lens = [], [], None, [], []
    for i, c in enumerate(clips):
        src = f"{d}/{c['src']}"; dur = stitch.probe(src)["dur"]
        cast = is_cast(fj["params"])
        if fj["kind"] == "chain" and not cast: head = W.GUIDE_FRAMES / W.FPS if c["seg"] > 1 else 0.0       # a guided segment repeats its predecessor's tail
        elif fj["kind"] == "keyframes" or cast: head = 1 / W.FPS if prev == c["seg"] - 1 else 0.0            # the boundary frame, after its own predecessor
        else: head = 0.0
        a = max(c["in"], head); b = min(c["out"] or dur, dur)
        if b - a < 0.2: raise RuntimeError(f"clip {i + 1} ({c['src']}) is shorter than 0.2 s after trimming")
        # a later clip lends up to 60 ms of the sound just before its in-point as a crossfade (picture/sound sync kept)
        x = min(0.06, a) if i else 0.0; xs.append(x); lens.append(b - a)
        inputs += ["-i", src]
        parts.append(f"[{i}:v]trim=start={a:.4f}:end={b:.4f},setpts=PTS-STARTPTS[v{i}];[{i}:a]atrim=start={a - x:.4f}:end={b:.4f},asetpts=PTS-STARTPTS[a{i}]")
        prev = c["seg"]
    achain, last, T = [], "a0", lens[0]
    for i in range(1, len(clips)):
        if xs[i] >= 0.005: achain.append(f"[{last}][a{i}]acrossfade=d={xs[i]:.4f}:c1=qsin:c2=qsin[ax{i}]")
        else: achain.append(f"[{last}]afade=t=out:st={T - 0.01:.4f}:d=0.01[af{i}];[a{i}]afade=t=in:st=0:d=0.01[ai{i}];[af{i}][ai{i}]concat=n=2:v=0:a=1[ax{i}]")   # nothing to lend: 10 ms fades
        last = f"ax{i}"; T += lens[i]
    graph = ";".join(parts + ["".join(f"[v{i}]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0[v]"] + achain)
    cut = f"{vd}/cut.mp4"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[v]", "-map", f"[{last}]", "-c:v", "libx264", "-preset", "slow", "-crf", "14",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", cut], capture_output=True, text=True)
    if r.returncode: raise RuntimeError("cut failed: " + (r.stderr or "")[-400:])
    job["params"].update({k: v for k, v in fj["params"].items() if k not in ("film",)}); job["params"].update(film=fj["id"], music=e.get("music", ""),
        music_style=e.get("music_style", ""), music_level=e.get("music_level", "medium"), voice=e.get("voice", ""), vo=e.get("vo", ""))
    main = cut
    if job["params"]["music"] in ("fast", "best"):
        wait_gpu(job)
        if not ensure_comfy(): raise RuntimeError("ComfyUI did not start")
        job["status"] = "running"; save(); main = add_score(job, main)
    main = add_voiceover(job, main)
    stitch.poster(main, f"{vd}/poster.jpg", at=min(1.5, max(0.2, stitch.probe(main)["dur"] / 3)))
    rel = f"v{n}/{os.path.basename(main)}"; dur = stitch.probe(main)["dur"]
    e.setdefault("versions", []).append({"v": n, "file": rel, "poster": f"v{n}/poster.jpg", "created": now(), "clips": len(clips), "dur": round(dur, 2)})
    fj["outputs"].insert(0, {"file": rel, "kind": f"edit v{n}", "poster": f"v{n}/poster.jpg"})
    job["outputs"] = [{"file": rel, "kind": f"edit v{n}"}]; save()


def noise_seed(seed, pictures):
    """an edit's noise is drawn for the seed AND the bytes of every input picture. With the plain seed, an edit whose seed equals the one
    its source was made with starts from that picture's own noise and walks back to it. Same seed + same pictures +
    same prompt still gives the same result. 48 bits so it travels as a whole number."""
    import hashlib
    h = hashlib.sha256(f"studio-edit:{int(seed)}".encode())
    for pic in pictures:
        with open(pic, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""): h.update(blk)
        h.update(b"|")
    return int.from_bytes(h.digest()[:6], "big")


def run_edit(job):
    """the references into ComfyUI's input dir, one Turbo edit per seed (count), saved as img01.."""
    p = job["params"]; d = f"{LIB}/{job['id']}"; os.makedirs(d, exist_ok=True)
    names = [to_input(f"{ROOT}/{im}", f"studio_{job['id']}_ref{i + 1}{os.path.splitext(im)[1]}") for i, im in enumerate(p["images"])]
    for k in range(p["count"]):
        job["progress"].update(seg=k + 1, segs=p["count"], phase=f"editing · {len(names)} reference(s) · {k + 1} of {p['count']}"); save()
        ns = noise_seed(p["seed"] + k, [f"{ROOT}/{im}" for im in p["images"]]); job.setdefault("noise_seeds", []).append(ns)
        pid = comfy_post(W.edit_graph(names, p["prompt"], ns, f"studio/{job['id']}_e{k + 1}", p.get("negative", ""), p.get("model", W.EDIT_MODEL)))
        files = [f for f in comfy_wait(pid, job) if f.lower().endswith(".png") and os.path.basename(f).startswith(f"{job['id']}_e{k + 1}")]
        if not files: raise RuntimeError("the edit produced no image")
        dst = f"{d}/img{k + 1:02d}.png"; shutil.move(files[0], dst); job["outputs"].append({"file": os.path.basename(dst), "kind": "image"}); save()
    for f in os.listdir(COMFY_IN):
        if f.startswith(f"studio_{job['id']}_"): os.remove(os.path.join(COMFY_IN, f))


def run_ming(job):
    """Ming-Image: the director's layout (optional) -> ComfyUI /free -> memory check (stop the director if still short) -> one Ming process"""
    p = job["params"]; d = f"{LIB}/{job['id']}"; os.makedirs(d, exist_ok=True); notes = []
    prompt = p["prompt"]
    if not prompt: raise RuntimeError("describe the design first")
    if p.get("layout"):
        job["progress"].update(phase="the director is laying out the design (Ming's own rewriter)"); save()
        try:
            prompt, secs = DIR.ming_layout(p["prompt"], p["w"], p["h"]); notes.append(f"director layout in {secs} s")
            open(f"{d}/ming_layout.json", "w").write(prompt)
        except Exception as e:
            notes.append(f"director layout failed ({str(e)[:120]}) — your words were used as written"); prompt = p["prompt"]
    if p.get("alpha"): prompt = "RGBA, 4-channel, transparent background\n" + prompt         # one of Ming's fixed RGBA phrases, at the very start
    open(f"{d}/ming_prompt.txt", "w").write(prompt)
    need = W.ming_need_gib(p["w"], p["h"]); mem = [f"free {meminfo_gib():.0f} GiB before (needs {need:.0f})"]
    job["progress"].update(phase="making room: ComfyUI unloads its models"); save()
    try:
        urllib.request.urlopen(urllib.request.Request(f"{COMFY}/free", data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=60).read(); time.sleep(4)
    except Exception: pass
    mem.append(f"{meminfo_gib():.0f} GiB after ComfyUI /free")
    if meminfo_gib() < need and C.DIRECTOR_SERVICE and DIR.status().get("director"):
        DIR.stop(); time.sleep(8); mem.append(f"{meminfo_gib():.0f} GiB after pausing the director")
        notes.append("the director was paused to make room (it restarts on its own when needed)")
    notes.append(" → ".join(mem))
    if meminfo_gib() < need:
        raise RuntimeError(f"Ming needs about {need:.0f} GiB free at {p['w']}x{p['h']} and only {meminfo_gib():.0f} GiB is — pick a smaller size, or stop another GPU service first")
    seeds = ",".join(str(int(p["seed"]) + i) for i in range(p["count"]))
    job["progress"].update(phase=f"Ming-Image · loading 53 GB (~6 min), then {p['count']} image(s)"); save()
    r = subprocess.run([W.MING_PY, W.MING_RUNNER, "--prompt-file", f"{d}/ming_prompt.txt", "--width", str(p["w"]), "--height", str(p["h"]),
                        "--seeds", seeds, "--out", f"{d}/ming_raw", "--model", W.MING_MODEL, "--code", W.MING_CODE],
                       capture_output=True, text=True, timeout=3600)
    lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode or not lines: raise RuntimeError("Ming-Image failed: " + (r.stderr or r.stdout)[-400:])
    res = json.loads(lines[-1])
    for i, f in enumerate(res["outputs"]):
        dst = f"{d}/img{i + 1:02d}.png"; shutil.move(f, dst); job["outputs"].append({"file": os.path.basename(dst), "kind": "image"})
    notes.append(f"Ming load {res.get('load_s')} s · per image {res.get('gen_s')} s · torch {res.get('torch')} · transformers {res.get('transformers')}")
    job["notes"] = " · ".join(notes); save()


def run_lrfast(job):
    """Fast (LocalRouter): the broker makes room -> LocalRouter up (started on demand) -> POST /v1/images/generations. True = done. Any
    refusal or error -> False with the reason in the job notes (LocalRouter's models unloaded first), and run_image renders on ComfyUI."""
    p = job["params"]; d = f"{LIB}/{job['id']}"; notes = []; t0 = time.time(); why = ""; pngs = []; res = {}
    def back(why):
        if lr_up(): notes.append(BR.release("localrouter", "fallback to ComfyUI"))
        job["notes"] = " · ".join(notes + [f"Fast (LocalRouter) not used: {why} — rendered on ComfyUI"]); save(); return False
    job["progress"].update(phase="Fast (LocalRouter) · making room"); save()
    # only ComfyUI's idle models give way: never LocalRouter itself (warm = its model is already loaded) and never the director (a 6 s image is
    # not worth a director restart; if ComfyUI's room is not enough, the job falls back to ComfyUI, ~3 s slower warm)
    t = SB.lr_tools(LR); warm = any(x.get("id") == W.LR_IMAGE[p["model"]] and x.get("state") in ("ready", "busy") for x in (t or {}).get("data") or [])
    r = BR.make_room(W.lr_need_gib(p["w"], p["h"], warm), "Fast (LocalRouter)" + (" (warm)" if warm else ""), may_release=("comfyui",)); notes += r["log"]
    if not r["ok"]: return back(r["reason"])
    with BR.using("localrouter"):
        LRF["busy"] = True
        try:
            job["progress"].update(phase="Fast (LocalRouter) · starting"); save()
            if not lr_ensure():
                why = (f"LocalRouter did not come up at {LR}" if (C.LR_SERVICE or C.LR_START)
                       else f"LocalRouter is not answering at {LR} (no LOCALROUTER_SERVICE or LOCALROUTER_START to start it)")
            else:
                job["progress"].update(phase=f"Fast (LocalRouter) · Qwen-Image 2.1 Turbo · {p['count']} image(s)"); save()
                body = {"model": W.LR_IMAGE[p["model"]], "prompt": p["prompt"], "size": f"{p['w']}x{p['h']}", "n": int(p["count"]), "seed": int(p["seed"]),
                        "response_format": "b64_json"}
                req = urllib.request.Request(f"{LR}/v1/images/generations", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                try:
                    res = json.load(urllib.request.urlopen(req, timeout=900))
                    pngs = [base64.b64decode(x.get("b64_json") or "") for x in (res.get("data") or [])]
                    if len(pngs) != int(p["count"]) or not all(b.startswith(b"\x89PNG\r\n\x1a\n") for b in pngs): why = f"{len(pngs)} image(s) came back, wanted {p['count']} PNG(s)"
                except urllib.error.HTTPError as e:
                    try: m = json.loads(e.read() or b"{}").get("error") or {}; why = f"HTTP {e.code} {m.get('type') or ''}: {str(m.get('message') or '')[:300]}"
                    except Exception: why = f"HTTP {e.code}"
                except Exception as e: why = f"{type(e).__name__}: {str(e)[:200]}"
        finally: LRF.update(busy=False, last=time.time())
    if why: return back(why)
    if job.get("cancel"): raise RuntimeError("cancelled")
    for i, b in enumerate(pngs):
        dst = f"{d}/img{i + 1:02d}.png"
        with open(dst + ".part", "wb") as f: f.write(b)
        os.replace(dst + ".part", dst); job["outputs"].append({"file": os.path.basename(dst), "kind": "image"})
    job["notes"] = " · ".join(notes + [f"Fast (LocalRouter) · Qwen-Image 2.1 Turbo FP8 · {time.time() - t0:.1f} s · seed {res.get('seed', p['seed'])}"]); save(); return True


def run_image(job):
    p = job["params"]; d = f"{LIB}/{job['id']}"; os.makedirs(d, exist_ok=True)
    if p.get("model") == W.MING: return run_ming(job)                                       # Ming runs outside ComfyUI
    if p.get("iengine") == "localrouter" and lr_ready():                                    # Fast (LocalRouter); ComfyUI below is the fallback
        why = W.lr_fits(p["model"], p["w"], p["h"], p["count"])
        if not why and run_lrfast(job): return
        if why: job["notes"] = f"Fast (LocalRouter) not used: {why} — rendered on ComfyUI"; save()
        elif not ensure_comfy(): raise RuntimeError(f"Fast (LocalRouter) fell back and ComfyUI is not answering at {COMFY}")   # the worker skipped it for this Fast job
    job["progress"].update(phase=f"{p['model']} · {p['count']} image(s)"); save()
    pid = comfy_post(W.image_graph(p["model"], p["prompt"], p["w"], p["h"], p["count"], p["seed"], f"studio/{job['id']}_img", p.get("negative", "")))
    files = [f for f in comfy_wait(pid, job) if f.lower().endswith((".png", ".jpg", ".webp")) and os.path.basename(f).startswith(f"{job['id']}_img")]
    if not files: raise RuntimeError("no image came back")
    for i, f in enumerate(files):
        dst = f"{d}/img{i + 1:02d}.png"; shutil.move(f, dst); job["outputs"].append({"file": os.path.basename(dst), "kind": "image"})


def worker():
    while True:
        job = None
        with LOCK:
            q = sorted((j for j in JOBS.values() if j["status"] in ("queued", "waiting_gpu")), key=lambda j: j["created"])
            job = q[0] if q else None
        if not job:
            GPU["busy"] = ""; time.sleep(2); continue
        try:
            lr_yield(job)                                                     # a warm LocalRouter unloads before any other job
            if job["kind"] == "export":                                       # ffmpeg only, unless it re-scores (run_export asks for the GPU then)
                job.update(status="running", started=now()); GPU["busy"] = job["id"]; save(); run_export(job)
            elif job["kind"] == "voice":
                vr = BR.make_room(8, "a voice", may_release=())                # the same 8 GiB check, through the broker; it releases nothing
                if not vr["ok"]: raise RuntimeError(f"only {vr['have_gib']:.0f} GiB free — a voice needs ~8 GiB (stop something first)")
                job.update(status="running", started=now()); GPU["busy"] = job["id"]; save(); run_voice(job)
            else:
                nv = job["params"].get("voice") if str(job["params"].get("vo") or "").strip() else ""   # a missing narrator voice stops the film before the render
                if nv and (not re.match(r"^[\w-]+$", str(nv)) or not os.path.exists(f"{VOICES}/{nv}/ref.json")):
                    raise RuntimeError("the narrator's voice isn't ready — its cloning failed or it was deleted. Pick another narrator (or none) and queue again")
                wait_gpu(job)
                if not lr_job(job) and not ensure_comfy(): raise RuntimeError(f"ComfyUI is not answering at {COMFY} (see COMFY_URL / COMFY_SERVICE in studio.env)")   # Fast jobs start ComfyUI only on fallback
                job.update(status="running", started=now()); GPU["busy"] = job["id"]; save()
                run_recreate(job) if job["kind"] == "recreate" else run_retake(job) if job["kind"] == "retake" else run_music(job) if job["kind"] == "music" else run_edit(job) if job["kind"] == "edit" else run_image(job) if job["kind"] == "image" else run_keyframes(job) if job["kind"] == "keyframes" else run_video(job)
            job.update(status="done", finished=now()); job["progress"]["phase"] = "done"
        except Exception as e:
            job.update(status="cancelled" if str(e) == "cancelled" else "failed", error=str(e)[:600], finished=now())
        GPU["busy"] = ""; save()


def free_gpu():
    """the user clicked 'stop <tenant> and render': run GPU_TENANT_STOP"""
    if not C.TENANT_STOP: return {"error": "no GPU_TENANT_STOP command is configured"}
    r = subprocess.run(C.TENANT_STOP, shell=True, capture_output=True, text=True, timeout=300)
    if r.returncode: return {"error": (r.stderr or r.stdout or "stop command failed")[-400:]}
    GPU["freed_tf"] = True; GPU["note"] = ""; _OWN["v"] = None; return {"done": [f"stopped {C.TENANT_NAME}"]}


def restore_tf():
    if any(j["status"] in ("queued", "waiting_gpu", "running") for j in JOBS.values()): return {"error": "renders are still queued — start it again when they finish"}
    if not C.TENANT_START: return {"error": "no GPU_TENANT_START command is configured"}
    if C.COMFY_SERVICE: sh(["systemctl", "--user", "stop", C.COMFY_SERVICE], 60)
    DIR.stop()                                                                  # the other service usually needs the memory back
    r = subprocess.run(C.TENANT_START, shell=True, capture_output=True, text=True, timeout=300)
    if r.returncode: return {"error": (r.stderr or r.stdout or "start command failed")[-400:]}
    GPU["freed_tf"] = False; return {"started": True, "note": f"{C.TENANT_NAME} is starting"}


# ── HTTP ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def state():
    o = gpu_owner(); h = heat()
    return {"jobs": sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)[:60], "gpu": {**o, **GPU, "comfy": comfy_up(), "heat": h},
            "presets": {k: {**v, "est5": W.estimate_s(k, 5), "est15": W.estimate_s(k, 15)} for k, v in W.VIDEO.items()},
            "image_models": W.image_models(), "edit_models": W.edit_models(), "tf_ready": W.tf_ready(), "vae8_ready": W.vae8_ready(), "cast_ready": W.cast_ready(), "music_ready": {e: MU.ready(e) for e in MU.ENGINES}, "gm": MT.GM, "kits": MT.KITS, "music_engines": MU.ENGINES, "music_keys": MU.KEYS, "image_preset": W.IMAGE_PRESET, "guide_frames": W.GUIDE_FRAMES, "director": DIR.status(), "voices": voices(),
            "voice_ready": os.path.exists(TTS_PY) and os.path.isdir(f"{C.MODELS}/Qwen3-TTS-12Hz-1.7B-Base"), "brand": {"tenant": C.TENANT_NAME},
            "drama3_ready": DR3.offered(), "drama3_local": DR3.local(),
            "drama3_server": {"built": DR3.fast_ready(), "up": d3_active(), "idle_stop_s": DR3.IDLE_STOP_S if DR3.UNIT else 0},
            "lr_ready": lr_ready(), "lr_models": list(W.LR_IMAGE), "lr_sizes": [f"{a}x{b}" for a, b in W.LR_GO_SIZES], "lr_n": W.LR_GO_N,
            "lr": {"busy": LRF["busy"], "idle_stop_s": LR_IDLE_S if lr_can_stop() else 0},
            "broker": BR.status()}


def vgraph(p, *a, **k):
    """the job's engine and decoder choice reach every graph (TensorFold covers FastH3 passes only)"""
    return W.video_graph(p["preset"], *a, engine=p.get("engine", "comfy"), vae8=bool(p.get("vae8")), **k)


def eng(b):
    # TensorFold + the 8-bit decoder unless the job asks otherwise
    return {"engine": "tensorfold" if b.get("engine", "tensorfold") == "tensorfold" and W.tf_ready() else "comfy",
            "vae8": bool(b.get("vae8", True)) and W.vae8_ready()}


def scorep(b):
    return {"music": b.get("music") if b.get("music") in ("fast", "best") else "", "music_style": str(b.get("music_style") or "")[:2000],
            "music_level": b.get("music_level") if b.get("music_level") in LEVELS else "medium"}


def midi_params(b):
    mid = str(b.get("midi") or ""); path = os.path.realpath(os.path.join(ROOT, mid))
    if not path.startswith(ROOT + "/uploads/") or not path.endswith(".mid") or not os.path.isfile(path): raise ValueError("upload a MIDI file first")
    parts = {}
    for k, v in (b.get("parts") or {}).items():
        ch = int(k)
        if not 0 <= ch <= 15: raise ValueError("bad MIDI channel")
        if isinstance(v, str) and v.strip().isdigit(): v = int(v)                    # a program number sent as text
        if v == "mute" or (isinstance(v, str) and v.startswith("kit:") and int(v[4:]) in MT.KITS): parts[str(ch)] = v
        elif isinstance(v, int) and 0 <= v <= 127: parts[str(ch)] = v
        else: raise ValueError(f"bad instrument for channel {ch}")
    restyle = bool(b.get("restyle")); style = str(b.get("style") or "").strip()[:2000]
    if restyle and not style: raise ValueError("describe the style for the AI restyle")
    names = ", ".join(("mute" if v == "mute" else MT.KITS[int(v[4:])] if isinstance(v, str) else MT.GM[v]) for v in parts.values())
    return {"engine": "midi", "midi": mid, "parts": parts, "transpose": max(-24, min(24, int(b.get("transpose") or 0))), "speed": max(0.5, min(2.0, float(b.get("speed") or 1.0))),
            "restyle": restyle, "style": style, "strength": max(0.3, min(0.8, float(b.get("strength") or 0.55))), "seed": int(b.get("seed") or int.from_bytes(os.urandom(3), "big")),
            "bpm": max(40, min(200, int(float(b.get("bpm") or 120)))), "seconds": float(b.get("seconds") or 30), "lyrics": "", "idea": "",
            "prompt": f"MIDI · {names}" + (f" · restyled: {style[:120]}" if restyle else "")}


def music_params(b):
    if b.get("engine") == "midi": return midi_params(b)
    eng = b.get("engine") if b.get("engine") in ("fast", "best") else "fast"
    style, idea = str(b.get("style") or "").strip()[:3000], str(b.get("idea") or "").strip()[:1000]
    if not style and not idea: raise ValueError("describe the music (or give the director an idea)")
    sec = max(5.0, min(240.0, float(b.get("seconds") or 60)))
    key = b.get("key") if b.get("key") in MU.KEYS else "C major"
    return {"engine": eng, "style": style, "idea": idea or style, "lyrics": str(b.get("lyrics") or "")[:6000], "seconds": sec,
            "seed": int(b.get("seed") or int.from_bytes(os.urandom(3), "big")), "bpm": max(40, min(200, int(b.get("bpm") or 100))), "key": key,
            "prompt": style or idea}


def recreate_params(b, preset):
    url = str(b.get("url") or "").strip()
    if not re.match(r"^https://(www\.|m\.)?(youtube\.com/(shorts/|watch\?v=)|youtu\.be/)[\w-]{6,}", url): raise ValueError("paste a YouTube Shorts (or YouTube video) link")
    ims = [str(x) for x in (b.get("cast_images") or [])][:3]
    if not ims: raise ValueError("add at least one face image for your cast")
    for im in ims:
        path = os.path.realpath(os.path.join(ROOT, im))
        if not path.startswith(ROOT + "/") or not os.path.isfile(path): raise ValueError(f"cast image not found: {im}")
    if b.get("cast_consent") is not True: raise ValueError("confirm you have the right to use this person's likeness")
    cv = str(b.get("cast_voice") or "")
    if cv and (not re.match(r"^[\w-]+$", cv) or not os.path.exists(f"{VOICES}/{cv}/ref.wav")): raise ValueError("that voice is not in the Voices tab")
    if not W.cast_ready(): raise ValueError("the reference-to-video model is not installed")
    return {"url": url, "preset": preset if preset in ("draft", "fasthd") else "draft", "max_seconds": max(5.0, min(60.0, float(b.get("max_seconds") or 30))),
            "notes": str(b.get("notes") or "")[:600], "keep_words": bool(b.get("keep_words", True)), "seed": int(b.get("seed") or int.from_bytes(os.urandom(3), "big")),
            "review": b.get("review") if b.get("review") in ("off", "note", "retake") else "retake", "cast_images": ims, "cast_voice": cv, "cast_consent": True,
            "prompt": "Remake of a Short", "seconds": float(b.get("max_seconds") or 30), "vae8": bool(b.get("vae8", True)) and W.vae8_ready(), **scorep(b),
            **soundp(b), "style": b.get("style") if b.get("style") in ("reimagine", "faithful") else "reimagine",
            "shape": "landscape" if b.get("shape") == "landscape" else "auto", "engine": eng(b)["engine"]}


def soundp(b):
    """new = H3 makes the sound · music = the Short's music under the remake's voices · original = the Short's audio.
    With the Short's sound there is no extra score (it would fight the original music)."""
    t = b.get("soundtrack") if b.get("soundtrack") in ("new", "music", "original") else "new"
    return {"soundtrack": t, **({"music": ""} if t != "new" else {})}


def mk_params(b):
    mode = b.get("mode", "video"); preset = b.get("preset", "fasthd")
    if mode == "music": return "music", music_params(b)
    if mode == "recreate": return "recreate", recreate_params(b, preset)
    if preset not in W.VIDEO: raise ValueError("unknown preset")
    seed = int(b.get("seed") or int.from_bytes(os.urandom(3), "big"))
    if mode == "edit":
        ims = [str(x) for x in (b.get("images") or [])][:4]
        if not ims: raise ValueError("add the image(s) to edit")
        for im in ims:
            path = os.path.realpath(os.path.join(ROOT, im))
            if not path.startswith(ROOT + "/") or not os.path.isfile(path): raise ValueError(f"image not found: {im}")
        pr = str(b.get("picture") or b.get("prompt") or "").strip()[:4000]
        if not pr: raise ValueError("write what to change")
        em = W.edit_models()                                                   # Qwen-Image 2.1 Turbo (default) or FLUX 3 Image Klein
        if not em: raise ValueError("no edit model is installed (Qwen-Image 2.1 Turbo or FLUX 3 Image Klein)")
        return "edit", {"model": b.get("model") if b.get("model") in em else em[0], "prompt": pr, "negative": str(b.get("negative") or ""), "images": ims, "count": max(1, min(4, int(b.get("count") or 1))),
                        "seed": seed, "preset": preset}
    if mode == "image":
        model = b.get("model") or W.IMAGE_PRESET[preset]
        if model not in W.image_models(): raise ValueError(f"image model not installed: {model}")
        w, h = int(b.get("w") or 1344), int(b.get("h") or 768)
        return "image", {"model": model, "prompt": b.get("picture", "").strip(), "negative": b.get("negative", ""), "w": w, "h": h,
                         "count": max(1, min(4, int(b.get("count") or 1))), "seed": seed, "preset": preset,
                         "layout": b.get("layout") is not False, "alpha": bool(b.get("alpha")),   # layout / alpha: Ming-only options
                         "iengine": "localrouter" if b.get("iengine") == "localrouter" else "comfy"}   # the image engine: ComfyUI or Fast (LocalRouter)
    if mode == "keyframes":
        ims = b.get("images") or []
        if not isinstance(ims, list) or not 2 <= len(ims) <= 12: raise ValueError("pick 2 to 12 images, in order")
        for im in ims:
            path = os.path.realpath(os.path.join(ROOT, str(im)))
            if not path.startswith(ROOT + "/") or not os.path.isfile(path): raise ValueError(f"image not found: {im}")
        g = b.get("gap") or 5
        gaps = [max(3.0, min(15.0, float(x))) for x in (g if isinstance(g, list) else [g] * (len(ims) - 1))][:len(ims) - 1]
        prompt = b.get("prompt") or W.compose_prompt(b.get("picture", ""), b.get("action", ""), b.get("sound", ""))
        if not prompt.strip(): raise ValueError("describe the scene and what happens between the images")
        return "keyframes", {"preset": preset, "prompt": prompt, "images": [str(x) for x in ims], "gaps": gaps, "seed": seed, "beats": b.get("beats", ""),
                             "seconds": round(sum(gaps) - (len(gaps) - 1) / W.FPS, 2), "picture": b.get("picture", ""), "action": b.get("action", ""), "sound": b.get("sound", ""),
                             "review": b.get("review") if b.get("review") in ("off", "note", "retake") else "retake",
                             "voice": str(b.get("voice") or ""), "vo": str(b.get("vo") or "")[:3000], "vo_engine": "drama3" if b.get("vo_engine") == "drama3" else "qwen", **eng(b), **scorep(b)}
    prompt = b.get("prompt") or W.compose_prompt(b.get("picture", ""), b.get("action", ""), b.get("sound", ""))
    if not prompt.strip(): raise ValueError("write what the picture shows")
    sec = float(b.get("seconds") or 5)
    p = {"preset": preset, "prompt": prompt, "seconds": sec, "seed": seed, "beats": b.get("beats", ""), "picture": b.get("picture", ""),
         "action": b.get("action", ""), "sound": b.get("sound", ""), "review": b.get("review") if b.get("review") in ("off", "note", "retake") else "retake",
         "voice": str(b.get("voice") or ""), "vo": str(b.get("vo") or "")[:3000], "vo_engine": "drama3" if b.get("vo_engine") == "drama3" else "qwen", **eng(b), **scorep(b)}
    ims = [str(x) for x in (b.get("cast_images") or [])][:3]
    for im in ims:
        path = os.path.realpath(os.path.join(ROOT, im))
        if not path.startswith(ROOT + "/") or not os.path.isfile(path): raise ValueError(f"cast image not found: {im}")
    cv = str(b.get("cast_voice") or "")
    if cv and (not re.match(r"^[\w-]+$", cv) or not os.path.exists(f"{VOICES}/{cv}/ref.wav")): raise ValueError("that voice is not in the Voices tab")
    if ims and b.get("cast_consent") is not True: raise ValueError("confirm you have the right to use this person's likeness")
    if ims or cv:
        if not W.cast_ready(): raise ValueError("the reference-to-video model is not installed")
        p.update(cast_images=ims, cast_voice=cv, cast_lines=str(b.get("cast_lines") or "")[:1200], cast_consent=bool(ims))
    fi = b.get("first_image")
    if fi:
        path = os.path.realpath(os.path.join(ROOT, fi))
        if not path.startswith(ROOT + "/") or not os.path.isfile(path): raise ValueError("first image not found")
        p["first_image_path"] = path
    return ("chain" if sec > 15 else "video"), p


class H(BaseHTTPRequestHandler):
    server_version = "studio/1"
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json", extra=None):
        b = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)
    def _file(self, path):
        size = os.path.getsize(path); rng = self.headers.get("Range"); ctype = {".mp4": "video/mp4", ".png": "image/png", ".jpg": "image/jpeg",
                                                                            ".webp": "image/webp", ".wav": "audio/wav"}.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, z = rng[6:].partition("-"); start = int(a or 0); end = int(z) if z else size - 1
        end = min(end, size - 1); n = end - start + 1
        self.send_response(206 if rng else 200); self.send_header("Content-Type", ctype); self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(n))
        if rng: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start); left = n
            while left > 0:
                c = f.read(min(1 << 20, left))
                if not c: break
                self.wfile.write(c); left -= len(c)
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/": return self._send(200, open(f"{HERE}/index.html", "rb").read(), "text/html; charset=utf-8")
        if u.path == "/api/state": return self._send(200, state())
        if u.path == "/api/trash": return self._send(200, {"items": trash_list()})
        if u.path == "/api/edit":
            try: return self._send(200, edit_state(film_job((parse_qs(u.query).get("id") or [""])[0])))
            except ValueError as e: return self._send(400, {"error": str(e)})
        if u.path.startswith("/files/"):
            path = os.path.realpath(os.path.join(ROOT, unquote(u.path[len("/files/"):])))
            if not path.startswith(ROOT + "/") or not os.path.isfile(path): return self._send(404, {"error": "not found"})
            return self._file(path)
        self._send(404, {"error": "not found"})
    def do_POST(self):
        if self.headers.get("X-Studio") != "1": return self._send(403, {"error": "missing X-Studio header"})   # CSRF: forces a CORS preflight we never allow
        u = urlparse(self.path); n = int(self.headers.get("Content-Length") or 0)
        if u.path == "/api/midi/upload":
            if n > 2 << 20: return self._send(413, {"error": "MIDI file too large (2 MB max)"})
            data = self.rfile.read(n)
            if not data.startswith(b"MThd"): return self._send(400, {"error": "that is not a standard MIDI file (.mid)"})
            name = f"uploads/{uuid.uuid4().hex[:10]}.mid"; open(f"{ROOT}/{name}", "wb").write(data)
            try: return self._send(200, {"file": name, **midi_worker("inspect", f"{ROOT}/{name}", t=60)})
            except Exception as e: return self._send(400, {"error": str(e)[:300]})
        if u.path == "/api/upload":
            if n > 40 << 20: return self._send(413, {"error": "image too large (40 MB max)"})
            ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get((self.headers.get("Content-Type") or "").split(";")[0])
            if not ext: return self._send(400, {"error": "png, jpg or webp only"})
            name = f"uploads/{uuid.uuid4().hex[:10]}{ext}"; open(f"{ROOT}/{name}", "wb").write(self.rfile.read(n))
            return self._send(200, {"file": name})
        try: b = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except ValueError: return self._send(400, {"error": "bad JSON"})
        if u.path == "/api/negation":                                         # negation.py: words the models would show
            return self._send(200, NG.check(str(b.get("text") or "")[:20000]))
        if u.path == "/api/jobs":
            try: kind, p = mk_params(b)
            except (ValueError, TypeError) as e: return self._send(400, {"error": str(e)})
            j = new_job(kind, p); ng = NG.find(" ".join(str(p.get(k) or "") for k in ("prompt", "notes", "changes")))
            if ng: j["notes"] = "Prompt names things to forbid them (" + ", ".join(f'"{x["phrase"]}"' for x in ng[:4]) + ") — the model tends to show them"; save()
            return self._send(200, j)
        if u.path == "/api/cancel":
            j = JOBS.get(b.get("id"))
            if not j: return self._send(404, {"error": "no such job"})
            if j["status"] in ("queued", "waiting_gpu"): j.update(status="cancelled", finished=now()); save()
            else: j["cancel"] = True
            return self._send(200, {"ok": True})
        if u.path == "/api/delete":
            j = JOBS.get(b.get("id"))
            if not j or j["status"] in ("queued", "waiting_gpu", "running"): return self._send(409, {"error": "only finished renders can be deleted"})
            return self._send(200, trash_job(j))                              # to the Trash, not deleted
        if u.path == "/api/gpu/free":
            if b.get("confirm") != "tf": return self._send(400, {"error": "confirm with {confirm: 'tf'}"})
            return self._send(200, free_gpu())
        if u.path == "/api/gpu/restore": return self._send(200, restore_tf())
        if u.path == "/api/voices":
            url = str(b.get("url") or "").strip()
            if not re.match(r"^https://(www\.|m\.)?(youtube\.com/(shorts/|watch\?v=)|youtu\.be/)[\w-]{6,}", url): return self._send(400, {"error": "paste a YouTube Shorts (or YouTube video) link"})
            if b.get("consent") is not True: return self._send(400, {"error": "confirm you have the right to use this person's voice"})
            vp = {"url": url, "who": str(b.get("who") or "")[:200], "label": str(b.get("label") or "")[:60]}
            if b.get("multi") is True:                                         # everyone who speaks: speakers 0 = auto, else the user's count
                vp.update(multi=True, max_speakers=6, speakers=max(0, min(6, int(b.get("speakers") or 0))))
            return self._send(200, new_job("voice", vp))
        if u.path == "/api/voices/delete":
            vid = str(b.get("id") or "")
            if not re.match(r"^[\w-]+$", vid) or not os.path.isdir(f"{VOICES}/{vid}"): return self._send(404, {"error": "no such voice"})
            return self._send(200, trash_voice(vid))
        if u.path == "/api/trash/restore":
            try: return self._send(200, trash_restore(str(b.get("id") or "")))
            except (ValueError, OSError) as e: return self._send(400, {"error": str(e)})
        if u.path == "/api/trash/empty":
            if b.get("confirm") != "empty": return self._send(400, {"error": "confirm with {confirm: 'empty'}"})
            return self._send(200, trash_empty(b.get("id")))
        if u.path in ("/api/edit/save", "/api/edit/rerender", "/api/edit/export"):
            try:
                fj = film_job(str(b.get("id") or ""))
                if u.path == "/api/edit/save": return self._send(200, edit_save(fj, b))
                if u.path == "/api/edit/rerender":
                    k = int(b.get("seg") or 0)
                    if k not in film_takes(fj["id"]): raise ValueError("no such segment")
                    pr = str(b.get("prompt") or "").strip()[:4000] or seg_prompt(fj["params"], k - 1)
                    fp = fj["params"]
                    return self._send(200, new_job("retake", {"film": fj["id"], "seg": k, "prompt": pr, "seed": int(b.get("seed") or int.from_bytes(os.urandom(3), "big")),
                                                              "preset": fp["preset"], "seconds": seg_seconds(fj, k - 1), "engine": fp.get("engine", "comfy"),
                                                              "review": bool(b.get("review", True))}))
                if not fj.get("edit"): edit_save(fj, b) if b.get("clips") else fj.setdefault("edit", edit_state(fj)["edit"])
                return self._send(200, new_job("export", {"film": fj["id"], "preset": fj["params"].get("preset"), "seconds": fj["params"].get("seconds"),
                                                          "prompt": "edit of " + fj["params"].get("prompt", "")[:150]}))
            except (ValueError, TypeError, KeyError) as e: return self._send(400, {"error": str(e)})
        if u.path == "/api/director/midi":                                    # the director arranges a MIDI file from words
            try: return self._send(200, midi_arrange_api(b))
            except ValueError as e: return self._send(400, {"error": str(e)[:300]})
            except Exception as e: return self._send(500, {"error": f"{type(e).__name__}: {str(e)[:300]}"})
        if u.path == "/api/director/music":
            try: return self._send(200, DIR.music_brief(str(b.get("idea") or "")[:1000], float(b.get("seconds") or 60), b.get("engine") if b.get("engine") in ("fast", "best") else "fast"))
            except Exception as e: return self._send(500, {"error": str(e)[:300]})
        if u.path == "/api/director/write":
            idea = str(b.get("idea") or "").strip()
            if not idea: return self._send(400, {"error": "write your idea in a sentence first"})
            sec = float(b.get("seconds") or 5); mode = b.get("mode") or "video"
            try: return self._send(200, DIR.write(idea, sec, len(plan_segments({"seconds": sec})) if sec > 15 else 1, mode, int(b.get("n_images") or 0)))
            except Exception as e: return self._send(502, {"error": f"{type(e).__name__}: {str(e)[:300]}"})
        self._send(404, {"error": "not found"})


def main():
    global JOBS
    try: JOBS = json.load(open(JOBS_F))
    except Exception: JOBS = {}
    for j in JOBS.values():                                    # a restart mid-render: that render is lost, say so
        if j["status"] == "running": j.update(status="failed", error="the studio restarted during this render")
    save(); WS.start(); threading.Thread(target=worker, daemon=True).start()
    if lr_ready() and lr_can_stop() and LR_IDLE_S > 0: threading.Thread(target=lr_idle, daemon=True).start()   # stops an idle LocalRouter
    if DR3.UNIT and DR3.IDLE_STOP_S > 0: threading.Thread(target=d3_idle, daemon=True).start()   # stops an idle Drama 3 speech server
    threading.Thread(target=mem_guard, daemon=True).start(); comfy_oom_first()  # the memory guard
    srv = ThreadingHTTPServer((C.HOST, PORT), H); print(f"Lion AI Studios on http://{C.HOST}:{PORT}", flush=True); srv.serve_forever()


if __name__ == "__main__":
    main()
