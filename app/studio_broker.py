# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""studio_broker.py — the studio's memory broker.
One registry of what holds the Spark's unified memory (ComfyUI, the director, LocalRouter, a Ming process, a voice job), each with its
need, priority, keep_warm and how it is released. make_room(need_gib, for_what) releases IDLE holders, lowest priority first (keep_warm ones
last), until MemAvailable >= need; it never touches a holder that is mid-job (or whose state it cannot read) and never one whose priority is
at or above the requester's. Fail closed: if it cannot prove the room it returns ok=False with the reason, and the caller fails (or falls back).
GB10 note: device memory comes out of system memory, so MemAvailable is the shared budget (LocalRouter's own scheduler reads the same number).
No threads, no timers: the studio's single worker calls it before it loads something heavy; status() feeds /api/state."""
import json, threading, time, urllib.request

FOREGROUND = 50          # a studio job the worker is about to run outranks every idle cache below


def meminfo(key="MemAvailable"):
    """GiB from /proc/meminfo; None when it cannot be read (the broker then refuses)"""
    try:
        with open("/proc/meminfo") as f: return int(next(l for l in f if l.startswith(key + ":")).split()[1]) / 1048576
    except Exception: return None


def _get(url, t=3):
    with urllib.request.urlopen(url, timeout=t) as r: return json.loads(r.read() or b"{}")


def _post(url, body=None, t=60):
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=t) as r: return json.loads(r.read() or b"{}")


class Holder:
    """something that can hold memory. busy()/held() return True, False or None (unknown — treated as busy, and as held).
    release() frees it and returns a short note; None = this broker never releases it (it ends on its own)."""
    def __init__(self, name, label, need_gib, priority, keep_warm=False, release=None, busy=None, held=None, settle_s=15):
        self.name, self.label, self.need_gib, self.priority, self.keep_warm = name, label, float(need_gib), int(priority), bool(keep_warm)
        self.release, self.busy, self.held, self.settle_s = release, busy or (lambda: False), held or (lambda: None), settle_s


def _probe(fn):
    try: return fn()
    except Exception: return None


class Broker:
    def __init__(self, mem=meminfo, sleep=time.sleep, clock=time.time):
        self.mem, self.sleep, self.clock = mem, sleep, clock
        self.holders = {}; self.inflight = {}; self.recent = []; self._lock = threading.Lock(); self._ilock = threading.Lock(); self._st = (0, None)

    def register(self, h):
        self.holders[h.name] = h; self.inflight.setdefault(h.name, 0); return h

    # ── in-flight marks: a call through track()/using() makes that holder busy for its duration ──
    def using(self, name):
        br = self
        class _U:
            def __enter__(s):
                with br._ilock: br.inflight[name] = br.inflight.get(name, 0) + 1
            def __exit__(s, *a):
                with br._ilock: br.inflight[name] = max(0, br.inflight.get(name, 0) - 1)
        return _U()

    def track(self, name, fn):
        def wrapped(*a, **k):
            with self.using(name): return fn(*a, **k)
        wrapped.__wrapped__ = fn; wrapped.__name__ = getattr(fn, "__name__", "tracked"); return wrapped

    def is_busy(self, h):
        if self.inflight.get(h.name, 0) > 0: return True
        b = _probe(h.busy)
        return True if b is None else bool(b)

    def _have(self):
        v = _probe(self.mem)
        return None if v is None else float(v)

    def release(self, name, why=""):
        """release one idle holder now (not through make_room — e.g. a warm LocalRouter giving way before another job); returns a log line"""
        h = self.holders.get(name)
        if not h or not h.release: return f"{name}: nothing to release"
        with self._lock:
            if _probe(h.held) is False: return f"{h.label}: holds nothing"
            if self.is_busy(h): return f"{h.label}: busy — left alone"
            try: note = h.release() or "released"
            except Exception as e: return f"{h.label}: release failed ({type(e).__name__}: {str(e)[:120]})"
            line = f"{h.label}: {note}{' (' + why + ')' if why else ''}"; self._remember({"t": self.clock(), "for": why or "release", "ok": True, "log": [line]}); return line

    def make_room(self, need_gib, for_what, priority=FOREGROUND, may_release=None):
        """-> {ok, need_gib, have_gib, log: [..], reason}. may_release: None = any eligible holder, () = check only, or a tuple of names."""
        need = float(need_gib); log = []
        with self._lock:
            have = self._have()
            if have is None:
                r = {"ok": False, "need_gib": need, "have_gib": 0.0, "log": log, "reason": f"cannot read /proc/meminfo — refusing {for_what} (fail closed)"}
                self._remember(r, for_what); return r
            log.append(f"{have:.1f} GiB free, {for_what} needs {need:.1f}")
            if have >= need:
                r = {"ok": True, "need_gib": need, "have_gib": have, "log": log, "reason": ""}; self._remember(r, for_what); return r
            skipped = []
            cands = sorted((h for h in self.holders.values() if h.release and (may_release is None or h.name in may_release)),
                           key=lambda h: (h.keep_warm, h.priority, h.name))
            for h in cands:
                if have >= need: break
                if h.priority >= priority: skipped.append(f"{h.label} (priority {h.priority} ≥ {priority})"); continue
                if _probe(h.held) is False: continue          # holds nothing (or is not running): nothing to free
                if self.is_busy(h): skipped.append(f"{h.label} (busy)"); continue
                try: note = h.release() or "released"
                except Exception as e: log.append(f"{h.label}: release failed ({type(e).__name__}: {str(e)[:120]})"); continue
                t0 = self.clock(); have = self._have()
                while have is not None and have < need and self.clock() - t0 < h.settle_s:
                    self.sleep(1); have = self._have()
                if have is None:                              # meminfo lost mid-loop: stop now — no further release, and refuse (fail closed)
                    log.append(f"{h.label}: {note} → /proc/meminfo unreadable — stopped")
                    r = {"ok": False, "need_gib": need, "have_gib": 0.0, "log": log, "reason": f"cannot read /proc/meminfo — refusing {for_what} (fail closed)"}
                    self._remember(r, for_what); return r
                log.append(f"{h.label}: {note} → {have:.1f} GiB free")
            if have >= need:
                r = {"ok": True, "need_gib": need, "have_gib": have, "log": log, "reason": ""}
            else:
                r = {"ok": False, "need_gib": need, "have_gib": have, "log": log,
                     "reason": f"{for_what} needs ~{need:.0f} GiB free and only {have:.0f} GiB is" + (f" — left alone: {', '.join(skipped)}" if skipped else "")}
            self._remember(r, for_what); return r

    def _remember(self, r, for_what=None):
        e = {"t": time.strftime("%H:%M:%S"), "for": for_what or r.get("for"), "ok": r.get("ok"), "log": list(r.get("log") or [])[:8], "reason": r.get("reason", "")}
        self.recent = (self.recent + [e])[-10:]

    def status(self, max_age=5):
        """for the page: memory, every holder (busy / holds memory), the last few broker actions. Cached max_age s (the page polls)."""
        if self._st[1] is not None and self.clock() - self._st[0] < max_age: return self._st[1]
        hs = [{"name": h.name, "label": h.label, "need_gib": h.need_gib, "priority": h.priority, "keep_warm": h.keep_warm, "releasable": bool(h.release),
               "busy": self.is_busy(h), "held": _probe(h.held), "inflight": self.inflight.get(h.name, 0)} for h in self.holders.values()]
        a, f = self._have(), _probe(lambda: meminfo("MemFree"))
        st = {"mem_available_gib": None if a is None else round(a, 1), "mem_free_gib": None if f is None else round(f, 1), "holders": hs, "recent": self.recent[-5:]}
        self._st = (self.clock(), st); return st


# ── the studio's registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def lr_tools(lr_url):
    """LocalRouter's GET /v1/tools ({data: [{id, state: unloaded|loading|ready|busy|failed, resident_bytes, ..}], mem_available_bytes, ..}); None if down"""
    try: return _get(f"{lr_url}/v1/tools", 3)
    except Exception: return None


def lr_busy(lr_url):
    t = lr_tools(lr_url)
    if t is None: return False                     # not running = not busy (and holds nothing)
    return any(x.get("state") in ("loading", "busy") for x in t.get("data") or [])


def lr_held(lr_url):
    t = lr_tools(lr_url)
    if t is None: return False
    return any(x.get("state") in ("loading", "ready", "busy") or int(x.get("resident_bytes") or 0) > 0 for x in t.get("data") or [])


def lr_release(lr_url, stop=None):
    """LocalRouter's own POST /v1/tools/release (unloads every model; the daemon stays, holding a few MB). If that call fails and the studio
    knows how to stop LocalRouter (stop() returns a short note), it is stopped."""
    try: _post(f"{lr_url}/v1/tools/release", {}, 60); return "models unloaded (POST /v1/tools/release)"
    except Exception as e:
        if stop is None: return f"release call failed ({type(e).__name__}); no stop command is configured"
        return f"release call failed ({type(e).__name__}); stopped: {stop()}"


def comfy_busy(comfy_url):
    q = _get(f"{comfy_url}/queue", 5)                # raises when unreadable → None → treated as busy
    return bool(q.get("queue_running") or q.get("queue_pending"))


def comfy_free(comfy_url):
    _post(f"{comfy_url}/free", {"unload_models": True, "free_memory": True}, 60); return "models unloaded (ComfyUI /free)"


def studio_registry(br, comfy_url, lr_url, lr_stop, director, current_kind, director_label="the director", director_can_stop=True, lr_reserve_gib=15):
    """the holders. lr_url = None when the Fast engine (LocalRouter) is off; lr_stop() stops LocalRouter (None = the studio cannot).
    director = the director module (stop(), status()); director_can_stop = the studio manages the director's service (DIRECTOR_SERVICE);
    current_kind() = the kind of the job the worker is running ('' idle).
    Priorities order the releases: LocalRouter (10, reloads in ~12 s) → ComfyUI (20, ~35 s) → the director (30, keep_warm: only when the first two
    were not enough; it restarts on its own when next needed). Ming and voice are never released here: they are the job itself and exit with it."""
    def comfy_up():
        try: _get(f"{comfy_url}/system_stats", 3); return True
        except Exception: return False
    def dir_stop(): director.stop(); return "stopped (it starts again on its own when next needed)"
    if lr_url:
        br.register(Holder("localrouter", "LocalRouter (Qwen-Image Turbo)", 36000 / 1024 + lr_reserve_gib, 10, False, lambda: lr_release(lr_url, lr_stop),
                           lambda: lr_busy(lr_url), lambda: lr_held(lr_url), settle_s=10))
    br.register(Holder("comfyui", "ComfyUI", 24, 20, False, lambda: comfy_free(comfy_url), lambda: comfy_busy(comfy_url),
                       lambda: comfy_up() or False, settle_s=15))
    br.register(Holder("director", director_label, 19, 30, True, dir_stop if director_can_stop else None, lambda: False,
                       lambda: bool(director.status().get("director")), settle_s=20))
    br.register(Holder("ming", "Ming-Image process", 58, 90, False, None, lambda: current_kind() == "ming", lambda: current_kind() == "ming"))
    br.register(Holder("voice", "voice / Demucs", 8, 90, False, None, lambda: current_kind() == "voice", lambda: current_kind() == "voice"))
    return br
