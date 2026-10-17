#!/usr/bin/env python3
# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""drama3.py — where the optional Fish Audio Drama 3 voice engine lives, in one place (voice.py, studio_server.py and the
studio-drama3 unit all read it). Every path comes from config.py (studio.env). Built with Fish Audio — the weights are under the Fish
Audio Community License v1.0: keep the speech server on loopback and clone only voices with consent. See docs/DRAMA3.md.

Two ways to run Drama 3, tried in this order by a Drama 3 voice-over:
  the fast server   drama-tensorfold (github.com/plotarmordev/drama-tensorfold, Apache-2.0): its guarded launcher recipe/pilot.py starts
                    a loopback speech server (OpenAI-style POST /v1/audio/speech, GET /health) on DRAMA3_PORT. Each cloned voice gets a
                    voice file (transcript + vq_codes) made once by the engine tree's tools/zig/drama_voice.py; the codes ride inline
                    with every request, so a new voice never needs a server restart.
  the per-take build  TensorFold's drama3-cuda port (drama3-generate in DRAMA3_DIR): one process per take.

  python3 drama3.py serve   what systemd/studio-drama3.service.example runs: exec pilot.py with a fresh evidence directory
  python3 drama3.py show    print the paths and what is present
Stdlib only (studio_server runs under the system python3, voice.py in the TTS venv)."""
import json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

CKPT = C.DRAMA3_MODEL                                                   # fishaudio/drama-3 @ e0a9218a (config.json, codec.pth, weights)
GIB_ON_DISK = 39                                                        # 41.5 GB

# the fast server (drama-tensorfold @ 3f44a09c: ashhart/TensorFold c7137bb3 + its patches, built by scripts/build-engine.sh)
SERVER_DIR = C.DRAMA3_SERVER_DIR
ENGINE_BIN = C.DRAMA3_ENGINE_BIN
PILOT = os.path.join(SERVER_DIR, "recipe/pilot.py")
VOICE_TOOL = os.path.join(SERVER_DIR, "build/tensorfold/tools/zig/drama_voice.py")   # ref clip + transcript -> voice file (vq_codes)
VOICE_ENV_PY = C.DRAMA3_VOICE_ENV_PY                                    # the TTS venv + sglang-omni's light deps (requirements/drama3-voice.txt)
SGL_REF = C.DRAMA3_SGLANG_OMNI_DIR                                      # sgl-project/sglang-omni at ab095ef9 (the encoder drama_voice.py uses)
WORK = os.path.join(C.STUDIO_HOME, "drama3")
VOICES = os.path.join(WORK, "voices")                                   # <voice id>.json voice files (drama3-voice/1)
RUNTIME = os.path.join(WORK, "runtime")                                 # pilot.py's runtime dir (caches, evidence runs)
GPU_LOCK = C.DRAMA3_GPU_LOCK                                            # pilot.py's shared GPU lock file
UNIT = C.DRAMA3_SERVICE                                                 # empty = the studio only uses a server that already answers
PORT = C.DRAMA3_PORT; URL = f"http://127.0.0.1:{PORT}"; MODEL = "fishaudio/drama-3"
NEED_GIB = 88                                                           # pilot.py admits a load only with this much MemAvailable
SERVE_ARGS = ["--max-frames", "4096", "--voices", VOICES, "--no-update-check"]
IDLE_STOP_S = C.DRAMA3_IDLE_S                                           # the studio stops the unit this long after its last voice-over
FETCH_CMD = C.DRAMA3_FETCH_CMD                                          # optional fetch hook: run when the checkpoint is missing

# the per-take build (TensorFold drama3-cuda): the fallback when the server path is not set up or fails
OLD_DIR = C.DRAMA3_DIR
OLD_BIN = os.path.join(OLD_DIR, "bin/drama3-generate"); OLD_CODEC = os.path.join(OLD_DIR, "codec-v1.safetensors")


def local():
    """the checkpoint is on this machine (its config and codec are present)"""
    return os.path.isfile(f"{CKPT}/config.json") and os.path.isfile(f"{CKPT}/codec.pth")


def server_built(): return all(os.path.exists(x) for x in (ENGINE_BIN, PILOT))
def voice_tool_ready(): return all(os.path.exists(x) for x in (VOICE_TOOL, VOICE_ENV_PY, SGL_REF))
def old_built(): return all(os.path.exists(x) for x in (OLD_BIN, OLD_CODEC))
def fast_ready(): return server_built() and voice_tool_ready()
def engine_ready(): return fast_ready() or old_built()


def offered():
    """the narrator's menu offers Drama 3: an engine is built and the weights are here (or the fetch hook can bring them)"""
    return engine_ready() and (local() or bool(FETCH_CMD))


def voice_name(vid):
    """a studio voice id -> a valid server voice-file name (1..64 of A-Za-z0-9._- led by a letter or digit, not 'default')"""
    n = re.sub(r"[^A-Za-z0-9._-]+", "_", str(vid)).lstrip("._-")[:64] or "voice"
    return "v_" + n[:62] if n.lower() == "default" else n


def serve():
    for d in (RUNTIME, f"{RUNTIME}/runs", VOICES): os.makedirs(d, exist_ok=True)
    if not local(): sys.exit(f"Drama 3 weights not found at {CKPT} (DRAMA3_MODEL) — see docs/DRAMA3.md")
    if not server_built(): sys.exit(f"drama-tensorfold is not built: {ENGINE_BIN} / {PILOT} (DRAMA3_SERVER_DIR, DRAMA3_ENGINE_BIN)")
    if not os.path.exists(GPU_LOCK):                                         # pilot.py refuses a missing lock file
        os.makedirs(os.path.dirname(GPU_LOCK), exist_ok=True); open(GPU_LOCK, "a").close()
    ev = f"{RUNTIME}/runs/{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"      # pilot.py refuses an existing evidence dir
    argv = [sys.executable, PILOT, "--native-bin", ENGINE_BIN, "--checkpoint", CKPT, "--runtime-dir", RUNTIME,
            "--gpu-lock", GPU_LOCK, "--evidence-dir", ev, "--port", str(PORT), "--", *SERVE_ARGS]
    print("exec " + " ".join(argv), flush=True)
    os.execv(argv[0], argv)


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "show"
    if c == "serve": serve()
    elif c == "show":
        print(json.dumps({"checkpoint": CKPT, "local": local(), "fetch_hook": bool(FETCH_CMD), "server_built": server_built(),
                          "voice_tool_ready": voice_tool_ready(), "old_built": old_built(), "unit": UNIT, "url": URL, "serve_args": SERVE_ARGS}, indent=1))
    else: sys.exit(__doc__)
