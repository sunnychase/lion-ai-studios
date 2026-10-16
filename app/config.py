# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
"""config.py — every machine-specific setting of Lion AI Studios, read from the environment.

Copy studio.env.example to studio.env and edit it; the systemd unit (or `set -a; . ./studio.env`) loads it.
Nothing here needs third-party packages, so the helper workers (voice.py, midi_tool.py) can import it from their own venvs."""
import os

E = os.environ.get


def _p(v):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(v)))


# ── the studio itself ──────────────────────────────────────────────────────────────────────────────────────
STUDIO_HOME = _p(E("STUDIO_HOME", "~/lion-studio"))       # library/, state/, uploads/, voices/, models/, envs/ live here
HOST = E("STUDIO_HOST", "127.0.0.1")                       # keep it on localhost; put an authenticating proxy in front for remote use
PORT = int(E("STUDIO_PORT", "8482"))
MODELS = os.path.join(STUDIO_HOME, "models")               # the studio's own models: TTS, Whisper, soundfonts

# ── ComfyUI (the renderer) ─────────────────────────────────────────────────────────────────────────────────
COMFY_DIR = _p(E("COMFY_DIR", "~/ComfyUI"))                # the ComfyUI checkout (input/ and output/ are used directly)
COMFY_URL = E("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_MODELS = _p(E("COMFY_MODELS", os.path.join(COMFY_DIR, "models")))
COMFY_SERVICE = E("COMFY_SERVICE", "comfyui.service")      # systemd --user unit the studio may start; empty = never start it

# ── the director (an OpenAI-compatible chat server with video input) ───────────────────────────────────────
DIRECTOR_URL = E("DIRECTOR_URL", "http://127.0.0.1:8486").rstrip("/")
DIRECTOR_MODEL = E("DIRECTOR_MODEL", "studio-director")   # the served model name
DIRECTOR_LABEL = E("DIRECTOR_LABEL", "the director")      # how the page names it
DIRECTOR_SERVICE = E("DIRECTOR_SERVICE", "")              # optional systemd --user unit started on first use and stopped to make room
# optional second, text-only writer that is preferred for writing whenever it answers (e.g. a bigger LLM you already run)
WRITER_URL = E("WRITER_URL", "").rstrip("/")
WRITER_MODEL = E("WRITER_MODEL", "")
WRITER_LABEL = E("WRITER_LABEL", "the writer")

# ── optional: another GPU service the studio may pause (ask first, then stop; offer to start it again after) ──
TENANT_NAME = E("GPU_TENANT_NAME", "")                     # e.g. "the vLLM server"; empty = feature off
TENANT_CHECK = E("GPU_TENANT_CHECK", "")                   # shell command, exit 0 = it is running
TENANT_STOP = E("GPU_TENANT_STOP", "")                     # shell command that stops it
TENANT_START = E("GPU_TENANT_START", "")                   # shell command that starts it again

# ── guards ─────────────────────────────────────────────────────────────────────────────────────────────────
NEED_GIB = float(E("STUDIO_NEED_GIB", "55"))               # free unified memory a render needs before ComfyUI is up
HOT_C = float(E("STUDIO_HOT_C", "95"))                     # pause between segments at this board temperature...
COOL_C = float(E("STUDIO_COOL_C", "85"))                   # ...and resume below this one
# memory guard while a ComfyUI render runs (the Spark's GPU and system share one pool; an overrun freezes the whole box):
MEM_FLOOR_GIB = float(E("STUDIO_MEM_FLOOR_GIB", "6"))      # below this much free memory the render is interrupted and the job fails with the reason
MEM_CRIT_GIB = float(E("STUDIO_MEM_CRIT_GIB", "3"))        # below this (or falling to it within ~1 s) ComfyUI is also stopped outright

# ── helper environments ────────────────────────────────────────────────────────────────────────────────────
TTS_PY = _p(E("STUDIO_TTS_PY", os.path.join(STUDIO_HOME, "envs/tts/bin/python")))     # qwen-tts, demucs, Whisper, yt-dlp
MIDI_PY = _p(E("STUDIO_MIDI_PY", os.path.join(STUDIO_HOME, "envs/midi/bin/python")))  # mido + tinysoundfont
SF2 = _p(E("STUDIO_SF2", os.path.join(MODELS, "soundfonts/MuseScore_General.sf2")))
TF_LIB = _p(E("TENSORFOLD_LIB", os.path.join(COMFY_DIR, "custom_nodes/tensorfold_h3/libtf_h3.so")))  # optional engine
DENO_BIN = E("DENO_BIN", "")                               # a Deno binary for yt-dlp's YouTube JS challenges; empty = `deno` on PATH, if any
TRASH = os.path.join(STUDIO_HOME, "trash")                  # deleted renders and voices wait here until the Trash is emptied

# ── optional: Fish Audio Drama 3 voice engine (TensorFold's CUDA port; see docs/DRAMA3.md) — Built with Fish Audio ──────────────────
# The narrator's "Voice engine" menu offers it only when the engine and the weights are both found.
DRAMA3_DIR = _p(E("DRAMA3_DIR", "~/.local/opt/tensorfold-studio/drama3"))   # bin/drama3-generate, lib/libtensorfold_drama3.so, codec-v1.safetensors
DRAMA3_MODEL = _p(E("DRAMA3_MODEL", os.path.join(MODELS, "drama-3")))       # huggingface.co/fishaudio/drama-3 (Fish Audio Community License)

# ── optional: Ming-Image-0.1-Design (runs outside ComfyUI, in its own venv; see docs/MING.md) ──────────────────────────────────
# The Image menu lists it only when the weights, the official code and the venv are all found.
MING_MODEL = _p(E("MING_MODEL", os.path.join(MODELS, "Ming-Image-0.1-Design")))   # huggingface.co/inclusionAI/Ming-Image-0.1-Design
MING_CODE = _p(E("MING_CODE", os.path.join(STUDIO_HOME, "ming/Ming-Image")))      # a checkout of github.com/inclusionAI/Ming-Image
MING_PY = _p(E("MING_PY", os.path.join(STUDIO_HOME, "envs/ming/bin/python")))     # the venv that runs it (torch + Ming's pins)
MING_RUNNER = _p(E("MING_RUNNER", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ming_runner.py")))

# ── optional: the "Fast" image engine, LocalRouter (github.com/jayleaton/localrouter; see docs/FAST_IMAGES.md) — off by default ──
# Plain Qwen-Image 2.1 Turbo text to image at 1024x1024, one image per job; anything it cannot take renders on ComfyUI as before.
LR_ENABLE = E("LOCALROUTER_ENABLE", "0").strip().lower() in ("1", "true", "yes", "on")
LR_URL = E("LOCALROUTER_URL", "http://127.0.0.1:8190").rstrip("/")
LR_MODEL = E("LOCALROUTER_MODEL", "qwen-image-2.1-turbo")  # LocalRouter's id for Qwen-Image 2.1 Turbo (its tools.json)
LR_SERVICE = E("LOCALROUTER_SERVICE", "")                  # systemd --user unit the studio starts for a Fast job and stops when idle
LR_START = E("LOCALROUTER_START", "")                      # or shell commands that start / stop it (used when LOCALROUTER_SERVICE is empty);
LR_STOP = E("LOCALROUTER_STOP", "")                        # with none of the three, the studio only uses it while it already answers
LR_IDLE_S = int(E("LOCALROUTER_IDLE_S", "600"))            # stop it after this long without a Fast job (0 = never)
LR_RESERVE_GIB = float(E("LOCALROUTER_RESERVE_GIB", "15")) # the reserve_bytes of its tools.json, in GiB
