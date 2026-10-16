# Notes for coding agents

Guidance for AI coding agents (and people) changing this repository. Read [CONTRIBUTING.md](CONTRIBUTING.md) too.

## What this is

A local media studio for one NVIDIA DGX Spark: a browser page, a small Python web server with a job queue, and ComfyUI as the
renderer, plus a director model and a few helper runners. See [How it works](README.md#how-it-works).

## Layout

| path | what it is |
|---|---|
| `app/config.py` | every machine-specific setting, read from the environment (`studio.env`); standard library only |
| `app/studio_server.py` | the web server, queue, worker and job runners; standard library only |
| `app/workflows.py` | ComfyUI API graphs, presets, image-model lists and estimates |
| `app/director.py` | talks to the director (an OpenAI-compatible chat server with video input) |
| `app/studio_broker.py` | the memory broker: who holds memory, and freeing idle holders before a big load |
| `app/voice.py`, `app/midi_tool.py`, `app/ming_runner.py` | helpers that run in their own Python environments |
| `app/music.py`, `app/stitch.py` | music graphs and ffmpeg joins |
| `app/negation.py`, `app/shot.py`, `app/wordcheck.py` | prompt negation check, frame-level shot check, spoken-word check; Apache-2.0 files from another repository (see NOTICE.md), keep their headers. `shot.py` needs numpy/Pillow only when it runs: the server calls it in-process if those exist, else in the voice venv |
| `app/index.html` | the whole page: HTML, CSS and JavaScript in one file, served fresh on every request |
| `comfyui_nodes/` | the studio's image node for ComfyUI |
| `scripts/` | install, download, helper environments, services, and `check.sh` |
| `systemd/` | unit templates; `*.example` files are examples with placeholders and are never installed by the scripts |
| `docs/`, `samples/`, `benchmarks/` | documentation, example outputs, raw measurements |

## Rules

1. **Config indirection.** Paths, URLs, ports, service names and shell commands come from `app/config.py`, which reads
   `studio.env`. Add a new key there with a safe default, and a commented line in `studio.env.example`. Never hard-code a
   machine's values.
2. **Nothing machine-specific or personal** in code, comments, docs or commit messages: no home-directory paths, host names,
   private addresses, user names, e-mail addresses or tokens. Use `$HOME`, `~` or a setting.
3. **Standard library only** in `studio_server.py`, `config.py`, `workflows.py`, `director.py`, `studio_broker.py` and the
   other modules the server imports. Code that needs packages runs as a subprocess in its own venv (pattern: `voice.py`,
   `ming_runner.py`) and passes results back as files and one JSON line.
4. **Optional means off.** A model is listed only when its files are present; an engine that needs extra software stays off
   until configured, and falls back to ComfyUI when it cannot do a job.
5. **Keep the guard rails**: consent checks for Cast, Recreate and voice cloning; `127.0.0.1` by default; the memory guard
   before loading models; the thermal guard between segments; asking before stopping another GPU service.
6. **Match the existing style**: compact functions, short comments that say why, user-facing messages in plain words.
   Element ids in `index.html` are used by the script; do not rename them casually.
7. **Measured numbers only.** Times and memory figures in docs and comments come from real runs; say where and how many.
8. **Document changes**: a dated [CHANGELOG.md](CHANGELOG.md) entry, and README / `docs/` updates when behaviour, settings
   or requirements change.

## Checks

```bash
bash scripts/check.sh      # py_compile on every .py, node --check on the page's scripts, hygiene grep
```

CI runs the same script. Most logic can be tested without a GPU: import the modules with a throw-away `STUDIO_HOME` and stub
the HTTP endpoints (ComfyUI, the director, LocalRouter) with a local `http.server`.

## Do not

- Commit `studio.env`, models, renders, `__pycache__/` or anything under a `STUDIO_HOME`.
- Add network services that listen beyond `127.0.0.1` by default, or features that call cloud APIs.
- Start, stop or restart services, download models or change system packages as a side effect of a code change.
