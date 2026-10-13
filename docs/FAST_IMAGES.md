# Fast images with LocalRouter (optional)

[LocalRouter](https://github.com/jayleaton/localrouter) (Apache-2.0) is a small daemon with hand-built engines for the
DGX Spark. One of them runs **Qwen-Image 2.1 Turbo** in FP8 with the checkpoint's own 8 steps, the same model the studio
runs in ComfyUI. The studio can use it as a second image engine, **Fast (LocalRouter)**, in the Image builder.

It is **off by default**. Nothing changes until you set `LOCALROUTER_ENABLE=1` in `studio.env`.

## What Fast does, and what it does not

- **Only plain text to image with Qwen-Image 2.1 Turbo at 1024×1024, one image per job.** That is the only case that was
  measured. Picking *Fast* sets the model, the size and *Variations* to match. A note under the menu says when a job
  does not fit, and such a job renders on ComfyUI as before. Edits, other sizes, 2K, several variations, other models,
  Ming-Image and the Video builder's scout images always use ComfyUI.
- **It falls back to ComfyUI on its own.** A Fast job falls back with the same model, size and seed whenever room cannot
  be made, LocalRouter does not come up, or it answers with an error (for example 503 `insufficient_memory`), times out
  or returns something other than one PNG. The reason goes into the job's notes. LocalRouter runs FP8 and ComfyUI runs
  bf16, so the same seed does not give the same pixels on both.
- **It starts on demand and stops when idle.** For a Fast job the studio starts LocalRouter (`LOCALROUTER_SERVICE` or
  `LOCALROUTER_START`) and waits up to about 90 s for its `/health`. LocalRouter's own `idle_ttl_s` unloads the model after
  5 minutes, and the studio stops LocalRouter after **10 minutes** without a Fast job (`LOCALROUTER_IDLE_S`). It never stops
  it while a Fast job runs or is queued, or while LocalRouter reports a model loading or busy.
- **It gives way.** Before any other job (video, ComfyUI images, Ming, voices) a warm LocalRouter unloads its model
  through its own `POST /v1/tools/release`, so it does not hold about 36 GiB under a render.

### The memory broker

`app/studio_broker.py` keeps one list of what holds the Spark's unified memory: LocalRouter, ComfyUI, the director, a Ming
process and a voice job, each with what it needs, a priority and how to release it. `make_room(need, …)` frees *idle*
holders, lowest priority first, until `MemAvailable` reaches the need. It never touches a holder that is busy (a ComfyUI
queue that is running or pending, a director request in flight) or whose state it cannot read, and it fails closed: if it
cannot read `/proc/meminfo` or cannot reach the need, it refuses and says why.

For a Fast job only ComfyUI's idle models may give way. LocalRouter's own estimate decides the need: about **50.7 GiB from
cold** (its 36,000 MiB resident estimate, the working memory and its reserve) and about **15.5 GiB when the model is
already loaded**. If ComfyUI cannot make enough room, the job falls back to ComfyUI rather than stopping the director.
The voice job's 8 GiB check also goes through the broker; it releases nothing. The Ming path keeps its own room-making.
`/api/state` shows the broker's holders and its last few actions under `broker`.

## Measured

One DGX Spark, Qwen-Image 2.1 Turbo, 1024×1024, 8 steps, 6 prompts × 2 seeds per engine (n = 12 each), wall time per image:

| | warm | first image (cold) | peak memory |
|---|---:|---:|---:|
| **Fast (LocalRouter, FP8)** | **6.2 s** (6.15–6.18) | about 12–13 s (12.05 s measured with the daemon already running) | 42.6 GB |
| ComfyUI (bf16) | 9.3 s (9.15–9.48) | 34.6 s | 24.4 GB |

The cold time is the model load with LocalRouter's daemon already up. Starting its container first adds time that was not
measured. Image quality was judged the same at this size.

## Setup

LocalRouter has its own installer, which downloads the official checkpoint, converts it and builds the Docker image.
Follow its README ([Install on a DGX Spark](https://github.com/jayleaton/localrouter#install-on-a-dgx-spark));
`MODELS=image` is enough for the studio (about 110 GB free during the install). Its own docs cover the weights, the
precisions and prebuilt packs. Then choose one of these two ways to run it.

### A. On demand, started and stopped by the studio (recommended)

The installer starts LocalRouter as an always-on service that keeps the Turbo model loaded (`keep_loaded`). On a Spark
that also renders video, that holds about 36 GiB all the time. To run it only when the studio needs it:

```bash
docker compose -p localrouter down                 # stop the installer's always-on service (see its README)
LR_HOME=/data/localrouter                          # wherever the installer put weights/, data/ and tools.json
cp docs/localrouter-tools.example.json "$LR_HOME/tools.studio.json"
sed -e "s#@LOCALROUTER_HOME@#$LR_HOME#g" -e "s#@LOCALROUTER_IMAGE@#localrouter:dev#g" \
    systemd/lion-localrouter.service.example > ~/.config/systemd/user/lion-localrouter.service
systemctl --user daemon-reload                     # no enable: the studio starts it when a Fast job needs it
```

`docs/localrouter-tools.example.json` serves only the Turbo model, does not keep it loaded (`keep_loaded: false`,
`idle_ttl_s: 300`) and reserves 15 GiB for everything else. Check the image name with `docker images` and the
`weights` path against your installer's `tools.json`. Then in `studio.env`:

```bash
LOCALROUTER_ENABLE=1
LOCALROUTER_SERVICE=lion-localrouter.service
```

and restart the studio. Instead of a unit you can give two shell commands, `LOCALROUTER_START` (it must return once
LocalRouter is starting, e.g. `docker start …` or `docker run -d …`) and `LOCALROUTER_STOP`.

### B. Your own always-on LocalRouter

Set only `LOCALROUTER_ENABLE=1` (and `LOCALROUTER_URL` if it is not on `127.0.0.1:8190`). The studio then uses LocalRouter
whenever it answers and never starts or stops it; when it does not answer, Fast jobs render on ComfyUI. Set
`keep_loaded` to `false` for the Turbo model in its `tools.json`: otherwise LocalRouter reloads the model whenever memory
frees, which competes with ComfyUI.

### Settings

| key | default | meaning |
|---|---|---|
| `LOCALROUTER_ENABLE` | `0` | turn the Fast engine on |
| `LOCALROUTER_URL` | `http://127.0.0.1:8190` | where LocalRouter answers |
| `LOCALROUTER_MODEL` | `qwen-image-2.1-turbo` | LocalRouter's id for Qwen-Image 2.1 Turbo |
| `LOCALROUTER_SERVICE` | empty | systemd --user unit the studio starts and stops |
| `LOCALROUTER_START` / `LOCALROUTER_STOP` | empty | shell commands used instead of a unit |
| `LOCALROUTER_IDLE_S` | `600` | stop it after this long without a Fast job (`0` = never) |
| `LOCALROUTER_RESERVE_GIB` | `15` | the `reserve_bytes` of its `tools.json`, in GiB |

LocalRouter has no authentication. Keep it on `127.0.0.1`, as the example unit does.

## Limits and open points

- Sizes other than 1024×1024 and more than one image per job are switched off in `app/workflows.py` (`LR_GO_SIZES`,
  `LR_GO_N`) until they are measured. LocalRouter itself accepts sides from 256 to its `max_side` in steps of 16 and 1 to 4
  images; how it seeds images 2 to n was not checked.
- LocalRouter's resident estimate (about 37.7 GB) is below the measured 42.6 GB peak. Its reserve covers the difference at
  1024×1024; larger sizes with the director loaded were not tested.
- The broker reads `MemAvailable`, which counts reclaimable page cache. If a large page cache makes a cold load fail,
  LocalRouter answers with an error and the job falls back to ComfyUI.
- A busy ComfyUI (any running or pending prompt, from any client) is never freed, so a cold Fast job may fall back then.
- The idle stop applies to the configured unit or stop command, whoever started LocalRouter.
- LocalRouter keeps its own copy of every output under its `data/jobs` for `keep_outputs_s`; the studio keeps its own copy
  in the library.
