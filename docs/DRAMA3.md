# Fish Audio Drama 3 (optional voice engine) — Built with Fish Audio

[Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3) is a voice-cloning text-to-speech model. In Lion AI Studios it is
an optional second engine for the **narrator** (Video builder → voice-over → *Voice engine*): Qwen3-TTS stays the default and
the fast choice; Drama 3 sounds the most natural. The menu offers it only when the engine and the weights below are found.

There are two ways to run it, neither in ComfyUI, and the studio tries them in this order:

1. **The fast server** ([drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold), below under
   [Fast server](#fast-server-drama-tensorfold)): an on-demand loopback speech server the studio starts for a voice-over and
   stops when idle. Each cloned voice becomes a voice file once; every line is then one request. Tested on a DGX Spark: server
   ready in 10 s, a cloned 18-word line word-perfect in about 19 s including the Whisper check (the per-take build: about 39 s).
2. **The per-take build**, TensorFold's native CUDA port (`drama3-generate`, sections 2 and 3): one process per take, about
   18 s to load, about 42 GB of memory at its peak.

If neither works for a job, the voice-over is spoken with Qwen3-TTS and the job notes say why; the film is never failed for it.
The studio asks ComfyUI to unload its models first either way. Like Qwen3-TTS takes, every Drama 3 take is checked by Whisper
(`app/wordcheck.py`): a take that drops words, loses its ending or reads reference words aloud is drawn again (up to 3 takes),
and the job keeps the best one.

## Licence: read this first

The weights are under the **Fish Audio Community License v1.0** (in the model repo), not under this repository's MIT licence.
In short (the licence text decides, not this summary):

- **Research and non-commercial use** (personal, hobby, evaluation) is free.
- **Commercial use** is free only while you (with your affiliates) have **under US$500,000 annual revenue**; above that you
  need a written licence from Fish Audio.
- **A hosted service** (letting other people generate speech or clone voices from text or samples they supply, through an
  API, a website or an app) **always needs a separate licence from Fish Audio**, whatever your revenue. Do not put the
  studio's Drama 3 engine in front of other users.
- **Consent**: never make speech that imitates an identifiable real person without their prior consent (or another legal
  right to do so). The studio's voice cloner asks for that confirmation; see [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).
- Anything built with it must display **"Built with Fish Audio"**. The studio shows it in the engine menu and records it with each job that used it.

This repository does **not** redistribute the weights, the codec or the engine. You download and build them yourself.

## What you need

| | default path | setting in `studio.env` |
|---|---|---|
| the weights, [`fishaudio/drama-3`](https://huggingface.co/fishaudio/drama-3) @ `e0a9218a`, 41.5 GB | `$STUDIO_HOME/models/drama-3` | `DRAMA3_MODEL` |
| the engine: `bin/drama3-generate`, `lib/libtensorfold_drama3.so` | `~/.local/opt/tensorfold-studio/drama3` | `DRAMA3_DIR` |
| the converted codec `codec-v1.safetensors` (made once from the weights' `codec.pth`) | `~/.local/opt/tensorfold-studio/drama3` | `DRAMA3_DIR` |
| *or / and* the fast server: a built [drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold) checkout | `~/src/drama-tensorfold` | `DRAMA3_SERVER_DIR`, `DRAMA3_ENGINE_BIN` |
| its voice-file encoder: [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni) @ `ab095ef9` + a small venv | `~/src/sglang-omni`, `$STUDIO_HOME/envs/drama3-voice` | `DRAMA3_SGLANG_OMNI_DIR`, `DRAMA3_VOICE_ENV_PY` |

The fast server and the per-take build share the weights. You need at least one of the two engines; with both, the per-take
build is the fallback.

## 1. The weights

```bash
cd lion-ai-studios && set -a && . ./studio.env && set +a
python3 scripts/download_models.py --group drama3      # opt-in; --all leaves it out
# or: hf download fishaudio/drama-3 --revision e0a9218aa48888270f06ef1ed9ea67622c0e17d2 --local-dir "$STUDIO_HOME/models/drama-3"
```

If you would rather keep the 39 GiB of weights elsewhere while they are not in use, set the optional fetch hook
`DRAMA3_FETCH_CMD` in `studio.env` to a shell command that puts them back into `$DRAMA3_MODEL` (it runs with `DRAMA3_MODEL` in
its environment; output goes to `$STUDIO_HOME/drama3/fetch.log`), for example an `rsync` from your own storage or the `hf download`
line above. A Drama 3 voice-over that finds no weights runs the hook first; with no hook it reports Drama 3 as not installed and
speaks with Qwen3-TTS.

## 2. The engine (TensorFold drama3-cuda, on a DGX Spark)

The CUDA port lives on the `drama3-cuda` branch of [drowzeys/TensorFold](https://github.com/drowzeys/TensorFold) (a fork of
[ashhart/TensorFold](https://github.com/ashhart/TensorFold); the branch is Apache-2.0). The studio was built and tested at
commit `81173e6c`. You need **Zig 0.17** ([ziglang.org/download](https://ziglang.org/download/)) and **CUDA 13.0** (`nvcc`);
the GB10 is `sm_121`. The steps follow the branch's own `tools/drama3/build_local.sh`, with paths of your choice.

```bash
git clone -b drama3-cuda https://github.com/drowzeys/TensorFold.git ~/src/tf-drama3
cd ~/src/tf-drama3 && git checkout 81173e6c
B=~/src/d3build; F=$B/fatbin; OUT=~/.local/opt/tensorfold-studio/drama3; ZIG=/path/to/zig-0.17.0/zig
export PATH=/usr/local/cuda/bin:$PATH ZIG_LOCAL_CACHE_DIR=$B/cache/zig-local ZIG_GLOBAL_CACHE_DIR=$B/cache/zig-global
mkdir -p $F

# a) every other kernel fatbin for sm_121, through the build's own "fatbins" step
$ZIG build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast --prefix $B/fatbuild -j4 fatbins
cp -n $B/fatbuild/fatbin/*.fatbin $F/

# b) the three Drama 3 fatbins with nvcc (the flags of tools/drama3/build_local.sh)
FL="-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
    --expt-relaxed-constexpr -std=c++20 -O3 -gencode=arch=compute_121,code=sm_121"
nvcc -fatbin $FL -o $F/drama3.fatbin zig/kernels/cuda/drama3.cu
nvcc -fatbin $FL -diag-suppress 177 -o $F/drama3_fp8.fatbin zig/kernels/cuda/drama3_fp8.cu
nvcc -fatbin $FL -o $F/drama3_codec.fatbin zig/kernels/cuda/drama3_codec.cu

# c) the generator and the library
$ZIG build -Dfatbins=$F -Dsm=121 -Doptimize=ReleaseFast --prefix $OUT -j8 drama3-generate drama3-lib
ls $OUT/bin/drama3-generate $OUT/lib/libtensorfold_drama3.so
```

## 3. The codec (one-time conversion)

`tools/drama3/codec_convert.py` turns the weights' `codec.pth` (Fish's modded DAC) into `codec-v1.safetensors` for the CUDA
codec. It runs on the CPU in a small venv of its own, and it needs **Fish's reference codec code**: it imports `build_codec`
from a module `drama3_ref` found in `ORACLE_DIR`. Upstream that module is the TensorFold author's **private** `drama3_ref`
oracle, which is not published. This studio's build used a **stand-in**: a short `drama3_ref.py` that builds the codec from
the copy of Fish's code vendored in [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni) (Apache-2.0,
tested at `49dc4ba9`, `sglang_omni/models/fishaudio_s2_pro/fish_speech`), the same way sglang-omni's own `load_codec` does.
The converted codec was checked end to end: Drama 3 takes made with it were transcribed word for word by Whisper.

```bash
python3 -m venv ~/drama3-env && P=~/drama3-env/bin/pip
$P install -U pip wheel && $P install torch --index-url https://download.pytorch.org/whl/cpu
$P install safetensors numpy hydra-core omegaconf einops "descript-audiotools==0.7.2" "descript-audio-codec==1.0.0"
git clone https://github.com/sgl-project/sglang-omni.git ~/src/sglang-omni && git -C ~/src/sglang-omni checkout 49dc4ba9
mkdir -p ~/src/drama3-oracle      # put the stand-in below into ~/src/drama3-oracle/drama3_ref.py
CUDA_VISIBLE_DEVICES= ORACLE_DIR=~/src/drama3-oracle SGLANG_OMNI_DIR=~/src/sglang-omni \
  ~/drama3-env/bin/python ~/src/tf-drama3/tools/drama3/codec_convert.py "$STUDIO_HOME/models/drama-3" \
  ~/.local/opt/tensorfold-studio/drama3/codec-v1.safetensors
```

The stand-in (`drama3_ref.py`; written for this studio, not part of TensorFold):

```python
"""Stand-in for TensorFold's private drama3_ref oracle (build_codec only): Fish's modded DAC from the fish_speech copy
vendored in sglang-omni, built as sglang-omni's load_codec builds it, without importing the sglang_omni package."""
import os, sys
import torch

_PKG = os.path.join(os.path.expanduser(os.environ.get("SGLANG_OMNI_DIR", "~/src/sglang-omni")), "sglang_omni/models/fishaudio_s2_pro")
sys.path.insert(0, _PKG)


def build_codec(codec_dir, device):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    text = open(os.path.join(_PKG, "fish_speech", "configs", "modded_dac_vq.yaml")).read()
    codec = instantiate(OmegaConf.create(text.replace("sglang_omni.models.fishaudio_s2_pro.fish_speech.", "fish_speech.")))
    sd = torch.load(os.path.join(codec_dir, "codec.pth"), map_location=str(device), mmap=True, weights_only=True)
    missing, unexpected = codec.load_state_dict(sd, strict=False, assign=True)
    print(f"[drama3_ref] codec.pth: {len(sd)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", file=sys.stderr)
    return codec.eval().to(device)
```

A good conversion reports `missing 0` (the six `unexpected` buffers, `freqs_cis` and `causal_mask`, are rebuilt at run time) and
writes 397 tensors. Run the engine's own `--codec-check` once on the first take if you change anything here.

## 4. Use it

Restart the studio (or reload the page). The narrator's **Voice engine** menu now enables *Fish Audio Drama 3*. Pick a cloned
voice, write the lines, choose the engine and render: the job notes record the engine, the path used (`say3s` = the fast
server, `say3` = the per-take build) and Whisper's word check of the take. If no engine is built, or the weights are missing
and no fetch hook is set, the option stays greyed out and Qwen3-TTS is used.

## Fast server (drama-tensorfold)

[plotarmordev/drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold) (Apache-2.0) patches
[ashhart/TensorFold](https://github.com/ashhart/TensorFold) to run Drama 3 and starts it as a local speech server with an
OpenAI-style `POST /v1/audio/speech` route. The studio was tested with drama-tensorfold at
`3f44a09ce245855f2e3b8f956ee0c44f591afb81` (engine: ashhart/TensorFold `c7137bb3` plus its patches).

> **Status: testing, not qualified.** drama-tensorfold's own README says so: the native engine is faster than the sglang-omni
> reference, but not all of its accuracy bars against that reference are met (backbone cosines, hidden-state L2, semantic KL,
> one-frame codec decode), and a long text can stall in silence (the server ends such a reply after 10 s with
> `X-Finish-Reason: silence`). The studio re-draws any reply that does not end in `stop` and word-checks every take, but treat
> its output as a test build. Read drama-tensorfold's [status](https://github.com/plotarmordev/drama-tensorfold#status) before
> relying on it.

How the studio uses it, per Drama 3 voice-over:

1. **Weights**: if `DRAMA3_MODEL` has none, the optional fetch hook runs (see section 1), else Drama 3 is skipped.
2. **ComfyUI** is asked to unload its models.
3. **Voice file** (`voice.py voice3`): the cloned voice's `ref.wav` and its transcript become a `drama3-voice/1` file in
   `$STUDIO_HOME/drama3/voices/` (the clip's codes from Fish's codec, made by the engine tree's `tools/zig/drama_voice.py` with
   sglang-omni's reference encoder; ~2.6 GB of GPU memory, 10-20 s). It is made once per voice and reused while the clip and the
   transcript stay the same.
4. **Server**: the studio starts `DRAMA3_SERVICE` if no server answers on `127.0.0.1:DRAMA3_PORT`. The launcher
   (`recipe/pilot.py`) admits a load only with **about 88 GiB free** and caps the server at 54 GiB; the studio first frees its own
   memory holders (ComfyUI, the optional LocalRouter, the director), never other services.
5. **Speech** (`voice.py say3s`): each line goes to the server with the voice file's transcript and codes inline (no restart per
   voice); Whisper checks every reply, and a reply with missing words or one that ends in `length` or `silence` is drawn again.
6. **Idle stop**: the unit is stopped `DRAMA3_IDLE_S` (120 s) after the last voice-over, or at once when a render starts.

Fallbacks: the server path fails → the per-take build (if built) → Qwen3-TTS, each with the reason in the job notes.

### Build the server

You need Linux with user systemd, CUDA 13 (`nvcc`), [Zig 0.17.0](https://ziglang.org/download/) and the weights from section 1
(verify them with the checksums in the checkout).

```bash
git clone https://github.com/plotarmordev/drama-tensorfold.git ~/src/drama-tensorfold
cd ~/src/drama-tensorfold && git checkout 3f44a09ce245855f2e3b8f956ee0c44f591afb81
(cd "$STUDIO_HOME/models/drama-3" && sha256sum -c ~/src/drama-tensorfold/recipe/checksums/checkpoint.sha256)
ZIG=/path/to/zig-0.17.0/zig NVCC=/usr/local/cuda/bin/nvcc scripts/build-engine.sh
ls build/tensorfold/zig-out/native/bin/tensorfold-native build/tensorfold/tools/zig/drama_voice.py
```

`scripts/build-engine.sh` fetches TensorFold at the pinned commit, applies the patches and builds the engine into
`build/tensorfold/`. If you put the checkout elsewhere, set `DRAMA3_SERVER_DIR` (and `DRAMA3_ENGINE_BIN` if the binary moves).

### The voice-file environment

`drama_voice.py` encodes a clip with sglang-omni's Fish reference encoder, so it needs a git checkout of
[sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni) (Apache-2.0) at the revision drama-tensorfold pins
(`ab095ef9`) and a few packages beside the TTS venv's CUDA torch. The venv below reuses the TTS venv's packages through a `.pth`
file and leaves the TTS venv itself unchanged; `requirements/drama3-voice.txt` pins every package and must be installed with
`--no-deps` (the descript packages would otherwise bring their own torch and old pins).

```bash
cd lion-ai-studios && set -a && . ./studio.env && set +a
git clone https://github.com/sgl-project/sglang-omni.git ~/src/sglang-omni
git -C ~/src/sglang-omni checkout ab095ef9e779e56af33b6ed2c863645c9903372d
"$STUDIO_TTS_PY" -m venv "$STUDIO_HOME/envs/drama3-voice"
SP=$("$STUDIO_HOME/envs/drama3-voice/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
"$STUDIO_TTS_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' > "$SP/zz_tts_env_overlay.pth"
"$STUDIO_HOME/envs/drama3-voice/bin/pip" install --no-deps -r requirements/drama3-voice.txt
"$STUDIO_HOME/envs/drama3-voice/bin/python" -c "import torch, hydra, dac, audiotools; print(torch.__version__, torch.cuda.is_available())"
```

### The unit and the settings

```bash
sed "s#@REPO@#$PWD#g" systemd/studio-drama3.service.example > ~/.config/systemd/user/studio-drama3.service
systemctl --user daemon-reload
```

Then in `studio.env` set `DRAMA3_SERVICE=studio-drama3.service` (and any paths that differ from the defaults:
`DRAMA3_SERVER_DIR`, `DRAMA3_ENGINE_BIN`, `DRAMA3_PORT`, `DRAMA3_VOICE_ENV_PY`, `DRAMA3_SGLANG_OMNI_DIR`, `DRAMA3_GPU_LOCK`,
`DRAMA3_IDLE_S`) and restart the studio. `DRAMA3_GPU_LOCK` is the launcher's shared GPU lock: if your machine already has one
that its other GPU jobs take, point it there; otherwise the studio creates `$STUDIO_HOME/drama3/gpu.lock`. With
`DRAMA3_SERVICE` empty, the studio uses the fast server only while one already answers on `DRAMA3_PORT` (for example one you
started by hand) and never starts or stops it.

### Test it

```bash
cd lion-ai-studios && set -a && . ./studio.env && set +a
python3 app/drama3.py show                                   # every path, and what is present
systemctl --user start studio-drama3.service                 # needs ~88 GiB free
until curl -sf http://127.0.0.1:${DRAMA3_PORT:-18793}/health >/dev/null; do sleep 2; done; echo ready
curl -sS http://127.0.0.1:${DRAMA3_PORT:-18793}/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "fishaudio/drama-3", "input": "The train leaves at seven fifteen."}' -D - -o /tmp/hello.wav | grep -i x-finish-reason
# a cloned voice (pick a folder under $STUDIO_HOME/voices/): make its voice file, then speak through the server
V=$STUDIO_HOME/voices/<voice-id>
"$STUDIO_TTS_PY" app/voice.py voice3 "$V" "$(basename "$V")"
"$STUDIO_TTS_PY" app/voice.py say3s "$V" "This line was spoken by the fast server and checked by Whisper." /tmp/say3s.wav "$(basename "$V")"
systemctl --user stop studio-drama3.service
```

`say3s` prints one JSON line: `"path": "say3s"`, the finish reason of each piece and Whisper's word check (`"action": "ok"` when
every word was heard). In the studio, a voice-over with *Voice engine: Fish Audio Drama 3* then records
`Fish Audio Drama 3 · speech server (drama-tensorfold)` in its job.
