# Fish Audio Drama 3 (optional voice engine) — Built with Fish Audio

[Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3) is a voice-cloning text-to-speech model. In Lion AI Studios it is
an optional second engine for the **narrator** (Video builder → voice-over → *Voice engine*): Qwen3-TTS stays the default and
the fast choice; Drama 3 sounds the most natural. The menu offers it only when the engine and the weights below are found.

It runs on **TensorFold's native CUDA port** of Drama 3 (`drama3-generate`), not in Python and not in ComfyUI. Each take is
one process: about 18 s to load, about 42 GB of memory at its peak, so the studio asks ComfyUI to unload its models first.
Like Qwen3-TTS takes, every Drama 3 take is checked by Whisper (`app/wordcheck.py`): a take that drops words, loses its
ending or reads reference words aloud is drawn again with a new seed (up to 3 takes), and the job keeps the best one.

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

## 1. The weights

```bash
cd lion-ai-studios && set -a && . ./studio.env && set +a
python3 scripts/download_models.py --group drama3      # opt-in; --all leaves it out
# or: hf download fishaudio/drama-3 --revision e0a9218aa48888270f06ef1ed9ea67622c0e17d2 --local-dir "$STUDIO_HOME/models/drama-3"
```

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
voice, write the lines, choose the engine and render: the job notes record the engine and Whisper's word check of the take.
If the engine or the weights are missing, the option stays greyed out and Qwen3-TTS is used.
