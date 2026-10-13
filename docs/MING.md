# Ming-Image 0.1 Design (optional image model)

[Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design) (inclusionAI, MIT, 6B) makes posters,
UI screens, infographics and other designs with a lot of text in them, and can make PNGs with a transparent background.
The studio lists it in the Image builder's model menu as **Ming-Image 0.1 Design (posters, UI, text, transparent)** once it
finds the weights, the official code and the venv described below.

It does not run in ComfyUI. Each job starts one process (`app/ming_runner.py`) in Ming's own venv. The process loads the
model once, renders every variation, saves the PNGs and exits, so all of its memory comes back when the job ends.

## What a Ming job does

1. **Layout (optional, on by default).** *Director lays out the design* sends your words to the director with Ming's own
   released text-to-image rewriter prompt (`assets/t2i_rewriter_system_prompt.txt` in the Ming-Image code). The director
   returns a Figma-style JSON layout, layer by layer, and its `aspect_ratio` is set to the canvas you picked. Ming's makers
   name Qwen3.8-27B as one of the two models for this step. If the director does not answer, your words are used as written
   and the job notes say so. The layout is saved next to the images as `ming_layout.json`.
2. **Transparent background (optional).** The prompt starts with `RGBA, 4-channel, transparent background`, one of Ming's
   fixed RGBA phrases, and the PNG keeps its alpha channel.
3. **Making room.** ComfyUI is asked to unload its models (`POST /free`). The job then needs
   **about 58 GiB + 5 GiB per megapixel** of free memory (about 63 GiB at 1024×1024, 78 GiB at 2048×2048). If memory is
   still short and the studio manages the director (`DIRECTOR_SERVICE`), the director is stopped; it starts again on its next use.
   If memory is still short after that, the job fails with a message instead of starting.
4. **Render.** One Ming process renders every variation (seed, seed+1, …) at Ming's defaults for text to image
   (12 steps, CFG 1.0) and exits.

Every job's notes record the load time, the time per image, and the torch and transformers versions it ran on.

**Measured on one DGX Spark** (single runs, model read from the internal NVMe):

| | |
|---|---|
| loading the 53 GB checkpoint | 309–363 s |
| one 1024×1024 image | 21–23 s |
| one 2048×2048 image | 122 s |
| peak memory | about 55 GiB at 1024×1024, about 70 GiB at 2048×2048 |

The load dominates, so a job with 4 variations costs little more than a job with 1.

## Setup

You need three things: the weights, the official code and a Python environment. The commands below put them where the
studio looks by default (`MING_MODEL`, `MING_CODE` and `MING_PY` in `studio.env` change that).

```bash
cd lion-ai-studios && set -a && . ./studio.env && set +a
H="${STUDIO_HOME:-$HOME/lion-studio}"
COMFY_VENV="${COMFY_VENV:-$HOME/comfyui-env}"          # the venv scripts/install_comfyui.sh made

# 1. the official code (the studio was tested at commit 3515cd7)
git clone https://github.com/inclusionAI/Ming-Image "$H/ming/Ming-Image"
git -C "$H/ming/Ming-Image" checkout 3515cd7

# 2. a venv that borrows torch from ComfyUI's venv and pins Ming's own libraries on top
python3 -m venv "$H/envs/ming"
COMFY_SP=$("$COMFY_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
MING_SP=$("$H/envs/ming/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "$COMFY_SP" > "$MING_SP/zz_comfyui_torch.pth"     # appended to sys.path, so Ming's pins below win
"$H/envs/ming/bin/pip" install transformers==4.57.1 peft==0.17.0 huggingface-hub==0.34.0 diffusers==0.36.0 tokenizers==0.22.2 \
    accelerate safetensors
"$H/envs/ming/bin/python" -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"

# 3. the weights: 53 GB, MIT (huggingface.co/inclusionAI/Ming-Image-0.1-Design)
"$H/envs/ming/bin/python" -c "from huggingface_hub import snapshot_download as s; \
s('inclusionAI/Ming-Image-0.1-Design', local_dir='$H/models/Ming-Image-0.1-Design')"
```

Then restart the studio (`systemctl --user restart lion-studio`) and pick the model in the Image builder.

**Notes on the environment**

- The tested runtime was torch **2.10.0 + cu130** (from ComfyUI's venv) with transformers **4.57.1**, peft **0.17.0**,
  huggingface-hub **0.34.0**, diffusers **0.36.0** and tokenizers **0.22.2**, on Python 3.12.
- Ming's own `requirements.txt` pins torch 2.4.0 and transformer-engine. The studio runs it on torch 2.10 with CUDA 13
  (what the GB10 needs) and without transformer-engine; text to image worked at 1024×1024 and 2048×2048.
- Ming needs transformers 4.57, while ComfyUI's venv may carry a newer one. The `.pth` line borrows only what the Ming venv
  does not have itself. pip may print "Not uninstalling … outside environment"; that is expected and leaves ComfyUI alone.
- Because torch is shared, upgrading ComfyUI's venv changes the torch Ming runs on. Each job records the versions it used,
  so a change shows up in the job notes. If you prefer to keep them apart, install `torch==2.10.0 torchvision` from
  `https://download.pytorch.org/whl/cu130` into the Ming venv instead of writing the `.pth` file (about 6 GB more disk).
- Keep the weights on local NVMe. Loading 53 GB over network storage takes far longer.

Layer decomposition (the separate *Ming-Image-0.1-Design-Layer* model) is not wired into the studio.
