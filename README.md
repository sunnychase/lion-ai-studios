<p align="center"><img src="docs/logo.png" width="148" alt="Lion AI Studios"></p>

<h1 align="center">Lion AI Studios</h1>
<p align="center"><b>Professional Media Suite for the NVIDIA DGX Spark</b><br>
Video with sound · images · music · voices · a director that writes and reviews every shot. Everything runs locally on one Spark.</p>

<p align="center"><img src="docs/screenshots/desktop-video.png" width="100%" alt="The Video builder"></p>

## What it does

| Builder | What you get |
|---|---|
| **Video** | **MiniMax H3** makes picture and sound together. Start from text, from scout images, from your own first frame, or from several keyframes that become one film. Films up to **5 minutes** are chained from 15 s segments; every seam carries the previous segment's last frames *and* audio, so cuts are invisible. Four presets: Draft, **Fast HD** (default), Standard, High. |
| **Image** | Ten image models behind one menu: Z-Image-Turbo, Qwen-Image 2.1 Turbo (2K, best at text), Krea 2, FLUX.2 klein, HiDream-O1, Chroma and more. **Edit** any image by describing the change (Qwen-Image 2.1 Turbo, up to 4 input images). |
| **Music** | **Fast**: ACE-Step 1.5 Turbo, a minute of music in seconds, with lyrics. **Best**: MiniMax Music 3. **MIDI**: upload a `.mid`, pick an instrument for every part (all 128 General MIDI instruments and 8 drum kits), and optionally let the AI restyle it. |
| **Director** | A vision-language model (Qwen3.8-27B) turns one sentence into a shot prompt with one beat per segment, **watches every rendered segment**, scores it, and re-takes a weak one once. |
| **Voices** | Clone a voice from a short clip (Qwen3-TTS) for narration or on-camera lines. |
| **Cast** | Put the person from up to 3 reference photos into the video (H3 reference-to-video, the official recipe). Consent checkbox required. |
| **Recreate a Short** | Paste a YouTube Short. The director watches it shot by shot and re-shoots it with your cast, keeping the timing and, optionally, the spoken words. |
| **Editor** | Re-take any segment, re-score the film, and export with crossfaded seams. |
| **Library / Queue** | Everything you make, filterable. Renders keep going if you close the page. |

A **TensorFold** int8 engine for FastH3 is optional and makes Draft **1.7–1.9x** and Fast HD **1.4x** faster ([benchmarks](docs/BENCHMARKS.md)).

## Samples

All of these were made with this repo on one DGX Spark. Prompts and settings are in [samples/README.md](samples/README.md).

| | |
|---|---|
| [<img src="samples/video/lion-roar-fasthd-10s.jpg" width="100%">](samples/video/lion-roar-fasthd-10s.mp4)<br>**Lion roar**: 10 s Fast HD with sound, written by the director from one sentence (review 8/10) | [<img src="samples/video/paper-boat-film-fasthd-33s.jpg" width="100%">](samples/video/paper-boat-film-fasthd-33s.mp4)<br>**Paper boat**: a 33 s film of 3 segments joined seamlessly, picture and sound continuous across the joins |
| <img src="samples/images/marquee-qwen-image-2.1-turbo.jpg" width="100%"><br>Qwen-Image 2.1 Turbo: text in the picture | <img src="samples/images/lighthouse-z-image-turbo.jpg" width="100%"><br>Z-Image-Turbo |
| <img src="samples/images/fisherman-krea2-realism.jpg" width="100%"><br>Krea 2 Turbo + realism LoRA | <img src="samples/images/reading-nook-flux2-klein.jpg" width="100%"><br>FLUX.2 klein 4B |

**Music:** [Run with the Lions](samples/music/run-with-the-lions-ace-step.mp3) (ACE-Step, a song with original lyrics) ·
[Trailer score](samples/music/trailer-score-minimax-music3.mp3) (MiniMax Music 3) ·
[Lion's Dawn, MIDI with chosen instruments](samples/music/lions-dawn-midi-instruments.mp3) and
[the same piece restyled by AI](samples/music/lions-dawn-midi-ai-restyle.mp3) ([the .mid](samples/music/lions-dawn.mid))

<p><img src="docs/screenshots/desktop-library.png" width="49%" alt="Library"> <img src="docs/screenshots/desktop-music.png" width="49%" alt="Music builder"></p>
<p><img src="docs/screenshots/phone-video.png" width="24%" alt="Phone: video"> <img src="docs/screenshots/phone-library.png" width="24%" alt="Phone: library"> <img src="docs/screenshots/desktop-image.png" width="49%" alt="Image builder"></p>

## Built for the DGX Spark

| | |
|---|---|
| **Hardware** | NVIDIA **DGX Spark** (GB10 Grace Blackwell, **128 GB unified memory**, aarch64). Video uses ~55 GB while rendering; the director needs ~19 GB more. Other GB10 machines with 128 GB should work but are untested. |
| **OS** | DGX OS / Ubuntu 24.04, NVIDIA driver 580+, CUDA 13.0 |
| **Disk** | ~150 GB for the core models, ~350 GB for everything, plus room for your library. Use the internal NVMe: models must not run off network storage. |
| **Software** | Python 3.12, `ffmpeg`, `git`, `curl`; Docker only for the TensorFold director recipe |
| **Renderer** | ComfyUI pinned to commit `b26625f2` (v0.39.0 + 18), PyTorch 2.10 + cu130 |

## Install

```bash
git clone https://github.com/sunnychase/lion-ai-studios.git && cd lion-ai-studios
cp studio.env.example studio.env            # edit paths, ports and the director's address
./scripts/install_comfyui.sh                # ComfyUI at the tested commit, its venv, and the studio's image node
python3 scripts/download_models.py          # core models, ~122 GB (--list to see everything, --all for ~331 GB)
./scripts/setup_envs.sh                     # optional: voice + MIDI helper environments
./scripts/install_services.sh               # systemd --user units; open http://127.0.0.1:8482
```

Then set up the director ([docs/DIRECTOR.md](docs/DIRECTOR.md)) and, optionally, the TensorFold engine
([tensorfold/README.md](tensorfold/README.md)). The web app itself is a single Python file that uses only the standard
library. It starts ComfyUI when the first render needs it.

PyTorch may warn that the GB10 (compute capability 12.1) is newer than the range it was built for (up to 12.0). That warning
is harmless; everything here was tested with it.

**Which models do I need?** See [docs/MODELS.md](docs/MODELS.md). Every file has its exact Hugging Face source, size,
destination folder and licence. Short version:

| You want | Download | Size |
|---|---|---:|
| Video with sound, all presets | `--group video` core files: MiniMax H3 int8, FastH3 int8, H3 Turbo LoRA, text encoder, video + audio decoders | 67 GB |
| Images | Z-Image-Turbo and Qwen-Image 2.1 Turbo (core); eight more models are optional | 45 GB core / 168 GB all |
| Music | ACE-Step 1.5 Turbo (core); MiniMax Music 3 and the MIDI soundfont are optional | 10 / 22 GB |
| Voices | Qwen3-TTS 1.7B Base + Whisper large-v3-turbo | 6 GB |
| Cast / Recreate | MiniMax H3 reference-to-video int8 | 21 GB |
| TensorFold engine | FastH3 bf16 | 44 GB |
| Director | Qwen3.8-27B, 4-bit (via its own recipe) | ~20 GB |

## Configuration

Everything machine-specific lives in `studio.env` ([template](studio.env.example)): paths, ports, the ComfyUI address,
the director's address and model name, the thermal guard (pause at 95 °C, resume below 85 °C), and an optional
*GPU tenant*. That is another service, such as a vLLM server, that the studio may stop before rendering and offer to start again
afterwards. It always asks you first.

**Remote access.** The app listens on `127.0.0.1` and has no login. To use it from your phone, put it behind an
authenticating reverse proxy or tunnel (for example Cloudflare Access, Tailscale, or nginx with auth). Never expose it bare.

## Benchmarks

One DGX Spark, 5 s clip with sound, single runs (n = 1). Details and caveats are in the full report.

| | ComfyUI | TensorFold engine |
|---|---:|---:|
| **Draft** 864x480, 8 steps | 132 s | **76 s** (1.74x) · **70 s** with the 8-bit decoder (1.89x) |
| **Fast HD** 1344x768, 8 steps | 272 s | **192 s** (1.42x) |
| **Standard** H3 + Turbo, 8 steps | 332–415 s | — |
| **High** H3 base 1344x768, 20 steps | 1052 s | — |

| Other | Time |
|---|---:|
| 20 s Draft film (2 segments, seamless) through the studio | 373 s |
| Qwen-Image 2.1 Turbo, 1344x768 / 2752x1536 | 16 s / 44 s |
| ACE-Step 1.5 Turbo, 30–60 s of music | 4–16 s (3.8–7.5x realtime) |
| MiniMax Music 3 | ~0.46x realtime |

On the same two prompts, the director scored **Fast HD 8/8** against **Standard 5/9**. Fast HD is also 1.6–1.7x faster and
renders at a higher resolution, which is why it is the default preset.

Full tables, methods and limitations are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md), and the raw results and scripts are in [benchmarks/](benchmarks/).

## Repository

```
app/               the studio: web server (stdlib only), ComfyUI graphs, director, music, MIDI and voice workers, the page
comfyui_nodes/     Lion AI Studios image node for ComfyUI (one menu for ten image models)
tensorfold/        optional TensorFold engine: setup notes + a small patch
scripts/           install ComfyUI, download models, helper envs, systemd units
systemd/           unit templates (studio, ComfyUI, optional director)
docs/              models, director, benchmarks, responsible use, screenshots
benchmarks/        raw results and the scripts that produced them
samples/           example videos, images and music made with the studio
```

## Responsible use

Voice cloning, Cast and Recreate can be misused. Use them only with people who have consented, respect copyright, and
label AI-generated media. Read [docs/RESPONSIBLE_USE.md](docs/RESPONSIBLE_USE.md).

## Credits and licence

The code in this repository is under the [MIT licence](LICENSE). It builds on ComfyUI, MiniMax H3 and Music 3, FastVideo's
FastH3, Qwen, ACE-Step, TensorFold and many open models; see [NOTICE.md](NOTICE.md). Each model keeps its own licence, and
several restrict commercial use.
