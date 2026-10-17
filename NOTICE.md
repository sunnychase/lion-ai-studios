# Third-party notices

The MIT licence in `LICENSE` covers the code in this repository. Everything below is separate, downloaded by you, and
under its own terms.

- **ComfyUI** ([comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI), GPL-3.0). The studio runs it as a
  separate process and talks to it over HTTP; `comfyui_nodes/lion_studio_image_gen.py` is a ComfyUI custom node.
- **TensorFold H3 node** ([drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3),
  Apache-2.0; formerly `drowzeys/keys-TensorFold-Studio-Mac-DGXSpark-64GB-VRam-Min`). `tensorfold/one-engine-resident.patch` modifies its `__init__.py`.
- **TensorFold engine** ([ashhart/TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0), built from the fork
  [drowzeys/TensorFold](https://github.com/drowzeys/TensorFold): branch `h3-cuda` for FastH3 and branch `drama3-cuda` for Drama 3,
  both under the upstream Apache-2.0 licence (the fork's `main` branch carries an MIT licence file). Not redistributed.
- **Director recipe** ([MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold)). Not redistributed.
- **Ming-Image** ([inclusionAI/Ming-Image](https://github.com/inclusionAI/Ming-Image), weights
  [inclusionAI/Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design), MIT). Not redistributed;
  `app/ming_runner.py` imports its `infer.py` unchanged, and the director uses its released rewriter prompt from that checkout.
- **LocalRouter** ([jayleaton/localrouter](https://github.com/jayleaton/localrouter), Apache-2.0). Optional; not redistributed.
  The studio talks to it over HTTP. `systemd/lion-localrouter.service.example` and `docs/localrouter-tools.example.json` are
  example settings for it.
- **FLUX 3 Image Klein prompts** (`comfyui_nodes/flux3_klein_system_prompts.json`): Black Forest Labs' prompt-upsampler system
  prompts, copied verbatim from the ComfyUI workflow templates [Comfy-Org/workflow_templates](https://github.com/Comfy-Org/workflow_templates)
  (MIT) `image_flux3_klein_distilled_t2i.json` and `image_flux3_klein_distilled_image_edit.json`. The model itself is not redistributed.
- **Python packages**: mido, tinysoundfont, qwen-tts, demucs, transformers, accelerate, soundfile, yt-dlp, yt-dlp-ejs (Unlicense AND MIT AND
  ISC; the solver scripts yt-dlp runs in a JavaScript runtime for YouTube), PyTorch, each under its own licence.
- **Deno** ([denoland/deno](https://github.com/denoland/deno), MIT). Optional, not redistributed: yt-dlp uses it as the JavaScript runtime
  for YouTube's challenges (`DENO_BIN`, else `deno` on PATH).
- **MuseScore General soundfont** (MIT).

## Code from drowzeys/keys-Wasserman-Slate-AI-Studio (Apache-2.0)

Three files in `app/` come from [drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3)
at commit `0677c2c`, licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Each keeps a header
naming its source; they remain under Apache-2.0, not under this repository's MIT licence.

- `app/wordcheck.py` (from `app/wordcheck.py`): copied unmodified below its header. Its own header credits TensorFold's Drama 3 port
  (`tensorfold/families/drama3/verify.py`, Apache-2.0) for the rules.
- `app/shot.py` (from `app/shot.py`): **modified** — `one_shot` also treats a written "cut to" / "smash cut" / "jump cut" / "montage"
  as asking for a cut, and `cuts` drops bursts of 3 or more flagged frames within 6 frames of each other (something crossing the
  lens, not a cut).
- `app/negation.py` (from `app/negation.py`): **modified** — adapted to this studio (its own sentences, its field names, its wording).

Ideas taken from the same repository and written anew for this studio: the edit **noise seed** drawn from the seed and the
input pictures' bytes (`edit_plan.py`), the **memory guard** with its floor and critical numbers (`spark/memory_guard.py`), and the
**Trash** for deleted renders and voices (`app/clear.py`).

## Fish Audio Drama 3 — Built with Fish Audio

The optional narrator engine uses **Fish Audio Drama 3** ([fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3)), licensed
under the **Fish Audio Community License, Version 1.0**. This repository does not redistribute the weights, the codec or the engine:
you download and build them yourself ([docs/DRAMA3.md](docs/DRAMA3.md)). In short, and the licence text decides: research and
non-commercial use is free; commercial use is free only while your annual revenue is under US$500,000; offering a hosted
text-to-speech or voice-cloning service to others always needs a separate licence from Fish Audio; never imitate an identifiable
person's voice without their consent; and anything built with it displays "Built with Fish Audio". If you distribute the Fish Audio
Materials yourself, the licence asks for this notice: "These Fish Audio Materials are licensed under the Fish Audio Community
License, Version 1.0. Copyright © 39 AI, INC. All Rights Reserved."

The engine is TensorFold's native CUDA port on the `drama3-cuda` branch of drowzeys/TensorFold (Apache-2.0, see above). Converting the
codec needs Fish's reference codec code; this studio's build used the copy vendored in
[sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni) (Apache-2.0), not redistributed here.

The optional fast server is **drama-tensorfold** ([plotarmordev/drama-tensorfold](https://github.com/plotarmordev/drama-tensorfold),
Apache-2.0, used at `3f44a09c`): patches to [ashhart/TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0) and its launcher
`recipe/pilot.py`. Not redistributed; the studio runs it as a separate process (`systemd/studio-drama3.service.example`) and talks to
it over HTTP on loopback, and calls its `tools/zig/drama_voice.py` to make voice files. That tool uses Fish's reference encoder from
[sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni) at `ab095ef9` (Apache-2.0), also not redistributed. The
voice-file venv (`requirements/drama3-voice.txt`) installs, among others, **descript-audiotools** and **descript-audio-codec**
([descriptinc](https://github.com/descriptinc), MIT), **hydra-core** (MIT), sglang (Apache-2.0) and transformers (Apache-2.0), each
under its own licence.
- **Models**: see [docs/MODELS.md](docs/MODELS.md). Several restrict commercial use.
- **README and repository-template layout** adapted from [morluto/rea](https://github.com/morluto/rea) (MIT): the structure
  only; the wording is this project's own.
- **Fonts**: Inter (SIL OFL 1.1), loaded from Google Fonts by the web page.
- **Sample media** in `samples/` was generated with this studio from the prompts listed in `samples/README.md`.
  Each output is subject to the licence of the model that made it.

Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
