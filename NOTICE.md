# Third-party notices

The MIT licence in `LICENSE` covers the code in this repository. Everything below is separate, downloaded by you, and
under its own terms.

- **ComfyUI** ([comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI), GPL-3.0). The studio runs it as a
  separate process and talks to it over HTTP; `comfyui_nodes/lion_studio_image_gen.py` is a ComfyUI custom node.
- **TensorFold H3 node** ([drowzeys/keys-TensorFold-Studio-Mac-DGXSpark-64GB-VRam-Min](https://github.com/drowzeys/keys-TensorFold-Studio-Mac-DGXSpark-64GB-VRam-Min), Apache-2.0).
  `tensorfold/one-engine-resident.patch` modifies its `__init__.py`.
- **TensorFold engine** ([drowzeys/TensorFold](https://github.com/drowzeys/TensorFold), MIT). Not redistributed.
- **Director recipe** ([MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold)). Not redistributed.
- **Ming-Image** ([inclusionAI/Ming-Image](https://github.com/inclusionAI/Ming-Image), weights
  [inclusionAI/Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design), MIT). Not redistributed;
  `app/ming_runner.py` imports its `infer.py` unchanged, and the director uses its released rewriter prompt from that checkout.
- **LocalRouter** ([jayleaton/localrouter](https://github.com/jayleaton/localrouter), Apache-2.0). Optional; not redistributed.
  The studio talks to it over HTTP. `systemd/lion-localrouter.service.example` and `docs/localrouter-tools.example.json` are
  example settings for it.
- **Python packages**: mido, tinysoundfont, qwen-tts, demucs, transformers, accelerate, soundfile, yt-dlp, PyTorch, each under its own licence.
- **MuseScore General soundfont** (MIT).
- **Models**: see [docs/MODELS.md](docs/MODELS.md). Several restrict commercial use.
- **README and repository-template layout** adapted from [morluto/rea](https://github.com/morluto/rea) (MIT): the structure
  only; the wording is this project's own.
- **Fonts**: Inter (SIL OFL 1.1), loaded from Google Fonts by the web page.
- **Sample media** in `samples/` was generated with this studio from the prompts listed in `samples/README.md`.
  Each output is subject to the licence of the model that made it.

Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
