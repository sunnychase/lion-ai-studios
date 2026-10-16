# Models

Everything the studio can use, where it goes, and where it comes from. `scripts/download_models.py` reads the same list
(`scripts/models.json`) and downloads with resume and size checks. Run `--list` to see it in your terminal.

- **core** (~122 GB): enough for video with sound (all four presets), the two default image models and fast music.
- **all** (~373 GB): adds the TensorFold checkpoint, the 8-bit decoder, Cast/Recreate, nine more image models, MiniMax Music 3, MIDI and voices.
- The director is a separate download (see [DIRECTOR.md](DIRECTOR.md)). Demucs fetches its own small model on first use.
- Ming-Image 0.1 Design is a separate download too, with its own code and venv (see [MING.md](MING.md)).
- Fish Audio Drama 3 (optional voice engine, 41.5 GB) is opt-in: `--all` leaves it out; fetch it with `--group drama3` (see [DRAMA3.md](DRAMA3.md)).

`comfy:<folder>` means `$COMFY_MODELS/<folder>/` and `studio:<folder>` means `$STUDIO_HOME/models/<folder>/`. The studio only offers
what is installed: an image model missing a file is left out of the list.

> **Licences.** Each model has its own licence. Several restrict commercial use, among them Juggernaut Z (CC BY-NC 4.0),
> Qwen-Image 2.1 (Qwen Research License), Krea 2 and MiniMax H3 (their community licences) and the FLUX.2 VAE's repo (non-commercial).
> FLUX 3 Image Klein is under the FLUX Kommunity License v1.0: non-commercial use, plus commercial use of outputs for businesses under
> US$5M gross annual revenue; anything else needs a licence from Black Forest Labs (bfl.ai/licensing).
> Read the model card before you sell or publish what you make.

## Video with sound (MiniMax H3 / FastH3)

| tier | file | size | goes to | source | licence | used for |
|---|---|---:|---|---|---|---|
| core | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 21.0 GB | `comfy:diffusion_models` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | MiniMax H3 Community License | MiniMax H3 — Standard/High and every image-start / film segment |
| core | `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors` | 22.1 GB | `comfy:diffusion_models` | [FastVideo/FastVideo-FastH3-Comfy](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy) | MiniMax H3 Community License | FastH3 — Draft / Fast HD on the ComfyUI engine |
| optional | `fastvideo_fasth3_8step_v2_pruned_bf16.safetensors` | 44.1 GB | `comfy:diffusion_models` | [FastVideo/FastVideo-FastH3-Comfy](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy) | MiniMax H3 Community License | FastH3 bf16 — needed only by the TensorFold engine |
| core | `minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | 2.0 GB | `comfy:loras` | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | apache-2.0 | H3 Turbo 8-step LoRA — Standard preset |
| core | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15.7 GB | `comfy:text_encoders` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | MiniMax H3 Community License | H3 text encoder |
| core | `minimax_h3_video_vae_fp16.safetensors` | 5.2 GB | `comfy:vae` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | MiniMax H3 Community License | H3 video decoder |
| core | `minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB | `comfy:vae` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | MiniMax H3 Community License | H3 audio decoder |
| optional | `minimax_h3_video_vae_int8_convrot.safetensors` | 2.8 GB | `comfy:vae` | [FastVideo/FastVideo-FastH3-Comfy](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy) | MiniMax H3 Community License | 8-bit video decoder (~2x faster decode) |
| optional | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 21.0 GB | `comfy:diffusion_models` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | MiniMax H3 Community License | H3 reference-to-video — Cast and Recreate a Short |

## Images

| tier | file | size | goes to | source | licence | used for |
|---|---|---:|---|---|---|---|
| core | `z_image_turbo_bf16.safetensors` | 12.3 GB | `comfy:diffusion_models` | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) | apache-2.0 | Z-Image-Turbo — default image model |
| core | `qwen_3_4b.safetensors` | 8.0 GB | `comfy:text_encoders` | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) | apache-2.0 | text encoder for Z-Image, Juggernaut Z, FLUX.2 klein |
| optional | `krea2_turbo_bf16.safetensors` | 26.3 GB | `comfy:diffusion_models` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) | Krea 2 Community License | Krea 2 Turbo |
| optional | `Krea2-realism-V2.safetensors` | 1.6 GB | `comfy:loras` | [RudySen/Krea2-realism-V2](https://huggingface.co/RudySen/Krea2-realism-V2) | mit | Krea 2 realism LoRA |
| optional | `qwen3vl_4b_fp8_scaled.safetensors` | 5.2 GB | `comfy:text_encoders` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) | Krea 2 Community License | text encoder for Krea 2 |
| core | `qwen_image_2.1_turbo_bf16.safetensors` | 14.2 GB | `comfy:diffusion_models` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) | Qwen Research License | Qwen-Image 2.1 Turbo — High preset, text in images, image editing |
| core | `qwen3vl_8b_int8_convrot.safetensors` | 9.4 GB | `comfy:text_encoders` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) | Qwen Research License | text encoder for Qwen-Image 2.1 |
| optional | `qwen_image_2.1_int8_convrot.safetensors` | 7.3 GB | `comfy:diffusion_models` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) | Qwen Research License | Qwen-Image 2.1 (25 steps) |
| core | `qwen_image_2.1_vae_bf16.safetensors` | 0.7 GB | `comfy:vae` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) | Qwen Research License | VAE for Qwen-Image 2.1 |
| optional | `qwen_image_2512_fp8_e4m3fn.safetensors` | 20.4 GB | `comfy:diffusion_models` | [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) | apache-2.0 | Qwen-Image 2512 |
| optional | `Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors` | 0.8 GB | `comfy:loras` | [lightx2v/Qwen-Image-2512-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning) | apache-2.0 | 8-step LoRA for Qwen-Image 2512 |
| optional | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | 9.4 GB | `comfy:text_encoders` | [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) | apache-2.0 | text encoder for Qwen-Image 2512 |
| optional | `Chroma1-HD.safetensors` | 17.8 GB | `comfy:diffusion_models` | [lodestones/Chroma1-HD](https://huggingface.co/lodestones/Chroma1-HD) | apache-2.0 | Chroma1-HD |
| optional | `t5xxl_fp8_e4m3fn_scaled.safetensors` | 5.2 GB | `comfy:text_encoders` | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) | apache-2.0 | text encoder for Chroma |
| optional | `hidream_o1_image_fp8_scaled.safetensors` | 8.1 GB | `comfy:checkpoints` | [Comfy-Org/HiDream-O1-Image](https://huggingface.co/Comfy-Org/HiDream-O1-Image) | mit | HiDream-O1 (all-in-one checkpoint) |
| optional | `Juggernaut_Z_V1_by_RunDiffusion_fp16.safetensors` | 12.3 GB | `comfy:diffusion_models` | [RunDiffusion/Juggernaut-Z-Image](https://huggingface.co/RunDiffusion/Juggernaut-Z-Image) | cc-by-nc-4.0 | Juggernaut Z (non-commercial licence) |
| optional | `Juggernaut_Z_V1_Fast_FP16.safetensors` | 12.3 GB | `comfy:diffusion_models` | [RunDiffusion/Juggernaut-Z-Image-Fast](https://huggingface.co/RunDiffusion/Juggernaut-Z-Image-Fast) | cc-by-nc-4.0 | Juggernaut Z Fast, 4 steps (non-commercial licence) |
| optional | `flux-2-klein-4b.safetensors` | 7.8 GB | `comfy:diffusion_models` | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | apache-2.0 | FLUX.2 klein 4B |
| core | `ae.safetensors` | 0.3 GB | `comfy:vae` | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) | apache-2.0 | VAE for Z-Image, Juggernaut Z, Chroma |
| optional | `qwen_image_vae.safetensors` | 0.3 GB | `comfy:vae` | [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) | apache-2.0 | VAE for Qwen-Image 2512 and Krea 2 |
| optional | `flux2-vae.safetensors` | 0.3 GB | `comfy:vae` | [Comfy-Org/flux2-dev](https://huggingface.co/Comfy-Org/flux2-dev) | repo card: flux-1-dev-non-commercial-license | VAE for FLUX.2 klein |
| optional | `flux3_image_klein_sd_bf16.safetensors` | 11.1 GB | `comfy:diffusion_models` | [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy) ¹ | FLUX Kommunity License | FLUX 3 Image Klein (distilled, bf16): images and edits |
| optional | `qwen3vl_4b_bf16.safetensors` | 8.9 GB | `comfy:text_encoders` | same repo ¹ | apache-2.0 | text encoder for FLUX 3 Image Klein |
| optional | `qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors` | 9.7 GB | `comfy:text_encoders` | same repo ¹ | apache-2.0 | BFL's prompt upsampler for FLUX 3 Image Klein |
| optional | `flux3_klein_vae.safetensors` | 0.3 GB | `comfy:vae` | same repo, `vae/flux2-vae.safetensors` ¹ | FLUX Kommunity License | VAE for FLUX 3 Image Klein, saved under its own name (it is a different file from FLUX.2's) |

¹ **Gated.** Accept the licence on the repo page while logged in to Hugging Face, then set `HF_TOKEN` for `download_models.py`.
FLUX 3 Image Klein also needs ComfyUI's FLUX 3 support: `scripts/install_comfyui.sh` adds the three upstream commits (it is built into ComfyUI v0.41+).
`scripts/models.json` lists the sha256 of each of these four files.

## Music

| tier | file | size | goes to | source | licence | used for |
|---|---|---:|---|---|---|---|
| core | `ace_step_1.5_turbo_aio.safetensors` | 10.0 GB | `comfy:checkpoints` | [Comfy-Org/ace_step_1.5_ComfyUI_files](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files) | apache-2.0 | ACE-Step 1.5 Turbo — the Fast music engine |
| optional | `minimax_music3_dit_int8_convrot.safetensors` | 2.5 GB | `comfy:diffusion_models` | [Comfy-Org/MiniMax-Music-3](https://huggingface.co/Comfy-Org/MiniMax-Music-3) | apache-2.0 | MiniMax Music 3 — the Best music engine |
| optional | `minimax_music3_text_encoder_pruned_int8_convrot.safetensors` | 9.2 GB | `comfy:text_encoders` | [Comfy-Org/MiniMax-Music-3](https://huggingface.co/Comfy-Org/MiniMax-Music-3) | apache-2.0 | MiniMax Music 3 text model |
| optional | `minimax_music3_dav.safetensors` | 0.2 GB | `comfy:vae` | [Comfy-Org/MiniMax-Music-3](https://huggingface.co/Comfy-Org/MiniMax-Music-3) | apache-2.0 | MiniMax Music 3 decoder |
| optional | `MuseScore_General.sf2` | 0.2 GB | `studio:soundfonts` | [download](https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2) | MIT | General MIDI soundfont for the MIDI engine |

## Voices

| tier | file | size | goes to | source | licence | used for |
|---|---|---:|---|---|---|---|
| optional | `whole repo` | 4.6 GB | `studio:Qwen3-TTS-12Hz-1.7B-Base` | [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | apache-2.0 | Qwen3-TTS voice cloning (whole repo) |
| optional | `whole repo` | 1.7 GB | `studio:whisper-large-v3-turbo` | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | mit | Whisper transcription (whole repo) |

## Fish Audio Drama 3 (optional voice engine, outside ComfyUI) — Built with Fish Audio

| tier | file | size | goes to | source | licence | used for |
|---|---|---:|---|---|---|---|
| optional, opt-in | `whole repo` @ `e0a9218a` | 41.5 GB | `DRAMA3_MODEL` (default `studio:drama-3`) | [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3) | Fish Audio Community License v1.0 | the narrator's most natural voice engine |

Research and non-commercial use is free; commercial use only under US$500,000 annual revenue; a hosted voice service always needs a
licence from Fish Audio; consent is required for real people's voices. It also needs the TensorFold engine and a one-time codec
conversion: [DRAMA3.md](DRAMA3.md).

## Ming-Image (optional, outside ComfyUI)

| weights | size | goes to | licence | used for |
|---|---:|---|---|---|
| [inclusionAI/Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design) (whole repo) | 52.9 GB | `MING_MODEL` (default `studio:Ming-Image-0.1-Design`) | MIT | posters, UI, infographics, text, transparent PNGs |

Setup, the code checkout and the venv: [MING.md](MING.md).

## Director

| option | weights | size |
|---|---|---:|
| TensorFold recipe (developed with) | 4-bit MLX Qwen3.8-27B + [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) drafter, fetched by the recipe | ~20 GB |
| any OpenAI-compatible server | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (official, Apache-2.0, vision) or a quantised build of it | 16–56 GB |
