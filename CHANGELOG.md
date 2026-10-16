# Changelog

Updates to Lion AI Studios, newest first. The project has no version numbers; each entry is dated.

## 2026-10-16 — Voices for everyone who speaks, Fish Audio Drama 3, shot and word checks, Trash, memory guard

Several of these come from [drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3](https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3)
(Apache-2.0): three files copied with credit, three ideas rewritten for this studio. See NOTICE.md.

- **Clone everyone who speaks.** The Voices tab can turn every person in one video into a voice of their own (up to 6): phrases are
  fingerprinted with Qwen3-TTS's speaker encoder and grouped; each reference is that person's longest solo stretch (or a few of their own
  turns joined) and is transcribed again from its own clip; the director names each voice from frames taken while that person speaks.
  *How many people?* overrides the automatic count. Single-voice references are now also re-transcribed from their own clip.
- **Word check for every spoken take** (`app/wordcheck.py`, copied). Whisper listens to each Qwen3-TTS take; a take that drops words,
  loses its ending or reads reference words before or after the line is drawn again (up to 3 takes), with a last x-vector-only take that
  cannot read the reference aloud. The voice-over job records the check.
- **Fish Audio Drama 3** (optional narrator engine, *Voice engine* menu; Built with Fish Audio). TensorFold's native CUDA port
  (`drama3-cuda` branch), one process per take, the same word check. Weights opt-in (`--group drama3`, 41.5 GB, Fish Audio Community
  License); build steps in docs/DRAMA3.md. New settings `DRAMA3_DIR`, `DRAMA3_MODEL`.
- **Shot check** (`app/shot.py`, copied and modified). Every take is frame-checked for a hard cut inside a one-shot clip and, in keyframe
  films, for a clip that does not open or close on its pictures; a failure triggers the single re-take, and a take that passes wins.
- **Negation warning** (`app/negation.py`, adapted). The page flags prompt phrases that name a thing to forbid it ("no cars"), `POST
  /api/negation` does the check, and queued jobs note them. The studio's own prompts (portrait card, face swap, Faithful continuity)
  now say what is there instead, and the director is told to write that way.
- **Edit noise seed.** Image edits draw their noise from the seed and the input pictures' bytes (the job records it), so an edit that
  reuses its source's seed no longer drifts back to the source.
- **Memory guard.** While a ComfyUI render runs, free memory is read 4x/s: under `STUDIO_MEM_FLOOR_GIB` (6) the render is interrupted
  and the job fails with the reason; under `STUDIO_MEM_CRIT_GIB` (3), or falling to it within a second, ComfyUI is stopped too. ComfyUI
  is also made the kernel's first choice if memory runs out.
- **Trash.** Deleting a render or a voice moves it to `$STUDIO_HOME/trash/`; Library → Trash restores it or deletes it for good
  (`GET /api/trash`, `POST /api/trash/restore`, `POST /api/trash/empty`). The delete prompts say so.
- **YouTube downloads** pass `--js-runtimes deno` when a Deno binary is configured (`DENO_BIN`) or on PATH, and retry a 403 up to 3 times;
  `yt-dlp-ejs` supplies the solver scripts.
- `scripts/download_models.py`: entries may pin a revision (`rev`) and be opt-in (`explicit`, left out of `--all`).
- NOTICE.md: the TensorFold H3 node's repository was renamed upstream; the TensorFold engine is Apache-2.0 (ashhart/TensorFold; the
  h3-cuda and drama3-cuda branches of the drowzeys fork carry it), not MIT as listed before.

## 2026-10-15 — FLUX 3 Image Klein

- **FLUX 3 Image Klein** (Black Forest Labs, 5.6B, distilled, bf16) is a new optional image model, at BFL's own ComfyUI recipe: 16 steps,
  euler / simple, CFG 1. Every prompt first goes through BFL's prompt upsampler (Qwen3.5-9B), as its model card asks, with BFL's system
  prompts verbatim (`comfyui_nodes/flux3_klein_system_prompts.json`). The model, its Qwen3-VL-4B encoder, the upsampler and its VAE are
  about 30 GB; the repo is gated and under the FLUX Kommunity License (see docs/MODELS.md).
- **Edits with FLUX 3 Image Klein.** With reference images loaded, the model menu now offers Qwen-Image 2.1 Turbo (still the default) or
  FLUX 3 Image Klein; models that cannot edit are greyed out. Klein edits follow BFL's edit template (the upsampler sees the references).
- `scripts/install_comfyui.sh` adds ComfyUI's FLUX 3 support on top of the tested commit (upstream 20b46065, 9fa63b0a, 6da3d1a5, all part
  of ComfyUI v0.41) and copies the prompts file next to the image node.
- `scripts/models.json` gains the four files, with their sha256. Three new samples and a new section 15 in docs/BENCHMARKS.md.

## 2026-10-13

- **Juggernaut Z fixed.** It is a fine-tune of Z-Image *Base*, not Turbo, but ran with Turbo settings (8 steps, CFG 1), which gave
  garbled pictures. It now uses its model card's settings: 35 steps, CFG 6. Slower (about a minute warm), clean output.
- **Juggernaut Z Fast (4 steps)**, RunDiffusion's fast release, is a new optional image model (12.3 GB, CC BY-NC 4.0) for the
  Juggernaut look at Turbo speed (4 steps, CFG 1). `scripts/models.json` and docs/MODELS.md list it.
- The Juggernaut Z row in docs/BENCHMARKS.md was measured at the old settings; it is marked as such until it is re-measured.

## 2026-10-12 — Benchmarks: every image model, image edit, Ming, voices, Cast

New measurements on one DGX Spark, in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) sections 10–14, with raw rows in `benchmarks/results/` and the scripts in
`benchmarks/scripts/`. Most cells are n = 1 or n = 2, so they are not error bars.
- **Every image model** the studio offers, at its own settings, 1344x768: one cold and two warm runs each. Warm: FLUX.2 klein 4 s, Z-Image Turbo and Juggernaut Z 8 s,
  Qwen-Image 2.1 Turbo 10 s (44 s at 2752x1536), Krea 2 16 s, Qwen-Image 2.1 18–20 s, HiDream-O1 26–28 s, Chroma1-HD 56–58 s. Juggernaut Z's pictures came
  out visibly broken at its settings.
- **Image edit** with 1, 2 and 4 references: 26 s cold for 1 (as the studio job), 16 / 24 / 40 s warm.
- **Ming-Image 0.1 Design** at 1024x1024: about 5.5 min per job, almost all of it loading the model; each further image +22 s. The director's optional
  layout step: 14 s.
- **Voices** (Qwen3-TTS) from a synthetic 5 s clip: cloning 67 s; a ~33 s narration 56 s in its own process (0.59x realtime), 1.3x realtime with the
  model already loaded.
- **Cast** (reference to video, official recipe), 5 s: Draft 480p 328 s; Fast HD 979 s, the same as the studio job.
- Not measured: Recreate a Short and the narrator clone from a YouTube Short. They need a source video the project may use.
- `benchlib.py` now records peak memory in use and GPU temperature once a second, ComfyUI's own execution time (`exec_s`, not rounded by polling), and waits
  while the studio has a job queued. `bench_graphs.py` gains copies of the studio's Cast and edit graphs.
- README: the TensorFold column for Standard and High now says why it is empty (the dense path was tested at 5–8 % faster and not shipped).

## 2026-10-12 — Ming-Image 0.1 Design, image downloads, narrator cloning, Fast images

**Images: optional Fast engine (LocalRouter)**
- Image builder → *Engine*: *ComfyUI* (default) or *Fast (LocalRouter)*. Fast sends plain Qwen-Image 2.1 Turbo text to image at
  1024×1024, one image per job, to [LocalRouter](https://github.com/jayleaton/localrouter)'s FP8 engine. Anything else renders on
  ComfyUI as before, and so does a Fast job that cannot be made (no room, LocalRouter not answering, an error or a bad reply), with
  the reason in the job notes. Fast jobs start ComfyUI only when they fall back.
- Off by default. New settings: `LOCALROUTER_ENABLE`, `LOCALROUTER_URL`, `LOCALROUTER_MODEL`, `LOCALROUTER_SERVICE` (or
  `LOCALROUTER_START` / `LOCALROUTER_STOP`), `LOCALROUTER_IDLE_S`, `LOCALROUTER_RESERVE_GIB`. The studio starts LocalRouter for a
  Fast job and stops it after 10 minutes without one; a warm LocalRouter unloads its model before every other job.
- An example on-demand unit (`systemd/lion-localrouter.service.example`) and model list (`docs/localrouter-tools.example.json`);
  setup in [docs/FAST_IMAGES.md](docs/FAST_IMAGES.md).
- New `app/studio_broker.py`, a small memory broker: one list of what holds memory (LocalRouter, ComfyUI, the director, Ming, a
  voice job) that frees idle holders, lowest priority first, and never a busy one; it refuses when it cannot read the memory.
  Fast jobs and the voice check use it; `/api/state` shows it under `broker`.
- Job cards in the Queue now show the job's notes (Ming and Fast write them).
- Measured on one DGX Spark at 1024×1024, n = 12 each: Fast 6.2 s warm and about 12–13 s from cold; ComfyUI 9.3 s warm.

**Video: clone the narrator's voice from a YouTube Short**
- *Music & voice-over* → *Clone a narrator voice from a YouTube Short*: paste a Shorts link, optionally say whose voice and name
  it, confirm consent, and press *Clone and use as narrator*. It is the same cloner as the Voices tab (yt-dlp, Demucs, Whisper,
  the director picks the speaker, Qwen3-TTS). The new voice is selected as narrator at once ("cloning…") and the film can be
  queued straight away: the queue runs in order, so the voice is made first. The page shows its progress and says if it fails.
- A film whose narrator voice is missing when its turn comes (the cloning failed, or the voice was deleted) now fails **before**
  rendering, with a message. It used to fail after the whole film had rendered, when the voice-over was added.

**Fixes**
- Video: the *Length* picker could not be changed. Every redraw wrote the saved length back into the picker before reading it, so
  a new choice was overwritten at once. The saved length is now restored only when the picker is first built.

**Library: download images**
- Image and edit cards now have a *Download* button for every picture (*Download 1…4* when a job made several). The file is
  saved as `studio_<job id>_<file>`. Video and music cards already had one.

**Images: Ming-Image 0.1 Design (optional)**
- inclusionAI's Ming-Image-0.1-Design (MIT, 6B) joins the Image menu for posters, UI screens, infographics and other text-heavy
  designs. It runs outside ComfyUI: one process per job in its own venv (`app/ming_runner.py`, the official inclusionAI/Ming-Image
  code unchanged) that loads the model, renders every variation and exits. Setup: [docs/MING.md](docs/MING.md).
- *Director lays out the design* (on by default) runs Ming's own released text-to-image rewriter prompt on the director
  (`director.ming_layout`) and sets the layout's aspect ratio to the chosen canvas. If the director does not answer, your words are
  used as written.
- *Transparent background* starts the prompt with Ming's fixed `RGBA, 4-channel, transparent background` phrase; the PNG keeps
  its alpha channel.
- Before it loads, ComfyUI unloads its models. The job needs about 58 GiB + 5 GiB per megapixel free (`workflows.ming_need_gib`);
  if memory is still short and the studio manages the director, the director is stopped, and otherwise the job fails with a message.
- New settings: `MING_MODEL`, `MING_CODE`, `MING_PY` (and `MING_RUNNER`). The model is listed only when all of them exist.
- Measured on one DGX Spark: 309–363 s to load, 21–23 s per 1024×1024 image, 122 s at 2048×2048.

## 2026-10-11 — Faithful Recreate, MIDI arranging

**Recreate a Short: Faithful style (now the default)**
- Follows the original frame by frame instead of re-describing it. The director picks out the performer once
  (`director.who_is_performer`), and one clean studio portrait of your cast is made with the image editor.
- Anchor frames are taken about 1.2 s apart *inside* every shot (never across a cut; the last one 3 frames before the cut, with an
  exact-seek fallback when a fast seek returns nothing). The director checks each one (`director.check_frame`): only where the
  performer's face is visible is the cast swapped in, using that frame and the portrait as references. Swapped frames are never
  chained into the next swap, because the sharpening compounds. Every other anchor passes through unchanged.
- H3 animates between neighbouring anchors (first + last frame) with a neutral "one continuous shot" prompt. Each clip is retimed
  to the exact gap between its anchors (`setpts` / `atempo`), and each shot is padded to its exact length (`tpad` / `apad`).
- The Short is captured at up to 1080p for sharper anchors (`voice.py capture URL DIR [max height]`).
- **Re-imagine** keeps the previous behaviour: the director rewrites each shot, and your *Changes* apply.
- **Shape**: output follows the Short's shape, so a vertical Short renders vertical (`workflows.shape_dims`; `video_graph` and
  `cast_graph` take `dims`). Choose *Landscape* to override.
- The Recreate soundtrack now defaults to *Original audio, exactly*. Choosing Faithful switches *New sound* to it.
- Measured on one DGX Spark: about 22 min for a 21 s Short at Draft, with audio sync 0.999 against the original.
- Known limitation: scene makeup (face paint, special-effects makeup) carries over on some swapped frames only.

**Music: the director arranges MIDI**
- *Attach a MIDI file* in the Music builder, describe the sound ("jazz trio: piano, upright bass, brushed drums"), and press
  *Director: arrange it*. The director reads every part (register, notes, track name) and picks a General MIDI instrument or drum
  kit for each, or mutes it. It also writes a style line for the AI restyle and decides whether the restyle is needed.
- New route: `POST /api/director/midi` with `{file, request}` returns `{parts, names, style, restyle, strength, notes, by}`.
  The director's names are matched to General MIDI programs and kits by `gm_pick`; an unknown name keeps the file's own instrument.

**Fixes**
- MIDI: the Standard drum kit (kit 0) could not be picked by name, because `gm_pick` treated kit 0 as "not found". It now
  compares with `None`.
- MIDI: choosing an instrument from a part's menu failed with "bad instrument". The page's digit test sat inside a template literal,
  where `\d` loses its backslash, so it now uses `[0-9]`. The server also accepts program numbers sent as digit strings.

## 2026-10-10 — Recreate soundtrack choice
- Soundtrack for Recreate: new sound, the original music under your cast's voices, or the original audio exactly. With the
  original sound every shot is cut to its original length (`voice.py stems`, with a CPU fallback).
- The director describes the scene's makeup, face paint, prosthetics, costume and expression as applied to the cast.

## 2026-10-10 — Director JSON
- Director replies are parsed leniently, with one retry. Recreate keeps the cast named in every shot.

## 2026-10-10 — First public release
- Video with sound (MiniMax H3 / FastH3, films up to 5 min), ten image models with editing, music (ACE-Step 1.5, MiniMax Music 3,
  MIDI), voices, the director, Cast, Recreate a Short, the Editor, setup scripts, the model manifest, benchmarks and samples.
