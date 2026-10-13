# Changelog

Updates to Lion AI Studios, newest first. The project has no version numbers; each entry is dated.

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
