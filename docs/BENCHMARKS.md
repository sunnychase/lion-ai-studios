# Benchmarks

Measured 2026-10-09 and 2026-10-10 on one machine while the studio was being built. Every number in the tables comes from a row
in [`benchmarks/results/`](../benchmarks/results/) and can be checked there. Numbers that were noted during the build but whose raw output was
not kept are listed separately under [Reported but not backed by a kept results file](#reported-but-not-backed-by-a-kept-results-file).

**Read this first:**
- **Most cells are a single run (n = 1).** There are no error bars. Run-to-run variation was not measured, and a 5–10 % difference between two
  single runs can be noise.
- **Wall times are end to end for one ComfyUI prompt.** The clock runs from submitting the graph to ComfyUI reporting it done, and ComfyUI is
  polled every 2 s (3 s for the baseline), so a time can be up to 2 s (3 s) longer than the true time, and never shorter. That covers text encoding, sampling, video and audio decoding and the
  MP4 encode. Model loading is included whenever the model was not already in memory. Warm or cold is noted per table where it is known.
- **"Director" scores come from an AI judge.** The studio's director is a 4-bit Qwen3.8-27B vision-language model (see [DIRECTOR.md](DIRECTOR.md)).
  It watches the clip downscaled to 640 px wide at 8 fps, sees the picture only (no audio) and the prompt, and returns a 1–10 score and a list of issues.
  It was not told which engine made a clip. It is useful for catching a clip that misses the prompt. It is not a human rating, it does not see
  fine softness or artefacts at full size, and one point either way is within its noise.
- **The same seed does not give the same video on both engines.** ComfyUI's FastH3 and TensorFold's FastH3 use different weight files
  (ComfyUI's int8 checkpoint vs the engine's own int8 conversion of the bf16 checkpoint) and different sparse-attention schedules. So the
  same prompt and seed give a different take, and pixel similarity between engines is not a quality measure.

## Test system

| | |
|---|---|
| Machine | 1x NVIDIA DGX Spark: GB10 Grace Blackwell, 128 GB unified memory, aarch64 |
| OS / driver | Ubuntu 24.04, NVIDIA driver 580, CUDA 13.0 |
| Renderer | ComfyUI v0.39.0 (+18, commit `b26625f2`), PyTorch 2.10+cu130, ComfyUI's native MiniMax H3 / ACE-Step / MiniMax Music 3 / Qwen-Image nodes |
| TensorFold engine | `libtf_h3.so` built from drowzeys/TensorFold `h3-cuda` @ `16cecde2` (Zig 0.17, CUDA 13.0) + ComfyUI node `TensorFoldH3Blocks` from drowzeys/keys-TensorFold-Studio-Mac-DGXSpark-64GB-VRam-Min @ `d99d089` |
| Other load | During the 2026-10-09 TensorFold runs the director model (~19 GiB) was loaded but idle. Apart from the baseline in section 1, the scripts wait for an empty ComfyUI queue before each run. Background load beyond that was not recorded. |

Video presets as the studio ships them (5 s = 124 frames at 24 fps, with sound):

| Preset | Model | Size | Steps | Engine in the studio |
|---|---|---|---|---|
| Draft | FastH3 8-step V2 | 864x480 | 8 | TensorFold (default) or ComfyUI |
| Fast HD (default preset) | FastH3 8-step V2 | 1344x768 | 8 | TensorFold (default) or ComfyUI |
| Standard | MiniMax H3 fl2va int8 + Turbo 8-step LoRA | 1248x704 | 8 | ComfyUI |
| High | MiniMax H3 fl2va int8, base | 1344x768 | 20 | ComfyUI |

---

## 1. Video presets on stock ComfyUI (baseline)

`video_presets_comfyui_baseline.csv`: 2026-10-09, n = 1 per preset, same prompt (a street musician with an accordion; sound included), seed 20261009,
fp16 decoder, no TensorFold. GPU and memory were sampled every second. "Step readings" are the two end-of-progress-bar values ComfyUI printed to its log
during the run (s per iteration).

| Preset | Size | Steps | Wall time | Step readings (s) | GPU util (mean) | Max power | Max GPU temp | Min free memory |
|---|---|---|---|---|---|---|---|---|
| Draft (FastH3 + VSA sparse attention) | 864x480 | 8 | **129.1 s** | 11.43 / 12.20 | 90.2 % | 89.79 W | 79 °C | 68.8 GiB |
| Standard (H3 + Turbo 8-step) | **1280x720** | 8 | **414.5 s** | 43.82 / 43.77 | 90.2 % | 88.91 W | 80 °C | 69.7 GiB |
| High (H3 base) | 1344x768 | 20 | **1051.9 s** | 52.39 / 50.18 | 95.3 % | 89.32 W | 81 °C | 67.0 GiB |

Standard ran at 1280x720 here. The studio later moved it to 1248x704 because H3 needs both sides divisible by 32, so 1280x720 broke first-frame
renders. Section 4 has Standard at 1248x704.

## 2. FastH3: ComfyUI vs TensorFold (text to video, 5 s with sound)

`video_fasth3_comfyui_vs_tensorfold.csv`: 2026-10-09, n = 1 per row, same prompt (a fisherman on a dock), seed 1009 unless noted. ComfyUI ran first.
Then ComfyUI's models were unloaded (`/free`) and the TensorFold runs followed.

| Preset | Engine | Decoder | Seed | Wall time | vs ComfyUI | Notes |
|---|---|---|---|---|---|---|
| Draft 864x480 | ComfyUI FastH3 | fp16 | 1009 | **132.3 s** | — | first run of the sequence |
| Draft 864x480 | TensorFold | fp16 | 1010 | **76.1 s** | **1.74x** | warm engine. Seed+1, because an identical graph would have come from ComfyUI's cache |
| Draft 864x480 | TensorFold | 8-bit | 1009 | **70.1 s** | **1.89x** | warm engine, same seed as the ComfyUI row |
| Fast HD 1344x768 | ComfyUI FastH3 | fp16 | 1009 | **272.3 s** | — | |
| Fast HD 1344x768 | TensorFold | fp16 | 1009 | **192.2 s** | **1.42x** | warm engine |
| Draft 864x480 | TensorFold, first run on the machine | fp16 | 1009 | 153.2 s | — | includes the engine's one-time bf16-to-int8 conversion and load (77.1 s more than the warm run) |
| Draft 864x480 | ComfyUI FastH3 | 8-bit | 1009 | 16.0 s | *not comparable* | same prompt and seed as the first row, so ComfyUI reused the cached sampler output. This times **decode + encode only** |

The 16.0 s row is in the raw file because it was measured. It is not a render time. Do not quote it as one.

## 3. Image starts, film continuation and keyframes (Draft, 480p)

`video_image_conditioning_480p.csv`: 2026-10-09, n = 1 per cell, 864x480, 8 steps, seed 4242, 8-bit decoder for all, same inputs for every engine.
The question was whether FastH3 can replace the slower H3 + Turbo path when a clip starts from an image, continues a previous clip, or travels between two keyframes.

- **Image start**: a first frame is given. The two inputs are Z-Image Turbo renders.
- **Film continuation**: the previous clip's last 22 frames and their audio are anchored at frame 0 (`MiniMaxH3AddGuide`).
- **Keyframes**: a first frame and a last frame are given.
- **SSIM** (ffmpeg `ssim`, "All" average, at 864x480) checks that the anchored frames came out as given. It compares the first and last frame with
  the input images, and the first 22 frames with the guide's last 22.

Each cell shows wall time · director score · SSIM.

| Test | H3 + Turbo 8-step (ComfyUI) | FastH3 (ComfyUI)* | FastH3 (TensorFold) |
|---|---|---|---|
| Image start 1 (railway platform) | 164.8 s · 8 · 0.920 | 120.2 s · 8 · 0.908 | **78.1 s** · 8 · 0.905 |
| Image start 2 (chef) | 130.2 s · 9 · 0.937 | 110.2 s · 9 · 0.930 | **76.1 s** · 8 · 0.930 |
| Film continuation (fisherman) | 154.0 s · 8 · 0.965 | 134.2 s · 7 · 0.956 | **98.1 s** · 6 · 0.955 |
| Keyframes, first / last (lighthouse) | 138.2 s · 6 · 0.755 / 0.840 | 120.2 s · 7 · 0.781 / 0.841 | **84.1 s** · 7 · 0.778 / 0.836 |
| **Mean wall time · director total (of 40)** | **146.8 s · 31** | **121.2 s · 31** | **84.1 s · 29** |

\* The studio never routes FastH3-with-inputs to ComfyUI. This column exists only for comparison.

On mean wall time, the TensorFold column is 1.75x faster than H3 + Turbo and 1.44x faster than FastH3 on ComfyUI. Anchored-frame SSIM on TensorFold is
within 0.005 of FastH3 on ComfyUI in every test. Against H3 + Turbo it runs from 0.015 lower (image start 1) to 0.023 higher (keyframes, first frame). The director gave TensorFold 2 points fewer in total, mostly from the film-continuation clip. Its issues list for that clip
says the fisherman is "already on the boat in the first frame". The build notes say a frame sheet showed the same action from all three engines, but
that check is not in the files.

After this test the studio sends Draft and Fast HD image starts, films and keyframe films to FastH3 on TensorFold.

## 4. Standard and High on TensorFold (experimental) vs Fast HD

`video_standard_high_fasthd.csv`: ComfyUI rows 2026-10-09, TensorFold rows 2026-10-10. n = 1 per row, 5 s, seed 9090, 8-bit decoder for all.
Two prompts: a glassblower (text only), and the chef with a first frame. "SSIM vs ComfyUI" is the mean over the first 120 frames of the TensorFold clip
against the ComfyUI clip made from the same seed. It shows how far the engines drift apart, **not** quality.

The TensorFold Standard and High rows used an **experimental, unshipped** path: full H3 in the engine's dense mode (no sparsity). It needs merged
bf16 checkpoints, with the Turbo LoRA merged in for Standard. Those checkpoints are not distributed.

| Run | Size | Steps | ComfyUI | TensorFold (experimental) | Director, ComfyUI / TensorFold | SSIM vs ComfyUI |
|---|---|---|---|---|---|---|
| Standard, text | 1248x704 | 8 | 332.4 s | 356.3 s ¹ | 5 / 6 | 0.669 |
| Standard, first frame | 1248x704 | 8 | 336.4 s | **310.4 s** (7.7 % less) | 9 / 8 | 0.622 |
| High, text | 1344x768 | 20 | 924.3 s | 878.8 s ¹ (4.9 % less) | 7 / 6 | 0.821 |

¹ The first run on that merged file. According to the build notes it includes the engine's one-time int8 conversion of that file. The conversion was not
timed on its own, so there is no warm number for these two rows.

Verdict at the time: dense attention dominates Standard and High, so the engine saves 5–8 % at best. That was not worth the extra merged
checkpoints (tens of GB each) and the model swaps, so it was not shipped. Standard and High stay on ComfyUI.

**The same two prompts on the shipped Fast HD preset** (FastH3 on TensorFold, higher resolution than Standard):

| Run | Size | Steps | Wall time | vs Standard on ComfyUI | Director | First-frame SSIM |
|---|---|---|---|---|---|---|
| Fast HD, text | 1344x768 | 8 | **198.2 s** | 1.68x faster (332.4 s) | 8 (Standard: 5) | — |
| Fast HD, first frame | 1344x768 | 8 | **212.2 s** | 1.59x faster (336.4 s) | 8 (Standard: 9) | 0.944 (Standard: 0.941) |

These two results are why Fast HD became the studio's default video preset. It is two prompts and an AI judge. Treat it as a
reason to try Fast HD first, not as proof that it beats Standard in general.

## 5. Through the studio, end to end

`studio_jobs_end_to_end.csv`: real jobs submitted through the studio's web app, one job per row (n = 1). Wall time runs from when the job started to
when it finished. It is taken from the job record (1 s resolution) and **includes** everything the job does: every segment render, director reviews
(the per-review seconds are in the CSV), joining, music mixing and posters. Queue waiting time is not included.

| Job | Preset / engine | Output | Wall time | Director |
|---|---|---|---|---|
| 5 s clip (A/B prompt 0) | Draft, ComfyUI, fp16 decoder | 864x480, 5.17 s | 124 s | 8 |
| 5 s clip (A/B prompt 2) | Draft, TensorFold, 8-bit decoder | 864x480, 5.17 s | 72 s | 8 |
| 10 s clip | Fast HD, TensorFold | 1344x768, 10.13 s | 419 s | 6 |
| **20 s film**, 2 guided segments | Draft, TensorFold | 864x480, 20.78 s (498 frames) | **373 s** | 8 / 6 |
| **Keyframe film**, 3 images, 2 x 5 s | Draft, TensorFold | 864x480, 10.33 s (247 frames) | **177 s** | 7 / 7 |
| 10 s clip + ACE-Step score, review off | Draft, TensorFold | 864x480, 10.13 s | 165 s | — |
| Editor: re-render segment 2 of the 20 s film | Draft, TensorFold | one new take | 127 s | 6 |
| Editor: export 20 s film (trim + new take + score) | — | 864x480, 19.75 s | 12 s | — |
| Editor: export keyframe film (cut only) | — | — | 2 s | — |
| Cast: H3 ref2va, official 20-step recipe, synthetic face | ref2va (ComfyUI) | 1344x768, 5.17 s | **979 s** | 8 |
| Music: MiniMax Music 3, 30 s from a one-line idea | — | 30.01 s | 72 s | — |
| Music: MIDI render + ACE-Step restyle (strength 0.55) | — | 37.39 s | 6 s | — |
| Image edit: Qwen-Image 2.1 Turbo, 1 reference | — | 1 image | 26 s | — |

The two A/B rows are the only ones left of a 3-prompt A/B (6 jobs). The other four job records are no longer in the studio's state. See the reported section.
The two rows use **different prompts**, so they are not a matched pair.

## 6. Images: Qwen-Image 2.1 Turbo

`images_qwen_image_21.csv`: 2026-10-10, n = 1 per cell, seed 77, two prompts (a text-heavy travel poster, a photographic portrait), through the studio's image node.

| Model | Weights | Steps / sampling | Size | Poster | Portrait |
|---|---|---|---|---|---|
| Qwen-Image 2.1 Turbo | bf16 | 8, the checkpoint's own sigma schedule, euler, CFG 1 | 1344x768 | **16.1 s** | **16.0 s** |
| Qwen-Image 2.1 Turbo | bf16 | same | 2752x1536 (4.1x the pixels) | 44.0 s | 44.0 s |
| Qwen-Image 2.1 | int8 | 25, euler / simple, CFG 1 | 1344x768 | 24.0 s | 24.0 s |

Runs went in table order for each prompt, so the base model's runs include switching models. The 16.1 s and 16.0 s Turbo runs (first and fourth in the
sequence) suggest that switching costs little, but load time was not measured separately. No timings were kept for the studio's other image models
(Z-Image Turbo, Krea 2, FLUX.2 klein and others).

## 7. Music

`music_ace_step_vs_minimax_music3.csv`: 2026-10-10, n = 1 per cell, three briefs (orchestral score, indie-pop song with lyrics, lo-fi instrumental),
seeds 1010–1012. ACE-Step 1.5 Turbo (8 steps) got style tags + BPM + key. MiniMax Music 3 (int8, 30 steps) got a structured brief.
**Realtime factor = seconds of music delivered / wall seconds** (above 1 = faster than realtime).

| Brief | Asked | ACE-Step 1.5 Turbo: got · wall · realtime | MiniMax Music 3: got · wall · realtime |
|---|---|---|---|
| Score (instrumental) | 60 s | 60.0 s · 16.0 s · **3.75x** | 32.4 s · 72.1 s · 0.45x |
| Song (lyrics) | 60 s | 60.0 s · 8.0 s · **7.49x** | 52.2 s · 112.3 s · 0.47x |
| Lo-fi (instrumental) | 30 s | 30.0 s · 4.0 s · **7.49x** | 9.5 s · 20.0 s · 0.47x |

- ACE-Step delivered exactly the asked length every time. MiniMax Music 3 treats the length as a maximum and plans its own song: it delivered 54 %, 87 % and 32 % of what was asked.
- The ACE-Step times are 4–16 s and can be up to 2 s high (polling), so treat their realtime factors as lower bounds. The 4.0 s lo-fi run took
  between 2 and 4 s, which is 7.5x–15x realtime.
- The first ACE-Step run (score, 16.0 s) was the first music run, so it probably includes loading the 10 GB checkpoint. The 60 s song took 8.0 s right after. Load time was not recorded separately.
- In the studio: Fast = ACE-Step (default), Best = MiniMax Music 3. A 30 s MiniMax Music 3 job took 72 s end to end (section 5).

**MIDI.** A MIDI file is rendered with a General MIDI soundfont (no GPU). It can then be "restyled" by ACE-Step, which re-samples the render at a chosen
strength. One studio job (4 parts, restyle strength 0.55, 37.39 s out) took 6 s end to end (section 5). The restyle-strength sweep is in the reported section.

## 8. Lip-sync with a reference voice

`lipsync_ref2va.csv`: 2026-10-10, n = 1 per row. **All inputs are synthetic:** the face is a Z-Image Turbo render, and the voice is the audio of a Draft clip
of a fictional speaker generated first. MiniMax H3 ref2va int8 at 1344x768, 124 frames, seed 2024.

| Run | Recipe | Wall time |
|---|---|---|
| Voice source clip | Draft, FastH3 on TensorFold, 8-bit decoder | 70 s (logged rounded) |
| Face + reference voice + quoted line | ref2va + ref2v Turbo 4-step LoRA v0.1 + VSA gate patch 0.75 | 228.3 s |
| Control: same face + line, no reference voice | same | 186.3 s |
| Studio Cast job (section 5), synthetic chef face, no reference voice | **official** ref2va recipe: 20 steps, res_multistep, no Turbo LoRA, no gate patch | 979 s |

The 4-step recipe is a speed shortcut from the community. Its picture quality was visibly worse at full size, so the studio's Cast feature uses MiniMax's
official recipe: about 4.3x the time of the 4-step run, for a better picture. Whether the line was spoken and the pitch comparison are in the reported section.

## 9. Seam continuity

No results file was kept for this one. The method is in `benchmarks/scripts/seam_check.py`, and the numbers noted at the time are in the reported section.
The script reads the studio library and renders nothing, so you can run it on your own films.

---

## Reported but not backed by a kept results file

These were measured during the build and written down then, but the raw output was not kept, or the studio job record it came from is gone.
They cannot be checked against data here. Treat them as notes, not results.

| Area | Noted result | Why it is not checkable |
|---|---|---|
| 8-bit video decoder | vs fp16 on the same latent: PSNR 44.5 dB mean (min 42.9 dB), SSIM 0.990 | comparison output not kept |
| FastH3 per step, 480p | TensorFold 6.55 s vs ComfyUI ~11.8 s | engine log not kept (the ComfyUI side matches the baseline's 11.43 / 12.20 s readings) |
| FastH3 one-time conversion | bf16 to int8 conversion 56 s, done once | not timed in the results (the first-run penalty in section 2 is 77.1 s including load) |
| Same seed, two engines | FastH3 ComfyUI vs TensorFold, Draft: SSIM 0.45 | output not kept |
| Standard / High per step | Standard 32 s TensorFold vs ~44 s ComfyUI; High 42 s vs ~51 s | engine log not kept (the ComfyUI side matches the baseline) |
| 3-prompt A/B through the studio | director ComfyUI 8 / 6 / 9 vs TensorFold 8 / 7 / 8 (total 23 each), mean 121 s vs 73 s | only 2 of the 6 job records remain (section 5) |
| Seam continuity | Guided (chain) joins on TensorFold: deepest 50 ms dip 4.4 dB vs 3.5 dB at ordinary points, so no seam dip beyond normal level wobble. Keyframe joins had a 24–27 dB, ~0.1 s hole caused by the studio's own 60 ms fade-out/fade-in. After switching to a one-frame crossfade: 11–12 dB on that pair, about the clips' own quiet edges (−7.7 / −4.4 dB). Film length unchanged (10.327 s / 247 frames). | `seam_check.py` printed but did not save |
| Music lyrics | Whisper transcript of the song brief: MiniMax Music 3 word error rate 0.08; ACE-Step 2 slipped words + "oh oh" filler (no WER) | transcripts not kept |
| Music, studio | ACE-Step 30 s song from the studio: 8 s | job record gone |
| MIDI restyle | chroma match with the MIDI render: 0.89 at strength 0.4, 0.69 at 0.6, 0.41 at 0.8, vs 0.12 for an unrelated piece | `midi_restyle_check.py` printed but did not save |
| MIDI render | plain MIDI render ~1 s | job record gone |
| Lip-sync | with the reference voice: median pitch 170 Hz vs reference 163 Hz (231 Hz without it); the line was transcribed word-perfect | analysis not kept |
| Cast, 4-step recipe in the studio | Draft, synthetic chef face: 87 s, director 8 | job record gone; recipe since retired for quality |
| Image edit | 20–26 s per edit | one job kept (26 s, section 5) |

## Where the build notes and the files disagree

Where they disagree, the tables above use the files.

- **Fast HD vs Standard speed-up.** Notes: "1.65x". Files: 1.68x (text) and 1.59x (first frame), or 1.63x on summed time.
- **Anchored-frame SSIM "within 0.01".** True against FastH3 on ComfyUI (within 0.005). Against H3 + Turbo it is 0.015 lower in one test and 0.023 higher
  in another (section 3).
- **"TensorFold Draft 1.74x at the same seed."** The 76.1 s run used seed 1010, not 1009 (to avoid ComfyUI's cache). The same-seed TensorFold run is the
  8-bit-decoder one: 70.1 s, 1.89x.
- **Standard on TensorFold "310 s warm vs 332/336 s".** Only the first-frame run (310.4 s vs 336.4 s) is warm. The text run took *longer* on TensorFold
  (356.3 s vs 332.4 s), which the notes put down to a one-time conversion that was not timed.
- **Qwen-Image 2.1 Turbo at 2752x1536.** Notes: "44–48 s". Files: 44.0 s for both prompts.
- **MIDI restyle "6 s for 37 s".** The job asked for 34.89 s, delivered 37.39 s, and took 6 s.
- **Fast HD 10 s test clip.** Notes: "409 s render". Job record: 419 s, which includes the 7.2 s director review and the job's other steps.
- **ComfyUI Draft + 8-bit decoder, 16.0 s.** This is in the raw file but is a cache effect (section 2), not a render time. The notes did not quote it either.

## What was left out, and why

- **Rows involving real people or third-party content were not published.** These are two voice-capture jobs whose input was a YouTube video, a
  "recreate a Short" test whose input was a YouTube Short, and one video job whose first frame was a user-uploaded photo. None of them was a benchmark.
- **LLM benchmarks** (director/writer model throughput and evals) are not part of the studio's media benchmarks and are not included.
- **All paths, host names and user names** were removed from the published rows. Output files are named by file name only. Studio job ids are kept so
  rows can be matched to each other.
- Prompts are kept. They are generic scene descriptions.

## How to reproduce

Scripts are in [`benchmarks/scripts/`](../benchmarks/scripts/). Each one starts with a comment that says exactly what it measures. They talk to ComfyUI's
HTTP API and read their settings from the environment, using the same names as `studio.env`:

```bash
set -a; . ./studio.env; set +a          # COMFY_URL, COMFY_DIR, COMFY_MODELS, STUDIO_HOME, STUDIO_URL, TENSORFOLD_LIB
export BENCH_OUT=$PWD/bench-out         # where results (.jsonl) and copies of the outputs go
cd benchmarks/scripts

python3 bench_presets_comfyui.py draft      # section 1 (then: standard, high)
python3 bench_tensorfold_fasth3.py          # section 2
python3 bench_image_conditioning.py         # section 3 (KEYFRAME_VIDEO=<a >=10 s clip> for the keyframe inputs)
python3 bench_standard_high.py              # section 4 (TensorFold Standard/High rows need the unshipped merged files)
python3 bench_qwen_image.py                 # section 6
python3 bench_music.py                      # section 7
python3 midi_restyle_check.py               # MIDI restyle sweep (MIDI_RENDER=<a MIDI render .wav>)
python3 bench_lipsync.py                    # section 8: synthetic face and voice only
python3 seam_check.py                       # section 9: reads the studio library, renders nothing
```

- Each script except `bench_presets_comfyui.py` waits for an empty ComfyUI queue. It pauses while the GPU is at 85 °C or more, and interrupts a render
  if free unified memory drops below 6 GiB. Run them on an otherwise idle machine.
- Director scores need the director running (`DIRECTOR_URL`). Without it, the review field records an error and the timings are unaffected.
- The model files are listed in [MODELS.md](MODELS.md). The TensorFold rows need `TENSORFOLD_LIB` and the FastH3 bf16 checkpoint. The engine converts the
  checkpoint to int8 on first use, so expect one slow first run (section 2).
- Expect different absolute numbers on different software versions, and some run-to-run spread on the same machine. Running each cell several times is the
  most useful thing anyone could add to these results.
