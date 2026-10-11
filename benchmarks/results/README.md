# Raw benchmark rows

The rows behind the tables in [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md). One row = one run. Measured 2026-10-09/10 on one DGX Spark.

| File | Section | Rows | Made by |
|---|---|---|---|
| `video_presets_comfyui_baseline.csv` | 1 | 3 | `scripts/bench_presets_comfyui.py` |
| `video_fasth3_comfyui_vs_tensorfold.csv` | 2 | 7 | `scripts/bench_tensorfold_fasth3.py` |
| `video_image_conditioning_480p.csv` | 3 | 12 | `scripts/bench_image_conditioning.py` |
| `video_standard_high_fasthd.csv` | 4 | 8 | `scripts/bench_standard_high.py` |
| `studio_jobs_end_to_end.csv` | 5 | 13 | the studio's own job records (jobs submitted through the web app) |
| `images_qwen_image_21.csv` | 6 | 6 | `scripts/bench_qwen_image.py` |
| `music_ace_step_vs_minimax_music3.csv` | 7 | 6 | `scripts/bench_music.py` |
| `lipsync_ref2va.csv` | 8 | 3 | `scripts/bench_lipsync.py` |

Common columns:
- `wall_s`: seconds from submitting the graph to ComfyUI reporting it done. Polled every 2 s (3 s in the baseline), so a value can be up to that
  much longer than the true time. In `studio_jobs_end_to_end.csv` it is the job's start-to-finish time instead (1 s resolution), which includes director reviews,
  joins and mixing.
- `director_score`: 1–10 from the studio's vision-language director, which watches the picture only (640 px, 8 fps). It is an AI judge.
  `director_issues` holds its issue list, joined with ` | `.
- `*_ssim`: ffmpeg `ssim` filter, "All" average. `ssim_vs_comfyui_first120` compares the two engines' clips from the same seed, which measures drift, not quality.
- `output`: file name only.
- `note`: read it. Two rows are not ordinary render times: `D_comfy_480_vae8` is a cache effect (decode only), and `B1_tf_480_first` includes a
  one-time conversion.

## Sanitizing

- Absolute paths, host names and user names were removed. Output files are given by file name only.
- Studio job ids (`1009-215213-dcbe` and the like) are kept so rows can be matched.
- Prompts are kept. They are generic scene descriptions. The faces in the lip-sync and Cast rows are Z-Image Turbo renders, and the reference voice is
  the audio of a generated clip of a fictional speaker.
- **Not published:** two voice-capture jobs whose input was a YouTube video, a "recreate a Short" test whose input was a YouTube Short, and one video
  job whose first frame was a user-uploaded photo. None of them was a benchmark run. LLM benchmarks are not part of the studio and are not included.
