# Optional: the TensorFold engine for FastH3

TensorFold's CUDA engine runs FastH3's 50 transformer blocks in int8 on the GB10's tensor cores. ComfyUI keeps the text
encoder, sampler and decoders. On a DGX Spark it renders **Draft 1.7–1.9x faster and Fast HD 1.4x faster** than
ComfyUI alone (see [docs/BENCHMARKS.md](../docs/BENCHMARKS.md)). Without it the studio uses ComfyUI for everything, and the
*Engine* menu greys TensorFold out.

Both projects are third-party and permissively licensed; this repo only adds a 13-line patch.

| part | source | tested at |
|---|---|---|
| ComfyUI node `TensorFoldH3Blocks` | [drowzeys/keys-TensorFold-Studio-Mac-DGXSpark-64GB-VRam-Min](https://github.com/drowzeys/keys-TensorFold-Studio-Mac-DGXSpark-64GB-VRam-Min) (Apache-2.0) | `d99d0895` |
| engine `libtf_h3.so` | [drowzeys/TensorFold](https://github.com/drowzeys/TensorFold) `h3-cuda` (MIT) | `16cecde2`, built with CUDA 13.0 + Zig 0.17.0 |

1. Install the node into `ComfyUI/custom_nodes/tensorfold_h3/` and build `libtf_h3.so` as that repo's README describes,
   at the commits above.
2. Apply the patch, which keeps only one TensorFold model in memory at a time (FastH3, H3 and H3 Turbo are about 21 GB each):
   `cd ComfyUI/custom_nodes/tensorfold_h3 && patch -p1 < /path/to/lion-ai-studios/tensorfold/one-engine-resident.patch`
3. Download the bf16 FastH3 checkpoint the engine reads: `python3 scripts/download_models.py --file fastvideo_fasth3_8step_v2_pruned_bf16.safetensors`.
   On first use the engine converts it once and saves `<file>.tf-int8` next to it.
4. Set `TENSORFOLD_LIB` in `studio.env` if the library is not at `ComfyUI/custom_nodes/tensorfold_h3/libtf_h3.so`, then
   restart ComfyUI.
