# Nemotron 3.5 Super VL vision reference (RADIO ViT-H, pixel shuffle, mlp1)

Small fixtures from `tools/dump_radio_reference.py`. The reference is the pack's own code (`modeling_radio.py`,
`modeling_nemotron_h_omni.py`, imported in place) run on the pack's own bf16 weights, read strictly through
`model.safetensors.index.json`. The language model is never built. Full dumps (about 23 MB) stay on node 4 and are not
committed.

## Files

- `radio_ref_fixture.safetensors` (603 KB). For each shape `512x512` and `320x448` (keys `<shape>.<name>`):
  - `pixels_bf16_rows` [3, 2, W]: channel by (row 0, row H-1) of the normalized bf16 input.
  - `block_rows` [33, 3, 1280]: stage 0 is the embeddings (cls and register prefix plus patches, position added), stages
    1 to 32 are the outputs of blocks 0 to 31. Rows are token 0 (cls), token 10 (first patch) and the last token.
  - `final_ln_rows` [2, 1280], `pixel_shuffle_rows` [2, 5120], `mlp1_rows` [2, 4096]: the first and last projector tokens.
    `pixel_shuffle` is the input to mlp1, after vision_final_layernorm and the pixel shuffle.
- `meta.json` (110 KB): the versions (Python 3.12.13, torch 2.14.0+cu130, CUDA 13.0, transformers 5.12.1, numpy 2.5.3,
  NVIDIA GB10), the SHA-256 of the six pack `.py` files, `model.safetensors.index.json`, `config.json` and the three vision
  shards, each checked against the pack's `CHECKSUMS.sha256`, the image recipe, the full-tensor SHA-256 and sums of every
  stage, and the reimplementation check.

The fixture rows were checked bitwise against the full dump (`512x512.block_rows` and `512x512.mlp1_rows`). `meta.json` is
byte-identical to the full run's.

## Reimplementation check (acceptance 1)

`metric`: rel_max = max|a - b| / max|b|, computed in fp32, with a the re-implementation and b the pack classes. rel_l2 =
||a - b|| / ||b||. The 1e-3 threshold is applied to rel_max, over all 37 stages (embeddings, 32 blocks, tower features,
final layernorm, pixel shuffle, mlp1), at 512x512 and 320x448, on the synthetic image and on random input.

Result: 0.0 at every stage, in bf16 and in fp32. Every element is bitwise equal (0 mismatches). The re-implementation
reads raw checkpoint tensors and uses no pack modules. The attention is eager in both (matmul, scale, softmax, matmul).

## Determinism (acceptance 2)

Two separate processes (`--out run1`, then `--out run2`) gave byte-identical directories: `meta.json` and both
`ref_*.safetensors` files, by `filecmp`. Each process also repeats the pack forward in-process, and every stage matched.
The first attempt differed in the safetensors header only: the metadata keys came out in a different order between
processes, while every tensor was bitwise equal. The writer now stores its metadata under one key.

## Noise floor

The bf16 pack against the fp32 pack on the same inputs, in the same metric. At mlp1: rel_max 4.0e-2 (512x512), 5.0e-2
(320x448); rel_l2 3.9e-2. At the tower output, rel_l2 is 1.4e-2 to 1.6e-2 (block 31: 1.3e-2 to 1.6e-2). A bf16 engine
should be judged against this floor, not against the 1e-3 threshold, which applies only to the re-implementation against
the pack in the same precision.

## Findings that differ from the handoff summary

- Position embedding. The pack does not resample the 128x128 grid directly to (h, w). It resamples to a square
  max(h, w) grid (bilinear, align_corners=False) and crops to (h, w). For a square grid the two are identical. For
  320x448 (20x28 patches) the square-then-crop differs from a direct resample by up to 0.74 in absolute terms.
  The engine must follow the pack's square-then-crop.
- vision_final_layernorm is applied, with bias, eps 1e-6, because llm_config.num_nextn_predict_layers is 1. The
  checkpoint carries `vision_projector.vision_final_layernorm.{weight,bias}`.
- Layer scale has no checkpoint tensors. The pack initializes lambda1 to 1.0, so it is an exact identity.
- The pixel shuffle is ps_version v2. Channels are ordered (row inside the 2x2 block, column inside it, channel).
- Normalization is external. The composition calls `make_preprocessor_external()` and the checkpoint's
  `input_conditioner` stats equal the config's norm_mean and norm_std (asserted within 1e-6). The image is normalized
  as `(u8 / 255 - mean) / std` in fp32, then cast to bf16.
- `config.json` uses the marker `{"__float__": "Infinity"}` in the LM `time_step_limit`, which the transformers 5.12.1
  config rejects. The dump decodes it to `float("inf")`. This only affects the LM config, not the vision path.

## Not covered

- The video path (`video_embedder` is loaded, but `forward_video` is not run).
- The image processor's resize and tiling. The inputs are the fixed 512x512 and 320x448 synthetic images.
- The SDPA attention path. The reference uses eager, and the pack's default SDPA path is not compared.

## Reproduce

On node 4 only, under the GPU lock, from a synced copy of this repository. The interpreter is a dedicated venv with
torch 2.14.0+cu130 and transformers 5.12.1 (`~/projects/nemo-vision-ref/venv`, not in the repository; the fork venv has no
transformers):

    python -I tools/dump_radio_reference.py --out DIR1 --fixture-dir DIR_FIX
    python -I tools/dump_radio_reference.py --out DIR2
    python -I tools/dump_radio_reference.py --compare DIR1 DIR2
