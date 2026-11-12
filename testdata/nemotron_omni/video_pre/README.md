# Nemotron 3.5 Super VL video: reference fixtures

Made by `tools/nemotron_omni/gen_video_fixture.py` from the pack's own HF code (`processing_nemotron_h_omni.py`,
`image_processing_nemotron_h_omni.py`, `modeling_nemotron_h_omni.py`, `modeling_radio.py`; transformers 5.12.1, torch 2.14.0+cu130, aarch64,
SVE128; versions and the SHA-256 of every pack file read are in `meta.json`), CPU only. Inputs are the repository's public test images and
integers: no weight and nothing derived from one is here. The embeddings are synthetic (a seeded integer recipe).

| file | what | used by |
|---|---|---|
| `evs.json` | `compute_retention_mask` of the model on bf16 embeddings: 14 inline cases (T = 1, N = 1, identical tubelets, scaled copies, ternary values, zero vectors, a NaN, a 3 x 4 grid, hidden 8 / 12 / 20 / 36 / 40 / 4096) and 9 seeded cases (hidden 4096, up to 16 tubelets of 256 tokens, 252-token grid 12 x 21, power-of-two dimension scales) rebuilt from their recipe and checked by SHA-256. Per case: the bf16 score tensor the function ranks by (bits) and the mask at 0.7, 0.5, 0.3, 0.9, 0.99, 0.1, 1.0 (keep count and LSB-first bitmap). Seeded cases also record how the scores and the mask move when only the order of the hidden axis changes (a permutation: fp32 accumulation order). | `crates/vision/tests/nemotron_video.rs` |
| `tubelets.json` | the real `forward_video` (stub tower) on 1, 2, 3, 4, 5 and 7 tiny frames: the channel stack and `_image_to_patches` of it; and 144 rows of (frame size, frame count) with the processor's tile, tokens, tubelets and the real function's kept count at 0.7, 0.5, 0.9 | same |
| `videos.json` | three videos from `rings_333x517.png` and `grad_1080x1920.png` (1 frame 333 x 517; 2 frames 1600 x 900; 5 frames 320 x 320, so three tubelets and a padded one), through the real processor: per-frame `pixel_values` (lattice, SHA-256), the tubelet stack (bf16 lattice, SHA-256), the rendered prompt, the expanded text and the ids (count, SHA-256, runs) with six metadata variants of the labels, the model's own pruned ids for the square video at 0.7, 0.5, 0.3, and the second `<video>` tag staying literal | `crates/vision/tests/nemotron_video.rs`, `crates/server/tests/nemotron_omni_video.rs` |
| `meta.json` | versions, pack file hashes, seeds | |

Regenerate (node 4, `~/venvs/nemo-vision-pre`, a few seconds):

```
HF_HUB_OFFLINE=1 python tools/nemotron_omni/gen_video_fixture.py --images <dir with rings_333x517.png and grad_1080x1920.png> --out <dir>
```

Re-check the EVS cases on another device (the M4 GPU worker's question, whether torch's CUDA accumulation order moves the mask):

```
HF_HUB_OFFLINE=1 python tools/nemotron_omni/gen_video_fixture.py --out testdata/nemotron_omni/video_pre --check-device cuda
```

About 0.9 MB.
