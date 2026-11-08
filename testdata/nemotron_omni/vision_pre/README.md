# Nemotron 3.5 Super VL image preprocessing: reference fixtures

Made by `tools/nemotron_omni/gen_vision_pre_fixture.py` from the pack's own HF processor (transformers 5.12.1, torch 2.14.0+cu130, aarch64,
SVE128; versions and the SHA-256 of every pack file read are in `meta.json`). Inputs are the repository's public test images and prompts written
in the script; no weight and nothing derived from one is here.

| file | what | used by |
|---|---|---|
| `table.json` | tile `[w_patches, h_patches, tokens]` (or `null` for a Python exception) of 3,600 `(w, h)` pairs: every multiple of 16 and its neighbours, extreme aspect ratios, photo sizes, log-uniform pairs (sides 1 to 8,000, ten pairs past that), for `max_model_len` 16,384 / 2,048 / 1,028 / 260, five images in one call at 1,028, and the video rule | `crates/vision/tests/nemotron_preprocess.rs` |
| `images.json` | eight PNGs of `testdata/vision` through the processor: tile, tokens, `pixel_values` on a lattice (float32, exact), per-channel sums, the SHA-256 of the whole tensor; and the 3 x 256 table of `(v / 255 - mean) / std` | same |
| `texts.json` | eight chat prompts (one image, two images, an image in a multi-turn chat, no image, a typed `<image>`, images in two messages, thinking off): the template's text, the text the processor hands the tokenizer, `input_ids` count and SHA-256, the runs of the `<image>` id | `crates/server/tests/nemotron_omni_vision.rs` |
| `meta.json` | versions, pack file hashes, configuration, self-check result | |
| `preprocessor_config.json` | the pack's file, unchanged | the config parser tests |

About 1.6 MB.
