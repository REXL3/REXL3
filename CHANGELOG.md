# Changelog

## [v0.1.0] - 2026-11-06

First production release of **REXL3**: an OpenAI-compatible Rust + CUDA 13 (`sm_121`) inference engine (`rexl3-serve` and two-node cluster coordinator `rexl3-cluster`) for NVIDIA GB10 (`aarch64`) hosts, shipping five hash-locked EXL3 runtime bundles qualified at their full context windows (`262,144` or `1,048,576` tokens) with lossless speculative decoding and prefix caching enabled by default.

> SAGE is our own mixed-precision quantization method for EXL3. The method is not published.

### Shipped models and measured GB10 performance

- **Qwen3.8-27B (`qwen3.8/qwen3.8-27b` `v3`, runtime `qwen38-27b-dflash2-exl3-4.00`, port `8889`, PR #66):**
  - Pack: `r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw` (`15.4 GiB`) + DFlash2 draft `r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw` (`1.2 GiB`).
  - Profile `qwen38-single-262k`: 1 GB10, `262,144`-token window, 6 slots, `--spec auto` (DFlash2), 40,960 MiB prefix cache, image input on, default reasoning effort `low`.
  - Measured on GB10: **62.5 tok/s** decode; **808.6 / 621.9 tok/s** prefill (50% / 100%); Q200: GSM8K `79/80` (`98.75%`), HumanEval `39/40` (`97.5%`), IFEval `35/40` (`87.5%`), hard reasoning `20/20` (`100%`), BFCL `15/20` (`75.0%`).

- **Qwen3.8-Flash-Next (`qwen3.8/qwen3.8-flash-next` `v2`, runtime `qwen38-flash-next-exl3-3.05`, port `8899`, PR #73):**
  - Pack: `turboderp/Qwen3.8-Flash-Next-exl3` (`3.05 bpw`, `79.3 GiB`).
  - Profile `qwen38-flash-next-single-262k`: 1 GB10, `262,144`-token window, 2 slots, `--spec mtp`, 22,528 MiB prefix cache, vision off (owner exception through `2027-01-31`).
  - Measured on GB10: **70.3 tok/s** decode; **1,047 / 1,006 tok/s** prefill (50% / 100%); Q200: GSM8K `78/80` (`97.5%`), HumanEval `40/40` (`100%`), IFEval `38/40` (`95.0%`), hard reasoning `20/20` (`100%`), BFCL `9/20` (`45.0%`).

- **GLM-5.3-Flash (`glm-5.3/glm-5.3-flash` `v1`, runtime `glm53-flash-2.25bpw-sm121`, port `8950`, PR #69):**
  - Pack: `r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121` (`2.25 bpw`, `91.8 GiB`).
  - Profile `glm53-flash-single` (default, 1 GB10, `262,144`-token window, 1 slot, `--spec auto` NextN, 3,584 MiB prefix cache, image input on): **39.7 tok/s** decode; **971.4 / 976.3 tok/s** prefill (50% / 100%).
  - Profile `glm53-flash-tp2-1m` (2 GB10s over RoCE, TP=2, `1,048,576`-token window, 1 slot, `--spec auto` NextN, 13,440 MiB prefix cache, image input on): **57.2 tok/s** decode; **1,130.6 / 979.3 tok/s** prefill (50% / 100%).
  - Q200 (both profiles): GSM8K `77/80` (`96.25%`), HumanEval `39/40` (`97.5%`), IFEval `37/40` (`92.5%`), hard reasoning `19/20` (`95.0%`), BFCL `12/20` (`60.0%`).

- **Nemotron-3.5-Super-VL (`nemotron-3.5/nemotron-3.5-super-vl` `v1`, runtime `nemotron35-super-vl-sage-3.85`, port `9100`, PR #71):**
  - Pack: `vcruz305/Nemotron-Super-VL-EXL3-3.85bpw` (mixed-precision SAGE EXL3 `3.85 bpw`, `55.8 GiB`).
  - Profile `nemotron35-super-vl-single-1m`: 1 GB10, `1,048,576`-token window, 16 slots, `--spec auto` (MTP), prefix cache on, image input on (up to 3,328 tokens/image).
  - Measured on GB10: **50.8 tok/s** decode; **879 / 589 tok/s** prefill (50% / 100% of 1M window); Q200: GSM8K `76/80` (`95.0%`), HumanEval `40/40` (`100%`), IFEval `28/40` (`70.0%`), hard reasoning `20/20` (`100%`), BFCL `15/20` (`75.0%`).

- **DeepSeek-V4.1-Flash-1017 (`deepseek-v4.1/deepseek-v4.1-flash-1017` `v1`, runtime `dsv41-flash-1017-sage-1.55`, port `8940`, PR #75):**
  - Pack: `vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw` (mixed-precision SAGE EXL3 `1.55 bpw`, `298.3 GiB`).
  - Profile `dsv41-single-262k`: 1 GB10, `262,144`-token window, 1 slot, `--kv-dtype q4`, `--spec auto` (MTP), prefix cache on, vision off (owner exception through `2026-12-31`). The two-node `dsv41-tp2-1m` profile is planned for `v0.1.1`.
  - Measured on GB10: **29.6 tok/s** decode; **1,274 / 1,034 tok/s** prefill (50% / 100%); Q200: GSM8K `78/80` (`97.5%`), HumanEval `40/40` (`100%`), IFEval `35/40` (`87.5%`), hard reasoning `16/20` (`80.0%`, owner-accepted for `v0.1.0`), BFCL `13/20` (`65.0%`).

### Engine, runtime, and release highlights

- Native Rust + CUDA 13 (`sm_121`) engine (`rexl3-serve` and `rexl3-cluster`) supporting uniform and mixed-bit-rate (`MixedK` K2–K8) EXL3 packs, lossless speculative decoding (`drafted == serial` `24/24`), block-hashed prefix caching (`84/84` exact reuse checks), and chunk-size-invariant prefill (`chunked == whole` across `512`/`256`/`128` chunks up to `60,000` tokens).
- Self-contained `./launch.sh` runtime bundles (`install-deps`, `doctor`, `setup`, `verify`, `start`, `smoke`, `status`, `logs`, `diagnostics`, `stop`) with strict boot-time verification of `release.lock.json`, binary SHA-256, and `model.lock.json` shard hashes.
- Change-scoped release qualification (`T0` carry-forward from freeze commit `799fa750f8eb7d4c45d757d9c44beeb7182b5d51` onto primary release engine `afc9b5dfb69d070a306ffe6696d9b4f96db4ddec`) and per-profile subset approvals (PR #64, PR #68).
- Models: qwen3.8/qwen3.8-27b v3, glm-5.3/glm-5.3-flash v1, nemotron-3.5/nemotron-3.5-super-vl v1, qwen3.8/qwen3.8-flash-next v2, deepseek-v4.1/deepseek-v4.1-flash-1017 v1.

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No unreleased changes yet. Each user-visible change lands as a fragment under
`changes/unreleased/` and moves here at release time (see `docs/RELEASING.md`).
