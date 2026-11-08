# REXL3 v0.1.0 release notes

`v0.1.0` is the first production release of **REXL3**: an OpenAI-compatible inference engine (`rexl3-serve` and two-node cluster coordinator `rexl3-cluster`, built in Rust and CUDA 13 for NVIDIA GB10 `aarch64` `sm_121`) paired with five hash-locked EXL3 model runtime bundles. Every model in `v0.1.0` ships at its full qualified context window (`262,144` tokens on one GB10 or `1,048,576` tokens on one or two GB10s) with lossless speculative decoding and prefix caching enabled by default, and is backed by a complete GB10 qualification evidence archive (`*-evidence.tar.gz`) and corresponding engine source (`rexl3-engine-*-source.tar.gz`).

> SAGE is our own mixed-precision quantization method for EXL3. The method is not published.

## Highlights

- **Qwen3.8-27B (`v3`, `qwen38-27b-dflash2-exl3-4.00`, port `8889`):** EXL3 `4.00 bpw` target pack (`r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw`, `15.4 GiB`) with a DFlash2 `4.00 bpw` draft pack (`r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw`, `1.2 GiB`) on one GB10 (`qwen38-single-262k`): **262,144-token** context window, 6 slots, speculative decoding (`--spec auto`), 40,960 MiB prefix cache, image input on, default reasoning effort `low`.
  - **GB10 throughput (3-run median):** **62.5 tok/s** generation; **808.6 tok/s** (50% window) and **621.9 tok/s** (100% window) prefill.
  - **Q200 quality baseline:** GSM8K `79/80` (`98.75%`), HumanEval `39/40` (`97.5%`), IFEval `35/40` (`87.5%`), hard reasoning `20/20` (`100%`), BFCL hard-20 `15/20` (`75.0%`). (PR #66)

- **Qwen3.8-Flash-Next (`v2`, `qwen38-flash-next-exl3-3.05`, port `8899`):** EXL3 `3.05 bpw` pack (`turboderp/Qwen3.8-Flash-Next-exl3`, `79.3 GiB`) on one GB10 (`qwen38-flash-next-single-262k`): **262,144-token** context window, 2 slots, speculative decoding with the pack's native MTP head (`--spec mtp`), 22,528 MiB prefix cache (86 entries, 488,960-token reach), vision off under an owner exception through `2027-01-31`.
  - **GB10 throughput (3-run median):** **70.3 tok/s** generation; **1,047 tok/s** (50% window) and **1,006 tok/s** (100% window) prefill.
  - **Q200 quality baseline:** GSM8K `78/80` (`97.5%`), HumanEval `40/40` (`100%`), IFEval `38/40` (`95.0%`), hard reasoning `20/20` (`100%`), BFCL hard-20 `9/20` (`45.0%`). (PR #73)

- **GLM-5.3-Flash (`v1`, `glm53-flash-2.25bpw-sm121`, port `8950`):** EXL3 `2.25 bpw` pack (`r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121`, `91.8 GiB`) with NextN MTP speculative decoding (`--spec auto`), prefix cache, and image input (up to 8,000 tokens/image) in two profiles:
  - **`glm53-flash-single` (default, 1 GB10, 262,144-token window, 1 slot, 3,584 MiB prefix cache):** **39.7 tok/s** generation; **971.4 tok/s** (50% window) and **976.3 tok/s** (100% window) prefill.
  - **`glm53-flash-tp2-1m` (2 GB10s over RoCE, TP=2, 1,048,576-token window, 1 slot, 13,440 MiB prefix cache):** **57.2 tok/s** generation; **1,130.6 tok/s** (50% window) and **979.3 tok/s** (100% window) prefill.
  - **Q200 quality baseline (identical on single-node 262K and two-node 1M):** GSM8K `77/80` (`96.25%`), HumanEval `39/40` (`97.5%`), IFEval `37/40` (`92.5%`), hard reasoning `19/20` (`95.0%`), BFCL hard-20 `12/20` (`60.0%`). (PR #69)

- **Nemotron-3.5-Super-VL (`v1`, `nemotron35-super-vl-sage-3.85`, port `9100`):** Mixed-precision SAGE EXL3 `3.85 bpw` pack (`vcruz305/Nemotron-Super-VL-EXL3-3.85bpw`, `55.8 GiB`) on one GB10 (`nemotron35-super-vl-single-1m`): **1,048,576-token** context window on a single node, 16 slots, MTP speculative decoding (`--spec auto`), prefix cache on, image input on (up to 3,328 tokens/image).
  - **GB10 throughput (3-run median):** **50.8 tok/s** generation; **879 tok/s** (50% of 1M window, ~524K tokens) and **589 tok/s** (100% of 1M window, ~1.047M tokens) prefill, with ~27 GiB free memory headroom and zero swap.
  - **Q200 quality baseline:** GSM8K `76/80` (`95.0%`), HumanEval `40/40` (`100%`), IFEval `28/40` (`70.0%`), hard reasoning `20/20` (`100%`), BFCL hard-20 `15/20` (`75.0%`). (PR #71)

- **DeepSeek-V4.1-Flash-1017 (`v1`, `dsv41-flash-1017-sage-1.55`, port `8940`):** Mixed-precision SAGE EXL3 `1.55 bpw` pack (`vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw`, `298.3 GiB`) on one GB10 (`dsv41-single-262k`): **262,144-token** context window, 1 slot, `q4` KV cache, MTP speculative decoding (`--spec auto`), prefix cache on, vision off under an owner exception through `2026-12-31`. *(The two-node `dsv41-tp2-1m` profile is not part of `v0.1.0` and is planned for `v0.1.1`.)*
  - **GB10 throughput (3-run median):** **29.6 tok/s** generation; **1,274 tok/s** (50% window) and **1,034 tok/s** (100% window) prefill.
  - **Q200 quality baseline:** GSM8K `78/80` (`97.5%`), HumanEval `40/40` (`100%`), IFEval `35/40` (`87.5%`), hard reasoning `16/20` (`80.0%`, owner-accepted for `v0.1.0`), BFCL hard-20 `13/20` (`65.0%`). (PR #75)

## Supported models

| Model | Architecture | Pack | Context | Speculative decoding | Prefix cache | Vision | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.8/qwen3.8-27b | Qwen3_5ForConditionalGeneration | EXL3 4.00 bpw + DFlash2 4.00 bpw | 262,144 (single-node) | DFlash2 (k=16, auto) | on | on (image) | qwen38-27b-dflash2-exl3-4.00-v3-evidence.tar.gz |
| glm-5.3/glm-5.3-flash | Glm5NextForConditionalGeneration | EXL3 2.25 bpw | 262,144 (single-node) / 1,048,576 (two-node) | MTP (auto) | on | on (image) | glm53-flash-2.25bpw-sm121-v1-evidence.tar.gz |
| nemotron-3.5/nemotron-3.5-super-vl | NemotronH_Omni_Reasoning_V3 | EXL3 3.85 bpw | 1,048,576 (single-node) | MTP (auto) | on | on (image) | nemotron35-super-vl-sage-3.85-v1-evidence.tar.gz |
| qwen3.8/qwen3.8-flash-next | Qwen4ExpForConditionalGeneration | EXL3 3.05 bpw | 262,144 (single-node) | MTP (auto) | on | off (owner exception through 2027-01-31) | qwen38-flash-next-exl3-3.05-v2-evidence.tar.gz |
| deepseek-v4.1/deepseek-v4.1-flash-1017 | DeepseekV41ForCausalLM | EXL3 1.55 bpw | 262,144 (single-node, dsv41-single-262k) | MTP (auto) | on | off (owner exception through 2026-12-31) | dsv41-flash-1017-sage-1.55-v1-evidence.tar.gz |

## New features and improvements

- **Native Rust + CUDA 13 (`sm_121`) GB10 inference engine (`rexl3-serve` and `rexl3-cluster`):**
  - Purpose-built for NVIDIA GB10 (`aarch64`, `sm_121`) unified-memory hosts, serving uniform and mixed-bit-rate (`MixedK`, K2–K8 per projection) EXL3 packs with zero register spilling on hot decode and prefill paths.
  - Family-specific fused CUDA kernels for Qwen3.8 (`Qwen3_5ForConditionalGeneration`), Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration` with GDN recurrence and hyper-connection mixing), GLM-5.3-Flash (`Glm5NextForConditionalGeneration`), Nemotron-3.5-Super-VL (`NemotronH_Omni_Reasoning_V3` Mamba-2 + latent MoE + vision tower), and DeepSeek-V4.1-Flash-1017 (`DeepseekV41ForCausalLM` MLA + `q4` KV cache + routed MoE).
  - Two-node tensor-parallel (`TP=2`) serving over RoCE via `rexl3-cluster` for `glm53-flash-tp2-1m` at `1,048,576` tokens.

- **Lossless speculative decoding (`DFlash2` and `MTP`) and exact prefix caching:**
  - Speculative decoding is enabled by default on every shipped profile (`--spec auto` or `--spec mtp`), verified bit-exact against serial greedy decoding (`drafted == serial`, `24/24` pairs in the exactness gate).
  - Automatic prefix caching (`--prefix-cache on`) reuses shared prompt prefixes across turns with bit-exact parity (`84/84` live prefix-cache verification checks) and chunk-size-invariant prefill (`chunked == whole` across `512`, `256`, and `128` chunk sizes from `1,500` to `60,000` tokens).

- **Self-contained, hash-locked runtime bundles (`./launch.sh`):**
  - Each model ships as a verifiable bundle (`<runtime-id>-v<N>-bundle.tar.gz`) with `launch.sh` lifecycle commands (`install-deps`, `doctor`, `setup --accept-model-download`, `verify`, `start`, `smoke`, `status`, `logs`, `diagnostics`, `stop`, `reset-state`).
  - Every boot verifies the bundle inventory (`release.lock.json`), binary SHA-256, and full model shard hashes (`model.lock.json`) before loading weights.

- **Change-scoped release qualification and per-profile subset approvals:**
  - Models whose family and kernel scope are untouched between the group freeze (`799fa750`) and the release engine (`afc9b5df`) carry forward their sealed `T0` approvals alongside their qualified engine binaries (`engine-v0.1.0-799fa750`), while models qualified on the release engine ship with `engine-v0.1.0-afc9b5df`.
  - A sealed approval may qualify a subset of the profiles packaged in a runtime tree (such as `dsv41-single-262k` alone), leaving unapproved sibling profiles outside the release contract. (PR #64, PR #68)

## Bug fixes

- **DeepSeek-V4.1 chunk-size invariance and long-context prefix-cache exactness (`afc9b5df`):**
  - Fixed the `KvFmt::Q4` window-cache transition across multi-chunk prefills (`start_pos == 0` vs `start_pos > 0` in layers 0–19), row-count dispatch in `dsv4_bf16` (`m <= 16` vs `m > 16`) and `dsv4_moe_group`, and partial-block tail replay when resuming from a cached prefix snapshot at `262,144` context. `check_chunked_eq_whole.py` (`4/4` across `1,500`–`60,000` tokens at `512`/`256`/`128` chunks) and `check_prefix_cache.py` (`84/84` checks, `34` cache hits) now pass bit-identically.
- **Release asset verification and multi-model staging (`tools/cut_release.py`, `tools/promote_runtime.py`):**
  - `cut_release.py --publish` stages each model's dev release in its own directory and downloads the exact engine binaries named in `approval.json["engine_binaries"]` from the corresponding dev engine release (`engine-v0.1.0-afc9b5df` or `engine-v0.1.0-799fa750`), verifying size and SHA-256 before publishing. (PR #49, PR #51, PR #52, PR #58)

## Breaking changes and upgrade notes

- `v0.1.0` uses the hierarchical `runtimes/<family>/<model>/v<N>/` layout and `<runtime-id>-v<N>-*` release asset naming. The pre-release flat `runtimes/<runtime-id>/` layout and per-model release tags are retired.
- Any profile that ships with vision disabled (`--no-vision`) requires a non-expired maintainer vision exception recorded in its sealed qualification receipt. (PR #53)

## Known issues

- **DeepSeek-V4.1-Flash-1017 (`dsv41-single-262k`):**
  - At `temperature: 0` (greedy decoding), the model can repeat a short phrase in its reasoning until `max_tokens`; in the `Q200` quality run (greedy, 8,192-token cap, thinking off) 7 of 180 text prompts hit the cap and were counted incorrect (including 4 in `hard_reasoning`, which scored `16/20`, one case below the `17/20` regression bar, and was accepted by the maintainers for `v0.1.0`). Use the pack's default sampling (`temperature: 1.0`, `top_p: 0.95`) for long reasoning and set `max_tokens` to bound replies.
  - Vision is off (`--no-vision`) under an owner exception expiring `2026-12-31`; image, video, and audio inputs are refused.
  - Free host memory at the `262,144`-token window reaches ~0.7 GiB `MemAvailable` under full load; do not run other workloads on the node.
  - The packaged `dsv41-tp2-1m` profile is **not** part of `v0.1.0` and is planned for `v0.1.1`.

- **GLM-5.3-Flash (`glm53-flash-single` and `glm53-flash-tp2-1m`):**
  - On one node (`glm53-flash-single`), the 3,584 MiB prefix cache holds 2 entries (one full-window conversation at a time); a second long conversation evicts the first, which will prefill cold on its next turn. Replies are unchanged.
  - The 1,048,576-token window requires the two-node profile (`glm53-flash-tp2-1m`) over a private, trusted RoCE network; `glm53-flash-tp2-1m` binds the leader node's network address without authentication or TLS, so run it only on a private fabric or behind a firewall/SSH tunnel.
  - Thinking has no off switch (`enable_thinking: false` is a no-op noted in `X-Rexl3-Note`); scale reasoning with `reasoning_effort`.
  - Under greedy decoding (`temperature: 0`) near the 1,048,576-token limit, the model can start a second turn after answering; leave `temperature` and `top_p` at their defaults (`1.0` and `0.95`).

- **Nemotron-3.5-Super-VL (`nemotron35-super-vl-single-1m`):**
  - Thinking is **off** by default when a request omits `enable_thinking` / `reasoning_effort`; when turning thinking on, set `max_tokens` generously (in `Q200` at `reasoning_effort: xhigh` with a 65,536-token budget, 8 of 40 IFEval prompts spent the full budget in thinking).
  - Accepts inline images up to 3,328 vision tokens per image; encoded video files (`mp4`, `webm`, `gif`) and audio are refused.

- **Qwen3.8-27B (`qwen38-single-262k`):**
  - Launches only with the target and DFlash2 draft packs downloaded into the launcher's managed state directory (`./launch.sh --yes setup --accept-model-download`); `model_dir` in `site.json` is not supported because the recipe locks two pack roles (`target` and `draft`).
  - Video, audio, and constrained JSON response formats (`json_object`, `json_schema`) are refused.

- **Qwen3.8-Flash-Next (`qwen38-flash-next-single-262k`):**
  - The served model name in `GET /v1/models` is `default`.
  - Vision is off (`--no-vision`) under an owner exception expiring `2027-01-31` while vision-tower support for `Qwen4ExpForConditionalGeneration` is in development; image input is refused with HTTP 400.
  - MTP speculative decoding drafts while one slot is decoding; when both slots decode concurrently, generation runs without drafts (output tokens are identical).

- **Cross-model API notes:**
  - Served model names (`model` in request bodies, matching `GET /v1/models`) are: `qwen38-27b-exl3` (Qwen3.8-27B), `default` (Qwen3.8-Flash-Next), `GLM-5.3-Flash-EXL3-2.25bpw-sm121` (GLM-5.3-Flash), `nemotron-3.5-super-vl` (Nemotron-3.5-Super-VL), and `DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw` (DeepSeek-V4.1-Flash-1017). Always read the served ID from `GET /v1/models` or `profile.json`.

## Assets and checksums

- `rexl3-cluster-0.1.0-afc9b5df-linux-aarch64-sm121` sha256 `29bf146f707e1a6acef7ef1f45a47b3671f6d7112cf66254c7574c5f13c4f306`
- `rexl3-engine-0.1.0-afc9b5df-source.tar.gz` sha256 `ade2a05b0629cbbbc8921ee945cad18ffff1e899a3f9c83dd74b24504fe7a046`
- `rexl3-serve-0.1.0-afc9b5df-linux-aarch64-sm121` sha256 `40e23e7116eff881e3a4545304cd82f18f84c532ae2f7b68506af02373e48318`
- `rexl3-serve-0.1.0-799fa750-linux-aarch64-sm121` sha256 `3e188724f538d25f7d354e34865a2205c266274171f1d000f764fa5093ad4041`
- `rexl3-cluster-0.1.0-799fa750-linux-aarch64-sm121` sha256 `360a04ea4bc99dfea3da90cf47c87cb1bf815de69049fd5ed538e37273b2c409`
- qwen3.8/qwen3.8-27b v3: approval sha256 `d26800e692a5ac147a774da8505cd36b77a555cfb79414b2329ca923c38441de`
- glm-5.3/glm-5.3-flash v1: approval sha256 `7d1cca67cc1d39871eabc9c786530dda50719c96339a2ee9d1ad3dcafc39492f`
- nemotron-3.5/nemotron-3.5-super-vl v1: approval sha256 `bb6a5a00929fd05185f74c91d9d4aca6f955e94c43827413eb0d66b89abd4548`
- qwen3.8/qwen3.8-flash-next v2: approval sha256 `fdbd312cbf6d7345bbb61c9c803d11771933511ed332b3218e1630ac50b7ab2e`
- deepseek-v4.1/deepseek-v4.1-flash-1017 v1: approval sha256 `2485106e16499278c95c49c846992018e1c5eb227b738ae0a2e37fb8cf649d73`

## Provenance

- Engine dev commit `afc9b5dfb69d070a306ffe6696d9b4f96db4ddec` tree `1580b494a8ae436fb938afeb8f77a76052d3f6e8`; dev engine release tag `engine-v0.1.0-afc9b5df`.
- Models sealed from the engine dev commit `afc9b5dfb69d070a306ffe6696d9b4f96db4ddec`: qwen3.8/qwen3.8-flash-next v2, deepseek-v4.1/deepseek-v4.1-flash-1017 v1.
- qwen3.8/qwen3.8-27b v3 carries forward its previously sealed approval at commit `799fa750f8eb7d4c45d757d9c44beeb7182b5d51` (engine release `engine-v0.1.0-799fa750`, tier `T0`).
- glm-5.3/glm-5.3-flash v1 carries forward its previously sealed approval at commit `799fa750f8eb7d4c45d757d9c44beeb7182b5d51` (engine release `engine-v0.1.0-799fa750`, tier `T0`).
- nemotron-3.5/nemotron-3.5-super-vl v1 carries forward its previously sealed approval at commit `799fa750f8eb7d4c45d757d9c44beeb7182b5d51` (engine release `engine-v0.1.0-799fa750`, tier `T0`).
