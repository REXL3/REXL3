# Qwen3.8-27B EXL3 4.00 bpw with the DFlash2 draft: frozen configuration, not qualification

One normalized choice, `qwen38-single-262k`: one GB10, 262,144 context, 6 slots, native speculative decoding
(`--spec auto` with the DFlash2 draft pack), prefix cache on, vision on. Capabilities: text, SSE, reasoning, tools, images,
prefix cache. Audio, video and constrained JSON response formats (`json_object`, `json_schema`) are refused by the server.
There is no tensor-parallel choice. Configuration freeze does not imply throughput, long-context, multimodal or quality
qualification on the GB10; `qualification.state` is `unverified` and no measurement is transferred from earlier runs.

## Where the arguments come from

The argv is normalized from the documented serve line in `docs/examples/rexl3-serve.service` (`profile.json` `source` names
that file at an existing commit of the development repository and its raw SHA-256). Changes made while normalizing:

- `--queue-depth 64`, `--queue-timeout 300` and `--shutdown-grace 8` are dropped: each equals the server default and none is a
  serving-contract option (CONTRIBUTING section 6).
- `--spec auto` and `--prefix-cache on` are pinned as contract options. The unit relies on the defaults; section 6.1 requires
  both to be explicit.
- The host, port and the two pack directories are managed tokens. The port is the documented 8889.
- `--kv-dtype` is not passed: the Qwen engine keeps a bf16 KV cache and the flag only selects the DeepSeek and Nemotron caches.
- `--no-video` is not passed: video input is turned on only for a family whose tower encodes video, and the Qwen family does
  not (`ModelFamily::encodes_video` defaults to `false` and `families/qwen.rs` does not override it), so a video part is refused
  by name already. `capabilities.video` is `false` for that reason.
- `--no-vision` is not passed (CONTRIBUTING section 6.1, rule 15).

## Launching: managed models only, no `--site model_dir`

The pack has two roles, `target` and `draft`. `runtimes/template-v1/scripts/runtime.py` (the `model_dir` overlay branch,
`EXISTING_MODEL_ROLES_MISMATCH`) accepts an existing model copy only for a single-role lock, and the single-node `site_fields`
carry no draft directory. So the profile validates and the launcher can run it with the packs it downloads itself
(`@MODEL_DIR@` and `@DRAFT_DIR@`), but it cannot yet be pointed at existing local copies through `--site model_dir`. Mapping a
draft root in the adapter is a change to the shared scripts and is not made here.

## Sampling defaults and thinking

- The pack ships `generation_config.json` with temperature 1.0, top_p 0.95 and top_k 20, and the server reads all three from
  it (`GenDefaults::from_dir`), so no engine fallback is involved. The base model card also lists min_p 0.0, presence penalty
  0.0 and repetition penalty 1.0 for thinking mode; the server refuses non-zero penalties. The branch that adds
  `--default-temperature`, `--default-top-p` and `--default-top-k` has not landed, so these values are documented here and
  are not passed as flags.
- **Default reasoning effort is `low`** (`--default-reasoning-effort low` in the profile; owner decision 2026-10-26: low is the recommended
  setting for this model). A request that names no `reasoning_effort` and does not set `enable_thinking: false` is served with thinking on at low effort (`enable_thinking: true` alone gets low as well).
  It is a default, not a fixed value: a request that names an effort (`low`, `medium`, `xhigh`) or `enable_thinking: false` keeps its own. Because requests that name nothing now think, the gate's reply budgets are raised in the recipe's `gate_params` (2,048 to 4,096 tokens) and the profile's smoke request has 2,048 tokens, as for the other always-thinking runtimes.
  The pack's template itself defaults to `xhigh`, and the engine's own default (`--default-thinking off`) would not think at all, so
  without this flag a client that sends nothing gets a different model behaviour from the recommended one. Earlier measurement, with the
  template's own `xhigh` default (`--default-thinking template`) on one GB10 node: every baseline stage passed except "NIAH 100% + tool"
  (no `report_codes` call): that stage fills the window to 261,696 tokens and leaves 256 tokens of reply, and thinking spent them before the
  tool call. Decode was 63.0 tok/s with thinking on. Low effort thinks much less; whether it leaves enough of those 256 tokens is what the
  GB10 receipt of the change that introduced this flag shows. If that stage fails, the fix is a window-aware reply budget, not a different
  default.

## Measured once on one node; not yet on a second

- `speed_floors` in `scripts/gates/recipes/qwen38-27b-dflash2-exl3-4.00.json` are 80% (rounded down) of what the baseline
  gate measured on one GB10 node (N2) before the fused prefill GEMM: decode 44.7 tok/s (sampled), prefill 773 tok/s at 50% of
  the window and 597 at 100%. The fused prefill measured 827 and 626, so the floors hold with or without it. A second node
  has not confirmed them.
- The full baseline gate passed on that node with this profile's contract (6 slots, 262,144 tokens, vision tower loaded,
  `--prefix-cache on`): the cache stayed engaged (stage 5 cached 7,958 tokens, stage 6 cached tokens rose with each turn)
  and the lowest free memory seen was 68 GiB.
- The `expect_log` and `forbid_log` markers matched a live boot (gate stages 1 and 5), and the vision stage passed.
- `readiness_timeout_s` (600) is a chosen ceiling. A cold boot measured about 5 s on a GB10 node with the pack on local disk.

## Model identity

Both packs are public. The lock pins the Hugging Face revisions
`3f1771b8c21f83cbb8e82169559ced9f38ca04e5` (target) and `265b5240592907d2d55ff0dc4d5f66569692604d` (draft). Every file digest
is the SHA-256 of the bytes of a read-only copy of the pack. The two safetensors shards of the target and the draft's
`model.safetensors` also equal the Hugging Face LFS object ids and the shard digests in `crates/model/src/fingerprint.rs`; the
small files were checked against the git blob ids of the same revisions. The pack fingerprints equal the engine's validated
pair, so `--allow-unvalidated-pack` is never needed. Only `README.md`, `LICENSE` and `.gitattributes` may sit beside the
locked files unlisted.

## Dependencies

`dependencies.lock.json` and `requirements-runtime.lock` are those of the DeepSeek runtime, which uses the same launcher
scripts and Hugging Face client, with one change: `libnccl2` is not listed, because the single-node engine does not link or
load it (it is loaded only by the tensor-parallel path). They have not been rehearsed on a clean host for this runtime.
