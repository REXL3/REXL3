# Nemotron-3.5-Super-VL SAGE EXL3 3.85 bpw: frozen configuration, not qualification

One profile, `nemotron35-super-vl-single-1m` (the default): one GB10, 1,048,576-token
context, 16 slots, speculation `auto` with the in-pack MTP head, prefix cache on, vision on.
Capabilities: text, SSE, reasoning, tools, images, prefix cache. Video is not a declared
capability of this profile (`capabilities.video` is `false`), so it is not part of the
release contract; the argv does not pass `--no-video`, and the engine's video limits are kept
below for reference. Audio and constrained JSON response formats (`json_object`,
`json_schema`) are refused by the server.

State: `final` (`handoff.json`). `profile.json`, `model.lock.json`, `dependencies.lock.json`
and `requirements-runtime.lock` are the final, sealed files of this entry. Configuration freeze
does not imply throughput, long-context, multimodal or quality qualification on the GB10:
`qualification.state` is `unverified`, no measurement is transferred from earlier runs, and the
gate results for the release binary travel with the release approval, not in this file.

The entry is `runtimes/nemotron-3.5/nemotron-3.5-super-vl/` (family/model layout,
`MODEL.json` in `rexl3.prod-model.v1`). The runtime id `nemotron35-super-vl-sage-3.85`
follows the SAGE convention (`dsv41-flash-1017-sage-1.55`,
`redsnow-5.3-flash-sage-2.49`); pack details are carried by `MODEL.json`, never by
folder names.

## Pack identity and validation

- Pack: `vcruz305/Nemotron-Super-VL-EXL3-3.85bpw` at revision
  `3963815246eda892a4c3b88e18d0d1c39288c8e0` (8 weight shards, 55.77 GiB; index sha256
  `75291cea51e97609ab93dc51f1ec5b6884e169fdc02baf3c367ca15db9cfc75a`).
  `architectures` = `NemotronH_Omni_Reasoning_V3`, `model_type` = `nemotron_h_omni`
  (verified via a public Hub resolve of the pack's `config.json` at the pinned
  revision); `llm_config.max_position_embeddings` = 1,048,576.
- Fingerprint `sha256:b17cec7004215df7f3b303bca1af552ca71b026dbb301a2fd94183eb766a1826`
  is already registered in `crates/model/src/fingerprint.rs` `VALIDATED_TEXT`
  (`NEMOTRON_MIXEDK` / `FP_NEMOTRON_MIXEDK`, from the v0.1.2 pin). The
  pack boots without `--allow-unvalidated-pack`; the flag is never passed and stays
  forbidden.

## Serving contract (window: 1,048,576 — the inventory recommendation)

The serving contract is the profile the v0.1.x campaigns qualified on one GB10
(1M window, 16 slots, speculation auto with the in-pack MTP head, prefix cache on,
vision on):

```
rexl3-serve --model PACK --slots 16 --max-model-len 1048576 --spec auto \
  --prefix-cache on --request-timeout 7200 \
  --queue-timeout 6900 \
  --served-model-name nemotron-3.5-super-vl
```

Pinned explicitly: model, served name (`nemotron-3.5-super-vl`), host/port,
`--max-model-len` 1048576, `--slots 16`, `--spec auto`, `--prefix-cache on`,
`--request-timeout 7200` and `--queue-timeout 6900`. `--spec auto` matches the engine
default and is kept because it is a serving-contract option. The final profile does not
pass `--prefill-chunk`: the engine default (4096) applies. Vision is on by default (no `--no-vision`; the vision tower is built in the
engine). No family environment variables are set.

**The window is a single clearly-marked value** in the recipe (`max_model_len`), the
cluster config (`max_model_len`) and the release requirements (`maximum_context` /
`required_context`). Default is 1,048,576 (the inventory recommendation: the pack's
native context claim, the qualified v0.1.x profile, and MixedK's measured 27 GiB min
MemAvailable clears the 20 GiB rule). Switching to 262,144 is a supported
fast-validation alternative (about 5 h less GB10 time; see
`reports/nemotron-runtime-prep.md` for the per-stage timing table); when switching,
also adjust the recipe `speed_floors`, the requirements `context_windows` and the
quality lane's `WINDOW`.

## Why the recipe flags are what they are

- `--request-timeout 7200`: the 1M cold prefill of the NIAH 100% fill takes ~1,780 s
  measured (v0.1.2, MixedK, NIAH 1,048,512 tokens, TTFT 1,781.3 s); the default 1,800 s
  deadline would end it (the v0.1.0 1M failure mode). With the recipe's speed floor the
  gate sizes its own per-request timeout from the floor; `--request-timeout 7200`
  covers the slowest allowed server.
- `--slots 16`: the qualified profile; the family default is 8.
- `--prefix-cache on`: pinned contract (the engine default; the v0.1.x campaign and
  the release requirements require prefix cache on). No `--prefix-cache-mib`: the
  MixedK 1M boot at the qualified profile succeeded with the Auto cache (leg 0/1);
  re-verify on the release binary (the v0.1.1 single-node 1M boot refused the bf16
  pack's flags unless `--prefix-cache-mib 6600` was given; the MixedK boot did not
  need it).
- `--spec auto`: the in-pack MTP head; speculation is always on (CONTRIBUTING 6.1).
- `--queue-timeout 6900`: Nemotron serves ONE shared KV pool sized at the window
  (plan `kv_tokens`), and admission reserves prompt + max_tokens against it (`plan.rs`
  `Admission::with_kv_tokens`; `gb_sched::Admission` `can_admit`/`admit`). A request
  that does not fit waits in FIFO order (head-of-line, `runtime.rs admit`), and
  `expire_queue` fails it after `--queue-timeout` counted from arrival (503 +
  `rexl3_queue_timeouts_total`). The pinned value is derived from Nemotron's worst-case
  service time: cold full-window prefill (measured 1,781 s at 1M) + the contract's
  largest output (8,192 tokens at the 42 tok/s decode floor ≈ 195 s) ≈ 1,976 s per
  request, ×3 requests ahead ≈ 5,928 s, kept 300 s under the explicit `--request-timeout
  7200` (see the recipe's `switch_justifications`). Without the pin, the server raises
  the default 300 s queue timeout to 14,400 s when `--max-model-len > 524288`
  (`main.rs:773`) — the pin documents the real requirement instead of that hidden raise.
  At the 262,144 alternative window the same rule gives ≈ 426 s per request and 1,500 s
  would apply.

## Vision limits (serving contract)

- **Images**: at most 3,328 tokens (`--max-image-tokens` capped; a 6000 x 4000 image
  becomes 3,290 tokens). Generated solid-colour images are answered correctly; one
  image+text request yields a correct tool call.
- **Video**: frames only — mp4/webm/gif are refused by name. At most 32 frames and
  4,096 tokens.
- **Audio**: refused by name (the pack has no sound tower).
- The vision tower is built in the engine (RADIO tower + pixel-shuffle projector; GPU
  kernels), so vision is on and tested by default; there is no vision exception.

## Known issues for v0.1.0

1. **The 24 h soak is NOT run for v0.1.0.** A maintainer decided to waive it for this
   release (the soak was never started), pending the other maintainer's confirmation.
   The stability evidence is the sustained-load legs of the baseline/qualifier runs plus
   the campaign evidence; a dedicated soak is not part of the v0.1.0 receipts unless the
   maintainers ask for it.
2. **Vision limits**: above (images ≤ 3,328 tokens; video frames only, ≤ 32 frames /
   4,096 tokens; audio refused). The refusal above 3,328 image tokens was not exercised
   in the v0.1.0 gate (that check is partial) — a GPU item.
3. **1M single-node memory**: the MixedK pack measured 27 GiB min MemAvailable at the
   1M single-node profile (above the 20 GiB rule), but the bf16/4-bit-head packs
   measured 18 GiB (below). The recipe pins the MixedK pack; re-measure MemAvailable at
   ready on the release binary and re-verify the 20 GiB rule (≥ 20 GiB).
4. **Prefix-cache cut policy**: the interactive cut policy ships with this runtime
   (engine fix, `engine.rs` `nemotron_prefix_config`): cuts that serve a second
   question about a long document and a short tool loop, with the environment's
   overrides still honored (`REXL3_PREFIX_TAIL`, `REXL3_PREFIX_MIN_CUT`, gap/gain).
   All v0.1.x campaign evidence used the previous default policy; the interactive
   policy must be re-proven at the recipe (the consolidated GB10 runs use it).
5. **Speculation under concurrent slots**: drafts are batched across streams (no
   Flash-Next-style single-stream gate), but the `drafted`/`accepted` counters above 0
   under concurrent slots still need proof on the release binary (a GPU item).
6. **The recipe `speed_floors`** are set from the v0.1.2 MixedK measurements (80% of
   measured, rounded down — the same margin convention as the Qwen/GLM recipes). The
   50% floor at 1M is derived from an interpolated rate (no direct 524K measurement
   exists); confirm on the GB10 baseline run.
7. **The quality lane and the release requirements** (`evals/releases/nemotron35-super-vl-sage-3.85-v2/`,
   `evals/q200_nemotron/`) are prepared and CPU-tested; no GB10 run of the quality lane
   exists yet.

## Open gaps at the configuration freeze (v0.1.0 scope)

1. **Consolidated GB10 run on the exact release head**: `scripts/gates/pre_pr.sh`
   once (build gate + baseline gate, producing the GB10-GATE v1 receipt), then at
   least two more baseline RESULT.json runs (the performance receipt requires ≥ 3
   distinct self-booted runs), the release qualifier, the quality lane (≥ 3 runs), the
   lifecycle driver and the exactness producers.
2. **Speculation/prefix-cache proofs at the real contract**: spec counters written 0
   under concurrent slots; `cached_tokens > 0` on a shared-prefix follow-up at the 1M
   window with 16 slots; long 1M resume (cached == cold output).
3. **Vision on the release binary**: `two_vision`-style image + video smoke with no
   `--no-vision`; the refusal-above-3,328-image-tokens case; `VISION SERVE CHECK`.
4. **Final profile and locks: done.** `profile.json` is final (authored from
   `docs/examples/cluster.nemotron35-super-vl-sage-3.85-single.json`), `model.lock.json`
   pins revision `39638152…` with a per-file SHA-256/size manifest and the fingerprint,
   and `dependencies.lock.json` and `requirements-runtime.lock` are filled; `handoff.json`
   is `final`, so strict generation (`build_runtime_repo.py --strict`) emits this entry.
5. **Nine gates on the freeze build**: doctor, lifecycle, capability, vision, context,
   exactness, prefix_cache, performance, quality (the 24 h soak is waived for v0.1.0;
   known issue 1).
6. **Seal and prod PRs**: `tools/approve_release.sh` seal on the GB10, then the prod
   `model/<model>-approval` and `model/<model>` branches per the promotion chain.
