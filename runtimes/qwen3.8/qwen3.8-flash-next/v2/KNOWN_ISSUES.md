# Qwen3.8-Flash-Next EXL3 3.05 bpw: frozen configuration, not qualification

One profile, `qwen38-flash-next-single-262k` (the default): one GB10, 262,144-token context,
2 slots, MTP speculative decoding (`--spec mtp`), prefix cache on (`--prefix-cache-mib 22528`),
vision off under a temporary, documented exception (see "Vision" below). Capabilities: text,
SSE, reasoning, tools, prefix cache. Images, audio, video and constrained JSON response formats
(`json_object`, `json_schema`) are refused by the server.

State: `final` (`handoff.json`). `profile.json`, `model.lock.json`, `dependencies.lock.json`
and `requirements-runtime.lock` are the final, sealed files of this entry. Configuration freeze
does not imply throughput, long-context or quality qualification on the GB10:
`qualification.state` is `unverified`, no measurement is transferred from earlier runs, and the
gate results for the release binary travel with the release approval, not in this file.

The entry is `runtimes/qwen3.8/qwen3.8-flash-next/` (family/model layout, `MODEL.json`
in `rexl3.prod-model.v1`). The runtime id `qwen38-flash-next-exl3-3.05` keeps the pack
identity (the naming regex `^[a-z0-9]+([.-][a-z0-9]+)*$`, pack details carried by
`MODEL.json`, never by folder names).

## Serving contract proven on a GB10 (PR #107)

The serving contract was proven on one NVIDIA GB10 (DGX Spark, 121 GiB unified) by PR
#107 `FEAT: Accelerated Qwen Flash Next prefill`:

```
rexl3-serve --model PACK --slots 2 --max-model-len 262144 --spec mtp \
  --prefix-cache on --prefix-cache-mib 22528 --served-model-name default --no-vision
```

with

```
REXL3_NGRAM_WARM=0
```

The grouped MoE prefill and the compensated-fp16 HCG pre-mix are the engine defaults for the family
(`REXL3_FLASH_NEXT_GROUPED_MOE_PREFILL` on when unset; explicit `0` is a documented debug opt-out),
so the profile sets neither prefill switch. The HCG pre-mix is the **declared prefill contract**: it
is unconditional and has no switch (see the prefill contract section below).

Pinned explicitly: model, served name (`default`), host/port, `--max-model-len`
262144, `--slots 2`, `--spec mtp`, `--prefix-cache on` (and `--prefix-cache-mib
22528`), and `--queue-timeout 1500` (a non-default pin: the 300 s default fails
requests queued behind a full-window request — see the shared-pool section below).
`--slots 2` and `--max-model-len 262144` match the engine defaults for the
family and are kept because they are serving-contract options. `REXL3_MTP_K` (default
6) and `--window-rows` (default) are at their defaults, so they are dropped.
Stateless MTP prefill and parallel planning are unconditional on main, so they are not
set.

## Why the runtime variables are set

- `REXL3_NGRAM_WARM=0`: the full-length 262,144-token prefix entry and the warm
  30.4 GiB n-gram table cannot both fit in 121 GiB. The 5-bit n-gram table is
  30.40 GiB (measured); warming it competes with the KV pool and the 22,528 MiB prefix
  cache the contract needs at the full window. `=0` keeps the table cold (mmap pages
  are reclaimable file cache, not a second anonymous allocation) and leaves the memory
  for the full-length prefix entry. This is the documented choice for a recipe that
  reserves memory for full-length cache entries (the same choice is documented for the
  family in `docs/QWEN_FLASH_NEXT_SERVER_QUALIFICATION.md`).
- The grouped MoE prefill is the engine default (no env needed), and the HCG pre-mix is the
  declared prefill contract (unconditional, no switch). See the prefill contract section below for
  the measurements and the exactness statement.

## Shared KV pool, FIFO admission and the queue timeout

The Flash-Next KV pool is **one shared budget capped at the window** (262,144 tokens;
`crates/model/src/families/qwen_flash_next/engine.rs:227`). A request reserves prompt plus
max output when it is admitted, and the server admits requests in strict FIFO order
(`crates/server/src/runtime.rs`, `admit` around line 1040): the front request is admitted
while a slot is free and the pool fits prompt plus max_new; a request that does not fit
holds the line (later, smaller requests never overtake it). When the server is busy the
front request waits; when idle and still no room, it fails. `--queue-depth` stays at the
default 64: past it the next request gets 429 (backpressure), and because admission is
head-of-line, a deeper queue only delays failures — it does not increase throughput.

`--queue-timeout` counts from arrival (`Queued::since`, `expire_queue`), and a request
queued behind a full-window request can wait for the whole front request's service. With
the 300 s default, a request queued behind one cold ~260K prefill (~245–276 s on the
GB10) plus decode would be failed with `the request waited 300s for a free slot` (503,
`rexl3_queue_timeouts_total`) — the bug this recipe fixes. The recipe pins
`--queue-timeout 1500`:

- Worst-case full-window service: cold ~260K prefill ~276 s + decode of the profile's
  largest output (8,192 tokens) at the 55 tok/s decode floor (~149 s) ≈ 425 s.
- 2–3 full-window requests ahead ≈ 850–1,275 s; 1,500 s covers 3 with ~18% headroom.
- 1,500 s stays 300 s under the 1,800 s `--request-timeout` (raised to 14,400 s only
  when `--max-model-len` > 524,288, which does not apply here), so a queued request is
  failed by the queue timeout before its end-to-end deadline — the documented
  interaction, and the queue-timeout 503/counter stays the observable failure mode.

The stability campaign's QUEUE stage exercises exactly this: while a full ~260K request
runs, a second request that cannot fit must wait (no 429/503/queue timeout), then
complete byte-identical to its serial reference, with the wait time recorded.

## Prefix pool capacity and eviction

The 22,528 MiB prefix pool is sized by the windowed planner (dev main #140,
`CacheBudget::fit_pages_first`): at the recipe's pin and window the plan is **86
entries / 1,910 pages / 488,960 tokens of reach (~21.99 GiB)** — planned (CPU),
recomputed on the merged code with the reviewer's rule re-implementation, to be
confirmed in the GB10 receipt run on this head. The previous (pre-#140) plan at
the same pin was 135 entries / 1,080 pages / 276,480 tokens (~21.97 GiB):
the cache memory envelope is essentially unchanged, but the shape changes
materially — far fewer snapshot slots and ~1.77× the reach.

One full-window window-store request is about 260K prompt tokens (measured
260,021–260,085 in the GB10 stability campaign), so the pool holds about 1.88
full-window entries. **Two full ~260K documents still do not fit** (about 520K >
488,960), so a second full-length insert still evicts the first (run 2/3 of the
GB10 stability campaign: `rexl3_prefix_cache_evicted_for_pages_total=3` at the
queue-front insert) and follow-ups on the evicted document run as a cold prefill
(about 4–5 min at 260K: 239.7–245.3 s TTFT alone, ~1,060–1,085 tok/s; up to
~339 s under heavy concurrent prefills). Documents whose total stays under ~489K
tokens can now coexist in the pool. Output is byte-identical (not corrupt); only
latency is affected.

## Pack identity and validation

- Pack: `turboderp/Qwen3.8-Flash-Next-exl3` at revision `3.05bpw_h5_ng5` (commit
  `69e33439ae950f17bcbe95c98f117d80f759ab6d`, resolved from the Hub). Seven weight
  shards + `model.safetensors.index.json` + the MTP mixer sidecar and the n-gram
  sidecar. `architectures` = `Qwen4ExpForConditionalGeneration`, `model_type` =
  `qwen4_exp` (verified via a public Hub resolve of the pack's `config.json`).
- Fingerprint `sha256:9e31f4ee36ba86dbe5eb5b263492216e6231e1808a220de2d42483e349b30824`
  is registered in `crates/model/src/fingerprint.rs` `VALIDATED_TEXT` (`QFN305` /
  `QFN305_FP`) by this branch's first commit, with a regression test. With that entry,
  `serve::check_packs` accepts the pack without `--allow-unvalidated-pack`; the flag is
  never passed and stays forbidden (it would disable the identity gate).
- The verified pack-validation record is `docs/validation/flash-next-3.05bpw.md`: the
  four-cell Spark rerun passed (1K mean cosine 0.995090, 8K mean cosine 0.998024; both
  above the oracle gates). The older 1K FAIL branch record is `unbound_unreproduced`
  and excluded from qualification.

## The prefill contract

The declared prefill contract for Flash-Next is the **compensated-fp16 tensor-core hyper-connection
pre-mix** (`hcg`): the same math as the fp32 `hcx::pre` reference, run as cuBLAS tensor-core GEMMs
whose fp16 operands are an **fp16 hi+lo split** (Dekker pairing) of the fp32 values, accumulating in
fp32. It applies to **prefill rows only** (tiled prefill chunks); decode and GEMV prefill always keep
the fp32 reference. It is **unconditional**: there is no switch (the environment reader was removed),
and the boot log names the active contract on the family's plan line
(`prefill_contract=hcg-compensated-fp16`), which the gate recipe asserts.

### Accuracy vs the fp32 reference and fp64 truth

Measured against a float64 ground truth on real 32K activations (shipped bf16 output resolution):

- fp32 reference path: **>= 99.996 %** bf16-equal per call, worst **<= 1** bf16 ulp;
- compensated HCG: **>= 99.93 %** bf16-equal per call, worst **<= 2** bf16 ulps;
- both are the same accuracy class; the compensation residual is the ~2^-22 split, not the ~2^-11
  of a single fp16 rounding (the original fp16-only pre-mix measured ~1e-3 relative vs the reference).

Row behavior:

- **single-row calls are bit-identical** to the fp32 reference;
- multi-row: mean_abs ~5–7e-8, rmse ~1.3–1.7e-5 vs the reference (the gate's 0.025 tolerance),
  with bit-exact determinism and chunk independence (tiled chunks equal the whole prefill).

### One deterministic near-tie behavior difference

The prompt `add 2 and 3 + call lookup on step 5` with parallel tools returns **two tool calls**
under the contract vs **one** under the fp32 reference path (deterministic, 3/3 runs each). The
fp64 study finds both arms numerically equivalent at bf16 resolution; the flip is a near-tie logit
margin (~5 bf16 units), not an accuracy regression. The pre-mix arithmetic order can differ by up to
2 ulp and can flip a decision only where the decision margin is smaller than the logit delta (rare).

### The gate's stage-3 probe sits on a measured near-tie

Measured at the declared contract (`prefill_contract=hcg-compensated-fp16`, temperature 0, on the
gate's own stage-3 parallel tool-call request, which asks for two add calls): the decision to emit
the second tool call or stop is made at generated position 34, where main's top-2 logit margin is
**0.375 bf16 logit units** (main #157, continuation wins). For comparison, the pre-#157 fp32
pre-mix path has a 2.125 margin there, and an experimental batched variant of the same compensated
pre-mix, more accurate against fp64, flips the decision to stop, returning one tool call instead of
two (stage 3 fails).

Inter-arm full-vocab |logit delta| at that token is 2.95, about 8x main's margin. Under the MTP
spec, main itself shows exact 0.000 ties in the same generation (draft rows 92 and 99). All arms
share the same runner-up token, so this is the model's own near-tie, not an artifact of one
arithmetic.

Consequence: any numeric change in Flash-Next prefill, even a more accurate one, can flip the
stage-3 result. A stage-3 flip (one tool call returned instead of two) after such a change should
be read as this near-tie first, not automatically as an accuracy regression. The disclosure above
(the `add 2 and 3 + call lookup on step 5` prompt flipping between one and two calls) is the same
phenomenon on a different prompt; this gate probe is it on the gate's own request.

### Verification

- `crates/cuda/tests/fn_mhc.rs` (GPU): hcg vs hcx within 2e-4 relative on non-square shapes at the
  family's dims; bit-exact determinism and chunk independence; the poisoned-neighbor test proves no
  read escapes a declared operand (an out-of-bounds read would feed NaN into the GEMMs and flip bits).
- Model unit tests pin that the path is unconditional (no env reader, no registry entry for
  `REXL3_FLASH_NEXT_HCG_PREFILL`).
- The gate recipe asserts `prefill_contract=hcg-compensated-fp16` in the boot log and runs the
  stage-3 parallel tool-call probe (passes 3/3 with the contract on).
- GB10 prefill speed with the contract on: 1,137 tok/s at the 131K fill and 1,086 tok/s at the
  ~260K fill (the gate's own method), ~1.88x the grouped-only configuration at 32K; decode unchanged.

There is no opt-out: the contract is always on for qualifying shapes (H % 4 == 0, hc >= 1, rank >= 1
— true of this family).

## Vision (temporary documented exception)

The pack ships the vision tower's tensors, but REXL3 does not build the Flash-Next
vision tower (`unbuilt_vision()`, `crates/server/src/main.rs`). `--no-vision` is used
under a temporary, documented vision exception granted by a maintainer on 2026-10-23,
expiring 2027-01-31, recorded in `scripts/gates/recipes/qwen38-flash-next-exl3-3.05.json`
`capability_exceptions.vision`. To ship past the expiry, either the tower is built and
vision is on and tested, or a vision-on boot at the same contract is refused for memory
(`does-not-fit` record).

## Open gaps at the configuration freeze (v0.1.0 scope)

1. **Consolidated GB10 run.** The `speed_floors` in the gate recipe are set from
   PR #107's GB10 measurements (80% of measured, rounded down — the same margin
   convention as the Qwen3.8-27B and GLM recipes) and are confirmed by the
   consolidated GB10 run on the exact release head: `scripts/gates/pre_pr.sh`
   once (builds and produces the GB10-GATE v1 receipt), then the 262K stability
   campaign. The recipe documents the derivation and the markers to confirm.
2. **Concurrent-slot speculation proof.** MTP drafting returns nothing while more than
   one request is decoding (`backend.rs`); the contract has 2 slots. The speculative
   decoding counters (`/metrics` drafted/accepted) above 0 under concurrent slots and
   the prefix-cache hit proof at the recipe's real context and slots are still needed
   on the GB10.
3. **Final profile and locks: done.** `profile.json` is final, `model.lock.json` pins the
   exact revision with a per-file SHA-256/size manifest and the fingerprint, and
   `dependencies.lock.json` and `requirements-runtime.lock` are filled; `handoff.json` is
   `final`, so strict generation (`build_runtime_repo.py --strict`) emits this entry.
4. **Nine gates on the freeze build.** doctor, lifecycle, capability, vision, context,
   exactness, prefix_cache, performance, quality on the release binary (the freeze
   commit shared by the v0.1.0 models).
5. **Seal and prod PRs.** `tools/approve_release.sh` seal on the GB10, then the prod
   `model/<model>-approval` and `model/<model>` branches per the promotion chain.
