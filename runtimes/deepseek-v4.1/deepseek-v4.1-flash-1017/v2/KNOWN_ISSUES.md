# DSV4.1 SAGE 1.55 bpw (single-node) — frozen configuration, not qualification

This entry is the single-node DeepSeek-V4.1-Flash-1017 runtime,
`dsv41-single-262k` (the release default on the 1.55bpw pack). The two-node
TP=2 profile `dsv41-tp2-1m` ships as its own runtime entry
`runtimes/deepseek-v4.1/deepseek-v4.1-flash-1017-tp2/` (runtime
`dsv41-flash-1017-sage-rexl3-1.55`), so the two entries are independent runtime
units and release/receipt separately. This file covers the single-node entry
only.

The single-node profile serves a 262,144-token context with q4, native
`spec=auto`, text, SSE, reasoning, tools and prefix cache, with lossless
`REXL3_DSV41_MOE_TOPP=1.0` and the approved FAST_MOE/SPEC_LEARN switches.
Vision (the owner's decision of 2026-11-01, "vision off for deepseek"): this
profile ships `--no-vision`. The tower is implemented and image input is encoded
exactly as the upstream reference (feature cosine 0.99996), but colour/shape
accuracy on this quantized pack is unreliable (3 of 10 solid colours correct on
the single-node probe), so images are refused under a temporary exception in the
recipe (`capability_exceptions.vision`, granted by a maintainer on 2026-11-01,
expires 2026-12-31), and the gate reports the vision stage as EXCEPTION. Vision
returns when the pack passes the colour probe. Audio, video and constrained JSON
response formats remain unsupported. Slots remain 1. Configuration freeze does
not imply full-window, throughput, multimodal or quality qualification.

## Pack revision replacement

The lock pins revision `c40cf25cc3148bae95ac27483505c883840bf8a7` because the
earlier `cba79c67582950ea44a4a449b3bb85b800748a35` revision is no longer
available from the Hub. All 24 locked files were independently verified to have
identical sizes and SHA-256 digests at the replacement revision, so the payload
manifest and engine fingerprint are unchanged. Existing external `model_dir`
copies remain usable after the same full payload verification. Release-evidence
carry-forward accepts the earlier revision only through the reviewed, exact
same-files revision alias; no other pack-record difference is excused.

## DeepSeek-V4.1 KV formats and `int8` fall-through

DeepSeek-V4.1 implements `q4` and `fp16` KV caches only. Both shipped profiles
select `q4`, which is the shipped, qualified cache; `fp16` is the other
implemented format. The `int8` choice in the shared `--kv-dtype` help text
applies to Nemotron's `REXL3_KV_DTYPE` path, not to DeepSeek-V4.1.

On DeepSeek-V4.1, `--kv-dtype int8` currently fails the family-specific parse
without surfacing an error, leaves `REXL3_DSV41_KV` unset, and therefore runs
the engine's default `q4` cache. It does not select an `int8` cache. The shared
build-info path still records `REXL3_KV_DTYPE=int8`, so `system_fingerprint`
can contain `kvi8q4`: `kvi8` records the shared setting while `q4` records the
cache that DeepSeek-V4.1 actually uses.

This fall-through explains @Weschera's single-node GB10 report in
REXL3/REXL3#111. Custom needle recall at 65,283 prompt tokens, Spark-Bench v7
`long_context` quality, prefill on a 33,015-token prompt across `n=3` runs per
flag, and decode across three short workloads matched between `--kv-dtype q4`
and `--kv-dtype int8` on recall, quality, memory and speed because both flags
executed the `q4` cache. Those measurements are not supporting evidence for an
`int8` implementation or its qualification. The Spark-Bench v7 result is not a
Q200 result.

## Single-node profile is the release default (1.55bpw pack)

The single-node profile (`dsv41-single-262k`) is the release default and serves
the 1.55bpw pack at 262,144 context with `--no-vision` (see above; the tower
fits, the decision is about answer quality). Measured on one GB10, with the
tower loaded
(report `reports/dsv155-fit-test.md`, 2026-10-29, the binary of that run, exact
profile config):

- Full expert residency: 15,360 / 15,360 routed experts resident, 0 streamed
  (slab 98.483 GiB planned, 102.501 GiB measured).
- Post-ready reserve 1.2 GiB held; vision tower loaded (0.904 GiB weights,
  0.437 GiB scratch, 156 MiB features).
- MemAvailable at ready 2.948 GiB; min over the whole run 2.605 GiB
  (gate floor is 1 GiB; the 1.62bpw pack measured 0.32–0.55 GiB at ready and
  could not pass).
- 262K NIAH correct at ~1,045 tok/s prefill; decode 20.9 tok/s with spec on;
  prefix-cache hit 261,807 / 261,855 tokens cached (99.98%).
- Vision answers are unreliable (3 of 10 colours correct) on both packs — the
  vision weights are byte-identical between the 1.55 and 1.62 packs; hence
  `--no-vision` under the 2026-11-01 exception.

The gate receipt numbers on the swap head come from the PR's receipt; a
docs-only follow-up will add them (the 1.54 receipt on the previous head
measured 1,458 / 1,147 tok/s prefill and 24.2 tok/s decode).

The two-node TP=2 profile is not in this entry: it ships as its own runtime
entry `runtimes/deepseek-v4.1/deepseek-v4.1-flash-1017-tp2/`
(`dsv41-tp2-1m`, runtime `dsv41-flash-1017-sage-rexl3-1.55`, gate recipe
`scripts/gates/recipes/dsv41-flash-1017-sage-rexl3-1.55.json`). It serves the
same REXL3 1.55bpw pack at the full 1,048,576-token window.

## Prefix-cache snapshot floor and post-ready reserve

Commit c7566b972 (cherry-picked from 1fee310ff) decouples the snapshot-capture
floor (0.25 GiB default) from the boot floor and adds a 1.2 GiB post-ready
reserve to the full-residency election, so captures are not silently skipped
when boot headroom is tight (prefix-cache stage 5/6 can fill while the reserve
stays free). The capture floor no longer reads `REXL3_DSV41_FLOOR_GIB`.

## Existing model copies

`--site` can select `model_dir` and `binary_dir` for single-node serving.
The model tree is read-only: no moves, downloads, chmod, cache or marker writes.
Every locked file is size/full-SHA verified on every preparation. Only the exact
irrelevant root documents `README.md`, `LICENSE`, `.gitattributes` may be extra;
they too must be regular, non-symlink, single-link files. The one tolerated
directory is the Hugging Face download bookkeeping `.cache/huggingface/` left by
`hf download --local-dir` (also on copies of the managed directory): it may hold
only directories and regular single-link files, is never hashed or read by the
server, and anything else under `.cache` is refused. Extra tensors, sidecars,
indexes and other directories fail closed, and the refusal names its code
(for example `EXISTING_MODEL_IDENTITY_FAILED: UNLISTED_MODEL_DIRECTORY`). All ancestors are no-follow traversed.
The operator must keep admitted trees unchanged while serving: a preflight hash
is a point-in-time observation, not isolation against a hostile same-UID writer.

## Reasoning-effort names and sampling defaults

**Reasoning effort.** The engine's native renderer (`crates/server/src/formats/deepseek_v41.rs`) accepts
`reasoning_effort` as a top-level field or as `chat_template_kwargs.reasoning_effort`, with one of these names:
`none` and `off` turn thinking off; `minimal`, `low`, `medium`, `high`, `max` and `xhigh` turn it on. Any other
name is refused with HTTP 400. Each name maps to the effort number the `Reasoning Effort: N` prefix renders:
`minimal` and `low` = 50, `medium` = 65, `high` = 75, `max` and `xhigh` = 100. The request validator refuses
non-name values, an integer included, with HTTP 400; the renderer's own mapping would clamp an integer 1–100 if
one reached it. Upstream's reference encoder (DeepSeek's `encoding/encoding.py`; upstream ships no Jinja
template) accepts only an integer 1–100 or `low`/`high`/`max` (50/75/100). The extra names are OpenAI-style
aliases of this runtime's control table, and `medium` = 65 is this runtime's own choice; the pack's
`chat_template.jinja` (for tools outside REXL3) follows upstream and raises on the extra names, and it does not
render tools. Thinking is off by default. With thinking on and no effort given, the effort is 75 (upstream's
`high`). The effort prefix renders only in thinking mode, at the start of the conversation.

**Sampling.** A request that leaves out `temperature`, `top_p` or `top_k` gets temperature 1.0 and top_p 0.95 from
the pack's `generation_config.json` (the file the model lock pins), and top_k 20, the server's fallback (`GenDefaults`
in `crates/server/src/api.rs`). Neither the pack nor upstream sets top_k. Where the pack and upstream differ (upstream
recommends top_p 1.0), the pack's file wins: the owner's rule of 2026-11-02 takes a pack's own config first and uses
the base model only where the pack sets nothing. Requests can set these fields explicitly, which overrides the defaults.

**Greedy decoding can loop.** At temperature 0 the model sometimes repeats a short phrase in its reasoning until
`max_tokens`. Q200 text180 was run on the two-node profile at effort `low` on 2026-11-02:
- **Greedy:** 25 of 180 prompts looped, with an 8-word run repeated 50 to 1,307 times. Most were instruction-following
  prompts; 22 of the 40 hit the 8,192-token cap.
- **The defaults above** (seed 1234): 8 of 180 looped, all instruction-following.

This is the model's behaviour, not this pack or engine: the 1.62bpw pack loops on the same prompts as often, and
turning off speculative decoding or using an fp16 KV cache does not stop it. For long reasoning, use the defaults
rather than temperature 0, and set `max_tokens` to bound a reply.

## Unverified boundaries

CPU fixture tests are not GPU or native-Rust execution evidence. Native compilation,
real SSH/host keys, all-rank hashing of the model, dependency/native identity
on actual hosts, cleanup under actual link failure, clean-host rehearsal, full-window
context (262K single-node) and
multimodal/quality/throughput qualification belong to the hardware lane.
No measurements from earlier profiles are transferred to this frozen configuration.
The single-node GB10 receipt on the swap head is pending (the 1.54 receipt on
the previous head passed at 1,458/1,147/24.2; the fit test is
reports/dsv155-fit-test.md). The two-node GB10 receipt on the REXL3 1.55bpw
pack is pending (the 1.62bpw pack's passed at 1,693/1,292/28.3).
