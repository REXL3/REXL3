# DSV4.1 SAGE 1.55 bpw — frozen configuration, not qualification

The two profiles are `dsv41-single-262k` (**single-node, the release default** on
the 1.55bpw pack) and `dsv41-tp2-1m` (two-node TP=2, the second shipped profile
on the REXL3 1.55bpw pack). The single-node profile serves a 262,144-token context;
the two-node profile serves the model's full 1,048,576-token window. Both use q4, native `spec=auto`, text, SSE, reasoning, tools and prefix
cache, with lossless `REXL3_DSV41_MOE_TOPP=1.0` and the
approved FAST_MOE/SPEC_LEARN switches. Vision (the owner's decision of
2026-11-01, "vision off for deepseek"): **both** profiles ship **`--no-vision`**.
The tower is implemented and image input is encoded exactly as the upstream
reference (feature cosine 0.99996), but colour/shape accuracy on this quantized
pack is unreliable (3 of 10 solid colours correct on the single-node probe, 0 of
6 on the two-node probe), so images are refused under a temporary exception in
each profile's recipe (`capability_exceptions.vision`, granted by a maintainer
on 2026-11-01, expires 2026-12-31), and the gate reports the vision stage as
EXCEPTION. Vision returns when the pack passes the colour probe. (Until then the
single-node profile had shipped vision on under the 2026-10-28 grant.) Audio, video and constrained JSON response
formats remain unsupported. Slots remain 1. Configuration freeze does not imply
full-window, throughput, multimodal or quality qualification.

## Single-node profile is the release default (1.55bpw pack)

The single-node profile (`dsv41-single-262k`) is the release default and serves
the 1.55bpw pack at 262,144 context with `--no-vision` (see above; the tower
fits, the decision is about answer quality). Measured on one GB10, with the
tower loaded
(report `reports/dsv155-fit-test.md`, 2026-10-29, dev-main binary, exact
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

## Two-node TP=2 profile is the second shipped profile (REXL3 1.55bpw pack)

The two-node TP=2 profile (`dsv41-tp2-1m`) ships on the REXL3 1.55bpw pack
(`vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw` at revision
`cba79c67582950ea44a4a449b3bb85b800748a35`: 320.3 GB, 17 shards, EXL3 with a
K chosen per tensor; routed experts use K 1 to 4, the trunk linears K 5 and 6).
It replaced the 1.62bpw pack on 2026-10-31 (maintainers' decision: it should
perform better). It splits the expert slab and KV across ranks (≈ half per
rank) and needs a RoCE link between the two GB10s. It is the same pack as the
single-node default: the entry lock pins it once, as its `target` role, and
both choices serve it. The TP2 choice keeps its own runtime_id,
`dsv41-flash-1017-sage-rexl3-1.55`, because its recipe pins its own window
(1,048,576), so the TP2 gate receipt pairs the recipe
`scripts/gates/recipes/dsv41-flash-1017-sage-rexl3-1.55.json` with the entry
profile via `--choice dsv41-tp2-1m` and `--base-url`/`--server-log`. The engine
lists the pack as validated (`crates/model/src/fingerprint.rs`, fingerprint
`sha256:c34045423c126e40d051943ff46bd57a67c07350d8b1743ed5185415db61351c`).
Vision is off on this profile (`--no-vision`, see above), so the tower is not
loaded and the gate's vision stage reports EXCEPTION.

Not yet done on this pack:

- Measurements before the change's receipt: speed, memory, quality and the 1M
  window were unmeasured on this pack at two nodes. The 1.62bpw pack's last 262K receipt
  (PR #144, head `2d23a4903`: 1,693 / 1,292 tok/s prefill, 28.3 tok/s decode)
  and the 1M runs below are history, not evidence for this pack.

## The two-node profile serves the full 1,048,576-token window

`dsv41-tp2-1m` serves `--max-model-len 1048576`, the model's advertised window.
Measured on the **1.62bpw pack** this profile served before (not on the REXL3
1.55bpw pack), with the baseline gate on two GB10s at the same binary (REXL3-dev
main `6fae166e`), one run per window, run directly and not through `pre_pr.sh`,
so these are measurements and not a receipt:

| | 262,144 | 1,048,576 |
|---|---|---|
| gate result | PASS (315 s) | PASS (2,463 s) |
| decode | 28.9 tok/s | 29.0 tok/s |
| prefill at 131K tokens | 1,817 tok/s | about 1,870 tok/s (NIAH on the 1M server) |
| prefill at half the window | 1,817 tok/s (131,092 tokens) | 935 tok/s (524,292 tokens, 561 s) |
| prefill at the full window | 1,372 tok/s (261,976 tokens) | 567 tok/s (1,048,396 tokens, 1,848 s) |
| lowest MemAvailable | 47.8 GiB (leader) | 44.3 GiB (leader), 45.8 GiB (follower) |
| speculative accept rate | 0.60 | 0.60 |

- The speed floors (decode 18, prefill 700 at half the window, 400 at the full
  window) held at 1M with margin on the 1.62bpw pack and are unchanged for the
  REXL3 1.55bpw pack.
- A prompt that fills the window takes about 31 minutes to its first token. A
  shorter prompt costs nothing extra.
- When `--max-model-len` is above 524,288 the server raises its request,
  prefill, decision and queue timeouts and the TP peer timeout to 14,400 s
  (`crates/server/src/main.rs`), so the profile's `--tp-peer-timeout 60` is
  raised to 14,400 s at this window: a stalled peer is declared lost only after
  that long (board item T2G6).
- Not measured on this profile: anything on the REXL3 1.55bpw pack (its 1M
  receipt and its Q200 quality baseline are pending), the 256K to 512K context
  ladder through the qualification tool, the Q200 quality lane at 1M, and a soak. The follower's
  memory was read once after the run (the gate samples the leader only).

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

## TP=2 adapter

An explicit site `nodes` mapping must contain exactly `leader` and `follower`.
Each node supplies literal IPv4 `address`, `fabric_address`, `user`, `model_dir`,
and `binary_dir`. The HTTP port, `tp_port`, `fabric_interface` and a non-empty `hca`
rail list must also be stated; the sealed cluster file's example network is never
used as a default (omitting one is refused, naming the field). Serving
flags/environment cannot be overridden. The leader binds
to its management address (restrict network access externally).

The adapter delegates validate/check/up/status/down to the pinned native cluster
binary. All ranks must pass full payload, binary, native-version and dependency
identity checks before up. A SHA-verified transmitted bundled helper uses isolated
Python, strict existing SSH host trust, and no forwarded agent/environment secrets.
Default SSH keys must already be provisioned; the adapter never edits trust files.
Native scripts use a clean environment and unique per-attempt state directories.
Readiness requires exact rank argv/process/socket ownership plus API model/limits
and an actual smoke. A health 200 alone is not readiness. Partial starts delegate
to native down; an unsuccessful cleanup leaves the active marker for intervention.

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
context (262K single-node, 1M two-node) and
multimodal/quality/throughput qualification belong to the hardware lane.
No measurements from earlier profiles are transferred to this frozen configuration.
The single-node GB10 receipt on the swap head is pending (the 1.54 receipt on
the previous head passed at 1,458/1,147/24.2; the fit test is
reports/dsv155-fit-test.md). The two-node GB10 receipt on the REXL3 1.55bpw
pack is pending (the 1.62bpw pack's passed on PR #144's head `2d23a4903`:
1,693/1,292/28.3).
