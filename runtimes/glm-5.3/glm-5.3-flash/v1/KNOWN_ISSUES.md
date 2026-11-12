# GLM-5.3-Flash EXL3 2.25 bpw - frozen configuration, not qualification

The first GLM release (v0.1.0) ships two choices: the default,
`glm53-flash-single` (one GB10, 262,144 tokens), and `glm53-flash-tp2-1m`
(two GB10s, tensor parallel, 1,048,576 tokens; section "Two-node profile" below).
Both serve text, SSE, reasoning, tools and images, with
speculative decoding (`--spec auto`, the pack's own NextN layer) and the prefix
cache on.
Audio, video and constrained JSON response formats are unsupported.
Configuration freeze does not imply full-window, throughput, multimodal or
quality qualification: every number below is a dated reference under the
conditions stated, and none is a floor of this profile.

| | `glm53-flash-single` | `glm53-flash-tp2-1m` |
| --- | --- | --- |
| Window (`--max-model-len`) | 262,144 | 1,048,576 |
| Slots | 1 | 1 |
| KV pool | planned by the engine; equals the window at 262,144 | pinned at the window (`--kv-tokens 1048576`, required on a GLM group) |
| Speculation | `--spec auto` | `--spec auto` |
| Prefix cache | on, pinned at 3,584 MiB (`--prefix-cache-mib 3584`, 2 entries, 1,036 pages); reuses prompts up to 265,216 tokens, one long conversation at a time | on, pinned at 13,440 MiB (`--prefix-cache-mib 13440`: 5 entries, 4,106 pages a rank, reaching 1,051,136 tokens; rank 0 booted with that plan in a diagnostic run) |
| Memory floor | `REXL3_MEMORY_FLOOR_GIB=4` (engine default 8) | the same |
| Images | on (24-layer vision tower, up to 8,000 tokens an image) | on (rank 0 holds the tower) |
| Prefill contract | P2 (the engine default since 2026-10-19) | P2 |

## Why the window is 262,144 and not the advertised 1,048,576

The pack and the base model advertise 1,048,576 tokens (`max_position_embeddings`).
One GB10 cannot hold that window with the features the rules require, so the
profile ships at a window that boots with the cache on and a margin on the
memory the engine plans from.

The KV pool for 1,048,576 tokens needs about 26.5 GiB free after the weights
(the engine's own planner, CPU run on this pack's shard headers: 25 GiB free
gives 923,904 tokens, 20 GiB gives 512,768; the KV cache is 11,968 bytes a
token, 13,056 with the drafter). A GB10 leaves about 20 GiB after the 90.6 GiB of
weights plus 2.1 GiB for the tower and its scratch.

Measured on a GB10 node (20.0 GiB free after the weights, images on, NextN
layer, one slot, `--spec auto --prefix-cache on`):

| `--max-model-len` | Effective window / KV pool | Prefix cache (`prefix_cache_plan`) |
| --- | --- | --- |
| 1,048,576 | effective 594,944, below the configured window | `entries=0`: no spare memory |
| 524,288 | effective 512,768, below the configured window | `entries=0` |
| 458,752 | equals the configured window | 2 entries, 0.7 GiB spare |
| **393,216** | **equals the configured window** | **4 entries (0.65 GiB), 1.5 GiB spare** |
| 327,680 | equals the configured window | 6 entries |

The engine plans from the node's available memory, rounded down to a whole GiB, and
that figure moved by a GiB between boots of the same profile (19.0, 20.0 and 21.0
`plan_free_gib` were seen). Measured with the earlier shipped argv (window 393,216, cache
not pinned) and a memory ballast to mimic a tighter host:

| Planned free | Window / KV pool | Prefix cache |
| --- | --- | --- |
| 21 GiB | 393,216 | 7 entries |
| 20 GiB | 393,216 | 4 entries (0.65 GiB), 1.5 GiB spare |
| 19 GiB | 393,216 | `entries=0` (cache auto-disabled), 3 of 3 boots |

and with the cache pinned at 1,024 MiB (window 393,216):

| Planned free | Window / KV pool | Prefix cache |
| --- | --- | --- |
| 20-21 GiB | 393,216 | 6 entries |
| 19 GiB | 350,464 (below the configured window) | 6 entries |
| 18 GiB | 268,032 (below the configured window) | 6 entries |

So pinning keeps the cache but moves the one-GiB sensitivity to the window; it does not
remove it. A first release pinned the cache at 1,024 MiB (48 pages, 6 entries) and set the
window to 262,144. That held the window but reused prompts only up to 48 pages of 256 tokens
(about 12,000 tokens). The planner ties the cache's pages to its snapshot slots at 8 pages a
slot, and a prompt is stored only when all its pages fit, so the reach is pages x 256 tokens,
not the cache's size in bytes. The shipped profile therefore lowers the floor to 4 GiB
(`REXL3_MEMORY_FLOOR_GIB=4`, as the RED-SNOW profile does) and pins the cache at 3,584 MiB.
Measured at window 262,144 with that floor and a memory ballast to mimic a tighter host
(this table was taken with the planner before the window-sized prefix sizing; the entry counts
are from that planner, and the ballast rows have not been repeated with the current one):

| Pin | 20-21 GiB planned free | 18-19 GiB | 16 GiB |
| --- | --- | --- | --- |
| 3,328 MiB (20 entries, 3.25 GiB) | window held, spare 4.8 | window held at 19, spare 2.8 | 245,760 |
| **3,584 MiB (21 entries, 3.41 GiB)** | **window held, spare 3.6 at 20** | **window held at 18, spare 1.6** | 232,192 |
| 4,096 MiB (24 entries, 3.90 GiB) | window held, spare 3.2 at 20 | window held at 19, spare 2.2 | 192,256 |

3,584 MiB holds the full window one GiB below the lowest planned-free value seen on any
boot (19). Below 18 GiB the engine shrinks the window and the launcher's identity check
refuses that as `MODEL_LIMIT_MISMATCH`. The cost of the lower floor is less memory left to
the rest of the host: measure the lifecycle on the target node.

The prefix cache is sized from the window (`docs/PREFIX_CACHE_SIZING.md`): at the 3,584 MiB
pin the engine plans 2 snapshot slots and 1,036 pages, which reaches 265,216 tokens, 12 pages
over one full 262,144-token window. Measured on a GB10 with the shipped argv: a 249,512-token
prompt asked again is served 99.97% from the cache in 0.25 s (cold 313 s), the plan is the same
over three boots, and the lowest MemAvailable seen was 10.6 GiB.

**One long conversation at a time.** With one slot and the two-snapshot floor the cache holds a
single long conversation. A different 30K-token prompt evicts the full-window entry, and the next
send of the long prompt is prefilled cold again (313 s). A user who alternates between a very long
document and unrelated chats will see that cold prefill each time they return to the document;
short tool loops and one long session are served from the cache. The pin is not raised: another
+3.5 GiB would use all of the spare the profile has at 20 GiB planned free and would shrink the
window below 262,144 on a host with less free memory, and it is not measured to stop the eviction.
Boot takes about 16 s. The real boot lines at this window are
`rexl3 event=kv_plan free_gib=.. kv_pages=.. kv_tokens=262144 slots=1` and
`rexl3 event=prefix_cache_plan entries=2 pages=1036 gib=.. spare_gib=..`.

A configured window above the pool is not served as configured: the server
reports a smaller effective `max_model_len`, and the launcher's identity check
refuses that as `MODEL_LIMIT_MISMATCH`. Also, with `entries=0` the gate counts
the prefix cache as auto-disabled, which fails (rule 14). Without the tower the
pool is larger by only the features reserve (25 GiB free gives 964,096 tokens), so
dropping vision would not reach the window either.

Measured earlier on this pack, below that window: `docs/GLM53.md` "Long context",
`--slots 1`, tower and NextN resident, `--spec none`, P1 prefill: 262,144 KV tokens
planned, a 262,144-token request admitted, needle retrieval exact (16 of 16)
from 32,768 to 261,120 tokens. `--spec auto` at context and P2 at that length were
not measured. Nothing is claimed for a single-node window above 262,144.

## Two-node profile (`glm53-flash-tp2-1m`)

GLM two-node is 1,048,576 tokens only: `glm53-flash-tp2-1m` is the entry's only
two-node choice, and it ships in v0.1.0 beside the single-node default.

What it is: the single-node choice's serving contract on a two-rank
tensor-parallel group of two GB10s joined by a RoCE link, at the pack's whole
1,048,576-token window. It uses the same pack, `runtime_id` and gate recipe
(`scripts/gates/recipes/glm53-flash-2.25bpw-sm121.json`); the recipe pins no
window, so the gate takes 1,048,576 from this argv. Its argv is the single-node
argv with `--max-model-len 1048576` and `--prefix-cache-mib 13440`, plus
`--kv-tokens 1048576` and the group flags. A GLM group plans before it loads and
refuses a KV pool left to the planner by name
(`crates/model/src/families/glm/tp_boot.rs`, `check_tp_options`). The NCCL
variables and the attested cluster adapter are the DeepSeek two-node profile's.
The sealed cluster file `config/cluster.tp2-1m.json` (pinned by sha256) carries
example network values; the site must state its nodes, fabric interface, HCAs and
TP port. Rank 0 serves HTTP, holds the vision tower and encodes every image; the
follower holds no tower (the server gives a follower no vision slot whatever its
flags), so there is no `--no-vision`. Each rank loads its share of the routed
experts and keeps its own prefix-cache snapshots; rank 0 decides the hits and
forwards each snapshot, restore and release to the follower.

The cache pin gives each rank a cache that reaches a whole window: the engine's
planner, run on the CPU on this pack's headers for rank 0 at one slot, from
110 GiB free and without the vision tower the leader also holds (about 2.1 GiB
with its scratch), plans 5 entries and 4,106 pages (1,051,136 tokens of reach)
with 23.6 GiB spare; 13,056 MiB reaches only 1,037,056 tokens
(`crates/model/src/families/glm/tp_boot.rs`,
`the_2p25bpw_ranks_plan_a_whole_window_cache_at_1m`).

It needs an engine that serves the prefix cache on a GLM group, as this
release's engine does. An older binary either refuses the group at boot (it asks
for `--prefix-cache off`) or does not print the single node's boot lines on the
leader, which fails four of the recipe's boot markers.

**Diagnostic run, not a receipt** (2026-10-30, two GB10s, a development build
of this engine, the baseline gate in `--base-url` mode with the leader's log, and
temporary copies of this choice's profile and recipe made before they were
committed):

| What | Value |
| --- | --- |
| Gate stages | every stage PASS |
| Boot plan (rank 0) | `prefix_cache_plan entries=5 pages=4106`, as planned above |
| Decode | 57.6 tok/s |
| Prefill | 1,104 tok/s at the 50% NIAH stage (524,309 tokens), 962 at 100% (1,044,466 tokens); both NIAH PASS |
| Leader min MemAvailable | 32.2 GiB |
| Prefix cache | 6 of 6, greedy output identical to the cache off |
| Vision | 5 of 5 |

None of these is a floor; the recipe's floors are the single node's at 262,144.
Boot time at this window is not recorded. Gate results for the release binary
travel with the release approval, not in this file.

`dependencies.lock.json` is the DeepSeek-V4.1 two-node lock: the single-node lock
plus a strict `libnccl2` pin, because a pair without `libnccl.so.2` exits at
boot. A single node installs NCCL too and does not use it.

## Draft and licence

The pack has no external draft: speculation uses the NextN layer stored in the
pack. The DFlash2 draft for this model is CC-BY-NC-ND and is not part of this
profile or of the lock (`docs/DISTRIBUTION.md`); `model.lock.json` has the
`target` role only and the argv has no `--draft`.

## Sampling and reasoning (source order: the pack, then the base card)

- The pack's README states no sampling values. The pack's own
  `generation_config.json` (locked) sets `temperature` 1.0 and `top_p` 0.95,
  which the server reads as its defaults.
- The base card (`zai-org/GLM-5.3-Flash`) reports evaluation runs at
  temperature 1.0 with top_p 0.95 (HLE, BabyVision) or 1.0 (NL2Repo, Terminal-Bench),
  and temperature 0.95 with top_p 1.0 (DeepSWE); it states no `top_k` and no
  recommended `max_tokens` (its longest generation was 163,840 tokens). Neither
  card states `top_k`; the server's `top_k` fallback is not sourced from either.
- The profile carries no sampling flag: the engine has no `--default-temperature`
  family of flags at this commit, so the values above are documentation.
- The chat template opens a think block on every generation prompt. It has no
  off switch: `enable_thinking: false` has no effect and the response says so in
  `X-Rexl3-Note` (`docs/EVAL_RESULTS.md` "Serve hardening"). `reasoning_effort`
  is a chat-template variable: this server takes `low`, `medium`, `high` and
  `xhigh` in `chat_template_kwargs` and refuses a literal `max` and a top-level
  `reasoning_effort` field (HTTP 400). The template renders `xhigh` as its `max`,
  which is the default and the setting the base card keeps for benchmark
  reproduction. `clear_thinking` defaults to `false`. The smoke request sends
  `chat_template_kwargs: {reasoning_effort: low}` and a 2,048-token budget so the
  think block ends before the budget does; the launcher sends it verbatim.

## The model copy

`model.lock.json` pins `r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121` at revision
`b3194b87af5665bd8b1dd6316c99a48607fc137d` (Hugging Face, public): 39 files,
98,519,480,006 bytes. The 31 shard hashes are the Hugging Face LFS object ids
and were recomputed on a node's copy (read only) with no difference; the
other eight files were compared with the bytes downloaded at that revision. The
pack fingerprint `sha256:96a51ffa...f52c5` is the one `rexl3-serve
--print-fingerprints` printed on that node copy and the one in
`crates/model/src/fingerprint.rs` (`VALIDATED_TEXT`); the "(text)" in that table
names the scope of the oracle validation, not a capability limit. Only the root
documents `README.md`, `LICENSE` and `.gitattributes` may be extra in a model
directory. The node copy's `README.md` is an older text than the one at the
locked revision; it is not locked.

## Reference figures (dated; not floors)

All on the 2.25 bpw pack, one GB10, greedy, `docs/GLM53.md` unless noted.

| What | Value | Conditions |
| --- | --- | --- |
| Decode, one stream, autoregressive | 26.5 tok/s short context, 25.8 at 2,300 tokens, 25.4 at 261,120 | 2026-10-02/03, P1 era, `--spec none` |
| Decode with MTP | `auto` median 40.8 tok/s over six prompt classes | 2026-10-02, 384 new tokens, K chosen by the arbiter |
| Prefill, P2, one node | 939.5 tok/s at 8k, 929.1 at 32k | measured at the P2 flip, 2026-10-19 |
| Prefill, P1, long context | 717-733 tok/s (chunk 512), 802-807 (chunk 2,048) from 32k to 261k | 2026-10-03 |

The gate's NIAH stages at 50% and 100% of a 262,144-token window are about 131k
and 262k tokens of prefill (several minutes each at the figures above); the gate's
default request timeout is 300 s, so raise `GB10_GATE_REQUEST_TIMEOUT_S` for them.

## Memory

Weights are 90.6 GiB with the NextN layer plus 2.1 GiB for the tower and its
scratch. One GB10 has about 121 GiB; the engine plans against `MemAvailable`
with an 8 GiB floor and refuses a boot that does not fit. On a node with only
13 GiB free after the weights (other tenants) a vision-on boot at one slot was
refused in `docs/GLM53.md` ("0.0 GiB allocatable after the floor"): the profile
is sized for a node with no other large tenant. If a vision-on boot is refused on
a tester GB10, the answer is a does-not-fit proof and a maintainer decision, not
a silent `--no-vision`.

## Unverified boundaries

CPU fixture tests are not GPU or native-Rust execution evidence. The boot lines
named above were seen at this window; not run for this frozen configuration: the
rest of the gate's boot markers in
`scripts/gates/recipes/glm53-flash-2.25bpw-sm121.json`; the smoke request;
speculation counters at the window; the prefix-cache hit proof; vision answers on
generated images and an image-plus-tools case; and the dependency lock
(`dependencies.lock.json` is the DeepSeek-V4.1 two-node lock, byte for byte:
the Qwen3.8-27B single-node lock plus the `libnccl2` pin the two-node choice
needs (section "Two-node profile"), same native engine build and base image;
it has not been re-resolved for this profile on a clean host).
`profile.json` `source` names a document at the base commit that the flags were
derived from, not the profile file itself.

## Quality reference: a recorded baseline, not a floor

The release quality gate guards against regression in this engine's serving; it is not a target
for the model to reach. Each category's floor is the baseline measured on the release binary
minus a stated noise allowance, and that baseline is recorded with the release.

A trial on a development build, at the model's default effort (`xhigh` here, which the template
renders as its `max` and the base card keeps for benchmark reproduction), scored the BFCL
multi-turn lane at 12 of 20 cases (60%), with every reply finishing on its own (`stop` or
`tool_calls`, none cut at the lane's fixed 8,192-token cap). That is the model's own score on
this 20-case lane (one case is 5 points), recorded as the baseline and a limit of this pack's
tool calling, not a failure of the profile. The release run on the frozen binary replaces
this figure.
