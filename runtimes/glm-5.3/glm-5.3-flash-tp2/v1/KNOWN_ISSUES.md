# GLM-5.3-Flash EXL3 2.25 bpw (TP2) — frozen configuration, not qualification

This entry is the two-node GLM-5.3-Flash runtime, `glm53-flash-tp2-1m` (two
GB10s, tensor parallel, 1,048,576 tokens). It is an independent runtime unit
from the single-node entry `runtimes/glm-5.3/glm-5.3-flash/`
(`glm53-flash-single`), so the single node can release and be gated without the
two-node group and vice versa. This file covers the two-node entry only.

GLM two-node is 1,048,576 tokens only: `glm53-flash-tp2-1m` is this entry's
only choice, and it ships in v0.1.0 beside the single-node default.

What it is: the single-node choice's serving contract on a two-rank
tensor-parallel group of two GB10s joined by a RoCE link, at the pack's whole
1,048,576-token window. It uses the same pack as the single-node entry and its
own runtime id (`glm53-flash-tp2-sage-2.25`) and gate recipe
(`scripts/gates/recipes/glm53-flash-tp2-sage-2.25.json`); the recipe pins no
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

`model.lock.json` pins `r0b0tlab/GLM-5.3-Flash-EXL3-SAGE-2.25bpw` at revision
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
These are single-node figures; the two-node diagnostic figures are in the
introduction above.

| What | Value | Conditions |
| --- | --- | --- |
| Decode, one stream, autoregressive | 26.5 tok/s short context, 25.8 at 2,300 tokens, 25.4 at 261,120 | 2026-10-02/03, P1 era, `--spec none` |
| Decode with MTP | `auto` median 40.8 tok/s over six prompt classes | 2026-10-02, 384 new tokens, K chosen by the arbiter |
| Prefill, P2, one node | 939.5 tok/s at 8k, 929.1 at 32k | measured at the P2 flip, 2026-10-19 |
| Prefill, P1, long context | 717-733 tok/s (chunk 512), 802-807 (chunk 2,048) from 32k to 261k | 2026-10-03 |

The gate's NIAH stages at 50% and 100% of a 1,048,576-token window are about 524k
and 1,048k tokens of prefill; the gate's default request timeout is 300 s, so
raise `GB10_GATE_REQUEST_TIMEOUT_S` for them (the diagnostic run above used the
1M server's raised timeouts).

## Memory

Weights are 90.6 GiB with the NextN layer plus 2.1 GiB for the tower and its
scratch. One GB10 has about 121 GiB; the engine plans against `MemAvailable`
with an 8 GiB floor and refuses a boot that does not fit. Rank 0 also holds the
vision tower and its scratch. The planner's 1M plan fits with 23.6 GiB spare at
110 GiB planned free (diagnostic boot above); a host with other large tenants
may not.

## Reply end at a generated chat role token (fixed in v0.1.1)

v0.1.0 shipped a workaround: under greedy decoding near the 1,048,576-token limit,
the model could emit a chat role token after answering and start a second turn, so
the note was to leave `temperature` and `top_p` at their generation-config defaults.
The cause was the engine stop set: `generation_config.json`'s eos ends a reply on
`<|user|>` and `<|observation|>` but not on `<|assistant|>` or `<|system|>`, which
a GLM reply can also generate to open a new turn. v0.1.1 adds the GLM family's
role-open tokens to the stop set, so a reply that generates one finishes with
`finish_reason` `stop` and no second turn; outputs are identical up to the role
token, and tool calls are unaffected (a tool turn still ends on `<|observation|>`).
No workaround is needed in v0.1.1; the greedy note is withdrawn.

## Unverified boundaries

CPU fixture tests are not GPU or native-Rust execution evidence. The diagnostic
two-node run above is not a receipt: not run for this frozen configuration: a
real two-node receipt on the release binary, the smoke request, speculation
counters at the window, the prefix-cache hit proof, vision answers on generated
images and an image-plus-tools case, and a clean-host dependency resolution
(`dependencies.lock.json` is the DeepSeek-V4.1 two-node lock, byte for byte: the
Qwen3.8-27B single-node lock plus the `libnccl2` pin; same native engine build
and base image).
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
