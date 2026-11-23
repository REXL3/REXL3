# GLM-5.3-Flash EXL3 2.25 bpw — frozen configuration, not qualification

The first GLM release (v0.1.0) ships two independent runtime entries: the
default, `glm53-flash-single` (one GB10, 262,144 tokens), and the two-node
`glm53-flash-tp2-1m` (two GB10s, tensor parallel, 1,048,576 tokens), which
lives in its own entry `runtimes/glm-5.3/glm-5.3-flash-tp2/` (section
"Two-node profile" in that entry's KNOWN_ISSUES.md). This file covers the
single-node entry only; the two entries are independent runtime units and
release/receipt separately.

The single-node profile serves text, SSE, reasoning, tools and images, with
speculative decoding (`--spec auto`, the pack's own NextN layer) and the prefix
cache on. Audio, video and constrained JSON response formats are unsupported.
Configuration freeze does not imply full-window, throughput, multimodal or
quality qualification: every number below is a dated reference under the
conditions stated, and none is a floor of this profile. The gate's speed floors
are in the recipe: decode 31 tok/s, prefill 770 tok/s at 50% and 750 at 100% of
the window.

| | `glm53-flash-single` |
| --- | --- |
| Window (`--max-model-len`) | 262,144 |
| Slots | 1 |
| KV pool | planned by the engine; equals the window at 262,144 |
| Speculation | `--spec auto` |
| Prefix cache | on, pinned at 3,584 MiB (`--prefix-cache-mib 3584`, 2 entries, 1,036 pages); reuses prompts up to 265,216 tokens, one long conversation at a time |
| Memory floor | `REXL3_MEMORY_FLOOR_GIB=4` (engine default 8) |
| Images | on (24-layer vision tower, up to 8,000 tokens an image) |
| Prefill contract | P2 (the engine default since 2026-10-19) |

## Release status and current known issues

Released as version 1 in v0.1.0, as one unit with the two-node profile, and as version 2 in v0.1.1. v0.1.2 shipped version 2 again, unchanged.

Where the sections below say "not yet run", "pending" or "to be confirmed", they describe the entry before its release qualification. The receipts of that qualification are in the evidence archive of each release named above.

Known issues as of 2026-11-19. Each says what happens, who is affected and what to do:

- **NVIDIA driver: the 580 branch at build 580.142 or later; builds after 580.178.04 are accepted but not yet qualified.** This applies to every host. `libnvidia-compute-580` and `nvidia-utils-580` must both be the same upstream build; Ubuntu's `-0ubuntu0.24.04.1` and NVIDIA's CUDA-repository `-1ubuntu1` packages of one build count as the same driver. On a build later than 580.178.04, `./launch.sh doctor` and `install-deps` continue and print `DRIVER_NOT_YET_QUALIFIED` (`doctor` reports `driver_qualification: newer-than-tested`): 580.142 is the tested floor and 580.178.04 the highest build this runtime was qualified on, and results on a later build are not covered by the release's qualification evidence. The CUDA driver, runtime and cuBLAS versions that `rexl3-serve --version` reports must be the qualified ones or later on the same major version: a later one is accepted and `doctor` prints `NATIVE_NOT_YET_QUALIFIED`, and an older one or another major version stops `doctor` with `NATIVE_VERSION_MISMATCH`. A build below 580.142, or two different builds on one host, stops `doctor` with `SYSTEM_DEPENDENCY_VERSION_MISMATCH`; another driver branch (570, 590) is not accepted. When a driver change is needed and the configured repositories no longer offer the `580.178.04-0ubuntu0.24.04.1` package, `install-deps` stops with `DRIVER_PACKAGE_UNAVAILABLE` and installs nothing. What to do: stay on the 580 branch at 580.142 or later; if the package is unavailable, install any 580 build at or above 580.142 with the host package manager (both packages at the same build), reboot and rerun `./launch.sh doctor`.
- **A pending restart no longer stops the launcher.** When the host has a restart pending (`/var/run/reboot-required`, set by a kernel or system library update), `doctor` and `install-deps` print `REBOOT_PENDING` and continue. A driver upgraded without a restart is refused by name: `doctor` stops with `DRIVER_MODULE_MISMATCH` when the loaded NVIDIA kernel module is not the installed driver build, and after `install-deps` has changed the driver packages itself it stops with `DRIVER_MODULE_MISMATCH` too (with `REBOOT_REQUIRED` when the loaded module's build cannot be read and a restart is pending). What to do: restart the host, then rerun `./launch.sh doctor`.
- **Thinking cannot be turned off.** The chat template opens a reasoning block on every reply; `enable_thinking: false` has no effect, and the response says so in the `X-Rexl3-Note` header. What to do: allow for the reasoning in `max_tokens`, or lower it with `chat_template_kwargs: {"reasoning_effort": "low"}`.
- **The prefix cache holds one long conversation at a time.** A different long prompt evicts a full-window entry, and the next send of the first prompt is prefilled from the start again. Who is affected: a user who alternates between very long prompts. Short tool loops and one long session are served from the cache (see "One long conversation at a time" below).
- **The opt-in choice `glm53-flash-2.48-dflash-single` is not release-qualified.** It has not been in a release. What to do: use the default choice `glm53-flash-single` for qualified serving.

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

So pinning keeps the cache but moves the one-giB sensitivity to the window; it does not
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

The two-node choice is not in this entry: it ships as its own runtime entry
`runtimes/glm-5.3/glm-5.3-flash-tp2/` (`glm53-flash-tp2-1m`,
runtime `glm53-flash-tp2-sage-2.25`, gate recipe
`scripts/gates/recipes/glm53-flash-tp2-sage-2.25.json`), so the single node
can release and be gated without the two-node group.

## Draft licence and distribution

The default `glm53-flash-single` profile and its `target` lock role keep the
original `r0b0tlab/GLM-5.3-Flash-EXL3-SAGE-2.25bpw` pack and its embedded NextN
layer. Its argv has no `--draft`, and `default_profile` remains
`glm53-flash-single`, so default users need no external draft.

### Opt-in SAGE 2.48 + DFlash2 choice

The additional `glm53-flash-2.48-dflash-single` choice uses the separate
`sage-2.48` lock role (`vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw` at revision
`7d0de24624020fc4f07923572be2cef591a4d1ed`) and the `draft` role
(`incoai/GLM-5.3-Flash-DFlash2` at revision
`6cc6c481eee930a5139619ed919f78726371adbc`, whose `model.safetensors` and
`config.json` are byte-identical to the earlier revision `bf582e4e`). Its own runtime id is
`glm53-flash-dflash2-sage-rexl3-2.48`; it runs one slot at 262,144 tokens with
`--spec auto --draft @DRAFT_DIR@`, vision on and the 3,584 MiB prefix cache.
The original 2.25 target and profile remain available unchanged.

The 2.48 pack's `chat_template.jinja` intentionally differs from the default 2.25
pack's template: its tool-result ordering block is rewritten. The engine renders
the selected pack's own template (`formats/glm/mod.rs`), so several tool results
use the 2.48 ordering for this choice; it does not fall back to the 2.25 template.

The DFlash2 creator is **Inco AI**. Its licence is **Apache-2.0**
(<https://www.apache.org/licenses/LICENSE-2.0>) since revision `6cc6c481`
(2026-11-10); the earlier revision `bf582e4e` was CC BY-NC-ND 4.0. The
operator obtains the published draft from `incoai/GLM-5.3-Flash-DFlash2`; no
draft weights are shipped in this repository. **The engine converts the
published weights in memory at load and writes nothing to disk.**
DFlash2 only proposes; the target verifies every committed token, and the
pack's NextN layer is not loaded for this opt-in choice.

This choice pins `REXL3_MEMORY_FLOOR_GIB=0.5`, instead of the original 2.25
profile's 4 GiB. Live plans with 4 and 1.5 GiB floors capped KV at 66,304 and
200,960 tokens respectively, so those floors cannot hold the mandatory 262,144
window with DFlash2 state, vision and the 3,584 MiB cache on this 99.46-GiB
pack. The 0.5-GiB floor is an explicit unallocated margin, not a claim of free
headroom: the gate still fails if the live plan shrinks or the memory monitor
trips. The PR body records the minimum MemAvailable measured over the receipted
run.

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
with this profile's 4 GiB floor (engine default 8) and refuses a boot that does not fit. On a node with only
13 GiB free after the weights (other tenants) a vision-on boot at one slot was
refused in `docs/GLM53.md` ("0.0 GiB allocatable after the floor"): the profile
is sized for a node with no other large tenant. If a vision-on boot is refused on
a tester GB10, the answer is a does-not-fit proof and a maintainer decision, not
a silent `--no-vision`.

## Reply end at a generated chat role token (fixed in v0.1.1)

GLM opens a new turn with a single role token, and `generation_config.json`'s eos
ends a reply on `<|user|>` and `<|observation|>` but not on `<|assistant|>` or
`<|system|>`. A greedy reply could generate one of those after answering and run on
into a second turn (observed on the two-node 1M profile). v0.1.1 adds the GLM
family's role-open tokens to the engine stop set, so such a reply finishes with
`finish_reason` `stop`; outputs are identical up to the role token, and tool calls
are unaffected (a tool turn still ends on `<|observation|>`).

## Unverified boundaries

CPU fixture tests are not GPU or native-Rust execution evidence. The release
gates (the boot markers in
`scripts/gates/recipes/glm53-flash-sage-2.25.json`, the smoke request,
speculation counters at the window, the prefix-cache hit proof, vision answers on
generated images and an image-plus-tools case) ran in the v0.1.0 and v0.1.1
qualifications; their receipts are in those releases' evidence archives. Not
done: a clean-host re-resolution of the dependency lock
(`dependencies.lock.json` is the Qwen3.8-27B single-node lock, byte for byte,
same native engine build and base image; it has not been re-resolved for this
profile on a clean host).
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
tool calling, not a failure of the profile. The v0.1.1 release notes record the same 12 of 20
for version 2 at effort `xhigh`.
