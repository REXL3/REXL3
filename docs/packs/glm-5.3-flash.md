# Pack sheet: `glm53-flash-2.25bpw-sm121`

Checked against release `v0.1.0`. This runtime was sealed on engine commit `799fa750f8eb` (the v0.1.0 freeze) and travels in the release unchanged (carried forward, tier T0); its sealed approval is `approve-glm53-flash-2.25bpw-sm121-1`, one approval covering both profiles.

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `glm53-flash-2.25bpw-sm121` | `profile.json` |
| Versions | v1 (the only version in this tree) | `runtimes/glm-5.3/glm-5.3-flash/` |
| Release tag | `v0.1.0` (one product release; there is no per-runtime tag) | `approval.json` |
| What it serves | GLM-5.3-Flash (`Glm5NextForConditionalGeneration`), EXL3 2.25 bits per weight; speculation uses the pack's own NextN layer | `MODEL.json`, `model.lock.json`, `KNOWN_ISSUES.md` |
| Model repository and revision | `r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121` at `b3194b87af5665bd8b1dd6316c99a48607fc137d` | `model.lock.json` |
| Draft or second model | none | `model.lock.json` |
| Model licence | MIT (base model `zai-org/GLM-5.3-Flash`; see the pack's LICENSE and model card) | `model.lock.json` |
| Download size | 39 files, 91.8 GiB; free space needed: the remaining bytes plus a reserve | `model.lock.json` |

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `glm53-flash-single` (default) | single node | 262,144 tokens | 1 | 8950 | `127.0.0.1` | at most 900 s (`readiness_timeout_s`, a ceiling) after model verification |
| `glm53-flash-tp2-1m` | two nodes, tensor parallel (2 ranks) | 1,048,576 tokens | 1 | 8950 | `127.0.0.1` on the first node | at most 900 s (`readiness_timeout_s`, a ceiling) after model verification |

Source: `profile.json`. The pack advertises a 1,048,576-token window. One GB10 cannot hold it with the required features, so the single-node profile ships at 262,144 (`KNOWN_ISSUES.md` "Why the window is 262,144"); the full window needs two GB10s joined by a RoCE link and the site's cluster file (`config/cluster.tp2-1m.json` carries example network values the site must replace: nodes, fabric interface, HCAs and TP port). Rank 0 serves HTTP and holds the vision tower.

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | yes | |
| Thinking (reasoning) | yes | Effort scale, no off switch: `enable_thinking: false` has no effect and the response says so in `X-Rexl3-Note`. The template default is `xhigh`; the server refuses the literal `max`. |
| Tools | yes | `tool_choice` `auto` and `none` only; at most 64 tools |
| Images | yes | Inline data URI only; up to 8,000 tokens an image |
| Prefix cache | yes | Automatic. Single node: pinned at 3,584 MiB (`--prefix-cache-mib 3584`), 2 entries, reuse of prompts up to 265,216 tokens, one long conversation at a time. Two nodes: pinned at 13,440 MiB (`--prefix-cache-mib 13440`), 5 entries, reach 1,051,136 tokens on each rank (`KNOWN_ISSUES.md`) |
| Speculative decoding | yes | Automatic (`--spec auto`, the pack's NextN layer) |

Source: `profile.json` `capabilities` and `KNOWN_ISSUES.md`. Audio, video and constrained JSON response formats are unsupported.

## 4. Sampling and output defaults

The pack's `generation_config.json` sets temperature 1.0 and top_p 0.95. The pack and base card state no `top_k`, and the server's fallback `top_k` is not sourced from either. The profile carries no sampling flag. Source: `KNOWN_ISSUES.md` "Sampling and reasoning".

## 5. Behaviour to expect

- Thinking cannot be switched off; use `reasoning_effort` to scale it. A smoke request in the sealed profile sends effort `low` with a 2,048-token budget.
- Prefix reuse is bounded by the cache's reach: a prompt longer than the reach is prefilled in full each time (`KNOWN_ISSUES.md`).
- With one slot and two cache snapshots, the single-node cache holds one long conversation: a different long prompt evicts it and returning to the first one prefills it cold again (`KNOWN_ISSUES.md`).
- The single-node profile lowers the memory floor to 4 GiB (`REXL3_MEMORY_FLOOR_GIB=4`); below about 18 GiB of planned free memory the engine shrinks the window and the launcher refuses the boot (`MODEL_LIMIT_MISMATCH`).

## 6. Measured behaviour

| Quantity | Value | Hardware and release | Recorded in |
| --- | --- | --- | --- |
| Generation speed, one stream, single node | 39.7 tokens/s (median of 3 runs: 39.7, 39.7, 39.7) | one GB10, engine `799fa750f8eb` | performance gate, `performance/performance-cases.json` in the approval's evidence asset |
| Prompt speed at 50% and 100% of the window, single node | 971.4 and 976.3 tokens/s (median of 3 runs each) | one GB10, engine `799fa750f8eb` | same file |
| Quality lane (Q200, reasoning effort `xhigh`, 262,144 window), single node | GSM8K 77/80 (96.25%), HumanEval 39/40 (97.5%), IFEval 37/40 (92.5%), hard reasoning 19/20 (95%), BFCL hard-20 12/20 (60%); the first four above the 70% floor; BFCL hard-20 is a scored set whose receipt status is PASS | one GB10, engine `799fa750f8eb` | `quality/quality-cases.json` in the same asset |
| Generation speed, one stream, two nodes | 57.2 tokens/s (median of 3 runs: 56.7, 57.9, 57.2) | two GB10s, engine `799fa750f8eb` | `tp2/performance/performance-cases.json` |
| Prompt speed at 50% and 100% of the 1,048,576 window, two nodes | 1,130.6 and 979.3 tokens/s (median of 3 runs each) | two GB10s, engine `799fa750f8eb` | same file |
| Quality lane (Q200, `xhigh`, 1,048,576 window), two nodes | same counts as the single-node run: GSM8K 77/80, HumanEval 39/40, IFEval 37/40, hard reasoning 19/20, BFCL hard-20 12/20 | two GB10s, engine `799fa750f8eb` | `tp2/quality/quality-cases.json` |

No figure from an earlier version or build is carried here. The decode floor in the recipe is 31 tokens/s for both profiles.

## 7. Limits of this release

One slot in both profiles. The single-node window is 262,144, not the advertised 1,048,576. The chain ran the gates on one GB10 (`glm53-flash-single`) and on two GB10s (`glm53-flash-tp2-1m`) and every gate in each profile's evidence passed; speed was measured on one node or one pair of nodes only. BFCL tool-calling (hard-20) is the weakest result at 12/20. The two-node profile needs a site cluster file and the NCCL library (its dependency lock pins `libnccl2`); the sealed `KNOWN_ISSUES.md` records the earlier diagnostic run for that profile and states that none of its figures is a floor. Not supported: audio, video and constrained JSON response formats.

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/glm-5.3/glm-5.3-flash/v1/KNOWN_ISSUES.md`
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
