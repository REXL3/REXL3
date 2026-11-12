# Pack sheet: `glm53-flash-sage-2.25`

Checked against version 2 (dev approval `approve-glm53-flash-sage-2.25-2`, source commit `75031112ba12`, engine release `engine-v0.1.1-75031112`, archive `glm53-flash-sage-2.25-2.tar.gz`). Version 2 holds the single-node profile `glm53-flash-single` only, under the runtime's current id `glm53-flash-sage-2.25` (renamed on dev from `glm53-flash-2.25bpw-sm121`). Version 1 (release `v0.1.0`, engine commit `799fa750f8eb`, approval `approve-glm53-flash-2.25bpw-sm121-1`) stays in the tree unchanged under the earlier id, with both the single-node and the two-node profile; production accepts it under the current id through the reviewed alias for `glm53-flash-single`. The two-node profile now ships as its own runtime unit, `glm-5.3/glm-5.3-flash-tp2` (runtime `glm53-flash-tp2-sage-2.25`), with its own pack sheet.

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `glm53-flash-sage-2.25` (version 2); version 1 was sealed as `glm53-flash-2.25bpw-sm121` | `profile.json`, `approval.json` |
| Versions | `v1` (v0.1.0), `v2` (latest) | `runtimes/glm-5.3/glm-5.3-flash/` |
| Release tag | version 1 shipped in `v0.1.0`; version 2 ships in the next product release (there is no per-runtime tag) | `approval.json` |
| What it serves | GLM-5.3-Flash (`Glm5NextForConditionalGeneration`), EXL3 2.25 bits per weight; speculation uses the pack's own NextN layer | `MODEL.json`, `model.lock.json`, `KNOWN_ISSUES.md` |
| Model repository and revision | `r0b0tlab/GLM-5.3-Flash-EXL3-SAGE-2.25bpw` at `b3194b87af5665bd8b1dd6316c99a48607fc137d` (the repository was renamed from `r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121`, which version 1's lock names; same revision and files) | `model.lock.json` |
| Draft or second model | none | `model.lock.json` |
| Model licence | MIT (base model `zai-org/GLM-5.3-Flash`; see the pack's LICENSE and model card) | `model.lock.json` |
| Download size | 39 files, 91.8 GiB; free space needed: the remaining bytes plus a reserve | `model.lock.json` |

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `glm53-flash-single` (default) | single node | 262,144 tokens | 1 | 8950 | `127.0.0.1` | at most 900 s (`readiness_timeout_s`, a ceiling) after model verification |

Source: version 2's `profile.json`. The pack advertises a 1,048,576-token window. One GB10 cannot hold it with the required features, so the single-node profile ships at 262,144 (`KNOWN_ISSUES.md` "Why the window is 262,144"). The full window on two GB10s is the separate runtime unit `glm-5.3/glm-5.3-flash-tp2`; version 1 of this folder still carries its v0.1.0 two-node profile `glm53-flash-tp2-1m`.

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | yes | |
| Thinking (reasoning) | yes | Effort scale, no off switch: `enable_thinking: false` has no effect and the response says so in `X-Rexl3-Note`. The template default is `xhigh`; the server refuses the literal `max`. |
| Tools | yes | `tool_choice` `auto` and `none` only; at most 64 tools |
| Images | yes | Inline data URI only; up to 8,000 tokens an image |
| Prefix cache | yes | Automatic. Single node: pinned at 3,584 MiB (`--prefix-cache-mib 3584`), 2 entries, reuse of prompts up to 265,216 tokens, one long conversation at a time. (`KNOWN_ISSUES.md`) |
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

All figures are from version 2's qualification, recorded in `evidence.tar.gz` (the dev approval's evidence asset): one GB10 node, engine `0.1.1` at commit `75031112ba12` (binary `rexl3-serve` sha256 `4fc2cb44…5fe5`), profile `glm53-flash-single`, window 262,144, 1 slot, vision on. Seven gates ran on that binary; `context` and `quality` are carried from version 1's sealed single-node qualification (engine `799fa750f8eb`, binary `3e188724…`), as recorded in the receipt `v011/composed/summary.json`.

| Quantity | Value | Recorded in |
| --- | --- | --- |
| Generation speed, one stream (median of 3 runs, each 5 streamed replies, speculation on, short context) | 38.9 tok/s (38.9 to 39.2); floor 31 | `v011/performance/performance-cases.json` |
| Prompt speed at 50% of the window (median of 3, cold) | 997 tok/s (987 to 1,000); floor 770 | same |
| Prompt speed at 100% of the window (median of 3, cold) | 994 tok/s (993 to 996); floor 750 | same |
| Draft acceptance during the baseline runs | 0.72 | `v011/performance/runs/run-0N-RESULT.json` |
| Lowest free memory during the baseline runs | 9.92 to 10.44 GiB, no swap | same |
| Boot to ready | 15.8 s first boot, 23.6 s restart | `v011/lifecycle/lifecycle-cases.json` |
| Exactness | 68 of 68 cases pass (drafted equals serial; chunked prefill equals whole prefill up to a 261,120-token prompt) | `v011/exactness/exactness-cases.json` |
| Prefix cache | gate passes (84 checks, 0 failures) | `v011/exactness/prefix-cache/check_prefix_cache.log` |
| Release gates | all PASS in one composed pass: fresh on the v0.1.1 binary — doctor 4/4, lifecycle 10/10, capability 5/5, boundary 5/5, vision 3/3, exactness 68/68, prefix cache 1/1, performance 4/4, performance smoke 3/3; carried — context 8/8, quality 5/5 | `v011/composed/summary.json` |

Quality, carried from version 1's single-node run (Q200, reasoning effort `xhigh`, 262,144 window, one GB10, engine `799fa750f8eb`): GSM8K 77/80 (96.25%), HumanEval 39/40 (97.5%), IFEval 37/40 (92.5%), hard reasoning 19/20 (95%), BFCL hard-20 12/20 (60%). These are this pack's regression baseline, not a claim of absolute quality. Recorded in version 1's `quality/quality-cases.json`.

Only `context` and `quality` are carried; every other figure above was measured on the v0.1.1 binary. Version 1's two-node figures are in version 1's evidence asset; they do not describe version 2.

## 7. Limits of this release

One slot. The window is 262,144, not the advertised 1,048,576; the two-node unit `glm-5.3/glm-5.3-flash-tp2` serves the full window. Version 2's gates ran on one GB10 and every gate in its evidence passed; speed was measured on one node only. BFCL tool-calling (hard-20) is the weakest result at 12/20. Version 2 was qualified for engine commit `75031112ba12` and its release binary; `context` and `quality` come from version 1's build under the carried-gate rule (CONTRIBUTING section 3). No 24-hour soak was run. Not supported: audio, video and constrained JSON response formats.

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/glm-5.3/glm-5.3-flash/v2/KNOWN_ISSUES.md` (single-node only; version 1's file also covers the two-node profile)
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
