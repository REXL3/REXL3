# Pack sheet: `qwen38-27b-dflash2-exl3-4.00`

Checked against release `v0.1.0`. This runtime was sealed on engine commit `799fa750f8eb` (the v0.1.0 freeze) and travels in the release unchanged (carried forward, tier T0); its sealed approval is `approve-qwen38-27b-dflash2-exl3-4.00-3`.

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `qwen38-27b-dflash2-exl3-4.00` | `profile.json` |
| Versions | v3 (latest, and the only version in this tree) | `runtimes/qwen3.8/qwen3.8-27b/` |
| Release tag | `v0.1.0` (one product release; there is no per-runtime tag) | `approval.json` |
| What it serves | Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`), EXL3 4.00 bits per weight, with a DFlash2 draft pack for speculative decoding | `MODEL.json`, `model.lock.json` |
| Model repository and revision | `r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw` at `3f1771b8c21f83cbb8e82169559ced9f38ca04e5` | `model.lock.json` |
| Draft or second model | `r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw` at `265b5240592907d2d55ff0dc4d5f66569692604d` | `model.lock.json` |
| Model licence | Apache-2.0 (model card metadata of each pack repository) | `model.lock.json` |
| Download size | 14 target files (15.4 GiB) and 3 draft files (1.2 GiB); free space needed: the remaining bytes plus a reserve | `model.lock.json` |

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen38-single-262k` (default) | single node | 262,144 tokens | 6 | 8889 | `127.0.0.1` | at most 600 s (`readiness_timeout_s`, a ceiling) after model verification |

Source: `profile.json`. There is no tensor-parallel choice for this runtime.

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | yes | |
| Thinking (reasoning) | yes | Default when a request names no setting: on, at `low` effort (`--default-reasoning-effort low`). A request may name `reasoning_effort` (`low`, `medium`, `xhigh`) or set `enable_thinking: false`. |
| Tools | yes | `tool_choice` `auto` and `none` only; at most 64 tools |
| Images | yes | Inline data URI only. Video and audio are refused. |
| Prefix cache | yes | Automatic; pinned at 40,960 MiB (`--prefix-cache-mib 40960`) |
| Speculative decoding | yes | Automatic (`--spec auto`, DFlash2 draft) |

Source: `profile.json` `capabilities` and `KNOWN_ISSUES.md`. Constrained JSON response formats (`json_object`, `json_schema`) are refused. Everything not listed as supported is refused (user guide section 5.10).

## 4. Sampling and output defaults

The pack's `generation_config.json` sets temperature 1.0, top_p 0.95 and top_k 20, and the server reads all three from it. Non-zero presence or repetition penalties are refused. Because a request that names nothing now thinks, give replies enough `max_tokens` for the thinking to finish before the answer. Source: `KNOWN_ISSUES.md` "Sampling defaults and thinking".

## 5. Behaviour to expect

- A request that names no reasoning setting thinks at `low` effort. The pack's own template defaults to `xhigh`, and the engine's own default would not think at all; the profile pins `low` so a client that sends nothing gets the recommended behaviour (`KNOWN_ISSUES.md`).
- The runtime launches only with the packs the launcher downloads itself. It cannot yet be pointed at existing local copies through the `model_dir` site field, because the pack has two roles (`KNOWN_ISSUES.md` "Launching").
- A video part is refused by name; the Qwen family does not encode video.
- The sealed tree's `KNOWN_ISSUES.md` was written before the chain ran and still says qualification is `unverified`; that sentence is out of date. The record is the sealed approval and the measurements in section 6. A sealed tree is not edited.

## 6. Measured behaviour

| Quantity | Value | Hardware and release | Recorded in |
| --- | --- | --- | --- |
| Generation speed, one stream | 62.5 tokens/s (median of 3 runs: 62.47, 62.50, 62.64) | one GB10, engine `799fa750f8eb` | performance gate, `performance/performance-cases.json` in the approval's evidence asset |
| Prompt speed at 50% and 100% of the window | 808.6 and 621.9 tokens/s (median of 3 runs each) | one GB10, engine `799fa750f8eb` | same file |
| Quality lane (Q200, reasoning effort `low`, 262,144 window) | GSM8K 79/80 (98.75%), HumanEval 39/40 (97.5%), IFEval 35/40 (87.5%), hard reasoning 20/20 (100%), BFCL hard-20 15/20 (75%); all above the 70% floor | one GB10, engine `799fa750f8eb` | `quality/quality-cases.json` in the same asset |

No figure from an earlier version or an earlier build is carried here.

## 7. Limits of this release

The chain ran the gates on one GB10 node (`r0b0tdgx1`) with this profile's contract (6 slots, 262,144 tokens, vision tower loaded, prefix cache on) and every gate in the evidence summary passed: doctor, lifecycle, capability, boundary, context, exactness, prefix cache, performance, performance smoke, vision and quality. Speed was measured on that one node only; a second node has not confirmed it. The quality figures are one run at `low` reasoning effort; the BFCL tool-calling set (20 cases) is the weakest result at 75%. Not supported: video, audio and constrained JSON response formats.

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/qwen3.8/qwen3.8-27b/v3/KNOWN_ISSUES.md` (v3 confirmed)
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
