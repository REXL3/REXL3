# Pack sheet: `qwen38-27b-dflash2-exl3-4.00`

Checked against version 4 (dev approval `approve-qwen38-27b-dflash2-exl3-4.00-4`, source commit `75031112ba12`, engine release `engine-v0.1.1-75031112`, archive `qwen38-27b-dflash2-exl3-4.00-4.tar.gz`). Version 3 (release `v0.1.0`, engine commit `799fa750f8eb`, approval `approve-qwen38-27b-dflash2-exl3-4.00-3`) stays in the tree unchanged. The runtime files of the two versions are identical (`profile.json`, `model.lock.json`, `KNOWN_ISSUES.md`); version 4 is the same packs and profile requalified on the v0.1.1 engine.

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `qwen38-27b-dflash2-exl3-4.00` | `profile.json` |
| Versions | `v3` (v0.1.0), `v4` (latest) | `runtimes/qwen3.8/qwen3.8-27b/` |
| Release tag | version 3 shipped in `v0.1.0`; version 4 ships in the next product release (there is no per-runtime tag) | `approval.json` |
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

All figures are from version 4's qualification, recorded in `evidence.tar.gz` (the dev approval's evidence asset): one GB10 node, engine `0.1.1` at commit `75031112ba12` (binary `rexl3-serve` sha256 `4fc2cb44…5fe5`), profile `qwen38-single-262k`. Seven gates ran on that binary; `context` and `quality` are carried from version 3's sealed qualification (engine `799fa750f8eb`, binary `3e188724…`), as recorded in the receipt `v011/composed/summary.json`.

| Quantity | Value | Recorded in |
| --- | --- | --- |
| Generation speed, one stream (median of 3 runs, each 5 streamed replies, DFlash2 draft on, short context) | 62.2 tok/s (runs 62.21, 62.19, 62.20); floor 35 | `v011/performance/performance-cases.json` |
| Prompt speed at 50% of the window (median of 3, cold) | 835.8 tok/s (834.9 to 840.6); floor 600 | same |
| Prompt speed at 100% of the window (median of 3, cold) | 634.7 tok/s (632.1 to 635.1); floor 470 | same |
| Lowest free memory during the baseline runs | 35.49 to 35.94 GiB, no swap | `v011/performance/runs/run-0N-RESULT.json` |
| Boot to ready | 4.5 s first boot, 5.5 s restart | `v011/lifecycle/lifecycle-cases.json` |
| Exactness | 68 of 68 cases pass (drafted equals serial; chunked prefill equals whole prefill) | `v011/exactness/exactness-cases.json` |
| Prefix cache | gate passes (83 checks, 0 failures, 25 cache hits counted) | `v011/exactness/prefix-cache/check_prefix_cache.log` |
| Release gates | all PASS in one composed pass: fresh on the v0.1.1 binary — doctor 4/4, lifecycle 10/10, capability 5/5, boundary 5/5, vision 3/3, exactness 68/68, prefix cache 1/1, performance 4/4, performance smoke 3/3; carried — context 8/8, quality 5/5 | `v011/composed/summary.json` |

Quality, carried from version 3 (Q200, reasoning effort `low`, 262,144 window, one GB10, engine `799fa750f8eb`): GSM8K 79/80 (98.75%), HumanEval 39/40 (97.5%), IFEval 35/40 (87.5%), hard reasoning 20/20 (100%), BFCL hard-20 15/20 (75%). Recorded in `quality/quality-cases.json` in the same asset. These are this pack's regression baseline, not a claim of absolute quality.

Only `context` and `quality` are carried; every other figure above was measured on the v0.1.1 binary.

## 7. Limits of this release

Version 4's gates ran on one GB10 node with this profile's contract (6 slots, 262,144 tokens, vision tower loaded, prefix cache on) and every gate in its evidence passed. Speed was measured on one node only. Version 4 was qualified for engine commit `75031112ba12` and its release binary; `context` and `quality` come from version 3's build under the carried-gate rule (CONTRIBUTING section 3). The quality figures are one run at `low` reasoning effort; the BFCL tool-calling set (20 cases) is the weakest result at 75%. No 24-hour soak was run. Not supported: video, audio and constrained JSON response formats.

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/qwen3.8/qwen3.8-27b/v4/KNOWN_ISSUES.md` (identical to version 3's)
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
