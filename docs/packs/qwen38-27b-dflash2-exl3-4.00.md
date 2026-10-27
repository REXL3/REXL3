# Pack sheet: `qwen38-27b-dflash2-exl3-4.00`

Checked against release `prod/qwen38-27b-dflash2-exl3-4.00/v1` (source commit `4208d8486937`, archive `qwen38-27b-dflash2-exl3-4.00-1.tar.gz`, promotion commit `b119ffe`).

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `qwen38-27b-dflash2-exl3-4.00` | `profile.json` |
| Versions | `v1` (latest) | `runtimes/qwen38-27b-dflash2-exl3-4.00/` |
| Release tag | `prod/qwen38-27b-dflash2-exl3-4.00/v1` | `approval.json` (approve tag `approve-qwen38-27b-dflash2-exl3-4.00-1`) |
| What it serves | Qwen3.8-27B, 4.00 bits per weight in the published EXL3 format, with a DFlash2 speculative-decoding draft | `model.lock.json` |
| Model repository and revision | `r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw` at `3f1771b8c21f83cbb8e82169559ced9f38ca04e5` | `model.lock.json` |
| Draft or second model | `r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw` at `265b5240592907d2d55ff0dc4d5f66569692604d` | `model.lock.json` |
| Model licence | Apache-2.0 for both packs (the lock records "model card metadata of the pack repository") | `model.lock.json` |
| Download size | 17 files, 16.57 GiB (target 14 files, 15.40 GiB; draft 3 files, 1.17 GiB); free space needed: the remaining bytes plus a reserve | `model.lock.json` |
| Approved by | `am423`, schema `rexl3.profile-approval.v2` | `approval.json` |
| Engine | version `0.1.2`, commit `4208d8486937`, native build fingerprint `fp_4208d8486937_e2657881_cb1311` | `approval.json`, evidence |

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen38-single-262k` (default) | single node | 262,144 tokens | 6 | 8889 | `127.0.0.1` | 600 s after model verification (a ceiling, not a measurement) |

Source: `profile.json`. Served model name `qwen38-27b-exl3`. Arguments of the profile: `--slots 6`, `--max-model-len 262144`, `--spec auto` with the draft pack, `--prefix-cache on`, `--prefix-cache-mib 40960`, `--default-reasoning-effort low`. There is no two-node profile.

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | yes | |
| Thinking (reasoning) | yes | Default when a request names no setting: on, at effort `low` (`--default-reasoning-effort low`). A request that names an effort or sets `enable_thinking: false` keeps its own. |
| Tools | yes | `tool_choice` `auto` and `none` only; at most 64 tools |
| Images | yes | Inline data URI only. Vision is on. |
| Prefix cache | yes | Automatic; pinned at 40,960 MiB |
| Speculative decoding | yes | Automatic, with the DFlash2 draft pack |
| Video, audio, `json_object`, `json_schema` | no | Refused by the server |

Source: `profile.json` `capabilities` and `KNOWN_ISSUES.md`. Everything not listed as supported is refused (user guide section 5.10).

## 4. Sampling and output defaults

The pack ships `generation_config.json` with temperature 1.0, top_p 0.95 and top_k 20, and the server reads all three from it when a request does not set them (`KNOWN_ISSUES.md`). The server refuses non-zero presence and repetition penalties. Because a request that names nothing now thinks at effort `low`, give thinking requests a generous `max_tokens`; the qualification run used a text budget of 32,768 tokens (`quality/quality-cases.json`).

## 5. Behaviour to expect

- The default reasoning effort is `low`. The pack's own template defaults to `xhigh`; this profile sets the recommended `low` as a server default that a request can override (`KNOWN_ISSUES.md`, "Sampling defaults and thinking").
- Video input is refused by name: the Qwen family does not encode video (`KNOWN_ISSUES.md`).
- The launcher runs this profile with the packs it downloads itself. It cannot yet be pointed at existing local copies through `--site model_dir`, because the pack has two roles and the single-node site fields carry no draft directory (`KNOWN_ISSUES.md`, "Launching").
- The sealed `profile.json` still records `qualification.state: unverified`, and `KNOWN_ISSUES.md` describes the frozen configuration; both were written before the qualification of this release. The qualification of this release is the evidence asset `evidence.tar.gz` (section 6). Where the two differ, the release evidence is the record of what was measured on this commit.
- The dependency lists have not been rehearsed on a clean host for this runtime in the sealed files (`KNOWN_ISSUES.md`, "Dependencies").

## 6. Measured behaviour

All figures are from this release's own qualification, recorded in `evidence.tar.gz` (the release asset): one GB10 node, engine `0.1.2` at commit `4208d8486937`, 6 slots, window 262,144, vision on, prefix cache pinned at 40,960 MiB.

| Quantity | Value | Recorded in |
| --- | --- | --- |
| Generation speed, one stream (median of 3 runs) | 62.6 tok/s (62.56 to 62.64); floor 35 | `performance/performance-cases.json` |
| Prompt speed at 50% window fill (median of 3) | 772.8 tok/s (772.1 to 773.5); floor 600 | same |
| Prompt speed at full window fill (median of 3) | 600.1 tok/s (599.1 to 600.7); floor 470 | same |
| Lowest free memory during the baseline runs | 36.25 to 36.46 GiB, no swap | `performance/runs/run-0N-RESULT.json` |
| Context gate | 8 of 8 rows pass at 131,072 and 262,144 tokens, with the cache warm | `single/summary.json` |
| Exactness | 67 of 67 cases pass (drafted equals serial; chunked prefill equals whole prefill) | `exactness/exactness-cases.json` |
| Release gates | all nine PASS in one composed pass (doctor 4/4, lifecycle 10/10, capability, vision, context, boundary, exactness, prefix cache, performance 4/4, quality 5/5) | `single/summary.json` |

Quality (the Q200 suite run at reasoning effort `low`, text budget 32,768 tokens), recorded in `quality/quality-cases.json`. These are this release's regression baseline, not a claim of absolute quality:

| Suite | Correct | Accuracy |
| --- | --- | --- |
| gsm8k | 79 of 80 | 98.75% |
| humaneval | 39 of 40 | 97.5% |
| ifeval | 35 of 40 | 87.5% |
| hard_reasoning | 20 of 20 (graded by an independent review agent) | 100% |
| BFCL multi-turn (20 cases) | 15 of 20 | 75.0% |

## 7. Limits of this release

- One GB10 node only; no tensor-parallel or two-node profile.
- The quality figures are one run on one node; a later release is held to them with a stated noise allowance (one case on the 20-case BFCL set), not to a fixed target.
- Audio, video and constrained JSON response formats are refused (section 3).
- The engine was qualified for this exact commit and binary; a different engine build is not covered by this sheet.
- A cold-boot time for this release is not stated in the sealed files; the readiness deadline is 600 s.

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/qwen38-27b-dflash2-exl3-4.00/v1/KNOWN_ISSUES.md`
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
