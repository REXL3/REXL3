# Pack sheet: `qwen38-flash-next-exl3-3.05`

Checked against version 3 (dev approval `approve-qwen38-flash-next-exl3-3.05-3`, source commit `75031112ba12`, engine release `engine-v0.1.1-75031112`, archive `qwen38-flash-next-exl3-3.05-3.tar.gz`). Version 2 (release `v0.1.0`, engine commit `afc9b5dfb69d`, approval `approve-qwen38-flash-next-exl3-3.05-2`) stays in the tree unchanged. Version 3 keeps version 2's default choice `qwen38-flash-next-single-262k` (same argv, requalified on the v0.1.1 engine) and adds an opt-in drafting choice, `qwen38-flash-next-single-262k-draft`, with its own runtime id `qwen38-flash-next-dflash2-exl3-3.05`.

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `qwen38-flash-next-exl3-3.05`; the drafting choice reports `qwen38-flash-next-dflash2-exl3-3.05` | `profile.json` |
| Versions | `v2` (v0.1.0), `v3` (latest) | `runtimes/qwen3.8/qwen3.8-flash-next/` |
| Release tag | version 2 shipped in `v0.1.0`; version 3 ships in `v0.1.1` (there is no per-runtime tag) | `releases/v0.1.1/manifest.json` |
| Release assets (version 3) | `qwen38-flash-next-exl3-3.05-v3-approval.json`, `qwen38-flash-next-exl3-3.05-v3-bundle.tar.gz`; evidence `qwen38-flash-next-exl3-3.05-v3-evidence.tar.gz` | the release page |
| What it serves | Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`), EXL3 3.05 bits per weight; the default choice speculates with the pack's own MTP head, the drafting choice with a DFlash2 draft pack | `MODEL.json`, `model.lock.json`, `KNOWN_ISSUES.md` |
| Model repository and revision | `turboderp/Qwen3.8-Flash-Next-exl3` at `69e33439ae950f17bcbe95c98f117d80f759ab6d` | `model.lock.json` |
| Draft or second model | drafting choice only: `r0b0tlab/Qwen3.8-Flash-Next-DFlash2-EXL3-4.00bpw` at `846786415744bf40a70f6c274d1ff3a7c1fe64ee` (a 4.00 bpw quantization of `incoai/Qwen3.8-Flash-Next-DFlash2`); the default choice downloads nothing new | `model.lock.json`, `KNOWN_ISSUES.md` |
| Model licence | Qwen Community License 1.0 for the target pack; Apache-2.0 for the draft pack (model card metadata of each pack repository) | `model.lock.json` |
| Download size | 22 target files, 79.3 GiB; the drafting choice adds 3 draft files, 0.48 GiB; free space needed: the remaining bytes plus a reserve | `model.lock.json` |

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen38-flash-next-single-262k` (default) | single node | 262,144 tokens | 2 | 8899 | `127.0.0.1` | at most 600 s (`readiness_timeout_s`, a ceiling) after model verification |
| `qwen38-flash-next-single-262k-draft` (opt-in) | single node | 262,144 tokens | 2 | 8899 | `127.0.0.1` | at most 600 s (`readiness_timeout_s`, a ceiling) after model verification |

Source: `profile.json`. Served model name `default` (not a name derived from the model; `GET /v1/models` lists it). Arguments of the profile: `--slots 2`, `--max-model-len 262144`, `--spec mtp`, `--prefix-cache on`, `--prefix-cache-mib 22528`, `--queue-timeout 1500`, `--no-vision`; the profile also sets `REXL3_NGRAM_WARM=0` so the pack's n-gram table stays cold and the memory is left for full-length prefix entries (`KNOWN_ISSUES.md`). 262,144 tokens is the pack's native window (model card). The drafting choice runs the same line with `--spec auto --draft @DRAFT_DIR@` in place of `--spec mtp`; every other flag, the environment and the capabilities are the default choice's. It has no `model_dir` site field, so it is served from the packs the launcher manages (`KNOWN_ISSUES.md`, "Opt-in drafting choice"). There is no tensor-parallel or two-node profile.

Both profiles are in the same bundle. Download it with the runtime id `qwen38-flash-next-exl3-3.05` (the asset prefix) and select the drafting choice with `--profile qwen38-flash-next-single-262k-draft`; the id `qwen38-flash-next-dflash2-exl3-3.05` is what that profile reports, not an asset name.

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | yes | |
| Thinking (reasoning) | yes | Default when a request names no setting: **off** (the server's default; `GET /v1/models` reports `thinks: false`), although the pack's chat template would think. A request names `chat_template_kwargs.enable_thinking` and/or `reasoning_effort` (`low`, `medium`, `high`, `xhigh`); `preserve_thinking` is also accepted. No thinking-token budget is offered for this model. |
| Tools | yes | `tool_choice` `auto` and `none` only; at most 64 tools |
| Images | **no** | Refused with HTTP 400. The pack ships the vision tower's tensors, but the engine does not build this model's tower yet; the profile runs `--no-vision` under a documented exception that expires on 2027-01-31 |
| Prefix cache | yes | Automatic; pinned at 22,528 MiB (`--prefix-cache-mib 22528`): 86 entries and a reach of 488,960 tokens |
| Speculative decoding | yes | Default choice: MTP head of the pack (`--spec mtp`). Drafting choice: DFlash2 draft (`--spec auto --draft`); with a draft the MTP head is not loaded |
| Video, audio, `json_object`, `json_schema` | no | Refused by the server |

Source: `profile.json` `capabilities`, `KNOWN_ISSUES.md`, and the `GET /v1/models` and prefix-cache records in the qualification evidence (`evidence.tar.gz`). Everything not listed as supported is refused (user guide section 5.10).

## 4. Sampling and output defaults

The pack's `generation_config.json` (at the pinned revision) sets temperature 1.0, top_p 0.95 and top_k 20, and the server applies each value only when a request leaves that field out. The model card recommends the same three values for thinking mode. For non-thinking mode the card recommends temperature 0.7, top_p 0.8 and a presence penalty of 1.5; the server refuses non-zero presence, frequency and repetition penalties, so that third setting cannot be used here. Because a request that names nothing does not think, and a thinking request can spend its whole output budget on thinking, set `max_tokens` generously when you turn thinking on; the quality runs used a text budget of 65,536 tokens and 8,192 for the tool-calling suite, at temperature 0 for the text suites and 0.001 for the tool-calling suite. Sources: the pack's `generation_config.json` and model card, the v0.1.0 model release page, `quality/q200/summary.json` in the evidence.

## 5. Behaviour to expect

- A request that names no reasoning setting does not think. This differs from the model card, which says the model thinks by default. To think, send `enable_thinking: true` (the qualification sent it together with `reasoning_effort: xhigh`).
- Thinking can use the whole output budget. In the quality run (effort `xhigh`, 65,536-token budget) one of 180 text prompts ran to the budget without finishing and was counted incorrect (`quality/q200/summary.json`).
- Image input is refused; see section 3.
- Speculative decoding drafts only while one request is decoding; when both slots decode, generation runs without drafts. Replies are the same either way (`KNOWN_ISSUES.md`).
- The two slots share one 262,144-token KV pool. Requests are admitted in order; a request that does not fit waits for the one ahead, and is failed after 1,500 s in the queue (`--queue-timeout 1500`). A cold prompt of about 260,000 tokens took about 265 s to prefill in the qualification run (section 6).
- The prefix cache holds about 1.88 full-window documents. Two different documents of about 260,000 tokens do not fit together: the second evicts the first, and a follow-up on the evicted one is prefilled again in full (about 4 to 5 minutes at that size). Output is identical, only latency changes (`KNOWN_ISSUES.md`).
- The drafting choice returns the default choice's outputs token for token (drafted equals serial and cached equals cold were measured at 32K and at the full window). Its gain is in concurrency and structured output: on one GB10 at 2 slots the dev measurement recorded about 3x the aggregate throughput for two to four concurrent streams and 5 to 28% faster single-stream code and structured output, and about 8% slower single-stream long English prose (59.7 vs 65.0 tok/s on a 2,048-token greedy essay). Keep the default choice for that workload (`KNOWN_ISSUES.md`, "Opt-in drafting choice").
- On a few prompts that ask for parallel tool calls the model's two best choices are nearly tied, so one build may return one call where another returns two (`KNOWN_ISSUES.md`).
- The sealed `profile.json` still records `qualification.state: unverified`, and `KNOWN_ISSUES.md` describes the frozen configuration and lists as open some items the release gates have since run. The qualification of version 3 is its evidence asset `evidence.tar.gz` (section 6). Where the two differ, the release evidence is the record of what was measured on this commit.

## 6. Measured behaviour

All figures are from version 3's qualification, recorded in the release asset `qwen38-flash-next-exl3-3.05-v3-evidence.tar.gz` (named `evidence.tar.gz` inside the sealed approval): one GB10 node, engine `0.1.1` at commit `75031112ba12` (binary `rexl3-serve` sha256 `4fc2cb44…5fe5`), 2 slots, window 262,144, vision off, prefix cache pinned at 22,528 MiB.

Default choice `qwen38-flash-next-single-262k`: seven gates ran on the v0.1.1 binary; `context` and `quality` are carried from version 2's sealed qualification (engine `afc9b5dfb69d`, binary `40e23e71…`), as recorded in the receipt `v011/composed/summary.json`. Drafting choice `qwen38-flash-next-single-262k-draft`: a fresh full gate chain on the v0.1.1 binary, nothing carried, receipt `draft/qualification/summary.json`.

| Quantity | Default choice | Drafting choice | Recorded in |
| --- | --- | --- | --- |
| Generation speed, one stream (median of 3 runs, each 5 streamed replies, short context) | 69.5 tok/s (69.5 to 69.8); floor 55 | 63.0 tok/s (62.8 to 63.2); floor 55 | `v011/` and `draft/performance/performance-cases.json` |
| Prompt speed at 50% window fill, about 131,000 tokens (median of 3, cold) | 1,047 tok/s (1,047 to 1,053); floor 860 | 1,027 tok/s (1,025 to 1,062); floor 860 | same |
| Prompt speed at full window fill, about 260,000 tokens (median of 3, cold) | 996 tok/s (992 to 1,005); floor 860 | 979 tok/s (978 to 999); floor 860 | same |
| Draft acceptance during the baseline runs | 0.72 to 0.73 (MTP) | 0.63 (DFlash2) | `performance/runs/run-0N-RESULT.json` |
| Lowest free memory during the baseline runs | 31.04 to 31.07 GiB, no swap | 31.65 to 31.74 GiB, no swap | same |
| Boot to ready | 22.2 s first boot, 40.7 s restart | 31.4 s first boot, 25.2 s restart | `lifecycle/lifecycle-cases.json` |
| Exactness | 68 of 68 cases pass | 70 of 70 cases pass | `exactness/exactness-cases.json` |
| Prefix cache | gate passes (83 checks, 0 failures, 25 cache hits counted) | the same | `exactness/prefix-cache/check_prefix_cache.log` |
| Release gates | all PASS: fresh — doctor 4/4, lifecycle 10/10, capability 5/5, boundary 5/5, vision 2/2 (under the exception), exactness 68/68, prefix cache 1/1, performance 4/4, performance smoke 3/3; carried — context 8/8, quality 5/5 | all PASS, fresh: doctor 4/4, lifecycle 10/10, capability 5/5, boundary 5/5, vision 2/2 (under the exception), context 8/8, exactness 70/70, prefix cache 1/1, performance 4/4, performance smoke 3/3, quality 5/5 | `v011/composed/summary.json`, `draft/qualification/summary.json` |

Quality (the Q200 suite at reasoning effort `xhigh`, text budget 65,536 tokens, tool-calling budget 8,192; temperature 0, and 0.001 for tool calling; one client worker). The default choice's figures are carried from version 2; the drafting choice ran its own Q200 on the v0.1.1 binary and scored the same counts. These are this pack's regression baseline, not a claim of absolute quality:

| Suite | Correct | Accuracy |
| --- | --- | --- |
| gsm8k | 78 of 80 | 97.5% |
| humaneval | 40 of 40 | 100% |
| ifeval | 38 of 40 (one prompt ran to the token budget and counts as incorrect) | 95.0% |
| hard_reasoning | 20 of 20 (reviewed manually by a review agent independent of the model, as recorded in the evidence) | 100% |
| BFCL tool calling (20 hard cases) | 9 of 20 | 45.0% |

Only the default choice's `context` and `quality` are carried; every other figure above was measured on the v0.1.1 binary.

## 7. Limits of this release

- One GB10 node only; two slots; window 262,144 tokens.
- No image input until the engine builds this model's vision tower; the exception that allows shipping without it expires on 2027-01-31 (`KNOWN_ISSUES.md`, "Vision").
- Audio, video and constrained JSON response formats are refused (section 3).
- No thinking-token budget parameter for this model.
- The gates above are the qualification of version 3; no separate long-run (soak) result is part of them.
- The quality figures are one run on one node; a later release is held to them with a stated noise allowance, not to a fixed target. The tool-calling suite scored 45%.
- Version 3 was qualified for engine commit `75031112ba12` and its release binary; a different engine build is not covered by this sheet. The default choice's `context` and `quality` come from version 2's build under the carried-gate rule (CONTRIBUTING section 3).
- The readiness deadline is 600 s; the measured boots are in section 6.
- Install on an updated host: this recipe pins the NVIDIA driver packages `libnvidia-compute-580` and `nvidia-utils-580` at exactly `580.173.02-0ubuntu0.24.04.1` (`dependencies.lock.json`). On an Ubuntu 24.04 arm64 host whose packages have moved past that build, `./launch.sh install-deps` and `./launch.sh doctor` refuse with exit 20. The release notes give a single-node workaround; it skips the launcher's checks and is outside the release's qualification evidence. See "Known issues" in the [v0.1.1 release notes](../../releases/v0.1.1/NOTES.md).

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/qwen3.8/qwen3.8-flash-next/v3/KNOWN_ISSUES.md` (version 2's plus the section "Opt-in drafting choice")
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
