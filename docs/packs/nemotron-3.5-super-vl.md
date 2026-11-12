# Pack sheet: `nemotron35-super-vl-sage-3.85`

Checked against release `v0.1.0` (source commit `799fa750f8eb`, archive `nemotron35-super-vl-sage-3.85-1.tar.gz`).

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `nemotron35-super-vl-sage-3.85` | `profile.json` |
| Versions | `v1` (latest) | `runtimes/nemotron-3.5/nemotron-3.5-super-vl/` |
| Release tag | `v0.1.0` (one product release; there is no per-runtime tag) | `approval.json` |
| What it serves | Nemotron-3.5-Super-VL (`NemotronH_Omni_Reasoning_V3`), EXL3 3.85 bits per weight; speculation uses the MTP head that ships in the pack | `MODEL.json`, `model.lock.json`, `KNOWN_ISSUES.md` |
| Model repository and revision | `vcruz305/Nemotron-Super-VL-EXL3-3.85bpw` at `3963815246eda892a4c3b88e18d0d1c39288c8e0` | `model.lock.json` |
| Draft or second model | none | `model.lock.json` |
| Model licence | Recorded in the lock as "NVIDIA Open Model License", taken from the base model card metadata (`nvidia/nemotron-3.5-super-vl`); the pack repository's own card metadata says only `license: other`. Read the licence on the base model card before use | `model.lock.json` |
| Download size | 23 files, 55.8 GiB; free space needed: the remaining bytes plus a reserve | `model.lock.json` |

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `nemotron35-super-vl-single-1m` (default) | single node | 1,048,576 tokens | 16 | 9100 | `127.0.0.1` | at most 600 s (`readiness_timeout_s`, a ceiling) after model verification |

Source: `profile.json`. Served model name `nemotron-3.5-super-vl`. Arguments of the profile: `--slots 16`, `--max-model-len 1048576`, `--spec auto`, `--prefix-cache on` (no size pin), `--request-timeout 7200`, `--queue-timeout 6900`. The pack's own configuration states a 1,048,576-token position limit, the same as the window. There is no tensor-parallel or two-node profile.

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | yes | |
| Thinking (reasoning) | yes | Default when a request names no setting: **off** (the server's default; `GET /v1/models` reports `thinks: false`), although the pack's chat template would think. Effort is set with `reasoning_effort` (`none`, `low`, `medium`, `high`, `xhigh`), at the top level or in `chat_template_kwargs`; `enable_thinking` also works. A thinking budget is accepted as `max_reasoning_tokens`; `low` carries a default budget of 1,024 tokens. |
| Tools | yes | `tool_choice` `auto` and `none` only; at most 64 tools |
| Images | yes | Inline data URI only. At most 3,328 tokens per image (a 6000 x 4000 image becomes 3,290 tokens) |
| Prefix cache | yes | Automatic, with no size pin; the qualification boot planned 8 entries. The reach in tokens is not stated |
| Speculative decoding | yes | Automatic (`--spec auto`, the pack's MTP head) |
| Video | not a declared capability | Video files (`data:video/mp4`, `webm`, `gif`) are refused; video sent as a list of image frames (at most 32 frames or 4,096 tokens) works but is not covered by the v0.1.0 qualification |
| Audio, `json_object`, `json_schema` | no | Refused by the server |

Source: `profile.json` `capabilities`, `KNOWN_ISSUES.md`, and the `GET /v1/models` and boot records in the qualification evidence (`evidence.tar.gz`). Everything not listed as supported is refused (user guide section 5.10).

## 4. Sampling and output defaults

The pack's `generation_config.json` (at the pinned revision) sets temperature 1.0 and top_p 0.95; it states no `top_k`. The server's fallback `top_k` of 20 applies when a request leaves it out, and that value is not sourced from the pack or the base model. The server applies each value only when a request leaves that field out, and refuses non-zero presence, frequency and repetition penalties. Because a request that names nothing does not think, and a thinking request can spend its whole output budget on thinking, set `max_tokens` generously when you turn thinking on; the qualification run used a text budget of 65,536 tokens and 8,192 for the tool-calling suite, at temperature 0 for the text suites and 0.001 for the tool-calling suite. Sources: the pack's `generation_config.json`, the v0.1.0 model release page, `quality/q200/summary.json` in the evidence.

## 5. Behaviour to expect

- A request that names no reasoning setting does not think. The pack's chat template would think by default; this server does not, unless the request asks (the qualification sent `enable_thinking: true` with `reasoning_effort: xhigh`).
- Thinking can use the whole output budget. In the quality run (effort `xhigh`, 65,536-token budget) 8 of 40 IFEval prompts ran to the budget without finishing and were counted incorrect (`quality/q200/summary.json`).
- A cold request that fills the window is slow: a prompt of 1,046,622 tokens took about 1,777 s (about 30 minutes) to prefill in the qualification run, which is why the profile sets a 7,200 s request timeout. The 16 slots share one KV pool sized at the window; a request that does not fit waits in order and is failed after 6,900 s in the queue (`KNOWN_ISSUES.md`).
- Image input is limited to 3,328 tokens per image. The refusal above that limit was not exercised in the v0.1.0 qualification (`KNOWN_ISSUES.md`).
- Video files are refused; frame lists are accepted but not covered by the qualification (section 3).
- A 24-hour soak was not run for v0.1.0 (`KNOWN_ISSUES.md`).
- Free memory at the full window: the lowest reading during the baseline runs was about 27 GiB, with no swap (section 6).
- The sealed `profile.json` still records `qualification.state: unverified`, and `KNOWN_ISSUES.md` describes the frozen configuration and lists as open some items the release gates have since run. The qualification of this release is the evidence asset `evidence.tar.gz` (section 6). Where the two differ, the release evidence is the record of what was measured on this commit.

## 6. Measured behaviour

All figures are from this release's own qualification, recorded in `evidence.tar.gz` (the release asset): one GB10 node, engine `0.1.0` at commit `799fa750f8eb`, 16 slots, window 1,048,576, vision on.

| Quantity | Value | Recorded in |
| --- | --- | --- |
| Generation speed, one stream (median of 3 runs) | 50.8 tok/s (50.3 to 50.8); floor 42 | `performance/performance-cases.json` |
| Prompt speed at 50% window fill, about 524,000 tokens (median of 3) | 879 tok/s (878 to 883); floor 683 | same |
| Prompt speed at full window fill, about 1,047,000 tokens (median of 3) | 589 tok/s (588 to 589); floor 471 | same |
| Lowest free memory during the baseline runs | 26.93 to 27.04 GiB, no swap | `performance/runs/run-0N-RESULT.json` |
| Context gate | 16 of 16 rows pass with prompts of 130,048, 261,120, 523,264 and 1,047,552 tokens | `single/summary.json` |
| Exactness | 67 of 67 cases pass (drafted equals serial; chunked prefill equals whole prefill) | `exactness/exactness-cases.json` |
| Prefix cache | gate passes; a shared prefix is served from the cache and the reply is identical (84 checks, 0 failures) | `exactness/prefix-cache/check_prefix_cache.log` |
| Release gates | all nine PASS in one composed pass (doctor 4/4, lifecycle 10/10, capability 5/5, vision 3/3, context 16/16, exactness 67/67, prefix cache 1/1, performance 4/4, quality 5/5) | `single/summary.json` |

Quality (the Q200 suite at reasoning effort `xhigh`, text budget 65,536 tokens, tool-calling budget 8,192; temperature 0, and 0.001 for tool calling), recorded in `quality/quality-cases.json`. These are this release's regression baseline, not a claim of absolute quality:

| Suite | Correct | Accuracy |
| --- | --- | --- |
| gsm8k | 76 of 80 | 95.0% |
| humaneval | 40 of 40 | 100% |
| ifeval | 28 of 40 (8 prompts ran to the token budget and count as incorrect) | 70.0% |
| hard_reasoning | 20 of 20 (reviewed manually by a review agent independent of the model, as recorded in the evidence) | 100% |
| BFCL tool calling (20 hard cases) | 15 of 20 | 75.0% |

No figure from an earlier version or an earlier build is carried here.

## 7. Limits of this release

- One GB10 node only; no tensor-parallel or two-node profile.
- Images up to 3,328 tokens each; video files and audio are refused; video as image frames is not covered by the qualification.
- Audio and constrained JSON response formats are refused (section 3).
- The nine gates are the qualification of this release; no 24-hour soak was run for v0.1.0, and no other long-run result is part of them.
- The quality figures are one run on one node; a later release is held to them with a stated noise allowance, not to a fixed target. The IFEval score is lowered by prompts that thinking took to the budget (section 5).
- The engine was qualified for this exact commit and binary; a different engine build is not covered by this sheet.
- A cold-boot time for this release is not stated in the sealed files; the readiness deadline is 600 s.

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/nemotron-3.5/nemotron-3.5-super-vl/v1/KNOWN_ISSUES.md`
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
