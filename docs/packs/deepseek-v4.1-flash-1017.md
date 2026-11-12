# Pack sheet: `dsv41-flash-1017-sage-1.55`

Checked against release `v0.1.0`. This runtime was sealed on the release engine, commit `afc9b5dfb69d` (the v0.1.0 freeze plus two DeepSeek fixes); its sealed approval is `approve-dsv41-flash-1017-sage-1.55-1`. The approval covers the single-node profile only (section 2).

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `dsv41-flash-1017-sage-1.55` | `profile.json` |
| Versions | `v1` (the only version in this tree) | `runtimes/deepseek-v4.1/deepseek-v4.1-flash-1017/` |
| Release tag | `v0.1.0` (one product release; there is no per-runtime tag) | `approval.json` |
| What it serves | DeepSeek-V4.1-Flash-1017 (`DeepseekV41ForCausalLM`), EXL3 1.55 bits per weight | `MODEL.json`, `model.lock.json` |
| Model repository and revision | `vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw` at `cba79c67582950ea44a4a449b3bb85b800748a35` | `model.lock.json` |
| Draft or second model | none | `model.lock.json` |
| Model licence | see the upstream model card and the LICENSE in the pack | `model.lock.json` |
| Download size | 24 files, 298.3 GiB; free space needed: the remaining bytes plus a reserve | `model.lock.json` |

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `dsv41-single-262k` (default) | single node | 262,144 tokens | 1 | 8940 | `127.0.0.1` | at most 900 s (`readiness_timeout_s`, a ceiling) after model verification |

Source: `profile.json`. The profile runs `--kv-dtype q4`, `--spec auto`, `--prefix-cache on`, `--no-vision` and `--no-video`. The source tree also packages a two-node profile, `dsv41-tp2-1m`, which is **not part of this release**: the sealed approval names `dsv41-single-262k` only, nothing in v0.1.0 qualified the two-node profile, and it ships separately in v0.1.1. Do not launch it from this release.

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | yes | |
| Thinking (reasoning) | yes | Default when a request names no setting: **off**. Effort names: `none` and `off` turn thinking off; `minimal`, `low`, `medium`, `high`, `max` and `xhigh` turn it on (any other name is refused with HTTP 400). With thinking on and no effort given, the effort is `high` (75) (`KNOWN_ISSUES.md`) |
| Tools | yes | `tool_choice` `auto` and `none` only; at most 64 tools |
| Images | **no** | Refused. The profile carries `--no-vision` and `capabilities.images` is false, under a temporary owner exception that expires on 2026-12-31 (answers about colour and shape on this quantized pack are unreliable: 3 of 10 solid colours correct) |
| Prefix cache | yes | Automatic (the profile pins no size) |
| Speculative decoding | yes | Automatic (`--spec auto`) |
| Video, audio, `json_object`, `json_schema` | no | Refused by the server |

Source: `profile.json` `capabilities` and `KNOWN_ISSUES.md`. Everything not listed as supported is refused (user guide section 5.10).

## 4. Sampling and output defaults

A request that leaves out `temperature` or `top_p` gets temperature 1.0 and top_p 0.95 from the pack's `generation_config.json` (the file the model lock pins), and `top_k` 20, the server's fallback; neither the pack nor the upstream model sets `top_k`. Requests can set these fields explicitly. Source: `KNOWN_ISSUES.md` "Reasoning-effort names and sampling defaults".

## 5. Behaviour to expect

- **Greedy decoding can loop.** At temperature 0 the model sometimes repeats a short phrase in its reasoning until `max_tokens`. In this release's quality run (greedy, 8,192-token cap, thinking off) 7 of 180 text prompts ran to the cap and were counted incorrect, four of them in the hard-reasoning set (section 6). For long reasoning use the sampling defaults rather than temperature 0, and set `max_tokens` to bound a reply (`KNOWN_ISSUES.md`).
- Image input is refused by design in this release (section 3).
- Free memory is tight at this window: the lowest reading during the quality run was about 0.7 GiB of MemAvailable on the test node (the gate's guard floor was 0.5 GiB). Keep other work off the node.
- Model files are verified in full on every start and the model tree is read-only (`KNOWN_ISSUES.md` "Existing model copies").
- The sealed `KNOWN_ISSUES.md` still describes a frozen configuration and lists qualification as pending; the release evidence (`evidence.tar.gz`) is the record of what was measured on this commit (section 6).

## 6. Measured behaviour

All figures are from this release's own qualification, recorded in `evidence.tar.gz` (the release asset): one GB10 node, engine `0.1.0` at commit `afc9b5dfb69d`, 1 slot, window 262,144, vision off. Every gate ran fresh on this binary; none was carried from an earlier build.

| Quantity | Value | Recorded in |
| --- | --- | --- |
| Generation speed, one stream (median of 3 runs) | 29.6 tok/s (29.6 to 29.7); floor 18.0 | `performance/performance-cases.json` |
| Prompt speed at 50% window fill, about 131,000 tokens (median of 3) | 1,274 tok/s (1,272 to 1,283); floor 700 | same |
| Prompt speed at full window fill, about 262,000 tokens (median of 3) | 1,034 tok/s (1,030 to 1,037); floor 400 | same |
| Exactness | 67 of 67 cases pass (drafted equals serial 24/24; chunked prefill equals whole prefill 4/4; prefix-cache check 84 checks, 0 failures, 34 hits) | `exactness/exactness-cases.json` |
| Release gates | every gate in the qualification summary passed (doctor 4/4, lifecycle 10/10, capability, vision under the exception, context at 131,072 and 262,144 tokens, exactness, prefix cache, performance) | `single/summary.json` |

Quality (the Q200 suite at reasoning effort off, greedy, 8,192-token reply cap; a reply cut by the cap counts as incorrect), recorded in `quality/quality-cases.json`. These are this release's regression baseline, not a claim of absolute quality:

| Suite | Correct | Accuracy | Bar the run used |
| --- | --- | --- | --- |
| gsm8k | 78 of 80 | 97.5% | 96.25% |
| humaneval | 40 of 40 | 100% | 95% |
| ifeval | 35 of 40 | 87.5% | 82.5% |
| hard_reasoning | 16 of 20 (judged manually by a review agent independent of the model; four answers were cut in repetition loops) | 80.0% | 80% |
| BFCL tool calling (20 hard cases) | 13 of 20 | 65.0% | 55% |

The hard_reasoning result is one case under the regression bar set for the release (baseline minus two cases, 85%, 17 of 20). The maintainer accepted it for v0.1.0 and the run was recorded with the accepted bar of 80%.

No figure from an earlier version, build or pack is carried here.

## 7. Limits of this release

- One GB10 node only; one slot; window 262,144 tokens. The two-node profile is not part of this release (section 2).
- Image input, video, audio and constrained JSON response formats are refused (section 3).
- The gates are the qualification of this release; no 24-hour soak was run.
- The quality figures are one run on one node. The hard_reasoning result is below its regression bar by one case and is accepted by the maintainer (section 6); greedy loops are a known behaviour of the model (section 5).
- The engine was qualified for this exact commit and binary; a different engine build is not covered by this sheet.

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/deepseek-v4.1/deepseek-v4.1-flash-1017/v1/KNOWN_ISSUES.md`
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
