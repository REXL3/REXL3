# Pack sheet: `qwen38-flash-next-exl3-3.05`

Checked against release `v0.1.0`. This runtime was sealed on the release engine, commit `afc9b5dfb69d` (the freeze plus two DeepSeek fixes that do not touch this family); its sealed approval is `approve-qwen38-flash-next-exl3-3.05-2`, which supersedes approval 1 sealed on the freeze commit.

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `qwen38-flash-next-exl3-3.05` | `profile.json` |
| Versions | `v2` (the only version in this tree) | `runtimes/qwen3.8/qwen3.8-flash-next/` |
| Release tag | `v0.1.0` (one product release; there is no per-runtime tag) | `approval.json` |
| What it serves | Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`), EXL3 3.05 bits per weight; speculation uses the pack's own MTP head | `MODEL.json`, `model.lock.json`, `KNOWN_ISSUES.md` |
| Model repository and revision | `turboderp/Qwen3.8-Flash-Next-exl3` at `69e33439ae950f17bcbe95c98f117d80f759ab6d` | `model.lock.json` |
| Draft or second model | none | `model.lock.json` |
| Model licence | Qwen Community License 1.0 (model card metadata of the pack repository) | `model.lock.json` |
| Download size | 22 files, 79.3 GiB; free space needed: the remaining bytes plus a reserve | `model.lock.json` |

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen38-flash-next-single-262k` (default) | single node | 262,144 tokens | 2 | 8899 | `127.0.0.1` | at most 600 s (`readiness_timeout_s`, a ceiling) after model verification |

Source: `profile.json`. Served model name `default` (not a name derived from the model; `GET /v1/models` lists it). Arguments of the profile: `--slots 2`, `--max-model-len 262144`, `--spec mtp`, `--prefix-cache on`, `--prefix-cache-mib 22528`, `--queue-timeout 1500`, `--no-vision`; the profile also sets `REXL3_NGRAM_WARM=0` so the pack's n-gram table stays cold and the memory is left for full-length prefix entries (`KNOWN_ISSUES.md`). 262,144 tokens is the pack's native window (model card). There is no tensor-parallel or two-node profile.

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | yes | |
| Thinking (reasoning) | yes | Default when a request names no setting: **off** (the server's default; `GET /v1/models` reports `thinks: false`), although the pack's chat template would think. A request names `chat_template_kwargs.enable_thinking` and/or `reasoning_effort` (`low`, `medium`, `high`, `xhigh`); `preserve_thinking` is also accepted. No thinking-token budget is offered for this model. |
| Tools | yes | `tool_choice` `auto` and `none` only; at most 64 tools |
| Images | **no** | Refused with HTTP 400. The pack ships the vision tower's tensors, but the engine does not build this model's tower yet; the profile runs `--no-vision` under a documented exception that expires on 2027-01-31 |
| Prefix cache | yes | Automatic; pinned at 22,528 MiB (`--prefix-cache-mib 22528`): 86 entries and a reach of 488,960 tokens |
| Speculative decoding | yes | MTP head of the pack (`--spec mtp`) |
| Video, audio, `json_object`, `json_schema` | no | Refused by the server |

Source: `profile.json` `capabilities`, `KNOWN_ISSUES.md`, and the `GET /v1/models` and prefix-cache records in the qualification evidence (`evidence.tar.gz`). Everything not listed as supported is refused (user guide section 5.10).

## 4. Sampling and output defaults

The pack's `generation_config.json` (at the pinned revision) sets temperature 1.0, top_p 0.95 and top_k 20, and the server applies each value only when a request leaves that field out. The model card recommends the same three values for thinking mode. For non-thinking mode the card recommends temperature 0.7, top_p 0.8 and a presence penalty of 1.5; the server refuses non-zero presence, frequency and repetition penalties, so that third setting cannot be used here. Because a request that names nothing does not think, and a thinking request can spend its whole output budget on thinking, set `max_tokens` generously when you turn thinking on; the qualification run used a text budget of 65,536 tokens and 8,192 for the tool-calling suite, at temperature 0 for the text suites and 0.001 for the tool-calling suite. Sources: the pack's `generation_config.json` and model card, the v0.1.0 model release page, `quality/q200/summary.json` in the evidence.

## 5. Behaviour to expect

- A request that names no reasoning setting does not think. This differs from the model card, which says the model thinks by default. To think, send `enable_thinking: true` (the qualification sent it together with `reasoning_effort: xhigh`).
- Thinking can use the whole output budget. In the quality run (effort `xhigh`, 65,536-token budget) one of 180 text prompts ran to the budget without finishing and was counted incorrect (`quality/q200/summary.json`).
- Image input is refused; see section 3.
- Speculative decoding drafts only while one request is decoding; when both slots decode, generation runs without drafts. Replies are the same either way (`KNOWN_ISSUES.md`).
- The two slots share one 262,144-token KV pool. Requests are admitted in order; a request that does not fit waits for the one ahead, and is failed after 1,500 s in the queue (`--queue-timeout 1500`). A cold prompt of about 260,000 tokens took about 265 s to prefill in the qualification run (section 6).
- The prefix cache holds about 1.88 full-window documents. Two different documents of about 260,000 tokens do not fit together: the second evicts the first, and a follow-up on the evicted one is prefilled again in full (about 4 to 5 minutes at that size). Output is identical, only latency changes (`KNOWN_ISSUES.md`).
- On a few prompts that ask for parallel tool calls the model's two best choices are nearly tied, so one build may return one call where another returns two (`KNOWN_ISSUES.md`).
- The sealed `profile.json` still records `qualification.state: unverified`, and `KNOWN_ISSUES.md` describes the frozen configuration and lists as open some items the release gates have since run. The qualification of this release is the evidence asset `evidence.tar.gz` (section 6). Where the two differ, the release evidence is the record of what was measured on this commit.

## 6. Measured behaviour

All figures are from this release's own qualification, recorded in `evidence.tar.gz` (the release asset): one GB10 node, engine `0.1.0` at commit `afc9b5dfb69d`, 2 slots, window 262,144, vision off, prefix cache pinned at 22,528 MiB.

| Quantity | Value | Recorded in |
| --- | --- | --- |
| Generation speed, one stream (median of 3 runs) | 70.3 tok/s (70.2 to 70.4); floor 55 | `performance/performance-cases.json` |
| Prompt speed at 50% window fill, about 131,000 tokens (median of 3) | 1,047 tok/s (1,047 to 1,048); floor 860 | same |
| Prompt speed at full window fill, about 260,000 tokens (median of 3) | 1,006 tok/s (1,005 to 1,007); floor 860 | same |
| Exactness | 67 of 67 cases pass (drafted equals serial; chunked prefill equals whole prefill) | `exactness/exactness-cases.json` |
| Prefix cache | gate passes; a shared prefix is served from the cache and the reply is identical (83 checks, 0 failures, 25 cache hits counted) | `exactness/prefix-cache/check_prefix_cache.log` |
| Release gates | every gate in the qualification summary passed on this engine binary, fresh and none carried from approval 1 (doctor, lifecycle, capability, vision under the exception, context, exactness, prefix cache, performance, quality) | `single/summary.json` |

Quality (the Q200 suite at reasoning effort `xhigh`, text budget 65,536 tokens, tool-calling budget 8,192; temperature 0, and 0.001 for tool calling), recorded in `quality/quality-cases.json`. These are this release's regression baseline, not a claim of absolute quality:

| Suite | Correct | Accuracy |
| --- | --- | --- |
| gsm8k | 78 of 80 | 97.5% |
| humaneval | 40 of 40 | 100% |
| ifeval | 38 of 40 (one prompt ran to the token budget and counts as incorrect) | 95.0% |
| hard_reasoning | 20 of 20 (reviewed manually by a review agent independent of the model, as recorded in the evidence) | 100% |
| BFCL tool calling (20 hard cases) | 9 of 20 | 45.0% |

The quality answers are byte-identical to the 180 text answers of approval 1 on the freeze engine, and the tool-calling score is the same. No other figure from an earlier version or build is carried here.

## 7. Limits of this release

- One GB10 node only; two slots; window 262,144 tokens.
- No image input until the engine builds this model's vision tower; the exception that allows shipping without it expires on 2027-01-31 (`KNOWN_ISSUES.md`, "Vision").
- Audio, video and constrained JSON response formats are refused (section 3).
- No thinking-token budget parameter for this model.
- The nine gates are the qualification of this release; no separate long-run (soak) result is part of them.
- The quality figures are one run on one node; a later release is held to them with a stated noise allowance, not to a fixed target. The tool-calling suite scored 45%.
- The engine was qualified for this exact commit and binary; a different engine build is not covered by this sheet.
- A cold-boot time for this release is not stated in the sealed files; the readiness deadline is 600 s.

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/qwen3.8/qwen3.8-flash-next/v2/KNOWN_ISSUES.md`
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
