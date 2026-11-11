# Pack sheet: `minicpmv47-35b-a3b-rexl3-sage-6.08`

Checked against model version `v1` (sealed approval `approve-minicpmv47-35b-a3b-rexl3-sage-6.08-1`, source commit `75031112ba12`, engine `0.1.1`, bundle archive `minicpmv47-35b-a3b-rexl3-sage-6.08-1.tar.gz`). This version is not yet part of a product release; the release tag is set when a release manifest names it.

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `minicpmv47-35b-a3b-rexl3-sage-6.08` | `profile.json` |
| Versions | `v1` (latest, and the only version in this tree) | `runtimes/minicpm-v-4.7/minicpm-v-4.7-35b-a3b/LATEST` |
| Release tag | not stated (no product release contains `v1` yet) | `releases/` |
| What it serves | MiniCPM-V-4.7-35B-A3B (`MiniCPMV4_7ForConditionalGeneration`), EXL3 6.08 bits per weight quantized with SAGE, with a DSpark draft pack (EXL3 6.00 bits per weight) for speculative decoding. SAGE is our own mixed-precision quantization method for EXL3. The method is not published. | `MODEL.json`, `model.lock.json`, `KNOWN_ISSUES.md` |
| Model repository and revision | `r0b0tlab/MiniCPM-V-4.7-35B-A3B-EXL3-6.08bpw-REXL3-SAGE` at `25b7aa3361dd8e319c7503ebc002947ceb1fd290` | `model.lock.json` |
| Draft or second model | `r0b0tlab/MiniCPM-V-4.7-35B-A3B-DSpark-EXL3-6.00bpw` at `e1ea1a3045277ea69a108b4d477983affae0f3c1` | `model.lock.json` |
| Model licence | Apache-2.0 (model card metadata of each pack repository) | `model.lock.json` |
| Download size | 13 target files (26.4 GiB) and 3 draft files (0.4 GiB); free space needed: the remaining bytes plus a reserve | `model.lock.json` |

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `minicpmv47-single-262k` (default) | single node | 262,144 tokens | 4 | 8889 | `127.0.0.1` | at most 600 s (`readiness_timeout_s`, a ceiling) after model verification |

Source: `profile.json`. Served model name `minicpm-v-4.7-35b-a3b-exl3`. Arguments of the profile: `--slots 4`, `--max-model-len 262144`, `--spec auto` with `--draft` (the DSpark pack), `--prefix-cache on`, `--prefix-cache-mib 32768`. Queue and request timeouts stay at the server defaults (300 s and 1,800 s, `KNOWN_ISSUES.md`). There is no tensor-parallel or two-node choice. The launcher downloads both packs itself; pointing the profile at an existing local copy through `--site model_dir` is not supported for this two-pack lock (`KNOWN_ISSUES.md`).

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | yes | |
| Thinking (reasoning) | yes | Default when a request names no setting: **off** (the server's default; `GET /v1/models` reports `thinks: false`), although the pack's chat template would think. Turn it on with `chat_template_kwargs: {"enable_thinking": true}`. The chat template has no effort variable; effort levels are not stated for this pack. |
| Tools | yes | `tool_choice` `auto` and `none` only (`required` is refused with HTTP 400 in the qualification run) |
| Images | yes | Inline `data:` URI only (PNG or JPEG). Up to 9 slices of 448 pixels plus a thumbnail, at most 4,096 tokens an image |
| Prefix cache | yes | Automatic; pinned at 32,768 MiB, which holds two full-window entries. An exact repeat is served from the prompt length less 64 tokens; prompts of 128 tokens or fewer are not cached |
| Speculative decoding | yes | Automatic (`--spec auto` with the DSpark draft; the checkpoint has no MTP head) |
| Video | not a declared capability | Video containers (`data:video/mp4`) are refused; frames sent as images (at most 32 frames, 4,096 video tokens) are not covered by the qualification |
| Audio, `json_object`, `json_schema` | no | Refused by the server |

Source: `profile.json` `capabilities`, `KNOWN_ISSUES.md`, and the `GET /v1/models` and tool-matrix records in the qualification evidence (`evidence.tar.gz`). Everything not listed as supported is refused (user guide section 5.10).

## 4. Sampling and output defaults

The pack's `generation_config.json` sets temperature 1.0, top_p 0.95 and top_k 20 (a copy of the base checkpoint's), and the server reads all three from it; each applies only when a request leaves that field out (`KNOWN_ISSUES.md`). A request that names nothing does not think. When you turn thinking on, leave room in `max_tokens` for the reasoning: the quality run used 8,192 tokens and no text row was cut by that budget (`quality/q200/summary.json`).

## 5. Behaviour to expect

- A request that names no reasoning setting does not think; send `enable_thinking: true` in `chat_template_kwargs` for reasoning (`KNOWN_ISSUES.md`, sampling defaults and thinking).
- The 4 slots share one 262,144-token KV pool and admission is in order. A request queued behind cold full-window prompts can wait: a full-window prefill took about 138 s in the qualification run (section 6), and the queue and request timeouts are the server defaults (300 s and 1,800 s). Full-window requests sent concurrently are not part of the qualification (`KNOWN_ISSUES.md`).
- Prefix-cache cuts never fall inside an image; a growing conversation is cached from its second turn (`KNOWN_ISSUES.md`, prefix cache).
- Images are limited to 4,096 tokens each; video containers and audio are refused (section 3).
- Free memory with the full window in use: the lowest reading during the baseline runs was 44.4 GiB, with no swap (section 6).
- The dependency set is that of the Qwen3.8-27B runtime and has not been rehearsed on a clean host for this runtime (`KNOWN_ISSUES.md`, dependencies).
- The sealed `profile.json` still records `qualification.state: unverified`, and `KNOWN_ISSUES.md` describes the frozen configuration before qualification. The qualification of this version is the evidence asset `evidence.tar.gz` (section 6); where the two differ, the evidence is the record of what was measured on this commit.

## 6. Measured behaviour

All figures are from this version's own qualification, recorded in `evidence.tar.gz` (the dev release asset of `approve-minicpmv47-35b-a3b-rexl3-sage-6.08-1`): one GB10 node, engine `0.1.1` at commit `75031112ba12`, profile `minicpmv47-single-262k` (4 slots, window 262,144, DSpark draft on, vision on).

Realistic workload first. During the Q200 text run (180 requests, at most two at a time, thinking on, greedy, `max_tokens` 8,192; client on the same host), one request generated a median of 113.3 tok/s, measured as completion tokens over the client's wall time for that request including its prompt (152,463 completion tokens in 1,410 s of request time; `quality/q200/summary.json`, `text180.e2e_throughput`).

Baseline gate (three runs; each figure is the median of the three run results; `performance/performance-cases.json`, per-run detail in `performance/runs/run-0N-RESULT.json`):

| Quantity | Method | Value |
| --- | --- | --- |
| Generation speed, one stream | 5 streamed replies of 256 tokens per run, speculative decoding on; per-run median of the 5 | 112.2 tok/s (111.9 to 112.2); floor 90 |
| Prompt speed at 50% window fill | one cold prompt of 131,097 tokens (a needle-in-a-haystack fill) per run | 2,706 tok/s (2,700 to 2,726); floor 2,188 |
| Prompt speed at full window fill | one cold prompt of 260,189 tokens (needle fill plus a tool call) per run; about 138 s | 1,884 tok/s (1,881 to 1,888); floor 1,512 |
| Lowest free memory during the baseline runs | minimum `MemAvailable` recorded in each run | 44.38 to 44.45 GiB, no swap |

Other gates (`single/summary.json`, every gate PASS in one composed pass):

| Gate | Result | Recorded in |
| --- | --- | --- |
| Doctor | 4 of 4 | `doctor/doctor-cases.json` |
| Lifecycle | 10 of 10 (boot, serve, stop and leak checks, then a restart in a new process with the same reply; boot took 11.1 s and 19.4 s) | `lifecycle/lifecycle-cases.json` |
| Capability, boundary | 5 of 5 each | `single/summary.json` |
| Vision | 3 of 3 (solid-colour images answered correctly; an image plus text request gave the correct tool call) | `single/summary.json` |
| Context | 8 of 8 (needles at 10%, 50%, 90% and 33/66/90% depth in windows of 131,072 and 262,144 tokens) | `single/summary.json` |
| Exactness | 68 of 68 cases (27 drafted equals serial, 6 chunked prefill equals whole prefill, 33 reply captures, 2 prefix cache) | `exactness/exactness-cases.json` |
| Prefix cache | gate passes; a shared prefix is served from the cache and the reply is identical (83 checks, 0 failures) | `exactness/prefix-cache/check_prefix_cache.log` |
| Performance, performance smoke | 4 of 4, 3 of 3 | `performance/performance-cases.json` |
| Quality | 5 of 5 suites at or above their floors | `quality/quality-cases.json` |

Quality: the Q200 suite, thinking on (`chat_template_kwargs: {"enable_thinking": true}`), one GB10, 4 slots. Text suites greedy (temperature 0, top_p 1) with `max_tokens` 8,192; the tool-calling suite at temperature 0.001 with `max_tokens` 8,192. Q200 total 183 of 200 (91.5%). Recorded in `quality/q200/summary.json` and `quality/quality-cases.json`. These are this version's regression baseline, not a claim of absolute quality:

| Suite | Correct | Accuracy |
| --- | --- | --- |
| gsm8k | 76 of 80 | 95.0% |
| humaneval | 40 of 40 | 100% |
| ifeval | 34 of 40 | 85.0% |
| hard_reasoning | 20 of 20 (graded by an independent manual review: a fresh reviewer given only the review packet, blind to the run, as recorded in the evidence) | 100% |
| BFCL tool calling (20 hard cases) | 13 of 20 | 65.0% |

All 180 text rows finished with `stop`; none was cut by the token budget. No figure from an earlier pack of this model or an earlier build is carried here.

## 7. Limits of this release

- One GB10 node only; no tensor-parallel or two-node profile.
- Images up to 4,096 tokens each; video containers and audio are refused; video as image frames is not covered by the qualification.
- Constrained JSON response formats are refused (section 3).
- Thinking is off unless a request asks for it; effort levels are not stated for this pack.
- Concurrent full-window requests were not qualified, and the queue and request timeouts are the server defaults (section 5).
- The gates in section 6 are the qualification of this version; no 24-hour soak is part of them.
- The quality figures are one run on one node; a later version is held to them with a stated noise allowance, not to a fixed target.
- The engine was qualified for this exact commit and binary; a different engine build is not covered by this sheet.

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/minicpm-v-4.7/minicpm-v-4.7-35b-a3b/v1/KNOWN_ISSUES.md`
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
