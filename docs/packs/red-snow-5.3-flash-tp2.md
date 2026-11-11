# Pack sheet: `redsnow-5.3-flash-tp2-sage-4.91`

Checked against model version `v1` (sealed approval `approve-redsnow-5.3-flash-tp2-sage-4.91-1`, source commit `75031112ba12`, engine `0.1.1`, bundle archive `redsnow-5.3-flash-tp2-sage-4.91-1.tar.gz`), brought in by #127. This version is not yet part of a product release; the release tag is set when a release manifest names it.

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `redsnow-5.3-flash-tp2-sage-4.91` | `profile.json` |
| Versions | `v1` (latest, and the only version in this tree) | `runtimes/glm-5.3/red-snow-5.3-flash-tp2/LATEST` |
| Release tag | not stated (no product release contains `v1` yet) | `releases/` |
| What it serves | RED-SNOW-5.3-Flash (`Glm5NextForConditionalGeneration`, a GLM-5.3-Flash derivative), EXL3 4.91 bits per weight quantized with SAGE, on two GB10 nodes (tensor parallel, 2 ranks). SAGE is our own mixed-precision quantization method for EXL3. The method is not published. The pack card describes the model as one for authorized offensive-security work and states its limits; read it before deployment. | `MODEL.json`, `model.lock.json`, `KNOWN_ISSUES.md` |
| Model repository and revision | `vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw` at `f6b8963784254095a54944f2963b2925fcf4b199` | `model.lock.json` |
| Draft or second model | none (speculative decoding uses the MTP layer stored in the pack) | `model.lock.json` |
| Model licence | MIT (model card metadata and `LICENSE` of the pack; its base `Blackfrost-AI/RED-SNOW-5.3-FLASH-BF16` and that model's base `zai-org/GLM-5.3-Flash` are MIT) | `model.lock.json` |
| Download size | 43 files (36 shards plus index, tokenizer and 5 small files), 199,538,523,078 bytes (185.8 GiB); both nodes need their own copy; free space needed: the remaining bytes plus a reserve | `model.lock.json`, `KNOWN_ISSUES.md` |

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `redsnow-tp2-512k` (default, and the only profile) | two nodes, tensor parallel (2 ranks) | 524,288 tokens | 1 | 8952 | `127.0.0.1` on the first node | at most 900 s (`readiness_timeout_s`, a ceiling) after model verification |

Source: `profile.json`. Arguments of the profile: `--slots 1`, `--max-model-len 524288`, `--kv-tokens 524288`, `--spec auto`, `--prefix-cache pool`, `--prefill-chunk 2048`, `--max-image-tokens 4096`. The pack advertises 1,048,576 tokens; at that window rank 0 cannot also hold the vision tower above its memory floor on this engine, so the profile ships at 524,288 (`KNOWN_ISSUES.md`, "Known issues"). The profile needs two GB10s joined by a RoCE link and a site file (user guide section 4.6): the sealed `config/cluster.tp2-512k.json` carries example network values the site must replace (nodes, fabric interface, HCAs and TP port). Rank 0 serves HTTP and holds the vision tower. Each node needs huge pages off and at least 114 GiB of idle `MemAvailable` before the group starts; the launcher's pre-flight enforces it (`KNOWN_ISSUES.md`, "Memory").

Only this two-node unit is in production. The sealed `KNOWN_ISSUES.md` mentions a single-node entry of a different pack; that entry is not part of this tree.

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | yes | |
| Thinking (reasoning) | yes | Default when a request names no setting: **off**. The pack's chat template opens a think block only when `chat_template_kwargs.enable_thinking` is true. The template never reads `reasoning_effort`; effort levels have no effect on this pack. |
| Tools | yes | `tool_choice` `auto` and `none` only (`required` is refused with HTTP 400 in the qualification run). See section 5 for tool behaviour with thinking off. |
| Images | yes | Inline data URI only; up to 4 images a request, at most 4,096 image tokens each (a larger image is scaled down, not refused) |
| Prefix cache | yes | Automatic; entries live in the KV pool's idle pages (`--prefix-cache pool`) and reach 524,288 tokens |
| Speculative decoding | yes | Automatic (`--spec auto`, the pack's MTP layer) |
| Audio, video, `json_object`, `json_schema` | no | Refused by the server |

Source: `profile.json` `capabilities`, `KNOWN_ISSUES.md`, and the tool-matrix and boundary records in the qualification evidence (`evidence.tar.gz`). Everything not listed as supported is refused (user guide section 5.10).

## 4. Sampling and output defaults

The pack's card and its `generation_config.json` give temperature 0.7 and top_p 0.95; the card states no top_k. The profile carries no sampling flag, so clients pass temperature and top_p themselves (`KNOWN_ISSUES.md`, "Reasoning and sampling"). A request that names nothing does not think. When you turn thinking on, leave room in `max_tokens` for the reasoning. The quality run (thinking off) used 8,192 tokens and no text row was cut by that budget (`tp2/quality/q200/summary.json`).

## 5. Behaviour to expect

- **With thinking off, the model may answer a tool request from memory instead of calling the tool.** This shows in the tool-calling suite (BFCL hard20 8 of 20 with thinking off, section 6). A replay of three such cases with thinking on (`enable_thinking: true`) called the right tool in each. If your client depends on tool calls, turn thinking on. The replay is not part of the sealed evidence.
- **Some replies after a tool call start with reasoning text that ends in a closing think tag (`</think>`).** A client that shows the reply as it arrives will show that text before the answer.
- A request that names no reasoning setting does not think; send `enable_thinking: true` in `chat_template_kwargs` for reasoning (`KNOWN_ISSUES.md`, "Reasoning and sampling").
- One slot. A cold full-window prompt takes about 515 s to prefill (section 6), and a request sent meanwhile waits behind it.
- A cache entry lives in the KV pool's idle pages: a full-window prompt that differs from the cached one evicts it before its prefill, while a follow-up turn that shares the cached prefix keeps it (`KNOWN_ISSUES.md`, "Known issues").
- The pack's chat template injects its own system prompt (about 176 tokens or more) into every prompt, so a short request may report a few cached tokens once another request has been served (`KNOWN_ISSUES.md`).
- The pack card says the vision components are packaged but not quality-tested by the publisher; the qualification here checks colour images and an image-plus-text request only (section 6).
- The dependency set is the GLM-5.3-Flash two-node lock and has not been rehearsed on a clean host for this runtime (`KNOWN_ISSUES.md`, "Unverified boundaries").
- The sealed `profile.json` still records `qualification.state: unverified`, and `KNOWN_ISSUES.md` describes the frozen configuration before qualification (its reference figures are from a development build, not the receipt). The two tool behaviours above were found in the qualification and are not in the sealed `KNOWN_ISSUES.md`. The qualification of this version is the evidence asset `evidence.tar.gz` (section 6); where the two differ, the evidence is the record of what was measured.

## 6. Measured behaviour

All figures are from the evidence asset `evidence.tar.gz` (the dev release asset of `approve-redsnow-5.3-flash-tp2-sage-4.91-1`): two GB10 nodes, tensor parallel (2 ranks), profile `redsnow-tp2-512k` (1 slot, window 524,288, MTP drafting on, vision on).

What ran where. Capability, boundary, doctor, exactness, lifecycle, performance, prefix cache and vision ran fresh on the release binary (engine `0.1.1`, commit `75031112ba12`). Context and quality were carried from a complete, all-pass qualification of the previous engine build (commit `2067531c80ec`, same pack and profile) under a bit-identity record (`bits/bit-identity.json`): 51 greedy and seeded requests replayed on both builds with the profile's arguments and environment gave token-identical outputs. The 51 are 32 reply captures, 15 chunked-prefill requests (up to 522,912 prompt tokens), one full-window request of 523,135 prompt tokens, one image, one reasoning and one tool request.

Realistic workload first (previous build, carried). During the Q200 text run (180 requests, one at a time, thinking off, greedy, `max_tokens` 8,192; client on a third host, remote from the leader), one request generated a median of 30.7 tok/s, measured as completion tokens over the client's wall time for that request including its prompt (28,915 completion tokens in 855 s of request time; `tp2/quality/q200/summary.json`, `text180.e2e_throughput`).

Baseline gate on the release binary (three runs, each a fresh server; each figure is the median of the three run results; `v011/performance/performance-cases.json`, per-run detail in `v011/performance/runs/run-0N-RESULT.json`):

| Quantity | Method | Value |
| --- | --- | --- |
| Generation speed, one stream | 5 streamed replies of 256 tokens per run, MTP drafting on, short context; per-run median of the 5 | 34.5 tok/s (34.2 to 34.5); floor 27 |
| Prompt speed at 50% window fill | one cold prompt of 262,337 tokens (a needle-in-a-haystack fill) per run, about 239 s | 1,097 tok/s (1,096 to 1,106); floor 851 |
| Prompt speed at full window fill | one cold prompt of 520,333 tokens (needle fill plus a tool call) per run, about 513 s | 1,014 tok/s (1,010 to 1,021); floor 788 |
| Lowest free memory during the baseline runs | minimum `MemAvailable` the gate sampled on the leader node (rank 0) in each run | 6.06 to 6.65 GiB, no swap |

Draft acceptance over each baseline run was 0.69 to 0.70 (accepted over drafted tokens, `speed`/`spec` in the run results).

Other gates (`v011/composed/summary.json`, every gate PASS in one composed pass):

| Gate | Result | Recorded in |
| --- | --- | --- |
| Doctor | 4 of 4 | `v011/doctor/doctor-cases.json` |
| Lifecycle | 10 of 10 (boot, serve, stop and leak checks on both ranks, then a restart in new processes with the same reply; boot to ready took 32.6 s and 31.2 s) | `v011/lifecycle/lifecycle-cases.json` |
| Capability, boundary | 5 of 5 each | `v011/composed/summary.json` |
| Vision | 3 of 3 (solid-colour images answered correctly; text after an image answered correctly) | `v011/composed/summary.json` |
| Exactness | 68 of 68 cases (27 drafted equals serial, 6 chunked prefill equals whole prefill including a 523,264-token case, 33 reply captures, 2 prefix cache) | `v011/exactness/exactness-cases.json` |
| Prefix cache | gate passes; a shared prefix is served from the cache and the reply is identical (86 checks, 0 failures) | `v011/exactness/prefix-cache/check_prefix_cache.log` |
| Performance, performance smoke | 4 of 4, 3 of 3 | `v011/performance/performance-cases.json` |
| Context (carried) | 12 of 12 (needles at 10%, 50%, 90% and 33/66/90% depth in windows of 131,072, 262,144 and 524,288 tokens) | `tp2/qualification/summary.json` |
| Quality (carried) | 5 of 5 suites at or above their floors | `tp2/quality/quality-cases.json` |

Quality (previous build, carried): the Q200 suite, thinking off (`chat_template_kwargs: {"enable_thinking": false}`), two GB10s, one slot, one client worker, client remote from the leader. Text suites greedy (temperature 0, top_p 1) with `max_tokens` 8,192; the tool-calling suite at temperature 0.001 with `max_tokens` 8,192. Q200 total 160 of 200 (80.0%). Recorded in `tp2/quality/q200/summary.json` and `tp2/quality/quality-cases.json`. These are this pack's own regression baseline, not a claim of absolute quality; the floors are each suite's score minus two cases, with an overall bar of 80%:

| Suite | Correct | Accuracy | Floor |
| --- | --- | --- | --- |
| gsm8k | 69 of 80 | 86.25% | 67 of 80 |
| humaneval | 39 of 40 | 97.5% | 37 of 40 |
| ifeval | 30 of 40 | 75.0% | 28 of 40 |
| hard_reasoning | 14 of 20 (graded by an independent manual review: a fresh reviewer given only the review packet, blind to the run) | 70.0% | 12 of 20 |
| BFCL tool calling (20 hard cases) | 8 of 20 | 40.0% | 6 of 20 |

All 180 text rows finished with `stop`; none was cut by the token budget. The tool-calling suite ran with thinking off only, where the model may answer a tool request from memory (section 5); its score with thinking on is not measured.

## 7. Limits of this release

- Two GB10 nodes only, tensor parallel; no single-node profile in this tree.
- Window 524,288 tokens, not the 1,048,576 the pack advertises (section 2).
- Images up to 4,096 tokens each (larger images are scaled down; the quality effect on fine detail is not measured); video, audio and constrained JSON response formats are refused (section 3).
- Thinking is off unless a request asks for it; with thinking off tool requests may be answered from memory (section 5).
- Q200 was run with thinking off only; quality with thinking on is not measured.
- One slot; concurrent requests queue.
- The gates in section 6 are the qualification of this version; no 24-hour soak is part of them.
- The quality figures are one run on one cluster; a later version is held to them with a stated noise allowance, not to a fixed target.
- The engine was qualified for this exact commit and binary (quality and context through the bit-identity record above); a different engine build is not covered by this sheet.

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/glm-5.3/red-snow-5.3-flash-tp2/v1/KNOWN_ISSUES.md`
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
