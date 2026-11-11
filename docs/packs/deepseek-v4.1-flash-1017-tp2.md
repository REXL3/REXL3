# Pack sheet: `dsv41-flash-1017-sage-rexl3-1.55`

Checked against model version `v1` (sealed approval `approve-dsv41-flash-1017-sage-rexl3-1.55-1`, source commit `75031112ba12`, engine `0.1.1`, bundle archive `dsv41-flash-1017-sage-rexl3-1.55-1.tar.gz`), brought in by #130. This version is not yet part of a product release; the release tag is set when a release manifest names it.

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `dsv41-flash-1017-sage-rexl3-1.55` | `profile.json` |
| Versions | `v1` (latest, and the only version in this tree) | `runtimes/deepseek-v4.1/deepseek-v4.1-flash-1017-tp2/LATEST` |
| Release tag | not stated (no product release contains `v1` yet) | `releases/` |
| What it serves | DeepSeek-V4.1-Flash-1017 (`DeepseekV41ForCausalLM`), EXL3 1.55 bits per weight quantized with SAGE, on two GB10 nodes (tensor parallel, 2 ranks). SAGE is our own mixed-precision quantization method for EXL3. The method is not published. | `MODEL.json`, `model.lock.json` |
| Model repository and revision | `vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw` at `cba79c67582950ea44a4a449b3bb85b800748a35` | `model.lock.json` |
| Draft or second model | none | `model.lock.json` |
| Model licence | see the upstream model card and the LICENSE in the pack | `model.lock.json` |
| Download size | 24 files, 320,275,894,170 bytes (298.3 GiB); both nodes need their own copy; free space needed: the remaining bytes plus a reserve | `model.lock.json`, `KNOWN_ISSUES.md` |

This is the same pack as the single-node entry `runtimes/deepseek-v4.1/deepseek-v4.1-flash-1017/` (runtime `dsv41-flash-1017-sage-1.55`, profile `dsv41-single-262k`). The two entries are separate runtime units with their own approvals; this sheet covers the two-node entry only.

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `dsv41-tp2-1m` (default, and the only profile) | two nodes, tensor parallel (2 ranks) | 1,048,576 tokens | 1 | 8940 | the leader's management `address` from the site file (the profile's `127.0.0.1` default is not used for two nodes; `KNOWN_ISSUES.md`, "TP=2 adapter") | at most 900 s (`readiness_timeout_s`, a ceiling) after model verification |

Source: `profile.json`. Arguments of the profile: `--slots 1`, `--max-model-len 1048576`, `--kv-dtype q4`, `--spec auto`, `--prefix-cache on`, `--no-vision`, `--no-video`. The profile needs two GB10s joined by a RoCE link and a site file (user guide section 4.6): the site supplies `nodes` (exactly `leader` and `follower`, each with its address, fabric address, user, model and binary directories), `http_port`, `tp_port`, `fabric_interface` and the `hca` rail list; the sealed `config/cluster.tp2-1m.json` example network is never used as a default (`KNOWN_ISSUES.md`, "TP=2 adapter"). The expert weights and the KV cache are split across the two ranks, about half on each.

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | yes | |
| Thinking (reasoning) | yes | Default when a request names no setting: **off**. Effort names, as a top-level `reasoning_effort` or in `chat_template_kwargs.reasoning_effort`: `none` and `off` turn thinking off; `minimal`, `low`, `medium`, `high`, `max` and `xhigh` turn it on; any other name, an integer included, is refused with HTTP 400. With thinking on and no effort given, the effort is `high` (75) (`KNOWN_ISSUES.md`, "Reasoning-effort names and sampling defaults") |
| Tools | yes | `tool_choice` `auto` and `none` only; at most 64 tools (65 are refused with HTTP 413 in the qualification run) |
| Images | **no** | Refused with HTTP 400. The profile carries `--no-vision` and `capabilities.images` is false, under a temporary exception granted by a maintainer on 2026-11-01 that expires on 2026-12-31: on this quantized pack the tower answered 0 of 6 solid colours correctly in a two-node probe. Vision returns when the pack passes that probe (`KNOWN_ISSUES.md`) |
| Prefix cache | yes | Automatic (the profile pins no size) |
| Speculative decoding | yes | Automatic (`--spec auto`) |
| Video, audio, `json_object`, `json_schema` | no | Refused by the server |

Source: `profile.json` `capabilities`, `KNOWN_ISSUES.md`, and the capability, boundary and edge-case records in the qualification evidence (`evidence.tar.gz`). Everything not listed as supported is refused (user guide section 5.10).

## 4. Sampling and output defaults

A request that leaves out `temperature`, `top_p` or `top_k` gets temperature 1.0 and top_p 0.95 from the pack's `generation_config.json` (the file the model lock pins), and `top_k` 20, the server's fallback; neither the pack nor the upstream model sets `top_k`. Requests can set these fields explicitly. Source: `KNOWN_ISSUES.md` "Reasoning-effort names and sampling defaults". For long reasoning use these defaults rather than temperature 0, and set `max_tokens` to bound a reply (section 5).

## 5. Behaviour to expect

- **Greedy decoding can loop.** At temperature 0 the model sometimes repeats a short phrase until `max_tokens` (`KNOWN_ISSUES.md`, "Greedy decoding can loop"). In this version's quality run (greedy, thinking off, 32,768-token budget) 7 of 180 text prompts ran to the budget and were counted incorrect: two gsm8k, one ifeval and four hard_reasoning (section 6).
- **A full-window prompt is slow to its first token.** A cold prompt of about 1,048,000 tokens took about 31.5 minutes to prefill in the baseline gate (section 6). One slot: a request sent meanwhile waits behind it. A shorter prompt costs nothing extra.
- **Long timeouts at this window.** Because `--max-model-len` is above 524,288, the server raises its request, prefill, decision and queue timeouts and the TP peer timeout to 14,400 s, so the profile's `--tp-peer-timeout 60` becomes 14,400 s: a stalled peer is declared lost only after that long (`KNOWN_ISSUES.md`).
- Image input is refused by design in this version (section 3).
- Model files are verified in full on every start and the model tree is read-only (`KNOWN_ISSUES.md` "Existing model copies"). All ranks must pass payload, binary, native-version and dependency identity checks before the group starts, and readiness requires an actual smoke reply, not a health 200 alone (`KNOWN_ISSUES.md`, "TP=2 adapter").
- The sealed `profile.json` still records `qualification.state: unverified`, and `KNOWN_ISSUES.md` describes the frozen configuration before qualification: it lists this pack's two-node receipt, 1,048,576-token window and Q200 baseline as pending, and its reference figures are from an earlier pack, not this one. The qualification of this version is the evidence asset `evidence.tar.gz` (section 6); where the two differ, the evidence is the record of what was measured.

## 6. Measured behaviour

All figures are from the evidence asset `evidence.tar.gz` (the dev release asset of `approve-dsv41-flash-1017-sage-rexl3-1.55-1`) unless stated: two GB10 nodes, tensor parallel (2 ranks), profile `dsv41-tp2-1m` (1 slot, window 1,048,576, KV cache q4, speculative decoding on, vision off). Every gate ran fresh on the release binary (engine `0.1.1`, commit `75031112ba12`) in one full chain; none was carried from an earlier build or pack.

Realistic workload first. During the Q200 text run (180 requests, one at a time, thinking off, greedy, `max_tokens` 32,768; client on a third host, remote from the leader), the 173 requests that finished with `stop` generated a median of 56.3 tok/s each, measured as completion tokens over the client's wall time for that request including its prompt (40,906 completion tokens in 839 s of request time, 48.8 tok/s in aggregate). The 7 requests that ran to the budget are left out of this figure because their repeated text would inflate it (`tp2/quality/q200/summary.json`, `text180.e2e_throughput.stopped_rows_only`). These are different prompts and replies from the 256-token gate replies below, so the two speeds are not comparable.

Baseline gate on the release binary (three runs, each a fresh server; each figure is the median of the three run results; `tp2/performance/performance-cases.json`, per-run detail in `tp2/performance/runs/run-0N-RESULT.json`):

| Quantity | Method | Value |
| --- | --- | --- |
| Generation speed, one stream | 5 streamed replies of 256 tokens per run, speculative decoding on, short context; per-run median of the 5 | 37.2 tok/s (37.2 to 37.3); floor 18 |
| Prompt speed at 50% window fill | one cold prompt of 524,292 tokens (a needle-in-a-haystack fill) per run, about 584 s | 898 tok/s (897 to 899); floor 700 |
| Prompt speed at full window fill | one cold prompt of 1,048,396 tokens (needle fill plus a tool call) per run, about 1,892 s | 554 tok/s (553 to 555); floor 400 |
| Lowest free memory during the baseline runs | minimum `MemAvailable` the gate sampled on the leader node (rank 0) in each run | 46.8 GiB in each run, no swap |

Draft acceptance over each baseline run was 0.60 (581 accepted of 969 drafted tokens, `spec` in the run results). The follower's memory is not sampled by the gate.

Other gates (`tp2/qualification/summary.json`, all 11 gates PASS):

| Gate | Result | Recorded in |
| --- | --- | --- |
| Doctor | 4 of 4 | `tp2/doctor/doctor-cases.json` |
| Lifecycle | 10 of 10 (boot, serve, stop and leak checks on both ranks, then a restart in new processes with the same reply; boot to ready took 31.4 s and 29.4 s, stop 5.0 s and 4.9 s) | `tp2/lifecycle/lifecycle-cases.json` |
| Capability, boundary | 5 of 5 each (text, streaming, thinking, tools, speculation; audio, video, `json_schema` and an over-window prompt of 1,048,596 tokens refused, text after a refusal answered) | `tp2/qualification/summary.json` |
| Vision | 2 of 2 under the exception in section 3: an image is refused with HTTP 400, and a text request after the refused image is answered correctly. No image is understood in this version | `tp2/qualification/summary.json` |
| Context | 16 of 16 (needles at 10%, 50%, 90% and 33/66/90% depth in windows of 131,072, 262,144, 524,288 and 1,048,576 tokens, each cold and then served again from the cache with the same answer) | `tp2/qualification/summary.json` |
| Exactness | 68 of 68 cases (27 drafted equals serial: 24 cases, 6 prompts each greedy and with three seeds, plus speculation, seed and coverage checks; 6 chunked prefill equals whole prefill at chunk sizes 128 and 256 and the default, the longest a 1,047,526-token prompt; 33 reply captures byte-identical to the pinned baseline; 2 prefix cache) | `tp2/exactness/exactness-cases.json` |
| Prefix cache | gate passes; a shared prefix is served from the cache and the reply is identical to the same request without the cache (84 checks, 0 failures, 25 hits) | `tp2/exactness/prefix-cache/check_prefix_cache.log` |
| Performance, performance smoke | 4 of 4, 3 of 3 | `tp2/performance/performance-cases.json` |
| Quality | 5 of 5 suites at or above their floors | `tp2/quality/quality-cases.json` |

An additional check outside the gate chain, on the same binary and profile at the full window of 1,048,576 tokens, found drafted output equal to serial output and cached output equal to cold output. Its record is not part of `evidence.tar.gz`.

Quality: the Q200 suite, reasoning effort off (`chat_template_kwargs: {"reasoning_effort": "off"}`), text suites greedy (temperature 0) with `max_tokens` 32,768; the tool-calling suite with `max_tokens` 32,768 and one thread (no separate temperature recorded); two GB10s, one slot, one client worker, client remote from the leader. Q200 total 182 of 200 (91.0%). A reply that ran to the 32,768-token budget counts as incorrect in this lane, whatever the grader said of the partial answer; 7 text rows did (section 5). Recorded in `tp2/quality/q200/summary.json` and `tp2/quality/quality-cases.json`. These are this pack's own regression baseline, not a claim of absolute quality; the floors are each suite's score minus two cases, with an overall bar of 90%:

| Suite | Correct | Accuracy | Floor |
| --- | --- | --- | --- |
| gsm8k | 78 of 80 (the two misses ran to the budget) | 97.5% | 76 of 80 |
| humaneval | 40 of 40 | 100% | 38 of 40 |
| ifeval | 35 of 40 (one miss ran to the budget) | 87.5% | 33 of 40 |
| hard_reasoning | 16 of 20 (graded by an independent manual review: a fresh reviewer given only the review packet, blind to the run; the four misses ran to the budget) | 80.0% | 14 of 20 |
| BFCL tool calling (20 hard cases) | 13 of 20 | 65.0% | 11 of 20 |

No figure from an earlier version, build or pack is carried here.

## 7. Limits of this release

- Two GB10 nodes only, tensor parallel; the single-node profile of this pack is a separate entry (section 1).
- One slot; concurrent requests queue, and a full-window prompt holds the slot for about half an hour (section 5).
- Image input, video, audio and constrained JSON response formats are refused (section 3).
- Q200 was run with reasoning effort off only; quality with thinking on is not measured.
- The gates in section 6 are the qualification of this version; no 24-hour soak is part of them. The follower node's memory is not sampled by the gate.
- The quality figures are one run on one cluster; a later version is held to them with a stated noise allowance, not to a fixed target.
- The engine was qualified for this exact commit and binary; a different engine build is not covered by this sheet.

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/deepseek-v4.1/deepseek-v4.1-flash-1017-tp2/v1/KNOWN_ISSUES.md`
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
