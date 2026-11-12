# Pack sheet: `glm53-flash-tp2-sage-2.25`

Checked against model version `v1` (sealed approval `approve-glm53-flash-tp2-sage-2.25-1`, source commit `75031112ba12`, engine `0.1.1`, bundle archive `glm53-flash-tp2-sage-2.25-1.tar.gz`), brought in by #145. This version is not yet part of a product release; the release tag is set when a release manifest names it.

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `glm53-flash-tp2-sage-2.25` | `profile.json` |
| Versions | `v1` (latest, and the only version in this tree) | `runtimes/glm-5.3/glm-5.3-flash-tp2/LATEST` |
| Release tag | not stated (no product release contains `v1` yet) | `releases/` |
| What it serves | GLM-5.3-Flash (`Glm5NextForConditionalGeneration`), EXL3 2.25 bits per weight quantized with SAGE, on two GB10 nodes (tensor parallel, 2 ranks). SAGE is our own mixed-precision quantization method for EXL3. The method is not published. Speculative decoding uses the NextN layer stored in the pack. | `MODEL.json`, `model.lock.json`, `KNOWN_ISSUES.md` |
| Model repository and revision | `r0b0tlab/GLM-5.3-Flash-EXL3-SAGE-2.25bpw` at `b3194b87af5665bd8b1dd6316c99a48607fc137d` | `model.lock.json` |
| Draft or second model | none (the NextN layer is part of the pack) | `model.lock.json` |
| Model licence | MIT (base model `zai-org/GLM-5.3-Flash`; see the pack's LICENSE and model card) | `model.lock.json` |
| Download size | 39 files, 98,519,480,006 bytes (91.8 GiB); each node needs its own copy (the site file names a model directory for each node); free space needed: the remaining bytes plus a reserve | `model.lock.json` |

This is the same pack as the single-node entry `runtimes/glm-5.3/glm-5.3-flash/` (profile `glm53-flash-single`, 262,144 tokens). The two entries are separate runtime units with their own approvals; this sheet covers the two-node entry only. In release `v0.1.0` the two-node profile `glm53-flash-tp2-1m` was part of the single-node entry's version 1 (runtime `glm53-flash-2.25bpw-sm121`); this entry is its own unit from now on.

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `glm53-flash-tp2-1m` (default, and the only profile) | two nodes, tensor parallel (2 ranks) | 1,048,576 tokens | 1 | 8950 (the site file's `http_port` replaces it) | the leader's `address` from the site file (`nodes.leader.address`), not `127.0.0.1`: the profile's `127.0.0.1` default is not used for two nodes, and the site file has no field to change the bind | at most 900 s (`readiness_timeout_s`, a ceiling) after model verification |

Source: `profile.json` and the launcher `scripts/runtime.py` (`configure_attested_cluster` sets the HTTP host to the leader's site address, and rank 0 is started with `--host` set to it). In the qualification run rank 0 was started that way, with `--host` set to the leader's site address.

**Network exposure.** Unlike the single-node profile, this profile listens on the leader's network address. The server has no API key and does not authenticate requests. Run it only on a private, trusted network, or behind a firewall or an SSH tunnel; it is not a public Internet gateway (the bundle's `README.md`: "Keep single-node endpoints on loopback and TP endpoints on a trusted private network").

Arguments of the profile: `--slots 1`, `--max-model-len 1048576`, `--kv-tokens 1048576`, `--spec auto`, `--prefix-cache on`, `--prefix-cache-mib 13440`, `--served-model-name GLM-5.3-Flash-EXL3-2.25bpw-sm121`, the group flags `--tp-ranks 2`, `--tp-connect-timeout 300`, `--tp-ready-timeout 900`, `--tp-peer-timeout 60`, `--shutdown-grace 5`, and `REXL3_MEMORY_FLOOR_GIB=4`. Requests name the model as `GLM-5.3-Flash-EXL3-2.25bpw-sm121`, the served name, which is unchanged from v0.1.0. The profile needs two GB10s joined by a RoCE link and a site file (user guide section 4.6): the site supplies `nodes` (exactly `leader` and `follower`), `http_port`, `tp_port`, `fabric_interface` and the `hca` rail list. The sealed `config/cluster.tp2-1m.json` carries example network values the site must replace. Rank 0 serves HTTP, holds the vision tower and encodes every image; each rank loads its share of the routed experts and keeps its own prefix-cache snapshots (`KNOWN_ISSUES.md`). The dependency lock pins `libnccl2`; a pair without `libnccl.so.2` exits at boot (`KNOWN_ISSUES.md`).

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | yes | |
| Thinking (reasoning) | yes | Default when a request names no setting: **on**, at effort `xhigh` (the template renders it as its `max`). There is no off switch: `enable_thinking: false` has no effect and the response says so in `X-Rexl3-Note`. Effort levels in `chat_template_kwargs.reasoning_effort`: `low`, `medium`, `high`, `xhigh`. A literal `max` and a top-level `reasoning_effort` field are refused with HTTP 400 (`KNOWN_ISSUES.md`, "Sampling and reasoning") |
| Tools | yes | `tool_choice` `auto` and `none` only; at most 64 tools |
| Images | yes | Inline data URI only; rank 0 encodes every image |
| Prefix cache | yes | Automatic. Pinned at 13,440 MiB (`--prefix-cache-mib 13440`): 5 entries and 4,106 pages on each rank, a reach of 1,051,136 tokens, a whole window (`KNOWN_ISSUES.md`) |
| Speculative decoding | yes | Automatic (`--spec auto`, the pack's NextN layer) |
| Video, audio, `json_object`, `json_schema` | no | Refused by the server with HTTP 400 |

Source: `profile.json` `capabilities`, `KNOWN_ISSUES.md`, and the capability and boundary records in the qualification evidence (`evidence.tar.gz`). Everything not listed as supported is refused (user guide section 5.10).

## 4. Sampling and output defaults

The pack's `generation_config.json` (locked) sets temperature 1.0 and top_p 0.95, which the server uses when a request does not set them. Neither the pack nor the base card states `top_k`; the server's fallback `top_k` is not sourced from either. The profile carries no sampling flag. The base card states no recommended `max_tokens`. Source: `KNOWN_ISSUES.md` "Sampling and reasoning". Every reply thinks first, so leave room in `max_tokens` for the reasoning: the sealed smoke request uses effort `low` with 2,048 tokens, and the quality run used `xhigh` with 65,536 tokens (section 6).

## 5. Behaviour to expect

- **Every reply thinks.** Thinking cannot be switched off; use `reasoning_effort` to scale it (section 3). At `xhigh` and a 65,536-token budget, 3 of 180 quality prompts ran to the budget and were counted incorrect (section 6).
- **A full-window prompt is slow to its first token.** A cold prompt of 1,044,466 tokens took about 1,066 s (about 18 minutes) to prefill in the baseline gate; a cold prompt of 524,309 tokens took about 462 s (section 6). One slot: a request sent meanwhile waits behind it.
- **A repeated long prompt is fast.** In the context gate, asking again with the same 1,044,480-token prompt was served from the prefix cache (1,044,416 tokens cached) with its first token in about 0.3 s, against about 1,055 s cold (section 6). A prompt longer than the cache's reach is prefilled in full each time (`KNOWN_ISSUES.md`).
- **The bind address is the leader's network address**, not loopback (section 2).
- The sealed `site.example.json` is a generic two-node example: its `http_port` (8940) and model directories are not this profile's values. Set `http_port` (the profile default is 8950) and each node's directories for your site.
- The sealed `profile.json` still records `qualification.state: unverified`, and `KNOWN_ISSUES.md` describes the frozen configuration before qualification: its two-node figures are from a diagnostic run on a development build, it lists a real two-node receipt as not yet run, and its sentence that this profile "ships in v0.1.0 beside the single-node default" describes the earlier combined entry. The qualification of this version is the evidence asset `evidence.tar.gz` (section 6); where the two differ, the evidence is the record of what was measured.

## 6. Measured behaviour

All figures are from the evidence asset `evidence.tar.gz` (the dev release asset of `approve-glm53-flash-tp2-sage-2.25-1`): two GB10 nodes, tensor parallel (2 ranks), profile `glm53-flash-tp2-1m` (1 slot, window 1,048,576, speculative decoding on, prefix cache on, vision on). Every gate ran fresh on the release binary (engine `0.1.1`, commit `75031112ba12`) in one full chain; none was carried from an earlier build.

Realistic workload first. During the Q200 text run (180 requests, one at a time, non-streamed, reasoning effort `xhigh`, greedy, `max_tokens` 65,536; client on a third host, remote from the leader), the 177 requests that finished with `stop` generated a median of 64.0 tok/s each, measured as completion tokens (thinking included) over the client's wall time for that request, prompt included (279,118 completion tokens in 4,484 s of request time, 62.2 tok/s in aggregate). The 3 requests that ran to the budget are left out of this figure (`tp2/quality/q200/summary.json`, `text180.e2e_throughput.stopped_rows_only`). These are different prompts and replies from the 256-token gate replies below, so the two speeds are not comparable.

Baseline gate on the release binary (three runs, each a fresh server; each figure is the median of the three run results; `tp2/performance/performance-cases.json`, per-run detail in `tp2/performance/runs/run-0N-RESULT.json`):

| Quantity | Method | Value |
| --- | --- | --- |
| Generation speed, one stream | 5 streamed replies of 256 tokens per run, speculative decoding on, short context; per-run median of the 5 | 56.8 tok/s (56.4 to 56.8) |
| Prompt speed at 50% window fill | one cold prompt of 524,309 tokens (a needle-in-a-haystack fill) per run, 461 to 463 s | 1,133 tok/s (1,132 to 1,137) |
| Prompt speed at full window fill | one cold prompt of 1,044,466 tokens (needle fill plus a tool call) per run, 1,065 to 1,067 s | 980 tok/s (979 to 981) |
| Lowest free memory during the baseline runs | minimum `MemAvailable` the gate sampled on the leader node (rank 0) in each run | 32.3 to 32.7 GiB, no swap |

The gate's floors (decode 31, prefill 770 and 750 tok/s) are the single-node recipe's, which this profile shares (`KNOWN_ISSUES.md`); every run passed them. Draft acceptance over each baseline run was 0.72 to 0.73 (2,237 to 2,243 accepted of 3,085 to 3,093 drafted tokens, `spec` in the run results). The follower's memory is not sampled by the gate.

Other gates (`tp2/qualification/summary.json`, all 11 gates PASS):

| Gate | Result | Recorded in |
| --- | --- | --- |
| Doctor | 4 of 4 | `tp2/doctor/doctor-cases.json` |
| Lifecycle | 10 of 10 (boot, serve, stop and leak checks on both ranks, then a restart in new processes with the same reply; boot to ready took 14.6 s and 21.1 s as the gate measures it, stop 5.7 s and 4.8 s, both ranks exited 0) | `tp2/lifecycle/lifecycle-cases.json` |
| Capability | 5 of 5 (text, streaming, thinking at efforts `low` and `xhigh`, tools, speculation) | `tp2/qualification/summary.json` |
| Boundary | 5 of 5 (audio, video, `json_schema` and an over-window prompt of 1,048,604 tokens refused with HTTP 400; text after a refusal answered) | `tp2/qualification/summary.json` |
| Vision | 3 of 3 (a solid red and a solid blue image named correctly; text after an image answered) | `tp2/qualification/summary.json` |
| Context | 16 of 16: needles at 10%, 50%, 90% and 33/66/90% depth in windows of 131,072, 262,144, 524,288 and 1,048,576 tokens (prompts of 126,976 to 1,044,480 tokens; server default sampling, seed 1729, 4,096-token budget), each cold and then asked again and served from the cache with the same answer | `tp2/qualification/summary.json` |
| Exactness | 68 of 68 cases (27 drafted equals serial: 24 cases, 6 prompt kinds each greedy and with three seeds, plus speculation, seed and coverage checks; 6 chunked prefill equals whole prefill at chunk sizes 256 and 512 and the default, greedy, 48 tokens, cache off, the longest a 1,047,682-token prompt; 33 reply captures byte-identical to the pinned baseline; 2 prefix cache) | `tp2/exactness/exactness-cases.json` |
| Prefix cache | gate passes; a shared prefix is served from the cache and the reply is identical to the same request without the cache (84 checks, 0 failures, 25 hits) | `tp2/exactness/prefix-cache/check_prefix_cache.log` |
| Performance, performance smoke | 4 of 4, 3 of 3 | `tp2/performance/performance-cases.json` |
| Quality | 5 of 5 suites at or above their floors | `tp2/quality/quality-cases.json` |

Quality: the Q200 suite, reasoning effort `xhigh` (`chat_template_kwargs: {"reasoning_effort": "xhigh"}`), text suites greedy (temperature 0, top_p 1) with `max_tokens` 65,536; the BFCL tool-calling suite with temperature 0.001, `max_tokens` 8,192 and one thread; two GB10s, one slot, one client worker, client remote from the leader. Q200 total 184 of 200 (92.0%); text 172 of 180. A reply that ran to the 65,536-token budget counts as incorrect in this lane, whatever the grader said of the partial answer; 3 text rows did (one each in humaneval, ifeval and hard_reasoning). Recorded in `tp2/quality/q200/summary.json` and `tp2/quality/quality-cases.json`. These are this pack's own regression baseline, not a claim of absolute quality. The floors are the pack's v0.1.0 two-node scores minus two cases per suite:

| Suite | Correct | Accuracy | Floor |
| --- | --- | --- | --- |
| gsm8k | 77 of 80 | 96.25% | 75 of 80 (93.75%) |
| humaneval | 39 of 40 (the miss ran to the budget) | 97.5% | 37 of 40 (92.5%) |
| ifeval | 37 of 40 (one miss ran to the budget) | 92.5% | 35 of 40 (87.5%) |
| hard_reasoning | 19 of 20 (graded by an independent manual review: a fresh reviewer given only the review packet, blind to the run; the miss ran to the budget) | 95.0% | 17 of 20 (85%) |
| BFCL tool calling (20 hard cases) | 12 of 20 | 60.0% | 10 of 20 (50%) |

No figure from an earlier version, build or pack is carried here.

## 7. Limits of this release

- Two GB10 nodes only, tensor parallel; the single-node profile of this pack is a separate entry at 262,144 tokens (section 1).
- The HTTP endpoint is on the leader's network address and has no API key (section 2).
- One slot; concurrent requests queue, and a cold full-window prompt holds the slot for about 18 minutes (section 5).
- Video, audio and constrained JSON response formats are refused (section 3).
- Thinking cannot be turned off. Q200 was run at effort `xhigh` only; quality at other efforts is not measured.
- BFCL tool calling (hard-20) is the weakest result at 12 of 20.
- The gates in section 6 are the qualification of this version; no 24-hour soak is part of them. The follower node's memory is not sampled by the gate.
- The quality figures are one run on one cluster; a later version is held to them with a stated noise allowance, not to a fixed target.
- The engine was qualified for this exact commit and binary; a different engine build is not covered by this sheet.

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/glm-5.3/glm-5.3-flash-tp2/v1/KNOWN_ISSUES.md`
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
