# Pack sheet: `ling31-flash-1019-sage-1.51`

Checked against model version `v1` (sealed approval `approve-ling31-flash-1019-sage-1.51-1`, source commit `75031112ba12`, engine `0.1.1`, bundle archive `ling31-flash-1019-sage-1.51-1.tar.gz`), brought in by #153. This version ships in product release `v0.1.1`.

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `ling31-flash-1019-sage-1.51` | `profile.json` |
| Versions | `v1` (latest, and the only version in this tree) | `runtimes/ling-3.1/ling-3.1-flash-1019/LATEST` |
| Release tag | `v0.1.1` | `releases/v0.1.1/manifest.json` |
| Release assets | `ling31-flash-1019-sage-1.51-v1-approval.json`, `ling31-flash-1019-sage-1.51-v1-bundle.tar.gz`; evidence `ling31-flash-1019-sage-1.51-v1-evidence.tar.gz` | the release page |
| What it serves | Ling-3.1-Flash-1019 (`BailingMoeV3ForCausalLM`), EXL3 1.51 bits per weight quantized with SAGE, on one GB10. SAGE is our own mixed-precision quantization method for EXL3. The method is not published. Speculative decoding uses the MTP head stored in the pack. Text only: the model has no vision tower. | `MODEL.json`, `model.lock.json`, `KNOWN_ISSUES.md` |
| Model repository and revision | `vcruz305/Ling-3.1-flash-1019-EXL3-SAGE-1.51bpw` at `437b03e6ddcc8b1ed653619aa8f266bbeeecd084` | `model.lock.json` |
| Draft or second model | none (the MTP head is part of the pack) | `model.lock.json` |
| Model licence | MIT (model card metadata of the pack repository; base model `inclusionAI/Ling-3.1-flash-1019`) | `model.lock.json` |
| Download size | 30 files (24 weight shards), 115,473,028,099 bytes (107.5 GiB); free space needed: the remaining bytes plus a reserve | `model.lock.json` |

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `ling31-single-262k` (default, and the only profile) | single node | 262,144 tokens | 1 | 8889 | `127.0.0.1` | at most 600 s (`readiness_timeout_s`, a ceiling) after model verification |

Source: `profile.json`. There is no tensor-parallel choice for this runtime.

Arguments of the profile: `--max-model-len 262144`, `--slots 1`, `--spec mtp`, `--prefix-cache on`, `--no-vision`, `--prefill-chunk 512`, `--heartbeat-stale-secs 1800`, `--request-timeout 7200`, `--served-model-name Ling-3.1-flash-1019-EXL3-SAGE-1.51bpw`, and `REXL3_MEMORY_FLOOR_GIB=2`. Requests name the model as `Ling-3.1-flash-1019-EXL3-SAGE-1.51bpw`. The weights take about 106 GiB of the GB10's memory, so whether the KV cache for the whole window fits depends on the free memory when the server starts (`KNOWN_ISSUES.md`, "Memory at 262,144 tokens"; section 5).

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | yes | |
| Thinking (reasoning) | yes | Default when a request names no setting: **off**. The qualification server's `/v1/models` reports `default: {"template_thinks": true, "server": "off", "thinks": false}`: the profile passes no `--default-thinking` or `--default-reasoning-effort`, so the server sets `enable_thinking: false` for a request that names neither, which is the case the thinking-off known issue below describes. The qualification itself always named the setting, so a request with no setting was not exercised. Switch: `chat_template_kwargs.enable_thinking` (`true` or `false`); the effort levels are `on` and `off`, accepted as `chat_template_kwargs.reasoning_effort` or as a top-level `reasoning_effort` (`/v1/models` `request_fields` and `aliases`); there is no reasoning budget. The qualification set `enable_thinking: true` in its gate requests and `chat_template_kwargs.reasoning_effort: "on"` in its quality run |
| Tools | yes | `tool_choice` `auto` and `none` only (`required` is refused with HTTP 400); at most 64 tools (65 are refused with HTTP 413) |
| Images | no | The model has no vision tower. An image part is refused with HTTP 400 and the server stays healthy |
| Prefix cache | yes | Automatic. Entries are kept on disk with a quota, not in GPU memory (the boot plan line reports `where=disk`); a hit restores the snapshot from disk (`KNOWN_ISSUES.md`, "Prefix cache") |
| Speculative decoding | yes | Automatic (`--spec mtp`, the pack's MTP head) |
| Video, audio, `json_object`, `json_schema` | no | Refused by the server with HTTP 400 |

Source: `profile.json` `capabilities`, `KNOWN_ISSUES.md`, and the capability, boundary and vision records in the qualification evidence (`evidence.tar.gz`). Everything not listed as supported is refused (user guide section 5.10).

## 4. Sampling and output defaults

The pack ships no `generation_config.json` (the lock lists none), so a request that leaves a field out gets the server's own fallback: temperature 1.0, top_p 0.95, top_k 20 (`crates/server/src/api.rs`, `GenDefaults`). The profile carries no sampling flag. With thinking on, leave room in `max_tokens` for the reasoning: the quality run used a 32,768-token budget and 8 of its 180 text replies ran to it (section 6); the context gate used 4,096 tokens.

## 5. Behaviour to expect

- With `chat_template_kwargs.enable_thinking` false, the model still writes reasoning text and a stray `</think>` into the answer (model behaviour; maintainer-granted thinking exception, expires 2026-12-31). Workaround: keep thinking on, or strip everything up to the last `</think>` from the content. The release qualifier's context, prefix-cache, tool, speculation and smoke checks ran with thinking on under that exception.
- `launch.sh start` re-hashes the ~107 GiB pack just before boot; the page cache this leaves can make the engine plan its KV cache below the 262,144-token window and refuse to start. Workaround: free the file cache before start (for example `sync; echo 1 | sudo tee /proc/sys/vm/drop_caches`). A launcher fix is planned for a later release; until then use the workaround.
- **Thinking is off unless the request asks for it** (section 3). Send `chat_template_kwargs: {"enable_thinking": true}` for the behaviour the quality figures in section 6 describe.
- **Little memory is left at the full window.** During the baseline runs the lowest free memory the gate sampled was 3.1 to 3.5 GiB, and swap grew by 0.64 GiB in one of the three runs (section 6). Do not run other large jobs on the same GB10. The engine refuses a window it cannot hold rather than starting with a smaller one.
- **A full-window prompt is slow to its first token.** A cold prompt of 260,148 tokens took about 1,165 s (about 19.5 minutes) to prefill in the baseline gate; one of 131,109 tokens took about 383 s. One slot: a request sent meanwhile waits behind it.
- **A repeated long prompt is fast.** In the context gate, asking again with the same 258,048-token prompt was served from the prefix cache (257,984 tokens cached) with its first token in about 5.7 s, against about 1,140 s cold (section 6).
- **No image input.** The pack has no vision tower, so the profile passes `--no-vision`. The release tools accept that switch for a pack without a tower only under a documented exception, which a maintainer granted (sealed in the receipt, expires 2027-03-31) (`KNOWN_ISSUES.md`, "Vision"). Nothing is withheld.
- The sealed `KNOWN_ISSUES.md` describes the frozen configuration before qualification: it says the release qualification is not yet complete and that the recipe has no prefill floors yet, and it has no section on the thinking-off behaviour. The sealed `profile.json` records `qualification.state: verified` for a configuration check on an earlier development build (five streamed replies, median decode 25.79 tok/s), not for this release's gates. The qualification of this version is the evidence asset `evidence.tar.gz` (section 6); where they differ, the evidence is the record of what was measured. A sealed tree is not edited.

## 6. Measured behaviour

All figures are from this version's evidence archive, the release asset `ling31-flash-1019-sage-1.51-v1-evidence.tar.gz` (named `evidence.tar.gz` inside the sealed approval): one GB10, profile `ling31-single-262k` (1 slot, window 262,144, speculative decoding on, prefix cache on, no vision). Every gate ran fresh on the release binary (engine `0.1.1`, commit `75031112ba12`, `rexl3-serve` sha256 `4fc2cb44…5fe5`); none was carried from an earlier build. `launch.sh doctor` and the smoke check ran on a second GB10 with NVIDIA driver 580.173.02 (`q148-release-notes.md` in the same asset); the gates below ran on the qualification host.

Realistic workload first. During the Q200 text run (180 requests, one at a time, non-streamed, thinking on with `reasoning_effort` `on`, temperature 0, `max_tokens` 32,768; client on the same host as the server), the 172 requests that finished with `stop` generated a median of 31.5 tok/s each, measured as completion tokens (thinking included) over the client's wall time for that request, prompt included (216,364 completion tokens in 7,285 s of request time, 29.7 tok/s in aggregate). The 8 requests that ran to the budget are left out of this figure (`quality/q200/summary.json`, `text180.e2e_throughput.stopped_rows_only`). These are different prompts and replies from the 256-token gate replies below, so the two speeds are not comparable.

Baseline gate on the release binary (three runs, each a fresh server; each figure is the median of the three run results; `performance/performance-cases.json`, per-run detail in `performance-input/runN-RESULT.json`):

| Quantity | Method | Value |
| --- | --- | --- |
| Generation speed, one stream | 5 streamed replies of 256 tokens per run, MTP on, short context; per-run median of the 5 | 27.1 tok/s (27.0 to 27.1) |
| Prompt speed at 50% window fill | one cold prompt of 131,109 tokens (a needle-in-a-haystack fill) per run, 382.5 to 384.1 s | 341.8 tok/s (341.3 to 342.8) |
| Prompt speed at full window fill | one cold prompt of 260,148 tokens (needle fill plus a tool call) per run, 1,165.1 to 1,168.5 s | 223.3 tok/s (222.6 to 223.3) |
| Lowest free memory during the baseline runs | minimum `MemAvailable` the gate sampled in each run | 3.1 to 3.5 GiB; swap grew by 0.64 GiB in run 1 and not in runs 2 and 3 |

The floors are the recipe's: decode 20 tok/s, prefill 250 tok/s at 50% and 150 tok/s at 100% of the window, with a 5% tolerance; every run passed them. The two prefill floors were added to the recipe after the three runs, so the performance evidence was derived again from the same three run results against the committed recipe; the measurements did not change, and the original run results are kept beside the derived ones (`performance-input/original-runN-RESULT.json`). Draft acceptance over each baseline run was 0.70 (611 accepted of 875 drafted tokens, `spec` in the run results).

Other gates (`single/summary.json`, all 11 gates PASS):

| Gate | Result | Recorded in |
| --- | --- | --- |
| Doctor | 4 of 4 (`launch.sh doctor`, static checks: status, runtime id, architecture, exact native version policy; second GB10, driver 580.173.02) | `doctor/doctor-cases.json` |
| Lifecycle | 10 of 10 (boot, serve, stop and leak checks, then a restart in a new process with the same reply; boot to ready took 33.2 s and 29.9 s as the gate measures it, stop 1.7 s and 1.4 s, exit 0) | `lifecycle/lifecycle-cases.json` |
| Capability | 5 of 5 (text, streaming, thinking in its excepted form, tools, speculation). The thinking row asks for 17 × 23: with thinking on the answer is `391` with the reasoning returned separately; the thinking-off reply (78 completion tokens, reasoning text and `</think>` in the content) is recorded under the thinking exception, not required. Tools and speculation ran with thinking on and a 4,096-token budget (speculation: 936 accepted of 1,147 drafted tokens) | `single/summary.json` |
| Boundary | 5 of 5 (audio, video, `json_schema` and an over-window prompt of 262,181 tokens refused with HTTP 400; text after a refusal answered) | `single/summary.json` |
| Vision | 2 of 2 (an image refused with HTTP 400 and the server healthy afterwards; text after the refused image answered) | `single/summary.json` |
| Context | 8 of 8: needles at 10%, 50%, 90% and 33/66/90% depth in windows of 131,072 and 262,144 tokens (prompts of 126,976 and 258,048 tokens; thinking on, server default sampling, seed 1729, 4,096-token budget), each cold and then asked again and served from the cache (126,912 and 257,984 tokens cached) with the same answer. First token cold about 362 to 366 s and 1,137 to 1,145 s; from the cache about 3.4 s and 5.7 s | `single/summary.json` |
| Exactness | 68 of 68 cases (27 drafted equals serial: 24 cases, plus speculation, seed and coverage checks; 6 chunked prefill equals whole prefill: 5 prompts of 1,486 to 261,369 tokens at chunk sizes 256, 384 and the shipped 512, greedy, 48 tokens, cache off, plus the summary; 33 reply captures byte-identical to the pinned baseline; 2 prefix cache) | `exactness/exactness-cases.json` |
| Prefix cache | gate passes: a 2,468-token prompt asked again is served from the cache (2,400 tokens cached) with the same reply, thinking on. The exactness gate's live check: 84 checks, 0 failures, 31 hits | `single/summary.json`, `exactness/prefix-cache/check_prefix_cache.log` |
| Performance, performance smoke | 4 of 4; 3 of 3 (three serial replies with thinking on and a 4,096-token budget, 30.6 to 31.9 tok/s at the client; a timing smoke, not a speed comparison) | `performance/performance-cases.json`, `single/summary.json` |
| Quality | 5 of 5 suites at or above their floors | `quality/quality-cases.json` |

Quality: the Q200 suite with thinking on (`chat_template_kwargs: {"reasoning_effort": "on"}`), text suites at temperature 0 with `max_tokens` 32,768 and a 3,600 s request timeout; the BFCL tool-calling suite with `max_tokens` 8,192 and one thread; one GB10, one slot, one client worker on the same host. Q200 total 179 of 200 (89.5%); text 167 of 180. A reply that ran to the 32,768-token budget counts as incorrect, never dropped, re-run or given a larger budget; 8 text rows did (2 in gsm8k, 6 in ifeval). Recorded in `quality/q200/summary.json`, `quality/q200/release-summary.json` and `quality/quality-cases.json`; the grader's own summary before that rule is kept as `quality/q200/raw-kit-summary.json`. These are this pack's own regression baseline, not a claim of absolute quality. The floors equal this run's scores minus two cases per suite:

| Suite | Correct | Accuracy | Floor |
| --- | --- | --- | --- |
| gsm8k | 75 of 80 (2 of the misses ran to the budget) | 93.75% | 73 of 80 (91.25%) |
| humaneval | 40 of 40 | 100% | 38 of 40 (95%) |
| ifeval | 32 of 40 (6 of the misses ran to the budget) | 80.0% | 30 of 40 (75%) |
| hard_reasoning | 20 of 20 (graded by an independent manual review by a maintainer) | 100% | 18 of 20 (90%) |
| BFCL tool calling (20 hard cases) | 12 of 20 | 60.0% | 10 of 20 (50%) |

No figure from an earlier version, build or pack is carried here.

## 7. Limits of this release

- One GB10 only; there is no tensor-parallel profile.
- One slot; concurrent requests queue, and a cold full-window prompt holds the slot for about 19.5 minutes (section 5).
- Text only: images, video, audio and constrained JSON response formats are refused (section 3).
- Thinking off is not clean: the thinking-off reply carries reasoning text and a stray `</think>` (section 5). Thinking is off by default. Quality was measured with thinking on only.
- The memory margin at the full window is small (3.1 to 3.5 GiB free during the baseline runs), and `launch.sh start` can refuse to start until the file cache is freed (section 5).
- BFCL tool calling (hard-20) is the weakest result at 12 of 20; ifeval lost 6 replies to the 32,768-token budget.
- Two maintainer-granted exceptions are sealed in the receipt: thinking (expires 2026-12-31) and vision for a pack with no vision tower (expires 2027-03-31).
- The gates in section 6 are the qualification of this version; no 24-hour soak is part of them.
- The quality figures are one run on one GB10; a later version is held to them with a stated noise allowance, not to a fixed target.
- The engine was qualified for this exact commit and binary; a different engine build is not covered by this sheet.
- Install on an updated host: this recipe pins the NVIDIA driver packages `libnvidia-compute-580` and `nvidia-utils-580` at exactly `580.173.02-0ubuntu0.24.04.1` (`dependencies.lock.json`). On an Ubuntu 24.04 arm64 host whose packages have moved past that build, `./launch.sh install-deps` and `./launch.sh doctor` refuse with exit 20. The release notes give a single-node workaround; it skips the launcher's checks and is outside the release's qualification evidence. See "Known issues" in the [v0.1.1 release notes](../../releases/v0.1.1/NOTES.md).

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/ling-3.1/ling-3.1-flash-1019/v1/KNOWN_ISSUES.md`
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
