# Pack sheet: `cyberfrost-3.8-sage-rexl3-3.35`

Checked against model version `v1` in release `v0.1.3` (source commit `0c8bc43cf373`, engine release `engine-v0.1.3-0c8bc43c`, bundle asset `cyberfrost-3.8-sage-rexl3-3.35-v1-bundle.tar.gz`, sealed approval sha256 `981d35c9f774737ea9d24b5dba53e9416da1f0d7c157502e309bcea65f39b108`). Version 1 is the first version of this runtime and is new in `v0.1.3`. One choice is approved and part of the release: the default, `cyberfrost-3.8-single-262k`.

## 1. Identity

| Field | Value | Source |
| --- | --- | --- |
| Runtime id | `cyberfrost-3.8-sage-rexl3-3.35` | `profile.json` |
| Versions | `v1` (latest, v0.1.3) | `runtimes/qwen3.8/cyber-frost-3.8/LATEST` |
| Release tag | version 1 ships in `v0.1.3` (there is no per-runtime tag) | `releases/v0.1.3/manifest.json` |
| Release assets (version 1) | `cyberfrost-3.8-sage-rexl3-3.35-v1-approval.json`, `cyberfrost-3.8-sage-rexl3-3.35-v1-bundle.tar.gz`. The qualification evidence archive `cyberfrost-3.8-sage-rexl3-3.35-v1-evidence.tar.gz` is not a release asset: it is held on the development release, and the sealed approval pins its sha256 and every file in it | the release page; `approval.json` (`evidence_asset`) |
| What it serves | CYBER-FROST 3.8 (`Qwen4ExpForConditionalGeneration`, the Qwen3.8-Flash-Next architecture and engine family), EXL3 3.35 bits per weight, SAGE mixed-precision quantization; it speculates with the pack's own MTP head | `MODEL.json`, `model.lock.json`, `KNOWN_ISSUES.md` |
| Model repository and revision | `vcruz305/CYBER-FROST-3.8-SAGE-REXL3-3.35bpw` at `33be4227ff69c85cbb94f940abe7f22ad3b504e1` | `model.lock.json` |
| Draft or second model | none for the approved choice. The lock also pins `r0b0tlab/Qwen3.8-Flash-Next-DFlash2-EXL3-4.00bpw` at `846786415744bf40a70f6c274d1ff3a7c1fe64ee` for a choice that is not part of v0.1.3 (section 2); the default choice does not download it | `model.lock.json`, the bundle's `README.md` |
| Model licence | Qwen Community License 1.0 (the pack repository `LICENSE`) | `model.lock.json` |
| Download size | 27 target files, 89.27 GiB (95,849,986,806 bytes); free space needed: the remaining bytes plus a 1 GiB reserve. The bundle's `README.md` gives a locked total of 89.75 GiB because it also counts the 0.48 GiB draft pack that the default choice does not use | `model.lock.json`, the bundle's `README.md` |

## 2. Profiles

| Profile | Topology | Context window | Slots | Port | Binds | Ready within |
| --- | --- | --- | --- | --- | --- | --- |
| `cyberfrost-3.8-single-262k` (default) | single node | 262,144 tokens | 2 | 8899 | `127.0.0.1` | at most 600 s (`readiness_timeout_s`, a ceiling) after model verification |

Source: `profile.json`; the approved choice is the one profile named in the sealed `approval.json`. Served model name `CYBER-FROST-3.8-SAGE-REXL3-3.35bpw` (`GET /v1/models` lists it; send it as `model`). Arguments of the profile: `--slots 2`, `--max-model-len 262144`, `--spec mtp`, `--default-thinking template`, `--default-reasoning-effort low`, `--prefix-cache on`, `--prefix-cache-mib 12288`, `--queue-timeout 1500`, `--no-vision`; the profile also sets `REXL3_NGRAM_WARM=0` so the pack's 36.36 GiB n-gram table stays cold and its pages remain reclaimable while the full-window KV and prefix-cache storage are reserved (`KNOWN_ISSUES.md`, "Serving and memory contract"). A site file is optional; the profile's site fields are `http_host`, `http_port` and `model_dir` (an existing local copy of the pack, verified read-only). There is no tensor-parallel or two-node profile.

The sealed `profile.json` also lists a second choice, `cyberfrost-3.8-dflash-single-262k`. It is **not part of v0.1.3**: the approval does not name it and it is not qualified. Its DFlash2 draft was trained for Qwen3.8-Flash-Next, not for this model. Use the default choice.

## 3. What the server accepts

| Capability | Supported | Notes |
| --- | --- | --- |
| Text and streaming | yes | |
| Thinking (reasoning) | yes | Default when a request names no setting: **on**, at effort `low` (the profile follows the pack's chat template with `--default-thinking template --default-reasoning-effort low`; `GET /v1/models` reports `thinks: true`, `effort: low`). A request names `chat_template_kwargs.enable_thinking` and/or `chat_template_kwargs.reasoning_effort` (`low`, `medium`, `high`, `xhigh`); `chat_template_kwargs.preserve_thinking` is also accepted. No thinking-token budget is offered for this model. `GET /v1/models` lists only these `chat_template_kwargs` fields and no aliases; whether a top-level `reasoning_effort` is accepted: not stated. |
| Tools | yes | `tool_choice` `auto` and `none` only; at most 64 tools |
| Images | **no** | Refused. The pack ships the vision tower's tensors, but the engine does not build this architecture's tower; the profile runs `--no-vision` under a documented temporary exception that expires on 2027-01-31 |
| Prefix cache | yes | Automatic; pinned at 12,288 MiB (`--prefix-cache-mib 12288`): 24 entries and a reach of 365,568 tokens |
| Speculative decoding | yes | MTP head of the pack (`--spec mtp`), automatic |
| Video, audio, constrained JSON response formats (`json_schema`) | no | Refused by the server |

Source: `profile.json` `capabilities`, `KNOWN_ISSUES.md`, the receipt `single/summary.json` (`vision_exception`), and the `GET /v1/models` and boot records in the qualification evidence (`single/evidence/`, `performance/runs/`). Everything not listed as supported is refused (user guide section 5.10).

## 4. Sampling and output defaults

The v0.1.3 release notes give the model's sampling defaults as temperature 1.0, top-p 0.95 and top-k 20, from the pack's `generation_config.json` (the sealed `model.lock.json` pins that file by hash; its values are not repeated in the sealed tree). **Use these defaults for long generations**: greedy decoding (temperature 0) can loop until `max_tokens` (section 5). Model-card recommendations for this pack and how the server treats presence, frequency and repetition penalties: not stated.

A request that names no reasoning setting thinks at effort `low`, and thinking counts against the output budget, so set `max_tokens` generously; the profile's own smoke request uses 2,048 for a one-word reply (`profile.json`). The quality runs used effort `xhigh`, a text budget of 65,536 tokens and 8,192 for the tool-calling suite, at temperature 0 for the text suites and 0.001 for the tool-calling suite (`quality/q200/summary.json` in the evidence).

## 5. Behaviour to expect

- **Greedy decoding can loop on long generations.** At temperature 0 the model can repeat until `max_tokens`. In the quality run (greedy, effort `xhigh`, 65,536-token text budget) 6 of the 180 text replies ran to the budget with no answer and were counted incorrect: four IFEval prompts, one HumanEval prompt and one hard-reasoning prompt (`quality/q200/summary.json`, `budget_cut_rows`). Use the sampling defaults in section 4. The sealed `KNOWN_ISSUES.md` does not list this point; the v0.1.3 release notes do.
- A request that names no reasoning setting thinks, at effort `low`. Reasoning is returned separately in `reasoning_content`. A small `max_tokens` can be spent entirely inside the reasoning, leaving no answer (`finish_reason: length`; the bundle's `docs/TROUBLESHOOTING.md`).
- The profile's default (template thinking at low effort) was chosen for tool use: on a 12-turn tool loop the thinking-off line called the tool again after the twelfth result, while template thinking at low effort returned all twelve results and stopped (`KNOWN_ISSUES.md`, "Serving and memory contract"). The file states this as a serving requirement, not a quality claim.
- Image input is refused; see section 3.
- The two slots share one 262,144-token KV pool; the flags do not give 262,144 tokens per slot. One request may use the whole window. Requests are admitted in order; a request that does not fit waits for the one ahead and is failed with HTTP 503 after 1,500 s in the queue (`--queue-timeout 1500`; `KNOWN_ISSUES.md`, "Shared KV pool and queueing"). A cold prompt of about 260,000 tokens took about 234 s to prefill in the qualification run (`performance/runs/run-01-RESULT.json`).
- The prefix cache reaches 365,568 tokens, above the 262,144-token window. `KNOWN_ISSUES.md` states that this is not a claim that two full-window entries coexist. How many full-window documents the cache holds at once: not stated.
- The second choice in `profile.json`, `cyberfrost-3.8-dflash-single-262k`, is not part of v0.1.3 (section 2). The sealed `KNOWN_ISSUES.md` says "One profile" and does not mention it.
- The sealed `profile.json` still records `qualification.state: unverified`, and `KNOWN_ISSUES.md` is titled as a frozen configuration, not a qualification. The qualification of version 1 is its evidence archive (section 6). Where the two differ, the release evidence is the record of what was measured on this commit.

## 6. Measured behaviour

All figures are from version 1's qualification, recorded in the evidence archive `cyberfrost-3.8-sage-rexl3-3.35-v1-evidence.tar.gz` (pinned by the sealed approval; not a release asset, see section 1): one GB10 node, engine `0.1.3` at commit `0c8bc43cf373` (binary `rexl3-serve` sha256 `bd77b147…70bb`), 2 slots, window 262,144, vision off, prefix cache pinned at 12,288 MiB, NVIDIA driver packages at build 580.178.04 (`doctor/doctor-output.json`). The receipt is `single/summary.json` (the path named by the sealed approval). Every gate ran fresh on the v0.1.3 binary; nothing is carried from another version or model.

| Quantity | Value | Recorded in |
| --- | --- | --- |
| Generation speed, one stream (median of 3 runs, each the median of 5 streamed replies of 256 tokens, short context) | 63.0 tok/s (62.8 to 63.1); floor 53 | `performance/performance-cases.json`, `performance/runs/run-0N-RESULT.json` |
| Prompt speed at 50% window fill, about 131,000 tokens (median of 3, cold) | 1,146 tok/s (1,135 to 1,158); floor 860 | same |
| Prompt speed at full window fill, about 260,000 tokens (median of 3, cold) | 1,105 tok/s (1,103 to 1,113); floor 860 | same |
| Draft acceptance during the baseline runs (MTP) | 0.73 (0.731 to 0.735) | `performance/runs/run-0N-RESULT.json` |
| Lowest free memory during the baseline runs | 36.41 to 36.49 GiB, no swap | same |
| Boot to ready | 14.4 s first boot, 13.6 s restart | `lifecycle/lifecycle-cases.json` |
| Exactness | 68 of 68 cases pass: drafted output equal to serial output (24 cases, greedy and sampled), pinned replies (32), prefix cache, and chunked prefill equal to whole prefill at five prompt lengths from 1,500 to 261,120 tokens with 256- and 512-token chunks | `exactness/exactness-cases.json`, `exactness/chunked/chunked.json` |
| Prefix cache | check passes (86 checks, 0 failures, 40 cache hits counted) | `exactness/prefix-cache/check_prefix_cache.log` |
| Context | 8 of 8 needle cases pass at 131,072 and 262,144 tokens | `single/summary.json` |
| Release gates | all PASS, fresh: doctor 9/9, lifecycle 10/10, capability 5/5, boundary 5/5, vision 2/2 (under the exception), context 8/8, exactness 68/68, prefix cache 1/1, performance 4/4, performance smoke 3/3, quality 5/5 | `single/summary.json` |

Quality (the Q200 suite at reasoning effort `xhigh`, text budget 65,536 tokens, tool-calling budget 8,192; temperature 0, and 0.001 for tool calling; one client worker on a separate host). The total is 178 of 200 (89.0%). These are this pack's regression baseline, not a claim of absolute quality:

| Suite | Correct | Accuracy | Floor |
| --- | --- | --- | --- |
| gsm8k | 77 of 80 | 96.25% | 93.75% |
| humaneval | 39 of 40 (one prompt ran to the token budget and counts as incorrect) | 97.5% | 92.5% |
| ifeval | 34 of 40 (four prompts ran to the token budget and count as incorrect) | 85.0% | 80.0% |
| hard_reasoning | 19 of 20 (graded by an independent reviewer, as recorded in the evidence; one prompt ran to the token budget and counts as incorrect) | 95.0% | 85.0% |
| BFCL tool calling (20 hard cases) | 9 of 20 | 45.0% | 35.0% |

Recorded in `quality/q200/summary.json` and `quality/quality-cases.json`. Notes on this qualification:

- This is the runtime's first version, so its quality floors were set from this run: each suite's score minus two cases.
- The Q200 run and its requirements file used a later revision of the evaluation tools; the engine and the binary are those of the v0.1.3 engine commit `0c8bc43cf373` (v0.1.3 release notes).
- One qualifier row, prefix cache cold versus warm, was rerun with a corrected probe. The earlier attempts at that row are kept in the sealed evidence: the receipt's `superseded` field and `single/superseded_rows.jsonl`.

## 7. Limits of this release

- One GB10 node only; two slots sharing one 262,144-token window.
- One approved choice. `cyberfrost-3.8-dflash-single-262k` is listed in `profile.json` but is not part of v0.1.3 and is not qualified.
- No image input; the exception that allows shipping without it expires on 2027-01-31 (`KNOWN_ISSUES.md`, "Vision"; receipt `vision_exception`).
- Audio, video and constrained JSON response formats are refused (section 3).
- No thinking-token budget parameter for this model.
- Greedy decoding (temperature 0) can loop on long generations until `max_tokens`; use the sampling defaults (sections 4 and 5).
- The gates above are the qualification of version 1; no separate long-run (soak) result is part of them.
- The quality figures are one run on one node; a later release is held to the floors in section 6, not to a fixed target. The tool-calling suite scored 45%.
- Version 1 was qualified for engine commit `0c8bc43cf373` and its release binary; a different engine build is not covered by this sheet.
- The readiness deadline is 600 s; the measured boots are in section 6.
- Host packages: `dependencies.lock.json` sets `host_policy: known-working`. For the NVIDIA driver packages `libnvidia-compute-580` and `nvidia-utils-580` it records a tested floor of 580.142 (`min_driver`) and a highest qualified build of 580.178.04 (`max_driver`). Any 580-branch build from 580.142 is accepted; a build after 580.178.04 prints `DRIVER_NOT_YET_QUALIFIED` and continues, and is not covered by this release's qualification evidence. Another driver branch, or a build below 580.142, is refused (the bundle's `docs/SETUP.md`). See "Known issues" in the [v0.1.3 release notes](../../releases/v0.1.3/NOTES.md).

## 8. Where to look next

- The release's `KNOWN_ISSUES.md`: `runtimes/qwen3.8/cyber-frost-3.8/v1/KNOWN_ISSUES.md` (it does not list the greedy-decoding loop or the second choice; the v0.1.3 release notes do)
- The user guide: [USER_GUIDE](../USER_GUIDE.md)
- Reporting a problem: user guide section 11
