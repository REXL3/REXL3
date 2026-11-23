# Ling-3.1-Flash-1019 EXL3 1.51 bpw: known issues

One choice, `ling31-single-262k`:
- **Hardware:** one GB10.
- **Context and slots:** 262,144 tokens, 1 slot.
- **Speculative decoding:** the pack's own MTP head (`--spec mtp`).
- **Prefix cache:** on (`--prefix-cache on`).
- **Prefill chunk:** 512 rows.
- **Memory floor:** `REXL3_MEMORY_FLOOR_GIB=1.5`.
- **Capabilities:** text, SSE, reasoning and tools. Images, video, audio and constrained JSON response formats are not served.
- **Thinking:** on by default (`--default-reasoning-effort on`). A request can turn it off; see "Thinking off" below before you do.

The model has no vision tower, so vision is not applicable. There is no tensor-parallel choice.

## Release status and current known issues

Released as version 1 in v0.1.1. v0.1.2 shipped version 1 again, unchanged.

Where the sections below say "not yet run", "pending" or "to be confirmed", they describe the entry before its release qualification. The receipts of that qualification are in the evidence archive of each release named above.

Known issues as of 2026-11-19. Each says what happens, who is affected and what to do:

- **NVIDIA driver: the 580 branch at build 580.142 or later; builds after 580.178.04 are accepted but not yet qualified.** This applies to every host. `libnvidia-compute-580` and `nvidia-utils-580` must both be the same upstream build; Ubuntu's `-0ubuntu0.24.04.1` and NVIDIA's CUDA-repository `-1ubuntu1` packages of one build count as the same driver. On a build later than 580.178.04, `./launch.sh doctor` and `install-deps` continue and print `DRIVER_NOT_YET_QUALIFIED` (`doctor` reports `driver_qualification: newer-than-tested`): 580.142 is the tested floor and 580.178.04 the highest build this runtime was qualified on, and results on a later build are not covered by the release's qualification evidence. The CUDA driver, runtime and cuBLAS versions that `rexl3-serve --version` reports must be the qualified ones or later on the same major version: a later one is accepted and `doctor` prints `NATIVE_NOT_YET_QUALIFIED`, and an older one or another major version stops `doctor` with `NATIVE_VERSION_MISMATCH`. A build below 580.142, or two different builds on one host, stops `doctor` with `SYSTEM_DEPENDENCY_VERSION_MISMATCH`; another driver branch (570, 590) is not accepted. When a driver change is needed and the configured repositories no longer offer the `580.178.04-0ubuntu0.24.04.1` package, `install-deps` stops with `DRIVER_PACKAGE_UNAVAILABLE` and installs nothing. What to do: stay on the 580 branch at 580.142 or later; if the package is unavailable, install any 580 build at or above 580.142 with the host package manager (both packages at the same build), reboot and rerun `./launch.sh doctor`.
- **A pending restart no longer stops the launcher.** When the host has a restart pending (`/var/run/reboot-required`, set by a kernel or system library update), `doctor` and `install-deps` print `REBOOT_PENDING` and continue. A driver upgraded without a restart is refused by name: `doctor` stops with `DRIVER_MODULE_MISMATCH` when the loaded NVIDIA kernel module is not the installed driver build, and after `install-deps` has changed the driver packages itself it stops with `DRIVER_MODULE_MISMATCH` too (with `REBOOT_REQUIRED` when the loaded module's build cannot be read and a restart is pending). What to do: restart the host, then rerun `./launch.sh doctor`.
- **A request that turns thinking off gets reasoning text and a stray `</think>` in its answer.** A request that names no thinking setting thinks (the profile passes `--default-reasoning-effort on`) and gets its reasoning back separately, so it is not affected. Who is affected: a request that sends `chat_template_kwargs.enable_thinking` false or `reasoning_effort` `off`. What to do: leave thinking on, or strip everything up to the last `</think>` from the content (see "Thinking off" below; the exception for it expires on 2026-12-31).
- **The server refuses to start at the 262,144-token window when the host is short of free memory.** The weights leave little room, and an idle GB10 is close to the line. The engine names the refusal and never starts at a smaller window. What to do: start on an idle host, and wait a few seconds after a large job ends. "Memory at 262,144 tokens" below gives the `MemAvailable` the boot needs and the measurements.
- **Port 8889 is also the default port of the Qwen3.8-27B and MiniCPM-V-4.7 runtimes.** What to do: run one of them at a time on a host, or set `http_port` in a site file.

## Vision (temporary documented exception)

The profile passes `--no-vision`, because it declares `images=false` and the release tools require the switch for that. `tools/check_switches.py` has no state yet for a pack with no vision tower: it accepts `--no-vision` only as a does-not-fit proof or under a documented exception. A maintainer (am423) granted that exception on 2026-11-09, expiring 2027-03-31, in `scripts/gates/recipes/ling31-flash-1019-sage-1.51.json` (`capability_exceptions.vision`). It is tracked by teaching the switch check the no-tower case. Nothing is withheld: the pack has no tower to serve.

## Thinking off (temporary documented exception)

**The default is thinking on.** The profile passes `--default-reasoning-effort on`. A request that names no thinking setting (neither `chat_template_kwargs.enable_thinking` nor `reasoning_effort`) is served exactly as if it had sent `enable_thinking` true: the reasoning comes back separately, in `reasoning_content`, and the answer is clean. This is the model's recommended setting: the pack's chat template thinks unless it is told not to, and the base model card says to set `enable_thinking` to true. `GET /v1/models` shows the default under `reasoning.default`.

- **Budget for the reasoning.** Reasoning tokens count against `max_tokens`. A request with a small `max_tokens` can end in its reasoning with `finish_reason` `length` and an empty answer.
- **Earlier builds** (engine `75031112`, runtime version 1) had no such flag in the profile, so the server's own default applied: thinking off. A request that named no thinking setting got the thinking-off behaviour below. On those builds, send `chat_template_kwargs: {"enable_thinking": true}` with every request.

**Thinking off is still a known issue.** With `chat_template_kwargs.enable_thinking` false, or `reasoning_effort` `off`, the model still writes reasoning text, followed by a stray `</think>`, into the answer. The engine renders the closed think block that the chat template asks for, so this is how the model behaves; the engine is not at fault. If you turn thinking off, strip everything up to the last `</think>` from the content.

**Measured** on one GB10, 2026-11-19, this profile's arguments, greedy (temperature 0), `max_tokens` 2048, one request each. The prompt is the release qualifier's: "Compute 17 * 23. Give only the final integer."

| Request | Answer (`content`) | Reasoning returned separately | Completion tokens (reasoning) |
|---|---|---|---|
| no thinking setting | `391` | yes, 206 characters | 113 (109) |
| `enable_thinking` true | `391` | yes, 206 characters | 113 (109) |
| `enable_thinking` false | 235 characters: reasoning text, `</think>`, then `391` | no | 78 (0) |
| `reasoning_effort` `off` | the same 235 characters | no | 78 (0) |

The profile's own smoke request ("Reply with the single word: ready", `max_tokens` 256) answered `ready` in 37 completion tokens, 35 of them reasoning.

**The exception.** A maintainer (am423) granted a thinking exception on 2026-11-11, expiring 2026-12-31. It is recorded in `scripts/gates/recipes/ling31-flash-1019-sage-1.51.json` (`capability_exceptions.thinking`) and in the release requirements `thinking` block. Under it, the qualifier runs `capability/thinking_excepted`, which records the thinking-off reply rather than requiring it. Every other check stays strict. The thinking-on default does not remove the need for it: the qualifier's thinking row still asks for a clean thinking-off reply.

The pack `vcruz305/Ling-3.1-flash-1019-EXL3-SAGE-1.51bpw` is pinned at revision `437b03e6ddcc8b1ed653619aa8f266bbeeecd084`:
- 24 shards, 107.5 GiB in total;
- fingerprint `sha256:84c8b957a7268a0a721ea81280dadf96fa803d3ce2231991c2f0f6f1df6f2547`, the value in the engine's validated list.

The release requirements file is `evals/releases/ling31-flash-1019-sage-1.51-v1/requirements.json`. The release qualification checks the choice against it, this profile and its lock.

## Memory at 262,144 tokens

The weights take about 106 GiB of a GB10's memory (106.25 GiB of device regions). When the server starts, the plan reads `MemAvailable`, takes out the weights, 2.0 GiB of boot overhead and the 1.5 GiB floor, and sizes the KV pool from what is left, to the byte. So whether the full window fits depends on `MemAvailable` at that moment.

- **What the full window needs.** 114.70 GiB of `MemAvailable` when the plan reads it. The server uses up to about 1.2 GiB itself before that reading (its process, the CUDA context, the pack's records). So start it with about **115.9 GiB** of `MemAvailable` or more.
- **Below that, the boot is refused by name.** The engine never starts at a smaller window. The message gives the pool it could have held, for example `the KV pool holds 258816 tokens, below the 262144-token window`.
- **Idle GB10s are close to the line.** Three idle GB10s read 115.7 to 117.4 GiB before the server starts. A node at the low end of that range can still be refused. A desktop session, another server or a build that has just finished can take the difference.
- **What to do:** start the server on an idle host, and wait a few seconds after a large job ends, because `MemAvailable` takes time to recover. Page cache is not the cause: the kernel counts reclaimable cache as available.
- **Memory while serving.** With the full window in use, the lowest `MemAvailable` we measured was 2.24 GiB, on a boot that read 115.35 GiB at the plan (see the table). A boot nearer the 114.70 GiB line has less: expect about 1.5 to 2 GiB, the plan's floor being 1.5 GiB.
- **Earlier builds** (engine `75031112`, runtime version 1) planned from whole GiB with a 2 GiB floor. They needed 115.25 GiB at the plan, and about one idle boot in seven was refused.

Measured on one GB10, 2026-11-18, this profile's arguments and environment, the node otherwise idle. Where a row says "held", a helper process held that much anonymous memory to lower `MemAvailable`. "Before start" is `MemAvailable` just before the server process starts; "at the plan" is the `mem_available_gib` of the server's `event=memory_view` line. The lowest reading is from a 0.1 s trace from ready to exit. Loaded boots served one 261,023-token prompt and then the same prefix again from the cache.

| Held | Before start (GiB) | At the plan (GiB) | KV pool (tokens), loaded | Lowest `MemAvailable` (GiB) |
|---|---|---|---|---|
| none | 117.31 | 116.33 | 262,144, no | 3.26 |
| none | 116.72 | 116.33 | 262,144, yes | 3.21 |
| none | 116.68 | 115.86 | 262,144, no | 3.52 |
| 0.79 GiB | 116.28 | 115.35 | 262,144, yes | 2.24 |
| 1.02 GiB | 115.83 | 114.91 | 262,144, no | 2.38 |
| 1.26 GiB | 115.89 | 114.67 | refused (pool 258,816) | - |
| 1.52 GiB | 115.75 | 114.58 | refused (pool 250,368) | - |
| 1.72 GiB | 115.57 | 114.43 | refused (pool 236,544) | - |
| 1.98 GiB | 115.38 | 114.45 | refused (pool 237,824) | - |

Nine boots: five planned the full window and four, all below 114.70 GiB at the plan, were refused. The server used 0.39 to 1.22 GiB between its start and the plan's reading.

## Prefix cache

Prefix-cache entries are kept on disk, with a quota, rather than in GPU memory: the boot plan line reports `where=disk`. A cache hit restores the snapshot from disk. The recipe forbids a boot that plans zero entries (`event=ling31_prefix_plan entries=0`).

## Measured so far

- **Configuration check:** one GB10, driver 580.178.04, engine `dc884cdaa0f465d4c89b294055f88ecd0da5199b`:
  - five streamed chat completions at the full 262,144-token contract;
  - ready in 38.0 s;
  - median decode 25.79 tok/s.
- **Speed floors:** the recipe sets a decode floor of 20 tok/s and prefill floors of 250 tok/s at 50% and 150 tok/s at 100% of the 262,144-token window (recorded 2026-11-11).
- **Release qualification:** completed for v0.1.1 (version 1). The figures above are from before it and are not its receipt.
