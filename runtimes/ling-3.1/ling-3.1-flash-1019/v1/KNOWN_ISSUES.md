# Ling-3.1-Flash-1019 EXL3 1.51 bpw: known issues

One choice, `ling31-single-262k`:
- **Hardware:** one GB10.
- **Context and slots:** 262,144 tokens, 1 slot.
- **Speculative decoding:** the pack's own MTP head (`--spec mtp`).
- **Prefix cache:** on (`--prefix-cache on`).
- **Prefill chunk:** 512 rows.
- **Memory floor:** `REXL3_MEMORY_FLOOR_GIB=2`.
- **Capabilities:** text, SSE, reasoning and tools. Images, video, audio and constrained JSON response formats are not served.

The model has no vision tower, so vision is not applicable. There is no tensor-parallel choice.

## Vision (temporary documented exception)

The profile passes `--no-vision`, because it declares `images=false` and the release tools require the switch for that. `tools/check_switches.py` has no state yet for a pack with no vision tower: it accepts `--no-vision` only as a does-not-fit proof or under a documented exception. A maintainer (am423) granted that exception on 2026-11-09, expiring 2027-03-31, in `scripts/gates/recipes/ling31-flash-1019-sage-1.51.json` (`capability_exceptions.vision`). It is tracked by teaching the switch check the no-tower case. Nothing is withheld: the pack has no tower to serve.

The pack `vcruz305/Ling-3.1-flash-1019-EXL3-SAGE-1.51bpw` is pinned at revision `437b03e6ddcc8b1ed653619aa8f266bbeeecd084`:
- 24 shards, 107.5 GiB in total;
- fingerprint `sha256:84c8b957a7268a0a721ea81280dadf96fa803d3ce2231991c2f0f6f1df6f2547`, the value in the engine's validated list.

There is no release requirements file. The release qualification checks the choice against this profile and its lock.

## Memory at 262,144 tokens

The weights take about 106 GiB of a GB10's memory. The plan reserves 2.0 GiB of boot overhead and a 2 GiB floor, then sizes the KV pool from what is left. So whether the full window fits depends on `MemAvailable` when the server starts.

- **Measured refusal.** One boot started with `MemAvailable` 113.8 GiB, of which 75.7 GiB was page cache from earlier work. The plan had 5.0 GiB free after the weights and the overhead. At chunk 512 the KV pool held 79,872 tokens, short of the window, so the engine refused the configuration by name. About 1.95 GiB was missing.
- **The same node, idle with little page cache:** `MemAvailable` was 117.2 GiB.
- **What to do:** start the server on a host with little page cache, for example right after boot or after the page cache is dropped. Don't start it right after reading another large pack. The refusal is fail-closed: the engine never starts at a window it cannot hold.
- **Unplanned memory**, measured on one boot at a 32,768-token window: 0.84 GiB at ready, and 1.20 GiB at the minimum during the first requests, after the planner's own memory reading. That is within the 2.0 GiB the plan reserves.

## Prefix cache

Prefix-cache entries are kept on disk, with a quota, rather than in GPU memory: the boot plan line reports `where=disk`. A cache hit restores the snapshot from disk. The recipe forbids a boot that plans zero entries (`event=ling31_prefix_plan entries=0`).

## Measured so far

- **Configuration check:** one GB10, driver 580.178.04, engine `dc884cdaa0f465d4c89b294055f88ecd0da5199b`:
  - five streamed chat completions at the full 262,144-token contract;
  - ready in 38.0 s;
  - median decode 25.79 tok/s.
- **Speed floor:** the recipe's decode floor is 20 tok/s. It has no prefill floors yet.
- **Release qualification:** not yet complete. These figures are not a release receipt.
