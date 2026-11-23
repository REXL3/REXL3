# Qwen3.8-27B EXL3 4.00 bpw with the DFlash2 draft: frozen configuration, not qualification

One normalized choice, `qwen38-single-262k`: one GB10, 262,144 context, 6 slots, native speculative decoding
(`--spec auto` with the DFlash2 draft pack), prefix cache on, vision on. Capabilities: text, SSE, reasoning, tools, images,
prefix cache. Audio, video and constrained JSON response formats (`json_object`, `json_schema`) are refused by the server.
There is no tensor-parallel choice. Configuration freeze does not imply throughput, long-context, multimodal or quality
qualification on the GB10; `qualification.state` is `unverified` and no measurement is transferred from earlier runs.

## Release status and current known issues

Released as version 3 in v0.1.0 and as version 4 in v0.1.1. v0.1.2 shipped version 4 again, unchanged.

Where the sections below say "not yet run", "pending" or "to be confirmed", they describe the entry before its release qualification. The receipts of that qualification are in the evidence archive of each release named above.

Known issues as of 2026-11-19. Each says what happens, who is affected and what to do:

- **NVIDIA driver: the 580 branch at build 580.142 or later; builds after 580.178.04 are accepted but not yet qualified.** This applies to every host. `libnvidia-compute-580` and `nvidia-utils-580` must both be the same upstream build; Ubuntu's `-0ubuntu0.24.04.1` and NVIDIA's CUDA-repository `-1ubuntu1` packages of one build count as the same driver. On a build later than 580.178.04, `./launch.sh doctor` and `install-deps` continue and print `DRIVER_NOT_YET_QUALIFIED` (`doctor` reports `driver_qualification: newer-than-tested`): 580.142 is the tested floor and 580.178.04 the highest build this runtime was qualified on, and results on a later build are not covered by the release's qualification evidence. The CUDA driver, runtime and cuBLAS versions that `rexl3-serve --version` reports must be the qualified ones or later on the same major version: a later one is accepted and `doctor` prints `NATIVE_NOT_YET_QUALIFIED`, and an older one or another major version stops `doctor` with `NATIVE_VERSION_MISMATCH`. A build below 580.142, or two different builds on one host, stops `doctor` with `SYSTEM_DEPENDENCY_VERSION_MISMATCH`; another driver branch (570, 590) is not accepted. When a driver change is needed and the configured repositories no longer offer the `580.178.04-0ubuntu0.24.04.1` package, `install-deps` stops with `DRIVER_PACKAGE_UNAVAILABLE` and installs nothing. What to do: stay on the 580 branch at 580.142 or later; if the package is unavailable, install any 580 build at or above 580.142 with the host package manager (both packages at the same build), reboot and rerun `./launch.sh doctor`.
- **A pending restart no longer stops the launcher.** When the host has a restart pending (`/var/run/reboot-required`, set by a kernel or system library update), `doctor` and `install-deps` print `REBOOT_PENDING` and continue. A driver upgraded without a restart is refused by name: `doctor` stops with `DRIVER_MODULE_MISMATCH` when the loaded NVIDIA kernel module is not the installed driver build, and after `install-deps` has changed the driver packages itself it stops with `DRIVER_MODULE_MISMATCH` too (with `REBOOT_REQUIRED` when the loaded module's build cannot be read and a restart is pending). What to do: restart the host, then rerun `./launch.sh doctor`.
- **An existing local copy of the packs cannot be used.** The lock has two packs (target and draft) and a site file names one model directory, so the launcher refuses `model_dir` in a site file. What to do: let the launcher download both packs (`./launch.sh setup --accept-model-download`).
- **Port 8889 is also the default port of the MiniCPM-V-4.7 and Ling-3.1 runtimes.** What to do: run one of them at a time on a host, or set `http_port` in a site file.

## Where the arguments come from

The argv is normalized from the documented serve line in `docs/examples/rexl3-serve.service` (`profile.json` `source` names
that file at an existing commit of the development repository and its raw SHA-256). Changes made while normalizing:

- `--queue-depth 64`, `--queue-timeout 300` and `--shutdown-grace 8` are dropped: each equals the server default and none is a
  serving-contract option (CONTRIBUTING section 6).
- `--spec auto` and `--prefix-cache on` are pinned as contract options. The unit relies on the defaults; section 6.1 requires
  both to be explicit.
- The host, port and the two pack directories are managed tokens. The port is the documented 8889.
- `--kv-dtype` is not passed: the Qwen engine keeps a bf16 KV cache and the flag only selects the DeepSeek and Nemotron caches.
- `--no-video` is not passed: video input is turned on only for a family whose tower encodes video, and the Qwen family does
  not (`ModelFamily::encodes_video` defaults to `false` and `families/qwen.rs` does not override it), so a video part is refused
  by name already. `capabilities.video` is `false` for that reason.
- `--no-vision` is not passed (CONTRIBUTING section 6.1, rule 15).

## Launching: managed models only, no `--site model_dir`

The pack has two roles, `target` and `draft`. `runtimes/template-v1/scripts/fetch_models.py` (the `model_dir` overlay branch,
`EXISTING_MODEL_ROLES_MISMATCH`) accepts an existing model copy only for a single-role lock, and the single-node `site_fields`
carry no draft directory. So the profile validates and the launcher can run it with the packs it downloads itself
(`@MODEL_DIR@` and `@DRAFT_DIR@`), but it cannot yet be pointed at existing local copies through `--site model_dir`. Mapping a
draft root in the adapter is a change to the shared scripts and is not made here.

## Sampling defaults and thinking

- The pack ships `generation_config.json` with temperature 1.0, top_p 0.95 and top_k 20, and the server reads all three from
  it (`GenDefaults::from_dir`), so no engine fallback is involved. The base model card also lists min_p 0.0, presence penalty
  0.0 and repetition penalty 1.0 for thinking mode; the server refuses non-zero penalties. The branch that adds
  `--default-temperature`, `--default-top-p` and `--default-top-k` has not landed, so these values are documented here and
  are not passed as flags.
- **Default reasoning effort is `low`** (`--default-reasoning-effort low` in the profile; the maintainers' decision of 2026-10-26: low is the recommended
  setting for this model). A request that names no `reasoning_effort` and does not set `enable_thinking: false` is served with thinking on at low effort (`enable_thinking: true` alone gets low as well).
  It is a default, not a fixed value: a request that names an effort (`low`, `medium`, `xhigh`) or `enable_thinking: false` keeps its own. Because requests that name nothing now think, the gate's reply budgets are raised in the recipe's `gate_params` (2,048 to 4,096 tokens) and the profile's smoke request has 2,048 tokens, as for the other always-thinking runtimes.
  The pack's template itself defaults to `xhigh`, and the engine's own default (`--default-thinking off`) would not think at all, so
  without this flag a client that sends nothing gets a different model behaviour from the recommended one. Earlier measurement, with the
  template's own `xhigh` default (`--default-thinking template`) on one GB10 node: every baseline stage passed except "NIAH 100% + tool"
  (no `report_codes` call): that stage fills the window to 261,696 tokens and leaves 256 tokens of reply, and thinking spent them before the
  tool call. Decode was 63.0 tok/s with thinking on. Low effort thinks much less; whether it leaves enough of those 256 tokens is what the
  GB10 receipt of the change that introduced this flag shows. If that stage fails, the fix is a window-aware reply budget, not a different
  default.

## Measured once on one node; not yet on a second

- `speed_floors` in `scripts/gates/recipes/qwen38-27b-dflash2-exl3-4.00.json` are 80% (rounded down) of what the baseline
  gate measured on one GB10 node before the fused prefill GEMM: decode 44.7 tok/s (sampled), prefill 773 tok/s at 50% of
  the window and 597 at 100%. The fused prefill measured 827 and 626, so the floors hold with or without it. The v0.1.0 and
  v0.1.1 qualifications re-measured them on the release binaries.
- The full baseline gate passed on that node with this profile's contract (6 slots, 262,144 tokens, vision tower loaded,
  `--prefix-cache on`): the cache stayed engaged (stage 5 cached 7,958 tokens, stage 6 cached tokens rose with each turn)
  and the lowest free memory seen was 68 GiB.
- The `expect_log` and `forbid_log` markers matched a live boot (gate stages 1 and 5), and the vision stage passed.
- `readiness_timeout_s` (600) is a chosen ceiling. A cold boot measured about 5 s on a GB10 node with the pack on local disk.

## Model identity

Both packs are public. The lock pins the Hugging Face revisions
`3f1771b8c21f83cbb8e82169559ced9f38ca04e5` (target) and `265b5240592907d2d55ff0dc4d5f66569692604d` (draft). Every file digest
is the SHA-256 of the bytes of a read-only copy of the pack. The two safetensors shards of the target and the draft's
`model.safetensors` also equal the Hugging Face LFS object ids and the shard digests in `crates/model/src/fingerprint.rs`; the
small files were checked against the git blob ids of the same revisions. The pack fingerprints equal the engine's validated
pair, so `--allow-unvalidated-pack` is never needed. Only `README.md`, `LICENSE` and `.gitattributes` may sit beside the
locked files unlisted.

## Dependencies

`dependencies.lock.json` and `requirements-runtime.lock` are those of the DeepSeek runtime, which uses the same launcher
scripts and Hugging Face client, with one change: `libnccl2` is not listed, because the single-node engine does not link or
load it (it is loaded only by the tensor-parallel path). They have not been rehearsed on a clean host for this runtime.
