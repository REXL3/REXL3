# DSV4.1 SAGE 1.55 bpw (TP2) — frozen configuration, not qualification

This entry is the two-node DeepSeek-V4.1-Flash-1017 runtime,
`dsv41-tp2-1m` (two-node TP=2, the second shipped profile on the REXL3
1.55bpw pack). It is an independent runtime unit from the single-node entry
`runtimes/deepseek-v4.1/deepseek-v4.1-flash-1017/` (`dsv41-single-262k`,
runtime `dsv41-flash-1017-sage-1.55`), so the single node can release and be
gated without the two-node group and vice versa. This file covers the two-node
entry only.

The two-node profile serves the model's full 1,048,576-token window with q4,
native `spec=auto`, text, SSE, reasoning, tools and prefix cache, with lossless
`REXL3_DSV41_MOE_TOPP=1.0` and the approved FAST_MOE/SPEC_LEARN switches.
Vision (the owner's decision of 2026-11-01, "vision off for deepseek"): this
profile ships `--no-vision`. The tower is implemented and image input is encoded
exactly as the upstream reference (feature cosine 0.99996), but colour/shape
accuracy on this quantized pack is unreliable (0 of 6 solid colours correct on
the two-node probe), so images are refused under a temporary exception in the
recipe (`capability_exceptions.vision`, granted by a maintainer on 2026-11-01,
expires 2026-12-31), and the gate reports the vision stage as EXCEPTION. Vision
returns when the pack passes the colour probe. Audio, video and constrained JSON
response formats remain unsupported. Slots remain 1. Configuration freeze does
not imply full-window, throughput, multimodal or quality qualification.

## Release status and current known issues

Released as version 1 in v0.1.1. v0.1.2 shipped version 1 again, unchanged. Version 1 pins the earlier pack revision `cba79c67`; this entry pins `c40cf25c` (see "Pack revision replacement").

Where the sections below say "not yet run", "pending" or "to be confirmed", they describe the entry before its release qualification. The receipts of that qualification are in the evidence archive of each release named above.

Known issues as of 2026-11-19. Each says what happens, who is affected and what to do:

- **NVIDIA driver: the 580 branch at build 580.142 or later; builds after 580.178.04 are accepted but not yet qualified.** This applies to every host, leader and follower. `libnvidia-compute-580` and `nvidia-utils-580` must both be the same upstream build, and every host must run that build; Ubuntu's `-0ubuntu0.24.04.1` and NVIDIA's CUDA-repository `-1ubuntu1` packages of one build count as the same driver. On a build later than 580.178.04, `./launch.sh doctor` and `install-deps` continue and print `DRIVER_NOT_YET_QUALIFIED` (`doctor` reports `driver_qualification: newer-than-tested`): 580.142 is the tested floor and 580.178.04 the highest build this runtime was qualified on, and results on a later build are not covered by the release's qualification evidence. The CUDA driver, runtime and cuBLAS versions that `rexl3-serve --version` reports must be the qualified ones or later on the same major version: a later one is accepted and `doctor` prints `NATIVE_NOT_YET_QUALIFIED`, and an older one or another major version stops `doctor` with `NATIVE_VERSION_MISMATCH`. A build below 580.142, or two different builds on one host, stops `doctor` with `SYSTEM_DEPENDENCY_VERSION_MISMATCH`; another driver branch (570, 590) is not accepted. `setup` and `start` check every host from the leader: a host that is not accepted stops them with `RANK_IDENTITY_FAILED` and a reason: `NATIVE_VERSION_MISMATCH` when the CUDA driver, runtime or cuBLAS version its `rexl3-serve --version` reports is older than the qualified one or of another major version (checked first), `IDENTITY_PROBE_FAILED` when a 580 package has never been installed on it (for example a host on another driver branch: the package query itself fails), `DEPENDENCY_MISSING` when one was installed and has been removed, and `DEPENDENCY_VERSION_MISMATCH` when its 580 build is below 580.142 or its two packages are different builds; hosts on different builds stop them with `DRIVER_VERSION_MISMATCH_ACROSS_RANKS`, and a later build on every host prints `DRIVER_NOT_YET_QUALIFIED`. When a driver change is needed and the configured repositories no longer offer the `580.178.04-0ubuntu0.24.04.1` package, `install-deps` stops with `DRIVER_PACKAGE_UNAVAILABLE` and installs nothing. What to do: stay on the 580 branch at 580.142 or later; if the package is unavailable, install any 580 build at or above 580.142 with the host package manager (both packages at the same build), reboot and rerun `./launch.sh doctor`.
- **NCCL (`libnccl2`): version 2 at 2.30.7 or later, the same on both hosts.** Qualified on 2.30.7 only. A later version, such as 2.32.6, is accepted and `doctor` prints `DEPENDENCY_NOT_YET_QUALIFIED`: this runtime has no qualification evidence for 2.32.6. A version below 2.30.7 or a `+cuda12` build stops `doctor` with `SYSTEM_DEPENDENCY_VERSION_MISMATCH`. Hosts on different versions stop `setup` and `start` with `DEPENDENCY_VERSION_MISMATCH_ACROSS_RANKS`. What to do: upgrade both hosts together, or hold the package on both.
- **A pending restart no longer stops the launcher.** When the host has a restart pending (`/var/run/reboot-required`, set by a kernel or system library update), `doctor` and `install-deps` print `REBOOT_PENDING` and continue. A driver upgraded without a restart is refused by name: `doctor` stops with `DRIVER_MODULE_MISMATCH` when the loaded NVIDIA kernel module is not the installed driver build, and after `install-deps` has changed the driver packages itself it stops with `DRIVER_MODULE_MISMATCH` too (with `REBOOT_REQUIRED` when the loaded module's build cannot be read and a restart is pending). What to do: restart the host, then rerun `./launch.sh doctor`.
- **The HTTP endpoint is on the network and has no authentication.** The leader listens on its address from the site file, not on `127.0.0.1`, and the server has no API key, so anyone who can reach the port can use it. What to do: run it only on a private, trusted network, or behind a firewall or an SSH tunnel.
- **Images are refused.** The profile runs `--no-vision` under a temporary exception that expires on 2026-12-31, because this quantized pack answers a solid-colour probe wrongly (see the introduction above). There is no workaround in this runtime.
- **At temperature 0 the model can repeat a phrase until `max_tokens`.** What to do: keep the sampling defaults for long reasoning and set `max_tokens` (see "Greedy decoding can loop" below).
- **`--kv-dtype int8` does not select an 8-bit cache.** Off-profile: this model has `q4` and `fp16` caches only, and with `int8` it runs `q4` without an error (see "DeepSeek-V4.1 KV formats" below). What to do: keep the profile's `q4`.

## Pack revision replacement

The lock pins revision `c40cf25cc3148bae95ac27483505c883840bf8a7` because the
earlier `cba79c67582950ea44a4a449b3bb85b800748a35` revision is no longer
available from the Hub. All 24 locked files were independently verified to have
identical sizes and SHA-256 digests at the replacement revision, so the payload
manifest and engine fingerprint are unchanged. Existing external `model_dir`
copies remain usable after the same full payload verification. Release-evidence
carry-forward accepts the earlier revision only through the reviewed, exact
same-files revision alias; no other pack-record difference is excused.

## DeepSeek-V4.1 KV formats and `int8` fall-through

DeepSeek-V4.1 implements `q4` and `fp16` KV caches only. Both shipped profiles
select `q4`, which is the shipped, qualified cache; `fp16` is the other
implemented format. The `int8` choice in the shared `--kv-dtype` help text
applies to Nemotron's `REXL3_KV_DTYPE` path, not to DeepSeek-V4.1.

On DeepSeek-V4.1, `--kv-dtype int8` currently fails the family-specific parse
without surfacing an error, leaves `REXL3_DSV41_KV` unset, and therefore runs
the engine's default `q4` cache. It does not select an `int8` cache. The shared
build-info path still records `REXL3_KV_DTYPE=int8`, so `system_fingerprint`
can contain `kvi8q4`: `kvi8` records the shared setting while `q4` records the
cache that DeepSeek-V4.1 actually uses.

This fall-through explains @Weschera's single-node GB10 report in
REXL3/REXL3#111. Custom needle recall at 65,283 prompt tokens, Spark-Bench v7
`long_context` quality, prefill on a 33,015-token prompt across `n=3` runs per
flag, and decode across three short workloads matched between `--kv-dtype q4`
and `--kv-dtype int8` on recall, quality, memory and speed because both flags
executed the `q4` cache. Those measurements are not supporting evidence for an
`int8` implementation or its qualification. The Spark-Bench v7 result is not a
Q200 result.

## Two-node TP=2 profile is the second shipped profile (REXL3 1.55bpw pack)

The two-node TP=2 profile (`dsv41-tp2-1m`) ships on the REXL3 1.55bpw pack
(`vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw` at revision
`c40cf25cc3148bae95ac27483505c883840bf8a7`: 320.3 GB, 17 shards, EXL3 with a
K chosen per tensor; routed experts use K 1 to 4, the trunk linears K 5 and 6).
It replaced the 1.62bpw pack on 2026-10-31 (maintainers' decision: it should
perform better). It splits the expert slab and KV across ranks (≈ half per
rank) and needs a RoCE link between the two GB10s. It is the same pack as the
single-node default: this entry's lock pins it once, as its `target` role. This
entry keeps its own runtime_id, `dsv41-flash-1017-sage-rexl3-1.55`, because its
recipe pins its own window (1,048,576), so the TP2 gate receipt pairs the recipe
`scripts/gates/recipes/dsv41-flash-1017-sage-rexl3-1.55.json` with the entry
profile via `--choice dsv41-tp2-1m` and `--base-url`/`--server-log`. The engine
lists the pack as validated (`crates/model/src/fingerprint.rs`, fingerprint
`sha256:c34045423c126e40d051943ff46bd57a67c07350d8b1743ed5185415db61351c`).
Vision is off on this profile (`--no-vision`, see above), so the tower is not
loaded and the gate's vision stage reports EXCEPTION.

Not yet done on this pack:

- Measurements before the change's receipt: speed, memory, quality and the 1M
  window were unmeasured on this pack at two nodes. The 1.62bpw pack's last 262K receipt
  (1,693 / 1,292 tok/s prefill, 28.3 tok/s decode)
  and the 1M runs below are history, not evidence for this pack.

## The two-node profile serves the full 1,048,576-token window

`dsv41-tp2-1m` serves `--max-model-len 1048576`, the model's advertised window.
Measured on the **1.62bpw pack** this profile served before (not on the REXL3
1.55bpw pack), with the baseline gate on two GB10s, one binary for both windows and one run per
window, run directly and not through `pre_pr.sh`,
so these are measurements and not a receipt:

| | 262,144 | 1,048,576 |
|---|---|---|
| gate result | PASS (315 s) | PASS (2,463 s) |
| decode | 28.9 tok/s | 29.0 tok/s |
| prefill at 131K tokens | 1,817 tok/s | about 1,870 tok/s (NIAH on the 1M server) |
| prefill at half the window | 1,817 tok/s (131,092 tokens) | 935 tok/s (524,292 tokens, 561 s) |
| prefill at the full window | 1,372 tok/s (261,976 tokens) | 567 tok/s (1,048,396 tokens, 1,848 s) |
| lowest MemAvailable | 47.8 GiB (leader) | 44.3 GiB (leader), 45.8 GiB (follower) |
| speculative accept rate | 0.60 | 0.60 |

- The speed floors (decode 18, prefill 700 at half the window, 400 at the full
  window) held at 1M with margin on the 1.62bpw pack and are unchanged for the
  REXL3 1.55bpw pack.
- A prompt that fills the window takes about 31 minutes to its first token. A
  shorter prompt costs nothing extra.
- When `--max-model-len` is above 524,288 the server raises its request,
  prefill, decision and queue timeouts and the TP peer timeout to 14,400 s
  (`crates/server/src/main.rs`), so the profile's `--tp-peer-timeout 60` is
  raised to 14,400 s at this window: a stalled peer is declared lost only after
  that long.
- Not measured on this profile: anything on the REXL3 1.55bpw pack (its 1M
  receipt and its Q200 quality baseline are pending), the 256K to 512K context
  ladder through the qualification tool, the Q200 quality lane at 1M, and a soak. The follower's
  memory was read once after the run (the gate samples the leader only).

## Prefix-cache snapshot floor and post-ready reserve

Commit c7566b972 (cherry-picked from 1fee310ff) decouples the snapshot-capture
floor (0.25 GiB default) from the boot floor and adds a 1.2 GiB post-ready
reserve to the full-residency election, so captures are not silently skipped
when boot headroom is tight (prefix-cache stage 5/6 can fill while the reserve
stays free). The capture floor no longer reads `REXL3_DSV41_FLOOR_GIB`.

## Existing model copies

The model tree is read-only: no moves, downloads, chmod, cache or marker writes.
Every locked file is size/full-SHA verified on every preparation. Only the exact
irrelevant root documents `README.md`, `LICENSE`, `.gitattributes` may be extra;
they too must be regular, non-symlink, single-link files. The one tolerated
directory is the Hugging Face download bookkeeping `.cache/huggingface/` left by
`hf download --local-dir` (also on copies of the managed directory): it may hold
only directories and regular single-link files, is never hashed or read by the
server, and anything else under `.cache` is refused. Extra tensors, sidecars,
indexes and other directories fail closed, and the refusal names its code
(for example `EXISTING_MODEL_IDENTITY_FAILED: UNLISTED_MODEL_DIRECTORY`). All ancestors are no-follow traversed.
The operator must keep admitted trees unchanged while serving: a preflight hash
is a point-in-time observation, not isolation against a hostile same-UID writer.

## TP=2 adapter

An explicit site `nodes` mapping must contain exactly `leader` and `follower`.
Each node supplies literal IPv4 `address`, `fabric_address`, `user`, `model_dir`,
and `binary_dir`. The HTTP port, `tp_port`, `fabric_interface` and a non-empty `hca`
rail list must also be stated; the sealed cluster file's example network is never
used as a default (omitting one is refused, naming the field). Serving
flags/environment cannot be overridden. The leader binds
to its management address (restrict network access externally).

The adapter delegates validate/check/up/status/down to the pinned native cluster
binary. All ranks must pass full payload, binary, native-version and dependency
identity checks before up. A SHA-verified transmitted bundled helper uses isolated
Python, strict existing SSH host trust, and no forwarded agent/environment secrets.
Default SSH keys must already be provisioned; the adapter never edits trust files.
Native scripts use a clean environment and unique per-attempt state directories.
Readiness requires exact rank argv/process/socket ownership plus API model/limits
and an actual smoke. A health 200 alone is not readiness. Partial starts delegate
to native down; an unsuccessful cleanup leaves the active marker for intervention.

## Reasoning-effort names and sampling defaults

**Reasoning effort.** The engine's native renderer (`crates/server/src/formats/deepseek_v41.rs`) accepts
`reasoning_effort` as a top-level field or as `chat_template_kwargs.reasoning_effort`, with one of these names:
`none` and `off` turn thinking off; `minimal`, `low`, `medium`, `high`, `max` and `xhigh` turn it on. Any other
name is refused with HTTP 400. Each name maps to the effort number the `Reasoning Effort: N` prefix renders:
`minimal` and `low` = 50, `medium` = 65, `high` = 75, `max` and `xhigh` = 100. The request validator refuses
non-name values, an integer included, with HTTP 400; the renderer's own mapping would clamp an integer 1–100 if
one reached it. Upstream's reference encoder (DeepSeek's `encoding/encoding.py`; upstream ships no Jinja
template) accepts only an integer 1–100 or `low`/`high`/`max` (50/75/100). The extra names are OpenAI-style
aliases of this runtime's control table, and `medium` = 65 is this runtime's own choice; the pack's
`chat_template.jinja` (for tools outside REXL3) follows upstream and raises on the extra names, and it does not
render tools. Thinking is off by default. With thinking on and no effort given, the effort is 75 (upstream's
`high`). The effort prefix renders only in thinking mode, at the start of the conversation.

**Sampling.** A request that leaves out `temperature`, `top_p` or `top_k` gets temperature 1.0 and top_p 0.95 from
the pack's `generation_config.json` (the file the model lock pins), and top_k 20, the server's fallback (`GenDefaults`
in `crates/server/src/api.rs`). Neither the pack nor upstream sets top_k. Where the pack and upstream differ (upstream
recommends top_p 1.0), the pack's file wins: the owner's rule of 2026-11-02 takes a pack's own config first and uses
the base model only where the pack sets nothing. Requests can set these fields explicitly, which overrides the defaults.

**Greedy decoding can loop.** At temperature 0 the model sometimes repeats a short phrase in its reasoning until
`max_tokens`. Q200 text180 was run on the two-node profile at effort `low` on 2026-11-02:
- **Greedy:** 25 of 180 prompts looped, with an 8-word run repeated 50 to 1,307 times. Most were instruction-following
  prompts; 22 of the 40 hit the 8,192-token cap.
- **The defaults above** (seed 1234): 8 of 180 looped, all instruction-following.

This is the model's behaviour, not this pack or engine: the 1.62bpw pack loops on the same prompts as often, and
turning off speculative decoding or using an fp16 KV cache does not stop it. For long reasoning, use the defaults
rather than temperature 0, and set `max_tokens` to bound a reply.

## Unverified boundaries

CPU fixture tests are not GPU or native-Rust execution evidence. Native compilation,
real SSH/host keys, all-rank hashing of the model, dependency/native identity
on actual hosts, cleanup under actual link failure, clean-host rehearsal, full-window
context (1M two-node) and
multimodal/quality/throughput qualification belong to the hardware lane.
No measurements from earlier profiles are transferred to this frozen configuration.
The two-node GB10 receipt on the REXL3 1.55bpw pack is pending (the 1.62bpw
pack's passed at 1,693/1,292/28.3). The single-node
GB10 receipt is pending too (the 1.54 receipt on the previous head passed at
1,458/1,147/24.2; the fit test is reports/dsv155-fit-test.md).
