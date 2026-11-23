# RED-SNOW-5.3-FLASH EXL3 4.91 bpw (TP2) - frozen configuration, not qualification

This entry is the two-node RED-SNOW-5.3-Flash runtime, `redsnow-tp2-512k` (two
GB10s, tensor parallel, 524,288 tokens). It is its own runtime unit, separate
from the single-node 2.49 bpw entry `runtimes/glm-5.3/red-snow-5.3-flash/`
(`redsnow-single`): a different pack, runtime id
(`redsnow-5.3-flash-tp2-sage-4.91`) and gate recipe
(`scripts/gates/recipes/redsnow-5.3-flash-tp2-sage-4.91.json`). The recipe pins
no window, so the gate takes 524,288 from this argv.

It serves text, SSE, reasoning, tools and **images**, with speculative decoding
(`--spec auto`, the pack's own MTP layer) and the pool-page prefix cache.
Audio, video and constrained JSON response formats are unsupported.
Configuration freeze does not imply full-window, throughput or quality
qualification: every number below is a dated reference under the conditions
stated, and none is a floor of this profile. The gate's speed floors are in the
recipe: decode 27 tok/s, prefill 851 tok/s at 50% and 788 at 100% of the window.

| | `redsnow-tp2-512k` |
| --- | --- |
| Window (`--max-model-len`) | 524,288 (the pack advertises 1,048,576; see "Known issues") |
| Slots | 1 |
| KV pool | `--kv-tokens 524288` on every rank (a GLM group requires it) |
| Speculation | `--spec auto` (the pack's MTP layer) |
| Prefix cache | `--prefix-cache pool`: entries live in the KV pool's idle pages, reach 524,288 tokens |
| Prefill | P2 contract (the engine default), `--prefill-chunk 2048` |
| Images | on, `--max-image-tokens 4096` (rank 0 holds the tower) |
| Memory floor | `REXL3_MEMORY_FLOOR_GIB=2` (engine default 8) |
| Activation reserve | `REXL3_GLM_RESERVE_MARGIN_GIB=4`: modelled scratch at 2,048 rows plus 4 GiB |
| Embedding | `REXL3_GLM_EMBED_IN_PLACE=1`: read in place from its shard (-1.18 GiB a rank) |
| Host | huge pages off on both nodes; idle `MemAvailable` at least 114 GiB on each node |

The pack is `vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw` (MIT, below), a
GLM-5.3-Flash derivative: architecture `Glm5NextForConditionalGeneration`
(`glm5_next` in the engine), one MTP layer inside the pack, experts of mixed
widths (4, 5 and 6 bits, mixed inside 41 of the 43 MoE layers, and one expert
whose gate, up and down projections differ). The pack card describes it as a
model for authorized offensive-security work and states its limits; read it
before deployment.

What it is on the wire: rank 0 serves HTTP, holds the vision tower and encodes
every image; the follower holds no tower (the server gives a follower no vision
slot whatever its flags), so there is no `--no-vision`. Each rank loads its share
of the routed experts. The NCCL variables and the attested cluster adapter are
the GLM and DeepSeek two-node profiles'. The sealed cluster file
`config/cluster.tp2-512k.json` (pinned by sha256) carries example network values;
the site must state its nodes, fabric interface, HCAs and TP port.

## Release status and current known issues

Released as version 1 in v0.1.1. v0.1.2 shipped version 1 again, unchanged.

Where the sections below say "not yet run", "pending" or "to be confirmed", they describe the entry before its release qualification. The receipts of that qualification are in the evidence archive of each release named above.

Known issues as of 2026-11-19. Each says what happens, who is affected and what to do:

- **NVIDIA driver: the 580 branch at build 580.142 or later; builds after 580.178.04 are accepted but not yet qualified.** This applies to every host, leader and follower. `libnvidia-compute-580` and `nvidia-utils-580` must both be the same upstream build, and every host must run that build; Ubuntu's `-0ubuntu0.24.04.1` and NVIDIA's CUDA-repository `-1ubuntu1` packages of one build count as the same driver. On a build later than 580.178.04, `./launch.sh doctor` and `install-deps` continue and print `DRIVER_NOT_YET_QUALIFIED` (`doctor` reports `driver_qualification: newer-than-tested`): 580.142 is the tested floor and 580.178.04 the highest build this runtime was qualified on, and results on a later build are not covered by the release's qualification evidence. The CUDA driver, runtime and cuBLAS versions that `rexl3-serve --version` reports must be the qualified ones or later on the same major version: a later one is accepted and `doctor` prints `NATIVE_NOT_YET_QUALIFIED`, and an older one or another major version stops `doctor` with `NATIVE_VERSION_MISMATCH`. A build below 580.142, or two different builds on one host, stops `doctor` with `SYSTEM_DEPENDENCY_VERSION_MISMATCH`; another driver branch (570, 590) is not accepted. `setup` and `start` check every host from the leader: a host that is not accepted stops them with `RANK_IDENTITY_FAILED` and a reason: `NATIVE_VERSION_MISMATCH` when the CUDA driver, runtime or cuBLAS version its `rexl3-serve --version` reports is older than the qualified one or of another major version (checked first), `IDENTITY_PROBE_FAILED` when a 580 package has never been installed on it (for example a host on another driver branch: the package query itself fails), `DEPENDENCY_MISSING` when one was installed and has been removed, and `DEPENDENCY_VERSION_MISMATCH` when its 580 build is below 580.142 or its two packages are different builds; hosts on different builds stop them with `DRIVER_VERSION_MISMATCH_ACROSS_RANKS`, and a later build on every host prints `DRIVER_NOT_YET_QUALIFIED`. When a driver change is needed and the configured repositories no longer offer the `580.178.04-0ubuntu0.24.04.1` package, `install-deps` stops with `DRIVER_PACKAGE_UNAVAILABLE` and installs nothing. What to do: stay on the 580 branch at 580.142 or later; if the package is unavailable, install any 580 build at or above 580.142 with the host package manager (both packages at the same build), reboot and rerun `./launch.sh doctor`.
- **NCCL (`libnccl2`): version 2 at 2.30.7 or later, the same on both hosts.** Qualified on 2.30.7 and 2.32.6. A later version is accepted and `doctor` prints `DEPENDENCY_NOT_YET_QUALIFIED`. A version below 2.30.7 or a `+cuda12` build stops `doctor` with `SYSTEM_DEPENDENCY_VERSION_MISMATCH`. Hosts on different versions stop `setup` and `start` with `DEPENDENCY_VERSION_MISMATCH_ACROSS_RANKS`. What to do: upgrade both hosts together, or hold the package on both.
- **A pending restart no longer stops the launcher.** When the host has a restart pending (`/var/run/reboot-required`, set by a kernel or system library update), `doctor` and `install-deps` print `REBOOT_PENDING` and continue. A driver upgraded without a restart is refused by name: `doctor` stops with `DRIVER_MODULE_MISMATCH` when the loaded NVIDIA kernel module is not the installed driver build, and after `install-deps` has changed the driver packages itself it stops with `DRIVER_MODULE_MISMATCH` too (with `REBOOT_REQUIRED` when the loaded module's build cannot be read and a restart is pending). What to do: restart the host, then rerun `./launch.sh doctor`.
- **The HTTP endpoint is on the network and has no authentication.** The leader listens on its address from the site file, not on `127.0.0.1`, and the server has no API key, so anyone who can reach the port can use it. What to do: run it only on a private, trusted network, or behind a firewall or an SSH tunnel.

The limits of this model and profile, tool use with thinking off among them, are listed under "Known issues" below.

## Known issues

- **The window is 524,288 tokens, not 1,048,576.** The pack advertises
  1,048,576. At 1,048,576 rank 0 cannot also hold the vision tower with 1 GiB over
  its 2 GiB floor on this engine: with vision on it was refused at boot (the tower
  needed 1.60 GiB and 3.36 GiB were left at a 2 GiB floor), and with vision off
  its `MemAvailable` went under the floor. 1,048,576 is planned follow-up work
  (engine memory fixes); this profile does not carry any of the opt-ins that
  work uses (for example the RoCE site sizing `REXL3_GLM_ROCE_SITES` or lending
  the tower's scratch).
- **Images over 4,096 image tokens are scaled down to fit.** The pack's processor
  allows 8,000 image tokens (about 6.3 megapixels); this profile caps an image at
  4,096 (about 3.2 megapixels, 784 pixels a token: 14-pixel patches merged 2 x 2,
  e.g. about 1,792 x 1,792). A larger image is scaled, not refused: the
  preprocessor lowers the processor's budget to the server limit. The quality
  effect on fine-detail images is not measured. Up to 4 images a request.
- **Huge pages must be off on both nodes.** The plan needs the memory a reserved
  huge-page pool would hold (`HugePages_Total 0`; `vm.nr_hugepages = 0`). With a
  pool reserved the host shows less `MemAvailable` and the boot is refused or the
  floor is not kept.
- **A cache entry lives in the KV pool's idle pages** (`--prefix-cache pool`), so
  it survives only until a request that does not share it needs those pages. At
  one slot a full-window prompt that differs from the cached one (or one sent with
  `"prefix_cache": false`) evicts the entry before its prefill; a follow-up turn
  that shares the cached prefix keeps it (its shared pages are credited, not
  evicted). Measured: a follow-up on a 518,780-token conversation hit 518,672
  cached tokens.
- The pack's chat template injects its own system prompt (about 176 tokens or
  more) into every prompt, so every prompt shares those tokens: a short request
  may report cached tokens up to a cut inside that shared prefix once another
  request has been served (128 and 176 were observed).
- The pack card says the vision components are packaged but not quality-tested
  and that the vision and audio towers were frozen during the security
  adaptation; multimodal performance is not established by the publisher.
- **With thinking off the model may answer a tool request from memory** instead
  of calling the tool; the same cases called the right tool with thinking on.
  Send `enable_thinking: true` for tool use.
- **Some replies after a tool call start with reasoning text that ends in a
  closing think tag.** Strip everything up to the last `</think>` if you need a
  clean answer.

## Memory

Each rank holds about 94 GiB of this pack's weights with the MTP layer
(rank 0 `weights_gib=94.02`, rank 1 93.99; the embedding table is read in place
from its shard, so it is not in that figure). Rank 0 also holds the vision tower
(1.050 GiB of weights and 0.549 GiB of encoder scratch at 4,096 image tokens)
and a 0.25 GiB features reserve. The engine plans against `MemAvailable` (cut
to a whole GiB) with this profile's 2 GiB floor and refuses a boot that does not
fit.

**Measured** (2026-11-08, two GB10s, huge pages off, idle `MemAvailable`
115.9 GiB on rank 0's node and 116.5 on rank 1's, a development build of this
engine with this argv, memory sampled every 0.2 s on both nodes):

| | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Plan | `kv_pages=2049`, tower 1.60, features 0.25, `free_after_gib=19.09`, cache `entries=8 pages=0 gib=0.55 spare_gib=3.8` | `free_after_gib=20.72` |
| `MemAvailable` after load | 6.76 GiB | 9.19 GiB |
| Minimum over the session | **5.96 GiB** (the load-end transient, at the tower load) | 8.61 GiB |
| Minimum while serving (image requests, a 518,746-token prompt with an image, a cache-hit follow-up) | 6.23 GiB | 8.97 GiB |
| Minimum minus the 2 GiB floor | **+3.96 GiB** | +6.61 GiB |

`REXL3_GLM_RESERVE_MARGIN_GIB=4` is measured, not modelled: a rank uses about
3.1 to 3.4 GiB beyond the plan's modelled items (pinned transport buffers, the
driver's page tracking, process memory and the load-end transient). At margin 4
rank 0's `MemAvailable` after load was about 0.6 GiB above the plan's prediction.

**Host requirement.** Rank 0 is the binding rank. Its plan's spare falls by
about 1 GiB for each GiB less that the planner reads, while the use beyond the
plan stays the same; the measured margin was +3.96 GiB over the floor at 115 GiB
planned. The bar this profile keeps is at least 1 GiB over the floor, so the
planner must read at least 113 GiB; about 0.8 GiB is taken before the plan
(process, CUDA context), so each node needs **at least 114 GiB of idle
`MemAvailable`** before the group starts. The cluster file's `memory.min_free_gib`
is 114, which the launcher's pre-flight (`rexl3-cluster check`) enforces. A host
with other large tenants is refused at boot, which is the engine working as
designed.

## What "validated" rests on for this pack

The pack is in `VALIDATED_TEXT` and its 36 shards are in `VALIDATED_SHARDS`
(`crates/model/src/fingerprint.rs`), so the profile does not pass
`--allow-unvalidated-pack`. The fingerprint is
`sha256:32167cfae738efe503ca13af3c7f107bb683aa2e73b5714849e3759721108f9b`. It
rests on:

- the P2 tests for six-bit and mixed-width experts (a rank holding several
  subbanks, and the split gate/up expert) and the two-rank loopback tests on this
  pack's real layers: two ranks equal one model byte for byte;
- a two-node session at this profile's window with this argv (above): boot,
  images, a 518,746-token prompt and a cache hit, with drafting on;
- every file of the node copies equal to the published ones: the SHA-256 of all
  43 locked files on both nodes equal `model.lock.json` (the Hugging Face LFS
  object ids for the shards and large files at the locked revision); each shard's
  CRC-32 taken from those bytes.

**There is no ExLlamaV3 oracle comparison for this pack**, unlike the GLM-5.3-Flash
2.25 bpw entry; the "(text)" in the table names the scope of the validation.

## Reasoning and sampling (source order: the pack, then the base card)

- The pack's card and its `generation_config.json` give `temperature` 0.7 and
  `top_p` 0.95 (the same file as the 2.49 bpw pack's). The profile carries no
  sampling flag; the server reads the file, so a request that leaves them out gets
  0.7 and 0.95 (`top_k` 20 is the server's fallback). The card
  states no `top_k`.
- The packaged `chat_template.jinja` (the same file as the 2.49 bpw pack's) opens a
  think block only when `enable_thinking` is true: a request that names neither
  `enable_thinking` nor `reasoning_effort` does not think. The template never
  reads `reasoning_effort`; the smoke request's `reasoning_effort: low` changes
  nothing for this pack.

## Draft and licence

No external draft: speculation uses the MTP layer stored in the pack, and the
argv has no `--draft`. `model.lock.json` has the `target` role only.

The pack's model card metadata and its `LICENSE` are MIT (the licence is copied
from the base release, copyright Z.AI); its base `Blackfrost-AI/RED-SNOW-5.3-FLASH-BF16`
and that model's base `zai-org/GLM-5.3-Flash` are MIT. This runtime pins the
published pack by revision and hashes and serves it; it does not redistribute
the weights.

## The model copy

`model.lock.json` pins `vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw` at revision
`f6b8963784254095a54944f2963b2925fcf4b199` (Hugging Face, public): 43 files,
199,538,523,078 bytes (36 shards, `model.safetensors.index.json`,
`tokenizer.json` and 5 small files). The 38 large files' hashes are the Hugging
Face LFS object ids at that revision and were recomputed on both node copies
(read only) with no difference; the 5 small files were compared with the bytes
downloaded at that revision.

The repository also publishes `README.md`, `LICENSE`, `.gitattributes` (these
three may be extra in a model directory, and are not locked), and `DEPLOY.md`,
`deploy.md` and two images in `ASSETS/`, which are **not locked and not
allowed**: the launcher's inventory refuses an unlisted file. The launcher's own
download fetches only the locked files, so this affects only a directory made
by cloning the whole repository: delete `DEPLOY.md`, `deploy.md` and `ASSETS/`
from such a copy.

## Reference figures (dated; not floors)

Two GB10s, this argv, a development build of this engine, 2026-11-08.

| What | Value | Conditions |
| --- | --- | --- |
| Prefill, uncached | 950 tok/s | 518,746-token prompt with an image (TTFT 545.8 s); 522,871 tokens of text in 539 to 550 s (four runs) |
| Prefill, baseline gate | 1,065 tok/s at 50% of the window, 985 at 100% | NIAH stages, about 262,000 and 523,000 tokens |
| Prefix-cache hit | 518,672 of 518,780 tokens cached, TTFT 1.17 s | follow-up turn on the same conversation |
| Decode with MTP | median 31.6 tok/s, acceptance 0.61 | 3 replies of 1,896 to 2,048 tokens, greedy, short context |
| Decode with MTP, baseline gate | median 33.8 tok/s (31.6 to 34.7), acceptance 0.64 | five 256-token samples |
| Boot to ready | 18 to 21 s | warm page cache |

## Unverified boundaries

The figures above are not a receipt. The release receipt is the v0.1.1
qualification (version 1), in that release's evidence archive. Not done: a
clean-host dependency resolution (`dependencies.lock.json` is the GLM-5.3-Flash
two-node lock, byte for byte). `profile.json` `source` names a document at the base commit that the
flags were derived from, not the profile file itself.
