# REXL3

REXL3 is an inference engine for **NVIDIA GB10** systems (DGX Spark class, `sm_121`) that serves
EXL3-quantized language models behind an **OpenAI-compatible HTTP API**. It is written in Rust with
CUDA kernels and runs recipes: sealed, hash-pinned per-model runtime configurations that pin the model
pack, the serving profile and every file in the tree.

This repository is the **production** repository. It holds:

- the sealed recipe trees under `runtimes/<family>/<model>/`, each produced by the dev repository's
  approval and promotion tooling and never edited by hand;
- the policy that decides which trees production accepts (`runtimes/templates/APPROVED.json` and
  `runtimes/templates/APPROVALS.json`);
- the stdlib-Python verifier and promotion tools in `tools/`.

Engine development, model validation and sealing happen in the development repository,
`REXL3/REXL3-dev`. There is no engine source here. Rules for contributing are in
[CONTRIBUTING.md](CONTRIBUTING.md); how a recipe gets here is in [docs/PROMOTION.md](docs/PROMOTION.md).

> **Status: no pack is released yet.** The first release (v0.1.0) is being prepared; none of the
> first-release models below is published. Everything in this README describes how the release will
> work and how recipes are structured, not something that is already available.

## First release (v0.1.0) models

| Model | Architecture | Status |
| --- | --- | --- |
| Qwen3.8-27B | `qwen3_5` (family `qwen`) | Furthest along: a v2-schema approval existed but was unreviewed, and its registry entry was removed in the restructure; it must be re-reviewed and re-promoted through the model-branch flow. |
| Qwen3.8-Flash-Next | `qwen4_exp` (family `qwen_flash_next`) | No runtime entry yet; the pack needs engine validation and a profile/gate recipe. |
| DeepSeek-V4.1-Flash-1017 | `deepseek_v41` | Runtime entry exists (single-node and two-node profiles); the vision-disabled profile needs restoring and re-sealing before release. |
| GLM-5.3-Flash | `glm` (glm5_next) | Runtime entry exists; live-server qualification on the release binary is still required. |
| Nemotron-3.5-Super-VL | `nemotron_h_omni` | Family code exists; no runtime entry yet. |

None of these is released. Qualification on the exact release binary, sealing and promotion are the
remaining steps; the release notes document what each model's evidence proves.

## Supported hardware

- **Single node: NVIDIA GB10** (DGX Spark class) — Ubuntu 24.04 on ARM64, 128 GB unified memory.
  The engine's kernels target `sm_121`. Other GPUs, OSes or architectures are not supported.
- **Two nodes (tensor parallel, TP=2):** two GB10 systems on a fabric NCCL can use (measured over
  RoCE). Only two ranks are operationally verified; larger groups are built but unmeasured on a real
  fabric and are not part of a release.

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for the full requirements list (GPU, memory, disk,
privileges, network, GitHub access).

## How releases work

There is **one versioned product release** — the engine and its recipes ship together — not a release
per runtime. Versioning is SemVer 0.x; the first release is **v0.1.0**.

- All models in a release are sealed from the same dev release-candidate commit.
- A release is a `v<X.Y.Z>` tag on the prod repository. Per-runtime releases are retired; a release
  contains the bundle archives for every recipe in it, their `approval.json` records, and the engine
  corresponding source.
- Releases are immutable: a bad release is superseded by a patch release, never edited.
- Cutting a release is a maintainer ceremony following [docs/RELEASING.md](docs/RELEASING.md); agents
  never create or publish releases.

## Install and run a recipe

A **recipe** is one sealed runtime tree: the model, its serving profile, and the launcher. Operators
install a released recipe from the release's assets, never from a development checkout. The steps:

```bash
# 1. Download the release assets (choose the exact tag, e.g. v0.1.0)
gh release download v0.1.0 -R REXL3/REXL3 -D /tmp/rexl3-release

# 2. Verify before extracting: the archive's SHA-256 must equal bundle.sha256 in approval.json
ARCHIVE=$(python3 -c "import json;print(json.load(open('/tmp/rexl3-release/approval.json'))['bundle']['archive'])")
sha256sum "/tmp/rexl3-release/$ARCHIVE"        # compare with bundle.sha256

# 3. Extract the bundle and run the launcher lifecycle
mkdir -p ~/rexl3 && tar -xzf "/tmp/rexl3-release/$ARCHIVE" -C ~/rexl3 && cd ~/rexl3
./launch.sh --yes install-deps                # pinned system packages and a private Python environment
./launch.sh doctor                            # read-only checks; must exit 0
./launch.sh --yes setup --accept-model-download   # downloads and verifies the pinned model
./launch.sh --yes start                       # foreground; verified READY before serving
```

Run the launcher as your normal user, never with `sudo`. `--yes` approves routine package
installation only; the model download needs the separate `--accept-model-download` consent. From a
second terminal:

```bash
curl -s http://127.0.0.1:8889/v1/models                    # the served model id and effective window
curl -s http://127.0.0.1:8889/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"<id from /v1/models>","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":512}'
```

The default port is 8889; a recipe may pin another in its profile. `./launch.sh status`, `logs`,
`stop`, `verify` and `diagnostics` operate from a second terminal. The server has no authentication and
no TLS; keep it on loopback or behind your own proxy (see [SECURITY.md](SECURITY.md)).

`launch.sh` commands: `doctor`, `install-deps`, `setup`, `print-command`, `start`, `status`, `stop`,
`logs`, `verify`, `diagnostics`, `reset-state`. `install-deps` and `--yes`/`--allow-driver-change`
precede the command; `--profile`, `--site` and `--accept-model-download` follow it.

## Recipe structure

Recipes live under `runtimes/<family>/<model>/`, lowercase, for example
`runtimes/qwen3.8/qwen3.8-27b/`. The `<model>` name never carries a quantization format, bit width or
draft method; those live in `MODEL.json` and the sealed approval.

```
runtimes/
  <family>/
    <model>/
      MODEL.json        # reviewed metadata (schema rexl3.prod-model.v1): schema, family, model,
                        # display_name, architectures, engine_family, runtime_id, and pack
                        # (required key: format; descriptive keys may include bits, draft,
                        # target_repo, draft_repo, vision_does_not_fit)
      LATEST            # one line: the current sealed version, e.g. v1
      v<N>/             # the sealed tree, produced by tools/promote_runtime.py; never edited
        approval.json   # who approved which bytes: source commit, tree, template, archive hashes
        profile.json    # the serving profile: argv, env, capabilities, smoke request
        model.lock.json # the pinned model repository, revision and per-file hashes
        launch.sh       # the launcher (the command lifecycle above)
        KNOWN_ISSUES.md # what was and was not qualified, behaviour to expect
        docs/           # SETUP and TROUBLESHOOTING for this recipe
        sbom/           # software bill of materials
        dependencies.lock.json, requirements-runtime.lock, LICENSE, NOTICE, ...
  templates/
    APPROVED.json        # template versions and digests production accepts
    APPROVALS.json       # approval registry: the protected anchor for each tree
```

`APPROVALS.json` keys are `<family>/<model>/v<N>`; each entry pins the approval hash, the source
commit and tree, and the runtime id. The verifier (`verify`, section "Checks and CI" below) requires
every tree under `runtimes/` to match a registry entry exactly — file inventory, hashes, sizes and
modes — and refuses a template digest that is not in `APPROVED.json`.

## Run your own model or recipe (advanced)

Advanced users can point the engine at their own model directory or pack with the server binary
`rexl3-serve` (built from the dev repository), e.g.:

```bash
rexl3-serve --model <model-dir> [--draft <draft-dir>] --host 127.0.0.1 --port 8889 \
  --slots 8 --max-model-len 262144 [--kv-dtype fp16|q4|int8] [--spec auto|mtp|copy|none] \
  [--prefix-cache on|off] [--prefix-cache-mib N] --served-model-name NAME
```

- **Pack fingerprints:** every boot checks the pack before any plan or weight load. Only packs on the
  engine's validated fingerprint list boot without extra flags; `--verify-pack` checks the shard
  CRC-32s, `--print-fingerprints` prints the fingerprints, and `--print-crc32` prints shard CRCs.
- **Unvalidated pack override:** a pack that is not on the validated list is refused unless you
  explicitly pass `--allow-unvalidated-pack`. This is a deliberate, explicit override — never a
  default.
- **Your own profile:** a runtime profile is a JSON file (`rexl3.runtime-profile.v1`) that pins argv,
  environment, capabilities and the model lock. Writing a new profile and getting it sealed is the
  promotion path described in this repository's [docs/PROMOTION.md](docs/PROMOTION.md) and the dev
  repository's `CONTRIBUTING.md`.

## Engine features

All details below are read from the engine's source on dev main.

### OpenAI-compatible API

| Route | What it does |
| --- | --- |
| `GET /v1/models` | served model id, effective/configured window, reasoning controls, pack validation status |
| `POST /v1/chat/completions` | chat; streaming (`"stream": true`, SSE ending in `[DONE]`); tool calls (`tool_choice` `auto` or `none`); images; video frames; reasoning returned as `reasoning_content` |
| `POST /v1/completions` | plain completion |
| `POST /v1/chat/completions/render` | the rendered prompt's token ids, nothing generated |
| `POST /v1/decisions` | decision endpoint |
| `POST /tokenize` | tokenizer |
| `GET /health/live` | 200 while the process answers |
| `GET /health`, `/health/ready`, `/v1/health` | 200 when ready; 503 while loading/draining/failed; under TP=2 the body carries `tensor_parallel` |
| `GET /metrics` | Prometheus text: requests, queue, tokens, speculation, prefix-cache counters, build info |

`response_format` values `json_object` and `json_schema` are refused (plain text only). `n` other
than 1, `logit_bias`, non-zero penalties, `tool_choice` `required`/named function,
`parallel_tool_calls` and per-token logprobs are refused by name with a 400.

### Speculative decoding

Drafting is on and is the engine's invariant: **a drafted reply is byte-identical to the serial
reply** — speculation changes speed, never text. Modes: `--spec auto` (default), `mtp`, `copy`,
`none` (serial; `--no-drafts`). `--spec-tree`/`--no-spec-tree` control the draft tree. Drafters are
per-family: native draft heads where the model has them (e.g. Qwen's DFlash2, GLM/NextN in-pack, the
DeepSeek DSpark stages, Nemotron's MTP head). `usage.spec` reports drafted/accepted token counts; the
metrics expose the same counters.

### Prefix caching

On by default (`auto`): the engine keeps state of earlier prompts and reuses a shared beginning.
`--prefix-cache on|off` and `--prefix-cache-mib N` (MiB; 0 disables) control it. `usage.prompt_tokens_details.cached_tokens` reports what was reused; a cached reply equals the uncached one. The automatic cap is at most half the spare memory and 8 GiB.

### KV cache formats

`--kv-dtype fp16` (default) | `q4` (DeepSeek-V4.1) | `int8` (Nemotron, opt-in). Environment
equivalents: `REXL3_DSV41_KV` (q4) and `REXL3_KV_DTYPE` (int8/fp16). fp8 was evaluated and rejected.

### Tensor parallel / cluster

TP=2 is measured over RoCE (NCCL): sharded attention, weights halved per node, vision tower on rank
0. Start a pair with `rexl3-cluster` and a cluster JSON file (validate/check/up/status/down/logs/
supervise over ssh), or by hand with `--tp-ranks 2 --tp-rank 0|1 --tp-peer <leader-ip>:<port>`
(env: `REXL3_TP_RANKS`, `REXL3_TP_RANK`, `REXL3_TP_PEER`, timeouts). A boot gate compares identity
between ranks and refuses a mismatch before any weight is read. Only two ranks are verified.

### Vision

Image input is inline base64 in a data URI (`image_url` part). Remote HTTPS image URLs are off unless
`--image-urls on` (policy-guarded). `--no-vision` disables the tower (accepted only with a recorded
does-not-fit justification); `--max-image-tokens` bounds image tokens. Towers exist for Qwen, GLM
(glm5_next, 24-layer), MiniCPM-V-4.7 and Nemotron Omni; Nemotron also accepts video as a frame list
(`--max-video-frames`, `--max-video-tokens`, `--video-pruning-rate`, `--video-evs-strict`,
`--no-video`). Audio is refused by every family.

### Reasoning controls

Reasoning models return their thinking separately in `message.reasoning_content` (streaming:
`delta.reasoning_content`); `message.content` holds the answer. Request fields: `reasoning_effort`
(the level names depend on the family; `/v1/models` reports the controls the model offers) and
`chat_template_kwargs.enable_thinking`. Server defaults: `--default-thinking off|template` and
`--default-reasoning-effort LEVEL` for requests that name neither; `max_reasoning_tokens` caps the
thought.

### Memory admission planning

Before reading a weight, the engine plans from `MemAvailable` and **refuses a configuration that does
not fit** (fail-closed). `REXL3_MEMORY_FLOOR_GIB` sets the free-memory floor (default 8, fractions
allowed, ≥0.25 validated). KV pool and slots are sized from the plan (`--slots`, `--kv-tokens`,
`--kv-headroom-tokens`); the prefix cache is carved from spare memory.

### EXL3 formats

The engine serves EXL3 quantized packs only and refuses other architectures by name. EXL3 in this
engine means trellis-coded tensors with a per-tensor bit width K from 1 to 8 (half-integer widths such
as 2.5 bpw included), mixed-precision and mixed-K pack forms, an EXL3 `lm_head`, F16 router gates and
latent projections, BF16 control surfaces, F8/E4M3 native masters handled at load, and NVMe-mmap'ed
engram tables where a pack uses them (e.g. DeepSeek-V4.1). `exl3-inspect`, `--verify-pack`,
`--print-fingerprints` and `--print-crc32` inspect and verify packs.

## The capability contract

Every released model must deliver, and prove at its real context and slot count:

- **speculative decoding on**, proven — drafted output is byte-identical to serial;
- **prefix cache on**, proven at the real contract;
- **vision on and tested by default**, off only with a proven does-not-fit justification.

`SAGE is our own mixed-precision quantization method for EXL3. The method is not published.`

## Checks and CI

Every PR to `main` must pass four required checks, each of which always reports (job-level conditions
that succeed as a no-op; never workflow-level path or branch filters):

- `verify` — the production gate: from the protected base revision, it downloads the authenticated dev
  release assets for the candidate trees (`tools/fetch_dev_assets.py`, tags derived from each registry
  entry's `runtime_id`), verifies every tree under `runtimes/` against the pinned registry
  (`tools/verify_runtime_tree.py`), and runs the unit tests. `MODEL.json` must agree with the sealed
  approval, and the registry must match the `TRUSTED_APPROVALS_SHA256` pin in
  `.github/workflows/verify.yml`.
- `pr-rules` — title (`TYPE: Past-tense summary`), branch name, path scope per branch type, and the
  `changes/unreleased/` fragment rule.
- `model-completeness` — runs `tools/check_model_branch.py` for `model/<model>` PRs; no-op success
  otherwise.
- `local-validation` — for model PRs, an APPROVED review by a maintainer other than the author whose
  body carries a `LOCAL-VALIDATION v1` block for the current head; no-op success for non-model PRs and
  `model/*-approval` PRs.

`governance-audit` is a scheduled daily plus manual, read-only audit (not required): it fails when live
releases, merged-PR reviews or live rulesets deviate from the committed policy on `main`.

The workflows read code and policy from the protected base checkout only (`pull_request_target`); a PR
head is checked out as inert data and never executed. `DEV_ASSETS_TOKEN` is the only secret, used by a
single download step; actions are pinned by full commit SHA.

## Reporting a problem

Open an issue in this repository (Issues, New issue, **Tester report**). The form asks for the release
tag and archive hash, the profile, the exact command with its exit code and error token, the output of
`./launch.sh diagnostics` (an allow-list report — read it before pasting), and what you expected.
Issues are visible to the maintainers and every invited tester; never paste raw logs, prompts,
credentials or tokens. Report a security problem via [SECURITY.md](SECURITY.md), never as an issue.

## Licence and corresponding source

AGPL-3.0-only for everything here that the REXL3 authors wrote (`LICENSE`, `NOTICE`). Model weights
are not in this repository: each recipe's `model.lock.json` pins the upstream repository and revision,
and that repository's own licence governs the weights. Third-party notices are in each tree's
`THIRD_PARTY_NOTICES.md` and `sbom/`.

Binaries in a release archive are built from the source commit recorded in the recipe's
`approval.json`, in the dev repository. Releases attach the engine corresponding source as a release
asset; anyone who receives a binary and needs its Corresponding Source can also ask the maintainers.
