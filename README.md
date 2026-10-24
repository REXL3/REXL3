# REXL3 production runtime packs

This repository holds sealed, hash-pinned runtime packs for REXL3 serving on NVIDIA GB10 (DGX Spark
class) hosts. A pack is the serve profile, the launch and contract scripts, the model and dependency
locks, and an `approval.json` that binds every file to the source commit and runtime template it was
built from. The git trees are provenance. The runnable bundle, which contains the CUDA binaries, is a
**release asset**.

Nothing in this repository is edited by hand after promotion, and contributors do not edit
`runtimes/`. Rules for changes are in [CONTRIBUTING.md](CONTRIBUTING.md); how a pack gets here is in
[docs/PROMOTION.md](docs/PROMOTION.md).

## Released packs

| Runtime | Version | Context | Topology | Vision |
| --- | --- | --- | --- | --- |
| `dsv41-flash-1017-sage-1.62` | `v1` | 32K (`--max-model-len 32768`) | single node, one slot | off (`--no-vision`) |

`v1` serves DeepSeek-V4.1-Flash-1017 (EXL3 1.62 bpw, SAGE) with a q4 KV cache, speculative decoding
(`--spec auto`) and prefix caching. Capabilities in its profile: text, SSE streaming, reasoning,
tools, prefix cache. Not supported: images, video, audio, `response_format` JSON schema, tensor
parallel across nodes. The server listens on `127.0.0.1` only. See
`runtimes/dsv41-flash-1017-sage-1.62/v1/KNOWN_ISSUES.md` for measured behaviour.

No other profile is released.

## Run `v1`

Requirements, from the bundle's `docs/SETUP.md` and `scripts/doctor.py`: Ubuntu 24.04 on ARM64, the GPU
compute capability the release pins (a GB10 host), `sudo` for system packages, network access, and
room for the model (`model.lock.json`: 23 files, 304.16 GiB). `doctor` checks all of these. The repository is private, so `gh` must be logged in with access to it.

```bash
RUNTIME=dsv41-flash-1017-sage-1.62
VER=v1                                  # or the contents of runtimes/$RUNTIME/LATEST

gh release download "prod/$RUNTIME/$VER" -R REXL3/REXL3 -D /tmp/pack
# The archive hash must equal bundle.sha256 in the release's approval.json (and in this repository's
# runtimes/$RUNTIME/$VER/approval.json).
sha256sum /tmp/pack/*.tar.gz
grep -A4 '"bundle"' /tmp/pack/approval.json

mkdir -p ~/rexl3 && tar -xzf /tmp/pack/*.tar.gz -C ~/rexl3 && cd ~/rexl3
./launch.sh --yes install-deps          # isolated, hash-pinned Python environment; no model download
./launch.sh doctor                      # read-only checks; must exit 0
./launch.sh setup --accept-model-download   # downloads and verifies the model (about 304 GiB)
./launch.sh start                       # foreground; stays attached to this terminal
```

Run `launch.sh` as your normal user, not with `sudo`. `--yes` approves routine package installation
only; the model download needs `--accept-model-download`. `start` also runs the installer step and
refuses to run as root.

From a second terminal:

```bash
curl -s http://127.0.0.1:8940/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":64}'
```

Reasoning is on by default, so a very small `max_tokens` can end inside the reasoning span and return
empty content (see `KNOWN_ISSUES.md`). Use `./launch.sh status`, `logs`, `stop`, `verify` and
`diagnostics` from the second terminal. `stop` stops only the process that launch owns. The server has
no authentication and no TLS; keep it on loopback or behind your own proxy (see [SECURITY.md](SECURITY.md)).

The bundle's own `README.md`, `docs/SETUP.md` and `docs/TROUBLESHOOTING.md` are the command reference.

### Reporting a problem

Run `./launch.sh diagnostics` and attach its output. It is an allow-list: it contains no prompts,
responses, raw logs, environment values or credentials. Read it before sharing. Send it to the
maintainers with the command you ran. Report a security problem as described in
[SECURITY.md](SECURITY.md), not in an issue.

## Layout

```
runtimes/
  templates/APPROVED.json        # runtime template versions and digests production accepts
  templates/APPROVALS.json       # approval registry: the protected anchor for each tree (CODEOWNERS)
  <runtime-id>/
    LATEST                       # the promoted version, e.g. v1
    v<N>/                        # one sealed tree; never edited
      approval.json              # who approved what: commit, tree, template, archive hashes
      source.manifest.json       # every file's sha256, bytes and mode
      profile.json, model.lock.json, dependencies.lock.json, requirements-runtime.lock,
      KNOWN_ISSUES.md, LICENSE, NOTICE, THIRD_PARTY_NOTICES.md, README.md,
      launch.sh, scripts/, docs/, sbom/, TEMPLATE.json
tools/                           # verifier, promoter and their tests (stdlib Python)
```

`bin/` and `release.lock.json` are not in git: they exist only in the release archive, because a
hosted CPU runner cannot build or qualify the CUDA binaries.

## What CI checks

`verify` runs on every pull request and every push to `main`. Its verifier and policy files come from
the protected base revision, never from the pull request. It requires every tree under `runtimes/` to
match an entry in `runtimes/templates/APPROVALS.json` (exact file inventory, hashes, sizes and modes),
and it refuses a template digest that is not in `APPROVED.json`. `v1` is checked as a legacy tree,
against its pinned inventory. A tree from the newer receipt-based format also needs the authenticated
dev release assets, which CI cannot download, so CI rejects it until a maintainer-approved asset
source exists. This is described in [docs/PROMOTION.md](docs/PROMOTION.md). CI checks provenance and
integrity; it does not run or qualify a model.

## Licence and source

AGPL-3.0-only for everything here that the REXL3 authors wrote (`LICENSE`, `NOTICE`). Model weights
are not in this repository: each pack's `model.lock.json` pins the upstream repository and revision,
and that repository's own licence governs the weights. Third-party notices are in each tree's
`THIRD_PARTY_NOTICES.md` and `sbom/`.

The binaries in a release archive are built from the `source_commit` recorded in the tree's
`approval.json`, in the private `REXL3/REXL3-dev` repository. The `v1` release has no source archive
attached. Releases made after the receipt-based process attach an `engine_source_asset` with that
source. Anyone who receives a binary and needs its Corresponding Source should ask the maintainers;
the gap for `v1` is tracked in [docs/PROMOTION.md](docs/PROMOTION.md).
