# REXL3 production runtime packs

This repository holds sealed, hash-pinned runtime packs for REXL3 serving on NVIDIA GB10 (DGX Spark
class) hosts. A pack is the serve profile, the launch and contract scripts, the model and dependency
locks, and an `approval.json` that binds every file to the source commit and runtime template it was
built from. The git trees are provenance. The runnable bundle, which contains the CUDA binaries, is a
**release asset**.

Nothing in this repository is edited by hand after promotion, and contributors do not edit
`runtimes/`. Rules for changes are in [CONTRIBUTING.md](CONTRIBUTING.md); how a pack gets here is in
[docs/PROMOTION.md](docs/PROMOTION.md).

**Testing a pack? Start with the [user guide](docs/USER_GUIDE.md).** It covers the whole journey from an
empty directory (access, download and verification, install, model download, start), using the API
(streaming, thinking, tools, images), running it day to day, troubleshooting and how to report a problem.
Each pack also has a one-page **pack sheet** with its model, context window, port and expectations.

## Released packs

No pack is released yet. The first product release will be `v0.1.0`.

No other profile is released.

## Run a pack

This is the short path; the [user guide](docs/USER_GUIDE.md) explains each step and what to do when one
fails. Requirements, from the bundle's `docs/SETUP.md` and `scripts/doctor.py`: Ubuntu 24.04 on ARM64, the
GPU compute capability the release pins (a GB10 host), `sudo` for system packages, network access, and
room for the model (its size is on the pack sheet). `doctor` checks the platform and the bundle, not free
disk or an idle GPU. The repository is private, so `gh` must be logged in with access to it.

```bash
RUNTIME=<runtime-id>
VER=v1                                  # or the contents of runtimes/$RUNTIME/LATEST

gh release download "prod/$RUNTIME/$VER" -R REXL3/REXL3 -D /tmp/pack
# The archive hash must equal bundle.sha256 in the release's approval.json (and in this repository's
# runtimes/$RUNTIME/$VER/approval.json).
ARCHIVE=$(python3 -c "import json;print(json.load(open('/tmp/pack/approval.json'))['bundle']['archive'])")   # the name approval.json records
sha256sum "/tmp/pack/$ARCHIVE"
grep -A4 '"bundle"' /tmp/pack/approval.json

mkdir -p ~/rexl3 && tar -xzf "/tmp/pack/$ARCHIVE" -C ~/rexl3
ENGINE=/tmp/pack-engine                              # only when approval.json has "engine_binaries" (see section 4.4a)
python3 - /tmp/pack/approval.json "$ENGINE" ~/rexl3 <<'PY'
import hashlib, json, os, shutil, subprocess, sys
approval, engine_dir, root = sys.argv[1:4]
for install, item in json.load(open(approval)).get('engine_binaries', {}).items():
    subprocess.run(['gh', 'release', 'download', item['release_tag'], '-R', 'REXL3/REXL3', '-p', item['asset'], '-D', engine_dir], check=True)
    path = os.path.join(engine_dir, item['asset'])
    if hashlib.sha256(open(path, 'rb').read()).hexdigest() != item['sha256'] or os.path.getsize(path) != item['bytes']:
        sys.exit('ENGINE_BINARY_HASH_MISMATCH: stop; do not run it')
    os.makedirs(os.path.join(root, os.path.dirname(install)), exist_ok=True)
    shutil.copyfile(path, os.path.join(root, install)); os.chmod(os.path.join(root, install), 0o755)
    print('placed', install, 'from', item['release_tag'])
PY
cd ~/rexl3
./launch.sh --yes install-deps          # isolated, hash-pinned Python environment; no model download
./launch.sh doctor                      # read-only checks; must exit 0
./launch.sh --yes setup --accept-model-download   # downloads and verifies the model (size: the pack sheet)
./launch.sh --yes start                 # foreground; stays attached to this terminal
```

Run `launch.sh` as your normal user, not with `sudo`. `--yes` approves routine package installation
only; the model download needs `--accept-model-download`. `start` also runs the installer step and
refuses to run as root.

From a second terminal:

```bash
PORT=8940                               # the port on the pack's sheet
curl -s http://127.0.0.1:$PORT/v1/models                  # the model id to use, and the effective window
curl -s http://127.0.0.1:$PORT/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"<id from /v1/models>","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":512}'
```

A pack that thinks before it answers can spend a small `max_tokens` budget inside the thinking and return
empty content (the pack sheet and its `KNOWN_ISSUES.md` say whether yours does). Use `./launch.sh status`, `logs`,
`stop`, `verify` and `diagnostics` from the second terminal. `stop` stops only the process that launch owns. The server has
no authentication and no TLS; keep it on loopback or behind your own proxy (see [SECURITY.md](SECURITY.md)).

The bundle's own `README.md`, `docs/SETUP.md` and `docs/TROUBLESHOOTING.md` are the command reference.

### Reporting a problem

Open an issue in this repository (Issues, New issue, **Tester report**). The form asks for the release
tag and archive hash, the profile, the command with its exit code and error token, the output of
`./launch.sh diagnostics`, and what you expected. `diagnostics` is an allow-list: it contains no prompts,
responses, raw logs, environment values or credentials. Read it before pasting. Issues are visible to the
maintainers and every invited tester. For a quick question or to talk with other testers, use the testers'
Discord server; anything you want tracked belongs in an issue. Report a security problem as described in
[SECURITY.md](SECURITY.md), never in an issue or a chat channel.

## Layout

```
docs/
  USER_GUIDE.md                  # the tester's guide, the same for every pack
  packs/<runtime-id>.md          # one sheet per released runtime, added after its release
  packs/TEMPLATE.md, README.md   # how to write a sheet
  PROMOTION.md                   # how a pack gets here
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
hosted CPU runner cannot build or qualify the CUDA binaries. A newer release may leave `bin/` out of its
archive and take the server binary from an engine release (`engine-v<version>-<commit>`) that its
`approval.json` names by hash under `engine_binaries`; the quick start above places it.

## What CI checks

`verify` runs on every pull request and every push to `main`. Its verifier and policy files come from
the protected base revision, never from the pull request. It requires every tree under `runtimes/` to
match an entry in `runtimes/templates/APPROVALS.json` (exact file inventory, hashes, sizes and modes),
and it refuses a template digest that is not in `APPROVED.json`. `v1` is checked as a legacy tree,
against its pinned inventory. A tree from the newer receipt-based format also needs the authenticated
dev release assets: `verify` downloads them with a repository secret before it checks the receipts, as
described in [docs/PROMOTION.md](docs/PROMOTION.md). CI checks provenance and
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
