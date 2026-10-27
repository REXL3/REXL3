# Promotion: dev approval to production

Production (`REXL3/REXL3`) holds sealed runtime trees under `runtimes/<family>/<model>/v<N>/`,
`MODEL.json` (schema `rexl3.prod-model.v1`) and `LATEST`. A dev-side approval authorizes a tree;
nothing is copied by hand, no workflow merges a pull request, and a maintainer reviews every
promotion. The authoritative receipt format and sealer live in `docs/PROMOTION.md` in
`REXL3/REXL3-dev`; this file is the production side.

## Layout and names

```
runtimes/
  <family>/
    <model>/
      MODEL.json      # schema rexl3.prod-model.v1: reviewed model metadata (below)
      LATEST          # one line: v<N>
      v<N>/           # sealed tree produced by tools/promote_runtime.py (never edited by hand)
  templates/          # APPROVED.json (template allow-list), APPROVALS.json (approval registry)
```

`<family>` and `<model>` are lowercase, regex `^[a-z0-9]+([.-][a-z0-9]+)*$`. `<model>` is the model
name WITHOUT quantization format, bit width, draft/speculation method or other pack details: those
live in `MODEL.json` and in the sealed approval. Dev runtime ids and dev tags never drive the folder
name. The old flat `runtimes/<runtime-id>/v<N>` layout is refused by the verifier.

`MODEL.json` required keys: `schema`, `family`, `model` (equal to the folder names), `display_name`,
`architectures`, `engine_family`, `runtime_id`, `pack` (with `format`). The exact engine architecture
is recorded here because one family can span architectures. The registry key is
`<family>/<model>/v<N>`; each entry carries `approval_sha256`, `source_commit`, `source_tree` and
`runtime_id`.

## Trust boundary

A hash proves that bytes did not change. It does not prove that a benchmark ran. A tree is accepted
only if all of these hold:

1. The approval is `rexl3.profile-approval.v2` and its exact bytes are listed in
   `runtimes/templates/APPROVALS.json` at `approvals["<family>/<model>/v<N>"]`, including a
   `runtime_id` that equals the approval's runtime id. A maintainer reviews that entry; it is the
   independent anchor.
2. The registry file matches the hash pinned in `.github/workflows/verify.yml`
   (`TRUSTED_APPROVALS_SHA256`), so changing the registry also changes a second reviewed file.
3. The verifier, the registry and the template allow-list (`runtimes/templates/APPROVED.json`) are
   read from the protected base revision of `main`, never from the pull request. `.github/CODEOWNERS`
   makes `@am423` and `@vcruz305` owners of `.github/`, `tools/`, `runtimes/` and `docs/`.
4. `MODEL.json` is valid per the schema and its `runtime_id` equals the sealed approval's. Dev
   release tags derive from the registry entry's `runtime_id` (`approve-<runtime_id>-<N>`), never
   from folder names.
5. The authenticated dev release assets match the approval, and their receipts pass the strict gate
   rules: nine gates (`doctor`, `lifecycle`, `capability`, `vision`, `context`, `exactness`,
   `prefix_cache`, `performance`, `quality`), each with `total > 0`, `passed == total`, no failure
   and no skip.
6. The tree equals the wrapper projection plus the original `approval.json`, byte for byte: no
   missing, extra, edited, linked or mode-changed file.

One secret, `DEV_ASSETS_TOKEN`, lets CI read the private dev release. There is no auto-merge anywhere.

## Steps (maintainer)

### 0. Register the approval (`model/<model>-approval`, merges first)

Open a PR that adds the `approvals["<family>/<model>/v<N>"]` entry to
`runtimes/templates/APPROVALS.json` and updates `TRUSTED_APPROVALS_SHA256` in
`.github/workflows/verify.yml` to the new file's sha256. A second maintainer reviews it and merges
it. CI reads the registry from the base branch, so this must merge before the model PR.

### 1. Prepare the model branch (`model/<model>`)

```bash
git fetch origin && git switch --detach origin/main
python3 tools/promote_runtime.py \
  --family <family> --model <model> --version <N> \
  --approve-tag approve-<runtime_id>-<N> \
  --target-commit "$(git rev-parse HEAD)" --prod-checkout "$PWD" \
  --assets-dir <durable-dev-assets> --download \
  --approved runtimes/templates/APPROVED.json \
  --trusted-approvals runtimes/templates/APPROVALS.json \
  --trusted-approvals-sha256 "$(sed -n 's/^ *TRUSTED_APPROVALS_SHA256: *//p' .github/workflows/verify.yml)" \
  --display-name "<Display Name>" --engine-family <dev-module> --pack-format exl3 \
  --architecture <HFArchitectureClass>
```

`--download` validates release and asset ids, sizes, the tag commit, every byte and all evidence and
caches the verified files. The tool then writes `MODEL.json` (or validates an existing one against
the derived metadata), extracts the tree, writes `LATEST`, verifies against the base registry,
commits branch `model/<model>` and opens a PR titled `FEAT: Promoted <model> v<N>`. `--dry-run`
verifies and prints without writing. An existing `MODEL.json` whose derived metadata differs is a
hard refusal, never an overwrite.

Add `docs/packs/<model>.md` and `changes/unreleased/<model>.md` to the same PR, then run the
completeness check (section 6) with `--rc-commit <dev release-candidate commit>` and paste its full
output. A pack sheet describes the sealed files and the capability contract (speculative decoding
on, prefix cache on, vision on unless a recorded does-not-fit justification exists in `MODEL.json`).

### 2. Review and merge

The other maintainer reviews the diff, the registry entry and the tool output, and merges with
squash. Model PRs additionally need the `local-validation` check: an APPROVED review by a maintainer
other than the author whose body carries a `LOCAL-VALIDATION v1` block for the current head (PR
number, head SHA, `<family>/<model>`, host, four PASS fields and gates). It re-evaluates on push and
on review submit/dismiss; stale-head or self approvals never count.

### 3. Product release (maintainer ceremony only)

Per-runtime production releases (`prod/<runtime>/v<N>`) are retired. Product releases are `v<X.Y.Z>`
and follow [RELEASING.md](RELEASING.md): a release PR, then `tools/cut_release.py` (dry run,
`--publish`, `--verify`) run by a maintainer personally.

## Verification in CI

`verify` runs on every PR and push: it checks out the protected base into `base/` and the PR head
into `candidate/` as data only, runs `base/tools/fetch_dev_assets.py` (which derives each dev tag
from the registry entry's `runtime_id`), then `base/tools/verify_runtime_tree.py` over `candidate/`,
then the unit tests from `base/`. `pr-rules` enforces title, branch and path scope.
`model-completeness` runs `tools/check_model_branch.py` for `model/<model>` PRs.
`local-validation` sets the head-SHA status described above. `governance-audit` is a daily read-only
audit, not required.

## Legacy v1

Legacy `profile-approval.v1` trees are not accepted in the family/model layout. Removed history
returns only as a new sealed version from the dev repository.

## Open items

- The asset download is exercised in mocks and in fixture trees; the first real promotion is the
  first end-to-end run. Rotate `DEV_ASSETS_TOKEN` before its 90-day expiry (update by day 80).
- Tools assume `git` and Python 3.12+; prod tooling is stdlib-only.
