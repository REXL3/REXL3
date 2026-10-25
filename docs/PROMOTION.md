# Promotion: dev approval to production

Production (`REXL3/REXL3`) holds sealed runtime trees that a dev-side approval authorised. Nothing is
copied by hand, no workflow merges a pull request, and a maintainer reviews every promotion. The
authoritative description of the receipt format and the sealer is `docs/PROMOTION.md` in
`REXL3/REXL3-dev`; this file is the production side of the same contract.

## Trust boundary

A hash proves that bytes did not change. It does not prove that a benchmark ran. A tree is accepted
only if all of these hold:

1. The approval is `rexl3.profile-approval.v2` and its exact bytes are listed in
   `runtimes/templates/APPROVALS.json` (`approvals["<runtime-id>/v<N>"]`: `approval_sha256`,
   `source_commit`, `source_tree`). A maintainer reviews that entry; it is the independent anchor.
2. The registry file matches the hash pinned in `.github/workflows/verify.yml`
   (`TRUSTED_APPROVALS_SHA256`), so changing the registry also changes a second reviewed file.
3. The verifier, the registry and the template allow-list (`runtimes/templates/APPROVED.json`) are read
   from the protected base revision of `main`, never from the pull request. A pull request cannot
   change the rules it is judged by. `.github/CODEOWNERS` makes `@am423` and `@vcruz305` owners of
   `.github/`, `tools/`, `runtimes/` and `docs/`.
4. The authenticated dev release assets (below) match the approval, and their receipts pass the strict
   gate rules: nine gates (`doctor`, `lifecycle`, `capability`, `vision`, `context`, `exactness`,
   `prefix_cache`, `performance`, `quality`), each with `total > 0`, `passed == total`, no failure and
   no skip. `FAIL`, `NOT_RUN`, zero tests and strings that look like a pass are refused.
5. The tree equals the wrapper projection plus the original `approval.json`, byte for byte: no missing,
   extra, edited, linked or mode-changed file.

No paid service or signing key is part of this. One secret, `DEV_ASSETS_TOKEN`, lets `verify` read the private dev release (see "Verification in production"). There is no auto-merge anywhere.

## Release assets

A v2 approval names four archives besides `approval.json`. All five are attached to the dev release
`approve-<runtime-id>-<N>` and again to the production release `prod/<runtime-id>/v<N>`.

| Asset | What it is |
| --- | --- |
| `approval.json` | The approval document. The production tree holds the same bytes. |
| `bundle` | The runnable bundle: wrapper, configuration and the qualified native binaries (`bin/`, `release.lock.json`). This is what a tester downloads. |
| `source_asset` | The **runtime wrapper projection**: exactly the bundle minus `bin/` and `release.lock.json`, including `source.manifest.json`. The field name is historical. This is what lives in `runtimes/<id>/v<N>/`. It is not the engine's corresponding source. |
| `engine_source_asset` | The **native engine corresponding source**: the original Git blobs and build inputs for the same commit and tree the binaries were built from. A missing engine source is a hard refusal. |
| `evidence_asset` | `evidence.tar.gz`: the qualification receipts, launcher observations and the raw evidence they reference. |

The bundle's `bin/` is not in git because a hosted CPU runner cannot build or qualify CUDA binaries.

## Steps

All commands run on a maintainer's machine with an authenticated `gh`, never in CI. They need a clean
checkout of this repository at the commit you pass as `--target-commit`.

### 0. Register the approval

Open a pull request that adds the `approvals["<runtime-id>/v<N>"]` entry to
`runtimes/templates/APPROVALS.json` and updates `TRUSTED_APPROVALS_SHA256` in
`.github/workflows/verify.yml` to the new file's sha256. Add the template digest to
`runtimes/templates/APPROVED.json` in the same pull request if the template version is new. A second
maintainer reviews it and merges it. This must be merged before step 1, because CI reads the registry
from the base branch.

### 1. Prepare the pull request (no `LATEST`)

```bash
git fetch origin && git switch --detach origin/main      # clean checkout; HEAD is the base commit
python3 tools/promote_runtime.py \
  --runtime-id <id> --approve-tag approve-<id>-<N> --version <N> \
  --target-commit "$(git rev-parse HEAD)" --prod-checkout "$PWD" \
  --assets-dir <durable-dev-assets> --download \
  --approved runtimes/templates/APPROVED.json \
  --trusted-approvals runtimes/templates/APPROVALS.json \
  --trusted-approvals-sha256 "$(sed -n 's/^ *TRUSTED_APPROVALS_SHA256: *//p' .github/workflows/verify.yml)"
```

The hash comes from the pin in the workflow, a second reviewed file, not from the registry it checks.

`--download` fetches the dev release with your `gh` login, validates release and asset ids, sizes, the
tag commit, every byte and all evidence, and caches the verified files. The tool then creates branch
`promote/<id>-v<N>`, extracts the tree after checking every tar header (absolute, `..`, duplicate,
link, device, sparse and oversized members are refused before any write), verifies it, commits it and
opens a pull request. `LATEST` is not written. `--dry-run` verifies and prints what it would do
without changing the tree.

Before asking for review, give the pull request a title in the form `FEAT: Promoted <id> v<N>` (the
tool's own title does not follow the rule in `CONTRIBUTING.md`; the title becomes the squash commit
subject) and paste the tool's `VERIFIED_PR` line into the description. Check the branch's commit
author; it must be a noreply identity.

### 2. Review and merge

The other maintainer reviews the diff, the registry entry from step 0 and the tool output, and merges
with **squash**; that is standard practice because a promotion changes what testers install. If only
one maintainer is available, that maintainer may merge the promotion they prepared once `verify` is green. Nothing merges the pull request automatically, and the `promote/`
branch prefix grants no privilege.

### 3. Publish the release

```bash
git fetch origin && git switch --detach <merged-commit-sha>
python3 tools/promote_runtime.py \
  --runtime-id <id> --approve-tag approve-<id>-<N> --version <N> \
  --publish-only --target-commit <merged-commit-sha> --prod-checkout "$PWD" \
  --assets-dir <durable-dev-assets> \
  --approved runtimes/templates/APPROVED.json \
  --trusted-approvals runtimes/templates/APPROVALS.json \
  --trusted-approvals-sha256 "$(sed -n 's/^ *TRUSTED_APPROVALS_SHA256: *//p' .github/workflows/verify.yml)"
```

`--publish-only` requires a clean checkout of exactly the merged commit, reachable from `origin/main`,
whose tree and approval bytes equal the verified assets. It creates `prod/<id>/v<N>` with
`--target <merged-commit-sha>`, `--latest=false` and the five assets, downloads them back and compares
them with the approval, and writes `publication-<id>-v<N>.json` into the assets directory. That file
pins the release id, asset ids, sizes and measured hashes. A retry must match it; a changed release,
asset or tag fails closed, and a fully verified retry is a no-op. An existing release is downloaded and
verified, never accepted blindly.

### 4. Move `LATEST`

The tools never write `LATEST`. After step 3 succeeds, open a separate pull request that changes
`runtimes/<id>/LATEST` to `v<N>`. `LATEST` therefore never points at a tree whose release is missing.

### 5. Publish the pack sheet

Testers read a pack's sheet before they run it, so a released version is not finished until its sheet is
current. After step 4, open a `DOCS:` pull request that adds or updates `docs/packs/<runtime-id>.md` from
the sealed files (copy `docs/packs/TEMPLATE.md` for a new runtime; its rules are in `docs/packs/README.md`)
and the runtime's row in the README's released-packs table. The pull request names the release tag it was
checked against. Docs pull requests never touch `runtimes/`.

## Verification in CI

`.github/workflows/verify.yml` (job `verify`) runs on `pull_request_target` and on pushes to `main`:

- It checks out the protected base into `base/` and the pull request head into `candidate/` as data
  only, runs `base/tools/verify_runtime_tree.py` over `candidate/`, then runs the unit tests from
  `base/`. The workflow token is read-only and actions are pinned by commit SHA. The one secret,
  `DEV_ASSETS_TOKEN`, is passed only to the step that downloads dev release assets; the verification and
  test steps have none. There is no `merge` job.
- A tree whose approval is `rexl3.profile-approval.v1` passes only in `--allow-legacy` mode and only
  if the registry's `legacy` entry matches its approval hash, source commit and tree, and complete
  file inventory.
- A v2 tree needs its authenticated assets in `<assets>/<id>/v<N>/`. They are in the private
  `REXL3/REXL3-dev` release. The step before verification runs `base/tools/fetch_dev_assets.py` with
  `DEV_ASSETS_TOKEN` (a fine-grained token that reads that one repository's contents): it takes the id
  and version from the directory names, derives the tag `approve-<id>-<N>`, and downloads through
  `promote_runtime.download_release`, which refuses any archive the pinned registry does not name. A
  candidate cannot choose its own tag. A v1 tree needs no download.
- Because the verifier, tests and policy come from `base/`, a pull request's own changes to `tools/`,
  the tests, the registry or the pin are **not** exercised by its `verify` run; the first CI run of
  them is the push to `main` after the merge. For those pull requests the author's pasted local run
  on the head commit (below) is the only evidence before merge, and reviewers should run it too.
- `tools/test_verify_mutations.py` proves the verifier rejects a FAIL gate, a SKIP gate, zero tests, a
  wrong template digest, a runtime file edited together with its manifest entry, traversal and link
  tars, a self-edited approval and a v1 tree without `--allow-legacy`, and that the committed v1 tree
  still passes in legacy mode.

Run the same checks locally (use `umask 022`, which is what makes the file modes match):

```bash
umask 022
python3 -m unittest discover -s tools -p 'test_*.py'
python3 -B -E -s tools/verify_runtime_tree.py --root . \
  --approved runtimes/templates/APPROVED.json \
  --trusted-approvals runtimes/templates/APPROVALS.json \
  --trusted-approvals-sha256 "$(sha256sum runtimes/templates/APPROVALS.json | cut -d' ' -f1)" \
  --assets-dir "$(mktemp -d)" --allow-legacy
```

## Legacy v1

`runtimes/dsv41-flash-1017-sage-1.62/v1` was sealed before the receipt format. Its approval is
`rexl3.profile-approval.v1` (gates recorded as strings, no receipts, no engine source asset). Its
bytes are not rewritten. The registry holds its `legacy` anchor, and CI verifies it as history, not as
a new qualification. The v1 sealer and the `promote.yml` workflow that produced it no longer exist, and
v1 is never accepted by the promoter.

## Tags

| Tag | Meaning |
| --- | --- |
| `approve-<runtime-id>-<N>` | dev release; `<N>` and the production `v<N>` are the same integer |
| `prod/<runtime-id>/v<N>` | production release; same archive bytes as the dev release |
| `prod/<runtime-id>/v<N>-withdrawn` | marker added when a version is withdrawn |

`.github/rulesets/protected-tags.json` proposes forbidding update and deletion of `prod/*` tags.

## Withdrawing a bad pack

History is append-only. Revert the promotion commit through a pull request, add the `-withdrawn` tag,
and point `runtimes/<id>/LATEST` at the previous version in a pull request. Never force-push `main`.
Then update the pack sheet and the README table in a `DOCS:` pull request so testers are not pointed at
the withdrawn version.

## Open items

- **The asset download has not yet met a real v2 release.** `fetch_dev_assets.py` and the secret are in
  place and tested with mocks only. The first v2 promotion pull request is the first real run. Because
  `verify` downloads the assets of every v2 tree on every run, two things follow: a run is as slow as the
  download, and once a v2 tree is on `main`, an expired or revoked `DEV_ASSETS_TOKEN` turns `verify` red
  for every later pull request. The token is created with a 90-day expiry; rotate it before day 80 (create
  a new token, update the secret, delete the old token). If `verify` is red because of the token and a fix
  must land, the owner can temporarily edit the `main-guard` ruleset, merge, and restore it; write the
  reason in the pull request.
- **Corresponding source for v1.** The v1 release has no `engine_source_asset`. Its binaries were built
  from `source_commit` in `approval.json` in the private dev repository. Decide how that source is
  provided.
- **Branch protection.** `.github/rulesets/main-guard.json` is a proposal (one code-owner approval,
  squash only, `verify` required). The previous workflow noted that this repository's plan could not
  use rulesets; if it still cannot, the review and the manual merge are the only control.
- **`LATEST` and stray files.** The verifier checks `runtimes/<id>/v<N>/` trees. It does not verify
  `LATEST` or files placed elsewhere under `runtimes/<id>/`; reviewers must.
- **Tool conventions.** `tools/promote_runtime.py` is copied unchanged from dev. It names the PR
  `promote <id> v<N>` and commits as `r0b0tlab@users.noreply.github.com`; both are corrected by hand
  as described in step 1 until the tool is changed in dev.
