# Promotion: dev approval to production

Production (`REXL3/REXL3`) holds sealed runtime trees under `runtimes/<family>/<model>/` that a
dev-side approval authorised. Nothing is copied by hand, no workflow merges a pull request, and a
maintainer reviews every promotion. The authoritative description of the receipt format and the sealer
is `docs/PROMOTION.md` in `REXL3/REXL3-dev`; this file is the production side of the same contract.

## Trust boundary

A hash proves that bytes did not change. It does not prove that a benchmark ran. A tree is accepted
only if all of these hold:

1. The approval is `rexl3.profile-approval.v2` and its exact bytes are listed in
   `runtimes/templates/APPROVALS.json` under `approvals["<family>/<model>/v<N>"]`: `approval_sha256`,
   `source_commit`, `source_tree`, `runtime_id`. A maintainer reviews that entry; it is the independent
   anchor.
2. The registry file matches the hash pinned in `.github/workflows/verify.yml`
   (`TRUSTED_APPROVALS_SHA256`), so changing the registry also changes a second reviewed file.
3. The verifier, the registry and the template allow-list (`runtimes/templates/APPROVED.json`) are
   read from the protected base revision of `main`, never from the pull request. A pull request cannot
   change the rules it is judged by. `.github/CODEOWNERS` makes `@am423` and `@vcruz305` owners of
   `.github/`, `tools/`, `runtimes/` and `docs/`.
4. The authenticated dev release assets match the approval, and their receipts pass the strict gate
   rules: nine gates (`doctor`, `lifecycle`, `capability`, `vision`, `context`, `exactness`,
   `prefix_cache`, `performance`, `quality`), each with `total > 0`, `passed == total`, no failure and
   no skip. `FAIL`, `NOT_RUN`, zero tests and strings that look like a pass are refused.
5. The tree equals the wrapper projection plus the original `approval.json`, byte for byte: no missing,
   extra, edited, linked or mode-changed file.

One secret, `DEV_ASSETS_TOKEN`, lets `verify` read the private dev release (see "What CI checks").
There is no auto-merge anywhere.

## Definition of done for a model in dev

Dev a model may be proposed for production, the dev repository must have, for the exact version to be
released:

- a **final runtime entry** (the final runtime for the release, not a work-in-progress one);
- the **capability contract proven**: speculative decoding on with counters above 0; prefix cache on
  and proven at the real contract (the real context and slots); vision on and tested, or off only
  under an unexpired owner vision exception sealed in each vision-off profile's qualification receipt
  (accepted in place of a proven does-not-fit);
- **exactness** passed;
- **long context** passed;
- **GB10 receipts on the sealed build**: doctor, smoke and a gate subset run on a GB10 host,
  attached to the engine commit the model's approval is sealed from;
- a `KNOWN_ISSUES` file present in the tree;
- the **dev approval sealed from that build** (the sealed approval names the engine commit it was
  sealed from). A model whose scope is untouched by newer engine commits may instead carry forward a
  registry-pinned prior approval (T0); it is not re-sealed for the release.

Dev tracks a model's qualification with labels `model:MODEL` and `engine`, and scope files under
`releases/v0.1.0/models` (added by dev tooling; plain text).

## The two-PR flow

1. **`model/<model>-approval`** — register the dev approval: add the entry to
   `runtimes/templates/APPROVALS.json` and update the `TRUSTED_APPROVALS_SHA256` pin in
   `.github/workflows/verify.yml` in the same commit (tool tests enforce the two agree). This merges
   first, because CI trusts only the registry on main.
2. **`model/<model>`** — bring the validated model into prod: `runtimes/<family>/<model>/**`
   (`MODEL.json`, `LATEST`, `v<N>/`), `docs/packs/<model>.md`, `changes/unreleased/<model>.md`. This
   PR needs the `local-validation` check (below).

There are no separate LATEST or pack-sheet PRs, no separate publish step, and no `prod/<runtime>/v<N>`
release. `LATEST` is part of the model PR's tree addition.

## promote_runtime

`tools/promote_runtime.py` produces the sealed tree at `runtimes/<family>/<model>/v<N>` from the dev
approval assets (dev release `approve-<runtime_id>-<N>`). It runs on a maintainer's machine with an
authenticated `gh`, never in CI. The tool derives dev release tags from `runtime_id` — read from
`MODEL.json` and the sealed approval, which must agree — never from folder names. A maintainer runs it;
an agent runs it only on an explicit, quoted maintainer instruction (CONTRIBUTING.md section 7).

```bash
python3 tools/promote_runtime.py \
  --family <family> --model <model> --version <N> \
  --approve-tag approve-<runtime_id>-<N> \
  --target-commit "$(git rev-parse HEAD)" --prod-checkout "$PWD" \
  --assets-dir <durable-dev-assets> [--download] \
  --approved runtimes/templates/APPROVED.json \
  --trusted-approvals runtimes/templates/APPROVALS.json \
  --trusted-approvals-sha256 "$(sed -n 's/^ *TRUSTED_APPROVALS_SHA256: *//p' .github/workflows/verify.yml)" \
  [--dry-run] [--display-name "Display Name" --engine-family <module> \
   --pack-format exl3 --architecture <HFClass> [--pack KEY=VALUE ...]]
```

`--download` validates release and asset ids, sizes, the tag commit, every byte and all evidence and
caches the verified files in `<assets-dir>/<family>/<model>/v<N>/` (with the engine release's executables in
its `engine/`), the directory the promotion then reads. The tool then writes `MODEL.json` (or validates an existing one against the
derived metadata; a mismatch is a hard refusal, never an overwrite), extracts the tree, writes
`LATEST`, verifies against the base registry, commits branch `model/<model>` and opens a PR titled
`FEAT: Promoted <model> v<N>`. `--dry-run` verifies and prints without writing.

## The model-branch completeness checker

`tools/check_model_branch.py` is the completeness gate for `model/<model>` PRs:

```
python3 tools/check_model_branch.py --repo-root PATH --base REF --head REF \
  [--branch REF] [--rc-commit SHA] [--tree-dir PATH] [--base-tree-dir PATH] \
  [--assets-dir PATH] [--local]
```

`--base`/`--head` are refs in `--repo-root`. `--tree-dir` defaults to `--repo-root`; `--base-tree-dir`
defaults to reading base files from git objects; `--assets-dir` defaults to `$REXL3_ASSETS_DIR` (CI
passes it explicitly). It prints one line per check (`PASS`/`FAIL name: reason`), exit 0 all pass, 1
any fail, 2 usage error. The check names are `branch/model`, `scope`, `MODEL.json`, `LATEST`, `tree`,
`source commit`, `capability contract`, `docs and fragment`. Checks:

1. Branch/model: the changed `runtimes/<family>/<model>/` folder is exactly one model and matches the
   branch.
2. Scope: changed paths are within CONTRIBUTING.md section 3 for `model/<model>`.
3. `MODEL.json` valid per CONTRIBUTING.md section 2 and consistent with its folder.
4. `LATEST` names an existing `v<N>`.
5. The tree passes `tools/verify_runtime_tree.py` against the registry on the base.
6. If `--rc-commit` is given, the sealed approval's source commit equals it.
7. Capability contract from the sealed profile: speculative decoding on, prefix cache on, vision on
   unless `MODEL.json` records a `vision_does_not_fit` justification in `pack` or every profile's
   sealed qualification receipt carries an unexpired owner vision exception (step 5 already refuses a
   vision-off receipt without a valid one).
8. `docs/packs/<model>.md` and `changes/unreleased/<model>.md` exist and are non-empty.

`--local` additionally downloads and digest-checks the dev assets via `tools/fetch_dev_assets.py`.

## The reviewer's local procedure (LOCAL-VALIDATION v1)

For a `model/<model>` PR, a maintainer other than the author reviews locally before approving:

1. Fetch the branch: `gh pr checkout <N>` (or fetch the head SHA).
2. Run the completeness checker with `--local` on the head:
   `python3 tools/check_model_branch.py --repo-root . --base main --head <HEAD> --local`.
3. Verify the assets: the downloaded dev assets match the approval byte for byte.
4. On a GB10 host, run the tree's `launch.sh doctor` and `launch.sh smoke`, plus a gate subset
   (capability, context, prefix_cache at minimum).
5. Post the `LOCAL-VALIDATION v1` block from CONTRIBUTING.md section 4 as an APPROVED review on the
   current head. CI evaluates it with
   `python3 tools/check_local_validation.py --repo OWNER/REPO --pr N --head-sha SHA --codeowners PATH --token TOKEN`,
   which sets the commit status `local-validation` on the head SHA and re-evaluates on push and on
   review submit/dismiss.

## What CI checks

Required on every PR to main (each ALWAYS reports; job-level conditions that succeed as a no-op, never
workflow-level path or branch filters):

- `verify` — the tree verifier plus unit tests, from the protected base revision, with the
  `DEV_ASSETS_TOKEN` download step for v2 trees.
- `pr-rules` — title, branch name, path scope per branch type, `changes/unreleased/` fragment presence.
- `model-completeness` — runs `tools/check_model_branch.py` for `model/<model>` PRs; no-op success
  otherwise.
- `local-validation` — the APPROVED review with the `LOCAL-VALIDATION v1` block for model PRs; no-op
  success for non-model PRs and `model/*-approval` PRs.

## Withdrawal and superseding

- Releases are immutable and a bad release is superseded by a patch release (`docs/RELEASING.md`);
  deletion needs both maintainers.
- A bad model tree is superseded by a corrected promoted version from dev. Removing a tree or its
  registry entry needs both maintainers; the registry entry and the `TRUSTED_APPROVALS_SHA256` pin
  change in the same commit.

## Legacy v1

Production once held `runtimes/dsv41-flash-1017-sage-1.62/v1`, sealed before the receipt format
(approval schema `rexl3.profile-approval.v1`: gates recorded as strings, no receipts, no engine source
asset). It was removed from production by maintainer decision, with its release and tag, and returns
only as a new promoted version when its runtime is finished. The verifier requires the family/model
layout and refuses the old flat `runtimes/<runtime-id>/v<N>` layout and v1-approval trees outright
there is no `--allow-legacy` switch; the registry keeps an empty `legacy` section for the removed
history, and a tree of that schema can only return as a new sealed version from the dev repository.
