# Promotion: dev approval to production

Production (`REXL3/REXL3`) holds sealed runtime trees under `runtimes/<family>/<model>/` that a
dev-side approval authorised. Nothing is copied by hand, no workflow merges a pull request, and a
maintainer reviews every promotion. The authoritative description of the receipt format and the sealer
is `docs/PROMOTION.md` in `REXL3/REXL3-dev`; this file is the production side of the same contract.

## Trust boundary

A hash proves that bytes did not change. It does not prove that a benchmark ran. A tree is accepted
only if all of these hold:

1. The approval is `rexl3.profile-approval.v2` and its exact bytes are listed under
   `approvals["<family>/<model>/v<N>"]`: `approval_sha256`, `source_commit`, `source_tree`, `runtime_id`.
   The anchor lives in the top-level `runtimes/templates/APPROVALS.json` or its modular
   `runtimes/templates/approvals/<family>/<model>.json` file. The loader merges both forms, refuses
   duplicate keys, and treats every reviewed entry as an independent anchor.
2. Existing base anchors in either `APPROVALS.json` or `runtimes/templates/approvals/<family>/<model>.json`
   are append-only: deleting or mutating an existing base anchor is refused. A newly added modular anchor
   is reviewed in the same `release/v<X.Y.Z>` PR as its tree and must be downloaded and fully verified
   (approval bytes, every asset and the strict receipt rules of item 4) in that same run
   (`TRUSTED_APPROVALS_SHA256` in `verify.yml` is retired to eliminate multi-model rebase conflicts). A
   module holds only `schema` and a non-empty `approvals` section whose keys all name its own
   `<family>/<model>`; a misfiled key, an empty module or another section is refused.
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
6. Diff-scoped verification skips only the remote download and archive revalidation for a tree whose
   bytes equal the protected base and whose anchor is unchanged. It still validates every candidate
   `MODEL.json`, `LATEST`, `v<N>/approval.json` and registry binding. Every newly added anchor must
   name a discovered tree that is downloaded and fully verified in that same run; orphan anchors and
   candidate-only carried-gate source anchors are refused. `--full` disables the transport skip.

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

## Promotion flows

The standard product cut is a **single-PR `release/v<X.Y.Z>` flow**, opened on explicit maintainer
instruction from **one unified `engine-v<X.Y.Z>-<8hex>` dev release** (never create separate
`approve-<runtime_id>-<N>` dev releases or `2N` per-model PRs for a product cut):

- One reviewed `release/v<X.Y.Z>` PR carries the corresponding engine source, every promoted model tree and
  its `MODEL.json`/`LATEST`, modular `runtimes/templates/approvals/<family>/<model>.json` anchors, pack docs,
  release manifest, notes and `CHANGELOG.md`.
- Its `LOCAL-VALIDATION v1` review covers every changed model at the exact PR head `commit_id`.
- A `release/v<X.Y.Z>` PR is the only branch that may combine engine source, sealed trees and registry
  anchors; an `engine/*` branch may change neither registry form.

Fresh models in one product cut use the unified internal dev release `engine-v<X.Y.Z>-<8hex>`, where
`<X.Y.Z>` is the native engine version and `<8hex>` is the first eight lowercase hex digits of the
approval's full `source_commit`. Each model has prefixed approval, bundle, source and evidence assets and
shares the release's bound `rexl3-engine-source-<X.Y.Z>-<8hex>.tar.gz` archive. Older `approve-<runtime_id>-<N>`
approvals remain valid for legacy and T0 carried-forward entries.

*(Single-model hotfix: the standalone `model/<model>-approval` -> `model/<model>` path cannot add a new
version today. `verify` refuses a new anchor whose tree is not in the same run (`new candidate registry
anchor has no discovered runtime tree`), and a `model/<model>` PR may not touch the registry. A hotfix
that needs a new anchor is a one-model `release/v<X.Y.Z>` patch release PR. A `model/<model>` PR still
works for a tree whose anchor is already on the base, in either registry form.)*

There are no separate LATEST or pack-sheet PRs, no separate publish step, and no `prod/<runtime>/v<N>`
release. `LATEST` is part of the tree addition. When a later version of a model already on `main` is
promoted, the tool moves `LATEST` forward from the earlier `v<M>` to the new `v<N>`.

## promote_runtime

`tools/promote_runtime.py` stages each sealed tree at `runtimes/<family>/<model>/v<N>` and its modular
anchor `runtimes/templates/approvals/<family>/<model>.json` onto `release/v<X.Y.Z>` from the unified
`engine-v<X.Y.Z>-<8hex>` dev release. It runs on a maintainer's machine with an authenticated `gh`,
never in CI:

```bash
python3 tools/promote_runtime.py \
  --family <family> --model <model> --version <N> \
  --approve-tag engine-v<X.Y.Z>-<8hex> \
  --release-branch release/v<X.Y.Z> \
  --target-commit "$(git rev-parse HEAD)" --prod-checkout "$PWD" \
  --assets-dir <durable-dev-assets> [--download] \
  --approved runtimes/templates/APPROVED.json \
  --trusted-approvals runtimes/templates/APPROVALS.json \
  [--new-anchor "$ANCHORS/<model>.json"] [--co-author "Name <id+login@users.noreply.github.com>" ...] \
  [--dry-run] [--display-name "Display Name" --engine-family <module> \
   --pack-format exl3 --architecture <HFClass> [--pack KEY=VALUE ...]]
```

**A new anchor.** The clean checkout cannot hold the anchor of a version it is about to promote
(the tool refuses a dirty checkout), so without help the first promotion of a cut refuses with
`approval not in independently trusted registry: <family>/<model>/v<N>`. The maintainer writes that
one anchor in a file **outside the checkout** and passes it with `--new-anchor` (accepted only with
`--release-branch`). The file has exactly the form of the module the promotion commits:

```json
{"schema": "rexl3.approval-registry.v1",
 "approvals": {"<family>/<model>/v<N>": {
   "approval_sha256": "<the approval_sha256= line the dev seal printed>",
   "source_commit": "<40-hex>", "source_tree": "<40-hex>", "runtime_id": "<runtime_id>",
   "approve_tag": "engine-v<X.Y.Z>-<8hex>"}}}
```

`approve_tag` is required for a unified `engine-v*` tag. `binary_sha256` and the per-profile receipt
pins are optional, as for any entry (CONTRIBUTING.md section 2). The file must hold exactly the promoted
key, the registry must not already hold it, and the entry passes the same schema check as every
registry entry. The downloaded approval is then authenticated against it exactly as against an anchor
already on main: its bytes must hash to `approval_sha256`, and its source commit, source tree,
runtime id and tag must match. Carried-gate sources still resolve only against the checkout's
registry, never against the new anchor. The promotion commits the entry as
`runtimes/templates/approvals/<family>/<model>.json`, and the release PR review is where the
maintainers authorise it.

**Tester credit.** Each `--co-author "Name <id+login@users.noreply.github.com>"` adds a
`Co-authored-by:` trailer to the promotion commit (GitHub noreply addresses only; one line each;
duplicates refused). The subject stays `FEAT: Promoted <model> v<N>`. docs/RELEASING.md says where
credit goes in a release cut.

`--download` validates release and asset ids, sizes, the tag commit, every byte and all evidence and
caches the verified files in `<assets-dir>/<family>/<model>/v<N>/` (with the engine release's executables in
its `engine/`), the directory the promotion then reads. The tool then writes `MODEL.json` (or validates an existing one against the
derived metadata; a mismatch is a hard refusal, never an overwrite, except a reviewed runtime rename:
MODEL.json moves to the runtime's current id, and a reviewed pack-repo rename, only when nothing else changes and
every sealed tree in the folder still matches it), extracts the tree, writes
`LATEST`, verifies against the registry, and commits `FEAT: Promoted <model> v<N>`. With
`--release-branch` it commits locally on `release/v<X.Y.Z>`, together with the new module anchor
when there is one, and never pushes. Without it, it commits branch `model/<model>` and opens a PR
with that title. `--dry-run` verifies and prints without writing.

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
   current head. Both the review body's `head` field and the review API's `commit_id` must equal that
   head; editing an old review body cannot revalidate a new commit. CI evaluates it with
   `python3 tools/check_local_validation.py --repo OWNER/REPO --pr N --head-sha SHA --codeowners PATH --token TOKEN`,
   which sets the commit status `local-validation` on the head SHA and re-evaluates on push and on
   review submit/dismiss.

## What CI checks

Required on every PR to main (each ALWAYS reports; job-level conditions that succeed as a no-op, never
workflow-level path or branch filters):

- `verify` — the tree verifier plus unit tests, from the protected base revision. With `--base`,
  unchanged trees use the diff-scoped transport skip described above; changed or newly anchored v2
  trees still require the authenticated `DEV_ASSETS_TOKEN` download and full verification.
- `pr-rules` — title, branch name, path scope per branch type, `changes/unreleased/` fragment presence.
- `model-completeness` — runs `tools/check_model_branch.py` for `model/<model>` PRs; no-op success
  otherwise.
- `local-validation` — the APPROVED review with the `LOCAL-VALIDATION v1` block for model PRs; no-op
  success for non-model PRs and `model/*-approval` PRs.

## Withdrawal and superseding

- Releases are immutable and a bad release is superseded by a patch release (`docs/RELEASING.md`);
  deletion needs both maintainers.
- A bad model tree is superseded by a corrected promoted version from dev. Existing sealed registry
  anchors are append-only: the diff-scoped verifier refuses their deletion or mutation, and a fix
  lands as a new `v<N>` anchor and tree.

## Legacy v1

Production once held `runtimes/dsv41-flash-1017-sage-1.62/v1`, sealed before the receipt format
(approval schema `rexl3.profile-approval.v1`: gates recorded as strings, no receipts, no engine source
asset). It was removed from production by maintainer decision, with its release and tag, and returns
only as a new promoted version when its runtime is finished. The verifier requires the family/model
layout and refuses the old flat `runtimes/<runtime-id>/v<N>` layout and v1-approval trees outright
there is no `--allow-legacy` switch; the registry keeps an empty `legacy` section for the removed
history, and a tree of that schema can only return as a new sealed version from the dev repository.
