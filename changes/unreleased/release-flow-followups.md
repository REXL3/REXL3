## Bug fixes

- `tools/promote_runtime.py` can add a version the registry does not hold yet. The documented
  `--release-branch` command refused the first promotion of a cut with
  `approval not in independently trusted registry`. A maintainer-written `--new-anchor FILE`, outside
  the checkout and holding exactly the promoted key, is now authenticated like any other registry
  entry and committed as that model's `runtimes/templates/approvals/<family>/<model>.json` (#163).
- `tools/check_model_branch.py` now reads the base registry together with its modular
  `approvals/<family>/<model>.json` files, not only the top-level `APPROVALS.json` (#163).
- Registry modules are bound to their path. Every key in `approvals/<family>/<model>.json` must be
  `<family>/<model>/v<N>`, and empty modules or modules with other sections are refused (#163).
- `pr-rules` no longer lets a `model/*-approval` PR change `.github/workflows/verify.yml`. The retired
  `TRUSTED_APPROVALS_SHA256` pin check used to pass any added line that contained that name (#163).

## New features and improvements

- `tools/promote_runtime.py --co-author "Name <id+login@users.noreply.github.com>"` (repeatable) adds
  `Co-authored-by:` trailers to a promotion commit, so tester credit reaches main in the single-PR
  release flow (docs/RELEASING.md) (#163).
- The release-cut `LOCAL-VALIDATION v1` forms (`models:` and the `sealed_receipts: PASS` waiver) are
  now documented, and tests show that the waiver never applies to a `model/<model>` PR (#163).
- The docs no longer promise the standalone `model/<model>-approval` route for a new version: verify
  refuses an anchor without its tree. A single-model hotfix that needs a new anchor is a one-model
  `release/v<X.Y.Z>` patch release PR (#163).
