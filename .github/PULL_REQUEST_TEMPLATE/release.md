## Purpose

<!-- One sentence, no "and": cut the product release v<X.Y.Z>. This PR is opened by a maintainer only (CONTRIBUTING.md section 3). The PR title is `TYPE: Past-tense summary`. -->

## What and why

<!-- The release contents: `releases/v<X.Y.Z>/manifest.json`, `releases/v<X.Y.Z>/NOTES.md`, the `CHANGELOG.md` entry, and the consumed `changes/unreleased/*.md` fragments (docs/RELEASING.md). -->

## How it was checked

Run on this head commit (SHA: ), with `umask 022`; paste the output:

- [ ] `git diff --check`
- [ ] `python3 -m unittest discover -s tools -p 'test_*.py'`
- [ ] Manifest validated against schema `rexl3.release-manifest.v1` and checked for consistency with main (engine dev commit/tree/tag/digests; every model entry matches the registry)
- [ ] Notes assembled via `python3 tools/assemble_release_notes.py --repo-root . --version vX.Y.Z` into `docs/templates/RELEASE_NOTES.md`; `CHANGELOG.md` entry present
- [ ] `python3 tools/cut_release.py --version vX.Y.Z` (dry run) passes on this head
- [ ] Rebased on `origin/main` (commit: ); one commit; the subject equals the PR title

## Manifest and checklist

- [ ] `version` = `vX.Y.Z`; tag `v<X.Y.Z>` absent; immutable releases enabled
- [ ] `engine` block: `dev_commit`, `dev_tree`, `engine_release_tag`, artifact digests
- [ ] `models`: every first-release model present with `family`, `model`, `version` (v<N>), `runtime_id`, `approval_sha256`; all sealed from the SAME dev release-candidate commit
- [ ] `notes` present; `releases/v<X.Y.Z>/NOTES.md` present; consumed `changes/unreleased/*.md` fragments deleted
- [ ] Approved by the other maintainer (squash merge); the cut_release ceremony then runs per docs/RELEASING.md

## Release notes fragment

- [ ] Not needed: this PR's notes live in `releases/v<X.Y.Z>/NOTES.md` and `CHANGELOG.md`
