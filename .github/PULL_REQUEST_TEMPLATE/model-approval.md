## Purpose

<!-- One sentence, no "and": register the dev approval for `model/<model>` version v<N>. The PR title is `TYPE: Past-tense summary` (CONTRIBUTING.md section 5). -->

## What and why

<!-- Which dev approval (release `approve-<runtime_id>-<N>`), which model entry, which registry change. Scope: `runtimes/templates/approvals/<family>/<model>.json` or `runtimes/templates/APPROVALS.json` only; `.github/workflows/verify.yml` is out of scope (CONTRIBUTING.md section 3). Note: `verify` refuses a new anchor whose tree is not in the same run, so this route cannot add a new version on its own; a hotfix with a new anchor is a one-model `release/v<X.Y.Z>` PR. -->

## How it was checked

Run on this head commit (SHA: ), with `umask 022`; paste the output:

- [ ] `git diff --check`
- [ ] `python3 -m unittest discover -s tools -p 'test_*.py'`
- [ ] Rebased on `origin/main` (commit: ); one commit; the subject equals the PR title

## Contracts

- [ ] The registry entry is keyed `<family>/<model>/v<N>` and carries `approval_sha256`, `source_commit`, `source_tree`, `runtime_id`
- [ ] `runtime_id` matches the dev release tag `approve-<runtime_id>-<N>` and the model's `MODEL.json`
- [ ] A module file holds only `schema` and `approvals`, and every key in it is this module's own `<family>/<model>/v<N>`
- [ ] No file under `runtimes/*/*/v*/` is edited; no other policy file changes
- [ ] No secrets, tokens, personal email addresses, internal project names or quantization-method internals

## Release notes fragment

- [ ] `changes/unreleased/<model>.md` present (rules in `changes/unreleased/README.md`)
