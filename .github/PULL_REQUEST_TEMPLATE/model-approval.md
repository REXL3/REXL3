## Purpose

<!-- One sentence, no "and": register the dev approval for `model/<model>` version v<N>. The PR title is `TYPE: Past-tense summary` (CONTRIBUTING.md section 5). -->

## What and why

<!-- Which dev approval (release `approve-<runtime_id>-<N>`), which model entry, which registry change. Scope: `runtimes/templates/APPROVALS.json` and the `TRUSTED_APPROVALS_SHA256` pin in `.github/workflows/verify.yml` (CONTRIBUTING.md section 3). This PR merges before the model PR. -->

## How it was checked

Run on this head commit (SHA: ), with `umask 022`; paste the output:

- [ ] `git diff --check`
- [ ] `python3 -m unittest discover -s tools -p 'test_*.py'`
- [ ] The pin in `.github/workflows/verify.yml` equals the sha256 of the new registry: `shasum -a 256 runtimes/templates/APPROVALS.json`
- [ ] Rebased on `origin/main` (commit: ); one commit; the subject equals the PR title

## Contracts

- [ ] The registry entry is keyed `<family>/<model>/v<N>` and carries `approval_sha256`, `source_commit`, `source_tree`, `runtime_id`
- [ ] `runtime_id` matches the dev release tag `approve-<runtime_id>-<N>` and the model's `MODEL.json`
- [ ] The pin matches the registry bytes; both change in this same commit
- [ ] No file under `runtimes/*/*/v*/` is edited; no other policy file changes
- [ ] No secrets, tokens, personal email addresses, internal project names or quantization-method internals

## Release notes fragment

- [ ] `changes/unreleased/<model>.md` present (rules in `changes/unreleased/README.md`)
