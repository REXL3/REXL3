## Purpose

<!-- One sentence, no "and": bring `model/<model>` into production. The PR title is `TYPE: Past-tense summary` (CONTRIBUTING.md section 5). -->

## What and why

<!-- Which model, which family folder, which dev approval and version. Scope: `runtimes/<family>/<model>/**`, `docs/packs/<model>.md`, `changes/unreleased/<model>.md` (CONTRIBUTING.md section 3). -->

## How it was checked

Run on this head commit (SHA: ), with `umask 022`; paste the output:

- [ ] `git diff --check`
- [ ] `python3 -m unittest discover -s tools -p 'test_*.py'`
- [ ] Completeness checker output, pasted in full: `python3 tools/check_model_branch.py --repo-root . --base main --head <HEAD>`
- [ ] Rebased on `origin/main` (commit: ); one commit; the subject equals the PR title

## Dev evidence

Link the dev-side evidence for the sealed approval:

- [ ] Dev approval release `approve-<runtime_id>-<N>` (link)
- [ ] Sealed approval source commit equals the dev release-candidate commit (the RC commit)
- [ ] Qualification receipts on the RC commit: GB10 doctor, smoke and gate subset
- [ ] `KNOWN_ISSUES` present; capability contract proven (speculative decoding, prefix cache at the real contract, vision tested or proven does-not-fit)

## Contracts

- [ ] The changed `runtimes/<family>/<model>/` folder is exactly one model and matches the branch
- [ ] `MODEL.json` valid per CONTRIBUTING.md section 2 (`family`, `model`, `runtime_id`, `pack.format` and the rest)
- [ ] `LATEST` points at an existing `v<N>`; no file under another model's tree changed
- [ ] `docs/packs/<model>.md` and `changes/unreleased/<model>.md` exist and are non-empty
- [ ] No secrets, tokens, personal email addresses, internal project names or quantization-method internals

## Release notes fragment

- [ ] `changes/unreleased/<model>.md` present (rules in `changes/unreleased/README.md`)

## Reviewer local validation

- [ ] A maintainer other than the author ran `python3 tools/check_model_branch.py --repo-root . --base main --head <HEAD> --local`, verified the assets, ran `launch.sh doctor` and `launch.sh smoke` plus a gate subset on a GB10, and posted a `LOCAL-VALIDATION v1` block as an APPROVED review (CONTRIBUTING.md section 4)
