## Purpose

<!-- One sentence, no "and". The PR title is `TYPE: Past-tense summary` (CONTRIBUTING.md section 3). -->

## What and why

<!-- What changes, why, and which part of production it touches: tools, policy, workflow, docs, or a promoted runtime tree. -->

## How it was checked

Run on this head commit (SHA: ), with `umask 022`; paste the output:

- [ ] `git diff --check`
- [ ] `python3 -m unittest discover -s tools -p 'test_*.py'`
- [ ] `python3 -B -E -s tools/verify_runtime_tree.py ... --allow-legacy` (the command in CONTRIBUTING.md section 7)
- [ ] Rebased on `origin/main` (commit: ); one commit; the subject equals the PR title

## Contracts (tick what applies, explain what does not)

- [ ] No file under `runtimes/*/v*/` is edited. A promotion adds a new `v<N>` only, produced by `tools/promote_runtime.py`, and pastes its `VERIFIED_PR` line
- [ ] Policy files (`runtimes/templates/APPROVED.json`, `APPROVALS.json`, the `TRUSTED_APPROVALS_SHA256` pin in `.github/workflows/verify.yml`) change only in a PR whose purpose is that change; the pin equals the sha256 of the registry
- [ ] The workflow still reads the verifier, tests and policy from the base revision, executes nothing from the PR head, uses `permissions: contents: read` and no secret, and pins every action by full commit SHA
- [ ] No test, gate or check was weakened in this PR; every new check has a mutation that fails a named test
- [ ] Files copied from the dev repository are byte-identical to the dev commit named in the commit message (or the difference is stated)
- [ ] Licences: copied code is MIT, Apache-2.0 or BSD-3-Clause and attributed; no model weights
- [ ] No secrets, personal email addresses, internal project names or quantization-method internals
- [ ] Docs updated: what was checked, how, and what is not done
