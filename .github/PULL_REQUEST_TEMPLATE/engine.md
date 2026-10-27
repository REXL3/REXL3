## Purpose

<!-- One sentence, no "and". The PR title is `TYPE: Past-tense summary` (CONTRIBUTING.md section 5). -->

## What and why

<!-- What changes, why, and which part of production it touches: tools, workflows, policy, docs, templates. Scope: anything except `runtimes/*/*/v*/**` and `runtimes/templates/APPROVALS.json` (CONTRIBUTING.md section 3). -->

## How it was checked

Run on this head commit (SHA: ), with `umask 022`; paste the output:

- [ ] `git diff --check`
- [ ] `python3 -m unittest discover -s tools -p 'test_*.py'`
- [ ] For behavior-changing tool/workflow PRs: `python3 -B -E -s tools/verify_runtime_tree.py --root . --approved runtimes/templates/APPROVED.json --trusted-approvals runtimes/templates/APPROVALS.json --trusted-approvals-sha256 "$(sed -n 's/^ *TRUSTED_APPROVALS_SHA256: *//p' .github/workflows/verify.yml)" --assets-dir "$(mktemp -d)"` (the command in CONTRIBUTING.md section 6), run on the head
- [ ] Rebased on `origin/main` (commit: ); one commit; the subject equals the PR title

## Contracts

- [ ] The title, branch name and changed paths match the branch type's scope (CONTRIBUTING.md section 3)
- [ ] No file under `runtimes/*/*/v*/` is edited; the approval registry does not change here
- [ ] No test, gate or check was weakened in this PR
- [ ] Workflow changes keep the security model: code and policy read from the protected base, nothing executes PR-head code, read-only permissions unless a narrow write is justified, actions pinned by full commit SHA
- [ ] No secrets, tokens, personal email addresses, internal project names or quantization-method internals
- [ ] Docs updated: what was checked, how, and what is not done

## Release notes fragment

- [ ] Behavior-changing PR: add `changes/unreleased/<slug>.md` (rules in `changes/unreleased/README.md`); not needed for DOCS-only PRs
