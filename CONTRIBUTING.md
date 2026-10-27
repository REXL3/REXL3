# Contributing to REXL3 production

This repository (`REXL3/REXL3`) is the **production** side of REXL3, published under AGPL-3.0-only. Its content is sealed, data-only runtime trees under `runtimes/`, the policy that decides which trees are accepted, and the small stdlib-Python tools that verify and promote them. There is no engine source here. The engine, the runtime template, model profiles and every qualification gate live in the dev repository, `REXL3/REXL3-dev`; follow that repository's `CONTRIBUTING.md` for engine and profile work.

This file is the single source of truth for humans and agents. Agents must also read `AGENTS.md`; where they differ, this file wins. The rules mirror the dev repository's where they apply.

## 1. Golden rules

| # | Rule |
|---|------|
| 1 | Never push to `main`: not directly, not by merge, not by force-push. Every change lands through a pull request. |
| 2 | One purpose per PR. If you cannot state it in one sentence without "and", split the PR. |
| 3 | Every PR title is `TYPE: Past-tense summary` (section 3). |
| 4 | Standard practice: the other maintainer (`@am423`, `@vcruz305`) reviews every PR before it merges. If only one maintainer is available, that maintainer may merge their own PR once `verify` is green and review threads are resolved. A PR from anyone else needs a maintainer's approval. An AI agent merges only a PR that the maintainer it works for has told it to merge. |
| 5 | **Never hand-edit `runtimes/<id>/v<N>/`.** A sealed tree changes only by a new promoted version, never in place. |
| 6 | Models enter production only through the dev repository's approval and `tools/promote_runtime.py`, run by a maintainer or by an agent that maintainer has explicitly told to run it (`docs/PROMOTION.md`, section 9). |
| 7 | Never weaken a test, a gate or the verifier in the same PR as the change it gates. That goes in its own TEST PR. |
| 8 | Exit code 0 is not evidence. Show which checks ran and what they printed. |
| 9 | No secrets, tokens, personal email addresses, internal project names or quantization-method internals in commits, PR text or docs. |
| 10 | When unsure, stop and ask a maintainer. Do not guess policy. |
| 11 | One logical commit per branch; its subject is the PR title. Base it on the latest `main`. |

## 2. PR types

Exactly seven. The TYPE is the uppercase prefix of the title.

| Type | Meaning | Typical paths here |
|---|---|---|
| FIX | Fixed a bug | `tools/**`, `.github/**` |
| FEAT | Added a feature, or promoted a runtime version | `tools/**`; a promotion adds `runtimes/<id>/v<N>/` |
| BUILD | Changed the build or packaging process | `.gitignore`, `tools/requirements*.txt` |
| CI | Changed CI or repository policy | `.github/**`, `tools/**` that the workflow runs, `runtimes/templates/**` |
| TEST | Added or fixed tests or checks | `tools/test_*.py` |
| BUMP | Bumped a pinned version (an action SHA, a Python requirement) | `.github/workflows/**`, `tools/requirements*.txt` |
| DOCS | Documentation only | `**/*.md` outside `runtimes/`, `docs/**` |

DOCS, TEST and BUMP pull requests must not touch `runtimes/`. A pull request that changes `runtimes/<id>/v<N>/` other than by adding a new version through `tools/promote_runtime.py` is rejected, whatever its type. A sealed tree also contains `.md` files; they are not documentation you may fix.

## 3. Title rule

Format `TYPE: Summary`:

- TYPE is uppercase and one of the seven types, followed by `": "`.
- The summary's first word is capitalized and past tense: it ends in "ed", or is an irregular past form (Built, Made, Wrote, Rewrote, Split, Set, Ran, Kept, Cut, Put, Found, Held, Led, Left, Read, Sent, Took, and similar).
- At most 90 characters. No trailing period. No "WIP".

Structural regex (the past-tense check is separate):

```regex
^(FIX|FEAT|BUILD|CI|TEST|BUMP|DOCS): ([A-Z][a-z]*)( .*)?$
```

Passes: `CI: Made production verify approval receipts from the protected base`, `DOCS: Rewrote the promotion guide for the two-phase path`, `FEAT: Promoted example-runtime-1.0 v7`.
Fails: `fix: add check` (lowercase type), `FEAT: Adds a check` (not past tense), `DOCS: Updated x.` (trailing period), `DOC: Added x` (not a type).

The title becomes the squash commit subject on `main`. A promotion pull request created by `tools/promote_runtime.py` is titled `promote <id> v<N>`; retitle it to `FEAT: Promoted <id> v<N>` before review.

## 4. Branch names

`<type lowercase>/<kebab-slug>`, for example `ci/prod-verifies-receipts-and-needs-review` or `docs/prod-contributor-rules-and-tester-readme`. `tools/promote_runtime.py` creates `promote/<id>-v<N>`; that prefix confers no privileges and nothing acts on it automatically.

## 5. Scope

There are no line or file-count limits. Scope is enforced by purpose: reviewers reject a diff that contains a change the title does not describe. Large work ships as a stack of PRs, each passing the checks on its own.

## 6. What you can change, and what you cannot

You can propose changes to:

- documentation (`README.md`, `docs/`, `SECURITY.md`, this file);
- `tools/` (verifier, promoter, their tests);
- `.github/` (workflows, CODEOWNERS, ruleset proposals, the PR template).

The following are **protected policy**. They decide what production accepts, they are read by CI from the base branch rather than from your pull request, and every change needs both maintainers' attention in review:

- `runtimes/templates/APPROVED.json` (template digests production accepts);
- `runtimes/templates/APPROVALS.json` (the approval registry), together with the `TRUSTED_APPROVALS_SHA256` pin in `.github/workflows/verify.yml`;
- `.github/workflows/verify.yml`, `.github/CODEOWNERS` and `tools/`.

You cannot add or change a runtime tree yourself. If you believe a pack is wrong, report it (section 12); the fix is a new approved version.

## 7. Checks before you open a PR

```bash
umask 022                     # file modes in the legacy inventory are 644/755
git diff --check
python3 -m unittest discover -s tools -p 'test_*.py'
python3 -B -E -s tools/verify_runtime_tree.py --root . \
  --approved runtimes/templates/APPROVED.json \
  --trusted-approvals runtimes/templates/APPROVALS.json \
  --trusted-approvals-sha256 "$(sed -n 's/^ *TRUSTED_APPROVALS_SHA256: *//p' .github/workflows/verify.yml)" \
  --assets-dir "$(mktemp -d)" --allow-legacy
```

The unit tests need only Python 3.12 and `git`. They use disposable Git repositories and synthetic fixtures; they make no network calls and run no GPU or model code. Paste the output into the PR description. Run them on the head commit you submit; a later push invalidates what you pasted.

CI runs the same checks as the required `verify` job from the protected base revision (`docs/PROMOTION.md`). A pull request that changes the workflow, the tools or the policy files is judged by the rules already on `main`, not by its own changes: its own tool, test, registry and pin changes are not run by its `verify` job (the first CI run of them is the push after the merge). For those pull requests your pasted local run on the head SHA is the only evidence before merge, and reviewers re-run it.

## 8. PR description

`.github/pull_request_template.md` is the template. Fill every section: the one-sentence purpose, what and why, the commands you ran with their output, and the contract checklist.

## 9. Agent protocol (binding)

Agents (AI coding assistants) must:

1. Read this file and `AGENTS.md` in full before any change.
2. Work on a `<type>/<kebab-slug>` branch and open a PR. Never push to `main`, force-push a shared branch, bypass hooks, or write to the GitHub API except to open or update your own PR. Merge a PR only when the maintainer you work for has told you to merge that PR.
3. Report evidence, not ticks: the command, the head SHA it ran on and its output.
4. Answer in the PR description: Can the purpose be stated in one sentence without "and"? Does the diff contain anything the title does not describe? Did the checks run on this head SHA? Did any test or threshold change, and is that in its own TEST PR?
5. Never touch `runtimes/*/v*/`. Never create or edit a registry entry, a template digest or an approval, and never run `tools/promote_runtime.py`, unless the maintainer you work for has explicitly told you to do that specific step (section 9.1). Without that instruction these are maintainer actions.

Stop and ask a maintainer when the intent is unclear, requirements conflict, the rules do not cover the situation, or you would need to weaken a test or gate.

### 9.1 Approving by instruction

The approvers are always the two maintainers, `@am423` and `@vcruz305`. Either may explicitly instruct the agent they work with to carry out an approval step on their behalf: create or edit a registry entry, a template digest or approval, run `tools/promote_runtime.py`, or grant a gate exception. The rules:

1. **The instruction names the step.** It says which step, for which runtime, version or digest. A general instruction ("handle the release"), earlier approval of related work, or silence is not an instruction to approve, and an agent never infers one. An instruction covers that step only, not another action and not a later session.
2. **The approver of record is the maintainer.** `--approved-by`, `approved_by` and `granted_by` carry the maintainer's login who gave the instruction, never an agent. The approval is that maintainer's act and responsibility.
3. **The agent shows the instruction.** The PR description quotes or links it, names the maintainer, and says which step it covered.
4. **Review is unchanged.** The PR still gets the standard review in section 10, and every other rule in this file still applies. In particular an agent never weakens a check, edits a sealed tree by hand or changes the registry pin without the registry change it belongs to.
5. **No instruction, no step.** If the instruction is missing, ambiguous or not from `@am423` or `@vcruz305`, stop and ask.

## 10. Review and merge

- `.github/CODEOWNERS` is `* @am423 @vcruz305`. Both are requested on every PR.
- Standard practice is that the other maintainer reviews the PR before it merges, and that review is expected for a promotion PR that adds a runtime version. It is acceptable for a maintainer to merge their own PR when the other maintainer is not available, once the required check (`verify`) is green and review threads are resolved; GitHub does not enforce a review so that either of the two maintainers can always merge. A PR from a non-maintainer needs a maintainer's approval. Stale approvals are dismissed on push.
- Squash-merge only, so the PR title is the commit subject on `main`. No force-push to `main`; linear history.
- Nothing merges a pull request automatically. The old `merge` job and the `gh pr merge --auto` step are gone.
- `.github/rulesets/` holds ruleset proposals for the maintainers to apply on GitHub. Contributors never call the settings API.

## 11. Licences, attribution, secrets

- This repository is AGPL-3.0-only (`LICENSE`, `NOTICE`); contributions are accepted under the same licence. Only MIT, Apache-2.0 and BSD-3-Clause code may be copied or re-expressed here, with attribution; AGPL, GPL and non-commercial code is ideas only.
- Tools copied from the dev repository stay byte-identical to the dev commit named in the commit message, except the adaptations that message states. Fix them in dev first, then re-copy.
- A runtime release archive contains AGPL-licensed binaries. Anyone who distributes one must also provide the Corresponding Source (`docs/PROMOTION.md`, `engine_source_asset`). Do not add model weights or anything derived from them to this repository; a model keeps the licence of its source.
- No secrets or tokens in commits (API keys, Hugging Face and GitHub tokens, private keys, `.env` files, passwords in URLs).
- Commit with a GitHub noreply address (`<id>+<login>@users.noreply.github.com`). Do not commit with a personal email address.

## 12. Reporting problems

A pack that misbehaves: send the output of `./launch.sh diagnostics` (an allow-list; read it before sharing) to the maintainers. A security problem: follow `SECURITY.md` and do not open a public issue.
