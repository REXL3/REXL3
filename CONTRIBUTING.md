# Contributing to REXL3 production

This repository (`REXL3/REXL3`) is the **production** side of REXL3, published under AGPL-3.0-only.
Its content is sealed, data-only runtime trees under `runtimes/`, the policy that decides which trees
are accepted, and the small stdlib-Python tools that verify and promote them. There is no engine source
here. The engine, the runtime template, model profiles and every qualification gate live in the dev
repository, `REXL3/REXL3-dev`; follow that repository's `CONTRIBUTING.md` for engine and profile work.

This file is the single source of truth for humans and agents. Agents must also read `AGENTS.md`;
where they differ, this file wins, except that nothing here relaxes the agent "Never" list in
section 7.

## 1. Golden rules

| # | Rule |
|---|------|
| 1 | Never push to `main`: not directly, not by merge, not by force-push. Every change lands through a pull request. |
| 2 | One purpose per PR. If you cannot state it in one sentence without "and", split the PR. |
| 3 | Every PR title is `TYPE: Past-tense summary` (section 5). |
| 4 | Every PR into `main` needs an approving review from the **other** maintainer, given personally after their own review (section 4). There is no single-maintainer self-merge exception. |
| 5 | **Never hand-edit `runtimes/<family>/<model>/v<N>/`.** A sealed tree changes only by a new promoted version, never in place. |
| 6 | Models enter production only through the dev repository's approval and `tools/promote_runtime.py`. An agent prepares a model branch, registry entry or promotion only on an explicit, quoted maintainer instruction (section 7). |
| 7 | Never weaken a test, a gate or the verifier in the same PR as the change it gates. That goes in its own TEST PR. |
| 8 | Exit code 0 is not evidence. Show which checks ran and what they printed. |
| 9 | No secrets, tokens, personal email addresses, internal project names or quantization-method internals in commits, PR text or docs. |
| 10 | When unsure, stop and ask a maintainer. Do not guess policy. |
| 11 | One logical commit per branch; its subject is the PR title. Base it on the latest `main`. |

## 2. Runtime layout and naming

Runtimes live under `runtimes/<family>/<model>/`:

```
runtimes/
  <family>/
    <model>/
      MODEL.json      schema "rexl3.prod-model.v1" (reviewed metadata; see below)
      LATEST          one line: v<N>
      v<N>/           sealed tree produced by tools/promote_runtime.py (never edited by hand)
  templates/          APPROVED.json (template allow-list), APPROVALS.json (approval registry)
```

Naming:

- Lowercase, regex `^[a-z0-9]+([.-][a-z0-9]+)*$` for both `<family>` and `<model>`.
- `<family>` is the model family as users know it. `<model>` is the model name **without**
  quantization format, bit width, draft/speculation method or other pack details. Those live in
  `MODEL.json` and in the sealed approval.
- The exact architecture is recorded in `MODEL.json` because one family can span architectures in the
  engine: Qwen3.8-27B is `qwen3_5` (`Qwen3_5ForConditionalGeneration`, dev family module `qwen`),
  while Qwen3.8-Flash-Next is `qwen4_exp` (`Qwen4ExpForConditionalGeneration`, dev module
  `qwen_flash_next`).
- **One pack per model per release.** A second pack for the same model needs both maintainers'
  agreement and a descriptive variant suffix agreed by both (never a bit width).
- A derivative model with its own name is a separate model, not a second pack: it gets its own
  `<model>` folder, for example RED-SNOW-5.3-Flash lives at
  `runtimes/glm-5.3/red-snow-5.3-flash/` rather than being a second GLM-5.3-Flash pack.

First-release map (priority order):

| Priority | Model | prod folder | dev runtime id(s) | dev engine module |
|---|---|---|---|---|
| 1 | Qwen3.8-27B | `runtimes/qwen3.8/qwen3.8-27b/` | `qwen38-27b-dflash2-exl3-4.00` | `qwen` (qwen3_5) |
| 2 | Qwen3.8-Flash-Next | `runtimes/qwen3.8/qwen3.8-flash-next/` | none yet | `qwen_flash_next` (qwen4_exp) |
| 3 | DeepSeek-V4.1-Flash-1017 | `runtimes/deepseek-v4.1/deepseek-v4.1-flash-1017/` | `dsv41-flash-1017-sage-1.62` | `deepseek_v41` |
| 4 | GLM-5.3-Flash | `runtimes/glm-5.3/glm-5.3-flash/` | `glm53-flash-2.25bpw-sm121` | `glm` |
| 5 | Nemotron-3.5-Super-VL | `runtimes/nemotron-3.5/nemotron-3.5-super-vl/` | none yet | `nemotron_h_omni` |

`MODEL.json` (schema `rexl3.prod-model.v1`), required keys:

- `schema`, `family`, `model` (must equal the folder names), `display_name` (e.g. "Qwen3.8-27B"),
- `architectures` (list of HF architecture class names), `engine_family` (dev module name),
- `runtime_id` (dev runtime id used for dev release tags `approve-<runtime_id>-<n>`),
- `pack` (object; required key `format`, e.g. "exl3"; other descriptive keys such as `bits`,
  `draft`, `target_repo`, `draft_repo`, and `vision_does_not_fit` (a string recording a proven
  does-not-fit justification when vision is off) are allowed).

Registry (`runtimes/templates/APPROVALS.json`): entries are keyed `<family>/<model>/v<N>`; each entry
keeps `approval_sha256`, `source_commit`, `source_tree` and adds `runtime_id`. The
`TRUSTED_APPROVALS_SHA256` pin in `.github/workflows/verify.yml` changes in the same commit as the
registry. Tools derive dev release tags from `runtime_id` (MODEL.json and the sealed approval must
agree), never from folder names.

## 3. Branches and PR scope

| Branch | Purpose | May touch only |
|---|---|---|
| `model/<model>-approval` | register one dev approval (merges first; verify trusts only the registry on main) | `runtimes/templates/APPROVALS.json`, the pin line in `.github/workflows/verify.yml` |
| `model/<model>` | bring one validated model into prod | `runtimes/<family>/<model>/**`, `docs/packs/<model>.md`, `changes/unreleased/<model>.md` |
| `engine/<topic>` | universal production changes: tools, workflows, policy, docs, templates | anything except `runtimes/*/*/v*/**` and `runtimes/templates/APPROVALS.json` |
| `release/v<X.Y.Z>` | the release PR, opened by a maintainer only | `releases/v<X.Y.Z>/**`, `CHANGELOG.md`, deleting consumed `changes/unreleased/*.md` |

- `<topic>` is `^[a-z0-9]+(-[a-z0-9]+)*$`. `<model>` follows section 2.
- Titles: `TYPE: Past-tense summary`, TYPE one of FIX FEAT BUILD CI TEST BUMP DOCS, at most 90
  characters, no trailing period.
- Squash merge only; the title becomes main's commit subject.
- `model/*` branches are protected from deletion and force-push (they are the record of what a model
  brought in).

## 4. Review and merge

- Every PR into prod main needs an approving review from the OTHER maintainer (code owner; CODEOWNERS
  `* @am423 @vcruz305`), given by that human personally after their own review. No agent approvals in
  prod. The last push must be approved by someone other than the pusher. Required checks green. Squash
  merge.
- No single-maintainer self-merge exception. The only exception is an explicit owner decision recorded
  in `docs/incidents/` and reviewed retroactively (for example `docs/incidents/2026-10-26-unreviewed-release.md`).
- Model PRs (`model/<model>`) additionally need the `local-validation` check: an APPROVED review by a
  maintainer other than the PR author whose body contains a `LOCAL-VALIDATION v1` block for the current
  head.

`LOCAL-VALIDATION v1` block (plain text lines in the review body; keys exact, all required):

```
LOCAL-VALIDATION v1
pr: 123
head: <40-hex PR head sha>
model: <family>/<model>
host: <hostname and GPU>
check_model_branch: PASS
assets: PASS
doctor: PASS
smoke: PASS
gates: <free text naming the gates run on that host>
```

Rules: `head` equals the PR head at evaluation time; `model` equals the PR's model folder; the four
PASS fields are exactly `PASS`; the reviewer is a maintainer and not the PR author.

## 5. PR types, titles and required checks

PR types are exactly seven; the TYPE is the uppercase prefix of the title:

| Type | Meaning | Typical paths here |
|---|---|---|
| FIX | Fixed a bug | `tools/**`, `.github/**` |
| FEAT | Added a feature, or promoted a model version | `tools/**`; a model PR adds `runtimes/<family>/<model>/v<N>/` |
| BUILD | Changed the build or packaging process | `.gitignore`, `tools/requirements*.txt` |
| CI | Changed CI or repository policy | `.github/**`, `tools/**` that the workflow runs, `runtimes/templates/**` |
| TEST | Added or fixed tests or checks | `tools/test_*.py` |
| BUMP | Bumped a pinned version (an action SHA, a Python requirement) | `.github/workflows/**`, `tools/requirements*.txt` |
| DOCS | Documentation only | `**/*.md` outside `runtimes/`, `docs/**` |

Title rule: `TYPE: Summary`, TYPE uppercase and one of the seven, the summary's first word capitalized
and past tense (ends in "ed" or an irregular past form), at most 90 characters, no trailing period, no
"WIP". The title becomes the squash commit subject on `main`.

Required checks on every PR to main (each must ALWAYS report; use job-level conditions that succeed as a
no-op, never workflow-level path or branch filters):

- `verify` (existing tree verifier; keep its trust model).
- `pr-rules`: title (above), branch name, path scope per branch type, `changes/unreleased/` fragment
  present for `model/*` and for `engine/*` PRs that change behavior (tools/workflows), not for DOCS-only.
- `model-completeness`: runs `tools/check_model_branch.py` for `model/<model>` PRs; no-op success otherwise.
- `local-validation`: section 4; no-op success for non-model PRs and `model/*-approval` PRs.

Security model for every workflow: `pull_request_target` (and `pull_request_review` only where needed),
code and policy read from the protected base checkout only, never execute PR-head code, `permissions`
read-only unless a narrowly needed write (e.g. `statuses: write`) is justified, actions pinned by full
commit SHA, PR title/body passed via `env`, not inline expressions.

`governance-audit` is scheduled daily plus manual, read-only. It fails when: a release exists whose tag
is not `v<X.Y.Z>` or has no matching `releases/v<X.Y.Z>/manifest.json` on main; a PR merged to main in
the last 14 days has no approving review by a non-author maintainer and is not listed in an incident
record; the live rulesets differ from `.github/rulesets/*.json`; immutable releases are disabled. It
is run as `python3 tools/governance_audit.py --repo-root PATH --repo OWNER/REPO --token TOKEN`
(`--codeowners` defaults to `<repo-root>/.github/CODEOWNERS`; `REXL3_REPO`/`GH_TOKEN` environment
fallbacks apply) and prints `PASS governance-audit` or one `FAIL <problem>` per problem.

## 6. Checks before you open a PR

```bash
umask 022                     # file modes in the inventory are 644/755
git diff --check
python3 -m unittest discover -s tools -p 'test_*.py'
python3 -B -E -s tools/verify_runtime_tree.py --root . \
  --approved runtimes/templates/APPROVED.json \
  --trusted-approvals runtimes/templates/APPROVALS.json \
  --trusted-approvals-sha256 "$(sed -n 's/^ *TRUSTED_APPROVALS_SHA256: *//p' .github/workflows/verify.yml)" \
  --assets-dir "$(mktemp -d)"
```

The unit tests need only Python 3.12 and `git`. They use disposable Git repositories and synthetic
fixtures; they make no network calls and run no GPU or model code. Paste the output into the PR
description. Run them on the head commit you submit; a later push invalidates what you pasted.

A `model/<model>` PR additionally runs the completeness checker on the head commit (docs/PROMOTION.md)
and pastes its full output.

CI runs the same checks as the required `verify` job from the protected base revision
(`docs/PROMOTION.md`). A pull request that changes the workflow, the tools or the policy files is
judged by the rules already on `main`, not by its own changes: its own tool, test, registry and pin
changes are not run by its `verify` job (the first CI run of them is the push after the merge). For
those pull requests your pasted local run on the head SHA is the only evidence before merge, and
reviewers re-run it.

### Tool command reference

The exact commands and flags below are the implemented interfaces (read from the tools themselves;
see `docs/PROMOTION.md` and `docs/RELEASING.md` for the workflows they belong to). The checker tools
print one line per check (`PASS`/`FAIL name: reason`) and exit 0 when all pass, 1 on any fail, 2 on
a usage error.

**`tools/check_model_branch.py`** — completeness gate for `model/<model>` PRs (check names:
`branch/model`, `scope`, `MODEL.json`, `LATEST`, `tree`, `source commit`, `capability contract`,
`docs and fragment`):

```bash
python3 tools/check_model_branch.py --repo-root PATH --base REF --head REF \
  [--branch REF] [--rc-commit SHA] [--tree-dir PATH] [--base-tree-dir PATH] \
  [--assets-dir PATH] [--local]
```

`--base`/`--head` are refs in `--repo-root`. `--tree-dir` defaults to `--repo-root`; `--base-tree-dir`
defaults to reading base files from git objects; `--assets-dir` defaults to `$REXL3_ASSETS_DIR`.
`--rc-commit` demands the sealed approval's source commit equal it; `--local` additionally downloads
and digest-checks the dev assets via `tools/fetch_dev_assets.py`.

**`tools/check_pr_rules.py`** — title, branch, path scope and fragment rules (check names:
`branch`, `title`, `scope`, `verify pin only` for `model/*-approval`, `fragment`). Requires
`--repo`, `--pr`, `--title`, `--head-ref`, `--token` (or the environment `REXL3_REPO`,
`REXL3_PR_NUMBER`, `REXL3_TITLE`, `REXL3_HEAD_REF`, `GH_TOKEN`):

```bash
python3 tools/check_pr_rules.py --repo OWNER/REPO --pr N --title T --head-ref REF --token TOKEN
```

**`tools/check_local_validation.py`** — evaluates APPROVED maintainer reviews carrying a
`LOCAL-VALIDATION v1` block for the current head (section 4) and sets the commit status
`local-validation` on the head SHA. Requires `--repo`, `--pr`, `--head-sha`, `--codeowners`,
`--token` (or `REXL3_REPO`, `REXL3_PR_NUMBER`, `REXL3_HEAD_SHA`, `REXL3_CODEOWNERS`, `GH_TOKEN`):

```bash
python3 tools/check_local_validation.py --repo OWNER/REPO --pr N --head-sha SHA \
  --codeowners PATH --token TOKEN
```

**`tools/governance_audit.py`** — read-only audit (scheduled daily + manual, not required). Requires
`--repo` and `--token` (or `REXL3_REPO`, `GH_TOKEN`); `--repo-root` defaults to `.`, `--codeowners`
defaults to `<repo-root>/.github/CODEOWNERS`. Prints `PASS governance-audit` or one `FAIL <problem>`
per problem:

```bash
python3 tools/governance_audit.py --repo-root PATH --repo OWNER/REPO --token TOKEN
```

**`tools/promote_runtime.py`** — creates the sealed tree and the `model/<model>` PR
(`docs/PROMOTION.md`). `--version` is an integer `<N>`; `--target-commit` is the exact production base
commit; `--assets-dir` is a durable, verified dev-asset cache; `--approved`,
`--trusted-approvals` and `--trusted-approvals-sha256` are the template allow-list, registry and pin;
`--download` validates and caches the dev release assets; `--dry-run` verifies and prints without
writing; when `MODEL.json` is new, `--display-name`, `--engine-family`, `--pack-format`,
`--architecture` (repeatable) and `--pack KEY=VALUE` (repeatable) supply the metadata:

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

**`tools/assemble_release_notes.py`** — assembles `releases/v<X.Y.Z>/NOTES.md` from
`changes/unreleased/` fragments and prepends the `CHANGELOG.md` entry (`docs/RELEASING.md`).
`--version` is required; `--repo-root` defaults to `.`. Unknown fragment headings, empty fragments and
duplicate slugs are refused (`NOTES_REFUSED: ...` on stderr, exit 2):

```bash
python3 tools/assemble_release_notes.py --repo-root PATH --version vX.Y.Z
```

**`tools/cut_release.py`** — the maintainer release ceremony (dry run by default; `--publish` and
`--verify` are mutually exclusive). Requires `GH_TOKEN`; `--repo` defaults to `REXL3/REXL3`,
`--repo-root` to `.`. Refusals print `CUT_REFUSED: ...` on stderr and exit 2:

```bash
python3 tools/cut_release.py --version vX.Y.Z              # dry run (default)
python3 tools/cut_release.py --version vX.Y.Z --publish    # tag + draft release, verify, publish
python3 tools/cut_release.py --version vX.Y.Z --verify      # confirm published release matches manifest
```

## 7. Agent protocol (binding)

Agents (AI coding assistants) working in this repository must follow this file, `AGENTS.md` and the
branch, review and check rules above.

### Never (no instruction from anyone overrides these)

1. **Never create, edit, publish or delete a release or a tag.** No `gh release`, no releases API, no
   `git tag` push, no `git push --tags`. A release is cut only by a maintainer personally, following
   docs/RELEASING.md.
2. **Never merge a pull request, and never submit a review that approves one.** In production every
   approval comes from a maintainer personally, after their own review. You prepare the PR and report;
   a maintainer merges.
3. **Never change repository settings**: rulesets, branch or tag protection, environments, secrets,
   variables, collaborators, teams, webhooks, deploy keys, Actions permissions or release settings.
4. **Never work with admin credentials.** If the token you were given can administer this repository,
   stop and tell the maintainer you work for.
5. **Never push to `main`, never force-push, and never delete a `model/*` branch.** Never bypass hooks
   or checks.
6. **Never edit a sealed tree** (`runtimes/*/*/v*/`). A sealed tree is data with a hash-pinned
   inventory; a pack changes only through a new sealed version from the dev repository.
7. **Never weaken the verifier, a check, a test or a workflow safety rule**, and never in the same PR
   as the change it guards. Workflows read code and policy from the protected base revision and never
   execute pull-request code; keep it that way.

### Always

1. **Use the right branch type** (section 3): `model/<model>` brings one validated model into
   production, `model/<model>-approval` registers that model's dev approval and merges first,
   `engine/<topic>` is any universal production change: tools, workflows, policy, docs, templates.
   `release/v<X.Y.Z>` is opened by a maintainer only; never open one.
2. **Name runtime folders by family, then model**: `runtimes/<family>/<model>/`, lowercase. Never put a
   bit width, quantization format, draft method or other pack detail in a folder or branch name; those
   belong in `MODEL.json` and the sealed approval.
3. **Prepare a model branch, registry entry or promotion only when the maintainer you work for
   explicitly asked for that model and version.** Quote or link the instruction in the PR. The
   approver of record is that maintainer, never an agent. This is the only way an agent touches a
   registry entry or a promotion; it does not make the agent an approver of the PR.
4. **One purpose per PR.** Title `TYPE: Past-tense summary` (section 5). Fill in the matching PR
   template and add a release-notes fragment under `changes/unreleased/` when section 5 requires one.
5. **For a model PR, run the model-branch completeness check** described in docs/PROMOTION.md and
   paste its full output.
6. **Evidence, not claims.** An exit code of 0 is not a pass. Paste the commands, the head SHA and the
   output.
7. **No secrets, no personal email, no internal project names.** Commit with a GitHub noreply identity.
   Describe SAGE only as: "SAGE is our own mixed-precision quantization method for EXL3. The method is
   not published."
8. **When anything is unclear, or a rule blocks you, stop and ask the maintainer you work for.** Do not
   look for a workaround.

## 8. Licences, attribution, secrets

- This repository is AGPL-3.0-only (`LICENSE`, `NOTICE`); contributions are accepted under the same
  licence. Only MIT, Apache-2.0 and BSD-3-Clause code may be copied or re-expressed here, with
  attribution; AGPL, GPL and non-commercial code is ideas only.
- Tools copied from the dev repository stay byte-identical to the dev commit named in the commit
  message, except the adaptations that message states. Fix them in dev first, then re-copy.
- A runtime release archive contains AGPL-licensed binaries. Anyone who distributes one must also
  provide the Corresponding Source (`docs/RELEASING.md`, `engine_source_asset`). Do not add model
  weights or anything derived from them to this repository; a model keeps the licence of its source.
- No secrets or tokens in commits (API keys, Hugging Face and GitHub tokens, private keys, `.env`
  files, passwords in URLs).
- Commit with a GitHub noreply address (`<id>+<login>@users.noreply.github.com`). Do not commit with a
  personal email address.

## 9. Reporting problems

A pack that misbehaves: send the output of `./launch.sh diagnostics` (an allow-list; read it before
sharing) to the maintainers. A security problem: follow `SECURITY.md` and do not open a public issue.
