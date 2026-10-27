# Incident: unreviewed release of a Qwen3.8-27B runtime

Recorded 2026-10-26 (PDT) under the owner-authorized emergency restructure. This record lists the PRs
merged without the other maintainer's review, for am423's retroactive review (CONTRIBUTING.md section
4; the emergency exception).

## What happened

On 2026-10-26, an unattended agent acting for am423 self-merged a series of unreviewed PRs
(#12–#28) into prod main and, immediately after merging PR #24, created and published a per-runtime
GitHub release `qwen38-27b-dflash2-exl3-4.00 v1` (tag `prod/qwen38-27b-dflash2-exl3-4.00/v1`) via
the releases API. The release carried five assets and became "latest". The production docs in force at
the time allowed a single-maintainer self-merge and per-runtime publishing, so the flow was authorized
by the then-current rules.

## Timeline (2026-10-26 PDT / 2026-10-27 UTC)

All events are actor `am423` unless noted.

| Time (PDT) | Time (UTC) | Event |
|---|---|---|
| 18:46 | 01:46:54 | PR #23 `ci/register-the-qwen3-8-27b-v1-approval` opened; verify green 18:47. |
| 18:48 | 01:48:03 | PR #23 merged (squash); registry gains `qwen38-27b-dflash2-exl3-4.00/v1`; verify.yml pin updated. |
| 18:50 | 01:50:47 | PR #24 `promote/qwen38-27b-dflash2-exl3-4.00-v1` opened (head `b7a0a90`, authored by `r0b0tlab`). |
| 18:52 | 01:52:53 | Verify run 50 on PR #24: success. |
| 18:53 | 01:53:15 | PR #24 merged (squash) → merge commit `b119ffe`. **Same second**, release id 424954752 created via API with `target_commitish=b119ffe` (the release auto-created the lightweight tag). |
| 18:53 | 01:53:18 | `push` event to main (`b119ffe`); verify run 51 success 18:54. |
| 18:54 | 01:54:11 | Release published (assets attached 18:53–18:54). |
| 20:27 | 03:27:27 | PR #25 (LATEST → `v1`) merged. |
| 20:27 | 03:27:52 | PR #26 (pack sheet) merged. |
| 21:14 | 04:14:37 | PR #27 (promote feature: bundle referencing engine release) merged. |
| 21:19 | 04:19:42 | PR #28 (registry v2 entry) merged. |
| 21:21 | 04:21:20 | `main-guard` ruleset updated to require 1 approval (+ extra approval for unattributed changes) — after the last unreviewed merge. |

## Root cause

- The production docs allowed a single-maintainer self-merge and per-runtime publishing, so each step
  followed a documented path.
- `main-guard` required **0 approvals** until 21:21 PDT. The merges #12–#28 happened before that
  change, so they were permitted by the rule in force at the time; they were not an admin bypass of
  the review rule.
- Tag/release creation was unrestricted: the `protected-tags` ruleset blocked update/deletion of
  existing `prod/*/*` tags but not creation of new ones, and anyone with write could create a release.
- The agent account (`am423`) holds repository admin, which remains the control gap. `main-guard`
  has **no bypass actors**, so the rulesets DO apply to admins — but repository admin can edit or
  disable rulesets and change repository settings, and no ruleset can prevent that. Agents must
  therefore run with non-admin identities (see Open follow-ups).

## Tonight's actions (2026-10-26, owner-authorized emergency restructure)

- **Backup**: before deletion, a maintainer archived the release metadata and asset bytes with digests
  (forensics §7.1 procedure).
- **Deleted the release and both tags**: release 424954752 (tag
  `prod/qwen38-27b-dflash2-exl3-4.00/v1` @ `b119ffe`) and the leftover legacy tag
  `prod/dsv41-flash-1017-sage-1.62/v1` @ `dfb5687` were removed after the tag ruleset was briefly
  suspended and restored.
- **Locked tag creation**: the `protected-tags` ruleset now restricts creation as well as update and
  deletion of `prod/*`-shaped tags, with no bypass actors.
- **Enabled immutable releases** (applies to future releases).
- **Added the `model-branches` ruleset**: `refs/heads/model/*` protected from deletion and
  non-fast-forward, no bypass actors.
- **Parked PR #30** (the v2 promotion); its content is superseded by the model-branch process.
- **Moved the unreviewed Qwen content off main** into `model/qwen3.8-27b` so it can be reviewed and
  re-promoted through the new flow.
- **Opened the restructure PRs** (rules, tooling, README, and this docs PR; listed below).

## Emergency-merged PRs

These PRs were merged under the emergency exception without am423's review. The merge coordinator
maintains the table; rows for PRs that are not yet merged are marked "filled at merge time".

| PR | Title | Head SHA | Merged UTC |
|---|---|---|---|
| #33 | FIX: Removed the unreviewed Qwen3.8-27B release from production | `677a092e` | 2026-10-27 05:52:22 |
| #34 | FIX: Removed the unreviewed Qwen3.8-27B approvals from the registry | `106c9680` | 2026-10-27 05:50:56 |
| #35 | FEAT: Restructured production runtime layout and added release and governance checks | `779e86cf` | 2026-10-27 06:42:04 |
| #36 | TEST: Pinned the promotion test git identity to the synthetic user | `804d793b` | 2026-10-27 06:49:38 |
| #37 | FIX: Made the governance audit tolerate workflow-token limits and cover the retro range | `a949ad5a` | 2026-10-27 07:00:23 |
| #39 | FIX: Fixed the incident range parsing for single PR numbers | `656f7f6b` | 2026-10-27 07:04:45 |
| #40 | FIX: Equated null-echoed ruleset lists and recorded the audit fix PRs | `25d4acfa` | 2026-10-27 07:09:17 |
| #32 | DOCS: Rewrote the production README around engine features and recipes | `51f99840` | 2026-10-27 06:51:53 |
| #31 | DOCS: Replaced the production rules with the model-branch release process | this PR | this PR |

Retroactive review list: every pre-restructure unreviewed merge (PRs #1-#28) plus every PR in the
table above. The review is a maintainer approval of the change itself; where a change has since been
superseded by the restructure, a review of its replacement suffices.

## Open follow-ups

- **Non-admin agent identities for both maintainers**: replace admin-scoped agent access with
  fine-grained, non-admin identities so rulesets are enforceable against agent actions.
- **AGENTS.md and CLAUDE.md awaiting owner consent**: the intended replacement texts are staged; the
  orchestrator writes them once the owner consents. CONTRIBUTING.md already carries the agent rules.
- **Secrets for am423 to review**: `REXL3_PROMOTE_PAT` (legacy, unused) and `XAI_OAUTH_STATE` —
  decide whether to rotate, restrict or remove.
- **Choose one GLM-5.3-Flash pack**: dev has two (`glm53-flash-2.25bpw-sm121`,
  `redsnow-5.3-flash-sage-2.49`); the first release carries one.
- **Version alignment at the RC**: set the dev workspace version to `v0.1.0` at the release candidate
  (earlier 0.1.1/0.1.2 labels were internal builds).
