# Incident: two PRs merged without a non-author review during the v0.1.0 public launch

Recorded 2026-11-07 (US Central) on the owner's explicit decision, for retroactive review under
CONTRIBUTING.md (no single-maintainer self-merge exception except an owner decision recorded here).

The owner accepted both merges on 2026-11-07.

## PRs covered

| PR | Title | Author | Merged by | Merged (Central) | Approving review |
|---|---|---|---|---|---|
| #77 | FEAT: Added v0.1.0 engine source tree, tester quickstart, and issue templates | vcruz305 | vcruz305 | 2026-11-06 22:55 | none |
| #81 | DOCS: Told testers that issues are public and that they report through issues only | am423 | vcruz305 | 2026-11-07 00:12 | none |

## What happened

Both PRs were part of making the repository public for invited testers on the night of
2026-11-06/07. A maintainer merged them without an approving review; the `main-guard` ruleset was
changed for the merges and restored afterwards. The live rulesets match the committed policy again.
The daily `governance-audit` then failed on both PRs (runs on 2026-11-07), as designed.

## Why it was accepted

The owner accepted the merges so the public launch could ship on schedule.
Neither PR changes engine code, a runtime, a seal or a release: #77 adds the published v0.1.0
engine source and tester docs, #81 is a docs change.

## Follow-up

- Retroactive review of #77 and #81 by am423 (this record lists them; the audit passes once it is on main).
- No change to the policy: every PR into `main` still needs an approving review from the other maintainer.
