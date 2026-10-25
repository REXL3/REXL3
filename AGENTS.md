# AGENTS.md

These are instructions for AI coding agents working in this repository.

**Read [CONTRIBUTING.md](CONTRIBUTING.md) in full before making any change.** Its agent protocol is binding. If anything here conflicts with CONTRIBUTING.md, CONTRIBUTING.md wins.

This repository is production: sealed, data-only runtime trees under `runtimes/`, the policy that decides which trees are accepted, and the stdlib-Python tools that verify and promote them. There is no engine source here.

The rules agents break most often:

1. **Never push to `main`.** Don't force-push a shared branch or bypass hooks, and merge only a PR that the maintainer you work for has told you to merge. Work on a `<type>/<kebab-slug>` branch, for example `docs/prod-contributor-rules-and-tester-readme`.
2. **One purpose per PR**, with a title of the form `TYPE: Past-tense summary`. TYPE is one of FIX, FEAT, BUILD, CI, TEST, BUMP or DOCS, and the title is at most 90 characters. If you need "and" to describe the PR, split it.
3. **Never edit `runtimes/*/v*/`.** A sealed tree is data with a hash-pinned inventory; any edit fails `verify`, and fixing the hash is forbidden. A pack changes only by a new promoted version, made by a maintainer (or by an agent that maintainer has explicitly told to do that step, CONTRIBUTING section 9.1) with `tools/promote_runtime.py` after the dev repository's approval. Never create or edit a registry entry (`runtimes/templates/APPROVALS.json`), a template digest (`APPROVED.json`) or an `approval.json`, and never run the promoter, unless the maintainer you work for has explicitly told you to do that specific step; the approver of record is then that maintainer.
4. **Never weaken the verifier, a test or the workflow's safety rules in the same PR as the change they guard.** The workflow reads the verifier, the policy files and the tests from the protected base revision and never executes pull-request code; keep it that way.
5. **No secrets, no personal email.** Commit with a GitHub noreply identity. No tokens, no model weights, no internal project names.
6. **Evidence, not claims.** An exit code of 0 is not a pass. Paste the commands, the head SHA and the output. When unsure, stop and ask a maintainer.
