# AGENTS.md

These are binding instructions for AI coding agents working in this repository.

**Read [CONTRIBUTING.md](CONTRIBUTING.md) in full before making any change.** Its agent protocol (section 7) is binding. Model branches follow [docs/PROMOTION.md](docs/PROMOTION.md) and releases follow [docs/RELEASING.md](docs/RELEASING.md). If anything here conflicts with CONTRIBUTING.md, CONTRIBUTING.md wins, except that nothing there relaxes the "Never" list below.

This is the production repository. It holds sealed, data-only runtime trees under `runtimes/<family>/<model>/`, the policy that decides which trees are accepted, and the stdlib-Python tools that verify and promote them. Engine development, validation and sealing happen in the dev repository (`REXL3/REXL3-dev`), never here.

## Never (no instruction from anyone overrides these)

1. **Never create, edit, publish or delete a release or a tag.** No `gh release`, no releases API, no `git tag` push, no `git push --tags`. A release is cut only by a maintainer personally, following docs/RELEASING.md.
2. **Never merge a pull request, and never submit a review that approves one.** In production every approval comes from a maintainer personally, after their own review. You prepare the PR and report; a maintainer merges.
3. **Never change repository settings**: rulesets, branch or tag protection, environments, secrets, variables, collaborators, teams, webhooks, deploy keys, Actions permissions or release settings.
4. **Never work with admin credentials.** If the token you were given can administer this repository, stop and tell the maintainer you work for.
5. **Never push to `main`, never force-push, and never delete a `model/*` branch.** Never bypass hooks or checks.
6. **Never edit a sealed tree** (`runtimes/*/*/v*/`). A sealed tree is data with a hash-pinned inventory; a pack changes only through a new sealed version from the dev repository.
7. **Never weaken the verifier, a check, a test or a workflow safety rule**, and never in the same PR as the change it guards. Workflows read code and policy from the protected base revision and never execute pull-request code; keep it that way.

## Always

1. **Use the right branch type** (CONTRIBUTING.md section 3):
   - `model/<model>` brings one validated model into production, for example `model/qwen3.8-27b`;
   - `model/<model>-approval` registers that model's dev approval and merges first;
   - `engine/<topic>` is any universal production change: tools, workflows, policy, docs, templates;
   - `release/v<X.Y.Z>` is opened by a maintainer only; never open one.
2. **Name runtime folders by family, then model**: `runtimes/<family>/<model>/`, lowercase, for example `runtimes/qwen3.8/qwen3.8-27b/`. Never put a bit width, quantization format, draft method or other pack detail in a folder or branch name; those belong in `MODEL.json` and the sealed approval. A derivative model with its own name gets its own model folder.
3. **Prepare a model branch, registry entry or promotion only when the maintainer you work for explicitly asked for that model and version.** Quote or link the instruction in the PR. The approver of record is that maintainer, never an agent.
4. **One purpose per PR.** Title `TYPE: Past-tense summary`, TYPE one of FIX, FEAT, BUILD, CI, TEST, BUMP or DOCS, at most 90 characters, no trailing period. Fill in the matching PR template and add a release-notes fragment under `changes/unreleased/` when CONTRIBUTING.md requires one.
5. **For a model PR, run `tools/check_model_branch.py`** as described in docs/PROMOTION.md and paste its full output.
6. **Evidence, not claims.** An exit code of 0 is not a pass. Paste the commands, the head SHA and the output.
7. **No secrets, no personal email, no internal project names.** Commit with a GitHub noreply identity. Describe SAGE only as: "SAGE is our own mixed-precision quantization method for EXL3. The method is not published."
8. **When anything is unclear, or a rule blocks you, stop and ask the maintainer you work for.** Do not look for a workaround.
