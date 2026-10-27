## New features and improvements

- Moved the runtime layout to `runtimes/<family>/<model>/v<N>/` with `MODEL.json`
  (`rexl3.prod-model.v1`); registry keys are `<family>/<model>/v<N>` and carry `runtime_id`. Dev
  release tags derive from `runtime_id`, never from folder names. The old flat layout is refused.
- Added `tools/check_model_branch.py` (model-branch completeness), `tools/check_pr_rules.py`,
  `tools/check_local_validation.py`, `tools/governance_audit.py`, `tools/assemble_release_notes.py`
  and `tools/cut_release.py`, with tests.
- Added the `pr-rules`, `model-completeness`, `local-validation` and `governance-audit` workflows
  and the target ruleset copies (`main-guard`, `protected-tags`, `model-branches`).
