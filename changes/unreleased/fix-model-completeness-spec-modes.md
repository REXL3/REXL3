## Bug fixes

- The model-branch completeness check (`model-completeness`) now reads speculative decoding the way
  the engine parses a profile's argv. A profile counts when it declares `--spec auto` or `--spec mtp`
  (the last `--spec` wins), a `--draft` pack, or a `draft` role in its model lock, and nothing turns
  drafting off (`--no-drafts`, a final `--spec none` or `--spec copy`). Before, the check looked only
  for `--draft` or any second pack in the model lock: profiles that run `--spec auto` or `--spec mtp`
  without a draft pack (GLM-5.3-Flash, Nemotron-3.5-Super-VL, Qwen3.8-Flash-Next, Ling) failed, and
  a model whose second pack is an alternate target rather than a draft passed whatever its argv said.
  (PR #51)
