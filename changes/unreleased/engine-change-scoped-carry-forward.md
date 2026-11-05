## New features and improvements

- A release may carry forward a model whose family/kernel scope is untouched by the newer engine
  commits in that release, instead of re-running its full qualification chain. A manifest model entry
  sets `carried_forward: true` together with the `qualified_engine_release_tag` it was sealed against,
  its `qualification_tier` (T0 untouched-scope carry-forward; T1 shared-glue smoke; T2 gate-scoped
  re-run), and the `scope_diff_base` commit. The release ceremony stages each carried model's engine
  binaries from the exact engine release its sealed approval names, so its approval verifies against
  the engine it was sealed with. A model whose math/weights/sampling changes, or a new model, never
  carries forward and must re-seal.
- Gate receipts stay bound to the model pack (`model_lock_sha256`) and profile as well as the engine
  build. Only a gate a receipt explicitly lists as carried may reuse raw evidence from an earlier
  engine build, and only when that build is named exactly (source engine commit/tree/binary) and at
  least one gate is re-run on the new build. `tools/release_evidence.py` refuses any cross-engine-build
  gate evidence outside such a declaration.
