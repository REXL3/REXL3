## New features and improvements

- A release may now carry forward a model whose family/kernel scope is untouched by the newer engine
  commits in that release, instead of re-running its full qualification chain (decision D37/D40). A
  manifest model entry may set `carried_forward: true` together with the `qualified_engine_release_tag`
  it was sealed against, its `qualification_tier` (T0 carry-forward, T1 golden-smoke, T2 gate-scoped
  re-run), and the `scope_diff_base` commit. The release ceremony stages each carried model's engine
  binaries from the exact engine release its sealed approval names, so its approval verifies against
  the engine it was sealed with. T3 (model math/weights/sampling changes, or a new model) never
  carries forward and must re-seal.
- Gate receipts are bound to the model pack (`model_lock_sha256`) and profile, not to the engine binary
  build (decision D40): a carried-forward approval's gate receipts (Q200 included) remain valid across
  engine builds, and only the gates a T2 re-run actually re-executes (e.g. #229 DeepSeek
  exactness/prefix_cache) are refreshed on the new binary. `tools/release_evidence.py` accepts
  cross-engine-build gate evidence when the pack/profile-bound identity matches.

## Breaking changes and upgrade notes

- `tools/cut_release.py` refuses a carried-forward model whose `qualified_engine_release_tag` equals
  the release's primary engine tag (self carry-forward is not a carry-forward), and refuses a carried
  approval that names another engine release than its `qualified_engine_release_tag` or whose tag does
  not name its own approval commit.
