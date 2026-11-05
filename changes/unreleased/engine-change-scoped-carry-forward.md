## New features and improvements

- A release may now carry forward a model whose family/kernel scope is untouched by the newer engine
  commits in that release, instead of re-running its full qualification chain (decision D37). A
  manifest model entry may set `carried_forward: true` together with the `qualified_engine_release_tag`
  it was sealed against, its `qualification_tier` (T0 carry-forward, T1 golden-smoke, T2 bit-exact
  kernel), and the `scope_diff_base` commit. The release ceremony stages each carried model's engine
  binaries from the exact engine release its sealed approval names, so its approval verifies against
  the engine it was sealed with. T3 (model math/weights/sampling changes, or a new model) never
  carries forward and must re-seal.

## Breaking changes and upgrade notes

- `tools/cut_release.py` refuses a carried-forward model whose `qualified_engine_release_tag` equals
  the release's primary engine tag (self carry-forward is not a carry-forward), and refuses a carried
  approval that names another engine release than its `qualified_engine_release_tag` or whose tag does
  not name its own approval commit.
