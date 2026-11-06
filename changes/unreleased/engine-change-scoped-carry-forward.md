## New features and improvements

- A release may carry forward a model whose family/kernel scope is untouched by the newer engine
  commits in that release, instead of re-running its full qualification chain. The release ceremony
  stages each carried model's engine binaries from the exact engine release its sealed approval
  names, so its approval verifies against the engine it was sealed with. A model whose
  math/weights/sampling changes, or a new model, never carries forward and must re-seal.
- Gate receipts stay bound to the model pack and profile as well as the engine build. Only a gate a
  receipt explicitly lists as carried may reuse raw evidence from an earlier engine build, and only
  when that build is pinned exactly (source engine commit/tree/binary/lock) and tied to a
  registry-pinned prior approval; the exactness and prefix-cache gates always re-run on the new
  build, and at least one gate must stay fresh on it. `tools/release_evidence.py` refuses any
  cross-engine-build gate evidence outside such a declaration.
