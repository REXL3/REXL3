## New features and improvements

- A sealed approval may qualify a subset of the profiles a runtime packages, as long as it includes the
  default profile. A packaged profile the approval does not name (for example the two-node choice beside
  the single-node one) is not approved and is not judged by the model-branch check. This lets a
  single-node profile ship on its own receipts without requalifying its two-node sibling, as the
  tp1/tp2 independent-runtime-unit rule intends. The sealer and `tools/release_evidence.py` still
  refuse an approval with no profiles, a profile the runtime does not package, or one that omits the
  default profile.
