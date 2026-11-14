## New features and improvements

- The production `verify` and `model-completeness` checks now accept a new approval anchor only from a `release/v<X.Y.Z>` branch of this repository (never a fork) whose release manifest lists that anchor with the same approval hash, and they refuse anything else before downloading development assets. A green `verify` proves that a tree equals a published development seal; the maintainers' review of the release pull request is what authorizes a new anchor. Pull requests that add no anchor are unaffected.
