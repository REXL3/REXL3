## New features and improvements

- A release manifest can now list `engine.source_archives`, so the engine corresponding-source archive that several models were sealed from is published once (`rexl3-engine-source-<X.Y.Z>-<8hex>.tar.gz`) instead of once per model. Each sealed approval still names its own archive digest and size; the release tools map it to the shared asset by that digest, refuse any mismatch, and check the published asset against the manifest. Releases without the key (`v0.1.0`, `v0.1.1`) are unchanged.
