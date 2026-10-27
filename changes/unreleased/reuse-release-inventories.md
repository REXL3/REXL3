## New features and improvements

- Promotion now reuses verified release inventories: a release archive is
  decompressed and re-inventoried at most twice per attempt instead of once per
  verification phase. Every phase still requires a fresh streaming SHA-256 of
  the compressed transport bytes equal to the approval record before any cached
  inventory is honored, and tampered or unlisted members still fail closed.
