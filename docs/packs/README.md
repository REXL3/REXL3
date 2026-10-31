# Pack sheets

A **pack sheet** is the one-page, tester-facing description of a released runtime: what model it serves,
how big it is, which context window, slots and port it uses, which request switches it honours, and what
behaviour to expect. The [user guide](../USER_GUIDE.md) is the same for every pack; the sheet holds
everything that differs.

There is one sheet per model, named `docs/packs/<model>.md` after the `runtimes/<family>/<model>/`
folder. The
[README](../../README.md)'s released-packs table links to each sheet.

No pack sheet exists yet; the first accompanies the `v0.1.0` product release.

## Rules for a sheet

1. **A sheet describes a sealed version, never a plan.** It is added in the `model/<model>` pull request
   that brings that version in (`tools/check_model_branch.py` requires `docs/packs/<model>.md`), from the
   sealed files in that pull request. Do not describe a model that has no
   `runtimes/<family>/<model>/v<N>/` tree.
2. **Every fact has a source.** Copy values from the release itself, not from memory: the sealed
   `profile.json` (window, slots, port, flags, capabilities, readiness timeout), `model.lock.json` (model
   repository, revision, file count, size, licence), the sealed `approval.json` (source commit, bundle
   hash), `releases/v<X.Y.Z>/manifest.json` (release tag), the release asset names
   (`<runtime-id>-v<N>-approval.json`, `<runtime-id>-v<N>-bundle.tar.gz`) and `KNOWN_ISSUES.md`
   (behaviour, limits, measurements). The sheet says which file each table
   comes from. Where the release does not state something, the sheet says "not stated", it does not guess.
3. **The release wins.** The sealed files and the bundle's own documents are authoritative. If a sheet and
   the release disagree, the release is right and the sheet is a bug to fix with a new DOCS pull request.
4. **No measured number without its context.** A speed or memory figure names the hardware, the release it
   was measured on and where it is recorded. Do not carry a figure from one version to another.
5. **One version history.** A new promoted version updates the sheet's version row and any value that
   changed, in that version's `model/<model>` pull request; the old version stays listed with its
   differences until it is withdrawn (`docs/PROMOTION.md`).
6. **No internal project names, personal email addresses, credentials or quantization-method internals.**
   These rules are in [CONTRIBUTING](../../CONTRIBUTING.md) section 1.

## Adding a sheet

1. Copy [`TEMPLATE.md`](TEMPLATE.md) to `docs/packs/<model>.md`.
2. Fill every section from the sealed files, as rule 2 says; delete none (write "not stated" instead).
3. Add its row to the table above and to the README's released-packs table, and update the README's
   "Run" command variables if the quick start changes.
4. Commit it in the `model/<model>` pull request (CONTRIBUTING section 3). A later correction to a
   sheet is a `DOCS:` pull request on an `engine/<topic>` branch; its description names the release tag
   and the head commit the sheet was checked against.
