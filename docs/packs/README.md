# Pack sheets

A **pack sheet** is the one-page, tester-facing description of a released runtime: what model it serves,
how big it is, which context window, slots and port it uses, which request switches it honours, and what
behaviour to expect. The [user guide](../USER_GUIDE.md) is the same for every pack; the sheet holds
everything that differs.

There is one sheet per runtime id, named `docs/packs/<runtime-id>.md`. The
[README](../../README.md)'s released-packs table links to each sheet.

| Sheet | Runtime | Latest version |
| --- | --- | --- |

## Rules for a sheet

1. **A sheet describes a released version, never a plan.** Add it in a DOCS pull request after the
   promotion pull request has merged and the release assets are published. Do not describe a runtime that
   has no `runtimes/<id>/v<N>/` tree.
2. **Every fact has a source.** Copy values from the release itself, not from memory: the sealed
   `profile.json` (window, slots, port, flags, capabilities, readiness timeout), `model.lock.json` (model
   repository, revision, file count, size, licence), `approval.json` (release tag, archive name, source
   commit) and `KNOWN_ISSUES.md` (behaviour, limits, measurements). The sheet says which file each table
   comes from. Where the release does not state something, the sheet says "not stated", it does not guess.
3. **The release wins.** The sealed files and the bundle's own documents are authoritative. If a sheet and
   the release disagree, the release is right and the sheet is a bug to fix with a new DOCS pull request.
4. **No measured number without its context.** A speed or memory figure names the hardware, the release it
   was measured on and where it is recorded. Do not carry a figure from one version to another.
5. **One version history.** A new promoted version updates the sheet's version row and any value that
   changed, in the DOCS pull request that follows the promotion; the old version stays listed with its
   differences until it is withdrawn (`docs/PROMOTION.md`).
6. **No internal project names, personal email addresses, credentials or quantization-method internals.**
   These rules are in [CONTRIBUTING](../../CONTRIBUTING.md) section 1.

## Adding a sheet

1. Copy [`TEMPLATE.md`](TEMPLATE.md) to `docs/packs/<runtime-id>.md`.
2. Fill every section from the sealed files, as rule 2 says; delete none (write "not stated" instead).
3. Add its row to the table above and to the README's released-packs table, and update the README's
   "Run" command variables if the quick start changes.
4. Open a `DOCS:` pull request (title and branch rules are in CONTRIBUTING sections 3 and 4). The
   description names the release tag and the head commit the sheet was checked against.
