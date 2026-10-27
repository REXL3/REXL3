# Releasing REXL3

One product version for the engine and its recipes. Releases are cut only by maintainers personally —
never by agents, never by automation. Agents never create, edit, publish or delete releases or tags
(CONTRIBUTING.md section 7).

## Versioning

- Semantic Versioning 0.x. The first release is `v0.1.0`.
  - MINOR: new models or architectures, new features, breaking changes (pre-1.0).
  - PATCH: fixes and performance work that keep recipes compatible.
- Prod tags: `v<X.Y.Z>` only. No per-runtime GitHub releases; `prod/<runtime>/v<N>` is retired.
- Dev releases `approve-*` and `engine-v*` are internal provenance, never product releases.

## The release candidate on dev

- Dev uses a release branch `release/v<X.Y.Z>`; candidates are tagged `v<X.Y.Z>-rc.<N>` on dev.
- **All models in a release are sealed from the SAME dev release-candidate commit.**
- The dev workspace version is set to `<X.Y.Z>` at the release candidate; the next dev cycle uses
  `<X.(Y+1).0>-dev`. Earlier 0.1.1/0.1.2 labels were internal builds.

## The release PR

A `release/v<X.Y.Z>` branch is opened by a maintainer only (CONTRIBUTING.md section 3), and contains:

- `releases/v<X.Y.Z>/manifest.json` — schema `rexl3.release-manifest.v1`: `version`, `engine`
  {`dev_commit`, `dev_tree`, `engine_release_tag`, artifact digests}, `models` [{`family`, `model`,
  `version` (v<N>), `runtime_id`, `approval_sha256`}], `notes`.
- `releases/v<X.Y.Z>/NOTES.md` — the assembled release notes.
- The `CHANGELOG.md` entry.
- It deletes the consumed `changes/unreleased/*.md` fragments.

## Notes assembly

- Each change contributes a fragment `changes/unreleased/<slug>.md` (rules in
  `changes/unreleased/README.md`).
- Fragments are assembled by `tools/assemble_release_notes.py` into the template
  `docs/templates/RELEASE_NOTES.md`:

  ```bash
  python3 tools/assemble_release_notes.py --repo-root PATH --version vX.Y.Z
  ```

  (`--repo-root` defaults to `.`.) The template sections are: Highlights; Supported models (table:
  model, architecture, pack, context, speculative decoding, prefix cache, vision, evidence); New
  features and improvements; Bug fixes; Breaking changes and upgrade notes; Known issues; Assets and
  checksums; Provenance. A fragment uses only the headings it needs (Highlights; New features and
  improvements; Bug fixes; Breaking changes and upgrade notes; Known issues); the Supported models,
  Assets and Provenance sections are generated from `releases/v<X.Y.Z>/manifest.json`. The tool writes
  `releases/v<X.Y.Z>/NOTES.md` and prepends the `CHANGELOG.md` entry; unknown headings, empty
  fragments and duplicate slugs are refused (`NOTES_REFUSED: ...` on stderr, exit 2).

## The cut_release ceremony (maintainers only)

No GitHub App. The ceremony is run by a maintainer personally on their own machine with an
authenticated `gh`; `tools/cut_release.py` requires `GH_TOKEN` in the environment (it never edits
rulesets or settings):

1. **Release PR approved by the other maintainer and merged.**
2. `python3 tools/cut_release.py --version vX.Y.Z` — dry run (the default) verifies: release PR
   merged with an approving review by a non-author maintainer, manifest valid and consistent with main,
   notes present, tag absent, immutable releases enabled.
3. A maintainer temporarily allows creation of exactly that tag in the tag ruleset.
4. `python3 tools/cut_release.py --version vX.Y.Z --publish` — creates the tag at the release PR's
   merge commit, a draft release with the notes and assets, verifies uploaded digests, then publishes.
5. The maintainer restores the tag ruleset; `python3 tools/cut_release.py --version vX.Y.Z --verify`
   confirms the lock is restored and the published release matches the manifest.

`--publish` and `--verify` are mutually exclusive; `--repo` defaults to `REXL3/REXL3` and
`--repo-root` to `.`; refusals print `CUT_REFUSED: ...` on stderr and exit 2.

## Immutability

- Releases are immutable. A bad release is superseded by a patch release and marked in its notes and
  `CHANGELOG.md`; deletion needs both maintainers.

## Release checklist

Before the candidate:

- [ ] Every first-release model meets the definition of done in `docs/PROMOTION.md` (final runtime
      entry, capability contract proven, exactness, long context, GB10 receipts on the RC commit,
      `KNOWN_ISSUES`, approval sealed from the RC commit).
- [ ] Dev workspace version set to `<X.Y.Z>`; release branch `release/v<X.Y.Z>` cut; candidate tagged
      `v<X.Y.Z>-rc.<N>`.
- [ ] All models in the release are sealed from the SAME dev release-candidate commit; GB10 receipts
      exist for that commit.

At the release PR:

- [ ] `releases/v<X.Y.Z>/manifest.json` is schema-valid (`rexl3.release-manifest.v1`) and consistent
      with main (engine dev commit/tree/tag/digests; every model entry matches the registry).
- [ ] `releases/v<X.Y.Z>/NOTES.md` assembled from fragments via
      `tools/assemble_release_notes.py`; `CHANGELOG.md` entry added; consumed
      `changes/unreleased/*.md` fragments deleted.
- [ ] Release PR approved by the other maintainer and merged (squash).

At the ceremony:

- [ ] `python3 tools/cut_release.py --version vX.Y.Z` (dry run) passes.
- [ ] Maintainer temporarily allows creation of exactly that tag in the tag ruleset.
- [ ] `python3 tools/cut_release.py --version vX.Y.Z --publish` publishes tag + draft release, verifies
      uploaded digests, publishes.
- [ ] Maintainer restores the tag ruleset; `python3 tools/cut_release.py --version vX.Y.Z --verify` confirms the
      lock is restored and the release matches the manifest.

After the release:

- [ ] Notes and changelog mark any superseded release; `governance-audit` stays green.
