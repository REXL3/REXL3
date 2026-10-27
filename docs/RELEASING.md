# Releases (maintainer ceremony)

One product version for the engine and its recipes. SemVer 0.x. First release: `v0.1.0`. MINOR for
new models or architectures, new features, breaking changes (pre-1.0); PATCH for fixes and
performance work that keep recipes compatible. Prod tags are `v<X.Y.Z>` only. Dev release branches
and `approve-*` / `engine-v*` releases are internal provenance, never product releases.

## Cutting a release

Only a maintainer runs this ceremony, personally. `tools/cut_release.py` never edits rulesets or
settings and requires `GH_TOKEN` in the environment.

1. **Release PR** on branch `release/v<X.Y.Z>`: `releases/v<X.Y.Z>/manifest.json` (schema
   `rexl3.release-manifest.v1`), `releases/v<X.Y.Z>/NOTES.md`, the `CHANGELOG.md` entry, and
   deletion of the consumed `changes/unreleased/*.md` fragments. Assemble notes with
   `python3 tools/assemble_release_notes.py --version vX.Y.Z` from the fragments into
   `docs/templates/RELEASE_NOTES.md` (Highlights; Supported models; New features and improvements;
   Bug fixes; Breaking changes and upgrade notes; Known issues; Assets and checksums; Provenance).
   All models in a release are sealed from the same dev release-candidate commit.
2. The other maintainer approves and merges the release PR. Then verify the dry run:
   `python3 tools/cut_release.py --version vX.Y.Z`. It refuses unless the release PR merged with an
   approving review by a non-author maintainer, the manifest is valid and consistent with main,
   notes and the CHANGELOG entry exist, the tag is absent and immutable releases are enabled.
3. Temporarily allow creation of exactly that tag in the tag ruleset (maintainer, on GitHub).
4. `python3 tools/cut_release.py --version vX.Y.Z --publish`: creates the tag at the release PR's
   merge commit, a draft release with the notes and assets, verifies uploaded digests, then
   publishes.
5. Restore the tag ruleset. `python3 tools/cut_release.py --version vX.Y.Z --verify` confirms the
   published release matches the manifest.

Releases are immutable. A bad release is superseded by a patch release and marked in its notes and
`CHANGELOG.md`; deletion needs both maintainers. Agents never create, edit, publish or delete
releases or tags and never edit rulesets or settings.
