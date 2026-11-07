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
- No model ever gets its own release or release candidate.

## Grouped releases

A release is cut from an explicitly agreed group of models listed in the release manifest
(`releases/v<X.Y.Z>/manifest.json`). v0.1.0's current manifest holds the five already-sealed
models: Qwen3.8-27B, Qwen3.8-Flash-Next, GLM-5.3-Flash, Nemotron-3.5-Super-VL and
DeepSeek-V4.1-Flash-1017.

- Each model enters prod through its own `model/<model>` branch and PR, which holds all the commits
  that make that model work. Each model PR must be validated (LOCAL-VALIDATION v1) and approved by
  the maintainer who did NOT submit it (CONTRIBUTING.md section 4).
- The release is cut ONLY after every in-scope model PR has been validated and merged.
- No model gets its own release or release candidate.
- Only AFTER v0.1.0 is released may a later release contain a single newly validated model as a new
  version; it is never phrased as a release candidate for one model.
- Change-scoped qualification tiers:
  - **T0** (untouched scope): the model's family/kernel scope is untouched by the newer engine
    commits, so its prior sealed approval travels as-is in the release manifest
    (`carried_forward: true`, `qualification_tier: "T0"`).
  - **T1 / T2** (shared-glue smoke / gate-scoped re-run): the required gates re-run on the new
    engine binary, unaffected gate receipts carry forward (`carried_gates`), and a NEW approval is
    sealed on the new engine build.
  - **T3** (new model, or pack weight/sampling/chat-template change): the full nine-gate chain plus
    `Q200` runs again; nothing carries forward.
- As the first release, v0.1.0 has no earlier engine release to carry from. The four originally
  sealed v0.1.0 models (Qwen3.8-27B, Qwen3.8-Flash-Next, GLM-5.3-Flash and
  Nemotron-3.5-Super-VL) were sealed on the `799fa750` freeze engine; DeepSeek-V4.1-Flash-1017
  seals fresh on the release engine `E = afc9b5df`. T0/T1/T2 change-scoped carry-forward applies
  WHENEVER a release manifest carries forward an approval sealed on an earlier freeze commit
  (`799fa750`) onto the release engine (`E = afc9b5df`), including within v0.1.0 itself. GLM's
  v0.1.0 approval is its
  already-sealed combined approval 1
  (`glm53-flash-2.25bpw-sm121`), covering both `glm53-flash-single` and
  `glm53-flash-tp2-1m`; the `<model>` (`tp1`) / `<model>-tp2` (`tp2`) split applies to models
  added after v0.1.0 and to subsequent GLM versions.
- Runtime units: single-node (`tp1`) and two-node (`tp2`) profiles are independent runtime units —
  `<model>` for `tp1`, `<model>-tp2` for `tp2` — packaged, sealed and promoted separately, so a
  `tp1` release never waits on its `tp2` sibling. Each unit seals its own qualification receipts on
  its own runtime id and topology; a receipt is never reused across runtime ids or topologies.
  While a source tree still packages both profiles in one `profile.json`, the sealed approval names the
  profiles it qualified (a non-empty subset of the packaged profiles that includes the default profile);
  a packaged profile without a receipt in the approval is not approved, is not judged by the model-branch
  check, and is not part of the release.

## The group freeze on dev

- Dev uses a release branch `release/v<X.Y.Z>`; there are no release-candidate tags. The only
  tags are the final `v<X.Y.Z>` tags (the prod release through the maintainer ceremony, and the
  matching dev source tag), created by maintainers after all in-scope model PRs are validated by
  the non-submitting maintainer and merged.
- The dev freeze commit exists only so the whole group is validated on one engine build; it is never
  a release and is never per model. Every model is either sealed on the freeze commit, or carries
  forward a previously sealed approval whose newer engine commits do not touch its scope (T0), or
  re-seals on the new engine build with carried gate receipts (T1/T2).
- The dev workspace version is set to `<X.Y.Z>` at the freeze; the next dev cycle uses
  `<X.(Y+1).0>-dev`. Earlier 0.1.1/0.1.2 labels were internal builds.

## The release PR

A `release/v<X.Y.Z>` branch is opened by a maintainer only (CONTRIBUTING.md section 3), and contains:

- `releases/v<X.Y.Z>/manifest.json` — schema `rexl3.release-manifest.v1`: `version`, `engine`
  {`dev_commit`, `dev_tree`, `engine_release_tag`, artifact digests}, `models` [{`family`, `model`,
  `version` (v<N>), `runtime_id`, `approval_sha256`, and for a carried-forward model
  `carried_forward`, `qualified_engine_release_tag`, `qualification_tier`, `scope_diff_base`}],
  `notes`. The release PR attaches the release-scope classification (`tools/release_scope.py
  classify` output) for each model, naming its tier and its scope base (the exact commit the tier
  was assessed against).
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
  `releases/v<X.Y.Z>/NOTES.md` and prepends the `CHANGELOG.md` entry. `changes/unreleased/README.md`
  holds the fragment rules and is skipped; any other file not named `<slug>.md` (lowercase kebab-case),
  unknown headings, empty fragments and duplicate slugs are refused (`NOTES_REFUSED: ...` on stderr,
  exit 2).

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

### Agents and the ceremony

An agent that prepares a release runs only the read-only steps: the dry run (step 2) and, after the maintainer has
published, `--verify`. For each write step (3, 4 and 5) it posts a decision packet (AGENTS.md, Always 9) with the exact
command, the tag, the release PR merge commit printed in the `DRY_RUN_OK` line, the checkout whose
`releases/<version>/NOTES.md` becomes the release body, and what is restored afterwards. `cut_release.py` tags the
release PR's merge commit but reads `NOTES.md` from the checkout it runs in, so the packet names that checkout and the
two must not be allowed to drift apart unnoticed. The maintainer runs exactly what the packet states; a different
command, tag or checkout needs a new packet.

`--publish` and `--verify` are mutually exclusive; `--repo` defaults to `REXL3/REXL3` and
`--repo-root` to `.`; refusals print `CUT_REFUSED: ...` on stderr and exit 2.

## Immutability

- Releases are immutable. A bad release is superseded by a patch release and marked in its notes and
  `CHANGELOG.md`; deletion needs both maintainers.

## Release checklist

Before the release:

- [ ] Every first-release model meets the definition of done in `docs/PROMOTION.md` (final runtime
      entry, capability contract proven, exactness, long context, GB10 receipts on the build its
      approval was sealed from, `KNOWN_ISSUES`, and an approval sealed from that build — or a
      registry-pinned prior approval carried forward under T0).
- [ ] Dev workspace version set to `<X.Y.Z>`; release branch `release/v<X.Y.Z>` cut at the merged
      bump (no release-candidate tags).
- [ ] Every model is sealed on the dev freeze commit, or carries forward a previously sealed approval
      whose newer engine commits do not touch its scope (T0), or re-seals on the new engine build
      with carried gate receipts (T1/T2); GB10 receipts exist for every sealed build. The freeze
      exists only so the whole group is validated on one engine build; it is never a release and is
      never per model.

At the release PR:

- [ ] `releases/v<X.Y.Z>/manifest.json` is schema-valid (`rexl3.release-manifest.v1`) and consistent
      with main (engine dev commit/tree/tag/digests; every model entry matches the registry).
- [ ] The manifest lists the agreed release group (all in-scope models; no model outside the group).
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
