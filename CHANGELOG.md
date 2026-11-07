# Changelog

## [v0.1.0] - 2026-11-06

- DeepSeek-V4.1-Flash-1017 (EXL3 1.55 bits per weight) on one GB10: 262,144-token window, one slot, speculative decoding and prefix cache on, image input off (`dsv41-single-262k`, port 8940). The two-node profile is not part of this release. Added by #75.


- GLM-5.3-Flash (EXL3 2.25 bits per weight) with speculation from the pack's NextN layer, prefix cache and image input on, in two profiles of one recipe: `glm53-flash-single` on one GB10 (262,144-token window, one slot) and `glm53-flash-tp2-1m` on two GB10 nodes with tensor parallelism (1,048,576-token window), both on port 8950. Added by #69.

- Nemotron-3.5-Super-VL (EXL3 3.85 bits per weight) on one GB10: 1,048,576-token window, 16 slots, speculative decoding with the model's MTP head, prefix cache and image input on (`nemotron35-super-vl-single-1m`, port 9100). Added by #71.

- Qwen3.8-27B (EXL3 4.00 bits per weight with a DFlash2 draft) on one GB10: 262,144-token window, six slots, speculative decoding with the DFlash2 draft, prefix cache and image input on, thinking at `low` effort by default (`qwen38-single-262k`, port 8889). Added by #66.

- Qwen3.8-Flash-Next (EXL3 3.05 bits per weight) on one GB10: 262,144-token window, two slots, speculative decoding with the model's MTP head (`--spec mtp`) and prefix cache on (`qwen38-flash-next-single-262k`, port 8899). Added by #73.


- A sealed approval may qualify a subset of the profiles a runtime packages, as long as it includes the
  default profile. A packaged profile the approval does not name (for example the two-node choice beside
  the single-node one) is not approved and is not judged by the model-branch check. This lets a
  single-node profile ship on its own receipts without requalifying its two-node sibling, as the
  tp1/tp2 independent-runtime-unit rule intends. The sealer and `tools/release_evidence.py` still
  refuse an approval with no profiles, a profile the runtime does not package, or one that omits the
  default profile.

- A release may carry forward a model whose family/kernel scope is untouched by the newer engine
  commits in that release, instead of re-running its full qualification chain. The release ceremony
  stages each carried model's engine binaries from the exact engine release its sealed approval
  names, so its approval verifies against the engine it was sealed with. A model whose
  math/weights/sampling changes, or a new model, never carries forward and must re-seal.
- Gate receipts stay bound to the model pack and profile as well as the engine build. Only a gate a
  receipt explicitly lists as carried may reuse raw evidence from an earlier engine build, and only
  when that build is pinned exactly (source engine commit/tree/binary and the source receipt bytes,
  canonical dev #238 shape) and tied to a registry-pinned prior approval of the same model; the
  release lock stays strict on the shipped build (D42 option B). The exactness and prefix-cache
  gates always re-run on the new build, and at least one gate must stay fresh on it.
  `tools/release_evidence.py` refuses any cross-engine-build gate evidence outside such a declaration.

- Production accepts an owner vision exception in place of a proven does-not-fit justification for a
  model that ships with vision off. The exception is a record in each vision-off profile's sealed
  qualification receipt: the maintainer who granted it (am423 or vcruz305), the grant date, an expiry
  date and the reason. The model-branch completeness check accepts vision off when every profile
  carries a valid one. (PR #53)


- Moved the runtime layout to `runtimes/<family>/<model>/v<N>/` with `MODEL.json`
  (`rexl3.prod-model.v1`); registry keys are `<family>/<model>/v<N>` and carry `runtime_id`. Dev
  release tags derive from `runtime_id`, never from folder names. The old flat layout is refused.
- Added `tools/check_model_branch.py` (model-branch completeness), `tools/check_pr_rules.py`,
  `tools/check_local_validation.py`, `tools/governance_audit.py`, `tools/assemble_release_notes.py`
  and `tools/cut_release.py`, with tests.
- Added the `pr-rules`, `model-completeness`, `local-validation` and `governance-audit` workflows
  and the target ruleset copies (`main-guard`, `protected-tags`, `model-branches`).

- Promotion now reuses verified release inventories: a release archive is
  decompressed and re-inventoried at most twice per attempt instead of once per
  verification phase. Every phase still requires a fresh streaming SHA-256 of
  the compressed transport bytes equal to the approval record before any cached
  inventory is honored, and tampered or unlisted members still fail closed.

- The release ceremony (`tools/cut_release.py --publish`) now verifies an approval whose bundle leaves
  the server and cluster executables to the dev engine release. It downloads the executables the approval
  names from that engine release, which must be the one the release manifest names, and checks their
  exact name, size, digest and bytes, and that the engine tag names the approved commit. Before this fix,
  every such approval, which includes every v0.1.0 model, was refused as "engine binary files are
  required". Each model's dev release is now also staged in its own directory, so a release with several
  models no longer stops at the second model on a "destination conflict" between the models' receipts.
  (PR #58)

- Production verification now accepts a recipe whose profiles each pin their own pack and runtime id,
  the layout dev uses for a model with a single-node and a two-node profile. Every receipt identity
  now names its `pack` role. A profile may name a lock role other than `draft`, and the model lock
  must carry a `target` pack and every role a profile names. Before, `verify` and the promotion tools
  derived every profile's identity from the approval's top-level runtime id and refused such a recipe
  as "approved profile identity drift". Approvals sealed before dev added the `pack` identity key no
  longer verify; none is registered in production. (PR #52)

- Made the governance audit tolerate GitHub Actions workflow-token limits: a 403 on the
  immutable-releases settings endpoint and incomplete/403 ruleset detail views are skipped with an
  informational line instead of failing the audit (real, readable drift still fails). The immutable
  releases setting remains verifiable by a maintainer token.
- The incident record now enumerates the full retroactive review range (`#1-#28`), and the audit
  parser accepts `through`-spelled ranges, so pre-restructure unreviewed merges are matched.

- Fixed the governance audit's incident-record parsing for single PR numbers (a listed PR without a
  range bound no longer crashes with `IndexError`).

- The model-branch completeness check (`model-completeness`) now reads speculative decoding the way
  the engine parses a profile's argv. A profile counts when it declares `--spec auto` or `--spec mtp`
  (the last `--spec` wins), a `--draft` pack, or a `draft` role in its model lock, and nothing turns
  drafting off (`--no-drafts`, a final `--spec none` or `--spec copy`). Before, the check looked only
  for `--draft` or any second pack in the model lock: profiles that run `--spec auto` or `--spec mtp`
  without a draft pack (GLM-5.3-Flash, Nemotron-3.5-Super-VL, Qwen3.8-Flash-Next) failed, and
  a model whose second pack is an alternate target rather than a draft passed whatever its argv said.
  (PR #51)

- `tools/promote_runtime.py` can now promote a new model with its documented command. Two defects stopped it.
  `--download` caches a dev release in `<assets-dir>/<family>/<model>/v<N>/`, but the promotion read
  `approval.json`, the archives and the engine executables from `<assets-dir>` itself and refused with
  "missing/nonregular/hardlinked file: approval.json"; it now reads that model version's cache directory, the
  same one the tree check uses. And the early check of the `MODEL.json` metadata flags left out the runtime id,
  so every promotion that passed `--display-name` and the other metadata flags, which a new model needs, was
  refused before it started; the check now uses the registry entry's runtime id.

- Pinned a deterministic git identity in the promotion test suite so `verify` passes on
  runners without a resolvable OS user name.

- Release assets are named `<runtime-id>-v<N>-approval.json` and `<runtime-id>-v<N>-bundle.tar.gz`
  (plus the `-source`, `-engine-source` and `-evidence` archives) as documented. The release tool
  had written the model version's `v` twice (`<runtime-id>-vv1-...`), and the changelog's model line
  had the same doubled `v`. (PR #49)

- The release-notes assembler (`tools/assemble_release_notes.py`) no longer reads
  `changes/unreleased/README.md`, the fragment rules, as a fragment. Its own headings had made every
  assembly on main fail with `NOTES_REFUSED: README.md: unknown heading 'When a fragment is required'`,
  so the documented release step could not run. A file there whose name is not a lowercase kebab-case
  `<slug>.md` is now refused by name instead of being parsed.

- Models: qwen3.8/qwen3.8-27b v3, glm-5.3/glm-5.3-flash v1, nemotron-3.5/nemotron-3.5-super-vl v1, qwen3.8/qwen3.8-flash-next v2, deepseek-v4.1/deepseek-v4.1-flash-1017 v1.

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No unreleased changes yet. Each user-visible change lands as a fragment under
`changes/unreleased/` and moves here at release time (see `docs/RELEASING.md`).
