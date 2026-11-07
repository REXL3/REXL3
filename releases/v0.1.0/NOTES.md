# REXL3 v0.1.0 release notes

<!-- Template for tools/assemble_release_notes.py (docs/RELEASING.md). The placeholders are filled
     from changes/unreleased/<slug>.md fragments and releases/v<X.Y.Z>/manifest.json; a fragment uses
     only the headings it needs. Supported models, Assets and Provenance are generated from the
     manifest. -->

## Highlights

- DeepSeek-V4.1-Flash-1017 (EXL3 1.55 bits per weight) on one GB10: 262,144-token window, one slot, speculative decoding and prefix cache on, image input off (`dsv41-single-262k`, port 8940). The two-node profile is not part of this release. Added by #75.


- GLM-5.3-Flash (EXL3 2.25 bits per weight) with speculation from the pack's NextN layer, prefix cache and image input on, in two profiles of one recipe: `glm53-flash-single` on one GB10 (262,144-token window, one slot) and `glm53-flash-tp2-1m` on two GB10 nodes with tensor parallelism (1,048,576-token window), both on port 8950. Added by #69.

- Nemotron-3.5-Super-VL (EXL3 3.85 bits per weight) on one GB10: 1,048,576-token window, 16 slots, speculative decoding with the model's MTP head, prefix cache and image input on (`nemotron35-super-vl-single-1m`, port 9100). Added by #71.

- Qwen3.8-27B (EXL3 4.00 bits per weight with a DFlash2 draft) on one GB10: 262,144-token window, six slots, speculative decoding with the DFlash2 draft, prefix cache and image input on, thinking at `low` effort by default (`qwen38-single-262k`, port 8889). Added by #66.

- Qwen3.8-Flash-Next (EXL3 3.05 bits per weight) on one GB10: 262,144-token window, two slots, speculative decoding with the model's MTP head (`--spec mtp`) and prefix cache on (`qwen38-flash-next-single-262k`, port 8899). Added by #73.

## Supported models

| Model | Architecture | Pack | Context | Speculative decoding | Prefix cache | Vision | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.8/qwen3.8-27b | Qwen3_5ForConditionalGeneration | EXL3 4.00 bpw + DFlash2 4.00 bpw | 262,144 (single-node) | DFlash2 (k=16, auto) | on | on (image) | qwen38-27b-dflash2-exl3-4.00-v3-evidence.tar.gz |
| glm-5.3/glm-5.3-flash | Glm5NextForConditionalGeneration | EXL3 2.25 bpw | 262,144 (single-node) / 1,048,576 (two-node) | MTP (auto) | on | on (image) | glm53-flash-2.25bpw-sm121-v1-evidence.tar.gz |
| nemotron-3.5/nemotron-3.5-super-vl | NemotronH_Omni_Reasoning_V3 | EXL3 3.85 bpw | 1,048,576 (single-node) | MTP (auto) | on | on (image) | nemotron35-super-vl-sage-3.85-v1-evidence.tar.gz |
| qwen3.8/qwen3.8-flash-next | Qwen4ExpForConditionalGeneration | EXL3 3.05 bpw | 262,144 (single-node) | MTP (auto) | on | off (owner exception through 2027-01-31) | qwen38-flash-next-exl3-3.05-v2-evidence.tar.gz |
| deepseek-v4.1/deepseek-v4.1-flash-1017 | DeepseekV41ForCausalLM | EXL3 1.55 bpw | 262,144 (single-node, dsv41-single-262k) | MTP (auto) | on | off (owner exception through 2026-12-31) | dsv41-flash-1017-sage-1.55-v1-evidence.tar.gz |

## New features and improvements

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

## Bug fixes

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

## Breaking changes and upgrade notes

- The verifier now refuses a vision-off qualification receipt with no owner vision exception, or with
  an expired or malformed one, and refuses an expired or malformed record on a vision-on receipt. The
  expiry is checked against the date of each verification run. (PR #53)

## Known issues

- At temperature 0 the model can repeat a phrase until the reply cap; use the sampling defaults for long reasoning. In the release quality run the hard-reasoning set scored 16 of 20, one case under its regression bar, and was accepted for v0.1.0.

- GLM-5.3-Flash on one node (`glm53-flash-single`) keeps 2 prefix-cache entries at its 262,144-token window, so the cache holds one long conversation at a time. A second long conversation evicts the first, and the next request of the first one recomputes its prompt. Replies are unchanged.
- GLM-5.3-Flash's 1,048,576-token window needs the two-node profile; one GB10 holds 262,144 tokens with the required features on.
- GLM-5.3-Flash's two-node profile listens on the leader node's network address with no API key and no request log. Run it only on a private, trusted network, or behind a firewall or an SSH tunnel.
- GLM-5.3-Flash's thinking has no off switch; scale it with `reasoning_effort`.
- With greedy decoding (`temperature` 0), on a prompt close to the 1,048,576-token window GLM-5.3-Flash can start a second turn after it answers, so the reply repeats its reasoning and its answer. Use the model's default sampling (leave `temperature` and `top_p` unset: 1.0 and 0.95). A fix that ends the reply at that point is planned for v0.1.1.

- Nemotron-3.5-Super-VL accepts images of up to 3,328 vision tokens each and refuses audio.
- Nemotron-3.5-Super-VL takes video only as a list of image frames (at most 32 frames or 4,096 tokens); video files (`data:video/mp4`, `webm`, `gif`) are refused. Video sent as frames works but is not covered by the v0.1.0 qualification.

- Qwen3.8-27B launches only with the packs the launcher downloads itself; an existing local copy cannot be selected through the `model_dir` site field.
- Qwen3.8-27B refuses video, audio and constrained JSON response formats.

- Qwen3.8-Flash-Next's served model name is `default`, not a name derived from the model. It is the name `GET /v1/models` lists and the one requests send as `model`.
- Qwen3.8-Flash-Next runs with vision off, under an owner vision exception that expires on 2027-01-31: the engine does not build this model's vision tower yet. Image input is refused.
- Qwen3.8-Flash-Next drafts with MTP only while one request is decoding; when both slots are decoding, generation runs without drafts. Replies are the same either way.

- Served model names (the `model` field of a request, as `GET /v1/models` lists it) differ in style from model to model in v0.1.0: Qwen3.8-27B `qwen38-27b-exl3`, Qwen3.8-Flash-Next `default`, GLM-5.3-Flash `GLM-5.3-Flash-EXL3-2.25bpw-sm121`, Nemotron-3.5-Super-VL `nemotron-3.5-super-vl`. Read the name from `GET /v1/models` or the recipe's `profile.json` instead of deriving it.

## Assets and checksums

- `rexl3-cluster-0.1.0-afc9b5df-linux-aarch64-sm121` sha256 `29bf146f707e1a6acef7ef1f45a47b3671f6d7112cf66254c7574c5f13c4f306`
- `rexl3-engine-0.1.0-afc9b5df-source.tar.gz` sha256 `ade2a05b0629cbbbc8921ee945cad18ffff1e899a3f9c83dd74b24504fe7a046`
- `rexl3-serve-0.1.0-afc9b5df-linux-aarch64-sm121` sha256 `40e23e7116eff881e3a4545304cd82f18f84c532ae2f7b68506af02373e48318`
- qwen3.8/qwen3.8-27b v3: approval sha256 `d26800e692a5ac147a774da8505cd36b77a555cfb79414b2329ca923c38441de`
- glm-5.3/glm-5.3-flash v1: approval sha256 `7d1cca67cc1d39871eabc9c786530dda50719c96339a2ee9d1ad3dcafc39492f`
- nemotron-3.5/nemotron-3.5-super-vl v1: approval sha256 `bb6a5a00929fd05185f74c91d9d4aca6f955e94c43827413eb0d66b89abd4548`
- qwen3.8/qwen3.8-flash-next v2: approval sha256 `fdbd312cbf6d7345bbb61c9c803d11771933511ed332b3218e1630ac50b7ab2e`
- deepseek-v4.1/deepseek-v4.1-flash-1017 v1: approval sha256 `2485106e16499278c95c49c846992018e1c5eb227b738ae0a2e37fb8cf649d73`

## Provenance

- Engine dev commit `afc9b5dfb69d070a306ffe6696d9b4f96db4ddec` tree `1580b494a8ae436fb938afeb8f77a76052d3f6e8`; dev engine release tag `engine-v0.1.0-afc9b5df`.
- Models sealed from the engine dev commit `afc9b5dfb69d070a306ffe6696d9b4f96db4ddec`: qwen3.8/qwen3.8-flash-next v2, deepseek-v4.1/deepseek-v4.1-flash-1017 v1.
- qwen3.8/qwen3.8-27b v3 carries forward its previously sealed approval at commit `799fa750f8eb7d4c45d757d9c44beeb7182b5d51` (engine release `engine-v0.1.0-799fa750`).
- glm-5.3/glm-5.3-flash v1 carries forward its previously sealed approval at commit `799fa750f8eb7d4c45d757d9c44beeb7182b5d51` (engine release `engine-v0.1.0-799fa750`).
- nemotron-3.5/nemotron-3.5-super-vl v1 carries forward its previously sealed approval at commit `799fa750f8eb7d4c45d757d9c44beeb7182b5d51` (engine release `engine-v0.1.0-799fa750`).
