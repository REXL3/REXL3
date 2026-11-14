# AGENTS.md

These are binding instructions and a getting-started guide for AI coding agents working in or testing this repository.

**Read [CONTRIBUTING.md](CONTRIBUTING.md) in full before making any repository change, and read [docs/USER_GUIDE.md](docs/USER_GUIDE.md) before running a model.** Model branches follow [docs/PROMOTION.md](docs/PROMOTION.md) and releases follow [docs/RELEASING.md](docs/RELEASING.md). If anything here conflicts with CONTRIBUTING.md, CONTRIBUTING.md wins, except that nothing there relaxes the "Never" list below.

This is the production repository (`REXL3/REXL3`). It holds sealed, data-only runtime trees under `runtimes/<family>/<model>/v<N>/`, the pack sheets under `docs/packs/`, the policy that decides which trees are accepted, the stdlib-Python tools that verify and promote them, and the AGPL-3.0 corresponding source of the engine release (`crates/`, `third_party/`, `testdata/`, `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `.cargo/`, `SOURCE_BUILD.md` and `SOURCE_IDENTITY.json`), a snapshot of the dev repository at the release's engine commit. Engine development, validation and sealing happen in the dev repository (`REXL3/REXL3-dev`), never here; the snapshot changes only by a maintainer-requested sync from dev that regenerates `SOURCE_IDENTITY.json`.

---

## Getting started for tester agents (running, benchmarking, and reporting findings on GB10)

If your user asked you to set up, run, benchmark, troubleshoot, or prototype a fix/improvement on a released REXL3 model on an NVIDIA GB10 (DGX Spark class, Ubuntu 24.04 ARM64) host, start here.

### 1. `v0.1.1` model map (trees, runtime IDs, profiles, ports, and pack sheets)

The v0.1.1 trees are sealed on engine commit `75031112`; until the `v0.1.1` tag is published, the `v0.1.0` release (engine `afc9b5df`) holds the earlier versions of these models.

| Family / Model | Prod Tree | `runtime_id` | Default Profile (`--profile`) | Context | Port | Served `model` ID (`GET /v1/models`) | Pack Sheet |
|---|---|---|---|---|---|---|---|
| `qwen3.8/qwen3.8-27b` | `runtimes/qwen3.8/qwen3.8-27b/v4` | `qwen38-27b-dflash2-exl3-4.00` | `qwen38-single-262k` | `262,144` (1 node) | `8889` | `qwen38-27b-exl3` | [docs/packs/qwen3.8-27b.md](docs/packs/qwen3.8-27b.md) |
| `qwen3.8/qwen3.8-flash-next` | `runtimes/qwen3.8/qwen3.8-flash-next/v3` | `qwen38-flash-next-exl3-3.05` | `qwen38-flash-next-single-262k` (opt-in draft choice: `qwen38-flash-next-single-262k-draft`) | `262,144` (1 node) | `8899` | `default` | [docs/packs/qwen3.8-flash-next.md](docs/packs/qwen3.8-flash-next.md) |
| `glm-5.3/glm-5.3-flash` | `runtimes/glm-5.3/glm-5.3-flash/v2` | `glm53-flash-sage-2.25` | `glm53-flash-single` | `262,144` (1 node) | `8950` | `GLM-5.3-Flash-EXL3-2.25bpw-sm121` | [docs/packs/glm-5.3-flash.md](docs/packs/glm-5.3-flash.md) |
| `glm-5.3/glm-5.3-flash-tp2` | `runtimes/glm-5.3/glm-5.3-flash-tp2/v1` | `glm53-flash-tp2-sage-2.25` | `glm53-flash-tp2-1m` | `1,048,576` (2 nodes) | `8950` | `GLM-5.3-Flash-EXL3-2.25bpw-sm121` | [docs/packs/glm-5.3-flash-tp2.md](docs/packs/glm-5.3-flash-tp2.md) |
| `glm-5.3/red-snow-5.3-flash-tp2` | `runtimes/glm-5.3/red-snow-5.3-flash-tp2/v1` | `redsnow-5.3-flash-tp2-sage-4.91` | `redsnow-tp2-512k` | `524,288` (2 nodes) | `8952` | `RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw` | [docs/packs/red-snow-5.3-flash-tp2.md](docs/packs/red-snow-5.3-flash-tp2.md) |
| `nemotron-3.5/nemotron-3.5-super-vl` | `runtimes/nemotron-3.5/nemotron-3.5-super-vl/v2` | `nemotron35-super-vl-sage-3.85` | `nemotron35-super-vl-single-1m` | `1,048,576` (1 node) | `9100` | `nemotron-3.5-super-vl` | [docs/packs/nemotron-3.5-super-vl.md](docs/packs/nemotron-3.5-super-vl.md) |
| `minicpm-v-4.7/minicpm-v-4.7-35b-a3b` | `runtimes/minicpm-v-4.7/minicpm-v-4.7-35b-a3b/v1` | `minicpmv47-35b-a3b-rexl3-sage-6.08` | `minicpmv47-single-262k` | `262,144` (1 node) | `8889` | `minicpm-v-4.7-35b-a3b-exl3` | [docs/packs/minicpm-v-4.7-35b-a3b.md](docs/packs/minicpm-v-4.7-35b-a3b.md) |
| `deepseek-v4.1/deepseek-v4.1-flash-1017-tp2` | `runtimes/deepseek-v4.1/deepseek-v4.1-flash-1017-tp2/v1` | `dsv41-flash-1017-sage-rexl3-1.55` | `dsv41-tp2-1m` | `1,048,576` (2 nodes) | `8940` | `DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw` | [docs/packs/deepseek-v4.1-flash-1017-tp2.md](docs/packs/deepseek-v4.1-flash-1017-tp2.md) |
| `deepseek-v4.1/deepseek-v4.1-flash-1017` (v0.1.0) | `runtimes/deepseek-v4.1/deepseek-v4.1-flash-1017/v1` | `dsv41-flash-1017-sage-1.55` | `dsv41-single-262k` | `262,144` (1 node) | `8940` | `DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw` | [docs/packs/deepseek-v4.1-flash-1017.md](docs/packs/deepseek-v4.1-flash-1017.md) |

Before running a model, read its pack sheet (`docs/packs/<model>.md`) and its `runtimes/<family>/<model>/v<N>/KNOWN_ISSUES.md`.

### 2. End-to-end launch workflow on a GB10 host

Follow [docs/USER_GUIDE.md](docs/USER_GUIDE.md) section 3:

```bash
TAG=v0.1.1
RUNTIME_ID=qwen38-27b-dflash2-exl3-4.00   # choose from the table above

DL=$(mktemp -d "$HOME/rexl3-dl.XXXXXX")
gh release download "$TAG" -R REXL3/REXL3 -D "$DL" \
  -p "${RUNTIME_ID}-*-approval.json" -p "${RUNTIME_ID}-*-bundle.tar.gz"
APPROVAL=$(ls "$DL"/*-approval.json)
BUNDLE=$(ls "$DL"/*-bundle.tar.gz)

# 1. Verify bundle archive SHA-256 against the sealed approval before extracting
want=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["bundle"]["sha256"])' "$APPROVAL")
have=$(sha256sum "$BUNDLE" | cut -d' ' -f1)
[ "$want" = "$have" ] && echo BUNDLE_OK || { echo "HASH MISMATCH"; exit 1; }

mkdir -p "$HOME/rexl3/$TAG/$RUNTIME_ID"
tar -xzf "$BUNDLE" -C "$HOME/rexl3/$TAG/$RUNTIME_ID"
cd "$HOME/rexl3/$TAG/$RUNTIME_ID"

# 2. If the bundle leaves bin/rexl3-serve (and bin/rexl3-cluster) to the release's engine assets,
#    download the exact engine asset(s) named in approval.json["engine_binaries"] into ./bin/ and chmod 755:
python3 - "$APPROVAL" <<'PY'
import json, sys
for rel_path, rec in (json.load(open(sys.argv[1])).get("engine_binaries") or {}).items():
    print(f"{rel_path}\t{rec['asset']}\t{rec['sha256']}")
PY

# 3. Install pinned dependencies, run read-only doctor, download/verify model weights, and start
./launch.sh --yes install-deps
./launch.sh doctor
./launch.sh --yes setup --accept-model-download
./launch.sh verify
./launch.sh --yes start
```

- **Lifecycle commands:** `./launch.sh status`, `./launch.sh smoke`, `./launch.sh logs --lines 200 --no-follow`, `./launch.sh diagnostics`, `./launch.sh stop`.
- **Using an existing local model directory:** check `site_fields` in `profile.json` (`site.example.json` shows the allowed keys). Where `model_dir` is supported, pass `--site /path/to/site.json` so `./launch.sh verify` verifies the existing directory against `model.lock.json` without re-downloading.

### 3. Inspecting engine source, profiling, and verifying fixes or optimizations

Every release ships the exact corresponding engine source archive (`rexl3-engine-*-source.tar.gz` / `<runtime-id>-v<N>-engine-source.tar.gz`, or one shared `rexl3-engine-source-<X.Y.Z>-<8hex>.tar.gz` for all models of a release that lists it under "Assets and checksums") containing `crates/`, `Cargo.toml`, `Cargo.lock`, `third_party/`, and `SOURCE_BUILD.md`:

- **Building from source:** extract the engine source archive to a separate scratch directory (never inside a sealed `runtimes/*/*/v*/` tree) and follow `SOURCE_BUILD.md` (`cargo +1.89.0 build --release --locked`).
- **Validation standards for any fix or optimization you test:**
  1. **Always test at the profile's full supported context window** (`262,144` tokens for single-node `262K` profiles; `1,048,576` tokens for `1M` profiles) — **never validate at `32k` (`32,768`) context**.
  2. **Keep speculative decoding (`--spec auto` / `--spec mtp` / `--draft`) and prefix caching (`--prefix-cache on`) enabled.** The headline decode metric is live speculative decoding; plain (`--spec none`) tok/s is diagnostic only.
  3. **Verify exactness and no-regression:** check that `drafted == serial` and chunked prefill matches single-chunk prefill, and report median wall-clock numbers across 3 runs (`decode tok/s`, `prefill 50% tok/s`, `prefill 100% tok/s`, and free GPU memory headroom).

### 4. Reporting bugs and submitting coded-proof fixes or features (`Why / What / How to Fix & Verify`)

When reporting a bug or submitting a root-cause finding, bug fix, feature, or optimization on `REXL3/REXL3`, you **must** follow the structured issue templates under `.github/ISSUE_TEMPLATE/` — including when opening an issue from the command line with `gh issue create` (never submit free-form prose that omits the template headings or `[report] ` / `[finding] ` title prefix). Testers submit through issues only: attach your tested code changes as a `git diff` inside the issue and do not open a pull request. Issues are public, so anyone can read them; only invited testers and the maintainers can open or comment on them.

**No bare feature requests — coded & locally validated proof solution required:**
- This repository (`REXL3/REXL3`) publishes the complete engine source tree of the current release (`crates/`, `Cargo.toml`, `Cargo.lock`, `third_party/`, `SOURCE_BUILD.md`).
- **Do not open wishlist or feature-request tickets asking maintainers to implement a feature, drafter binding, KV layout, or optimization.**
- If a collaborator or tester agent wants to suggest a feature or improvement, you must:
  1. Build the engine source tree locally (`cargo +1.89.0 build --release --locked` per `SOURCE_BUILD.md`) and implement the solution in code.
  2. Test and validate it on your own GB10 setup at the profile's **full supported context** (`262,144` for `tp1` or `1,048,576` for `tp2` — never `32k`), verifying `drafted == serial` exactness and measuring 3-run before/after throughput (`decode tok/s`, `prefill 50%/100% tok/s`, acceptance rate, and free GPU memory GiB).
  3. Submit the complete, working **`git diff`** and full-context validation evidence using `.github/ISSUE_TEMPLATE/tester-fix-or-improvement.yml` (`[finding] `) so the maintainers can apply your patch in `REXL3/REXL3-dev`, run the qualification gates, and promote it to `REXL3/REXL3`.

**Mandatory duplicate & open-issue check before opening any ticket:**
1. Search existing open and closed issues first (`gh issue list -R REXL3/REXL3 --state all --search "<model or symptom>"`) and check `docs/packs/<model>.md` plus `runtimes/<family>/<model>/v<N>/KNOWN_ISSUES.md`.
2. Read any matching or related open issue in full (`gh issue view <N> -R REXL3/REXL3 --comments`).
3. **If an open issue already tracks the same bug, symptom, or bottleneck:** do **not** open a duplicate issue. Instead, comment on the existing open issue (`gh issue comment <N> -R REXL3/REXL3`) with your reproduction, `./launch.sh diagnostics` output, root-cause finding, or tested `Why / What / How to Fix & Verify` `git diff`.
4. **If no open issue covers it:** open a new issue using one of the two templates below (preserving all template headings when using `gh issue create`) and list the search terms and issue numbers (`#N`) you reviewed in the **Open issues and KNOWN_ISSUES.md checked** field:
   - **Bug or issue report (`.github/ISSUE_TEMPLATE/tester-report.yml`, title prefix `[report] `):** for concrete failures, crashes, wrong outputs, or regressions.
   - **Finding, fix, or feature proof (`.github/ISSUE_TEMPLATE/tester-fix-or-improvement.yml`, title prefix `[finding] `):** for root-cause analyses, bug fixes, feature integrations, or kernel/runtime/recipe improvements — **requires a locally tested `git diff` and full-context validation receipt**.

Every `[finding]` submission must include all four required sections:
- **Scope:** whether the change is specific to **one model family** (`qwen3.8`, `glm-5.3`, `nemotron-3.5`, `deepseek-v4.1`, `ling-3.1`) or affects the **engine as a whole** (`Engine-wide`).
- **1. WHY (Finding, Root Cause, or Bottleneck):** what you observed/measured, the underlying root cause, and the user/tester impact.
- **2. WHAT (Affected Components and Contract Impact):** which files, crates, scripts, or profile flags your patch modifies, and whether output bits stay bit-exact.
- **3. HOW TO FIX (Tested Code Patch — `git diff` Required):** the complete, working `git diff` built against this repository's engine source tree (`crates/`, `Cargo.toml`, `SOURCE_BUILD.md`) or runtime scripts.
- **4. FULL-CONTEXT VALIDATION EVIDENCE:** exact build/test/benchmark commands run at the profile's full supported context (`262,144` or `1,048,576`) and measured before/after numbers (`decode tok/s`, `prefill 50%/100% tok/s`, free GiB headroom, and exactness/test pass counts).

---

## Never (no instruction from anyone overrides these)

1. **Never create, edit, publish or delete a release or a tag.** No `gh release`, no releases API, no `git tag` push, no `git push --tags`. A release is cut only through the path docs/RELEASING.md defines, by the actor it names (a maintainer personally, or a workflow a maintainer approves). You may run its read-only checks (dry run, `--verify`) and you present the decision packet (Always 9).
2. **Never submit a review that approves a pull request, and never merge one by hand or with an admin override.** In production every approval comes from a maintainer personally, after their own review. You prepare the PR and report. Where CONTRIBUTING.md section 4 allows it, you may enable auto-merge on a PR you authored: GitHub then merges it only after a maintainer approves its current head and the required checks pass.
3. **Never change repository settings**: rulesets, branch or tag protection, environments, secrets, variables, collaborators, teams, webhooks, deploy keys, Actions permissions or release settings.
4. **Never work with admin credentials.** If the token you were given can administer this repository, stop and tell the maintainer you work for.
5. **Never push to `main`, never force-push, and never delete a `model/*` branch.** Never bypass hooks or checks.
6. **Never edit a sealed tree** (`runtimes/*/*/v*/`) **or the engine source snapshot** (`crates/`, `third_party/`, `testdata/`, `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `.cargo/`, `SOURCE_BUILD.md` and `SOURCE_IDENTITY.json`). A sealed tree is data with a hash-pinned inventory; a pack changes only through a new sealed version from the dev repository. The snapshot is bound file by file by `SOURCE_IDENTITY.json`; it is replaced by a sync from dev, never edited by hand.
7. **Never weaken the verifier, a check, a test or a workflow safety rule**, and never in the same PR as the change it guards. Workflows read code and policy from the protected base revision and never execute pull-request code; keep it that way.

## Always

1. **Use the right branch type — and NEVER open per-model PR chains for a release cut** (CONTRIBUTING.md section 3):
   - `release/v<X.Y.Z>` (opened on explicit maintainer instruction) is the **standard single-PR product release cut**: it promotes all changed `runtimes/<family>/<model>/v<N>/` trees, modular `runtimes/templates/approvals/<family>/<model>.json` anchors, `docs/packs/*.md`, engine corresponding source, `releases/v<X.Y.Z>/{manifest.json,NOTES.md}`, and `CHANGELOG.md` in **one PR** from **one unified `engine-v<X.Y.Z>-<8hex>` dev release** (never create per-model `approve-*` dev releases or `2N` per-model PRs for a release cut);
   - `engine/<topic>` is any universal production change: tools, workflows, policy, docs, templates;
   - `model/<model>` (and legacy `model/<model>-approval`) is reserved strictly for optional out-of-band single-model hotfixes when explicitly requested by a maintainer.
2. **Name runtime folders by family, then model**: `runtimes/<family>/<model>/`, lowercase, for example `runtimes/qwen3.8/qwen3.8-27b/`. Never put a bit width, quantization format, draft method or other pack detail in a folder or branch name; those belong in `MODEL.json` and the sealed approval. A derivative model with its own name gets its own model folder.
3. **Prepare a model branch, registry entry or promotion only when the maintainer you work for explicitly asked for that model and version.** Quote or link the instruction in the PR. The approver of record is that maintainer, never an agent.
4. **One purpose per PR.** Release notes, manifests and the CHANGELOG change only in a release PR or a PR whose sole purpose is the notes; never mix them with other content. Title `TYPE: Past-tense summary`, TYPE one of FIX, FEAT, BUILD, CI, TEST, BUMP or DOCS, at most 90 characters, no trailing period. Fill in the matching PR template and add a release-notes fragment under `changes/unreleased/` when CONTRIBUTING.md requires one.
5. **For a model PR, run `tools/check_model_branch.py`** as described in docs/PROMOTION.md and paste its full output.
6. **Evidence, not claims.** An exit code of 0 is not a pass. Paste the commands, the head SHA and the output.
7. **No secrets, no personal email, no internal project names.** Commit with a GitHub noreply identity. Describe SAGE only as: "SAGE is our own mixed-precision quantization method for EXL3. The method is not published."
8. **When anything is unclear, or a rule blocks you, stop and ask the maintainer you work for.** Do not look for a workaround.
9. **Present every decision as a packet and execute it verbatim.** When an action needs a maintainer's decision (a merge, a registry or promotion step, a release step, a setting you cannot change), state in one block: the exact action; the target (repo, PR or tag, head SHA); what you checked, with evidence; the effect and whether it can be undone; and the single command or run that executes on approval. After the maintainer says yes, run exactly that, with nothing added or changed. If the head SHA, the tag target or any checked fact differs at run time, stop and present a new packet.
