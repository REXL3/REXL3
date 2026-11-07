# AGENTS.md

These are binding instructions and a getting-started guide for AI coding agents working in or testing this repository.

**Read [CONTRIBUTING.md](CONTRIBUTING.md) in full before making any repository change, and read [docs/USER_GUIDE.md](docs/USER_GUIDE.md) before running a model.** Model branches follow [docs/PROMOTION.md](docs/PROMOTION.md) and releases follow [docs/RELEASING.md](docs/RELEASING.md). If anything here conflicts with CONTRIBUTING.md, CONTRIBUTING.md wins, except that nothing there relaxes the "Never" list below.

This is the production repository (`REXL3/REXL3`). It holds sealed, data-only runtime trees under `runtimes/<family>/<model>/v<N>/`, the pack sheets under `docs/packs/`, the policy that decides which trees are accepted, and the stdlib-Python tools that verify and promote them. Engine development, validation and sealing happen in the dev repository (`REXL3/REXL3-dev`), never here.

---

## Getting started for tester agents (running, benchmarking, and reporting findings on GB10)

If your user asked you to set up, run, benchmark, troubleshoot, or prototype a fix/improvement on a released REXL3 model on an NVIDIA GB10 (DGX Spark class, Ubuntu 24.04 ARM64) host, start here.

### 1. `v0.1.0` model map (trees, runtime IDs, profiles, ports, and pack sheets)

| Family / Model | Prod Tree | `runtime_id` | Default Profile (`--profile`) | Context | Port | Served `model` ID (`GET /v1/models`) | Pack Sheet |
|---|---|---|---|---|---|---|---|
| `qwen3.8/qwen3.8-27b` | `runtimes/qwen3.8/qwen3.8-27b/v3` | `qwen38-27b-dflash2-exl3-4.00` | `qwen38-single-262k` | `262,144` (1 node) | `8889` | `qwen38-27b-exl3` | [docs/packs/qwen3.8-27b.md](docs/packs/qwen3.8-27b.md) |
| `qwen3.8/qwen3.8-flash-next` | `runtimes/qwen3.8/qwen3.8-flash-next/v2` | `qwen38-flash-next-exl3-3.05` | `qwen38-flash-next-single-262k` | `262,144` (1 node) | `8899` | `default` | [docs/packs/qwen3.8-flash-next.md](docs/packs/qwen3.8-flash-next.md) |
| `glm-5.3/glm-5.3-flash` | `runtimes/glm-5.3/glm-5.3-flash/v1` | `glm53-flash-2.25bpw-sm121` | `glm53-flash-single` (`tp1`) / `glm53-flash-tp2-1m` (`tp2`) | `262,144` (1 node) / `1,048,576` (2 nodes) | `8950` | `GLM-5.3-Flash-EXL3-2.25bpw-sm121` | [docs/packs/glm-5.3-flash.md](docs/packs/glm-5.3-flash.md) |
| `nemotron-3.5/nemotron-3.5-super-vl` | `runtimes/nemotron-3.5/nemotron-3.5-super-vl/v1` | `nemotron35-super-vl-sage-3.85` | `nemotron35-super-vl-single-1m` | `1,048,576` (1 node) | `9100` | `nemotron-3.5-super-vl` | [docs/packs/nemotron-3.5-super-vl.md](docs/packs/nemotron-3.5-super-vl.md) |
| `deepseek-v4.1/deepseek-v4.1-flash-1017` | `runtimes/deepseek-v4.1/deepseek-v4.1-flash-1017/v1` | `dsv41-flash-1017-sage-1.55` | `dsv41-single-262k` | `262,144` (1 node) | `8940` | `DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw` | [docs/packs/deepseek-v4.1-flash-1017.md](docs/packs/deepseek-v4.1-flash-1017.md) |

Before running a model, read its pack sheet (`docs/packs/<model>.md`) and its `runtimes/<family>/<model>/v<N>/KNOWN_ISSUES.md`.

### 2. End-to-end launch workflow on a GB10 host

Follow [docs/USER_GUIDE.md](docs/USER_GUIDE.md) section 3:

```bash
TAG=v0.1.0
RUNTIME_ID=dsv41-flash-1017-sage-1.55   # choose from the table above

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

Every release ships the exact corresponding engine source archive (`rexl3-engine-*-source.tar.gz` / `<runtime-id>-v<N>-engine-source.tar.gz`) containing `crates/`, `Cargo.toml`, `Cargo.lock`, `third_party/`, and `SOURCE_BUILD.md`:

- **Building from source:** extract the engine source archive to a separate scratch directory (never inside a sealed `runtimes/*/*/v*/` tree) and follow `SOURCE_BUILD.md` (`cargo +1.89.0 build --release --locked`).
- **Validation standards for any fix or optimization you test:**
  1. **Always test at the profile's full supported context window** (`262,144` tokens for single-node `262K` profiles; `1,048,576` tokens for `1M` profiles) — **never validate at `32k` (`32,768`) context**.
  2. **Keep speculative decoding (`--spec auto` / `--spec mtp` / `--draft`) and prefix caching (`--prefix-cache on`) enabled.** The headline decode metric is live speculative decoding; plain (`--spec none`) tok/s is diagnostic only.
  3. **Verify exactness and no-regression:** check that `drafted == serial` and chunked prefill matches single-chunk prefill, and report median wall-clock numbers across 3 runs (`decode tok/s`, `prefill 50% tok/s`, `prefill 100% tok/s`, and free GPU memory headroom).

### 4. Reporting bugs, findings, fixes, and improvements (`Why / What / How to Fix`)

When reporting an issue or submitting a root-cause finding, bug fix, or optimization on `REXL3/REXL3`, use the structured issue templates under `.github/ISSUE_TEMPLATE/`:

1. **Bug or issue report (`.github/ISSUE_TEMPLATE/tester-report.yml`):** for failures, crashes, wrong outputs, or regressions.
2. **Finding, fix, or improvement (`.github/ISSUE_TEMPLATE/tester-fix-or-improvement.yml`):** for root-cause analyses, patches/diffs, kernel/runtime speedups, or recipe improvements.

Every submission must clearly state:
- **Scope:** whether the finding/fix is specific to **one model family** (`qwen3.8`, `glm-5.3`, `nemotron-3.5`, `deepseek-v4.1`, `ling-3.1`) or affects the **engine as a whole** (`Engine-wide`).
- **1. WHY (Finding, Root Cause, or Bottleneck):** what you observed/measured, the underlying root cause, and the user/tester impact.
- **2. WHAT (Affected Components and Contract Impact):** which files, crates, scripts, or profile flags change, and whether output bits stay bit-exact.
- **3. HOW TO FIX AND VERIFY (Concrete Steps, Patch/Diff, and Full-Context Verification):** step-by-step fix or diff, exact commands run at the profile's full supported context (`262,144` or `1,048,576`), and before/after numbers.

---

## Never (no instruction from anyone overrides these)

1. **Never create, edit, publish or delete a release or a tag.** No `gh release`, no releases API, no `git tag` push, no `git push --tags`. A release is cut only by a maintainer personally, following docs/RELEASING.md.
2. **Never merge a pull request, and never submit a review that approves one.** In production every approval comes from a maintainer personally, after their own review. You prepare the PR and report; a maintainer merges.
3. **Never change repository settings**: rulesets, branch or tag protection, environments, secrets, variables, collaborators, teams, webhooks, deploy keys, Actions permissions or release settings.
4. **Never work with admin credentials.** If the token you were given can administer this repository, stop and tell the maintainer you work for.
5. **Never push to `main`, never force-push, and never delete a `model/*` branch.** Never bypass hooks or checks.
6. **Never edit a sealed tree** (`runtimes/*/*/v*/`). A sealed tree is data with a hash-pinned inventory; a pack changes only through a new sealed version from the dev repository.
7. **Never weaken the verifier, a check, a test or a workflow safety rule**, and never in the same PR as the change it guards. Workflows read code and policy from the protected base revision and never execute pull-request code; keep it that way.

## Always

1. **Use the right branch type** (CONTRIBUTING.md section 3):
   - `model/<model>` brings one validated model into production, for example `model/qwen3.8-27b`;
   - `model/<model>-approval` registers that model's dev approval and merges first;
   - `engine/<topic>` is any universal production change: tools, workflows, policy, docs, templates;
   - `release/v<X.Y.Z>` is opened by a maintainer only; never open one.
2. **Name runtime folders by family, then model**: `runtimes/<family>/<model>/`, lowercase, for example `runtimes/qwen3.8/qwen3.8-27b/`. Never put a bit width, quantization format, draft method or other pack detail in a folder or branch name; those belong in `MODEL.json` and the sealed approval. A derivative model with its own name gets its own model folder.
3. **Prepare a model branch, registry entry or promotion only when the maintainer you work for explicitly asked for that model and version.** Quote or link the instruction in the PR. The approver of record is that maintainer, never an agent.
4. **One purpose per PR.** Title `TYPE: Past-tense summary`, TYPE one of FIX, FEAT, BUILD, CI, TEST, BUMP or DOCS, at most 90 characters, no trailing period. Fill in the matching PR template and add a release-notes fragment under `changes/unreleased/` when CONTRIBUTING.md requires one.
5. **For a model PR, run `tools/check_model_branch.py`** as described in docs/PROMOTION.md and paste its full output.
6. **Evidence, not claims.** An exit code of 0 is not a pass. Paste the commands, the head SHA and the output.
7. **No secrets, no personal email, no internal project names.** Commit with a GitHub noreply identity. Describe SAGE only as: "SAGE is our own mixed-precision quantization method for EXL3. The method is not published."
8. **When anything is unclear, or a rule blocks you, stop and ask the maintainer you work for.** Do not look for a workaround.
