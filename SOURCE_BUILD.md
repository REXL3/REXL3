# REXL3 native corresponding-source build

- Engine source commit: `afc9b5dfb69d070a306ffe6696d9b4f96db4ddec`
- Git tree: `1580b494a8ae436fb938afeb8f77a76052d3f6e8`
- Rust/Cargo toolchain: `1.89.0` (from `rust-toolchain.toml`).

This is common REXL3 engine source, including the DSV4.1 implementation, under
**AGPL-3.0-only**. Model weights are not licensed by this archive; obtain each
model pack separately under its own license. See `LICENSE`, `NOTICE`, and
`THIRD_PARTY_NOTICES.md` for the engine and bundled third-party components.

## What this archive establishes

`SOURCE_IDENTITY.json` binds each original file to its committed Git blob, SHA-256,
size, and Git/archive mode. Original source files have not been rewritten or
sanitized by the exporter. `.rexl3_commit` contains the full source commit.
The generated instruction file is bound to this committed template. The identity
manifest intentionally excludes its own digest; hash the entire archive when
binding it to a binary release. No Git history, evaluation runs, host logs,
model packs, or mutable worktree files are exported. Test source and non-parameter
fixtures are retained. Tensor/checkpoint containers (`.safetensors`, `.pt`, `.bin`,
and the other suffixes enumerated in the identity manifest) and opaque archives
are excluded before reading their blobs, even if named "fixture". Such files may
contain real learned parameters, not just synthetic inputs or expected outputs.
The repository originals are never removed or changed by this exporter.

Tests that load these assets at runtime are **external-asset-dependent**: obtain
the exact assets separately under their applicable licenses before running them.
Their source remains included, but this archive does **not** provide a self-contained
`cargo test --workspace`: even compiling asset-dependent tests requires the separate
fixtures. No tests, defaults, model code or original Rust bytes have been changed.

### EXTERNAL_TEST_DEPENDENCIES

The optional `external_test_dependencies` list in the committed input manifest
records the only reviewed compile-time exceptions. Every record binds the source
SHA-256, exact one-based line and byte-column of `include_bytes!`, literal reference,
resolved target path, and target Git blob ID/size from the same committed tree.
The target identity is **metadata-only provenance**, not a payload read, tensor
classification, license grant or proof of fixture contents. Containers remain
excluded even when reviewed. The identity manifest repeats these verified records:

- `crates/model/src/families/nemotron_h_omni/bridge.rs:1620:35` -> `testdata/nemotron_omni/moe/layer1_fixture.safetensors` (cfg_test_module; excluded Git blob `80e7cd803548e25a7c0645861850efd11eb92ecb`, 349008 bytes).
- `crates/model/src/families/nemotron_h_omni/bridge.rs:1625:9` -> `testdata/nemotron_omni/moe/layer1_fixture_repaired.safetensors` (cfg_test_module; excluded Git blob `ab067a95c0022a1452765a70fcba17560c0917e5`, 349096 bytes).
- `crates/model/tests/nemotron_omni_moe.rs:23:24` -> `testdata/nemotron_omni/moe/layer1_fixture.safetensors` (cargo_integration_test; excluded Git blob `80e7cd803548e25a7c0645861850efd11eb92ecb`, 349008 bytes).
- `crates/model/tests/nemotron_omni_moe.rs:27:33` -> `testdata/nemotron_omni/moe/layer1_fixture_repaired.safetensors` (cargo_integration_test; excluded Git blob `ab067a95c0022a1452765a70fcba17560c0917e5`, 349096 bytes).

The exporter independently verifies test scope, not just a caller's label: a
balanced, plain top-level `#[cfg(test)] mod tests` with pinned attribute/module
lines, or an immediate Cargo `tests/*.rs` auto-discovered integration target with
an exact manifest hash. Comments, strings, raw strings and character literals
cannot create scope. Unsupported/ambiguous scope fails closed. Integration sources
reused by Cargo targets/build scripts or Rust includes/path attributes are rejected.
Unknown references, changed source/manifest/provenance, stale locations, and any
production dependency on an excluded container remain fatal. Serialization repeats
this verification from original blobs; editing diagnostic fields grants nothing.

The native `gb-server` / `gb-cluster` release source closure is complete under this
static/reviewed contract (`native_build_source_complete` in the identity manifest).
These release targets do not compile the verified test-only includes above. This
is **not** evidence that a native build or the full test suite has been run. Before
running the affected tests, obtain the exact separate fixtures at the listed paths
and verify their Git blob identities; the exporter does not fetch or restore them.

The explicit projection retains every crate, CUDA/C/assembly/header source,
vendored `third_party/` source input, Cargo config/lockfile, literal Rust include
and `#[path]` closure, and quoted C/CUDA/assembly header closure. References outside
the allowed projection, missing references, and unsupported dynamic spellings
are rejected. Angle-bracket headers are external system/toolchain prerequisites.
This conservative parser is not a compiler or a general preprocessor.

`tools/corresponding_source_inputs.json` enumerates the bounded runtime-test
inputs (cluster example configurations, dashboard checker and panel definitions).
Every listed file must exist in the same commit and be included; this does not
export the operational docs tree. Every enabled Cargo build script, including a
custom `package.build`, must have an exact-SHA-256 review in that manifest with a
finite repository-input list, include search directories and external prerequisites.
Changed scripts and stale reviews fail closed. Dynamic dependency completeness
is a human-reviewed contract, not something arbitrary build-script code can prove
about itself. Review this manifest independently before release; update it when
runtime tests or build dependencies change. No build script is executed here.
The manifest records the omitted-entry count and digest without exposing their
private names. This is a source snapshot, not a full development repository.
Cargo registry dependencies are pinned by `Cargo.lock` but **not vendored**;
a first dependency fetch requires registry access, or an independently populated
Cargo cache. Compiler toolchains, CUDA libraries, and other system libraries are
external prerequisites, not files supplied by this source archive.

A clean privacy scan is a conservative packaging gate, not a guarantee that
source is public-safe. A human release review must also bind the binary's actual
build commit to this source commit. Do not repin an existing binary to a new
sanitization commit; rebuild it from the new committed source. This archive is
**not a GPU qualification**, performance result, byte-identical binary rebuild
claim, or a certification of license compliance.

## Native Linux build (aarch64 / GB10)

Provision the pinned Rust toolchain, a C17/C++17 compiler, assembler, `ar`, linker,
and CUDA 13.0 toolkit with `nvcc` supporting `sm_121`. The native build scripts
compile the CUDA kernels and vendored tokenizer. CUDA runtime, cuBLAS, cuBLASLt,
and libstdc++ must be available to the linker and runtime loader. To compile the
RoCE transport rather than its unavailable-backend fallback, also provide RDMA
development headers (`infiniband/verbs.h`, commonly `libibverbs-dev`):
`crates/cuda/csrc/roce_shim.cu` selects that implementation with `__has_include`.
Runtime TP transports additionally need their matching NCCL or RDMA libraries;
none are needed merely to construct this source archive. `.cargo/config.toml` tunes aarch64
Rust code for the build host, so use the intended target hardware/toolchain.

Run from a fresh archive extraction outside another Git worktree. Select the
installed toolkit explicitly (the conventional installation below is an example,
not a captured private build-host path):

```sh
set -eu
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
test -x "$CUDA_HOME/bin/nvcc"
"$CUDA_HOME/bin/nvcc" --version
command -v cc
command -v ar
rustup toolchain install 1.89.0 --profile minimal
cargo +1.89.0 --version
rustc +1.89.0 --version
export REXL3_CUDA_ARCH=sm_121
export REXL3_COMMIT=afc9b5dfb69d070a306ffe6696d9b4f96db4ddec
# A new target directory avoids cached CPU stubs and stale toolkit discovery.
export CARGO_TARGET_DIR="$PWD/target-native"
test ! -e "$CARGO_TARGET_DIR"
env -u REXL3_NO_CUDA -u GB_NO_CUDA -u GB_CUDA_ARCH \
  cargo +1.89.0 build --release --locked -p gb-server --bin rexl3-serve
env -u REXL3_NO_CUDA -u GB_NO_CUDA -u GB_CUDA_ARCH \
  cargo +1.89.0 build --release --locked -p gb-cluster
"$CARGO_TARGET_DIR/release/rexl3-serve" --version
```

The `REXL3_COMMIT` override is essential without Git metadata (and prevents a
surrounding repository from supplying the wrong stamp). Inspect the final
`--version` result: the commit must match above, the CUDA architecture must be
`sm_121`, and `nvcc` / `kernel_hash` must not be `none`. The source build may
otherwise succeed as a CPU-only stub. This check does not load a model or
qualify GPU behavior. Native build scripts may spawn many compiler processes
regardless of Cargo's job count; build only with adequate CPU/memory capacity.

## Recreate and verify the source archive

From a repository containing this exact commit and the matching exporter:

```sh
python3 -B tools/build_corresponding_source.py --repo . \
  --commit afc9b5dfb69d070a306ffe6696d9b4f96db4ddec --output corresponding-source.tar.gz
python3 -B tools/build_corresponding_source.py --repo . \
  --commit afc9b5dfb69d070a306ffe6696d9b4f96db4ddec --output corresponding-source.tar.gz --check
```

The exporter reads Git objects, not the working tree. `--check` regenerates the
archive and compares every byte without overwriting it. Gzip/tar metadata is
fixed (mtime/uid/gid zero, sorted paths, regular-file modes 0644/0755); compression
uses Python's zlib at level 9. Use the same exporter/Python/zlib implementation
when reproducing compressed bytes across systems. Existing output is never
replaced. Creation requires Linux `O_TMPFILE` and descriptor-based `linkat`:
an anonymous inode under the anchored parent is fully written and file-fsynced
before its atomic, no-replacement installation. Unsupported primitives fail closed.
The output is a regular 0600 file with one link. Both create and `--check` bind
exact bytes, mode, link count, timestamps and final pathname identity to the open
descriptor; creation also fsyncs the directory. Parent movement is rejected.
Failure cleanup captures owned output into a private 0700 recovery directory
before unlinking. A captured foreign replacement is restored without replacement,
or retained (at most one capture per invocation) with an explicit
`output_cleanup_quarantined` failure if restoration collides. Inspect any
`.source-recovery-*` directory after a cleanup failure; do not blindly delete it.
The private recovery namespace must not be tampered with by same-UID processes.
These are point-in-time guarantees, not permanent immutability against the file
owner, power-loss transactionality with stdout, or control of later mutations.

Symlinks/submodules, unsafe paths, unsupported includes, missing build
inputs, and resource-limit failures are rejected rather than silently omitted.

`--diagnose` emits locations/categories only, never matched secret values, and
**never creates an archive**. A blocked scan exits 3. Commit reviewed source
sanitization first, then rebuild the binary and export the new exact commit.
The exporter never substitutes scrubbed source beneath an old commit identity.

The optional committed `tools/corresponding_source_test_allowances.json` has schema
`rexl3.source-test-allowances.v1` and an `entries` array. Each record specifies an
exact relative `path`, whole-file lowercase SHA-256 `sha256`, nonempty `reason`,
and `findings` containing positive integer `line` and known scanner `category`.
Only those exact locations/categories in those exact bytes are reviewed. Missing
policy grants zero allowances. Changed bytes, stale entries, duplicates, unknown
categories and malformed records fail closed; new findings remain unreviewed and
block export. Filename findings cannot be allowed (their line is zero). The policy
cannot allow itself and is included as an original committed file in the archive
and identity manifest. There is no ignore/export-anyway switch.

Raw `findings`, `reviewed_findings` and `unreviewed_findings` remain separately
visible. `PASS_REVIEWED_TEST_FIXTURES` means reviewed findings are present and zero
unreviewed findings remain; it never means that all findings were erased. `PASS`
means no findings. `publishable` is only a packaging-gate result, never release
authorization (`release_authorized` remains false). Independently review the
source and allowance policy before release; a committed policy is an input, not
self-authorizing publication approval.
