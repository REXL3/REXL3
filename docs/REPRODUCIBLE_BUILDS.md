# Reproducible builds

This page applies to the engine source of v0.1.3 (engine commit `0c8bc43c`) and later. How to build the
engine is in [SOURCE_BUILD.md](../SOURCE_BUILD.md); this page says when two builds give the same files.

One commit builds to the same release binaries on every node, in any directory, for any user, at any time, from an empty
target directory or an old one. So each node of a cluster may build its own binaries, or take a copy: the sha256 is the
same either way.

```sh
cargo build --release --locked --workspace
sha256sum target/release/rexl3-serve target/release/rexl3-cluster target/release/rexl3-router \
          target/release/exl3-inspect target/release/rexl3-quant
```

Run the two commands on each node and compare the lines. `rexl3-cluster check` compares the server binary's sha256
across a group for you.

## What must be equal for the files to be equal

The commit is not the only input. Two builds agree when all of these agree:

- the commit, with a clean tree, and the same `REXL3_COMMIT` if that is set (a dirty tree is stamped `-dirty`);
- the Rust toolchain (pinned in `rust-toolchain.toml`) and `Cargo.lock` (`--locked`);
- the CUDA toolkit version (`nvcc --version`) and the host C compiler and linker (`gcc --version`, `ld --version`);
- the CPU: `.cargo/config.toml` builds for the native Arm core on aarch64, so every GB10 agrees and another Arm
  machine may not;
- the build profile and the `REXL3_CUDA_ARCH` setting.

A node that differs in one of these builds a different file. That is a real difference in the build, not noise, and
the fix is to align the node or to copy the binaries from a node that is aligned.

## How it is done

Three things used to tie a binary to the place and time it was built. Each is now fixed as an input before the
compiler runs; nothing is patched afterwards.

| What used to vary | Where it ended up in the file | What fixes it |
| --- | --- | --- |
| The checkout's path and Cargo's home, in Rust code | panic locations and debug information (`debug = 1`) | `.cargo/rustc-remap`, the `rustc-wrapper` in `.cargo/config.toml`, maps both to `/rexl3` and `/cargo`. It is a small Python program, so building needs `python3` on the path |
| The path of Cargo's home, in the C code of a dependency | that code's debug information | `CFLAGS = "-g0"` in `.cargo/config.toml` (`[env]`): no debug information for dependency C code |
| A kernel source's path and modification time, and nvcc's process id | the kernel's name when it lives in an anonymous namespace, the device code's line tables (`-lineinfo`), an ELF file symbol | `crates/cuda/kernel_build.rs`: nvcc compiles a copy of `csrc/` staged at `/tmp/rexl3-kernel-src-<crate>-<hash of the sources>` with one fixed modification time, and keeps its intermediate files in a directory of ours |

No nvcc flag in `NVCC_FLAGS` changed, so the kernel hash and the generated device code are what they were;
`-lineinfo` stays, so profiles still show source lines.

## Limits

- **Your own `RUSTC_WRAPPER` or `CFLAGS` win over the repository's.** A build with `RUSTC_WRAPPER=sccache`, for
  example, works and is not reproducible across paths. Do not build release binaries that way.
- **The staging directory is per user.** If `/tmp/rexl3-kernel-src-...` for these sources already belongs to another
  user, or cannot be created, the build prints a warning, compiles the kernels in place and is not reproducible across
  paths. Two users building the same commit on one host at the same time is the case to avoid.
- **A target directory from an engine before v0.1.3** holds dependency objects with the old paths in them. Start the
  first release build of v0.1.3 or later from an empty target directory.
- **Debuggers** see `/rexl3/...` and `/cargo/...` in the debug information: in gdb, `set substitute-path /rexl3 <checkout>`.
- The staged copies under `/tmp` are small (a few megabytes per distinct set of kernel sources) and are left in place
  for the next build; the system's own cleaning of `/tmp` removes them.

## The check that guards it

`crates/cuda/tests/kernel_build_identity.rs` compiles kernel sources from two checkouts at different paths with
different file times and requires identical object bytes, and a second test shows that the same sources compiled in
place differ. Both need nvcc, so they run on a host with the CUDA toolkit; they are ignored by name in a build
without it.
