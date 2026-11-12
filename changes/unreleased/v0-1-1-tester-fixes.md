## Bug fixes

- Engine build: `crates/cuda/build.rs` runs at most Cargo's job count of `nvcc` processes instead of one per stale kernel file, and drops the AArch64-only host flag when the target is another architecture (#83, reported by @Blackwellboy).
- Engine source archive: it now ships `rustfmt.toml` and `tools/switch_registry.json`, and workspace clippy passes, so the archive passes its own `cargo fmt --check`, clippy and unit tests (#84, reported by @Blackwellboy; the exporter half landed in #102).
- `--allow-unvalidated-pack` is required for every family, including unlisted DeepSeek-V4.1 and Qwen3.5-MoE packs (#86, reported by @Blackwellboy).
- DeepSeek-V4.1: Engram table mappings are unmapped at the address and length `mmap` returned (#90); a `prompt_logprobs` readout over 64 tokens and the n-gram splice on a short context no longer panic (#91); the device total counts the shared RoPE tables once (#97); three comments and the boot-margin note match the code (#98). Reported by @Blackwellboy.
- `rexl3-cluster`: `--all` on `up`, `down`, `supervise`, `logs`, `exec` and `env` is refused by name, and a short top-level `memory.min_free_gib` list is refused instead of checking the missing ranks against 0 GiB (#92, reported by @Blackwellboy).
- `/v1/decisions`: an explicit `null` description or yes/no text reads as absent (#93, reported by @Blackwellboy).
- Speculation: the draft cost model prices 9- to 15-row blocks correctly (#96), and the arbiter's copy probe can no longer take a round to `max_rows + 1` (#100). Reported by @Blackwellboy.
- Server: a panic on a tensor-parallel group's engine thread leaks the engine like every other fatal exit, instead of dropping it before `die()` answers (#101, reported by @Blackwellboy).
- Tests: the vision mutation tests pass in a default debug build (#99, reported by @Blackwellboy).
