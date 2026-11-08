//! Source guard for the DSpark drafter's lazy device writes (`crates/cuda/src/dsv4_mtp_gpu.rs`).
//!
//! The GPU MTP tests run the drafter on a non-blocking stream. A legacy-stream memset (`DevBuf::zeroed*`) is not ordered
//! with work on such a stream, so the two buffers the drafter allocates lazily must not be zeroed that way: the round
//! params are written whole by a stream-ordered copy, and the EXL3 decode workspace needs no initialization.

const SRC: &str = include_str!("../src/dsv4_mtp_gpu.rs");

fn assignments(prefix: &str) -> Vec<&'static str> {
    SRC.lines().map(str::trim).filter(|l| l.starts_with(prefix)).collect()
}

#[test]
fn the_lazily_allocated_round_params_and_workspace_are_not_zeroed_on_the_legacy_stream() {
    let params = assignments("self.round_params_dev = Some(");
    let workspace = assignments("*ws = DevBuf::");
    assert_eq!(params.len(), 1, "expected one round-params allocation, found {params:?}");
    assert_eq!(workspace.len(), 1, "expected one workspace growth, found {workspace:?}");
    for line in params.iter().chain(&workspace) {
        assert!(!line.contains("zeroed"), "a legacy-stream memset is back on a lazily allocated buffer: {line}");
    }
}
