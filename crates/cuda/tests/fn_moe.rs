//! Synthetic packed EXL3 MoE for CoopMK and FrostMoE, with CPU packed-weight reference: Qwen3.8-Flash-Next's layer shape (hidden 2560,
//! expert intermediate 640, 2 routed experts, top-1, every matrix 2 bits). tests/minicpm_moe.rs runs the same cases at MiniCPM-V-4.7's.
mod common;
const H: usize = 2560;
const I: usize = 640;
const NE: usize = 2;
const TK: usize = 1;
const KC: usize = 10;
const SEED_STRIDE: usize = 10;
fn bits_of(_matrix: usize, _expert: usize) -> usize {
    2
}
include!("fn_moe/cases.rs");
