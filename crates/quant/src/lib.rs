//! REXL3's EXL3 quantizer (`rexl3-quant`). Produces `-REXL3-SAGE` packs.
//!
//! Stages, in pipeline order: `source` (read the model), `calib` (calibration sets), `teacher` (reference forward),
//! `capture` (input statistics), `sense` and `predict` (per-tensor loss model), `alloc` (bit allocation), `core` and
//! `encode` (per-tensor quantization), `polish`, `eval`, `pack` (the written pack) and `auto` (the guided pipeline).
//! `exl3_trellis` is the tile encoder (`csrc/exl3_trellis.cu`) and `gpu` holds the cuBLAS / cuSOLVER bindings this crate
//! uses; nothing here is linked into the serving binary.

#![cfg_attr(gb_quant_no_cuda, allow(dead_code, unused_imports))]

pub mod alloc;
pub mod auto;
pub mod bank;
pub mod bench;
pub mod calib;
pub mod capture;
pub mod core;
pub mod encode;
pub mod eval;
pub mod exl3_trellis;
pub mod gpu;
pub mod pack;
pub mod polish;
pub mod predict;
pub mod sense;
pub mod source;
pub mod teacher;

/// The pack name suffix of this quantizer's output.
pub const PACK_SUFFIX: &str = "-REXL3-SAGE";
