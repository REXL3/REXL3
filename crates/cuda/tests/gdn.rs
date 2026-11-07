//! Gated delta net kernels: f64 CPU reference, bit-exact replay/row-invariance gates, ExLlamaV3 dumps, benchmarks.
//!
//! The 48-value-head build of the kernels (Qwen3.8-27B); tests/gdn_hv32.rs runs the same cases on the 32-head build.
#![cfg(not(gb_no_cuda))]

mod common;

const SHAPE: gb_cuda::gdn::GdnShape = gb_cuda::gdn::GdnShape::Hv48;
include!("gdn/cases.rs");
