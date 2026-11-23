//! The shared attention cases on the config-selected 16-query-head, 4-KV-head Qwen kernel. The 27B Q24/Kv4
//! launcher and arithmetic are not recompiled through this shim.
#![cfg(not(gb_no_cuda))]

mod common;

const SHAPE: gb_cuda::attention::AttnShape = gb_cuda::attention::AttnShape::Q16Kv4;
include!("attention/cases.rs");
