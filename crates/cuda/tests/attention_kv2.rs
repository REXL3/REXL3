//! The cases of tests/attention.rs on the 2-KV-head build of the kernels (csrc/attention_kv2.cu: 24 query heads in groups of 12,
//! Qwen3.8-Flash-Next): CPU f64 references, bit-exact row invariance, verify window == serial, chunk invariance, tree windows,
//! and the golden hashes. GPU tests skip when there is no device.
#![cfg(not(gb_no_cuda))]

mod common;

const SHAPE: gb_cuda::attention::AttnShape = gb_cuda::attention::AttnShape::Q24Kv2;
include!("attention/cases.rs");
