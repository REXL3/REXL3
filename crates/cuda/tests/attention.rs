//! Paged KV cache + attention kernels: CPU f64 references, bit-exact row invariance, chunk invariance, the
//! ExLlamaV3 layer-3 dump, and (ignored) benchmarks. GPU tests skip when there is no device.
//!
//! The 24-query-head, 4-KV-head build of the kernels (Qwen3.8-27B); tests/attention_kv2.rs, tests/attention_q16kv2.rs and
//! tests/attention_q8kv2.rs run the same cases on the other builds.
#![cfg(not(gb_no_cuda))]

mod common;

const SHAPE: gb_cuda::attention::AttnShape = gb_cuda::attention::AttnShape::Q24Kv4;
include!("attention/cases.rs");
