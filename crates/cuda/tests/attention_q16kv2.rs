//! The cases of tests/attention.rs on the 16-query-head, 2-KV-head build of the kernels (csrc/attention_q16kv2.cu: groups of 8, head
//! dim 256: the full-attention layers of the Qwen3.5/3.6-35B-A3B text backbone of MiniCPM-V-4.7): CPU f64 references, bit-exact row
//! invariance, verify window == serial, the draft-style block of rows sharing one bound, chunk invariance, tree windows, the prefill
//! variants (2 and 3 do not exist at groups of 8 and are refused) and the golden hashes. GPU tests skip when there is no device.
#![cfg(not(gb_no_cuda))]

mod common;

const SHAPE: gb_cuda::attention::AttnShape = gb_cuda::attention::AttnShape::Q16Kv2;
include!("attention/cases.rs");
