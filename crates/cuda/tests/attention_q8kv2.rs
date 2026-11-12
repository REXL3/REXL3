//! The cases of tests/attention.rs on the 8-query-head, 2-KV-head build of the kernels (csrc/attention_q8kv2.cu: groups of 4, head dim
//! 256: the layers of MiniCPM-V-4.7's DSpark draft): CPU f64 references, bit-exact row invariance (a decode row is one of up to 12 a
//! CTA serves at this group size), verify window == serial, the block of up to 15 rows that share one visibility bound, chunk
//! invariance, tree windows, the prefill variants (2 and 3 do not exist at groups of 4 and are refused) and the golden hashes.
//! GPU tests skip when there is no device.
#![cfg(not(gb_no_cuda))]

mod common;

const SHAPE: gb_cuda::attention::AttnShape = gb_cuda::attention::AttnShape::Q8Kv2;
include!("attention/cases.rs");
