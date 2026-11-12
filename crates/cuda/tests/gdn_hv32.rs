//! The cases of tests/gdn.rs on the 32-value-head build of the kernels (csrc/gdn_hv32.cu: 16 key x 32 value heads of 128, 8192 conv
//! channels, value head h reads key head h / 2: the Qwen3.5/3.6-35B-A3B backbone of MiniCPM-V-4.7): the f64 reference of every stage,
//! window == serial bit for bit, batching and row invariance, deferred commit, tree windows, prefill chunk invariance, and the golden
//! hashes. GPU tests skip when there is no device.
#![cfg(not(gb_no_cuda))]

mod common;

const SHAPE: gb_cuda::gdn::GdnShape = gb_cuda::gdn::GdnShape::Hv32;
include!("gdn/cases.rs");
