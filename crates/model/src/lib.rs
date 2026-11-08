//! Model families on the GB10 (Qwen3.8 serves today, GLM-5.3-Flash is being added): configuration, memory model, weights, the
//! forward pass and drafting.
pub mod backend;
pub mod bootdiag;
pub mod config;
pub mod costs;
pub mod costs_grid;
pub mod costs_grid_tree;
pub mod draft;
pub mod error;
pub mod exit;
pub mod families;
pub mod family;
pub mod feature_budget;
pub mod fingerprint;
pub mod forward;
pub mod hostmem;
pub mod hosttrace;
pub mod intervention;
pub mod kvtables;
pub mod loadctx;
pub mod lockstep;
pub mod memory;
pub mod mrope;
pub mod mrope_canvas;
pub mod pack;
pub mod plan;
pub mod prefill_p2;
pub mod rows;
pub mod scratch;
pub mod serve;
pub mod shapes;
pub mod shard;
pub mod snapshot;
pub mod state;
pub mod tp_status;
pub mod verify;
pub mod vision;
pub mod weights;
pub mod window_graph;
pub mod wq;

/// The loader's phase timers, exported on `/metrics` by the server.
pub use gb_exl3::loadstats;
