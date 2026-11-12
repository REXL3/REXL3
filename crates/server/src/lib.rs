//! OpenAI-compatible serving for the engine.
#![deny(unsafe_code)]

pub mod api;
pub mod budget;
pub mod build_info;
pub mod chat_format;
mod decision_casefold;
pub mod decisions;
pub mod field_policy;
pub mod flow;
pub mod formats;
pub mod graph_metrics;
pub mod http;
pub mod latency;
pub mod listen;
pub mod observability;
pub mod prefix_metrics;
pub mod preparation;
pub mod reasoning;
pub mod reasoning_caps;
pub mod reasoning_controls;
pub mod runtime;
pub mod spec_metrics;
pub mod template;
pub mod tokenizer;
pub mod tools;
pub mod tp_metrics;
pub mod vision;
pub mod vision_ingress;
