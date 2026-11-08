//! The registered model families: one module each, one line below.
//!
//! Adding a family is a module here (its config parser, layer recipe, memory model, chat format, drafter offer and `build`),
//! one line in [`FAMILIES`], and its own fixtures and tests. The engine core (`gb_engine`, `gb_sched`, `gb_sampling`, `gb_spec`,
//! the server) never names a family, and `crate::family::Registry` refuses a list in which two entries claim one name.

use crate::family::ModelFamily;

pub mod bailing_moe_v3;
pub mod deepseek_v41;
pub mod glm;
pub mod mimo;
pub mod minicpmv4_7;
pub mod nemotron_h_omni;
pub mod qwen;
pub mod qwen_flash_next;

/// Every family this release can probe. Qwen first (the one that runs), then the families being brought up.
pub static FAMILIES: &[&'static dyn ModelFamily] = &[
    &qwen::QwenFamily,
    &glm::GlmFamily,
    &qwen_flash_next::FlashNextFamily,
    &mimo::MiMoFamily,
    &nemotron_h_omni::NemotronOmniFamily,
    &minicpmv4_7::MiniCpmFamily,
    &deepseek_v41::DeepseekV41Family,
    &bailing_moe_v3::BailingFamily,
];
