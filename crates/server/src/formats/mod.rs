//! The chat formats the server ships, one module per family. A family is registered by one line in [`BUILTIN`]; see
//! [`crate::chat_format`] for the contract and for how a family is found.

pub mod bailing_moe_v3;
pub mod deepseek_v41;
pub mod glm;
pub mod mimo;
pub mod minicpmv;
pub mod nemotron_omni;
pub mod qwen;

use crate::chat_format::FormatEntry;

/// Every registered family. Adding one: write its module, add `<module>::ENTRY` here.
pub const BUILTIN: &[FormatEntry] =
    &[qwen::ENTRY, glm::ENTRY, mimo::ENTRY, nemotron_omni::ENTRY, minicpmv::ENTRY, deepseek_v41::ENTRY, bailing_moe_v3::ENTRY];
