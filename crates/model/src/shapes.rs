//! Startup checks of what a pack's configuration says against what the kernels were compiled for (release plan R11, R13).
//!
//! The config selects among the attention and GDN kernels compiled into the engine (including 24/4 and 16/4 attention and 48- or
//! 32-value-head GDN). DFlash2 keeps its fixed layer/head geometry while its target width, FFN width, vocabulary and tap indices come
//! from the draft config. Anything without a compiled implementation fails at startup, before weights are read, with the offending
//! field and geometry named.

use crate::config::QwenConfig;
use crate::error::ModelError;
use serde_json::Value;
use std::path::Path;

/// The oldest CUDA version the driver may report (`cudaDriverGetVersion`): the kernels are built with nvcc 13.0 for sm_121.
pub const MIN_DRIVER_CUDA: (u32, u32) = (13, 0);

/// Collects mismatches so one startup error lists every one.
#[derive(Default)]
struct Mismatches(Vec<String>);

impl Mismatches {
    fn eq<T: PartialEq + std::fmt::Display>(&mut self, what: &str, has: T, compiled: T) {
        if has != compiled {
            self.0.push(format!("{what} is {has}, the kernels are built for {compiled}"));
        }
    }
    fn finish(self, whose: &str) -> Result<(), String> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(format!("{whose} does not match the compiled kernels: {}", self.0.join("; ")))
        }
    }
}

/// Select the target's attention and GDN kernels from its config, and reject dimensions the shared glue kernels do not implement.
/// Qwen3.8-27B therefore selects its original Q24/Kv4 and Hv48 kernels; another supported geometry needs no source edit.
pub fn check_target(c: &QwenConfig) -> Result<(), String> {
    c.attn_shape()?;
    c.gdn_shape()?;
    if c.head_dim != gb_cuda::attention::HEAD_DIM {
        return Err(format!(
            "unsupported attention geometry: head_dim is {}, the compiled Qwen attention kernels require {}",
            c.head_dim,
            gb_cuda::attention::HEAD_DIM
        ));
    }
    if c.conv_kernel != gb_cuda::gdn::CONV_TAPS {
        return Err(format!(
            "unsupported GDN geometry: linear_conv_kernel_dim is {}, the compiled Qwen GDN kernels require {}",
            c.conv_kernel,
            gb_cuda::gdn::CONV_TAPS
        ));
    }
    if !matches!(c.rotary_dim, 0 | 64 | 128 | 256) || c.rotary_dim > c.head_dim {
        return Err(format!(
            "unsupported attention RoPE geometry: rotary dim {} (head_dim x partial_rotary_factor) is not one of 0, 64, 128, 256 within head_dim {}",
            c.rotary_dim, c.head_dim
        ));
    }
    if c.has_vision {
        if c.rotary_dim != 64 {
            return Err(format!(
                "unsupported vision M-RoPE geometry: rope_parameters rotary dim is {}, the vision kernels require 64",
                c.rotary_dim
            ));
        }
        if c.mrope_section != Some([11, 11, 10]) {
            return Err(format!(
                "unsupported vision M-RoPE geometry: rope_parameters.mrope_section is {:?}, the vision kernels require [11, 11, 10]",
                c.mrope_section
            ));
        }
    }
    Ok(())
}

/// The DFlash2 draft's `config.json`, the fields the kernels depend on.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub rms_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: usize,
    pub num_target_layers: usize,
    pub block_size: usize,
    pub conv_group: usize,
    pub conv_kernel: usize,
    pub mask_token: u64,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub target_layer_ids: Vec<usize>,
    pub config_causal: bool,
}

impl DraftConfig {
    pub fn from_json(root: &Value) -> Result<Self, String> {
        let uz = |v: &Value, k: &str| -> Result<usize, String> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| format!("the draft's config.json: missing or invalid `{k}`"))
        };
        let fz = |v: &Value, k: &str| -> Result<f64, String> {
            v.get(k).and_then(|x| x.as_f64()).ok_or_else(|| format!("the draft's config.json: missing or invalid `{k}`"))
        };
        let d = root
            .get("dflash_config")
            .ok_or("the draft's config.json: missing `dflash_config` (is this a DFlash2 draft pack?)")?;
        let ids = d
            .get("target_layer_ids")
            .and_then(|x| x.as_array())
            .ok_or("the draft's config.json: missing `dflash_config.target_layer_ids`")?
            .iter()
            .map(|x| x.as_u64().map(|x| x as usize).ok_or("the draft's config.json: `target_layer_ids` must be integers"))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            hidden: uz(root, "hidden_size")?,
            layers: uz(root, "num_hidden_layers")?,
            heads: uz(root, "num_attention_heads")?,
            kv_heads: uz(root, "num_key_value_heads")?,
            head_dim: uz(root, "head_dim")?,
            intermediate: uz(root, "intermediate_size")?,
            vocab: uz(root, "vocab_size")?,
            rms_eps: fz(root, "rms_norm_eps")?,
            rope_theta: root
                .get("rope_parameters")
                .and_then(|r| r.get("rope_theta"))
                .and_then(|x| x.as_f64())
                .or_else(|| root.get("rope_theta").and_then(|x| x.as_f64()))
                .ok_or("the draft's config.json: missing `rope_parameters.rope_theta`")?,
            sliding_window: uz(root, "sliding_window")?,
            num_target_layers: uz(root, "num_target_layers")?,
            block_size: uz(d, "block_size")?,
            conv_group: uz(d, "conv_group_size")?,
            conv_kernel: uz(d, "conv_kernel_size")?,
            mask_token: uz(d, "mask_token_id")? as u64,
            selector_rank: uz(d, "selector_rank")?,
            selector_top_k: uz(d, "selector_top_k")?,
            target_layer_ids: ids,
            config_causal: root
                .get("is_causal")
                .and_then(|x| x.as_bool())
                .ok_or("the draft's config.json: missing or invalid `is_causal`")?,
        })
    }

    /// The validated draft geometry. Target hidden width, vocabulary, FFN width, mask token and tap indices all come from this file.
    pub fn geometry(&self) -> Result<crate::draft::DraftGeom, String> {
        let tap_layers: [usize; crate::draft::TAPS] = self.target_layer_ids.clone().try_into().map_err(|ids: Vec<usize>| {
            format!(
                "the draft's config.json: dflash_config.target_layer_ids has {} entries, the DFlash2 fc takes {}",
                ids.len(),
                crate::draft::TAPS
            )
        })?;
        let geom = crate::draft::DraftGeom {
            name: "qwen-config",
            vocab: self.vocab,
            tap_layers,
            config_causal: self.config_causal,
            hidden: self.hidden,
            intermediate: self.intermediate,
            mask_token: self.mask_token as i32,
            eps: self.rms_eps as f32,
            rope_theta: self.rope_theta,
            block_causal: true,
        };
        for known in crate::draft::DraftGeom::KNOWN {
            let mut candidate = geom;
            candidate.name = known.name;
            candidate.block_causal = known.block_causal;
            if candidate == *known {
                return Ok(*known);
            }
        }
        // An unknown draft has no published exception to the causal engine mask.
        if geom.config_causal != geom.block_causal {
            return Err(format!(
                "the draft's config.json: is_causal is {}, but the generic DFlash2 engine mask is causal; only a matched DraftGeom::KNOWN geometry may override it",
                geom.config_causal
            ));
        }
        Ok(geom)
    }

    /// The draft against the kernels and against the target it drafts for.
    pub fn check(&self, target: &QwenConfig) -> Result<(), String> {
        use crate::draft as d;
        let geom = self.geometry()?;
        let mut m = Mismatches::default();
        m.eq("num_hidden_layers", self.layers, d::LAYERS);
        m.eq("num_attention_heads", self.heads, d::Q_HEADS);
        m.eq("num_key_value_heads", self.kv_heads, d::KV_HEADS);
        m.eq("head_dim", self.head_dim, d::HEAD_DIM);
        m.eq("sliding_window", self.sliding_window, gb_cuda::dflash2::DraftCtxLayout::WINDOW);
        // the pack's block is what the checkpoint was TRAINED with (8): the runtime knob `REXL3_DRAFT_BLOCK` (8..=16) runs the
        // block longer, it does not retune the pack, so a pack trained with another block is still a different draft
        if self.block_size != d::BLOCK {
            m.0.push(format!(
                "dflash_config.block_size is {}, the draft was trained with a block of {} (REXL3_DRAFT_BLOCK may extend the block at runtime; the pack itself is the trained one)",
                self.block_size, d::BLOCK
            ));
        }
        m.eq("dflash_config.conv_group_size", self.conv_group, d::CONV_GROUP);
        m.eq("dflash_config.conv_kernel_size", self.conv_kernel, 2);
        m.eq("dflash_config.selector_rank", self.selector_rank, d::SELECTOR_RANK);
        m.eq("dflash_config.selector_top_k", self.selector_top_k, d::SELECTOR_TOP_K);
        if let Some(&l) = geom.tap_layers.iter().find(|&&l| l >= target.layers.len()) {
            m.0.push(format!("dflash_config.target_layer_ids contains layer {l}, the target has {} layers", target.layers.len()));
        }
        if geom.tap_layers.windows(2).any(|w| w[0] >= w[1]) {
            m.0.push(format!(
                "dflash_config.target_layer_ids {:?} must be strictly increasing with no duplicates",
                geom.tap_layers
            ));
        }
        // the draft must belong to this target
        if self.hidden != target.hidden {
            m.0.push(format!("hidden_size {} differs from the target's {}", self.hidden, target.hidden));
        }
        if self.vocab != target.vocab {
            m.0.push(format!("vocab_size {} differs from the target's {}", self.vocab, target.vocab));
        }
        if self.num_target_layers != target.layers.len() {
            m.0.push(format!(
                "num_target_layers {} differs from the target's {} layers",
                self.num_target_layers,
                target.layers.len()
            ));
        }
        if self.mask_token > i32::MAX as u64 {
            m.0.push(format!("dflash_config.mask_token_id {} exceeds the kernel's signed 32-bit token id", self.mask_token));
        } else if self.mask_token as usize >= target.vocab {
            m.0.push(format!("the mask token {} is outside the target's vocabulary of {}", self.mask_token, target.vocab));
        }
        m.finish("the draft's config.json")
    }
}

/// Read and validate `dir/config.json`, returning the draft geometry the loader and target forward must use.
pub fn read_draft_dir(dir: &Path, target: &QwenConfig) -> Result<DraftConfig, ModelError> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| ModelError::Config(format!("{}: {e}", path.display())))?;
    let root: Value = serde_json::from_str(&text).map_err(|e| ModelError::Config(format!("{}: {e}", path.display())))?;
    let draft = DraftConfig::from_json(&root).map_err(ModelError::Config)?;
    draft.check(target).map_err(ModelError::Config)?;
    Ok(draft)
}

/// Compatibility helper for callers that only need validation.
pub fn check_draft_dir(dir: &Path, target: &QwenConfig) -> Result<(), ModelError> {
    read_draft_dir(dir, target).map(|_| ())
}

/// The driver must report CUDA 13.0 or newer. A missing driver is its own message.
pub fn check_driver(v: &gb_cuda::rt::Versions) -> Result<(), String> {
    let Some(d) = v.driver.as_deref() else {
        return Err("no CUDA driver is available (cudaDriverGetVersion failed): the GPU backend needs the NVIDIA driver".into());
    };
    let (maj, min) = d
        .split_once('.')
        .and_then(|(a, b)| Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?)))
        .ok_or_else(|| format!("cannot read the CUDA driver version `{d}`"))?;
    if (maj, min) < MIN_DRIVER_CUDA {
        return Err(format!(
            "the NVIDIA driver supports CUDA {maj}.{min}, but this build needs CUDA {}.{} or newer (it was built with nvcc {} for {}): update the driver",
            MIN_DRIVER_CUDA.0, MIN_DRIVER_CUDA.1, v.nvcc, v.arch
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn target() -> QwenConfig {
        QwenConfig::from_json(&serde_json::from_str(include_str!("../../../testdata/qwen38/config.json")).unwrap()).unwrap()
    }

    fn draft_json() -> Value {
        serde_json::from_str(include_str!("../../../testdata/qwen38/draft_config.json")).unwrap()
    }

    #[test]
    fn the_real_target_and_draft_configs_match_the_kernels() {
        check_target(&target()).unwrap();
        DraftConfig::from_json(&draft_json()).unwrap().check(&target()).unwrap();
    }

    #[test]
    fn qwen38_draft_keeps_its_known_geometry_and_unknown_masks_fail_closed() {
        let draft = DraftConfig::from_json(&draft_json()).unwrap();
        assert_eq!(draft.geometry().unwrap(), crate::draft::DraftGeom::QWEN38);
        assert_eq!(draft.geometry().unwrap().name, "qwen38");

        let mut unknown = draft_json();
        unknown["hidden_size"] = json!(4096);
        let e = DraftConfig::from_json(&unknown).unwrap().geometry().unwrap_err();
        assert!(e.contains("is_causal") && e.contains("engine mask"), "{e}");
    }

    #[test]
    fn vision_mrope_must_match_the_compiled_sections() {
        let mut root: Value = serde_json::from_str(include_str!("../../../testdata/qwen38/config.json")).unwrap();
        root["text_config"]["rope_parameters"]["partial_rotary_factor"] = json!(0.5);
        let e = check_target(&QwenConfig::from_json(&root).unwrap()).unwrap_err();
        assert!(e.contains("vision M-RoPE") && e.contains("rotary dim is 128"), "{e}");

        let mut root: Value = serde_json::from_str(include_str!("../../../testdata/qwen38/config.json")).unwrap();
        root["text_config"]["rope_parameters"]["mrope_section"] = json!([11, 10, 11]);
        let e = check_target(&QwenConfig::from_json(&root).unwrap()).unwrap_err();
        assert!(e.contains("vision M-RoPE") && e.contains("mrope_section") && e.contains("[11, 11, 10]"), "{e}");
    }

    #[test]
    fn unsupported_target_geometries_are_refused_by_field() {
        for (field, value, needle) in [
            ("num_attention_heads", json!(32), "unsupported attention geometry"),
            ("num_key_value_heads", json!(8), "num_key_value_heads=8"),
            ("head_dim", json!(128), "head_dim is 128"),
            ("linear_num_key_heads", json!(8), "unsupported GDN geometry"),
            ("linear_key_head_dim", json!(64), "linear_key_head_dim=64"),
            ("linear_conv_kernel_dim", json!(5), "linear_conv_kernel_dim is 5"),
        ] {
            let mut root: Value = serde_json::from_str(include_str!("../../../testdata/qwen38/config.json")).unwrap();
            root["text_config"][field] = value;
            let cfg = QwenConfig::from_json(&root).unwrap_or_else(|e| panic!("{field}: the parser refuses it first: {e}"));
            let e = check_target(&cfg).unwrap_err();
            assert!(e.contains(needle), "{field}: {e}");
        }
        // A supported non-27B shape is selected, not rejected as a mismatch with 27B.
        let mut root: Value = serde_json::from_str(include_str!("../../../testdata/qwen38/config.json")).unwrap();
        root["text_config"]["num_attention_heads"] = json!(16);
        root["text_config"]["linear_num_value_heads"] = json!(32);
        let cfg = QwenConfig::from_json(&root).unwrap();
        check_target(&cfg).unwrap();
        assert_eq!(cfg.attn_shape().unwrap(), gb_cuda::attention::AttnShape::Q16Kv4);
        assert_eq!(cfg.gdn_shape().unwrap(), gb_cuda::gdn::GdnShape::Hv32);
    }

    #[test]
    fn a_draft_with_other_shapes_or_for_another_target_is_refused_by_field() {
        for (path, value, needle) in [
            ("hidden_size", json!(4096), "hidden_size 4096 differs from the target's 5120"),
            ("num_hidden_layers", json!(4), "num_hidden_layers is 4, the kernels are built for 5"),
            ("num_attention_heads", json!(16), "num_attention_heads is 16"),
            ("head_dim", json!(64), "head_dim is 64"),
            ("sliding_window", json!(1024), "sliding_window is 1024"),
            ("vocab_size", json!(150000), "vocab_size 150000 differs from the target's 248320"),
            ("num_target_layers", json!(48), "num_target_layers 48 differs"),
        ] {
            let mut j = draft_json();
            j["is_causal"] = json!(true);
            j[path] = value;
            let e = DraftConfig::from_json(&j).unwrap().check(&target()).unwrap_err();
            assert!(e.contains(needle), "{path}: {e}");
        }
        for (path, value, needle) in [
            ("block_size", json!(16), "block_size is 16, the draft was trained with a block of 8"),
            ("selector_rank", json!(128), "selector_rank is 128"),
            ("conv_kernel_size", json!(3), "conv_kernel_size is 3"),
            ("target_layer_ids", json!([4, 19, 33, 47, 64]), "target_layer_ids contains layer 64"),
        ] {
            let mut j = draft_json();
            j["is_causal"] = json!(true);
            j["dflash_config"][path] = value;
            let e = DraftConfig::from_json(&j).unwrap().check(&target()).unwrap_err();
            assert!(e.contains(needle), "{path}: {e}");
        }
        // a config without the DFlash2 section is not a DFlash2 draft
        let mut j = draft_json();
        j.as_object_mut().unwrap().remove("dflash_config");
        assert!(DraftConfig::from_json(&j).unwrap_err().contains("dflash_config"));
    }

    #[test]
    fn a_draft_folder_with_no_readable_config_is_an_error() {
        let dir = std::env::temp_dir().join(format!("rexl3-draft-cfg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(check_draft_dir(&dir, &target()).unwrap_err().to_string().contains("config.json"));
        std::fs::write(dir.join("config.json"), "{ not json").unwrap();
        assert!(check_draft_dir(&dir, &target()).is_err());
        std::fs::write(dir.join("config.json"), include_str!("../../../testdata/qwen38/draft_config.json")).unwrap();
        assert!(check_draft_dir(&dir, &target()).is_ok());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_driver_below_cuda_13_0_is_a_clear_error() {
        let mut v = gb_cuda::rt::versions();
        v.driver = Some("13.0".into());
        assert!(check_driver(&v).is_ok());
        v.driver = Some("13.2".into());
        assert!(check_driver(&v).is_ok());
        v.driver = Some("14.0".into());
        assert!(check_driver(&v).is_ok());
        v.driver = Some("12.9".into());
        let e = check_driver(&v).unwrap_err();
        assert!(e.contains("supports CUDA 12.9") && e.contains("13.0 or newer") && e.contains("update the driver"), "{e}");
        v.driver = None;
        assert!(check_driver(&v).unwrap_err().contains("no CUDA driver"));
        v.driver = Some("garbage".into());
        assert!(check_driver(&v).unwrap_err().contains("cannot read"));
    }
}
