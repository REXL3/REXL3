//! X126: Generic Qwen geometry comes from the target and draft configs, never a retargeted 27B binary.

use gb_cuda::attention::AttnShape;
use gb_cuda::gdn::GdnShape;
use gb_model::config::QwenConfig;
use gb_model::shapes::{check_target, DraftConfig};
use serde_json::{json, Value};

fn target_27b() -> Value {
    serde_json::from_str(include_str!("../../../testdata/qwen38/config.json")).unwrap()
}

fn draft_27b() -> Value {
    serde_json::from_str(include_str!("../../../testdata/qwen38/draft_config.json")).unwrap()
}

fn target_4b() -> Value {
    let mut root = target_27b();
    let t = &mut root["text_config"];
    t["hidden_size"] = json!(2560);
    t["intermediate_size"] = json!(9216);
    t["num_hidden_layers"] = json!(32);
    t["num_attention_heads"] = json!(16);
    t["linear_num_value_heads"] = json!(32);
    t["layer_types"] = json!((0..32).map(|i| if i % 4 == 3 { "full_attention" } else { "linear_attention" }).collect::<Vec<_>>());
    root.as_object_mut().unwrap().remove("vision_config");
    root
}

#[test]
fn qwen27b_selects_the_unchanged_default_kernels_and_prep() {
    let cfg = QwenConfig::from_json(&target_27b()).unwrap();
    check_target(&cfg).unwrap();
    assert_eq!(cfg.attn_shape().unwrap(), AttnShape::Q24Kv4);
    assert_eq!(cfg.gdn_shape().unwrap(), GdnShape::Hv48);
    assert_eq!(cfg.attn_prep(), gb_cuda::glue::AttnPrepCfg::QWEN_TARGET);
    assert_eq!(cfg.conv_channels(), 10240);
    let draft = DraftConfig::from_json(&draft_27b()).unwrap();
    draft.check(&cfg).unwrap();
    assert_eq!(draft.geometry().unwrap().tap_layers, gb_model::forward::TAP_LAYERS);
}

#[test]
fn a_second_geometry_and_its_draft_plan_from_config_alone() {
    let cfg = QwenConfig::from_json(&target_4b()).unwrap();
    check_target(&cfg).unwrap();
    assert_eq!(cfg.attn_shape().unwrap(), AttnShape::Q16Kv4);
    assert_eq!(cfg.gdn_shape().unwrap(), GdnShape::Hv32);
    assert_eq!(cfg.conv_channels(), 8192);
    assert_eq!((cfg.full_layers(), cfg.linear_layers()), (8, 24));
    assert_eq!(cfg.attn_prep().heads, 16);
    gb_model::families::qwen::memory_profile(&cfg, &gb_model::memory::MemoryOptions::serving_defaults()).unwrap();

    let mut j = draft_27b();
    j["is_causal"] = json!(true);
    j["hidden_size"] = json!(2560);
    j["intermediate_size"] = json!(9216);
    j["num_target_layers"] = json!(32);
    j["dflash_config"]["target_layer_ids"] = json!([2, 8, 15, 23, 30]);
    let draft = DraftConfig::from_json(&j).unwrap();
    draft.check(&cfg).unwrap();
    let geom = draft.geometry().unwrap();
    assert_eq!(geom.tap_layers, [2, 8, 15, 23, 30]);
    assert_eq!((geom.hidden, geom.intermediate), (2560, 9216));
    assert!(gb_model::draft::config_mismatches(&j, &geom).is_empty());
}

#[test]
fn unsupported_geometry_and_bad_taps_fail_closed_with_named_fields() {
    let mut bad = target_4b();
    bad["text_config"]["num_attention_heads"] = json!(32);
    let e = check_target(&QwenConfig::from_json(&bad).unwrap()).unwrap_err();
    assert!(e.contains("unsupported attention geometry") && e.contains("num_attention_heads=32"), "{e}");

    let mut bad = target_4b();
    bad["text_config"]["linear_num_value_heads"] = json!(80);
    let e = check_target(&QwenConfig::from_json(&bad).unwrap()).unwrap_err();
    assert!(e.contains("unsupported GDN geometry") && e.contains("linear_num_value_heads=80"), "{e}");

    let cfg = QwenConfig::from_json(&target_4b()).unwrap();
    let mut j = draft_27b();
    j["is_causal"] = json!(true);
    j["hidden_size"] = json!(2560);
    j["intermediate_size"] = json!(9216);
    j["num_target_layers"] = json!(32);
    j["dflash_config"]["target_layer_ids"] = json!([2, 8, 15, 23, 32]);
    let e = DraftConfig::from_json(&j).unwrap().check(&cfg).unwrap_err();
    assert!(e.contains("target_layer_ids") && e.contains("layer 32") && e.contains("32 layers"), "{e}");

    j["dflash_config"]["target_layer_ids"] = json!([2, 8, 30]);
    let e = DraftConfig::from_json(&j).unwrap().check(&cfg).unwrap_err();
    assert!(e.contains("target_layer_ids") && e.contains("3 entries"), "{e}");
}

#[cfg(not(rexl3_no_cuda))]
mod cuda {
    use super::{json, target_4b, Value};
    use gb_exl3::f16::f64_to_f16;
    use gb_exl3::{Bits, MARKER_MUL1};
    use std::path::{Path, PathBuf};

    struct Tensor {
        name: String,
        dtype: &'static str,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    fn tensor(name: String, dtype: &'static str, shape: &[usize], data: Vec<u8>) -> Tensor {
        Tensor { name, dtype, shape: shape.to_vec(), data }
    }

    fn le16(values: impl IntoIterator<Item = u16>) -> Vec<u8> {
        values.into_iter().flat_map(u16::to_le_bytes).collect()
    }

    fn le32(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
        values.into_iter().flat_map(f32::to_le_bytes).collect()
    }

    fn safetensors(tensors: &[Tensor]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        for t in tensors {
            let lo = data.len();
            data.extend_from_slice(&t.data);
            header.insert(t.name.clone(), json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [lo, data.len()]}));
        }
        let h = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
        bytes.extend(h);
        bytes.extend(data);
        bytes
    }

    fn write_f32_pack(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let tensors = [
            ("model.language_model.layers.0.linear_attn.A_log", 32usize),
            ("model.language_model.layers.0.linear_attn.dt_bias", 32usize),
            ("model.language_model.layers.0.linear_attn.norm.weight", 128usize),
        ];
        let tensors: Vec<_> = tensors
            .into_iter()
            .map(|(name, n)| tensor(name.into(), "F32", &[n], le32((0..n).map(|i| (i + 1) as f32 * 0.25))))
            .collect();
        std::fs::write(dir.join("model.safetensors"), safetensors(&tensors)).unwrap();
    }

    fn exl3(prefix: String, k: usize, n: usize, seed: u64) -> Vec<Tensor> {
        assert_eq!((k % 16, n % 16), (0, 0));
        let bits = Bits::from_twice(8).unwrap(); // four-bit Mul1, the served pack's format
        let words = (k / 16) * (n / 16) * bits.tile_words();
        let mut state = seed;
        let trellis = (0..words).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 32) as u16
        });
        let scale = f64_to_f16(0.01);
        vec![
            tensor(format!("{prefix}.trellis"), "I16", &[k / 16, n / 16, bits.tile_words()], le16(trellis)),
            tensor(format!("{prefix}.suh"), "F16", &[k], le16(std::iter::repeat_n(scale, k))),
            tensor(format!("{prefix}.svh"), "F16", &[n], le16(std::iter::repeat_n(scale, n))),
            tensor(format!("{prefix}.mul1"), "I32", &[], MARKER_MUL1.to_le_bytes().to_vec()),
        ]
    }

    fn plain16(tensors: &mut Vec<Tensor>, name: String, dtype: &'static str, shape: &[usize], value: u16) {
        let n = shape.iter().product();
        tensors.push(tensor(name, dtype, shape, le16(std::iter::repeat_n(value, n))));
    }

    /// A two-layer Q16/Kv4 + Hv32 pack: one recurrent layer, one full-attention layer, and a small vocabulary. All
    /// dimensions other than the deliberately small layer count, FFN width and vocabulary are the second geometry's.
    fn write_second_geometry_pack(dir: &Path) {
        let mut root = target_4b();
        let t = &mut root["text_config"];
        t["num_hidden_layers"] = json!(2);
        t["intermediate_size"] = json!(1024);
        t["vocab_size"] = json!(256);
        t["layer_types"] = json!(["linear_attention", "full_attention"]);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&root).unwrap()).unwrap();

        let (d, inter, vocab) = (2560usize, 1024usize, 256usize);
        let (heads, kv_heads, head_dim) = (16usize, 4usize, 256usize);
        let (hv, value_dim, conv) = (32usize, 128usize, 8192usize);
        let z = hv * value_dim;
        let p = "model.language_model.";
        let mut ts = Vec::new();
        // Nonzero, finite bf16 embeddings; norms are stored as gamma-1, so zero loads as the identity gamma.
        plain16(&mut ts, format!("{p}embed_tokens.weight"), "BF16", &[vocab, d], 0x3b80);
        for layer in 0..2 {
            let l = format!("{p}layers.{layer}.");
            plain16(&mut ts, format!("{l}input_layernorm.weight"), "BF16", &[d], 0);
            plain16(&mut ts, format!("{l}post_attention_layernorm.weight"), "BF16", &[d], 0);
            for (name, k, n) in [("gate_proj", d, inter), ("up_proj", d, inter), ("down_proj", inter, d)] {
                ts.extend(exl3(format!("{l}mlp.{name}"), k, n, 100 + layer * 20 + ts.len() as u64));
            }
            if layer == 0 {
                let a = format!("{l}linear_attn.");
                for (name, k, n) in [("in_proj_qkv", d, conv), ("in_proj_z", d, z), ("out_proj", z, d)] {
                    ts.extend(exl3(format!("{a}{name}"), k, n, 200 + ts.len() as u64));
                }
                plain16(&mut ts, format!("{a}in_proj_a.weight"), "F16", &[hv, d], 0x2c00);
                plain16(&mut ts, format!("{a}in_proj_b.weight"), "F16", &[hv, d], 0x2c00);
                plain16(&mut ts, format!("{a}conv1d.weight"), "BF16", &[conv, 1, 4], 0x3b80);
                ts.push(tensor(format!("{a}A_log"), "F32", &[hv], le32(std::iter::repeat_n(0.25, hv))));
                ts.push(tensor(format!("{a}dt_bias"), "F32", &[hv], le32(std::iter::repeat_n(0.0, hv))));
                ts.push(tensor(format!("{a}norm.weight"), "F32", &[value_dim], le32(std::iter::repeat_n(1.0, value_dim))));
            } else {
                let a = format!("{l}self_attn.");
                let qdim = heads * head_dim;
                let kvdim = kv_heads * head_dim;
                for (name, k, n) in [("q_proj", d, 2 * qdim), ("k_proj", d, kvdim), ("v_proj", d, kvdim), ("o_proj", qdim, d)] {
                    ts.extend(exl3(format!("{a}{name}"), k, n, 300 + ts.len() as u64));
                }
                plain16(&mut ts, format!("{a}q_norm.weight"), "BF16", &[head_dim], 0);
                plain16(&mut ts, format!("{a}k_norm.weight"), "BF16", &[head_dim], 0);
            }
        }
        plain16(&mut ts, format!("{p}norm.weight"), "BF16", &[d], 0);
        ts.extend(exl3("lm_head".into(), d, vocab, 999));
        std::fs::write(dir.join("model.safetensors"), safetensors(&ts)).unwrap();
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rexl3-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn f32_a_log_and_gdn_norm_load_byte_identically_in_both_loader_modes() {
        assert!(gb_cuda::device_available(), "this CUDA-gated test requires a real GPU");
        use gb_model::loadctx::{LoadCtx, LoadMode};
        use gb_model::pack::Pack;
        let dir = temp_dir("qwen-config-f32");
        write_f32_pack(&dir);
        let pack = Pack::open(&dir).unwrap();
        let names = [
            ("model.language_model.layers.0.linear_attn.A_log", 32usize),
            ("model.language_model.layers.0.linear_attn.dt_bias", 32usize),
            ("model.language_model.layers.0.linear_attn.norm.weight", 128usize),
        ];
        for mode in [LoadMode::Serial, LoadMode::Pipeline] {
            let mut ctx = LoadCtx::new(mode);
            let buffers: Vec<_> = names.iter().map(|&(name, n)| ctx.f32_plain(&pack, name, &[n], false).unwrap()).collect();
            ctx.finish().unwrap();
            for (buf, &(name, n)) in buffers.iter().zip(&names) {
                let got = buf.to_vec::<f32>().unwrap();
                let want: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.25).collect();
                assert_eq!(got, want, "{mode:?}: {name}");
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn assert_same_candidates(got: &gb_engine::RowOutput, want: &gb_engine::RowOutput, what: &str) {
        assert_eq!(got.ids, want.ids, "{what}: greedy/candidate token ids");
        assert_eq!(got.values.len(), want.values.len(), "{what}: candidate count");
        for (i, (&a, &b)) in got.values.iter().zip(&want.values).enumerate() {
            assert!(a.is_finite() && b.is_finite(), "{what}: non-finite candidate {i}: {a} vs {b}");
            assert!((a - b).abs() <= 1e-5, "{what}: candidate {i} logit/value {a} differs from {b}");
        }
    }

    #[test]
    fn second_geometry_plain_and_tree_commits_match_serial_continuation() {
        assert!(gb_cuda::device_available(), "this CUDA-gated test requires a real GPU");
        use gb_engine::{Backend, RowRequest, TreeRequest};
        use gb_model::backend::QwenBackend;
        use gb_model::config::QwenConfig;
        use gb_model::forward::Model;
        use gb_model::pack::Pack;
        use gb_model::weights::Weights;

        let dir = temp_dir("qwen-q16kv4-two-layer");
        write_second_geometry_pack(&dir);
        let cfg = QwenConfig::from_dir(&dir).unwrap();
        assert_eq!(cfg.layers.len(), 2);
        assert_eq!(cfg.attn_shape().unwrap(), gb_cuda::attention::AttnShape::Q16Kv4);
        assert_eq!(cfg.gdn_shape().unwrap(), gb_cuda::gdn::GdnShape::Hv32);
        let pack = Pack::open(&dir).unwrap();
        let weights = Weights::load(&pack, &cfg, |_, _| {}).unwrap();
        let mut model = Model::new(cfg, weights, 1, 16, 256, 128).unwrap();
        model.enable_trees(false).unwrap();
        let mut backend = QwenBackend::new(model, 128);

        // Serial reference: the second step follows commit(0, 1), so it exercises plain flush_pending.
        let prefill = backend.prefill(0, 0, &[1, 2, 3], 4);
        assert!(backend.take_error().is_none());
        assert_eq!(prefill.ids.len(), 4);
        let serial1 =
            backend.step(&[RowRequest { slot: 0, tokens: vec![prefill.ids[0]], first_position: 3 }], 4).remove(0).remove(0);
        assert!(backend.take_error().is_none());
        backend.commit(0, 1);
        let serial2 =
            backend.step(&[RowRequest { slot: 0, tokens: vec![serial1.ids[0]], first_position: 4 }], 4).remove(0).remove(0);
        assert!(backend.take_error().is_none());
        backend.commit(0, 1);
        let serial3 =
            backend.step(&[RowRequest { slot: 0, tokens: vec![serial2.ids[0]], first_position: 5 }], 4).remove(0).remove(0);
        assert!(backend.take_error().is_none());
        backend.commit(0, 1);

        // The same prompt and accepted rows in one plain speculative window must continue identically.
        let restarted = backend.prefill(0, 0, &[1, 2, 3], 4);
        assert_same_candidates(&restarted, &prefill, "plain restart prefill");
        let speculative =
            backend.step(&[RowRequest { slot: 0, tokens: vec![restarted.ids[0], serial1.ids[0]], first_position: 3 }], 4);
        assert!(backend.take_error().is_none());
        assert_same_candidates(&speculative[0][0], &serial1, "plain speculative root");
        assert_same_candidates(&speculative[0][1], &serial2, "plain speculative draft row");
        backend.commit(0, 2);
        let continued =
            backend.step(&[RowRequest { slot: 0, tokens: vec![serial2.ids[0]], first_position: 5 }], 4).remove(0).remove(0);
        assert!(backend.take_error().is_none());
        assert_same_candidates(&continued, &serial3, "plain multi-row commit continuation");
        backend.commit(0, 1);

        // A real tree (two children of the root) takes commit_conv_path_shaped on its accepted path.
        let restarted = backend.prefill(0, 0, &[1, 2, 3], 4);
        assert_same_candidates(&restarted, &prefill, "tree restart prefill");
        let tree = backend.step_tree(
            &[TreeRequest {
                slot: 0,
                tokens: vec![restarted.ids[0], serial1.ids[0], 42],
                parents: vec![-1, 0, 0],
                first_position: 3,
            }],
            4,
        );
        assert!(backend.take_error().is_none());
        assert_same_candidates(&tree[0][0], &serial1, "tree root");
        assert_same_candidates(&tree[0][1], &serial2, "tree accepted child");
        backend.commit_path(0, &[0, 1]);
        let continued =
            backend.step(&[RowRequest { slot: 0, tokens: vec![serial2.ids[0]], first_position: 5 }], 4).remove(0).remove(0);
        assert!(backend.take_error().is_none());
        assert_same_candidates(&continued, &serial3, "tree-path commit continuation");
        backend.commit(0, 1);
        drop(backend);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
