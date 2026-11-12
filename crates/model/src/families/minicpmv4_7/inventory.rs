//! The tensors a MiniCPM-V-4.7 recipe needs, checked against a pack inventory: a pack whose layers do not match its own
//! `layer_types`, MoE sizes, head width or vision config is refused by naming the tensor, before a byte is read. The check is
//! closed-world: a tensor the recipe does not claim is an error too (an `mtp.*` head the family does not run, a 41st layer, a
//! `.mcg` codebook marker beside the `.mul1` one).
//!
//! The committed table (`testdata/minicpmv4_7/tensor_inventory.tsv`, written by `tools/gen_minicpmv4_7_inventory.py`) is the
//! EXPECTED layout of the EXL3 pack, derived by rule from the Hugging Face checkpoint's tensors and the converter's naming (the
//! layout of the published Qwen3.5-35B-A3B pack, which has the same backbone): no pack of this model has been written yet. The
//! converter smoke (`Smoke B` of the plan) writes the first real one, and `tools/gen_minicpmv4_7_inventory.py diff` compares the two.
//! A real pack mixes per-tensor bit widths; the table states the nominal ones (4 bits for the experts, 6 for the attention, GDN,
//! shared-expert and head linears: `-b 4.00 -hq -hb 6`), and this check accepts any depth from 1 to 8 bits and tests the tile grid.
//!
//! What the pack holds, per layer `model.language_model.layers.N.`: the two RMSNorm weights, then either the GDN mixer
//! (`linear_attn.{in_proj_qkv,in_proj_z,out_proj}` as EXL3 groups, `in_proj_a`/`in_proj_b` plain fp16, `conv1d`, `A_log`,
//! `dt_bias`, `norm`) or the attention mixer (`self_attn.{q,k,v,o}_proj` as groups, `q_norm`, `k_norm`), then the MoE: the
//! router `mlp.gate.weight` plain, 256 experts of three groups each, the shared expert's three groups and its plain gate. 3,106
//! tensors a layer. At the top: `embed_tokens.weight` (bf16, 248,144 rows), the final `norm.weight` and `lm_head` as a group whose
//! output width is the padded 248,192. The tower and the projector (459 tensors, 1.106 GB, bf16) are copied through under
//! `model.vision_tower.` and `model.merger.`.

use super::config::{DSparkConfig, MiniCpmConfig};
pub use crate::families::mimo::inventory::{expand_table, expanded_count, Tensors};
use crate::family::Mixer;
use std::collections::BTreeSet;

/// The tensors of the tower and the projector in a pack, and their size, as the checkpoint ships them (bf16).
pub const VISION_TENSORS: usize = 459;
pub const VISION_BYTES: u64 = 1_105_972_064;
/// A layer's tensor count in the pack (norms 2, mixer 18, router 1, experts 3,072, shared expert 12 and its gate 1).
pub const TENSORS_PER_LAYER: usize = 3_106;

pub const TEXT_PREFIX: &str = "model.language_model";
pub const VISION_PREFIX: &str = "model.vision_tower";
pub const MERGER_PREFIX: &str = "model.merger";

struct Check<'a> {
    t: &'a Tensors,
    errors: Vec<String>,
    claimed: BTreeSet<String>,
}

impl Check<'_> {
    fn plain_half(&mut self, name: &str, allow_bf16: bool, shape: &[usize]) {
        self.claimed.insert(name.to_string());
        match self.t.get(name) {
            None => self.errors.push(format!("{name}: missing")),
            Some((d, _)) if d != "F16" && !(allow_bf16 && d == "BF16") => {
                self.errors.push(format!("{name}: dtype {d}, the recipe needs F16"))
            }
            Some((_, s)) if s != shape => self.errors.push(format!("{name}: shape {s:?}, the recipe needs {shape:?}")),
            Some(_) => {}
        }
    }

    fn plain(&mut self, name: &str, dtype: &str, shape: &[usize]) {
        self.claimed.insert(name.to_string());
        match self.t.get(name) {
            None => self.errors.push(format!("{name}: missing")),
            Some((d, _)) if d != dtype => self.errors.push(format!("{name}: dtype {d}, the recipe needs {dtype}")),
            Some((_, s)) if s != shape => self.errors.push(format!("{name}: shape {s:?}, the recipe needs {shape:?}")),
            Some(_) => {}
        }
    }

    /// An EXL3 linear `k -> n`: a trellis of 16 x 16 tiles (`[k / 16, n / 16, 16 x bits]`, 1 to 8 bits), its fp16 input and output
    /// scales and the `mul1` codebook marker.
    fn group(&mut self, name: &str, k: usize, n: usize) {
        for suffix in ["trellis", "suh", "svh", "mul1"] {
            self.claimed.insert(format!("{name}.{suffix}"));
        }
        let Some((dtype, tr)) = self.t.get(&format!("{name}.trellis")) else {
            return self.errors.push(format!("{name}: missing EXL3 linear"));
        };
        if tr.len() != 3 || (tr[0] * 16, tr[1] * 16) != (k, n) || tr[2] % 16 != 0 || !(1..=8).contains(&(tr[2] / 16)) {
            let note = if name == "lm_head" { " (the head is zero-padded to ceil(vocab / 128) x 128 output rows)" } else { "" };
            self.errors.push(format!("{name}.trellis: shape {tr:?}, the recipe needs [{}, {}, 16 x bits]{note}", k / 16, n / 16));
        } else if dtype != "I16" {
            self.errors.push(format!("{name}.trellis: dtype {dtype}, the recipe needs I16"));
        }
        self.plain(&format!("{name}.suh"), "F16", &[k]);
        self.plain(&format!("{name}.svh"), "F16", &[n]);
        match self.t.get(&format!("{name}.mul1")) {
            None => self.errors.push(format!("{name}.mul1: missing (the codebook marker)")),
            Some((d, s)) if d != "I32" || !s.is_empty() => {
                self.errors.push(format!("{name}.mul1: dtype {d} shape {s:?}, the marker is a scalar I32"))
            }
            Some(_) => {}
        }
    }

    /// A plain floating-point tensor whose dtype the pack may state as bf16, fp16 or fp32 (the draft's norms and heads: a draft
    /// pack's conversion decides, and the loader widens whichever it finds).
    fn plain_float(&mut self, name: &str, shape: &[usize]) {
        self.claimed.insert(name.to_string());
        match self.t.get(name) {
            None => self.errors.push(format!("{name}: missing")),
            Some((d, _)) if !matches!(d.as_str(), "BF16" | "F16" | "F32") => {
                self.errors.push(format!("{name}: dtype {d}, the recipe needs a floating-point tensor"))
            }
            Some((_, s)) if s != shape => self.errors.push(format!("{name}: shape {s:?}, the recipe needs {shape:?}")),
            Some(_) => {}
        }
    }

    /// A linear `k -> n` that is either an EXL3 group (a converted draft) or the checkpoint's `.weight` (`[n, k]`, floating point).
    fn linear(&mut self, name: &str, k: usize, n: usize) {
        if self.t.contains_key(&format!("{name}.trellis")) {
            self.group(name, k, n);
        } else {
            self.plain_float(&format!("{name}.weight"), &[n, k]);
        }
    }

    /// Nothing may exist under `prefix`.
    fn absent(&mut self, prefix: &str, why: &str) {
        for (n, _) in self.t.range(prefix.to_string()..).take_while(|(n, _)| n.starts_with(prefix)).take(3) {
            self.errors.push(format!("{n}: present, but {why}"));
        }
    }

    fn mixer(&mut self, p: &str, mixer: Mixer, c: &MiniCpmConfig) {
        let h = c.hidden;
        match mixer {
            Mixer::Gqa => {
                let (q, kv) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
                // queries and, when the layer is gated, their per-head output gate in one group (interleaved per head)
                self.group(&format!("{p}.self_attn.q_proj"), h, if c.attn_output_gate { 2 * q } else { q });
                self.group(&format!("{p}.self_attn.k_proj"), h, kv);
                self.group(&format!("{p}.self_attn.v_proj"), h, kv);
                self.group(&format!("{p}.self_attn.o_proj"), q, h);
                self.plain(&format!("{p}.self_attn.q_norm.weight"), "BF16", &[c.head_dim]);
                self.plain(&format!("{p}.self_attn.k_norm.weight"), "BF16", &[c.head_dim]);
                self.absent(&format!("{p}.linear_attn."), "the layer is full attention");
            }
            Mixer::Gdn => {
                let (z, ch, vh) = (c.lin_value_heads * c.lin_value_dim, c.conv_channels(), c.lin_value_heads);
                self.group(&format!("{p}.linear_attn.in_proj_qkv"), h, ch);
                self.group(&format!("{p}.linear_attn.in_proj_z"), h, z);
                self.group(&format!("{p}.linear_attn.out_proj"), z, h);
                self.plain_half(&format!("{p}.linear_attn.in_proj_a.weight"), c.is_qwen3_5_moe_shape(), &[vh, h]);
                self.plain_half(&format!("{p}.linear_attn.in_proj_b.weight"), c.is_qwen3_5_moe_shape(), &[vh, h]);
                self.plain(&format!("{p}.linear_attn.A_log"), "BF16", &[vh]);
                self.plain(&format!("{p}.linear_attn.dt_bias"), "BF16", &[vh]);
                self.plain(&format!("{p}.linear_attn.conv1d.weight"), "BF16", &[ch, 1, c.lin_conv_kernel]);
                self.plain(&format!("{p}.linear_attn.norm.weight"), "BF16", &[c.lin_value_dim]);
                self.absent(&format!("{p}.self_attn."), "the layer is a gated delta net");
            }
            other => self.errors.push(format!("{p}: mixer {other:?} is not part of MiniCPM-V-4.7")),
        }
    }

    fn moe(&mut self, p: &str, c: &MiniCpmConfig) {
        let (h, i, si) = (c.hidden, c.moe_intermediate, c.shared_intermediate);
        self.plain_half(&format!("{p}.mlp.gate.weight"), c.is_qwen3_5_moe_shape(), &[c.experts, h]);
        for e in 0..c.experts {
            self.group(&format!("{p}.mlp.experts.{e}.gate_proj"), h, i);
            self.group(&format!("{p}.mlp.experts.{e}.up_proj"), h, i);
            self.group(&format!("{p}.mlp.experts.{e}.down_proj"), i, h);
        }
        self.absent(&format!("{p}.mlp.experts.{}.", c.experts), "the config has fewer experts");
        self.group(&format!("{p}.mlp.shared_expert.gate_proj"), h, si);
        self.group(&format!("{p}.mlp.shared_expert.up_proj"), h, si);
        self.group(&format!("{p}.mlp.shared_expert.down_proj"), si, h);
        self.plain_half(&format!("{p}.mlp.shared_expert_gate.weight"), c.is_qwen3_5_moe_shape(), &[1, h]);
    }

    /// The tower, the window merger and the projector: bf16 tensors copied through from the checkpoint.
    fn vision(&mut self, c: &MiniCpmConfig) {
        let (v, h) = (&c.vision, c.hidden);
        let (vh, vi) = (v.hidden, v.intermediate);
        let t = VISION_PREFIX;
        let both = |ck: &mut Self, base: &str, w: &[usize], b: &[usize]| {
            ck.plain(&format!("{base}.weight"), "BF16", w);
            ck.plain(&format!("{base}.bias"), "BF16", b);
        };
        both(self, &format!("{t}.embeddings.patch_embedding"), &[vh, 3, v.patch_size, v.patch_size], &[vh]);
        self.plain(&format!("{t}.embeddings.position_embedding.weight"), "BF16", &[v.position_rows(), vh]);
        for l in 0..v.layers {
            let p = format!("{t}.encoder.layers.{l}");
            for n in ["layer_norm1", "layer_norm2"] {
                both(self, &format!("{p}.{n}"), &[vh], &[vh]);
            }
            for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                both(self, &format!("{p}.self_attn.{n}"), &[vh, vh], &[vh]);
            }
            both(self, &format!("{p}.mlp.fc1"), &[vi, vh], &[vi]);
            both(self, &format!("{p}.mlp.fc2"), &[vh, vi], &[vh]);
        }
        both(self, &format!("{t}.post_layernorm"), &[vh], &[vh]);
        // the 2x2 window merger after encoder layer `insert_layer_id`: window self-attention, then 4 x 1152 -> 4 x 4304 -> 1152
        let m = format!("{t}.vit_merger");
        both(self, &format!("{m}.layer_norm1"), &[vh], &[vh]);
        for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            both(self, &format!("{m}.self_attn.{n}"), &[vh, vh], &[vh]);
        }
        both(self, &format!("{m}.pre_norm"), &[4 * vh], &[4 * vh]);
        both(self, &format!("{m}.linear_1"), &[4 * vi, 4 * vh], &[4 * vi]);
        both(self, &format!("{m}.linear_2"), &[vh, 4 * vi], &[vh]);
        // the projector into the text width
        let g = format!("{MERGER_PREFIX}.mlp.0");
        both(self, &format!("{g}.pre_norm"), &[4 * vh], &[4 * vh]);
        both(self, &format!("{g}.linear_1"), &[4 * vh, 4 * vh], &[4 * vh]);
        both(self, &format!("{g}.linear_2"), &[h, 4 * vh], &[h]);
    }
}

/// What a pack holding `tensors` lacks, holds in the wrong shape or dtype, or holds beyond the recipe of `cfg`. Empty when they
/// agree. Every error names the tensor.
pub fn check(cfg: &MiniCpmConfig, tensors: &Tensors) -> Vec<String> {
    check_with_claims(cfg, tensors).0
}

/// [`check`] and the names the recipe claimed (every tensor of every group, the plain tensors, the tower and the projector), which
/// is what the weight map (`super::weights`) binds one by one: a claimed name that exists is a slot, and no other name is.
pub fn check_with_claims(cfg: &MiniCpmConfig, tensors: &Tensors) -> (Vec<String>, BTreeSet<String>) {
    let mut ck = Check { t: tensors, errors: Vec::new(), claimed: BTreeSet::new() };
    let h = cfg.hidden;
    ck.plain(&format!("{TEXT_PREFIX}.embed_tokens.weight"), "BF16", &[cfg.vocab, h]);
    ck.plain(&format!("{TEXT_PREFIX}.norm.weight"), "BF16", &[h]);
    ck.group("lm_head", h, cfg.lm_head_width());
    for (i, spec) in cfg.layer_specs().iter().enumerate() {
        let p = format!("{TEXT_PREFIX}.layers.{i}");
        ck.plain(&format!("{p}.input_layernorm.weight"), "BF16", &[h]);
        ck.plain(&format!("{p}.post_attention_layernorm.weight"), "BF16", &[h]);
        ck.mixer(&p, spec.mixer, cfg);
        ck.moe(&p, cfg);
    }
    if cfg.is_qwen3_5_moe_shape() {
        for name in tensors.keys() {
            if name.starts_with("model.visual.") || name.starts_with("mtp.") {
                ck.claimed.insert(name.clone());
            }
        }
        ck.absent(&format!("{TEXT_PREFIX}.layers.{}.", cfg.layers), "the config has fewer layers");
    } else {
        ck.vision(cfg);
        ck.absent(&format!("{TEXT_PREFIX}.layers.{}.", cfg.layers), "the config has fewer layers");
        ck.absent(
            "mtp.",
            "this checkpoint ships no MTP head (the config's mtp_num_hidden_layers is not backed by tensors, the family runs none)",
        );
    }
    // closed world: whatever the recipe did not claim is wrong, whatever its name
    let mut errors = ck.errors;
    let stray: Vec<&String> = tensors.keys().filter(|n| !ck.claimed.contains(*n) && !n.starts_with("mtp.")).collect();
    for n in stray.iter().take(5) {
        if errors.iter().any(|e| e.starts_with(&format!("{n}:"))) {
            continue;
        }
        errors.push(format!("{n}: not part of the MiniCPM-V-4.7 recipe"));
    }
    if stray.len() > 5 {
        errors.push(format!("... and {} more tensors outside the recipe", stray.len() - 5));
    }
    (errors, ck.claimed)
}

/// The tensors under the tower and projector prefixes and their bf16 size (checks the constants above against a table).
pub fn vision_totals(t: &Tensors) -> (usize, u64) {
    let mut n = 0;
    let mut bytes = 0u64;
    for (name, (dtype, shape)) in t {
        if name.starts_with(VISION_PREFIX) || name.starts_with(MERGER_PREFIX) {
            n += 1;
            bytes += shape.iter().map(|&d| d as u64).product::<u64>() * gb_exl3::checkpoint::dtype_size(dtype).unwrap_or(0);
        }
    }
    (n, bytes)
}

/// What a DSpark draft holding `tensors` lacks, holds in the wrong shape, or holds beyond the recipe of `cfg` (the draft's own names,
/// no `model.` prefix). The body, `fc`, the Markov tables and the confidence head; **no embedding and no LM head**, which are the
/// target's. Each linear may be the checkpoint's bf16 weight or an EXL3 group; the norms and heads are floating point. Empty when
/// they agree.
pub fn check_draft(cfg: &DSparkConfig, tensors: &Tensors) -> Vec<String> {
    let mut ck = Check { t: tensors, errors: Vec::new(), claimed: BTreeSet::new() };
    let h = cfg.hidden;
    let (q, kv) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
    ck.linear("fc", cfg.fc_in(), h);
    ck.plain_float("hidden_norm.weight", &[h]);
    ck.plain_float("norm.weight", &[h]);
    for i in 0..cfg.layers {
        let p = format!("layers.{i}");
        ck.plain_float(&format!("{p}.input_layernorm.weight"), &[h]);
        ck.plain_float(&format!("{p}.post_attention_layernorm.weight"), &[h]);
        ck.linear(&format!("{p}.self_attn.q_proj"), h, q);
        ck.linear(&format!("{p}.self_attn.k_proj"), h, kv);
        ck.linear(&format!("{p}.self_attn.v_proj"), h, kv);
        ck.linear(&format!("{p}.self_attn.o_proj"), q, h);
        ck.plain_float(&format!("{p}.self_attn.q_norm.weight"), &[cfg.head_dim]);
        ck.plain_float(&format!("{p}.self_attn.k_norm.weight"), &[cfg.head_dim]);
        ck.linear(&format!("{p}.mlp.gate_proj"), h, cfg.intermediate);
        ck.linear(&format!("{p}.mlp.up_proj"), h, cfg.intermediate);
        ck.linear(&format!("{p}.mlp.down_proj"), cfg.intermediate, h);
    }
    // the Markov transition tables: `w1` embeds the previous token (target vocabulary), `w2` projects it into the draft vocabulary
    ck.plain_float("markov_head.markov_w1.weight", &[cfg.vocab, cfg.markov_rank]);
    ck.plain_float("markov_head.markov_w2.weight", &[cfg.draft_vocab, cfg.markov_rank]);
    if cfg.enable_confidence_head {
        ck.plain_float("confidence_head.proj.weight", &[1, cfg.confidence_in()]);
        ck.plain_float("confidence_head.proj.bias", &[1]);
    } else {
        ck.absent("confidence_head.", "the config disables the confidence head");
    }
    ck.absent(&format!("layers.{}.", cfg.layers), "the config has fewer layers");
    for shared in ["embed_tokens", "lm_head"] {
        ck.absent(&format!("{shared}."), "the draft shares the target's embedding and head and ships neither");
    }
    let mut errors = ck.errors;
    for n in tensors.keys().filter(|n| !ck.claimed.contains(*n)).take(5) {
        if !errors.iter().any(|e| e.starts_with(&format!("{n}:"))) {
            errors.push(format!("{n}: not part of the DSpark draft recipe"));
        }
    }
    errors
}

/// The layers of `t` that carry a layer-index tensor (a count by `layers.N.input_layernorm.weight`).
pub fn layers_in(t: &Tensors) -> usize {
    t.keys().filter(|n| n.starts_with(&format!("{TEXT_PREFIX}.layers.")) && n.ends_with(".input_layernorm.weight")).count()
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::real;
    use super::*;

    const TABLE: &str = include_str!("../../../../../testdata/minicpmv4_7/tensor_inventory.tsv");

    fn table() -> Tensors {
        expand_table(TABLE).unwrap()
    }

    fn has(errs: &[String], needle: &str) -> bool {
        errs.iter().any(|e| e.contains(needle))
    }

    #[test]
    fn the_committed_inventory_matches_the_recipe_of_its_config() {
        let t = table();
        // 40 layers x 3,106 + lm_head 4 + embedding 1 + final norm 1 = 124,246 text tensors, and 459 tower and projector tensors
        assert_eq!(TENSORS_PER_LAYER, 2 + 18 + 1 + 3 * 4 * 256 + 3 * 4 + 1);
        assert_eq!(t.len(), 40 * 3_106 + 6 + 459);
        assert_eq!(t.len(), 124_705);
        assert_eq!(expanded_count(&t), 124_705);
        assert_eq!(layers_in(&t), 40);
        let errors = check(&real(), &t);
        assert!(errors.is_empty(), "{:#?}", &errors[..errors.len().min(10)]);
    }

    #[test]
    fn the_tower_and_the_projector_are_459_bf16_tensors_of_1_106_gb() {
        let (n, bytes) = vision_totals(&table());
        assert_eq!((n, bytes), (VISION_TENSORS, VISION_BYTES));
        // 27 layers x 16 + embeddings 3 + post norm 2 + merger 16 + projector 6
        assert_eq!(27 * 16 + 3 + 2 + 16 + 6, 459);
    }

    #[test]
    fn the_head_is_padded_and_the_embedding_is_not() {
        let t = table();
        assert_eq!(t["lm_head.trellis"].1, vec![128, 15_512, 96]);
        assert_eq!(t["lm_head.svh"].1, vec![248_192]);
        assert_eq!(t["lm_head.suh"].1, vec![2048]);
        assert_eq!(t["model.language_model.embed_tokens.weight"].1, vec![248_144, 2048]);
        // a head at the unpadded vocabulary (what a converter that does not pad would write) is refused, naming the padding
        let mut t = table();
        t.get_mut("lm_head.trellis").unwrap().1 = vec![128, 15_509, 96];
        t.get_mut("lm_head.svh").unwrap().1 = vec![248_144];
        let e = check(&real(), &t);
        assert!(has(&e, "lm_head.trellis: shape [128, 15509, 96], the recipe needs [128, 15512, 16 x bits]"), "{e:?}");
        assert!(has(&e, "zero-padded"), "{e:?}");
        assert!(has(&e, "lm_head.svh: shape [248144]"), "{e:?}");
        // and an embedding at the padded width (the vocabulary the sampler sees is 248,144 ids) is refused as well
        let mut t = table();
        t.get_mut("model.language_model.embed_tokens.weight").unwrap().1 = vec![248_192, 2048];
        let e = check(&real(), &t);
        assert!(has(&e, "embed_tokens.weight: shape [248192, 2048], the recipe needs [248144, 2048]"), "{e:?}");
    }

    /// Each edit of the table is refused by naming the tensor it broke.
    #[test]
    fn a_pack_that_disagrees_with_its_config_is_refused_by_name() {
        let cfg = real();
        // an attention projection missing on layer 7
        let mut t = table();
        t.remove("model.language_model.layers.7.self_attn.v_proj.trellis");
        let e = check(&cfg, &t);
        assert!(has(&e, "layers.7.self_attn.v_proj: missing EXL3 linear"), "{e:?}");
        // the router of layer 0 with the wrong width (255 experts)
        let mut t = table();
        t.get_mut("model.language_model.layers.0.mlp.gate.weight").unwrap().1 = vec![255, 2048];
        let e = check(&cfg, &t);
        assert!(has(&e, "layers.0.mlp.gate.weight: shape [255, 2048], the recipe needs [256, 2048]"), "{e:?}");
        // an expert's down projection of the wrong width
        let mut t = table();
        t.get_mut("model.language_model.layers.20.mlp.experts.123.down_proj.trellis").unwrap().1 = vec![64, 128, 64];
        let e = check(&cfg, &t);
        assert!(has(&e, "layers.20.mlp.experts.123.down_proj.trellis: shape [64, 128, 64]"), "{e:?}");
        // a 257th expert, and a 41st layer
        let mut t = table();
        t.insert("model.language_model.layers.5.mlp.experts.256.gate_proj.trellis".into(), ("I16".into(), vec![128, 32, 64]));
        let e = check(&cfg, &t);
        assert!(has(&e, "layers.5.mlp.experts.256.gate_proj.trellis: present, but the config has fewer experts"), "{e:?}");
        let mut t = table();
        t.insert("model.language_model.layers.40.input_layernorm.weight".into(), ("BF16".into(), vec![2048]));
        let e = check(&cfg, &t);
        assert!(has(&e, "layers.40.input_layernorm.weight: present, but the config has fewer layers"), "{e:?}");
        // a GDN layer that also has an attention projection (a layer_types that does not match the tensors)
        let mut t = table();
        t.insert("model.language_model.layers.4.self_attn.q_norm.weight".into(), ("BF16".into(), vec![256]));
        let e = check(&cfg, &t);
        assert!(has(&e, "layers.4.self_attn.q_norm.weight: present, but the layer is a gated delta net"), "{e:?}");
        // the codebook marker of the other family of packs
        let mut t = table();
        let mul1 = t.remove("model.language_model.layers.3.self_attn.k_proj.mul1").unwrap();
        t.insert("model.language_model.layers.3.self_attn.k_proj.mcg".into(), mul1);
        let e = check(&cfg, &t);
        assert!(has(&e, "layers.3.self_attn.k_proj.mul1: missing (the codebook marker)"), "{e:?}");
        assert!(has(&e, "layers.3.self_attn.k_proj.mcg: not part of the MiniCPM-V-4.7 recipe"), "{e:?}");
        // a tower tensor of the wrong shape, and the position table of another grid
        let mut t = table();
        t.get_mut("model.vision_tower.vit_merger.linear_1.weight").unwrap().1 = vec![4304, 4608];
        t.get_mut("model.vision_tower.embeddings.position_embedding.weight").unwrap().1 = vec![2304, 1152];
        let e = check(&cfg, &t);
        assert!(has(&e, "vit_merger.linear_1.weight: shape [4304, 4608], the recipe needs [17216, 4608]"), "{e:?}");
        assert!(has(&e, "position_embedding.weight: shape [2304, 1152]"), "{e:?}");
        // a norm in the wrong dtype (the loader reads the pack's gamma as bf16)
        let mut t = table();
        t.get_mut("model.language_model.norm.weight").unwrap().0 = "F16".into();
        let e = check(&cfg, &t);
        assert!(has(&e, "model.language_model.norm.weight: dtype F16, the recipe needs BF16"), "{e:?}");
    }

    #[test]
    fn mtp_tensors_are_refused_because_the_family_runs_none() {
        // Qwen3.6 packs carry `mtp.*`; this checkpoint does not, although its config declares one layer
        let mut t = table();
        t.insert("mtp.fc.weight".into(), ("BF16".into(), vec![2048, 4096]));
        t.insert("mtp.layers.0.mlp.gate.weight".into(), ("F16".into(), vec![256, 2048]));
        let e = check(&real(), &t);
        assert!(has(&e, "mtp.fc.weight: present, but this checkpoint ships no MTP head"), "{e:?}");
        assert_eq!(e.len(), 2, "{e:?}");
        assert!(table().keys().all(|n| !n.starts_with("mtp.")));
    }

    const DRAFT_TABLE: &str = include_str!("../../../../../testdata/minicpmv4_7/draft_tensors.tsv");

    fn draft_table() -> Tensors {
        expand_table(DRAFT_TABLE).unwrap()
    }

    #[test]
    fn the_real_draft_checkpoint_matches_its_config() {
        use super::super::config::tests::draft;
        let t = draft_table();
        // 62 tensors, 389,223,169 parameters: fc, hidden_norm, norm, 5 x 11, the two Markov tables and the confidence head
        assert_eq!(t.len(), 62);
        let params: u64 = t.values().map(|(_, s)| s.iter().map(|&d| d as u64).product::<u64>()).sum();
        assert_eq!(params, 389_223_169);
        let e = check_draft(&draft(), &t);
        assert!(e.is_empty(), "{e:?}");
        // no embedding, no LM head, no mask embedding: they are the target's (the mask id is a row of the target's embedding)
        assert!(t.keys().all(|n| !n.contains("embed_tokens") && !n.contains("lm_head") && !n.contains("mask_embedding")));
    }

    #[test]
    fn a_converted_draft_with_exl3_linears_is_accepted_too() {
        use super::super::config::tests::draft;
        let mut t = Tensors::new();
        let mut converted = 0;
        for (name, (dtype, shape)) in draft_table() {
            let linear = name == "fc.weight" || name.ends_with("_proj.weight");
            match name.strip_suffix(".weight") {
                Some(prefix) if linear => {
                    let (n, k) = (shape[0], shape[1]);
                    t.insert(format!("{prefix}.trellis"), ("I16".into(), vec![k / 16, n / 16, 6 * 16]));
                    t.insert(format!("{prefix}.suh"), ("F16".into(), vec![k]));
                    t.insert(format!("{prefix}.svh"), ("F16".into(), vec![n]));
                    t.insert(format!("{prefix}.mul1"), ("I32".into(), vec![]));
                    converted += 1;
                }
                _ => {
                    t.insert(name, (dtype, shape));
                }
            }
        }
        // fc and 5 x 7 linears, four tensors each
        assert_eq!(converted, 36);
        assert_eq!(t.len(), 62 - 36 + 36 * 4);
        let e = check_draft(&draft(), &t);
        assert!(e.is_empty(), "{e:?}");
    }

    #[test]
    fn a_draft_that_disagrees_with_its_config_is_refused_by_name() {
        use super::super::config::tests::draft;
        let cfg = draft();
        // a Markov table missing, and one of the wrong rank
        let mut t = draft_table();
        t.remove("markov_head.markov_w2.weight");
        t.get_mut("markov_head.markov_w1.weight").unwrap().1 = vec![248_144, 128];
        let e = check_draft(&cfg, &t);
        assert!(has(&e, "markov_head.markov_w2.weight: missing"), "{e:?}");
        assert!(has(&e, "markov_head.markov_w1.weight: shape [248144, 128], the recipe needs [248144, 256]"), "{e:?}");
        // `fc` reading four taps instead of five
        let mut t = draft_table();
        t.get_mut("fc.weight").unwrap().1 = vec![2048, 8192];
        let e = check_draft(&cfg, &t);
        assert!(has(&e, "fc.weight: shape [2048, 8192], the recipe needs [2048, 10240]"), "{e:?}");
        // a confidence head that does not read the Markov embedding
        let mut t = draft_table();
        t.get_mut("confidence_head.proj.weight").unwrap().1 = vec![1, 2048];
        let e = check_draft(&cfg, &t);
        assert!(has(&e, "confidence_head.proj.weight: shape [1, 2048], the recipe needs [1, 2304]"), "{e:?}");
        // an embedding or an LM head of its own, and a sixth layer
        let mut t = draft_table();
        t.insert("embed_tokens.weight".into(), ("BF16".into(), vec![248_144, 2048]));
        t.insert("lm_head.weight".into(), ("BF16".into(), vec![248_144, 2048]));
        t.insert("layers.5.input_layernorm.weight".into(), ("BF16".into(), vec![2048]));
        let e = check_draft(&cfg, &t);
        assert!(has(&e, "embed_tokens.weight: present, but the draft shares the target's embedding"), "{e:?}");
        assert!(has(&e, "lm_head.weight: present, but the draft shares"), "{e:?}");
        assert!(has(&e, "layers.5.input_layernorm.weight: present, but the config has fewer layers"), "{e:?}");
        // an integer where a norm should be
        let mut t = draft_table();
        t.get_mut("norm.weight").unwrap().0 = "I32".into();
        let e = check_draft(&cfg, &t);
        assert!(has(&e, "norm.weight: dtype I32"), "{e:?}");
        // an unknown tensor
        let mut t = draft_table();
        t.insert("rotary_emb.inv_freq".into(), ("F32".into(), vec![128]));
        assert_eq!(check_draft(&cfg, &t), vec!["rotary_emb.inv_freq: not part of the DSpark draft recipe".to_string()]);
    }

    #[test]
    fn a_tensor_outside_the_recipe_is_refused_by_name() {
        let mut t = table();
        t.insert("model.language_model.rotary_emb.inv_freq".into(), ("F32".into(), vec![32]));
        let e = check(&real(), &t);
        assert_eq!(e, vec!["model.language_model.rotary_emb.inv_freq: not part of the MiniCPM-V-4.7 recipe".to_string()]);
    }
}
