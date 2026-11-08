//! The tensors a MiMo recipe needs, checked against a pack inventory: a pack whose layers do not match its own
//! `hybrid_layer_pattern`, MoE sizes or attention dims is refused by name, before a byte is read. The committed table
//! (`testdata/mimo/tensor_inventory.tsv`, written by `tools/gen_mimo_inventory.py`) is the real vcruz305 2.20bpw
//! pack's inventory; `tests/mimo_pack.rs` checks it against the pack when `REXL3_MIMO_DIR` is set (M2's weights slice
//! runs it on the anonymized host-d pack).
//!
//! The pack has NO sidecars: no `preserved-fp.safetensors` (the MTP layers), no `dflash/` and no `audio_tokenizer/` —
//! verified against the Hugging Face file list 2026-10-04 (36 files). Those weights load from the original
//! XiaomiMiMo release beside the pack in M6/M8/M9 (mixed provenance, recorded in the fingerprint entry).

use super::config::MiMoConfig;
use std::collections::BTreeMap;

/// Every tensor of a pack by name: (dtype, shape). The compact table's format extends GLM's and Flash-Next's with a
/// fifth `except` column (`{L}:{E}=trellis-depth` overrides), because this pack mixes per-(layer, expert) bit widths.
pub type Tensors = BTreeMap<String, (String, Vec<usize>)>;

/// Expand the compact table: `experts N`, then rows of `pattern, dtype, shape, layers[, except]`. `{L}` takes each
/// listed layer, `{E}` each expert id below N; a shape of `-` is a scalar; an `except` entry `l:e=d` overrides the
/// trellis depth of `pattern[l]`'s expert `e` to `d` words.
pub fn expand_table(text: &str) -> Result<Tensors, String> {
    let mut experts = None;
    let mut out = Tensors::new();
    for (no, line) in text.lines().enumerate() {
        let line = line.trim_end();
        let at = |m: &str| format!("tensor table line {}: {m}", no + 1);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(n) = line.strip_prefix("experts ") {
            experts = Some(n.trim().parse::<usize>().map_err(|_| at("bad expert count"))?);
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 4 {
            return Err(at("expected 4 or 5 columns"));
        }
        let (pattern, dtype, shape, layers) = (cols[0], cols[1], cols[2], cols[3]);
        let shape: Vec<usize> = if shape == "-" {
            vec![]
        } else {
            shape.split(',').map(|d| d.parse().map_err(|_| at("bad shape"))).collect::<Result<_, _>>()?
        };
        let mut ids: Vec<Option<usize>> = vec![None];
        if layers != "-" {
            ids.clear();
            for part in layers.split(',') {
                let (a, b) = part.split_once('-').unwrap_or((part, part));
                let (a, b): (usize, usize) = (a.parse().map_err(|_| at("bad layers"))?, b.parse().map_err(|_| at("bad layers"))?);
                ids.extend((a..=b).map(Some));
            }
        }
        for l in ids {
            let p = l.map_or(pattern.to_string(), |l| pattern.replace("{L}", &l.to_string()));
            let names: Vec<String> = if p.contains("{E}") {
                (0..experts.ok_or_else(|| at("`{E}` before the `experts` line"))?)
                    .map(|e| p.replace("{E}", &e.to_string()))
                    .collect()
            } else {
                vec![p]
            };
            for n in names {
                if out.insert(n.clone(), (dtype.to_string(), shape.clone())).is_some() {
                    return Err(at(&format!("{n} is listed twice")));
                }
            }
        }
        // the except column: override trellis depths of named (layer, expert) pairs
        if let Some(exc) = cols.get(4).filter(|s| !s.is_empty()) {
            for one in exc.split(',') {
                let (pair, depth) = one.split_once('=').ok_or_else(|| at("bad except entry"))?;
                let (l, e): (usize, usize) = pair
                    .split_once(':')
                    .and_then(|(l, e)| Some((l.parse().ok()?, e.parse().ok()?)))
                    .ok_or_else(|| at("bad except pair"))?;
                let n = pattern.replace("{L}", &l.to_string()).replace("{E}", &e.to_string());
                let entry = out.get_mut(&n).ok_or_else(|| at("except names a tensor the row does not cover"))?;
                if entry.1.len() == 3 {
                    entry.1[2] = depth.parse().map_err(|_| at("bad except depth"))?;
                } else {
                    return Err(at("except on a non-trellis tensor"));
                }
            }
        }
    }
    Ok(out)
}

/// The count of tensors the expanded table names (the generator's header names the raw count; they must agree).
pub fn expanded_count(t: &Tensors) -> usize {
    t.len()
}

/// Check a pack inventory against the recipe the config declares. Every error names the tensor.
pub fn check(cfg: &MiMoConfig, t: &Tensors) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let mut c = Check { t, errors: &mut errors };
    // top level
    c.plain("model.embed_tokens.weight", &[cfg.vocab, cfg.hidden]);
    c.plain("model.norm.weight", &[cfg.hidden]);
    c.group("lm_head", cfg.hidden, cfg.vocab);
    for l in 0..cfg.layers {
        let p = format!("model.layers.{l}");
        c.plain(&format!("{p}.input_layernorm.weight"), &[cfg.hidden]);
        c.plain(&format!("{p}.post_attention_layernorm.weight"), &[cfg.hidden]);
        // attention: q, k, v separate. IMPORTANT (verified from the shard headers 2026-10-04): the v_proj weights
        // are STORED at the K head dim (192), like k_proj — the [hidden, kv_heads x 192] linear is sliced to the
        // 128-wide V per head at load (M2's job); the inventory checks the stored width.
        let swa = cfg.hybrid_pattern[l] == 1;
        let (kv_h, k_dim) = if swa { (cfg.swa_kv_heads, cfg.swa_head_dim) } else { (cfg.kv_heads, cfg.head_dim) };
        let v_dim = if swa { cfg.swa_v_head_dim } else { cfg.v_head_dim };
        let v_stored_dim = kv_h * k_dim; // stored wide; sliced to v_dim per head at load
        c.group(&format!("{p}.self_attn.q_proj"), cfg.hidden, cfg.heads * k_dim);
        c.group(&format!("{p}.self_attn.k_proj"), cfg.hidden, kv_h * k_dim);
        c.group(&format!("{p}.self_attn.v_proj"), cfg.hidden, v_stored_dim);
        c.group(&format!("{p}.self_attn.o_proj"), cfg.heads * v_dim, cfg.hidden);
        if swa && cfg.swa_sink {
            c.plain(&format!("{p}.self_attn.attention_sink_bias"), &[cfg.heads]);
        }
        // FFN
        if cfg.moe_freq[l] == 0 {
            c.group(&format!("{p}.mlp.gate_proj"), cfg.hidden, cfg.dense_intermediate);
            c.group(&format!("{p}.mlp.up_proj"), cfg.hidden, cfg.dense_intermediate);
            c.group(&format!("{p}.mlp.down_proj"), cfg.dense_intermediate, cfg.hidden);
        } else {
            c.plain(&format!("{p}.mlp.gate.weight"), &[cfg.experts, cfg.hidden]);
            c.plain(&format!("{p}.mlp.gate.e_score_correction_bias"), &[cfg.experts]);
            for e in 0..cfg.experts {
                let q = format!("{p}.mlp.experts.{e}");
                c.group(&format!("{q}.gate_proj"), cfg.hidden, cfg.moe_intermediate);
                c.group(&format!("{q}.up_proj"), cfg.hidden, cfg.moe_intermediate);
                c.group(&format!("{q}.down_proj"), cfg.moe_intermediate, cfg.hidden);
            }
        }
    }
    // the pack is text-only: the MTP/DFlash/vision sidecars are absent BY DESIGN (they load from the original
    // release in M6/M8/M9); nothing to check here
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

struct Check<'a> {
    t: &'a Tensors,
    errors: &'a mut Vec<String>,
}

impl Check<'_> {
    fn plain(&mut self, name: &str, shape: &[usize]) {
        match self.t.get(name) {
            None => self.errors.push(format!("{name}: missing")),
            Some((_, s)) if s != shape => self.errors.push(format!("{name}: shape {s:?}, the recipe needs {shape:?}")),
            Some(_) => {}
        }
    }
    /// An EXL3 linear `k -> n`: a trellis of 16 x 16 tiles (`[k / 16, n / 16, 16 x bits]`), fp16 scales, marker.
    /// This pack mixes per-layer bit widths (32/48/64 words) — the trellis depth is whatever the pack chose, so only
    /// the tile grid is checked, not the depth.
    fn group(&mut self, name: &str, k: usize, n: usize) {
        let Some((_, tr)) = self.t.get(&format!("{name}.trellis")) else {
            return self.errors.push(format!("{name}: missing EXL3 linear"));
        };
        if tr.len() != 3 || (tr[0] * 16, tr[1] * 16) != (k, n) || tr[2] % 16 != 0 || !(1..=8).contains(&(tr[2] / 16)) {
            return self.errors.push(format!(
                "{name}.trellis: shape {tr:?}, the recipe needs [{}, {}, 16 x bits]",
                k / 16,
                n / 16
            ));
        }
        self.plain(&format!("{name}.suh"), &[k]);
        self.plain(&format!("{name}.svh"), &[n]);
        if !self.t.contains_key(&format!("{name}.mul1")) {
            self.errors.push(format!("{name}.mul1: missing (the codebook marker)"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::MIMO_CONFIG;
    use super::*;

    const TABLE: &str = include_str!("../../../../../testdata/mimo/tensor_inventory.tsv");

    #[test]
    fn the_committed_table_is_the_real_packs_inventory_and_satisfies_the_recipe() {
        let cfg = MiMoConfig::from_json(&serde_json::from_str(MIMO_CONFIG).unwrap()).unwrap();
        let t = expand_table(TABLE).unwrap();
        // the generator's header: 145,399 tensors in 83 rows (the pack's index count; re-verified at generation)
        assert_eq!(expanded_count(&t), 145_399, "the committed table must expand to the pack's tensor count");
        check(&cfg, &t).unwrap();
    }

    #[test]
    fn a_table_missing_a_tensor_or_with_a_wrong_shape_is_refused_by_name() {
        let cfg = MiMoConfig::from_json(&serde_json::from_str(MIMO_CONFIG).unwrap()).unwrap();
        let mut t = expand_table(TABLE).unwrap();
        let victim = "model.layers.7.self_attn.attention_sink_bias".to_string();
        t.remove(&victim);
        let errs = check(&cfg, &t).unwrap_err();
        assert!(errs.iter().any(|e| e.contains(&victim)), "{errs:?}");
        t.insert(victim, ("BF16".into(), vec![8])); // wrong shape: the recipe needs [64]
        let errs = check(&cfg, &t).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("shape [8]")), "{errs:?}");
        // and an expert's EXL3 group gone missing
        t.remove("model.layers.20.mlp.experts.123.down_proj.trellis");
        let errs = check(&cfg, &t).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("experts.123.down_proj")), "{errs:?}");
    }

    #[test]
    fn the_pack_has_no_sidecars_by_design() {
        let t = expand_table(TABLE).unwrap();
        for absent in ["model.mtp.layers.0.eh_proj.weight", "dflash.draft.layers.0", "visual.patch_embed"] {
            assert!(!t.contains_key(absent), "{absent} must not be in the 2.20bpw pack's index");
        }
    }
}
