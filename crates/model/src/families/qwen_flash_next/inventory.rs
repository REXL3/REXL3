//! The tensors the recipe of a Flash-Next config needs, checked against a pack inventory: a pack whose layers do not match its
//! own `layer_types`, MoE sizes, hyper-connection rank, PLE layer or MTP layer is refused by naming the tensor, before a byte is
//! read. The committed table (`testdata/qwen_flash_next/tensor_inventory.tsv`, written by `tools/gen_flash_next_inventory.py`)
//! is the real pack's inventory; `tests/flash_next_pack.rs` checks it against the pack when `REXL3_FLASH_NEXT_DIR` is set.
//!
//! Two sidecar files sit beside the indexed shards: the n-gram table (`ngram_embedding.safetensors`: the table shards, which
//! are packed rows read by the PLE stage and not EXL3 linears, and their hash parameters) and the MTP head's input mixer.

use super::config::FlashNextConfig;
use crate::family::Mixer;
use std::collections::BTreeMap;

/// Every tensor of a pack by name: (safetensors dtype, shape).
pub type Tensors = BTreeMap<String, (String, Vec<usize>)>;

pub const NGRAM_FILE: &str = "ngram_embedding.safetensors";
pub const MTP_MIXER_FILE: &str = "mtp_hyper_connection_mixer_patch.safetensors";
/// Name of n-gram table shard `i` under the PLE layer's prefix.
pub const NGRAM_SHARD: &str = "ple.ple_embedding.ngram_embedding.shard_";

/// Width of one packed n-gram table row: an fp16 scale and 160 codes of `bits` bits, in u16 words.
pub fn ngram_row_words(bits: usize) -> usize {
    1 + 160 * bits / 16
}

/// Expand the compact table (`experts N`, then rows of name pattern, dtype, shape, layers; `{L}` takes each listed layer, `{E}`
/// each expert id below N, a shape of `-` is a scalar). The format is the one GLM's `PackInventory::from_table` reads.
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
        let [pattern, dtype, shape, layers] = line.split('\t').collect::<Vec<_>>()[..] else {
            return Err(at("expected 4 columns"));
        };
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
    }
    Ok(out)
}

struct Check<'a> {
    t: &'a Tensors,
    errors: Vec<String>,
    claimed: std::collections::BTreeSet<String>,
}

impl Check<'_> {
    fn plain(&mut self, name: &str, shape: &[usize]) {
        self.claimed.insert(name.to_string());
        match self.t.get(name) {
            None => self.errors.push(format!("{name}: missing")),
            Some((_, s)) if s != shape => self.errors.push(format!("{name}: shape {s:?}, the recipe needs {shape:?}")),
            Some(_) => {}
        }
    }
    /// An EXL3 linear `k -> n`: a trellis of 16 x 16 tiles (`[k / 16, n / 16, 16 bits]`, 1 to 8 bits), its fp16 input and
    /// output scales and the codebook marker.
    fn group(&mut self, name: &str, k: usize, n: usize) {
        for suffix in ["trellis", "suh", "svh", "mul1"] {
            self.claimed.insert(format!("{name}.{suffix}"));
        }
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
    fn absent(&mut self, prefix: &str, why: &str) {
        if let Some(n) = self.t.range(prefix.to_string()..).next().map(|(n, _)| n).filter(|n| n.starts_with(prefix)) {
            self.errors.push(format!("{n}: present, but {why}"));
        }
    }

    fn hyper(&mut self, p: &str, c: &FlashNextConfig) {
        let wide = c.hc_streams * c.hidden;
        for side in ["attn", "mlp"] {
            let q = format!("{p}.{side}_hyper_connection");
            self.plain(&format!("{q}.hc_norm.weight"), &[wide]);
            self.plain(&format!("{q}.block_inject_weight.weight"), &[c.hc_streams, wide]);
            self.plain(&format!("{q}.input_mix_weight_down.weight"), &[c.hc_lowrank, wide]);
            self.plain(&format!("{q}.input_mix_weight_up.weight"), &[wide, c.hc_lowrank]);
        }
    }

    fn mixer(&mut self, p: &str, mixer: Mixer, c: &FlashNextConfig) {
        let h = c.hidden;
        match mixer {
            Mixer::Gqa => {
                let (q, kv, ix) = (c.heads * c.head_dim, c.kv_heads * c.head_dim, c.indexer_head_dim);
                self.group(&format!("{p}.self_attn.q_proj"), h, 2 * q); // queries and their output gate
                self.group(&format!("{p}.self_attn.k_proj"), h, kv);
                self.group(&format!("{p}.self_attn.v_proj"), h, kv);
                self.group(&format!("{p}.self_attn.o_proj"), q, h);
                self.plain(&format!("{p}.self_attn.q_norm.weight"), &[c.head_dim]);
                self.plain(&format!("{p}.self_attn.k_norm.weight"), &[c.head_dim]);
                self.group(&format!("{p}.self_attn.indexer.index_qk_proj"), h, (c.indexer_heads + c.indexer_kv_heads) * ix);
                self.plain(&format!("{p}.self_attn.indexer.q_layernorm.weight"), &[ix]);
                self.plain(&format!("{p}.self_attn.indexer.k_layernorm.weight"), &[ix]);
                self.absent(&format!("{p}.linear_attn."), "the layer is full attention");
            }
            Mixer::Gdn => {
                let (v, ch, vh) = (c.lin_value_heads * c.lin_value_dim, c.conv_channels(), c.lin_value_heads);
                self.group(&format!("{p}.linear_attn.in_proj_qkv"), h, ch);
                self.group(&format!("{p}.linear_attn.in_proj_z"), h, v);
                self.group(&format!("{p}.linear_attn.out_proj"), v, h);
                self.plain(&format!("{p}.linear_attn.in_proj_a.weight"), &[vh, h]);
                self.plain(&format!("{p}.linear_attn.in_proj_b.weight"), &[vh, h]);
                self.plain(&format!("{p}.linear_attn.A_log"), &[vh]);
                self.plain(&format!("{p}.linear_attn.dt_bias"), &[vh]);
                self.plain(&format!("{p}.linear_attn.conv1d.weight"), &[ch, 1, c.lin_conv_kernel]);
                self.plain(&format!("{p}.linear_attn.norm.weight"), &[c.lin_value_dim]);
                self.absent(&format!("{p}.self_attn."), "the layer is a gated delta net");
            }
            other => self.errors.push(format!("{p}: mixer {other:?} is not part of Flash-Next")),
        }
    }

    fn moe(&mut self, p: &str, c: &FlashNextConfig) {
        let (h, i, si) = (c.hidden, c.moe_intermediate, c.shared_intermediate);
        self.plain(&format!("{p}.mlp.gate.weight"), &[c.experts, h]);
        for e in 0..c.experts {
            self.group(&format!("{p}.mlp.experts.{e}.gate_proj"), h, i);
            self.group(&format!("{p}.mlp.experts.{e}.up_proj"), h, i);
            self.group(&format!("{p}.mlp.experts.{e}.down_proj"), i, h);
        }
        self.absent(&format!("{p}.mlp.experts.{}.", c.experts), "the config has fewer experts");
        self.group(&format!("{p}.mlp.shared_expert.gate_proj"), h, si);
        self.group(&format!("{p}.mlp.shared_expert.up_proj"), h, si);
        self.group(&format!("{p}.mlp.shared_expert.down_proj"), si, h);
        self.plain(&format!("{p}.mlp.shared_expert_gate.weight"), &[1, h]);
    }

    /// The PLE stage's weights and the n-gram table; returns the table's shard count and code width (bits).
    fn ple(&mut self, p: &str, c: &FlashNextConfig) -> Option<(usize, usize)> {
        let (h, wide, heads) = (c.hidden, c.hc_streams * c.hidden, (c.ngram_size - 1) * c.heads_per_ngram);
        self.plain(&format!("{p}.ple.key_proj.weight"), &[wide, c.ple_embed_dim]);
        self.plain(&format!("{p}.ple.value_proj.weight"), &[h, c.ple_embed_dim]);
        self.plain(&format!("{p}.ple.conv1d.weight"), &[wide, 1, c.ple_conv_kernel]);
        for n in ["norm_key", "norm_query", "norm_conv"] {
            self.plain(&format!("{p}.ple.{n}.weight"), &[wide]);
        }
        let t = format!("{p}.ple.ple_embedding.ngram_embedding");
        self.plain(&format!("{t}.head_bias"), &[heads, c.ple_embed_dim / heads]);
        self.plain(&format!("{t}.head_offsets"), &[heads]);
        self.plain(&format!("{t}.head_vocab_sizes"), &[heads]);
        self.plain(&format!("{t}.layer_multipliers"), &[c.ngram_size]);
        // the shards: one width for all, a whole number of bits per code
        let shards: Vec<&Vec<usize>> =
            (0..).map_while(|i| self.t.get(&format!("{p}.{NGRAM_SHARD}{i}.trellis")).map(|(_, s)| s)).collect();
        for i in 0..shards.len() {
            self.claimed.insert(format!("{p}.{NGRAM_SHARD}{i}.trellis"));
        }
        let words = shards.first().map(|s| s.get(1).copied().unwrap_or(0));
        let bits = words.and_then(|w| (1..=8).find(|&b| ngram_row_words(b) == w));
        match (shards.len(), bits) {
            (0, _) => self.errors.push(format!("{p}.{NGRAM_SHARD}0.trellis: missing (the n-gram table, {NGRAM_FILE})")),
            (_, None) => self.errors.push(format!("{p}.{NGRAM_SHARD}0.trellis: rows of {words:?} words are not 1 + 160 K / 16")),
            (n, Some(_)) if shards.iter().any(|s| s.len() != 2 || Some(s[1]) != words) => {
                self.errors.push(format!("{p}.{NGRAM_SHARD}*: the {n} shards differ in row width"))
            }
            _ => {}
        }
        Some((shards.len(), bits?))
    }
}

/// What a pack holding `tensors` lacks, or holds in the wrong shape, for the recipe of `cfg`. Empty when they agree.
pub fn check(cfg: &FlashNextConfig, tensors: &Tensors) -> Vec<String> {
    check_with_claims(cfg, tensors).0
}

/// Validate the recipe and return every text tensor claimed by its unique slot. Vision tensors belong to a separate component.
pub fn check_with_claims(cfg: &FlashNextConfig, tensors: &Tensors) -> (Vec<String>, std::collections::BTreeSet<String>) {
    let mut ck = Check { t: tensors, errors: Vec::new(), claimed: Default::default() };
    let (h, lm) = (cfg.hidden, "model.language_model");
    ck.plain(&format!("{lm}.embed_tokens.weight"), &[cfg.vocab, h]);
    ck.group("lm_head", h, cfg.vocab);
    let wide = cfg.hc_streams * h;
    for mix in [format!("{lm}.hyper_connection_mixer"), "mtp.hyper_connection_mixer".to_string()]
        .iter()
        .take(1 + (cfg.mtp_layers > 0) as usize)
    {
        ck.plain(&format!("{mix}.hc_norm.weight"), &[wide]);
        ck.plain(&format!("{mix}.input_mix_weight_down.weight"), &[cfg.hc_lowrank, wide]);
        ck.plain(&format!("{mix}.input_mix_weight_up.weight"), &[wide, cfg.hc_lowrank]);
    }
    let ple_layer = cfg.ple_layer_ids.first().map(|&i| i - 1);
    for (i, spec) in cfg.layer_specs().iter().enumerate() {
        let p = format!("{lm}.layers.{i}");
        ck.mixer(&p, spec.mixer, cfg);
        ck.moe(&p, cfg);
        ck.hyper(&p, cfg);
        if Some(i) == ple_layer {
            ck.ple(&p, cfg);
        } else {
            ck.absent(&format!("{p}.ple."), "the config puts the PLE stage on another layer");
        }
    }
    for (j, spec) in cfg.mtp_specs().iter().enumerate() {
        let p = format!("mtp.layers.{j}");
        ck.mixer(&p, spec.mixer, cfg);
        ck.moe(&p, cfg);
        ck.hyper(&p, cfg);
    }
    if cfg.mtp_layers > 0 {
        ck.group("mtp.fc_embedding", h, h);
        ck.group("mtp.fc_hidden", h, h);
        ck.plain("mtp.pre_fc_norm_embedding.weight", &[h]);
        ck.plain("mtp.pre_fc_norm_hidden.weight", &[wide]);
    }
    ck.absent(&format!("{lm}.layers.{}.", cfg.layers), "the config has fewer layers");
    ck.absent(&format!("mtp.layers.{}.", cfg.mtp_layers), "the config has fewer MTP layers");
    (ck.errors, ck.claimed)
}

/// The n-gram table of a pack that passed [`check`]: (shards, code bits).
pub fn ngram_table(cfg: &FlashNextConfig, tensors: &Tensors) -> Option<(usize, usize)> {
    let p = format!("model.language_model.layers.{}", cfg.ple_layer_ids.first()? - 1);
    Check { t: tensors, errors: Vec::new(), claimed: Default::default() }.ple(&p, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::qwen_flash_next::config::tests::FLASH_NEXT_CONFIG;

    pub const TABLE: &str = include_str!("../../../../../testdata/qwen_flash_next/tensor_inventory.tsv");

    fn cfg() -> FlashNextConfig {
        FlashNextConfig::from_json(&serde_json::from_str(FLASH_NEXT_CONFIG).unwrap()).unwrap()
    }

    #[test]
    fn the_committed_inventory_matches_the_recipe_of_its_config() {
        let t = expand_table(TABLE).unwrap();
        assert_eq!(t.len(), 304_240);
        let errors = check(&cfg(), &t);
        assert!(errors.is_empty(), "{:#?}", &errors[..errors.len().min(10)]);
        // the table: 128 shards of 2,500,012 rows at 5 bits (320,001,536 rows of 102 B, 32.6 GB)
        assert_eq!(ngram_table(&cfg(), &t), Some((128, 5)));
        let rows: usize = (0..128).map(|i| t[&format!("model.language_model.layers.1.{NGRAM_SHARD}{i}.trellis")].1[0]).sum();
        assert_eq!(rows, 320_001_536);
        // and GLM's reader accepts the same table: every EXL3 linear parses, the only refusals are the table's shards
        let inv = crate::families::glm::PackInventory::from_table(TABLE).unwrap();
        assert_eq!(inv.refused.len(), 128);
        assert!(inv.refused.keys().all(|n| n.contains(NGRAM_SHARD)));
    }

    /// Each edit of the table is refused by naming the tensor it broke.
    #[test]
    fn a_pack_that_disagrees_with_its_config_is_refused_by_name() {
        let edit = |f: &dyn Fn(&str) -> String| -> Vec<String> {
            let text: String = TABLE.lines().map(|l| f(l) + "\n").collect();
            check(&cfg(), &expand_table(&text).unwrap())
        };
        let has = |errs: &[String], needle: &str| errs.iter().any(|e| e.contains(needle));
        // the indexer of layer 47 missing
        let e = edit(&|l| if l.contains("indexer.") { l.replace(",43,47", ",43") } else { l.to_string() });
        assert!(has(&e, "layers.47.self_attn.indexer.index_qk_proj: missing"), "{e:?}");
        // one expert fewer everywhere
        let e = edit(&|l| if l.starts_with("experts ") { "experts 511".into() } else { l.to_string() });
        assert!(has(&e, "layers.0.mlp.experts.511.gate_proj: missing"), "{e:?}");
        // the PLE stage on the wrong layer
        let e = edit(&|l| match l.strip_suffix("\t1") {
            Some(head) if l.contains(".ple.") => format!("{head}\t2"),
            _ => l.to_string(),
        });
        assert!(has(&e, "layers.1.ple.key_proj.weight: missing") && has(&e, "layers.2.ple."), "{e:?}");
        // a misshapen hyper-connection rank
        let e = edit(&|l| {
            if l.starts_with("model.language_model.layers.{L}.attn_hyper_connection.input_mix_weight_down") {
                l.replace("320,10240", "256,10240")
            } else {
                l.to_string()
            }
        });
        assert!(has(&e, "layers.0.attn_hyper_connection.input_mix_weight_down.weight: shape [256, 10240]"), "{e:?}");
        // an attention projection of the wrong width
        let e =
            edit(&|l| if l.contains("self_attn.k_proj.trellis") { l.replace("160,32,80", "160,64,80") } else { l.to_string() });
        assert!(has(&e, "layers.3.self_attn.k_proj.trellis: shape [160, 64, 80]"), "{e:?}");
    }
}
