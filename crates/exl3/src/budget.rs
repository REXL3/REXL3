//! Bytes a decode step must read, which bounds speed on a bandwidth-limited box.
//!
//! A mixture-of-experts model reads every non-expert weight on every row but only the routed experts a row
//! selects. Counting all experts (as a dense model would) overstates the traffic by roughly
//! `n_routed / top_k` and gives a ceiling an order of magnitude too low.

use serde_json::Value;

/// What a tensor is, for byte accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Category {
    Vision,
    /// A multi-token-prediction / NextN layer: read only when drafting with it.
    Mtp,
    Head,
    Embedding,
    /// One routed expert's projection: read only when the router picks that expert.
    RoutedExpert,
    /// Decoder weights read on every row: attention, dense and shared-expert MLPs, router, norms.
    Decoder,
    Other,
}

impl Category {
    pub fn label(&self) -> &'static str {
        match self {
            Category::Vision => "vision",
            Category::Mtp => "mtp",
            Category::Head => "head",
            Category::Embedding => "embedding",
            Category::RoutedExpert => "routed-experts",
            Category::Decoder => "decoder",
            Category::Other => "other",
        }
    }
}

/// The index after `layers.` in a tensor name, if any.
fn layer_index(name: &str) -> Option<usize> {
    let rest = name.split("layers.").nth(1)?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// `.experts.<number>.` marks one routed expert (a shared expert is `shared_experts`, which does not match).
fn is_routed_expert(name: &str) -> bool {
    name.match_indices(".experts.").any(|(i, m)| name[i + m.len()..].chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false))
}

/// Classify a tensor or group name. `num_layers` is the text model's depth from config.json: a decoder-style
/// tensor at or past it (GLM stores its NextN layer as `layers.<num_layers>`) belongs to the MTP head.
pub fn categorize(name: &str, num_layers: Option<usize>) -> Category {
    let l = name.to_ascii_lowercase();
    if l.contains("visual") || l.contains("vision") {
        return Category::Vision;
    }
    if l.starts_with("mtp") || l.contains(".mtp.") || l.contains("nextn") {
        return Category::Mtp;
    }
    if l.contains("lm_head") {
        return Category::Head;
    }
    if l.contains("embed_tokens") {
        return Category::Embedding;
    }
    if l.contains("layers.") {
        if let (Some(i), Some(n)) = (layer_index(&l), num_layers) {
            if i >= n {
                return Category::Mtp;
            }
        }
        return if is_routed_expert(&l) { Category::RoutedExpert } else { Category::Decoder };
    }
    Category::Other
}

/// Mixture-of-experts shape from a model's config.json (top level or `text_config`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeShape {
    pub n_routed: u64,
    pub top_k: u64,
}

fn text_field<'a>(config: &'a Value, key: &str) -> Option<&'a Value> {
    config.get("text_config").and_then(|t| t.get(key)).or_else(|| config.get(key))
}

pub fn moe_shape(config: &Value) -> Option<MoeShape> {
    let n_routed = text_field(config, "n_routed_experts").or_else(|| text_field(config, "num_experts"))?.as_u64()?;
    let top_k = text_field(config, "num_experts_per_tok")?.as_u64()?;
    (n_routed > 0 && top_k > 0 && top_k <= n_routed).then_some(MoeShape { n_routed, top_k })
}

pub fn num_layers(config: &Value) -> Option<usize> {
    text_field(config, "num_hidden_layers")?.as_u64().and_then(|n| usize::try_from(n).ok())
}

impl MoeShape {
    /// Expected fraction of a layer's routed experts touched by `rows` independent rows, if each row picks
    /// `top_k` of `n_routed` uniformly at random. Real routing is correlated across neighbouring tokens, so
    /// the true fraction is lower: treat this as an upper estimate, and measure.
    pub fn touched_fraction(&self, rows: u64) -> f64 {
        let miss = 1.0 - self.top_k as f64 / self.n_routed as f64;
        1.0 - miss.powf(rows as f64)
    }
}

/// Bytes on disk per category, as `inspect` sums them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bytes {
    pub routed: u64,
    pub decoder: u64,
    pub head: u64,
    pub other: u64,
    pub mtp: u64,
    pub embedding: u64,
    pub vision: u64,
}

impl Bytes {
    pub fn add(&mut self, c: Category, n: u64) {
        let slot = match c {
            Category::RoutedExpert => &mut self.routed,
            Category::Decoder => &mut self.decoder,
            Category::Head => &mut self.head,
            Category::Other => &mut self.other,
            Category::Mtp => &mut self.mtp,
            Category::Embedding => &mut self.embedding,
            Category::Vision => &mut self.vision,
        };
        *slot = slot.saturating_add(n);
    }

    /// Weights read on every row: decoder, head and other. One embedding row is negligible; vision runs per
    /// image; the MTP head runs only when it drafts.
    pub fn always_read(&self) -> u64 {
        self.decoder.saturating_add(self.head).saturating_add(self.other)
    }

    /// Every weight of a dense model, or all experts of a MoE read as if each row hit every expert.
    pub fn all_experts_bound(&self) -> u64 {
        self.always_read().saturating_add(self.routed)
    }

    /// Bytes one step of `rows` rows reads: the always-read weights plus the routed experts the rows touch
    /// (uniform-routing estimate). Without a MoE shape every byte counts.
    pub fn step_bytes(&self, moe: Option<MoeShape>, rows: u64) -> f64 {
        match moe {
            Some(m) => self.always_read() as f64 + self.routed as f64 * m.touched_fraction(rows),
            None => self.all_experts_bound() as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn classification_follows_the_glm_pack_naming() {
        let n = Some(45);
        assert_eq!(categorize("model.language_model.layers.7.mlp.experts.12.gate_proj", n), Category::RoutedExpert);
        assert_eq!(categorize("model.language_model.layers.7.mlp.shared_experts.gate_proj", n), Category::Decoder);
        assert_eq!(categorize("model.language_model.layers.7.self_attn.qkv_proj", n), Category::Decoder);
        assert_eq!(categorize("model.language_model.layers.44.mlp.experts.0.down_proj", n), Category::RoutedExpert);
        // the NextN layer is stored as layers.45, past the 45 decoder layers, and its experts are MTP bytes too
        assert_eq!(categorize("model.language_model.layers.45.eh_proj", n), Category::Mtp);
        assert_eq!(categorize("model.language_model.layers.45.mlp.experts.3.up_proj", n), Category::Mtp);
        assert_eq!(categorize("model.visual.blocks.0.attn.qkv", n), Category::Vision);
        assert_eq!(categorize("lm_head", n), Category::Head);
        assert_eq!(categorize("model.language_model.embed_tokens", n), Category::Embedding);
        assert_eq!(categorize("model.language_model.norm", n), Category::Other);
        // without a config, layers.45 cannot be told from a decoder layer
        assert_eq!(categorize("model.language_model.layers.45.eh_proj", None), Category::Decoder);
        // an expert id above 9 and a name that merely contains "experts" are handled
        assert!(!is_routed_expert("layers.1.mlp.experts.gate_up"));
        assert!(is_routed_expert("layers.1.mlp.experts.255.down_proj"));
    }

    #[test]
    fn moe_shape_reads_text_config_or_top_level() {
        let nested = json!({"text_config": {"n_routed_experts": 288, "num_experts_per_tok": 8, "num_hidden_layers": 45}});
        assert_eq!(moe_shape(&nested), Some(MoeShape { n_routed: 288, top_k: 8 }));
        assert_eq!(num_layers(&nested), Some(45));
        let flat = json!({"num_experts": 256, "num_experts_per_tok": 8});
        assert_eq!(moe_shape(&flat), Some(MoeShape { n_routed: 256, top_k: 8 }));
        assert_eq!(moe_shape(&json!({"hidden_size": 4096})), None);
        assert_eq!(
            moe_shape(&json!({"n_routed_experts": 4, "num_experts_per_tok": 9})),
            None,
            "top_k above the expert count is nonsense"
        );
    }

    #[test]
    fn touched_fraction_behaves() {
        let m = MoeShape { n_routed: 288, top_k: 8 };
        assert!((m.touched_fraction(1) - 8.0 / 288.0).abs() < 1e-12);
        assert_eq!(m.touched_fraction(0), 0.0);
        let mut last = 0.0;
        for r in 1..200 {
            let f = m.touched_fraction(r);
            assert!(f > last && f < 1.0);
            last = f;
        }
        // six independent rows touch about 15.6% of the experts, the figure behind the verify-cost estimate
        assert!((m.touched_fraction(6) - 0.1559).abs() < 5e-4);
    }

    #[test]
    fn a_moe_step_reads_far_less_than_all_experts() {
        let b =
            Bytes { routed: 87_400_000_000, decoder: 3_000_000_000, head: 476_000_000, other: 100_000_000, ..Default::default() };
        let m = Some(MoeShape { n_routed: 288, top_k: 8 });
        let one = b.step_bytes(m, 1);
        assert!(one < b.all_experts_bound() as f64 / 10.0, "one row should read under a tenth of everything");
        assert!(b.step_bytes(m, 6) > 2.0 * one && b.step_bytes(m, 6) < 6.0 * one);
        assert_eq!(b.step_bytes(None, 1), b.all_experts_bound() as f64);
    }
}
