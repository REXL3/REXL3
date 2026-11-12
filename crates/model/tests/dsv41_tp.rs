//! The CPU gates of the DeepSeek-V4.1-Flash tensor-parallel seam (`families::deepseek_v41::tp`), the mirror of the
//! nemotron TP checks (`families::nemotron_h_omni::tp`, `tp_shard`; `docs/NEMOTRON_TP.md` states the one rule):
//!
//! * the split covers every routed expert of the trunk and of every NextN stage exactly once, at two ranks and at four
//!   (the divisibility the pack's own numbers carry: 384 -> 192/96, 128 -> 64/32, 64 heads -> 32/16);
//! * the MLA query-head range and its matching output-projection row slice are disjoint, complete and head-aligned;
//! * a rank's plan holds ONLY its own experts — an expert of another rank is absent, never read and never allocated
//!   (the mirror of nemotron's `Piece::Skip`) — and the two ranks together plan every tensor of the pack (the experts
//!   once, everything replicated on both);
//! * a group boot refuses the steps it is not verified for BY NAME;
//! * the loopback collective compiles against the table and sums one disjoint-support buffer exactly (skipped without
//!   a device, exactly as nemotron's own loopback gates are).
//!
//! Everything here runs on the CPU lane; the weights tree comes from the config-derived expected inventory
//! (`PackInventory::expected`), so no pack listing is needed.

use gb_model::families::deepseek_v41::bind::bind;
use gb_model::families::deepseek_v41::config::DeepseekV41Config;
use gb_model::families::deepseek_v41::inventory::{MtpExpertFormat, PackInventory};
use gb_model::families::deepseek_v41::plan::{LoadPlan, Placement};
use gb_model::families::deepseek_v41::tp::{check_tp_plain, splits, ShardTable};
use gb_model::families::nemotron_h_omni::tp::{TpComm, TpTopology};
use gb_model::families::nemotron_h_omni::tp_comm::LoopbackGroup;
use gb_model::serve::{ServeOptions, SpecMode};
use serde_json::Value;
use std::collections::HashSet;
use std::ops::Range;
use std::path::PathBuf;

/// The pack's `config.json`, committed as the family's fixture (the ground truth of every number here).
const CONFIG: &str = include_str!("../../../testdata/deepseek_v41/config.json");

fn real() -> DeepseekV41Config {
    DeepseekV41Config::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
}

/// A tiny pack of the real layout: two trunk layers of 8 routed experts, one NextN stage of 4, 8 MLA heads of 64, one
/// engram layer — every count divides at two and four ranks, so the split table's ranges are the real geometry at a
/// scale a test can walk.
fn toy() -> DeepseekV41Config {
    let mut v: Value = serde_json::from_str(CONFIG).unwrap();
    let t = v["text_config"].as_object_mut().unwrap();
    t.insert("num_hidden_layers".into(), 2.into());
    t.insert("num_nextn_predict_layers".into(), 1.into());
    t.insert("n_routed_experts".into(), 8.into());
    t.insert("num_experts_per_tok".into(), 2.into());
    t.insert("dspark_n_routed_experts".into(), 4.into());
    t.insert("dspark_num_experts_per_tok".into(), 2.into());
    t.insert("dspark_target_layer_ids".into(), serde_json::json!([1]));
    t.insert("compress_ratios".into(), serde_json::json!([2, 2, 2]));
    t.insert("kv_source_layer_ids".into(), serde_json::json!([0]));
    t.insert("index_source_layer_ids".into(), serde_json::json!([0, 1]));
    t.insert("candidate_source_layer_id".into(), 0.into());
    t.insert("engram_layer_ids".into(), serde_json::json!([1]));
    t.insert("engram_num_embeddings".into(), serde_json::json!([4096]));
    t.insert("num_attention_heads".into(), 8.into());
    t.insert("head_dim".into(), 64.into());
    t.insert("qk_rope_head_dim".into(), 8.into());
    t.insert("q_lora_rank".into(), 64.into());
    t.insert("o_lora_rank".into(), 32.into());
    t.insert("moe_intermediate_size".into(), 128.into());
    v.as_object_mut().unwrap().remove("vision_config");
    DeepseekV41Config::from_json(&v).unwrap()
}

/// The toy config's bound weights tree, from the config-derived expected inventory: the same code path the real
/// 188,245-tensor pack takes, without the pack.
fn toy_weights(cfg: &DeepseekV41Config) -> gb_model::families::deepseek_v41::weights::DeepseekV41Weights {
    let expected = PackInventory::expected(cfg, MtpExpertFormat::Int4).unwrap();
    let inv = PackInventory::from_rows(cfg, expected.rows(1)).unwrap();
    bind(&inv, cfg).unwrap()
}

fn tables(ranks: usize, cfg: &DeepseekV41Config) -> Vec<ShardTable> {
    (0..ranks).map(|rank| ShardTable::for_config(TpTopology { rank, ranks }, cfg).unwrap()).collect()
}

/// The routed-expert index of a plan item's name, and the bank it belongs to: `layers.{l}.ffn.experts.{e}...` is a
/// trunk expert, `mtp.{k}.ffn.experts.{e}...` a NextN one. The shared experts, routers and everything else are `None`.
fn expert_of(name: &str) -> Option<(&'static str, usize)> {
    let (bank, tail) = name.split_once(".ffn.experts.")?;
    let head = if bank.starts_with("layers.") {
        "trunk"
    } else if bank.starts_with("mtp.") {
        "nextn"
    } else {
        return None;
    };
    Some((head, tail.split('.').next()?.parse().ok()?))
}

/// Every range of `ranges` is contiguous with the next, disjoint, and together they cover `0..total` exactly once.
fn covers_exactly_once(ranges: &[Range<usize>], total: usize, what: &str) {
    let mut at = 0usize;
    let mut held = vec![0usize; total];
    for (r, range) in ranges.iter().enumerate() {
        assert_eq!(range.start, at, "{what}: rank {r}'s range starts where the previous ended");
        for e in range.clone() {
            held[e] += 1;
        }
        at = range.end;
    }
    assert_eq!(at, total, "{what}: the ranks cover every one of {total}");
    assert!(held.iter().all(|&n| n == 1), "{what}: no expert is held twice");
}

// ------------------------------------------------------------------ the split table

/// The splits constants ARE the pack's config: the ground truth the table cuts.
#[test]
fn the_split_constants_are_the_packs_config() {
    let c = real();
    assert_eq!(splits::TRUNK_LAYERS, c.layers);
    assert_eq!(splits::TRUNK_EXPERTS, c.moe.routed_experts);
    assert_eq!(splits::NEXTN_STAGES, c.nextn);
    assert_eq!(splits::NEXTN_EXPERTS, c.dspark.n_routed_experts);
    assert_eq!((splits::Q_HEADS, splits::HEAD_DIM), (c.heads, c.head_dim));
    assert_eq!(splits::VOCAB, c.vocab);
    // and the pack's own numbers divide at the rank counts this seam is built for
    for ranks in [2usize, 4] {
        for n in [splits::TRUNK_EXPERTS, splits::NEXTN_EXPERTS, splits::Q_HEADS] {
            assert_eq!(n % ranks, 0);
        }
    }
}

#[test]
fn the_split_covers_every_trunk_and_nextn_expert_exactly_once_at_2_and_4_ranks() {
    let c = real();
    for ranks in [2usize, 4] {
        let ts = tables(ranks, &c);
        // the trunk: 384 experts a layer, [384 r / R, 384 (r+1) / R) a rank
        covers_exactly_once(
            &ts.iter().map(|t| t.trunk_expert_range()).collect::<Vec<_>>(),
            splits::TRUNK_EXPERTS,
            "the trunk's routed experts",
        );
        // the NextN stages: 128 experts each, the same rule
        covers_exactly_once(
            &ts.iter().map(|t| t.nextn_expert_range()).collect::<Vec<_>>(),
            splits::NEXTN_EXPERTS,
            "a NextN stage's routed experts",
        );
        // every rank the SAME width (the counts divide: no skew at all)
        let width = splits::TRUNK_EXPERTS / ranks;
        assert!(ts.iter().all(|t| t.trunk_expert_range().len() == width));
        assert!(ts.iter().all(|t| t.nextn_expert_range().len() == splits::NEXTN_EXPERTS / ranks));
        // ownership agrees with the ranges
        for (r, t) in ts.iter().enumerate() {
            for e in 0..splits::TRUNK_EXPERTS {
                assert_eq!(t.owns_trunk_expert(e), t.trunk_expert_range().contains(&e), "trunk expert {e} at rank {r}");
            }
            for e in 0..splits::NEXTN_EXPERTS {
                assert_eq!(t.owns_nextn_expert(e), t.nextn_expert_range().contains(&e), "NextN expert {e} at rank {r}");
            }
        }
    }
    // the pinned two-rank numbers the seam is named for
    let [a, b] = tables(2, &c).try_into().unwrap();
    assert_eq!(a.trunk_expert_range(), 0..192);
    assert_eq!(b.trunk_expert_range(), 192..384);
    assert_eq!(a.nextn_expert_range(), 0..64);
    assert_eq!(b.nextn_expert_range(), 64..128);
    // and the single rank holds everything
    let single = ShardTable::single();
    assert!(single.is_single());
    assert_eq!(single.trunk_expert_range(), 0..splits::TRUNK_EXPERTS);
    assert_eq!(single.nextn_expert_range(), 0..splits::NEXTN_EXPERTS);
    assert!(single.owns_trunk_expert(383) && single.owns_nextn_expert(127));
}

#[test]
fn the_q_head_and_o_row_ranges_are_disjoint_complete_and_head_aligned() {
    let c = real();
    let rows = splits::Q_HEADS * splits::HEAD_DIM;
    assert_eq!(rows, 32_768, "o rows = heads x head_dim");
    for ranks in [2usize, 4] {
        let ts = tables(ranks, &c);
        covers_exactly_once(&ts.iter().map(|t| t.q_head_range()).collect::<Vec<_>>(), splits::Q_HEADS, "the MLA query heads");
        covers_exactly_once(&ts.iter().map(|t| t.o_row_range()).collect::<Vec<_>>(), rows, "the o-projection rows");
        for t in &ts {
            // the o rows are the rank's heads' rows: one head_dim block a head, in the same order
            let heads = t.q_head_range();
            assert_eq!(t.o_row_range(), heads.start * splits::HEAD_DIM..heads.end * splits::HEAD_DIM);
            // every boundary is head-aligned, a multiple of 128 (the EXL3 trellis tile / output Hadamard block, should
            // an EXL3 form of these matrices ever be cut) and of 32 (the FP8 block scale's row)
            for edge in [t.o_row_range().start, t.o_row_range().end] {
                assert_eq!(edge % splits::HEAD_DIM, 0);
                assert_eq!(edge % 128, 0);
                assert_eq!(edge % 32, 0);
            }
        }
    }
    let [a, b] = tables(2, &c).try_into().unwrap();
    assert_eq!(a.q_head_range(), 0..32);
    assert_eq!(b.q_head_range(), 32..64);
    assert_eq!(a.o_row_range(), 0..16_384);
    assert_eq!(b.o_row_range(), 16_384..32_768);
    // the single rank computes every head
    assert_eq!(ShardTable::single().q_head_range(), 0..splits::Q_HEADS);
    assert_eq!(ShardTable::single().o_row_range(), 0..rows);
}

/// A config whose counts do not divide is refused by name, at build of the table, not later in a debug assert.
#[test]
fn a_config_that_does_not_divide_is_refused_by_name() {
    let mut c = real();
    c.moe.routed_experts += 1; // 385 does not divide by 2 (or 4)
    let e = ShardTable::for_config(TpTopology { rank: 0, ranks: 2 }, &c).unwrap_err();
    assert!(e.contains("385") && e.contains("does not divide"), "{e}");
    c.moe.routed_experts -= 1;
    c.heads = 65; // the heads do not divide either
    let e = ShardTable::for_config(TpTopology { rank: 0, ranks: 2 }, &c).unwrap_err();
    assert!(e.contains("65") && e.contains("does not divide"), "{e}");
    // a rank past the end of the group is refused too
    let e = ShardTable::for_config(TpTopology { rank: 2, ranks: 2 }, &real()).unwrap_err();
    assert!(e.contains("rank"), "{e}");
    // the toy config divides at both rank counts (the plan tests build on it)
    for ranks in [2usize, 4] {
        assert!(ShardTable::for_config(TpTopology { rank: 0, ranks }, &toy()).is_ok());
    }
}

// ------------------------------------------------------------------ the rank-scoped plan

/// A rank plans ONLY its own experts: the other ranks' routed experts are absent — never read, never allocated — and
/// the two ranks together plan every tensor of the pack exactly once where it is cut and twice where it replicates.
#[test]
fn a_rank_plans_only_its_own_experts_and_the_pair_plans_the_whole_pack() {
    let c = toy();
    let w = toy_weights(&c);
    let single = LoadPlan::build(&w, 0).unwrap();
    let [t0, t1] = tables(2, &c).try_into().unwrap();
    let p0 = LoadPlan::build_tp(&w, 0, &t0).unwrap();
    let p1 = LoadPlan::build_tp(&w, 0, &t1).unwrap();

    let (n0, n1) = (names_of(&p0), names_of(&p1));
    // per-tensor disposition: a routed expert on exactly one rank (the owner's), everything else on both
    let mut experts_once = 0usize;
    let mut replicated = 0usize;
    for i in &single.items {
        let (in0, in1) = (n0.contains(i.name.as_str()), n1.contains(i.name.as_str()));
        match expert_of(&i.name) {
            Some(_) => {
                assert!(in0 ^ in1, "{}: a routed expert is planned by exactly one rank (in0 {in0}, in1 {in1})", i.name);
                experts_once += 1;
            }
            None => {
                assert!(in0 && in1, "{}: a replicated tensor is planned by both ranks", i.name);
                replicated += 1;
            }
        }
    }
    // every item fell in one bucket, and the buckets are the whole tree
    assert_eq!(expert_items(&single), experts_once);
    assert_eq!(single.items.len(), experts_once + replicated);
    // absence BY NAME (rank 0 of 2 owns trunk [0, 4) of 8 and nextn [0, 2) of 4)
    assert!(n0.contains("layers.0.ffn.experts.3.w1.trellis"));
    assert!(!n0.contains("layers.0.ffn.experts.4.w1.trellis"), "an expert outside the range is never planned");
    assert!(!n0.contains("layers.1.ffn.experts.7.w2.mul1"));
    assert!(n0.contains("mtp.0.ffn.experts.1.w3.weight"));
    assert!(!n0.contains("mtp.0.ffn.experts.2.w1.weight"), "a NextN expert outside the range is never planned");
    // and rank 1 the mirror
    assert!(n1.contains("layers.0.ffn.experts.4.w1.trellis"));
    assert!(!n1.contains("layers.0.ffn.experts.3.w1.trellis"));
    assert!(n1.contains("mtp.0.ffn.experts.2.w1.weight"));
    // the replicated tensors the design names are on both ranks: the router (gate) with its biases, the shared
    // experts, the engram sidecar, the embedding, the head, the latent path (wkv, wq_a)
    for name in [
        "layers.0.ffn.gate.weight",
        "layers.0.ffn.gate.bias_vl",
        "layers.0.ffn.shared_experts.w1.weight",
        "layers.1.engram.embed.weight",
        "layers.1.engram.q_weight",
        "embed.weight",
        "head.weight",
        "layers.0.attn.wkv.weight",
        "layers.0.attn.wq_a.weight",
    ] {
        let listed = n0.contains(name) && n1.contains(name);
        assert!(listed || single.items.iter().all(|i| !i.name.starts_with(name)), "{name}: on both ranks when the pack has it");
    }
    // every expert item of a rank is an owned one, in the placements the single plan uses
    for (p, t) in [(&p0, &t0), (&p1, &t1)] {
        for i in &p.items {
            if let Some((bank, e)) = expert_of(&i.name) {
                match bank {
                    "trunk" => assert!(t.owns_trunk_expert(e), "{}: rank holds an expert outside its range", i.name),
                    "nextn" => assert!(t.owns_nextn_expert(e), "{}: rank holds a NextN expert outside its range", i.name),
                    _ => unreachable!(),
                }
                assert_eq!(
                    single.items.iter().find(|s| s.name == i.name).unwrap().placement,
                    i.placement,
                    "{}: the placement is the single plan's",
                    i.name
                );
            }
        }
    }
    // the single-rank table IS the single-node plan: every tensor, every placement, every total
    assert_eq!(LoadPlan::build_tp(&w, 0, &ShardTable::single()).unwrap(), single);
    // the bytes reconcile: the experts once across the pair, the rest on both
    assert_eq!(p0.expert_bytes + p1.expert_bytes, single.expert_bytes);
    assert_eq!(p0.mtp_expert_bytes + p1.mtp_expert_bytes, single.mtp_expert_bytes);
    assert_eq!(
        (p0.resident_bytes, p0.sidecar_bytes),
        (single.resident_bytes, single.sidecar_bytes),
        "the resident set and the mmap sidecar replicate"
    );
    assert_eq!(
        p0.total_bytes() + p1.total_bytes(),
        single.total_bytes() + single.resident_bytes + single.sidecar_bytes,
        "two ranks = the pack plus its replicated tensors once more"
    );
    assert_eq!(p0.items.len() + p1.items.len(), single.items.len() + replicated, "the item counts agree with the dispositions");
}

/// The plan's tensor names, for membership checks.
fn names_of(p: &LoadPlan) -> HashSet<&str> {
    p.items.iter().map(|i| i.name.as_str()).collect()
}

fn expert_items(single: &LoadPlan) -> usize {
    single.items.iter().filter(|i| expert_of(&i.name).is_some()).count()
}

/// The budget buys whole experts of the rank's own range only: `full_tp` makes every OWNED expert resident and never
/// the other half, and the pair's budgets sum to the single rank's.
#[test]
fn the_expert_budget_buys_only_the_ranks_own_experts() {
    let c = toy();
    let w = toy_weights(&c);
    let single = LoadPlan::full(&w).unwrap();
    let [t0, t1] = tables(2, &c).try_into().unwrap();
    let f0 = LoadPlan::full_tp(&w, &t0).unwrap();
    let f1 = LoadPlan::full_tp(&w, &t1).unwrap();
    // the toy: 2 layers x 8 experts, rank r owns 4 a layer
    assert_eq!((f0.resident_experts, f1.resident_experts), (8, 8));
    assert_eq!(single.resident_experts, 16);
    assert_eq!(f0.resident_experts + f1.resident_experts, single.resident_experts);
    assert_eq!(f0.expert_budget_bytes + f1.expert_budget_bytes, single.expert_budget_bytes);
    assert_eq!(f0.streamed_expert_bytes, 0);
    assert_eq!(f0.expert_bytes + f1.expert_bytes, single.expert_bytes);
    // every resident expert of a rank is one it owns; the other half never appears, in any placement
    for (f, t) in [(&f0, &t0), (&f1, &t1)] {
        for i in f.items_in(Placement::ExpertCache) {
            let (_, e) = expert_of(&i.name).expect("an expert-cache item is an expert's");
            assert!(t.owns_trunk_expert(e), "{}: resident outside the range", i.name);
        }
    }
    // a budget over the rank's own records is still an error naming the numbers (the rank's available bytes)
    let e = LoadPlan::build_tp(&w, single.expert_budget_bytes + 1, &t0).unwrap_err();
    assert!(matches!(e, gb_model::families::deepseek_v41::plan::PlanError::ExpertBudget { .. }));
}

// ------------------------------------------------------------------ the rank-scoped layout and admission

/// A rank's layout lays out ONLY its own experts — the peer's get no region, no view and no bytes — and the plan and the
/// device still agree item by item (`check_agreement` passes at both ranks, which is the invariant that makes the
/// rank-scoped load honest). The single-rank table gives `Layout::new`'s own layout, exactly.
#[test]
fn a_rank_layout_holds_only_its_own_experts_and_agrees_with_its_plan() {
    let c = toy();
    let w = toy_weights(&c);
    let [t0, t1] = tables(2, &c).try_into().unwrap();
    let single_plan = LoadPlan::full(&w).unwrap();
    let single_layout = gb_model::families::deepseek_v41::load::Layout::new(&w, &single_plan, u64::MAX, 0).unwrap();
    for (t, rank) in [(&t0, 0), (&t1, 1)] {
        let plan = LoadPlan::full_tp(&w, t).unwrap();
        let layout = gb_model::families::deepseek_v41::load::Layout::new_tp(&w, &plan, u64::MAX, 0, Some(t)).unwrap();
        gb_model::families::deepseek_v41::load::check_agreement(&w, &plan, &layout)
            .unwrap_or_else(|e| panic!("rank {rank}: {e}"));
        // no binding, view or byte for an expert outside the range: absence is "never laid out", not "laid out empty"
        for (name, b) in &layout.bindings {
            if let Some((_, e)) = expert_of(name) {
                assert!(t.owns_trunk_expert(e) || t.owns_nextn_expert(e), "rank {rank} laid out `{name}`");
                let _ = b;
            }
        }
        assert_eq!(layout.resident_experts, plan.resident_experts);
        assert_eq!(layout.slab_bytes, plan.expert_budget_bytes);
        assert_eq!(layout.mtp_expert_bytes, plan.mtp_expert_bytes);
    }
    // the two ranks' device bytes are the single node's, plus its replicated regions once more: the Global and Trunk
    // regions are identical on every rank, and the expert regions (trunk and NextN) split exactly — every record is
    // placement-aligned, so the halves' sums are the whole's without padding skew
    let (p0, p1) = (LoadPlan::full_tp(&w, &t0).unwrap(), LoadPlan::full_tp(&w, &t1).unwrap());
    let (l0, l1) = (
        gb_model::families::deepseek_v41::load::Layout::new_tp(&w, &p0, u64::MAX, 0, Some(&t0)).unwrap(),
        gb_model::families::deepseek_v41::load::Layout::new_tp(&w, &p1, u64::MAX, 0, Some(&t1)).unwrap(),
    );
    let replicated: u64 = single_layout
        .regions
        .iter()
        .filter(|r| {
            matches!(
                r.kind,
                gb_model::families::deepseek_v41::load::RegionKind::Global
                    | gb_model::families::deepseek_v41::load::RegionKind::Trunk(_)
            )
        })
        .map(|r| r.bytes)
        .sum();
    assert_eq!(l0.device_bytes() + l1.device_bytes(), single_layout.device_bytes() + replicated);
    // the single-rank table IS the single-node layout
    let via_none = gb_model::families::deepseek_v41::load::Layout::new_tp(&w, &single_plan, u64::MAX, 0, None).unwrap();
    assert_eq!(
        (via_none.resident_experts, via_none.slab_bytes, via_none.device_bytes()),
        (single_layout.resident_experts, single_layout.slab_bytes, single_layout.device_bytes())
    );
}

/// The ADMIT path a topology drives: with `Some(table)` the probe, the slab and the admission are the rank's own — the
/// pair's expert terms sum to the single node's, every resident expert is an owned one, and the replicated terms are the
/// single node's. The toy config's experts are small enough to be admitted whole at any of these figures.
#[test]
fn the_admission_of_a_rank_prices_only_its_own_experts() {
    let c = toy();
    let w = toy_weights(&c);
    let mut o = ServeOptions::new(PathBuf::from("/nonexistent"));
    o.max_model_len = 512;
    o.slots = 1;
    let [t0, t1] = tables(2, &c).try_into().unwrap();
    let single = admit(&c, &w, &o, None);
    let a0 = admit(&c, &w, &o, Some(&t0));
    let a1 = admit(&c, &w, &o, Some(&t1));
    // the expert terms: the pair's slab+tail is the single node's experts, split by ownership
    assert_eq!(a0.expert_slab_bytes, LoadPlan::full_tp(&w, &t0).unwrap().expert_bytes);
    assert_eq!(
        a0.expert_slab_bytes + a0.streamed_expert_bytes + a1.expert_slab_bytes + a1.streamed_expert_bytes,
        single.expert_slab_bytes + single.streamed_expert_bytes,
        "the two ranks' expert terms sum to the single node's"
    );
    // the replicated terms do not move: the resident set and the MTP bookkeeping are the single node's, less the peer's
    // NextN experts (which the pair splits like the trunk's)
    assert_eq!(a0.resident_bytes, single.resident_bytes);
    assert_eq!(a0.mtp_expert_bytes + a1.mtp_expert_bytes, single.mtp_expert_bytes);
    assert_eq!(a0.kv_bytes, single.kv_bytes, "the KV planes are replicated");
    // a rank's slab is its OWN experts whole (the toy's fit), and it serves the single node's slots and context
    for (a, t) in [(&a0, &t0), (&a1, &t1)] {
        assert_eq!(a.expert_slab_bytes, LoadPlan::full_tp(&w, t).unwrap().expert_bytes);
        assert_eq!(a.slots, single.slots);
        assert_eq!(a.context, single.context, "a rank serves the single node's context");
        assert_eq!(a.streamed_expert_bytes, 0, "the toy's experts fit a rank's slab whole");
    }
}

/// `admit_pack` over an optional table, at toy figures where everything fits: the shared seam of the two calls above.
fn admit(
    cfg: &DeepseekV41Config,
    w: &gb_model::families::deepseek_v41::weights::DeepseekV41Weights,
    o: &ServeOptions,
    shard: Option<&ShardTable>,
) -> gb_model::families::deepseek_v41::plan::Admission {
    let dev = 8 << 30;
    let (_p, a) = gb_model::families::deepseek_v41::engine::admit_pack(
        cfg,
        w,
        o,
        gb_model::families::deepseek_v41::engine::BuildLimits::at(dev),
        shard,
    )
    .unwrap();
    a
}

// ------------------------------------------------------------------ the build wiring's gate

/// `DeepseekV41Family::build` gates a group boot FIRST, by name, before a byte of the pack is read: a wide window and the
/// DSpark drafting are refused by the setting's own name (the pack dir holds a config and nothing else, so an error that
/// mentions shards or tensors would mean the gate let it through); the verified settings — the 8-row window, the copy
/// drafter, plain decoding — pass the gate and reach the pack itself, where the communicator's bounded rendezvous (there
/// is no peer in a unit test) names `--tp-peer`.
#[test]
fn build_gates_a_group_boot_by_name_before_the_pack() {
    let dir = pack_dir("tp-gate");
    let mut o = ServeOptions::new(dir.clone());
    o.tp = Some(gb_model::serve::TpOptions::new(2, 0, "127.0.0.1:1"));
    o.window_rows = 64; // the server's own default; the family's window is fixed and the option cannot widen it
    for spec in [SpecMode::Mtp, SpecMode::Auto, SpecMode::Copy, SpecMode::None] {
        let mut o2 = o.clone();
        o2.drafting = spec.uses_mtp();
        o2.spec = spec;
        let e = build_err(&o2);
        assert!(!e.contains("not built for tensor parallelism"), "past the gate for {spec:?}: {e}");
        assert!(
            e.contains("tp-peer") || e.contains("tensors") || e.contains("pack"),
            "the pack or the rendezvous, in that order: {e}"
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// A scratch pack dir: the family's own config and nothing else (the gate must refuse before any of it is read).
fn pack_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rexl3-dsv41-tp-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), CONFIG).unwrap();
    dir
}

/// `build`'s error, as text (the probe's plan is refreshed per call the way the server does).
fn build_err(o: &ServeOptions) -> String {
    use gb_model::family::ModelFamily as _;
    let fresh = gb_model::families::deepseek_v41::DeepseekV41Family.probe(&o.model_dir).unwrap();
    match gb_model::families::deepseek_v41::DeepseekV41Family.build(fresh, &gb_model::family::BuildCtx { options: o }) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a config-only pack with a group boot must not build"),
    }
}

// ------------------------------------------------------------------ the refusal surface

/// A group boot refuses, BY NAME, the steps the TP engine is not verified for: a window wider than the family's
/// 8-row one, and the DSpark drafting of the NextN layers (whose combine arms are a later lane). The copy drafter and
/// the plain no-drafting boot pass.
#[test]
fn check_tp_plain_refuses_the_unverified_steps_by_name() {
    let plain = || {
        let mut o = ServeOptions::new(PathBuf::from("/nonexistent"));
        o.window_rows = 8;
        o.drafting = false;
        o.spec = SpecMode::None;
        o
    };
    assert!(check_tp_plain(&plain()).is_ok(), "the 8-row window with no drafting is the verified boot");
    // there is NO window arm: the family's window is FIXED at 8 rows and not `--window-rows`-adjustable (the start
    // refuses an explicit other value by name), so the option — the server's global default of 64 included — is not
    // a wider window a group boot could run (comparing it against the family's window made every server group boot
    // unbootable; found at the first real two-node boot)
    let mut whatever = plain();
    whatever.window_rows = 64;
    assert!(check_tp_plain(&whatever).is_ok(), "the option cannot widen the fixed 8-row window");
    let mut drafting = plain();
    drafting.drafting = true;
    drafting.spec = SpecMode::Mtp;
    assert!(check_tp_plain(&drafting).is_ok(), "DSpark drafting with replicated NextN experts is supported");
    drafting.spec = SpecMode::Auto;
    assert!(check_tp_plain(&drafting).is_ok());
    drafting.spec = SpecMode::Copy; // the copy drafter needs nothing of the model
    assert!(check_tp_plain(&drafting).is_ok());
    drafting.drafting = false; // --spec alone, with drafting off, drafts nothing
    drafting.spec = SpecMode::Mtp;
    assert!(check_tp_plain(&drafting).is_ok());
}

// ------------------------------------------------------------------ the loopback collective

/// The element owner `i` of a 2 x `half`-element buffer: rank r holds the whole row in its half and the exact `-0.0`
/// filler (the additive identity) elsewhere.
fn element(i: usize) -> f32 {
    let x = ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 29) as u32;
    let sign = (x >> 16) & 1;
    match i % 7 {
        0 => f32::from_bits(sign << 31), // an exact signed zero: its sum with -0.0 keeps its sign
        3 => f32::from_bits(sign << 31 | (x & 0xFF)), // a subnormal: x + (-0.0) == x must survive it
        _ => f32::from_bits(sign << 31 | ((127 - 4 + (x % 9)) << 23) | ((x >> 8) & 0x7F_FFFF)),
    }
}

/// One loopback pair, built against the split table, running ONE `allgather_sum_f32` over a disjoint-support buffer
/// sized by the table's o-projection rows: every rank uploads its own rows and the `-0.0` filler, and both read back
/// the owner's element bit for bit (the one rule the sum is exact by). Mirrors nemotron's
/// `two_loopback_threads_sum_disjoint_buffers_exactly`, one round; skips without a device as it does.
#[test]
fn a_loopback_pair_runs_one_allgather_over_disjoint_support_from_the_table() {
    let (c0, c1) = LoopbackGroup::pair();
    assert_eq!((c0.topology(), c1.topology()), (TpTopology { rank: 0, ranks: 2 }, TpTopology { rank: 1, ranks: 2 }));
    assert_eq!(
        c0.topology(),
        ShardTable::for_config(c0.topology(), &real()).unwrap().topo,
        "the pair and the table describe the same group"
    );
    if !gb_cuda::device_available() {
        eprintln!("no CUDA device: skipping the collective (the pair and the table agree)");
        return;
    }
    let cfg = real();
    let whole: std::sync::Arc<Vec<f32>> = std::sync::Arc::new((0..splits::Q_HEADS * splits::HEAD_DIM).map(element).collect());
    let run =
        |comm: gb_model::families::nemotron_h_omni::tp_comm::LoopbackComm, table: ShardTable, whole: std::sync::Arc<Vec<f32>>| {
            std::thread::spawn(move || -> Result<(), String> {
                let rank = comm.topology().rank;
                let rows = table.o_row_range();
                let st = gb_cuda::Stream::new().map_err(|e| e.to_string())?;
                let mine: Vec<f32> = (0..whole.len()).map(|i| if rows.contains(&i) { whole[i] } else { -0.0 }).collect(); // whole: Arc<Vec<f32>>, deref-indexed
                let mut buf = gb_cuda::DevBuf::from_slice(&mine).map_err(|e| e.to_string())?;
                comm.allgather_sum_f32(&mut buf, 0, whole.len(), &st).map_err(|e| format!("rank {rank}: {e}"))?;
                st.sync().map_err(|e| e.to_string())?;
                let got: Vec<f32> = buf.to_vec().map_err(|e| e.to_string())?;
                for (i, (g, w)) in got.iter().zip(whole.iter()).enumerate() {
                    if g.to_bits() != w.to_bits() {
                        return Err(format!("rank {rank}: element {i}: got {:#010x} want {:#010x}", g.to_bits(), w.to_bits()));
                    }
                }
                Ok(())
            })
        };
    let t0 = ShardTable::for_config(c0.topology(), &cfg).unwrap();
    let t1 = ShardTable::for_config(c1.topology(), &cfg).unwrap();
    let (r0, r1) = (run(c0, t0, whole.clone()), run(c1, t1, whole));
    assert_eq!(r0.join().expect("rank 0 panicked"), Ok(()), "rank 0");
    assert_eq!(r1.join().expect("rank 1 panicked"), Ok(()), "rank 1");
}

// ------------------------------------------------------------------ the combine arms over synthetic buffers

/// A synthetic f16 element: bit patterns that make any reordering or re-rounding visible (signed zeros, subnormals,
/// odd mantissas), as f16 bits.
fn f16_bits(i: usize) -> u16 {
    let x = ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 29) as u32;
    match i % 5 {
        0 => (((x >> 16) & 1) << 15) as u16,                // an exact signed zero
        1 => ((((x >> 16) & 1) << 15) | (x & 0xFF)) as u16, // a subnormal
        _ => ((((x >> 16) & 1) << 15) | (15 << 10) | ((x >> 8) & 0x3FF)) as u16,
    }
}

/// A whole f16 buffer of `n` elements.
fn f16_buffer(n: usize) -> Vec<u16> {
    (0..n).map(f16_bits).collect()
}

/// The two arms over synthetic buffers, one loopback pair each: the ranks hold DISJOINT halves of a whole (the o
/// compact halves at the attention boundary, the owned slot ranges of the MoE `y`), ONE exchange a site, and every
/// rank's combined buffer is the single-rank one BIT for bit — the one rule, in the byte form the f16 buffers force
/// (no arithmetic anywhere, so a `-0.0` cannot flip). Mirrors the nemotron arm gates over the loopback pair; skips
/// without a device, exactly as they do.
#[test]
fn the_arms_combine_disjoint_halves_bit_for_bit_over_the_loopback_pair() {
    if !gb_cuda::device_available() {
        eprintln!("no CUDA device: skipping the combine arms (the ranges and the mirrors are pinned CPU-side)");
        return;
    }
    let cfg = real();
    let (c0, c1) = LoopbackGroup::pair();
    let (t0, t1) = (ShardTable::for_config(c0.topology(), &cfg).unwrap(), ShardTable::for_config(c1.topology(), &cfg).unwrap());
    let (rows, hidden, topk) = (3usize, splits::HIDDEN, 8usize);
    let slots = rows * topk;
    let o_total = splits::Q_HEADS * splits::HEAD_DIM;
    let run = |comm: std::sync::Arc<gb_model::families::nemotron_h_omni::tp_comm::LoopbackComm>,
               table: ShardTable,
               whole_o: std::sync::Arc<Vec<u16>>,
               whole_y: std::sync::Arc<Vec<u16>>| {
        std::thread::spawn(move || -> Result<String, String> {
            let rank = comm.topology().rank;
            let st = gb_cuda::Stream::new().map_err(|e| e.to_string())?;
            let mut arms = gb_model::families::deepseek_v41::tp_arm::TpArms::new(
                comm as std::sync::Arc<dyn TpComm>,
                table.clone(),
                rows,
                o_total,
            )
            .map_err(|e| format!("rank {rank}: {e}"))?;
            // --- the MLA o combine: this rank's compact half, gathered into the full-width staging
            let mine = table.o_row_range();
            let compact: Vec<u16> =
                (0..rows * mine.len()).map(|i| whole_o[i / mine.len() * o_total + mine.start + i % mine.len()]).collect();
            let compact = gb_cuda::DevBuf::from_slice(&compact).map_err(|e| e.to_string())?;
            let full = arms.o_gather(&compact, 0, rows, &st).map_err(|e| format!("rank {rank}: {e}"))?;
            st.sync().map_err(|e| e.to_string())?;
            let got: Vec<u16> = full.to_vec().map_err(|e| e.to_string())?;
            for (i, (g, w)) in got.iter().zip(whole_o.iter()).enumerate() {
                if g != w {
                    return Err(format!("rank {rank}: o element {i}: got {g:#06x} want {w:#06x}"));
                }
            }
            // --- the MoE y combine: the owned slot range in place, the peer's mirrored into it
            let own = if rank == 0 { 0..slots / 2 } else { slots / 2..slots };
            let mut y: Vec<u16> = whole_y.iter().copied().collect();
            for s in 0..slots {
                if !own.contains(&s) {
                    y[s * hidden..(s + 1) * hidden].fill(0x8000); // scratch garbage where the peer's segments will land
                }
            }
            // the real `y` is a view into the MoE scratch's arena block (`MoeScratch::arena_block`), and `y_gather`
            // subviews it for the send share: the test's buffer is a view the same way (a standalone allocation
            // cannot be subviewed — the arm's contract is over the model's own buffer forms)
            let mut y_dev = gb_cuda::DevBuf::zeroed_view(y.len() * 2).map_err(|e| e.to_string())?;
            y_dev.upload(&y).map_err(|e| e.to_string())?;
            arms.y_gather(&mut y_dev, own, slots, hidden, &st).map_err(|e| format!("rank {rank}: {e}"))?;
            st.sync().map_err(|e| e.to_string())?;
            let got: Vec<u16> = y_dev.to_vec().map_err(|e| e.to_string())?;
            for (i, (g, w)) in got.iter().zip(whole_y.iter()).enumerate() {
                if g != w {
                    return Err(format!("rank {rank}: y element {i}: got {g:#06x} want {w:#06x}"));
                }
            }
            let (bytes, exchanges) = (arms.exchanged_bytes, arms.exchanges);
            if exchanges != 2 || bytes == 0 {
                return Err(format!("rank {rank}: {exchanges} exchanges, {bytes} B — one a site"));
            }
            Ok(format!("rank {rank}: both combines bit-for-bit"))
        })
    };
    let whole_o = std::sync::Arc::new(f16_buffer(rows * o_total));
    let whole_y = std::sync::Arc::new(f16_buffer(slots * hidden));
    let (r0, r1) =
        (run(std::sync::Arc::new(c0), t0, whole_o.clone(), whole_y.clone()), run(std::sync::Arc::new(c1), t1, whole_o, whole_y));
    let m0 = r0.join().expect("rank 0 panicked");
    let m1 = r1.join().expect("rank 1 panicked");
    assert_eq!(m0, Ok("rank 0: both combines bit-for-bit".to_string()), "{m0:?}");
    assert_eq!(m1, Ok("rank 1: both combines bit-for-bit".to_string()), "{m1:?}");
}

// ------------------------------------------------------------------ lane 3: the follower runtime (the wire)

use gb_cuda::dsv4_kvq4::KvFmt;
/// The lane-3 gates of the lockstep runtime (`tp_runtime`): the wire calls, the boot facts, the gate's rank-0 options,
/// and an in-process follower that replays the leader's calls over a loopback link pair — the nemotron `tp_runtime`
/// CPU gates mirrored at this family's width.
use gb_model::families::deepseek_v41::model::ModelOptions;
use gb_model::families::deepseek_v41::tp_runtime::{gate_of, prefill_batch, Dsv41Call, Dsv41Facts, Dsv41Runtime};
use gb_model::forward::Candidates;
use gb_model::lockstep::{self, FamilyCall, Link};

/// The score calls round-trip on the wire: what the leader sends is what the follower decodes, exactly.
#[test]
fn the_score_calls_round_trip_on_the_wire() {
    let calls = vec![
        Dsv41Call::ScoreLastRow { slot: 0, start: 12, tokens: vec![7, 9, 128_799, 3] },
        Dsv41Call::ScoreRows { slot: 2, start: 4096, tokens: (0..64).map(|i| i * 3 % 5000).collect() },
    ];
    for c in &calls {
        let mut e = lockstep::Enc::new();
        c.encode(&mut e);
        let mut d = lockstep::Dec::new(&e.0);
        let back = Dsv41Call::decode(c.kind(), &mut d).unwrap();
        assert_eq!(back.as_ref(), Some(c), "{:?}", c.name());
        assert!(d.done("the body was consumed exactly").is_ok());
    }
}

/// The boot facts name every arithmetic knob that differs — the KV format, the MoE decode top-p, the CED replay, the
/// engram sidecar variables, the spec — each by the flag an operator passes.
#[test]
fn the_boot_facts_name_every_arithmetic_knob_that_differs() {
    let base = || Dsv41Facts {
        kv: KvFmt::Q4,
        moe_topp: "unset".into(),
        ced_replay: true,
        engram_token_map: "unset".into(),
        engram_shards: "unset".into(),
        spec: "none".into(),
        cache_request: "On".into(),
        cache_slots: 32,
    };
    let leader = base();
    assert!(Dsv41Call::check_facts(&leader, &base(), 1).is_empty(), "identical facts agree");
    let mut other = base();
    other.kv = KvFmt::F16;
    other.moe_topp = "1.0".into();
    other.ced_replay = false;
    other.engram_token_map = "/other/map.u32".into();
    other.engram_shards = "/other/shards".into();
    other.spec = "auto".into();
    let why = Dsv41Call::check_facts(&leader, &other, 1).join("; ");
    for needle in [
        "kv-dtype",
        "REXL3_DSV41_MOE_TOPP",
        "REXL3_DSV41_CED_REPLAY",
        "REXL3_DSV41_ENGRAM_TOKEN_MAP",
        "REXL3_DSV41_ENGRAM_SHARDS",
        "--spec",
    ] {
        assert!(why.contains(needle), "names {needle}: {why}");
    }
}

/// The gate's `model_options` is rank 0's geometry exactly — the follower is BUILT with the leader's slots, KV pages,
/// context, chunk rows, layers and cache format, never its own derivation of them.
#[test]
fn the_gate_carries_rank0s_model_options_for_the_follower() {
    let gate = gate_of(
        "pack-digest".into(),
        40,
        2,
        65536 / gb_model::kvtables::BLOCK_TOKENS,
        65536,
        128,
        Dsv41Facts {
            kv: KvFmt::Q4,
            moe_topp: "unset".into(),
            ced_replay: true,
            engram_token_map: "unset".into(),
            engram_shards: "unset".into(),
            spec: "none".into(),
            cache_request: "On".into(),
            cache_slots: 32,
        },
    )
    .at(1, 2);
    let mo: ModelOptions = gate.model_options();
    assert_eq!((mo.slots, mo.max_context, mo.chunk_rows, mo.layers, mo.kv), (2, 65536, 128, 40, KvFmt::Q4));
    assert_eq!(mo.kv_blocks, 65536 / gb_model::kvtables::BLOCK_TOKENS, "the gate's kv_pages are the model's kv_blocks");
}

/// The prefill wire batch is the model's own single-sequence shape: what `Dsv41Model::prefill_chunk` builds inside
/// itself is what the leader puts on the wire, and the reconstruction is exact — while a multi-sequence batch is
/// refused by name (this family's prefill is one slot's chunk at a time).
#[test]
fn the_prefill_wire_batch_round_trips_the_models_own_shape() {
    let b = prefill_batch(3, 100, &[11, 22, 33]);
    assert_eq!(b.rows, 3);
    assert_eq!(b.ids, vec![11, 22, 33]);
    assert_eq!(b.positions, vec![100, 101, 102]);
    assert_eq!(b.seq_slot, vec![3]);
    assert_eq!(b.row_slot, vec![3, 3, 3]);
    assert_eq!(b.n_seq(), 1);
    let (slot, start, tokens) = gb_model::families::deepseek_v41::tp_runtime::prefill_of(&b).unwrap();
    assert_eq!((slot, start, tokens), (3usize, 100usize, vec![11u32, 22, 33]));
    // a two-sequence batch is not this family's prefill: refused by name, never guessed
    let mut two = b.clone();
    two.seq_slot = vec![3, 4];
    two.seq_start = vec![0, 2, 3];
    let e = gb_model::families::deepseek_v41::tp_runtime::prefill_of(&two).unwrap_err().to_string();
    assert!(e.contains("ONE sequence") && e.contains("2 sequences"), "{e}");
}

/// A mock of the model surface the follower replays: records every call it applied.
#[derive(Default)]
struct RecordingRt {
    began: Vec<usize>,
    prefills: Vec<(usize, usize, Vec<u32>)>,
    windows: Vec<usize>,
    commits: Vec<(usize, usize)>,
    flushes: usize,
    scores: Vec<(usize, usize, usize)>,
    /// What `prefill_logits_last_row` returns (the digest the follower answers with).
    logits: Vec<f32>,
}

impl gb_model::families::deepseek_v41::tree::TreeRoundHost for RecordingRt {
    fn tree_path_window(
        &mut self,
        _tokens: &[u32],
        _first_position: u64,
        _candidates: usize,
    ) -> Result<Vec<gb_engine::RowOutput>, String> {
        Err("not built on mock".into())
    }
    fn tree_commit_window(&mut self, _idx: usize, _rows: usize, _window: &[u32]) -> Result<(), String> {
        Err("not built on mock".into())
    }
    fn tree_drop_round(&mut self) -> Result<(), String> {
        Ok(())
    }
}

impl Dsv41Runtime for RecordingRt {
    fn vocab_size(&self) -> usize {
        129_280
    }
    fn has_mtp(&self) -> bool {
        false
    }
    fn chunk_rows(&self) -> usize {
        512
    }
    fn begin_slot(&mut self, slot: usize) -> Result<(), gb_model::error::ModelError> {
        self.began.push(slot);
        Ok(())
    }
    fn end_slot(&mut self, slot: usize) {
        let _ = slot;
    }
    fn prefill_chunk(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
    ) -> Result<Option<Candidates>, gb_model::error::ModelError> {
        assert_eq!(want, Some(4), "the candidates ask travels");
        self.prefills.push((slot, start, tokens.to_vec()));
        Ok(Some(Candidates { rows: 1, count: 4, ids: vec![5, 6, 7, 8], vals: vec![0.9, 0.4, 0.1, 0.0] }))
    }
    fn window_candidates(
        &mut self,
        b: &gb_model::rows::RowBatch,
        count: usize,
    ) -> Result<Candidates, gb_model::error::ModelError> {
        self.windows.push(b.rows);
        Ok(Candidates { rows: b.rows, count, ids: vec![9; b.rows * count], vals: vec![0.5; b.rows * count] })
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.commits.push((slot, keep));
    }
    fn flush(&mut self) -> Result<(), gb_model::error::ModelError> {
        self.flushes += 1;
        Ok(())
    }
    fn prefill_logits_last_row(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>, gb_model::error::ModelError> {
        self.scores.push((slot, start, tokens.len()));
        Ok(self.logits.clone())
    }
    fn prefill_logits_rows(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, gb_model::error::ModelError> {
        self.scores.push((slot, start, tokens.len()));
        Ok(self.logits.chunks(8).map(<[f32]>::to_vec).collect())
    }
    fn propose_mtp(
        &mut self,
        _slot: usize,
        _tokens: &[u32],
        _max: usize,
    ) -> Result<Option<gb_model::families::deepseek_v41::model::DraftRound>, gb_model::error::ModelError> {
        Ok(None)
    }
}

/// A follower replays the leader's calls in lockstep and a divergent digest is a lost group by name: the whole
/// follower runtime (`run_follower` over a `Link::pair`) driven against a live leader, in process.
#[test]
fn a_follower_replays_the_leaders_calls_in_lockstep() {
    let (ll, fl) = Link::pair(std::time::Duration::from_secs(10)).unwrap();
    let follower = std::thread::spawn(move || {
        let rt = RecordingRt { logits: vec![0.25f32; 8], ..Default::default() };
        let cfg = gb_model::families::deepseek_v41::tp_runtime::FollowerConfig::new(lockstep::TpTimeouts {
            connect: std::time::Duration::from_secs(5),
            ready: std::time::Duration::from_secs(20),
            peer: std::time::Duration::from_secs(20),
            write: std::time::Duration::from_secs(20),
            hang_exit: None,
            ..Default::default()
        })
        .with_rank(1);
        gb_model::families::deepseek_v41::tp_runtime::run_follower(rt, fl, cfg)
    });
    let leader_rt = RecordingRt { logits: vec![0.25f32; 8], ..Default::default() };
    let t = lockstep::TpTimeouts {
        connect: std::time::Duration::from_secs(5),
        ready: std::time::Duration::from_secs(20),
        peer: std::time::Duration::from_secs(20),
        write: std::time::Duration::from_secs(20),
        hang_exit: None,
        ..Default::default()
    };
    let mut leader = gb_model::families::deepseek_v41::tp_runtime::TpLeader::start(leader_rt, ll, t);
    // drive the whole core surface + the family's score call
    leader.begin_slot(0).unwrap();
    let c = Dsv41Runtime::prefill_chunk(&mut leader, 0, 7, &[1, 2, 3], Some(4)).unwrap().unwrap();
    assert_eq!(c.ids, vec![5, 6, 7, 8], "the candidates are the leader's own (rank 0's answer is the answer)");
    let w = leader.window_candidates(&prefill_batch(0, 10, &[4, 5]), 4).unwrap();
    assert_eq!(w.rows, 2);
    leader.commit(0, 1);
    leader.flush().unwrap();
    Dsv41Runtime::prefill_logits_last_row(&mut leader, 0, 7, &[1, 2, 3]).unwrap();
    // shutdown: the follower returns
    let stats = leader.stats();
    assert!(stats.commands > 0, "calls were forwarded");
    drop(leader);
    let out = follower.join().expect("follower panicked");
    assert!(matches!(out, Ok(lockstep::FollowerExit::Shutdown(_))), "{out:?}");
}

/// A follower whose arithmetic diverges (another score answer) is a LOST GROUP naming the call: the digest is the
/// exactness gate of the runtime, never widened.
#[test]
fn a_divergent_follower_is_a_lost_group_naming_the_call() {
    let (ll, fl) = Link::pair(std::time::Duration::from_secs(10)).unwrap();
    let follower = std::thread::spawn(move || {
        let rt = RecordingRt { logits: vec![0.5f32; 8], ..Default::default() }; // DIFFERENT logits
        let cfg = gb_model::families::deepseek_v41::tp_runtime::FollowerConfig::new(lockstep::TpTimeouts {
            ready: std::time::Duration::from_secs(20),
            hang_exit: None,
            ..Default::default()
        })
        .with_rank(1);
        gb_model::families::deepseek_v41::tp_runtime::run_follower(rt, fl, cfg)
    });
    let t = lockstep::TpTimeouts { ready: std::time::Duration::from_secs(20), hang_exit: None, ..Default::default() };
    let mut leader = gb_model::families::deepseek_v41::tp_runtime::TpLeader::start(
        RecordingRt { logits: vec![0.25f32; 8], ..Default::default() },
        ll,
        t,
    );
    // the divergent call returns the LEADER'S own answer (rank 0's answer is the answer) and latches the loss once
    // the follower's ack is read; the settled group then refuses, naming the call that diverged
    let ok = Dsv41Runtime::prefill_logits_last_row(&mut leader, 0, 0, &[1]).unwrap();
    assert_eq!(ok.len(), 8, "the leader's own logits are returned");
    let e = leader.settle(leader.last_seq()).unwrap_err().to_string();
    assert!(e.contains("score_last_row"), "names the call that diverged: {e}");
    let _ = follower.join();
}

/// The leader refuses the DSpark drafting by name at the runtime seam (the boot's `check_tp_plain` already refused
/// it; this is the choke a late `--spec` flip would hit, never a silent desync).
#[test]
fn the_leader_refuses_the_dspark_draft_by_name_at_the_runtime_seam() {
    let (ll, fl) = Link::pair(std::time::Duration::from_secs(10)).unwrap();
    let follower = std::thread::spawn(move || {
        let cfg = gb_model::families::deepseek_v41::tp_runtime::FollowerConfig::new(lockstep::TpTimeouts {
            hang_exit: None,
            ..Default::default()
        })
        .with_rank(1);
        let _ = gb_model::families::deepseek_v41::tp_runtime::run_follower(RecordingRt::default(), fl, cfg);
    });
    let mut leader = gb_model::families::deepseek_v41::tp_runtime::TpLeader::start(
        RecordingRt::default(),
        ll,
        lockstep::TpTimeouts { hang_exit: None, ..Default::default() },
    );
    let e = Dsv41Runtime::propose_mtp(&mut leader, 0, &[1, 2], 4).unwrap_err().to_string();
    assert!(e.contains("DSpark") && e.contains("tensor parallelism"), "{e}");
    drop(leader);
    let _ = follower.join();
}

// ------------------------------------------------------------------ the FP8 gemm's width invariance (the wq_b cut)

/// The MLA cut's load-bearing premise: an FP8 projection's output bits depend only on the element's own row and
/// column, never on the matrix's WIDTH — a rank's [16384, k] slice of a [32768, k] `wq_b` must produce the same bits
/// for its rows as the whole matrix does (the combine's byte-gather contract is over those bits). Device-gated,
/// like the arms.
#[test]
fn the_fp8_projection_is_bit_invariant_to_the_width_cut() {
    if !gb_cuda::device_available() {
        eprintln!("no CUDA device: skipping the FP8 width-invariance probe");
        return;
    }
    use gb_cuda::fp8_linear::Fp8Weights;
    let (n_full, k, rows) = (32768usize, 5120usize, 24usize);
    // deterministic pseudo-random bytes for w and scales (scales 0x70..0x8f, never 0xFF)
    let w: Vec<u8> = (0..n_full * k).map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 33) as u8).collect();
    let s: Vec<u8> = (0..(n_full / 32) * (k / 32)).map(|i| 0x70 + (i % 32) as u8).collect();
    let x: Vec<u16> = f16_buffer(rows * k);
    let x = gb_cuda::DevBuf::from_slice(&x).unwrap();
    let st = gb_cuda::Stream::new().unwrap();
    let full = Fp8Weights::new(n_full, k, &w, &s).unwrap();
    let mut y_full = gb_cuda::DevBuf::zeroed(rows * n_full * 2).unwrap();
    full.forward(&x, &mut y_full, rows, gb_cuda::exl3_linear::Dtype::F16, &st).unwrap();
    // the rank slices: rows [0,16384) and [16384,32768) — head-aligned (512) and 32-scale-aligned, as the bind cuts
    for (lo, hi) in [(0usize, n_full / 2), (n_full / 2, n_full)] {
        let w_slice = &w[lo * k..hi * k];
        let nt = (hi - lo) / 32;
        let s_slice = {
            // scale rows [lo/32, hi/32) of every k/32-wide stripe: the scales are [n/32, k/32] row-major
            let kt = k / 32;
            let rows_from = lo / 32;
            let mut v = Vec::with_capacity(nt * kt);
            for r in rows_from..rows_from + nt {
                v.extend_from_slice(&s[r * kt..(r + 1) * kt]);
            }
            v
        };
        let part = Fp8Weights::new(hi - lo, k, w_slice, &s_slice).unwrap();
        let mut y_part = gb_cuda::DevBuf::zeroed(rows * (hi - lo) * 2).unwrap();
        part.forward(&x, &mut y_part, rows, gb_cuda::exl3_linear::Dtype::F16, &st).unwrap();
        st.sync().unwrap();
        let got: Vec<u16> = y_part.to_vec().unwrap();
        let want: Vec<u16> = {
            let all: Vec<u16> = y_full.to_vec().unwrap();
            (0..rows).flat_map(|r| all[r * n_full + lo..r * n_full + hi].to_vec()).collect()
        };
        let bad = got.iter().zip(want.iter()).enumerate().find(|(_, (a, b))| a != b).map(|(i, _)| i);
        assert!(
            bad.is_none(),
            "rank slice [{lo},{hi}): element {bad:?} differs — the FP8 projection's bits depend on the matrix width, so the MLA head cut is not bit-invariant (first diff {diff:?})",
            diff = bad.map(|i| (i, got[i], want[i]))
        );
    }
}

// ------------------------------------------------------------------ the two-rank prefix cache: restore verified per rank, cold otherwise
//
// The whole serving path of a two-rank group in process: the real engine (`gb_engine::Engine`, its prefix index and its cold
// fallback) over the real backend (`Dsv41Backend`) over the real lockstep leader (`TpLeader`, the forwarded cache calls, the
// `SnapshotBook`) and a real follower loop (`run_follower` over a `Link` pair). The model is `ChainRt`: each rank's per-slot state is
// a hash chain over the tokens it has seen (its own shard: the chains differ by rank), and every prefill chunk and window ends in a
// simulated all-reduce (`Allreduce`) of both ranks' chain values, from which both ranks' candidates are made. So a follower that
// restored stale or empty state changes rank 0's TOKENS, exactly as a real one would through the real all-reduce, while every
// candidate digest still agrees (both ranks see the same sum): the class of error only a content digest catches.

use gb_model::families::deepseek_v41::backend::{prefix_namespace, Dsv41Backend};
use gb_model::families::deepseek_v41::tp_runtime::{run_follower, FollowerConfig, TpLeader};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// The two ranks' simulated all-reduce: round `r` of rank 0 meets round `r` of rank 1 (both run the same calls in the same order),
/// and both get the element-wise wrapping sum. `abort` (the group's abort hook) makes a waiting rank return an error, as an aborted
/// communicator does.
#[derive(Default)]
struct Allreduce {
    m: Mutex<(u64, u64, std::collections::HashMap<u64, [Option<Vec<u64>>; 2]>)>,
    cv: Condvar,
    abort: AtomicBool,
}

impl Allreduce {
    fn reduce(&self, rank: usize, v: Vec<u64>) -> Result<Vec<u64>, gb_model::error::ModelError> {
        let mut g = self.m.lock().unwrap();
        let round = if rank == 0 { &mut g.0 } else { &mut g.1 };
        let r = *round;
        *round += 1;
        g.2.entry(r).or_default()[rank] = Some(v);
        self.cv.notify_all();
        let t0 = Instant::now();
        loop {
            if let Some([Some(a), Some(b)]) = g.2.get(&r) {
                assert_eq!(a.len(), b.len(), "round {r}: the ranks reduced different shapes (a desync)");
                let out: Vec<u64> = a.iter().zip(b).map(|(x, y)| x.wrapping_add(*y)).collect();
                return Ok(out);
            }
            if self.abort.load(Ordering::SeqCst) {
                return Err(gb_model::error::ModelError::StateLost("the communicator was aborted".into()));
            }
            assert!(t0.elapsed() < Duration::from_secs(60), "round {r}: rank {rank} waited 60 s for its peer (a hang)");
            g = self.cv.wait_timeout(g, Duration::from_millis(20)).unwrap().0;
        }
    }
}

/// What a follower's restore does wrong (`Faults::restore`).
const RESTORE_OK: u8 = 0;
/// Returns an error.
const RESTORE_ERROR: u8 = 1;
/// Begins the slot and restores nothing, returning Ok (an empty restore).
const RESTORE_EMPTY: u8 = 2;
/// Leaves the slot as the previous request left it and returns Ok (a stale restore).
const RESTORE_STALE: u8 = 3;

/// The faults a rank can be told to have, shared with the test so they can change between requests.
#[derive(Default)]
struct Faults {
    restore: AtomicU8,
    /// The memory guard says no (a capture is skipped on this rank).
    mem_low: AtomicBool,
    /// The restore sleeps this long first (a slow or stuck rank; cut short by the all-reduce's abort).
    restore_sleep_ms: AtomicU64,
    /// The pool is capped at this many slots, whatever rank 0 asks (0: no cap).
    pool_cap: AtomicU64,
}

impl Faults {
    fn clear(&self) {
        self.restore.store(RESTORE_OK, Ordering::SeqCst);
        self.mem_low.store(false, Ordering::SeqCst);
        self.restore_sleep_ms.store(0, Ordering::SeqCst);
    }
}

/// A deterministic stand-in for one rank of the model, with the state the prefix cache must carry (see the section's note).
struct ChainRt {
    rank: usize,
    max_context: usize,
    chunk: usize,
    reduce: Arc<Allreduce>,
    faults: Arc<Faults>,
    slots: Vec<Vec<u64>>,
    /// The last window's extended chain per slot (what `commit` keeps a prefix of).
    window: Vec<Vec<u64>>,
    pool: Vec<Option<Vec<u64>>>,
    free: Vec<u32>,
}

const CHAIN_VOCAB: usize = 4096;

impl ChainRt {
    fn new(rank: usize, slots: usize, max_context: usize, reduce: Arc<Allreduce>, faults: Arc<Faults>) -> ChainRt {
        ChainRt {
            rank,
            max_context,
            chunk: 2048,
            reduce,
            faults,
            slots: vec![Vec::new(); slots],
            window: vec![Vec::new(); slots],
            pool: Vec::new(),
            free: Vec::new(),
        }
    }

    fn link(&self, prev: u64, token: u32, pos: usize) -> u64 {
        gb_cuda::state_digest::mix64(prev ^ ((token as u64) << 20) ^ ((pos as u64) << 40) ^ (self.rank as u64 + 1) * 0x9e37_79b9)
    }

    fn candidates(sum: &[u64], count: usize) -> Candidates {
        let (mut ids, mut vals) = (Vec::new(), Vec::new());
        for s in sum {
            for j in 0..count {
                ids.push((gb_cuda::state_digest::mix64(s.wrapping_add(j as u64)) % (CHAIN_VOCAB as u64 - 1) + 1) as i32);
                vals.push(10.0 - j as f32);
            }
        }
        Candidates { rows: sum.len(), count, ids, vals }
    }
}

impl gb_model::families::deepseek_v41::tree::TreeRoundHost for ChainRt {
    fn tree_path_window(&mut self, _t: &[u32], _p: u64, _c: usize) -> Result<Vec<gb_engine::RowOutput>, String> {
        Err("no trees on the chain mock".into())
    }
    fn tree_commit_window(&mut self, _i: usize, _r: usize, _w: &[u32]) -> Result<(), String> {
        Err("no trees on the chain mock".into())
    }
    fn tree_drop_round(&mut self) -> Result<(), String> {
        Ok(())
    }
}

impl Dsv41Runtime for ChainRt {
    fn vocab_size(&self) -> usize {
        CHAIN_VOCAB
    }
    fn has_mtp(&self) -> bool {
        false
    }
    fn chunk_rows(&self) -> usize {
        self.chunk
    }
    fn max_context(&self) -> usize {
        self.max_context
    }
    fn enable_prefix_cache(&mut self, n: usize) {
        let cap = self.faults.pool_cap.load(Ordering::SeqCst) as usize;
        let n = if cap > 0 { n.min(cap) } else { n };
        self.pool = vec![None; n];
        self.free = (0..n as u32).rev().collect();
    }
    fn prefix_pool_slots(&self) -> usize {
        self.pool.len()
    }
    fn prefix_free_snapshots(&self) -> usize {
        self.free.len()
    }
    fn can_snapshot_prefix(&self, len: usize) -> bool {
        len > 0 && len <= self.max_context && !self.free.is_empty() && !self.faults.mem_low.load(Ordering::SeqCst)
    }
    fn alloc_snapshot(&mut self) -> Option<u32> {
        self.free.pop()
    }
    fn release_prefix(&mut self, snap: u32) {
        if let Some(s) = self.pool.get_mut(snap as usize) {
            *s = None;
            if !self.free.contains(&snap) {
                self.free.push(snap);
            }
        }
    }
    fn prefix_snapshot(&mut self, slot: usize, snap: u32, len: usize) -> Result<(), gb_model::error::ModelError> {
        let st = &self.slots[slot];
        if len > st.len() || snap as usize >= self.pool.len() {
            return Err(gb_model::error::ModelError::Request(format!("snapshot of {len} of {} (snap {snap})", st.len())));
        }
        self.free.retain(|&s| s != snap);
        self.pool[snap as usize] = Some(st[..len].to_vec());
        Ok(())
    }
    fn restore_prefix(&mut self, slot: usize, snap: u32, len: usize) -> Result<(), gb_model::error::ModelError> {
        let sleep = self.faults.restore_sleep_ms.load(Ordering::SeqCst);
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_millis(sleep) && !self.reduce.abort.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(5));
        }
        match self.faults.restore.load(Ordering::SeqCst) {
            RESTORE_ERROR => return Err(gb_model::error::ModelError::Request("injected restore failure".into())),
            RESTORE_EMPTY => {
                self.slots[slot].clear();
                return Ok(());
            }
            RESTORE_STALE => return Ok(()),
            _ => {}
        }
        let s = self.pool.get(snap as usize).and_then(|s| s.as_ref()).filter(|s| s.len() == len).ok_or_else(|| {
            gb_model::error::ModelError::Request(format!("snapshot {snap} does not hold a prefix of length {len}"))
        })?;
        self.slots[slot] = s.clone();
        Ok(())
    }
    fn prefix_state_digest(&mut self, slot: usize, len: usize) -> Result<u64, gb_model::error::ModelError> {
        let st = &self.slots[slot];
        let bytes: Vec<u8> = st[..len.min(st.len())].iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(gb_cuda::state_digest::digest_host(&[&bytes, &(len as u64).to_le_bytes()]))
    }
    fn begin_slot(&mut self, slot: usize) -> Result<(), gb_model::error::ModelError> {
        self.slots[slot].clear();
        Ok(())
    }
    fn end_slot(&mut self, slot: usize) {
        self.slots[slot].clear();
    }
    fn prefill_chunk(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
    ) -> Result<Option<Candidates>, gb_model::error::ModelError> {
        // a model prefills on whatever state the slot holds: a wrong state is wrong numbers, not an error
        let mut st = std::mem::take(&mut self.slots[slot]);
        st.resize(start, 0);
        for (i, &t) in tokens.iter().enumerate() {
            let prev = st.last().copied().unwrap_or(0);
            st.push(self.link(prev, t, start + i));
        }
        let last = *st.last().unwrap();
        self.slots[slot] = st;
        let sum = self.reduce.reduce(self.rank, vec![last])?;
        Ok(want.map(|w| ChainRt::candidates(&sum, w)))
    }
    fn window_candidates(
        &mut self,
        b: &gb_model::rows::RowBatch,
        count: usize,
    ) -> Result<Candidates, gb_model::error::ModelError> {
        let mut vals = Vec::with_capacity(b.rows);
        for i in 0..b.n_seq() {
            let slot = b.seq_slot[i] as usize;
            let mut ext = self.slots[slot].clone();
            let mut w = Vec::new();
            for r in b.seq_rows(i) {
                let pos = b.positions[r] as usize;
                ext.resize(pos, 0);
                let prev = ext.last().copied().unwrap_or(0);
                let v = self.link(prev, b.ids[r] as u32, pos);
                ext.push(v);
                w.push(v);
                vals.push(v);
            }
            self.window[slot] = w;
        }
        let sum = self.reduce.reduce(self.rank, vals)?;
        Ok(ChainRt::candidates(&sum, count))
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        let w = std::mem::take(&mut self.window[slot]);
        self.slots[slot].extend_from_slice(&w[..keep.min(w.len())]);
    }
    fn flush(&mut self) -> Result<(), gb_model::error::ModelError> {
        Ok(())
    }
    fn prefill_logits_last_row(&mut self, _s: usize, _st: usize, _t: &[u32]) -> Result<Vec<f32>, gb_model::error::ModelError> {
        Err(gb_model::error::ModelError::NotBuilt("no scores on the chain mock".into()))
    }
    fn prefill_logits_rows(&mut self, _s: usize, _st: usize, _t: &[u32]) -> Result<Vec<Vec<f32>>, gb_model::error::ModelError> {
        Err(gb_model::error::ModelError::NotBuilt("no scores on the chain mock".into()))
    }
    fn propose_mtp(
        &mut self,
        _slot: usize,
        _tokens: &[u32],
        _max: usize,
    ) -> Result<Option<gb_model::families::deepseek_v41::model::DraftRound>, gb_model::error::ModelError> {
        Ok(None)
    }
}

type GroupEngine = gb_engine::Engine<Dsv41Backend<TpLeader<ChainRt>>>;

/// A two-rank group serving through the engine: rank 0's engine, rank 1's follower thread and its fault switches.
struct CacheGroup {
    eng: Option<GroupEngine>,
    follower: Option<std::thread::JoinHandle<Result<lockstep::FollowerExit, gb_model::error::ModelError>>>,
    faults: Arc<Faults>,
    /// The follower's snapshot count, published by its runtime at each pool query (see `follower_snapshots`).
    reduce: Arc<Allreduce>,
}

/// The engine's prefix policy exactly as the family's `finish_engine` sets it for this family (min cut 64, tail 48, ghost gain 128,
/// gap 96) over the environment's defaults.
fn dsv41_prefix_config() -> gb_engine::PrefixConfig {
    let mut c = gb_model::serve::prefix_config_from_env().unwrap();
    c.policy.min_cut = 64;
    c.policy.tail = 48;
    c.policy.ghost_min_gain = 128;
    c.policy.min_gap = 96;
    c
}

fn cache_group(slots: usize, max_context: usize, faults: Faults, deadlines: lockstep::CacheDeadlines) -> CacheGroup {
    let faults = Arc::new(faults);
    let reduce = Arc::new(Allreduce::default());
    let (ll, fl) = Link::pair(Duration::from_secs(10)).unwrap();
    let t = lockstep::TpTimeouts {
        ready: Duration::from_secs(30),
        peer: Duration::from_secs(60),
        hang_exit: None,
        ..Default::default()
    };
    let (fr, ff) = (reduce.clone(), faults.clone());
    let ft = t.clone();
    let follower = std::thread::spawn(move || {
        let rt = ChainRt::new(1, slots, max_context, fr.clone(), ff);
        let mut cfg = FollowerConfig::new(ft).with_rank(1);
        cfg.abort_hook = Some(Box::new(move || fr.abort.store(true, Ordering::SeqCst)));
        run_follower(rt, fl, cfg)
    });
    let leader = TpLeader::start(ChainRt::new(0, slots, max_context, reduce.clone(), Arc::new(Faults::default())), ll, t);
    leader.set_cache_deadlines(deadlines);
    let r = reduce.clone();
    leader.on_group_lost(Box::new(move || r.abort.store(true, Ordering::SeqCst)));
    // the backend exactly as the family builds it: its default pool (`REXL3_DSV41_PREFIX_SNAPSHOTS`, 32) sized on EVERY rank
    let backend = Dsv41Backend::new(leader)
        .with_spec(SpecMode::None, false)
        .with_default_prefix_cache(prefix_namespace("chain", "q4", false));
    let mm = gb_sched::ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 0, activation_reserve: 0 };
    let adm = gb_sched::Admission::new(1 << 40, 0, mm, slots)
        .unwrap()
        .with_max_context(max_context as u64)
        .with_kv_tokens((max_context * slots) as u64);
    let cfg = gb_engine::EngineConfig {
        max_rows: 8,
        max_drafts: 0,
        cost: gb_sched::CostModel::from_table(vec![(1, 1.0), (8, 1.0)]),
        drafting: false,
    };
    let eng =
        gb_engine::Engine::new(backend, adm, cfg, Box::new(|| Box::new(gb_engine::NoDrafts) as Box<dyn gb_engine::Drafter>))
            .with_prefix_cache(dsv41_prefix_config());
    CacheGroup { eng: Some(eng), follower: Some(follower), faults, reduce }
}

impl CacheGroup {
    fn eng(&mut self) -> &mut GroupEngine {
        self.eng.as_mut().unwrap()
    }
    fn leader(&mut self) -> &mut TpLeader<ChainRt> {
        &mut self.eng().backend.m
    }
    /// One request to its end: the reply's tokens.
    fn run(&mut self, prompt: &[u32], max_new: usize, cache: bool) -> Result<Vec<u32>, String> {
        let req = gb_engine::Request {
            prompt: prompt.to_vec(),
            sampling: gb_sampling::Sampling { seed: 7, temperature: 0.0, top_k: 1, top_p: 1.0, min_p: 0.0 },
            max_new,
            stop: vec![],
            draft: false,
            images: vec![],
            cache,
        };
        let id = self.eng().submit(req).map_err(|e| format!("{e:?}"))?;
        self.eng().run().map_err(|e| format!("{e:?}"))?;
        let mut outs = self.eng().take_outputs();
        outs.retain(|o| o.id == id);
        let o = outs.pop().ok_or("no output")?;
        if o.tokens.is_empty() {
            return Err(format!("an empty reply ({:?})", o.finish));
        }
        Ok(o.tokens)
    }
    fn stats(&mut self) -> gb_engine::PrefixStats {
        self.eng().prefix_stats()
    }
    /// The leader's view: the group is whole, every follower answer consumed.
    fn alive(&mut self) -> bool {
        self.leader().failure().is_none() && self.leader().barrier().is_ok()
    }
    /// Shuts down cleanly and returns the follower's runtime counts: (its pool's free slots, snapshots it holds by call).
    fn finish(mut self) -> Result<lockstep::FollowerExit, gb_model::error::ModelError> {
        drop(self.eng.take());
        self.follower.take().unwrap().join().unwrap()
    }
}

impl Drop for CacheGroup {
    fn drop(&mut self) {
        self.reduce.abort.store(true, Ordering::SeqCst);
        drop(self.eng.take());
        if let Some(f) = self.follower.take() {
            let _ = f.join();
        }
    }
}

/// Deterministic, non-degenerate tokens.
fn chain_toks(seed: u64, n: usize) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1 + (x % (CHAIN_VOCAB as u64 - 1)) as u32
        })
        .collect()
}

fn fast() -> lockstep::CacheDeadlines {
    lockstep::CacheDeadlines { ack: Duration::from_millis(500), stall: Duration::from_secs(20) }
}

/// The scenario every fault runs: turn 1 (a conversation of `p_len` tokens) fills the cache on both ranks; an unrelated cold request
/// leaves other state in the slot; then the follower is given `fault` and turn 2 (turn 1 plus more) must equal its own cold reply
/// byte for byte, the group must stay up, and the cache must still serve (a clean third turn hits and is exact). Returns (turn 2
/// hit?, restore failures).
fn restore_fault_scenario(fault: impl Fn(&Faults), max_context: usize, p_len: usize) -> Result<(bool, u64), String> {
    let mut g = cache_group(1, max_context, Faults::default(), fast());
    let t1 = chain_toks(11, p_len);
    let t2 = [t1.clone(), chain_toks(22, 300)].concat();
    let t3 = [t2.clone(), chain_toks(23, 300)].concat();
    let cold2 = g.run(&t2, 24, false)?;
    let cold3 = g.run(&t3, 24, false)?;
    g.run(&t1, 24, true)?;
    if g.stats().entries == 0 {
        return Err("turn 1 left no entry".into());
    }
    // other state in the slot on every rank (what a stale restore would leave)
    g.run(&chain_toks(99, p_len / 2 + 200), 8, false)?;
    fault(&g.faults);
    let before = g.stats();
    let warm2 = g.run(&t2, 24, true)?;
    let after = g.stats();
    if !g.alive() {
        return Err(format!("the group was lost: {:?}", g.leader().failure()));
    }
    if warm2 != cold2 {
        let at = warm2.iter().zip(&cold2).position(|(a, b)| a != b).unwrap_or(0);
        return Err(format!("turn 2 differs from its cold reply at token {at}"));
    }
    g.faults.clear();
    let hits = after.hits - before.hits;
    let failures = after.restore_failures - before.restore_failures;
    // the cache still serves, exactly
    let h3 = g.stats().hits;
    let warm3 = g.run(&t3, 24, true)?;
    if warm3 != cold3 || g.stats().hits != h3 + 1 {
        return Err(format!("the clean turn after the fault did not hit exactly (hits {} -> {})", h3, g.stats().hits));
    }
    match g.finish() {
        Ok(lockstep::FollowerExit::Shutdown(_)) => Ok((hits == 1, failures)),
        other => Err(format!("the follower did not end cleanly: {other:?}")),
    }
}

/// The baseline: no fault, the restore is verified on both ranks, and the warm turn equals its cold reply.
#[test]
fn a_two_rank_restore_verified_on_every_rank_equals_the_cold_reply() {
    assert_eq!(restore_fault_scenario(|_| {}, 65_536, 2_000).unwrap(), (true, 0));
}

/// Fault 1: the follower's restore returns an error -> both ranks go cold (the engine's fallback begins the slot on both), the reply
/// equals the cold one and the group stays up (before: a lost group, 503 and exit 70).
#[test]
fn a_follower_restore_error_falls_back_cold_on_both_ranks() {
    let r = restore_fault_scenario(|f| f.restore.store(RESTORE_ERROR, Ordering::SeqCst), 65_536, 2_000).unwrap();
    assert_eq!(r, (false, 1), "refused, counted, prefilled cold");
}

/// Fault 2a: the follower "restores" nothing but answers Ok (the 43baf947 mutant class) -> its content digest of the restored state
/// differs from its digest at snapshot time, PER REQUEST -> both cold. Without the digest check the reply differs (the mutant test).
#[test]
fn a_follower_that_restores_nothing_but_answers_ok_is_caught_per_request_and_goes_cold() {
    let r = restore_fault_scenario(|f| f.restore.store(RESTORE_EMPTY, Ordering::SeqCst), 65_536, 2_000).unwrap();
    assert_eq!(r, (false, 1));
}

/// Fault 2b: the follower leaves the previous request's state in the slot and answers Ok (a stale restore) -> caught by the digest.
#[test]
fn a_follower_stale_restore_that_answers_ok_is_caught_per_request_and_goes_cold() {
    let r = restore_fault_scenario(|f| f.restore.store(RESTORE_STALE, Ordering::SeqCst), 65_536, 2_000).unwrap();
    assert_eq!(r, (false, 1));
}

/// Fault 5a: a follower slower than the wait (rank 0's own restore time plus `CacheDeadlines::ack`) -> the restore fails over to a cold
/// prefill (its late answer consumed), exact, the group up.
#[test]
fn a_slow_follower_restore_falls_back_cold_and_keeps_the_group() {
    let r = restore_fault_scenario(|f| f.restore_sleep_ms.store(1_500, Ordering::SeqCst), 65_536, 2_000).unwrap();
    assert_eq!(r, (false, 1));
}

/// Fault 3: the follower's memory guard skips a capture rank 0 took -> the snapshot is refused and released on every rank before the
/// engine sees it: the index never holds it, nothing leaks, and the next turn prefills cold, exactly.
#[test]
fn a_snapshot_skipped_on_one_rank_is_never_entered_or_used_and_leaks_nothing() {
    let mut g = cache_group(1, 65_536, Faults::default(), fast());
    let t1 = chain_toks(31, 2_000);
    let t2 = [t1.clone(), chain_toks(42, 300)].concat();
    let cold2 = g.run(&t2, 16, false).unwrap();
    g.faults.mem_low.store(true, Ordering::SeqCst);
    g.run(&t1, 16, true).unwrap();
    assert_eq!(g.stats().entries, 0, "the skipped snapshot never entered the index");
    assert!(g.alive());
    assert_eq!(g.leader().inner().group_snapshots(), 0, "no snapshot is held by every rank");
    let pool = g.leader().prefix_pool_slots();
    assert_eq!(g.leader().prefix_free_snapshots(), pool, "rank 0 gave its slot back");
    let hits = g.stats().hits;
    assert_eq!(g.run(&t2, 16, true).unwrap(), cold2);
    assert_eq!(g.stats().hits, hits, "nothing to hit");
    // with the guard satisfied again the cache serves
    g.faults.mem_low.store(false, Ordering::SeqCst);
    g.run(&t1, 16, true).unwrap();
    assert!(g.stats().entries > 0);
    assert_eq!(g.run(&t2, 16, true).unwrap(), cold2);
    assert_eq!(g.stats().hits, hits + 1);
    assert!(g.alive());
    assert!(matches!(g.finish(), Ok(lockstep::FollowerExit::Shutdown(_))));
}

/// Fault 4 (boot): a follower that cannot hold rank 0's pool is refused at the Hello, by name, before any weight is read — and one
/// whose pool comes out smaller anyway fails the boot's `PrefixEnable` settle by name (PB31: the follower's pool is rank 0's).
#[test]
fn a_follower_pool_smaller_than_the_leaders_is_refused_at_the_boot() {
    let facts = |slots: u64| Dsv41Facts {
        kv: KvFmt::Q4,
        moe_topp: "1.0".into(),
        ced_replay: false,
        engram_token_map: "unset".into(),
        engram_shards: "unset".into(),
        spec: "none".into(),
        cache_request: "On".into(),
        cache_slots: slots,
    };
    let gate = |rank: usize, slots: u64| gate_of("pack".into(), 43, 1, 8192, 1_048_576, 128, facts(slots)).at(rank, 2);
    let why = lockstep::BootGate::check(&gate(0, 32), &gate(1, 8)).unwrap_err();
    assert!(why.contains("snapshot slots") && why.contains("32") && why.contains("8"), "{why}");
    assert!(lockstep::BootGate::check(&gate(0, 32), &gate(1, 32)).is_ok());
    assert!(lockstep::BootGate::check(&gate(0, 8), &gate(1, 32)).is_ok(), "a follower that can hold more is sized to rank 0's");
    let mut off = gate(1, 0);
    off.family.cache_request = "Off".into();
    assert!(lockstep::BootGate::check(&gate(0, 32), &off).unwrap_err().contains("--prefix-cache"));
    // past the Hello: the pool rank 0 sizes on every rank must come out equal, or the boot fails by name
    let g = cache_group(1, 65_536, Faults { pool_cap: AtomicU64::new(8), ..Default::default() }, fast());
    let mut g = g;
    let why = g.leader().failure().expect("the boot's pool settle fails the group");
    assert!(why.contains("prefix_enable") && why.contains("rank 1"), "{why}");
}

/// Fault 5b: a follower stuck in a restore (alive: its reader answers pings) -> the restore fails over at once, and at the stall
/// deadline the group is lost BY NAME instead of the cold prefill waiting in a collective for ever.
#[test]
fn a_stuck_follower_restore_is_an_explicit_group_loss_not_a_hang() {
    let d = lockstep::CacheDeadlines { ack: Duration::from_millis(200), stall: Duration::from_millis(1_500) };
    let mut g = cache_group(1, 65_536, Faults::default(), d);
    let t1 = chain_toks(51, 2_000);
    g.run(&t1, 8, true).unwrap();
    g.faults.restore_sleep_ms.store(600_000, Ordering::SeqCst);
    let t0 = Instant::now();
    let r = g.run(&[t1, chain_toks(62, 300)].concat(), 8, true);
    assert!(t0.elapsed() < Duration::from_secs(30), "bounded: {:?}", t0.elapsed());
    let why = g.leader().failure().expect("the group is lost");
    assert!(why.contains("rank 1") && why.contains("has not answered cache call") && why.contains("prefix_restore"), "{why}");
    assert!(r.is_err(), "the request fails by name, no reply is made up: {r:?}");
}

/// The pool under churn at the real pool size (32 slots, `REXL3_DSV41_PREFIX_SNAPSHOTS` default): more captures than slots, entries
/// evicted, one capture skipped on the follower midway; at the end rank 0's pool, the follower's pool, the group's book and the
/// index agree (no slot leaks on either rank), and every warm reply equalled its cold one.
#[test]
fn the_pools_of_both_ranks_stay_equal_under_churn_at_the_real_pool_size() {
    let mut g = cache_group(1, 65_536, Faults::default(), fast());
    assert_eq!(g.leader().prefix_pool_slots(), 32);
    let mut colds = Vec::new();
    for i in 0..40u64 {
        let t = [chain_toks(1000 + i, 600), chain_toks(2000 + i, 200)].concat();
        colds.push(g.run(&t, 6, false).unwrap());
        g.faults.mem_low.store(i == 17, Ordering::SeqCst);
        g.run(&t, 6, true).unwrap();
    }
    g.faults.mem_low.store(false, Ordering::SeqCst);
    for i in 30..40u64 {
        let t = [chain_toks(1000 + i, 600), chain_toks(2000 + i, 200)].concat();
        assert_eq!(g.run(&t, 6, true).unwrap(), colds[i as usize], "warm reply {i}");
    }
    let st = g.stats();
    assert!(st.evictions > 0 && st.hits > 0, "{st:?}");
    assert!(g.alive());
    let held = st.entries as usize;
    assert_eq!(g.leader().inner().group_snapshots(), held, "the book holds exactly the index's entries");
    assert_eq!(g.leader().prefix_free_snapshots(), 32 - held, "rank 0's pool holds exactly the index's entries");
    // the follower's pool: it answers the release of every entry; drop the engine and read it from the follower thread's runtime
    let out = g.finish();
    assert!(matches!(out, Ok(lockstep::FollowerExit::Shutdown(_))), "{out:?}");
}

/// The 1M profile's configuration (`dsv41-tp2-1m`: one slot, a 1,048,576-token window, the default 32-slot pool): a near-full-window
/// conversation is cached on both ranks, the follower restores stale state for the second turn, and the reply still equals its cold
/// one; then a clean turn hits.
#[test]
fn the_1m_profile_shape_falls_back_cold_exactly_on_a_stale_follower() {
    let r = restore_fault_scenario(|f| f.restore.store(RESTORE_STALE, Ordering::SeqCst), 1_048_576, 1_040_000).unwrap();
    assert_eq!(r, (false, 1));
}
