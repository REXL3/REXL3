//! GPU tests for stage 1 of the GLM serial forward: the plan realised on the device. Every test is `#[ignore]` and env-gated: it needs
//! REXL3_GLM_DIR (the pack) and a GPU, runs under the GPU lock (`nrun NAME --gpu -- ...`), refuses to start when `MemAvailable` is low and
//! aborts the process when it falls under 5 GiB. Without REXL3_GLM_DIR a test prints "not run" and passes.
//!
//! * `a_prefix_lands_where_the_plan_says`: the first four layers and the global region (embedding, final norm, `lm_head`) loaded through the
//!   pipelined loader; EVERY item compared with the shard's bytes through its conversion.
//! * `the_moe_layer_from_a_region_equals_the_one_from_groups`: layer 3's router, routed experts and shared expert built from the region
//!   (`MoeExperts::from_region`) and from the per-expert groups (`MoeExperts::from_groups`), run on the same rows: same picks, same bits.
//! * `the_whole_pack_boots` (REXL3_GLM_FULL=1): all 45 layers; cold boot time, memory, a sampled byte check.
//!
//! `REXL3_GLM_DIR=... cargo test --release -p gb-model --test glm_device -- --ignored --nocapture --test-threads=1`

mod glm_support;

use gb_cuda::exl3_linear::Dtype;
use gb_cuda::moe::{MoeDims, MoeLayer, MoeScratch, SharedExpert, SharedScratch, SigmoidTable};
use gb_cuda::{DevBuf, Stream};
use gb_exl3::checkpoint::scan;
use gb_model::families::glm::device::FfnDevice;
use gb_model::families::glm::load::{boot, LoadOptions};
use glm_support::*;

fn rand_f16(seed: u64, n: usize) -> Vec<u16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            let u = (z >> 40) as f32 / 16777216.0;
            gb_exl3::f16::f64_to_f16(((u * 2.0 - 1.0) * 3.0) as f64)
        })
        .collect()
}

#[test]
#[ignore]
fn a_prefix_lands_where_the_plan_says() {
    let Some(dir) = pack_dir() else { return };
    require_available(20 * GIB);
    let guard = Guard::start(5 * GIB);
    let b = boot(&dir, Some(4), &LoadOptions::from_env(8 * GIB)).unwrap();
    let min = guard.finish();
    eprintln!(
        "4 layers: {} items, {:.2} GiB in {} regions; scan+map {:.2} s, alloc {:.2} s, load {:.2} s ({} workers, {} spans, {} direct); MemAvailable min {:.1} GiB",
        b.plan.items.len(),
        gib(b.device.regions.bytes()),
        b.device.regions.count(),
        b.scan_s,
        b.alloc_s,
        b.load_s,
        b.report.workers,
        b.report.spans,
        b.report.direct_spans,
        gib(min)
    );
    assert_eq!(b.device.layers.len(), 4);
    let all: Vec<usize> = (0..b.plan.items.len()).collect();
    let n = verify_items(
        &b.plan,
        &dir,
        |it| b.device.regions.view(it.region, it.dst, it.dst_bytes).unwrap().to_vec::<u8>().unwrap(),
        &all,
    );
    eprintln!("verified {n} items byte for byte against the shards (conversions applied)");
    assert!(b.plan.markers.len() > 800, "layer 3's 864 expert groups are among the markers the load checked");
}

#[test]
#[ignore]
fn the_moe_layer_from_a_region_equals_the_one_from_groups() {
    let Some(dir) = pack_dir() else { return };
    require_available(24 * GIB);
    let guard = Guard::start(5 * GIB);
    let b = boot(&dir, Some(4), &LoadOptions::from_env(8 * GIB)).unwrap();
    let FfnDevice::Moe { layer: from_region, shared: shared_region } = &b.device.layers[3].ffn else {
        panic!("layer 3 is a MoE layer")
    };
    let ck = scan(&dir, true).unwrap();
    let from_groups = MoeLayer::from_checkpoint(&ck, 3, MoeDims::GLM53).unwrap();
    let shared_groups = SharedExpert::from_checkpoint(&ck, 3, MoeDims::GLM53.limit).unwrap();
    assert_eq!(from_groups.experts.bits_gu, from_region.experts.bits_gu);
    assert_eq!(from_groups.experts.expert_bytes, from_region.experts.expert_bytes);
    // the region holds the whole bank, the groups one allocation per tensor: the same bytes of weights
    eprintln!(
        "bank: {} B in the region, {} B from groups",
        from_region.experts.device_bytes(),
        from_groups.experts.device_bytes()
    );
    let st = Stream::new().unwrap();
    let tab = SigmoidTable::new().unwrap();
    for rows in [1usize, 9, 40, 128] {
        let x = DevBuf::from_slice(&rand_f16(rows as u64, rows * 4096)).unwrap();
        type Run = (Vec<f32>, Vec<i32>, Vec<u16>, Vec<f32>);
        let mut outs: Vec<Run> = Vec::new();
        for (layer, shared) in [(&from_groups, &shared_groups), (from_region, shared_region)] {
            let mut s = MoeScratch::new(&MoeDims::GLM53, 128).unwrap();
            let mut ss = SharedScratch::new(shared, 128).unwrap();
            let mut sh_out = DevBuf::zeroed(rows * 4096 * 4).unwrap();
            let mut out = DevBuf::zeroed(rows * 4096 * 4).unwrap();
            shared.forward(&x, Dtype::F16, rows, &mut sh_out, &mut ss, &st).unwrap();
            layer.forward(&x, Dtype::F16, rows, &tab, Some(&sh_out), &mut out, &mut s, &st).unwrap();
            st.sync().unwrap();
            outs.push((
                out.to_vec().unwrap(),
                s.pick.to_vec::<i32>().unwrap()[..rows * 8].to_vec(),
                s.wts.to_vec::<u16>().unwrap()[..rows * 8].to_vec(),
                sh_out.to_vec().unwrap(),
            ));
        }
        let (a, c) = (&outs[0], &outs[1]);
        assert_eq!(a.1, c.1, "{rows} rows: the router picked different experts");
        assert_eq!(a.2, c.2, "{rows} rows: the router weights differ");
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.3), bits(&c.3), "{rows} rows: the shared expert differs");
        assert_eq!(bits(&a.0), bits(&c.0), "{rows} rows: the layer output differs");
        eprintln!("{rows} rows: router, shared expert and routed experts from a region == from groups, bit for bit");
    }
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// A RANK's load (`GlmFamily::map_pack_owned` with experts 144..288): its expert regions hold only the owned experts (about half the
/// single node's), they bind as subbanks, and their pair outputs equal the single bank's `y` on every owned pair (`-0.0` elsewhere) for
/// picks routed by the replicated router.
#[test]
#[ignore]
fn a_rank_loads_only_its_experts_and_its_pairs_equal_the_single_banks() {
    use gb_cuda::moe::{ExpertOwnership, OwnScratch};
    use gb_model::families::glm::load::boot_mapped;
    use gb_model::families::glm::GlmFamily;
    let Some(dir) = pack_dir() else { return };
    require_available(24 * GIB);
    let guard = Guard::start(5 * GIB);
    let own = ExpertOwnership::range(288, 1, 2).unwrap();
    let pack = GlmFamily::map_pack_owned(&dir, &own).unwrap();
    let whole_pack = GlmFamily::map_pack(&dir).unwrap();
    let b = boot_mapped(&dir, pack, Some(4), &LoadOptions::from_env(8 * GIB), 0.0).unwrap();
    let min = guard.finish();
    let experts_region = |p: &gb_model::families::glm::plan::LoadPlan| {
        p.regions.iter().find(|r| r.kind == gb_model::families::glm::plan::RegionKind::Experts(3)).unwrap().bytes
    };
    let (rank_b, whole_b) = (experts_region(&b.plan), experts_region(&whole_pack.plan));
    let routed = |p: &gb_model::families::glm::plan::LoadPlan| -> u64 {
        p.items.iter().filter(|i| i.name.contains(".mlp.experts.")).map(|i| i.bytes).sum()
    };
    eprintln!(
        "rank 1 of 2, 4 layers: expert region of layer 3 {:.3} GiB (single node {:.3} GiB); routed bytes read {:.3} GiB of {:.3}; \
         MemAvailable min {:.1} GiB",
        gib(rank_b),
        gib(whole_b),
        gib(routed(&b.plan)),
        gib(routed(&whole_pack.plan.select_regions(gb_model::families::glm::load::keep_prefix(4)))),
        gib(min)
    );
    assert!(rank_b * 2 <= whole_b + 4096 && rank_b * 2 + 4096 >= whole_b);
    assert!(b.plan.items.iter().filter(|i| i.name.contains(".mlp.experts.")).all(|i| {
        let e: usize = i.name.split(".experts.").nth(1).unwrap().split('.').next().unwrap().parse().unwrap();
        own.owns(e)
    }));
    let FfnDevice::MoeSubbanks { layer: rank, .. } = &b.device.layers[3].ffn else {
        panic!("layer 3 binds as subbanks on a rank")
    };
    assert_eq!(rank.experts.ownership(), &own);
    assert_eq!(rank.experts.banks().len(), 1, "a uniform layer is one subbank");
    let ck = scan(&dir, true).unwrap();
    let whole = MoeLayer::from_checkpoint(&ck, 3, MoeDims::GLM53).unwrap();
    let st = Stream::new().unwrap();
    let tab = SigmoidTable::new().unwrap();
    for rows in [1usize, 9, 40, 128] {
        let x = DevBuf::from_slice(&rand_f16(rows as u64 + 7, rows * 4096)).unwrap();
        let p = rows * 8;
        let mut s = MoeScratch::new(&MoeDims::GLM53, 128).unwrap();
        let mut out = DevBuf::zeroed(rows * 4096 * 4).unwrap();
        whole.route(&x, Dtype::F16, rows, &tab, &mut s, &st).unwrap();
        whole.experts.run(&x, Dtype::F16, rows, None, &mut out, &mut s, true, &st).unwrap();
        st.sync().unwrap();
        let picks: Vec<i32> = s.pick.to_vec::<i32>().unwrap()[..p].to_vec();
        let y_ref: Vec<u32> = s.y_pairs.to_vec::<u32>().unwrap()[..p * 4096].to_vec();
        let mut sr = MoeScratch::new(&MoeDims::GLM53, 128).unwrap();
        sr.poison(0x7f, &st).unwrap();
        rank.route(&x, Dtype::F16, rows, &tab, &mut sr, &st).unwrap();
        let mut os = OwnScratch::new(&MoeDims::GLM53, 128).unwrap();
        rank.experts.run_pairs(&x, Dtype::F16, rows, &mut sr, &mut os, &st).unwrap();
        st.sync().unwrap();
        assert_eq!(sr.pick.to_vec::<i32>().unwrap()[..p], picks[..], "rows {rows}: the replicated router");
        let y: Vec<u32> = sr.y_pairs.to_vec::<u32>().unwrap()[..p * 4096].to_vec();
        let mut held = 0;
        for q in 0..p {
            let row = &y[q * 4096..(q + 1) * 4096];
            if own.owns(picks[q] as usize) {
                held += 1;
                assert_eq!(row, &y_ref[q * 4096..(q + 1) * 4096], "rows {rows} pair {q}");
            } else {
                assert!(row.iter().all(|&v| v == 0x8000_0000), "rows {rows} pair {q}: -0.0");
            }
        }
        eprintln!("rows {rows}: {held} of {p} pairs on this rank, byte-identical to the single bank");
    }
}

#[test]
#[ignore]
fn the_whole_pack_boots() {
    let Some(dir) = pack_dir() else { return };
    if std::env::var("REXL3_GLM_FULL").is_err() {
        eprintln!("REXL3_GLM_FULL is not set: not run (the whole pack is 92 GiB resident)");
        return;
    }
    require_available(100 * GIB);
    evict_pack(&dir);
    let before = mem_available();
    let guard = Guard::start(5 * GIB);
    let t = std::time::Instant::now();
    let b = boot(&dir, None, &LoadOptions::from_env(8 * GIB)).unwrap();
    let wall = t.elapsed();
    let after = mem_available();
    let min = guard.finish();
    eprintln!(
        "boot {:.2?}: scan+map {:.2} s, alloc {:.2} s ({} regions, {:.2} GiB), load {:.2} s ({:.2} GB/s, {} workers, {} spans of which {} direct, {} markers checked); MemAvailable {:.1} -> {:.1} GiB (min {:.1})",
        wall,
        b.scan_s,
        b.alloc_s,
        b.device.regions.count(),
        gib(b.device.regions.bytes()),
        b.load_s,
        b.report.bytes as f64 / 1e9 / b.load_s,
        b.report.workers,
        b.report.spans,
        b.report.direct_spans,
        b.plan.markers.len(),
        gib(before),
        gib(after),
        gib(min)
    );
    assert_eq!(b.device.layers.len(), 45);
    let picks = pick_items(&b.plan, 400);
    let n = verify_items(
        &b.plan,
        &dir,
        |it| b.device.regions.view(it.region, it.dst, it.dst_bytes).unwrap().to_vec::<u8>().unwrap(),
        &picks,
    );
    eprintln!("verified {n} items byte for byte (conversions applied)");
    let t = std::time::Instant::now();
    drop(b);
    eprintln!("freed in {:.2?}; MemAvailable {:.1} GiB", t.elapsed(), gib(mem_available()));
}
