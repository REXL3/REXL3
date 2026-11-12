//! The Ling-3.1 backend's device allocations against its memory model, on the whole model (the real pack, MTP on, the profile's one slot and
//! its 32,768-token window): every buffer `BailingModel::new` and `BailingDevice::build` allocate beside the pack's regions is an item of
//! `alloc::AllocPlan` of the same bytes, and every item is allocated. `#[ignore]`, env-gated (`REXL3_LING_DIR`), one model per process, with
//! a 1 GiB `MemAvailable` guard that aborts the process.

mod ling31_support;

use gb_model::families::bailing_moe_v3::alloc::{AllocParams, AllocPlan};
use gb_model::families::bailing_moe_v3::engine::post_alloc_reserve_bytes;
use gb_model::families::bailing_moe_v3::{load, memory};
use gb_model::memory::MemoryOptions;
use ling31_support::*;

/// The profile's pool: the window and a page for the slot (`plan::plan_memory_cached_windowed` caps it there).
const KV_BLOCKS: usize = 32_768 / 256 + 1;

#[test]
#[ignore]
fn allocations_equal_the_plan() {
    let Some(m) = load(1, KV_BLOCKS * 256, true) else { return };
    let p = AllocParams {
        slots: 1,
        kv_blocks: KV_BLOCKS,
        prefill_rows: m.prefill_rows(),
        mtp: true,
        embed_in_place: load::embed_in_place(),
    };
    let plan = AllocPlan::new(&m.cfg, &p);
    for (name, bytes) in m.allocations() {
        eprintln!("{name:24} {:10.2} MiB", bytes as f64 / 1048576.0);
    }
    m.check_allocations(&plan, &p).unwrap_or_else(|e| panic!("the memory model and the allocations disagree: {e}"));
    // the boot's arithmetic is the same plan: the profile (slot state, pages, window records, scratch) and the after-plan reserve
    let o =
        MemoryOptions { prefill_rows: 4096, max_context: 32_768, mtp: true, vision: None, ..MemoryOptions::serving_defaults() };
    let profile = memory::memory_profile(&m.cfg, &memory::engine_options(&o)).unwrap();
    let booted = profile.state_bytes_per_slot()
        + KV_BLOCKS as u64 * profile.kv_page_bytes()
        + profile.window_record_bytes()
        + profile.scratch_bytes()
        + post_alloc_reserve_bytes(&m.cfg, 1, true, p.embed_in_place, 32_768);
    assert_eq!(booted, plan.total(&p), "the boot prices what the backend allocates");
    let all: u64 = m.allocations().iter().map(|(_, b)| b).sum();
    eprintln!(
        "allocated after the plan {:.3} GiB; MemAvailable {:.2} GiB (low {:.2})",
        gib(all),
        gib(mem_available()),
        gib(guard().low())
    );
}

/// The Ling-3.1 pipelined weight load against the serial reference (`REXL3_LOAD=serial`): every device region must hold the same bytes.
///
/// * `a_synthetic_plan_loads_the_same_bytes_both_ways`: a hand-made plan over two shards (plain copies small and larger than a staging
///   span, a bf16 widening, a `kv_b_proj` repack, three parts into one fused slot, alignment padding that stays zero) is loaded serially
///   and once per pipeline configuration (tiny spans, one slot, many workers, `O_DIRECT` forced and refused); each region is compared
///   with the image the test builds by hand. Needs a CUDA device; skips (and passes) without one.
/// * `the_real_pack_loads_the_same_bytes_both_ways`: the same comparison on the real pack (`#[ignore]`, `REXL3_LING_DIR`): the pack is
///   loaded serially, each region's sha256 is recorded, the regions are freed, the pack is loaded through the pipeline, and the digests
///   must be equal region by region. One 106 GiB image at a time. Run with `--ignored --nocapture --test-threads=1`.
mod load_pipeline {
    use gb_cuda::loader::{DirectMode, Options};
    use gb_exl3::budget::Category;
    use gb_model::families::bailing_moe_v3::load::{convert, load_plan_with, Regions};
    use gb_model::families::bailing_moe_v3::weights::ALIGN;
    use gb_model::families::bailing_moe_v3::{ItemPart, LoadItem, LoadPlan, Region, RegionKind, Xform};
    use gb_model::loadctx::LoadMode;
    use std::path::{Path, PathBuf};

    fn gpu() -> bool {
        if gb_cuda::device_available() {
            true
        } else {
            eprintln!("no CUDA device: skipping");
            false
        }
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.next() as u8).collect()
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("ling31_load_pipeline").join(name);
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn item(name: &str, file: u16, offset: u64, bytes: u64, xform: Xform, region: u32, dst: u64) -> LoadItem {
        LoadItem {
            name: name.into(),
            part: ItemPart::Tensor,
            category: Category::Decoder,
            file,
            offset,
            bytes,
            xform,
            dst_bytes: xform.dst_bytes(bytes),
            region,
            dst,
        }
    }

    fn region(id: u32, kind: RegionKind, bytes: u64) -> Region {
        Region { id, kind, bytes }
    }

    /// The device image a region must hold: `parts` placed at their offsets over zeros.
    fn image(bytes: u64, parts: &[(u64, Vec<u8>)]) -> Vec<u8> {
        let mut v = vec![0u8; bytes as usize];
        for (at, p) in parts {
            v[*at as usize..*at as usize + p.len()].copy_from_slice(p);
        }
        v
    }

    fn dump(r: &Regions, id: u32) -> Vec<u8> {
        let b = r.get(id).expect("region");
        let mut out = vec![0u8; b.bytes()];
        b.download(&mut out).unwrap();
        out
    }

    /// The plan, the shards it reads (written under `dir`) and the image each region must hold.
    fn synthetic(dir: &Path) -> (LoadPlan, Vec<(u32, Vec<u8>)>) {
        let mut rng = Rng(0x1CE);
        let files = vec!["shard-1.safetensors".to_string(), "shard-2.safetensors".to_string()];
        // shard 0: header bytes, then small tensors (some within a staging gap of each other), a 20 MiB tensor (several spans), a bf16
        // gamma, then the kv_b_proj (64 * 256 * 512 fp16, 16 MiB)
        let (lead, a, b, big, gamma_n, kvb_n) = (3333usize, 4096usize, 777usize, 20 << 20, 1000usize, 64 * 256 * 512 * 2);
        let s0 = rng.bytes(lead + a + 40_000 + b + 500_000 + big + 9 + gamma_n * 2 + 5000 + kvb_n + 17);
        let at_a = lead as u64;
        let at_b = at_a + a as u64 + 40_000;
        let at_big = at_b + b as u64 + 500_000;
        let at_gamma = at_big + big as u64 + 9;
        let at_kvb = at_gamma + gamma_n as u64 * 2 + 5000;
        // shard 1: three conv parts and one odd-sized tensor
        let (conv, odd) = (8192 * 4 * 2usize, 12_345usize);
        let s1 = rng.bytes(100 + 3 * conv + 50 + odd + 100);
        std::fs::write(dir.join(&files[0]), &s0).unwrap();
        std::fs::write(dir.join(&files[1]), &s1).unwrap();
        let pad = |n: u64| n.div_ceil(ALIGN) * ALIGN;
        // region 0 (global): a, big, gamma widened, b with alignment gaps between them
        let (off_a, off_big, off_gamma, off_b) = (0u64, pad(a as u64) + 256, pad(a as u64) + 256 + pad(big as u64) + 512, 0);
        let r0_bytes = off_gamma + pad(gamma_n as u64 * 4) + 256 + pad(b as u64);
        let off_b = off_gamma + pad(gamma_n as u64 * 4) + 256 + off_b;
        // region 1 (layer 0): kv_b repack, then a fused 3-part slot, then the odd tensor
        let (off_kvb, off_conv, off_odd) = (0u64, pad(kvb_n as u64) + 256, pad(kvb_n as u64) + 256 + pad(3 * conv as u64) + 256);
        let r1_bytes = off_odd + pad(odd as u64);
        let sl = |s: &[u8], at: u64, n: usize| s[at as usize..at as usize + n].to_vec();
        let conv_at = |i: u64| 100 + i * conv as u64;
        let odd_at = 100 + 3 * conv as u64 + 50;
        let mut items = vec![
            item("a", 0, at_a, a as u64, Xform::None, 0, off_a),
            item("b", 0, at_b, b as u64, Xform::None, 0, off_b),
            item("big", 0, at_big, big as u64, Xform::None, 0, off_big),
            item("gamma", 0, at_gamma, gamma_n as u64 * 2, Xform::Bf16ToF32, 0, off_gamma),
            item("kv_b", 0, at_kvb, kvb_n as u64, Xform::PackKvB, 1, off_kvb),
        ];
        for i in 0..3 {
            items.push(item("conv", 1, conv_at(i), conv as u64, Xform::None, 1, off_conv + i * conv as u64));
        }
        items.push(item("odd", 1, odd_at, odd as u64, Xform::None, 1, off_odd));
        items.sort_by_key(|i| (i.file, i.offset));
        let plan = LoadPlan {
            files,
            regions: vec![region(0, RegionKind::Global, r0_bytes), region(1, RegionKind::Trunk(0), r1_bytes)],
            items,
            skipped: Default::default(),
            markers: Vec::new(),
        };
        plan.validate().expect("the hand-made plan is consistent");
        // the images, built from the formulas the old inline loader used (independently of `convert`)
        let gamma_f32: Vec<u8> = s0[at_gamma as usize..][..gamma_n * 2]
            .chunks_exact(2)
            .flat_map(|c| ((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).to_le_bytes())
            .collect();
        let kvb_rows: Vec<u16> =
            s0[at_kvb as usize..][..kvb_n].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let (uk, uvt) =
            gb_cuda::mla_gated::pack_kv_b(&kvb_rows, gb_cuda::exl3_linear::Dtype::F16, gb_cuda::exl3_linear::Dtype::BF16);
        let kvb_dev: Vec<u8> = [&uk, &uvt].into_iter().flat_map(|v| v.iter().flat_map(|x| x.to_le_bytes())).collect();
        assert_eq!(
            kvb_dev,
            convert(Xform::PackKvB, &s0[at_kvb as usize..][..kvb_n]).unwrap(),
            "`convert` is the old inline repack"
        );
        assert_eq!(
            gamma_f32,
            convert(Xform::Bf16ToF32, &s0[at_gamma as usize..][..gamma_n * 2]).unwrap(),
            "`convert` is the old inline widening"
        );
        let img0 = image(
            r0_bytes,
            &[(off_a, sl(&s0, at_a, a)), (off_big, sl(&s0, at_big, big)), (off_gamma, gamma_f32), (off_b, sl(&s0, at_b, b))],
        );
        let img1 = image(
            r1_bytes,
            &[
                (off_kvb, kvb_dev),
                (off_conv, [sl(&s1, conv_at(0), conv), sl(&s1, conv_at(1), conv), sl(&s1, conv_at(2), conv)].concat()),
                (off_odd, sl(&s1, odd_at, odd)),
            ],
        );
        (plan, vec![(0, img0), (1, img1)])
    }

    fn options(workers: usize, slots: usize, span_kib: usize, direct: DirectMode) -> Options {
        Options { workers, slots, span_bytes: span_kib << 10, gap_bytes: 256 << 10, direct }
    }

    #[test]
    fn a_synthetic_plan_loads_the_same_bytes_both_ways() {
        if !gpu() {
            return;
        }
        let dir = tmp("synthetic");
        let (plan, want) = synthetic(&dir);
        let check = |label: &str, r: &Regions| {
            for (id, img) in &want {
                assert!(dump(r, *id) == *img, "{label}: region {id} differs from the hand-built image");
            }
        };
        let (serial, report) = load_plan_with(&dir, &plan, LoadMode::Serial, &Options::from_env(), &mut |_, _| {}).unwrap();
        assert!(report.is_none(), "the serial reference has no loader report");
        check("serial", &serial);
        drop(serial);
        for (label, o) in [
            ("default", options(8, 2, 8 << 10, DirectMode::Auto)),
            ("tiny spans, one slot, one worker", options(1, 1, 64, DirectMode::Never)),
            ("tiny spans, many workers", options(16, 2, 128, DirectMode::Never)),
            ("O_DIRECT forced", options(4, 2, 1024, DirectMode::Always)),
        ] {
            let mut calls = Vec::new();
            let (r, report) = load_plan_with(&dir, &plan, LoadMode::Pipeline, &o, &mut |d, t| calls.push((d, t))).unwrap();
            let report = report.expect("the pipeline reports");
            assert_eq!(report.jobs, plan.items.len(), "{label}: one job per item");
            assert_eq!(calls, vec![(plan.items.len(), plan.items.len())], "{label}: progress once, at the end");
            check(label, &r);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_conversion_error_names_the_item_and_fails_the_load() {
        if !gpu() {
            return;
        }
        let dir = tmp("badconv");
        let (mut plan, _) = synthetic(&dir);
        // a kv_b_proj of the wrong size is refused by name, not loaded
        let i = plan.items.iter().position(|i| i.name == "gamma").unwrap();
        plan.items[i].bytes -= 1;
        plan.items[i].dst_bytes = plan.items[i].xform.dst_bytes(plan.items[i].bytes);
        let e = load_plan_with(&dir, &plan, LoadMode::Pipeline, &Options::from_env(), &mut |_, _| {})
            .err()
            .expect("an odd bf16 byte count");
        assert!(e.to_string().contains("gamma"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // ------------------------------------------------------------------------------------------------------------ the real pack

    mod real {
        use super::*;
        use sha2::{Digest, Sha256};
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// sha256 of every region, `threads` regions at a time, each streamed through a 128 MiB host buffer.
        fn digests(r: &Regions, threads: usize) -> Vec<(u32, String)> {
            let regions: Vec<(u32, usize, usize)> = r.bufs.iter().map(|(&id, b)| (id, b.ptr() as usize, b.bytes())).collect();
            let next = AtomicUsize::new(0);
            let out = std::sync::Mutex::new(Vec::new());
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        let mut buf = vec![0u8; 128 << 20];
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some(&(id, ptr, bytes)) = regions.get(i) else { break };
                            let mut h = Sha256::new();
                            let mut at = 0usize;
                            while at < bytes {
                                let n = buf.len().min(bytes - at);
                                // SAFETY: `ptr + at` is device memory of the region, `buf` holds `n` bytes
                                unsafe { gb_cuda::rt::download_raw(ptr + at, buf.as_mut_ptr().cast(), n) }.unwrap();
                                h.update(&buf[..n]);
                                at += n;
                            }
                            out.lock().unwrap().push((id, format!("{:x}", h.finalize())));
                        }
                    });
                }
            });
            let mut v = out.into_inner().unwrap();
            v.sort();
            v
        }

        #[test]
        #[ignore = "needs the real pack (REXL3_LING_DIR) and a GB10"]
        fn the_real_pack_loads_the_same_bytes_both_ways() {
            let Ok(dir) = std::env::var("REXL3_LING_DIR") else {
                eprintln!("REXL3_LING_DIR is not set: not run");
                return;
            };
            let dir = PathBuf::from(dir);
            let cfg = gb_model::families::bailing_moe_v3::config::BailingConfig::from_dir(&dir).unwrap();
            let ck = gb_exl3::checkpoint::scan(&dir, false).unwrap();
            let inv = gb_model::families::bailing_moe_v3::inventory::PackInventory::from_checkpoint(&ck).unwrap();
            let pack = gb_model::families::bailing_moe_v3::load::BailingPack::from_inventory(&inv, cfg).unwrap();
            let o = gb_model::families::bailing_moe_v3::load::LoadOptions::from_env().loader;
            let t = std::time::Instant::now();
            let (serial, _) = load_plan_with(&dir, &pack.plan, LoadMode::Serial, &o, &mut |_, _| {}).unwrap();
            let t_serial = t.elapsed().as_secs_f64();
            let ds = digests(&serial, 8);
            let gib = serial.bytes() as f64 / (1u64 << 30) as f64;
            eprintln!(
                "serial: {} regions, {gib:.2} GiB, load {t_serial:.1} s, digests done {:.1} s",
                ds.len(),
                t.elapsed().as_secs_f64()
            );
            drop(serial);
            let t = std::time::Instant::now();
            let (piped, report) = load_plan_with(&dir, &pack.plan, LoadMode::Pipeline, &o, &mut |_, _| {}).unwrap();
            let report = report.unwrap();
            eprintln!(
                "pipeline: load {:.1} s (workers {}, spans {}, direct spans {})",
                t.elapsed().as_secs_f64(),
                report.workers,
                report.spans,
                report.direct_spans
            );
            let dp = digests(&piped, 8);
            assert_eq!(ds.len(), dp.len());
            let bad: Vec<_> = ds.iter().zip(&dp).filter(|(a, b)| a != b).map(|(a, _)| a.0).collect();
            let all = {
                let mut h = Sha256::new();
                for (id, d) in &dp {
                    h.update(id.to_le_bytes());
                    h.update(d.as_bytes());
                }
                format!("{:x}", h.finalize())
            };
            eprintln!("DIGEST regions {} gib {gib:.2} all {all}", dp.len());
            assert!(bad.is_empty(), "regions that differ between the serial and the pipelined load: {bad:?}");
        }
    }
}
