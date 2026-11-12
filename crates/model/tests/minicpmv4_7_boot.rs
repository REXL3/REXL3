//! The MiniCPM-V-4.7 weights loader and the boot with no forward (E2 of `docs/MINICPMV4_7_INTEGRATION_PLAN.md`), on a real pack.
//!
//! `REXL3_MINICPM_DIR` names the EXL3 pack: the converter's 8-layer smoke pack (its config says 8 layers, so the boot is asked for the
//! truncated layer count) or the full 40-layer pack. The tests skip without it.
//!
//! * the header-only tests (no device, no weight read) bind the pack, refuse it by name in the ways a corrupted pack differs from
//!   the recipe, and refuse an unknown pack without `--allow-unvalidated-pack`;
//! * the device test (`#[ignore]`, GPU, run it under `flock -s /tmp/rexl3-gpu.lock`) boots the pack cold and warm, compares
//!   the weights on the device with their bytes in the shards (EXL3 parts, converted norms, the pointer tables), and checks that
//!   the state the boot allocated and the state at 1 to 8 slots are exactly what `memory::allocations` prices.
//!
//! ```text
//! REXL3_MINICPM_DIR=~/models/minicpm-v-4.7-35b-a3b/work/smokeB/o cargo test --release -p gb-model --test minicpmv4_7_boot -- --nocapture
//! flock -s -w 1800 /tmp/rexl3-gpu.lock env REXL3_MINICPM_DIR=... cargo test --release -p gb-model --test minicpmv4_7_boot -- --ignored --nocapture
//! ```
use gb_model::families::minicpmv4_7::boot::{boot_with, BootExtras, KC, WINDOW_ROWS};
use gb_model::families::minicpmv4_7::config::MiniCpmConfig;
use gb_model::families::minicpmv4_7::device::{MiniCpmWeights, MixerWeights};
use gb_model::families::minicpmv4_7::inventory::{TENSORS_PER_LAYER, VISION_BYTES, VISION_TENSORS};
use gb_model::families::minicpmv4_7::memory::allocations;
use gb_model::families::minicpmv4_7::state::{compare_with_plan, compare_with_plan_cached, ModelState};
use gb_model::families::minicpmv4_7::weights::{Source, WeightMap};
use gb_model::pack::Pack;
use gb_model::serve::ServeOptions;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The fingerprint of the converter's 8-layer smoke pack (`work/smokeB/o`, written 2026-10-12 19:33 CDT). TEST-ONLY: the validated
/// list `fingerprint::VALIDATED_MINICPMV4_7` stays empty until a full pack has passed its gates (E7). The pin makes this suite fail
/// loudly if the pack it runs on is not the one the numbers in `docs/MINICPMV4_7_INTEGRATION_PLAN.md` were taken on.
const SMOKE_8L_FINGERPRINT: &str = "sha256:fc7f0aa6b95a5b1ca529be0c52b284561ecb7122671c5d7bf1b90076fcbfc0da";

fn dir() -> Option<PathBuf> {
    match std::env::var("REXL3_MINICPM_DIR") {
        Ok(d) if Path::new(&d).join("config.json").exists() => Some(PathBuf::from(d)),
        _ => {
            eprintln!("not run: REXL3_MINICPM_DIR is not set to a pack");
            None
        }
    }
}

fn options(dir: &Path, slots: usize) -> ServeOptions {
    let mut o = ServeOptions::new(dir.to_path_buf());
    o.slots = slots;
    o.slots_auto = false;
    o.allow_unvalidated_pack = true;
    // no prefix cache unless a test asks (`prefix_options`): the default `auto` would take up to 8 GiB of pages and snapshots
    o.prefix_cache = gb_model::serve::PrefixCacheOption::Off;
    // a smaller KV pool for a box that others share (the pool is 5 GiB at the model's 262,144 tokens); the plan is the same arithmetic
    if let Some(n) = std::env::var("REXL3_MINICPM_KV_TOKENS").ok().and_then(|v| v.parse::<usize>().ok()) {
        o.kv_tokens = Some(n);
    }
    o
}

fn extras(cfg: &MiniCpmConfig) -> BootExtras {
    BootExtras { allow_truncated_layers: cfg.layers < 40 }
}

fn gb(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn file_bytes(dir: &Path, src: &Source) -> Vec<u8> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(dir.join(&src.file)).unwrap();
    f.seek(SeekFrom::Start(src.offset)).unwrap();
    let mut v = vec![0u8; src.bytes as usize];
    f.read_exact(&mut v).unwrap();
    v
}

/// Ask the kernel to drop the pack's pages from the page cache (a cold boot reads the disk).
fn evict(dir: &Path) {
    use std::os::unix::io::AsRawFd;
    for e in std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()) {
        if e.path().extension().is_some_and(|x| x == "safetensors") {
            let f = std::fs::File::open(e.path()).unwrap();
            // SAFETY: a valid descriptor; the advice is a hint
            unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        }
    }
}

// ----------------------------------------------------------------------------------------------------- header-only tests

#[test]
fn the_real_pack_binds_with_every_tensor_a_slot_and_its_extent_recorded() {
    let Some(dir) = dir() else { return };
    let cfg = MiniCpmConfig::from_dir(&dir).unwrap();
    cfg.check_shapes_for(true).unwrap();
    let t = Instant::now();
    let map = WeightMap::from_dir(&dir, &cfg).unwrap_or_else(|e| panic!("{} problems: {:?}", e.len(), &e[..e.len().min(8)]));
    eprintln!("bound {} tensors in {:.2} s", map.slots.len(), t.elapsed().as_secs_f64());
    assert_eq!(map.slots.len(), cfg.layers * TENSORS_PER_LAYER + 6 + VISION_TENSORS);
    let plan = map.load_plan();
    assert!(plan.items.iter().all(|s| s.source.is_some()), "every uploaded slot has its file extent");
    assert!(plan.markers.iter().all(|s| s.source.is_some()));
    assert_eq!((plan.recorded.len(), plan.recorded_bytes()), (VISION_TENSORS, VISION_BYTES));
    assert_eq!((map.lm_head_valid(), map.lm_head_width), (248_144, 248_192));
    // every extent of an uploaded slot is as long as the shape and dtype say (an f32 widening is made on the device)
    for s in &plan.items {
        let src = s.source.as_ref().unwrap();
        let elems: u64 = s.shape.iter().map(|&d| d as u64).product();
        let size = gb_exl3::checkpoint::dtype_size(&s.dtype).unwrap();
        assert_eq!(src.bytes, elems * size, "{}", s.name);
    }
    eprintln!(
        "{} layers: {} groups, bits {:?}, weights {:.3} GiB planned ({:.3} GiB payload), tower {:.3} GiB recorded in {} tensors",
        cfg.layers,
        map.groups.len(),
        map.bit_histogram(),
        gb(map.upload_bytes()),
        gb(map.payload_bytes()),
        gb(plan.recorded_bytes()),
        plan.recorded.len()
    );
}

#[test]
fn an_unknown_pack_is_refused_unless_the_operator_allows_it_and_the_family_names_what_it_refuses() {
    let Some(dir) = dir() else { return };
    let cfg = MiniCpmConfig::from_dir(&dir).unwrap();
    let x = extras(&cfg);
    let pack = Pack::open(&dir).unwrap();
    let fp = gb_model::fingerprint::fingerprint(&pack.ck);
    eprintln!("fingerprint {fp}");
    if let Some(name) = gb_model::fingerprint::validated_pack(&fp, None) {
        // a pack the validated list names (E7: the shipping pack) passes the check without the operator's flag; the boot of such a pack
        // is a device test (`minicpmv4_7_validated.rs`), so only the check itself runs here
        gb_model::serve::check_packs(&pack, None, false).unwrap_or_else(|e| panic!("{name} is listed and was refused: {e}"));
    } else {
        // a pack with no entry is refused before the device is touched
        let mut o = options(&dir, 1);
        o.allow_unvalidated_pack = false;
        let e = boot_with(&o, x).err().expect("an unvalidated pack booted").to_string();
        assert!(e.contains("not one this release was validated on") && e.contains("--allow-unvalidated-pack"), "{e}");
        assert!(e.contains(&fp), "the refusal names the pack by its fingerprint: {e}");
    }
    if cfg.layers == 8 {
        assert_eq!(fp, SMOKE_8L_FINGERPRINT, "this is not the smoke pack the suite's numbers were taken on");
    }
    // a truncated pack is no model: the serving boot refuses its layer count by key and value
    if cfg.layers < 40 {
        let e = boot_with(&options(&dir, 1), BootExtras::default()).err().expect("an 8-layer pack booted").to_string();
        assert!(e.contains("num_hidden_layers") && e.contains("(the kernels implement 40)"), "{e}");
    }
    // a draft that is no draft (the target's own directory) is refused by its config before anything is loaded; another window is
    // refused by the component that is not built
    let mut o = options(&dir, 1);
    o.draft_dir = Some(dir.clone());
    let e = boot_with(&o, x).err().unwrap().to_string();
    assert!(e.contains("--draft"), "{e}");
    let mut o = options(&dir, 1);
    o.window_rows = 32;
    assert!(boot_with(&o, x).err().unwrap().to_string().contains("--window-rows must be 64"));
}

/// The real pack's layout with one tensor more than the recipe knows: an MTP head in an extra shard the index lists. The scan
/// reads headers only, so this costs nothing and still runs the whole of the boot's refusal path on the real tensor names.
#[test]
fn a_pack_with_an_mtp_head_is_refused_by_the_tensor_name() {
    let Some(dir) = dir() else { return };
    let cfg = MiniCpmConfig::from_dir(&dir).unwrap();
    let tmp = std::env::temp_dir().join(format!("rexl3-minicpm-mtp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    for e in std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap()) {
        let name = e.file_name();
        if name == "model.safetensors.index.json" {
            continue;
        }
        std::os::unix::fs::symlink(e.path(), tmp.join(&name)).unwrap();
    }
    let mut index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).unwrap()).unwrap();
    index["weight_map"]["mtp.fc.weight"] = serde_json::json!("model-mtp.safetensors");
    std::fs::write(tmp.join("model.safetensors.index.json"), index.to_string()).unwrap();
    let head = br#"{"mtp.fc.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#;
    let mut shard = (head.len() as u64).to_le_bytes().to_vec();
    shard.extend_from_slice(head);
    shard.extend_from_slice(&[0u8; 8]);
    std::fs::write(tmp.join("model-mtp.safetensors"), shard).unwrap();
    let e = WeightMap::from_dir(&tmp, &cfg).unwrap_err();
    assert!(e.iter().any(|m| m.contains("mtp.fc.weight: present, but this checkpoint ships no MTP head")), "{e:?}");
    let e = boot_with(&options(&tmp, 1), extras(&cfg)).err().expect("a pack with an MTP head booted").to_string();
    assert!(e.contains("MiniCPM-V-4.7 pack") && e.contains("mtp.fc.weight"), "{e}");
    // and `build`'s capability report carries the same verdict (headers only)
    let v = gb_model::families::minicpmv4_7::boot::pack_verdict(&tmp).unwrap();
    assert!(v.starts_with("pack: ") && v.contains("problem") && v.contains("mtp.fc.weight"), "{v}");
    std::fs::remove_dir_all(tmp).unwrap();
}

// ------------------------------------------------------------------------------------------------------------ device test

fn widened(raw: &[u8], add_one: bool) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16) + if add_one { 1.0 } else { 0.0 })
        .collect()
}

#[test]
#[ignore]
fn the_pack_boots_cold_and_warm_with_the_plan_equal_to_the_allocations() {
    let Some(dir) = dir() else { return };
    if !gb_cuda::device_available() {
        return eprintln!("skipping: no CUDA device");
    }
    let cfg = MiniCpmConfig::from_dir(&dir).unwrap();
    let x = extras(&cfg);
    let map = WeightMap::from_dir(&dir, &cfg).unwrap();
    eprintln!(
        "pack {} ({} layers), {:.3} GiB of weights planned, {:.3} GiB free",
        dir.display(),
        cfg.layers,
        gb(map.upload_bytes()),
        gb(gb_model::serve::mem_available().unwrap_or(0))
    );

    // ---- cold: the pack's pages dropped, then the boot
    evict(&dir);
    let cold = boot_with(&options(&dir, 4), x).unwrap_or_else(|e| panic!("cold boot: {e}"));
    let r = &cold.report;
    eprintln!("COLD {}", r.line());
    assert_eq!(r.tensors, cfg.layers * TENSORS_PER_LAYER + 6 + VISION_TENSORS);
    assert_eq!((cold.valid_vocab(), cold.logits_width()), (248_144, 248_192));
    assert_eq!(KC, 4);
    assert_eq!(r.slots, 4);
    // the arena carved the planned bytes exactly (the loader's views are the plan's items), and the buffers sum to the payload
    assert_eq!(r.weights_arena_used, map.upload_bytes());
    assert_eq!(r.weights_planned, map.upload_bytes());
    assert_eq!(cold.weights.bytes as u64, map.payload_bytes() + r.pointer_tables);
    assert!(r.weights_arena_reserved >= r.weights_arena_used);
    // the tower is recorded and not on the device
    assert_eq!((cold.weights.tower.tensors, cold.weights.tower.bytes), (VISION_TENSORS, VISION_BYTES));
    // the state the boot allocated is what the plan prices, at the slots and pages the plan granted
    let items = allocations(&cfg, None);
    let diffs = compare_with_plan(&items, &cold.state.bytes(), r.slots, r.kv_pages, WINDOW_ROWS);
    assert!(diffs.is_empty(), "{diffs:?}");
    assert_eq!(cold.state.bytes(), r.state);
    // the kernel gates ran over the six shapes of the target
    let shapes: Vec<(usize, usize)> = r.shapes.iter().map(|s| (s.k, s.n)).collect();
    assert_eq!(shapes, vec![(512, 2048), (2048, 512), (2048, 4096), (2048, 8192), (2048, 248_192), (4096, 2048)]);
    // the capability line and the engine
    let line = cold.capability_line();
    assert!(line.contains("text forward built") && line.contains("vision_minicpmv") && line.contains("DSpark draft"), "{line}");
    // memory: what the box lost against what the plan holds
    let fell = r.host_drop().unwrap_or(0);
    eprintln!(
        "RESIDENT {:.3} GiB (arena blocks {:.3} + tables {:.3} + state {:.3}); MemAvailable fell {:.3} GiB; plan: weights {:.3} + \
         slot state {} x {:.3} MiB + KV {} pages x {:.3} MiB; engine reserve {:.3} GiB (window buffers {:.3} GiB inside it)",
        gb(r.resident_bytes()),
        gb(r.weights_arena_reserved),
        gb(r.pointer_tables),
        gb(r.state.total()),
        fell as f64 / (1u64 << 30) as f64,
        gb(r.weights_planned),
        r.slots,
        cold.plan.memory.state_bytes_per_slot as f64 / (1 << 20) as f64,
        r.kv_pages,
        cold.plan.memory.block_bytes as f64 / (1 << 20) as f64,
        gb(r.engine_reserve),
        gb(r.state.gdn_records + r.state.gdn_raw_qkv)
    );
    assert!(
        (fell - r.resident_bytes() as i64).unsigned_abs() < 3 << 30,
        "MemAvailable fell {fell} B against {} B resident",
        r.resident_bytes()
    );

    // ---- the weights on the device against their bytes in the shards: every item, then the named samples and the pointer tables
    check_every_item(&dir, &map, &cold.weights);
    check_weights(&dir, &cfg, &map, &cold.weights);

    // ---- the engine around the booted model (E3): a request through it, and the options it has no component for refused by name
    let mut eng = cold.into_engine(&options(&dir, 4)).unwrap_or_else(|e| panic!("the engine: {e}"));
    let prompt: Vec<u32> = (0..40).map(|i| 100 + i as u32 * 7).collect();
    let greedy = gb_sampling::Sampling { seed: 0, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 };
    let req =
        gb_engine::Request { prompt, sampling: greedy, max_new: 6, stop: vec![], draft: false, images: vec![], cache: false };
    eng.submit_deferred(req).unwrap();
    for _ in 0..400 {
        if !eng.busy() {
            break;
        }
        eng.step().unwrap();
    }
    let out = eng.take_outputs();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].tokens.len(), 6);
    assert!(out[0].tokens.iter().all(|&t| t < 248_144), "{:?}", out[0].tokens);
    drop(eng);

    // ---- warm: the pages are in the cache now
    let warm = boot_with(&options(&dir, 4), x).unwrap_or_else(|e| panic!("warm boot: {e}"));
    eprintln!("WARM {}", warm.report.line());
    assert_eq!(warm.report.weights_arena_used, map.upload_bytes());
    assert_eq!(warm.report.state, warm.state.bytes());
    drop(warm);

    // ---- the plan equals the allocations at 1 to 8 slots, on real buffers
    for slots in 1..=8usize {
        let pages = 3 + slots;
        let mut st = ModelState::alloc(&cfg, slots, pages, WINDOW_ROWS).unwrap();
        // the pools are laid out for the 16 x 32 head shape and the cache for 16 query / 2 KV heads of 256, at the pool's pages
        assert!(st
            .kv
            .iter()
            .all(|c| c.capacity_tokens() == pages * 256 && c.shape == gb_model::families::minicpmv4_7::state::ATTN));
        assert!(st
            .window
            .records
            .iter()
            .all(|r| r.shape == gb_model::families::minicpmv4_7::state::GDN && r.rows == WINDOW_ROWS));
        // a slot reset zeroes that slot's state in every layer and no other's
        if slots >= 2 {
            let stream = gb_cuda::Stream::new().unwrap();
            // the first layer's state and the last layer's history stand for every layer (the reset walks them all)
            let n = st.gdn.s[0].bytes();
            st.gdn.s[0].upload(&vec![0xFFu8; n]).unwrap();
            let last = st.gdn.conv.len() - 1;
            let n = st.gdn.conv[last].bytes();
            st.gdn.conv[last].upload(&vec![0xFFu8; n]).unwrap();
            st.gdn.reset_slot(1, &stream).unwrap();
            stream.sync().unwrap();
            let (s_slot, c_slot) = (2_097_152usize, 49_152usize);
            let s0 = st.gdn.s[0].to_vec::<u8>().unwrap();
            let c0 = st.gdn.conv.last().unwrap().to_vec::<u8>().unwrap();
            assert!(s0[s_slot..2 * s_slot].iter().all(|&b| b == 0), "slot 1 of the delta-rule state was not zeroed");
            assert!(s0[..s_slot].iter().chain(&s0[2 * s_slot..]).all(|&b| b == 0xFF), "another slot's state was touched");
            assert!(c0[c_slot..2 * c_slot].iter().all(|&b| b == 0), "slot 1 of the convolution history was not zeroed");
            assert!(c0[..c_slot].iter().chain(&c0[2 * c_slot..]).all(|&b| b == 0xFF), "another slot's history was touched");
        }
        let diffs = compare_with_plan(&items, &st.bytes(), slots, pages, WINDOW_ROWS);
        assert!(diffs.is_empty(), "{slots} slots: {diffs:?}");
        // E6b: the same state with a prefix cache's arena of `slots + 1` snapshot slots (the pool's pages include the cache's)
        let snapshots = slots + 1;
        let cached = ModelState::alloc_cached(&cfg, slots, pages + 8 * snapshots, WINDOW_ROWS, snapshots).unwrap();
        let held = cached.bytes();
        assert_eq!(held.prefix_snapshots, snapshots as u64 * cfg.linear_layers() as u64 * (2_097_152 + 49_152));
        assert_eq!(cached.snapshots.as_ref().unwrap().bytes() as u64, held.prefix_snapshots);
        let diffs = compare_with_plan_cached(&items, &held, slots, pages + 8 * snapshots, WINDOW_ROWS, snapshots);
        assert!(diffs.is_empty(), "{slots} slots, {snapshots} snapshot slots: {diffs:?}");
        // a plan that counts one snapshot slot fewer or more than the device holds is named
        for wrong in [snapshots - 1, snapshots + 1] {
            let d = compare_with_plan_cached(&items, &held, slots, pages + 8 * snapshots, WINDOW_ROWS, wrong);
            assert!(d.iter().any(|m| m.starts_with("prefix_snapshot: the device holds")), "{d:?}");
        }
        drop(cached);
        let b = st.bytes();
        assert_eq!(b.gdn_state, cfg.linear_layers() as u64 * slots as u64 * 2_097_152);
        assert_eq!(b.gdn_conv_history, cfg.linear_layers() as u64 * slots as u64 * 49_152);
        assert_eq!(b.kv_target, pages as u64 * 256 * cfg.full_layers() as u64 * 2_048);
        assert_eq!(b.gdn_records + b.gdn_raw_qkv, WINDOW_ROWS as u64 * cfg.linear_layers() as u64 * 33_024);
    }
    eprintln!("plan == allocations at 1..=8 slots ({} GDN layers, {} attention layers)", cfg.linear_layers(), cfg.full_layers());
}

/// Every uploaded tensor of the pack, on the device, against its bytes in the shards through the conversion the map names (the
/// copy as stored, or bf16 widened to fp32 with or without the `+1`); and the set of buffers the structs hold is exactly the load
/// plan's items, so no slot was left out and none was invented.
fn check_every_item(dir: &Path, map: &WeightMap, w: &MiniCpmWeights) {
    use gb_model::families::minicpmv4_7::weights::Transform;
    let t = Instant::now();
    let held = w.named_buffers();
    let want: std::collections::BTreeSet<String> = map.load_plan().items.iter().map(|s| s.name.clone()).collect();
    let got: std::collections::BTreeSet<String> = held.iter().map(|(n, _)| n.clone()).collect();
    assert_eq!(held.len(), got.len(), "a buffer is named twice");
    assert_eq!(got, want, "the buffers the structs hold are not the load plan's items");
    let mut bytes = 0u64;
    for (name, dev) in &held {
        let slot = &map.slots[name];
        let raw = file_bytes(dir, slot.source.as_ref().unwrap());
        let got = dev.to_vec::<u8>().unwrap();
        match slot.transform {
            Transform::None => assert!(got == raw, "{name}: the device copy differs from the shard"),
            t @ (Transform::Bf16ToF32 | Transform::Bf16AddOneToF32) => {
                let want: Vec<u8> = widened(&raw, t == Transform::Bf16AddOneToF32).iter().flat_map(|f| f.to_le_bytes()).collect();
                assert!(got == want, "{name}: the widened copy differs ({t:?})");
            }
            Transform::Bf16ToF16 => eprintln!("{name}: Bf16ToF16 is not byte-checked by this comparison"),
        }
        bytes += got.len() as u64;
    }
    eprintln!(
        "every upload item ({} tensors, {:.3} GiB) equals its bytes in the shards, through its conversion: {:.1} s",
        held.len(),
        gb(bytes),
        t.elapsed().as_secs_f64()
    );
}

/// The device's copy of a sample of the weights equals what the shards hold (through the conversion the map names).
fn check_weights(dir: &Path, cfg: &MiniCpmConfig, map: &WeightMap, w: &MiniCpmWeights) {
    let src = |name: &str| map.slots[name].source.clone().unwrap_or_else(|| panic!("{name}: no extent"));
    let raw = |name: &str| file_bytes(dir, &src(name));
    let lm = "model.language_model";
    let last = cfg.layers - 1;
    // plain tensors: copied as stored
    let embed = w.embed.to_vec::<u8>().unwrap();
    assert!(embed == raw(&format!("{lm}.embed_tokens.weight")), "embedding differs");
    let l0 = &w.layers[0];
    let MixerWeights::Linear(m0) = &l0.mixer else { panic!("layer 0 is a gated delta net") };
    let eq = |dev: &gb_cuda::DevBuf, name: &str| {
        let got = dev.to_vec::<u8>().unwrap();
        assert!(got == raw(name), "{name}: the device copy differs from the shard");
    };
    eq(&m0.conv_w, &format!("{lm}.layers.0.linear_attn.conv1d.weight"));
    eq(&m0.in_proj_a, &format!("{lm}.layers.0.linear_attn.in_proj_a.weight"));
    eq(&m0.in_proj_b, &format!("{lm}.layers.0.linear_attn.in_proj_b.weight"));
    eq(&l0.moe.router, &format!("{lm}.layers.0.mlp.gate.weight"));
    eq(&l0.moe.shared_gate, &format!("{lm}.layers.0.mlp.shared_expert_gate.weight"));
    // widened norms: with and without the `+1`
    let f32s = |b: &gb_cuda::DevBuf| b.to_vec::<f32>().unwrap();
    assert_eq!(f32s(&l0.input_norm), widened(&raw(&format!("{lm}.layers.0.input_layernorm.weight")), true));
    assert_eq!(f32s(&l0.post_norm), widened(&raw(&format!("{lm}.layers.0.post_attention_layernorm.weight")), true));
    assert_eq!(f32s(&m0.norm), widened(&raw(&format!("{lm}.layers.0.linear_attn.norm.weight")), false));
    assert_eq!(f32s(&m0.a_log), widened(&raw(&format!("{lm}.layers.0.linear_attn.A_log")), false));
    assert_eq!(f32s(&m0.dt_bias), widened(&raw(&format!("{lm}.layers.0.linear_attn.dt_bias")), false));
    assert_eq!(f32s(&w.final_norm), widened(&raw(&format!("{lm}.norm.weight")), true));
    let MixerWeights::Full(a3) = &w.layers[3].mixer else { panic!("layer 3 is full attention") };
    assert_eq!(f32s(&a3.q_norm), widened(&raw(&format!("{lm}.layers.3.self_attn.q_norm.weight")), true));
    assert_eq!(f32s(&a3.k_norm), widened(&raw(&format!("{lm}.layers.3.self_attn.k_norm.weight")), true));
    // EXL3 groups: trellis and the fp16 scales as stored, for a mixer, the shared expert, the first and last expert and the head
    let checked = std::cell::Cell::new(0usize);
    let group = |g: &gb_cuda::exl3_linear::Exl3Weights, prefix: &str| {
        checked.set(checked.get() + 1);
        for (part, dev) in g.buffers() {
            let got = dev.to_vec::<u8>().unwrap();
            assert!(got == raw(&format!("{prefix}.{part}")), "{prefix}.{part}: the device copy differs from the shard");
        }
        let grp = &map.groups[prefix];
        assert_eq!((g.k, g.n, g.bits), (grp.k, grp.n, (grp.bits.twice() / 2) as usize), "{prefix}");
    };
    group(&m0.in_proj_qkv, &format!("{lm}.layers.0.linear_attn.in_proj_qkv"));
    group(&a3.q_proj, &format!("{lm}.layers.3.self_attn.q_proj"));
    assert_eq!((a3.q_proj.k, a3.q_proj.n), (2048, 8192), "q_proj is one group of query and gate");
    group(&l0.moe.shared_expert.up, &format!("{lm}.layers.0.mlp.shared_expert.up_proj"));
    for e in [0usize, 137, 255] {
        group(&w.layers[last].moe.experts[e].down, &format!("{lm}.layers.{last}.mlp.experts.{e}.down_proj"));
        group(&w.layers[0].moe.experts[e].gate, &format!("{lm}.layers.0.mlp.experts.{e}.gate_proj"));
    }
    group(&w.lm_head, "lm_head");
    assert_eq!((w.lm_head.k, w.lm_head.n), (2048, 248_192));
    // the pointer tables the MoE kernels index: expert e's trellis, scales and widths, and the shared expert past the routed ones
    for layer in [0usize, last] {
        let m = &w.layers[layer].moe;
        let tr = m.g_trellis_ptrs.to_vec::<u64>().unwrap();
        let su = m.u_suh_ptrs.to_vec::<u64>().unwrap();
        let sv = m.d_svh_ptrs.to_vec::<u64>().unwrap();
        let kt = m.k_tab.to_vec::<i32>().unwrap();
        assert_eq!((tr.len(), kt.len()), (257, 257 * 3));
        for e in [0usize, 1, 100, 255] {
            let x = &m.experts[e];
            assert_eq!(tr[e], x.gate.buffers()[0].1.ptr() as u64);
            assert_eq!(su[e], x.up.buffers()[1].1.ptr() as u64);
            assert_eq!(sv[e], x.down.buffers()[2].1.ptr() as u64);
            assert_eq!(&kt[e * 3..e * 3 + 3], &[x.gate.bits as i32, x.up.bits as i32, x.down.bits as i32]);
        }
        let s = &m.shared_expert;
        assert_eq!(tr[256], s.gate.buffers()[0].1.ptr() as u64);
        assert_eq!(&kt[256 * 3..], &[s.gate.bits as i32, s.up.bits as i32, s.down.bits as i32]);
    }
    eprintln!(
        "weights: embedding, norms, parameters, {} EXL3 groups and the pointer tables of layers 0 and {last} equal the shards",
        checked.get()
    );
}

// ------------------------------------------------------------------------------------------- a synthetic full-size pack

const TABLE: &str = include_str!("../../../testdata/minicpmv4_7/tensor_inventory.tsv");
const CONFIG: &str = include_str!("../../../testdata/minicpmv4_7/config.json");

/// One sparse shard holding the committed inventory's 124,705 tensors at their names, dtypes and shapes (the nominal widths: experts 4
/// bits, the rest 6), every weight zero and every codebook marker as the converter writes it: a pack of the real pack's layout and
/// size that costs no disk and no conversion, so the 40-layer boot (counts, arena blocks, plan, pointer tables) can be exercised
/// before and beside the real pack. Returns the shard's logical size.
fn write_synthetic_pack(dir: &Path) -> u64 {
    use std::os::unix::fs::FileExt;
    let tensors = gb_model::families::minicpmv4_7::inventory::expand_table(TABLE).unwrap();
    let mut head = serde_json::Map::new();
    head.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
    let (mut at, mut markers) = (0u64, Vec::new());
    for (name, (dtype, shape)) in &tensors {
        let bytes = shape.iter().map(|&d| d as u64).product::<u64>() * gb_exl3::checkpoint::dtype_size(dtype).unwrap();
        head.insert(name.clone(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [at, at + bytes]}));
        if name.ends_with(".mul1") {
            markers.push(at);
        }
        at += bytes;
    }
    let text = serde_json::Value::Object(head).to_string().into_bytes();
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), CONFIG).unwrap();
    let f = std::fs::File::create(dir.join("model-00001-of-00001.safetensors")).unwrap();
    f.write_all_at(&(text.len() as u64).to_le_bytes(), 0).unwrap();
    f.write_all_at(&text, 8).unwrap();
    let base = 8 + text.len() as u64;
    for m in markers {
        f.write_all_at(&gb_exl3::MARKER_MUL1.to_le_bytes(), base + m).unwrap();
    }
    f.set_len(base + at).unwrap();
    base + at
}

/// The 40-layer path at full size through the serving options (no truncated-layer extra): `REXL3_MINICPM_SYNTH_DIR` names a
/// directory on a disk with room for nothing but a sparse file's metadata.
#[test]
#[ignore]
fn a_synthetic_full_size_pack_boots_through_the_serving_path() {
    use gb_model::family::Registry;
    let Ok(parent) = std::env::var("REXL3_MINICPM_SYNTH_DIR") else {
        return eprintln!("not run: REXL3_MINICPM_SYNTH_DIR is not set");
    };
    if !gb_cuda::device_available() {
        return eprintln!("skipping: no CUDA device");
    }
    let dir = Path::new(&parent).join(format!("minicpm-synth40-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let t = Instant::now();
    let size = write_synthetic_pack(&dir);
    eprintln!("synthetic pack: {:.2} GiB logical, written in {:.2} s", gb(size), t.elapsed().as_secs_f64());

    // the registry's probe accepts it (40 layers) with every component of the text engine built (E3), and the family offers the
    // prefix cache (E6b); `build` itself is the boot below plus the engine (E3's boot test goes through `into_engine`)
    let (family, plan) = Registry::global().probe(&dir).unwrap();
    assert!(plan.missing_components().is_empty(), "{:?}", plan.missing_components());
    assert!(family.prefix_cache());
    let o = options(&dir, 4);

    // the boot, with the production options: the unknown pack needs the operator's word, the layer count is the model's
    let mut refused = options(&dir, 4);
    refused.allow_unvalidated_pack = false;
    let e = boot_with(&refused, BootExtras::default()).err().expect("an unvalidated pack booted").to_string();
    assert!(e.contains("--allow-unvalidated-pack"), "{e}");
    let cold = boot_with(&o, BootExtras::default()).unwrap_or_else(|e| panic!("boot: {e}"));
    let r = &cold.report;
    eprintln!("SYNTH {}", r.line());
    assert_eq!((r.layers, r.tensors, r.groups), (40, 124_705, 30_971));
    assert_eq!(r.bit_histogram.get("4"), Some(&30_720));
    assert_eq!(r.bit_histogram.get("6"), Some(&(30 * 3 + 10 * 4 + 40 * 3 + 1)));
    assert_eq!((r.slots, r.kv_pages, r.kv_tokens), (4, 262_144 / 256 + 4, 262_144));
    // the weights: the arena carved the planned bytes, the tables are 40 x (9 x 257 x 8 + 257 x 3 x 4) bytes
    assert_eq!(r.weights_arena_used, r.weights_planned);
    assert_eq!(r.pointer_tables, 40 * (9 * 257 * 8 + 257 * 3 * 4));
    assert_eq!(cold.weights.bytes as u64, r.weights_payload + r.pointer_tables);
    assert_eq!((cold.weights.tower.tensors, cold.weights.tower.bytes), (VISION_TENSORS, VISION_BYTES));
    // the plan's own constants, on the real buffers: 64,389,120 B a slot, 20,480 B a token, 990,720 B a window row (one record set)
    let s = &r.state;
    assert_eq!(s.gdn_state + s.gdn_conv_history, 4 * 64_389_120);
    assert_eq!(s.kv_target, (262_144 / 256 + 4) as u64 * 256 * 20_480);
    assert_eq!(s.gdn_records + s.gdn_raw_qkv, 64 * 990_720);
    let cfg = MiniCpmConfig::from_dir(&dir).unwrap();
    assert!(compare_with_plan(&allocations(&cfg, None), s, 4, r.kv_pages, 64).is_empty());
    // the expert pointer tables of the last layer point at distinct trellises
    let tr = cold.weights.layers[39].moe.g_trellis_ptrs.to_vec::<u64>().unwrap();
    assert_eq!(tr.len(), 257);
    assert!(tr.iter().all(|&p| p != 0) && tr.windows(2).all(|w| w[0] != w[1]));
    let line = cold.capability_line();
    assert!(line.contains("124705 tensors") && line.contains("images and video built"), "{line}");
    drop(cold);
    let warm = boot_with(&o, BootExtras::default()).unwrap();
    eprintln!("SYNTH-WARM {}", warm.report.line());
    drop(warm);

    // E6b: the same boot with a prefix cache of 1,500 MiB. The pool at the default 262,144 tokens is the admitted share plus the
    // cache's pages, the arena is the plan's snapshot slots, every item is priced, and the engine built around it runs the cache.
    let mut oc = options(&dir, 4);
    oc.prefix_cache = gb_model::serve::PrefixCacheOption::Mib(1_500);
    let cached = boot_with(&oc, BootExtras::default()).unwrap_or_else(|e| panic!("boot with the cache: {e}"));
    let (cache, r) = (cached.plan.memory.cache, &cached.report);
    eprintln!("SYNTH-CACHE {}", r.line());
    let unit = 64_389_120u64 + 8 * 256 * 20_480;
    assert_eq!(
        (cache.snapshot_slots as u64, cache.page_blocks as u64),
        (1_500u64 * (1 << 20) / unit, 8 * (1_500u64 * (1 << 20) / unit))
    );
    assert_eq!(r.kv_tokens, 262_144, "the admitted pool is the one without a cache");
    assert_eq!(
        (r.slots, r.kv_pages, r.snapshot_slots, r.cache_pages),
        (4, 262_144 / 256 + 4 + cache.page_blocks, cache.snapshot_slots, cache.page_blocks)
    );
    assert_eq!(r.state.prefix_snapshots, cache.snapshot_slots as u64 * 64_389_120);
    assert_eq!(r.state.kv_target, r.kv_pages as u64 * 256 * 20_480);
    assert_eq!(cached.state.snapshots.as_ref().map(|b| b.bytes() as u64), Some(r.state.prefix_snapshots));
    let diffs = compare_with_plan_cached(&allocations(&cfg, None), &r.state, 4, r.kv_pages, 64, cache.snapshot_slots);
    assert!(diffs.is_empty(), "{diffs:?}");
    // the plan's own total: slots' state, pool, cache (snapshots and pages) and the engine reserve
    let planned = cached.plan.memory.planned_bytes(4);
    assert_eq!(
        planned,
        cached.plan.memory.engine_reserve_bytes
            + r.state.gdn_state
            + r.state.gdn_conv_history
            + r.state.kv_target
            + r.state.prefix_snapshots
    );
    let mut eng = cached.into_engine(&oc).unwrap_or_else(|e| panic!("the engine with the cache: {e}"));
    let st = eng.prefix_stats();
    assert!(st.enabled, "the engine runs the cache the plan carved out");
    assert_eq!((st.snapshot_slots, st.page_blocks), (cache.snapshot_slots as u64, cache.page_blocks as u64));
    // a prompt of more than 128 tokens twice: the second is a hit (the cache is the same code path as on a real pack)
    let prompt: Vec<u32> = (0..600).map(|i| 100 + (i as u32 * 7) % 5000).collect();
    let greedy = gb_sampling::Sampling { seed: 0, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 };
    let mk = |cache| gb_engine::Request {
        prompt: prompt.clone(),
        sampling: greedy,
        max_new: 4,
        stop: vec![],
        draft: false,
        images: vec![],
        cache,
    };
    let (a, b) = (eng.submit_deferred(mk(true)).unwrap(), eng.submit_deferred(mk(true)).unwrap());
    for _ in 0..4000 {
        if !eng.busy() {
            break;
        }
        eng.step().unwrap();
    }
    let c = eng.submit_deferred(mk(true)).unwrap();
    for _ in 0..4000 {
        if !eng.busy() {
            break;
        }
        eng.step().unwrap();
    }
    let outs = eng.take_outputs();
    assert_eq!(outs.len(), 3, "{:?}", (a, b, c));
    assert!(outs.iter().all(|o| o.tokens == outs[0].tokens), "a hit and a miss answer alike");
    assert!(outs[2].cached_tokens > 0, "the third identical prompt hit the entry the first left");
    assert!(eng.prefix_stats().hits >= 1);
    drop(eng);
    std::fs::remove_dir_all(&dir).unwrap();
}
