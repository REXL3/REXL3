//! Reading real tensors from the pack. Set REXL3_QWEN_DIR (tests skip without it).
use gb_model::pack::Pack;

#[test]
fn real_pack_tensors_have_the_expected_forms() {
    let Some(d) = std::env::var("REXL3_QWEN_DIR").ok().map(std::path::PathBuf::from).filter(|p| p.join("config.json").exists())
    else {
        return eprintln!("REXL3_QWEN_DIR not set: skipping");
    };
    let p = Pack::open(&d).unwrap();
    let l = "model.language_model.layers.0.";
    // plain tensors, with shape and dtype checks
    assert_eq!(p.bf16(&format!("{l}linear_attn.conv1d.weight"), &[10240, 1, 4]).unwrap().len(), 40960);
    assert_eq!(p.f16(&format!("{l}linear_attn.in_proj_a.weight"), &[48, 5120]).unwrap().len(), 48 * 5120);
    assert!(p.bf16(&format!("{l}linear_attn.in_proj_a.weight"), &[]).is_err(), "dtype mismatches are errors");
    assert!(p.bf16(&format!("{l}linear_attn.A_log"), &[47]).is_err(), "shape mismatches are errors");
    assert!(p.bf16("no.such.tensor", &[]).is_err());
    // norm weights: stored gamma-1, widened and shifted in fp32
    let raw = p.f32_from_bf16(&format!("{l}input_layernorm.weight"), &[5120], false).unwrap();
    let folded = p.f32_from_bf16(&format!("{l}input_layernorm.weight"), &[5120], true).unwrap();
    assert!(raw.iter().zip(&folded).all(|(a, b)| (b - a - 1.0).abs() < 1e-6));
    // gdn norm is plain gamma
    assert_eq!(p.f32_from_bf16(&format!("{l}linear_attn.norm.weight"), &[128], false).unwrap().len(), 128);
    // an EXL3 group
    let g = p.group(&format!("{l}mlp.gate_proj")).unwrap();
    assert_eq!((g.group.k, g.group.n, g.group.bits.twice()), (5120, 17408, 8));
    assert!(p.group(&format!("{l}mlp.nope")).is_err());
    assert!(p.has_group("lm_head"));
}

/// The packs on this machine are the validated pair, and a changed config would not be (REXL3_QWEN_DIR and REXL3_DRAFT_DIR).
#[test]
fn real_packs_match_the_validated_fingerprints() {
    let dir = |v: &str| std::env::var(v).ok().map(std::path::PathBuf::from).filter(|p| p.join("config.json").exists());
    let (Some(t), Some(d)) = (dir("REXL3_QWEN_DIR"), dir("REXL3_DRAFT_DIR")) else {
        return eprintln!("REXL3_QWEN_DIR / REXL3_DRAFT_DIR not set: skipping");
    };
    use gb_model::fingerprint::{fingerprint, validated};
    let mut tp = Pack::open(&t).unwrap();
    let dp = Pack::open(&d).unwrap();
    let (tf, df) = (fingerprint(&tp.ck), fingerprint(&dp.ck));
    assert!(validated(&tf, Some(&df)).is_some(), "target {tf} draft {df} are not in VALIDATED");
    assert!(validated(&df, Some(&tf)).is_none(), "target and draft are not interchangeable");
    tp.ck.config["text_config"]["num_hidden_layers"] = serde_json::json!(63);
    assert!(validated(&fingerprint(&tp.ck), Some(&df)).is_none(), "an edited config must not validate");
}

/// `--verify-pack` on the real packs: every shard matches the checksum pinned for the validated pack (a second or so, warm), the
/// shards `crc32.txt` does not mention are covered by the pins, and a changed pin is a corrupt-pack error.
#[test]
fn real_packs_verify_against_their_pins() {
    use gb_model::fingerprint::{fingerprint, pinned_shards};
    use gb_model::verify::{verify_pack, VerifyError};
    let dir = |v: &str| std::env::var(v).ok().map(std::path::PathBuf::from).filter(|p| p.join("config.json").exists());
    let (Some(t), Some(d)) = (dir("REXL3_QWEN_DIR"), dir("REXL3_DRAFT_DIR")) else {
        return eprintln!("REXL3_QWEN_DIR / REXL3_DRAFT_DIR not set: skipping");
    };
    for (path, shards) in [(t, 2usize), (d, 1)] {
        let pack = Pack::open(&path).unwrap();
        let pins: Vec<(&str, u32)> = pinned_shards(&fingerprint(&pack.ck)).iter().map(|&(n, _, c)| (n, c)).collect();
        assert_eq!(pins.len(), shards, "{}: pinned shards", path.display());
        let v = verify_pack(&path, &pins, &pack.shard_files()).unwrap();
        assert!(v.unverified.is_empty(), "{}: shards no checksum covers: {:?}", path.display(), v.unverified);
        assert!(v.files >= shards && v.bytes > 1 << 30, "{}: verified only {} files, {} bytes", path.display(), v.files, v.bytes);
        // one wrong digit in a pin is a corrupt pack
        let mut bad = pins.clone();
        bad[0].1 ^= 1;
        assert!(matches!(verify_pack(&path, &bad, &pack.shard_files()), Err(VerifyError::Corrupt(_))), "{}", path.display());
    }
}

/// Loads every text weight of the target onto the GPU (about 14 GiB, a minute or so): off by default.
/// `REXL3_QWEN_DIR=... cargo test --release -p gb-model --test real_pack -- --ignored weights_load`
#[test]
#[ignore]
fn weights_load_to_the_gpu() {
    use gb_model::config::QwenConfig;
    use gb_model::weights::{Mixer, Weights};
    let d = std::path::PathBuf::from(std::env::var("REXL3_QWEN_DIR").expect("set REXL3_QWEN_DIR"));
    if !gb_cuda::device_available() {
        return eprintln!("no CUDA device: skipping");
    }
    let c = QwenConfig::from_dir(&d).unwrap();
    let p = Pack::open(&d).unwrap();
    let t0 = std::time::Instant::now();
    let w = Weights::load(&p, &c, |i, n| {
        if i % 16 == 0 {
            eprintln!("layer {i}/{n}  {:.0}s", t0.elapsed().as_secs_f64());
        }
    })
    .unwrap();
    eprintln!("loaded {:.2} GiB in {:.1}s", w.bytes as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64());
    assert_eq!(w.layers.len(), 64);
    assert_eq!(w.layers.iter().filter(|l| matches!(l.mixer, Mixer::Full(_))).count(), 16);
    // 14.65 GiB of text weights per the spec (layers 11.39 + embedding 2.37 + head 0.89), plus small plain tensors
    let gib = w.bytes as f64 / (1u64 << 30) as f64;
    assert!((14.0..15.5).contains(&gib), "{gib} GiB");
}

#[test]
fn scratch_sizes_are_sane() {
    use gb_model::config::QwenConfig;
    use gb_model::scratch::Scratch;
    let Some(d) = std::env::var("REXL3_QWEN_DIR").ok().map(std::path::PathBuf::from).filter(|p| p.join("config.json").exists())
    else {
        return eprintln!("REXL3_QWEN_DIR not set: skipping");
    };
    if !gb_cuda::device_available() {
        return eprintln!("no CUDA device: skipping");
    }
    let c = QwenConfig::from_dir(&d).unwrap();
    let s = Scratch::new(&c, 4096).unwrap();
    eprintln!("scratch for 4096 rows: {:.2} GiB", s.bytes() as f64 / (1u64 << 30) as f64);
    assert!(s.bytes() < 8 << 30);
    let small = Scratch::new(&c, 128).unwrap();
    eprintln!("scratch for 128 rows: {:.2} GiB", small.bytes() as f64 / (1u64 << 30) as f64);
    assert!(small.bytes() < s.bytes() / 4, "the decode-sized scratch must be much smaller than the prefill-sized one");
}
