//! Gate 3: the whole target forward pass against ExLlamaV3's per-layer dumps (testdata/ref).
//!
//! ExLlamaV3 is not a bit-exact oracle (fp32 residual, fp16 sublayers; spec 3.3), so layers are compared by cosine and
//! relative RMS. Heavy (loads 14.65 GiB): needs REXL3_QWEN_DIR, a GPU and the dumps.
//!
//!     REXL3_QWEN_DIR=... cargo test --release -p gb-model --test gate3 -- --ignored --nocapture

mod refs;
use gb_model::config::QwenConfig;
use gb_model::forward::Model;
use gb_model::pack::Pack;
use gb_model::weights::Weights;
use refs::{cos_rel_rms, have_refs, ref_npy};

fn load_model(slots: usize, kv_blocks: usize, max_context: usize, prefill_rows: usize) -> Option<Model> {
    let d = std::path::PathBuf::from(std::env::var("REXL3_QWEN_DIR").ok()?);
    if !gb_cuda::device_available() || !have_refs() {
        return None;
    }
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let pack = Pack::open(&d).unwrap();
    let w = Weights::load(&pack, &cfg, |_, _| {}).unwrap();
    Some(Model::new(cfg, w, slots, kv_blocks, max_context, prefill_rows).unwrap())
}

fn bf16_to_f32(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()
}

#[test]
#[ignore]
fn prefill_matches_exllamav3_layer_by_layer() {
    let Some(mut m) = load_model(2, 64, 4096, 256) else {
        return eprintln!("needs REXL3_QWEN_DIR, a GPU and testdata/ref: skipping");
    };
    for prompt in ["short", "long"] {
        let ids: Vec<u32> = ref_npy(&format!("{prompt}/ids.npy")).to_i64().into_iter().map(|x| x as u32).collect();
        let t = ids.len();
        m.begin_slot(0).unwrap();
        m.capture = true;
        let cand = m.prefill_chunk(0, 0, &ids, ids.len(), Some(10), 0).unwrap().unwrap();
        m.capture = false;
        assert_eq!(m.captured.len(), 64);
        let mut worst = (1.0f64, 0.0f64, 0usize);
        eprintln!("--- {prompt} (T={t})");
        for i in 0..64 {
            let want = ref_npy(&format!("{prompt}/layer_out_{i:02}.npy")).to_f32();
            let got = bf16_to_f32(&m.captured[i]);
            assert_eq!(got.len(), want.len(), "layer {i}: shape");
            let (cos, rel) = cos_rel_rms(&got, &want);
            if i % 8 == 7 || i < 2 || cos < 0.999 {
                eprintln!("layer {i:2}: cos {cos:.6}  relRMS {rel:.4}");
            }
            if cos < worst.0 {
                worst = (cos, rel, i);
            }
        }
        eprintln!("worst layer {}: cos {:.6} relRMS {:.4}", worst.2, worst.0, worst.1);
        assert!(worst.0 >= 0.999, "layer {} cosine {} is below 0.999", worst.2, worst.0);
        // final logits of the last prompt row: EX's top-1 must be among our top-4, and the top-10 should mostly agree
        let logits = ref_npy(&format!("{prompt}/logits.npy"));
        let v = m.cfg.vocab;
        let all = logits.to_f32();
        let last = &all[(t - 1) * v..t * v];
        let mut order: Vec<usize> = (0..v).collect();
        order.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap().then(a.cmp(&b)));
        let ours: Vec<i32> = cand.ids[..10].to_vec();
        eprintln!("EX top-5 {:?}   ours {:?}", &order[..5], &ours[..5]);
        assert!(ours[..4].contains(&(order[0] as i32)), "EX's top-1 {} is not in our top-4 {:?}", order[0], &ours[..4]);
        let overlap = order[..10].iter().filter(|&&x| ours.contains(&(x as i32))).count();
        assert!(overlap >= 7, "top-10 overlap {overlap}/10");
    }
}
