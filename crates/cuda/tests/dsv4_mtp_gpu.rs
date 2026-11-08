//! The GPU DSpark drafter (`src/dsv4_mtp_gpu.rs`, `csrc/dsv4_mtp_gpu.cu`, `csrc/fp4_linear.cu`) against the host
//! reference in `src/dsv4_mtp.rs`.
//!
//! # What is compared, and against what
//!
//! Every test builds one set of random weights and then binds it **twice**: packed MXFP4 / FP8 on the device, and the
//! *dequantised* `f32` values in [`MtpLayerHost`]. So the comparison is not "the GPU against a re-derivation of the
//! kernel" -- it is the device arithmetic against the host reference over the same numbers, and the tolerance covers only
//! the activation dtype (`f16` on the device, `f32` on the host) and the fp32 accumulation order.
//!
//! The host reference is itself pinned against a direct f64 transcription of `model.py` in `tests/dsv4_mtp.rs`; this file
//! does not repeat that, it inherits it.
//!
//! # What runs without a GPU
//!
//! Every test starts with `dev()`, which prints `not run: no device` and returns on a CPU-only box, so the binary is green
//! either way. `v41_chain_timing` is `#[ignore]`d: it allocates ~9 GB of device weight.
use gb_cuda::dsv4_attn::Fp8Host;
use gb_cuda::dsv4_moe::{hc_mixes_host, HcCfg};
use gb_cuda::dsv4_mtp::{
    accept_prefix, get_dspark_topk_row, ConfidenceHeadHost, DsparkChain, MarkovHeadHost, MtpConfig, MtpLayerHost,
};
use gb_cuda::dsv4_mtp_gpu::{
    hc_mixes_dev, hc_post_dev, hc_pre_dev, schedule_verify_len_dev, set_round_graphs_enabled, DsparkGpu, MtpStage, MtpStagePack,
    MtpWalkSampling, RoutedExperts,
};
use gb_cuda::exl3_linear::{Dtype, Exl3Dims, Exl3Weights};
use gb_cuda::fp4_linear::{dequant_host, pack_f32, Mxfp4Weights};
use gb_cuda::fp8_linear::pack_bf16;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::Codebook;
use gb_sampling::{choose, top_candidates, Sampling};

// ------------------------------------------------------------------------------------------------ fixtures

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * u
    }
    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform(lo, hi)).collect()
    }
}

/// A weight in both forms: what the device reads, and what the host reference is given.
struct W {
    n: usize,
    k: usize,
    bytes: Vec<u8>,
    s: Vec<u8>,
    deq: Vec<f32>,
}

impl W {
    fn t(&self) -> (usize, usize, &[u8], &[u8]) {
        (self.n, self.k, &self.bytes, &self.s)
    }
}

/// f32 -> bf16 -> the pack's fp8 32x32 block layout -> back to f32.
fn fp8_w(n: usize, k: usize, r: &mut Rng, lo: f32, hi: f32) -> W {
    let f = r.vec(n * k, lo, hi);
    let raw: Vec<u8> = f.iter().flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
    let (bytes, s) = pack_bf16(n, k, &raw).expect("pack");
    let deq = Fp8Host::new(n, k, &bytes, &s).w;
    W { n, k, bytes, s, deq }
}

/// f32 -> MXFP4 (E2M1 + E8M0 per 32) -> back to f32.
fn fp4_w(n: usize, k: usize, r: &mut Rng, lo: f32, hi: f32) -> W {
    let f = r.vec(n * k, lo, hi);
    let (bytes, s) = pack_f32(n, k, &f);
    let deq = dequant_host(n, k, &bytes, &s);
    W { n, k, bytes, s, deq }
}

/// `MtpConfig::toy`, widened so every matrix is a multiple of 32 for the fp8 tile **and** of 128 for the EXL3 gemv
/// (which is what the pack's MTP experts are now: K2, 2.00 bpw).
fn toy() -> MtpConfig {
    let mut c = MtpConfig::toy();
    c.vocab = 128;
    c.markov_rank = 128;
    c.inter_dim = 128;
    c.o_lora = 128;
    c
}

/// A random EXL3 K2 weight: `[k, n]` trellis + `suh` / `svh`, plus the dequantised `[n, k]` f32 form the host
/// reference reads. Every 2-bit field of a random trellis is a valid codebook index, so random bytes are a valid weight.
fn exl3_w(k: usize, n: usize, r: &mut Rng, st: &Stream) -> (Exl3Weights, Vec<f32>) {
    let bits = 2usize;
    let tb = (k / 16) * (n / 16) * 16 * bits * 2;
    let trellis: Vec<u16> = (0..tb / 2).map(|_| r.next_u64() as u16).collect();
    let suh: Vec<u16> = (0..k).map(|_| f64_to_f16(r.uniform(0.2, 0.3) as f64)).collect();
    let svh: Vec<u16> = (0..n).map(|_| f64_to_f16(r.uniform(0.2, 0.3) as f64)).collect();
    let w = Exl3Weights::from_dims(
        Exl3Dims { k, n, bits, codebook: Codebook::Mul1 },
        DevBuf::from_slice(&trellis).expect("trellis"),
        DevBuf::from_slice(&suh).expect("suh"),
        DevBuf::from_slice(&svh).expect("svh"),
        None,
    )
    .expect("exl3 weight");
    let mut deq = vec![0f32; n * k];
    let mut e_in = vec![0u16; k];
    let mut d_in = DevBuf::zeroed(k * 2).expect("d_in");
    let mut d_out = DevBuf::zeroed(n * 2).expect("d_out");
    let mut d_ws = DevBuf::zeroed(gb_cuda::exl3_linear::linear_workspace_bytes(1, k, n, 2)).expect("d_ws");
    for i in 0..k {
        if i > 0 {
            e_in[i - 1] = 0;
        }
        e_in[i] = f64_to_f16(1.0);
        d_in.upload(&e_in).expect("h2d");
        settle();
        w.forward(&d_in, &mut d_out, 1, Dtype::F16, Dtype::F16, &mut d_ws, st).expect("forward");
        st.sync().expect("sync");
        let row = d_out.to_vec::<u16>().expect("d2h");
        for j in 0..n {
            deq[j * k + i] = f16_to_f64(row[j]) as f32;
        }
    }
    (w, deq)
}

/// A whole three-stage fixture: the host stages over the dequantised weights, the GPU chain over the packed bytes.
struct Fx {
    host: Vec<MtpLayerHost>,
    gpu: DsparkGpu,
    tapw: usize,
}

/// `rows` random taps as (f32 for the host, f16 bits for the device).
fn taps(cfg: &MtpConfig, rows: usize, r: &mut Rng) -> (Vec<f32>, DevBuf) {
    let tapw = cfg.tap_width();
    let f = r.vec(rows * tapw, -0.4, 0.4);
    let bits: Vec<u16> = f.iter().map(|&v| f64_to_f16(v as f64)).collect();
    let host: Vec<f32> = bits.iter().map(|&b| f16_to_f64(b) as f32).collect();
    // a block view, so `prefill_seed` can chunk it
    let mut buf = DevBuf::zeroed_view(bits.len() * 2).expect("taps");
    buf.upload(&bits).expect("upload taps");
    settle();
    (host, buf)
}

fn fixture(cfg: &MtpConfig, seed: u64, st: &Stream) -> Fx {
    fixture_mode(cfg, seed, st, false)
}

/// `fixture` with the Markov tables handed to the device as pre-packed bf16 zeros (the shipped pack's dtype),
/// so a draft round rides the fast BF16 keyed walk instead of the argmax fallback loop; the host reference also
/// sees the zero tables, so the walks agree exactly.
fn fixture_bf16_markov(cfg: &MtpConfig, seed: u64, st: &Stream) -> Fx {
    fixture_mode(cfg, seed, st, true)
}

fn fixture_mode(cfg: &MtpConfig, seed: u64, st: &Stream, bf16_markov: bool) -> Fx {
    let mut r = Rng(seed);
    let d = cfg.dim;
    let tapw = cfg.tap_width();
    let embed_f16: Vec<u16> = (0..cfg.vocab * d).map(|_| f64_to_f16(r.uniform(-0.5, 0.5) as f64)).collect();
    let embed_f32: Vec<f32> = embed_f16.iter().map(|&h| f16_to_f64(h) as f32).collect();
    let lm = fp8_w(cfg.vocab, d, &mut r, -0.3, 0.3);
    let attn_norm = r.vec(d, 0.8, 1.2);
    let ffn_norm = r.vec(d, 0.8, 1.2);

    let mut hosts = Vec::new();
    let mut stages = Vec::new();
    let mut host_experts: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
    for stage in 0..cfg.n_stages {
        host_experts.clear();
        let wq_a = fp8_w(cfg.q_lora, d, &mut r, -0.2, 0.2);
        let wq_b = fp8_w(cfg.n_heads * cfg.head_dim, cfg.q_lora, &mut r, -0.2, 0.2);
        let wkv = fp8_w(cfg.head_dim, d, &mut r, -0.2, 0.2);
        let wo_a = fp8_w(cfg.o_groups * cfg.o_lora, cfg.o_in_per_group(), &mut r, -0.2, 0.2);
        let wo_b = fp8_w(d, cfg.o_groups * cfg.o_lora, &mut r, -0.2, 0.2);
        let q_norm = r.vec(cfg.q_lora, 0.8, 1.2);
        let kv_norm = r.vec(cfg.head_dim, 0.8, 1.2);
        let sink = r.vec(cfg.n_heads, -0.5, 0.5);
        let gate = r.vec(cfg.n_routed_experts * d, -0.3, 0.3);
        let gate_bias = r.vec(cfg.n_routed_experts, -0.1, 0.1);
        let mut experts = Vec::with_capacity(cfg.n_routed_experts);
        for _ in 0..cfg.n_routed_experts {
            let (w1, d1) = exl3_w(d, cfg.inter_dim, &mut r, st);
            let (w3, d3) = exl3_w(d, cfg.inter_dim, &mut r, st);
            let (w2, d2) = exl3_w(cfg.inter_dim, d, &mut r, st);
            experts.push([w1, w2, w3]);
            host_experts.push((d1, d2, d3));
        }
        let sw1 = fp8_w(cfg.inter_dim, d, &mut r, -0.3, 0.3);
        let sw3 = fp8_w(cfg.inter_dim, d, &mut r, -0.3, 0.3);
        let sw2 = fp8_w(d, cfg.inter_dim, &mut r, -0.3, 0.3);
        // non-zero mixer weights: a zero mixer makes the four streams identical and hides a collapse bug
        let hca: Vec<f32> = r.vec(24 * 4 * d, -0.05, 0.05).iter().map(|v| v + 0.01).collect();
        let hcb = r.vec(24, -0.1, 0.1);
        let hcs = [0.7f32, 0.9, 1.1];
        let hfa: Vec<f32> = r.vec(24 * 4 * d, -0.05, 0.05).iter().map(|v| v + 0.01).collect();
        let hfb = r.vec(24, -0.1, 0.1);
        let hfs = [1.1f32, 0.8, 1.05];

        let gate_bf16: Vec<u16> = gate.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();
        let gate_deq: Vec<f32> = gate_bf16.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();

        // --- the host stage, over the dequantised weights
        let mut h = MtpLayerHost::zeros(cfg, stage, cfg.window + cfg.block_size + 8);
        h.wq_a = wq_a.deq.clone();
        h.wq_b = wq_b.deq.clone();
        h.wkv = wkv.deq.clone();
        h.wo_a = wo_a.deq.clone();
        h.wo_b = wo_b.deq.clone();
        h.q_norm = q_norm.clone();
        h.kv_norm = kv_norm.clone();
        h.sink = sink.clone();
        h.gate = gate_deq;
        h.gate_bias = gate_bias.clone();
        // [experts * inter, dim] and [experts * dim, inter], concatenated over the experts
        h.w1 = host_experts.iter().flat_map(|e| e.0.iter().copied()).collect();
        h.w3 = host_experts.iter().flat_map(|e| e.2.iter().copied()).collect();
        h.w2 = host_experts.iter().flat_map(|e| e.1.iter().copied()).collect();
        h.shared_w1 = sw1.deq.clone();
        h.shared_w3 = sw3.deq.clone();
        h.shared_w2 = sw2.deq.clone();
        h.hc_attn_fn = hca.clone();
        h.hc_attn_base = hcb.clone();
        h.hc_attn_scale = hcs;
        h.hc_ffn_fn = hfa.clone();
        h.hc_ffn_base = hfb.clone();
        h.hc_ffn_scale = hfs;
        let mut h = h.with_trunk_tables(embed_f32.clone(), lm.deq.clone(), attn_norm.clone(), ffn_norm.clone());

        // --- the GPU stage, over the same numbers
        let mut main_proj = None;
        let mut main_norm = None;
        let mut norm = None;
        let mut markov_embed = None;
        let mut markov_head = None;
        let mut markov_embed_bf16 = None;
        let mut markov_head_bf16 = None;
        let mut conf_proj = None;
        let mut lm_head = None;
        if stage == 0 {
            let mp = fp8_w(d, tapw, &mut r, -0.2, 0.2);
            h.main_proj = Some(mp.deq.clone());
            let mn = r.vec(d, 0.8, 1.2);
            h.main_norm = Some(mn.clone());
            main_proj = Some(mp);
            main_norm = Some(mn);
        }
        if stage + 1 == cfg.n_stages {
            let nm = r.vec(d, 0.8, 1.2);
            h.norm = Some(nm.clone());
            let mut me = r.vec(cfg.vocab * cfg.markov_rank, -0.3, 0.3);
            let mut mh = r.vec(cfg.vocab * cfg.markov_rank, -0.3, 0.3);
            let cp = r.vec(cfg.conf_in(), -0.1, 0.1);
            if bf16_markov {
                // zero tables: the bias is zero, so the head logits ARE the walked logits and the host
                // oracle reproduces the walk exactly (no f16 rounding skew between the two sides)
                me.iter_mut().for_each(|v| *v = 0.0);
                mh.iter_mut().for_each(|v| *v = 0.0);
            }
            h.markov = Some(MarkovHeadHost::new(cfg, me.clone(), mh.clone()));
            h.confidence_head = Some(ConfidenceHeadHost::new(cfg, cp.clone(), 0.0));
            norm = Some(nm);
            let (me_bf16, mh_bf16) = if bf16_markov {
                (
                    Some(me.iter().map(|&v| f64_to_f16(v as f64)).collect::<Vec<u16>>()),
                    Some(mh.iter().map(|&v| f64_to_f16(v as f64)).collect::<Vec<u16>>()),
                )
            } else {
                (None, None)
            };
            markov_embed = Some(me);
            markov_head = Some(mh);
            markov_embed_bf16 = me_bf16;
            markov_head_bf16 = mh_bf16;
            conf_proj = Some(cp);
            lm_head = Some(lm.t());
        }
        hosts.push(h);
        stages.push(
            MtpStage::new(
                cfg,
                MtpStagePack {
                    stage,
                    seq: cfg.window + cfg.block_size + 8,
                    wq_a: wq_a.t(),
                    wq_b: wq_b.t(),
                    wkv: wkv.t(),
                    wo_a: wo_a.t(),
                    wo_b: wo_b.t(),
                    q_norm: &q_norm,
                    kv_norm: &kv_norm,
                    sink: &sink,
                    gate_bf16: &gate_bf16,
                    gate_bias: &gate_bias,
                    routed: RoutedExperts::Exl3(experts),
                    shared_w1: sw1.t(),
                    shared_w3: sw3.t(),
                    shared_w2: sw2.t(),
                    hc_attn: (&hca, &hcb, &hcs),
                    hc_ffn: (&hfa, &hfb, &hfs),
                    attn_norm: &attn_norm,
                    ffn_norm: &ffn_norm,
                    main_proj: main_proj.as_ref().map(|w| w.t()),
                    main_norm: main_norm.as_deref(),
                    norm: norm.as_deref(),
                    markov_embed: markov_embed.as_deref(),
                    markov_head: markov_head.as_deref(),
                    markov_embed_bf16: markov_embed_bf16.as_deref(),
                    markov_head_bf16: markov_head_bf16.as_deref(),
                    conf_proj: conf_proj.as_deref(),
                    lm_head,
                },
            )
            .expect("stage"),
        );
    }
    let gpu = DsparkGpu::new(cfg.clone(), stages, &embed_f16).expect("chain");
    settle();
    Fx { host: hosts, gpu, tapw }
}

/// Process-wide device lock (see `dev()`): static, so every call takes the *same* lock. Poison is recovered —
/// a panicked test must not wedge the remaining tests of the binary.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A non-blocking stream plus the process-wide device lock guard. `Deref<Target = Stream>` keeps the
/// `let Some(st) = dev() else { return };` call sites unchanged: `&st` coerces to `&Stream` and `st.sync()`
/// finds the stream methods through deref.
struct DevStream {
    stream: Stream,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl std::ops::Deref for DevStream {
    type Target = Stream;
    fn deref(&self) -> &Stream {
        &self.stream
    }
}

fn dev() -> Option<DevStream> {
    if !device_available() {
        println!("not run: no device");
        return None;
    }
    // Serialize the device tests of this binary: CUDA-graph captures and eager kernels on non-blocking streams in
    // the same process are not safe to overlap (a capture can be invalidated by another thread's synchronous
    // copies, and inference kernels from concurrent tests are free to race the device's own scheduling), so each
    // test holds one process-wide device lock for its whole body. The lock is taken here, in `dev()`, which every
    // device test calls first.
    let lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Non-blocking: a CUDA-graph capture on a legacy stream is invalidated by other threads' synchronous
    // copies on the legacy default stream (the parallel tests here do exactly that via `DevBuf::upload`).
    Stream::new_nonblocking().ok().map(|stream| DevStream { stream, _lock: lock })
}

/// `max|a - b| / max|b|`: the error relative to the tensor's own scale. A per-element relative error would be dominated
/// by the elements that cancel to ~0, which says nothing about the arithmetic.
fn scaled_err(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "lengths");
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6) as f64;
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs())) / scale
}

fn f16s(v: &DevBuf) -> Vec<f32> {
    v.to_vec::<u16>().expect("d2h").iter().map(|&h| f16_to_f64(h) as f32).collect()
}

/// Waits for every write queued on the legacy default stream: `DevBuf::from_slice` / `upload` (a pageable `cudaMemcpy`,
/// which can return before its DMA lands) and `DevBuf::zeroed` / `zeroed_view` (an asynchronous `cudaMemset`). `dev()`'s
/// stream is non-blocking, so nothing orders those writes before its kernels: without this a kernel can read a stale
/// input, or a late memset can zero its scratch or output. `st.sync()` does not help, it waits only for `st`. Called
/// after each batch of host writes, before the first launch that reads them.
fn settle() {
    gb_cuda::rt::device_sync().expect("device sync");
}

// ------------------------------------------------------------------------------------------------ tests

/// One EXL3 K2 matrix: `forward` against an f64 matmul over the `reconstruct`ed weight. This is the fixture the routed
/// block is built on, so it pins the fixture itself rather than the drafter.
#[test]
fn one_exl3_k2_matrix_runs() {
    let Some(st) = dev() else { return };
    let mut r = Rng(3);
    let (k, n) = (128usize, 128usize);
    let (w, deq) = exl3_w(k, n, &mut r, &st);
    let x: Vec<f32> = r.vec(k, -1.0, 1.0);
    let xb: Vec<u16> = x.iter().map(|&v| f64_to_f16(v as f64)).collect();
    let xv = DevBuf::from_slice(&xb).expect("x");
    let mut y = DevBuf::zeroed(n * 2).expect("y");
    let mut ws = DevBuf::zeroed(w.workspace_bytes(1).max(4 << 20)).expect("ws");
    settle();
    w.forward(&xv, &mut y, 1, Dtype::F16, Dtype::F16, &mut ws, &st).expect("gemv");
    st.sync().expect("sync");
    let got = f16s(&y);
    let mut want = vec![0f32; n];
    for c in 0..n {
        let mut acc = 0f64;
        for i in 0..k {
            acc += x[i] as f64 * deq[c * k + i] as f64;
        }
        want[c] = acc as f32;
    }
    println!("exl3 got[..4] {:?}", &got[..4]);
    println!("exl3 want[..4] {:?}", &want[..4]);
    assert!(want.iter().fold(0f32, |m, v| m.max(v.abs())) > 1e-3, "the fixture weight is not zero");
    let e = scaled_err(&got, &want);
    assert!(e < 5e-3, "exl3 k2 gemv: err/scale {e:.3e}");
}

/// The MXFP4 GEMV at every decode row count, against an f64 matmul over the dequantised weight.
#[test]
fn the_mxfp4_gemm_matches_the_dequantised_weight() {
    let Some(st) = dev() else { return };
    let mut r = Rng(7);
    let (n, k) = (64usize, 128usize);
    let w = fp4_w(n, k, &mut r, -0.5, 0.5);
    let mw = Mxfp4Weights::new(n, k, &w.bytes, &w.s).expect("weights");
    for m in 1..=8usize {
        let x: Vec<f32> = r.vec(m * k, -1.0, 1.0);
        let xb: Vec<u16> = x.iter().map(|&v| f64_to_f16(v as f64)).collect();
        let xv = DevBuf::from_slice(&xb).expect("x");
        let mut y = DevBuf::zeroed(m * n * 2).expect("y");
        settle();
        mw.forward(&xv, &mut y, m, Dtype::F16, &st).expect("gemv");
        st.sync().expect("sync");
        let got = f16s(&y);
        let mut want = vec![0f32; m * n];
        for row in 0..m {
            for c in 0..n {
                let mut acc = 0f64;
                for i in 0..k {
                    acc += x[row * k + i] as f64 * w.deq[c * k + i] as f64;
                }
                want[row * n + c] = acc as f32;
            }
        }
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())) as f64;
        let e = got.iter().zip(&want).fold(0f64, |a, (x, y)| a.max((*x as f64 - *y as f64).abs())) / scale.max(1e-6);
        assert!(e < 1e-3, "m = {m}: max err / max|y| = {e:.3e}");
    }
}

/// `forward_embed`: `main_x` from the taps, and the block embedding `[t, NOISE, ..]` over `hc_mult` copies.
#[test]
fn the_block_embedding_is_the_anchor_plus_noise() {
    let Some(st) = dev() else { return };
    let cfg = toy();
    let mut fx = fixture(&cfg, 11, &st);
    let (taps_f32, taps_dev) = taps(&cfg, 1, &mut Rng(3));
    let anchor = 5i32;
    let (h_want, main_want) = fx.host[0].forward_embed(&taps_f32, 1, &[anchor]);
    fx.gpu.forward_embed(&taps_dev, anchor, &st).expect("embed");
    st.sync().expect("sync");
    let e = scaled_err(&f16s(&fx.gpu.scratch.main_x)[..cfg.dim], &main_want);
    assert!(e < 3e-3, "main_x: err/scale {e:.3e}");
    let got = fx.gpu.scratch.h.to_vec::<f32>().expect("h");
    let e = scaled_err(&got, &h_want);
    assert!(e < 3e-3, "h: err/scale {e:.3e}");
    // slot 1 is the noise token's embedding, and it differs from the anchor's
    for c in 0..4 {
        let a = &got[c * cfg.dim..][..cfg.dim];
        let b = &got[(4 + c) * cfg.dim..][..cfg.dim];
        assert_ne!(a, b, "stream {c}: slot 1 must be the noise embedding");
    }
}

/// The hyper-connection mixer at a non-5120 width: `pre` / `post` / `comb` and the collapse.
#[test]
fn the_hyper_connection_mixer_matches_the_host() {
    let Some(st) = dev() else { return };
    let cfg = toy();
    let mut fx = fixture(&cfg, 23, &st);
    let rows = cfg.block_size;
    let h: Vec<f32> = Rng(5).vec(rows * 4 * cfg.dim, -0.6, 0.6);
    fx.gpu.scratch.h.upload(&h).expect("h");
    let (mut pre, mut post, mut comb) =
        (DevBuf::zeroed(rows * 4 * 4).unwrap(), DevBuf::zeroed(rows * 4 * 4).unwrap(), DevBuf::zeroed(rows * 16 * 4).unwrap());
    settle();
    hc_mixes_dev(&fx.gpu.scratch.h, &fx.gpu.stages[0].hc_attn, rows, cfg.dim, HcCfg::V41, &mut pre, &mut post, &mut comb, &st)
        .expect("mixes");
    st.sync().expect("sync");
    let (pre_w, post_w, comb_w) = hc_mixes_host(
        &h,
        rows,
        cfg.dim,
        &fx.host[0].hc_attn_fn,
        &fx.host[0].hc_attn_scale,
        &fx.host[0].hc_attn_base,
        4 * cfg.dim,
        HcCfg::V41,
    );
    let e = scaled_err(&pre.to_vec::<f32>().unwrap(), &pre_w);
    assert!(e < 1e-4, "pre: err/scale {e:.3e}");
    let e = scaled_err(&post.to_vec::<f32>().unwrap(), &post_w);
    assert!(e < 1e-4, "post: err/scale {e:.3e}");
    let e = scaled_err(&comb.to_vec::<f32>().unwrap(), &comb_w);
    assert!(e < 1e-3, "comb: err/scale {e:.3e}");

    // and the collapse
    fx.gpu.scratch.pre_mix.upload(&gb_cuda::dsv4_mtp::DsparkChain::identity_pre_mix(&cfg, rows)).unwrap();
    settle();
    hc_pre_dev(&fx.gpu.scratch.h, &fx.gpu.scratch.pre_mix, &mut fx.gpu.scratch.y, rows, cfg.dim, &st).expect("pre");
    st.sync().expect("sync");
    let want = gb_cuda::dsv4_moe::hc_pre_host(&h, rows, cfg.dim, &gb_cuda::dsv4_mtp::DsparkChain::identity_pre_mix(&cfg, rows));
    let e = scaled_err(&f16s(&fx.gpu.scratch.y), &want);
    assert!(e < 3e-3, "collapse: err/scale {e:.3e}");

    // and the residual mix
    let y = fx.gpu.scratch.y.to_vec::<u16>().unwrap();
    let yf: Vec<f32> = y.iter().map(|&b| f16_to_f64(b) as f32).collect();
    hc_post_dev(&fx.gpu.scratch.h, &fx.gpu.scratch.y, &post, &comb, rows, cfg.dim, &st).expect("post");
    st.sync().expect("sync");
    let want = gb_cuda::dsv4_moe::hc_post_host(&h, &yf, rows, cfg.dim, &post_w, &comb_w);
    let e = scaled_err(&fx.gpu.scratch.h.to_vec::<f32>().unwrap(), &want);
    assert!(e < 3e-3, "residual: err/scale {e:.3e}");
}

/// The routed block: router, top-3 MXFP4 experts, shared expert, scatter.
#[test]
fn the_routed_experts_match_the_host() {
    let Some(st) = dev() else { return };
    let cfg = toy();
    let mut fx = fixture(&cfg, 31, &st);
    let rows = cfg.block_size;
    let x: Vec<f32> = Rng(9).vec(rows * cfg.dim, -0.8, 0.8);
    let bits: Vec<u16> = x.iter().map(|&v| f64_to_f16(v as f64)).collect();
    fx.gpu.scratch.xn.upload(&bits).expect("xn");
    settle();
    let want = fx.host[0].moe(&x, rows);
    fx.gpu.moe(0, rows, &st).expect("moe");
    st.sync().expect("sync");
    let e = scaled_err(&f16s(&fx.gpu.scratch.y), &want);
    println!("moe err/scale {e:.3e}");
    assert!(e < 2e-2, "moe: err/scale {e:.3e}");
    // the fp32 copy the mHC post consumes agrees with the f16 one
    let f32v = fx.gpu.scratch.moe32.to_vec::<f32>().expect("moe32");
    for (a, b) in f32v.iter().zip(f16s(&fx.gpu.scratch.y)) {
        assert!((a - b).abs() <= 2e-3 * a.abs().max(1.0), "{a} vs {b}");
    }
}

/// One full stage (attention + MoE between hyper-connections) at `start_pos > 0`, with a part-filled window.
#[test]
fn one_stage_matches_the_host() {
    let Some(st) = dev() else { return };
    let cfg = toy();
    let mut fx = fixture(&cfg, 41, &st);
    let mut r = Rng(13);
    let rows = cfg.block_size;
    let start = 3usize;
    let (pf, pfbuf) = taps(&cfg, start, &mut r);
    fx.gpu.prefill_seed(&pfbuf, 0, start, &st).expect("seed");
    let (tap1, tap1buf) = taps(&cfg, 1, &mut r);
    fx.gpu.forward_embed(&tap1buf, 2, &st).expect("embed");
    st.sync().expect("sync");

    let mut host = fx.host[0].clone();
    // one prefill call over the whole chunk: `seed_cache` rotates only when it is handed every row at once
    let mut mx_all = Vec::new();
    for p in 0..start {
        let (_, mx) = host.forward_embed(&pf[p * fx.tapw..][..fx.tapw], 1, &[0]);
        mx_all.extend_from_slice(&mx);
    }
    host.seed_cache(&mx_all, start, 0);
    let (h0, mx1) = fx.host[0].forward_embed(&tap1, 1, &[2]);
    fx.gpu.scratch.h.upload(&h0).expect("h");
    settle();
    let pre_mix = DsparkChain::identity_pre_mix(&cfg, rows);
    let (x2, _) = host.forward(&h0, start, &pre_mix, &mx1);
    let n_kv = cfg.window.min(start + 1) + cfg.block_size;
    fx.gpu.stage_forward(0, start, n_kv, &st).expect("stage");
    st.sync().expect("sync");
    let e = scaled_err(&fx.gpu.scratch.h.to_vec::<f32>().expect("h"), &x2);
    println!("stage err/scale {e:.3e}");
    assert!(e < 4e-2, "stage: err/scale {e:.3e}");
}

/// The whole chain: ids, confidence and logits against `DsparkChain::step`.
#[test]
fn the_full_chain_matches_the_host() {
    let Some(st) = dev() else { return };
    let cfg = toy();
    let mut fx = fixture(&cfg, 53, &st);
    let mut r = Rng(17);
    let start = 6usize;
    let (pf, pfbuf) = taps(&cfg, start, &mut r);
    fx.gpu.prefill_seed(&pfbuf, 0, start, &st).expect("seed");
    let (tap, tapbuf) = taps(&cfg, 1, &mut r);
    let anchor = 9i32;

    let mut chain = DsparkChain::new(cfg.clone(), fx.host.clone());
    chain.prefill_at(&pf, 0, start);
    let want = chain.step(&tap, &[anchor]);

    fx.gpu.draft(&tapbuf, anchor, start, &st).expect("draft");
    st.sync().expect("sync");
    let (ids, conf) = fx.gpu.read().expect("read");
    assert_eq!(ids[0], anchor, "slot 0 is the anchor");
    assert_eq!(ids.len(), cfg.block_size + 1);
    let e = scaled_err(&conf, &want.confidence);
    println!(
        "confidence err/scale {e:.3e}, logits err/scale {:.3e}",
        scaled_err(&fx.gpu.scratch.logits.to_vec::<f32>().unwrap(), &want.logits)
    );
    assert!(e < 6e-2, "confidence: err/scale {e:.3e}");
    let logits = fx.gpu.scratch.logits.to_vec::<f32>().expect("logits");
    let e = scaled_err(&logits, &want.logits);
    assert!(e < 5e-2, "logits: err/scale {e:.3e}");

    // the acceptance rule, on both sides
    let thr = 0.5f32;
    assert_eq!(fx.gpu.accept(thr, &st).expect("accept"), accept_prefix(&want.confidence, thr));
    assert_eq!(DsparkGpu::accept_host(&conf, thr), accept_prefix(&conf, thr));
}

/// The SPEC_TREE lattice capture (`draft_n_lattice`, T2.5a): the unary scores are the head's own PRE-BIAS logits
/// (recovered on the host by subtracting the markov bias of the walk's own parent from the chain's biased logits),
/// every level's candidate set carries the walk's pick, the edge terms are the markov head against the parent
/// candidates' embeddings, and `unary + edge` at the picked parent is the walk's own argmax -- the picked candidate
/// beats every other captured candidate of its level.
#[test]
fn the_lattice_capture_carries_the_walk_and_the_edges() {
    let Some(st) = dev() else { return };
    let cfg = toy();
    let mut fx = fixture(&cfg, 71, &st);
    let mut r = Rng(23);
    let start = 6usize;
    let (pf, pfbuf) = taps(&cfg, start, &mut r);
    fx.gpu.prefill_seed(&pfbuf, 0, start, &st).expect("seed");
    let (tap, tapbuf) = taps(&cfg, 1, &mut r);
    let anchor = 9i32;

    let mut chain = DsparkChain::new(cfg.clone(), fx.host.clone());
    chain.prefill_at(&pf, 0, start);
    let want = chain.step(&tap, &[anchor]);

    let lat = fx
        .gpu
        .draft_n_lattice(&tapbuf, anchor, start, cfg.block_size, 8, &st)
        .expect("draft")
        .expect("the toy stack carries markov tables");
    st.sync().expect("sync");

    let (ids, _conf) = fx.gpu.read().expect("read");
    let rows = lat.rows;
    assert_eq!(rows, cfg.block_size);
    assert!(lat.k >= 2 && lat.k <= 8);
    // the picks are the walk the plain round drafts
    assert_eq!(lat.picks, ids[1..=rows].iter().map(|&x| x as u32).collect::<Vec<u32>>());
    assert_eq!(lat.picks, want.drafted().iter().map(|&x| x as u32).collect::<Vec<u32>>());

    // the host markov head, for the bias and the unary recovery
    let mk = fx.host[cfg.n_stages - 1].markov.as_ref().expect("the last stage owns the markov head");
    let bias_of = |tok: i32| mk.logits(&mk.embed_tokens(&[tok]), 1);
    for d in 0..rows {
        let cand: Vec<u32> = lat.ids[d * lat.k..(d + 1) * lat.k].to_vec();
        assert_eq!(cand[0], lat.picks[d], "the pick rides its level's head");
        assert!(cand.iter().filter(|&&t| t != u32::MAX).collect::<Vec<_>>().len() >= 2, "no real candidates captured");
        // the host's pre-bias logits of this level: the chain's biased row minus the walk's own parent's bias
        let parent = ids[d];
        let bias = bias_of(parent);
        for (c, &t) in cand.iter().enumerate() {
            if t == u32::MAX {
                assert!(lat.unary[d * lat.k + c] == f32::NEG_INFINITY, "padding must be unreachable");
                continue;
            }
            let host_unary = want.logits[d * cfg.vocab + t as usize] - bias[t as usize];
            let e = (lat.unary[d * lat.k + c] - host_unary).abs() / host_unary.abs().max(1.0).max(lat.unary[d * lat.k + c].abs());
            assert!(e < 5e-2, "level {d} candidate {t}: unary {} vs host {host_unary} (rel {e:.3e})", lat.unary[d * lat.k + c]);
            // the edge of this candidate against every captured parent of the previous level (level 0: the anchor)
            let parents: Vec<u32> = if d == 0 { vec![ids[0] as u32] } else { lat.ids[(d - 1) * lat.k..d * lat.k].to_vec() };
            for (pi, &pt) in parents.iter().enumerate() {
                if pt == u32::MAX {
                    continue;
                }
                let host_edge = bias_of(pt as i32)[t as usize];
                let e = (lat.edge[(d * lat.k + pi) * lat.k + c] - host_edge).abs() / host_edge.abs().max(1.0);
                assert!(
                    e < 1e-3,
                    "level {d} parent {pt} candidate {t}: edge {} vs host {host_edge}",
                    lat.edge[(d * lat.k + pi) * lat.k + c]
                );
            }
        }
        // the walk's own score: at the walked parent (candidate 0 of the level before, the anchor at level 0 --
        // parent row 0 either way) the pick's unary + edge beats every other captured candidate of the level
        for (c, &ct) in cand.iter().enumerate() {
            if ct == u32::MAX {
                continue;
            }
            let pick_score = lat.unary[d * lat.k] + lat.edge[d * lat.k * lat.k];
            let other = lat.unary[d * lat.k + c] + lat.edge[d * lat.k * lat.k + c];
            assert!(pick_score >= other - 1e-2, "level {d}: candidate {c} ({ct}) outscores the pick: {other} > {pick_score}");
        }
    }
}

/// `prefill_seed` puts position `p` in slot `p % window` -- the ring the reference's rotated split produces.
#[test]
fn prefill_seed_matches_the_host_ring() {
    let Some(st) = dev() else { return };
    let cfg = toy();
    let mut fx = fixture(&cfg, 61, &st);
    let n = cfg.window + 2;
    let (taps_f32, taps_dev) = taps(&cfg, n, &mut Rng(19));
    fx.gpu.prefill_seed(&taps_dev, 0, n, &st).expect("seed");
    st.sync().expect("sync");
    let mut host = fx.host[0].clone();
    let mut mx_all = Vec::new();
    for p in 0..n {
        let (_, mx) = host.forward_embed(&taps_f32[p * fx.tapw..][..fx.tapw], 1, &[0]);
        mx_all.extend_from_slice(&mx);
    }
    host.seed_cache(&mx_all, n, 0);
    let e = scaled_err(&f16s(&fx.gpu.stages[0].window), &host.window_kv);
    println!("ring err/scale {e:.3e}");
    {
        let g = f16s(&fx.gpu.stages[0].window);
        let hd = cfg.head_dim;
        for slot in 0..cfg.window {
            let a = &g[slot * hd..][..hd];
            let b = &host.window_kv[slot * hd..][..hd];
            println!("slot {slot}: gpu {:?} host {:?}", &a[..3], &b[..3]);
        }
    }
    assert!(e < 3e-3, "ring: err/scale {e:.3e}");
}

/// The slot list is the host's row, for a window that is not yet full and one that is.
#[test]
fn the_slot_list_is_the_host_row() {
    let Some(st) = dev() else { return };
    let cfg = toy();
    let mut fx = fixture(&cfg, 67, &st);
    let taps_dev = DevBuf::from_slice(&vec![0x3000u16; fx.tapw]).expect("taps");
    settle();
    for start in [1usize, 2, cfg.window, cfg.window + 3] {
        let reach = cfg.window.min(start + 1);
        let n_kv = reach + cfg.block_size;
        fx.gpu.draft(&taps_dev, 0, start, &st).expect("draft");
        st.sync().expect("sync");
        let got = fx.gpu.scratch.idxs.to_vec::<i32>().expect("idxs");
        let want = get_dspark_topk_row(cfg.window, cfg.block_size, start);
        for q in 0..cfg.block_size {
            assert_eq!(&got[q * n_kv..][..n_kv], &want[..], "start {start} query {q}");
        }
    }
}

/// The keyed walk's picks equal the host sampler's on the same per-row logits: the same splitmix64 noise per
/// (seed, absolute position, token id) and the same temperature / top_p / min_p transform. That is the
/// coupling contract: a sampled request's drafts draw the target's own noise per position, so a draft is
/// accepted exactly when it equals the target's keyed sample there.
///
/// The fast BF16 Markov walk (rank 256, large vocab) is the path the shipped pack rides; the fixture widens
/// the toy config to it and hands the Markov tables as pre-packed bf16 zeros, so the keyed kernel — not the
/// fallback argmax loop — is what is exercised, and the capture's unary scores ARE the walked logits.
#[test]
fn the_keyed_walk_draws_the_targets_noise_per_position() {
    let Some(st) = dev() else { return };
    let mut cfg = toy();
    cfg.markov_rank = 256;
    cfg.vocab = 4096;
    // the fast BF16-keyed walk needs the scratch `q`/`o` segments to cover walk_rows*1024*4 bytes; those
    // segments are `block_size * n_heads * head_dim * 2`, so widen the heads (the fallback argmax loop would
    // otherwise run silently)
    cfg.n_heads = 64;
    let mut fx = fixture_bf16_markov(&cfg, 73, &st);
    let mut r = Rng(29);
    let start = 6usize;
    let (_, pfbuf) = taps(&cfg, start, &mut r);
    fx.gpu.prefill_seed(&pfbuf, 0, start, &st).expect("seed");
    let (_, tapbuf) = taps(&cfg, 1, &mut r);
    let anchor = 9i32;

    let sampling = Sampling { seed: 0x5EED_7C0F_1234_ABCD, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 };
    let cand_k = sampling.candidate_count(cfg.vocab).clamp(1, 64);
    // the walk's transform is `try_choose` with top_k = cand_k (it cuts once, on its own candidate list);
    // the oracle must mirror that, not the request's top_k=20 (which is the target sampler's own cut)
    let oracle = Sampling { top_k: cand_k, ..sampling };
    let base_pos = start as u64 + 1; // the anchor sits at index `start`; drafts predict absolute positions from there
    let walk = MtpWalkSampling {
        cand_k,
        seed: sampling.seed,
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        min_p: sampling.min_p,
        base_pos,
    };
    fx.gpu.draft_n_keyed_slot(0, &tapbuf, anchor, start, cfg.block_size, Some(walk), &st).expect("keyed draft");
    st.sync().expect("sync");
    let (ids, _conf) = fx.gpu.read().expect("read");

    // the Markov tables here are zero, so the walked logits are the head logits: read them back and run the
    // host sampler on the same per-row candidates (top-cand_k, ordered by value desc / id asc)
    let logits = fx.gpu.scratch.logits.to_vec::<f32>().expect("logits");
    let vocab = cfg.vocab;
    for d in 0..cfg.block_size {
        let row = &logits[d * vocab..(d + 1) * vocab];
        let (cids, cvals) = top_candidates(row, cand_k);
        let want_tok = choose(&cvals, &cids, base_pos + d as u64, &oracle);
        assert_eq!(ids[d + 1] as u32, want_tok, "draft {d} at absolute position {}", base_pos + d as u64);
    }
}

/// The CUDA-graph draft round replays bit-identically to the eager round across advancing `start_pos`
/// (including across a window wrap boundary), with different anchors and keyed-walk seeds/base positions.
#[test]
fn the_graphed_draft_round_matches_eager_across_advancing_positions() {
    let Some(st) = dev() else { return };
    let mut cfg = toy();
    cfg.markov_rank = 256;
    cfg.vocab = 4096;
    cfg.n_heads = 64;
    // the fast BF16 walk requires q/o scratch >= walk_rows*1024*4 (rows*1024 fp32); toy hd=8 is too small, so widen
    cfg.head_dim = 32;

    // A deterministic schedule of rounds: (start_pos, anchor, keyed-walk sampling).
    // start_pos crosses the window boundary (cfg.window == 4) at 4 and 8; anchors differ; keyed walks
    // carry different seeds and base_pos per round.
    let mut r = Rng(91);
    let rounds: Vec<(usize, i32, Option<MtpWalkSampling>)> = (1..=12)
        .map(|start| {
            let anchor = (r.next_u64() % 40) as i32;
            let keyed = (start % 2) == 0;
            let walk = keyed.then(|| {
                let sampling = Sampling { seed: r.next_u64(), temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 };
                let cand_k = sampling.candidate_count(cfg.vocab).clamp(1, 64);
                MtpWalkSampling {
                    cand_k,
                    seed: sampling.seed,
                    temperature: sampling.temperature,
                    top_p: sampling.top_p,
                    min_p: sampling.min_p,
                    base_pos: start as u64 + 1,
                }
            });
            (start, anchor, walk)
        })
        .collect();

    // one tap buffer, reused for every round of both legs: a captured graph replays the *absolute* pointers it
    // was captured with, so replay must see the same buffer (with whatever contents it holds now), not a fresh one
    let (_, tapbuf) = taps(&cfg, 1, &mut r);

    // eager reference: round graphs off
    let prev = set_round_graphs_enabled(false);
    let mut fx_eager = fixture_bf16_markov(&cfg, 131, &st);
    let mut eager_out: Vec<(Vec<i32>, Vec<f32>)> = Vec::new();
    for (start, anchor, walk) in &rounds {
        fx_eager.gpu.draft_n_keyed_slot(0, &tapbuf, *anchor, *start, cfg.block_size, *walk, &st).expect("keyed draft eager");
        st.sync().expect("sync");
        let (eid, econ) = fx_eager.gpu.read().expect("read eager");
        eager_out.push((eid, econ));
    }
    set_round_graphs_enabled(prev);

    // graphed: round graphs on (the feature is default-on; re-assert)
    set_round_graphs_enabled(true);
    let mut fx_graph = fixture_bf16_markov(&cfg, 131, &st);
    let mut graph_out: Vec<(Vec<i32>, Vec<f32>)> = Vec::new();
    for (start, anchor, walk) in &rounds {
        fx_graph.gpu.draft_n_keyed_slot(0, &tapbuf, *anchor, *start, cfg.block_size, *walk, &st).expect("keyed draft graphed");
        st.sync().expect("sync");
        let (gid, gcon) = fx_graph.gpu.read().expect("read graphed");
        graph_out.push((gid, gcon));
    }
    set_round_graphs_enabled(prev);

    assert_eq!(eager_out.len(), graph_out.len(), "round count");
    for (i, ((e_ids, e_conf), (g_ids, g_conf))) in eager_out.iter().zip(graph_out.iter()).enumerate() {
        assert_eq!(e_ids, g_ids, "draft ids differ at start_pos={} anchor={}", rounds[i].0, rounds[i].1);
        // conf is only written by the *keyed* walk; the unkeyed fast walk leaves it uninitialized (garbage/NaN),
        // so compare it only where it is defined.
        if rounds[i].2.is_some() {
            assert_eq!(e_conf, g_conf, "draft confidences differ at start_pos={} anchor={}", rounds[i].0, rounds[i].1);
        }
    }
}

/// `bytes()` prices what `new` allocates, and a round does not move the arena.
#[test]
fn the_prices_are_the_allocations() {
    let Some(st) = dev() else { return };
    let cfg = toy();
    let mut fx = fixture(&cfg, 71, &st);
    assert_eq!(fx.gpu.scratch.bytes_held(), DsparkGpu::bytes(&cfg));
    assert!(fx.gpu.weight_bytes() > 0);
    let p0 = fx.gpu.scratch.h.ptr() as usize;
    let taps_dev = DevBuf::from_slice(&vec![0x3000u16; fx.tapw]).expect("taps");
    settle();
    fx.gpu.draft(&taps_dev, 1, 4, &st).expect("draft");
    st.sync().expect("sync");
    assert_eq!(fx.gpu.scratch.h.ptr() as usize, p0, "no reallocation across rounds");
}

/// **Timing at v41 shapes.** `cargo test --release -p gb-cuda --test dsv4_mtp_gpu v41 -- --ignored --nocapture`.
///
/// The pack's real shapes: three stages of 128 experts x 5120 x 2304 MXFP4 (2.26 GiB a stage), a 129,280 x 5120 fp8
/// head, two 129,280 x 256 f32 markov tables -- ~9 GB of device weight, which is why it is ignored by default. The
/// weights are a constant pattern: a kernel's cost does not depend on the values and a random 2.26 GiB expert set would
/// take minutes to generate.
#[test]
#[ignore]
fn v41_chain_timing() {
    let Some(st) = dev() else { return };
    let cfg = MtpConfig::v41();
    let d = cfg.dim;
    let tapw = cfg.tap_width();
    let fill = |n: usize| vec![0x38u8; n];
    let embed: Vec<u16> = vec![0x3000u16; cfg.vocab * d];
    let f8 = |n: usize, k: usize| W { n, k, bytes: fill(n * k), s: vec![120u8; (n / 32) * (k / 32)], deq: Vec::new() };
    let mut stages = Vec::new();
    for stage in 0..cfg.n_stages {
        let wq_a = f8(cfg.q_lora, d);
        let wq_b = f8(cfg.n_heads * cfg.head_dim, cfg.q_lora);
        let wkv = f8(cfg.head_dim, d);
        let wo_a = f8(cfg.o_groups * cfg.o_lora, cfg.o_in_per_group());
        let wo_b = f8(d, cfg.o_groups * cfg.o_lora);
        let sw1 = f8(cfg.inter_dim, d);
        let sw3 = f8(cfg.inter_dim, d);
        let sw2 = f8(d, cfg.inter_dim);
        let q_norm = vec![1f32; cfg.q_lora];
        let kv_norm = vec![1f32; cfg.head_dim];
        let sink = vec![0f32; cfg.n_heads];
        let gate = vec![0u16; cfg.n_routed_experts * d];
        let gate_bias = vec![0f32; cfg.n_routed_experts];
        let hc = vec![0.01f32; 24 * 4 * d];
        let hcb = vec![0f32; 24];
        let hcs = [1f32, 1.0, 1.0];
        let an = vec![1f32; d];
        // K2: (k/16)(n/16)16*2*2 bytes a matrix
        let k2 = |k: usize, n: usize| -> Exl3Weights {
            let tb = (k / 16) * (n / 16) * 16 * 2 * 2;
            Exl3Weights::from_dims(
                Exl3Dims { k, n, bits: 2, codebook: Codebook::Mul1 },
                DevBuf::from_slice(&vec![0x1111u16; tb / 2]).expect("trellis"),
                DevBuf::from_slice(&vec![0x2c00u16; k]).expect("suh"),
                DevBuf::from_slice(&vec![0x2c00u16; n]).expect("svh"),
                None,
            )
            .expect("exl3 weight")
        };
        let mut experts: Vec<[Exl3Weights; 3]> = Vec::with_capacity(cfg.n_routed_experts);
        for _ in 0..cfg.n_routed_experts {
            experts.push([k2(d, cfg.inter_dim), k2(cfg.inter_dim, d), k2(d, cfg.inter_dim)]);
        }
        let routed = RoutedExperts::Exl3(experts);
        let (mut main_proj, mut main_norm) = (None, None);
        let (mut norm, mut markov_embed, mut markov_head, mut conf_proj, mut lm_head) = (None, None, None, None, None);
        if stage == 0 {
            main_proj = Some(f8(d, tapw));
            main_norm = Some(vec![1f32; d]);
        }
        if stage + 1 == cfg.n_stages {
            norm = Some(vec![1f32; d]);
            markov_embed = Some(vec![0.01f32; cfg.vocab * cfg.markov_rank]);
            markov_head = Some(vec![0.01f32; cfg.vocab * cfg.markov_rank]);
            conf_proj = Some(vec![0.01f32; cfg.conf_in()]);
            lm_head = Some(f8(cfg.vocab, d));
        }
        stages.push(
            MtpStage::new(
                &cfg,
                MtpStagePack {
                    stage,
                    seq: 8192,
                    wq_a: wq_a.t(),
                    wq_b: wq_b.t(),
                    wkv: wkv.t(),
                    wo_a: wo_a.t(),
                    wo_b: wo_b.t(),
                    q_norm: &q_norm,
                    kv_norm: &kv_norm,
                    sink: &sink,
                    gate_bf16: &gate,
                    gate_bias: &gate_bias,
                    routed,
                    shared_w1: sw1.t(),
                    shared_w3: sw3.t(),
                    shared_w2: sw2.t(),
                    hc_attn: (&hc, &hcb, &hcs),
                    hc_ffn: (&hc, &hcb, &hcs),
                    attn_norm: &an,
                    ffn_norm: &an,
                    main_proj: main_proj.as_ref().map(|w| w.t()),
                    main_norm: main_norm.as_deref(),
                    norm: norm.as_deref(),
                    markov_embed: markov_embed.as_deref(),
                    markov_head: markov_head.as_deref(),
                    markov_embed_bf16: None,
                    markov_head_bf16: None,
                    conf_proj: conf_proj.as_deref(),
                    lm_head: lm_head.as_ref().map(|w| w.t()),
                },
            )
            .expect("stage"),
        );
    }
    let mut gpu = DsparkGpu::new(cfg.clone(), stages, &embed).expect("chain");
    let taps_dev = DevBuf::from_slice(&vec![0x3000u16; tapw]).expect("taps");
    settle();
    for _ in 0..2 {
        gpu.draft(&taps_dev, 100, 200, &st).expect("draft");
    }
    st.sync().expect("sync");
    let reps = 10;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        gpu.draft(&taps_dev, 100, 200, &st).expect("draft");
    }
    st.sync().expect("sync");
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    println!("v41 chain: {ms:.3} ms per draft round ({} tokens a round)", cfg.block_size + 1);
    println!("weights: {:.2} GiB", gpu.weight_bytes() as f64 / (1u64 << 30) as f64);
}

/// The device verify-length schedule (`schedule_verify_len_dev`) must be bit-for-bit the host's
/// (`DsparkGpu::schedule_verify_len`) for the same confidence vector: same clamp, same hard floor break, same
/// expected-token accumulation order, same cycle table, same `>=` tie-break. This is the byte-identity guard of
/// the on-GPU scheduler (T2.5b): a divergence here is a committed-text difference downstream.
#[test]
fn the_device_verify_len_schedule_mirrors_the_host_bit_for_bit() {
    let Some(st) = dev() else { return };
    let cycle = DevBuf::from_slice(&DsparkGpu::VERIFY_CYCLE_MS).expect("cycle");
    let mut r = Rng(0x5EED_CAFE);
    let mut cases: Vec<(Vec<f32>, f32, usize)> = vec![
        (vec![], 0.38, 7),
        (vec![0.5], 0.38, 7),
        (vec![0.9; 7], 0.38, 7),
        (vec![1.0; 7], 0.38, 7),
        (vec![0.0; 7], 0.38, 7),
        (vec![0.9, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9], 0.38, 7),
        (vec![0.9, 0.9, 0.9], 0.38, 7),
        (vec![0.9; 7], 0.38, 3),
        (vec![0.9; 7], 0.0, 7),
        (vec![0.9; 7], 1.5, 7),
        (vec![0.39, 0.37, 0.9, 0.9], 0.38, 7),
        (vec![1.2, -0.1, 0.5, 0.5], 0.38, 7),
    ];
    for _ in 0..200 {
        let n = 1 + (r.next_u64() % 7) as usize;
        cases.push(((0..n).map(|_| r.uniform(-0.1, 1.1)).collect(), 0.38, 7));
    }
    for (conf, min_conf, max) in &cases {
        let buf = DevBuf::from_slice(conf).expect("conf");
        let out = DevBuf::zeroed(4).expect("out");
        settle();
        schedule_verify_len_dev(&buf, conf.len(), *min_conf, *max, &cycle, &out, &st).expect("launch");
        st.sync().expect("sync");
        let mut v = [0i32; 1];
        out.download(&mut v).expect("d2h");
        let got = v[0].max(0) as usize;
        let want = DsparkGpu::schedule_verify_len(conf, *min_conf, *max);
        assert_eq!(got, want, "conf={conf:?} min_conf={min_conf} max={max}");
    }
}
