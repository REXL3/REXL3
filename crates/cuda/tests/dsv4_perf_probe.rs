//! Native CUDA performance probes, not CPU correctness tests.
//! Raw kernel symbols are unavailable in NO_CUDA builds (even behind runtime guards).
#![cfg(not(gb_no_cuda))]

use gb_cuda::dsv4_attn::*;
use gb_cuda::dsv4_kvq4::KvFmt;
use gb_cuda::dsv4_moe::*;
use gb_cuda::exl3_linear::{Dtype, Exl3Dims, Exl3Weights};
use gb_cuda::fp8_linear::Fp8Weights;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::Codebook;
use std::time::Instant;

fn sync_time(st: &Stream, iters: usize, mut f: impl FnMut(usize)) -> f64 {
    for i in 0..3 {
        f(i);
    }
    st.sync().unwrap();
    let t0 = Instant::now();
    for i in 0..iters {
        f(i);
    }
    st.sync().unwrap();
    t0.elapsed().as_secs_f64() * 1e3 / iters as f64
}

#[test]
fn probe_all_dsv41_kernels() {
    if !device_available() {
        return;
    }
    let st = Stream::new().unwrap();

    eprintln!("=== 1. FP8 GEMM shapes (cold, rotated >= 256 MB) ===");
    let fp8_shapes = [
        ("wq_a", 1280usize, 5120usize, 40usize),
        ("wq_b", 32768, 1280, 40),
        ("wkv", 512, 5120, 40),
        ("wo_b", 5120, 8192, 40),
        ("ix_wqb", 4096, 1280, 8),
        ("shared_w13", 2304, 5120, 80),
        ("shared_w2", 5120, 2304, 40),
    ];
    for (name, n, k, mult) in fp8_shapes {
        let bytes = n * k + (n / 32) * (k / 32);
        let copies = ((256 << 20) / bytes).clamp(4, 64);
        let w = vec![0x38u8; n * k];
        let s = vec![0x7Fu8; (n / 32) * (k / 32)];
        let mats: Vec<Fp8Weights> = (0..copies).map(|_| Fp8Weights::new(n, k, &w, &s).unwrap()).collect();
        let x = DevBuf::zeroed(64 * k * 2).unwrap();
        let mut y = DevBuf::zeroed(64 * n * 2).unwrap();
        let mut line = format!("{name:<12} [{n:>5}, {k:>4}] x{mult:<2} ({:>5.1} MB/call) |", bytes as f64 / 1e6);
        for m in [1usize, 2, 4, 6, 8, 16, 32, 64] {
            let ms = sync_time(&st, copies * 2, |i| {
                mats[i % copies].forward(&x, &mut y, m, Dtype::F16, &st).unwrap();
            });
            let gbps = (bytes as f64 / 1e9) / (ms * 1e-3);
            let total_40 = ms * mult as f64;
            line += &format!(" m={m}: {ms:>6.3}ms ({gbps:>5.1} GB/s, x{mult}={total_40:>5.2}ms) |");
        }
        eprintln!("{line}");
    }

    eprintln!("\n=== 2. wo_a grouped [8, 1024, 4096] (cold, rotated >= 256 MB) ===");
    {
        let (groups, o_lora, o_in) = (8usize, 1024usize, 4096usize);
        let n = groups * o_lora;
        let k = o_in;
        let bytes = n * k + (n / 32) * (k / 32);
        let copies = ((256 << 20) / bytes).clamp(4, 32);
        let w = vec![0x38u8; n * k];
        let s = vec![0x7Fu8; (n / 32) * (k / 32)];
        let mats: Vec<(DevBuf, DevBuf)> =
            (0..copies).map(|_| (DevBuf::from_slice(&w).unwrap(), DevBuf::from_slice(&s).unwrap())).collect();
        let o = DevBuf::zeroed(64 * groups * o_in * 2).unwrap();
        let out = DevBuf::zeroed(64 * groups * o_lora * 2).unwrap();
        let mut line = format!("wo_a         [{n:>5}, {k:>4}] x40 ({:>5.1} MB/call) |", bytes as f64 / 1e6);
        for m in [1usize, 2, 4, 6, 8, 16, 32, 64] {
            let ms = sync_time(&st, copies * 2, |i| {
                let (wb, sb) = &mats[i % copies];
                unsafe {
                    extern "C" {
                        fn gb_dsv4_woa_grouped(
                            o: *const core::ffi::c_void,
                            w: *const u8,
                            s: *const u8,
                            out: *mut core::ffi::c_void,
                            rows: i32,
                            groups: i32,
                            o_lora: i32,
                            o_in: i32,
                            dtype: i32,
                            use_classic: i32,
                            stream: *mut core::ffi::c_void,
                        ) -> i32;
                    }
                    gb_dsv4_woa_grouped(
                        o.ptr(),
                        wb.ptr() as *const u8,
                        sb.ptr() as *const u8,
                        out.ptr(),
                        m as i32,
                        groups as i32,
                        o_lora as i32,
                        o_in as i32,
                        0,
                        0,
                        st.raw(),
                    );
                }
            });
            let gbps = (bytes as f64 / 1e9) / (ms * 1e-3);
            line += &format!(" m={m}: {ms:>6.3}ms ({gbps:>5.1} GB/s, x40={:>5.2}ms) |", ms * 40.0);
        }
        eprintln!("{line}");
    }

    eprintln!("\n=== 3. MLA non-GEMM kernels (x40 layers equivalent) ===");
    {
        let cfg = Dsv4Config::v41();
        let rope = SharedRope::new(&cfg, 4096).unwrap();
        let rpair = rope.get(&cfg, 2);
        let q = DevBuf::zeroed(8 * 64 * 512 * 2).unwrap();
        let kv = DevBuf::zeroed(8 * 512 * 2).unwrap();
        let qr = DevBuf::zeroed(8 * 1280 * 2).unwrap();
        let q_norm = DevBuf::from_slice(&vec![1.0f32; 1280]).unwrap();
        let kv_norm = DevBuf::from_slice(&vec![1.0f32; 512]).unwrap();
        let sink = DevBuf::from_slice(&vec![0.0f32; 64]).unwrap();
        let planes = Dsv4Planes::new(&cfg, 4096, 2, KvFmt::F16).unwrap();
        let sc = Dsv4Scratch::new(&cfg, 8, 4096).unwrap();
        for m in [1usize, 4, 6] {
            let t_rope_q = sync_time(&st, 100, |_| {
                rope_rows_dev(&q, m, 64 * 512, 64, 512, 64, rpair.pair(), 128, 1, false, &st).unwrap();
            });
            let t_quant_o = sync_time(&st, 100, |_| {
                quant_dev(&q, m, 64 * 512, 32, 8, true, &st).unwrap();
            });
            let t_qnorm = sync_time(&st, 100, |_| {
                rmsnorm_dev(&qr, &q_norm, m, 1280, 1e-20, &st).unwrap();
            });
            let t_kvnorm = sync_time(&st, 100, |_| {
                rmsnorm_dev(&kv, &kv_norm, m, 512, 1e-20, &st).unwrap();
            });
            let t_attn = sync_time(&st, 100, |_| {
                gb_cuda::dsv4_kvq4::attn_dev(
                    &q,
                    &planes.window,
                    &planes.compress,
                    128 + m,
                    &sink,
                    &planes.topk,
                    planes.sel_stride,
                    128 + 256,
                    &sc.o,
                    m,
                    64,
                    512,
                    KvFmt::F16,
                    KvFmt::F16,
                    1.0 / 512.0f32.sqrt(),
                    &st,
                )
                .unwrap();
            });
            eprintln!(
                "m={m}: rope_q(x80)={:.2}ms quant_o(x40)={:.2}ms qnorm(x40)={:.2}ms kvnorm(x40)={:.2}ms attn(x40,len=384)={:.2}ms",
                t_rope_q * 80.0,
                t_quant_o * 40.0,
                t_qnorm * 40.0,
                t_kvnorm * 40.0,
                t_attn * 40.0
            );
        }
    }

    eprintln!("\n=== 4. Router + mHC (cold, x40 layers / x80 sites) ===");
    {
        let gate_bf16 = vec![0x3c00u16; 384 * 5120];
        let bias = vec![0.0f32; 384];
        let routers: Vec<Dsv4Router> =
            (0..40).map(|_| Dsv4Router::new(&gate_bf16, &bias, &bias, Dsv4Dims::V41).unwrap()).collect();
        let fnw = vec![0.01f32; 24 * 4 * 5120];
        let base = vec![0.0f32; 24];
        let scale = [0.5f32, 0.25, 0.125];
        let sites: Vec<HcSite> = (0..80).map(|_| HcSite::new(&fnw, &base, &scale).unwrap()).collect();
        let x_f16 = DevBuf::zeroed(8 * 5120 * 2).unwrap();
        let mut logits = DevBuf::zeroed(8 * 384 * 4).unwrap();
        let mut picks = DevBuf::zeroed(8 * 6 * 4).unwrap();
        let mut wts = DevBuf::zeroed(8 * 6 * 2).unwrap();
        let mut streams = DevBuf::zeroed(8 * 4 * 5120 * 4).unwrap();
        let mut work = HcWork::new(8).unwrap();
        let hc = Dsv4Hc { cfg: HcCfg::V41 };
        for m in [1usize, 4, 6] {
            let t_rtr_log = sync_time(&st, 40, |i| {
                routers[i % 40].logits(&x_f16, Dtype::F16, m, &mut logits, &st).unwrap();
            });
            let t_rtr_sel = sync_time(&st, 40, |i| {
                routers[i % 40].select(&logits, m, None, &mut picks, &mut wts, &st).unwrap();
            });
            let t_hc_part = sync_time(&st, 80, |i| {
                hc.partials(&streams, &sites[i % 80], m, &mut work, &st).unwrap();
            });
            let t_hc_fin = sync_time(&st, 80, |i| {
                hc.finalize(&streams, &sites[i % 80], m, &mut work, &st).unwrap();
            });
            let t_hc_app = sync_time(&st, 80, |_| {
                hc.apply(&mut streams, &x_f16, Dtype::F16, m, &work, &st).unwrap();
            });
            eprintln!(
                "m={m}: router_logits(x40)={:.2}ms router_select(x40)={:.2}ms hc_partials(x80)={:.2}ms hc_finalize(x80)={:.2}ms hc_apply(x80)={:.2}ms",
                t_rtr_log * 40.0,
                t_rtr_sel * 40.0,
                t_hc_part * 80.0,
                t_hc_fin * 80.0,
                t_hc_app * 80.0
            );
        }
    }

    eprintln!("\n=== 5. HeadDevice::Exl3 [5120, 129280] K=8 ===");
    {
        let (k, n, bits) = (5120usize, 129280usize, 8usize);
        let d = Exl3Dims { k, n, bits, codebook: Codebook::Mul1 };
        let trellis = DevBuf::zeroed((k / 16) * (n / 16) * 16 * bits * 2).unwrap();
        let suh = DevBuf::zeroed(k * 2).unwrap();
        let svh = DevBuf::zeroed(n * 2).unwrap();
        let w = Exl3Weights::from_dims(d, trellis, suh, svh, None).unwrap();
        let x = DevBuf::zeroed(64 * k * 2).unwrap();
        let mut y = DevBuf::zeroed(64 * n * 2).unwrap();
        let mut ws = DevBuf::zeroed(gb_cuda::exl3_linear::linear_workspace_bytes(64, k, n, bits)).unwrap();
        for m in [1usize, 2, 4, 6, 8, 16, 32, 64] {
            let ms = sync_time(&st, 20, |_| {
                w.forward(&x, &mut y, m, Dtype::F16, Dtype::F16, &mut ws, &st).unwrap();
            });
            let gbps = (w.weight_bytes() as f64 / 1e9) / (ms * 1e-3);
            eprintln!("lm_head m={m}: {ms:.3} ms ({gbps:.1} GB/s)");
        }
    }

    eprintln!("\n=== 6. moe_experts_resident (cold, 384 experts x 1 layer, scaled x40) ===");
    {
        let (h, inter, n_exp, topk) = (5120usize, 2304usize, 384usize, 6usize);
        let mut exps: Vec<[Exl3Weights; 3]> = Vec::with_capacity(n_exp);
        for e in 0..n_exp {
            let b13 = if e % 3 == 0 { 2 } else { 1 };
            let b2 = 2;
            let d1 = Exl3Dims { k: h, n: inter, bits: b13, codebook: Codebook::Mul1 };
            let d2 = Exl3Dims { k: inter, n: h, bits: b2, codebook: Codebook::Mul1 };
            let mk = |d: Exl3Dims| {
                Exl3Weights::from_dims(
                    d,
                    DevBuf::zeroed((d.k / 16) * (d.n / 16) * 16 * d.bits * 2).unwrap(),
                    DevBuf::zeroed(d.k * 2).unwrap(),
                    DevBuf::zeroed(d.n * 2).unwrap(),
                    None,
                )
                .unwrap()
            };
            exps.push([mk(d1), mk(d2), mk(d1)]);
        }
        let refs: Vec<&[Exl3Weights]> = exps.iter().map(|g| g.as_slice()).collect();
        let descs = build_expert_descs(&refs, h, inter).unwrap().unwrap();
        let mut s = MoeScratch::new(512, topk, h, inter, n_exp).unwrap();
        let hidden_in = DevBuf::zeroed(512 * h * 2).unwrap();
        let mut shared_f16 = DevBuf::zeroed(512 * h * 2).unwrap();
        let mut moe_out = DevBuf::zeroed(512 * h * 4).unwrap();
        let wts = DevBuf::from_slice(&vec![0x3c00u16; 512 * topk]).unwrap();
        for m in [1usize, 2, 4, 6, 8, 16, 32, 64, 128, 512] {
            let distinct: Vec<i32> = (0..(m * topk)).map(|i| ((i * 7) % n_exp) as i32).collect();
            let picks_d = DevBuf::from_slice(&distinct).unwrap();
            let ms_d = sync_time(&st, 20, |iter| {
                let shift: Vec<i32> = distinct.iter().map(|&x| (x + (iter as i32 * 37)) % (n_exp as i32)).collect();
                let _ = shift;
                moe_experts_resident(&mut s, &descs, m, &hidden_in, &mut shared_f16, &picks_d, &wts, &mut moe_out, 10.0, &st)
                    .unwrap();
            });
            eprintln!("moe_experts_resident m={m} ({} slots): {ms_d:.3} ms/layer -> x40 = {:.2} ms", m * topk, ms_d * 40.0);
        }
    }
}

#[test]
fn probe_mla_full_block() {
    if !device_available() {
        return;
    }
    let st = Stream::new().unwrap();
    let cfg = Dsv4Config::v41();
    let rope = SharedRope::new(&cfg, 4096).unwrap();
    let wq_a = vec![0x38u8; 1280 * 5120];
    let wq_a_s = vec![0x7Fu8; (1280 / 32) * (5120 / 32)];
    let wq_b = vec![0x38u8; 32768 * 1280];
    let wq_b_s = vec![0x7Fu8; (32768 / 32) * (1280 / 32)];
    let wkv = vec![0x38u8; 512 * 5120];
    let wkv_s = vec![0x7Fu8; (512 / 32) * (5120 / 32)];
    let wo_a = vec![0x38u8; 8192 * 4096];
    let wo_a_s = vec![0x7Fu8; (8192 / 32) * (4096 / 32)];
    let wo_b = vec![0x38u8; 5120 * 8192];
    let wo_b_s = vec![0x7Fu8; (5120 / 32) * (8192 / 32)];
    let q_norm = vec![1.0f32; 1280];
    let kv_norm = vec![1.0f32; 512];
    let sink = vec![0.0f32; 64];
    let c_wkv = vec![0u8; 512 * 5120 * 2];
    let c_wgate = vec![0u8; 512 * 5120 * 2];
    let c_norm = vec![1.0f32; 512];
    let i_wqb = vec![0x38u8; 4096 * 1280];
    let i_wqb_s = vec![0x7Fu8; (4096 / 32) * (1280 / 32)];
    let i_wproj = vec![0u8; 32 * 5120 * 2];
    let i_wk = vec![0u8; 128 * 512 * 2];
    let i_knorm = vec![1.0f32; 128];

    // Build 4 representative layers:
    // layer 0 (window only, x2 in model), layer 2 (kv+index source, x4 in model),
    // layer 3 (reader, x30 in model), layer 24 (index-only source with candidates, x4 in model)
    let make_layer = |layer: usize| -> MlaDsv4 {
        let comp = if cfg.is_kv_source(layer) {
            Some(CompressorPack { wkv: &c_wkv, wgate: if cfg.ratio(layer) > 1 { Some(&c_wgate) } else { None }, norm: &c_norm })
        } else {
            None
        };
        let ix = if cfg.is_index_source(layer) {
            Some(IndexerPack {
                wq_b: (&i_wqb, &i_wqb_s),
                weights_proj: &i_wproj,
                wk: if cfg.is_kv_source(layer) { Some(&i_wk) } else { None },
                k_norm: if cfg.is_kv_source(layer) { &i_knorm } else { &[] },
            })
        } else {
            None
        };
        let pack = MlaDsv4Pack {
            wq_a: (&wq_a, &wq_a_s),
            wq_b: (&wq_b, &wq_b_s),
            wkv: (&wkv, &wkv_s),
            wo_a: (&wo_a, &wo_a_s),
            wo_b: (&wo_b, &wo_b_s),
            q_norm: &q_norm,
            kv_norm: &kv_norm,
            sink: &sink,
            compressor: comp,
            indexer: ix,
        };
        MlaDsv4::new_shared(&cfg, layer, &pack, &rope).unwrap()
    };

    let mut m0 = make_layer(0);
    let mut m2 = make_layer(2);
    let mut m3 = make_layer(3);
    let mut m20 = make_layer(20);
    let mut m24 = make_layer(24);
    let p0 = Dsv4Planes::new(&cfg, 4096, 0, KvFmt::F16).unwrap();
    let p2 = Dsv4Planes::new(&cfg, 4096, 2, KvFmt::F16).unwrap();
    let p3 = Dsv4Planes::new(&cfg, 4096, 3, KvFmt::F16).unwrap();
    let p20 = Dsv4Planes::new(&cfg, 4096, 20, KvFmt::F16).unwrap();
    let p24 = Dsv4Planes::new(&cfg, 4096, 24, KvFmt::F16).unwrap();
    let mut sc = Dsv4Scratch::new(&cfg, 8, 4096).unwrap();
    let x = DevBuf::zeroed(8 * 5120 * 2).unwrap();

    // Seed planes at start_pos = 0..280 first so topk has realistic indices
    for pos in (0..280).step_by(4) {
        m0.forward(&x, 4, pos, &p0, None, &mut sc, &st).unwrap();
        m2.forward(&x, 4, pos, &p2, None, &mut sc, &st).unwrap();
        m3.forward(&x, 4, pos, &p3, Some(&p2), &mut sc, &st).unwrap();
        m20.forward(&x, 4, pos, &p20, None, &mut sc, &st).unwrap();
        m24.forward(&x, 4, pos, &p24, Some(&p20), &mut sc, &st).unwrap();
    }
    st.sync().unwrap();

    eprintln!("\n=== 7b. Full MlaDsv4::forward by layer role at start_pos=280 ===");
    for m in [1usize, 4, 6, 8] {
        let t0 = sync_time(&st, 50, |_| m0.forward(&x, m, 280, &p0, None, &mut sc, &st).unwrap());
        let t2 = sync_time(&st, 50, |_| m2.forward(&x, m, 280, &p2, None, &mut sc, &st).unwrap());
        let t3 = sync_time(&st, 50, |_| m3.forward(&x, m, 280, &p3, Some(&p2), &mut sc, &st).unwrap());
        let t20 = sync_time(&st, 50, |_| m20.forward(&x, m, 280, &p20, None, &mut sc, &st).unwrap());
        let t24 = sync_time(&st, 50, |_| m24.forward(&x, m, 280, &p24, Some(&p20), &mut sc, &st).unwrap());
        let weighted = 2.0 * t0 + 3.0 * t2 + 1.0 * t20 + 4.0 * t24 + 30.0 * t3;
        eprintln!(
            "m={m} @ pos=280: L0(win,x2)={t0:.3}ms L2(kv+ix,x3)={t2:.3}ms L20(cand,x1)={t20:.3}ms L24(ix,x4)={t24:.3}ms L3(read,x30)={t3:.3}ms => weighted_40={weighted:.2}ms"
        );
    }
    // Seed planes at start_pos = 0..64 first so topk has realistic indices
    for pos in (0..64).step_by(4) {
        m0.forward(&x, 4, pos, &p0, None, &mut sc, &st).unwrap();
        m2.forward(&x, 4, pos, &p2, None, &mut sc, &st).unwrap();
        m3.forward(&x, 4, pos, &p3, Some(&p2), &mut sc, &st).unwrap();
        m20.forward(&x, 4, pos, &p20, None, &mut sc, &st).unwrap();
        m24.forward(&x, 4, pos, &p24, Some(&p20), &mut sc, &st).unwrap();
    }
    st.sync().unwrap();

    eprintln!("\n=== 7. Full MlaDsv4::forward by layer role at start_pos=64 ===");
    for m in [1usize, 4, 6] {
        let t0 = sync_time(&st, 50, |_| m0.forward(&x, m, 64, &p0, None, &mut sc, &st).unwrap());
        let t2 = sync_time(&st, 50, |_| m2.forward(&x, m, 64, &p2, None, &mut sc, &st).unwrap());
        let t3 = sync_time(&st, 50, |_| m3.forward(&x, m, 64, &p3, Some(&p2), &mut sc, &st).unwrap());
        let t20 = sync_time(&st, 50, |_| m20.forward(&x, m, 64, &p20, None, &mut sc, &st).unwrap());
        let t24 = sync_time(&st, 50, |_| m24.forward(&x, m, 64, &p24, Some(&p20), &mut sc, &st).unwrap());
        let weighted = 2.0 * t0 + 3.0 * t2 + 1.0 * t20 + 4.0 * t24 + 30.0 * t3;
        eprintln!(
            "m={m}: L0(win,x2)={t0:.3}ms L2(kv+ix,x3)={t2:.3}ms L20(cand,x1)={t20:.3}ms L24(ix,x4)={t24:.3}ms L3(read,x30)={t3:.3}ms => weighted_40={weighted:.2}ms"
        );
    }
}

#[test]
fn probe_mtp_gpu_chain() {
    if !device_available() {
        return;
    }
    let st = Stream::new().unwrap();
    use gb_cuda::dsv4_mtp::MtpConfig;
    use gb_cuda::dsv4_mtp_gpu::{DsparkGpu, HcPack, MtpDense, MtpStage, RoutedExperts};

    let mcfg = MtpConfig {
        dim: 5120,
        n_heads: 64,
        head_dim: 512,
        rope_dim: 64,
        q_lora: 1280,
        o_lora: 1024,
        o_groups: 8,
        window: 128,
        block_size: 5,
        noise_token_id: 128814,
        n_stages: 3,
        target_layer_ids: vec![37, 38, 39],
        vocab: 129280,
        markov_rank: 256,
        n_routed_experts: 128,
        topk: 3,
        inter_dim: 2304,
        swiglu_limit: 10.0,
        route_scale: 1.5,
        gate_temp: 1.0,
        hc: HcCfg::V41,
        eps: 1e-20,
        rope_theta: 10000.0,
        temperature: 0.0,
    };
    let dense = |n: usize, k: usize| -> MtpDense {
        MtpDense::from_parts(
            n,
            k,
            DevBuf::zeroed(n * k).unwrap(),
            DevBuf::from_slice(&vec![0x7Fu8; (n / 32) * (k / 32)]).unwrap(),
        )
        .unwrap()
    };
    let mut stages = Vec::new();
    for idx in 0..3 {
        let last = idx == 2;
        let mut exps = Vec::new();
        for _ in 0..128 {
            let d1 = Exl3Dims { k: 5120, n: 2304, bits: 2, codebook: Codebook::Mul1 };
            let d2 = Exl3Dims { k: 2304, n: 5120, bits: 2, codebook: Codebook::Mul1 };
            let mk = |d: Exl3Dims| {
                Exl3Weights::from_dims(
                    d,
                    DevBuf::zeroed((d.k / 16) * (d.n / 16) * 16 * d.bits * 2).unwrap(),
                    DevBuf::zeroed(d.k * 2).unwrap(),
                    DevBuf::zeroed(d.n * 2).unwrap(),
                    None,
                )
                .unwrap()
            };
            exps.push([mk(d1), mk(d2), mk(d1)]);
        }
        let hc_site = || {
            HcPack::from_parts(
                5120,
                DevBuf::zeroed(24 * 4 * 5120 * 4).unwrap(),
                DevBuf::zeroed(24 * 4).unwrap(),
                DevBuf::from_slice(&[0.5f32, 0.25, 0.125]).unwrap(),
            )
            .unwrap()
        };
        let dh = Exl3Dims { k: 5120, n: 129280, bits: 8, codebook: Codebook::Mul1 };
        stages.push(MtpStage {
            stage: idx,
            wq_a: dense(1280, 5120),
            wq_b: dense(64 * 512, 1280),
            wkv: dense(512, 5120),
            wo_a: dense(8 * 1024, 4096),
            wo_b: dense(5120, 8192),
            q_norm: DevBuf::from_slice(&vec![1.0f32; 1280]).unwrap(),
            kv_norm: DevBuf::from_slice(&vec![1.0f32; 512]).unwrap(),
            sink: DevBuf::zeroed(64 * 4).unwrap(),
            rope: RopeTables::new(RopeSpec {
                dim: 64,
                seq: 4096,
                original_seq_len: 0,
                base: 10000.0,
                factor: 1.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
            })
            .unwrap(),
            window: DevBuf::zeroed(128 * 512 * 2).unwrap(),
            gate: DevBuf::zeroed(128 * 5120 * 2).unwrap(),
            gate_bias: DevBuf::zeroed(128 * 4).unwrap(),
            routed: RoutedExperts::Exl3(exps),
            shared_w1: dense(2304, 5120),
            shared_w3: dense(2304, 5120),
            shared_w2: dense(5120, 2304),
            hc_attn: hc_site(),
            hc_ffn: hc_site(),
            attn_norm: DevBuf::from_slice(&vec![1.0f32; 5120]).unwrap(),
            ffn_norm: DevBuf::from_slice(&vec![1.0f32; 5120]).unwrap(),
            main_proj: if idx == 0 { Some(dense(5120, 15360)) } else { None },
            main_norm: if idx == 0 { Some(DevBuf::from_slice(&vec![1.0f32; 5120]).unwrap()) } else { None },
            norm: if last { Some(DevBuf::from_slice(&vec![1.0f32; 5120]).unwrap()) } else { None },
            markov_embed: if last { Some(DevBuf::zeroed(129280 * 256 * 2).unwrap()) } else { None },
            markov_head: if last { Some(DevBuf::zeroed(129280 * 256 * 2).unwrap()) } else { None },
            conf_proj: if last { Some(DevBuf::zeroed((5120 + 256) * 2).unwrap()) } else { None },
            conf_bias: if last { Some(DevBuf::zeroed(4).unwrap()) } else { None },
            lm_head: None,
            lm_head_exl3: if last {
                Some(
                    Exl3Weights::from_dims(
                        dh,
                        DevBuf::zeroed((dh.k / 16) * (dh.n / 16) * 16 * dh.bits * 2).unwrap(),
                        DevBuf::zeroed(dh.k * 2).unwrap(),
                        DevBuf::zeroed(dh.n * 2).unwrap(),
                        None,
                    )
                    .unwrap(),
                )
            } else {
                None
            },
        });
    }
    let mut gpu =
        DsparkGpu::from_parts(mcfg, stages, DevBuf::zeroed(129280 * 5120 * 2).unwrap(), Dtype::BF16, Dtype::BF16).unwrap();
    let taps = DevBuf::zeroed(15360 * 2).unwrap();
    let t_embed = sync_time(&st, 20, |_| gpu.forward_embed(&taps, 100, &st).unwrap());
    let t_stg0 = sync_time(&st, 20, |_| gpu.stage_forward(0, 64, 69, &st).unwrap());
    let t_attn0 = sync_time(&st, 20, |_| gpu.attn(0, 64, 69, &st).unwrap());
    let t_moe0 = sync_time(&st, 20, |_| {
        gpu.moe(0, 5, &st).unwrap();
    });
    let t_head = sync_time(&st, 20, |_| gpu.head(&st).unwrap());
    let t_full = sync_time(&st, 20, |_| gpu.draft(&taps, 100, 64, &st).unwrap());
    eprintln!(
        "=== 8. DsparkGpu (EXL3 K=2 experts + EXL3 K=8 head + BF16 markov) ===\nembed={t_embed:.3}ms stage0={t_stg0:.3}ms (attn={t_attn0:.3}ms moe={t_moe0:.3}ms hc={:.3}ms) x3={:.3}ms head={t_head:.3}ms => full={t_full:.3}ms",
        t_stg0 - t_attn0 - t_moe0,
        t_stg0 * 3.0
    );
}

#[test]
fn probe_mla_subkernels_pos280() {
    if !device_available() {
        return;
    }
    let st = Stream::new().unwrap();
    let cfg = Dsv4Config::v41();
    let rope = SharedRope::new(&cfg, 4096).unwrap();
    let _rpair = rope.get(&cfg, 20);
    let q = DevBuf::zeroed(8 * 64 * 512 * 2).unwrap();
    let _kv = DevBuf::zeroed(8 * 512 * 2).unwrap();
    let _qr = DevBuf::zeroed(8 * 1280 * 2).unwrap();
    let _q_norm = DevBuf::from_slice(&vec![1.0f32; 1280]).unwrap();
    let _kv_norm = DevBuf::from_slice(&vec![1.0f32; 512]).unwrap();
    let sink = DevBuf::from_slice(&vec![0.0f32; 64]).unwrap();
    let planes = Dsv4Planes::new(&cfg, 4096, 20, KvFmt::F16).unwrap();
    // Fill planes.topk with valid indices 0..416 for all 8 rows!
    let mut topk_host = vec![-1i32; 8 * planes.sel_stride];
    for r in 0..8 {
        for j in 0..128 {
            topk_host[r * planes.sel_stride + j] = ((j * 7 + r) % 128) as i32;
        }
        for j in 0..280 {
            topk_host[r * planes.sel_stride + 128 + j] = (128 + 8 + j) as i32;
        }
    }
    let topk_buf = DevBuf::from_slice(&topk_host).unwrap();
    let sc = Dsv4Scratch::new(&cfg, 8, 4096).unwrap();
    for m in [1usize, 4, 6, 8] {
        let t_attn_128 = sync_time(&st, 100, |_| {
            gb_cuda::dsv4_kvq4::attn_dev(
                &q,
                &planes.window,
                &planes.compress,
                128 + m,
                &sink,
                &topk_buf,
                planes.sel_stride,
                128,
                &sc.o,
                m,
                64,
                512,
                KvFmt::F16,
                KvFmt::F16,
                1.0 / 512.0f32.sqrt(),
                &st,
            )
            .unwrap();
        });
        let t_attn_268 = sync_time(&st, 100, |_| {
            gb_cuda::dsv4_kvq4::attn_dev(
                &q,
                &planes.window,
                &planes.compress,
                128 + m,
                &sink,
                &topk_buf,
                planes.sel_stride,
                128 + 140,
                &sc.o,
                m,
                64,
                512,
                KvFmt::F16,
                KvFmt::F16,
                1.0 / 512.0f32.sqrt(),
                &st,
            )
            .unwrap();
        });
        let t_attn_408 = sync_time(&st, 100, |_| {
            gb_cuda::dsv4_kvq4::attn_dev(
                &q,
                &planes.window,
                &planes.compress,
                128 + m,
                &sink,
                &topk_buf,
                planes.sel_stride,
                128 + 280,
                &sc.o,
                m,
                64,
                512,
                KvFmt::F16,
                KvFmt::F16,
                1.0 / 512.0f32.sqrt(),
                &st,
            )
            .unwrap();
        });
        let t_attn_640 = sync_time(&st, 100, |_| {
            gb_cuda::dsv4_kvq4::attn_dev(
                &q,
                &planes.window,
                &planes.compress,
                128 + m,
                &sink,
                &topk_buf,
                planes.sel_stride,
                128 + 512,
                &sc.o,
                m,
                64,
                512,
                KvFmt::F16,
                KvFmt::F16,
                1.0 / 512.0f32.sqrt(),
                &st,
            )
            .unwrap();
        });
        eprintln!(
            "m={m}: attn_dev x40: len=128 -> {:.2}ms | len=268 -> {:.2}ms | len=408 -> {:.2}ms | len=640 -> {:.2}ms",
            t_attn_128 * 40.0,
            t_attn_268 * 40.0,
            t_attn_408 * 40.0,
            t_attn_640 * 40.0
        );
    }
}

#[test]
fn probe_detailed_breakdown() {
    if !device_available() {
        return;
    }
    let st = Stream::new().unwrap();
    let cfg = Dsv4Config::v41();
    let rope = SharedRope::new(&cfg, 8192).unwrap();
    let rpair = rope.get(&cfg, 20);
    let x = DevBuf::zeroed(8 * 5120 * 2).unwrap();
    let q = DevBuf::zeroed(8 * 64 * 512 * 2).unwrap();
    let kv = DevBuf::zeroed(8 * 512 * 2).unwrap();
    let qr = DevBuf::zeroed(8 * 1280 * 2).unwrap();
    let index_q = DevBuf::zeroed(8 * 32 * 128 * 2).unwrap();
    let index_w = DevBuf::zeroed(8 * 32 * 4).unwrap();
    let q_norm = DevBuf::from_slice(&vec![1.0f32; 1280]).unwrap();
    let kv_norm = DevBuf::from_slice(&vec![1.0f32; 512]).unwrap();
    let planes = Dsv4Planes::new(&cfg, 8192, 20, KvFmt::F16).unwrap();
    let sc = Dsv4Scratch::new(&cfg, 8, 8192).unwrap();
    let w_bf16_512 = Bf16W::new(512, 5120, &vec![0u8; 512 * 5120 * 2]).unwrap();
    let w_bf16_32 = Bf16W::new(32, 5120, &vec![0u8; 32 * 5120 * 2]).unwrap();
    let w_bf16_128 = Bf16W::new(128, 512, &vec![0u8; 128 * 512 * 2]).unwrap();

    for m in [1usize, 4, 6, 8] {
        let t_rope_q = sync_time(&st, 100, |_| {
            rope_rows_dev(&q, m, 64 * 512, 64, 512, 64, rpair.pair(), 280, 1, false, &st).unwrap();
        });
        let t_quant_o = sync_time(&st, 100, |_| {
            quant_dev(&q, m, 64 * 512, 32, 8, true, &st).unwrap();
        });
        let t_quant_kv = sync_time(&st, 100, |_| {
            quant_dev(&kv, m, 512, 32, 8, true, &st).unwrap();
        });
        let t_quant_iq = sync_time(&st, 100, |_| {
            quant_dev(&index_q, m, 4096, 32, 4, true, &st).unwrap();
        });
        let t_bf16_512 = sync_time(&st, 100, |_| {
            let mut out = DevBuf::zeroed(8 * 512 * 2).unwrap();
            w_bf16_512.forward(&x, &mut out, m, Dtype::F16, &st).unwrap();
        });
        let t_bf16_32 = sync_time(&st, 100, |_| {
            let mut out = DevBuf::zeroed(8 * 32 * 4).unwrap();
            w_bf16_32.forward(&x, &mut out, m, Dtype::F32, &st).unwrap();
        });
        eprintln!(
            "m={m}: rope_q(x80)={:.2}ms quant_o(x40)={:.2}ms quant_kv(x40)={:.2}ms quant_iq(x8)={:.2}ms",
            t_rope_q * 80.0,
            t_quant_o * 40.0,
            t_quant_kv * 40.0,
            t_quant_iq * 8.0
        );
        let _ = (t_bf16_512, t_bf16_32, &w_bf16_128, &q_norm, &kv_norm, &planes, &sc, &qr, &index_w);
    }

    for pos in [64usize, 280, 512] {
        for m in [1usize, 4, 6, 8] {
            let n_pos = pos + m;
            let bscore = DevBuf::zeroed(m * n_pos.div_ceil(8) * 4).unwrap();
            let t_ix_score = sync_time(&st, 100, |_| unsafe {
                extern "C" {
                    fn gb_dsv4_index_score(
                        q: *const core::ffi::c_void,
                        index_k: *const core::ffi::c_void,
                        w: *const f32,
                        mask: *const u8,
                        mask_stride: i32,
                        score: *mut f32,
                        rows: i32,
                        n_pos: i32,
                        n_heads: i32,
                        dim: i32,
                        scale: f32,
                        start_pos: i32,
                        ratio: i32,
                        dtype: i32,
                        stream: *mut core::ffi::c_void,
                    ) -> i32;
                }
                gb_dsv4_index_score(
                    index_q.ptr(),
                    planes.index_k.ptr(),
                    index_w.ptr() as *const f32,
                    core::ptr::null(),
                    planes.cand_stride as i32,
                    sc.score.ptr() as *mut f32,
                    m as i32,
                    n_pos as i32,
                    32,
                    128,
                    1.0 / 1024.0f32.sqrt(),
                    pos as i32,
                    1,
                    0,
                    st.raw(),
                );
            });
            let t_cand = sync_time(&st, 100, |_| unsafe {
                extern "C" {
                    fn gb_dsv4_candidate_blocks(
                        logits: *const f32,
                        bscore: *mut f32,
                        mask: *mut u8,
                        mask_stride: i32,
                        rows: i32,
                        width: i32,
                        start_pos: i32,
                        ratio: i32,
                        topk_blocks: i32,
                        block_size: i32,
                        stream: *mut core::ffi::c_void,
                    ) -> i32;
                }
                gb_dsv4_candidate_blocks(
                    sc.score.ptr() as *const f32,
                    bscore.ptr() as *mut f32,
                    planes.candidates.ptr() as *mut u8,
                    planes.cand_stride as i32,
                    m as i32,
                    n_pos as i32,
                    pos as i32,
                    1,
                    2048,
                    8,
                    st.raw(),
                );
            });
            let t_topk = sync_time(&st, 100, |_| unsafe {
                extern "C" {
                    fn gb_dsv4_topk_select(
                        score: *const f32,
                        out: *mut i32,
                        rows: i32,
                        n_pos: i32,
                        topk: i32,
                        col_offset: i32,
                        val_offset: i32,
                        row_stride: i32,
                        stream: *mut core::ffi::c_void,
                    ) -> i32;
                }
                gb_dsv4_topk_select(
                    sc.score.ptr() as *const f32,
                    planes.topk.ptr() as *mut i32,
                    m as i32,
                    n_pos as i32,
                    512,
                    128,
                    128 + m as i32,
                    planes.sel_stride as i32,
                    st.raw(),
                );
            });
            eprintln!(
                "pos={pos:<3} m={m}: ix_score(x8)={:.2}ms cand(x1)={:.2}ms topk(x8)={:.2}ms",
                t_ix_score * 8.0,
                t_cand,
                t_topk * 8.0
            );
        }
    }
}

#[test]
fn probe_mla_prefill_breakdown() {
    if !device_available() {
        return;
    }
    let st = Stream::new().unwrap();
    let cfg = Dsv4Config::v41();
    let max_seq = 8192usize;
    let rope = SharedRope::new(&cfg, max_seq).unwrap();
    let wq_a = vec![0x38u8; 1280 * 5120];
    let wq_a_s = vec![0x7Fu8; (1280 / 32) * (5120 / 32)];
    let wq_b = vec![0x38u8; 32768 * 1280];
    let wq_b_s = vec![0x7Fu8; (32768 / 32) * (1280 / 32)];
    let wkv = vec![0x38u8; 512 * 5120];
    let wkv_s = vec![0x7Fu8; (512 / 32) * (5120 / 32)];
    let wo_a = vec![0x38u8; 8192 * 4096];
    let wo_a_s = vec![0x7Fu8; (8192 / 32) * (4096 / 32)];
    let wo_b = vec![0x38u8; 5120 * 8192];
    let wo_b_s = vec![0x7Fu8; (5120 / 32) * (8192 / 32)];
    let q_norm = vec![1.0f32; 1280];
    let kv_norm = vec![1.0f32; 512];
    let sink = vec![0.0f32; 64];
    let c_wkv = vec![0u8; 512 * 5120 * 2];
    let c_wgate = vec![0u8; 512 * 5120 * 2];
    let c_norm = vec![1.0f32; 512];
    let i_wqb = vec![0x38u8; 4096 * 1280];
    let i_wqb_s = vec![0x7Fu8; (4096 / 32) * (1280 / 32)];
    let i_wproj = vec![0u8; 32 * 5120 * 2];
    let i_wk = vec![0u8; 128 * 512 * 2];
    let i_knorm = vec![1.0f32; 128];

    let make_layer = |layer: usize| -> MlaDsv4 {
        let comp = if cfg.is_kv_source(layer) {
            Some(CompressorPack { wkv: &c_wkv, wgate: if cfg.ratio(layer) > 1 { Some(&c_wgate) } else { None }, norm: &c_norm })
        } else {
            None
        };
        let ix = if cfg.is_index_source(layer) {
            Some(IndexerPack {
                wq_b: (&i_wqb, &i_wqb_s),
                weights_proj: &i_wproj,
                wk: if cfg.is_kv_source(layer) { Some(&i_wk) } else { None },
                k_norm: if cfg.is_kv_source(layer) { &i_knorm } else { &[] },
            })
        } else {
            None
        };
        let pack = MlaDsv4Pack {
            wq_a: (&wq_a, &wq_a_s),
            wq_b: (&wq_b, &wq_b_s),
            wkv: (&wkv, &wkv_s),
            wo_a: (&wo_a, &wo_a_s),
            wo_b: (&wo_b, &wo_b_s),
            q_norm: &q_norm,
            kv_norm: &kv_norm,
            sink: &sink,
            compressor: comp,
            indexer: ix,
        };
        MlaDsv4::new_shared(&cfg, layer, &pack, &rope).unwrap()
    };

    let mut m0 = make_layer(0);
    let mut m2 = make_layer(2);
    let mut m3 = make_layer(3);
    let mut m20 = make_layer(20);
    let mut m24 = make_layer(24);
    let p0 = Dsv4Planes::new(&cfg, max_seq, 0, KvFmt::Q4).unwrap();
    let p2 = Dsv4Planes::new(&cfg, max_seq, 2, KvFmt::Q4).unwrap();
    let p3 = Dsv4Planes::new(&cfg, max_seq, 3, KvFmt::Q4).unwrap();
    let p20 = Dsv4Planes::new(&cfg, max_seq, 20, KvFmt::Q4).unwrap();
    let p24 = Dsv4Planes::new(&cfg, max_seq, 24, KvFmt::Q4).unwrap();
    let mut sc = Dsv4Scratch::new(&cfg, gb_cuda::dsv4_attn::MAX_ROWS, max_seq).unwrap();
    let x = DevBuf::zeroed(gb_cuda::dsv4_attn::MAX_ROWS * 5120 * 2).unwrap();

    for &start_pos in &[0usize, 512] {
        for &m in &[128usize, gb_cuda::dsv4_attn::MAX_ROWS] {
            if m > gb_cuda::dsv4_attn::MAX_ROWS {
                continue;
            }
            // Seed once
            m0.forward(&x, m, start_pos, &p0, None, &mut sc, &st).unwrap();
            m2.forward(&x, m, start_pos, &p2, None, &mut sc, &st).unwrap();
            m3.forward(&x, m, start_pos, &p3, Some(&p2), &mut sc, &st).unwrap();
            m20.forward(&x, m, start_pos, &p20, None, &mut sc, &st).unwrap();
            m24.forward(&x, m, start_pos, &p24, Some(&p20), &mut sc, &st).unwrap();
            st.sync().unwrap();

            let t0 = sync_time(&st, 10, |_| m0.forward(&x, m, start_pos, &p0, None, &mut sc, &st).unwrap());
            let t2 = sync_time(&st, 10, |_| m2.forward(&x, m, start_pos, &p2, None, &mut sc, &st).unwrap());
            let t3 = sync_time(&st, 10, |_| m3.forward(&x, m, start_pos, &p3, Some(&p2), &mut sc, &st).unwrap());
            let t20 = sync_time(&st, 10, |_| m20.forward(&x, m, start_pos, &p20, None, &mut sc, &st).unwrap());
            let t24 = sync_time(&st, 10, |_| m24.forward(&x, m, start_pos, &p24, Some(&p20), &mut sc, &st).unwrap());
            let weighted = 2.0 * t0 + 3.0 * t2 + 1.0 * t20 + 4.0 * t24 + 30.0 * t3;
            eprintln!(
                "PREFILL Q4 m={m} @ pos={start_pos}: L0(x2)={t0:.3}ms L2(x3)={t2:.3}ms L20(x1)={t20:.3}ms L24(x4)={t24:.3}ms L3(x30)={t3:.3}ms => weighted_40={weighted:.2}ms"
            );

            // Sub-kernel breakdown at m
            let t_wqa = sync_time(&st, 20, |_| m20.wq_a.forward(&x, &mut sc.qr, m, Dtype::F16, &st).unwrap());
            let t_wqb = sync_time(&st, 20, |_| m20.wq_b.forward(&sc.qr, &mut sc.q, m, Dtype::F16, &st).unwrap());
            let t_wkv = sync_time(&st, 20, |_| m20.wkv.forward(&x, &mut sc.kv, m, Dtype::F16, &st).unwrap());
            let t_woa = sync_time(&st, 20, |_| unsafe {
                extern "C" {
                    fn gb_dsv4_woa_grouped(
                        o: *const core::ffi::c_void,
                        w: *const u8,
                        s: *const u8,
                        out: *mut core::ffi::c_void,
                        rows: i32,
                        groups: i32,
                        o_lora: i32,
                        o_in: i32,
                        dtype: i32,
                        use_classic: i32,
                        stream: *mut core::ffi::c_void,
                    ) -> i32;
                }
                gb_dsv4_woa_grouped(
                    sc.o.ptr(),
                    m20.wo_a.w.ptr() as *const u8,
                    m20.wo_a.s.ptr() as *const u8,
                    sc.olora.ptr(),
                    m as i32,
                    8,
                    1024,
                    4096,
                    0,
                    0,
                    st.raw(),
                );
            });
            let t_wob = sync_time(&st, 20, |_| m20.wo_b.forward(&sc.olora, &mut sc.out, m, Dtype::F16, &st).unwrap());
            let t_rope_q = sync_time(&st, 20, |_| {
                rope_rows_dev(&sc.q, m, 32768, 64, 512, 64, m20.rope.pair(), start_pos, 1, false, &st).unwrap()
            });
            let t_rot_q = sync_time(&st, 20, |_| gb_cuda::dsv4_kvq4::rot_dev(&sc.q, m, 32768, 448, 512, 64, &st).unwrap());
            let t_quant_o = sync_time(&st, 20, |_| quant_dev(&sc.o, m, 32768, 32, 8, true, &st).unwrap());
            let split = if start_pos == 0 { m } else { 128 + m };
            let win_slots = if start_pos == 0 { m.min(128) } else { 128 };
            let n_cmp = (start_pos + m).min(512);
            let idx_len = win_slots + n_cmp;
            let win_stage_bytes = (128 + m) * 512 * 2;
            let topk_stage_off = (win_stage_bytes + 4095) & !4095;
            let cand_stage_off = topk_stage_off + ((m * p20.sel_stride * 4 + 4095) & !4095);
            let topk_ptr = if m <= 128 { p20.topk.ptr() as *mut i32 } else { sc.oq.ptr_at(topk_stage_off) as *mut i32 };
            let cand_ptr = if m <= 128 { p20.candidates.ptr() as *mut u8 } else { sc.oq.ptr_at(cand_stage_off) as *mut u8 };
            let t_attn = sync_time(&st, 20, |_| {
                gb_cuda::dsv4_kvq4::attn_raw_dev(
                    &sc.q,
                    &sc.oq,
                    &p20.compress,
                    split,
                    &m20.sink,
                    topk_ptr as *const i32,
                    p20.sel_stride,
                    idx_len,
                    &sc.o,
                    m,
                    64,
                    512,
                    KvFmt::F16,
                    KvFmt::Q4,
                    1.0 / 512.0f32.sqrt(),
                    &st,
                )
                .unwrap();
            });
            let comp20 = m20.compressor.as_ref().unwrap();
            let t_comp_wkv = sync_time(&st, 20, |_| comp20.wkv.forward(&x, &mut sc.ckv, m, Dtype::F16, &st).unwrap());
            let ix20 = m20.indexer.as_ref().unwrap();
            let t_ix_wqb = sync_time(&st, 20, |_| ix20.wq_b.forward(&sc.qr, &mut sc.index_q, m, Dtype::F16, &st).unwrap());
            let t_ix_wproj = sync_time(&st, 20, |_| ix20.weights_proj.forward(&x, &mut sc.index_w, m, Dtype::F32, &st).unwrap());
            let n_pos = start_pos + m;
            let bscore = DevBuf::zeroed(m * n_pos.div_ceil(8) * 4).unwrap();
            let t_ix_score = sync_time(&st, 20, |_| {
                gb_cuda::dsv4_kvq4::index_score_dev(
                    &sc.index_q,
                    &p20.index_k,
                    &sc.index_w,
                    None,
                    p20.cand_stride,
                    &sc.score,
                    m,
                    n_pos,
                    32,
                    128,
                    1.0 / 4096.0f32.sqrt(),
                    start_pos,
                    1,
                    KvFmt::Q4,
                    &st,
                )
                .unwrap();
            });
            let t_cand = sync_time(&st, 20, |_| unsafe {
                extern "C" {
                    fn gb_dsv4_candidate_blocks(
                        logits: *const f32,
                        bscore: *mut f32,
                        mask: *mut u8,
                        mask_stride: i32,
                        rows: i32,
                        width: i32,
                        start_pos: i32,
                        ratio: i32,
                        topk_blocks: i32,
                        block_size: i32,
                        stream: *mut core::ffi::c_void,
                    ) -> i32;
                }
                gb_dsv4_candidate_blocks(
                    sc.score.ptr() as *const f32,
                    bscore.ptr() as *mut f32,
                    cand_ptr,
                    p20.cand_stride as i32,
                    m as i32,
                    n_pos as i32,
                    start_pos as i32,
                    1,
                    2048,
                    8,
                    st.raw(),
                );
            });
            let t_topk = sync_time(&st, 20, |_| unsafe {
                extern "C" {
                    fn gb_dsv4_topk_select(
                        score: *const f32,
                        out: *mut i32,
                        rows: i32,
                        n_pos: i32,
                        topk: i32,
                        col_offset: i32,
                        val_offset: i32,
                        row_stride: i32,
                        stream: *mut core::ffi::c_void,
                    ) -> i32;
                }
                gb_dsv4_topk_select(
                    sc.score.ptr() as *const f32,
                    topk_ptr,
                    m as i32,
                    n_pos as i32,
                    512,
                    win_slots as i32,
                    split as i32,
                    p20.sel_stride as i32,
                    st.raw(),
                );
            });
            eprintln!(
                "  x40: wq_a={:.2}ms wq_b={:.2}ms wkv={:.2}ms wo_a={:.2}ms wo_b={:.2}ms | rope(x80)={:.2}ms rot(x80)={:.2}ms quant_o(x40)={:.2}ms attn(x40,len={idx_len})={:.2}ms",
                t_wqa * 40.0, t_wqb * 40.0, t_wkv * 40.0, t_woa * 40.0, t_wob * 40.0,
                t_rope_q * 80.0, t_rot_q * 80.0, t_quant_o * 40.0, t_attn * 40.0
            );
            eprintln!(
                "  comp/ix: comp_wkv(x7)={:.2}ms ix_wqb(x8)={:.2}ms ix_wproj(x8)={:.2}ms ix_score(x8,npos={n_pos})={:.2}ms cand(x1)={:.2}ms topk(x8)={:.2}ms",
                t_comp_wkv * 7.0, t_ix_wqb * 8.0, t_ix_wproj * 8.0, t_ix_score * 8.0, t_cand, t_topk * 8.0
            );
        }
    }
}

#[test]
fn probe_moe_prefill_breakdown() {
    if !device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    let st = Stream::new().unwrap();
    {
        let gate_bf16 = vec![0x3c00u16; 384 * 5120];
        let bias = vec![0.0f32; 384];
        let routers: Vec<Dsv4Router> =
            (0..40).map(|_| Dsv4Router::new(&gate_bf16, &bias, &bias, Dsv4Dims::V41).unwrap()).collect();
        let fnw = vec![0.01f32; 24 * 4 * 5120];
        let base = vec![0.0f32; 24];
        let scale = [0.5f32, 0.25, 0.125];
        let sites: Vec<HcSite> = (0..80).map(|_| HcSite::new(&fnw, &base, &scale).unwrap()).collect();
        let x_f16 = DevBuf::zeroed(512 * 5120 * 2).unwrap();
        let mut logits = DevBuf::zeroed(512 * 384 * 4).unwrap();
        let mut picks = DevBuf::zeroed(512 * 6 * 4).unwrap();
        let mut wts = DevBuf::zeroed(512 * 6 * 2).unwrap();
        let mut streams = DevBuf::zeroed(512 * 4 * 5120 * 4).unwrap();
        let mut moe_out = DevBuf::zeroed(512 * 5120 * 4).unwrap();
        let mut hidden = DevBuf::zeroed(512 * 5120 * 2).unwrap();
        let norm_w = DevBuf::from_slice(&vec![1.0f32; 5120]).unwrap();
        let pre_src = DevBuf::zeroed(512 * 4 * 4).unwrap();
        let mut work = HcWork::new(512).unwrap();
        let hc = Dsv4Hc { cfg: HcCfg::V41 };
        let m = 512usize;
        let t_rtr_log = sync_time(&st, 40, |i| {
            routers[i % 40].logits(&x_f16, Dtype::F16, m, &mut logits, &st).unwrap();
        });
        let t_rtr_sel = sync_time(&st, 40, |i| {
            routers[i % 40].select(&logits, m, None, &mut picks, &mut wts, &st).unwrap();
        });
        let t_hc_part = sync_time(&st, 80, |i| {
            hc.partials_and_collapse(&streams, &sites[i % 80], &pre_src, &mut moe_out, m, &mut work, &st).unwrap();
        });
        let t_hc_fin = sync_time(&st, 80, |i| {
            hc.finalize(&streams, &sites[i % 80], m, &mut work, &st).unwrap();
        });
        let t_hc_col = 0.0f64;
        let t_rms = sync_time(&st, 80, |_| {
            gb_cuda::dsv4_attn::rmsnorm_f32_to_f16_dev(&moe_out, &norm_w, &mut hidden, m, 5120, 1e-20, &st).unwrap();
        });
        let t_hc_app = sync_time(&st, 80, |_| {
            hc.apply(&mut streams, &x_f16, Dtype::F16, m, &work, &st).unwrap();
        });
        eprintln!(
            "PREFILL m={m}: router_logits(x40)={:.2}ms router_select(x40)={:.2}ms hc_partials(x80)={:.2}ms hc_finalize(x80)={:.2}ms hc_collapse(x80)={:.2}ms rms_norm(x80)={:.2}ms hc_apply(x80)={:.2}ms",
            t_rtr_log * 40.0,
            t_rtr_sel * 40.0,
            t_hc_part * 80.0,
            t_hc_fin * 80.0,
            t_hc_col * 80.0,
            t_rms * 80.0,
            t_hc_app * 80.0
        );
    }
    let (h, inter, n_exp, topk) = (5120usize, 2304usize, 384usize, 6usize);
    let mut exps: Vec<[Exl3Weights; 3]> = Vec::with_capacity(n_exp);
    for e in 0..n_exp {
        let b13 = if e % 3 == 0 { 2 } else { 1 };
        let b2 = 2;
        let d1 = Exl3Dims { k: h, n: inter, bits: b13, codebook: Codebook::Mul1 };
        let d2 = Exl3Dims { k: inter, n: h, bits: b2, codebook: Codebook::Mul1 };
        let mk = |d: Exl3Dims| {
            Exl3Weights::from_dims(
                d,
                DevBuf::zeroed((d.k / 16) * (d.n / 16) * 16 * d.bits * 2).unwrap(),
                DevBuf::zeroed(d.k * 2).unwrap(),
                DevBuf::zeroed(d.n * 2).unwrap(),
                None,
            )
            .unwrap()
        };
        exps.push([mk(d1), mk(d2), mk(d1)]);
    }
    let refs: Vec<&[Exl3Weights]> = exps.iter().map(|g| g.as_slice()).collect();
    let descs = build_expert_descs(&refs, h, inter).unwrap().unwrap();
    let mut sc = MoeScratch::new(512, topk, h, inter, n_exp).unwrap();
    let hidden_in = DevBuf::zeroed(512 * h * 2).unwrap();
    let mut shared_f16 = DevBuf::zeroed(512 * h * 2).unwrap();
    let mut moe_out = DevBuf::zeroed(512 * h * 4).unwrap();
    let wts = DevBuf::from_slice(&vec![0x3c00u16; 512 * topk]).unwrap();

    for (label, skew) in [("uniform", false), ("skewed_real", true)] {
        for &m in &[128usize, 256, 512] {
            let mut picks_host = Vec::with_capacity(m * topk);
            for r in 0..m {
                for k in 0..topk {
                    let e = if !skew {
                        (r * topk + k) % n_exp
                    } else if r % 2 == 0 {
                        ((r / 2) * topk + k * 5) % 32
                    } else {
                        (r * topk + k * 7) % n_exp
                    };
                    picks_host.push(e as i32);
                }
            }
            let picks = DevBuf::from_slice(&picks_host).unwrap();
            let t_ms = sync_time(&st, 20, |_| {
                moe_experts_resident(&mut sc, &descs, m, &hidden_in, &mut shared_f16, &picks, &wts, &mut moe_out, 10.0, &st)
                    .unwrap();
            });
            eprintln!(
                "MOE [{label}] m={m}: {t_ms:.3} ms/layer => x40 = {:.2} ms ({:.3} ms/tok)",
                t_ms * 40.0,
                (t_ms * 40.0) / (m as f64)
            );
        }
    }
}
