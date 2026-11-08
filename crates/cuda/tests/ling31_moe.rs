//! The Ling-3.1-flash-1019 MoE cases on FrostMoE: hidden 4096, expert intermediate 1280, 512 routed experts + the
//! shared expert (id 512, sigmoid gate), top-8 with the group-limited noaux_tc router (`BailingMoeV3Gate`: sigmoid
//! scores + expert_bias, top-2 per group, the top-4 of 8 groups of 64 experts, normalized unbiased sigmoid weights
//! times 2.5) and the SwiGLU clamps (gate clamped to at most L and up to +-L before `silu(gate) * up`; L = 10 on the
//! routed experts, and separately on the shared expert). Every expert's gate/up share a width, across the layer the
//! widths are mixed half-integer K2 in {2, 3, 4, 5} (1.0, 1.5, 2.0, 2.5 bpw); down is K2 in {3, 4, 5}.
//!
//! The decode and prefill kernels run against the packed-weight CPU reference and bit-exact row invariance (one row's
//! result is the same in a call of 1, 7 or 32 rows, at any position, whatever the other rows pick). GPU tests skip when
//! there is no device.
mod common;

const H: usize = 4096;
const I: usize = 1280;
const NE: usize = 512;
const TK: usize = 8;
/// FrostMoE's K chunks of the gate/up pass at H = 4096 (32 blocks of 128; valid counts 8, 16, 32).
const KC: usize = 8;
const N_GROUP: usize = 8;
const TOPK_GROUP: usize = 4;
const SCALE: f32 = 2.5;
const LIMIT: f32 = 10.0;
const SEED_STRIDE: usize = 1000;

use common::Rng;
use gb_cuda::{fmoe, DevBuf, Stream};
use gb_exl3::checkpoint::{Group, LoadedGroup};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::{Bits, Codebook};

fn gpu() -> Option<Stream> {
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    gb_cuda::rt::init_context().ok()?;
    Stream::new().ok()
}
fn bf(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn bff(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}
fn dl<T: Copy + Default>(v: &DevBuf, st: &Stream) -> Vec<T> {
    st.sync().unwrap();
    v.to_vec().unwrap()
}
fn f16(v: f64) -> u16 {
    f64_to_f16(v)
}

/// K2 of an expert's gate/up (shared across both) and down: the Ling-3.1 mixed profile.
fn k2_of(matrix: usize, expert: usize) -> usize {
    if matrix < 2 {
        [2usize, 3, 4, 5][expert % 4]
    } else {
        [3usize, 4, 5][expert % 3]
    }
}

/// A deterministic random packed trellis at a half-integer K2 (K2 = 2 * bits; the tile's i16 words are 8 * K2).
fn synth_group_k2(k: usize, n: usize, k2: usize, cb: Codebook, seed: u64) -> LoadedGroup {
    let mut r = Rng(seed);
    let words = (k / 16) * (n / 16) * 8 * k2;
    let trellis: Vec<i16> = (0..words).map(|_| r.next_u64() as i16).collect();
    let group = Group {
        prefix: "synthetic".into(),
        bits: Bits::from_twice(k2 as u8).unwrap(),
        codebook: cb,
        k,
        n,
        in_scales: "suh".into(),
        out_scales: "svh".into(),
        bias: false,
        files: vec![],
        bytes: 0,
    };
    let svh = (0..n)
        .map(|_| {
            f16_to_f64(f64_to_f16(
                (if r.next_u64() & 1 == 1 { -1.0 } else { 1.0 }) / (k as f64).sqrt()
                    * (0.5 + (r.next_u64() >> 40) as f64 / 16777216.0),
            ))
        })
        .collect();
    LoadedGroup {
        group,
        trellis,
        kt: k / 16,
        nt: n / 16,
        suh: (0..k)
            .map(|_| {
                f16_to_f64(f64_to_f16(
                    (if r.next_u64() & 1 == 1 { -8.0 } else { 8.0 }) * (0.5 + (r.next_u64() >> 40) as f64 / 16777216.0),
                ))
            })
            .collect(),
        svh,
        bias: None,
    }
}

const E: usize = NE + 1;

struct Owned {
    keep: Vec<DevBuf>,
    tab: [DevBuf; 9],
    kt: DevBuf,
    gs: [Vec<LoadedGroup>; 3],
}

impl Owned {
    fn new() -> Self {
        let gs = std::array::from_fn(|p| {
            (0..E)
                .map(|e| {
                    synth_group_k2(
                        if p == 2 { I } else { H },
                        if p == 2 { H } else { I },
                        k2_of(p, e),
                        Codebook::Mul1,
                        0x31a4 + (p * SEED_STRIDE + e) as u64,
                    )
                })
                .collect::<Vec<_>>()
        });
        let mut keep = Vec::new();
        let mut tables: Vec<DevBuf> = Vec::new();
        for proj in &gs {
            for kind in 0..3 {
                let mut ptrs = Vec::new();
                for g in proj {
                    let b = match kind {
                        0 => DevBuf::from_slice(&g.trellis).unwrap(),
                        1 => DevBuf::from_slice(&g.suh.iter().map(|x| f16(*x)).collect::<Vec<_>>()).unwrap(),
                        _ => DevBuf::from_slice(&g.svh.iter().map(|x| f16(*x)).collect::<Vec<_>>()).unwrap(),
                    };
                    ptrs.push(b.ptr() as u64);
                    keep.push(b);
                }
                tables.push(DevBuf::from_slice(&ptrs).unwrap());
            }
        }
        let tab: [DevBuf; 9] = tables.try_into().ok().unwrap();
        let kt: Vec<i32> = (0..E * 3).map(|i| k2_of(i % 3, i / 3) as i32).collect();
        let kt = DevBuf::from_slice(&kt).unwrap();
        Owned { keep, tab, kt, gs }
    }
    fn tables(&self) -> gb_cuda::fn_moe::ExpertTables<'_> {
        let a = &self.tab;
        gb_cuda::fn_moe::ExpertTables {
            g_t: &a[0],
            g_su: &a[1],
            g_sv: &a[2],
            u_t: &a[3],
            u_su: &a[4],
            u_sv: &a[5],
            d_t: &a[6],
            d_su: &a[7],
            d_sv: &a[8],
            ktab: &self.kt,
        }
    }
}

struct Inputs {
    x: Vec<u16>,
    logits: Vec<u16>,
    sg: Vec<u16>,
    bias: Vec<f32>,
}

/// The field of an extreme token scale so the projected gate/up values regularly exceed the 10.0 clamps.
fn inputs() -> Inputs {
    let mut r = Rng(0x51e31);
    Inputs {
        x: (0..32 * H).map(|_| bf(r.uniform(-2.0, 2.0))).collect(),
        logits: (0..32 * NE).map(|_| bf(r.uniform(-2.0, 2.0))).collect(),
        sg: (0..32).map(|_| bf(r.uniform(-1.0, 1.0))).collect(),
        bias: (0..NE).map(|_| r.uniform(-0.5, 0.5)).collect(),
    }
}

/// The router on the CPU in the kernel's exact semantics: per group the top-2 biased scores summed, the top-4 groups by
/// (value DESC, index ASC), top-8 over the masked biased scores by (value DESC, index ASC), weights = the unbiased
/// sigmoid scores of the selection divided by their sum + 1e-20 and scaled.
fn cpu_route_group(
    logits: &[f32],
    bias: &[f32],
    tk: usize,
    n_group: usize,
    topk_group: usize,
    scale: f32,
) -> (Vec<usize>, Vec<f64>) {
    let ne = logits.len();
    let glen = ne / n_group;
    let mut sc = vec![0f32; ne];
    let mut sb = vec![0f32; ne];
    for e in 0..ne {
        sc[e] = 1.0 / (1.0 + (-logits[e]).exp());
        sb[e] = sc[e] + bias[e];
    }
    let mut gsum = vec![0f32; n_group];
    for g in 0..n_group {
        let (mut t1, mut t2) = (f32::MIN, f32::MIN);
        for j in 0..glen {
            let v = sb[g * glen + j];
            if v > t1 {
                t2 = t1;
                t1 = v;
            } else if v > t2 {
                t2 = v;
            }
        }
        gsum[g] = t1 + t2;
    }
    let mut order: Vec<usize> = (0..n_group).collect();
    order.sort_by(|&a, &b| gsum[b].partial_cmp(&gsum[a]).unwrap().then(a.cmp(&b)));
    let sel_g: std::collections::HashSet<usize> = order[..topk_group].iter().copied().collect();
    let mut masked = vec![f32::NEG_INFINITY; ne];
    for e in 0..ne {
        if sel_g.contains(&(e / glen)) {
            masked[e] = sb[e];
        }
    }
    let mut all: Vec<usize> = (0..ne).collect();
    all.sort_by(|&a, &b| masked[b].partial_cmp(&masked[a]).unwrap().then(a.cmp(&b)));
    let sel: Vec<usize> = all[..tk].to_vec();
    let den: f64 = sel.iter().map(|&i| sc[i] as f64).sum::<f64>() + 1e-20;
    let w = sel.iter().map(|&i| sc[i] as f64 / den * scale as f64).collect();
    (sel, w)
}

fn cpu_projection(g: &LoadedGroup, x: &[f64]) -> Vec<f64> {
    let cb = gb_exl3::codebook(g.group.codebook);
    gb_exl3::forward(x, 1, &g.trellis, g.kt, g.nt, g.group.bits, &cb, &g.suh, &g.svh, None)
}

fn cpu_row(o: &Owned, inp: &Inputs, sel: &[usize], rw: &[f64], shared_w: f64, routed_limit: f32, shared_limit: f32) -> Vec<f32> {
    let x = inp.x[..H].iter().map(|v| bff(*v) as f64).collect::<Vec<_>>();
    let mut total = vec![0f32; H];
    for slot in 0..TK + 1 {
        let e = if slot < TK { sel[slot] } else { NE };
        let w = if slot < TK { rw[slot] } else { shared_w };
        let limit = if e == NE { shared_limit } else { routed_limit };
        let gate = cpu_projection(&o.gs[0][e], &x);
        let up = cpu_projection(&o.gs[1][e], &x);
        let act = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| {
                let mut g = *g;
                let mut u = *u;
                if limit > 0.0 {
                    g = g.min(limit as f64);
                    u = u.max(-(limit as f64)).min(limit as f64);
                }
                let silu = g / (1.0 + (-g).exp());
                bff(bf((silu * u) as f32)) as f64
            })
            .collect::<Vec<_>>();
        let down = cpu_projection(&o.gs[2][e], &act);
        for j in 0..H {
            total[j] += (down[j] as f32) * w as f32;
        }
    }
    total
}

fn run(o: &Owned, inp: &Inputs, t: usize, routed_limit: f32, shared_limit: f32, st: &Stream) -> (Vec<u16>, Vec<u16>) {
    let x = DevBuf::from_slice(&inp.x[..t * H]).unwrap();
    let logits = DevBuf::from_slice(&inp.logits[..t * NE]).unwrap();
    let sg = DevBuf::from_slice(&inp.sg[..t]).unwrap();
    let bias = DevBuf::from_slice(&inp.bias).unwrap();
    let gr = fmoe::GroupRoute { expert_bias: Some(&bias), n_group: N_GROUP, topk_group: TOPK_GROUP, scale: SCALE };
    let limits = fmoe::SwigluLimit { routed: routed_limit, shared: shared_limit };
    let shape = fmoe::Shape { t, ne: NE, top_k: TK, h: H, i: I, kc: KC, max_tokens: 32 };
    let mut frost = DevBuf::zeroed(t * H * 2).unwrap();
    let mut ws = fmoe::workspace(32, TK, H, I, KC).unwrap();
    fmoe::forward(&x, &logits, &sg, &o.tables(), &mut frost, shape, Some(gr), limits, &mut ws, st).unwrap();
    let mut prefill = DevBuf::zeroed(t * H * 2).unwrap();
    let mut pws = fmoe::prefill_workspace(32, TK, NE, H, I).unwrap();
    fmoe::prefill(&x, &logits, &sg, &o.tables(), &mut prefill, shape, Some(gr), limits, &mut pws, st).unwrap();
    (dl(&frost, st), dl(&prefill, st))
}

fn rel(got: &[u16], want: &[f32]) -> f64 {
    let (e, n) = got
        .iter()
        .zip(want)
        .fold((0f64, 0f64), |(e, n), (&a, &b)| (e + (f64::from(bff(a) - b)).powi(2), n + (f64::from(b)).powi(2)));
    (e / n.max(1e-30)).sqrt()
}

/// Isolate the row dependence: same mixed-K layer and clamps but the LEGACY softmax router (gr = None). If this is row
/// invariant, the group route is the culprit; if not, the K2 decode or the clamps are.
#[test]
#[ignore]
fn legacy_router_row_invariance_probe() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    let inp = inputs();
    let limits = fmoe::SwigluLimit { routed: LIMIT, shared: LIMIT };
    let shape = fmoe::Shape { t: 32, ne: NE, top_k: TK, h: H, i: I, kc: KC, max_tokens: 32 };
    let x = DevBuf::from_slice(&inp.x).unwrap();
    let logits = DevBuf::from_slice(&inp.logits).unwrap();
    let sg = DevBuf::from_slice(&inp.sg).unwrap();
    let mut ws = fmoe::workspace(32, TK, H, I, KC).unwrap();
    let mut out32 = DevBuf::zeroed(32 * H * 2).unwrap();
    fmoe::forward(&x, &logits, &sg, &o.tables(), &mut out32, shape, None, limits, &mut ws, &st).unwrap();
    let full = dl::<u16>(&out32, &st);
    let shape1 = fmoe::Shape { t: 1, ne: NE, top_k: TK, h: H, i: I, kc: KC, max_tokens: 32 };
    let x1 = DevBuf::from_slice(&inp.x[..H]).unwrap();
    let l1 = DevBuf::from_slice(&inp.logits[..NE]).unwrap();
    let g1 = DevBuf::from_slice(&inp.sg[..1]).unwrap();
    let mut out1 = DevBuf::zeroed(H * 2).unwrap();
    fmoe::forward(&x1, &l1, &g1, &o.tables(), &mut out1, shape1, None, limits, &mut ws, &st).unwrap();
    let one = dl::<u16>(&out1, &st);
    assert_eq!(&one[..H], &full[..H], "legacy router: row0 of a 1-row call must equal row0 of a 32-row call");
}

/// The mixed-K group-limited router, decode and prefill against the CPU reference, with the SwiGLU clamps of both the
/// routed (L65..71-style: 10) and the routed-0/shared-10 layer (L64-style), and row invariance at 1, 7 and 32 rows.
#[test]
#[ignore]
fn ling31_mixed_k_moe_group_route_swiglu_limit() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    let inp = inputs();
    // the router itself: selection and weights must match the CPU exactly (group top-4 + top-8, tie -> lower index)
    let logits = DevBuf::from_slice(&inp.logits).unwrap();
    let sg = DevBuf::from_slice(&inp.sg).unwrap();
    let bias = DevBuf::from_slice(&inp.bias).unwrap();
    let mut ws = DevBuf::zeroed(32 * (TK + 1) * 8 + (NE + 1) * 4).unwrap();
    let (sel, wts) = fmoe::group_route(&logits, &sg, &bias, SCALE, 32, NE, TK, N_GROUP, TOPK_GROUP, &mut ws, &st).unwrap();
    for row in 0..32 {
        let l: Vec<f32> = (0..NE).map(|e| bff(inp.logits[row * NE + e])).collect();
        let (want_sel, want_w) = cpu_route_group(&l, &inp.bias, TK, N_GROUP, TOPK_GROUP, SCALE);
        for k in 0..TK {
            assert_eq!(sel[row * (TK + 1) + k] as usize, want_sel[k], "router selection k={k} of row {row}");
            assert!((wts[row * (TK + 1) + k] as f64 - want_w[k]).abs() < 5e-6, "router weight k={k} of row {row}");
        }
        assert_eq!(sel[row * (TK + 1) + TK] as usize, NE, "shared expert entry of row {row}");
        assert!((wts[row * (TK + 1) + TK] - 1.0 / (1.0 + (-bff(inp.sg[row])).exp())).abs() < 1e-6, "shared weight of row {row}");
    }
    for (routed_limit, shared_limit) in [(LIMIT, LIMIT), (0.0, LIMIT), (LIMIT, 0.0), (0.0, 0.0)] {
        let (frost, prefill) = run(&o, &inp, 32, routed_limit, shared_limit, &st);
        // row invariance: a call of t rows equals the first t rows of the 32-row call
        for t in [1usize, 7, 32] {
            let (f_s, p_s) = run(&o, &inp, t, routed_limit, shared_limit, &st);
            if &f_s[..t * H] != &frost[..t * H] {
                let n = f_s[..t * H].iter().zip(&frost[..t * H]).take_while(|(a, b)| a == b).count();
                eprintln!(
                    "decode diff at t={t} limits ({routed_limit},{shared_limit}): first diff elem {n} (f_s={} frost={})",
                    f_s[n], frost[n]
                );
            }
            if &p_s[..t * H] != &prefill[..t * H] {
                let n = p_s[..t * H].iter().zip(&prefill[..t * H]).take_while(|(a, b)| a == b).count();
                eprintln!(
                    "prefill diff at t={t} limits ({routed_limit},{shared_limit}): first diff elem {n} (p_s={} prefill={})",
                    p_s[n], prefill[n]
                );
            }
            assert_eq!(&f_s[..t * H], &frost[..t * H], "decode row invariance at t={t} limits ({routed_limit},{shared_limit})");
            assert_eq!(
                &p_s[..t * H],
                &prefill[..t * H],
                "prefill row invariance at t={t} limits ({routed_limit},{shared_limit})"
            );
        }
        // the CPU reference for row 0 with the kernel's route
        let l0: Vec<f32> = (0..NE).map(|e| bff(inp.logits[e])).collect();
        let (sel0, w0) = cpu_route_group(&l0, &inp.bias, TK, N_GROUP, TOPK_GROUP, SCALE);
        let shared_w = 1.0 / (1.0 + (-bff(inp.sg[0])).exp()) as f64;
        let cpu0 = cpu_row(&o, &inp, &sel0, &w0, shared_w, routed_limit, shared_limit);
        for (name, got) in [("decode", &frost[..H]), ("prefill", &prefill[..H])] {
            let err = rel(got, &cpu0);
            assert!(err < 0.03, "{name} vs CPU reference at limits ({routed_limit},{shared_limit}): rel_l2 {err}");
        }
    }
    // the clamps must actually bind for this seed: the clamped output differs from the unclamped one
    let (clamped, _) = run(&o, &inp, 32, LIMIT, LIMIT, &st);
    let (plain, _) = run(&o, &inp, 32, 0.0, 0.0, &st);
    let neq = clamped.iter().zip(&plain).filter(|(a, b)| a != b).count();
    eprintln!("clamp sensitivity: {neq} of {} elements differ", clamped.len());
    // CPU gate/up magnitudes of row 0's first routed expert, to confirm values exceed 10
    {
        let l0: Vec<f32> = (0..NE).map(|e| bff(inp.logits[e])).collect();
        let (sel0, _) = cpu_route_group(&l0, &inp.bias, TK, N_GROUP, TOPK_GROUP, SCALE);
        let x = inp.x[..H].iter().map(|v| bff(*v) as f64).collect::<Vec<_>>();
        let gate = cpu_projection(&o.gs[0][sel0[0]], &x);
        let up = cpu_projection(&o.gs[1][sel0[0]], &x);
        let over_g = gate.iter().filter(|&&v| v > 10.0).count();
        let over_u = up.iter().filter(|&&v| v > 10.0).count();
        eprintln!("clamp stats row0 expert {}: gate>10 {over_g}/{} up>10 {over_u}/{}", sel0[0], gate.len(), up.len());
    }
    assert_ne!(&clamped, &plain, "the SwiGLU clamps of 10 must change the output for the extreme-x seed");
}

/// The mixed-K decoder is bit-exact against the CPU unpack at the half-integer widths (K2 3 and 5).
#[test]
#[ignore]
fn ling31_mixed_k_unpack_matches_cpu() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    for k2 in [3usize, 5] {
        let g = synth_group_k2(H, I, k2, Codebook::Mul1, 0xbeef + k2 as u64);
        let trellis = DevBuf::from_slice(&g.trellis).unwrap();
        let mut unpack = DevBuf::zeroed(H * I * 2).unwrap();
        fmoe::unpack(&trellis, &mut unpack, H, I, k2, &st).unwrap();
        let cpu = gb_exl3::unpack(&g.trellis, g.kt, g.nt, g.group.bits, &gb_exl3::codebook(g.group.codebook));
        assert_eq!(dl::<u16>(&unpack, &st), cpu, "mixed-K K2={k2} decoder must match CPU unpack");
    }
    let _ = &o.keep;
}

/// The server's prefill call shape: T=34 rows with a max_tokens=64 workspace, plus a NaN-logit run that must not fault.
#[test]
#[ignore]
fn ling31_moe_prefill_size34_probe() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    let mut r = Rng(0x51e33);
    let t = 34;
    let x: Vec<u16> = (0..t * H).map(|_| bf(r.uniform(-2.0, 2.0))).collect();
    let l: Vec<u16> = (0..t * NE).map(|_| bf(r.uniform(-2.0, 2.0))).collect();
    let sg: Vec<u16> = (0..t).map(|_| bf(r.uniform(-1.0, 1.0))).collect();
    let bias: Vec<f32> = (0..NE).map(|_| r.uniform(-0.5, 0.5)).collect();
    let xb = DevBuf::from_slice(&x).unwrap();
    let lb = DevBuf::from_slice(&l).unwrap();
    let sgb = DevBuf::from_slice(&sg).unwrap();
    let bb = DevBuf::from_slice(&bias).unwrap();
    let gr = fmoe::GroupRoute { expert_bias: Some(&bb), n_group: N_GROUP, topk_group: TOPK_GROUP, scale: SCALE };
    let limits = fmoe::SwigluLimit { routed: LIMIT, shared: LIMIT };
    let shape = fmoe::Shape { t, ne: NE, top_k: TK, h: H, i: I, kc: KC, max_tokens: 64 };
    let mut ws = fmoe::workspace(64, TK, H, I, KC).unwrap();
    let mut frost = DevBuf::zeroed(t * H * 2).unwrap();
    fmoe::forward(&xb, &lb, &sgb, &o.tables(), &mut frost, shape, Some(gr), limits, &mut ws, &st).unwrap();
    // NaN logits must not fault (the guards clamp bg/bi), even though the weights become NaN
    let lnan: Vec<u16> = (0..t * NE).map(|_| bf(f32::NAN)).collect();
    let lnb = DevBuf::from_slice(&lnan).unwrap();
    let mut frost2 = DevBuf::zeroed(t * H * 2).unwrap();
    fmoe::forward(&xb, &lnb, &sgb, &o.tables(), &mut frost2, shape, Some(gr), limits, &mut ws, &st).unwrap();
}
