// Synthetic packed EXL3 MoE for CoopMK and FrostMoE, with CPU packed-weight reference, for the layer shape the including test target sets:
// `H` (hidden), `I` (expert intermediate), `NE` routed experts, `TK` experts a token picks, `KC` (FrostMoE's K chunks of the gate/up
// pass), `SEED_STRIDE` and `bits_of(matrix, expert)` (the EXL3 bits of gate (0), up (1) and down (2) of an expert). Included, not a
// module, so every target keeps the plain test names; each declares `mod common` and the constants.
//   tests/fn_moe.rs        Qwen3.8-Flash-Next: H 2560, I 640, 2 experts, top-1, every matrix 2 bits (its golden is committed).
//   tests/minicpm_moe.rs   MiniCPM-V-4.7 (Qwen3.5/3.6-35B-A3B backbone): H 2048, I 512, top-8 + the shared expert, bits 3..8 mixed.
use common::{exl3_synth::synth_group, Rng};
use gb_cuda::{fmoe, fn_moe, DevBuf, Stream};
use gb_exl3::checkpoint::LoadedGroup;
use gb_exl3::f16::f64_to_f16;
use gb_exl3::Codebook;
const E: usize = NE + 1;
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
fn fnv<T: Copy>(v: &[T]) -> u64 {
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) };
    b.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ u64::from(*x)).wrapping_mul(0x100000001b3))
}
struct Owned {
    keep: Vec<DevBuf>,
    tab: [DevBuf; 9],
    kt: DevBuf,
    gs: [Vec<LoadedGroup>; 3],
}
impl Owned {
    fn new() -> Self {
        Self::with_bits(bits_of)
    }
    /// The layer with `bits(matrix, expert)` bits a matrix instead of the including target's `bits_of` (the prefill bench's uniform 6).
    #[allow(dead_code)] // only the MiniCPM prefill target builds anything but `bits_of`
    fn with_bits(bits: impl Fn(usize, usize) -> usize) -> Self {
        let bits_of = bits;
        let gs = std::array::from_fn(|p| {
            (0..E)
                .map(|e| {
                    synth_group(
                        if p == 2 { I } else { H },
                        if p == 2 { H } else { I },
                        bits_of(p, e),
                        Codebook::Mul1,
                        0x90240 + (p * SEED_STRIDE + e) as u64,
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
                        1 => DevBuf::from_slice(&g.suh.iter().map(|x| f64_to_f16(*x)).collect::<Vec<_>>()).unwrap(),
                        _ => DevBuf::from_slice(&g.svh.iter().map(|x| f64_to_f16(*x)).collect::<Vec<_>>()).unwrap(),
                    };
                    ptrs.push(b.ptr() as u64);
                    keep.push(b);
                }
                tables.push(DevBuf::from_slice(&ptrs).unwrap());
            }
        }
        let tab: [DevBuf; 9] = tables.try_into().ok().unwrap();
        // ktab holds the EXL3 K2 (twice the bits)
        let kt: Vec<i32> = (0..E * 3).map(|i| (2 * bits_of(i % 3, i / 3)) as i32).collect();
        let kt = DevBuf::from_slice(&kt).unwrap();
        Owned { keep, tab, kt, gs }
    }
    fn tables(&self) -> fn_moe::ExpertTables<'_> {
        let a = &self.tab;
        fn_moe::ExpertTables {
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
}
fn inputs() -> Inputs {
    let mut r = Rng(0x129884);
    Inputs {
        x: (0..64 * H).map(|_| bf(r.uniform(-0.2, 0.2))).collect(),
        logits: (0..64 * NE).map(|_| bf(r.uniform(-2., 2.))).collect(),
        sg: (0..64).map(|_| bf(r.uniform(-1., 1.))).collect(),
    }
}
struct Out {
    sel: Vec<u32>,
    rw: Vec<f32>,
    coop: Vec<u16>,
    frost: Vec<u16>,
    prefill: Vec<u16>,
}
/// FrostMoE decode (`fmoe::forward`) with `kc` K chunks of the gate/up pass.
#[allow(dead_code)] // the Flash-Next target only runs `run`; the MiniCPM one sweeps the chunk counts
fn run_frost(o: &Owned, inp: &Inputs, t: usize, kc: usize, st: &Stream) -> Result<Vec<u16>, gb_cuda::CudaError> {
    let x = DevBuf::from_slice(&inp.x[..t * H]).unwrap();
    let logits = DevBuf::from_slice(&inp.logits[..t * NE]).unwrap();
    let sg = DevBuf::from_slice(&inp.sg[..t]).unwrap();
    let shape = fmoe::Shape { t, ne: NE, top_k: TK, h: H, i: I, kc, max_tokens: 64 };
    let mut frost = DevBuf::zeroed(t * H * 2).unwrap();
    let mut ws = fmoe::workspace(64, TK, H, I, kc)?;
    fmoe::forward(&x, &logits, &sg, &o.tables(), &mut frost, shape, None, fmoe::SwigluLimit::default(), &mut ws, st)?;
    Ok(dl(&frost, st))
}
fn run(o: &Owned, inp: &Inputs, t: usize, st: &Stream) -> Out {
    let x = DevBuf::from_slice(&inp.x[..t * H]).unwrap();
    let logits = DevBuf::from_slice(&inp.logits[..t * NE]).unwrap();
    let sg = DevBuf::from_slice(&inp.sg[..t]).unwrap();
    let mut sel = DevBuf::zeroed(t * (TK + 1) * 4).unwrap();
    let mut rw = DevBuf::zeroed(t * (TK + 1) * 4).unwrap();
    let mut coop = DevBuf::zeroed(t * H * 2).unwrap();
    fn_moe::moe_topk_softmax(&logits, Some(&sg), &mut sel, &mut rw, NE, TK, t, st).unwrap();
    let mut act = DevBuf::zeroed(fn_moe::coopmk_workspace_bytes(t, TK + 1, I)).unwrap();
    fn_moe::coopmk_decode(&x, &sel, &rw, &o.tables(), &mut act, &mut coop, H, I, TK + 1, t, st).unwrap();
    let shape = fmoe::Shape { t, ne: NE, top_k: TK, h: H, i: I, kc: KC, max_tokens: 64 };
    let mut frost = DevBuf::zeroed(t * H * 2).unwrap();
    let mut ws = fmoe::workspace(64, TK, H, I, KC).unwrap();
    fmoe::forward(&x, &logits, &sg, &o.tables(), &mut frost, shape, None, fmoe::SwigluLimit::default(), &mut ws, st).unwrap();
    let mut prefill = DevBuf::zeroed(t * H * 2).unwrap();
    let mut pws = fmoe::prefill_workspace(64, TK, NE, H, I).unwrap();
    fmoe::prefill(&x, &logits, &sg, &o.tables(), &mut prefill, shape, None, fmoe::SwigluLimit::default(), &mut pws, st).unwrap();
    Out { sel: dl(&sel, st), rw: dl(&rw, st), coop: dl(&coop, st), frost: dl(&frost, st), prefill: dl(&prefill, st) }
}
fn cpu_projection(g: &LoadedGroup, x: &[f64]) -> Vec<f64> {
    let cb = gb_exl3::codebook(g.group.codebook);
    gb_exl3::forward(x, 1, &g.trellis, g.kt, g.nt, g.group.bits, &cb, &g.suh, &g.svh, None)
}
fn cpu_row(o: &Owned, inp: &Inputs, sel: &[u32], rw: &[f32]) -> Vec<f32> {
    let x = inp.x[..H].iter().map(|v| bff(*v) as f64).collect::<Vec<_>>();
    let mut total = vec![0f32; H];
    for slot in 0..TK + 1 {
        let e = sel[slot] as usize;
        let gate = cpu_projection(&o.gs[0][e], &x);
        let up = cpu_projection(&o.gs[1][e], &x);
        let act = gate.iter().zip(up).map(|(g, u)| bff(bf((g / (1. + (-g).exp()) * u) as f32)) as f64).collect::<Vec<_>>();
        let down = cpu_projection(&o.gs[2][e], &act);
        for j in 0..H {
            total[j] += (down[j] as f32) * rw[slot];
        }
    }
    total
}
fn rel(got: &[u16], want: &[f32]) -> f64 {
    let (e, n) = got
        .iter()
        .zip(want)
        .fold((0f64, 0f64), |(e, n), (&a, &b)| (e + (f64::from(bff(a) - b)).powi(2), n + (f64::from(b)).powi(2)));
    (e / n.max(1e-30)).sqrt()
}
/// The router on the CPU: per row the top-`tk` of `ne` logits by (value DESC, index ASC), then the softmax over the selected values
/// (the shared expert, entry `tk`, is `ne` weighted by the sigmoid of its gate, which the caller checks).
fn cpu_route(logits: &[f32], tk: usize) -> (Vec<usize>, Vec<f64>) {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b)));
    let sel = idx[..tk].to_vec();
    let m = logits[sel[0]] as f64;
    let den: f64 = sel.iter().map(|&i| (logits[i] as f64 - m).exp()).sum();
    let w = sel.iter().map(|&i| (logits[i] as f64 - m).exp() / den).collect();
    (sel, w)
}
#[test]
#[ignore]
fn packed_experts_cpu_reference_and_row_invariance() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    let inp = inputs();
    let full = run(&o, &inp, 64, &st);
    let mut changed = inputs();
    changed.x[H..].fill(bf(0.75));
    changed.logits[NE..].fill(bf(-0.5));
    changed.sg[1..].fill(bf(0.25));
    changed.x[36 * H..37 * H].copy_from_slice(&inp.x[..H]);
    changed.logits[36 * NE..37 * NE].copy_from_slice(&inp.logits[..NE]);
    changed.sg[36] = inp.sg[0];
    let mixed = run(&o, &changed, 64, &st);
    assert_eq!(&full.sel[..TK + 1], &mixed.sel[..TK + 1], "router other-row invariance");
    assert_eq!(&full.sel[..TK + 1], &mixed.sel[36 * (TK + 1)..37 * (TK + 1)], "router position invariance");
    for (name, a, b) in
        [("CoopMK", &full.coop, &mixed.coop), ("FrostMoE", &full.frost, &mixed.frost), ("prefill", &full.prefill, &mixed.prefill)]
    {
        assert_eq!(&a[..H], &b[..H], "{name} other-row invariance");
        assert_eq!(&a[..H], &b[36 * H..37 * H], "{name} position invariance");
    }

    assert_eq!(o.keep.len(), E * 9);
    let mut worst = [0f64; 3];
    for t in [1, 7, 37, 64] {
        let g = run(&o, &inp, t, &st);
        assert_eq!(g.sel, &full.sel[..t * (TK + 1)], "router row invariance");
        assert_eq!(g.rw, &full.rw[..t * (TK + 1)], "router weights row invariance");
        assert_eq!(g.coop, &full.coop[..t * H], "CoopMK row invariance");
        assert_eq!(g.frost, &full.frost[..t * H], "FrostMoE row invariance");
        assert_eq!(g.prefill, &full.prefill[..t * H], "FrostMoE prefill row invariance");
        for row in 0..t {
            let l: Vec<f32> = (0..NE).map(|e| bff(inp.logits[row * NE + e])).collect();
            let (want_sel, want_w) = cpu_route(&l, TK);
            for k in 0..TK {
                assert_eq!(g.sel[row * (TK + 1) + k] as usize, want_sel[k], "CPU top-{} of row {row}", k + 1);
                assert!((g.rw[row * (TK + 1) + k] as f64 - want_w[k]).abs() < 5e-6, "router weight {k} of row {row}");
            }
            assert_eq!(g.sel[row * (TK + 1) + TK] as usize, NE, "CPU shared ID");
            assert!((g.rw[row * (TK + 1) + TK] - 1. / (1. + (-bff(inp.sg[row])).exp())).abs() < 1e-6);
        }
        let cpu = cpu_row(&o, &inp, &g.sel, &g.rw);
        for (i, (name, v, tol)) in
            [("CoopMK", &g.coop, 0.03), ("FrostMoE", &g.frost, 0.03), ("FrostMoE prefill", &g.prefill, 0.03)]
                .into_iter()
                .enumerate()
        {
            let err = rel(&v[..H], &cpu);
            worst[i] = worst[i].max(err);
            assert!(err < tol, "{name} vs packed EXL3 CPU reference: rel_l2={err}, tol={tol}");
        }
    }
    eprintln!("worst rel_l2 vs the CPU reference: CoopMK {:.3e}, FrostMoE {:.3e}, prefill {:.3e}", worst[0], worst[1], worst[2]);
    let g = &o.gs[0][0];
    let trellis = DevBuf::from_slice(&g.trellis).unwrap();
    let mut unpack = DevBuf::zeroed(H * I * 2).unwrap();
    fmoe::unpack(&trellis, &mut unpack, H, I, 2 * bits_of(0, 0), &st).unwrap();
    let cpu = gb_exl3::unpack(&g.trellis, g.kt, g.nt, g.group.bits, &gb_exl3::codebook(g.group.codebook));
    assert_eq!(dl::<u16>(&unpack, &st), cpu, "FrostMoE packed EXL3 decoder matches CPU unpack");
}
#[test]
#[ignore]
fn golden_fn_moe_hashes() {
    let Some(st) = gpu() else { return };
    let o = Owned::new();
    let v = run(&o, &inputs(), 37, &st);
    let lines = [
        format!("route {:016x}", fnv(&v.sel)),
        format!("weights {:016x}", fnv(&v.rw)),
        format!("coopmk {:016x}", fnv(&v.coop)),
        format!("frost {:016x}", fnv(&v.frost)),
        format!("prefill {:016x}", fnv(&v.prefill)),
    ];
    std::fs::write(std::env::var("REXL3_GOLDEN_OUT").unwrap(), format!("{}\n", lines.join("\n"))).unwrap();
}
