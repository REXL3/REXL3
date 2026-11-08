//! Mamba2 SSD kernels of the Nemotron 3.5 Super VL backbone (`csrc/nemotron_ssm.cu`), checked three ways:
//!
//! 1. Bit for bit: a verify window of T rows gives the same core output and gated output as T serial decode steps (a 1-row
//!    verify, then commit with keep = 1), and every commit(keep), keep in 0..=T, gives the pools the serial decode has after keep
//!    steps. The verify pass must leave the committed pools byte-identical. Multi-sequence batches with unequal lengths, non-identity
//!    slots and an unused slot are covered too.
//! 2. Against the CPU serial oracle below: the crates/model `ssm_ref` formula with the live path's bf16 rounding points (the crate
//!    cannot import `ssm_ref` directly: gb-model depends on gb-cuda). Checked within bf16 rounding (scale-normalized).
//! 3. Against the torch fixture `testdata/nemotron_omni/ssm/layer0_fixture.safetensors` (16 tokens, fp32 torch reference, see its
//!    README), within the bf16 rounding that the kernels apply. The fixture's safetensors header is read with a small JSON reader,
//!    because gb-cuda has no serde_json.
//!
//! The bf16 rounding points are the ones the .cu file documents. All reference math is fp32 on the host, with `mul_add` where the
//! kernel uses `__fmaf_rn`, so the oracle agrees with the kernels up to exp implementations and summation order.

#![cfg(not(gb_no_cuda))]

use gb_cuda::gdn::{Batch, BatchBuf};
use gb_cuda::nemotron_ssm::*;
use gb_cuda::num::{bf16_to_f32, bf16_to_f32_vec, f32_to_bf16};
use gb_cuda::{device_available, DevBuf, Stream};

macro_rules! need_gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

const H: usize = HEADS;
const N: usize = STATE;
const P: usize = HEAD_DIM;
const G: usize = GROUPS;
const HPG: usize = HEADS / GROUPS;
const CONV: usize = CONV_DIM;
const NG: usize = INNER / GROUPS;

/// Relative error bound for the oracle and the fixture, as a fraction of the reference's max magnitude. One bf16 ulp of an element is
/// 2^-8 of that element, so a flipped rounding at the largest element costs about 2^-8 of the scale. The bound allows a few of them.
const BF16_TOL: f64 = 2e-2;
/// The oracle agrees with the kernel up to exp implementations and summation order, so rounding flips are rare and tiny.
const ORACLE_TOL: f64 = 1e-2;

fn rb(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}

// ------------------------------------------------------------------------------------------------ deterministic inputs

fn splitmix64(z: u64) -> u64 {
    let z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let v = splitmix64(self.0);
        self.0 = self.0.wrapping_add(1);
        v
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let k = (self.next_u64() >> 40) as f32 / 16_777_216.0;
        lo + (hi - lo) * k
    }
    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform(lo, hi)).collect()
    }
}

/// The per-layer parameters on the host. The bf16 ones hold bf16-exact values; the f32 ones are f32.
#[derive(Clone)]
struct Params {
    conv_w: Vec<f32>, // [10240][4]
    conv_b: Vec<f32>, // [10240]
    a_log: Vec<f32>,  // [128]
    dt_bias: Vec<f32>,
    d: Vec<f32>,
    norm_w: Vec<f32>, // [8192]
}

impl Params {
    fn random(rng: &mut Rng) -> Self {
        let bf = |v: Vec<f32>| -> Vec<f32> { v.iter().map(|&x| rb(x)).collect() };
        Self {
            conv_w: bf(rng.vec(CONV * 4, -0.5, 0.5)),
            conv_b: bf(rng.vec(CONV, -0.5, 0.5)),
            a_log: rng.vec(H, -1.0, 1.0),
            dt_bias: rng.vec(H, -1.0, 1.0),
            d: rng.vec(H, -1.0, 1.0),
            norm_w: rng.vec(INNER, 0.5, 1.5),
        }
    }
}

/// Rows of the post-in_proj streams: xBC f32 `[n][10240]`, dt_raw f32 `[n][128]`, z f32 `[n][8192]`.
struct Rows {
    xbc: Vec<f32>,
    dt: Vec<f32>,
    z: Vec<f32>,
}

impl Rows {
    fn random(rng: &mut Rng, n: usize) -> Self {
        Self { xbc: rng.vec(n * CONV, -2.0, 2.0), dt: rng.vec(n * H, -2.0, 2.0), z: rng.vec(n * INNER, -2.0, 2.0) }
    }
    fn xbc(&self, r: usize) -> &[f32] {
        &self.xbc[r * CONV..(r + 1) * CONV]
    }
    fn dt(&self, r: usize) -> &[f32] {
        &self.dt[r * H..(r + 1) * H]
    }
    fn z(&self, r: usize) -> &[f32] {
        &self.z[r * INNER..(r + 1) * INNER]
    }
}

// ------------------------------------------------------------------------------------------------ CPU serial oracle

/// The host state of one sequence: S `[128 heads][128 state][64 head_dim]` and the conv history `[3][10240]` (bf16 values, oldest first).
#[derive(Clone)]
struct HostState {
    s: Vec<f32>,
    hist: Vec<f32>,
}

impl HostState {
    fn zero() -> Self {
        Self { s: vec![0.0; H * N * P], hist: vec![0.0; 3 * CONV] }
    }
}

/// One serial row of the reference, with the live path's rounding points (see the .cu header). Returns the core output (bf16 values)
/// and the gated output (bf16 values), each `[8192]`.
fn oracle_row(p: &Params, st: &mut HostState, xbc: &[f32], dt_raw: &[f32], z: &[f32], round: bool) -> (Vec<f32>, Vec<f32>) {
    // `q` is the bf16 rounding point of the live path; with `round` off the same formula runs in pure fp32 (the torch fixture's arithmetic).
    let q = |v: f32| if round { rb(v) } else { v };
    let x: Vec<f32> = xbc.iter().map(|&v| q(v)).collect();
    let mut conv = vec![0.0f32; CONV];
    for c in 0..CONV {
        let w = &p.conv_w[c * 4..c * 4 + 4];
        let mut acc = p.conv_b[c];
        acc = w[0].mul_add(st.hist[c], acc);
        acc = w[1].mul_add(st.hist[CONV + c], acc);
        acc = w[2].mul_add(st.hist[2 * CONV + c], acc);
        acc = w[3].mul_add(x[c], acc);
        conv[c] = q(acc / (1.0 + (-acc).exp()));
    }
    st.hist.copy_within(CONV..3 * CONV, 0);
    st.hist[2 * CONV..].copy_from_slice(&x);

    let mut dt_bf = vec![0.0f32; H];
    let mut decay = vec![0.0f32; H];
    for h in 0..H {
        let av = dt_raw[h] + p.dt_bias[h];
        let sp = if av > 20.0 { av } else { av.exp().ln_1p() };
        let dt = if sp < 0.0 { 0.0 } else { sp };
        let g = -p.a_log[h].exp() * dt;
        dt_bf[h] = q(dt);
        decay[h] = g.exp();
    }

    let mut o = vec![0.0f32; INNER];
    for h in 0..H {
        let grp = h / HPG;
        let bvec = &conv[INNER + grp * N..INNER + grp * N + N];
        let cvec = &conv[INNER + BC + grp * N..INNER + BC + grp * N + N];
        for j in 0..P {
            let xj = conv[h * P + j];
            let xdt = xj * dt_bf[h];
            let mut a = [0.0f32; 4];
            for i in 0..N {
                let idx = (h * N + i) * P + j;
                let kept = st.s[idx] * decay[h];
                let inj = bvec[i] * xdt;
                let sv = kept + inj;
                st.s[idx] = sv;
                a[i % 4] += cvec[i] * sv;
            }
            let y = ((a[0] + a[1]) + (a[2] + a[3])) + p.d[h] * xj;
            o[h * P + j] = q(y);
        }
    }

    let mut out = vec![0.0f32; INNER];
    for grp in 0..G {
        let base = grp * NG;
        let mut v = vec![0.0f32; NG];
        let mut sq = 0.0f32;
        for k in 0..NG {
            let zb = q(z[base + k]);
            v[k] = o[base + k] * (zb / (1.0 + (-zb).exp()));
            sq += v[k] * v[k];
        }
        let inv = 1.0 / (sq / NG as f32 + NORM_EPS).sqrt();
        for k in 0..NG {
            out[base + k] = q((v[k] * p.norm_w[base + k]) * inv);
        }
    }
    (o, out)
}

// ------------------------------------------------------------------------------------------------ device helpers

fn upload_bf16(v: &[f32]) -> DevBuf {
    let u: Vec<u16> = v.iter().map(|&x| f32_to_bf16(x)).collect();
    DevBuf::from_slice(&u).unwrap()
}

fn upload_f32(v: &[f32]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}

/// The parameters on the device, in the layouts the kernels take.
struct GpuParams {
    conv_w: DevBuf,
    conv_b: DevBuf,
    a_log: DevBuf,
    dt_bias: DevBuf,
    d: DevBuf,
    norm_w: DevBuf,
}

impl GpuParams {
    fn new(p: &Params) -> Self {
        Self {
            conv_w: upload_bf16(&p.conv_w),
            conv_b: upload_bf16(&p.conv_b),
            a_log: upload_f32(&p.a_log),
            dt_bias: upload_f32(&p.dt_bias),
            d: upload_f32(&p.d),
            norm_w: upload_f32(&p.norm_w),
        }
    }
    fn weights(&self) -> Weights<'_> {
        Weights { conv_w: &self.conv_w, conv_b: &self.conv_b, a_log: &self.a_log, dt_bias: &self.dt_bias, d: &self.d }
    }
}

struct Ctx {
    st: Stream,
    gp: GpuParams,
    host: Params,
}

impl Ctx {
    fn new(host: Params) -> Self {
        Self { st: Stream::new().unwrap(), gp: GpuParams::new(&host), host }
    }
}

/// Both pools of a layer, with `n_slots` slots.
struct Pools {
    s: DevBuf,
    conv: DevBuf,
}

impl Pools {
    fn zeros(n_slots: usize) -> Self {
        Self {
            s: DevBuf::zeroed(n_slots * S_SLOT_ELEMS * 4).unwrap(),
            conv: DevBuf::zeroed(n_slots * CONV_SLOT_ELEMS * 2).unwrap(),
        }
    }
}

/// The bytes of both pools.
type Snap = (Vec<u8>, Vec<u8>);

fn snap(p: &Pools) -> Snap {
    (p.s.to_vec::<u8>().unwrap(), p.conv.to_vec::<u8>().unwrap())
}

fn restore(s: &Snap) -> Pools {
    Pools { s: DevBuf::from_slice(&s.0).unwrap(), conv: DevBuf::from_slice(&s.1).unwrap() }
}

/// Verify: `pre` then `window`. Reads the pools and writes the record and the core output only.
fn verify(ctx: &Ctx, pools: &Pools, rec: &mut Record, batch: &Batch, xb: &DevBuf, db: &DevBuf, o: &mut DevBuf) {
    let w = ctx.gp.weights();
    pre(&ctx.st, xb, db, &w, &pools.conv, batch, rec).unwrap();
    window(&ctx.st, rec, &pools.s, &ctx.gp.d, batch, o).unwrap();
}

/// Commit: the state half and the conv half, both with the same keep.
fn commit(ctx: &Ctx, pools: &mut Pools, rec: &Record, batch: &Batch, xb: &DevBuf, keep: &DevBuf) {
    commit_state(&ctx.st, rec, &mut pools.s, batch, keep).unwrap();
    commit_conv(&ctx.st, xb, &mut pools.conv, batch, keep).unwrap();
}

/// One serial decode step of a sequence in `slot`: a 1-row verify, the gated norm, then commit(keep = 1). Returns the core output
/// and the gated output of the row, as bf16 bits.
fn decode_step(
    ctx: &Ctx,
    pools: &mut Pools,
    n_slots: usize,
    slot: usize,
    xbc: &[f32],
    dt: &[f32],
    z: &[f32],
) -> (Vec<u16>, Vec<u16>) {
    let bb = BatchBuf::with_pool_slots(&[1], &[slot], n_slots).unwrap();
    let batch = bb.view();
    let xb = upload_f32(xbc);
    let db = upload_f32(dt);
    let zb = upload_f32(z);
    let mut rec = Record::alloc(1).unwrap();
    let mut o = DevBuf::zeroed(INNER * 2).unwrap();
    let mut out = DevBuf::zeroed(INNER * 2).unwrap();
    verify(ctx, pools, &mut rec, &batch, &xb, &db, &mut o);
    gated_norm(&ctx.st, &o, &zb, &ctx.gp.norm_w, 1, &mut out).unwrap();
    let keep = DevBuf::from_slice(&[1i32]).unwrap();
    commit(ctx, pools, &rec, &batch, &xb, &keep);
    ctx.st.sync().unwrap();
    (o.to_vec::<u16>().unwrap(), out.to_vec::<u16>().unwrap())
}

// ------------------------------------------------------------------------------------------------ comparisons

fn first_diff(a: &[u16], b: &[u16]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y).or(if a.len() != b.len() { Some(a.len().min(b.len())) } else { None })
}

/// max |got - want| / max |want|.
fn scale_err(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, v| m.max((*v as f64).abs()));
    let mut e = 0.0f64;
    for (a, b) in got.iter().zip(want) {
        e = e.max(((*a - *b) as f64).abs());
    }
    e / scale
}

// ------------------------------------------------------------------------------------------------ the window cases

/// Checks one window of `t_len` rows after `prefix` serial rows of a single sequence in slot 0.
/// Returns the scale-normalized errors of the core output and the gated output against the CPU oracle.
fn check_window_case(ctx: &Ctx, prefix: usize, t_len: usize, seed: u64) -> (f64, f64) {
    let n = prefix + t_len;
    let rows = Rows::random(&mut Rng(seed), n);

    // The CPU oracle over every row; the window's rows are the last t_len.
    let mut hs = HostState::zero();
    let mut o_ref = Vec::new();
    let mut out_ref = Vec::new();
    for r in 0..n {
        let (o, out) = oracle_row(&ctx.host, &mut hs, rows.xbc(r), rows.dt(r), rows.z(r), true);
        if r >= prefix {
            o_ref.extend(o);
            out_ref.extend(out);
        }
    }

    // A committed prefix, built with serial decode steps on the device.
    let mut base = Pools::zeros(1);
    for r in 0..prefix {
        decode_step(ctx, &mut base, 1, 0, rows.xbc(r), rows.dt(r), rows.z(r));
    }
    let base_snap = snap(&base);

    // The verify window: it must not write the committed pools.
    let bb = BatchBuf::with_pool_slots(&[t_len], &[0], 1).unwrap();
    let batch = bb.view();
    let win = &rows.xbc[prefix * CONV..];
    let xb = upload_f32(win);
    let db = upload_f32(&rows.dt[prefix * H..]);
    let zb = upload_f32(&rows.z[prefix * INNER..]);
    let pools = restore(&base_snap);
    let mut rec = Record::alloc(t_len).unwrap();
    let mut o = DevBuf::zeroed(t_len * INNER * 2).unwrap();
    let mut out = DevBuf::zeroed(t_len * INNER * 2).unwrap();
    verify(ctx, &pools, &mut rec, &batch, &xb, &db, &mut o);
    gated_norm(&ctx.st, &o, &zb, &ctx.gp.norm_w, t_len, &mut out).unwrap();
    ctx.st.sync().unwrap();
    assert!(snap(&pools) == base_snap, "T={t_len}: the verify window wrote the committed pools");
    let o_win: Vec<u16> = o.to_vec().unwrap();
    let out_win: Vec<u16> = out.to_vec().unwrap();

    // The serial decode chain, and commit(keep) for every keep in 0..=T.
    let mut chain = restore(&base_snap);
    let mut serial_o: Vec<u16> = Vec::new();
    let mut serial_out: Vec<u16> = Vec::new();
    for keep in 0..=t_len {
        let mut q = restore(&base_snap);
        let mut rec_q = Record::alloc(t_len).unwrap();
        let mut oq = DevBuf::zeroed(t_len * INNER * 2).unwrap();
        verify(ctx, &q, &mut rec_q, &batch, &xb, &db, &mut oq);
        let kb = DevBuf::from_slice(&[keep as i32]).unwrap();
        commit(ctx, &mut q, &rec_q, &batch, &xb, &kb);
        ctx.st.sync().unwrap();
        let got = snap(&q);
        let want = snap(&chain);
        assert!(got.1 == want.1, "T={t_len} keep={keep}: the conv history differs from serial decode");
        assert!(got.0 == want.0, "T={t_len} keep={keep}: the S state differs from serial decode");
        if keep == 0 {
            assert!(got == base_snap, "T={t_len}: keep = 0 must leave the state untouched");
        }
        if keep < t_len {
            let r = prefix + keep;
            let (o1, out1) = decode_step(ctx, &mut chain, 1, 0, rows.xbc(r), rows.dt(r), rows.z(r));
            serial_o.extend(o1);
            serial_out.extend(out1);
        }
    }

    // The window's outputs equal the serial decode's, bit for bit.
    if let Some(i) = first_diff(&o_win, &serial_o) {
        panic!("T={t_len}: core output differs from serial decode at element {i}");
    }
    if let Some(i) = first_diff(&out_win, &serial_out) {
        panic!("T={t_len}: gated output differs from serial decode at element {i}");
    }

    let e_o = scale_err(&bf16_to_f32_vec(&o_win), &o_ref);
    let e_out = scale_err(&bf16_to_f32_vec(&out_win), &out_ref);
    (e_o, e_out)
}

#[test]
fn window_equals_serial_decode_and_every_commit_matches() {
    need_gpu!();
    let ctx = Ctx::new(Params::random(&mut Rng(7)));
    // 128 and 256 are the widths of a prefill slice once the large-row MoE prefill is on: the same kernels, the same bits
    for (k, &t) in [1usize, 2, 3, 5, 16, 33, 64, 128, 256].iter().enumerate() {
        let (e_o, e_out) = check_window_case(&ctx, 5, t, 1000 + k as u64);
        eprintln!("T={t:>2}: bits == serial decode; vs CPU oracle: core {e_o:.3e}, gated {e_out:.3e} (scale-normalized)");
        assert!(e_o <= ORACLE_TOL && e_out <= ORACLE_TOL, "T={t}: oracle error {e_o:e} / {e_out:e} over {ORACLE_TOL:e}");
    }
}

#[test]
fn window_from_zero_state_and_a_single_row_match_too() {
    need_gpu!();
    let ctx = Ctx::new(Params::random(&mut Rng(11)));
    for &t in &[1usize, 64] {
        let (e_o, e_out) = check_window_case(&ctx, 0, t, 2000 + t as u64);
        eprintln!("zero prefix, T={t}: core {e_o:.3e}, gated {e_out:.3e}");
        assert!(e_o <= ORACLE_TOL && e_out <= ORACLE_TOL);
    }
}

/// Several sequences in one launch: unequal lengths, non-identity slots, and an unused slot that must stay zero.
#[test]
fn multi_sequence_batch_matches_per_sequence_serial_decode() {
    need_gpu!();
    let ctx = Ctx::new(Params::random(&mut Rng(21)));
    let n_slots = 4usize;
    let lens = [3usize, 7, 1];
    let slots = [2usize, 0, 3];
    let keeps = [2i32, 0, 1];
    let prefix = 3usize;

    let rows: Vec<Rows> = (0..lens.len()).map(|k| Rows::random(&mut Rng(3000 + k as u64), prefix + lens[k])).collect();

    // Oracle per sequence.
    let mut o_ref = Vec::new();
    let mut out_ref = Vec::new();
    for k in 0..lens.len() {
        let mut hs = HostState::zero();
        for r in 0..prefix + lens[k] {
            let (o, out) = oracle_row(&ctx.host, &mut hs, rows[k].xbc(r), rows[k].dt(r), rows[k].z(r), true);
            if r >= prefix {
                o_ref.extend(o);
                out_ref.extend(out);
            }
        }
    }

    // Committed prefix of each sequence in its slot.
    let mut base = Pools::zeros(n_slots);
    for k in 0..lens.len() {
        for r in 0..prefix {
            decode_step(&ctx, &mut base, n_slots, slots[k], rows[k].xbc(r), rows[k].dt(r), rows[k].z(r));
        }
    }
    let base_snap = snap(&base);

    // Serial reference: each sequence's window rows decoded one by one in its slot, and the expected state after keep steps.
    let mut serial_o: Vec<u16> = Vec::new();
    let mut serial_out: Vec<u16> = Vec::new();
    let mut chain = restore(&base_snap);
    for k in 0..lens.len() {
        for t in 0..lens[k] {
            let r = prefix + t;
            let (o1, out1) = decode_step(&ctx, &mut chain, n_slots, slots[k], rows[k].xbc(r), rows[k].dt(r), rows[k].z(r));
            serial_o.extend(o1);
            serial_out.extend(out1);
        }
    }
    let mut expect = restore(&base_snap);
    for k in 0..lens.len() {
        for t in 0..keeps[k] as usize {
            let r = prefix + t;
            decode_step(&ctx, &mut expect, n_slots, slots[k], rows[k].xbc(r), rows[k].dt(r), rows[k].z(r));
        }
    }

    // The batch.
    let total: usize = lens.iter().sum();
    let mut xbc = Vec::new();
    let mut dt = Vec::new();
    let mut z = Vec::new();
    for row in &rows {
        xbc.extend_from_slice(&row.xbc[prefix * CONV..]);
        dt.extend_from_slice(&row.dt[prefix * H..]);
        z.extend_from_slice(&row.z[prefix * INNER..]);
    }
    let bb = BatchBuf::with_pool_slots(&lens, &slots, n_slots).unwrap();
    let batch = bb.view();
    let xb = upload_f32(&xbc);
    let db = upload_f32(&dt);
    let zb = upload_f32(&z);
    let mut pools = restore(&base_snap);
    let mut rec = Record::alloc(total).unwrap();
    let mut o = DevBuf::zeroed(total * INNER * 2).unwrap();
    let mut out = DevBuf::zeroed(total * INNER * 2).unwrap();
    verify(&ctx, &pools, &mut rec, &batch, &xb, &db, &mut o);
    gated_norm(&ctx.st, &o, &zb, &ctx.gp.norm_w, total, &mut out).unwrap();
    let kb = upload_i32(&keeps);
    ctx.st.sync().unwrap();
    assert!(snap(&pools) == base_snap, "the batched verify wrote the committed pools");
    let o_win: Vec<u16> = o.to_vec().unwrap();
    let out_win: Vec<u16> = out.to_vec().unwrap();
    if let Some(i) = first_diff(&o_win, &serial_o) {
        panic!("batch: core output differs from serial decode at element {i}");
    }
    if let Some(i) = first_diff(&out_win, &serial_out) {
        panic!("batch: gated output differs from serial decode at element {i}");
    }

    commit(&ctx, &mut pools, &rec, &batch, &xb, &kb);
    ctx.st.sync().unwrap();
    let got = snap(&pools);
    let want = snap(&expect);
    assert!(got.1 == want.1, "batch commit: conv history differs from serial decode");
    assert!(got.0 == want.0, "batch commit: S state differs from serial decode");
    // Slot 1 is unused by every sequence: it must still be zero.
    assert!(got.0[S_SLOT_ELEMS * 4..2 * S_SLOT_ELEMS * 4].iter().all(|&b| b == 0), "the unused slot was written");

    let e_o = scale_err(&bf16_to_f32_vec(&o_win), &o_ref);
    let e_out = scale_err(&bf16_to_f32_vec(&out_win), &out_ref);
    eprintln!(
        "batch {lens:?} slots {slots:?} keep {keeps:?}: bits == serial decode; vs CPU oracle: core {e_o:.3e}, gated {e_out:.3e}"
    );
    assert!(e_o <= ORACLE_TOL && e_out <= ORACLE_TOL);
}

fn upload_i32(v: &[i32]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}

// ------------------------------------------------------------------------------------------------ the torch fixture

/// The generator of `tools/gen_ssm_fixture.py`: element `idx` of stream `stream` is `k / 2^23 - 1` with `k` the top 24 bits of
/// splitmix64(SEED + stream << 48 + idx). Returns the values and the sum of the `k`s (the fixture's checksum).
fn gen_stream(stream: u64, rows: usize, cols: usize) -> (Vec<f32>, i64) {
    const SEED: u64 = 20261010;
    let mut out = Vec::with_capacity(rows * cols);
    let mut sum = 0i64;
    for idx in 0..(rows * cols) as u64 {
        let k = splitmix64(SEED.wrapping_add(stream << 48).wrapping_add(idx)) >> 40;
        sum += k as i64;
        out.push((k as f32 - 8_388_608.0) / 8_388_608.0);
    }
    (out, sum)
}

// A minimal JSON reader for the safetensors header (objects, arrays, strings, numbers; literals are skipped).
#[derive(Debug)]
enum Json {
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
    Other,
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn nums(&self) -> Vec<usize> {
        match self {
            Json::Arr(a) => a
                .iter()
                .map(|v| match v {
                    Json::Num(n) => *n as usize,
                    other => panic!("expected a number, got {other:?}"),
                })
                .collect(),
            other => panic!("expected an array, got {other:?}"),
        }
    }
}

struct JsonP<'a> {
    b: &'a [u8],
    i: usize,
}

impl JsonP<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn string(&mut self) -> String {
        assert_eq!(self.b[self.i], b'"', "json: expected a string");
        self.i += 1;
        let mut s = Vec::new();
        while self.b[self.i] != b'"' {
            if self.b[self.i] == b'\\' {
                self.i += 1;
            }
            s.push(self.b[self.i]);
            self.i += 1;
        }
        self.i += 1;
        String::from_utf8(s).unwrap()
    }
    fn value(&mut self) -> Json {
        self.ws();
        match self.b[self.i] {
            b'{' => {
                self.i += 1;
                let mut m = Vec::new();
                loop {
                    self.ws();
                    if self.b[self.i] == b'}' {
                        self.i += 1;
                        break;
                    }
                    let k = self.string();
                    self.ws();
                    assert_eq!(self.b[self.i], b':');
                    self.i += 1;
                    let v = self.value();
                    m.push((k, v));
                    self.ws();
                    match self.b[self.i] {
                        b',' => self.i += 1,
                        b'}' => {
                            self.i += 1;
                            break;
                        }
                        c => panic!("json: unexpected {}", c as char),
                    }
                }
                Json::Obj(m)
            }
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                loop {
                    self.ws();
                    if self.b[self.i] == b']' {
                        self.i += 1;
                        break;
                    }
                    a.push(self.value());
                    self.ws();
                    match self.b[self.i] {
                        b',' => self.i += 1,
                        b']' => {
                            self.i += 1;
                            break;
                        }
                        c => panic!("json: unexpected {}", c as char),
                    }
                }
                Json::Arr(a)
            }
            b'"' => Json::Str(self.string()),
            b't' | b'f' | b'n' => {
                while self.b[self.i].is_ascii_alphabetic() {
                    self.i += 1;
                }
                Json::Other
            }
            _ => {
                let st = self.i;
                while self.i < self.b.len() && (self.b[self.i].is_ascii_digit() || b"+-.eE".contains(&self.b[self.i])) {
                    self.i += 1;
                }
                Json::Num(std::str::from_utf8(&self.b[st..self.i]).unwrap().parse().expect("json number"))
            }
        }
    }
}

struct Fixture {
    bytes: Vec<u8>,
    header: Json,
    data_start: usize,
}

impl Fixture {
    fn load() -> Self {
        let path = repo_root().join("testdata/nemotron_omni/ssm/layer0_fixture.safetensors");
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header = JsonP { b: &bytes[8..8 + n], i: 0 }.value();
        Fixture { bytes, header, data_start: 8 + n }
    }

    /// The raw bytes of one tensor, after checking its dtype and shape.
    fn raw(&self, name: &str, dtype: &str, shape: &[usize]) -> &[u8] {
        let e = self.header.get(name).unwrap_or_else(|| panic!("fixture has no tensor `{name}`"));
        match e.get("dtype") {
            Some(Json::Str(d)) => assert_eq!(d, dtype, "{name} dtype"),
            _ => panic!("{name}: no dtype"),
        }
        assert_eq!(e.get("shape").unwrap().nums(), shape, "{name} shape");
        let off = e.get("data_offsets").unwrap().nums();
        &self.bytes[self.data_start + off[0]..self.data_start + off[1]]
    }

    fn f32(&self, name: &str, shape: &[usize]) -> Vec<f32> {
        self.raw(name, "F32", shape).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn bf16(&self, name: &str, shape: &[usize]) -> Vec<f32> {
        self.raw(name, "BF16", shape)
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    }

    fn i64(&self, name: &str, shape: &[usize]) -> Vec<i64> {
        self.raw(name, "I64", shape).chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
    }
}

#[test]
fn layer0_fixture_matches_torch_within_bf16_rounding() {
    need_gpu!();
    let fx = Fixture::load();
    const T: usize = 16;
    let host = Params {
        conv_w: fx.bf16("conv1d.weight", &[CONV, 1, 4]),
        conv_b: fx.bf16("conv1d.bias", &[CONV]),
        a_log: fx.bf16("A_log", &[H]),
        dt_bias: fx.bf16("dt_bias", &[H]),
        d: fx.bf16("D", &[H]),
        norm_w: fx.bf16("norm.weight", &[INNER]),
    };
    let (z, sum_z) = gen_stream(0, T, INNER);
    let (xbc, sum_xbc) = gen_stream(1, T, CONV);
    let (dt, sum_dt) = gen_stream(2, T, H);
    assert_eq!(
        fx.i64("input_checksum", &[3]),
        vec![sum_z, sum_xbc, sum_dt],
        "the input generator drifted from the torch generator"
    );

    let ctx = Ctx::new(host);
    let bb = BatchBuf::with_pool_slots(&[T], &[0], 1).unwrap();
    let batch = bb.view();
    let xb = upload_f32(&xbc);
    let db = upload_f32(&dt);
    let zb = upload_f32(&z);
    let mut pools = Pools::zeros(1);
    let mut rec = Record::alloc(T).unwrap();
    let mut o = DevBuf::zeroed(T * INNER * 2).unwrap();
    let mut out = DevBuf::zeroed(T * INNER * 2).unwrap();
    verify(&ctx, &pools, &mut rec, &batch, &xb, &db, &mut o);
    gated_norm(&ctx.st, &o, &zb, &ctx.gp.norm_w, T, &mut out).unwrap();
    let keep = upload_i32(&[T as i32]);
    commit(&ctx, &mut pools, &rec, &batch, &xb, &keep);
    ctx.st.sync().unwrap();

    let conv_last = bf16_to_f32_vec(&rec.xbc.to_vec::<u16>().unwrap()[(T - 1) * CONV..T * CONV]);
    let dt_last = bf16_to_f32_vec(&rec.dt.to_vec::<u16>().unwrap()[(T - 1) * H..T * H]);
    let g_all: Vec<f32> = rec.g.to_vec().unwrap();
    let g_last = &g_all[(T - 1) * H..T * H];
    let gated: Vec<f32> = bf16_to_f32_vec(&out.to_vec::<u16>().unwrap());
    let s_all: Vec<f32> = pools.s.to_vec().unwrap();
    let state_heads = &s_all[..2 * N * P];

    let e_conv = scale_err(&conv_last, &fx.f32("conv_last", &[CONV]));
    let e_dt = scale_err(&dt_last, &fx.f32("dt_last", &[H]));
    let e_g = scale_err(g_last, &fx.f32("g_last", &[H]));
    let e_gated = scale_err(&gated, &fx.f32("gated_ref", &[T, INNER]));
    let e_state = scale_err(state_heads, &fx.f32("state_ref_heads", &[2, N, P]));
    eprintln!("fixture (scale-normalized max error, max |torch| as the scale):");
    eprintln!("  conv_last (silu conv, row 15, bf16 output)     {e_conv:.3e}");
    eprintln!("  dt_last   (bf16 dt)                            {e_dt:.3e}");
    eprintln!("  g_last    (fp32 g)                             {e_g:.3e}");
    eprintln!("  gated_ref (16 x 8192, bf16 output)             {e_gated:.3e}");
    eprintln!("  state_ref_heads 0-1 (fp32 state after 16 rows) {e_state:.3e}");

    // Where the gap comes from. The same formula with no rounding points, on the same fixture inputs, must match torch to the
    // README's 1e-4 gate (it is the same arithmetic). The rounding oracle is what the kernel implements, so the kernel must match it
    // to the oracle's own tolerance. The remaining gap to torch is the bf16 rounding the semantics require.
    let mut hs32 = HostState::zero();
    let mut hs_r = HostState::zero();
    let mut gated32 = Vec::with_capacity(T * INNER);
    let mut gated_r = Vec::with_capacity(T * INNER);
    for r in 0..T {
        let xr = &xbc[r * CONV..(r + 1) * CONV];
        let dr = &dt[r * H..(r + 1) * H];
        let zr = &z[r * INNER..(r + 1) * INNER];
        gated32.extend(oracle_row(&ctx.host, &mut hs32, xr, dr, zr, false).1);
        gated_r.extend(oracle_row(&ctx.host, &mut hs_r, xr, dr, zr, true).1);
    }
    let e_fp32 = scale_err(&gated32, &fx.f32("gated_ref", &[T, INNER]));
    let e_fp32_state = scale_err(&hs32.s[..2 * N * P], &fx.f32("state_ref_heads", &[2, N, P]));
    let e_kernel_vs_oracle = scale_err(&gated, &gated_r);
    eprintln!("  CPU oracle, pure fp32 (no rounding points) vs torch: gated {e_fp32:.3e}, state {e_fp32_state:.3e}");
    eprintln!("  kernel vs CPU oracle with the rounding points: gated {e_kernel_vs_oracle:.3e}");

    assert!(e_g <= 1e-4, "g is computed in fp32 from the same inputs: {e_g:e}");
    assert!(e_fp32 <= 1e-4 && e_fp32_state <= 1e-4, "the fp32 oracle must match torch: {e_fp32:e} {e_fp32_state:e}");
    assert!(e_kernel_vs_oracle <= ORACLE_TOL, "kernel vs rounding oracle: {e_kernel_vs_oracle:e}");
    assert!(e_conv <= BF16_TOL && e_dt <= BF16_TOL, "conv or dt outside bf16 rounding: {e_conv:e} {e_dt:e}");
    assert!(e_gated <= BF16_TOL, "gated output outside bf16 rounding: {e_gated:e}");
    assert!(e_state <= BF16_TOL, "state outside bf16 rounding: {e_state:e}");
}

/// The repository root: `REXL3_REPO_ROOT` when set (a prebuilt test binary run from an installed bin directory that carries the
/// tree's `testdata/`, e.g. on a node without this checkout), else the checkout this binary was built from.
fn repo_root() -> std::path::PathBuf {
    std::env::var_os("REXL3_REPO_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}
