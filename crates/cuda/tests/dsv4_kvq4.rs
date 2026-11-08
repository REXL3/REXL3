//! The q4 KV cache (`src/dsv4_kvq4.rs`, `csrc/dsv4_kvq4.cu`): the layout and its error bounds on the host, the read
//! kernels against their host twins on the device, the indexer's selection agreement, and the decode microbenchmark.
//!
//! Everything but the `gpu_*` tests runs without a GPU; those print `not run: no device` and return.
use gb_cuda::dsv4_attn::{compress_slots, compress_slots_max, topk_indices, Dsv4Config, Dsv4Planes};
use gb_cuda::dsv4_kvq4::*;
use gb_cuda::{device_available, DevBuf, Stream};

// ------------------------------------------------------------------------------------------------ helpers

/// xorshift64*: the same test data everywhere, no dependency.
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
    /// Box-Muller, in f64 so the tails are clean.
    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform(lo, hi)).collect()
    }
    fn gauss(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

/// `n` rows of `hd` gaussian values, RMS-normalised (what the plane's writer hands the store kernel). `heavy` scales
/// eight dims of every row by 6..13x first: a heavy-tailed latent, which is what a projection off a real activation
/// looks like and what an isotropic sample does not.
fn latents(r: &mut Rng, n: usize, hd: usize, heavy: bool) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n * hd).map(|_| r.gauss()).collect();
    if heavy {
        for row in 0..n {
            for k in 0..8 {
                let d = (k * 61 + row * 7) % hd;
                v[row * hd + d] *= 6.0 + k as f32;
            }
        }
    }
    for row in 0..n {
        let s: f32 = v[row * hd..][..hd].iter().map(|x| x * x).sum::<f32>() / hd as f32;
        let inv = 1.0 / s.sqrt();
        for x in v[row * hd..][..hd].iter_mut() {
            *x *= inv;
        }
    }
    v
}

/// The plane's stored values for `rows`: `rotate` only picks the encoding, so the values come back in the rotated domain.
fn q4_plane(rows: &[f32], hd: usize, rotate: bool) -> Vec<f32> {
    let n = rows.len() / hd;
    let mut out = vec![0f32; rows.len()];
    let mut buf = vec![0u8; kvq4_row_bytes(hd)];
    for j in 0..n {
        let (nope, rope) = rows[j * hd..][..hd].split_at(hd - ROPE_DIMS);
        pack_q4_row(nope, rope, rotate, &mut buf);
        unpack_q4_row(&buf, hd - ROPE_DIMS, false, &mut out[j * hd..][..hd]);
    }
    out
}

/// `v` rounded through f16 and uploaded as the kernel's element type: every activation plane is f16, so a device test
/// must hand the kernel f16 values or it reads an f32 bit pattern as two halves.
fn up(v: &[f32]) -> DevBuf {
    let h: Vec<u16> = v.iter().map(|&x| half_bits(x)).collect();
    DevBuf::from_slice(&h).unwrap()
}

/// A f16 plane read back as f32.
fn read16(b: &DevBuf) -> Vec<f32> {
    let h: Vec<u16> = b.to_vec().unwrap();
    h.iter().map(|&x| half_value(x)).collect()
}

/// `v` as the kernel holds it (f16), back in f32.
fn down(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| half_value(half_bits(x))).collect()
}

/// Undo the layout's rotation on every row of a 512-wide plane.
fn unrotate(rows: &mut [f32], hd: usize) {
    for row in rows.chunks_mut(hd) {
        hadamard64(&mut row[..hd - ROPE_DIMS]);
    }
}

/// The fp16 plane as the device stores it today: block-16 E2M1, non-power-of-two scale, then f16.
fn f16_plane(rows: &[f32], hd: usize) -> Vec<f32> {
    let mut v = rows.to_vec();
    for r in v.chunks_mut(hd) {
        qdq_e2m1(r, 16, false);
        for x in r.iter_mut() {
            *x = half_value(half_bits(*x));
        }
    }
    v
}

// ------------------------------------------------------------------------------------------------ layout

#[test]
fn the_row_layout_is_368_bytes_and_16_byte_aligned() {
    assert_eq!(kvq4_row_bytes(512), 368);
    assert_eq!(kvq4_index_row_bytes(128), 80);
    assert_eq!(kvq4_row_bytes(512) % 16, 0);
    assert_eq!(kvq4_index_row_bytes(128) % 16, 0);
    // a plane of any number of rows keeps every row 16-byte aligned
    for n in [1usize, 7, 1_024, 32_769] {
        assert_eq!((n * 368) % 16, 0);
    }
    // the fp16 alternative, for the ratio: 1,024 -> 368 is 2.78x, not 4x (the RoPE tail is 35% of a q4 row)
    assert_eq!(KvFmt::F16.row_bytes(512, ROPE_DIMS), 1_024);
    assert_eq!(KvFmt::Q4.row_bytes(128, 0), 80);
    assert!((1_024.0f64 / 368.0 - 2.7826).abs() < 1e-3);
}

#[test]
fn the_format_parses_and_refuses_typos() {
    assert_eq!(KvFmt::parse("q4"), Ok(KvFmt::Q4));
    assert_eq!(KvFmt::parse(" Q4 "), Ok(KvFmt::Q4));
    assert_eq!(KvFmt::parse("f16"), Ok(KvFmt::F16));
    assert_eq!(KvFmt::parse("fp16"), Ok(KvFmt::F16));
    assert!(KvFmt::parse("q8").is_err(), "a typo must not silently serve fp16");
    assert_eq!(KvFmt::default(), KvFmt::F16, "fp16 is the default");
    std::env::set_var(KV_FMT_ENV, "q4");
    assert_eq!(configured_fmt(KvFmt::F16), Ok(KvFmt::Q4));
    std::env::set_var(KV_FMT_ENV, "nonsense");
    assert!(configured_fmt(KvFmt::F16).is_err());
    std::env::remove_var(KV_FMT_ENV);
    assert_eq!(configured_fmt(KvFmt::F16), Ok(KvFmt::F16));
}

#[test]
fn e8m0_is_a_power_of_two_and_the_scale_never_clips() {
    for b in 1u16..=254 {
        let v = e8m0_value(b as u8);
        assert!(v > 0.0 && v.is_finite(), "byte {b} must be a positive power of two");
        assert_eq!(v.log2().fract(), 0.0);
    }
    // the byte a group of magnitude `amax` gets leaves `amax` inside the E2M1 range
    for amax in [1e-4f32, 1e-3, 0.02, 0.5, 1.0, 7.0, 1e3] {
        let b = e8m0_byte(amax);
        assert!(amax / e8m0_value(b) <= 6.0 + 1e-5, "amax {amax} clips at byte {b}");
    }
}

// ------------------------------------------------------------------------------------------------ round trip

#[test]
fn the_round_trip_error_is_bounded_by_the_group_scale() {
    let hd = 512;
    let n = 256;
    let mut r = Rng(0x5eed_0000_0000_0001);
    for heavy in [false, true] {
        let rows = latents(&mut r, n, hd, heavy);
        let mut buf = vec![0u8; kvq4_row_bytes(hd)];
        let mut back = vec![0f32; hd];
        let mut worst: f32 = 0.0;
        for j in 0..n {
            let (nope, rope) = rows[j * hd..][..hd].split_at(hd - ROPE_DIMS);
            pack_q4_row(nope, rope, false, &mut buf);
            unpack_q4_row(&buf, hd - ROPE_DIMS, false, &mut back);
            // the fp16 tail is exact
            for i in 0..ROPE_DIMS {
                assert_eq!(back[hd - ROPE_DIMS + i], half_value(half_bits(rope[i])));
            }
            // every value of a group is within one step of the group's scale
            for g in 0..(hd - ROPE_DIMS) / KV_GROUP {
                let scale = e8m0_value(buf[(hd - ROPE_DIMS) / 2 + g]);
                for i in g * KV_GROUP..(g + 1) * KV_GROUP {
                    let e = (back[i] - nope[i]).abs();
                    assert!(e <= scale + 1e-6, "group {g} dim {i}: {e} over one step ({scale})");
                    worst = worst.max(e / scale);
                }
            }
        }
        assert!(worst <= 1.0, "the worst error is {worst} steps");
        println!("[round trip, heavy={heavy}] worst error {worst:.3} of a group step");
    }
}

#[test]
fn the_rotation_is_orthonormal_and_its_own_inverse() {
    let mut r = Rng(0x1234_0000_0000_0002);
    let mut x: Vec<f32> = (0..448).map(|_| r.gauss()).collect();
    let y = x.clone();
    hadamard64(&mut x);
    hadamard64(&mut x);
    let (mx, mean, cos) = err_stats(&x, &y);
    assert!(mx < 1e-5 && mean < 1e-6 && cos > 0.999_999, "H^2 must be I (max {mx:e})");
    // and it preserves a dot product, which is what makes it valid for both K and V
    let mut a: Vec<f32> = (0..448).map(|_| r.gauss()).collect();
    let mut b: Vec<f32> = (0..448).map(|_| r.gauss()).collect();
    let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    hadamard64(&mut a);
    hadamard64(&mut b);
    let got: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    assert!((got - want).abs() < 1e-3 * want.abs().max(1.0), "dot {got} != {want}");
    // the transform really moves the values (a no-op would pass the two tests above)
    let mut z: Vec<f32> = (0..448).map(|_| r.gauss()).collect();
    let before = z.clone();
    hadamard64(&mut z);
    assert!(max_abs_diff(&z, &before) > 0.1);
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

// ------------------------------------------------------------------------------------------------ attention error

/// q4 against the fp16 plane, on both an isotropic and a heavy-tailed latent: max and mean relative error of the
/// attention output, and its cosine similarity. The rotation is measured, not assumed.
#[test]
fn the_attention_output_error_of_q4() {
    let hd = 512;
    let (n, heads) = (640, 64);
    let scale = 1.0 / (hd as f32).sqrt();
    for (name, heavy) in [("gauss", false), ("outlier", true)] {
        let mut r = Rng(0xfeed_0000_0000_0003);
        let rows = latents(&mut r, n, hd, heavy);
        let q = r.vec(heads * hd, -1.0, 1.0);
        let sink = vec![-1.0f32; heads];
        let want = attn_host(&q, heads, hd, &rows, &sink, scale, false);
        for rot in [false, true] {
            let stored = q4_plane(&rows, hd, rot);
            let got = attn_host(&q, heads, hd, &stored, &sink, scale, rot);
            let (mx, mean, cos) = err_stats(&got, &want);
            println!("[attention {name} rot={rot}] max {mx:e} mean {mean:e} cos {cos:.6}");
            assert!(cos > 0.98, "{name} rot={rot}: cosine {cos} is not a q4 cache");
            assert!(mean < 2e-2, "{name} rot={rot}: mean relative error {mean}");
        }
        // the rotation must be the better of the two on a heavy-tailed latent, and no worse on an isotropic one
        let plain = err_stats(&attn_host(&q, heads, hd, &q4_plane(&rows, hd, false), &sink, scale, false), &want);
        let rot = err_stats(&attn_host(&q, heads, hd, &q4_plane(&rows, hd, true), &sink, scale, true), &want);
        if heavy {
            assert!(rot.1 < 0.8 * plain.1, "the rotation must cut the mean error: {} vs {}", rot.1, plain.1);
            assert!(rot.0 < 0.6 * plain.0, "and the max: {} vs {}", rot.0, plain.0);
        } else {
            assert!(rot.1 <= 1.05 * plain.1, "the rotation must not cost an isotropic latent: {} vs {}", rot.1, plain.1);
        }
        // and the f16 plane's own error, for the ratio: q4 with the rotation is close to it
        let f16p = f16_plane(&rows, hd);
        let (_, fmean, fcos) = err_stats(&attn_host(&q, heads, hd, &f16p, &sink, scale, false), &want);
        println!("[attention {name} fp16 plane] mean {fmean:e} cos {fcos:.6}");
    }
}

// ------------------------------------------------------------------------------------------------ indexer

/// The indexer's keys are already E2M1 on a power-of-two scale per 32 before they reach the plane, so the q4 encoding
/// is exact and the selection cannot move. E4M3 with a per-row scale is the alternative, measured here.
#[test]
fn the_indexer_selection_agrees() {
    let (id, npos, ih) = (128usize, 32_768usize, 32usize);
    let mut r = Rng(0xdead_0000_0000_0004);
    let mut keys = latents(&mut r, npos, id, true);
    // what the device stores today: block-32 E2M1 with a power-of-two scale
    for p in 0..npos {
        qdq_e2m1(&mut keys[p * id..][..id], 32, true);
    }
    let q4 = {
        let mut out = vec![0f32; keys.len()];
        let mut buf = vec![0u8; kvq4_index_row_bytes(id)];
        for p in 0..npos {
            pack_q4_row(&keys[p * id..][..id], &[], false, &mut buf);
            unpack_q4_row(&buf, id, false, &mut out[p * id..][..id]);
        }
        out
    };
    // the alternative: one E4M3 byte a value, one scale a row
    let fp8 = {
        let mut out = keys.clone();
        for p in 0..npos {
            let amax = out[p * id..][..id].iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-4);
            let s = amax / 448.0;
            for x in out[p * id..][..id].iter_mut() {
                let q = *x / s;
                let e = q.abs().max(1e-30).log2().floor();
                let step = 2f32.powf(e - 3.0);
                *x = ((q / step).round() * step).clamp(-448.0, 448.0) * s;
            }
        }
        out
    };
    assert_eq!(q4, keys, "the q4 key plane must be the fp16 plane's values bit for bit");
    let qh = r.vec(ih * id, -1.0, 1.0);
    let w: Vec<f32> = (0..ih).map(|h| 0.1 + 0.01 * h as f32).collect();
    let score = |k: &[f32]| -> Vec<f32> {
        (0..npos)
            .map(|p| {
                let mut acc = 0f32;
                for h in 0..ih {
                    let mut d = 0f32;
                    for i in 0..id {
                        d += qh[h * id + i] * k[p * id + i];
                    }
                    acc += d.max(0.0) * w[h];
                }
                acc
            })
            .collect()
    };
    let base = topk_indices(&score(&keys), 512);
    for (name, k) in [("q4", &q4), ("fp8", &fp8)] {
        let t = topk_indices(&score(k), 512);
        let inter = t.iter().filter(|x| base.binary_search(x).is_ok()).count();
        println!("[indexer] {name}: top-512 overlap {inter}/512 ({:.2}%)", 100.0 * inter as f32 / 512.0);
        if name == "q4" {
            assert_eq!(inter, 512, "the q4 plane is a re-encoding: the selection must not move");
        }
    }
    println!("[indexer] bytes an entry: f16 256, q4 80, fp8 144 (128 values + a row scale, 16-aligned)");
}

// ------------------------------------------------------------------------------------------------ ratio_hint

/// The bug the per-layer slot count fixes: a ratio-1 layer's compressed plane holds twice what a ratio-2 layer's does,
/// and every plane used to be sized from the largest ratio in the table.
#[test]
fn a_ratio_one_layer_gets_its_own_plane() {
    let cfg = Dsv4Config::v41();
    let ctx = 32_768;
    assert_eq!(cfg.ratio_hint(), 2, "the table's largest ratio");
    assert_eq!(compress_slots(&cfg, 2, ctx), 16_385);
    assert_eq!(compress_slots(&cfg, 20, ctx), 32_769, "a ratio-1 layer pools one entry a token");
    assert_eq!(compress_slots_max(&cfg, ctx), 32_769);
    // the old formula would have overflowed layer 20's plane by 16,384 entries
    let old = ctx.div_ceil(cfg.ratio_hint()) + 1;
    assert_eq!(old, 16_385);
    assert!(ctx.div_ceil(1) > old, "a ratio-1 layer writes past a plane sized from ratio_hint");
    // and the allocation follows the per-layer count
    let a = Dsv4Planes::bytes(&cfg, ctx, 2, KvFmt::F16);
    let b = Dsv4Planes::bytes(&cfg, ctx, 20, KvFmt::F16);
    let per = 512 * 2 + 128 * 2 + 128; // the compressed entry, the index key and the level-1 mask byte
    assert_eq!(b - a, (32_769 - 16_385) * per, "layer 20's plane is the larger one");
    // q4 is 2.83x smaller on the same shape
    let q = Dsv4Planes::bytes(&cfg, ctx, 20, KvFmt::Q4);
    println!("[planes] layer 20 at 32K: f16 {b} B, q4 {q} B ({:.2}x)", b as f64 / q as f64);
    assert!(q * 2 < b);
}

/// **The price is the allocation.** At the serving shape (32K, one slot) `Dsv4Planes::bytes` is exactly what `new`
/// allocates, on a ratio-2 layer and a ratio-1 one, at both formats — and q4 is 2.4x smaller over the whole five planes
/// (2.86x over the compressed entries and index keys; the fp16 window ring and the i32 selection do not move).
#[test]
fn the_planes_price_is_what_new_allocates() {
    let cfg = Dsv4Config::v41();
    let ctx = 32_768;
    // the whole trunk, by the constructor's formula
    let total = |fmt: KvFmt| -> u64 { (0..40).map(|l| Dsv4Planes::bytes(&cfg, ctx, l, fmt) as u64).sum() };
    let (f16, q4) = (total(KvFmt::F16), total(KvFmt::Q4));
    println!("[planes] 32K x 1 slot: f16 {f16} B, q4 {q4} B ({:.3}x)", f16 as f64 / q4 as f64);
    assert!((f16 as f64 / q4 as f64) >= 2.4, "{f16} vs {q4}");
    // the payload alone: (512 + 128) values an entry at 32K over each layer's own ratio
    let rows = |fmt: KvFmt| -> u64 {
        let (e, k) = (fmt.row_bytes(512, ROPE_DIMS) as u64, fmt.row_bytes(128, 0) as u64);
        (0..40u64).map(|l| (e + k) * (ctx as u64 / cfg.ratio(l as usize).max(1) as u64 + 1)).sum()
    };
    println!(
        "[planes] entries + keys: f16 {} B, q4 {} B ({:.3}x)",
        rows(KvFmt::F16),
        rows(KvFmt::Q4),
        rows(KvFmt::F16) as f64 / rows(KvFmt::Q4) as f64
    );
    assert!((rows(KvFmt::F16) as f64 / rows(KvFmt::Q4) as f64) >= 2.5);

    if !device_available() {
        eprintln!("not run: no device (the allocation half of this test)");
        return;
    }
    // and `new` allocates exactly the price: all five buffers, both formats, a ratio-2 layer and a ratio-1 one
    for fmt in [KvFmt::F16, KvFmt::Q4] {
        for layer in [2usize, 20] {
            let p = Dsv4Planes::new(&cfg, ctx, layer, fmt).unwrap();
            assert_eq!(p.held_bytes(), Dsv4Planes::bytes(&cfg, ctx, layer, fmt), "{fmt:?} layer {layer}");
            assert_eq!(p.n_cmp, compress_slots(&cfg, layer, ctx));
            assert_eq!(p.cand_stride, p.n_cmp, "the mask is a column an entry, not a column a max-ratio entry");
        }
    }
}

// ------------------------------------------------------------------------------------------------ device

/// The store kernel against the host packer, byte for byte.
#[test]
fn gpu_store_matches_the_host() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    let hd = 512;
    let n = 5;
    let mut r = Rng(0xaaaa_0000_0000_0005);
    let rows = down(&latents(&mut r, n, hd, true));
    let src = up(&rows);
    let dst = DevBuf::zeroed(n * kvq4_row_bytes(hd)).unwrap();
    store_dev(&src, 0, &dst, n, 0, hd, ROPE_DIMS, 0, &st).unwrap();
    // the index-key form (no fp16 tail)
    let id = 128;
    let keys = down(&latents(&mut r, n, id, true));
    let ksrc = up(&keys);
    let kdst = DevBuf::zeroed(n * kvq4_index_row_bytes(id)).unwrap();
    store_dev(&ksrc, 0, &kdst, n, 0, id, 0, 0, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<u8> = dst.to_vec().unwrap();
    let kgot: Vec<u8> = kdst.to_vec().unwrap();
    let mut want = vec![0u8; kvq4_row_bytes(hd)];
    let mut kwant = vec![0u8; kvq4_index_row_bytes(id)];
    for j in 0..n {
        let (nope, rope) = rows[j * hd..][..hd].split_at(hd - ROPE_DIMS);
        pack_q4_row(nope, rope, false, &mut want);
        assert_eq!(&got[j * kvq4_row_bytes(hd)..][..kvq4_row_bytes(hd)], &want[..], "row {j} differs");
        pack_q4_row(&keys[j * id..][..id], &[], false, &mut kwant);
        assert_eq!(&kgot[j * kvq4_index_row_bytes(id)..][..kvq4_index_row_bytes(id)], &kwant[..]);
    }
}

/// The rotation kernel against [`hadamard64`].
#[test]
fn gpu_rotation_matches_the_host() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    let (hd, heads, n) = (512, 3, 2);
    let mut r = Rng(0xbbbb_0000_0000_0006);
    let host: Vec<f32> = down(&(0..n * heads * hd).map(|_| r.gauss()).collect::<Vec<f32>>());
    let x = up(&host);
    rot_dev(&x, n, heads * hd, hd - ROPE_DIMS, hd, heads, &st).unwrap();
    st.sync().unwrap();
    let got = read16(&x);
    let mut want = host.clone();
    for row in want.chunks_mut(heads * hd) {
        for h in 0..heads {
            hadamard64(&mut row[h * hd..][..hd - ROPE_DIMS]);
        }
    }
    let (mx, _, cos) = err_stats(&got, &want);
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    println!("[rot] max {mx:e} of {scale} (cos {cos:.7})");
    assert!(mx < 2e-3 * scale, "rotation differs from the host by {mx:e} of {scale}");
}

/// The q4 attention kernel against [`attn_host`] on the same dequantised keys, and against the fp16 kernel's output.
#[test]
fn gpu_attn_matches_the_host() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    let hd = 512;
    let (n_cmp, window, heads) = (96usize, 16usize, 64usize);
    let mut r = Rng(0xcccc_0000_0000_0007);
    let cmp_rows = down(&latents(&mut r, n_cmp, hd, true));
    let win_rows = down(&latents(&mut r, window, hd, true));
    let mut q = down(&latents(&mut r, 1, hd, false));
    q.extend(down(&latents(&mut r, heads - 1, hd, false)));
    let sink: Vec<f32> = (0..heads).map(|h| -0.5 - 0.01 * h as f32).collect();
    // the window rows first (index < split), then every compressed entry (offset by split), then a hole
    let mut idx: Vec<i32> = (0..window as i32).collect();
    idx.extend((0..n_cmp as i32).map(|j| window as i32 + j));
    idx.push(-1);
    let scale = 1.0 / (hd as f32).sqrt();

    let sinkb = DevBuf::from_slice(&sink).unwrap();
    let idxb = DevBuf::from_slice(&idx).unwrap();
    let out = DevBuf::zeroed(heads * hd * 2).unwrap();
    // the q4 path: the query rotated, both planes rotated and encoded
    let qrot = up(&q);
    rot_dev(&qrot, 1, heads * hd, hd - ROPE_DIMS, hd, heads, &st).unwrap();
    let mut cmp_rot = cmp_rows.clone();
    unrotate(&mut cmp_rot, hd);
    let mut cmp_enc = vec![0f32; cmp_rot.len()];
    let mut buf = vec![0u8; kvq4_row_bytes(hd)];
    for j in 0..n_cmp {
        // `cmp_rot` is already in the rotated domain (the kernel rotates with `rot_dev`), so pack without rotating
        let (nope, rope) = cmp_rot[j * hd..][..hd].split_at(hd - ROPE_DIMS);
        pack_q4_row(nope, rope, false, &mut buf);
        unpack_q4_row(&buf, hd - ROPE_DIMS, false, &mut cmp_enc[j * hd..][..hd]);
    }
    let mut win_rot = win_rows.clone();
    unrotate(&mut win_rot, hd);
    let mut win_enc = vec![0f32; win_rows.len()];
    for j in 0..window {
        let (nope, rope) = win_rot[j * hd..][..hd].split_at(hd - ROPE_DIMS);
        pack_q4_row(nope, rope, false, &mut buf);
        unpack_q4_row(&buf, hd - ROPE_DIMS, false, &mut win_enc[j * hd..][..hd]);
    }
    let cmp_q4 = DevBuf::zeroed(n_cmp * kvq4_row_bytes(hd)).unwrap();
    let win_q4 = DevBuf::zeroed(window * kvq4_row_bytes(hd)).unwrap();
    // the store takes the unrotated (= the rotated-domain row after `rot_dev`) values
    let cmp_src = up(&cmp_rows);
    rot_dev(&cmp_src, n_cmp, hd, hd - ROPE_DIMS, hd, 1, &st).unwrap();
    store_dev(&cmp_src, 0, &cmp_q4, n_cmp, 0, hd, ROPE_DIMS, 0, &st).unwrap();
    let win_src = up(&win_rows);
    rot_dev(&win_src, window, hd, hd - ROPE_DIMS, hd, 1, &st).unwrap();
    store_dev(&win_src, 0, &win_q4, window, 0, hd, ROPE_DIMS, 0, &st).unwrap();
    attn_dev(
        &qrot,
        &win_q4,
        &cmp_q4,
        window,
        &sinkb,
        &idxb,
        idx.len(),
        idx.len(),
        &out,
        1,
        heads,
        hd,
        KvFmt::Q4,
        KvFmt::Q4,
        scale,
        &st,
    )
    .unwrap();
    st.sync().unwrap();
    let got = read16(&out);
    // the host twin over the same dequantised planes, in the rotated domain
    let mut keys = Vec::with_capacity((n_cmp + window) * hd);
    keys.extend_from_slice(&win_enc);
    keys.extend_from_slice(&cmp_enc);
    let qhost = read16(&qrot);
    let want = attn_host(&qhost, heads, hd, &keys, &sink, scale, false);
    let (mx, mean, cos) = err_stats(&got, &want);
    println!("[gpu attn q4] vs host: max {mx:e} mean {mean:e} cos {cos:.7}");
    // the host accumulates the softmax in f64 and the device online in f32, so the agreement is a tolerance, not bits
    assert!(mx < 0.1, "device and host differ by {mx:e}");
    assert!(mean < 5e-3 && cos > 0.9998, "mean {mean:e}, cosine {cos}");
    // and the output really is a rotated value sum: unrotating it lands near the fp16 kernel's answer
    let mut un = got.clone();
    for h in 0..heads {
        hadamard64(&mut un[h * hd..][..hd - ROPE_DIMS]);
    }
    let mut kv16 = Vec::with_capacity((n_cmp + window) * hd);
    kv16.extend_from_slice(&win_rows);
    kv16.extend_from_slice(&cmp_rows);
    let want16 = attn_host(&q, heads, hd, &kv16, &sink, scale, false);
    let (mx, mean, cos) = err_stats(&un, &want16);
    println!("[gpu attn q4] vs fp16: max {mx:e} mean {mean:e} cos {cos:.6}");
    assert!(cos > 0.98, "q4 output is not the fp16 output: cosine {cos}");
}

/// The indexer's score kernel against its host twin.
#[test]
fn gpu_index_score_matches_the_host() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    let (id, ih, n, npos) = (128usize, 32usize, 3usize, 40usize);
    let mut r = Rng(0xdddd_0000_0000_0008);
    let q = down(&r.vec(n * ih * id, -1.0, 1.0));
    let w: Vec<f32> = (0..n * ih).map(|_| r.uniform(-0.2, 0.2)).collect();
    let mut keys = down(&latents(&mut r, npos, id, true));
    for p in 0..npos {
        qdq_e2m1(&mut keys[p * id..][..id], 32, true);
    }
    let keybuf = DevBuf::zeroed(npos * kvq4_index_row_bytes(id)).unwrap();
    let ksrc = up(&keys);
    store_dev(&ksrc, 0, &keybuf, npos, 0, id, 0, 0, &st).unwrap();
    let score = DevBuf::zeroed(n * npos * 4).unwrap();
    index_score_dev(
        &up(&q),
        &keybuf,
        &DevBuf::from_slice(&w).unwrap(),
        None,
        npos,
        &score,
        n,
        npos,
        ih,
        id,
        1.0 / (id as f32).sqrt(),
        0,
        1,
        KvFmt::Q4,
        &st,
    )
    .unwrap();
    st.sync().unwrap();
    let got: Vec<f32> = score.to_vec().unwrap();
    let mut want = vec![0f32; n * npos];
    for row in 0..n {
        let len = row + 1;
        for p in 0..npos {
            if p >= len {
                want[row * npos + p] = f32::NEG_INFINITY;
                continue;
            }
            let mut acc = 0f32;
            for h in 0..ih {
                let mut d = 0f32;
                for i in 0..id {
                    d += q[(row * ih + h) * id + i] * keys[p * id + i];
                }
                acc += d.max(0.0) * w[row * ih + h];
            }
            want[row * npos + p] = acc / (id as f32).sqrt();
        }
    }
    let mut worst = 0f32;
    for (g, e) in got.iter().zip(&want) {
        if e.is_finite() {
            worst = worst.max((g - e).abs() / e.abs().max(1e-3));
        } else {
            assert_eq!(*g, f32::NEG_INFINITY, "the reach mask must be -inf");
        }
    }
    println!("[gpu index score q4] worst relative error {worst:e}");
    assert!(worst < 5e-3, "device and host differ by {worst:e}");
}

/// The decode read at 32K: 128 window rows and a 512-entry selection, per head, q4 against fp16.
#[test]
fn gpu_attn_microbench() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    let hd = 512;
    let (window, topk, heads) = (128usize, 512usize, 64usize);
    let mut r = Rng(0xeeee_0000_0000_0009);
    let mut one: Vec<i32> = (0..(window + topk) as i32).collect();
    one[3] = -1;
    // eight rows of the same selection: `idx_len` is a row stride, and the server runs eight slots
    let idx: Vec<i32> = (0..8).flat_map(|_| one.iter().copied()).collect();
    let idxb = DevBuf::from_slice(&idx).unwrap();
    let q = up(&latents(&mut r, 8 * heads, hd, false));
    let sink = DevBuf::from_slice(&vec![-1.0f32; heads]).unwrap();
    let out = DevBuf::zeroed(8 * heads * hd * 2).unwrap();
    let scale = 1.0 / (hd as f32).sqrt();
    let win16 = up(&latents(&mut r, window, hd, true));
    let cmp16 = up(&latents(&mut r, topk, hd, true));
    let win8 = DevBuf::zeroed(window * kvq4_row_bytes(hd)).unwrap();
    let cmp8 = DevBuf::zeroed(topk * kvq4_row_bytes(hd)).unwrap();
    store_dev(&win16, 0, &win8, window, 0, hd, ROPE_DIMS, 0, &st).unwrap();
    store_dev(&cmp16, 0, &cmp8, topk, 0, hd, ROPE_DIMS, 0, &st).unwrap();
    st.sync().unwrap();

    let iters = 200;
    // one slot and the server's eight: at eight rows the grid is eight times wider, which is the case where the q4
    // plane's 2.78x fewer bytes has a chance to matter
    let run = |rows: usize, wf: KvFmt, cf: KvFmt| -> f64 {
        // warm up, then time
        for _ in 0..20 {
            attn_dev(&q, &win16, &cmp16, window, &sink, &idxb, one.len(), one.len(), &out, rows, heads, hd, wf, cf, scale, &st)
                .unwrap();
        }
        st.sync().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let (w, c) = if wf == KvFmt::Q4 { (&win8, &cmp8) } else { (&win16, &cmp16) };
            attn_dev(&q, w, c, window, &sink, &idxb, one.len(), one.len(), &out, rows, heads, hd, wf, cf, scale, &st).unwrap();
        }
        st.sync().unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    };
    for rows in [1usize, 8] {
        let f16 = run(rows, KvFmt::F16, KvFmt::F16);
        let mixed = run(rows, KvFmt::F16, KvFmt::Q4);
        let q4 = run(rows, KvFmt::Q4, KvFmt::Q4);
        println!(
            "[microbench] {rows} row(s), 640 keys x 64 heads, one layer: fp16 {f16:.1} us, window fp16 + q4 {mixed:.1} us, q4 {q4:.1} us"
        );
        println!(
            "[microbench] {rows} row(s): a 40-layer decode step fp16 {:.2} ms, q4 {:.2} ms",
            f16 * 40.0 / 1000.0,
            q4 * 40.0 / 1000.0
        );
        assert!(q4 < 10.0 * f16, "q4 must not be an order of magnitude slower");
    }
}
