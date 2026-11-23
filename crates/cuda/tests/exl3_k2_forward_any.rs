//! GPU: the EXL3 half-integer-width (k2 = 3/5/7) decode and prefill paths against the reconstruct + GEMM
//! reference of the existing integer tests.
//!
//! The EXL3 linear runs in the Hadamard-rotated domain, so the reference weight is the *effective* matrix the
//! kernels implement, measured exactly the way the existing integer tests measure it: feed each basis vector
//! through the decode path and record the column it produces ([`deq_of`] — the same method as `exl3_w` in
//! `dsv4_mtp_gpu.rs`). Then a random input must produce the same output through `forward_any` (decode) and
//! `forward_prefill_pinned_any` (prefill) as through an f64 GEMM over that reconstructed matrix, within the
//! documented tolerance of the existing integer tests (5e-3 scaled). In addition, the two GPU reconstructors —
//! `fmoe::unpack` and `gb_exl3_reconstruct_k2` — must agree bit-for-bit on the same trellis.
//!
//! What runs without a GPU: nothing (the test returns early), so the binary is green on a CPU-only box.
//!
//! Every launch here runs on a non-blocking stream, which does not wait for the legacy default stream, while the setup
//! (`DevBuf::from_slice`, a pageable `cudaMemcpy` that can return before its DMA lands, and `DevBuf::zeroed`, an asynchronous
//! `cudaMemset`) runs on the legacy stream. So each launch waits for its setup first ([`setup_landed`]). Without that, under
//! the gate's parallel CUDA lanes a workspace memset could land after the kernel had written the rotated input into it, and
//! the decode output came back as the bias alone (X148: `odd-k2 bias m=1: err/scale 9.883e-1`).
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::fmoe;
use gb_cuda::rt::{CudaError, DevBuf};
use gb_cuda::{device_available, Stream};
use gb_exl3::checkpoint::Group;
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::{Bits, Codebook};

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
}

/// `max|a - b| / max|b|`: the error relative to the tensor's own scale (the existing EXL3 tests' metric).
fn scaled_err(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "lengths");
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6) as f64;
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs())) / scale
}

/// Waits for the device, so that the legacy-stream setup above a launch (uploads and memsets) has landed before a kernel on
/// a non-blocking stream reads or writes those buffers (see the module docs and `HostStage` in `rt.rs`).
fn setup_landed() {
    gb_cuda::rt::device_sync().expect("device sync");
}

/// A synthetic k2 group: random trellis words (every 2-bit field of a random word is a valid `mul1` codebook
/// index), fp16 `suh`/`svh` scales and an optional non-zero bias. Returns the loaded weight and its trellis
/// buffer (the unpack reconstructors take it).
fn synth_k2(k: usize, n: usize, k2: usize, seed: u64, with_bias: bool, _st: &Stream) -> (Exl3Weights, DevBuf) {
    let mut r = Rng(seed);
    let words = k * n * k2 / 32;
    let trellis: Vec<i16> = (0..words).map(|_| r.next_u64() as i16).collect();
    let suh: Vec<u16> = (0..k).map(|_| f64_to_f16(r.uniform(0.2, 0.3) as f64)).collect();
    let svh: Vec<u16> = (0..n).map(|_| f64_to_f16(r.uniform(0.2, 0.3) as f64)).collect();
    let bias: Option<Vec<u16>> = with_bias.then(|| (0..n).map(|_| f64_to_f16(r.uniform(0.1, 0.2) as f64)).collect());
    let g = Group {
        prefix: String::new(),
        bits: Bits::from_twice(k2 as u8).expect("bits"),
        codebook: Codebook::Mul1,
        k,
        n,
        in_scales: "suh",
        out_scales: "svh",
        bias: with_bias,
        files: vec![],
        bytes: 0,
    };
    let trellis_d = DevBuf::from_slice(&trellis).expect("trellis");
    let w = Exl3Weights::from_parts(
        &g,
        DevBuf::from_slice(&trellis).expect("trellis (weight)"),
        DevBuf::from_slice(&suh).expect("suh"),
        DevBuf::from_slice(&svh).expect("svh"),
        bias.as_deref().map(|b| DevBuf::from_slice(b).expect("bias")),
    )
    .expect("from_parts");
    setup_landed();
    (w, trellis_d)
}

/// The effective `[k, n]` fp16 matrix the decode path implements, by feeding the basis vector `e_i` through
/// `forward_any` and recording the output column (the existing integer tests' reconstruct method).
fn deq_of(w: &Exl3Weights, k: usize, n: usize, st: &Stream) -> Vec<f32> {
    let mut deq = vec![0f32; n * k];
    let mut e_in = vec![0u16; k];
    let mut d_in = DevBuf::zeroed(k * 2).expect("d_in");
    let mut d_out = DevBuf::zeroed(n * 2).expect("d_out");
    let mut d_ws = DevBuf::zeroed(w.workspace_bytes(1).max(4 << 20)).expect("d_ws");
    for i in 0..k {
        if i > 0 {
            e_in[i - 1] = 0;
        }
        e_in[i] = f64_to_f16(1.0);
        d_in.upload(&e_in).expect("h2d");
        setup_landed();
        w.forward_any(&d_in, &mut d_out, 1, Dtype::F16, Dtype::F16, &mut d_ws, st).expect("forward");
        st.sync().expect("sync");
        let row = d_out.to_vec::<u16>().expect("d2h");
        for j in 0..n {
            deq[j * k + i] = f16_to_f64(row[j]) as f32;
        }
    }
    deq
}

/// `y [m, n] = W [n, k] x [m, k]` in f64 over `deq`.
fn gemm_ref(deq: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut want = vec![0f32; m * n];
    for row in 0..m {
        for c in 0..n {
            let mut acc = 0f64;
            for i in 0..k {
                acc += x[row * k + i] as f64 * deq[c * k + i] as f64;
            }
            want[row * n + c] = acc as f32;
        }
    }
    want
}

fn f16s(v: &DevBuf) -> Vec<f32> {
    v.to_vec::<u16>().expect("d2h").iter().map(|&h| f16_to_f64(h) as f32).collect()
}

/// The integrated-memory probe uses CUDA's `cudaDevAttrIntegrated` (18), not the similarly named boolean-looking
/// attributes around it. A discrete PCIe device reports false; an integrated device such as GB10 reports true.
#[test]
fn device_integration_matches_cuda_attribute() {
    #[cfg(gb_no_cuda)]
    assert!(!gb_cuda::rt::device_is_integrated());

    #[cfg(not(gb_no_cuda))]
    {
        if !device_available() {
            println!("not run: no device");
            return;
        }
        extern "C" {
            fn cudaDeviceGetAttribute(value: *mut i32, attribute: i32, device: i32) -> i32;
        }
        let mut value = 0i32;
        // SAFETY: `value` is a valid output pointer and device 0 exists per `device_available`.
        let rc = unsafe { cudaDeviceGetAttribute(&mut value, 18, 0) };
        assert_eq!(rc, 0, "cudaDeviceGetAttribute(cudaDevAttrIntegrated) failed: {rc}");
        assert_eq!(gb_cuda::rt::device_is_integrated(), value == 1);
    }
}

/// `forward_any` (decode) at `m` rows and `forward_prefill_pinned_any` (prefill) against the f64 GEMM over the
/// reconstructed matrix, for k2 = 2 through 8 (integer and half-integer widths). F16 in/out matches the existing
/// integer tests' dtype and tolerance.
#[test]
fn half_integer_k2_forward_any_matches_reconstruct_gemm() {
    if !device_available() {
        println!("not run: no device");
        return;
    }
    let st = Stream::new_nonblocking().expect("stream");
    let (k, n) = (5120usize, 5120usize);
    for k2 in [2usize, 3, 4, 5, 6, 7, 8] {
        let (w, trellis_d) = synth_k2(k, n, k2, 0xbeef + k2 as u64, false, &st);
        // the two GPU reconstructors must agree on the same trellis, bit for bit
        let mut wq_a = DevBuf::zeroed(k * n * 2).expect("wq_a");
        let wq_b = DevBuf::zeroed(k * n * 2).expect("wq_b");
        setup_landed();
        fmoe::unpack(&trellis_d, &mut wq_a, k, n, k2, &st).expect("unpack");
        unsafe {
            gb_cuda::exl3_linear::reconstruct_k2_raw(
                trellis_d.ptr(),
                wq_b.ptr(),
                k as i32,
                n as i32,
                k2 as i32,
                2, // mul1
                st.raw(),
            )
        }
        .expect("reconstruct_k2_raw");
        st.sync().expect("sync");
        let a = wq_a.to_vec::<u16>().expect("d2h");
        let b = wq_b.to_vec::<u16>().expect("d2h");
        assert_eq!(a, b, "k2={k2}: fmoe::unpack and gb_exl3_reconstruct_k2 differ");

        let deq = deq_of(&w, k, n, &st);
        for &m in &[1usize, 8, 64, 128] {
            let mut r = Rng(0x1234 + k2 as u64 + m as u64);
            let x: Vec<f32> = (0..m * k).map(|_| r.uniform(-1.0, 1.0)).collect();
            let xb: Vec<u16> = x.iter().map(|&v| f64_to_f16(v as f64)).collect();
            let xv = DevBuf::from_slice(&xb).expect("x");
            let mut y = DevBuf::zeroed(m * n * 2).expect("y");
            let mut ws = DevBuf::zeroed(w.workspace_bytes(m).max(4 << 20)).expect("ws");
            let want = gemm_ref(&deq, &x, m, k, n);
            setup_landed();
            w.forward_any(&xv, &mut y, m, Dtype::F16, Dtype::F16, &mut ws, &st).expect("forward_any");
            st.sync().expect("sync");
            let got = f16s(&y);
            let e = scaled_err(&got, &want);
            println!("k2={k2} decode m={m}: err/scale {e:.3e}");
            assert!(e < 5e-3, "forward_any k2={k2} m={m}: err/scale {e:.3e}");
        }
        // prefill: the pinned parts path (rows > MAX_DECODE_ROWS), the same reference
        let m = 256usize;
        let mut r = Rng(0x5678 + k2 as u64);
        let x: Vec<f32> = (0..m * k).map(|_| r.uniform(-1.0, 1.0)).collect();
        let xb: Vec<u16> = x.iter().map(|&v| f64_to_f16(v as f64)).collect();
        let xv = DevBuf::from_slice(&xb).expect("x");
        let mut y = DevBuf::zeroed(m * n * 2).expect("y");
        let mut scratch = DevBuf::zeroed_view(w.prefill_scratch_bytes(m)).expect("scratch");
        let want = gemm_ref(&deq, &x, m, k, n);
        setup_landed();
        w.forward_prefill_pinned_any(&xv, &mut y, m, Dtype::F16, &mut scratch, &st).expect("prefill_pinned_any");
        st.sync().expect("sync");
        let got = f16s(&y);
        let e = scaled_err(&got, &want);
        println!("k2={k2} prefill m={m}: err/scale {e:.3e}");
        assert!(e < 5e-3, "forward_prefill_pinned_any k2={k2} m={m}: err/scale {e:.3e}");
    }
}

#[test]
fn odd_k2_bias_reaches_decode_and_pinned_prefill() {
    if !device_available() {
        println!("not run: no device");
        return;
    }
    let st = Stream::new_nonblocking().expect("stream");
    let (k, n, k2) = (5120usize, 5120usize, 5usize);
    let (plain, _) = synth_k2(k, n, k2, 0xb1a5, false, &st);
    let (biased, _) = synth_k2(k, n, k2, 0xb1a5, true, &st);
    let bias = f16s(biased.buffers().into_iter().find(|(name, _)| *name == "bias").expect("bias").1);
    assert!(bias.iter().any(|&b| b != 0.0));
    let deq = deq_of(&plain, k, n, &st);

    for m in [1usize, 256] {
        let mut r = Rng(0xb1a5 + m as u64);
        let x: Vec<f32> = (0..m * k).map(|_| r.uniform(-1.0, 1.0)).collect();
        let xb: Vec<u16> = x.iter().map(|&v| f64_to_f16(v as f64)).collect();
        let xv = DevBuf::from_slice(&xb).expect("x");
        let mut want = gemm_ref(&deq, &x, m, k, n);
        for row in 0..m {
            for c in 0..n {
                want[row * n + c] += bias[c];
            }
        }
        let mut y = DevBuf::zeroed(m * n * 2).expect("y");
        if m == 1 {
            let mut ws = DevBuf::zeroed(biased.workspace_bytes(m).max(4 << 20)).expect("ws");
            setup_landed();
            biased.forward_any(&xv, &mut y, m, Dtype::F16, Dtype::F16, &mut ws, &st).expect("forward_any");
        } else {
            let mut scratch = DevBuf::zeroed_view(biased.prefill_scratch_bytes(m)).expect("scratch");
            setup_landed();
            biased.forward_prefill_pinned_any(&xv, &mut y, m, Dtype::F16, &mut scratch, &st).expect("prefill");
        }
        st.sync().expect("sync");
        let e = scaled_err(&f16s(&y), &want);
        assert!(e < 5e-3, "odd-k2 bias m={m}: err/scale {e:.3e}");
    }
}

#[test]
fn odd_k2_integer_forward_fails_closed() {
    if !device_available() {
        println!("not run: no device");
        return;
    }
    let st = Stream::new_nonblocking().expect("stream");
    let (k, n) = (128usize, 128usize);
    let (w, _) = synth_k2(k, n, 5, 0xfa11, false, &st);
    let x = DevBuf::zeroed(k * 2).expect("x");
    let mut y = DevBuf::zeroed(n * 2).expect("y");
    let mut ws = DevBuf::zeroed(w.workspace_bytes(1).max(4 << 20)).expect("ws");
    assert_eq!(w.forward(&x, &mut y, 1, Dtype::F16, Dtype::F16, &mut ws, &st), Err(CudaError(1)));
}

#[test]
fn odd_k2_selected_strips_use_exact_trellis_bytes_and_parent_columns() {
    if !device_available() {
        println!("not run: no device");
        return;
    }
    let st = Stream::new_nonblocking().expect("stream");
    let (k, n, k2) = (128usize, 80 * 128, 5usize);
    let (parent, _) = synth_k2(k, n, k2, 0x57121, false, &st);
    let cut = parent.select_strips(&[(1, 65)]).expect("select_strips");
    let expected_bytes = (k / 16) * 64 * 64 * k2 * 2;
    let trellis_bytes = cut.buffers().into_iter().find(|(name, _)| *name == "trellis").expect("trellis").1.bytes();
    assert_eq!(trellis_bytes, expected_bytes, "strip bytes must use k2, not 2 * (k2 / 2)");

    let x: Vec<u16> = (0..k).map(|i| f64_to_f16((i as f64 + 1.0) / k as f64)).collect();
    let xv = DevBuf::from_slice(&x).expect("x");
    let mut full = DevBuf::zeroed(n * 2).expect("full");
    let mut ws = DevBuf::zeroed(parent.workspace_bytes(1).max(4 << 20)).expect("ws");
    let mut selected = DevBuf::zeroed(cut.n * 2).expect("selected");
    let mut cut_ws = DevBuf::zeroed(cut.workspace_bytes(1).max(4 << 20)).expect("cut ws");
    setup_landed();
    parent.forward_any(&xv, &mut full, 1, Dtype::F16, Dtype::F16, &mut ws, &st).expect("parent");
    cut.forward_any(&xv, &mut selected, 1, Dtype::F16, Dtype::F16, &mut cut_ws, &st).expect("cut");
    st.sync().expect("sync");
    let full = full.to_vec::<u16>().expect("full d2h");
    let selected = selected.to_vec::<u16>().expect("selected d2h");
    assert_eq!(selected, full[128..65 * 128], "selected strips differ from the parent columns");
}
