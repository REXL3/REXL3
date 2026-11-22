//! Gated delta net kernels: f64 CPU reference, bit-exact replay/row-invariance gates, ExLlamaV3 dumps, benchmarks.
//!
//! The 48-value-head build of the kernels (Qwen3.8-27B); tests/gdn_hv32.rs runs the same cases on the 32-head build.
#![cfg(not(gb_no_cuda))]

mod common;

const SHAPE: gb_cuda::gdn::GdnShape = gb_cuda::gdn::GdnShape::Hv48;
include!("gdn/cases.rs");

/// End-to-end GDN core timing at the Cyberfrost shape. Each layer owns a distinct slot, so the 36 working sets
/// exceed L2 just as they do in the model instead of repeatedly timing one cache-resident state.
///
/// cargo test --release -p gb-cuda --test gdn bench_l40s_36_layer_prefill_decode -- --ignored --nocapture
#[test]
#[ignore]
fn bench_l40s_36_layer_prefill_decode() {
    need_gpu!();
    const LAYERS: usize = 36;
    let mut rng = Rng(0x12c_6d6e);
    let ctx = Ctx::new(Weights::random(&mut rng));

    let rows = 4096;
    let inp = Inp::random(&mut rng, rows);
    let x = DevInp::new(&inp);
    let batches: Vec<BatchBuf> = (0..LAYERS).map(|slot| BatchBuf::new(&[rows], &[slot]).unwrap()).collect();
    let mut pools = DevPools::new(&HostPools::zeros(LAYERS));
    let mut ws = DevBuf::zeroed(prefill_workspace_bytes(rows)).unwrap();
    let mut o = DevBuf::zeroed(rows * VH * D * 2).unwrap();
    let prefill_s = timed(&ctx.st, 3, |_| {
        for batch in &batches {
            prefill(&ctx.st, &x.view(), &ctx.weights(), &mut pools.s, &mut pools.conv, &batch.view(), &mut ws, &mut o).unwrap();
        }
    });

    let decode_rows = 8;
    let decode_inp = Inp::random(&mut rng, decode_rows);
    let decode_x = DevInp::new(&decode_inp);
    let decode_batches: Vec<BatchBuf> = (0..LAYERS).map(|slot| BatchBuf::new(&[decode_rows], &[slot]).unwrap()).collect();
    let mut decode_pools = DevPools::new(&HostPools::zeros(LAYERS));
    let mut rec = Record::alloc(decode_rows).unwrap();
    let mut decode_o = DevBuf::zeroed(decode_rows * VH * D * 2).unwrap();
    pre(&ctx.st, &decode_x.view(), &ctx.weights(), &decode_pools.conv, &decode_batches[0].view(), &mut rec).unwrap();
    let decode_s = timed(&ctx.st, 20, |_| {
        for batch in &decode_batches {
            window(&ctx.st, &rec, &decode_pools.s, &batch.view(), &mut decode_o).unwrap();
        }
    });
    let row0 = DevBuf::from_slice(&[0i32]).unwrap();
    let keep = DevBuf::from_slice(&[decode_rows as i32]).unwrap();
    let owed = Owed { rec: &rec, row0: &row0, keep: &keep };
    let owed_s = timed(&ctx.st, 20, |_| {
        for batch in &decode_batches {
            window_owed(&ctx.st, &rec, &owed, &mut decode_pools.s, &batch.view(), &mut decode_o).unwrap();
        }
    });

    eprintln!(
        "L40S GDN {LAYERS} layers | prefill {rows} rows: {:.3} ms ({:.3} ms/layer; 86.5 ms baseline: {:.2}x) | decode {decode_rows} rows: {:.3} ms ({:.2} us/layer; 1.35 ms baseline: {:.2}x) | decode with owed commit: {:.3} ms ({:.2} us/layer)",
        prefill_s * 1e3,
        prefill_s * 1e3 / LAYERS as f64,
        86.5 / (prefill_s * 1e3),
        decode_s * 1e3,
        decode_s * 1e6 / LAYERS as f64,
        1.35 / (decode_s * 1e3),
        owed_s * 1e3,
        owed_s * 1e6 / LAYERS as f64,
    );
}
