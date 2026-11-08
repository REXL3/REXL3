//! The DeepSeek-V4.1 prefix state digest on the real model (`Dsv41Model::prefix_state_digest`), the check a two-node group runs per
//! rank on every restore. GPU and pack gated: `REXL3_DSV41_DIR` names a DeepSeek-V4.1-Flash pack (the experts that do not fit stream
//! from it); without it every test prints `not run` and passes.
//!
//! * the digest of a slot's state after a snapshot equals its digest after the restore of that snapshot into a slot that held another
//!   prompt, and the warm continuation equals the cold one bit for bit;
//! * a slot that holds other (stale) state, and a restore from a snapshot one byte of which was changed, digest differently;
//! * the cold fallback after a failed restore (`begin_slot`, then the prompt from 0) equals a fresh cold prefill bit for bit: whatever
//!   a restore wrote, `begin_slot` and the prefill rewrite;
//! * the digest's cost at the 1M profile's geometry (report only).
//!
//! `REXL3_DSV41_DIR=... cargo test --release -p gb-model --test dsv41_prefix_digest -- --ignored --nocapture --test-threads=1`
use gb_engine::{Backend, Captured};
use gb_model::families::deepseek_v41::backend::Dsv41Backend;
use gb_model::families::deepseek_v41::engine::{build_engine, Dsv41Engine};
use gb_model::families::deepseek_v41::model::Dsv41Model;
use gb_model::serve::{PrefixCacheOption, ServeOptions, SpecMode};
use std::path::PathBuf;
use std::time::Instant;

fn pack() -> Option<PathBuf> {
    std::env::var_os("REXL3_DSV41_DIR").map(PathBuf::from).filter(|p| p.join("config.json").exists())
}

fn engine(max_model_len: usize) -> Option<Dsv41Engine> {
    let Some(dir) = pack() else {
        eprintln!("not run: REXL3_DSV41_DIR is not set to a DeepSeek-V4.1 pack");
        return None;
    };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    let mut o = ServeOptions::new(dir);
    o.slots = 1;
    o.max_model_len = max_model_len;
    o.vision = None;
    o.spec = SpecMode::None;
    o.drafting = false;
    o.spec_tree = false;
    o.prefix_cache = PrefixCacheOption::On;
    Some(build_engine(&o).unwrap_or_else(|e| panic!("cannot build the engine: {e}")))
}

fn toks(seed: u64, n: usize) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1000 + (x % 100_000) as u32
        })
        .collect()
}

/// The prompt from `from` to its end through the backend, as the engine drives it (`prefill_slice_cuts`: its slices, its window rows,
/// a snapshot at each of `cuts`): the last row's candidates and the snapshots taken.
fn prefill(b: &mut Dsv41Backend<Dsv41Model>, prompt: &[u32], from: usize, cuts: &[usize]) -> (Vec<(u32, u32)>, Vec<Captured>) {
    let (mut done, mut caps, mut row) = (from, Vec::new(), None);
    while done < prompt.len() {
        let s = b.prefill_slice_cuts(0, prompt, &[], done, 8, cuts);
        assert!(s.step.done > done, "the prefill advances");
        caps.extend(s.captured);
        done = s.step.done;
        row = s.step.row;
    }
    assert!(b.take_error().is_none(), "the prefill ran clean");
    let r = row.expect("the last slice has a row");
    (r.ids.iter().zip(&r.values).map(|(i, v)| (*i, v.to_bits())).collect(), caps)
}

#[test]
#[ignore = "GPU and REXL3_DSV41_DIR"]
fn the_prefix_state_digest_is_equal_after_a_restore_and_differs_for_stale_or_altered_state() {
    let Some(mut e) = engine(16_384) else { return };
    let b = &mut e.backend;
    let (p, q, other) = (toks(1, 1536), toks(2, 256), toks(3, 1800));
    let pq = [p.clone(), q.clone()].concat();
    let n = p.len();
    // the cold reply: the whole prompt from 0
    let (cold, _) = prefill(b, &pq, 0, &[]);
    // a snapshot of the first `n` tokens (a cut, as the engine plans one), its digest; then a snapshot of another prompt
    let (_, caps) = prefill(b, &pq, 0, &[n]);
    let hp = caps.into_iter().find(|c| c.len == n).expect("the cut was captured").handle;
    let (_, _) = prefill(b, &other, 0, &[other.len() - 64]);
    // the snapshot's digest: restore it into the slot, digest, and again (deterministic)
    b.prefix_restore(0, &hp, n).unwrap();
    let d_snap = b.m.prefix_state_digest(0, n).unwrap();
    assert_eq!(d_snap, b.m.prefix_state_digest(0, n).unwrap(), "deterministic");
    // the slot holding another prompt (what a stale restore leaves) differs
    let (_, caps_o) = prefill(b, &other, 0, &[other.len() - 64]);
    assert_ne!(b.m.prefix_state_digest(0, n).unwrap(), d_snap, "a slot holding another prompt (a stale restore) differs");
    // the restore writes back exactly what was captured, and the warm continuation equals the cold reply
    b.prefix_restore(0, &hp, n).unwrap();
    assert_eq!(b.m.prefix_state_digest(0, n).unwrap(), d_snap, "after the restore, the digest of the snapshot");
    let (warm, _) = prefill(b, &pq, n, &[]);
    assert_eq!(warm, cold, "the warm continuation equals the cold reply bit for bit");
    // the capture right after a prefill digests as its restore does (what a group compares: snapshot time vs restore time)
    let (_, caps2) = prefill(b, &pq, 0, &[n]);
    let h2 = caps2.into_iter().find(|c| c.len == n).unwrap().handle;
    let d_at_capture = {
        // the slot went on past the cut: digest the capture by restoring it, as the slot was at the cut
        b.prefix_restore(0, &h2, n).unwrap();
        b.m.prefix_state_digest(0, n).unwrap()
    };
    assert_eq!(d_at_capture, d_snap, "the same prefix captured twice digests the same");
    // an altered snapshot (one byte of one layer's window) is seen after its restore
    if let Some(s) = b.m.snaps.slots[hp.snap as usize].as_mut() {
        s.windows[3][77] ^= 0x01;
    }
    b.prefix_restore(0, &hp, n).unwrap();
    assert_ne!(b.m.prefix_state_digest(0, n).unwrap(), d_snap, "one altered byte of the restored state differs");
    // the cold fallback after a failed restore: the slot holds a restore's (wrong) state, then the prompt from 0 begins it again
    let ho = caps_o.into_iter().next().unwrap().handle;
    b.prefix_restore(0, &ho, other.len() - 64).unwrap();
    let (fallback, _) = prefill(b, &pq, 0, &[]);
    assert_eq!(fallback, cold, "the cold fallback over a restored slot equals a fresh cold prefill bit for bit");
    // the cost at this geometry (report)
    b.prefix_restore(0, &h2, n).unwrap();
    let t = Instant::now();
    for _ in 0..10 {
        b.m.prefix_state_digest(0, n).unwrap();
    }
    eprintln!("prefix_state_digest len={n}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / 10.0);
    for h in [hp, h2, ho] {
        b.prefix_release(h);
    }
}

/// The digest's cost per hit at the 1M profile's geometry (a 1,048,576-token window, planes in pageable host memory): a snapshot
/// and a restore of a 1,040,000-token prefix (whatever the slot holds), and the digest after each. Report only.
#[test]
#[ignore = "GPU and REXL3_DSV41_DIR; allocates a 1M-token slot and a ~1 GiB host snapshot"]
fn the_prefix_state_digest_cost_at_the_1m_geometry() {
    let Some(mut e) = engine(1_048_576) else { return };
    let len = 1_040_000;
    let _ = prefill(&mut e.backend, &toks(9, 4096), 0, &[]);
    let m = &mut e.backend.m;
    let snap = m.alloc_snapshot().unwrap();
    let t = Instant::now();
    m.prefix_snapshot(0, snap, len).unwrap();
    let snap_ms = t.elapsed().as_secs_f64() * 1e3;
    let bytes: usize = {
        let s = m.snaps.slots[snap as usize].as_ref().unwrap();
        s.kv_sources.iter().map(|k| k.compress.len() + k.index_k.len()).sum::<usize>()
            + s.windows.iter().map(Vec::len).sum::<usize>()
            + s.mtp_taps.len()
    };
    let t = Instant::now();
    let d0 = m.prefix_state_digest(0, len).unwrap();
    let dig_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    m.restore_prefix(0, snap, len).unwrap();
    let rest_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let d1 = m.prefix_state_digest(0, len).unwrap();
    let dig2_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(d0, d1, "the restore of a 1M-token snapshot digests as its capture did");
    eprintln!(
        "1M geometry len={len}: snapshot {:.2} GiB host copy {snap_ms:.1} ms; digest {dig_ms:.1} ms; restore {rest_ms:.1} ms; digest after restore {dig2_ms:.1} ms",
        bytes as f64 / (1u64 << 30) as f64
    );
    m.release_prefix(snap);
}
