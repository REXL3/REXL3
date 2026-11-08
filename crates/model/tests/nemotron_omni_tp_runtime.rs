//! TP-R on the real model: the lockstep runtime (`families::nemotron_h_omni::tp_runtime`) over two layer-limited Nemotron models in one
//! process and one device, a leader and a follower on a socket pair, driven by the real `NemotronBackend` (the engine's own call
//! pattern). Needs a GPU and the pack; no second node and no collectives (the shards are whole models: this proves the RUNTIME, not the
//! sharding).
//!
//! * `the_leader_changes_no_bit`: the same prompts and windows through `NemotronBackend<NemotronModel>` and through
//!   `NemotronBackend<TpLeader<NemotronModel>>` give identical candidate ids and value bits.
//! * the follower, a second model that only ever sees the call stream, reports a digest of every candidate block it computed; the leader
//!   compares it with its own, so every prefill slice and window of the run was checked to be bit-equal across the two models (a mismatch
//!   fails the group, and this test).
//! * `a_follower_fed_another_stream_is_caught`: a follower whose model differs in one call (an extra begin/commit the leader never sent)
//!   is a diverged group at the next compared result, named by call.
//!
//! Env: `REXL3_NEMO_OMNI_DIR` (the pack), `REXL3_NEMO_ORACLE_DIR` (real text; random tokens without), `REXL3_NEMO_TP_LAYERS` (default 8).
//! `#[ignore]`, `--test-threads=1`, under the GPU lock.
#![allow(clippy::ptr_arg)]

mod nemotron_omni_mtp_support;

use gb_engine::{Backend, ImageSpan, RowOutput, RowRequest};
use gb_model::families::nemotron_h_omni::backend::{ModelOptions, NemotronBackend};
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::model::NemotronModel;
use gb_model::families::nemotron_h_omni::tp_runtime::{
    run_follower, DeviceFeatures, FeatureBridge, FollowerConfig, FollowerExit, Link, TpLeader, TpTimeouts,
};
use gb_model::families::nemotron_h_omni::vision::{NemotronImageFeatures, SpliceOp};
use gb_model::families::nemotron_h_omni::weights::LayerPlan;
use gb_model::rows::RowBatch;
use nemotron_omni_mtp_support::{config, oracle_ids, pack_dir};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

const CANDS: usize = 8;

fn layers() -> usize {
    std::env::var("REXL3_NEMO_TP_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
}

fn opts() -> ModelOptions {
    ModelOptions { slots: 3, kv_pages: 64, max_context: 2048, prefill_rows: 64, layers: Some(layers()), mtp: false }
}

fn load(dir: &PathBuf) -> NemotronModel {
    let cfg = config(dir);
    let inv = Inventory::read(dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    NemotronModel::load(dir, &cfg, &plan, opts()).unwrap_or_else(|e| panic!("load: {e}"))
}

fn timeouts() -> TpTimeouts {
    TpTimeouts { peer: Duration::from_secs(30), hang_exit: None, ..TpTimeouts::default() }
}

/// What the run produced, as bits: for every returned row its candidate ids and value bits.
type Rows = Vec<(Vec<u32>, Vec<u32>)>;

fn bits(r: &RowOutput) -> (Vec<u32>, Vec<u32>) {
    (r.ids.clone(), r.values.iter().map(|v| v.to_bits()).collect())
}

fn tokens() -> (Vec<u32>, Vec<u32>) {
    let base = oracle_ids().unwrap_or_else(|| {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        (0..2000)
            .map(|_| {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as u32 % 100_000 + 1000
            })
            .collect()
    });
    (base[..150].to_vec(), base[400..560].to_vec())
}

/// The engine's call pattern for two sequences: prefill both (the second in 64-row slices from its start), then windows of one and of
/// several rows over both, with commits that keep fewer rows than the window had, then release.
fn script<B: Backend>(b: &mut B) -> Rows {
    let (p0, p1) = tokens();
    let mut out: Rows = Vec::new();
    let r0 = b.prefill(0, 0, &p0, CANDS);
    out.push(bits(&r0));
    let r1 = b.prefill(1, 0, &p1, CANDS);
    out.push(bits(&r1));
    let (mut n0, mut n1) = (p0.len(), p1.len());
    let (mut t0, mut t1) = (r0.ids[0], r1.ids[0]);
    for step in 0..6usize {
        // a window of one row on slot 0 and of four on slot 1, then the same slots with the draft-shaped windows
        let rows = if step % 2 == 0 {
            vec![
                RowRequest { slot: 0, first_position: n0 as u64, tokens: vec![t0] },
                RowRequest { slot: 1, first_position: n1 as u64, tokens: vec![t1, t1 + 1, t1 + 2, t1 + 3] },
            ]
        } else {
            vec![RowRequest { slot: 1, first_position: n1 as u64, tokens: vec![t1, t1 + 5] }]
        };
        let res = b.step(&rows, CANDS);
        assert_eq!(res.len(), rows.len());
        for (req, per_row) in rows.iter().zip(&res) {
            for r in per_row {
                out.push(bits(r));
            }
            // keep one row of a one-row window and two of a longer one
            let keep = if req.tokens.len() == 1 { 1 } else { 2 };
            b.commit(req.slot, keep);
            let next = per_row[keep - 1].ids[0];
            if req.slot == 0 {
                (n0, t0) = (n0 + keep, next);
            } else {
                (n1, t1) = (n1 + keep, next);
            }
        }
    }
    // a slot that is begun again from position 0 replaces its sequence
    let r = b.prefill(1, 0, &p0[..70], CANDS);
    out.push(bits(&r));
    b.release(0);
    b.release(1);
    assert!(b.take_error().is_none(), "{:?}", b.take_error());
    out
}

fn gpu() -> Option<PathBuf> {
    let dir = pack_dir()?;
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    Some(dir)
}

#[test]
#[ignore]
fn the_leader_changes_no_bit_and_the_follower_agrees_on_every_candidate_block() {
    let Some(dir) = gpu() else { return };
    // the reference: the model alone
    let solo = {
        let m = load(&dir);
        let mut b = NemotronBackend::with_runtime(m, 64, &opts()).unwrap();
        script(&mut b)
    };
    // the group: a follower that loads in its own thread and only ever sees the call stream, and the leader over its own model
    let (a, link_b) = Link::pair(Duration::from_secs(30)).unwrap();
    let dir_f = dir.clone();
    let follower = std::thread::spawn(move || {
        let m = load(&dir_f);
        run_follower(m, link_b, FollowerConfig::new(timeouts()))
    });
    let leader = TpLeader::start(load(&dir), a, timeouts());
    let mut b = NemotronBackend::with_runtime(leader, 64, &opts()).unwrap();
    let group = script(&mut b);
    b.m.barrier().unwrap_or_else(|e| panic!("the group is not in agreement: {e}"));
    assert_eq!(b.m.failure(), None);
    let stats = b.m.stats();
    eprintln!("tp runtime: {stats:?}");
    assert!(stats.commands > 20 && stats.acked >= stats.commands - 1, "{stats:?}");
    assert_eq!(stats.tolerated, 0);
    drop(b);
    match follower.join().unwrap() {
        Ok(FollowerExit::Shutdown(_)) => {}
        other => panic!("{other:?}"),
    }
    assert_eq!(solo.len(), group.len());
    for (i, (s, g)) in solo.iter().zip(&group).enumerate() {
        assert_eq!(s, g, "row {i} of the run differs between the model alone and the leader");
    }
}

#[test]
#[ignore]
fn a_follower_fed_another_stream_is_caught_at_the_call_that_shows_it() {
    let Some(dir) = gpu() else { return };
    let (a, link_b) = Link::pair(Duration::from_secs(30)).unwrap();
    let dir_f = dir.clone();
    // a follower whose model was also given a private prompt in slot 1: it holds a sequence the leader never began
    let follower = std::thread::spawn(move || {
        let mut m = load(&dir_f);
        m.begin_slot(1).unwrap();
        let b = RowBatch::from_requests(&[RowRequest { slot: 1, first_position: 0, tokens: vec![5, 6, 7] }]);
        m.prefill_chunk(&b, None).unwrap();
        run_follower(m, link_b, FollowerConfig::new(timeouts()))
    });
    let leader = TpLeader::start(load(&dir), a, timeouts());
    let mut b = NemotronBackend::with_runtime(leader, 64, &opts()).unwrap();
    let (p0, _) = tokens();
    // the leader begins slot 1: it succeeds here and the follower refuses a slot that is already begun, so the group is lost at that call
    let _ = b.prefill(1, 0, &p0, CANDS);
    let r = b.m.barrier();
    let why = b.m.failure();
    eprintln!("barrier: {r:?}; failure: {why:?}");
    assert!(r.is_err(), "a diverged follower went unnoticed");
    let why = why.expect("the group's failure is latched");
    assert!(why.contains("begin_slot"), "{why}");
    drop(b);
    let _ = follower.join();
}

/// bf16 bits of a deterministic stream of small values (what an encoder's rows look like to the model: finite, order of one).
fn bf16_rows(rows: usize, hidden: usize, seed: u32) -> Vec<u16> {
    let mut s = seed.wrapping_mul(2_654_435_761) | 1;
    (0..rows * hidden)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let v = ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0;
            (v.to_bits() >> 16) as u16
        })
        .collect()
}

#[test]
#[ignore]
fn device_features_round_trip_the_rows_of_a_splice() {
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    const ROW: usize = 4096;
    let tokens = 7;
    let data = bf16_rows(tokens, ROW, 7);
    let feats = Arc::new(NemotronImageFeatures {
        buf: gb_cuda::DevBuf::from_slice(&data).unwrap(),
        tokens,
        merged: (7, 1),
        permit: None,
    });
    let op = SpliceOp { feats: feats.clone(), dst_row: 3, src_row: 2, n: 4 };
    let bytes = DeviceFeatures.export(&op).unwrap();
    let want: Vec<u8> = data[2 * ROW..6 * ROW].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(bytes, want, "the exported bytes are rows 2..6 of the features");
    let back = DeviceFeatures.import(3, 4, bytes).unwrap();
    assert_eq!((back.dst_row, back.src_row, back.n, back.feats.tokens), (3, 0, 4, 4));
    let got: Vec<u16> = back.feats.buf.to_vec().unwrap();
    assert_eq!(got, data[2 * ROW..6 * ROW], "the imported buffer holds exactly those rows");
    // rows past the features are refused by name, not read
    assert!(DeviceFeatures.export(&SpliceOp { feats, dst_row: 0, src_row: 5, n: 4 }).is_err());
}

/// Image rows spliced into a prompt that is sliced inside the image: the follower rebuilds the features from the bytes it is sent and the
/// two ranks agree on every candidate block (the ledger fails the group otherwise), and the leader answers as the model alone does.
#[test]
#[ignore]
fn image_rows_reach_the_follower_and_every_candidate_block_agrees() {
    let Some(dir) = gpu() else { return };
    const IMG: u32 = 18; // `img_context_token_id` of the pack
    let (text, _) = tokens();
    let mut prompt: Vec<u32> = text[..100].iter().map(|&t| if t == IMG { IMG + 1 } else { t }).collect();
    let (start, n) = (20usize, 32usize);
    for t in &mut prompt[start..start + n] {
        *t = IMG;
    }
    let make_span = || {
        let feats = Arc::new(NemotronImageFeatures {
            buf: gb_cuda::DevBuf::from_slice(&bf16_rows(n, 4096, 11)).unwrap(),
            tokens: n,
            merged: (n / 4, 4),
            permit: None,
        });
        ImageSpan { start, tokens: n, merged: (n / 4, 4), features: feats, key: 0 }
    };
    // 32-row slices: the image starts inside the first slice and ends inside the second
    let run = |b: &mut dyn Backend| -> Rows {
        let r = b.prefill_mm(0, 0, &prompt, &[make_span()], CANDS);
        let mut out = vec![bits(&r)];
        let rows = b.step(&[RowRequest { slot: 0, first_position: prompt.len() as u64, tokens: vec![r.ids[0]] }], CANDS);
        out.push(bits(&rows[0][0]));
        b.commit(0, 1);
        b.release(0);
        assert!(b.take_error().is_none(), "{:?}", b.take_error());
        out
    };
    let solo = {
        let mut b = NemotronBackend::with_runtime(load(&dir), 32, &opts()).unwrap().with_image_token(IMG);
        run(&mut b)
    };
    let (a, link_b) = Link::pair(Duration::from_secs(30)).unwrap();
    let dir_f = dir.clone();
    let follower = std::thread::spawn(move || {
        let m = load(&dir_f);
        run_follower(m, link_b, FollowerConfig::new(timeouts()))
    });
    let leader = TpLeader::start(load(&dir), a, timeouts());
    let mut b = NemotronBackend::with_runtime(leader, 32, &opts()).unwrap().with_image_token(IMG);
    let group = run(&mut b);
    b.m.barrier().unwrap_or_else(|e| panic!("the group is not in agreement: {e}"));
    let stats = b.m.stats();
    eprintln!("tp runtime (image run): {stats:?}");
    // the image's rows cross once: 32 rows of bf16 [4096] are 256 KiB
    assert!(stats.bytes > (n * 4096 * 2) as u64, "{stats:?}");
    assert_eq!(b.m.failure(), None);
    drop(b);
    assert!(matches!(follower.join().unwrap(), Ok(FollowerExit::Shutdown(_))));
    assert_eq!(solo, group, "the leader changed a bit of an image prompt");
}

// ------------------------------------------------------------------------------------------------------------------ R sharded ranks

/// Group size of the N-rank tests below: `REXL3_NEMO_TP_RANKS`, 3 by default (the point is a group of more than two).
fn group_ranks() -> usize {
    let r: usize = std::env::var("REXL3_NEMO_TP_RANKS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(3);
    assert!((2..=8).contains(&r), "REXL3_NEMO_TP_RANKS must be 2 to 8, got {r}");
    r
}

/// Runs `drive` against the leader of a sharded group of `ranks`: R `NemotronModel` shards of the serving plan (`load_tp`, each its own
/// rank of an in-process `LoopbackGroup`, so every sharded linear, the experts and the `lm_head` run their collectives), rank 0 under a
/// `TpLeader` over R - 1 socket pairs behind the backend (`chunk` rows a prefill slice; `image_token`: the id image rows are spliced
/// at; `min_commands` the least the run must have sent), ranks 1.. in `run_follower` on their own threads. Returns what `drive`
/// returned, after `barrier()` (every follower's candidate digest agreed with rank 0's on every block: the ledger fails the group
/// otherwise); every follower must exit on the leader's Shutdown.
fn run_sharded_group<T>(
    dir: &Path,
    ranks: usize,
    chunk: usize,
    image_token: Option<u32>,
    min_commands: u64,
    drive: impl FnOnce(&mut NemotronBackend<TpLeader<NemotronModel>>) -> T,
) -> T {
    use gb_model::families::nemotron_h_omni::tp::TpComm;
    use gb_model::families::nemotron_h_omni::tp_comm::{HostBarrier, LoopbackGroup};
    use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

    let cfg = config(dir);
    let inv = Inventory::read(dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let mut comms = LoopbackGroup::new(ranks).into_iter().map(|c| Arc::new(c) as Arc<dyn TpComm>);
    let comm0 = comms.next().expect("rank 0");
    let ready = Arc::new(HostBarrier::new(ranks));
    std::thread::scope(|s| {
        let mut links = Vec::new();
        let mut followers = Vec::new();
        for (i, comm) in comms.enumerate() {
            let rank = i + 1;
            let (leader_side, follower_side) = Link::pair(Duration::from_secs(30)).unwrap();
            links.push(leader_side);
            let (dir, cfg, plan, ready) = (dir, &cfg, &plan, ready.clone());
            followers.push(s.spawn(move || {
                gb_cuda::rt::init_context().unwrap();
                // a rank that fails to load must not leave the others waiting at the barrier
                let loaded = catch_unwind(AssertUnwindSafe(|| {
                    NemotronModel::load_tp(dir, cfg, plan, opts(), comm).unwrap_or_else(|e| panic!("rank {rank}: load: {e}"))
                }));
                let mut m = match loaded {
                    Ok(m) => m,
                    Err(p) => {
                        ready.break_now();
                        resume_unwind(p)
                    }
                };
                // the loopback group synchronises its stream and the host inside a collective: not capturable, graphs off
                m.set_graphs(false);
                ready.wait(Duration::from_secs(600)).expect("the group did not finish loading");
                let mut fc = FollowerConfig::new(timeouts());
                fc.rank = rank;
                run_follower(m, follower_side, fc)
            }));
        }
        let loaded = catch_unwind(AssertUnwindSafe(|| {
            NemotronModel::load_tp(dir, &cfg, &plan, opts(), comm0).unwrap_or_else(|e| panic!("rank 0: load: {e}"))
        }));
        let mut m0 = match loaded {
            Ok(m) => m,
            Err(p) => {
                ready.break_now();
                resume_unwind(p)
            }
        };
        m0.set_graphs(false);
        ready.wait(Duration::from_secs(600)).expect("the group did not finish loading");
        let leader = TpLeader::start_group(m0, links, timeouts());
        let mut b = NemotronBackend::with_runtime(leader, chunk, &opts()).unwrap();
        if let Some(t) = image_token {
            b = b.with_image_token(t);
        }
        let out = drive(&mut b);
        b.m.barrier().unwrap_or_else(|e| panic!("the group is not in agreement: {e}"));
        assert_eq!(b.m.failure(), None);
        let stats = b.m.stats();
        eprintln!("tp runtime R={ranks}: {stats:?}");
        assert!(
            stats.commands >= min_commands && stats.acked >= (stats.commands - 1) * (ranks as u64 - 1),
            "{stats:?} (at least {min_commands} commands, every follower's ack)"
        );
        assert_eq!(stats.tolerated, 0);
        drop(b);
        for (i, f) in followers.into_iter().enumerate() {
            match f.join().unwrap() {
                Ok(FollowerExit::Shutdown(_)) => {}
                other => panic!("rank {}: {other:?}", i + 1),
            }
        }
        out
    })
}

/// The sharded group through the lockstep runtime, driven by the real `NemotronBackend` (the engine's own call pattern): the leader
/// answers as the single-node model does, bit for bit, and every follower agreed with rank 0 on every candidate block. The first run of
/// the N-follower runtime on the real model and sharded weights.
#[test]
#[ignore]
fn an_r_rank_sharded_group_through_the_runtime_changes_no_bit_and_every_follower_agrees() {
    let Some(dir) = gpu() else { return };
    let ranks = group_ranks();
    // the reference: the single-node model alone, dropped before the group loads
    let solo = {
        let m = load(&dir);
        let mut b = NemotronBackend::with_runtime(m, 64, &opts()).unwrap();
        script(&mut b)
    };
    let group = run_sharded_group(&dir, ranks, 64, None, 20, script);
    assert_eq!(solo.len(), group.len());
    for (i, (s, g)) in solo.iter().zip(&group).enumerate() {
        assert_eq!(s, g, "row {i} of the run differs between the single-node model and the {ranks}-rank sharded group");
    }
    eprintln!(
        "TP RUNTIME OK: R={ranks}, {} rows identical to the single-node model, every follower agreed on every block",
        solo.len()
    );
}

/// Image rows spliced into a prompt sliced inside the image, through a sharded group of R: every follower (R - 1 of them, each its own
/// link) rebuilds the features from the bytes it is sent, runs its shard of the splice step's layers, and agrees with rank 0 on every
/// candidate block; the leader answers as the single-node model does.
#[test]
#[ignore]
fn image_rows_reach_every_follower_of_an_r_rank_sharded_group_and_every_block_agrees() {
    let Some(dir) = gpu() else { return };
    const IMG: u32 = 18; // `img_context_token_id` of the pack
    let ranks = group_ranks();
    let (text, _) = tokens();
    let mut prompt: Vec<u32> = text[..100].iter().map(|&t| if t == IMG { IMG + 1 } else { t }).collect();
    let (start, n) = (20usize, 32usize);
    for t in &mut prompt[start..start + n] {
        *t = IMG;
    }
    let make_span = || {
        let feats = Arc::new(NemotronImageFeatures {
            buf: gb_cuda::DevBuf::from_slice(&bf16_rows(n, 4096, 11)).unwrap(),
            tokens: n,
            merged: (n / 4, 4),
            permit: None,
        });
        ImageSpan { start, tokens: n, merged: (n / 4, 4), features: feats, key: 0 }
    };
    let run = |b: &mut dyn Backend| -> Rows {
        let r = b.prefill_mm(0, 0, &prompt, &[make_span()], CANDS);
        let mut out = vec![bits(&r)];
        let rows = b.step(&[RowRequest { slot: 0, first_position: prompt.len() as u64, tokens: vec![r.ids[0]] }], CANDS);
        out.push(bits(&rows[0][0]));
        b.commit(0, 1);
        b.release(0);
        assert!(b.take_error().is_none(), "{:?}", b.take_error());
        out
    };
    let solo = {
        let mut b = NemotronBackend::with_runtime(load(&dir), 32, &opts()).unwrap().with_image_token(IMG);
        run(&mut b)
    };
    let group = run_sharded_group(&dir, ranks, 32, Some(IMG), 5, |b| run(b));
    assert_eq!(solo, group, "the sharded group of {ranks} changed a bit of an image prompt");
    eprintln!("TP RUNTIME IMAGE OK: R={ranks}, the image prompt's rows identical, every follower agreed");
}
