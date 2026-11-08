//! Two-process test of the ONE-SHOT collective (`NcclComm::oneshot_sum_f32`: `ncclGroupStart`, `ncclSend` + `ncclRecv` per peer,
//! `ncclGroupEnd`, then one rank-ordered sum kernel) against the ring allreduce, across two nodes, eager and inside a CUDA graph.
//! Same environment as `nccl_pair.rs` (one process per rank, each under its node's GPU lock; unset it prints `not run:` and passes):
//!
//! ```text
//! REXL3_NCCL_RANK=0|1  REXL3_NCCL_RANKS=2  REXL3_NCCL_PEER=192.0.2.1:29500   (rank 0's address: it listens, rank 1 connects)
//! NCCL fabric settings: source /srv/rexl3/nccl/nccl.env, and NCCL_MAX_NCHANNELS=4
//! optional: REXL3_NCCL_TIMEOUT_S (watchdog, default 600), REXL3_ONESHOT_PHASES=eager,latency,capture,step (default all)
//! ```
//!
//! Phases, on both ranks:
//!
//! * `eager`: one-shot and ring over disjoint-support (exact -0.0 filler, subnormals and signed zeros in the data) and dense buffers,
//!   4 B to 4 MB, byte offsets 0 and 256: both are bit-equal to the host's `a + b` and to each other, and the bytes around the range are
//!   untouched.
//! * `latency`: per collective, eager launches back to back and graph replay, ring against one-shot, 4 KB to 4 MB (the crossover at R = 2).
//! * `capture`: THE PROOF that send/recv can be captured on this fabric: 100 one-shot collectives of 16 KB and of 440 KiB, each on its
//!   own region of a buffer, captured into ONE CUDA graph and replayed 1,000 times, the result checked at every 100th replay and on 20
//!   fresh seeds, bit-equal to the ring's result (the same structure, ring) and to the host sum. A capture that fails is reported
//!   precisely (exit status 7); a replay that deadlocks ends at the watchdog (status 3).
//! * `step`: the collectives of one engine step of the serving plan (81: 40 `in_proj`, 40 MoE `y`, the head) as one graph, per step,
//!   at 1, 5, 16 and 64 rows: all ring, all one-shot, and `auto` (one-shot up to 700 KiB, ring above).
//!
//! The doc is `docs/NEMOTRON_TP.md` section 10.6.
#![allow(clippy::needless_range_loop)]

use gb_cuda::graph::Graph;
use gb_cuda::nccl::{NcclComm, NcclUniqueId};
use gb_cuda::{device_available, CudaError, DevBuf, Stream};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A finite f32 with a random sign and mantissa; every fifth a SUBNORMAL, every 23rd an exact signed zero (see `nccl_pair.rs`).
fn gen_f32(state: &mut u64) -> f32 {
    let r = splitmix(state);
    let mant = (r & 0x7F_FFFF) as u32;
    let sign = ((r >> 40) & 1) as u32;
    if (r >> 52) % 23 == 0 {
        return f32::from_bits(sign << 31);
    }
    if (r >> 48) % 5 == 0 {
        return f32::from_bits(sign << 31 | (mant | 1));
    }
    let exp = 127 - 8 + ((r >> 32) % 16) as u32;
    f32::from_bits(sign << 31 | exp << 23 | mant)
}

fn vector(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| gen_f32(&mut s)).collect()
}

fn owner(i: usize) -> usize {
    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 63) as usize
}

/// This rank's input and the host's expected sum of one case (`disjoint`: own elements and exact -0.0 elsewhere).
fn case(seed: u64, n: usize, rank: usize, disjoint: bool) -> (Vec<f32>, Vec<f32>) {
    let a = vector(seed, n);
    let b = vector(seed ^ 0xA5A5_A5A5, n);
    let (a, b): (Vec<f32>, Vec<f32>) = if disjoint {
        (
            (0..n).map(|i| if owner(i) == 0 { a[i] } else { -0.0 }).collect(),
            (0..n).map(|i| if owner(i) == 1 { b[i] } else { -0.0 }).collect(),
        )
    } else {
        (a, b)
    };
    let sum = (0..n).map(|i| a[i] + b[i]).collect();
    (if rank == 0 { a } else { b }, sum)
}

fn first_mismatch(got: &[f32], want: &[f32]) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!("length {} want {}", got.len(), want.len()));
    }
    got.iter().zip(want).position(|(g, w)| g.to_bits() != w.to_bits()).map(|i| {
        format!("element {i}: got {:#010x} ({}) want {:#010x} ({})", got[i].to_bits(), got[i], want[i].to_bits(), want[i])
    })
}

struct Side {
    sock: TcpStream,
}

impl Side {
    /// Both ranks call it at the same point; neither returns before the other has arrived.
    fn barrier(&mut self) {
        self.sock.write_all(&[0x42]).expect("lockstep write");
        let mut b = [0u8; 1];
        self.sock.read_exact(&mut b).expect("lockstep read");
        assert_eq!(b[0], 0x42);
    }
}

fn rendezvous(rank: usize, peer: &str, id: Option<&NcclUniqueId>) -> (Side, NcclUniqueId) {
    if rank == 0 {
        let port = peer.rsplit(':').next().expect("REXL3_NCCL_PEER is HOST:PORT");
        let l = TcpListener::bind(format!("0.0.0.0:{port}")).expect("bind the rendezvous port");
        let (mut s, from) = l.accept().expect("accept the peer");
        eprintln!("nccl_oneshot_pair rank 0: peer connected from {from}");
        s.set_nodelay(true).ok();
        let id = id.expect("rank 0 has the id");
        s.write_all(id.as_bytes()).expect("send the id");
        (Side { sock: s }, id.clone())
    } else {
        let t0 = Instant::now();
        let mut s = loop {
            match TcpStream::connect(peer) {
                Ok(s) => break s,
                Err(e) if t0.elapsed() < Duration::from_secs(120) => {
                    eprintln!("nccl_oneshot_pair rank 1: waiting for rank 0 at {peer}: {e}");
                    std::thread::sleep(Duration::from_secs(2));
                }
                Err(e) => panic!("rank 1 could not reach rank 0 at {peer}: {e}"),
            }
        };
        s.set_nodelay(true).ok();
        let mut b = [0u8; 128];
        s.read_exact(&mut b).expect("receive the id");
        (Side { sock: s }, NcclUniqueId(b))
    }
}

fn ck<T>(r: Result<T, CudaError>, what: &str) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{what}: {e}"),
    }
}

/// Which algorithm one call of [`run_coll`] uses.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Algo {
    Ring,
    OneShot,
    /// One-shot up to 700 KiB, ring above (the `auto` policy of `tp_comm.rs`).
    Auto,
}

fn algo_name(a: Algo) -> &'static str {
    match a {
        Algo::Ring => "ring",
        Algo::OneShot => "oneshot",
        Algo::Auto => "auto",
    }
}

fn run_coll(comm: &NcclComm, algo: Algo, buf: &mut DevBuf, off: usize, count: usize, st: &Stream) -> Result<(), CudaError> {
    let one = match algo {
        Algo::Ring => false,
        // forced one-shot: every message the staging holds (the policy's `oneshot` mode), the ring above
        Algo::OneShot => count * 4 <= comm.oneshot_capacity_bytes(),
        Algo::Auto => count * 4 <= 700 * 1024,
    };
    if one {
        comm.oneshot_sum_f32(buf, off, count, st)
    } else {
        comm.allreduce_sum_f32(buf, off, count, st)
    }
}

/// Capture `f` on `st` into a graph, or report the failure precisely and end the process with status 7 (the one place a failure to
/// capture a send/recv is a finding and not a test bug).
fn capture(st: &Stream, rank: usize, what: &str, f: impl FnOnce() -> Result<(), CudaError>) -> Graph {
    let cap = ck(Graph::begin(st), "begin capture");
    let ran = f();
    match cap.finish(ran) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("nccl_oneshot_pair rank {rank}: ONESHOT CAPTURE FAILED ({what}): {e}");
            std::process::exit(7);
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn phase_wanted(name: &str) -> bool {
    match std::env::var("REXL3_ONESHOT_PHASES") {
        Ok(v) if !v.trim().is_empty() => v.split(',').any(|p| p.trim() == name),
        _ => true,
    }
}

#[test]
fn nccl_oneshot_pair_eager_captured_and_timed() {
    let Ok(rank) = std::env::var("REXL3_NCCL_RANK") else {
        eprintln!("not run: REXL3_NCCL_RANK is not set");
        return;
    };
    let rank: usize = rank.parse().expect("REXL3_NCCL_RANK is 0 or 1");
    let ranks: usize = std::env::var("REXL3_NCCL_RANKS").ok().map(|v| v.parse().expect("REXL3_NCCL_RANKS")).unwrap_or(2);
    let peer = std::env::var("REXL3_NCCL_PEER").expect("REXL3_NCCL_PEER=HOST:PORT (rank 0's address)");
    assert_eq!(ranks, 2, "this test is the two-rank pair");
    assert!(rank < ranks);
    if !device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    let timeout: u64 = std::env::var("REXL3_NCCL_TIMEOUT_S").ok().and_then(|v| v.parse().ok()).unwrap_or(600);
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(timeout));
        eprintln!("nccl_oneshot_pair: watchdog expired after {timeout} s (a deadlock?), exiting with status 3");
        std::process::exit(3);
    });

    ck(gb_cuda::rt::init_context(), "init_context");
    let st = ck(Stream::new_nonblocking(), "compute stream");
    eprintln!(
        "nccl_oneshot_pair rank {rank}: libnccl {} version code {}; env {}",
        NcclComm::library_path(),
        ck(NcclComm::version(), "ncclGetVersion"),
        gb_cuda::nccl::nccl_environment().join(" ")
    );
    let id = if rank == 0 { Some(ck(NcclComm::new_unique_id(), "ncclGetUniqueId")) } else { None };
    let (mut side, id) = rendezvous(rank, &peer, id.as_ref());
    let t0 = Instant::now();
    let comm = ck(NcclComm::init(rank, ranks, &id), "ncclCommInitRank");
    eprintln!("nccl_oneshot_pair rank {rank}: communicator up in {:.2} s", t0.elapsed().as_secs_f64());
    // staging for the largest message of any phase: the logits of 16 rows (8.4 MB) and MoE y of 64 rows (5.8 MB) are in `step`,
    // 4 MB in `eager`; 24 MiB is the policy's forced limit (MoE y at 256 rows)
    let staging = ck(comm.reserve_oneshot_staging(24 << 20), "reserve staging");
    eprintln!("nccl_oneshot_pair rank {rank}: one-shot staging {} MB", staging as f64 / 1e6);
    side.barrier();

    // connect both kinds before anything is captured (NCCL connects peers lazily on the first collective of each kind)
    {
        let mut b = ck(DevBuf::zeroed(4096 * 4), "alloc");
        ck(gb_cuda::rt::device_sync(), "device sync");
        ck(comm.allreduce_sum_f32(&mut b, 0, 4096, &st), "connect ring");
        ck(comm.oneshot_sum_f32(&mut b, 0, 4096, &st), "connect one-shot");
        ck(st.sync(), "sync");
        side.barrier();
    }

    const CANARY: f32 = f32::from_bits(0x7FC0_1234);
    // ---- eager: one-shot against ring against the host --------------------------------------------------------------------------
    if phase_wanted("eager") {
        let counts = [1usize, 7, 1021, 1024, 4096, 16384, 22528, 65536, 112_640, 262144, 524_288, 1048576];
        let mut cases = 0;
        for (ci, &count) in counts.iter().enumerate() {
            for &off in &[0usize, 256] {
                for &disjoint in &[true, false] {
                    let seed = 0x1000 + (ci as u64) * 64 + (off as u64 / 256) * 2 + disjoint as u64;
                    let (own, want) = case(seed, count, rank, disjoint);
                    let pad = off / 4;
                    let mut host = vec![CANARY; pad + count + 16];
                    host[pad..pad + count].copy_from_slice(&own);
                    let mut results: Vec<Vec<f32>> = Vec::new();
                    for algo in [Algo::Ring, Algo::OneShot] {
                        let mut buf = ck(DevBuf::from_slice(&host), "upload");
                        ck(gb_cuda::rt::device_sync(), "device sync");
                        ck(run_coll(&comm, algo, &mut buf, off, count, &st), "collective");
                        ck(st.sync(), "sync");
                        let got: Vec<f32> = ck(buf.to_vec(), "download");
                        if let Some(m) = first_mismatch(&got[pad..pad + count], &want) {
                            panic!("rank {rank}: {} count {count} off {off} disjoint {disjoint}: {m}", algo_name(algo));
                        }
                        assert!(
                            got[..pad].iter().chain(&got[pad + count..]).all(|v| v.to_bits() == CANARY.to_bits()),
                            "rank {rank}: count {count} off {off}: the collective wrote outside its range"
                        );
                        results.push(got);
                    }
                    for r in &results[1..] {
                        assert_eq!(
                            results[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            r.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "rank {rank}: count {count}: ring and one-shot differ"
                        );
                    }
                    cases += 1;
                }
            }
        }
        // count 0 is a no-op
        {
            let mut buf = ck(DevBuf::from_slice(&[1.0f32; 4]), "upload");
            ck(comm.oneshot_sum_f32(&mut buf, 0, 0, &st), "one-shot of 0 elements");
            ck(st.sync(), "sync");
            assert_eq!(ck(buf.to_vec::<f32>(), "download"), vec![1.0f32; 4]);
        }
        eprintln!("nccl_oneshot_pair rank {rank}: (eager) {cases} cases: one-shot == ring == the host's a+b, bit for bit");
        side.barrier();
    }

    // ---- latency: ring against one-shot, per collective --------------------------------------------------------------------------
    if phase_wanted("latency") {
        let mut rows = Vec::new();
        for &count in &[1024usize, 4096, 16384, 22528, 65536, 112_640, 262144, 524_288, 1048576] {
            let mut cell = format!("{:.0} KB", count as f64 * 4.0 / 1024.0);
            for algo in [Algo::Ring, Algo::OneShot] {
                let mut buf = ck(DevBuf::zeroed(count * 4), "alloc");
                ck(gb_cuda::rt::device_sync(), "device sync");
                for _ in 0..50 {
                    ck(run_coll(&comm, algo, &mut buf, 0, count, &st), "warmup");
                }
                ck(st.sync(), "sync");
                side.barrier();
                // five rounds: the median is the number (the pair shares its CPUs and fabric with other work, and one round can be 10x off)
                let n = 100;
                let mut rounds = Vec::new();
                for _ in 0..5 {
                    let t = Instant::now();
                    for _ in 0..n {
                        ck(run_coll(&comm, algo, &mut buf, 0, count, &st), "timed");
                    }
                    ck(st.sync(), "sync");
                    rounds.push(t.elapsed().as_secs_f64() * 1e6 / n as f64);
                    side.barrier();
                }
                let eager = median(&mut rounds);
                let (inner, reps) = (50usize, 40usize);
                let g = capture(&st, rank, "latency graph", || {
                    for _ in 0..inner {
                        run_coll(&comm, algo, &mut buf, 0, count, &st)?;
                    }
                    Ok(())
                });
                for _ in 0..3 {
                    ck(g.launch(&st), "warm replay");
                }
                ck(st.sync(), "sync");
                side.barrier();
                let mut rounds = Vec::new();
                for _ in 0..5 {
                    let t = Instant::now();
                    for _ in 0..reps {
                        ck(g.launch(&st), "replay");
                    }
                    ck(st.sync(), "sync");
                    rounds.push(t.elapsed().as_secs_f64() * 1e6 / (reps * inner) as f64);
                    side.barrier();
                }
                let graph = median(&mut rounds);
                drop(g);
                side.barrier();
                cell.push_str(&format!(" | {} eager {eager:.1} / graph {graph:.1}", algo_name(algo)));
            }
            rows.push(cell);
        }
        eprintln!("nccl_oneshot_pair rank {rank}: latency per collective, us:\n  {}", rows.join("\n  "));
    }

    // ---- capture: 100 one-shots in ONE graph, replayed 1,000 times ---------------------------------------------------------------
    if phase_wanted("capture") {
        const INNER: usize = 100;
        const REPLAYS: usize = 1000;
        for &n in &[4096usize, 112_640] {
            let region = n * 4;
            let total = INNER * region;
            let case_of = |salt: u64, disjoint: bool| -> (Vec<f32>, Vec<f32>) {
                let (mut own, mut sum) = (Vec::with_capacity(INNER * n), Vec::with_capacity(INNER * n));
                for k in 0..INNER {
                    let (o, s) = case(salt + k as u64, n, rank, disjoint);
                    own.extend(o);
                    sum.extend(s);
                }
                (own, sum)
            };
            let (own, want) = case_of(0x7000 + n as u64, true);
            let mut seed = ck(DevBuf::from_slice(&own), "seed");
            let mut work_os = ck(DevBuf::zeroed(total), "work one-shot");
            let mut work_ring = ck(DevBuf::zeroed(total), "work ring");
            ck(gb_cuda::rt::device_sync(), "device sync");
            side.barrier();
            let g_os = capture(&st, rank, &format!("{INNER} one-shots of {} KB", region / 1024), || {
                work_os.copy_from(&seed, total, &st)?;
                for k in 0..INNER {
                    comm.oneshot_sum_f32(&mut work_os, k * region, n, &st)?;
                }
                Ok(())
            });
            let g_ring = capture(&st, rank, &format!("{INNER} ring allreduces of {} KB", region / 1024), || {
                work_ring.copy_from(&seed, total, &st)?;
                for k in 0..INNER {
                    comm.allreduce_sum_f32(&mut work_ring, k * region, n, &st)?;
                }
                Ok(())
            });
            eprintln!(
                "nccl_oneshot_pair rank {rank}: (capture) {INNER} one-shot collectives of {} KB captured into one graph (and the ring's)",
                region / 1024
            );
            side.barrier();
            for _ in 0..5 {
                ck(g_os.launch(&st), "warm replay");
                ck(g_ring.launch(&st), "warm replay ring");
            }
            ck(st.sync(), "sync");
            side.barrier();
            let check = |work: &DevBuf, want: &[f32], what: &str, at: usize| {
                let got: Vec<f32> = ck(work.to_vec(), "download");
                if let Some(m) = first_mismatch(&got, want) {
                    panic!("rank {rank}: {what}: replay {at}: {m}");
                }
            };
            let t = Instant::now();
            for rep in 0..REPLAYS {
                ck(g_os.launch(&st), "replay one-shot graph");
                if rep % 100 == 99 {
                    ck(st.sync(), "sync");
                    check(&work_os, &want, "one-shot graph", rep + 1);
                    // the ring's graph on the same seed: bit-equal to the one-shot's
                    ck(g_ring.launch(&st), "replay ring graph");
                    ck(st.sync(), "sync");
                    check(&work_ring, &want, "ring graph", rep + 1);
                }
            }
            ck(st.sync(), "sync");
            let el = t.elapsed();
            side.barrier();
            // timed replays of each, back to back
            let t = Instant::now();
            for _ in 0..200 {
                ck(g_os.launch(&st), "replay one-shot graph");
            }
            ck(st.sync(), "sync");
            let os_us = t.elapsed().as_secs_f64() * 1e6 / (200 * INNER) as f64;
            side.barrier();
            let t = Instant::now();
            for _ in 0..200 {
                ck(g_ring.launch(&st), "replay ring graph");
            }
            ck(st.sync(), "sync");
            let ring_us = t.elapsed().as_secs_f64() * 1e6 / (200 * INNER) as f64;
            side.barrier();
            // fresh seeds, disjoint and dense, each replay checked: ring and one-shot agree with each other and with the host
            for k in 0..20u64 {
                let (own, want) = case_of(0xC0DE00 + k * 1000 + n as u64, k % 2 == 0);
                ck(seed.upload(&own), "upload seed");
                ck(gb_cuda::rt::device_sync(), "device sync");
                ck(g_os.launch(&st), "replay");
                ck(g_ring.launch(&st), "replay ring");
                ck(st.sync(), "sync");
                check(&work_os, &want, "one-shot graph, fresh seed", k as usize);
                check(&work_ring, &want, "ring graph, fresh seed", k as usize);
                side.barrier();
            }
            eprintln!(
                "nccl_oneshot_pair rank {rank}: (capture) {} KB x {INNER} one-shots in one graph replayed {REPLAYS} times ({:.1} ms per replay incl. checks): \
                 bit-equal to the host sum and to the ring's graph at every 100th replay and on 20 fresh seeds; \
                 graph {os_us:.2} us per one-shot against {ring_us:.2} us per ring allreduce",
                region / 1024,
                el.as_secs_f64() * 1e3 / REPLAYS as f64
            );
            drop(g_os);
            drop(g_ring);
            side.barrier();
        }
    }

    // ---- step: the serving plan's collectives of one step as one graph, ring / one-shot / auto -------------------------------------
    if phase_wanted("step") {
        for &rows in &[1usize, 5, 16, 64] {
            let mut sizes: Vec<usize> = Vec::new();
            for l in 0..88 {
                match l % 11 {
                    0 | 2 | 4 | 6 | 8 => sizes.push(rows * 18560),
                    5 => {}
                    _ => sizes.push(rows * 22 * 1024),
                }
            }
            sizes.push(rows * 131072);
            let maxn = *sizes.iter().max().unwrap();
            let mut sbuf = ck(DevBuf::zeroed(maxn * 4), "step buffer");
            ck(gb_cuda::rt::device_sync(), "device sync");
            let mut line =
                format!("rows {rows}: {} collectives, {:.2} MB", sizes.len(), sizes.iter().sum::<usize>() as f64 * 4.0 / 1e6);
            for algo in [Algo::Ring, Algo::OneShot, Algo::Auto] {
                let name = algo_name(algo);
                // the forced one-shot is limited by the staging (24 MiB): the logits of 46+ rows would not fit, none is asked here
                for &n in &sizes[..3] {
                    ck(run_coll(&comm, algo, &mut sbuf, 0, n, &st), "step warm-up");
                }
                ck(st.sync(), "sync");
                side.barrier();
                // the host time of capturing the whole step (first-use cost of a window shape in the engine) and of a second capture
                let mut capture_ms = Vec::new();
                let mut g = None;
                for _ in 0..3 {
                    let t = Instant::now();
                    let cg = capture(&st, rank, &format!("a step of {} collectives ({name})", sizes.len()), || {
                        for &n in &sizes {
                            run_coll(&comm, algo, &mut sbuf, 0, n, &st)?;
                        }
                        Ok(())
                    });
                    capture_ms.push(t.elapsed().as_secs_f64() * 1e3);
                    g = Some(cg); // the previous graph of the three is dropped here
                    side.barrier();
                }
                let g = g.unwrap();
                for _ in 0..5 {
                    ck(g.launch(&st), "step replay");
                }
                ck(st.sync(), "sync");
                side.barrier();
                let reps = if rows >= 64 { 20 } else { 60 };
                let mut rounds = Vec::new();
                for _ in 0..5 {
                    let t = Instant::now();
                    for _ in 0..reps {
                        ck(g.launch(&st), "step replay");
                    }
                    ck(st.sync(), "sync");
                    rounds.push(t.elapsed().as_secs_f64() * 1e3 / reps as f64);
                    side.barrier();
                }
                let min = rounds.iter().cloned().fold(f64::MAX, f64::min);
                let ms = median(&mut rounds);
                let first_capture = capture_ms[0];
                let median_capture = median(&mut capture_ms);
                line.push_str(&format!(
                    " | {name} {ms:.2} (min {min:.2}) ms/step, capture {first_capture:.1} ms (first) {median_capture:.1} ms (median of 3)"
                ));
                drop(g);
                side.barrier();
            }
            eprintln!("nccl_oneshot_pair rank {rank}: (step) {line}");
        }
    }

    drop(comm);
    eprintln!("nccl_oneshot_pair rank {rank}: PASS");
}
