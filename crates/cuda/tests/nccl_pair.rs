//! Two-process NCCL test: the collective of the two-rank tensor-parallel engine, across two nodes, eager and inside a CUDA graph.
//!
//! One process per rank, started together (one per node, each under that node's GPU lock). Gated on the environment, so a plain
//! `cargo test` prints `not run: REXL3_NCCL_RANK is not set` and passes:
//!
//! ```text
//! REXL3_NCCL_RANK=0|1  REXL3_NCCL_RANKS=2  REXL3_NCCL_PEER=192.0.2.1:29500   (rank 0's address: it listens, rank 1 connects)
//! NCCL fabric settings: source /srv/rexl3/nccl/nccl.env     (NCCL_NET=IB, NCCL_SOCKET_IFNAME==enP7s7, ...)
//! optional: REXL3_NCCL_LIB (libnccl path), REXL3_NCCL_TIMEOUT_S (watchdog, default 240)
//! ```
//!
//! Rank 0 creates the `ncclUniqueId` and sends its 128 bytes over the TCP connection; the same connection is the lockstep channel between
//! the phases. What it checks, on both ranks:
//!
//! (a) eager allreduces over buffers of DISJOINT SUPPORT (each rank holds its own elements and exact -0.0 elsewhere: `x + -0.0 == x`
//!     for EVERY x, including -0.0, where `+0.0` would turn an exact -0.0 into +0.0) and over dense
//!     buffers, from 4 B to 4 MB at byte offsets 0 and 256: the result is bit-equal to the host's `a + b`, and the bytes around the range
//!     are untouched;
//! (b) 100 allreduces of 16 KB captured into one CUDA graph on the compute stream (the graph also copies the seed in first, so every replay
//!     starts from the same bits), replayed 1,000 times: the final bits are `(a + b) * 2^99` (a rank that has the sum doubles it each
//!     round, exactly), then 20 replays with fresh seeds each checked; the time per allreduce is printed for the graph and for eager launches;
//! (c) the collectives of one engine step (81 of the serving plan, 97 with the attention sliced, 137 with every dense shape sliced; for
//!     1, 5, 16 and 64 rows) captured as ONE graph and replayed 100 times: the per-step tax, graph and eager.
//!
//! The latency depends on NCCL's own tuning, which is pathological on this pair by default (see section 10): run it under the settings
//! the engine will use (`NCCL_MAX_NCHANNELS`, `NCCL_PROTO`).
//!
//! The doc is `docs/NEMOTRON_TP.md` section 10.
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

/// A finite f32 with a random sign and mantissa: magnitude in [2^-8, 2^8); every fifth element a SUBNORMAL (exponent 0), since
/// `x + -0.0 == x` must hold for a subnormal too (a flush-to-zero path anywhere would turn it into 0.0); and every 23rd element an exact
/// signed zero, +0.0 or -0.0, since the sum of an owner's -0.0 with the filler must keep its sign.
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

/// Which rank owns element `i` in a disjoint-support buffer.
fn owner(i: usize) -> usize {
    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 63) as usize
}

/// This rank's input and the expected sum for one case: `disjoint` keeps each rank's own elements and exact -0.0 elsewhere (the filler
/// of the engine's scatter: `x + -0.0 == x` for every x), and then the sum IS the owner's element, bit for bit.
fn case(seed: u64, n: usize, rank: usize, disjoint: bool) -> (Vec<f32>, Vec<f32>) {
    let a = vector(seed, n);
    let b = vector(seed ^ 0xA5A5_A5A5, n);
    let (a, b): (Vec<f32>, Vec<f32>) = if disjoint {
        let owned: Vec<f32> = (0..n).map(|i| if owner(i) == 0 { a[i] } else { b[i] }).collect();
        let sum = (0..n).map(|i| if owner(i) == 0 { a[i] + -0.0 } else { -0.0 + b[i] }).collect::<Vec<f32>>();
        assert_eq!(
            bits(&sum),
            bits(&owned),
            "a disjoint-support sum is a copy of the owner's element, signed zeros and subnormals included"
        );
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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn first_mismatch(got: &[f32], want: &[f32]) -> Option<String> {
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
        eprintln!("nccl_pair rank 0: peer connected from {from}");
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
                    eprintln!("nccl_pair rank 1: waiting for rank 0 at {peer}: {e}");
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

#[test]
fn nccl_pair_allreduce_eager_and_captured() {
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
    // a hung peer must not hold a GPU lock: the watchdog ends the process
    let timeout: u64 = std::env::var("REXL3_NCCL_TIMEOUT_S").ok().and_then(|v| v.parse().ok()).unwrap_or(240);
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(timeout));
        eprintln!("nccl_pair: watchdog expired after {timeout} s, exiting");
        std::process::exit(3);
    });

    ck(gb_cuda::rt::init_context(), "init_context");
    let st = ck(Stream::new_nonblocking(), "compute stream");
    eprintln!(
        "nccl_pair rank {rank}: libnccl {} version code {}",
        NcclComm::library_path(),
        ck(NcclComm::version(), "ncclGetVersion")
    );
    let id = if rank == 0 { Some(ck(NcclComm::new_unique_id(), "ncclGetUniqueId")) } else { None };
    let (mut side, id) = rendezvous(rank, &peer, id.as_ref());
    let t0 = Instant::now();
    let comm = ck(NcclComm::init(rank, ranks, &id), "ncclCommInitRank");
    eprintln!("nccl_pair rank {rank}: communicator up in {:.2} s", t0.elapsed().as_secs_f64());
    side.barrier();

    // ---- (a) eager, disjoint and dense, sizes 4 B .. 4 MB, offsets 0 and 256 -------------------------------------------------------
    const CANARY: f32 = f32::from_bits(0x7FC0_1234); // a NaN payload: any write over it is visible bit for bit
    let counts = [1usize, 7, 1021, 1024, 4096, 16384, 65536, 262144, 1048576];
    let mut cases = 0;
    for (ci, &count) in counts.iter().enumerate() {
        for &off in &[0usize, 256] {
            for &disjoint in &[true, false] {
                let seed = 0x1000 + (ci as u64) * 64 + (off as u64 / 256) * 2 + disjoint as u64;
                let (own, want) = case(seed, count, rank, disjoint);
                let pad = off / 4;
                let mut host = vec![CANARY; pad + count + 16];
                host[pad..pad + count].copy_from_slice(&own);
                let mut buf = ck(DevBuf::from_slice(&host), "upload");
                // the upload is a synchronous copy on the legacy stream; the compute stream does not order with it
                ck(gb_cuda::rt::device_sync(), "device sync");
                ck(comm.allreduce_sum_f32(&mut buf, off, count, &st), "allreduce");
                ck(st.sync(), "sync");
                let got: Vec<f32> = ck(buf.to_vec(), "download");
                if let Some(m) = first_mismatch(&got[pad..pad + count], &want) {
                    panic!("rank {rank}: count {count} off {off} disjoint {disjoint}: {m}");
                }
                assert!(
                    bits(&got[..pad]).iter().chain(bits(&got[pad + count..]).iter()).all(|&b| b == CANARY.to_bits()),
                    "rank {rank}: count {count} off {off}: the allreduce wrote outside its range"
                );
                cases += 1;
            }
        }
    }
    // count 0 is a no-op on both ranks
    {
        let mut buf = ck(DevBuf::from_slice(&[1.0f32; 4]), "upload");
        ck(comm.allreduce_sum_f32(&mut buf, 0, 0, &st), "allreduce of 0 elements");
        ck(st.sync(), "sync");
        assert_eq!(ck(buf.to_vec::<f32>(), "download"), vec![1.0f32; 4]);
    }
    eprintln!("nccl_pair rank {rank}: (a) {cases} eager cases bit-equal to the host sum");
    side.barrier();

    // ---- latency table: eager (the binding's call path, back to back, one sync at the end) and CUDA-graph replay per collective -----
    // the sizes of the step: 16 KB / 74 KB per row of out and in_proj, 90 KB (1 row) and 450 KB (5 rows) of MoE y, 2 MB, 4 MB
    // `REXL3_NCCL_LAT_COUNTS` (a comma list of f32 counts) replaces the sizes, for the prefill exchange's 32 to 256 MiB; above 1 M
    // elements the repetitions shrink with the size (at least 5 eager calls and a graph of 2 x 2), so a 256 MiB size takes seconds
    {
        let mut rows = Vec::new();
        let counts: Vec<usize> = match std::env::var("REXL3_NCCL_LAT_COUNTS") {
            Ok(v) => v.split(',').map(|x| x.trim().parse::<usize>().expect("REXL3_NCCL_LAT_COUNTS: f32 counts")).collect(),
            Err(_) => vec![1024usize, 4096, 16384, 22528, 65536, 112_640, 262144, 524_288, 1048576],
        };
        let scaled = |base: usize, floor: usize, count: usize| -> usize {
            if count <= 1 << 20 {
                base
            } else {
                (base * (1 << 20) / count).max(floor)
            }
        };
        for &count in &counts {
            let mut buf = ck(DevBuf::zeroed(count * 4), "alloc");
            ck(gb_cuda::rt::device_sync(), "device sync");
            for _ in 0..scaled(50, 3, count) {
                ck(comm.allreduce_sum_f32(&mut buf, 0, count, &st), "warmup");
            }
            ck(st.sync(), "sync");
            side.barrier();
            let n = scaled(200, 5, count);
            let t = Instant::now();
            for _ in 0..n {
                ck(comm.allreduce_sum_f32(&mut buf, 0, count, &st), "timed");
            }
            ck(st.sync(), "sync");
            let eager = t.elapsed().as_secs_f64() * 1e6 / n as f64;
            side.barrier();
            // the same collectives replayed from a graph of 50 (per collective)
            let (inner, reps) = (scaled(50, 2, count), scaled(40, 2, count));
            let g = {
                let cap = ck(Graph::begin(&st), "begin capture");
                let ran = (|| -> Result<(), CudaError> {
                    for _ in 0..inner {
                        comm.allreduce_sum_f32(&mut buf, 0, count, &st)?;
                    }
                    Ok(())
                })();
                match cap.finish(ran) {
                    Ok(g) => g,
                    Err(e) => panic!("rank {rank}: capturing {inner} allreduces of {count} f32 failed: {e}"),
                }
            };
            for _ in 0..3 {
                ck(g.launch(&st), "warm replay");
            }
            ck(st.sync(), "sync");
            side.barrier();
            let t = Instant::now();
            for _ in 0..reps {
                ck(g.launch(&st), "replay");
            }
            ck(st.sync(), "sync");
            let graph = t.elapsed().as_secs_f64() * 1e6 / (reps * inner) as f64;
            drop(g);
            side.barrier();
            rows.push(format!(
                "{:.0} KB eager {eager:.1} / graph {graph:.1} us ({n} eager, {inner} x {reps} replayed)",
                count as f64 * 4.0 / 1024.0
            ));
        }
        eprintln!("nccl_pair rank {rank}: allreduce latency per collective: {}", rows.join("; "));
    }

    // ---- (b) 100 allreduces of 16 KB in one CUDA graph, replayed 1,000 times --------------------------------------------------------
    const N: usize = 4096; // 16 KB of f32
    const INNER: usize = 100;
    const REPLAYS: usize = 1000;
    let (seed_host, want_c) = case(0xBEEF, N, rank, true);
    let scale = 2.0f32.powi(INNER as i32 - 1); // round 1 gives a + b; every later round both ranks hold it and double it
    let want: Vec<f32> = want_c.iter().map(|c| c * scale).collect();
    let mut seed = ck(DevBuf::from_slice(&seed_host), "seed");
    let mut work = ck(DevBuf::zeroed(N * 4), "work");
    ck(gb_cuda::rt::device_sync(), "device sync");
    // the first collective connects the peers: do it eagerly, before the capture
    ck(work.copy_from(&seed, N * 4, &st), "copy");
    ck(comm.allreduce_sum_f32(&mut work, 0, N, &st), "warm-up allreduce");
    ck(st.sync(), "sync");
    side.barrier();

    let graph = {
        let cap = ck(Graph::begin(&st), "begin capture");
        let ran = (|| -> Result<(), CudaError> {
            work.copy_from(&seed, N * 4, &st)?;
            for _ in 0..INNER {
                comm.allreduce_sum_f32(&mut work, 0, N, &st)?;
            }
            Ok(())
        })();
        match cap.finish(ran) {
            Ok(g) => g,
            Err(e) => panic!("rank {rank}: capturing {INNER} allreduces failed: {e}"),
        }
    };
    eprintln!("nccl_pair rank {rank}: (b) captured {INNER} allreduces of {} KB into one graph", N * 4 / 1024);
    side.barrier();

    // eager reference with the same work (copy + INNER allreduces, REPLAYS/INNER repetitions), for the per-call overhead of the graph
    let reps = REPLAYS / INNER;
    let t = Instant::now();
    for _ in 0..reps {
        ck(work.copy_from(&seed, N * 4, &st), "copy");
        for _ in 0..INNER {
            ck(comm.allreduce_sum_f32(&mut work, 0, N, &st), "eager");
        }
    }
    ck(st.sync(), "sync");
    let eager_us = t.elapsed().as_secs_f64() * 1e6 / (reps * INNER) as f64;
    let got: Vec<f32> = ck(work.to_vec(), "download");
    if let Some(m) = first_mismatch(&got, &want) {
        panic!("rank {rank}: eager reference: {m}");
    }
    side.barrier();

    // replay: warm, then 1,000 timed replays back to back
    for _ in 0..10 {
        ck(graph.launch(&st), "warm replay");
    }
    ck(st.sync(), "sync");
    side.barrier();
    let t = Instant::now();
    for _ in 0..REPLAYS {
        ck(graph.launch(&st), "replay");
    }
    ck(st.sync(), "sync");
    let el = t.elapsed();
    let graph_us = el.as_secs_f64() * 1e6 / (REPLAYS * INNER) as f64;
    let got: Vec<f32> = ck(work.to_vec(), "download");
    if let Some(m) = first_mismatch(&got, &want) {
        panic!("rank {rank}: after {REPLAYS} replays: {m}");
    }
    side.barrier();

    // fresh seeds: every replay's result follows the new inputs (the graph reads its inputs from the device buffers)
    for k in 0..20u64 {
        let (own, c) = case(0xC0DE + k, N, rank, k % 2 == 0);
        ck(seed.upload(&own), "upload seed");
        ck(gb_cuda::rt::device_sync(), "device sync");
        ck(graph.launch(&st), "replay");
        ck(st.sync(), "sync");
        let want: Vec<f32> = c.iter().map(|v| v * scale).collect();
        let got: Vec<f32> = ck(work.to_vec(), "download");
        if let Some(m) = first_mismatch(&got, &want) {
            panic!("rank {rank}: fresh seed {k}: {m}");
        }
        side.barrier();
    }
    eprintln!(
        "nccl_pair rank {rank}: (b) {REPLAYS} replays x {INNER} allreduces of {} KB: graph {graph_us:.2} us per allreduce \
         ({:.1} ms per replay), eager {eager_us:.2} us per allreduce; final bits equal the host's (a+b)*2^{}; 20 fresh-seed replays equal",
        N * 4 / 1024,
        el.as_secs_f64() * 1e3 / REPLAYS as f64,
        INNER - 1
    );

    // ---- (c) the per-step tax: the collectives of one engine step, as one graph -----------------------------------------------------
    // For `rows` rows of the 88-layer model (40 Mamba, 8 attention, 40 MoE), in the layer order of a typical trunk (the order only matters
    // for what runs back to back), f32, three plans (`docs/NEMOTRON_TP.md` sections 3 and 9):
    //   serving   per Mamba layer the in_proj [rows, 18560], per MoE layer the routed-expert buffer [rows, 22, 1024], the lm_head
    //             [rows, 131072] once: 81 collectives (out_proj and the attention whole)
    //   attn      the serving plan with the attention q|k|v [rows, 4608] and o [rows, 4096] sliced: 97
    //   all       every dense shape sliced, out_proj [rows, 4096] too: 137 (the first design)
    // The graph holds one whole step and is replayed 100 times (30 at 64 rows).
    for &(plan, mamba_out, attn) in &[("serving", false, false), ("attn", false, true), ("all", true, true)] {
        for &rows in &[1usize, 5, 16, 64] {
            if plan != "serving" && rows == 64 {
                continue;
            }
            let mut sizes: Vec<usize> = Vec::new();
            for l in 0..88 {
                match l % 11 {
                    // 40 Mamba, 8 attention, 40 MoE: eleven layers hold 5 Mamba, 1 attention, 5 MoE
                    0 | 2 | 4 | 6 | 8 => {
                        sizes.push(rows * 18560);
                        if mamba_out {
                            sizes.push(rows * 4096);
                        }
                    }
                    5 => {
                        if attn {
                            sizes.extend([rows * 4608, rows * 4096]);
                        }
                    }
                    _ => sizes.push(rows * 22 * 1024),
                }
            }
            sizes.push(rows * 131072);
            let maxn = *sizes.iter().max().unwrap();
            let total_bytes: usize = sizes.iter().sum::<usize>() * 4;
            let mut sbuf = ck(DevBuf::zeroed(maxn * 4), "step buffer");
            ck(gb_cuda::rt::device_sync(), "device sync");
            for &n in &sizes[..3] {
                ck(comm.allreduce_sum_f32(&mut sbuf, 0, n, &st), "step warm-up");
            }
            ck(st.sync(), "sync");
            side.barrier();
            let g = {
                let cap = ck(Graph::begin(&st), "begin step capture");
                let ran = (|| -> Result<(), CudaError> {
                    for &n in &sizes {
                        comm.allreduce_sum_f32(&mut sbuf, 0, n, &st)?;
                    }
                    Ok(())
                })();
                match cap.finish(ran) {
                    Ok(g) => g,
                    Err(e) => panic!("rank {rank}: capturing the step graph failed: {e}"),
                }
            };
            for _ in 0..5 {
                ck(g.launch(&st), "step replay");
            }
            ck(st.sync(), "sync");
            side.barrier();
            let reps = if rows >= 64 { 30 } else { 100 };
            let t = Instant::now();
            for _ in 0..reps {
                ck(g.launch(&st), "step replay");
            }
            ck(st.sync(), "sync");
            let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            side.barrier();
            // the same sequence launched eagerly
            let t = Instant::now();
            for _ in 0..reps / 4 {
                for &n in &sizes {
                    ck(comm.allreduce_sum_f32(&mut sbuf, 0, n, &st), "step eager");
                }
            }
            ck(st.sync(), "sync");
            let eager_ms = t.elapsed().as_secs_f64() * 1e3 / (reps / 4) as f64;
            side.barrier();
            eprintln!(
                "nccl_pair rank {rank}: (c) step tax, plan {plan}, rows {rows}: {} collectives, {:.2} MB summed: graph {ms:.2} ms per step, eager {eager_ms:.2} ms per step",
                sizes.len(),
                total_bytes as f64 / 1e6
            );
            drop(g);
            side.barrier();
        }
    }

    // the graph must be gone before the communicator it captured
    drop(graph);
    side.barrier();
    drop(comm);
    eprintln!("nccl_pair rank {rank}: PASS");
}
