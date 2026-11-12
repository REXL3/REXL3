//! Two-process test of REXL3's RoCE all-gather (`gb_cuda::roce`, `docs/ROCE_GATHER.md`) across two nodes, checked bit for bit against
//! NCCL moving the same bytes, eager and inside CUDA graphs, plus the failure paths.
//!
//! One process per rank, started together (one per node, each under that node's GPU lock). Gated on the environment, so a plain
//! `cargo test` prints `not run: REXL3_ROCE_RANK is not set` and passes:
//!
//! ```text
//! REXL3_ROCE_RANK=0|1  REXL3_ROCE_PEER=192.0.2.1:29640   (rank 0's address: it listens, rank 1 connects)
//! NCCL fabric settings: source /srv/rexl3/nccl/nccl.env; REXL3_NCCL_LIB=<libnccl.so.2 of the CRS812 build>
//! optional: REXL3_ROCE_* (gb_cuda::roce::RoceConfig), REXL3_ROCE_TEST_TIMEOUT_S (watchdog, default 600)
//! ```
//!
//! What it checks, on both ranks:
//! (a) eager all-gathers from 4 B to 24 MiB (one rail, both rails, chunked bulk), at source / destination offsets 0, 4 and 256, of
//!     random 32-bit patterns with -0.0, +0.0, subnormals, infinities and NaN payloads mixed in: equal to NCCL's bytes (ncclSend /
//!     ncclRecv of the same buffers) and to the host's `[rank 0 | rank 1]`, and nothing written outside the destination range;
//! (b) the in-place f32 sum, over disjoint-support buffers (-0.0 filler, the engine's exchange) and dense ones: bit-equal to
//!     `ncclAllReduce` (and, disjoint, to the owner's element);
//! (c) a swap with different sizes each way (5,000 B one way, 70,000 B the other);
//! (c2) zero-copy swaps (lane T, the GLM P2 part swap): each rank writes its 16 MiB straight into its half of the site's next exchange
//!     (`RoceComm::next_halves`), an exchange with no source publishes it, and the peer's half is read in place; six in a row with
//!     standalone swaps of the same site between them (the host's parity follows both), every byte checked; after (d) a site an
//!     exchange was captured at has no zero-copy halves;
//! (d) a CUDA graph of 12 exchanges on two sites (both parities, back to back) replayed 300 times with fresh data every 30th replay
//!     and eager exchanges between the replays (the device-resident sequences carry across);
//! (d2) two streams at once: 16 KiB gathers on one site and 8 MiB gathers on another, queued in a different interleaving on each
//!     rank (order is per site), every byte checked;
//! (e) a lost peer: rank 1 skips one exchange; rank 0's wait times out (2 s), records the error, writes nothing, and the transport
//!     refuses every later exchange; then a peer whose transport is GONE (its QPs destroyed): rank 0's sends fail and surface;
//! (f) abort from another thread while a wait spins: the stream completes at once and later exchanges fail;
//! (g) restart after the aborts: fresh transports on both ranks run correct exchanges again;
//! (h) two users of ONE size class (lane T, per-user site lists): a transport with two 8 MiB sites, user A's zero-copy swaps on one
//!     stream and user B's standalone swaps on another, interleaved differently on the two ranks, every byte checked (each user's
//!     site keeps its own order and parity; `site_for_among` picks within a user's ids).
#![allow(clippy::needless_range_loop)]

use gb_cuda::graph::Graph;
use gb_cuda::nccl::{NcclComm, NcclUniqueId};
use gb_cuda::roce::{self, RoceComm, RoceConfig};
use gb_cuda::{device_available, CudaError, DevBuf, Stream};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Random 32-bit words with the special patterns every byte-exact path must keep: -0.0, +0.0, subnormals, infinities, NaN payloads.
fn words(seed: u64, n: usize) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let r = splitmix(&mut s) as u32;
            match i % 29 {
                3 => 0x8000_0000,                      // -0.0
                7 => 0x0000_0000,                      // +0.0
                11 => (r & 0x807F_FFFF) | 1,           // a subnormal of either sign
                13 => 0x7F80_0000 | (r & 0x8000_0000), // an infinity
                17 => 0x7F80_0001 | (r & 0x803F_FFFF), // a NaN with a payload (signalling or quiet)
                _ => r,
            }
        })
        .collect()
}

fn bytes_of(w: &[u32], n: usize) -> Vec<u8> {
    let mut v: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    v.truncate(n);
    v
}

/// Finite f32 data for the sums (dense ones include subnormals and NaN payloads: the GPU returns the canonical NaN either way).
fn floats(seed: u64, n: usize, nan: bool) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let r = splitmix(&mut s);
            let mant = (r & 0x7F_FFFF) as u32;
            let sign = ((r >> 40) & 1) as u32;
            if nan && i % 97 == 5 {
                return f32::from_bits(0x7FC0_0000 | mant);
            }
            if i % 23 == 4 {
                return f32::from_bits(sign << 31);
            }
            if i % 5 == 1 {
                return f32::from_bits(sign << 31 | (mant | 1));
            }
            let exp = 127 - 8 + ((r >> 32) % 16) as u32;
            f32::from_bits(sign << 31 | exp << 23 | mant)
        })
        .collect()
}

fn owner(i: usize) -> usize {
    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 63) as usize
}

const CANARY: u8 = 0xA5;

struct Side {
    sock: TcpStream,
}

impl Side {
    fn barrier(&mut self) {
        self.sock.write_all(&[0x42]).expect("lockstep write");
        let mut b = [0u8; 1];
        self.sock.read_exact(&mut b).expect("lockstep read");
        assert_eq!(b[0], 0x42);
    }
    fn swap(&mut self, mine: &[u8]) -> Vec<u8> {
        self.sock.write_all(&(mine.len() as u64).to_le_bytes()).expect("write");
        self.sock.write_all(mine).expect("write");
        let mut n = [0u8; 8];
        self.sock.read_exact(&mut n).expect("read");
        let mut v = vec![0u8; u64::from_le_bytes(n) as usize];
        self.sock.read_exact(&mut v).expect("read");
        v
    }
}

fn rendezvous(rank: usize, peer: &str) -> Side {
    if rank == 0 {
        let port = peer.rsplit(':').next().expect("REXL3_ROCE_PEER is HOST:PORT");
        let l = TcpListener::bind(format!("0.0.0.0:{port}")).expect("bind the rendezvous port");
        let (s, from) = l.accept().expect("accept the peer");
        eprintln!("roce_pair rank 0: peer connected from {from}");
        s.set_nodelay(true).ok();
        Side { sock: s }
    } else {
        let t0 = Instant::now();
        let s = loop {
            match TcpStream::connect(peer) {
                Ok(s) => break s,
                Err(e) if t0.elapsed() < Duration::from_secs(120) => {
                    eprintln!("roce_pair rank 1: waiting for rank 0 at {peer}: {e}");
                    std::thread::sleep(Duration::from_millis(500));
                }
                Err(e) => panic!("rank 1 could not reach rank 0 at {peer}: {e}"),
            }
        };
        s.set_nodelay(true).ok();
        Side { sock: s }
    }
}

fn ck<T>(r: Result<T, CudaError>, what: &str) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{what}: {e}"),
    }
}

/// Sites of every transport this test builds: decode-sized, mid, bulk.
const CAPS: [usize; 3] = [64 << 10, 2 << 20, 24 << 20];

/// A connected, started transport (both ranks call it at the same point).
fn transport(rank: usize, cfg: &RoceConfig, side: &mut Side) -> Arc<RoceComm> {
    transport_caps(rank, cfg, side, &CAPS)
}

fn transport_caps(rank: usize, cfg: &RoceConfig, side: &mut Side, caps: &[usize]) -> Arc<RoceComm> {
    let t0 = Instant::now();
    let mut c = RoceComm::new(rank, cfg, caps).unwrap_or_else(|e| panic!("rank {rank}: RoceComm::new: {e}"));
    let peer = side.swap(&c.blob());
    c.connect(&peer).unwrap_or_else(|e| panic!("rank {rank}: connect: {e}"));
    side.barrier(); // both QPs ready before either proxy posts
    c.start().unwrap_or_else(|e| panic!("rank {rank}: start: {e}"));
    side.barrier();
    eprintln!(
        "roce_pair rank {rank}: transport up in {:.2} s ({} rails, region {:.1} MiB, {})",
        t0.elapsed().as_secs_f64(),
        c.rails(),
        c.region_bytes() as f64 / 1048576.0,
        cfg.fingerprint()
    );
    Arc::new(c)
}

#[test]
fn roce_pair_against_nccl_graphs_and_failures() {
    let Ok(rank) = std::env::var("REXL3_ROCE_RANK") else {
        eprintln!("not run: REXL3_ROCE_RANK is not set");
        return;
    };
    let rank: usize = rank.parse().expect("REXL3_ROCE_RANK is 0 or 1");
    assert!(rank < 2);
    let peer = std::env::var("REXL3_ROCE_PEER").expect("REXL3_ROCE_PEER=HOST:PORT (rank 0's address)");
    if !device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    if let Err(e) = roce::available() {
        panic!("rank {rank}: the transport is not available here: {e}");
    }
    let timeout: u64 = std::env::var("REXL3_ROCE_TEST_TIMEOUT_S").ok().and_then(|v| v.parse().ok()).unwrap_or(600);
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(timeout));
        eprintln!("roce_pair: watchdog expired after {timeout} s, exiting");
        std::process::exit(3);
    });
    ck(gb_cuda::rt::init_context(), "init_context");
    let st = ck(Stream::new_nonblocking(), "stream");
    let mut side = rendezvous(rank, &peer);
    // NCCL, the reference that moves the same bytes
    let id = if rank == 0 { ck(NcclComm::new_unique_id(), "ncclGetUniqueId").as_bytes().to_vec() } else { Vec::new() };
    let got = side.swap(&id);
    let id = NcclUniqueId::from_bytes(if rank == 0 { &id } else { &got }).expect("128-byte id");
    let nccl = ck(NcclComm::init(rank, 2, &id), "ncclCommInitRank");
    eprintln!("roce_pair rank {rank}: nccl {} up", ck(NcclComm::version(), "version"));
    let cfg = RoceConfig::from_env().expect("REXL3_ROCE_* configuration");
    let rc = transport(rank, &cfg, &mut side);

    // ---- (a) eager gathers against NCCL and the host -------------------------------------------------------------------------------
    let sizes = [4usize, 12, 4096, 4100, 65536, 300_004, 1 << 20, (2 << 20) - 64, 5 << 20, 24 << 20];
    let mut cases = 0;
    for (si, &n) in sizes.iter().enumerate() {
        for &(so, ro) in &[(0usize, 0usize), (4, 260), (256, 4)] {
            if n > (2 << 20) && so != 0 {
                continue; // the bulk sizes once, at offset 0
            }
            let seed = 0x5000 + si as u64 * 16 + so as u64;
            let mine = bytes_of(&words(seed ^ rank as u64, n.div_ceil(4)), n);
            let theirs = bytes_of(&words(seed ^ (1 - rank) as u64, n.div_ceil(4)), n);
            let mut send_h = vec![CANARY; so + n + 64];
            send_h[so..so + n].copy_from_slice(&mine);
            let send = ck(DevBuf::from_slice(&send_h), "upload send");
            let mut out_r = ck(DevBuf::from_slice(&vec![CANARY; ro + 2 * n + 64]), "upload out");
            let mut out_n = ck(DevBuf::from_slice(&vec![CANARY; ro + 2 * n + 64]), "upload out");
            ck(gb_cuda::rt::device_sync(), "sync uploads");
            let site = rc.site_for(n).expect("a site holds it");
            ck(rc.allgather(site, &send, so, &mut out_r, ro, n, &st), "roce allgather");
            // NCCL: this rank's slot by a device copy, the peer's by ncclSend / ncclRecv
            ck(out_n.copy_range_from(ro + rank * n, &send, so, n, &st), "own slot");
            ck(nccl.sendrecv_bytes(&send, so, n, &mut out_n, ro + (1 - rank) * n, n, &st), "nccl sendrecv");
            ck(st.sync(), "sync");
            if let Some(e) = rc.async_error() {
                panic!("rank {rank}: {e}");
            }
            let r: Vec<u8> = ck(out_r.to_vec(), "download");
            let q: Vec<u8> = ck(out_n.to_vec(), "download");
            let (a, b) = if rank == 0 { (&mine, &theirs) } else { (&theirs, &mine) };
            assert!(
                r[ro..ro + n] == a[..] && r[ro + n..ro + 2 * n] == b[..],
                "rank {rank}: {n} B at {so}/{ro}: not [rank 0 | rank 1]"
            );
            assert!(r == q, "rank {rank}: {n} B at {so}/{ro}: differs from NCCL's bytes");
            assert!(
                r[..ro].iter().chain(r[ro + 2 * n..].iter()).all(|&x| x == CANARY),
                "rank {rank}: {n} B: written outside the destination"
            );
            cases += 1;
        }
    }
    side.barrier();
    eprintln!("roce_pair rank {rank}: (a) {cases} eager gathers equal to NCCL and the host, 4 B to 24 MiB");

    // ---- (b) in-place f32 sums against ncclAllReduce --------------------------------------------------------------------------------
    let mut sums = 0;
    for (ci, &count) in [1usize, 1023, 4096, 16384, 100_003, 262_144, 1 << 20, 5 << 20].iter().enumerate() {
        for &disjoint in &[true, false] {
            for &off in &[0usize, 16] {
                let seed = 0x9000 + ci as u64 * 8 + disjoint as u64 * 2 + (off / 16) as u64;
                let a = floats(seed, count, !disjoint);
                let b = floats(seed ^ 0xABCD, count, !disjoint);
                let mine: Vec<f32> = (0..count)
                    .map(|i| {
                        let v = if rank == 0 { a[i] } else { b[i] };
                        if disjoint && owner(i) != rank {
                            -0.0
                        } else {
                            v
                        }
                    })
                    .collect();
                let pad = off / 4;
                let mut h = vec![f32::from_bits(0x7FC0_1234); pad + count + 8];
                h[pad..pad + count].copy_from_slice(&mine);
                let mut br = ck(DevBuf::from_slice(&h), "upload");
                let mut bn = ck(DevBuf::from_slice(&h), "upload");
                ck(gb_cuda::rt::device_sync(), "sync");
                let site = rc.site_for(count * 4).expect("site");
                ck(rc.sum_f32(site, &mut br, off, count, &st), "roce sum");
                ck(nccl.allreduce_sum_f32(&mut bn, off, count, &st), "nccl allreduce");
                ck(st.sync(), "sync");
                let r: Vec<u32> = ck(br.to_vec::<f32>(), "download").iter().map(|x| x.to_bits()).collect();
                let q: Vec<u32> = ck(bn.to_vec::<f32>(), "download").iter().map(|x| x.to_bits()).collect();
                if let Some(i) = r.iter().zip(&q).position(|(x, y)| x != y) {
                    panic!(
                        "rank {rank}: sum of {count} (disjoint {disjoint}, off {off}): element {i} roce {:#010x} nccl {:#010x}",
                        r[i], q[i]
                    );
                }
                if disjoint {
                    for i in 0..count {
                        let want = if owner(i) == 0 { a[i] } else { b[i] };
                        assert_eq!(r[pad + i], want.to_bits(), "rank {rank}: disjoint sum element {i} is not the owner's");
                    }
                }
                sums += 1;
            }
        }
    }
    side.barrier();
    eprintln!("roce_pair rank {rank}: (b) {sums} in-place sums bit-equal to ncclAllReduce (disjoint: the owner's element)");

    // ---- (c) swap, different sizes each way -----------------------------------------------------------------------------------------
    {
        let (n0, n1) = (5000usize, 70_000usize);
        let (send_n, recv_n) = if rank == 0 { (n0, n1) } else { (n1, n0) };
        let mine = bytes_of(&words(0xC0DE + rank as u64, send_n.div_ceil(4)), send_n);
        let theirs = bytes_of(&words(0xC0DE + (1 - rank) as u64, recv_n.div_ceil(4)), recv_n);
        let send = ck(DevBuf::from_slice(&mine), "upload");
        let mut out = ck(DevBuf::from_slice(&vec![CANARY; recv_n + 64]), "upload");
        ck(gb_cuda::rt::device_sync(), "sync");
        let site = rc.site_for(n1).expect("site");
        ck(rc.swap(site, &send, 0, send_n, &mut out, 0, recv_n, &st), "roce swap");
        ck(st.sync(), "sync");
        let r: Vec<u8> = ck(out.to_vec(), "download");
        assert!(r[..recv_n] == theirs[..] && r[recv_n..].iter().all(|&x| x == CANARY), "rank {rank}: swap");
    }
    side.barrier();
    eprintln!("roce_pair rank {rank}: (c) a swap of 5,000 B one way and 70,000 B the other");

    // ---- (c2) zero-copy swaps on the largest site, standalone swaps of the same site between them ------------------------------------
    let site_zc = rc.site_for(24 << 20).expect("site");
    {
        const N: usize = 16 << 20;
        let mut zc = 0;
        for k in 0..9u64 {
            let mine = bytes_of(&words(0xC2C2 + 16 * k + rank as u64, N / 4), N);
            let theirs = bytes_of(&words(0xC2C2 + 16 * k + (1 - rank) as u64, N / 4), N);
            if k % 3 == 1 {
                // a standalone swap at the same site: the parity the host tracks moves with it
                let send = ck(DevBuf::from_slice(&mine), "upload");
                let mut out = ck(DevBuf::zeroed(N), "out");
                ck(gb_cuda::rt::device_sync(), "sync");
                ck(rc.swap(site_zc, &send, 0, N, &mut out, 0, N, &st), "roce swap");
                ck(st.sync(), "sync");
                let r: Vec<u8> = ck(out.to_vec(), "download");
                assert!(r == theirs, "rank {rank}: standalone swap {k} between the zero-copy ones");
                continue;
            }
            let mut halves = rc.next_halves(site_zc).expect("an eager site has zero-copy halves");
            ck(halves[rank].upload(&mine), "write own half");
            ck(gb_cuda::rt::device_sync(), "sync");
            ck(rc.exchange(site_zc, None, None, N, N, 0, roce::Mode::None, &st), "zero-copy publish");
            ck(st.sync(), "sync");
            if let Some(e) = rc.async_error() {
                panic!("rank {rank}: {e}");
            }
            // a half is the site's whole cap (24 MiB): the message is its first N bytes
            let r: Vec<u8> = ck(halves[1 - rank].to_vec(), "download peer half");
            assert!(r[..N] == theirs[..], "rank {rank}: zero-copy swap {k}: the peer's half differs");
            let o: Vec<u8> = ck(halves[rank].to_vec(), "download own half");
            assert!(o[..N] == mine[..], "rank {rank}: zero-copy swap {k}: the own half changed");
            zc += 1;
        }
        side.barrier();
        eprintln!("roce_pair rank {rank}: (c2) {zc} zero-copy swaps of 16 MiB with standalone swaps of the same site between: every byte right");
    }

    // ---- (d) a graph of 12 exchanges on two sites, replayed with fresh data --------------------------------------------------------
    {
        const G: usize = 16 << 10; // 16 KiB gathers (one rail) and 256 KiB sums (both rails)
        const S: usize = 64 << 10; // f32 elements of the sum
        let mut seed_g = ck(DevBuf::zeroed(G), "seed");
        let mut seed_s = ck(DevBuf::zeroed(S * 4), "seed");
        let mut send = ck(DevBuf::zeroed(G), "send");
        let mut outs: Vec<DevBuf> = (0..4).map(|_| ck(DevBuf::zeroed(2 * G), "out")).collect();
        let mut sum = ck(DevBuf::zeroed(S * 4), "sum");
        let site_g = 0usize;
        let site_s = rc.site_for(S * 4).expect("site");
        assert_ne!(site_g, site_s);
        let upload = |seed: u64, seed_g: &mut DevBuf, seed_s: &mut DevBuf| -> (Vec<u8>, Vec<u8>, Vec<f32>) {
            let m = bytes_of(&words(seed + rank as u64, G / 4), G);
            let t = bytes_of(&words(seed + (1 - rank) as u64, G / 4), G);
            let a = floats(seed, S, false);
            let b = floats(seed ^ 0x77, S, false);
            let mine: Vec<f32> = (0..S)
                .map(|i| {
                    if owner(i) == rank {
                        if rank == 0 {
                            a[i]
                        } else {
                            b[i]
                        }
                    } else {
                        -0.0
                    }
                })
                .collect();
            let want: Vec<f32> = (0..S).map(|i| if owner(i) == 0 { a[i] } else { b[i] }).collect();
            ck(seed_g.upload(&m), "upload");
            ck(seed_s.upload(&mine), "upload");
            ck(gb_cuda::rt::device_sync(), "sync");
            (m, t, want)
        };
        let check = |what: &str, outs: &[DevBuf], sum: &DevBuf, m: &[u8], t: &[u8], want: &[f32]| {
            let (a, b) = if rank == 0 { (m, t) } else { (t, m) };
            for (k, o) in outs.iter().enumerate() {
                let r: Vec<u8> = ck(o.to_vec(), "download");
                assert!(r[..G] == a[..] && r[G..] == b[..], "rank {rank}: {what}: gather {k} wrong");
            }
            let r: Vec<f32> = ck(sum.to_vec(), "download");
            assert!(r.iter().zip(want).all(|(x, y)| x.to_bits() == y.to_bits()), "rank {rank}: {what}: sum wrong");
        };
        let (mut m, mut t, mut want) = upload(0xD000, &mut seed_g, &mut seed_s);
        side.barrier();
        let graph = {
            let cap = ck(Graph::begin(&st), "begin capture");
            let ran = (|| -> Result<(), CudaError> {
                for k in 0..4 {
                    send.copy_from(&seed_g, G, &st)?;
                    rc.allgather(site_g, &send, 0, &mut outs[k], 0, G, &st)?;
                    sum.copy_from(&seed_s, S * 4, &st)?;
                    rc.sum_f32(site_s, &mut sum, 0, S, &st)?;
                    rc.allgather(site_g, &send, 0, &mut outs[k], 0, G, &st)?; // the same again: the other parity
                }
                Ok(())
            })();
            match cap.finish(ran) {
                Ok(g) => g,
                Err(e) => panic!("rank {rank}: capture failed: {e}"),
            }
        };
        let mut replays = 0;
        let t0 = Instant::now();
        for r in 0..300 {
            if r % 30 == 0 && r > 0 {
                ck(st.sync(), "sync");
                check(&format!("replay {r}"), &outs, &sum, &m, &t, &want);
                side.barrier();
                (m, t, want) = upload(0xD000 + r as u64 * 7, &mut seed_g, &mut seed_s);
                // an eager exchange between replays: the sequences live on the device and carry across
                ck(send.copy_from(&seed_g, G, &st), "copy");
                ck(rc.allgather(site_g, &send, 0, &mut outs[0], 0, G, &st), "eager between replays");
                side.barrier();
            }
            ck(graph.launch(&st), "replay");
            replays += 1;
        }
        ck(st.sync(), "sync");
        let per = t0.elapsed().as_secs_f64() * 1e6 / (replays * 12) as f64;
        check("last replay", &outs, &sum, &m, &t, &want);
        if let Some(e) = rc.async_error() {
            panic!("rank {rank}: {e}");
        }
        drop(graph);
        // lane T: the host no longer knows these sites' parity (replays advanced them): no zero-copy halves; the eager site keeps its
        assert!(
            rc.next_halves(site_g).is_none() && rc.next_halves(site_s).is_none(),
            "rank {rank}: a captured site offered halves"
        );
        assert!(rc.next_halves(site_zc).is_some(), "rank {rank}: the eager site lost its halves");
        side.barrier();
        eprintln!(
            "roce_pair rank {rank}: (d) {replays} replays of a 12-exchange graph (both parities, two sites), fresh data every 30 and eager \
             exchanges between: all bit-exact ({per:.1} us per exchange incl. the copies; proxy {:?})",
            rc.proxy_stats()
        );
    }
    // ---- (d2) two streams at once: small gathers on one site, bulk gathers on another, interleaved DIFFERENTLY on the two ranks -------
    // (order is per site: each site sees the same sequence of exchanges on both ranks; across sites any interleaving is allowed)
    {
        let st2 = ck(Stream::new_nonblocking(), "stream 2");
        const NS: usize = 16 << 10;
        const NB: usize = 8 << 20;
        const KS: usize = 64;
        const KB: usize = 6;
        let site_s = rc.site_for(NS).expect("site");
        let site_b = rc.site_for(NB).expect("site");
        assert_ne!(site_s, site_b);
        let ms = bytes_of(&words(0xE100 + rank as u64, NS / 4 * KS), NS * KS);
        let ts = bytes_of(&words(0xE100 + (1 - rank) as u64, NS / 4 * KS), NS * KS);
        let mb = bytes_of(&words(0xE200 + rank as u64, NB / 4 * KB), NB * KB);
        let tb = bytes_of(&words(0xE200 + (1 - rank) as u64, NB / 4 * KB), NB * KB);
        let send_s = ck(DevBuf::from_slice(&ms), "upload");
        let send_b = ck(DevBuf::from_slice(&mb), "upload");
        let mut out_s = ck(DevBuf::zeroed(2 * NS * KS), "out");
        let mut out_b = ck(DevBuf::zeroed(2 * NB * KB), "out");
        ck(gb_cuda::rt::device_sync(), "sync");
        side.barrier();
        let mut b = 0;
        let bulk =
            |b: usize, out_b: &mut DevBuf| ck(rc.allgather(site_b, &send_b, b * NB, out_b, 2 * NB * b, NB, &st2), "bulk gather");
        if rank == 0 {
            while b < KB {
                bulk(b, &mut out_b);
                b += 1;
            }
        }
        for k in 0..KS {
            ck(rc.allgather(site_s, &send_s, k * NS, &mut out_s, 2 * NS * k, NS, &st), "small gather");
            if rank == 1 && k % 8 == 3 && b < KB {
                bulk(b, &mut out_b);
                b += 1;
            }
        }
        while b < KB {
            bulk(b, &mut out_b);
            b += 1;
        }
        ck(st.sync(), "sync");
        ck(st2.sync(), "sync 2");
        if let Some(e) = rc.async_error() {
            panic!("rank {rank}: {e}");
        }
        let (a, z) = if rank == 0 { (&ms, &ts) } else { (&ts, &ms) };
        let r: Vec<u8> = ck(out_s.to_vec(), "download");
        for k in 0..KS {
            let o = 2 * NS * k;
            assert!(
                r[o..o + NS] == a[k * NS..(k + 1) * NS] && r[o + NS..o + 2 * NS] == z[k * NS..(k + 1) * NS],
                "rank {rank}: two streams: small gather {k} wrong"
            );
        }
        let (a, z) = if rank == 0 { (&mb, &tb) } else { (&tb, &mb) };
        let r: Vec<u8> = ck(out_b.to_vec(), "download");
        for k in 0..KB {
            let o = 2 * NB * k;
            assert!(
                r[o..o + NB] == a[k * NB..(k + 1) * NB] && r[o + NB..o + 2 * NB] == z[k * NB..(k + 1) * NB],
                "rank {rank}: two streams: bulk gather {k} wrong"
            );
        }
        side.barrier();
        eprintln!(
            "roce_pair rank {rank}: (d2) two streams at once, {KS} gathers of 16 KiB on one site and {KB} of 8 MiB on another, queued \
             in a different interleaving on each rank: all bit-exact"
        );
    }
    drop(rc);
    side.barrier();

    // ---- (e) lost peer: a skipped exchange times out; then a peer whose transport is gone --------------------------------------------
    let mut short = cfg.clone();
    short.timeout_ms = 2000;
    {
        let rc = transport(rank, &short, &mut side);
        let send = ck(DevBuf::from_slice(&vec![1u8; 4096]), "upload");
        let mut out = ck(DevBuf::from_slice(&vec![CANARY; 8192]), "upload");
        ck(gb_cuda::rt::device_sync(), "sync");
        if rank == 0 {
            let t0 = Instant::now();
            ck(rc.allgather(0, &send, 0, &mut out, 0, 4096, &st), "the exchange the peer skips");
            ck(st.sync(), "sync after the timeout");
            let dt = t0.elapsed().as_secs_f64();
            let e = rc.async_error().expect("the timeout is recorded");
            eprintln!("roce_pair rank 0: (e) the skipped exchange ended after {dt:.2} s: {e}");
            assert!((1.5..10.0).contains(&dt), "the wait ended after {dt} s, not ~2 s");
            let r: Vec<u8> = ck(out.to_vec(), "download");
            assert!(r[4096..].iter().all(|&x| x == CANARY), "a failed exchange wrote the peer's half");
            assert!(rc.allgather(0, &send, 0, &mut out, 0, 4096, &st).is_err(), "a poisoned transport refuses");
        }
        side.barrier();
    }
    side.barrier();
    {
        let rc = transport(rank, &short, &mut side);
        if rank == 1 {
            drop(rc); // its QPs, MR and region are gone
            side.barrier();
            side.barrier();
        } else {
            side.barrier();
            let send = ck(DevBuf::from_slice(&vec![2u8; 65536]), "upload");
            let mut out = ck(DevBuf::zeroed(2 * 65536), "out");
            ck(gb_cuda::rt::device_sync(), "sync");
            let t0 = Instant::now();
            ck(rc.allgather(0, &send, 0, &mut out, 0, 65536, &st), "an exchange with a dead peer");
            ck(st.sync(), "sync");
            let e = rc.async_error().expect("the dead peer surfaces");
            eprintln!("roce_pair rank 0: (e) an exchange with a gone peer ended after {:.2} s: {e}", t0.elapsed().as_secs_f64());
            side.barrier();
        }
    }
    side.barrier();

    // ---- (f) abort from another thread while a wait spins ----------------------------------------------------------------------------
    {
        let rc = transport(rank, &cfg, &mut side);
        if rank == 0 {
            let send = ck(DevBuf::from_slice(&vec![3u8; 4096]), "upload");
            let mut out = ck(DevBuf::zeroed(8192), "out");
            ck(gb_cuda::rt::device_sync(), "sync");
            ck(rc.allgather(0, &send, 0, &mut out, 0, 4096, &st), "the exchange that gets aborted");
            let r2 = rc.clone();
            let t_abort = Instant::now();
            let h = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(300));
                r2.abort();
                Instant::now()
            });
            ck(st.sync(), "sync after the abort");
            let at = h.join().expect("abort thread");
            let after = at.elapsed().as_secs_f64();
            eprintln!(
                "roce_pair rank 0: (f) abort after {:.2} s released the wait {:.3} s later",
                (at - t_abort).as_secs_f64(),
                after
            );
            assert!(after < 1.0, "the stream took {after} s to finish after the abort");
            assert!(rc.is_aborted());
            assert!(rc.allgather(0, &send, 0, &mut out, 0, 4096, &st).is_err(), "an aborted transport refuses");
        }
        side.barrier();
    }
    side.barrier();

    // ---- (g) restart: fresh transports work again ------------------------------------------------------------------------------------
    {
        let rc = transport(rank, &cfg, &mut side);
        for &n in &[4096usize, 1 << 20] {
            let mine = bytes_of(&words(0xE000 + n as u64 + rank as u64, n / 4), n);
            let theirs = bytes_of(&words(0xE000 + n as u64 + (1 - rank) as u64, n / 4), n);
            let send = ck(DevBuf::from_slice(&mine), "upload");
            let mut out = ck(DevBuf::zeroed(2 * n), "out");
            ck(gb_cuda::rt::device_sync(), "sync");
            ck(rc.allgather(rc.site_for(n).unwrap(), &send, 0, &mut out, 0, n, &st), "after restart");
            ck(st.sync(), "sync");
            let r: Vec<u8> = ck(out.to_vec(), "download");
            let (a, b) = if rank == 0 { (&mine, &theirs) } else { (&theirs, &mine) };
            assert!(r[..n] == a[..] && r[n..] == b[..], "rank {rank}: wrong after restart");
        }
        assert!(rc.async_error().is_none());
        side.barrier();
        eprintln!("roce_pair rank {rank}: (g) restart after the aborts: correct");
    }
    // ---- (h) two users of one size class: their own sites, two streams, interleaved differently on the two ranks ---------------------
    {
        const N: usize = 6 << 20;
        const K: usize = 6;
        let rc = transport_caps(rank, &cfg, &mut side, &[8 << 20, 8 << 20]);
        assert!(
            !rc.is_aborted() && rc.async_error().is_none(),
            "rank {rank}: (h) the new transport is not healthy: {:?}",
            rc.async_error()
        );
        let (user_a, user_b) = ([0usize], [1usize]);
        let site_a = rc.site_for_among(&user_a, N).expect("user A's site");
        let site_b = rc.site_for_among(&user_b, N).expect("user B's site");
        assert!(site_a != site_b && rc.site_for_among(&user_a, (8 << 20) + 1).is_none());
        let st_b = ck(Stream::new_nonblocking(), "stream B");
        let ma = bytes_of(&words(0xF100 + rank as u64, N / 4 * K), N * K);
        let ta = bytes_of(&words(0xF100 + (1 - rank) as u64, N / 4 * K), N * K);
        let mb = bytes_of(&words(0xF200 + rank as u64, N / 4 * K), N * K);
        let tb = bytes_of(&words(0xF200 + (1 - rank) as u64, N / 4 * K), N * K);
        let send_a = ck(DevBuf::from_slice(&ma), "upload");
        let send_b = ck(DevBuf::from_slice(&mb), "upload");
        let mut out_b = ck(DevBuf::zeroed(N * K), "out");
        let mut got_a = vec![0u8; N * K];
        ck(gb_cuda::rt::device_sync(), "sync");
        side.barrier();
        // user B's swaps queued on stream B all at once (rank 0) or after user A's (rank 1); user A's zero-copy swaps one at a time
        let queue_b = |out_b: &mut DevBuf| {
            for k in 0..K {
                ck(rc.swap(site_b, &send_b, k * N, N, out_b, k * N, N, &st_b), "user B swap");
            }
        };
        if rank == 0 {
            queue_b(&mut out_b);
        }
        for k in 0..K {
            let mut h = rc.next_halves(site_a).expect("user A's eager site has halves");
            // stream-ordered (no device-wide sync: user B's swaps may be spinning on stream B, waiting for a peer that is busy here)
            ck(h[rank].copy_range_from(0, &send_a, k * N, N, &st), "write own half");
            if let Err(e) = rc.exchange(site_a, None, None, N, N, 0, roce::Mode::None, &st) {
                panic!(
                    "rank {rank}: (h) user A zero-copy publish {k}: {e} (aborted {}, async error {:?}, proxy {:?})",
                    rc.is_aborted(),
                    rc.async_error(),
                    rc.proxy_stats()
                );
            }
            ck(st.sync(), "sync");
            let r: Vec<u8> = ck(h[1 - rank].to_vec(), "download peer half");
            got_a[k * N..(k + 1) * N].copy_from_slice(&r[..N]);
        }
        if rank == 1 {
            queue_b(&mut out_b);
        }
        ck(st_b.sync(), "sync B");
        if let Some(e) = rc.async_error() {
            panic!("rank {rank}: {e}");
        }
        assert!(got_a == ta, "rank {rank}: user A's zero-copy swaps wrong");
        // the device's per-site counts (the path counters' source): K exchanges at each user's site
        let seqs = ck(rc.site_sequences(), "site sequences");
        assert_eq!(seqs, vec![K as u64, K as u64], "rank {rank}: the sites' device sequences");
        let r: Vec<u8> = ck(out_b.to_vec(), "download");
        assert!(r == tb, "rank {rank}: user B's swaps wrong");
        side.barrier();
        eprintln!(
            "roce_pair rank {rank}: (h) two users of one size class (8 MiB sites {site_a} / {site_b}): {K} zero-copy + {K} standalone swaps of \
             6 MiB on two streams, interleaved differently on the two ranks: every byte right"
        );
    }
    side.barrier();
    eprintln!("roce_pair rank {rank}: PASS");
}
