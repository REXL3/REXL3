//! Two-process test of the SHAPE of GLM's MoE exchange (lane E, Phase 2 step 2, `glmperf/E/PHASE2.md`): today a rank sums its pair
//! buffer `y_pairs` `[rows * topk, hidden]` f32 in place with NCCL's ring all-reduce (`families/glm/model/tp.rs`), every slot owned by
//! exactly one rank and exact `-0.0` in the other's copy. The claim: the GATHER form (each rank receives the peer's buffer, then adds the
//! two in rank order: `NcclComm::oneshot_sum_f32`, `REXL3_TP_COLLECTIVE=oneshot`) and the OWNER SELECT a compacted exchange delivers
//! (each slot copied from its owner, no add: the owned-pairs gather of the expert-partition gate's option (b)) give the same bits, and
//! the slot combine (`moe::combine_pairs`) leaves the same `mixer_out` from each. Same environment as `nccl_pair.rs` (one process per
//! rank, each under its node's GPU lock; unset it prints `not run:` and passes):
//!
//! ```text
//! REXL3_NCCL_RANK=0|1  REXL3_NCCL_RANKS=2  REXL3_NCCL_PEER=192.0.2.3:29640   (rank 0's address: it listens, rank 1 connects)
//! NCCL fabric settings: source /srv/rexl3/nccl/nccl.env, and NCCL_MAX_NCHANNELS=4
//! optional: REXL3_NCCL_TIMEOUT_S (watchdog, default 1200), REXL3_SHAPE_PHASES=scalar,glm (default both), REXL3_SHAPE_ROWS=1,4,..
//! ```
//!
//! * `scalar`: EVERY f32 bit pattern `x` (2^32, in chunks of 2^26), placed on rank 0 with `-0.0` on rank 1 and the other way round,
//!   through the ring and through the gather form: every non-NaN `x` must come back bit for bit from both (`x + -0.0 == x`, either
//!   order, signed zeros and subnormals included); the NaNs are tallied (payload kept, canonical `0x7fffffff`, other) per form.
//! * `glm`: GLM-5.3-Flash shapes (288 experts, top-8, hidden 4096, the TP=2 contiguous ownership 0..144 | 144..288) at 1 .. 2048 rows,
//!   once with finite values (normals, subnormals, signed zeros) and once with +-inf and NaN payloads mixed in: ring == gather == owner
//!   select on every element (NaNs tallied), then `combine_pairs` over each (the same picks, fp16 weights and fp32 shared) compared bit
//!   for bit.
#![allow(clippy::needless_range_loop)]

use gb_cuda::moe::{combine_pairs, MoeDims};
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

/// A pair-buffer value: a finite f32 with a random sign and mantissa; every fifth a SUBNORMAL, every 23rd an exact signed zero; with
/// `special`, also about one in 4,099 an infinity and one in 4,111 a NaN with a random payload and sign.
fn gen_f32(state: &mut u64, special: bool) -> f32 {
    let r = splitmix(state);
    let mant = (r & 0x7F_FFFF) as u32;
    let sign = ((r >> 40) & 1) as u32;
    if special {
        if (r >> 20) % 4111 == 0 {
            return f32::from_bits(sign << 31 | 0xFF << 23 | (mant | 1)); // NaN (quiet or signalling: any payload)
        }
        if (r >> 24) % 4099 == 0 {
            return f32::from_bits(sign << 31 | 0xFF << 23); // +-inf
        }
    }
    if (r >> 52) % 23 == 0 {
        return f32::from_bits(sign << 31);
    }
    if (r >> 48) % 5 == 0 {
        return f32::from_bits(sign << 31 | (mant | 1));
    }
    let exp = 127 - 8 + ((r >> 32) % 16) as u32;
    f32::from_bits(sign << 31 | exp << 23 | mant)
}

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
}

fn rendezvous(rank: usize, peer: &str, id: Option<&NcclUniqueId>) -> (Side, NcclUniqueId) {
    if rank == 0 {
        let port = peer.rsplit(':').next().expect("REXL3_NCCL_PEER is HOST:PORT");
        let l = TcpListener::bind(format!("0.0.0.0:{port}")).expect("bind the rendezvous port");
        let (mut s, from) = l.accept().expect("accept the peer");
        eprintln!("glm_exchange_shape rank 0: peer connected from {from}");
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
                    eprintln!("glm_exchange_shape rank 1: waiting for rank 0 at {peer}: {e}");
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

fn phase_wanted(name: &str) -> bool {
    match std::env::var("REXL3_SHAPE_PHASES") {
        Ok(v) if !v.trim().is_empty() => v.split(',').any(|p| p.trim() == name),
        _ => true,
    }
}

const CANONICAL_NAN: u32 = 0x7FFF_FFFF;

/// How a form treated the NaN inputs it was given: payload and sign kept, NVIDIA's canonical NaN, or another NaN / a non-NaN.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct NanTally {
    kept: u64,
    canonical: u64,
    other: u64,
}

impl NanTally {
    fn add(&mut self, input: u32, got: u32) {
        if got == input {
            self.kept += 1;
        } else if got == CANONICAL_NAN {
            self.canonical += 1;
        } else {
            self.other += 1;
        }
    }
}

/// One collective over `host` (this rank's operand) on a fresh device copy: `oneshot` = the gather form, else NCCL's ring.
fn exchange(comm: &NcclComm, host: &[f32], oneshot: bool, buf: &mut DevBuf, st: &Stream) -> Vec<f32> {
    ck(buf.upload(host), "upload");
    if oneshot {
        ck(comm.oneshot_sum_f32(buf, 0, host.len(), st), "one-shot");
    } else {
        ck(comm.allreduce_sum_f32(buf, 0, host.len(), st), "ring");
    }
    ck(st.sync(), "sync");
    let mut out = vec![0f32; host.len()];
    ck(buf.download(&mut out), "download");
    out
}

/// A positive fp16 weight as bits (exponents 2^-7 .. 2^0, random mantissa), one in 61 an exact zero.
fn gen_w16(state: &mut u64) -> u16 {
    let r = splitmix(state);
    if r % 61 == 0 {
        return 0;
    }
    let exp = 8 + ((r >> 8) % 7) as u16; // fp16 bias 15: 2^-7 .. 2^-1
    (exp << 10) | ((r >> 16) as u16 & 0x3FF)
}

#[test]
fn glm_exchange_shape_pair() {
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
    let timeout: u64 = std::env::var("REXL3_NCCL_TIMEOUT_S").ok().and_then(|v| v.parse().ok()).unwrap_or(1200);
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(timeout));
        eprintln!("glm_exchange_shape: watchdog expired after {timeout} s (a deadlock?), exiting with status 3");
        std::process::exit(3);
    });

    ck(gb_cuda::rt::init_context(), "init_context");
    let st = ck(Stream::new_nonblocking(), "compute stream");
    eprintln!(
        "glm_exchange_shape rank {rank}: libnccl {} version code {}; env {}",
        NcclComm::library_path(),
        ck(NcclComm::version(), "ncclGetVersion"),
        gb_cuda::nccl::nccl_environment().join(" ")
    );
    let id = if rank == 0 { Some(ck(NcclComm::new_unique_id(), "ncclGetUniqueId")) } else { None };
    let (mut side, id) = rendezvous(rank, &peer, id.as_ref());
    let comm = ck(NcclComm::init(rank, ranks, &id), "ncclCommInitRank");
    // staging for the largest message: y_pairs of a 2048-row chunk (2048 x 8 x 4096 f32 = 256 MiB), the scalar chunks the same
    const CHUNK: usize = 1 << 26;
    let staging = ck(comm.reserve_oneshot_staging(CHUNK * 4), "reserve staging");
    eprintln!("glm_exchange_shape rank {rank}: one-shot staging {} MiB", staging >> 20);
    {
        let mut b = ck(DevBuf::zeroed(4096 * 4), "alloc");
        ck(comm.allreduce_sum_f32(&mut b, 0, 4096, &st), "connect ring");
        ck(comm.oneshot_sum_f32(&mut b, 0, 4096, &st), "connect one-shot");
        ck(st.sync(), "sync");
    }
    side.barrier();
    let mut failures: Vec<String> = Vec::new();

    // ---- scalar: every bit pattern, both placements, both forms --------------------------------------------------------------------
    if phase_wanted("scalar") {
        let t0 = Instant::now();
        let mut buf = ck(DevBuf::alloc(CHUNK * 4), "alloc chunk");
        let neg_zero = vec![-0.0f32; CHUNK];
        let mut x = vec![0f32; CHUNK];
        let mut checked = 0u64;
        let mut nan_in = 0u64;
        // [placement][form]: form 0 ring, 1 gather
        let mut tally = [[NanTally::default(); 2]; 2];
        let mut bad = [[0u64; 2]; 2];
        let mut ring_vs_gather_nan = 0u64;
        for placement in 0..2usize {
            for c in 0..(1usize << 32) / CHUNK {
                let base = (c * CHUNK) as u32;
                for (i, v) in x.iter_mut().enumerate() {
                    *v = f32::from_bits(base.wrapping_add(i as u32));
                }
                let mine: &[f32] = if rank == placement { &x } else { &neg_zero };
                let ring = exchange(&comm, mine, false, &mut buf, &st);
                let gath = exchange(&comm, mine, true, &mut buf, &st);
                for i in 0..CHUNK {
                    let xb = x[i].to_bits();
                    let (rb, gb) = (ring[i].to_bits(), gath[i].to_bits());
                    if x[i].is_nan() {
                        if placement == 0 && rank == 0 {
                            nan_in += 1;
                        }
                        tally[placement][0].add(xb, rb);
                        tally[placement][1].add(xb, gb);
                        if rb != gb {
                            ring_vs_gather_nan += 1;
                        }
                    } else {
                        if rb != xb {
                            bad[placement][0] += 1;
                            if bad[placement][0] <= 3 {
                                eprintln!("scalar: ring placement {placement}: x {xb:#010x} -> {rb:#010x}");
                            }
                        }
                        if gb != xb {
                            bad[placement][1] += 1;
                            if bad[placement][1] <= 3 {
                                eprintln!("scalar: gather placement {placement}: x {xb:#010x} -> {gb:#010x}");
                            }
                        }
                        if placement == 0 && rank == 0 {
                            checked += 1;
                        }
                    }
                }
            }
            side.barrier();
        }
        for p in 0..2 {
            for (f, name) in ["ring", "gather"].iter().enumerate() {
                if bad[p][f] != 0 {
                    failures.push(format!("scalar: {name} placement {p}: {} non-NaN values changed", bad[p][f]));
                }
            }
        }
        eprintln!(
            "glm_exchange_shape rank {rank}: SCALAR every f32 bit pattern, x on rank p and -0.0 on the other, p = 0 and 1: non-NaN x \
             returned bit for bit: ring {} / {} bad, gather {} / {} bad (of {checked} non-NaN values a placement); NaN inputs ({nan_in}): \
             ring p0 {:?} p1 {:?}, gather p0 {:?} p1 {:?}; NaN results that differ between ring and gather: {ring_vs_gather_nan}; {:.1} s",
            bad[0][0],
            bad[1][0],
            bad[0][1],
            bad[1][1],
            tally[0][0],
            tally[1][0],
            tally[0][1],
            tally[1][1],
            t0.elapsed().as_secs_f64()
        );
    }

    // ---- glm: the y_pairs layout, ring vs gather vs owner select, and the combine over each ----------------------------------------
    if phase_wanted("glm") {
        let d = MoeDims::GLM53;
        let half = d.experts / 2; // ExpertOwnership::range at TP=2: rank 0 owns 0..144, rank 1 144..288
        let rows_list: Vec<usize> = std::env::var("REXL3_SHAPE_ROWS")
            .ok()
            .map(|v| v.split(',').map(|x| x.trim().parse().expect("REXL3_SHAPE_ROWS")).collect())
            .unwrap_or_else(|| vec![1, 4, 16, 64, 512, 2048]);
        for &special in &[false, true] {
            for &rows in &rows_list {
                let t0 = Instant::now();
                let p = rows * d.topk;
                let n = p * d.hidden;
                let mut s = 0x5EED_0000 ^ (rows as u64) << 8 ^ special as u64;
                // the replicated routing: 8 distinct experts a row, fp16 weights, fp32 shared
                let mut pick = vec![0i32; p];
                for r in 0..rows {
                    let mut k = 0;
                    while k < d.topk {
                        let e = (splitmix(&mut s) % d.experts as u64) as i32;
                        if !pick[r * d.topk..r * d.topk + k].contains(&e) {
                            pick[r * d.topk + k] = e;
                            k += 1;
                        }
                    }
                }
                let wts: Vec<u16> = (0..p).map(|_| gen_w16(&mut s)).collect();
                let shared: Vec<f32> = (0..rows * d.hidden).map(|_| gen_f32(&mut s, false)).collect();
                // the true pair buffer (what each slot's owner computes), and this rank's copy: its own slots, -0.0 elsewhere
                let truth: Vec<f32> = (0..n).map(|_| gen_f32(&mut s, special)).collect();
                let owner = |slot: usize| if (pick[slot] as usize) < half { 0 } else { 1 };
                let mine: Vec<f32> = (0..n).map(|i| if owner(i / d.hidden) == rank { truth[i] } else { -0.0 }).collect();
                let owned0 = (0..p).filter(|&q| owner(q) == 0).count();
                let mut buf = ck(DevBuf::alloc(n * 4), "alloc y");
                let ring = exchange(&comm, &mine, false, &mut buf, &st);
                let gath = exchange(&comm, &mine, true, &mut buf, &st);
                // the owner select (a compacted exchange's result after its scatter): each slot copied from its owner, no add
                let sel = &truth;
                let mut nan_t = [NanTally::default(); 2];
                let mut bad = [0u64; 3]; // ring vs gather, ring vs select (non-NaN), gather vs select (non-NaN)
                let mut nans = 0u64;
                for i in 0..n {
                    let (rb, gb, sb) = (ring[i].to_bits(), gath[i].to_bits(), sel[i].to_bits());
                    if rb != gb {
                        bad[0] += 1;
                    }
                    if sel[i].is_nan() {
                        nans += 1;
                        nan_t[0].add(sb, rb);
                        nan_t[1].add(sb, gb);
                    } else {
                        bad[1] += (rb != sb) as u64;
                        bad[2] += (gb != sb) as u64;
                    }
                }
                // the combine over each form's buffer: the same picks, weights and shared
                let pick_d = ck(DevBuf::from_slice(&pick), "pick");
                let wts_d = ck(DevBuf::from_slice(&wts), "wts");
                let sh_d = ck(DevBuf::from_slice(&shared), "shared");
                let mut outs: Vec<Vec<u32>> = Vec::new();
                for y in [&ring, &gath, sel] {
                    let y_d = ck(DevBuf::from_slice(y), "y");
                    let mut out_d = ck(DevBuf::zeroed(rows * d.hidden * 4), "out");
                    ck(combine_pairs(&y_d, &pick_d, &wts_d, Some(&sh_d), &mut out_d, rows, &d, &st), "combine_pairs");
                    ck(st.sync(), "sync");
                    let o: Vec<f32> = ck(out_d.to_vec(), "out");
                    outs.push(o.iter().map(|v| v.to_bits()).collect());
                }
                let diff = |a: &[u32], b: &[u32]| a.iter().zip(b).filter(|(x, y)| x != y).count();
                let (c_rg, c_rs) = (diff(&outs[0], &outs[1]), diff(&outs[0], &outs[2]));
                let out_nan = outs[0].iter().filter(|&&b| f32::from_bits(b).is_nan()).count();
                let out_nan_payloads: std::collections::BTreeSet<u32> =
                    outs.iter().flatten().copied().filter(|&b| f32::from_bits(b).is_nan()).collect();
                eprintln!(
                    "glm_exchange_shape rank {rank}: GLM rows {rows} ({}): {n} elements, {owned0} / {p} slots owned by rank 0; ring != gather \
                     {}; non-NaN ring != select {}, gather != select {}; NaN inputs {nans}: ring {:?}, gather {:?}; combine out (rows x 4096): \
                     ring != gather {c_rg}, ring != select {c_rs}, NaN outputs {out_nan} (payloads {:x?}); {:.1} s",
                    if special { "with inf and NaN payloads" } else { "finite" },
                    bad[0],
                    bad[1],
                    bad[2],
                    nan_t[0],
                    nan_t[1],
                    out_nan_payloads.iter().take(4).collect::<Vec<_>>(),
                    t0.elapsed().as_secs_f64()
                );
                if bad[1] != 0 || bad[2] != 0 {
                    failures.push(format!("glm rows {rows} special={special}: non-NaN elements differ from the owner's value"));
                }
                if !special && (bad[0] != 0 || c_rg != 0 || c_rs != 0) {
                    failures.push(format!("glm rows {rows} finite: the forms or their combine outputs differ"));
                }
                if special && (c_rg != 0 || c_rs != 0) {
                    failures.push(format!(
                        "glm rows {rows} with NaN payloads: combine outputs differ (ring/gather {c_rg}, ring/select {c_rs})"
                    ));
                }
                side.barrier();
            }
        }
    }
    for f in &failures {
        eprintln!("glm_exchange_shape rank {rank}: FAILED {f}");
    }
    assert!(failures.is_empty(), "{} failure(s)", failures.len());
    eprintln!("glm_exchange_shape rank {rank}: all checks passed");
}
