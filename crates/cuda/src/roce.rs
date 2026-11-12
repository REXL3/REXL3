//! REXL3's two-rank RoCE all-gather (`docs/ROCE_GATHER.md`): one-shot RDMA writes between two GB10 nodes over their ConnectX-7 rails,
//! signalled from inside REXL3's own kernels, with a proxy thread posting the work requests. It replaces NCCL for the two-rank
//! exchanges of the tensor-parallel engine at the sizes it measures faster (all of them on the CRS812 pair, `docs/ROCE_GATHER.md` 5).
//!
//! The pieces:
//! - `csrc/roce_shim.cu` (host): libibverbs opened at run time (no link-time dependency, the way [`crate::nccl`] opens NCCL), the queue
//!   pairs of every rail, and the proxy thread that turns GPU-written descriptors into RDMA writes, each piece followed by an inline
//!   flag write on the same QP.
//! - `csrc/roce_dev.cuh` (device): what a kernel calls to publish its output to the peer (ONE system-scope release per message) and
//!   to wait for the peer's (`ld.acquire.sys` on a flag, abort word, `%globaltimer` deadline). Sequences live in device memory, so a
//!   CUDA graph replays exchanges with new sequences and eager and graph launches mix.
//! - `csrc/roce_gather.cu` (device): the standalone exchange behind [`RoceComm::exchange`]: stage in, publish, wait, gather / swap /
//!   f32 sum, in ONE kernel launched with programmatic dependent launch.
//!
//! [`RoceComm`] owns the registered region (`cudaHostAlloc` mapped memory registered on every rail, never with relaxed ordering),
//! the device control words and the shim context. Both ranks build it from the SAME site list (the layout hash is checked at
//! connect), exchange [`RoceComm::blob`]s over any side channel, [`RoceComm::connect`], meet at a barrier, then [`RoceComm::start`]
//! the proxy. Every exchange of a site must be issued in the same order on both ranks, on one stream per site (the NCCL rule).
//!
//! Exactness: a gather and a swap are byte copies; the sum is `add.rn.f32` without flush-to-zero, rank 0's operand first, so over
//! buffers of disjoint support (`-0.0` filler) it is bit-identical to NCCL's two-rank ring (`crates/cuda/tests/roce_pair.rs`).

use crate::rt::{Block, CudaError, DevBuf, Stream};
use core::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// `CudaError` code base of this module's errors (an `NCCL_ERROR_BASE`-style band of its own).
pub const ROCE_ERROR_BASE: i32 = 21_000;
/// The transport is not built (no `<infiniband/verbs.h>` on the build host), `libibverbs.so.1` is missing, or no RDMA device opened.
pub const ERR_UNAVAILABLE: i32 = ROCE_ERROR_BASE + 1;
/// An argument this binding refuses before launching anything (a size above the site's cap, a site that does not exist, ...).
pub const ERR_BAD_ARGUMENT: i32 = ROCE_ERROR_BASE + 2;
/// The transport was aborted ([`RoceComm::abort`]) or recorded an error ([`RoceComm::async_error`]): every later exchange fails.
pub const ERR_ABORTED: i32 = ROCE_ERROR_BASE + 3;
/// Setup failed (registration, queue pairs, the peer's blob).
pub const ERR_SETUP: i32 = ROCE_ERROR_BASE + 4;

/// Header words (byte offsets into the region), as `roce_abi.h`.
const H_ABORT: usize = 64;
const H_ERROR: usize = 128;
const H_PROXY: usize = 1024;
const HEADER_BYTES: usize = 4096;
const RING_ENTRIES: u32 = 256;
const C_SITE0: usize = 16;
const C_PER_SITE: usize = 4;
/// Version of the protocol; part of the layout hash and the blob.
pub const VERSION: u32 = 1;
/// Bytes of one rank's blob.
pub const BLOB_BYTES: usize = 512;

/// The exchange's result (the kernel's `mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Mode {
    /// `dst = [rank 0's bytes | rank 1's bytes]`, `stride` bytes apart (`ncclAllGather`'s layout).
    Gather = 0,
    /// `dst = the peer's bytes` (`TpComm::swap_bytes`).
    Swap = 1,
    /// `dst[i] = rank0[i] + rank1[i]`, f32, `add.rn` without flush-to-zero (`TpComm::allgather_sum_f32`, in place allowed).
    SumF32 = 2,
    /// No result: a fused consumer reads the site's halves itself.
    None = 3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GbRoceDev {
    pub region: usize,
    pub ctl: usize,
    pub gen_hi: u64,
    pub timeout_ns: u64,
    pub ring_off: u64,
    pub flags_off: u64,
    pub bulk_min: u64,
    pub chunk: u64,
    pub chunk_min: u64,
    pub ring_mask: u32,
    pub rails: u32,
    pub rank: u32,
    pub stripe_min: u32,
    pub ctl_chunks: u32,
    pub ctl_ready: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GbRoceSite {
    /// Byte offsets `[parity][rank]` of a rank's half in the region.
    pub half: [[u64; 2]; 2],
    pub cap: u64,
    pub flag: [u32; 2],
    pub id: u32,
    pub max_pieces: u32,
}

#[repr(C)]
struct GbRoceOpen {
    region: *mut c_void,
    region_bytes: u64,
    ring_off: u64,
    flags_off: u64,
    ring_entries: u32,
    flag_slots: u32,
    layout_hash: u64,
    hcas: *const core::ffi::c_char,
    gid_index: i32,
    tclass: i32,
    sq_depth: i32,
    rank: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GbRoceError {
    pub code: u32,
    pub site: u32,
    pub seq: u32,
    pub detail: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GbRoceProxyStat {
    pub descs: u64,
    pub wrs: u64,
    pub completions: u64,
    pub max_sq: u64,
    pub running: u32,
    pub cpu: u32,
}

#[cfg(not(gb_no_cuda))]
mod ffi {
    use super::*;
    extern "C" {
        pub fn gb_roce_available(err: *mut core::ffi::c_char, errlen: i32) -> i32;
        pub fn gb_roce_open(o: *const GbRoceOpen, out: *mut *mut c_void, err: *mut core::ffi::c_char, errlen: i32) -> i32;
        pub fn gb_roce_blob(c: *mut c_void, blob: *mut c_void, len: i32) -> i32;
        pub fn gb_roce_connect(
            c: *mut c_void,
            peer: *const c_void,
            len: i32,
            generation: *mut u64,
            err: *mut core::ffi::c_char,
            errlen: i32,
        ) -> i32;
        pub fn gb_roce_start(c: *mut c_void, cpu: i32, err: *mut core::ffi::c_char, errlen: i32) -> i32;
        pub fn gb_roce_isolate(c: *mut c_void, err: *mut core::ffi::c_char, errlen: i32) -> i32;
        pub fn gb_roce_stop(c: *mut c_void);
        pub fn gb_roce_close(c: *mut c_void);
        pub fn gb_roce_rails(c: *const c_void) -> i32;
        pub fn gb_roce_exchange(
            d: *const GbRoceDev,
            s: *const GbRoceSite,
            src: *const c_void,
            dst: *mut c_void,
            send: u64,
            recv: u64,
            stride: u64,
            mode: i32,
            sms: i32,
            bulk_blocks: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_roce_abi(out: *mut u64, n: i32) -> i32;
        pub fn cudaDeviceGetAttribute(v: *mut i32, attr: i32, dev: i32) -> i32;
        pub fn cudaGetDevice(dev: *mut i32) -> i32;
        pub fn cudaStreamIsCapturing(stream: *mut c_void, status: *mut i32) -> i32;
    }
}

fn text(buf: &[u8]) -> String {
    let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

/// Whether this build and host can run the transport (`Err` says why not: no verbs header at build time, no `libibverbs.so.1`).
pub fn available() -> Result<(), String> {
    #[cfg(not(gb_no_cuda))]
    {
        let mut e = [0u8; 256];
        // SAFETY: e is writable for its length
        let rc = unsafe { ffi::gb_roce_available(e.as_mut_ptr() as *mut _, e.len() as i32) };
        if rc == 0 {
            Ok(())
        } else {
            Err(text(&e))
        }
    }
    #[cfg(gb_no_cuda)]
    Err("no CUDA in this build".into())
}

/// The C side's `sizeof`/`offsetof` of the mirrored records (the layout test compares them with this module's).
pub fn c_abi() -> Vec<u64> {
    #[cfg(not(gb_no_cuda))]
    {
        let mut v = vec![0u64; 32];
        // SAFETY: v holds 32 entries
        let n = unsafe { ffi::gb_roce_abi(v.as_mut_ptr(), v.len() as i32) };
        v.truncate(n.max(0) as usize);
        v
    }
    #[cfg(gb_no_cuda)]
    Vec::new()
}

/// What the transport is configured with. Both ranks must agree on every field but `proxy_cpu` (the fingerprint, which the
/// launcher puts into the boot gate, says which).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoceConfig {
    /// RDMA devices, comma separated, at most two (`REXL3_ROCE_HCA`, else `NCCL_IB_HCA` without `=`/`^` and `:port`).
    pub hcas: String,
    /// GID index (`REXL3_ROCE_GID_INDEX`); `None`: the RoCE v2 IPv4-mapped GID of each device, found in sysfs.
    pub gid_index: Option<i32>,
    /// GRH traffic class (`REXL3_ROCE_TC`, else `NCCL_IB_TC`, else 106: DSCP 26 = PFC priority 3 on the CRS812 fabric, ECN capable).
    pub tclass: i32,
    /// The core the proxy thread is pinned to (`REXL3_ROCE_CPU`, default 19, an X925 core; `none`: not pinned).
    pub proxy_cpu: Option<usize>,
    /// Take the proxy's core out of every other thread's affinity mask once the proxy runs (`REXL3_ROCE_ISOLATE`, default 1; `0`:
    /// leave the engine's threads alone). Per node, not in the fingerprint.
    pub isolate_proxy: bool,
    /// Messages of at least this many bytes go over both rails (`REXL3_ROCE_STRIPE_MIN`, default 32 KiB).
    pub stripe_min: usize,
    /// Messages of at least this many bytes are chunked onto the bulk QPs (`REXL3_ROCE_BULK_MIN`, default 1 MiB: at 512 KiB the
    /// two-piece latency path measured 2-3.5 us faster than the chunked one on N3+N2).
    pub bulk_min: usize,
    /// Bulk chunk bounds (`REXL3_ROCE_CHUNK`, default 2 MiB; `REXL3_ROCE_CHUNK_MIN`, default 128 KiB): about 8 chunks a message.
    pub chunk: usize,
    pub chunk_min: usize,
    /// Blocks of a bulk exchange (`REXL3_ROCE_BULK_BLOCKS`, default 8: half stage, half drain; more blocks slow the NIC's DMA,
    /// measured: 64 MiB 3.04 ms with 8, 3.35-3.50 with 16, N3+N2).
    pub bulk_blocks: usize,
    /// A wait gives up after this long (`REXL3_ROCE_TIMEOUT_MS`, default 30,000).
    pub timeout_ms: u64,
    /// Send queue depth per QP.
    pub sq_depth: i32,
}

impl Default for RoceConfig {
    fn default() -> Self {
        RoceConfig {
            hcas: "rocep1s0f0,roceP2p1s0f0".into(),
            gid_index: None,
            tclass: 106,
            proxy_cpu: Some(19),
            isolate_proxy: true,
            stripe_min: 32 << 10,
            bulk_min: 1 << 20,
            chunk: 2 << 20,
            chunk_min: 128 << 10,
            bulk_blocks: 8,
            timeout_ms: 30_000,
            sq_depth: 1024,
        }
    }
}

impl RoceConfig {
    /// The configuration from a variable lookup (see the field docs); a malformed value is an error naming it.
    pub fn from_vars(get: impl Fn(&str) -> Option<String>) -> Result<RoceConfig, String> {
        let mut c = RoceConfig::default();
        let nonempty = |k: &str| get(k).filter(|v| !v.trim().is_empty());
        if let Some(h) = nonempty("REXL3_ROCE_HCA").or_else(|| nonempty("NCCL_IB_HCA")) {
            let names: Vec<String> = h
                .split(',')
                .map(|w| w.trim().trim_start_matches(['=', '^']).split(':').next().unwrap_or("").to_string())
                .filter(|w| !w.is_empty())
                .collect();
            if names.is_empty() || names.len() > 2 {
                return Err(format!("REXL3_ROCE_HCA / NCCL_IB_HCA = {h:?}: one or two RDMA devices"));
            }
            c.hcas = names.join(",");
        }
        let num = |k: &str| -> Result<Option<u64>, String> {
            match nonempty(k) {
                None => Ok(None),
                Some(v) => v.trim().parse::<u64>().map(Some).map_err(|_| format!("{k} = {v:?} is not a number")),
            }
        };
        if let Some(v) = num("REXL3_ROCE_GID_INDEX")? {
            c.gid_index = Some(v as i32);
        }
        if let Some(v) = num("REXL3_ROCE_TC")?.or(num("NCCL_IB_TC")?) {
            if v > 255 {
                return Err(format!("traffic class {v} is not a byte"));
            }
            c.tclass = v as i32;
        }
        match nonempty("REXL3_ROCE_CPU").as_deref().map(str::trim) {
            None => {}
            Some("none") => c.proxy_cpu = None,
            Some(v) => c.proxy_cpu = Some(v.parse().map_err(|_| format!("REXL3_ROCE_CPU = {v:?}: a core number or none"))?),
        }
        match nonempty("REXL3_ROCE_ISOLATE").as_deref().map(str::trim) {
            None | Some("1") => {}
            Some("0") => c.isolate_proxy = false,
            Some(v) => return Err(format!("REXL3_ROCE_ISOLATE = {v:?}: 0 or 1")),
        }
        if let Some(v) = num("REXL3_ROCE_STRIPE_MIN")? {
            c.stripe_min = v as usize;
        }
        if let Some(v) = num("REXL3_ROCE_BULK_MIN")? {
            c.bulk_min = v as usize;
        }
        if let Some(v) = num("REXL3_ROCE_CHUNK")? {
            c.chunk = v as usize;
        }
        if let Some(v) = num("REXL3_ROCE_CHUNK_MIN")? {
            c.chunk_min = v as usize;
        }
        if let Some(v) = num("REXL3_ROCE_BULK_BLOCKS")? {
            c.bulk_blocks = v as usize;
        }
        if let Some(v) = num("REXL3_ROCE_TIMEOUT_MS")? {
            c.timeout_ms = v;
        }
        c.validate()?;
        Ok(c)
    }

    pub fn from_env() -> Result<RoceConfig, String> {
        Self::from_vars(|k| std::env::var(k).ok())
    }

    pub fn validate(&self) -> Result<(), String> {
        let pow2 = |x: usize| x.is_power_of_two();
        if !pow2(self.chunk) || !pow2(self.chunk_min) || self.chunk_min > self.chunk || self.chunk_min < 4096 {
            return Err(format!("chunk bounds {} / {} must be powers of two, 4 KiB <= min <= max", self.chunk_min, self.chunk));
        }
        if self.stripe_min > u32::MAX as usize || self.bulk_min < self.stripe_min || self.bulk_blocks == 0 || self.timeout_ms == 0
        {
            return Err("stripe_min <= bulk_min, at least one bulk block and a nonzero timeout".into());
        }
        Ok(())
    }

    /// What both ranks must agree on, as one line (the proxy core is per node and not in it).
    pub fn fingerprint(&self) -> String {
        format!(
            "roce:v{VERSION},tc={},stripe={},bulk={},chunk={}-{},blocks={}",
            self.tclass, self.stripe_min, self.bulk_min, self.chunk_min, self.chunk, self.bulk_blocks
        )
    }

    /// The most pieces (flag slots) a message of up to `cap` bytes has (`gb_roce_shape` / `gb_roce_chunk`).
    fn max_pieces(&self, cap: usize) -> u32 {
        (cap.div_ceil(self.chunk)).max(8) as u32
    }

    /// The chunk of a bulk message of `bytes` (`gb_roce_chunk`, mirrored for the tests and the host-side planner).
    pub fn chunk_of(&self, bytes: usize) -> usize {
        let want = bytes.div_ceil(8);
        let mut c = self.chunk_min;
        while c < want && c < self.chunk {
            c <<= 1;
        }
        c
    }

    /// Pieces of one rank's message of `bytes` (`gb_roce_shape`): 1 below stripe_min (or with one rail), 2 below bulk_min, else chunks.
    pub fn pieces(&self, bytes: usize, rails: usize) -> usize {
        if bytes >= self.bulk_min {
            bytes.div_ceil(self.chunk_of(bytes))
        } else if rails > 1 && bytes >= self.stripe_min {
            2
        } else {
            1
        }
    }
}

/// The region's layout: the same on both ranks for the same site list and configuration.
#[derive(Clone, Debug)]
struct Layout {
    bytes: usize,
    ring_off: usize,
    flags_off: usize,
    flag_slots: u32,
    sites: Vec<GbRoceSite>,
    hash: u64,
}

fn fnv(h: &mut u64, v: u64) {
    for i in 0..8 {
        *h ^= (v >> (8 * i)) & 0xff;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

impl Layout {
    fn of(cfg: &RoceConfig, caps: &[usize]) -> Layout {
        let ring_off = HEADER_BYTES;
        let flags_off = ring_off + 64 * RING_ENTRIES as usize;
        let mut slots = 0u32;
        let mut sites = Vec::with_capacity(caps.len());
        for (i, &cap) in caps.iter().enumerate() {
            let cap = cap.max(1).div_ceil(4096) * 4096;
            let mp = cfg.max_pieces(cap);
            sites.push(GbRoceSite {
                half: [[0; 2]; 2],
                cap: cap as u64,
                flag: [slots, slots + mp],
                id: i as u32,
                max_pieces: mp,
            });
            slots += 2 * mp;
        }
        let mut off = (flags_off + 8 * slots as usize).div_ceil(65536) * 65536;
        let mut h = 0xcbf29ce484222325u64;
        for v in [
            VERSION as u64,
            RING_ENTRIES as u64,
            cfg.stripe_min as u64,
            cfg.bulk_min as u64,
            cfg.chunk as u64,
            cfg.chunk_min as u64,
        ] {
            fnv(&mut h, v);
        }
        for s in &mut sites {
            for p in 0..2 {
                for r in 0..2 {
                    s.half[p][r] = off as u64;
                    off += s.cap as usize;
                }
            }
            fnv(&mut h, s.cap);
        }
        Layout { bytes: off.div_ceil(65536) * 65536, ring_off, flags_off, flag_slots: slots, sites, hash: h }
    }
}

/// One rank's end of the transport. Not `Clone`; share it behind an `Arc`.
pub struct RoceComm {
    rank: usize,
    cfg: RoceConfig,
    layout: Layout,
    /// The registered region (host memory; the shim registered its host address, kernels use its device alias).
    region: Arc<Block>,
    /// Device control words: ticket, per-site words, per-flag-slot arrive counters and ready words (kept alive here; kernels
    /// reach them through `dev.ctl`).
    _ctl: DevBuf,
    dev: GbRoceDev,
    ctx: *mut c_void,
    sms: i32,
    connected: bool,
    started: AtomicBool,
    aborted: AtomicBool,
    isolated: std::sync::atomic::AtomicUsize,
    /// Per site: the exchanges this host queued there (lane T, zero-copy callers: the next exchange's sequence is this + 1, so its
    /// parity is known without the device), and whether one was queued into a stream capture (a graph replays exchanges the host
    /// does not see: the count is then not the device's, and [`RoceComm::next_halves`] refuses the site).
    issued: Vec<AtomicU64>,
    captured: Vec<AtomicBool>,
}

// The shim context is used only through calls the shim makes thread-safe (the proxy thread is the only one that posts); the region
// and the control words are addresses.
unsafe impl Send for RoceComm {}
unsafe impl Sync for RoceComm {}

impl RoceComm {
    /// Rank `rank` (0 or 1) of a pair, with one site per entry of `caps` (bytes per rank of the largest message each site carries).
    /// Allocates and zeroes the region (`cudaHostAlloc`, ~0.6 ms per MiB) and opens the rails; call it before any graph capture.
    pub fn new(rank: usize, cfg: &RoceConfig, caps: &[usize]) -> Result<RoceComm, String> {
        if rank > 1 {
            return Err(format!("rank {rank}: the transport has two ranks"));
        }
        cfg.validate()?;
        if caps.is_empty() {
            return Err("no sites".into());
        }
        available()?;
        let layout = Layout::of(cfg, caps);
        let region =
            Arc::new(Block::alloc_mapped_host(layout.bytes).map_err(|e| format!("cudaHostAlloc({}): {e}", layout.bytes))?);
        let host = region.host_base().ok_or("the region has no host address")?;
        // SAFETY: the region is `layout.bytes` of host memory just allocated
        unsafe { core::ptr::write_bytes(host as *mut u8, 0, layout.bytes) };
        let ctl_words = C_SITE0 + C_PER_SITE * layout.sites.len() + 2 * layout.flag_slots as usize;
        let ctl = DevBuf::zeroed(ctl_words * 4).map_err(|e| format!("control words: {e}"))?;
        #[cfg_attr(gb_no_cuda, allow(unused_mut))]
        let mut sms = 48;
        #[cfg(not(gb_no_cuda))]
        // SAFETY: plain attribute queries into locals
        unsafe {
            let mut d = 0;
            ffi::cudaGetDevice(&mut d);
            ffi::cudaDeviceGetAttribute(&mut sms, 16 /* cudaDevAttrMultiProcessorCount */, d);
        }
        let hcas = std::ffi::CString::new(cfg.hcas.clone()).map_err(|_| "HCA list holds a NUL")?;
        let open = GbRoceOpen {
            region: host,
            region_bytes: layout.bytes as u64,
            ring_off: layout.ring_off as u64,
            flags_off: layout.flags_off as u64,
            ring_entries: RING_ENTRIES,
            flag_slots: layout.flag_slots,
            layout_hash: layout.hash,
            hcas: hcas.as_ptr(),
            gid_index: cfg.gid_index.unwrap_or(-1),
            tclass: cfg.tclass,
            sq_depth: cfg.sq_depth,
            rank: rank as i32,
        };
        #[cfg_attr(gb_no_cuda, allow(unused_mut))]
        let mut ctx: *mut c_void = core::ptr::null_mut();
        #[cfg(not(gb_no_cuda))]
        {
            let mut e = [0u8; 512];
            // SAFETY: `open` and its strings outlive the call; ctx and e are writable
            let rc = unsafe { ffi::gb_roce_open(&open, &mut ctx, e.as_mut_ptr() as *mut _, e.len() as i32) };
            if rc != 0 {
                return Err(format!("rank {rank}: {}", text(&e)));
            }
        }
        #[cfg(gb_no_cuda)]
        let _ = &open;
        #[cfg(not(gb_no_cuda))]
        // SAFETY: ctx is the live context just opened
        let rails = unsafe { ffi::gb_roce_rails(ctx) } as u32;
        #[cfg(gb_no_cuda)]
        let rails = 0u32;
        let dev = GbRoceDev {
            region: region.ptr_at(0) as usize,
            ctl: ctl.ptr() as usize,
            gen_hi: 0,
            timeout_ns: cfg.timeout_ms * 1_000_000,
            ring_off: layout.ring_off as u64,
            flags_off: layout.flags_off as u64,
            bulk_min: cfg.bulk_min as u64,
            chunk: cfg.chunk as u64,
            chunk_min: cfg.chunk_min as u64,
            ring_mask: RING_ENTRIES - 1,
            rails,
            rank: rank as u32,
            stripe_min: cfg.stripe_min as u32,
            ctl_chunks: (C_SITE0 + C_PER_SITE * layout.sites.len()) as u32,
            ctl_ready: (C_SITE0 + C_PER_SITE * layout.sites.len()) as u32 + layout.flag_slots,
        };
        let nsites = layout.sites.len();
        Ok(RoceComm {
            rank,
            cfg: cfg.clone(),
            layout,
            region,
            _ctl: ctl,
            dev,
            ctx,
            sms,
            connected: false,
            started: AtomicBool::new(false),
            aborted: AtomicBool::new(false),
            isolated: std::sync::atomic::AtomicUsize::new(0),
            issued: (0..nsites).map(|_| AtomicU64::new(0)).collect(),
            captured: (0..nsites).map(|_| AtomicBool::new(false)).collect(),
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn rails(&self) -> usize {
        self.dev.rails as usize
    }

    pub fn config(&self) -> &RoceConfig {
        &self.cfg
    }

    /// Bytes of the registered region.
    pub fn region_bytes(&self) -> usize {
        self.layout.bytes
    }

    /// This rank's half of what the peer needs to connect (send it over the side channel).
    pub fn blob(&self) -> Vec<u8> {
        #[cfg_attr(gb_no_cuda, allow(unused_mut))]
        let mut b = vec![0u8; BLOB_BYTES];
        #[cfg(not(gb_no_cuda))]
        // SAFETY: b holds BLOB_BYTES; the context is live
        unsafe {
            ffi::gb_roce_blob(self.ctx, b.as_mut_ptr() as *mut c_void, b.len() as i32);
        }
        b
    }

    /// Connect the queue pairs to the peer's blob (its region, keys, QPs; its layout must equal this one). Both ranks must have
    /// connected before either [`start`](Self::start)s (a barrier over the side channel between the two).
    pub fn connect(&mut self, peer: &[u8]) -> Result<(), String> {
        if self.connected {
            return Err("already connected".into());
        }
        #[cfg(not(gb_no_cuda))]
        {
            let mut e = [0u8; 512];
            let mut gen = 0u64;
            // SAFETY: peer is readable for its length; gen and e are writable
            let rc = unsafe {
                ffi::gb_roce_connect(
                    self.ctx,
                    peer.as_ptr() as *const c_void,
                    peer.len() as i32,
                    &mut gen,
                    e.as_mut_ptr() as *mut _,
                    e.len() as i32,
                )
            };
            if rc != 0 {
                return Err(format!("rank {}: {}", self.rank, text(&e)));
            }
            self.dev.gen_hi = gen << 32;
            self.connected = true;
            Ok(())
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = peer;
            Err("no CUDA in this build".into())
        }
    }

    /// Start the proxy thread (pinned to `config().proxy_cpu`). After this rank's and the peer's [`connect`](Self::connect).
    pub fn start(&self) -> Result<(), String> {
        if !self.connected {
            return Err("start before connect".into());
        }
        if self.started.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        #[cfg(not(gb_no_cuda))]
        {
            let mut e = [0u8; 256];
            let cpu = self.cfg.proxy_cpu.map_or(-1, |c| c as i32);
            // SAFETY: the context is live; e is writable
            let rc = unsafe { ffi::gb_roce_start(self.ctx, cpu, e.as_mut_ptr() as *mut _, e.len() as i32) };
            if rc != 0 {
                return Err(text(&e));
            }
            if self.cfg.isolate_proxy && cpu >= 0 {
                // SAFETY: as above
                let n = unsafe { ffi::gb_roce_isolate(self.ctx, e.as_mut_ptr() as *mut _, e.len() as i32) };
                if n < 0 {
                    return Err(text(&e));
                }
                self.isolated.store(n as usize, Ordering::Release);
            }
        }
        Ok(())
    }

    /// How many other threads had the proxy's core taken out of their affinity mask at [`start`](Self::start).
    pub fn isolated_threads(&self) -> usize {
        self.isolated.load(Ordering::Acquire)
    }

    /// The device context fused kernels take by value (`GbRoceDev` of `roce_dev.cuh`). Valid after [`connect`](Self::connect).
    pub fn dev(&self) -> GbRoceDev {
        self.dev
    }

    /// Site `i` as fused kernels take it (`GbRoceSite`).
    pub fn site(&self, i: usize) -> Option<GbRoceSite> {
        self.layout.sites.get(i).copied()
    }

    pub fn sites(&self) -> usize {
        self.layout.sites.len()
    }

    /// Zero-copy (lane T, the GLM P2 part swap): site `i`'s two halves `[rank 0's, rank 1's]` of the NEXT exchange queued there, so
    /// that a producer writes this rank's message straight into its own half, [`exchange`](Self::exchange) with no source publishes
    /// it, and a consumer reads both halves in place after the exchange (`Mode::None`). The parity comes from the host's count of the
    /// site's exchanges, so it is `None` for a site any exchange was queued at during a stream capture (a replay advances the device
    /// sequence without the host). The caller queues nothing else at the site between this and its exchange, and reads the halves
    /// before its next exchange at the site of the same parity (the transport's double buffering: two exchanges later).
    pub fn next_halves(&self, i: usize) -> Option<[DevBuf; 2]> {
        if self.captured.get(i)?.load(Ordering::Acquire) {
            return None;
        }
        let parity = ((self.issued[i].load(Ordering::Acquire) + 1) & 1) as usize;
        Some([self.half(i, parity, 0)?, self.half(i, parity, 1)?])
    }

    /// A view of site `i`'s half of `rank` for `parity` (for a test or an eager caller that tracks the parity itself: the parity of
    /// an exchange is its sequence & 1, and sequences start at 1 and advance by one per exchange of the site).
    pub fn half(&self, i: usize, parity: usize, rank: usize) -> Option<DevBuf> {
        let s = self.layout.sites.get(i)?;
        (parity < 2 && rank < 2).then(|| DevBuf::view(&self.region, s.half[parity][rank] as usize, s.cap as usize))
    }

    fn header<T: Copy>(&self, off: usize) -> T {
        let host = self.region.host_base().expect("host region");
        // SAFETY: the header page is inside the region; a racy read of words the GPU or the proxy writes is what the poll wants
        unsafe { core::ptr::read_volatile((host as *const u8).add(off) as *const T) }
    }

    /// The first error the GPU or the proxy recorded, as text, or `None` while healthy. Never blocks (a host read of the header).
    pub fn async_error(&self) -> Option<String> {
        let e: GbRoceError = self.header(H_ERROR);
        let what = match e.code {
            0 => return None,
            1 => "a wait timed out (the peer did not deliver)",
            2 => "aborted",
            3 => "a send completed with an error (the peer or the link is gone)",
            4 => "ibv_post_send failed",
            5 => "the proxy refused a descriptor",
            6 => "the send queue did not drain",
            _ => "unknown",
        };
        Some(format!("roce rank {}: {what} (code {}, site {}, seq {}, detail {})", self.rank, e.code, e.site, e.seq, e.detail))
    }

    /// The proxy's counters.
    pub fn proxy_stats(&self) -> GbRoceProxyStat {
        self.header(H_PROXY)
    }

    /// Give up: raise the abort word (every spinning kernel of this rank returns within 256 polls, skipping its result) and stop
    /// the proxy. From any thread; idempotent. The transport is unusable afterwards (make a new one to restart).
    pub fn abort(&self) {
        if self.aborted.swap(true, Ordering::AcqRel) {
            return;
        }
        let host = self.region.host_base().expect("host region");
        // SAFETY: the abort word is inside the header page
        unsafe {
            let w = (host as *mut u8).add(H_ABORT) as *mut u32;
            core::sync::atomic::AtomicU32::from_ptr(w).store(1, Ordering::Release);
        }
    }

    pub fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::Acquire) || self.header::<u32>(H_ABORT) != 0
    }

    /// The exchange at site `site`, queued on `st`: this rank sends `send` bytes, the peer `recv` (equal for a gather and a sum);
    /// `src` is `(buffer, byte offset)` of this rank's bytes, `None` when the caller already wrote them into this rank's half of the
    /// site for the exchange's parity (zero-copy); `dst` is where `mode` puts the result (`None` with [`Mode::None`]). A gather
    /// writes rank r's bytes at `dst + r * stride`. Every rank issues the same exchanges of a site in the same order.
    #[allow(clippy::too_many_arguments)]
    pub fn exchange(
        &self,
        site: usize,
        src: Option<(&DevBuf, usize)>,
        dst: Option<(&mut DevBuf, usize)>,
        send: usize,
        recv: usize,
        stride: usize,
        mode: Mode,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let s = self.layout.sites.get(site).ok_or(CudaError(ERR_BAD_ARGUMENT))?;
        if send == 0
            || recv == 0
            || send as u64 > s.cap
            || recv as u64 > s.cap
            || (send >= self.cfg.bulk_min) != (recv >= self.cfg.bulk_min)
        {
            eprintln!("roce: exchange of {send} / {recv} bytes at site {site} (cap {}) refused", s.cap);
            return Err(CudaError(ERR_BAD_ARGUMENT));
        }
        if !self.started.load(Ordering::Acquire) || self.is_aborted() {
            return Err(CudaError(ERR_ABORTED));
        }
        let srcp = match src {
            Some((b, off)) => {
                if off.checked_add(send).is_none_or(|e| e > b.bytes()) {
                    return Err(CudaError(ERR_BAD_ARGUMENT));
                }
                b.ptr_at(off) as *const c_void
            }
            None => core::ptr::null(),
        };
        let out_bytes = match mode {
            Mode::Gather => stride.saturating_add(send.max(recv)),
            Mode::Swap => recv,
            Mode::SumF32 => send,
            Mode::None => 0,
        };
        let dstp = match dst {
            Some((b, off)) => {
                if mode == Mode::None || off.checked_add(out_bytes).is_none_or(|e| e > b.bytes()) {
                    return Err(CudaError(ERR_BAD_ARGUMENT));
                }
                b.ptr_at(off)
            }
            None => {
                if mode != Mode::None {
                    return Err(CudaError(ERR_BAD_ARGUMENT));
                }
                core::ptr::null_mut()
            }
        };
        if mode == Mode::SumF32 && (send % 4 != 0 || send != recv) {
            return Err(CudaError(ERR_BAD_ARGUMENT));
        }
        if mode == Mode::Gather && (send != recv || stride < send) {
            return Err(CudaError(ERR_BAD_ARGUMENT));
        }
        #[cfg(not(gb_no_cuda))]
        {
            // the host's view of the site's sequence (next_halves): a captured exchange makes it unknowable
            let mut cap = 0i32;
            // SAFETY: a status query of a live stream into a local
            if unsafe { ffi::cudaStreamIsCapturing(st.raw(), &mut cap) } != 0 || cap != 0 {
                self.captured[site].store(true, Ordering::Release);
            }
            // SAFETY: every pointer was range-checked against its buffer above; the kernel is queued on `st`
            let rc = unsafe {
                ffi::gb_roce_exchange(
                    &self.dev,
                    s,
                    srcp,
                    dstp,
                    send as u64,
                    recv as u64,
                    stride as u64,
                    mode as i32,
                    self.sms,
                    self.cfg.bulk_blocks as i32,
                    st.raw(),
                )
            };
            CudaError::check(rc)?;
            self.issued[site].fetch_add(1, Ordering::AcqRel);
            Ok(())
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (srcp, dstp, st);
            Err(CudaError(-1))
        }
    }

    /// `ncclAllGather` of `bytes` a rank: `recv[recv_off..] = [rank 0's | rank 1's]`, this rank's from `send[send_off..]`.
    #[allow(clippy::too_many_arguments)]
    pub fn allgather(
        &self,
        site: usize,
        send: &DevBuf,
        send_off: usize,
        recv: &mut DevBuf,
        recv_off: usize,
        bytes: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.exchange(site, Some((send, send_off)), Some((recv, recv_off)), bytes, bytes, bytes, Mode::Gather, st)
    }

    /// `TpComm::swap_bytes`: this rank's `send_bytes` to the peer, the peer's `recv_bytes` into `recv[recv_off..]`.
    #[allow(clippy::too_many_arguments)]
    pub fn swap(
        &self,
        site: usize,
        send: &DevBuf,
        send_off: usize,
        send_bytes: usize,
        recv: &mut DevBuf,
        recv_off: usize,
        recv_bytes: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.exchange(site, Some((send, send_off)), Some((recv, recv_off)), send_bytes, recv_bytes, 0, Mode::Swap, st)
    }

    /// `TpComm::allgather_sum_f32` / `allreduce_sum_f32` with two ranks: `buf[off..off + 4 count] = rank0 + rank1`, in place, in
    /// rank order, without flush-to-zero.
    pub fn sum_f32(&self, site: usize, buf: &mut DevBuf, off: usize, count: usize, st: &Stream) -> Result<(), CudaError> {
        let bytes = count * 4;
        // in place: the source is the destination (each element's operands are read before its sum is written, by the same thread)
        let alias = buf.alias();
        self.exchange(site, Some((&*alias, off)), Some((buf, off)), bytes, bytes, 0, Mode::SumF32, st)
    }

    /// The site with the smallest cap that holds `bytes` (sites are matched by cap, so callers can keep one site per size class).
    pub fn site_for(&self, bytes: usize) -> Option<usize> {
        self.layout.sites.iter().filter(|s| s.cap as usize >= bytes).min_by_key(|s| s.cap).map(|s| s.id as usize)
    }

    /// Each site's exchanges completed so far, read from the device (lane T, the path counters: graph replays included, which the
    /// host does not see). A device-to-host copy of the control words: call it at a quiet point (it does not synchronise any stream).
    pub fn site_sequences(&self) -> Result<Vec<u64>, CudaError> {
        let words: Vec<u32> = self._ctl.to_vec()?;
        Ok((0..self.layout.sites.len()).map(|i| words.get(C_SITE0 + C_PER_SITE * i).copied().unwrap_or(0) as u64).collect())
    }

    /// Each site's cap (bytes a rank).
    pub fn site_caps(&self) -> Vec<usize> {
        self.layout.sites.iter().map(|s| s.cap as usize).collect()
    }

    /// The site with the smallest cap that holds `bytes` among `ids` (lane T: one user's sites, so that two users of one size class
    /// never share a site, its order or its zero-copy parity). Ids that are not sites are ignored.
    pub fn site_for_among(&self, ids: &[usize], bytes: usize) -> Option<usize> {
        ids.iter()
            .filter_map(|&i| self.layout.sites.get(i))
            .filter(|s| s.cap as usize >= bytes)
            .min_by_key(|s| s.cap)
            .map(|s| s.id as usize)
    }
}

impl Drop for RoceComm {
    fn drop(&mut self) {
        #[cfg(not(gb_no_cuda))]
        // SAFETY: the context is live until here; stop joins the proxy before close frees the QPs; the region outlives both
        unsafe {
            if !self.ctx.is_null() {
                ffi::gb_roce_stop(self.ctx);
                ffi::gb_roce_close(self.ctx);
                self.ctx = core::ptr::null_mut();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_records_match_the_c_layout() {
        let c = c_abi();
        if c.is_empty() {
            eprintln!("not run: no CUDA in this build");
            return;
        }
        assert_eq!(c[0] as usize, core::mem::size_of::<GbRoceDev>(), "GbRoceDev");
        assert_eq!(c[1] as usize, core::mem::size_of::<GbRoceSite>(), "GbRoceSite");
        assert_eq!(c[2] as usize, core::mem::size_of::<GbRoceOpen>(), "GbRoceOpen");
        assert_eq!(c[3] as usize, core::mem::size_of::<GbRoceError>(), "GbRoceError");
        assert_eq!(c[4] as usize, core::mem::size_of::<GbRoceProxyStat>(), "GbRoceProxyStat");
        assert_eq!(c[5] as usize, BLOB_BYTES, "GbRoceBlob");
        assert_eq!(c[6] as usize, core::mem::offset_of!(GbRoceDev, ctl_ready));
        assert_eq!(c[7] as usize, core::mem::offset_of!(GbRoceSite, max_pieces));
        assert_eq!(c[8] as usize, core::mem::offset_of!(GbRoceOpen, rank));
        assert_eq!([c[9], c[10], c[12]], [H_ABORT as u64, H_ERROR as u64, H_PROXY as u64]);
        assert_eq!([c[13], c[14]], [C_SITE0 as u64, C_PER_SITE as u64]);
    }

    #[test]
    fn the_config_reads_nccl_hcas_and_rejects_nonsense() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| pairs.iter().find(|p| p.0 == k).map(|p| p.1.to_string())
        };
        let c = RoceConfig::from_vars(env(&[("NCCL_IB_HCA", "=rocep1s0f0:1,roceP2p1s0f0:1"), ("NCCL_IB_TC", "106")])).unwrap();
        assert_eq!(c.hcas, "rocep1s0f0,roceP2p1s0f0");
        assert_eq!(c.tclass, 106);
        let c = RoceConfig::from_vars(env(&[("REXL3_ROCE_CPU", "none"), ("REXL3_ROCE_HCA", "a")])).unwrap();
        assert_eq!((c.proxy_cpu, c.hcas.as_str()), (None, "a"));
        assert!(RoceConfig::from_vars(env(&[("REXL3_ROCE_CHUNK", "3000000")])).is_err());
        assert!(RoceConfig::from_vars(env(&[("REXL3_ROCE_TC", "300")])).is_err());
        assert!(RoceConfig::from_vars(env(&[("REXL3_ROCE_HCA", "a,b,c")])).is_err());
        assert!(RoceConfig::default().fingerprint().starts_with("roce:v1,tc=106,"));
        // the proxy core's isolation is per node: on by default, never in the fingerprint
        assert!(RoceConfig::default().isolate_proxy);
        let off = RoceConfig::from_vars(env(&[("REXL3_ROCE_ISOLATE", "0")])).unwrap();
        assert!(!off.isolate_proxy && off.fingerprint() == RoceConfig::default().fingerprint());
        assert!(RoceConfig::from_vars(env(&[("REXL3_ROCE_ISOLATE", "maybe")])).is_err());
    }

    #[test]
    fn the_shape_has_one_two_or_about_eight_pieces() {
        let c = RoceConfig::default();
        assert_eq!(c.pieces(4096, 2), 1);
        assert_eq!(c.pieces(4096, 1), 1);
        assert_eq!(c.pieces(64 << 10, 2), 2);
        assert_eq!(c.pieces(64 << 10, 1), 1);
        assert_eq!(c.chunk_of(1 << 20), 128 << 10);
        assert_eq!(c.pieces(1 << 20, 2), 8);
        assert_eq!(c.pieces(64 << 20, 2), 32);
        assert_eq!(c.pieces((64 << 20) + 1, 2), 33);
        for bytes in [512 << 10, 777_777, 3 << 20, 16 << 20, 64 << 20] {
            assert!(c.pieces(bytes, 2) as u32 <= c.max_pieces(bytes), "{bytes}");
        }
    }

    #[test]
    fn both_ranks_build_the_same_layout() {
        let c = RoceConfig::default();
        let a = Layout::of(&c, &[512 << 10, 64 << 20]);
        let b = Layout::of(&c, &[512 << 10, 64 << 20]);
        assert_eq!((a.hash, a.bytes), (b.hash, b.bytes));
        assert_eq!(a.sites, b.sites);
        let s = &a.sites[1];
        assert!(s.half[0][1] >= s.half[0][0] + s.cap && s.half[1][0] >= s.half[0][1] + s.cap);
        assert_ne!(Layout::of(&c, &[512 << 10]).hash, a.hash);
        let mut d = c.clone();
        d.stripe_min = 16 << 10;
        assert_ne!(Layout::of(&d, &[512 << 10, 64 << 20]).hash, a.hash);
    }
}
