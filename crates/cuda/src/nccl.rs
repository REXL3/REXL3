//! NCCL for the two-rank tensor-parallel engine: a `dlopen` binding of exactly the calls it needs, so the crate has no link-time NCCL
//! dependency (the measured runs used a native 2.30.7 build with no headers installed) and the stub build (`REXL3_NO_CUDA=1`) keeps compiling: in a stub build every function returns an error.
//!
//! The collective is one: an f32 sum-allreduce over a device buffer, in place, queued on a stream, so it can be captured into a CUDA graph
//! and replayed (`crates/cuda/tests/nccl_pair.rs` proves the capture across two nodes). With two ranks the result is `a + b`, which is
//! commutative, so both ranks hold the same bits. See `docs/NEMOTRON_TP.md` sections 4 and 10.
//!
//! It has two forms. The ring [`NcclComm::allreduce_sum_f32`] costs 2(R-1) network hops. The one-shot [`NcclComm::oneshot_sum_f32`]
//! sends this rank's range to every peer and receives theirs into staging slots (`ncclGroupStart` / `ncclSend` + `ncclRecv` per peer /
//! `ncclGroupEnd`: one hop whatever R is), then ONE kernel ([`sum_slots_f32`]) adds the R operands in fixed rank order. Over buffers of
//! disjoint support (exact `-0.0` outside a rank's own elements: `x + -0.0 == x` for every x) that is exact in any order, so it is the
//! same bits as the ring; the bandwidth is (R-1) x S a rank, so the ring wins above a few hundred KB at large R (`docs/NEMOTRON_TP.md`).
//!
//! The library is `$REXL3_NCCL_LIB` when set and not empty (a path or a soname given to `dlopen`), else `libnccl.so.2` found by the
//! dynamic loader's normal search (`LD_LIBRARY_PATH`, the run path, the `ldconfig` cache, the default directories): see
//! [`resolve_library`]. A node whose NCCL build is outside the loader's path exports `REXL3_NCCL_LIB`. NCCL reads its own `NCCL_*`
//! environment (the fabric settings); this binding sets none of them.
//!
//! Errors are [`CudaError`] like the rest of the crate: an NCCL result `r` is code `NCCL_ERROR_BASE + r`, a missing library or symbol is
//! [`ERR_NO_LIBRARY`], a bad argument [`ERR_BAD_ARGUMENT`]. None of them is in `CudaError::is_fatal`'s retryable list, so all fail closed.
//! The message goes to stderr when it happens (the code alone does not carry NCCL's text).

use crate::rt::{CudaError, DevBuf, Stream};
use core::ffi::c_void;
use std::sync::Mutex;

/// `CudaError` code of NCCL result `r` (`ncclResult_t`: 1 unhandled CUDA error, 2 system error, 3 internal error, 4 invalid argument,
/// 5 invalid usage, 6 remote error, 7 in progress) is `NCCL_ERROR_BASE + r`.
pub const NCCL_ERROR_BASE: i32 = 20_000;
/// `libnccl` could not be loaded, or lacks a symbol this binding needs.
pub const ERR_NO_LIBRARY: i32 = NCCL_ERROR_BASE + 100;
/// A rank, a count or an offset this binding refuses before calling NCCL.
pub const ERR_BAD_ARGUMENT: i32 = NCCL_ERROR_BASE + 101;
/// The loopback group's peer failed, or did not arrive in time (`tp_comm.rs`).
pub const ERR_PEER: i32 = NCCL_ERROR_BASE + 102;
/// The communicator was aborted ([`NcclComm::abort`]): every later collective fails with this.
pub const ERR_ABORTED: i32 = NCCL_ERROR_BASE + 103;
/// The most ranks one sum-kernel launch (and so one one-shot collective) takes the operands of: the pointers travel in its parameters.
pub const MAX_ONESHOT_RANKS: usize = 16;

/// The name of a code this module produces, `None` for any other.
pub fn error_name(code: i32) -> Option<&'static str> {
    Some(match code - NCCL_ERROR_BASE {
        1 => "ncclUnhandledCudaError",
        2 => "ncclSystemError",
        3 => "ncclInternalError",
        4 => "ncclInvalidArgument",
        5 => "ncclInvalidUsage",
        6 => "ncclRemoteError",
        7 => "ncclInProgress",
        100 => "libnccl not loadable",
        101 => "bad argument to a collective",
        102 => "loopback peer failed or timed out",
        103 => "communicator aborted",
        _ => return None,
    })
}

/// A code that says "a collective could not finish", which is what a lost peer looks like from the rank that stayed: an NCCL result
/// (1 unhandled CUDA error, 2 system error, 3 internal error, 4 invalid argument, 5 invalid usage, 6 remote error, 7 in progress),
/// the loopback group's failed peer ([`ERR_PEER`]) or an aborted communicator ([`ERR_ABORTED`], what every collective returns after the
/// group's abort hook ran). A missing library and an argument this binding refused before launching anything are not.
pub fn is_collective_failure(code: i32) -> bool {
    (NCCL_ERROR_BASE + 1..=NCCL_ERROR_BASE + 7).contains(&code) || code == ERR_PEER || code == ERR_ABORTED
}

/// What a poll of `ncclCommGetAsyncError` means: the call's own result `rc` and the async result `r` it wrote. `ncclSuccess` (0) and
/// `ncclInProgress` (7, a non-blocking operation still running) are healthy; any other async result is the error, as a [`CudaError`]
/// code. A call that itself failed says nothing about the communicator (`None`).
pub fn async_error_code(rc: i32, r: i32) -> Option<i32> {
    (rc == 0 && r != 0 && r != 7).then_some(NCCL_ERROR_BASE + r)
}

/// `ncclUniqueId`: 128 opaque bytes that rank 0 creates and every other rank receives out of band (the TCP side channel of the lockstep
/// runtime). It carries the address of rank 0's bootstrap listener, so it is valid for one `init` round of one group.
#[derive(Clone, PartialEq, Eq)]
pub struct NcclUniqueId(pub [u8; 128]);

impl NcclUniqueId {
    pub const BYTES: usize = 128;

    pub fn as_bytes(&self) -> &[u8; 128] {
        &self.0
    }

    /// The id from exactly 128 bytes (`None` for any other length).
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        <[u8; 128]>::try_from(b).ok().map(Self)
    }
}

impl core::fmt::Debug for NcclUniqueId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "NcclUniqueId({:02x}{:02x}..)", self.0[0], self.0[1])
    }
}

/// The variable that names the NCCL library to `dlopen` (a path or a soname).
pub const LIBRARY_ENV: &str = "REXL3_NCCL_LIB";
/// The soname `dlopen` resolves through the dynamic loader's normal search when [`LIBRARY_ENV`] is unset or empty.
pub const DEFAULT_LIBRARY: &str = "libnccl.so.2";

/// The library name handed to `dlopen`, in this order: the value of [`LIBRARY_ENV`] when it is set and not empty, else
/// [`DEFAULT_LIBRARY`], a bare soname, so the dynamic loader searches its usual places. No directory is guessed.
pub fn resolve_library(env_value: Option<&str>) -> String {
    match env_value {
        Some(p) if !p.is_empty() => p.to_string(),
        _ => DEFAULT_LIBRARY.to_string(),
    }
}

#[cfg(not(gb_no_cuda))]
mod dl {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::sync::OnceLock;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct RawId {
        pub internal: [u8; 128],
    }

    /// `ncclFloat32` and `ncclSum` of `nccl.h`.
    pub const NCCL_FLOAT32: i32 = 7;
    /// `ncclUint8` of `nccl.h`.
    pub const NCCL_UINT8: i32 = 1;
    pub const NCCL_SUM: i32 = 0;

    pub struct Api {
        pub path: String,
        pub get_unique_id: unsafe extern "C" fn(*mut RawId) -> i32,
        pub comm_init_rank: unsafe extern "C" fn(*mut *mut c_void, i32, RawId, i32) -> i32,
        pub all_reduce: unsafe extern "C" fn(*const c_void, *mut c_void, usize, i32, i32, *mut c_void, *mut c_void) -> i32,
        pub group_start: unsafe extern "C" fn() -> i32,
        pub group_end: unsafe extern "C" fn() -> i32,
        /// `ncclSend(sendbuff, count, datatype, peer, comm, stream)`
        pub send: unsafe extern "C" fn(*const c_void, usize, i32, i32, *mut c_void, *mut c_void) -> i32,
        /// `ncclRecv(recvbuff, count, datatype, peer, comm, stream)`
        pub recv: unsafe extern "C" fn(*mut c_void, usize, i32, i32, *mut c_void, *mut c_void) -> i32,
        pub comm_destroy: unsafe extern "C" fn(*mut c_void) -> i32,
        pub comm_abort: unsafe extern "C" fn(*mut c_void) -> i32,
        pub get_error_string: unsafe extern "C" fn(i32) -> *const core::ffi::c_char,
        pub get_version: unsafe extern "C" fn(*mut i32) -> i32,
        /// `ncclCommGetAsyncError(comm, &result)`: optional (a library without it still loads; the poll then reports nothing).
        pub get_async_error: Option<unsafe extern "C" fn(*mut c_void, *mut i32) -> i32>,
    }

    // function pointers into a library that is never unloaded
    unsafe impl Send for Api {}
    unsafe impl Sync for Api {}

    fn dlerror_text() -> String {
        // SAFETY: dlerror returns null or a NUL-terminated string valid until the next dl call on this thread
        unsafe {
            let e = libc::dlerror();
            if e.is_null() {
                "unknown dlopen error".into()
            } else {
                CStr::from_ptr(e).to_string_lossy().into_owned()
            }
        }
    }

    /// The library to load: `$REXL3_NCCL_LIB`, else the loader's own search ([`resolve_library`]).
    pub fn library_path() -> String {
        resolve_library(std::env::var(LIBRARY_ENV).ok().as_deref())
    }

    unsafe fn sym<T: Copy>(lib: *mut c_void, name: &str) -> Result<T, String> {
        assert_eq!(core::mem::size_of::<T>(), core::mem::size_of::<*mut c_void>());
        let c = CString::new(name).expect("symbol name");
        let p = libc::dlsym(lib, c.as_ptr());
        if p.is_null() {
            return Err(format!("symbol {name} not found: {}", dlerror_text()));
        }
        Ok(core::mem::transmute_copy::<*mut c_void, T>(&p))
    }

    fn load() -> Result<Api, String> {
        let path = library_path();
        let c = CString::new(path.clone()).map_err(|_| format!("library path {path:?} holds a NUL"))?;
        // SAFETY: plain dlopen of a path; the handle is kept for the life of the process
        let lib = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if lib.is_null() {
            return Err(format!("dlopen({path}): {}", dlerror_text()));
        }
        // SAFETY: every symbol is a function of the C ABI with the signature `nccl.h` declares
        unsafe {
            Ok(Api {
                get_unique_id: sym(lib, "ncclGetUniqueId")?,
                comm_init_rank: sym(lib, "ncclCommInitRank")?,
                all_reduce: sym(lib, "ncclAllReduce")?,
                group_start: sym(lib, "ncclGroupStart")?,
                group_end: sym(lib, "ncclGroupEnd")?,
                send: sym(lib, "ncclSend")?,
                recv: sym(lib, "ncclRecv")?,
                comm_destroy: sym(lib, "ncclCommDestroy")?,
                comm_abort: sym(lib, "ncclCommAbort")?,
                get_error_string: sym(lib, "ncclGetErrorString")?,
                get_version: sym(lib, "ncclGetVersion")?,
                get_async_error: sym(lib, "ncclCommGetAsyncError").ok(),
                path,
            })
        }
    }

    /// The file the library was actually loaded from (`dladdr` of one of its symbols: the soname resolved by the loader, symlinks
    /// followed by the loader's own map), `None` when it is not loaded.
    pub fn loaded_file(api: &Api) -> Option<String> {
        // SAFETY: dladdr reads the loader's map for an address inside a loaded object; `info` is valid for a write
        unsafe {
            let mut info: libc::Dl_info = core::mem::zeroed();
            if libc::dladdr(api.get_version as *const c_void, &mut info) == 0 || info.dli_fname.is_null() {
                return None;
            }
            let p = CStr::from_ptr(info.dli_fname).to_string_lossy().into_owned();
            Some(std::fs::canonicalize(&p).map(|c| c.to_string_lossy().into_owned()).unwrap_or(p))
        }
    }

    /// The loaded library (loaded once; a failure is remembered and reported once on stderr).
    pub fn api() -> Result<&'static Api, CudaError> {
        static API: OnceLock<Result<Api, String>> = OnceLock::new();
        match API.get_or_init(|| {
            let r = load();
            if let Err(e) = &r {
                eprintln!("nccl: {e}");
            }
            r
        }) {
            Ok(a) => Ok(a),
            Err(_) => Err(CudaError(ERR_NO_LIBRARY)),
        }
    }

    /// `rc` of an NCCL call as a `CudaError`, with NCCL's own text on stderr.
    pub fn check(api: &Api, what: &str, rc: i32) -> Result<(), CudaError> {
        if rc == 0 {
            return Ok(());
        }
        // SAFETY: ncclGetErrorString returns a static string
        let text = unsafe {
            let p = (api.get_error_string)(rc);
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        eprintln!("nccl: {what} failed: result {rc} ({text})");
        Err(CudaError(NCCL_ERROR_BASE + rc))
    }
}

/// The `NCCL_*` variables in this process's environment as `NAME=value` pairs, sorted: the setting a run was measured under (the fabric
/// environment and the channel / protocol caps change the latency of this collective by up to 10x, see `docs/NEMOTRON_TP.md` section 10),
/// so the launcher logs it next to the communicator.
pub fn nccl_environment() -> Vec<String> {
    let mut v: Vec<String> =
        std::env::vars().filter(|(k, _)| k.starts_with("NCCL_")).map(|(k, val)| format!("{k}={val}")).collect();
    v.sort();
    v
}

/// One rank's end of an NCCL communicator over one GPU (the current CUDA device when it is created). The communicator is not thread safe,
/// so the enqueues of one `NcclComm` are serialised by a lock; they are cheap host calls.
pub struct NcclComm {
    comm: *mut c_void,
    rank: usize,
    ranks: usize,
    gate: Mutex<()>,
    /// Orders [`NcclComm::async_error`] (read) against [`NcclComm::abort`] (write), which frees the handle: a poll never reads a freed
    /// communicator, and an abort waits at most for one poll (a host call that returns at once). Not the enqueue `gate`, which a stuck
    /// collective holds.
    handle: std::sync::RwLock<()>,
    failed: core::sync::atomic::AtomicBool,
    /// Set once by [`NcclComm::abort`]: the handle is gone (aborted), so nothing may touch it again, not even the destructor.
    aborted: core::sync::atomic::AtomicBool,
    /// The staging slots of the one-shot collective ([`reserve_oneshot_staging`](Self::reserve_oneshot_staging)), `None` until
    /// reserved. A field, so it is freed AFTER `Drop::drop` has destroyed or aborted the communicator: no queued receive lands in
    /// freed memory.
    staging: Mutex<Option<DevBuf>>,
}

// The communicator handle is an opaque pointer used only under `gate`.
unsafe impl Send for NcclComm {}
unsafe impl Sync for NcclComm {}

impl NcclComm {
    /// The library this process loads (or loaded) NCCL from.
    pub fn library_path() -> String {
        #[cfg(not(gb_no_cuda))]
        {
            match dl::api() {
                Ok(api) => api.path.clone(),
                Err(_) => dl::library_path(),
            }
        }
        #[cfg(gb_no_cuda)]
        {
            String::from("(no CUDA in this build)")
        }
    }

    /// `NCCL_VERSION_CODE` of the loaded library (`23007` is 2.30.7).
    pub fn version() -> Result<i32, CudaError> {
        #[cfg(not(gb_no_cuda))]
        {
            let api = dl::api()?;
            let mut v = 0i32;
            // SAFETY: `v` is valid for a write
            dl::check(api, "ncclGetVersion", unsafe { (api.get_version)(&mut v) })?;
            Ok(v)
        }
        #[cfg(gb_no_cuda)]
        Err(CudaError(-1))
    }

    /// Rank 0 calls this once per group and hands the id to every other rank (any channel: the engine uses its TCP side channel).
    pub fn new_unique_id() -> Result<NcclUniqueId, CudaError> {
        #[cfg(not(gb_no_cuda))]
        {
            let api = dl::api()?;
            let mut raw = dl::RawId { internal: [0; 128] };
            // SAFETY: `raw` is a 128-byte ncclUniqueId
            dl::check(api, "ncclGetUniqueId", unsafe { (api.get_unique_id)(&mut raw) })?;
            Ok(NcclUniqueId(raw.internal))
        }
        #[cfg(gb_no_cuda)]
        Err(CudaError(-1))
    }

    /// Join the group `id` names as `rank` of `ranks`. Blocks until every rank has called it (each from its own process), so the ranks
    /// must be started together. Do it before the capture of any graph that uses the communicator, and run one eager collective before
    /// capturing (NCCL connects its peers lazily on the first one).
    pub fn init(rank: usize, ranks: usize, id: &NcclUniqueId) -> Result<NcclComm, CudaError> {
        if ranks == 0 || rank >= ranks || ranks > i32::MAX as usize {
            eprintln!("nccl: init: rank {rank} of {ranks} is not a valid position");
            return Err(CudaError(ERR_BAD_ARGUMENT));
        }
        #[cfg(not(gb_no_cuda))]
        {
            let api = dl::api()?;
            let mut comm: *mut c_void = core::ptr::null_mut();
            let raw = dl::RawId { internal: id.0 };
            // SAFETY: `comm` is valid for a write; the id is passed by value as `ncclUniqueId`
            let rc = unsafe { (api.comm_init_rank)(&mut comm, ranks as i32, raw, rank as i32) };
            dl::check(api, "ncclCommInitRank", rc)?;
            if comm.is_null() {
                return Err(CudaError(NCCL_ERROR_BASE + 3));
            }
            Ok(NcclComm {
                comm,
                rank,
                ranks,
                gate: Mutex::new(()),
                handle: std::sync::RwLock::new(()),
                failed: core::sync::atomic::AtomicBool::new(false),
                aborted: core::sync::atomic::AtomicBool::new(false),
                staging: Mutex::new(None),
            })
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = id;
            Err(CudaError(-1))
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    /// `ncclCommAbort`, from any thread, once: the collectives this communicator has in flight (a kernel waiting for a peer that is gone)
    /// return, a stream that was stuck behind them completes with an error, and every later [`allreduce_sum_f32`] fails with
    /// [`ERR_ABORTED`]. It does NOT take the enqueue lock (a collective that is stuck connecting to a dead peer holds it). The handle is
    /// freed by the abort, so this is for a process on its way out: nothing may use the communicator afterwards, and dropping it does not
    /// destroy it again.
    ///
    /// [`allreduce_sum_f32`]: Self::allreduce_sum_f32
    pub fn abort(&self) {
        #[cfg(not(gb_no_cuda))]
        {
            use core::sync::atomic::Ordering;
            // the write side: an in-flight poll finishes first, and no poll starts on a handle being freed
            let _w = self.handle.write().unwrap_or_else(|p| p.into_inner());
            if self.comm.is_null() || self.aborted.swap(true, Ordering::AcqRel) {
                return;
            }
            if let Ok(api) = dl::api() {
                // SAFETY: the handle is live (it is freed only here, and `aborted` makes this run once) and ncclCommAbort is the call
                // NCCL documents for unblocking a communicator from another thread
                let rc = unsafe { (api.comm_abort)(self.comm) };
                eprintln!("nccl: rank {} aborted its communicator (result {rc})", self.rank);
            }
        }
    }

    pub fn ranks(&self) -> usize {
        self.ranks
    }

    /// The file the NCCL library was loaded from (what the boot gate hashes), `None` when it is not loaded or this is the stub build.
    pub fn loaded_library_file() -> Option<String> {
        #[cfg(not(gb_no_cuda))]
        {
            dl::api().ok().and_then(dl::loaded_file)
        }
        #[cfg(gb_no_cuda)]
        None
    }

    /// `ncclCommGetAsyncError`: `Some(code)` (a [`CudaError`] code, `NCCL_ERROR_BASE + result`) when NCCL has recorded an
    /// asynchronous error on this communicator (a network transport that failed under a collective: what a RoCE link that went down
    /// looks like while the TCP command link stays up), `None` when it is healthy, aborted, being aborted, or the library lacks the call.
    /// Never blocks: NCCL documents the call as safe beside a running collective, it does not take the enqueue lock, and a poll that
    /// meets an abort in progress returns `None`.
    pub fn async_error(&self) -> Option<i32> {
        #[cfg(not(gb_no_cuda))]
        {
            use core::sync::atomic::Ordering;
            let _r = self.handle.try_read().ok()?;
            if self.comm.is_null() || self.aborted.load(Ordering::Acquire) {
                return None;
            }
            let api = dl::api().ok()?;
            let f = api.get_async_error?;
            let mut r = 0i32;
            // SAFETY: the handle is live (not aborted, and the abort waits for this read lock); `r` is valid for a write
            let rc = unsafe { f(self.comm, &mut r) };
            async_error_code(rc, r)
        }
        #[cfg(gb_no_cuda)]
        None
    }

    /// Bytes a staging slot takes for a message of `bytes`: rounded up to 256, so every slot is 16-byte aligned (the sum kernel is
    /// vectorised when every operand is).
    pub const fn oneshot_slot_bytes(bytes: usize) -> usize {
        bytes.div_ceil(256) * 256
    }

    /// Staging bytes a one-shot collective of messages up to `max_message_bytes` needs on a group of `ranks`: `ranks - 1` slots.
    pub const fn oneshot_staging_bytes(ranks: usize, max_message_bytes: usize) -> usize {
        ranks.saturating_sub(1) * Self::oneshot_slot_bytes(max_message_bytes)
    }

    /// Allocates the staging buffer for one-shot collectives of up to `max_message_bytes` (`ranks - 1` slots of that size) on the
    /// current CUDA device, once. A second call asking for no more is a no-op; one asking for more is refused (the buffer cannot
    /// grow: a graph that captured it holds its address). Call it before the first collective and before any capture (a `cudaMalloc`
    /// inside a capture is illegal). Returns the bytes held. A failure is an error to surface at boot, not a reason to fall back on
    /// one rank: the choice of algorithm must be the same on every rank.
    pub fn reserve_oneshot_staging(&self, max_message_bytes: usize) -> Result<usize, CudaError> {
        if self.ranks > MAX_ONESHOT_RANKS || max_message_bytes == 0 {
            eprintln!(
                "nccl: one-shot staging for {} ranks and {max_message_bytes} bytes is not supported (1 to {MAX_ONESHOT_RANKS} ranks)",
                self.ranks
            );
            return Err(CudaError(ERR_BAD_ARGUMENT));
        }
        let need = Self::oneshot_staging_bytes(self.ranks, max_message_bytes);
        let mut g = self.staging.lock().unwrap_or_else(|e| e.into_inner());
        match g.as_ref() {
            Some(b) if b.bytes() >= need => Ok(b.bytes()),
            Some(b) => {
                eprintln!("nccl: one-shot staging is {} bytes and cannot grow to {need}", b.bytes());
                Err(CudaError(ERR_BAD_ARGUMENT))
            }
            None => {
                *g = Some(DevBuf::alloc(need.max(256))?);
                Ok(need)
            }
        }
    }

    /// The largest message (bytes) the reserved staging serves with a one-shot collective; 0 when none is reserved or there is one rank.
    pub fn oneshot_capacity_bytes(&self) -> usize {
        if self.ranks < 2 {
            return 0;
        }
        let g = self.staging.lock().unwrap_or_else(|e| e.into_inner());
        g.as_ref().map_or(0, |b| b.bytes() / (self.ranks - 1) / 256 * 256)
    }

    /// The one-shot collective: sum `count` f32 elements of `buf` starting `off_bytes` bytes in, across every rank, in place, queued on
    /// `st`, with ONE network hop whatever the rank count. Every rank sends its range to every peer and receives the peers' ranges
    /// into the staging slots (`ncclGroupStart`, per peer `ncclSend` + `ncclRecv`, `ncclGroupEnd`), then one kernel writes
    /// `s[0] + s[1] + ... + s[R-1]` (rank order; this rank's own operand is `buf` itself) over the range. Over buffers of disjoint
    /// support (exact `-0.0` outside a rank's own elements) the result is bit-identical to
    /// [`allreduce_sum_f32`](Self::allreduce_sum_f32) at any R; with two ranks it is `a + b` for any data. Every rank must issue the
    /// same sequence of collectives with the same `count`s AND the same choice of algorithm for each (a ring on one rank and a
    /// one-shot on another deadlocks the group).
    ///
    /// The staging buffer is shared by every one-shot of this communicator, so they must be queued on ONE stream (the rule NCCL has
    /// for ordering collectives across ranks anyway); the enqueue is serialised by the communicator's lock. Capturable into a CUDA
    /// graph once the peers are connected (run one eager collective of each kind first). Needs
    /// [`reserve_oneshot_staging`](Self::reserve_oneshot_staging) for at least `4 * count` bytes first, else `ERR_BAD_ARGUMENT`
    /// (nothing is launched).
    pub fn oneshot_sum_f32(&self, buf: &mut DevBuf, off_bytes: usize, count: usize, st: &Stream) -> Result<(), CudaError> {
        let bytes = count.checked_mul(4).and_then(|b| b.checked_add(off_bytes));
        if off_bytes % 4 != 0 || bytes.is_none_or(|e| e > buf.bytes()) {
            eprintln!(
                "nccl: one-shot of {count} f32 at byte offset {off_bytes} does not fit a {}-byte buffer (or is not 4-byte aligned)",
                buf.bytes()
            );
            return Err(CudaError(ERR_BAD_ARGUMENT));
        }
        if count == 0 || self.ranks == 1 {
            return Ok(());
        }
        #[cfg(not(gb_no_cuda))]
        {
            use core::sync::atomic::Ordering;
            if self.aborted.load(Ordering::Acquire) {
                return Err(CudaError(ERR_ABORTED));
            }
            let api = dl::api()?;
            let stride = Self::oneshot_slot_bytes(count * 4);
            let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
            let stg = self.staging.lock().unwrap_or_else(|e| e.into_inner());
            let Some(staging) = stg.as_ref() else {
                eprintln!("nccl: a one-shot collective needs reserve_oneshot_staging first");
                return Err(CudaError(ERR_BAD_ARGUMENT));
            };
            if (self.ranks - 1) * stride > staging.bytes() {
                eprintln!(
                    "nccl: a one-shot of {} bytes needs {} bytes of staging, {} are reserved",
                    count * 4,
                    (self.ranks - 1) * stride,
                    staging.bytes()
                );
                return Err(CudaError(ERR_BAD_ARGUMENT));
            }
            let mine = buf.ptr_at(off_bytes);
            let slot_of = |peer: usize| if peer < self.rank { peer } else { peer - 1 };
            let fail = |what: &str, rc: i32| -> CudaError {
                self.failed.store(true, Ordering::Relaxed);
                dl::check(api, what, rc).err().unwrap_or(CudaError(NCCL_ERROR_BASE + 3))
            };
            // SAFETY: `mine` addresses `count` f32 inside `buf` and every slot `count` f32 inside `staging` (both checked above); the
            // group is closed before this block ends, on every path
            unsafe {
                let rc = (api.group_start)();
                if rc != 0 {
                    return Err(fail("ncclGroupStart", rc));
                }
                let mut err: Option<CudaError> = None;
                for peer in 0..self.ranks {
                    if peer == self.rank {
                        continue;
                    }
                    let rc = (api.send)(mine, count, dl::NCCL_FLOAT32, peer as i32, self.comm, st.raw());
                    if rc != 0 {
                        err.get_or_insert_with(|| fail("ncclSend", rc));
                        break;
                    }
                    let dst = staging.ptr_at(slot_of(peer) * stride);
                    let rc = (api.recv)(dst, count, dl::NCCL_FLOAT32, peer as i32, self.comm, st.raw());
                    if rc != 0 {
                        err.get_or_insert_with(|| fail("ncclRecv", rc));
                        break;
                    }
                }
                // always close the group: NCCL's per-thread group state must not be left open by an error
                let rc = (api.group_end)();
                if rc != 0 {
                    err.get_or_insert_with(|| fail("ncclGroupEnd", rc));
                }
                if let Some(e) = err {
                    return Err(e);
                }
            }
            // the sum in rank order: operand r is peer r's slot, and `buf` itself for this rank
            let mut ptrs: [*const c_void; MAX_ONESHOT_RANKS] = [core::ptr::null(); MAX_ONESHOT_RANKS];
            for (r, p) in ptrs.iter_mut().enumerate().take(self.ranks) {
                *p = if r == self.rank { mine as *const c_void } else { staging.ptr_at(slot_of(r) * stride) as *const c_void };
            }
            // SAFETY: every operand is `count` f32 inside its buffer (checked above); `mine` is both an operand and the output, which
            // the kernel supports (each element is read by the one thread that writes it)
            CudaError::check(unsafe {
                add_ffi::gb_tp_sum_slots_f32(ptrs.as_ptr(), self.ranks as i32, mine, count as i64, st.raw())
            })
            .inspect_err(|_| self.failed.store(true, Ordering::Relaxed))
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = st;
            Err(CudaError(-1))
        }
    }

    /// Two ranks only (lane S, contract P2): sends `send_bytes` bytes of `send` (from byte `send_off`) to the peer and receives the
    /// peer's `recv_bytes` bytes into `recv` (from byte `recv_off`), one NCCL group of a send and a receive, queued on `st`. No arithmetic:
    /// a byte copy across the pair. Both ranks must make the same call sequence with matching sizes (this rank's send is the peer's recv).
    #[allow(clippy::too_many_arguments)]
    pub fn sendrecv_bytes(
        &self,
        send: &DevBuf,
        send_off: usize,
        send_bytes: usize,
        recv: &mut DevBuf,
        recv_off: usize,
        recv_bytes: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if self.ranks != 2 || send_off + send_bytes > send.bytes() || recv_off + recv_bytes > recv.bytes() {
            eprintln!("nccl: sendrecv of {send_bytes} / {recv_bytes} bytes does not fit (or the group is not two ranks)");
            return Err(CudaError(ERR_BAD_ARGUMENT));
        }
        #[cfg(not(gb_no_cuda))]
        {
            use core::sync::atomic::Ordering;
            if self.aborted.load(Ordering::Acquire) {
                return Err(CudaError(ERR_ABORTED));
            }
            let api = dl::api()?;
            let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
            let peer = (1 - self.rank) as i32;
            let fail = |what: &str, rc: i32| -> CudaError {
                self.failed.store(true, Ordering::Relaxed);
                dl::check(api, what, rc).err().unwrap_or(CudaError(NCCL_ERROR_BASE + 3))
            };
            // SAFETY: both ranges are inside their buffers (checked above); the group is closed on every path
            unsafe {
                let rc = (api.group_start)();
                if rc != 0 {
                    return Err(fail("ncclGroupStart", rc));
                }
                let mut err: Option<CudaError> = None;
                if send_bytes > 0 {
                    let rc = (api.send)(send.ptr_at(send_off), send_bytes, dl::NCCL_UINT8, peer, self.comm, st.raw());
                    if rc != 0 {
                        err.get_or_insert_with(|| fail("ncclSend", rc));
                    }
                }
                if err.is_none() && recv_bytes > 0 {
                    let rc = (api.recv)(recv.ptr_at(recv_off), recv_bytes, dl::NCCL_UINT8, peer, self.comm, st.raw());
                    if rc != 0 {
                        err.get_or_insert_with(|| fail("ncclRecv", rc));
                    }
                }
                let rc = (api.group_end)();
                if rc != 0 {
                    err.get_or_insert_with(|| fail("ncclGroupEnd", rc));
                }
                if let Some(e) = err {
                    return Err(e);
                }
            }
            Ok(())
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (st, send, recv);
            Err(CudaError(-1))
        }
    }

    /// Sum `count` f32 elements of `buf` starting `off_bytes` bytes in, across every rank, in place, queued on `st`. The call returns when
    /// the collective is enqueued (or recorded, inside a graph capture on `st`); the result is there when `st` reaches this point. Every
    /// rank must issue the same sequence of collectives with the same `count`s. With two ranks the result is exactly `a + b`.
    pub fn allreduce_sum_f32(&self, buf: &mut DevBuf, off_bytes: usize, count: usize, st: &Stream) -> Result<(), CudaError> {
        let bytes = count.checked_mul(4).and_then(|b| b.checked_add(off_bytes));
        if off_bytes % 4 != 0 || bytes.is_none_or(|e| e > buf.bytes()) {
            eprintln!(
                "nccl: allreduce of {count} f32 at byte offset {off_bytes} does not fit a {}-byte buffer (or is not 4-byte aligned)",
                buf.bytes()
            );
            return Err(CudaError(ERR_BAD_ARGUMENT));
        }
        if count == 0 {
            return Ok(());
        }
        #[cfg(not(gb_no_cuda))]
        {
            if self.aborted.load(core::sync::atomic::Ordering::Acquire) {
                return Err(CudaError(ERR_ABORTED));
            }
            let api = dl::api()?;
            let p = buf.ptr_at(off_bytes);
            let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: `p` addresses `count` f32 inside `buf` (checked above); in place (send == recv) is NCCL's supported form
            let rc = unsafe { (api.all_reduce)(p, p, count, dl::NCCL_FLOAT32, dl::NCCL_SUM, self.comm, st.raw()) };
            let r = dl::check(api, "ncclAllReduce", rc);
            if r.is_err() {
                self.failed.store(true, core::sync::atomic::Ordering::Relaxed);
            }
            r
        }
        #[cfg(gb_no_cuda)]
        {
            let _ = (buf, st);
            Err(CudaError(-1))
        }
    }
}

impl Drop for NcclComm {
    fn drop(&mut self) {
        #[cfg(not(gb_no_cuda))]
        if !self.comm.is_null() && !self.aborted.load(core::sync::atomic::Ordering::Acquire) {
            if let Ok(api) = dl::api() {
                // a communicator that saw an error is aborted (destroy would wait for the peers); a healthy one is destroyed, which
                // waits for the work queued on it. Every graph that captured it must be gone first.
                let rc = unsafe {
                    if self.failed.load(core::sync::atomic::Ordering::Relaxed) {
                        (api.comm_abort)(self.comm)
                    } else {
                        (api.comm_destroy)(self.comm)
                    }
                };
                if rc != 0 {
                    eprintln!("nccl: communicator teardown returned {rc}");
                }
            }
        }
    }
}

mod add_ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_tp_add_f32(a: *const c_void, b: *const c_void, out: *mut c_void, n: i64, stream: *mut c_void) -> i32;
        pub fn gb_tp_sum_slots_f32(srcs: *const *const c_void, n_srcs: i32, out: *mut c_void, n: i64, stream: *mut c_void) -> i32;
    }
}

/// One operand of [`sum_slots_f32`].
#[derive(Clone, Copy)]
pub enum SumOperand<'a> {
    /// The destination range itself (an in-place operand: a rank's own buffer).
    Dst,
    /// The range of `count` f32 of this buffer starting this many bytes in.
    Buf(&'a DevBuf, usize),
}

/// `dst[dst_off..] = ops[0] + ops[1] + ... + ops[n-1]` over `count` f32, in operand order, left to right, one IEEE add per element per
/// operand (PTX `add.rn.f32`, no flush-to-zero, no fma), queued on `st`: the local half of the one-shot collective and the whole of
/// the loopback group's reduction. Over operands of disjoint support (exact `-0.0` outside an owner's elements) the result is the
/// owner's element in ANY operand order (`x + -0.0 == x` for every x), so the order is a convention that makes every rank compute the
/// same bits on dense data too. `Dst` may be any operand (in place); a `Buf` must not overlap `dst`. One to [`MAX_ONESHOT_RANKS`]
/// operands.
pub fn sum_slots_f32(
    dst: &mut DevBuf,
    dst_off: usize,
    ops: &[SumOperand<'_>],
    count: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    let fits = |buf: &DevBuf, off: usize| {
        off % 4 == 0 && count.checked_mul(4).and_then(|b| b.checked_add(off)).is_some_and(|e| e <= buf.bytes())
    };
    if ops.is_empty() || ops.len() > MAX_ONESHOT_RANKS || !fits(dst, dst_off) {
        return Err(CudaError(ERR_BAD_ARGUMENT));
    }
    let mut ptrs: [*const c_void; MAX_ONESHOT_RANKS] = [core::ptr::null(); MAX_ONESHOT_RANKS];
    for (p, op) in ptrs.iter_mut().zip(ops) {
        *p = match op {
            SumOperand::Dst => dst.ptr_at(dst_off) as *const c_void,
            SumOperand::Buf(b, off) => {
                if !fits(b, *off) {
                    return Err(CudaError(ERR_BAD_ARGUMENT));
                }
                b.ptr_at(*off) as *const c_void
            }
        };
    }
    if count == 0 {
        return Ok(());
    }
    // SAFETY: every operand and the output are `count` f32 inside their buffers (checked above); the launcher reads the operand
    // pointers on the host and the ranges on the device
    CudaError::check(unsafe {
        add_ffi::gb_tp_sum_slots_f32(ptrs.as_ptr(), ops.len() as i32, dst.ptr_at(dst_off), count as i64, st.raw())
    })
}

/// `dst[dst_off..] += src[src_off..]` over `count` f32, queued on `st` (one IEEE add per element; the reduction of the in-process
/// loopback group). `dst` and `src` must be different buffers or disjoint ranges of one.
pub fn add_f32_into(
    dst: &mut DevBuf,
    dst_off: usize,
    src: &DevBuf,
    src_off: usize,
    count: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    let (d_end, s_end) =
        (count.checked_mul(4).and_then(|b| b.checked_add(dst_off)), count.checked_mul(4).and_then(|b| b.checked_add(src_off)));
    if dst_off % 4 != 0 || src_off % 4 != 0 || d_end.is_none_or(|e| e > dst.bytes()) || s_end.is_none_or(|e| e > src.bytes()) {
        return Err(CudaError(ERR_BAD_ARGUMENT));
    }
    if count == 0 {
        return Ok(());
    }
    let d = dst.ptr_at(dst_off);
    let s = src.ptr_at(src_off);
    // SAFETY: both ranges lie inside their buffers (checked above); the launcher reads `s` and `d` and writes `d`
    CudaError::check(unsafe { add_ffi::gb_tp_add_f32(d as *const c_void, s as *const c_void, d, count as i64, st.raw()) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_library_is_the_override_else_the_loaders_soname() {
        // 1. $REXL3_NCCL_LIB, a path or a soname, verbatim
        assert_eq!(resolve_library(Some("/opt/nccl/lib/libnccl.so.2")), "/opt/nccl/lib/libnccl.so.2");
        assert_eq!(resolve_library(Some("libnccl.so.2.30.7")), "libnccl.so.2.30.7");
        // 2. unset or empty: the bare soname, so dlopen runs the dynamic loader's normal search (no directory is guessed)
        assert_eq!(resolve_library(None), DEFAULT_LIBRARY);
        assert_eq!(resolve_library(Some("")), DEFAULT_LIBRARY);
        assert_eq!(DEFAULT_LIBRARY, "libnccl.so.2");
        assert!(!resolve_library(None).contains('/'), "a bare soname: the loader searches");
    }

    #[test]
    fn an_async_poll_is_healthy_on_success_and_in_progress_and_names_any_other_result() {
        assert_eq!(async_error_code(0, 0), None);
        assert_eq!(async_error_code(0, 7), None);
        assert_eq!(async_error_code(0, 6), Some(NCCL_ERROR_BASE + 6));
        assert_eq!(error_name(async_error_code(0, 2).unwrap()), Some("ncclSystemError"));
        // the poll itself failed: no verdict about the communicator
        assert_eq!(async_error_code(4, 6), None);
    }

    #[test]
    fn the_stub_build_reports_no_library_file_and_no_async_error() {
        if cfg!(gb_no_cuda) {
            assert_eq!(NcclComm::loaded_library_file(), None);
        }
    }

    #[test]
    fn a_unique_id_is_128_bytes_and_round_trips() {
        let mut b = [0u8; 128];
        for (i, v) in b.iter_mut().enumerate() {
            *v = i as u8;
        }
        let id = NcclUniqueId(b);
        let back = NcclUniqueId::from_bytes(id.as_bytes()).unwrap();
        assert_eq!(id, back);
        assert!(NcclUniqueId::from_bytes(&b[..127]).is_none());
        assert!(NcclUniqueId::from_bytes(&[0u8; 129]).is_none());
        assert_eq!(NcclUniqueId::BYTES, core::mem::size_of::<NcclUniqueId>());
    }

    #[test]
    fn error_codes_are_named_and_fail_closed() {
        for r in 1..=7 {
            assert!(error_name(NCCL_ERROR_BASE + r).is_some(), "{r}");
            assert!(CudaError(NCCL_ERROR_BASE + r).is_fatal());
        }
        assert!(error_name(ERR_NO_LIBRARY).is_some() && error_name(ERR_BAD_ARGUMENT).is_some() && error_name(ERR_PEER).is_some());
        assert!(error_name(ERR_ABORTED).is_some());
        assert!(error_name(2).is_none());
        // what a collective that lost its peer returns, and only that
        for c in [NCCL_ERROR_BASE + 1, NCCL_ERROR_BASE + 2, NCCL_ERROR_BASE + 6, NCCL_ERROR_BASE + 7, ERR_PEER, ERR_ABORTED] {
            assert!(is_collective_failure(c), "{c}");
        }
        for c in [0, 2, 700, 999, NCCL_ERROR_BASE, NCCL_ERROR_BASE + 8, ERR_NO_LIBRARY, ERR_BAD_ARGUMENT, 20104, -1] {
            assert!(!is_collective_failure(c), "{c}");
        }
        for c in [ERR_NO_LIBRARY, ERR_BAD_ARGUMENT, ERR_PEER, ERR_ABORTED] {
            assert!(CudaError(c).is_fatal(), "{c}");
        }
    }

    #[test]
    fn an_invalid_position_is_refused_before_any_library_is_touched() {
        let id = NcclUniqueId([0; 128]);
        assert_eq!(NcclComm::init(2, 2, &id).err(), Some(CudaError(ERR_BAD_ARGUMENT)));
        assert_eq!(NcclComm::init(0, 0, &id).err(), Some(CudaError(ERR_BAD_ARGUMENT)));
    }

    #[cfg(gb_no_cuda)]
    #[test]
    fn the_stub_build_returns_errors() {
        assert!(NcclComm::new_unique_id().is_err());
        assert!(NcclComm::version().is_err());
        assert!(NcclComm::init(0, 2, &NcclUniqueId([0; 128])).is_err());
    }

    #[test]
    fn one_shot_staging_is_r_minus_one_aligned_slots() {
        assert_eq!(NcclComm::oneshot_slot_bytes(0), 0);
        assert_eq!(NcclComm::oneshot_slot_bytes(1), 256);
        assert_eq!(NcclComm::oneshot_slot_bytes(256), 256);
        assert_eq!(NcclComm::oneshot_slot_bytes(257), 512);
        assert_eq!(NcclComm::oneshot_staging_bytes(1, 1 << 20), 0);
        assert_eq!(NcclComm::oneshot_staging_bytes(2, 1000), 1024);
        assert_eq!(NcclComm::oneshot_staging_bytes(8, 23_068_672), 7 * 23_068_672);
    }

    // ---- the sum kernel (skips without a device) ----------------------------------------------------------------------------------

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Finite f32s: normal, every fifth a subnormal, every 23rd an exact signed zero.
    fn data(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                let r = splitmix(&mut s);
                let sign = ((r >> 40) & 1) as u32;
                let mant = (r & 0x7F_FFFF) as u32;
                if (r >> 52) % 23 == 0 {
                    f32::from_bits(sign << 31)
                } else if (r >> 48) % 5 == 0 {
                    f32::from_bits(sign << 31 | (mant | 1))
                } else {
                    f32::from_bits(sign << 31 | (127 - 8 + ((r >> 32) % 16) as u32) << 23 | mant)
                }
            })
            .collect()
    }

    #[test]
    fn the_sum_kernel_adds_left_to_right_without_flushing_subnormals() {
        if !crate::device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
        let st = Stream::new().unwrap();
        for &n_ops in &[1usize, 2, 3, 4, 8, MAX_ONESHOT_RANKS] {
            for &count in &[1usize, 3, 4, 7, 1024, 4099, 65_541] {
                for &off in &[0usize, 4, 16] {
                    let host: Vec<Vec<f32>> = (0..n_ops).map(|r| data(1000 + r as u64 * 7 + count as u64, count)).collect();
                    // left to right, host f32 adds (no flush-to-zero on the host)
                    let want: Vec<f32> = (0..count).map(|i| host.iter().skip(1).fold(host[0][i], |a, v| a + v[i])).collect();
                    let bufs: Vec<DevBuf> = host.iter().map(|v| DevBuf::from_slice(v).unwrap()).collect();
                    // out of place: the operands at offset 0, the output at `off` bytes in a canary-filled buffer
                    let mut dst = DevBuf::from_slice(&vec![f32::from_bits(0x7FC0_1234); count + 8]).unwrap();
                    let ops: Vec<SumOperand> = bufs.iter().map(|b| SumOperand::Buf(b, 0)).collect();
                    sum_slots_f32(&mut dst, off, &ops, count, &st).unwrap();
                    st.sync().unwrap();
                    let got: Vec<f32> = dst.to_vec().unwrap();
                    let lo = off / 4;
                    for i in 0..count {
                        assert_eq!(got[lo + i].to_bits(), want[i].to_bits(), "ops {n_ops} count {count} off {off} element {i}");
                    }
                    assert!(
                        got[..lo].iter().chain(&got[lo + count..]).all(|v| v.to_bits() == 0x7FC0_1234),
                        "wrote outside its range"
                    );
                    // in place: operand 0 is the destination itself
                    let mut inplace = DevBuf::from_slice(&host[0]).unwrap();
                    let ops: Vec<SumOperand> =
                        std::iter::once(SumOperand::Dst).chain(bufs.iter().skip(1).map(|b| SumOperand::Buf(b, 0))).collect();
                    sum_slots_f32(&mut inplace, 0, &ops, count, &st).unwrap();
                    st.sync().unwrap();
                    let got: Vec<f32> = inplace.to_vec().unwrap();
                    assert!(
                        got.iter().zip(&want).all(|(g, w)| g.to_bits() == w.to_bits()),
                        "in place: ops {n_ops} count {count}"
                    );
                }
            }
        }
        // arguments outside the contract are refused without a launch
        let a = DevBuf::zeroed(64).unwrap();
        let mut d = DevBuf::zeroed(64).unwrap();
        assert_eq!(sum_slots_f32(&mut d, 0, &[], 4, &st), Err(CudaError(ERR_BAD_ARGUMENT)));
        assert_eq!(sum_slots_f32(&mut d, 0, &[SumOperand::Buf(&a, 0)], 17, &st), Err(CudaError(ERR_BAD_ARGUMENT)));
        assert_eq!(sum_slots_f32(&mut d, 2, &[SumOperand::Buf(&a, 0)], 4, &st), Err(CudaError(ERR_BAD_ARGUMENT)));
        let many: Vec<SumOperand> = (0..MAX_ONESHOT_RANKS + 1).map(|_| SumOperand::Buf(&a, 0)).collect();
        assert_eq!(sum_slots_f32(&mut d, 0, &many, 4, &st), Err(CudaError(ERR_BAD_ARGUMENT)));
    }
}
