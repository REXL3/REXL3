//! Tokenizer access (`actual-computer/toks` v0.3.2 C + NEON/AVX2 assembly fast path with `tokenizers` fallback)
//! and incremental detokenization.
#![allow(unsafe_code)]

use crate::api::ApiError;
use crate::budget::{TokenizerCharge, TokenizerGate};
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use tokenizers::Tokenizer;

#[allow(non_camel_case_types, dead_code)]
mod ffi {
    use super::{c_char, c_void};

    pub const TOKS_TIER_AUTO: u32 = 0;
    pub const TOKS_ADDED_ALL: u32 = 0;
    pub const TOKS_NO_POSTPROCESS: u32 = 4;
    pub const TOKS_NO_TRUNCATE: u32 = 16;
    pub const TOKS_NO_PAD: u32 = 32;
    pub const TOKS_SKIP_SPECIAL: u32 = 1;
    pub const TOKS_SCRATCH_MEMO_SET: u32 = 1 << 20;

    pub const fn toks_scratch_memo_mib(m: u32) -> u32 {
        TOKS_SCRATCH_MEMO_SET | (m & 0xFFF)
    }

    pub const fn toks_scratch_cache_mib(n: u32) -> u32 {
        (n & 0xFF) << 12
    }

    #[repr(C)]
    pub struct toks_ctx {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct toks_par {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct toks_diag {
        pub code: i64,
        pub what: [c_char; 248],
    }

    #[repr(C)]
    pub struct toks_load_opts {
        pub size: u32,
        pub tier: u32,
        pub flags: u32,
        pub rsv: u32,
        pub diag: *mut toks_diag,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct toks_stream {
        pub opaque: [u64; 8],
    }

    #[repr(C)]
    pub struct toks_info {
        pub size: u32,
        pub abi_major: u32,
        pub abi_minor: u32,
        pub algorithm: u32,
        pub tier: u32,
        pub n_ids: u32,
        pub n_added: u32,
        pub paths: u32,
        pub cpu_features: u64,
        pub max_text: u64,
        pub control_isolation: u32,
        pub rsv: u32,
        pub source_sha256: [u8; 32],
        pub image_sha256: [u8; 32],
        pub name: [c_char; 64],
        pub trunc_on: u32,
        pub trunc_max: u32,
        pub trunc_stride: u32,
        pub pad_on: u32,
        pub pad_fixed: u32,
        pub pad_id: u32,
        pub pad_type_id: u32,
        pub pad_len: u32,
        pub pad_multiple: u32,
        pub pad_left: u32,
        pub n_template_prefix: u32,
        pub n_template_suffix: u32,
        pub seq_type_id: u32,
        pub rsv2: u32,
    }

    extern "C" {
        pub fn toks_load(out: *mut *mut toks_ctx, path: *const c_char, opts: *const toks_load_opts) -> i64;
        pub fn toks_unload(ctx: *mut toks_ctx);
        pub fn toks_scratch_bytes(ctx: *const toks_ctx, max_len: u64, flags: u32) -> u64;
        pub fn toks_scratch_init(ctx: *const toks_ctx, scr: *mut c_void, bytes: u64, flags: u32) -> i64;
        pub fn toks_encode_bound(ctx: *const toks_ctx, len: u64) -> u64;
        pub fn toks_encode(
            ctx: *const toks_ctx,
            text: *const c_void,
            len: u64,
            flags: u32,
            out: *mut u32,
            cap: u64,
            scr: *mut c_void,
        ) -> i64;
        pub fn toks_decode(ctx: *const toks_ctx, ids: *const u32, n: u64, flags: u32, out: *mut u8, cap: u64) -> i64;
        pub fn toks_stream_init(ctx: *const toks_ctx, st: *mut toks_stream, flags: u32);
        pub fn toks_stream_bound(ctx: *const toks_ctx, n_ids: u64) -> u64;
        pub fn toks_stream_push(
            ctx: *const toks_ctx,
            st: *mut toks_stream,
            ids: *const u32,
            n: u64,
            out: *mut u8,
            cap: u64,
        ) -> i64;
        pub fn toks_stream_flush(ctx: *const toks_ctx, st: *mut toks_stream, out: *mut u8, cap: u64) -> i64;
        pub fn toks_get_info(ctx: *const toks_ctx, out: *mut toks_info) -> i64;
        pub fn toks_version() -> *const c_char;
        pub fn toks_par_create(out: *mut *mut toks_par, ctx: *const toks_ctx, n_threads: u32, scratch_flags: u32) -> i64;
        pub fn toks_par_destroy(par: *mut toks_par);
        pub fn toks_par_encode(par: *mut toks_par, text: *const c_void, len: u64, flags: u32, out: *mut u32, cap: u64) -> i64;
    }
}

struct ScratchBuf {
    buf: Vec<u8>,
    max_len: usize,
    flags: u32,
}

impl ScratchBuf {
    fn new(ctx: *const ffi::toks_ctx, max_len: usize, flags: u32) -> Result<Self, String> {
        let cap = max_len.max(64 * 1024);
        let bytes = unsafe { ffi::toks_scratch_bytes(ctx, cap as u64, flags) } as usize;
        let mut buf = vec![0u8; bytes.max(256)];
        let rc = unsafe { ffi::toks_scratch_init(ctx, buf.as_mut_ptr() as *mut c_void, buf.len() as u64, flags) };
        if rc != 0 {
            return Err(format!("toks_scratch_init failed: {rc}"));
        }
        Ok(Self { buf, max_len: cap, flags })
    }

    fn ensure(&mut self, ctx: *const ffi::toks_ctx, len: usize) -> Result<(), String> {
        if len <= self.max_len {
            return Ok(());
        }
        let next_cap = len.next_power_of_two().max(64 * 1024);
        *self = Self::new(ctx, next_cap, self.flags)?;
        Ok(())
    }
}

pub struct ToksEngine {
    ctx: *mut ffi::toks_ctx,
    par: Mutex<*mut ffi::toks_par>,
    scratches: Mutex<Vec<ScratchBuf>>,
    scratch_flags: u32,
}

unsafe impl Send for ToksEngine {}
unsafe impl Sync for ToksEngine {}

impl Drop for ToksEngine {
    fn drop(&mut self) {
        if let Ok(mut p) = self.par.lock() {
            if !p.is_null() {
                unsafe { ffi::toks_par_destroy(*p) };
                *p = core::ptr::null_mut();
            }
        }
        if !self.ctx.is_null() {
            unsafe { ffi::toks_unload(self.ctx) };
            self.ctx = core::ptr::null_mut();
        }
    }
}

impl ToksEngine {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        if std::env::var_os("REXL3_DISABLE_TOKS").is_some_and(|v| v == "1") {
            return Err("disabled via REXL3_DISABLE_TOKS=1".into());
        }
        let c_path = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        let mut diag = ffi::toks_diag { code: 0, what: [0; 248] };
        let opts = ffi::toks_load_opts {
            size: std::mem::size_of::<ffi::toks_load_opts>() as u32,
            tier: ffi::TOKS_TIER_AUTO,
            flags: 0,
            rsv: 0,
            diag: &mut diag,
        };
        let mut ctx: *mut ffi::toks_ctx = core::ptr::null_mut();
        let rc = unsafe { ffi::toks_load(&mut ctx, c_path.as_ptr(), &opts) };
        if rc != 0 || ctx.is_null() {
            let reason = unsafe { CStr::from_ptr(diag.what.as_ptr()).to_string_lossy() };
            return Err(format!("toks_load({}): code {rc} ({reason})", path.display()));
        }
        let scratch_flags = ffi::toks_scratch_memo_mib(4) | ffi::toks_scratch_cache_mib(4);
        let par_threads = std::thread::available_parallelism().map_or(4, |n| n.get().clamp(2, 8)) as u32;
        let mut par: *mut ffi::toks_par = core::ptr::null_mut();
        let prc = unsafe { ffi::toks_par_create(&mut par, ctx, par_threads, ffi::toks_scratch_memo_mib(0)) };
        if prc != 0 {
            par = core::ptr::null_mut();
        }
        let mut info: ffi::toks_info = unsafe { std::mem::zeroed() };
        info.size = std::mem::size_of::<ffi::toks_info>() as u32;
        unsafe { ffi::toks_get_info(ctx, &mut info) };
        let ver = unsafe { CStr::from_ptr(ffi::toks_version()).to_string_lossy() };
        let tier_name = match info.tier {
            1 => "scalar",
            2 => "neon",
            3 => "avx2",
            4 => "avx512",
            _ => "auto",
        };
        eprintln!(
            "rexl3 event=toks_ready version={ver} tier={tier_name} algo={} n_ids={} n_added={} paths=0x{:x} par_threads={par_threads}",
            info.algorithm, info.n_ids, info.n_added, info.paths
        );
        let init_scr = ScratchBuf::new(ctx, 256 * 1024, scratch_flags)?;
        Ok(Self { ctx, par: Mutex::new(par), scratches: Mutex::new(vec![init_scr]), scratch_flags })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let len = text.len();
        let bound = unsafe { ffi::toks_encode_bound(self.ctx, len as u64) } as usize;
        let mut out = vec![0u32; bound.max(1)];
        let flags = ffi::TOKS_ADDED_ALL | ffi::TOKS_NO_POSTPROCESS | ffi::TOKS_NO_TRUNCATE | ffi::TOKS_NO_PAD;

        if len >= 32 * 1024 {
            if let Ok(par_guard) = self.par.try_lock() {
                if !par_guard.is_null() {
                    let n = unsafe {
                        ffi::toks_par_encode(
                            *par_guard,
                            text.as_ptr() as *const c_void,
                            len as u64,
                            flags,
                            out.as_mut_ptr(),
                            bound as u64,
                        )
                    };
                    if n >= 0 && (n as usize) <= bound {
                        out.truncate(n as usize);
                        return Ok(out);
                    }
                }
            }
        }

        let mut scr = self
            .scratches
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .map_or_else(|| ScratchBuf::new(self.ctx, len, self.scratch_flags), Ok)?;
        scr.ensure(self.ctx, len)?;
        let n = unsafe {
            ffi::toks_encode(
                self.ctx,
                text.as_ptr() as *const c_void,
                len as u64,
                flags,
                out.as_mut_ptr(),
                bound as u64,
                scr.buf.as_mut_ptr() as *mut c_void,
            )
        };
        let mut pool = self.scratches.lock().unwrap_or_else(|e| e.into_inner());
        if pool.len() < 8 {
            pool.push(scr);
        }
        drop(pool);
        if n < 0 || (n as usize) > bound {
            return Err(format!("toks_encode failed: {n}"));
        }
        out.truncate(n as usize);
        Ok(out)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Option<String> {
        if ids.is_empty() {
            return Some(String::new());
        }
        let flags = if skip_special { ffi::TOKS_SKIP_SPECIAL } else { 0 };
        let needed = unsafe { ffi::toks_decode(self.ctx, ids.as_ptr(), ids.len() as u64, flags, core::ptr::null_mut(), 0) };
        if needed < 0 {
            return None;
        }
        if needed == 0 {
            return Some(String::new());
        }
        let cap = needed as usize;
        let mut buf = vec![0u8; cap];
        let got = unsafe { ffi::toks_decode(self.ctx, ids.as_ptr(), ids.len() as u64, flags, buf.as_mut_ptr(), cap as u64) };
        if got < 0 {
            return None;
        }
        buf.truncate((got as usize).min(cap));
        Some(String::from_utf8_lossy(&buf).into_owned())
    }

    pub fn stream_init(&self, skip_special: bool) -> ffi::toks_stream {
        let mut st = ffi::toks_stream { opaque: [0; 8] };
        let flags = if skip_special { ffi::TOKS_SKIP_SPECIAL } else { 0 };
        unsafe { ffi::toks_stream_init(self.ctx, &mut st, flags) };
        st
    }

    pub fn stream_push(&self, st: &mut ffi::toks_stream, new: &[u32]) -> Option<String> {
        if new.is_empty() {
            return Some(String::new());
        }
        let cap = (unsafe { ffi::toks_stream_bound(self.ctx, new.len() as u64) } as usize).max(64);
        let mut buf = vec![0u8; cap];
        let n = unsafe { ffi::toks_stream_push(self.ctx, st, new.as_ptr(), new.len() as u64, buf.as_mut_ptr(), cap as u64) };
        if n < 0 {
            return None;
        }
        buf.truncate(n as usize);
        Some(String::from_utf8_lossy(&buf).into_owned())
    }

    pub fn stream_flush(&self, st: &mut ffi::toks_stream) -> Option<String> {
        let cap = (unsafe { ffi::toks_stream_bound(self.ctx, 1) } as usize).max(128);
        let mut buf = vec![0u8; cap];
        let n = unsafe { ffi::toks_stream_flush(self.ctx, st, buf.as_mut_ptr(), cap as u64) };
        if n < 0 {
            return None;
        }
        buf.truncate(n as usize);
        Some(String::from_utf8_lossy(&buf).into_owned())
    }
}

pub struct Tok {
    inner: Tokenizer,
    toks: Option<ToksEngine>,
    /// Ids that end a reply (`generation_config.json` eos list).
    pub eos: Vec<u32>,
    pub think_close: Option<u32>,
    pub vocab: usize,
    /// The processing reserve every encode is charged against (`budget::TokenizerGate`); set once by `http::router`. Without it (a `Tok`
    /// used outside a server) an encode is not charged.
    gate: OnceLock<Arc<TokenizerGate>>,
}

impl Tok {
    /// Load `tokenizer.json` and the eos ids from `generation_config.json` in `dir`. A pack that ships no
    /// `generation_config.json` takes them from `config.json` (top level, then `text_config`) — what Hugging Face's
    /// `GenerationConfig.from_model_config` does for such a pack. A `generation_config.json` that exists but names no
    /// eos id is not completed from anywhere: the caller refuses it.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let tok_file = dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&tok_file).map_err(|e| format!("tokenizer.json: {e}"))?;
        let toks = ToksEngine::from_file(&tok_file).ok();
        let mut eos = Vec::new();
        let ids = |v: Option<&serde_json::Value>, eos: &mut Vec<u32>| match v {
            Some(serde_json::Value::Number(n)) => eos.extend(n.as_u64().map(|x| x as u32)),
            Some(serde_json::Value::Array(a)) => eos.extend(a.iter().filter_map(|x| x.as_u64().map(|x| x as u32))),
            _ => {}
        };
        if let Ok(text) = std::fs::read_to_string(dir.join("generation_config.json")) {
            let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("generation_config.json: {e}"))?;
            ids(v.get("eos_token_id"), &mut eos);
        } else {
            if let Ok(text) = std::fs::read_to_string(dir.join("config.json")) {
                let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("config.json: {e}"))?;
                ids(v.get("eos_token_id"), &mut eos);
                if eos.is_empty() {
                    ids(v.get("text_config").and_then(|t| t.get("eos_token_id")), &mut eos);
                }
            }
            if eos.is_empty() {
                eos.push(1);
            }
        }
        let think_close = inner.token_to_id("</think>");
        let vocab = inner.get_vocab_size(true);
        Ok(Self { inner, toks, eos, think_close, vocab, gate: OnceLock::new() })
    }

    /// Charge every encode of this tokenizer against `gate` from now on. The first gate set stays: a tokenizer serves one server.
    pub fn charge_to(&self, gate: Arc<TokenizerGate>) {
        let _ = self.gate.set(gate);
    }

    /// The priced bytes of tokenizing `text_bytes` bytes, held until the returned charge is dropped (so across the encode and the copy of its
    /// ids). One charge at a time per thread: an encode never calls another.
    fn charged(&self, text_bytes: usize) -> Result<Option<TokenizerCharge>, ApiError> {
        self.gate.get().map(|g| g.charge(text_bytes)).transpose()
    }

    /// Token ids of `text`; special tokens written in the text (`<|im_start|>`) become their ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ApiError> {
        let _charge = self.charged(text.len())?;
        if let Some(t) = &self.toks {
            if let Ok(ids) = t.encode(text) {
                return Ok(ids);
            }
        }
        self.inner.encode(text, false).map(|e| e.get_ids().to_vec()).map_err(|e| ApiError::bad(e.to_string()))
    }

    /// Input and output token buffers are bounded independently. No chunking/truncation: BPE boundaries remain exact.
    pub fn encode_bounded(&self, text: &str, max_bytes: usize) -> Result<Vec<u32>, ApiError> {
        if text.len() > max_bytes {
            return Err(ApiError::bad("tokenizer input exceeds rendered prompt byte limit"));
        }
        let _charge = self.charged(text.len())?;
        let max_ids = max_bytes.checked_add(1024).ok_or_else(|| ApiError::bad("tokenizer token arithmetic overflow"))?;
        if let Some(t) = &self.toks {
            if let Ok(ids) = t.encode(text) {
                if ids.len() > max_ids {
                    return Err(ApiError::bad("tokenizer token budget exceeded"));
                }
                return Ok(ids);
            }
        }
        let encoding = self.inner.encode(text, false).map_err(|e| ApiError::bad(e.to_string()))?;
        if encoding.len() > max_ids {
            return Err(ApiError::bad("tokenizer token budget exceeded"));
        }
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        if let Some(t) = &self.toks {
            if let Some(s) = t.decode(ids, skip_special) {
                return s;
            }
        }
        self.inner.decode(ids, skip_special).unwrap_or_default()
    }

    pub fn token_to_id(&self, t: &str) -> Option<u32> {
        self.inner.token_to_id(t)
    }

    pub fn has_toks(&self) -> bool {
        self.toks.is_some()
    }
}

/// A counting gate: at most `n` holders at a time, the rest wait on a condition variable (std only: it is entered from a thread that may block).
pub struct Gate {
    free: std::sync::Mutex<usize>,
    freed: std::sync::Condvar,
}

/// Holds one place of a [`Gate`] until dropped.
pub struct Place<'a>(&'a Gate);

impl Gate {
    pub fn new(n: usize) -> Self {
        Self { free: std::sync::Mutex::new(n.max(1)), freed: std::sync::Condvar::new() }
    }
    /// Wait for a place.
    pub fn enter(&self) -> Place<'_> {
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        while *free == 0 {
            free = self.freed.wait(free).unwrap_or_else(|e| e.into_inner());
        }
        *free -= 1;
        Place(self)
    }
}

impl Drop for Place<'_> {
    fn drop(&mut self) {
        *self.0.free.lock().unwrap_or_else(|e| e.into_inner()) += 1;
        self.0.freed.notify_one();
    }
}

/// Request work that runs at the same time: one place per core, which is what the async workers themselves used to bound it before it left them.
fn render_gate() -> &'static Gate {
    static G: std::sync::OnceLock<Gate> = std::sync::OnceLock::new();
    G.get_or_init(|| Gate::new(std::thread::available_parallelism().map_or(4, |n| n.get())))
}

/// Run CPU-heavy request work (rendering the chat template, tokenizing) from an async handler without holding one of the runtime's workers.
pub fn off_workers<R>(f: impl FnOnce() -> R) -> R {
    off_workers_in(render_gate(), f)
}

fn off_workers_in<R>(gate: &Gate, f: impl FnOnce() -> R) -> R {
    match tokio::runtime::Handle::try_current().map(|h| h.runtime_flavor()) {
        Ok(tokio::runtime::RuntimeFlavor::MultiThread) => tokio::task::block_in_place(|| {
            let _place = gate.enter();
            f()
        }),
        _ => {
            let _place = gate.enter();
            f()
        }
    }
}

/// Turns a growing token list into text pieces without splitting a multi-byte character: uses `toks_stream` when
/// `toks` is active (zero-allocation state machine), falling back to prefix-window re-decoding when not.
pub struct StreamDecoder {
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
    skip_special: bool,
    toks_st: Option<ffi::toks_stream>,
    toks_failed: bool,
}

impl StreamDecoder {
    pub fn new(skip_special: bool) -> Self {
        Self { ids: Vec::new(), prefix: 0, read: 0, skip_special, toks_st: None, toks_failed: false }
    }

    /// Add tokens; returns the new text that is safe to show (possibly empty).
    pub fn push(&mut self, tok: &Tok, new: &[u32]) -> String {
        if !self.toks_failed {
            if let Some(t) = &tok.toks {
                let st = self.toks_st.get_or_insert_with(|| t.stream_init(self.skip_special));
                if let Some(piece) = t.stream_push(st, new) {
                    return piece;
                }
                self.toks_failed = true;
            }
        }
        self.ids.extend_from_slice(new);
        let before = tok.decode(&self.ids[self.prefix..self.read], self.skip_special);
        let after = tok.decode(&self.ids[self.prefix..], self.skip_special);
        if after.len() > before.len() && !after.ends_with('\u{FFFD}') {
            let piece = after[before.len()..].to_string();
            self.prefix = self.read.saturating_sub(5);
            self.read = self.ids.len();
            piece
        } else {
            String::new()
        }
    }

    /// Whatever is still held back (an incomplete character at the very end becomes U+FFFD, as in a full decode).
    pub fn finish(&mut self, tok: &Tok) -> String {
        if !self.toks_failed {
            if let (Some(t), Some(st)) = (&tok.toks, self.toks_st.as_mut()) {
                if let Some(piece) = t.stream_flush(st) {
                    return piece;
                }
            }
        }
        let before = tok.decode(&self.ids[self.prefix..self.read], self.skip_special);
        let after = tok.decode(&self.ids[self.prefix..], self.skip_special);
        self.prefix = self.ids.len();
        self.read = self.ids.len();
        if after.len() > before.len() {
            after[before.len()..].to_string()
        } else {
            String::new()
        }
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn heavy_work_leaves_the_workers_and_runs_at_most_as_many_at_a_time_as_the_gate_has_places() {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        let gate = Arc::new(Gate::new(3));
        let (now, most) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        rt.block_on(async {
            let tasks: Vec<_> = (0..24)
                .map(|_| {
                    let (gate, now, most) = (gate.clone(), now.clone(), most.clone());
                    tokio::spawn(async move {
                        off_workers_in(&gate, || {
                            let n = now.fetch_add(1, SeqCst) + 1;
                            most.fetch_max(n, SeqCst);
                            std::thread::sleep(Duration::from_millis(30));
                            now.fetch_sub(1, SeqCst);
                        })
                    })
                })
                .collect();
            for t in tasks {
                t.await.unwrap();
            }
        });
        assert_eq!(most.load(SeqCst), 3, "24 jobs through a gate of 3 places on a 2-worker runtime");
        assert_eq!(now.load(SeqCst), 0);
    }

    #[test]
    fn a_gate_of_zero_places_is_one_and_a_place_is_given_back_on_drop() {
        let g = Gate::new(0);
        drop(g.enter());
        drop(g.enter());
        let _a = g.enter();
    }
}
