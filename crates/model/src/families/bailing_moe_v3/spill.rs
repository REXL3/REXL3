//! The NVMe spill store for Ling-3.1 prefix snapshots (design: `plans/2026-11-01-rexl3-roadmap/notes/2026-11-06-ling-262k-full-cache-strategy.md`, section 4).
//!
//! A snapshot is one file under `$XDG_CACHE_HOME/rexl3/prefix/<namespace>` (else `~/.cache/rexl3/prefix/<namespace>`), purged
//! when the store opens (a boot), with a size quota. The quota is `--prefix-cache-mib` when the operator passed it, else
//! `min(free_disk_bytes / 4, 64 GiB)`. Writes and reads stream through one reusable staging buffer
//! ([`SPILL_STAGING_BYTES`], 64 MiB); every block of the payload is checksummed in the header
//! (FNV-1a 64-bit), and a read verifies every block — a missing or corrupted file is an error the caller turns into a cache
//! miss (cold prefill), never silent corruption. When no cache directory is writable [`SpillConfig::probe`] returns `None`
//! and the caller falls back to the existing host-RAM cache with a logged reason.
//!
//! File layout (little-endian), one file per snapshot:
//!
//! ```text
//! header: magic[8] "LINGSPL1" | version u32 | hdr_len u32 | namespace u64 | len_tokens u64
//!         | n_pages u64 | n_kda_layers u64 | mtp u8 | pad[7] | n_mla_caches u64 | page_block_bytes u64
//!         | payload_len u64 | nblocks u64 | checksums[nblocks] u64
//! payload: per MLA cache, in order: n_pages x page_block_bytes of raw page bytes
//!          per KDA layer, in order: u64 s_len | s f32 LE | i32 parity | u64 conv_len | conv u16 LE
//!          if mtp: u64 taps_len | taps u16 LE
//!                   | u64 absorbed | u64 valid_from | u8 last_out_present | u64 last_out_len | last_out u16 LE (if present)
//! ```
//!
//! The payload is one byte stream cut into `nblocks` blocks of [`SPILL_STAGING_BYTES`] (the last one partial); each block's
//! checksum covers exactly its bytes. `hdr_len` is the offset of the payload, at least `88 + 8 * nblocks`; an upper-bound
//! begin can reserve a larger checksum vector, zero-padded when the exact finish writes fewer blocks.
//!
//! Quota eviction is oldest-mtime-first (the index's LRU order is the engine's; a store-level eviction only ever removes the
//! oldest files when a new snapshot would exceed the quota, so the index's handles for removed files, if any, fail their read
//! and fall back cold cleanly).

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// The magic of a spill snapshot file.
pub const MAGIC: [u8; 8] = *b"LINGSPL1";
/// The file format version.
pub const VERSION: u32 = 1;
/// The staging buffer size (per write and per read).
pub const SPILL_STAGING_BYTES: usize = 64 << 20;
/// The default quota when the disk free cannot be read: `min(free_disk / 4, 64 GiB)`.
pub const DEFAULT_QUOTA_BYTES: u64 = 64 << 30;
/// The suffix of a spill snapshot file.
pub const EXT: &str = "ling";

/// The planned spill cache: where it lives and what it may use.
#[derive(Clone, Debug)]
pub struct SpillConfig {
    /// The directory of this namespace's snapshots (created on open).
    pub dir: PathBuf,
    /// The disk quota in bytes (the cache's whole footprint).
    pub quota_bytes: u64,
    /// The single staging buffer size (write and read).
    pub staging_bytes: usize,
}

impl SpillConfig {
    /// Probe for a usable spill cache in `<cache>/rexl3/prefix/<namespace>`:
    /// `$XDG_CACHE_HOME`, else `~/.cache`. `explicit_quota_mib` is `--prefix-cache-mib` when the operator passed it,
    /// else the default `min(free_disk_bytes / 4, 64 GiB)`.
    /// `None`: no writable directory (the caller logs the reason and keeps the RAM cache).
    pub fn probe(namespace_hex: &str, explicit_quota_mib: Option<u64>) -> Option<SpillConfig> {
        let base = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
        let dir = base.join("rexl3").join("prefix").join(namespace_hex);
        if fs::create_dir_all(&dir).is_err() {
            return None;
        }
        if !dir.is_dir() {
            return None;
        }
        let quota_bytes = match explicit_quota_mib {
            Some(mib) => mib << 20,
            None => free_disk_bytes(&dir).map(|free| (free / 4).min(DEFAULT_QUOTA_BYTES)).unwrap_or(DEFAULT_QUOTA_BYTES),
        };
        Some(SpillConfig { dir, quota_bytes, staging_bytes: SPILL_STAGING_BYTES })
    }
}

/// What a snapshot's payload records, validated on read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpillMeta {
    pub namespace: u64,
    /// Token length of the committed prefix.
    pub len_tokens: u64,
    /// MLA pages (256 tokens each).
    pub n_pages: u64,
    /// KDA layers in the pools' order.
    pub n_kda_layers: u64,
    /// The MTP drafter's tap ring and metadata are in the file.
    pub mtp: bool,
    /// MLA caches (each page is `page_block_bytes` per cache).
    pub n_mla_caches: u64,
    pub page_block_bytes: u64,
}

#[derive(Debug)]
pub enum SpillError {
    Io(io::Error),
    /// A header field did not match what the caller asked for, or a block checksum failed.
    Corrupt(String),
    /// The snapshot would exceed the quota even after evicting every other file.
    Space,
}

impl From<io::Error> for SpillError {
    fn from(e: io::Error) -> Self {
        SpillError::Io(e)
    }
}

impl std::fmt::Display for SpillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpillError::Io(e) => write!(f, "spill io: {e}"),
            SpillError::Corrupt(m) => write!(f, "spill corrupt: {m}"),
            SpillError::Space => write!(f, "spill quota exceeded"),
        }
    }
}

impl std::error::Error for SpillError {}

/// FNV-1a 64-bit.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct Statvfs {
    f_bsize: u64,
    f_frsize: u64,
    f_blocks: u64,
    f_bfree: u64,
    f_bavail: u64,
    f_files: u64,
    f_ffree: u64,
    f_favail: u64,
    f_fsid: u64,
    f_flag: u64,
    f_namemax: u64,
    // glibc's struct statvfs ends with int __f_spare[6] (24 bytes); omitting it made the
    // statvfs(2) call write 112 bytes into an 88-byte stack slot, clobbering the saved
    // return address and faulting the next ret (SIGSEGV at pc=0) on this box.
    f_spare: [i32; 6],
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn statvfs(path: *const i8, buf: *mut Statvfs) -> i32;
}

/// Free bytes on the filesystem holding `dir`, when the platform reports them.
#[cfg(target_os = "linux")]
fn free_disk_bytes(dir: &Path) -> Option<u64> {
    use std::ffi::CString;
    let c = CString::new(dir.as_os_str().as_encoded_bytes()).ok()?;
    let mut st = Statvfs {
        f_bsize: 0,
        f_frsize: 0,
        f_blocks: 0,
        f_bfree: 0,
        f_bavail: 0,
        f_files: 0,
        f_ffree: 0,
        f_favail: 0,
        f_fsid: 0,
        f_flag: 0,
        f_namemax: 0,
        f_spare: [0; 6],
    };
    if unsafe { statvfs(c.as_ptr() as *const i8, &mut st) } != 0 {
        return None;
    }
    st.f_bavail.checked_mul(st.f_frsize)
}

#[cfg(not(target_os = "linux"))]
fn free_disk_bytes(_dir: &Path) -> Option<u64> {
    None
}

/// A writer's in-flight state.
struct Writer {
    file: File,
    path: PathBuf,
    total: u64,
    header_len: usize,
    block: usize,
    buf_used: usize,
    cksums: Vec<u64>,
    meta: SpillMeta,
}

/// A reader's in-flight state.
struct Reader {
    file: File,
    pos: u64,
    remaining: u64,
    block: usize,
    nblocks: usize,
    cksums: Vec<u64>,
    filled: usize,
    consumed: usize,
    hash: u64,
    staging: Vec<u8>,
}

/// The spill store: one directory of snapshot files, one staging buffer, a used-byte counter.
pub struct SpillStore {
    cfg: SpillConfig,
    staging: Vec<u8>,
    used: u64,
    writer: Option<Writer>,
    reader: Option<Reader>,
    /// The path of the last snapshot written (its `id`).
    last_path: PathBuf,
}

impl SpillStore {
    /// Open the store: create the directory and purge every snapshot file in it (a boot). A fresh store is empty.
    pub fn open(cfg: SpillConfig) -> Result<Self, SpillError> {
        fs::create_dir_all(&cfg.dir)?;
        let staging = vec![0u8; cfg.staging_bytes.max(1)];
        let mut store = SpillStore { cfg, staging, used: 0, writer: None, reader: None, last_path: PathBuf::new() };
        store.purge()?;
        Ok(store)
    }

    /// Open a store in `dir` with `quota_bytes` and `staging_bytes` (tests).
    pub fn open_at(dir: PathBuf, quota_bytes: u64, staging_bytes: usize) -> Result<Self, SpillError> {
        Self::open(SpillConfig { dir, quota_bytes, staging_bytes })
    }

    pub fn config(&self) -> &SpillConfig {
        &self.cfg
    }

    /// Bytes the store currently holds (snapshot payloads plus headers).
    pub fn used_bytes(&self) -> u64 {
        self.used
    }

    fn path_of(&self, id: u64) -> PathBuf {
        self.cfg.dir.join(format!("{id:016x}.{EXT}"))
    }

    fn purge(&mut self) -> Result<(), io::Error> {
        for e in fs::read_dir(&self.cfg.dir)? {
            let e = e?;
            if e.path().extension().is_some_and(|x| x == EXT) {
                let _ = fs::remove_file(e.path());
            }
        }
        self.used = 0;
        Ok(())
    }

    /// Remove a snapshot file (a release or an eviction).
    pub fn delete(&mut self, path: &Path) -> Result<(), SpillError> {
        if path.extension().is_some_and(|x| x == EXT) {
            if let Ok(m) = fs::metadata(path) {
                self.used = self.used.saturating_sub(m.len());
            }
            let _ = fs::remove_file(path);
        }
        Ok(())
    }

    /// Evict the oldest snapshot files (by mtime) until `extra` bytes fit under the quota. `false` when even an empty store
    /// cannot hold `extra` (the caller refuses the capture).
    fn evict_for(&mut self, extra: u64) -> bool {
        if self.used + extra <= self.cfg.quota_bytes {
            return true;
        }
        let mut files: Vec<(PathBuf, u64)> = Vec::new();
        if let Ok(rd) = fs::read_dir(&self.cfg.dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == EXT) {
                    let mtime = e
                        .metadata()
                        .and_then(|m| m.modified())
                        .map(|t| t.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0))
                        .unwrap_or(0);
                    files.push((p, mtime));
                }
            }
        }
        files.sort_by_key(|(_, t)| *t);
        for (p, _) in files {
            if self.used + extra <= self.cfg.quota_bytes {
                break;
            }
            let _ = self.delete(&p);
        }
        self.used + extra <= self.cfg.quota_bytes
    }

    /// Begin writing a snapshot of `meta` with a payload of `payload_len` bytes. The quota is enforced here (evicting the
    /// oldest files as needed); [`SpillError::Space`] when one snapshot alone cannot fit.
    pub fn start_write(&mut self, id: u64, meta: SpillMeta, payload_len: u64) -> Result<(), SpillError> {
        if self.writer.is_some() {
            return Err(SpillError::Corrupt("a write is already in flight".into()));
        }
        if !self.evict_for(payload_len) {
            return Err(SpillError::Space);
        }
        let header_len = 88 + (payload_len.div_ceil(self.cfg.staging_bytes as u64) as usize) * 8;
        let path = self.path_of(id);
        let mut file = File::create(&path)?;
        let mut hdr = vec![0u8; header_len];
        hdr[..8].copy_from_slice(&MAGIC);
        hdr[8..12].copy_from_slice(&VERSION.to_le_bytes());
        hdr[12..16].copy_from_slice(&(header_len as u32).to_le_bytes());
        hdr[16..24].copy_from_slice(&meta.namespace.to_le_bytes());
        hdr[24..32].copy_from_slice(&meta.len_tokens.to_le_bytes());
        hdr[32..40].copy_from_slice(&meta.n_pages.to_le_bytes());
        hdr[40..48].copy_from_slice(&meta.n_kda_layers.to_le_bytes());
        hdr[48] = u8::from(meta.mtp);
        hdr[56..64].copy_from_slice(&meta.n_mla_caches.to_le_bytes());
        hdr[64..72].copy_from_slice(&meta.page_block_bytes.to_le_bytes());
        hdr[72..80].copy_from_slice(&payload_len.to_le_bytes());
        hdr[80..88].copy_from_slice(&(payload_len.div_ceil(self.cfg.staging_bytes as u64)).to_le_bytes());
        if let Err(e) = file.write_all(&hdr) {
            drop(file);
            let _ = fs::remove_file(&path);
            return Err(e.into());
        }
        self.writer = Some(Writer {
            file,
            path: path.clone(),
            total: payload_len,
            header_len,
            block: 0,
            buf_used: 0,
            cksums: Vec::new(),
            meta,
        });
        self.last_path = path;
        Ok(())
    }

    /// The path the in-flight write will land at.
    pub fn last_path(&self) -> &Path {
        &self.last_path
    }

    /// Abandon an in-flight write and remove its partial snapshot. The next capture can start immediately.
    pub fn abort_write(&mut self) {
        if let Some(w) = self.writer.take() {
            drop(w.file);
            let _ = fs::remove_file(w.path);
        }
    }

    /// Push `bytes` into the payload (they stream through the staging buffer). Any I/O error aborts the write and
    /// removes its partial file, so a later capture is not wedged behind the failed writer.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> Result<(), SpillError> {
        let result = self.push_bytes_inner(bytes);
        if result.is_err() {
            self.abort_write();
        }
        result
    }

    fn push_bytes_inner(&mut self, bytes: &[u8]) -> Result<(), SpillError> {
        let cap = self.cfg.staging_bytes;
        let mut pos = 0usize;
        while pos < bytes.len() {
            let w = self.writer.as_mut().ok_or_else(|| SpillError::Corrupt("no write in flight".into()))?;
            let take = (bytes.len() - pos).min(cap - w.buf_used);
            self.staging[w.buf_used..w.buf_used + take].copy_from_slice(&bytes[pos..pos + take]);
            w.buf_used += take;
            pos += take;
            if w.buf_used == cap {
                Self::flush_block(&mut self.staging, w)?;
            }
        }
        Ok(())
    }

    fn flush_block(staging: &mut Vec<u8>, w: &mut Writer) -> Result<(), SpillError> {
        let cap = staging.len();
        let n = w.buf_used.min(cap);
        if n == 0 {
            return Ok(());
        }
        w.cksums.push(fnv1a64(&staging[..n]));
        w.file.write_all(&staging[..n])?;
        w.buf_used = 0;
        w.block += 1;
        Ok(())
    }

    /// Finish the write: flush the tail, patch the header (payload length, block count, checksums) and close. Any error
    /// removes the partial file; the writer has already been taken, so the store is ready for another capture.
    pub fn finish_write(&mut self) -> Result<(), SpillError> {
        let mut w = self.writer.take().ok_or_else(|| SpillError::Corrupt("no write in flight".into()))?;
        let result = self.finish_write_inner(&mut w);
        if result.is_err() {
            drop(w.file);
            let _ = fs::remove_file(&w.path);
        }
        result
    }

    fn finish_write_inner(&mut self, w: &mut Writer) -> Result<(), SpillError> {
        let actual = w.block as u64 * self.cfg.staging_bytes as u64 + w.buf_used as u64;
        Self::flush_block(&mut self.staging, w)?;
        w.total = actual;
        let nblocks = w.cksums.len();
        let header_len = w.header_len;
        let mut hdr = vec![0u8; header_len];
        hdr[..8].copy_from_slice(&MAGIC);
        hdr[8..12].copy_from_slice(&VERSION.to_le_bytes());
        hdr[12..16].copy_from_slice(&(header_len as u32).to_le_bytes());
        hdr[16..24].copy_from_slice(&w.meta.namespace.to_le_bytes());
        hdr[24..32].copy_from_slice(&w.meta.len_tokens.to_le_bytes());
        hdr[32..40].copy_from_slice(&w.meta.n_pages.to_le_bytes());
        hdr[40..48].copy_from_slice(&w.meta.n_kda_layers.to_le_bytes());
        hdr[48] = u8::from(w.meta.mtp);
        hdr[56..64].copy_from_slice(&w.meta.n_mla_caches.to_le_bytes());
        hdr[64..72].copy_from_slice(&w.meta.page_block_bytes.to_le_bytes());
        hdr[72..80].copy_from_slice(&w.total.to_le_bytes());
        hdr[80..88].copy_from_slice(&(nblocks as u64).to_le_bytes());
        for (i, c) in w.cksums.iter().enumerate() {
            hdr[88 + i * 8..88 + (i + 1) * 8].copy_from_slice(&c.to_le_bytes());
        }
        w.file.seek(SeekFrom::Start(0))?;
        w.file.write_all(&hdr)?;
        w.file.sync_all()?;
        self.used = self.used.saturating_add(w.total + header_len as u64);
        Ok(())
    }

    /// Begin reading a snapshot: validate the header against `expect` (the caller's knowledge of what it stored).
    pub fn begin_read(&mut self, path: &Path, expect: SpillMeta) -> Result<(), SpillError> {
        if self.reader.is_some() {
            return Err(SpillError::Corrupt("a read is already in flight".into()));
        }
        let mut file = File::open(path)?;
        let mut fixed = [0u8; 88];
        file.read_exact(&mut fixed)?;
        if fixed[..8] != MAGIC {
            return Err(SpillError::Corrupt("bad magic".into()));
        }
        if u32::from_le_bytes(fixed[8..12].try_into().unwrap()) != VERSION {
            return Err(SpillError::Corrupt("bad version".into()));
        }
        let header_len = u32::from_le_bytes(fixed[12..16].try_into().unwrap()) as usize;
        if header_len < 88 {
            return Err(SpillError::Corrupt("bad header length".into()));
        }
        let got = SpillMeta {
            namespace: u64::from_le_bytes(fixed[16..24].try_into().unwrap()),
            len_tokens: u64::from_le_bytes(fixed[24..32].try_into().unwrap()),
            n_pages: u64::from_le_bytes(fixed[32..40].try_into().unwrap()),
            n_kda_layers: u64::from_le_bytes(fixed[40..48].try_into().unwrap()),
            mtp: fixed[48] != 0,
            n_mla_caches: u64::from_le_bytes(fixed[56..64].try_into().unwrap()),
            page_block_bytes: u64::from_le_bytes(fixed[64..72].try_into().unwrap()),
        };
        if got != expect {
            return Err(SpillError::Corrupt(format!("header {got:?} != expected {expect:?}")));
        }
        let payload_len = u64::from_le_bytes(fixed[72..80].try_into().unwrap());
        let nblocks = u64::from_le_bytes(fixed[80..88].try_into().unwrap()) as usize;
        let need = 88 + nblocks * 8;
        if header_len < need || (header_len - need) % 8 != 0 {
            return Err(SpillError::Corrupt(format!("header len {header_len} is before or misaligned with {need}")));
        }
        let mut cksums = vec![0u64; nblocks];
        let mut raw = vec![0u8; nblocks * 8];
        file.read_exact(&mut raw)?;
        for (i, c) in cksums.iter_mut().enumerate() {
            *c = u64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
        }
        file.seek(SeekFrom::Start(header_len as u64))?;
        self.reader = Some(Reader {
            file,
            pos: 0,
            remaining: payload_len,
            block: 0,
            nblocks,
            cksums,
            filled: 0,
            consumed: 0,
            hash: 0xcbf2_9ce4_8422_2325,
            staging: std::mem::take(&mut self.staging),
        });
        Ok(())
    }

    /// Abandon an in-flight read and return its staging buffer to the store. Safe after any read error, including an
    /// error returned by the caller between `begin_read` and `finish_read`.
    pub fn abort_read(&mut self) {
        if let Some(mut r) = self.reader.take() {
            self.staging = std::mem::take(&mut r.staging);
        }
    }

    /// Read exactly `out.len()` payload bytes, verifying every block checksum the payload passes through.
    pub fn read_exact_into(&mut self, out: &mut [u8]) -> Result<(), SpillError> {
        let result = self.read_exact_inner(out);
        if result.is_err() {
            self.abort_read();
        }
        result
    }

    fn read_exact_inner(&mut self, out: &mut [u8]) -> Result<(), SpillError> {
        let mut copied = 0usize;
        while copied < out.len() {
            let r = self.reader.as_mut().ok_or_else(|| SpillError::Corrupt("no read in flight".into()))?;
            if r.filled == r.consumed {
                if r.remaining == 0 {
                    return Err(SpillError::Corrupt("payload underrun".into()));
                }
                let cap = r.staging.len();
                let take = (r.remaining as usize).min(cap);
                r.staging[..take].fill(0);
                r.file.read_exact(&mut r.staging[..take])?;
                r.filled = take;
                r.consumed = 0;
            }
            let avail = r.filled - r.consumed;
            let take = avail.min(out.len() - copied);
            let start = r.consumed;
            out[copied..copied + take].copy_from_slice(&r.staging[start..start + take]);
            for &b in &r.staging[start..start + take] {
                r.hash ^= u64::from(b);
                r.hash = r.hash.wrapping_mul(0x100_0000_01b3);
            }
            r.consumed += take;
            r.pos += take as u64;
            r.remaining -= take as u64;
            copied += take;
            if r.consumed == r.filled {
                if r.block >= r.nblocks {
                    return Err(SpillError::Corrupt("too many blocks".into()));
                }
                if r.hash != r.cksums[r.block] {
                    return Err(SpillError::Corrupt(format!("block {} checksum mismatch", r.block)));
                }
                r.hash = 0xcbf2_9ce4_8422_2325;
                r.block += 1;
                r.filled = 0;
                r.consumed = 0;
            }
        }
        Ok(())
    }

    /// Finish the read: every block was verified and the payload is exactly consumed.
    pub fn finish_read(&mut self) -> Result<(), SpillError> {
        let mut r = self.reader.take().ok_or_else(|| SpillError::Corrupt("no read in flight".into()))?;
        self.staging = std::mem::take(&mut r.staging);
        if r.remaining != 0 {
            return Err(SpillError::Corrupt(format!("payload overrun: {} bytes left", r.remaining)));
        }
        if r.block != r.nblocks {
            return Err(SpillError::Corrupt(format!("{} of {} blocks read", r.block, r.nblocks)));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("ling-spill-{name}-{}", std::process::id()))
    }

    fn remove_dir(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    const M: SpillMeta = SpillMeta {
        namespace: 0x1234_5678_9abc_def0,
        len_tokens: 512,
        n_pages: 2,
        n_kda_layers: 3,
        mtp: true,
        n_mla_caches: 10,
        page_block_bytes: 1_024,
    };

    fn payload() -> Vec<u8> {
        // 2 pages x 1024 = 2048; 3 KDA layers: s(256 f32) + parity + conv(128 u16); taps 64 u16; MTP metadata
        let mut p = Vec::new();
        p.extend((0..2048u32).map(|i| (i % 251) as u8));
        for l in 0..3u32 {
            let s: Vec<f32> = (0..256u32).map(|i| (i as f32) * 0.5 + l as f32).collect();
            p.extend_from_slice(&(s.len() as u64).to_le_bytes());
            for v in &s {
                p.extend_from_slice(&v.to_le_bytes());
            }
            p.extend_from_slice(&(-7i32).to_le_bytes());
            let conv: Vec<u16> = (0..128u32).map(|i| (i + l * 10) as u16).collect();
            p.extend_from_slice(&(conv.len() as u64).to_le_bytes());
            for v in &conv {
                p.extend_from_slice(&v.to_le_bytes());
            }
        }
        let taps: Vec<u16> = (0..64u32).map(|i| (i as u16) << 3).collect();
        p.extend_from_slice(&(taps.len() as u64).to_le_bytes());
        for v in &taps {
            p.extend_from_slice(&v.to_le_bytes());
        }
        p.extend_from_slice(&511u64.to_le_bytes()); // absorbed
        p.extend_from_slice(&496u64.to_le_bytes()); // valid_from
        p.push(1); // last_out present
        let last: Vec<u16> = (0..16u32).map(|i| i as u16).collect();
        p.extend_from_slice(&(last.len() as u64).to_le_bytes());
        for v in &last {
            p.extend_from_slice(&v.to_le_bytes());
        }
        p
    }

    #[test]
    fn a_snapshot_roundtrips_bit_exact() {
        let dir = test_dir("roundtrip");
        remove_dir(&dir);
        let mut st = SpillStore::open_at(dir.clone(), 1 << 30, 64 << 10).unwrap();
        let p = payload();
        st.start_write(1, M, p.len() as u64).unwrap();
        st.push_bytes(&p[..1000]).unwrap();
        st.push_bytes(&p[1000..]).unwrap();
        st.finish_write().unwrap();
        let path = st.last_path().to_path_buf();
        assert!(path.exists());
        assert_eq!(st.used_bytes(), (p.len() as u64) + 88 + 1 * 8, "one 64KiB block");

        st.begin_read(&path, M).unwrap();
        let mut got = vec![0u8; p.len()];
        st.read_exact_into(&mut got).unwrap();
        st.finish_read().unwrap();
        assert_eq!(got, p, "the payload is bit-exact");

        st.delete(&path).unwrap();
        assert!(!path.exists());
        remove_dir(&dir);
    }

    #[test]
    fn a_corrupted_block_is_detected() {
        let dir = test_dir("corrupt");
        remove_dir(&dir);
        let mut st = SpillStore::open_at(dir.clone(), 1 << 30, 256).unwrap(); // tiny staging: many blocks
        let p = payload();
        st.start_write(2, M, p.len() as u64).unwrap();
        st.push_bytes(&p).unwrap();
        st.finish_write().unwrap();
        let path = st.last_path().to_path_buf();
        // flip a payload byte in the middle (not the header)
        let mut raw = fs::read(&path).unwrap();
        let nblocks = p.len().div_ceil(256);
        let header_len = 88 + nblocks * 8;
        raw[header_len + p.len() / 2] ^= 0xff;
        fs::write(&path, &raw).unwrap();
        let e = st.begin_read(&path, M).and_then(|_| {
            let mut got = vec![0u8; p.len()];
            st.read_exact_into(&mut got)
        });
        assert!(matches!(e, Err(SpillError::Corrupt(_))), "corruption detected: {e:?}");
        assert!(st.reader.is_none());
        assert_eq!(st.staging.len(), 256, "the failed reader returned the staging buffer");

        st.start_write(21, M, p.len() as u64).unwrap();
        st.push_bytes(&p).unwrap();
        st.finish_write().unwrap();
        let replacement = st.last_path().to_path_buf();
        st.begin_read(&replacement, M).unwrap();
        let mut got = vec![0u8; p.len()];
        st.read_exact_into(&mut got).unwrap();
        st.finish_read().unwrap();
        assert_eq!(got, p);
        remove_dir(&dir);
    }

    #[test]
    fn a_failed_block_write_is_aborted_and_the_next_write_succeeds() {
        let dir = test_dir("failed-block-write-reuse");
        remove_dir(&dir);
        let mut st = SpillStore::open_at(dir.clone(), 1 << 30, 256).unwrap();
        st.start_write(22, M, 256).unwrap();
        let path = st.last_path().to_path_buf();
        // Replace the writable handle with a read-only one without closing an fd behind File's back.
        let w = st.writer.as_mut().unwrap();
        w.file = File::open(&w.path).unwrap();
        assert!(matches!(st.push_bytes(&vec![7u8; 256]), Err(SpillError::Io(_))));
        assert!(st.writer.is_none());
        assert!(!path.exists(), "the partial snapshot was removed");

        let p = payload();
        st.start_write(23, M, p.len() as u64).unwrap();
        st.push_bytes(&p).unwrap();
        st.finish_write().unwrap();
        let replacement = st.last_path().to_path_buf();
        st.begin_read(&replacement, M).unwrap();
        let mut got = vec![0u8; p.len()];
        st.read_exact_into(&mut got).unwrap();
        st.finish_read().unwrap();
        assert_eq!(got, p);
        remove_dir(&dir);
    }

    #[test]
    fn a_failed_finish_write_is_aborted_and_the_next_write_succeeds() {
        let dir = test_dir("failed-finish-write-reuse");
        remove_dir(&dir);
        let mut st = SpillStore::open_at(dir.clone(), 1 << 30, 256).unwrap();
        st.start_write(24, M, 128).unwrap();
        st.push_bytes(&[7u8; 128]).unwrap();
        let path = st.last_path().to_path_buf();
        let w = st.writer.as_mut().unwrap();
        w.file = File::open(&w.path).unwrap();
        assert!(matches!(st.finish_write(), Err(SpillError::Io(_))));
        assert!(st.writer.is_none());
        assert!(!path.exists(), "the partial snapshot was removed");

        let p = payload();
        st.start_write(25, M, p.len() as u64).unwrap();
        st.push_bytes(&p).unwrap();
        st.finish_write().unwrap();
        remove_dir(&dir);
    }

    #[test]
    fn finish_write_records_the_actual_payload_after_an_upper_bound_begin() {
        let dir = test_dir("actual-payload-len");
        remove_dir(&dir);
        let mut st = SpillStore::open_at(dir.clone(), 1 << 30, 256).unwrap();
        let p = payload();
        st.start_write(20, M, p.len() as u64 + 8192).unwrap();
        st.push_bytes(&p).unwrap();
        st.finish_write().unwrap();
        let path = st.last_path().to_path_buf();

        st.begin_read(&path, M).unwrap();
        let mut got = vec![0u8; p.len()];
        st.read_exact_into(&mut got).unwrap();
        st.finish_read().unwrap();
        assert_eq!(got, p);
        remove_dir(&dir);
    }

    #[test]
    fn a_failed_read_leaves_the_store_reusable() {
        let dir = test_dir("failed-read-reuse");
        remove_dir(&dir);
        let mut st = SpillStore::open_at(dir.clone(), 1 << 30, 256).unwrap();
        let p = payload();
        st.start_write(30, M, p.len() as u64).unwrap();
        st.push_bytes(&p).unwrap();
        st.finish_write().unwrap();
        let path = st.last_path().to_path_buf();
        let mut raw = fs::read(&path).unwrap();
        raw.truncate(raw.len() - 1);
        fs::write(&path, &raw).unwrap();
        st.begin_read(&path, M).unwrap();
        let mut got = vec![0u8; p.len()];
        assert!(matches!(st.read_exact_into(&mut got), Err(SpillError::Io(_))), "the truncated payload fails the read");

        st.start_write(31, M, p.len() as u64).unwrap();
        st.push_bytes(&p).unwrap();
        st.finish_write().unwrap();
        remove_dir(&dir);
    }

    #[test]
    fn an_unfinished_read_leaves_the_store_reusable() {
        let dir = test_dir("unfinished-read-reuse");
        remove_dir(&dir);
        let mut st = SpillStore::open_at(dir.clone(), 1 << 30, 256).unwrap();
        let p = payload();
        st.start_write(40, M, p.len() as u64).unwrap();
        st.push_bytes(&p).unwrap();
        st.finish_write().unwrap();
        let path = st.last_path().to_path_buf();
        st.begin_read(&path, M).unwrap();
        assert!(st.finish_read().is_err(), "an unread payload cannot finish");

        st.start_write(41, M, p.len() as u64).unwrap();
        st.push_bytes(&p).unwrap();
        st.finish_write().unwrap();
        remove_dir(&dir);
    }

    #[test]
    fn a_missing_or_mismatched_snapshot_is_an_error() {
        let dir = test_dir("missing");
        remove_dir(&dir);
        let mut st = SpillStore::open_at(dir.clone(), 1 << 30, 64 << 10).unwrap();
        let missing = dir.join("deadbeef00000000.ling");
        assert!(matches!(st.begin_read(&missing, M), Err(SpillError::Io(_))));
        let p = payload();
        st.start_write(3, M, p.len() as u64).unwrap();
        st.push_bytes(&p).unwrap();
        st.finish_write().unwrap();
        let path = st.last_path().to_path_buf();
        let mut wrong = M;
        wrong.len_tokens = 511;
        assert!(matches!(st.begin_read(&path, wrong), Err(SpillError::Corrupt(_))));
        remove_dir(&dir);
    }

    #[test]
    fn quota_eviction_removes_the_oldest_first() {
        let dir = test_dir("quota");
        remove_dir(&dir);
        // three snapshots of ~200 bytes each; quota fits one
        let mut st = SpillStore::open_at(dir.clone(), 300, 64 << 10).unwrap();
        let small = vec![7u8; 200];
        for id in [10u64, 11, 12] {
            let meta = SpillMeta { len_tokens: id, ..M };
            st.start_write(id, meta, small.len() as u64).unwrap();
            st.push_bytes(&small).unwrap();
            st.finish_write().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        // 3 x ~288 bytes > 600: the oldest were evicted as each of the last writes needed room
        let remaining: Vec<String> =
            fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        assert_eq!(remaining.len(), 1, "only the newest survives: {remaining:?}");
        assert!(remaining[0].starts_with("000000000000000c"), "the newest id is 12: {remaining:?}");
        remove_dir(&dir);
    }
}
