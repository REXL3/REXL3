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
//! Direct I/O ([`SpillIo::Direct`], which [`SpillStore::open`] picks wherever the filesystem takes `O_DIRECT`; Linux only): the
//! files bypass the page cache. A buffered snapshot write asks the kernel for page-cache folios of up to 2 MiB (orders 7 to 9 on
//! ext4). On a GB10 serving at a few GiB of MemAvailable, the free memory beside the weights holds no free block that large,
//! so each such write ran direct reclaim, and that reclaim wrote the server's own idle heap pages to swap (about 1 GiB in a
//! gate run). A direct write allocates no page cache. The file format is unchanged: `hdr_len` is rounded up to
//! [`DIRECT_ALIGN`] (the reader already accepts a zero-padded checksum vector), and the last block is written padded and the
//! file then cut to its exact length, so every transfer is aligned. A direct store reads a buffered store's file (an
//! unaligned `hdr_len`) through the page cache. Other hosts have no `O_DIRECT`: there [`direct_supported`] is false, so
//! [`SpillStore::open`] stays buffered and [`SpillStore::open_io`] refuses [`SpillIo::Direct`].
//!
//! Quota eviction is oldest-mtime-first (the index's LRU order is the engine's; a store-level eviction only ever removes the
//! oldest files when a new snapshot would exceed the quota, so the index's handles for removed files, if any, fail their read
//! and fall back cold cleanly).

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
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
/// The alignment of every direct transfer: buffer address, file offset and length.
pub const DIRECT_ALIGN: usize = 4096;

/// How a store moves bytes to and from its files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpillIo {
    /// Through the page cache.
    Buffered,
    /// `O_DIRECT`: no page cache (the module notes say why the server uses it).
    Direct,
}

impl SpillIo {
    pub fn name(self) -> &'static str {
        match self {
            SpillIo::Buffered => "buffered",
            SpillIo::Direct => "direct",
        }
    }
}

fn align_up(n: usize) -> usize {
    n.div_ceil(DIRECT_ALIGN) * DIRECT_ALIGN
}

/// A zeroed byte buffer whose first byte is [`DIRECT_ALIGN`]-aligned (a `Vec` over-allocated by one alignment, never grown).
#[derive(Default)]
struct Staging {
    buf: Vec<u8>,
    off: usize,
    len: usize,
}

impl Staging {
    fn new(len: usize) -> Self {
        let buf = vec![0u8; len + DIRECT_ALIGN];
        let off = buf.as_ptr().align_offset(DIRECT_ALIGN);
        Staging { buf, off, len }
    }
}

impl std::ops::Deref for Staging {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.buf[self.off..self.off + self.len]
    }
}

impl std::ops::DerefMut for Staging {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.buf[self.off..self.off + self.len]
    }
}

#[cfg(target_os = "linux")]
fn open_flags(io: SpillIo) -> i32 {
    match io {
        SpillIo::Buffered => 0,
        SpillIo::Direct => libc::O_DIRECT,
    }
}

/// Off Linux no store is direct ([`direct_supported`] is false, so [`SpillStore::open_io`] refuses [`SpillIo::Direct`]).
#[cfg(not(target_os = "linux"))]
fn open_flags(io: SpillIo) -> i32 {
    match io {
        SpillIo::Buffered => 0,
        SpillIo::Direct => unreachable!("direct spill I/O is Linux-only"),
    }
}

/// Whether `dir`'s filesystem takes an aligned `O_DIRECT` write (tmpfs, for one, refuses the open). Linux only.
#[cfg(target_os = "linux")]
pub fn direct_supported(dir: &Path) -> bool {
    let probe = dir.join(format!(".direct-probe-{}", std::process::id()));
    let ok = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(libc::O_DIRECT)
        .open(&probe)
        .and_then(|mut f| f.write_all(&Staging::new(DIRECT_ALIGN)))
        .is_ok();
    let _ = fs::remove_file(&probe);
    ok
}

/// Off Linux there is no `O_DIRECT`: every store is buffered.
#[cfg(not(target_os = "linux"))]
pub fn direct_supported(_dir: &Path) -> bool {
    false
}

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
    /// Direct reads (`pread` at `next_off`); else buffered sequential reads.
    direct: bool,
    next_off: u64,
    pos: u64,
    remaining: u64,
    block: usize,
    nblocks: usize,
    cksums: Vec<u64>,
    filled: usize,
    consumed: usize,
    hash: u64,
    staging: Staging,
}

/// The spill store: one directory of snapshot files, one staging buffer, a used-byte counter.
pub struct SpillStore {
    cfg: SpillConfig,
    io: SpillIo,
    staging: Staging,
    used: u64,
    writer: Option<Writer>,
    reader: Option<Reader>,
    /// The path of the last snapshot written (its `id`).
    last_path: PathBuf,
}

impl SpillStore {
    /// Open the store: create the directory and purge every snapshot file in it (a boot). A fresh store is empty. Its files use
    /// direct I/O when the staging size is aligned and the directory's filesystem takes `O_DIRECT`, else the page cache.
    pub fn open(cfg: SpillConfig) -> Result<Self, SpillError> {
        fs::create_dir_all(&cfg.dir)?;
        let io = if cfg.staging_bytes > 0 && cfg.staging_bytes % DIRECT_ALIGN == 0 && direct_supported(&cfg.dir) {
            SpillIo::Direct
        } else {
            SpillIo::Buffered
        };
        Self::open_io(cfg, io)
    }

    /// [`Self::open`] with the I/O mode named. [`SpillIo::Direct`] needs a [`DIRECT_ALIGN`]-multiple staging size and a
    /// filesystem that takes `O_DIRECT`, else the open fails.
    pub fn open_io(cfg: SpillConfig, io: SpillIo) -> Result<Self, SpillError> {
        fs::create_dir_all(&cfg.dir)?;
        if io == SpillIo::Direct {
            if cfg.staging_bytes == 0 || cfg.staging_bytes % DIRECT_ALIGN != 0 {
                return Err(SpillError::Corrupt(format!(
                    "direct I/O needs a staging size that is a multiple of {DIRECT_ALIGN}, not {}",
                    cfg.staging_bytes
                )));
            }
            if !direct_supported(&cfg.dir) {
                return Err(SpillError::Io(io::Error::new(io::ErrorKind::Unsupported, "the spill directory refuses O_DIRECT")));
            }
        }
        let staging = Staging::new(cfg.staging_bytes.max(1));
        let mut store = SpillStore { cfg, io, staging, used: 0, writer: None, reader: None, last_path: PathBuf::new() };
        store.purge()?;
        Ok(store)
    }

    /// Open a buffered store in `dir` with `quota_bytes` and `staging_bytes` (tests).
    pub fn open_at(dir: PathBuf, quota_bytes: u64, staging_bytes: usize) -> Result<Self, SpillError> {
        Self::open_io(SpillConfig { dir, quota_bytes, staging_bytes }, SpillIo::Buffered)
    }

    /// The I/O mode of this store's files.
    pub fn io(&self) -> SpillIo {
        self.io
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
        let need = 88 + (payload_len.div_ceil(self.cfg.staging_bytes as u64) as usize) * 8;
        let header_len = if self.io == SpillIo::Direct { align_up(need) } else { need };
        let path = self.path_of(id);
        let mut file =
            OpenOptions::new().write(true).create(true).truncate(true).custom_flags(open_flags(self.io)).open(&path)?;
        let mut hdr = Staging::new(header_len);
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
                Self::flush_block(&mut self.staging, w, self.io)?;
            }
        }
        Ok(())
    }

    fn flush_block(staging: &mut Staging, w: &mut Writer, io: SpillIo) -> Result<(), SpillError> {
        let cap = staging.len();
        let n = w.buf_used.min(cap);
        if n == 0 {
            return Ok(());
        }
        w.cksums.push(fnv1a64(&staging[..n]));
        if io == SpillIo::Direct {
            // only the last block is partial: written padded to the alignment, the file is cut to its length at the finish
            let padded = align_up(n);
            staging[n..padded].fill(0);
            w.file.write_all(&staging[..padded])?;
        } else {
            w.file.write_all(&staging[..n])?;
        }
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
        Self::flush_block(&mut self.staging, w, self.io)?;
        w.total = actual;
        let nblocks = w.cksums.len();
        let header_len = w.header_len;
        let mut hdr = Staging::new(header_len);
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
        if self.io == SpillIo::Direct {
            w.file.set_len(header_len as u64 + actual)?;
        }
        w.file.sync_all()?;
        self.used = self.used.saturating_add(w.total + header_len as u64);
        Ok(())
    }

    /// Begin reading a snapshot: validate the header against `expect` (the caller's knowledge of what it stored).
    pub fn begin_read(&mut self, path: &Path, expect: SpillMeta) -> Result<(), SpillError> {
        if self.reader.is_some() {
            return Err(SpillError::Corrupt("a read is already in flight".into()));
        }
        let mut direct = self.io == SpillIo::Direct;
        let mut file = if direct {
            OpenOptions::new().read(true).custom_flags(open_flags(SpillIo::Direct)).open(path)?
        } else {
            File::open(path)?
        };
        let mut fixed = [0u8; 88];
        let mut head = None;
        if direct {
            let mut b = Staging::new(DIRECT_ALIGN);
            if read_full_at(&file, &mut b, 0)? < fixed.len() {
                return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
            }
            fixed.copy_from_slice(&b[..88]);
            head = Some(b);
        } else {
            file.read_exact(&mut fixed)?;
        }
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
        let raw = if direct && header_len % DIRECT_ALIGN == 0 {
            let b = match head {
                Some(b) if header_len <= b.len() => b,
                _ => {
                    let mut b = Staging::new(header_len);
                    if read_full_at(&file, &mut b, 0)? < need {
                        return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
                    }
                    b
                }
            };
            b[88..need].to_vec()
        } else {
            if direct {
                // a buffered store's file (its payload is not aligned): read it through the page cache
                file = File::open(path)?;
                direct = false;
            }
            file.seek(SeekFrom::Start(88))?;
            let mut raw = vec![0u8; nblocks * 8];
            file.read_exact(&mut raw)?;
            file.seek(SeekFrom::Start(header_len as u64))?;
            raw
        };
        for (i, c) in cksums.iter_mut().enumerate() {
            *c = u64::from_le_bytes(raw[i * 8..(i + 1) * 8].try_into().unwrap());
        }
        self.reader = Some(Reader {
            file,
            direct,
            next_off: header_len as u64,
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
                if r.direct {
                    // every block but the last is the aligned staging size; the last is read padded and stops at the end of file
                    let padded = align_up(take).min(cap);
                    if read_full_at(&r.file, &mut r.staging[..padded], r.next_off)? < take {
                        return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
                    }
                    r.next_off += take as u64;
                } else {
                    r.file.read_exact(&mut r.staging[..take])?;
                }
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

/// Read into `buf` (a direct file: aligned buffer, offset and length) from `off` until it is full or the file ends; the bytes
/// read. A read that ends off the alignment reached the end of the file (the next read would be an unaligned direct read).
fn read_full_at(file: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        match file.read_at(&mut buf[got..], off + got as u64) {
            Ok(0) => break,
            Ok(n) if n % DIRECT_ALIGN != 0 => {
                got += n;
                break;
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(got)
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

    /// A directory on the build's own filesystem (the workspace target dir): `temp_dir()` can be tmpfs, which refuses `O_DIRECT`.
    #[cfg(target_os = "linux")]
    fn disk_dir(name: &str) -> PathBuf {
        let base = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target"));
        base.join(format!("ling-spill-direct-{name}-{}", std::process::id()))
    }

    /// Pages of `path` resident in the page cache (`mincore` over a read-only mapping that is never touched).
    #[cfg(target_os = "linux")]
    fn cached_pages(path: &Path) -> usize {
        use std::os::fd::AsRawFd;
        let f = File::open(path).unwrap();
        let len = f.metadata().unwrap().len() as usize;
        let pages = len.div_ceil(4096);
        let mut vec = vec![0u8; pages];
        // SAFETY: a fresh read-only shared mapping of `len` bytes of an open file, queried with `mincore` into a vector of one
        // byte per page and unmapped; nothing dereferences it.
        unsafe {
            let addr = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0);
            assert_ne!(addr, libc::MAP_FAILED);
            assert_eq!(libc::mincore(addr, len, vec.as_mut_ptr() as _), 0);
            libc::munmap(addr, len);
        }
        vec.iter().filter(|&&b| b & 1 == 1).count()
    }

    #[cfg(target_os = "linux")]
    fn direct_store(name: &str, staging: usize) -> (PathBuf, SpillStore) {
        let dir = disk_dir(name);
        remove_dir(&dir);
        let st =
            SpillStore::open_io(SpillConfig { dir: dir.clone(), quota_bytes: 1 << 30, staging_bytes: staging }, SpillIo::Direct)
                .expect("the build filesystem takes O_DIRECT");
        assert_eq!(st.io(), SpillIo::Direct);
        (dir, st)
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn the_served_store_picks_direct_io_on_a_disk_and_buffered_on_an_unaligned_staging() {
        let dir = disk_dir("pick");
        remove_dir(&dir);
        let st =
            SpillStore::open(SpillConfig { dir: dir.clone(), quota_bytes: 1 << 30, staging_bytes: SPILL_STAGING_BYTES }).unwrap();
        assert_eq!(st.io(), SpillIo::Direct, "the served staging size on a disk filesystem is direct");
        assert_eq!(st.staging.as_ptr() as usize % DIRECT_ALIGN, 0, "the staging buffer is aligned");
        drop(st);
        let st = SpillStore::open(SpillConfig { dir: dir.clone(), quota_bytes: 1 << 30, staging_bytes: 256 }).unwrap();
        assert_eq!(st.io(), SpillIo::Buffered, "an unaligned staging size stays buffered");
        assert!(matches!(
            SpillStore::open_io(SpillConfig { dir: dir.clone(), quota_bytes: 1 << 30, staging_bytes: 256 }, SpillIo::Direct),
            Err(SpillError::Corrupt(_))
        ));
        remove_dir(&dir);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn a_direct_snapshot_roundtrips_bit_exact_with_aligned_transfers() {
        // a 4 KiB staging: the payload spans two blocks, the second partial
        let (dir, mut st) = direct_store("roundtrip", 4096);
        let p = payload();
        assert!(p.len() > 4096 && p.len() < 8192);
        st.start_write(1, M, p.len() as u64).unwrap();
        st.push_bytes(&p[..1000]).unwrap();
        st.push_bytes(&p[1000..]).unwrap();
        st.finish_write().unwrap();
        let path = st.last_path().to_path_buf();
        let raw = fs::read(&path).unwrap();
        let header_len = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
        assert_eq!(header_len, DIRECT_ALIGN, "the header is padded to the alignment");
        assert_eq!(raw.len(), header_len + p.len(), "the padded tail block was cut off");
        assert_eq!(&raw[header_len..], &p[..], "the payload sits after the padded header");
        assert_eq!(st.used_bytes(), (header_len + p.len()) as u64);
        // a second snapshot, never read through `fs::read`, for the page-cache checks
        st.start_write(6, M, p.len() as u64).unwrap();
        st.push_bytes(&p).unwrap();
        st.finish_write().unwrap();
        let unread = st.last_path().to_path_buf();
        assert_eq!(cached_pages(&unread), 0, "a direct write leaves nothing in the page cache");

        st.begin_read(&unread, M).unwrap();
        let mut got = vec![0u8; p.len()];
        st.read_exact_into(&mut got[..10]).unwrap();
        st.read_exact_into(&mut got[10..]).unwrap();
        st.finish_read().unwrap();
        assert_eq!(got, p, "the payload is bit-exact");
        assert_eq!(cached_pages(&unread), 0, "a direct read leaves nothing in the page cache");
        remove_dir(&dir);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn a_direct_store_detects_corruption_and_a_truncated_payload_and_stays_reusable() {
        let (dir, mut st) = direct_store("corrupt", 4096);
        let p = payload();
        st.start_write(2, M, p.len() as u64).unwrap();
        st.push_bytes(&p).unwrap();
        st.finish_write().unwrap();
        let path = st.last_path().to_path_buf();
        let good = fs::read(&path).unwrap();
        let mut bad = good.clone();
        bad[DIRECT_ALIGN + p.len() / 2] ^= 0xff;
        fs::write(&path, &bad).unwrap();
        let e = st.begin_read(&path, M).and_then(|_| st.read_exact_into(&mut vec![0u8; p.len()]));
        assert!(matches!(e, Err(SpillError::Corrupt(_))), "corruption detected: {e:?}");
        assert_eq!(st.staging.len(), 4096, "the failed reader returned the staging buffer");

        fs::write(&path, &good[..good.len() - 1]).unwrap();
        st.begin_read(&path, M).unwrap();
        assert!(matches!(st.read_exact_into(&mut vec![0u8; p.len()]), Err(SpillError::Io(_))), "a truncated payload fails");

        st.start_write(3, M, p.len() as u64).unwrap();
        st.push_bytes(&p).unwrap();
        st.finish_write().unwrap();
        let again = st.last_path().to_path_buf();
        st.begin_read(&again, M).unwrap();
        let mut got = vec![0u8; p.len()];
        st.read_exact_into(&mut got).unwrap();
        st.finish_read().unwrap();
        assert_eq!(got, p);
        remove_dir(&dir);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn a_direct_store_reads_a_buffered_file_and_a_multi_block_payload() {
        let (dir, mut st) = direct_store("mixed", 4096);
        // a buffered store's file (unaligned header) in the same directory
        let other = disk_dir("mixed-buffered");
        remove_dir(&other);
        let mut b = SpillStore::open_at(other.clone(), 1 << 30, 4096).unwrap();
        let p = payload();
        b.start_write(4, M, p.len() as u64).unwrap();
        b.push_bytes(&p).unwrap();
        b.finish_write().unwrap();
        assert!(cached_pages(b.last_path()) > 0, "a buffered write stays in the page cache (the probe sees it)");
        st.begin_read(b.last_path(), M).unwrap();
        let mut got = vec![0u8; p.len()];
        st.read_exact_into(&mut got).unwrap();
        st.finish_read().unwrap();
        assert_eq!(got, p, "a buffered file reads back through a direct store");

        // exactly three full blocks: no partial tail, nothing to pad or cut
        let big: Vec<u8> = (0..3 * 4096u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        st.start_write(5, M, big.len() as u64).unwrap();
        st.push_bytes(&big).unwrap();
        st.finish_write().unwrap();
        let path = st.last_path().to_path_buf();
        assert_eq!(fs::metadata(&path).unwrap().len() as usize, DIRECT_ALIGN + big.len());
        st.begin_read(&path, M).unwrap();
        let mut got = vec![0u8; big.len()];
        st.read_exact_into(&mut got).unwrap();
        st.finish_read().unwrap();
        assert_eq!(got, big);
        remove_dir(&dir);
        remove_dir(&other);
    }

    /// Off Linux the served store falls back to the page cache: no `O_DIRECT` probe, a buffered pick at the served staging size,
    /// a refused direct open, and a bit-exact buffered roundtrip.
    #[test]
    #[cfg(not(target_os = "linux"))]
    fn off_linux_the_served_store_is_buffered_and_refuses_direct_io() {
        let dir = test_dir("off-linux");
        remove_dir(&dir);
        assert!(!direct_supported(&dir));
        let cfg = SpillConfig { dir: dir.clone(), quota_bytes: 1 << 30, staging_bytes: SPILL_STAGING_BYTES };
        assert!(matches!(
            SpillStore::open_io(cfg.clone(), SpillIo::Direct),
            Err(SpillError::Io(e)) if e.kind() == io::ErrorKind::Unsupported
        ));
        let mut st = SpillStore::open(cfg).unwrap();
        assert_eq!(st.io(), SpillIo::Buffered, "the served staging size is buffered off Linux");
        let p = payload();
        st.start_write(1, M, p.len() as u64).unwrap();
        st.push_bytes(&p).unwrap();
        st.finish_write().unwrap();
        let path = st.last_path().to_path_buf();
        st.begin_read(&path, M).unwrap();
        let mut got = vec![0u8; p.len()];
        st.read_exact_into(&mut got).unwrap();
        st.finish_read().unwrap();
        assert_eq!(got, p, "the buffered payload is bit-exact");
        remove_dir(&dir);
    }
}
