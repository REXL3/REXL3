//! `--verify-pack`: check a pack's files against their checksums before serving it.
//!
//! Two references are used, because the one the pack ships does not cover its weights:
//!
//! * `crc32.txt`, when the pack has one: lines of `<8 hex digits>  <file name>` (CRC-32/IEEE, the `crc32` of zlib, what
//!   `crc32` and `rexl3-serve --print-crc32` print). Every file it lists is read and compared.
//! * Pinned CRC-32s of the shards of a validated pack (`fingerprint::VALIDATED_SHARDS`). Each pin sits beside the SHA-256
//!   the pack is published under (the Hugging Face LFS object id), which the local copies were checked against once when
//!   the pin was taken, so the pin is the published content's checksum, in the fast form.
//!
//! A `.safetensors` file that is missing, or whose checksum differs from either reference, is an error: the pack is
//! corrupt and must not be loaded. A mismatch on another listed file (a tokenizer or a template) is reported as a warning
//! with both values, because a hand-edited `config` or template makes a stale `crc32.txt`, which the shipped packs have
//! (see `docs/EVAL_RESULTS.md`, "Boot: the weight loader pipeline"); it is never silent. Shards that neither reference covers are listed as unverified.
//!
//! The check reads every byte of every covered file, one thread per file. CRC-32 runs on the CPU's CRC instructions at
//! about 10 GB/s per thread, so the 16 GiB target takes about a second warm; SHA-256 would take 15 s (the `sha2` crate has
//! no aarch64 hardware path without its `asm` feature) and buys nothing against accidental corruption. It runs only when
//! asked for; the default boot does none of it.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// `crc32.txt` is unreadable or has a line that is not `<crc> <name>`.
    Manifest(String),
    /// Shards that are missing or whose checksum differs (one line each: file, expected, found).
    Corrupt(Vec<String>),
    Io(String),
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifyError::Manifest(m) => write!(f, "crc32.txt: {m}"),
            VerifyError::Io(m) => write!(f, "{m}"),
            VerifyError::Corrupt(v) => write!(f, "the pack is corrupt: {}", v.join("; ")),
        }
    }
}

impl std::error::Error for VerifyError {}

/// What a successful check covered.
#[derive(Debug, Default, PartialEq)]
pub struct Verified {
    /// Files read and found correct.
    pub files: usize,
    pub bytes: u64,
    pub seconds: f64,
    /// Non-shard files whose `crc32.txt` entry did not match (see the module docs): one line each.
    pub warnings: Vec<String>,
    /// Shards no reference covers.
    pub unverified: Vec<String>,
}

/// The entries of a `crc32.txt`: (file name, crc). Blank lines are skipped; names must be plain file names.
pub fn parse_manifest(text: &str) -> Result<Vec<(String, u32)>, VerifyError> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let bad = |why: &str| VerifyError::Manifest(format!("line {}: {why}: {line:?}", i + 1));
        let (crc, name) = line.split_once(char::is_whitespace).ok_or_else(|| bad("expected `<crc>  <file>`"))?;
        if crc.len() != 8 || !crc.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(bad("the checksum is not 8 hexadecimal digits"));
        }
        let name = name.trim_start().trim_start_matches('*');
        if name.is_empty() || name.contains(['/', '\\']) || name == "." || name == ".." {
            return Err(bad("the file name is not a plain name inside the pack"));
        }
        out.push((name.to_string(), u32::from_str_radix(crc, 16).expect("eight hex digits")));
    }
    Ok(out)
}

/// One file to check: CRC-32 and/or SHA-256 against what is expected.
struct Job<'a> {
    name: &'a str,
    crc: Option<u32>,
    /// Checksum pinned for the validated pack.
    pin: Option<u32>,
}

struct Outcome {
    bytes: u64,
    crc: u32,
}

/// Fraction of `f`'s pages that are in the page cache (1.0 when it cannot be told, so such a file is never dropped).
#[cfg(unix)]
pub fn cache_resident_fraction(f: &File) -> f64 {
    use std::os::unix::io::AsRawFd;
    let Ok(len) = f.metadata().map(|m| m.len() as usize) else { return 1.0 };
    if len == 0 {
        return 1.0;
    }
    // SAFETY: a read-only shared mapping used only for `mincore` (its pages are never touched), unmapped before returning
    unsafe {
        let addr = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0);
        if addr == libc::MAP_FAILED {
            return 1.0;
        }
        // one byte per VM page: 4 KiB on the GB10, 16 KiB on Apple Silicon
        let page = match libc::sysconf(libc::_SC_PAGESIZE) {
            n if n > 0 => n as usize,
            _ => 4096,
        };
        let pages = len.div_ceil(page);
        let mut vec = vec![0u8; pages];
        let rc = libc::mincore(addr, len, vec.as_mut_ptr().cast());
        libc::munmap(addr, len);
        if rc != 0 {
            return 1.0;
        }
        vec.iter().filter(|&&b| b & 1 == 1).count() as f64 / pages as f64
    }
}

#[cfg(not(unix))]
pub fn cache_resident_fraction(_f: &File) -> f64 {
    1.0
}

/// A page-aligned buffer for `O_DIRECT` reads.
#[cfg(unix)]
struct Aligned {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

#[cfg(unix)]
impl Aligned {
    fn new(bytes: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(bytes, 4096).expect("a valid layout");
        // SAFETY: the size is not zero
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self { ptr, layout }
    }
    fn slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` points at `layout.size()` initialised bytes this value owns
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
    }
}

#[cfg(unix)]
impl Drop for Aligned {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}

/// CRC-32 of a file read with `O_DIRECT` (no page cache; large aligned reads keep the SSD at its own speed, about 10 GB/s here
/// against 1 GB/s for one buffered reader of a cold file). `None` when the filesystem refuses direct reads.
#[cfg(unix)]
fn hash_direct(path: &Path) -> std::io::Result<Option<Outcome>> {
    use std::os::unix::fs::FileExt;
    #[cfg(target_os = "linux")]
    use std::os::unix::fs::OpenOptionsExt;
    #[cfg(target_os = "linux")]
    let Ok(f) = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path) else {
        return Ok(None);
    };
    // macOS: no O_DIRECT; F_NOCACHE on a plain descriptor bypasses the page cache the same way
    #[cfg(target_os = "macos")]
    let f = {
        let Ok(f) = std::fs::OpenOptions::new().read(true).open(path) else { return Ok(None) };
        // SAFETY: fcntl on a descriptor we own; F_NOCACHE takes an int flag
        if unsafe { libc::fcntl(std::os::unix::io::AsRawFd::as_raw_fd(&f), libc::F_NOCACHE, 1) } == -1 {
            return Ok(None);
        }
        f
    };
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    return Ok(None);
    let len = f.metadata()?.len();
    let mut buf = Aligned::new(16 << 20);
    let mut crc = crc32fast::Hasher::new();
    let mut off = 0u64;
    while off < len {
        match f.read_at(buf.slice(), off) {
            Ok(0) => break,
            Ok(n) => {
                crc.update(&buf.slice()[..n]);
                off += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            // a filesystem that opens O_DIRECT but refuses the read: use the buffered path
            Err(e) if off == 0 && e.raw_os_error() == Some(libc::EINVAL) => return Ok(None),
            Err(e) => return Err(e),
        }
    }
    Ok(Some(Outcome { bytes: off, crc: crc.finalize() }))
}

/// CRC-32 of a whole file. Reading a pack that is not in the page cache would put all of it there, and on this unified
/// pool cached pages are memory the KV plan does not get (a 16 GiB pack shrank the pool measured at boot from 1,030 pages to
/// about 230), so a file that is mostly uncached is read with `O_DIRECT`, which leaves the cache as it was; where that is
/// refused it is read buffered and dropped from the cache afterwards. A file that is cached is read through the cache.
fn hash_file(path: &Path) -> std::io::Result<Outcome> {
    let mut f = File::open(path)?;
    let was_cold = cache_resident_fraction(&f) < 0.5;
    #[cfg(unix)]
    if was_cold {
        if let Some(o) = hash_direct(path)? {
            return Ok(o);
        }
    }
    let mut buf = vec![0u8; 4 << 20];
    let mut crc = crc32fast::Hasher::new();
    let mut bytes = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        crc.update(&buf[..n]);
        bytes += n as u64;
    }
    #[cfg(unix)]
    if was_cold {
        use std::os::unix::io::AsRawFd;
        // SAFETY: advice about a file descriptor we own; failure only means the pages stay cached
        #[cfg(target_os = "linux")]
        unsafe {
            libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
        }
        // macOS has no fadvise; the pages age out of the unified buffer cache on their own
        #[cfg(not(target_os = "linux"))]
        let _ = f.as_raw_fd();
    }
    #[cfg(not(unix))]
    let _ = was_cold;
    Ok(Outcome { bytes, crc: crc.finalize() })
}

fn is_shard(name: &str) -> bool {
    name.ends_with(".safetensors")
}

/// Check the pack in `dir`: every file `crc32.txt` lists, and every `(file, crc32)` in `pinned`. `shards` names the
/// `.safetensors` files the pack has, so the ones no reference covers can be reported.
pub fn verify_pack(dir: &Path, pinned: &[(&str, u32)], shards: &[String]) -> Result<Verified, VerifyError> {
    let t0 = Instant::now();
    let manifest = match std::fs::read_to_string(dir.join("crc32.txt")) {
        Ok(t) => parse_manifest(&t)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(VerifyError::Manifest(e.to_string())),
    };
    let mut jobs: Vec<Job> = manifest.iter().map(|(n, c)| Job { name: n, crc: Some(*c), pin: None }).collect();
    for (name, crc) in pinned {
        match jobs.iter_mut().find(|j| j.name == *name) {
            Some(j) => j.pin = Some(*crc),
            None => jobs.push(Job { name, crc: None, pin: Some(*crc) }),
        }
    }
    let results: Vec<(usize, std::io::Result<Outcome>)> = std::thread::scope(|s| {
        let handles: Vec<_> = jobs.iter().enumerate().map(|(i, j)| (i, s.spawn(move || hash_file(&dir.join(j.name))))).collect();
        handles
            .into_iter()
            .map(|(i, h)| (i, h.join().unwrap_or_else(|_| Err(std::io::Error::other("a checksum thread panicked")))))
            .collect()
    });
    let mut v = Verified::default();
    let mut corrupt = Vec::new();
    for (i, res) in results {
        let j = &jobs[i];
        let o = match res {
            Ok(o) => o,
            Err(e) if is_shard(j.name) => {
                corrupt.push(format!("{}: {e}", j.name));
                continue;
            }
            Err(e) => {
                v.warnings.push(format!("{}: listed in crc32.txt but unreadable: {e}", j.name));
                continue;
            }
        };
        let mut problems = Vec::new();
        if let Some(want) = j.crc.filter(|&w| w != o.crc) {
            problems.push(format!("crc32 expected {want:08x}, found {:08x}", o.crc));
        }
        if let Some(want) = j.pin.filter(|&w| w != o.crc) {
            problems.push(format!("crc32 expected {want:08x} (pinned for the validated pack), found {:08x}", o.crc));
        }
        if problems.is_empty() {
            v.files += 1;
            v.bytes += o.bytes;
        } else if is_shard(j.name) || j.pin.is_some() {
            corrupt.push(format!("{}: {}", j.name, problems.join(", ")));
        } else {
            v.warnings.push(format!("{}: {} (crc32.txt may be stale)", j.name, problems.join(", ")));
        }
    }
    if !corrupt.is_empty() {
        corrupt.sort();
        return Err(VerifyError::Corrupt(corrupt));
    }
    v.unverified = shards.iter().filter(|s| !jobs.iter().any(|j| j.name == s.as_str())).cloned().collect();
    v.unverified.sort();
    v.seconds = t0.elapsed().as_secs_f64();
    Ok(v)
}

/// Lines for a fresh `crc32.txt` covering every regular file in `dir` except `crc32.txt` itself and hidden files, sorted by
/// name (`rexl3-serve --print-crc32`): what a pack author publishes so that `--verify-pack` covers the shards too.
pub fn manifest_for(dir: &Path) -> Result<String, VerifyError> {
    let io = |e: std::io::Error| VerifyError::Io(format!("{}: {e}", dir.display()));
    let mut names: Vec<String> = Vec::new();
    for e in std::fs::read_dir(dir).map_err(io)? {
        let e = e.map_err(io)?;
        let name = e.file_name().to_string_lossy().to_string();
        if e.file_type().map_err(io)?.is_file() && !name.starts_with('.') && name != "crc32.txt" {
            names.push(name);
        }
    }
    names.sort();
    let lines: Vec<(String, std::io::Result<Outcome>)> = std::thread::scope(|s| {
        let handles: Vec<_> = names.iter().map(|n| (n.clone(), s.spawn(move || hash_file(&dir.join(n))))).collect();
        handles
            .into_iter()
            .map(|(n, h)| (n, h.join().unwrap_or_else(|_| Err(std::io::Error::other("a checksum thread panicked")))))
            .collect()
    });
    let mut out = String::new();
    for (n, r) in lines {
        let o = r.map_err(|e| VerifyError::Io(format!("{n}: {e}")))?;
        out.push_str(&format!("{:08x}  {n}\n", o.crc));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gb-verify-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn crc(b: &[u8]) -> u32 {
        crc32fast::hash(b)
    }

    /// A pack with two shards and a tokenizer, and a crc32.txt that lists all three.
    fn pack(d: &Path) -> (Vec<u8>, Vec<u8>) {
        let a: Vec<u8> = (0..300_000u32).map(|i| (i.wrapping_mul(2654435761) >> 11) as u8).collect();
        let b: Vec<u8> = (0..70_001u32).map(|i| (i ^ (i >> 3)) as u8).collect();
        std::fs::write(d.join("model-00001.safetensors"), &a).unwrap();
        std::fs::write(d.join("model-00002.safetensors"), &b).unwrap();
        std::fs::write(d.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(
            d.join("crc32.txt"),
            format!(
                "{:08x}  model-00001.safetensors\n{:08x} *model-00002.safetensors\n\n{:08X}  tokenizer.json\n",
                crc(&a),
                crc(&b),
                crc(b"{}")
            ),
        )
        .unwrap();
        (a, b)
    }

    fn shards() -> Vec<String> {
        vec!["model-00001.safetensors".into(), "model-00002.safetensors".into()]
    }

    #[test]
    fn a_pack_that_matches_its_manifest_verifies() {
        let d = dir("ok");
        let (a, b) = pack(&d);
        let v = verify_pack(&d, &[], &shards()).unwrap();
        assert_eq!((v.files, v.bytes), (3, (a.len() + b.len() + 2) as u64));
        assert!(v.warnings.is_empty() && v.unverified.is_empty());
    }

    #[test]
    fn one_flipped_bit_in_a_shard_is_an_error_naming_the_shard() {
        let d = dir("flip");
        let (mut a, _) = pack(&d);
        a[123_456] ^= 0x10;
        std::fs::write(d.join("model-00001.safetensors"), &a).unwrap();
        let e = verify_pack(&d, &[], &shards()).unwrap_err();
        match &e {
            VerifyError::Corrupt(v) => {
                assert_eq!(v.len(), 1, "{e}");
                assert!(v[0].starts_with("model-00001.safetensors: crc32 expected"), "{e}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_missing_or_truncated_shard_is_an_error() {
        let d = dir("gone");
        pack(&d);
        std::fs::remove_file(d.join("model-00002.safetensors")).unwrap();
        assert!(
            matches!(verify_pack(&d, &[], &shards()), Err(VerifyError::Corrupt(v)) if v[0].starts_with("model-00002.safetensors"))
        );
        let d = dir("short");
        let (a, _) = pack(&d);
        std::fs::write(d.join("model-00001.safetensors"), &a[..a.len() - 1]).unwrap();
        assert!(matches!(verify_pack(&d, &[], &shards()), Err(VerifyError::Corrupt(_))));
    }

    #[test]
    fn a_stale_entry_for_a_non_shard_is_a_warning_with_both_values() {
        let d = dir("stale");
        pack(&d);
        std::fs::write(d.join("tokenizer.json"), b"{\"edited\": 1}").unwrap();
        let v = verify_pack(&d, &[], &shards()).unwrap();
        assert_eq!(v.files, 2);
        assert_eq!(v.warnings.len(), 1);
        assert!(v.warnings[0].starts_with("tokenizer.json: crc32 expected"), "{:?}", v.warnings);
    }

    #[test]
    fn a_pinned_checksum_catches_what_the_manifest_does_not_cover() {
        let d = dir("pin");
        let (a, b) = pack(&d);
        std::fs::write(d.join("crc32.txt"), "").unwrap(); // a manifest that lists nothing
        let good = [("model-00001.safetensors", crc(&a)), ("model-00002.safetensors", crc(&b))];
        assert_eq!(verify_pack(&d, &good, &shards()).unwrap().files, 2);
        let mut bad = good;
        bad[1].1 ^= 1;
        let e = verify_pack(&d, &bad, &shards()).unwrap_err();
        assert!(
            matches!(&e, VerifyError::Corrupt(v) if v.len() == 1 && v[0].starts_with("model-00002.safetensors: crc32 expected") && v[0].contains("pinned")),
            "{e}"
        );
        // without a pin or a manifest line the shards are reported as unverified, not as fine
        let v = verify_pack(&d, &[], &shards()).unwrap();
        assert_eq!((v.files, v.unverified.len()), (0, 2));
    }

    #[test]
    fn a_malformed_manifest_is_refused() {
        for text in ["zzzzzzzz  a\n", "1234567  a\n", "12345678\n", "12345678  ../a\n", "12345678  sub/a\n", "12345678  \n"] {
            assert!(matches!(parse_manifest(text), Err(VerifyError::Manifest(_))), "{text:?}");
        }
        assert_eq!(
            parse_manifest("0000000a  x.bin\n\n  DEADBEEF *y\n").unwrap(),
            vec![("x.bin".to_string(), 10), ("y".to_string(), 0xDEADBEEF)]
        );
    }

    #[test]
    fn a_generated_manifest_verifies_the_pack_it_came_from() {
        let d = dir("gen");
        pack(&d);
        std::fs::write(d.join("README.md"), b"hello").unwrap();
        std::fs::write(d.join(".hidden"), b"x").unwrap();
        let m = manifest_for(&d).unwrap();
        assert_eq!(m.lines().count(), 4, "{m}");
        assert!(!m.contains("crc32.txt") && !m.contains(".hidden"));
        std::fs::write(d.join("crc32.txt"), &m).unwrap();
        assert_eq!(verify_pack(&d, &[], &shards()).unwrap().files, 4);
    }
}
