//! `--verify-pack` reads whole files; on this unified pool the page cache it would fill is memory the KV plan does not get.
//! A file that was not cached must not be left cached by the check, and one that was must stay cached.

use gb_model::verify::{cache_resident_fraction, verify_pack};
use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

fn tmp(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("verify_cache").join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn drop_cache(path: &Path) {
    let f = File::open(path).unwrap();
    // SAFETY: advice about a descriptor we own
    unsafe { libc_fadvise(f.as_raw_fd()) };
}

unsafe fn libc_fadvise(fd: i32) {
    #[cfg(target_os = "linux")]
    extern "C" {
        fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    }
    /// macOS has no `posix_fadvise` (and no way to drop one file's pages): a no-op, so a "cold" read here may be warm.
    #[cfg(not(target_os = "linux"))]
    unsafe fn posix_fadvise(_fd: i32, _offset: i64, _len: i64, _advice: i32) -> i32 {
        0
    }
    posix_fadvise(fd, 0, 0, 4); // POSIX_FADV_DONTNEED
}

fn fraction(path: &Path) -> f64 {
    cache_resident_fraction(&File::open(path).unwrap())
}

fn touch(path: &Path) {
    let mut v = Vec::new();
    File::open(path).unwrap().read_to_end(&mut v).unwrap();
}

#[test]
fn a_cold_file_is_left_cold_and_a_warm_one_stays_warm() {
    let d = tmp("cache");
    let bytes: Vec<u8> = (0..24u32 << 20).map(|i| (i.wrapping_mul(2654435761) >> 9) as u8).collect();
    let (cold, warm) = (d.join("cold.safetensors"), d.join("warm.safetensors"));
    for p in [&cold, &warm] {
        let mut f = File::create(p).unwrap();
        f.write_all(&bytes).unwrap();
        f.sync_all().unwrap(); // dirty pages cannot be dropped
    }
    let crc = crc32fast::hash(&bytes);
    std::fs::write(d.join("crc32.txt"), format!("{crc:08x}  cold.safetensors\n{crc:08x}  warm.safetensors\n")).unwrap();
    drop_cache(&cold);
    touch(&warm);
    if fraction(&cold) > 0.5 {
        return eprintln!("note: this filesystem does not drop cached pages on request: skipping");
    }
    assert!(fraction(&warm) > 0.9, "the warm file is not cached");
    let shards = vec!["cold.safetensors".to_string(), "warm.safetensors".to_string()];
    let v = verify_pack(&d, &[], &shards).unwrap();
    assert_eq!(v.files, 2);
    assert!(fraction(&cold) < 0.5, "the check left the uncached file in the page cache ({:.2} resident)", fraction(&cold));
    assert!(fraction(&warm) > 0.9, "the check dropped a file that was cached ({:.2} resident)", fraction(&warm));
}
