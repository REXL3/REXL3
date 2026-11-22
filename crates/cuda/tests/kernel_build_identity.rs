//! One commit compiles to one set of kernel objects (`kernel_build.rs`): the staging directory is a function of the
//! source bytes, staged files carry one fixed modification time, and nvcc, run on the staged copies, writes the same
//! object bytes for two checkouts at different paths with different file times. The nvcc tests need the CUDA toolkit
//! and are ignored, by name, in a build without it.

#[allow(dead_code)]
#[path = "../build.rs"]
mod build_script;

use build_script::kernel_build::{stage_dir, stage_sources, Compile, STAGE_MTIME_SECS, STAGE_ROOT};
use build_script::{find_cuda, NVCC_FLAGS};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// The kernel sources the nvcc tests compile: `moe_probe.cu` keeps its kernels in an anonymous namespace (nvcc names
/// those after the source's real path), `smoke.cu` does not, and both include `common.cuh`.
const SOURCES: &[&str] = &["moe_probe.cu", "smoke.cu", "common.cuh"];

/// This test's own directory under the system's temporary directory, removed when the test ends.
struct Scratch(PathBuf);

impl Scratch {
    fn new(test: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("rexl3-kernel-identity-{}-{test}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    /// A fresh, empty directory `name` inside it.
    fn dir(&self, name: &str) -> PathBuf {
        let dir = self.0.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A checkout-like directory `name` holding `csrc/` with the crate's real sources, every file stamped `mtime_secs`.
    fn checkout(&self, name: &str, mtime_secs: u64) -> PathBuf {
        let dir = self.dir(name);
        let csrc = dir.join("csrc");
        std::fs::create_dir_all(&csrc).unwrap();
        for file in SOURCES {
            let to = csrc.join(file);
            std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc").join(file), &to).unwrap();
            let time = SystemTime::UNIX_EPOCH + Duration::from_secs(mtime_secs);
            std::fs::File::options().write(true).open(&to).unwrap().set_modified(time).unwrap();
        }
        dir
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn files(pairs: &[(&str, &str)]) -> Vec<(String, Vec<u8>)> {
    pairs.iter().map(|(n, b)| ((*n).to_string(), b.as_bytes().to_vec())).collect()
}

#[test]
fn the_staging_directory_depends_on_the_source_bytes_and_on_nothing_else() {
    let root = Path::new("/tmp");
    let a = stage_dir(root, "cuda", &files(&[("a.cu", "one"), ("b.cuh", "two")]));
    assert_eq!(a, stage_dir(root, "cuda", &files(&[("b.cuh", "two"), ("a.cu", "one")])), "the order files are listed in");
    assert_ne!(a, stage_dir(root, "cuda", &files(&[("a.cu", "one"), ("b.cuh", "twO")])), "one changed byte");
    assert_ne!(a, stage_dir(root, "cuda", &files(&[("a.cu", "onet"), ("b.cuh", "wo")])), "a byte moved between files");
    assert_ne!(a, stage_dir(root, "cuda", &files(&[("a.cu", "one"), ("c.cuh", "two")])), "a renamed file");
    assert_ne!(a, stage_dir(root, "quant", &files(&[("a.cu", "one"), ("b.cuh", "two")])), "the crate");
    assert!(a.starts_with("/tmp") && a.file_name().unwrap().to_string_lossy().starts_with("rexl3-kernel-src-cuda-"));
}

#[cfg(unix)]
#[test]
fn staged_files_carry_the_fixed_time_whatever_the_checkout_says() {
    let tmp = Scratch::new("times");
    let root = tmp.dir("stage-root");
    let fixed = SystemTime::UNIX_EPOCH + Duration::from_secs(STAGE_MTIME_SECS);
    let first = stage_sources(&root, "t", &tmp.checkout("a", 1_700_000_000).join("csrc")).unwrap();
    let second = stage_sources(&root, "t", &tmp.checkout("b", 1_800_000_000).join("csrc")).unwrap();
    assert_eq!(first, second, "two checkouts of the same bytes share one staging directory");
    for file in SOURCES {
        let staged = first.join("csrc").join(file);
        assert_eq!(std::fs::metadata(&staged).unwrap().modified().unwrap(), fixed, "{file}");
        let original = Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc").join(file);
        assert_eq!(std::fs::read(&staged).unwrap(), std::fs::read(original).unwrap(), "{file}");
    }
    // A staged file someone altered is put right, bytes and time, by the next build.
    let victim = first.join("csrc").join("smoke.cu");
    std::fs::write(&victim, "tampered").unwrap();
    stage_sources(&root, "t", &tmp.checkout("c", 1_900_000_000).join("csrc")).unwrap();
    assert_ne!(std::fs::read(&victim).unwrap(), b"tampered");
    assert_eq!(std::fs::metadata(&victim).unwrap().modified().unwrap(), fixed);
}

#[cfg(unix)]
#[test]
fn a_staging_directory_another_user_could_write_is_refused() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = Scratch::new("refuse");
    let root = tmp.dir("stage-root");
    let csrc = tmp.checkout("a", 1_700_000_000).join("csrc");
    let dir = stage_sources(&root, "t", &csrc).unwrap();
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777)).unwrap();
    let err = stage_sources(&root, "t", &csrc).unwrap_err().to_string();
    assert!(err.contains("writable by other users"), "{err}");
    // A link in the directory's place is refused too: the sources nvcc reads must be the ones we wrote.
    std::fs::remove_dir_all(&dir).unwrap();
    std::os::unix::fs::symlink(tmp.dir("elsewhere"), &dir).unwrap();
    let err = stage_sources(&root, "t", &csrc).unwrap_err().to_string();
    assert!(err.contains("is not a directory"), "{err}");
}

/// Compiles `file` of the sources under `from` (a directory holding `csrc/`) and returns the object's bytes.
fn object(from: &Path, file: &str, out: &Path) -> Vec<u8> {
    let cuda = find_cuda().expect("the CUDA toolkit");
    let obj = out.join(Path::new(file).with_extension("o"));
    let ok = Compile {
        nvcc: &cuda.join("bin/nvcc"),
        cuda_bin: &cuda.join("bin"),
        stage: Some(from),
        flags: NVCC_FLAGS,
        arch: "sm_121",
        extra: &[],
        file,
        object: &obj,
    }
    .run();
    assert!(ok, "nvcc failed on {file}");
    std::fs::read(&obj).unwrap()
}

#[cfg(unix)]
#[test]
#[cfg_attr(gb_no_cuda, ignore = "needs nvcc: a build without the CUDA toolkit compiles no kernel")]
fn two_checkouts_with_different_paths_and_file_times_compile_to_the_same_objects() {
    let tmp = Scratch::new("same");
    let (a, b) = (tmp.checkout("a", 1_700_000_000), tmp.checkout("b/nested/deeper", 1_800_000_123));
    let staged_a = stage_sources(Path::new(STAGE_ROOT), "identity-test", &a.join("csrc")).unwrap();
    let staged_b = stage_sources(Path::new(STAGE_ROOT), "identity-test", &b.join("csrc")).unwrap();
    assert_eq!(staged_a, staged_b);
    for file in ["moe_probe.cu", "smoke.cu"] {
        let first = object(&staged_a, file, &tmp.dir("out-a"));
        let second = object(&staged_b, file, &tmp.dir("out-b"));
        assert!(first == second, "{file}: the two objects differ ({} and {} bytes)", first.len(), second.len());
    }
    let _ = std::fs::remove_dir_all(staged_a);
}

/// The mutation of the test above: hand nvcc the checkouts themselves and the objects differ, in the kernel names
/// (the path) and in the line tables (the times). If this ever fails, the test above has stopped proving anything.
#[cfg(unix)]
#[test]
#[cfg_attr(gb_no_cuda, ignore = "needs nvcc: a build without the CUDA toolkit compiles no kernel")]
fn compiled_in_place_the_same_sources_give_different_objects() {
    let tmp = Scratch::new("in-place");
    let (a, b) = (tmp.checkout("a", 1_700_000_000), tmp.checkout("b/nested/deeper", 1_800_000_123));
    let first = object(&a, "moe_probe.cu", &tmp.dir("out-a"));
    let second = object(&b, "moe_probe.cu", &tmp.dir("out-b"));
    assert!(first != second, "nvcc no longer writes the source's path or time into the object");
}
