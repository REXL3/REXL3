//! How a kernel source file is handed to nvcc, shared by this crate's build script and `crates/quant/build.rs` (which
//! includes this file by path), so that one commit compiles to the same object bytes in every checkout, on every node, at
//! any time.
//!
//! nvcc 13 writes three things into an object that have nothing to do with the code (measured on 13.0.88; see
//! `docs/REPRODUCIBLE_BUILDS.md`):
//!
//! 1. the name of its temporary file, which carries the process id of that nvcc run (an ELF `FILE` symbol);
//! 2. with `-lineinfo`, each source file's modification time and directory (the device code's line tables);
//! 3. a hash of the source file's real path, in the name of every kernel that lives in an anonymous namespace.
//!
//! All three are inputs, so they are fixed before nvcc runs and nothing is patched afterwards:
//!
//! - the sources are copied to a directory whose path is a function of their bytes only ([`stage_sources`]), with one
//!   fixed modification time, and nvcc compiles the copies (2 and 3; a symbolic link does not help, nvcc resolves it);
//! - nvcc keeps its intermediate files in a directory of ours (`--keep --keep-dir`), where it names them after the source
//!   and not after its process id (1). The directory is removed after the compile.
//!
//! No flag in the build script's `NVCC_FLAGS` changes, so the kernel hash and the generated code are what they were.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime};
use std::{env, fs, io};

/// The modification time every staged source gets: 1980-01-01 UTC. A constant, not `SOURCE_DATE_EPOCH`: two nodes
/// with different environments must still produce the same object.
pub const STAGE_MTIME_SECS: u64 = 315_532_800;

/// Where staged sources live. It must be the same real path for every checkout and every node, which rules out
/// `OUT_DIR`; `/tmp` exists on every Linux host that has nvcc.
pub const STAGE_ROOT: &str = "/tmp";

fn fnv1a(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= u64::from(*b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// The directory the sources `files` (name, bytes; any order) of crate `tag` are staged in. The name depends on the
/// bytes alone, so two checkouts of one commit share a directory, and two different commits never do: builds of
/// different commits can run at the same time on one node without seeing each other's sources.
pub fn stage_dir(root: &Path, tag: &str, files: &[(String, Vec<u8>)]) -> PathBuf {
    let mut sorted: Vec<&(String, Vec<u8>)> = files.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for (name, bytes) in sorted {
        fnv1a(&mut hash, name.as_bytes());
        fnv1a(&mut hash, &[0]);
        fnv1a(&mut hash, &(bytes.len() as u64).to_le_bytes());
        fnv1a(&mut hash, bytes);
    }
    root.join(format!("rexl3-kernel-src-{tag}-{hash:016x}"))
}

/// The user this process runs as, read from the owner of `/proc/self` (no libc in a build script).
#[cfg(unix)]
fn own_uid() -> io::Result<u32> {
    use std::os::unix::fs::MetadataExt;
    Ok(fs::metadata("/proc/self")?.uid())
}

/// `dir` must be a real directory (not a link) that only this user can write: another user could otherwise swap a source
/// between our check and nvcc's read.
#[cfg(unix)]
fn check_private_dir(dir: &Path) -> io::Result<()> {
    use std::os::unix::fs::MetadataExt;
    let meta = fs::symlink_metadata(dir)?;
    let refuse = |why: &str| Err(io::Error::other(format!("{} {why}", dir.display())));
    if !meta.file_type().is_dir() {
        return refuse("is not a directory");
    }
    if meta.uid() != own_uid()? {
        return refuse("belongs to another user");
    }
    if meta.mode() & 0o022 != 0 {
        return refuse("is writable by other users");
    }
    Ok(())
}

#[cfg(unix)]
fn create_private_dir(dir: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    match fs::DirBuilder::new().mode(0o755).create(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(e),
    }?;
    check_private_dir(dir)
}

fn stage_time() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(STAGE_MTIME_SECS)
}

/// Writes `bytes` to `path` with the staged modification time, unless the file already is exactly that. The write goes
/// through a temporary name and a rename, so a compile running beside us in another checkout of the same commit never
/// reads half a file.
fn stage_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let current = fs::read(path).ok();
    let time_ok = fs::metadata(path).and_then(|m| m.modified()).is_ok_and(|t| t == stage_time());
    if current.as_deref() == Some(bytes) && time_ok {
        return Ok(());
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    fs::write(&tmp, bytes)?;
    fs::File::options().write(true).open(&tmp)?.set_modified(stage_time())?;
    fs::rename(&tmp, path)
}

/// Copies every file of `csrc` to `<root>/rexl3-kernel-src-<tag>-<hash of the files>/csrc` and returns that directory's
/// parent (the directory nvcc runs in, so that `-I csrc` and `csrc/<name>.cu` mean what they mean in the crate).
///
/// # Errors
/// Any I/O error, and a staging directory that is a link, belongs to another user or is writable by others. The caller
/// then compiles in place and says so; the build still works, it is only not reproducible across paths.
#[cfg(unix)]
pub fn stage_sources(root: &Path, tag: &str, csrc: &Path) -> io::Result<PathBuf> {
    let mut files = Vec::new();
    for entry in fs::read_dir(csrc)? {
        let path = entry?.path();
        if path.is_file() {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            files.push((name, fs::read(&path)?));
        }
    }
    let dir = stage_dir(root, tag, &files);
    create_private_dir(&dir)?;
    let staged = dir.join("csrc");
    create_private_dir(&staged)?;
    for (name, bytes) in &files {
        stage_file(&staged.join(name), bytes)?;
    }
    Ok(dir)
}

/// Staging needs `/proc/self` and Unix ownership checks; elsewhere the caller compiles in place.
#[cfg(not(unix))]
pub fn stage_sources(_root: &Path, _tag: &str, _csrc: &Path) -> io::Result<PathBuf> {
    Err(io::Error::other("kernel sources are staged on Unix hosts only"))
}

/// The staged sources of `csrc` for crate `tag`, or `None` with a build warning that names the reason.
pub fn stage_or_warn(tag: &str, csrc: &Path) -> Option<PathBuf> {
    match stage_sources(Path::new(STAGE_ROOT), tag, csrc) {
        Ok(dir) => Some(dir),
        Err(e) => {
            println!(
                "cargo:warning=kernel sources could not be staged under {STAGE_ROOT} ({e}): compiling them in place, so \
                 this build's binaries depend on the checkout path and the source modification times"
            );
            None
        }
    }
}

/// One nvcc compile of `csrc/<file>` into `object`.
pub struct Compile<'a> {
    pub nvcc: &'a Path,
    /// The CUDA `bin` directory, put first on `PATH` so nvcc finds its own tools.
    pub cuda_bin: &'a Path,
    /// The staged sources ([`stage_sources`]), or `None` to compile the crate's own `csrc` in place.
    pub stage: Option<&'a Path>,
    /// Flags for every file, then `-arch`, then the file's own.
    pub flags: &'a [&'a str],
    pub arch: &'a str,
    pub extra: &'a [String],
    /// The source's file name inside `csrc`.
    pub file: &'a str,
    pub object: &'a Path,
}

impl Compile<'_> {
    /// Runs nvcc. Returns whether it succeeded.
    ///
    /// # Panics
    /// If nvcc cannot be started at all.
    pub fn run(&self) -> bool {
        let mut cmd = Command::new(self.nvcc);
        let mut paths = vec![self.cuda_bin.to_path_buf()];
        if let Some(orig) = env::var_os("PATH") {
            paths.extend(env::split_paths(&orig));
        }
        if let Ok(joined) = env::join_paths(paths) {
            cmd.env("PATH", joined);
        }
        cmd.args(self.flags).arg(format!("-arch={}", self.arch)).args(self.extra);
        // nvcc names kept intermediate files after the source; its default temporary names carry its process id, and the
        // host compiler copies that name into the object.
        let keep = self.object.with_extension("nvcc-keep");
        let _ = fs::remove_dir_all(&keep);
        if let Some(stage) = self.stage {
            if fs::create_dir_all(&keep).is_ok() {
                cmd.arg("--keep").arg("--keep-dir").arg(&keep);
            }
            cmd.current_dir(stage);
        }
        cmd.arg("-o").arg(self.object).arg(Path::new("csrc").join(self.file));
        let ok = cmd.status().expect("run nvcc").success();
        let _ = fs::remove_dir_all(&keep);
        ok
    }
}
