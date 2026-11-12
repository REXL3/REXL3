//! Stamps the engine commit into the binary as `REXL3_COMMIT` ([`stamp`] says from where), and compiles the vendored `actual-computer/toks`
//! (`v0.3.2`, Apache-2.0) C + NEON/AVX2 assembly tokenizer into `libtoks.a`.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The engine commit stamped into the binary as `REXL3_COMMIT`, a module of its own so `tests/build_commit.rs` (which includes this file)
/// can check it. In order, the stamp is:
///
/// 1. `REXL3_COMMIT` from the environment, when set and not blank;
/// 2. the HEAD of the git checkout whose top level IS this workspace (with `-dirty` when tracked files differ). A git repository that only
///    encloses the workspace (an export unpacked inside some other checkout) is not this source, and its commit is never taken;
/// 3. the `.rexl3_commit` file of an export (the workspace root's, then the crate's);
/// 4. `unknown`, with a build warning.
///
/// `REXL3_BUILD_COMMIT` is not a name this build reads. When it is set and `REXL3_COMMIT` is not, the build warns and names `REXL3_COMMIT`.
pub mod stamp {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    /// The stamp of a build with no source of its commit.
    pub const UNKNOWN: &str = "unknown";

    /// Where the stamp came from.
    #[allow(dead_code)] // `File`'s path is read by `tests/build_commit.rs`, not by `main`
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum Source {
        Env,
        Git,
        File(PathBuf),
        Unknown,
    }

    /// The stamp, its source, what the build should warn about, and the files whose change should re-run the build script.
    #[derive(Clone, Debug)]
    pub struct Stamp {
        pub commit: String,
        pub source: Source,
        pub warnings: Vec<String>,
        pub rerun: Vec<PathBuf>,
    }

    fn git_in(dir: &Path, args: &[&str]) -> Option<String> {
        let o = Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
        o.status.success().then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
    }

    fn same_dir(a: &Path, b: &Path) -> bool {
        match (a.canonicalize(), b.canonicalize()) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    }

    fn nonblank(v: Option<&str>) -> Option<&str> {
        v.map(str::trim).filter(|v| !v.is_empty())
    }

    /// The stamp of the crate at `manifest_dir` in the workspace at `root`. `env_commit` is `REXL3_COMMIT`, `wrong_name` is
    /// `REXL3_BUILD_COMMIT` (read only to warn).
    pub fn stamp(root: &Path, manifest_dir: &Path, env_commit: Option<&str>, wrong_name: Option<&str>) -> Stamp {
        let mut warnings = Vec::new();
        let mut rerun = Vec::new();
        let (commit, source) = match nonblank(env_commit) {
            Some(c) => (c.to_string(), Source::Env),
            None => {
                let top = git_in(root, &["rev-parse", "--show-toplevel"]);
                let ours = top.as_deref().is_some_and(|t| same_dir(Path::new(t), root));
                if let (Some(t), false) = (top.as_deref(), ours) {
                    warnings.push(format!(
                        "the git repository at {t} encloses this workspace ({}) but is not its checkout: its commit is not this source's",
                        root.display()
                    ));
                }
                match ours.then(|| git_in(root, &["rev-parse", "--short=12", "HEAD"])).flatten() {
                    Some(c) => {
                        if let Some(dir) = git_in(root, &["rev-parse", "--absolute-git-dir"]) {
                            for f in ["HEAD", "index", "packed-refs", "refs/heads"] {
                                let p = Path::new(&dir).join(f);
                                if p.exists() {
                                    rerun.push(p);
                                }
                            }
                        }
                        let dirty =
                            git_in(root, &["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
                        (if dirty { format!("{c}-dirty") } else { c }, Source::Git)
                    }
                    None => {
                        let files = [root.join(".rexl3_commit"), manifest_dir.join(".rexl3_commit")];
                        rerun.push(files[0].clone());
                        let found = files.iter().find_map(|f| {
                            let s = std::fs::read_to_string(f).ok()?;
                            let s = s.trim();
                            (!s.is_empty()).then(|| (s.to_string(), f.clone()))
                        });
                        match found {
                            Some((c, f)) => (c, Source::File(f)),
                            None => {
                                warnings.push(format!(
                                    "no engine commit: set REXL3_COMMIT=<sha>, build from a git checkout, or put the commit in {}; \
                                     stamped \"{UNKNOWN}\"",
                                    files[0].display()
                                ));
                                (UNKNOWN.to_string(), Source::Unknown)
                            }
                        }
                    }
                }
            }
        };
        if nonblank(wrong_name).is_some() && source != Source::Env {
            warnings
                .push(format!("REXL3_BUILD_COMMIT is not read by this build: set REXL3_COMMIT=<sha> instead (stamped {commit})"));
        }
        Stamp { commit, source, warnings, rerun }
    }
}

fn collect_ext(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|x| x.path())).collect();
    entries.sort();
    for p in entries {
        if p.extension().and_then(|s| s.to_str()) == Some(ext) {
            out.push(p);
        }
    }
}

fn build_toks() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let toks_root = manifest_dir.join("csrc/toks");
    if !toks_root.join("include/toks.h").exists() {
        return;
    }
    println!("cargo:rerun-if-changed={}", toks_root.join("include/toks.h").display());

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let isa = match target_arch.as_str() {
        "aarch64" => "arm64",
        "x86_64" => "x86_64",
        _ => "scalar",
    };

    let mut c_srcs = Vec::new();
    for sub in ["src/core", "src/platform", "src/gen", "src/par"] {
        collect_ext(&toks_root.join(sub), "c", &mut c_srcs);
    }
    let mut s_srcs = Vec::new();
    if isa != "scalar" {
        collect_ext(&toks_root.join("src/asm").join(isa), "S", &mut s_srcs);
    }

    let mut have_flags = Vec::new();
    for s in &s_srcs {
        if let Some(stem) = s.file_stem().and_then(|x| x.to_str()) {
            if stem.starts_with('k') {
                have_flags.push(format!("-DTOKS_HAVE_{}=1", stem.to_ascii_uppercase()));
            }
        }
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj_dir = out_dir.join("toks_obj");
    std::fs::create_dir_all(&obj_dir).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());

    let mut jobs: Vec<(PathBuf, PathBuf, Vec<String>)> = Vec::new();
    for src in &c_srcs {
        println!("cargo:rerun-if-changed={}", src.display());
        let rel = src.strip_prefix(&toks_root).unwrap();
        let obj = obj_dir.join(rel).with_extension("o");
        if let Some(p) = obj.parent() {
            std::fs::create_dir_all(p).unwrap();
        }
        let mut args = vec![
            "-std=c17".to_string(),
            "-O3".to_string(),
            "-fno-strict-aliasing".to_string(),
            "-fwrapv".to_string(),
            "-fno-builtin-strlen".to_string(),
            "-fno-builtin-bcmp".to_string(),
            "-fPIC".to_string(),
            "-fvisibility=hidden".to_string(),
            "-Wno-attributes".to_string(),
            "-Wno-array-bounds".to_string(),
            "-Wno-stringop-overflow".to_string(),
            "-pthread".to_string(),
            format!("-I{}", toks_root.join("include").display()),
            format!("-I{}", toks_root.join("src/core").display()),
            format!("-I{}", toks_root.join("src/platform").display()),
        ];
        if isa == "arm64" {
            args.push("-mbranch-protection=bti".to_string());
        }
        args.extend(have_flags.iter().cloned());
        args.push("-c".to_string());
        args.push(src.display().to_string());
        args.push("-o".to_string());
        args.push(obj.display().to_string());
        jobs.push((src.clone(), obj, args));
    }

    for src in &s_srcs {
        println!("cargo:rerun-if-changed={}", src.display());
        let rel = src.strip_prefix(&toks_root).unwrap();
        let obj = obj_dir.join(rel).with_extension("o");
        if let Some(p) = obj.parent() {
            std::fs::create_dir_all(p).unwrap();
        }
        let args = vec![
            "-fPIC".to_string(),
            format!("-I{}", toks_root.join("include").display()),
            format!("-I{}", toks_root.join("src/core").display()),
            format!("-I{}", toks_root.join("src/platform").display()),
            format!("-I{}", toks_root.join("src/asm").join(isa).display()),
            "-c".to_string(),
            src.display().to_string(),
            "-o".to_string(),
            obj.display().to_string(),
        ];
        jobs.push((src.clone(), obj, args));
    }

    let need_compile: Vec<(PathBuf, PathBuf, Vec<String>)> = jobs
        .iter()
        .filter(|(src, obj, _)| {
            let Ok(om) = std::fs::metadata(obj).and_then(|m| m.modified()) else {
                return true;
            };
            let Ok(sm) = std::fs::metadata(src).and_then(|m| m.modified()) else {
                return true;
            };
            sm > om
        })
        .cloned()
        .collect();

    if !need_compile.is_empty() {
        let workers = std::thread::available_parallelism().map_or(4, |n| n.get()).min(need_compile.len());
        let idx = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                handles.push(s.spawn(|| loop {
                    let i = idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= need_compile.len() {
                        break;
                    }
                    let (_, _, args) = &need_compile[i];
                    let out = Command::new(&cc).args(args).output().expect("failed to invoke C compiler for toks");
                    if !out.status.success() {
                        panic!("toks compile failed: {}\n{}", args.join(" "), String::from_utf8_lossy(&out.stderr));
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });
    }

    let lib_path = out_dir.join("libtoks.a");
    if !need_compile.is_empty() || !lib_path.exists() {
        let _ = std::fs::remove_file(&lib_path);
        let mut ar_args = vec!["rcs".to_string(), lib_path.display().to_string()];
        for (_, obj, _) in &jobs {
            ar_args.push(obj.display().to_string());
        }
        let status = Command::new("ar").args(&ar_args).status().expect("failed to invoke ar for libtoks.a");
        assert!(status.success(), "ar rcs libtoks.a failed");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=toks");
}

fn main() {
    build_toks();
    println!("cargo:rerun-if-env-changed=REXL3_COMMIT");
    println!("cargo:rerun-if-env-changed=REXL3_BUILD_COMMIT");
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest_dir.join("../..");
    let s = stamp::stamp(
        &root,
        &manifest_dir,
        std::env::var("REXL3_COMMIT").ok().as_deref(),
        std::env::var("REXL3_BUILD_COMMIT").ok().as_deref(),
    );
    for f in &s.rerun {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    for w in &s.warnings {
        println!("cargo:warning={w}");
    }
    println!("cargo:rustc-env=REXL3_COMMIT={}", s.commit);
}
