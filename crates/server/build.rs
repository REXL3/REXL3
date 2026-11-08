//! Stamps the engine commit into the binary as `REXL3_COMMIT`, and compiles the vendored `actual-computer/toks`
//! (`v0.3.2`, Apache-2.0) C + NEON/AVX2 assembly tokenizer into `libtoks.a`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let o = Command::new("git").args(args).output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
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
    let commit = match std::env::var("REXL3_COMMIT").ok().filter(|c| !c.trim().is_empty()) {
        Some(c) => c.trim().to_string(),
        None => match git(&["rev-parse", "--short=12", "HEAD"]) {
            Some(c) => {
                let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
                if let Some(dir) = git(&["rev-parse", "--absolute-git-dir"]) {
                    for f in ["HEAD", "index", "packed-refs", "refs/heads"] {
                        if Path::new(&dir).join(f).exists() {
                            println!("cargo:rerun-if-changed={dir}/{f}");
                        }
                    }
                }
                if dirty {
                    format!("{c}-dirty")
                } else {
                    c
                }
            }
            None => {
                println!("cargo:rerun-if-changed=../../.rexl3_commit");
                std::fs::read_to_string("../../.rexl3_commit")
                    .or_else(|_| std::fs::read_to_string(".rexl3_commit"))
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "40b7524dde1e".into())
            }
        },
    };
    println!("cargo:rustc-env=REXL3_COMMIT={commit}");
}
