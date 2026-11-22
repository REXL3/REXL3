//! The engine commit stamp (`build.rs`, its `stamp` module), on real git repositories in a scratch directory: the environment wins;
//! a checkout of the workspace gives its HEAD; a git repository that only ENCLOSES the workspace (an export unpacked inside another
//! checkout) is never taken, and the export's `.rexl3_commit` is used instead; with nothing, the stamp is `unknown` with a warning (not a
//! made-up commit); `REXL3_BUILD_COMMIT` is not read, and is named in a warning.

// the build script itself (its tokenizer build and `main` are not called here)
#[allow(dead_code)]
#[path = "../build.rs"]
mod build_script;

use build_script::stamp::{stamp, Source, UNKNOWN};
use build_script::{tokenizer_aux_mtime, tokenizer_inputs, tokenizer_object_is_fresh};
use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rexl3-build-commit-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn git(dir: &Path, args: &[&str]) -> String {
    let o = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"])
        .args(args)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .output()
        .expect("git");
    assert!(o.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).trim().to_string()
}

/// A git repository at `dir` with one commit; returns its HEAD, 12 hex digits.
fn repo(dir: &Path) -> String {
    git(dir, &["init", "-q"]);
    std::fs::write(dir.join("f.txt"), "x\n").unwrap();
    git(dir, &["add", "f.txt"]);
    git(dir, &["commit", "-q", "-m", "c"]);
    git(dir, &["rev-parse", "--short=12", "HEAD"])
}

/// A workspace layout: `root/crates/server`.
fn workspace(root: &Path) -> PathBuf {
    let m = root.join("crates/server");
    std::fs::create_dir_all(&m).unwrap();
    m
}

#[test]
fn the_environment_commit_wins_and_a_blank_one_does_not() {
    let d = scratch("env");
    let m = workspace(&d);
    let s = stamp(&d, &m, Some(" abc123 "), None);
    assert_eq!((s.commit.as_str(), &s.source), ("abc123", &Source::Env));
    assert!(s.warnings.is_empty(), "{:?}", s.warnings);
    let s = stamp(&d, &m, Some("  "), None);
    assert_ne!(s.source, Source::Env);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_checkout_of_the_workspace_stamps_its_head_and_marks_it_dirty() {
    let d = scratch("ours");
    let head = repo(&d);
    let m = workspace(&d);
    let s = stamp(&d, &m, None, None);
    assert_eq!((s.commit.as_str(), &s.source), (head.as_str(), &Source::Git));
    assert!(s.warnings.is_empty(), "{:?}", s.warnings);
    assert!(s.rerun.iter().any(|p| p.ends_with("HEAD")), "{:?}", s.rerun);
    std::fs::write(d.join("f.txt"), "y\n").unwrap();
    assert_eq!(stamp(&d, &m, None, None).commit, format!("{head}-dirty"));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn an_enclosing_repository_is_not_the_source_and_the_exports_file_is_used() {
    // an export (no .git of its own) unpacked inside another checkout
    let d = scratch("foreign");
    let foreign = repo(&d);
    let root = d.join("export");
    let m = workspace(&root);
    let s = stamp(&root, &m, None, None);
    assert_ne!(s.commit, foreign, "the enclosing repository's commit was taken");
    assert_eq!((s.commit.as_str(), &s.source), (UNKNOWN, &Source::Unknown));
    assert!(s.warnings.iter().any(|w| w.contains("encloses this workspace")), "{:?}", s.warnings);
    assert!(s.warnings.iter().any(|w| w.contains("REXL3_COMMIT") && w.contains(UNKNOWN)), "{:?}", s.warnings);
    // with the export's commit file (the corresponding-source archive and the production repository carry one)
    std::fs::write(root.join(".rexl3_commit"), "0123456789abcdef0123456789abcdef01234567\n").unwrap();
    let s = stamp(&root, &m, None, None);
    assert_eq!(s.commit, "0123456789abcdef0123456789abcdef01234567");
    assert_eq!(s.source, Source::File(root.join(".rexl3_commit")));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn no_source_of_the_commit_stamps_unknown_never_a_made_up_commit() {
    let d = scratch("none");
    // a scratch directory may itself sit inside a checkout (a CI temp dir does not, a developer's may): the stamp is never its commit
    let m = workspace(&d);
    let s = stamp(&d, &m, None, None);
    assert_eq!((s.commit.as_str(), &s.source), (UNKNOWN, &Source::Unknown));
    // the crate's own file is the second place an export may carry it
    std::fs::write(m.join(".rexl3_commit"), "feedfacecafe\n").unwrap();
    let s = stamp(&d, &m, None, None);
    assert_eq!((s.commit.as_str(), &s.source), ("feedfacecafe", &Source::File(m.join(".rexl3_commit"))));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn rexl3_build_commit_is_not_read_and_the_warning_names_rexl3_commit() {
    let d = scratch("wrong");
    let m = workspace(&d);
    let s = stamp(&d, &m, None, Some("41ded37e"));
    assert_eq!(s.commit, UNKNOWN, "REXL3_BUILD_COMMIT must not be read as the commit");
    assert!(
        s.warnings.iter().any(|w| w.contains("REXL3_BUILD_COMMIT is not read") && w.contains("REXL3_COMMIT=")),
        "{:?}",
        s.warnings
    );
    // set alongside REXL3_COMMIT: the right name is in use, nothing to say about it
    let s = stamp(&d, &m, Some("41ded37e"), Some("41ded37e"));
    assert_eq!(s.commit, "41ded37e");
    assert!(s.warnings.is_empty(), "{:?}", s.warnings);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn tokenizer_rerun_inputs_include_directories_sources_headers_and_includes() {
    let d = scratch("toks-inputs");
    for rel in [
        "include/toks.h",
        "src/core/bpe.h",
        "src/core/tables.inc",
        "src/platform/decode.hpp",
        "src/gen/tables.cc",
        "src/asm/arm64/asm.h",
        "src/core/bpe.c",
        "src/asm/arm64/kernel.S",
        "README.md",
    ] {
        let path = d.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"input").unwrap();
    }
    let got: Vec<String> =
        tokenizer_inputs(&d).into_iter().map(|p| p.strip_prefix(&d).unwrap().to_string_lossy().replace('\\', "/")).collect();
    for expected in [
        "",
        "include",
        "include/toks.h",
        "src/asm/arm64",
        "src/asm/arm64/asm.h",
        "src/asm/arm64/kernel.S",
        "src/core",
        "src/core/bpe.c",
        "src/core/bpe.h",
        "src/core/tables.inc",
        "src/gen/tables.cc",
        "src/platform/decode.hpp",
    ] {
        assert!(got.iter().any(|p| p == expected), "missing {expected:?}: {got:?}");
    }
    assert!(!got.iter().any(|p| p == "README.md"));
    let real = Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/toks");
    assert!(tokenizer_inputs(&real).iter().any(|p| p.ends_with("src/core/bpe.h")));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn editing_only_a_tokenizer_header_or_include_marks_newer_objects_stale() {
    for extension in ["h", "inc"] {
        let d = scratch(&format!("toks-{extension}-mtime"));
        let src = d.join("src/core/bpe.c");
        let aux = d.join(format!("src/core/bpe.{extension}"));
        let obj = d.join("out/bpe.o");
        std::fs::create_dir_all(src.parent().unwrap()).unwrap();
        std::fs::create_dir_all(obj.parent().unwrap()).unwrap();
        std::fs::write(&src, "source").unwrap();
        std::fs::write(&aux, "old header").unwrap();
        std::fs::write(&obj, "object").unwrap();
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(120);
        let times = std::fs::FileTimes::new().set_modified(old);
        for path in [&src, &aux] {
            std::fs::File::options().write(true).open(path).unwrap().set_times(times).unwrap();
        }
        assert!(tokenizer_object_is_fresh(&src, &obj, tokenizer_aux_mtime(&d), false));

        // The C source and object mtimes do not change; only a header/include moves ahead of the object.
        std::fs::write(&aux, "edited header").unwrap();
        let src_mtime = std::fs::metadata(&src).unwrap().modified().unwrap();
        let obj_mtime = std::fs::metadata(&obj).unwrap().modified().unwrap();
        assert!(obj_mtime >= src_mtime, "this test requires the object's own source to remain older");
        assert!(!tokenizer_object_is_fresh(&src, &obj, tokenizer_aux_mtime(&d), false), ".{extension} edit was ignored");
        let _ = std::fs::remove_dir_all(&d);
    }
}
