//! The engine commit stamp (`build.rs`, its `stamp` module), on real git repositories in a scratch directory: the environment wins;
//! a checkout of the workspace gives its HEAD; a git repository that only ENCLOSES the workspace (an export unpacked inside another
//! checkout) is never taken, and the export's `.rexl3_commit` is used instead; with nothing, the stamp is `unknown` with a warning (not a
//! made-up commit); `REXL3_BUILD_COMMIT` is not read, and is named in a warning.

// the build script itself (its tokenizer build and `main` are not called here)
#[allow(dead_code)]
#[path = "../build.rs"]
mod build_script;

use build_script::stamp::{stamp, Source, UNKNOWN};
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
