//! `rexl3-quant` builds, answers `--help`, and in a GPU build links cuSOLVER and loads it.

use std::process::Command;

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rexl3-quant")).args(args).output().expect("run rexl3-quant")
}

#[test]
fn help_and_unknown_commands() {
    let out = run(&["--help"]);
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("info"));
    let out = run(&["nope"]);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(run(&[]).status.code(), Some(2));
}

#[test]
fn info_reports_the_libraries_or_the_stub_build() {
    let out = run(&["info"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains(gb_quant::PACK_SUFFIX));
    if cfg!(gb_quant_no_cuda) || !gb_cuda::rt::device_available() {
        return;
    }
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("cusolver "), "{stdout}");
}

#[test]
fn gpu_build_links_cusolver() {
    let Ok(out) = Command::new("readelf").arg("-d").arg(env!("CARGO_BIN_EXE_rexl3-quant")).output() else {
        return;
    };
    let text = String::from_utf8_lossy(&out.stdout);
    if text.contains("libcudart") {
        assert!(text.contains("libcusolver"), "{text}");
    }
}
